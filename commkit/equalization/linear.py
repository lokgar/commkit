"""Linear block equalization: channel estimation, ZF / MMSE and fixed taps.

:func:`estimate_transfer_function` estimates a (MIMO) channel from a known
reference, :func:`zf_equalizer` inverts a channel estimate in the frequency
domain (zero-forcing, or MMSE with a noise variance), and :func:`apply_taps`
applies frozen equalizer taps.
"""

from __future__ import annotations

import operator

import numpy as np

from .. import _fft
from .._array import as_2d, restore_1d
from .._overlap_save import ols_backward, ols_forward
from ..backend import ArrayType, dispatch
from ..core._signal_adapter import adapt_signal, require_integer_sps
from ..core.signal import Signal
from ..logger import logger
from ._common import _build_padded_samples, _normalize_inputs

__all__ = ["apply_taps", "estimate_transfer_function", "zf_equalizer"]

# -----------------------------------------------------------------------------
# BLOCK EQUALIZATION (Signal-aware)
# -----------------------------------------------------------------------------


def zf_equalizer(
    samples: ArrayType | Signal,
    channel_estimate: ArrayType,
    *,
    noise_variance: float = 0.0,
    center: int = 0,
) -> ArrayType | Signal:
    """Equalize a known channel in the frequency domain: zero-forcing or MMSE.

    Inverts a channel impulse response estimate per frequency bin over
    overlap-save blocks, for SISO or MIMO channels: ``W = Hᴴ (H Hᴴ + σ² I)⁻¹``.
    ``noise_variance=0`` gives zero-forcing (with a ``1e-12`` floor);
    ``noise_variance > 0`` gives MMSE, which does not amplify the noise in
    spectral nulls.

    Parameters
    ----------
    samples : array_like or Signal
        Received samples, ``(N,)`` for SISO or ``(C, N)`` for MIMO.  A
        :class:`Signal` returns a new equalized :class:`Signal`.
    channel_estimate : array_like
        Channel impulse response, ``(L,)`` for SISO or ``(C, C, L)`` for
        MIMO, where ``[i, j]`` is the FIR from input ``j`` to output ``i``.
        Index ``center`` holds lag 0.  Cast to the precision of ``samples``.
    noise_variance : float, default 0.0
        Noise variance ``σ²`` of the MMSE regularization; ``0.0`` gives ZF.
    center : int, default 0
        Index of lag 0 in ``channel_estimate``, in ``[0, L)``. ``0`` means a
        causal response (``rx = convolve(tx, h)[:N]``). A centered estimate
        from :func:`estimate_transfer_function` with ``num_taps=L`` has
        ``center=L // 2``; taps before ``center`` are precursors (negative
        lags). The output is aligned with the transmitted samples (no delay).

    Returns
    -------
    ndarray or Signal
        Equalized samples, same shape, rate and device as the input.

    Raises
    ------
    ValueError
        If ``center`` is outside ``[0, L)``.
    """
    signal_adapter = adapt_signal(samples, function_name="zf_equalizer()")
    samples = signal_adapter.array

    logger.info("ZF/MMSE equalizer: noise_variance=%.2e", noise_variance)
    samples, xp, _ = dispatch(samples)
    # In the samples' precision before the FFT: a float64 estimate would
    # promote the whole overlap-save spectrum and its inverse to complex128.
    channel_estimate = xp.asarray(channel_estimate).astype(
        xp.result_type(samples.dtype, xp.complex64), copy=False
    )

    # Capture BEFORE any reshape so the later frequency-domain branch
    # can still take the efficient scalar path for SISO channels.
    siso_channel = channel_estimate.ndim == 1

    samples, was_1d = as_2d(samples, name="samples")
    if was_1d and siso_channel:
        channel_estimate = channel_estimate[None, None, :]

    num_ch, N = samples.shape
    L = channel_estimate.shape[-1] if channel_estimate.ndim > 0 else 1
    center = operator.index(center)
    if not 0 <= center < L:
        raise ValueError(
            f"center must lie in [0, {L}) for a {L}-tap channel_estimate, got {center}"
        )
    reg = noise_variance if noise_variance > 0 else 1e-12

    # N_fft must satisfy discard = N_fft//4 >= L so the IIR filter's causal
    # and anti-causal tails both decay within the discarded guard regions.
    N_fft = max(1024, 1 << (max(1, 4 * L) - 1).bit_length())

    logger.debug(
        "ZF/MMSE internals: N=%s, L=%s, num_ch=%s, N_fft=%s", N, L, num_ch, N_fft
    )

    if center:
        # Move lag 0 to index 0 circularly, so taps before `center` wrap to the
        # end of the FFT window as negative lags.
        pad = [(0, 0)] * (channel_estimate.ndim - 1) + [(0, N_fft - L)]
        channel_estimate = xp.roll(xp.pad(channel_estimate, pad), -center, axis=-1)

    # --- Shared OLS forward pass: pad -> stride_tricks -> batch FFT ---
    Y, meta = ols_forward(samples, N_fft)  # Y: (num_ch, num_blocks, N_fft)

    if siso_channel:
        # SISO: scalar frequency-domain ZF/MMSE inversion.
        # channel_estimate was reshaped to (1,1,L) for was_1d; flatten to 1D.
        H = _fft.fft(channel_estimate.reshape(-1), n=N_fft)
        W = xp.conj(H) / (xp.abs(H) ** 2 + reg)
        X_hat_f = Y * W
    else:
        # MIMO: per-bin (CxC) matrix inversion - cannot reduce to scalar multiply.
        H_f = _fft.fft(channel_estimate, n=N_fft, axis=-1)
        Hk = xp.transpose(H_f, (2, 0, 1))  # (N_fft, C_rx, C_tx)
        Hk_H = xp.conj(xp.transpose(Hk, (0, 2, 1)))  # (N_fft, C_tx, C_rx)
        HHh = Hk @ Hk_H  # (N_fft, C_rx, C_rx)

        eye = xp.eye(num_ch, dtype=samples.dtype)[None, :, :]
        if num_ch == 2:
            # Fast-path: Explicit 2x2 matrix inversion via Cramer's Rule avoiding
            # GPU LAPACK solver overheads (e.g. cuSOLVER batch inversion).
            H_reg = HHh + reg * eye
            a = H_reg[:, 0, 0]
            b = H_reg[:, 0, 1]
            c = H_reg[:, 1, 0]
            d = H_reg[:, 1, 1]
            det = a * d - b * c

            inv_term = xp.empty_like(H_reg)
            inv_term[:, 0, 0] = d / det
            inv_term[:, 0, 1] = -b / det
            inv_term[:, 1, 0] = -c / det
            inv_term[:, 1, 1] = a / det
        else:
            inv_term = xp.linalg.inv(HHh + reg * eye)

        Wk = Hk_H @ inv_term  # (N_fft, C_tx, C_rx)

        # Vectorized batch matrix multiplication across frequency bins and blocks.
        # Uses explicit batched GEMM instead of einsum for better GPU utilization.
        Y_t = xp.transpose(Y, (2, 0, 1))  # (N_fft, num_ch, num_blocks)
        X_hat_k = Wk @ Y_t  # (N_fft, C_tx, C_rx) @ (N_fft, C_rx, num_blocks)
        X_hat_f = xp.transpose(X_hat_k, (1, 2, 0))  # (num_ch, num_blocks, N_fft)

    # --- Shared OLS backward pass: batch IFFT -> symmetric discard -> reshape ---
    out = ols_backward(X_hat_f, meta)

    return signal_adapter.wrap_samples(restore_1d(was_1d, out))


def apply_taps(
    samples: ArrayType | Signal,
    weights: ArrayType,
    *,
    sps: int | None = None,
    normalize: bool = True,
    input_norm_factor: float | np.ndarray | None = None,
    samples_prefix: ArrayType | None = None,
    pad_mode: str = "zeros",
) -> ArrayType | Signal:
    """Apply frozen equalizer taps: the butterfly filter without adaptation.

    Reuses the weights of an earlier equalizer run on new samples, one
    output symbol per ``sps`` input samples::

        y[i, n] = sum_j sum_t conj(W[i, j, t]) * x[j, n*sps - T_c + t]

    with the center tap ``T_c = num_taps // 2``: the filter of the adaptive
    equalizers, as one batched ``einsum`` on the input's device.

    Parameters
    ----------
    samples : array_like or Signal
        Input samples at ``sps`` samples per symbol, ``(N,)`` for SISO or
        ``(C, N)`` for MIMO.  A :class:`Signal` returns a new :class:`Signal`
        at the symbol rate (``sampling_rate = symbol_rate``).
    weights : array_like
        Frozen tap weights, typically ``EqualizerResult.weights`` of an
        earlier run: ``(num_taps,)`` for SISO or ``(C, C, num_taps)`` for
        the MIMO butterfly.
    sps : int, optional
        Samples per symbol, an integer.  Taken from the Signal; required for
        array input.  A value that disagrees with the Signal raises.
    normalize : bool, default True
        Normalize ``samples`` to unit symbol power first, as the adaptive
        equalizers do.  ``False`` when the caller has already scaled the
        input as in the run that produced ``weights``.
    input_norm_factor : float or array_like, optional
        Normalization factor to divide by instead of measuring one (with
        ``normalize=True``), typically ``EqualizerResult.input_norm_factor``
        of the run that produced ``weights``.
    samples_prefix : array_like, optional
        Input samples that precede ``samples`` (e.g. the end of the previous
        record), at least ``num_taps // 2`` per channel: its last
        ``num_taps // 2`` samples fill the left edge of the first windows
        instead of padding, so a record split into pieces is filtered as one.
    pad_mode : {'zeros', 'edge', 'reflect'}, default 'zeros'
        Padding without ``samples_prefix``: zeros, the edge samples repeated,
        or the record mirrored about its end samples.

    Returns
    -------
    ndarray or Signal
        Equalized symbols, ``(N_sym,)`` for SISO or ``(C, N_sym)`` for MIMO
        with ``N_sym = N // sps``, on the input's device.

    Examples
    --------
    Freeze taps from a training run and apply to a new capture::

        result = equalization.lms(train_signal, training_symbols, num_taps=31)
        y = equalization.apply_taps(new_signal, result.weights,
                                    input_norm_factor=result.input_norm_factor)
    """
    signal_adapter = adapt_signal(samples, function_name="apply_taps()")
    samples = signal_adapter.array
    metadata = {}
    sps = require_integer_sps(signal_adapter.resolve_fact("sps", sps), "apply_taps()")
    if signal_adapter.signal is not None:
        metadata["sampling_rate"] = signal_adapter.signal.symbol_rate

    samples, xp, _ = dispatch(samples)
    weights = xp.asarray(weights)

    # Promote SISO -> MIMO shapes for uniform butterfly code path
    samples, was_1d = as_2d(samples, name="samples")
    if weights.ndim == 1:
        weights = weights[None, None, :]  # (T,) -> (1, 1, T)

    C, N = samples.shape
    num_taps = weights.shape[-1]
    n_sym = N // sps

    _at_eq_norm = None
    if normalize:
        samples, _, _at_eq_norm = _normalize_inputs(
            samples, None, sps, input_norm_factor=input_norm_factor
        )

    # Pad left by center tap so window[0] is center-aligned with sample 0,
    # pad right to fill the last window completely - mirrors LMS pre-processing.
    c_tap = num_taps // 2
    pad_left = c_tap
    pad_right = n_sym * sps - N + num_taps - 1 - pad_left
    if samples_prefix is None and pad_mode == "zeros":
        samples_padded = xp.pad(samples, ((0, 0), (pad_left, pad_right)))
    else:
        # _build_padded_samples is backend-generic: pad on the input device
        # instead of round-tripping the full signal through the host.
        samples_padded = xp.ascontiguousarray(
            _build_padded_samples(
                samples.astype(xp.complex64),
                pad_left,
                pad_right,
                samples_prefix,
                pad_mode,
                _at_eq_norm,
                sps,
            )
        )

    # Zero-copy sliding-window view: (C, N_sym, num_taps)
    # windows[c, n, t] = samples_padded[c, n*sps + t]
    s0, s1 = samples_padded.strides
    windows = xp.lib.stride_tricks.as_strided(
        samples_padded,
        shape=(C, n_sym, num_taps),
        strides=(s0, sps * s1, s1),
    )

    # Butterfly forward pass - single batched GEMM (dispatches to cuBLAS on GPU)
    # y[i, n] = Σ_j Σ_t conj(W[i,j,t]) * windows[j, n, t]
    y = xp.einsum("ijt,jnt->in", xp.conj(weights), windows)  # (C, N_sym)

    return signal_adapter.wrap_samples(restore_1d(was_1d, y), **metadata)


# -----------------------------------------------------------------------------
# DATA-AIDED CHANNEL ESTIMATION (Welch H1 transfer function; Signal-aware)
# -----------------------------------------------------------------------------


def estimate_transfer_function(
    reference: ArrayType | Signal,
    received: ArrayType | Signal,
    *,
    n_fft: int = 512,
    reg: float = 1e-2,
    max_windows: int = 4096,
    num_taps: int | None = None,
) -> ArrayType:
    r"""Estimate a (MIMO) channel from a known reference by Welch's H1 method.

    Estimates the frequency response
    :math:`B(f) \approx S_{yx}(f)\,S_{xx}(f)^{-1}` from a known ``reference``
    ``x`` to the measured ``received`` ``y``, where :math:`S_{xx}` and
    :math:`S_{yx}` are the Welch auto- and cross-spectra (averaged
    Hann-windowed periodograms).  This is the H1 channel-sounding estimator,
    extended to the full ``(C, C)`` cross-spectral matrix, so the MIMO
    response comes out of one pass: the reference is transformed once and
    every channel pair is formed in one product.

    Invert the estimated channel with :func:`zf_equalizer`, passing the center
    index of the taps::

        h = estimate_transfer_function(tx, rx, num_taps=L)
        tx_hat = zf_equalizer(rx, h, center=L // 2)  # aligned with tx, no delay

    The taps are a channel response, not equalizer weights, so they are not
    input for :func:`apply_taps`.

    Parameters
    ----------
    reference, received : array_like or Signal
        Channel input and output, time-aligned at the same rate, ``(N,)`` or
        ``(C, N)``.  A Signal contributes its samples.
    n_fft : int, default 512
        Welch segment length (frequency resolution ``fs / n_fft``).
    reg : float, default 1e-2
        Relative Tikhonov loading of the :math:`S_{xx}` inverse: a diagonal
        term scaled to the mean auto-spectrum.  It stabilizes the estimate
        where the reference has little power (band edges) independently of
        the signal level.  The H1 ratio does not depend on the overall scale
        or the number of windows.
    max_windows : int, default 4096
        Cap on the number of averaged segments.  An LTI channel is static, so a few
        thousand windows spread over the record give a stable estimate; capping
        bounds the vectorized FFT batch (the averaging only trades variance, not
        bias, so the cap does not move the estimate).
    num_taps : int, optional
        Return the centered, Hann-tapered impulse response truncated to
        ``num_taps`` taps, ``(C, C, num_taps)`` for MIMO or ``(num_taps,)``
        for SISO.  Lag 0 is at index ``num_taps // 2`` (the taps before it are
        negative lags), so pass ``center=num_taps // 2`` to
        :func:`zf_equalizer`.  By default the frequency response ``B(f)`` is
        returned in FFT-bin order, ``(n_fft, C, C)`` for MIMO or ``(n_fft,)``
        for SISO.

    Returns
    -------
    ndarray
        Frequency response or impulse response (see ``num_taps``), on the
        input's device.
    """
    reference = adapt_signal(
        reference, function_name="estimate_transfer_function(reference)"
    ).array
    received = adapt_signal(
        received, function_name="estimate_transfer_function(received)"
    ).array
    x, xp, _ = dispatch(reference)
    y = xp.asarray(received)
    single = x.ndim == 1
    if single:
        x, y = x[None, :], y[None, :]
    # Windows, FFTs and the window sums of the cross-spectra in the inputs'
    # precision (complex64 by default; complex128 changed the estimate by
    # under 1e-6, far below its statistical error); the small per-bin solve runs in complex128.
    cdtype = np.result_type(x.dtype, y.dtype, np.complex64)
    x = x.astype(cdtype, copy=False)
    y = y.astype(cdtype, copy=False)
    C, N = x.shape
    n_fft = int(min(n_fft, N))
    win = xp.hanning(n_fft).astype(np.empty(0, cdtype).real.dtype)

    # Welch auto/cross spectra, vectorized over windows (capped). One batched FFT
    # of the reference and the received, then every (C, C) cross term via einsum -
    # no per-window Python loop, no redundant re-FFT of shared channels.
    step = max(n_fft // 2, (N - n_fft) // max(max_windows - 1, 1))
    starts = xp.arange(0, N - n_fft + 1, step)
    idx = starts[:, None] + xp.arange(n_fft)[None, :]  # (n_win, n_fft)
    X = _fft.fft(x[:, idx] * win, axis=-1)  # (C, n_win, n_fft)
    Y = _fft.fft(y[:, idx] * win, axis=-1)
    Sxx = xp.einsum("cwf,dwf->fcd", X, X.conj()).astype(xp.complex128)  # (n_fft, C, C)
    Syx = xp.einsum("cwf,dwf->fcd", Y, X.conj()).astype(xp.complex128)

    eye = xp.eye(C, dtype=xp.complex128)
    reg_xx = reg * float(xp.einsum("fcc->f", Sxx).real.mean() / C)
    B = Syx @ xp.linalg.inv(Sxx + reg_xx * eye)  # (n_fft, C, C), H1 estimate

    logger.info(
        "estimate_transfer_function (Welch H1): n_fft=%d, n_win=%d, reg=%.2e "
        "[C=%d, N=%d, %s]",
        n_fft,
        int(starts.shape[0]),
        reg,
        C,
        N,
        "taps" if num_taps is not None else "freq-response",
    )

    if num_taps is None:
        return B[:, 0, 0] if single else B

    # Centered, Hann-tapered impulse response, truncated to num_taps: lag 0 sits
    # at index num_taps // 2 for odd and even lengths alike.
    n_taps = int(min(num_taps, n_fft))
    b_t = xp.fft.fftshift(_fft.ifft(B, axis=0), axes=0)  # (n_fft, C, C), lag axis 0
    center = n_fft // 2
    half = n_taps // 2
    taps = b_t[center - half : center - half + n_taps]  # (L, C, C)
    # Symmetric Hann about lag 0 (unit weight there); an even length drops the
    # last sample of the (2*half + 1)-point window.
    window = np.hanning(2 * half + 1)[:n_taps]
    taps = taps * xp.asarray(window)[:, None, None]
    if single:
        return taps[:, 0, 0]  # (L,)
    return xp.transpose(taps, (1, 2, 0))  # (C, C, L), matches zf_equalizer layout
