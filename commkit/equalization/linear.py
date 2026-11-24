"""Linear block equalization: channel estimation, ZF / MMSE, fixed taps and gain.

:func:`estimate_channel` estimates a (MIMO) channel from a known reference
(a :class:`ChannelEstimate`), :func:`correct_channel` inverts it in the
frequency domain (zero-forcing, or MMSE with a noise variance), and
:func:`apply_taps` applies frozen equalizer taps.  :func:`estimate_gain`
measures the data-aided amplitude gain of symbols against a reference (a
:class:`GainEstimate`, the gain the metrics divide by) and
:func:`correct_gain` removes it, putting the symbols on the reference's
scale.
"""

from __future__ import annotations

import logging
import operator
from dataclasses import dataclass
from typing import Any

import numpy as np

from .. import _fft
from .._array import as_2d, broadcast_channels, require_finite, require_int, restore_1d
from .._overlap_save import ols_apply
from .._value import _Record
from ..backend import ArrayType, dispatch, to_device
from ..core._signal_input import S, SignalInput
from ..core.signal import Signal
from ..logger import logger
from ..math import _data_aided_gain, _next_pow2
from ._common import _build_padded_samples, _normalize_inputs

__all__ = [
    "ChannelEstimate",
    "GainEstimate",
    "apply_taps",
    "correct_channel",
    "correct_gain",
    "estimate_channel",
    "estimate_gain",
]

# -----------------------------------------------------------------------------
# BLOCK EQUALIZATION (Signal-aware)
# -----------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class ChannelEstimate(_Record):
    """A (MIMO) channel estimated from a known reference, from
    :func:`estimate_channel`, inverted by :func:`correct_channel`.

    The constructor takes the same fields.

    Attributes
    ----------
    taps : ndarray
        Channel impulse response, ``(L,)`` for SISO or ``(C, C, L)`` for
        MIMO, where ``[i, j]`` is the FIR from input ``j`` to output ``i``,
        Hann-tapered about lag 0.
    center_tap : int
        Index of lag 0 in ``taps``; the taps before it are precursors
        (negative lags).
    response : ndarray
        Frequency response on the Welch grid in FFT-bin order, ``(nfft,)``
        for SISO or ``(nfft, C, C)`` for MIMO, untapered.
    """

    taps: ArrayType
    center_tap: int
    response: ArrayType


def correct_channel(
    samples: S,
    how: ChannelEstimate | ArrayType,
    *,
    noise_var: float = 0.0,
    center_tap: int | None = None,
) -> S:
    """Equalize a known channel in the frequency domain: zero-forcing or MMSE.

    Inverts a channel impulse response per frequency bin over overlap-save
    blocks, for SISO or MIMO channels: ``W = Hᴴ (H Hᴴ + σ² I)⁻¹``.
    ``noise_var=0`` gives zero-forcing (with a ``1e-12`` floor);
    ``noise_var > 0`` gives MMSE, which does not amplify the noise in
    spectral nulls.

    Parameters
    ----------
    samples : array_like or Signal
        Received samples, ``(N,)`` for SISO or ``(C, N)`` for MIMO.  A
        :class:`Signal` returns a new equalized :class:`Signal`.
    how : ChannelEstimate or array_like
        The channel: an estimate from :func:`estimate_channel`, which
        carries its centre tap, or an impulse response, ``(L,)`` for SISO or
        ``(C, C, L)`` for MIMO, where ``[i, j]`` is the FIR from input ``j``
        to output ``i``.  Cast to the precision of ``samples``.
    noise_var : float, default 0.0
        Noise variance ``σ²`` of the MMSE regularization; ``0.0`` gives ZF.
    center_tap : int, optional
        Index of lag 0 in an impulse response ``how``, in ``[0, L)``;
        ``None`` takes 0, a causal response (``rx = convolve(tx, h)[:N]``).
        Taps before it are precursors (negative lags).  An estimate carries
        its own.  The output is aligned with the transmitted samples (no
        delay).

    Returns
    -------
    ndarray or Signal
        Equalized samples, same shape, rate and device as the input.

    Raises
    ------
    ValueError
        If ``noise_var`` is not finite and ``>= 0``; if the impulse response
        is empty or not ``(L,)`` or ``(C, C, L)`` for ``C`` channels of
        ``samples``; or if ``center_tap`` is outside ``[0, L)`` or given
        with an estimate.
    TypeError
        If ``noise_var`` is not a real number.
    """
    inp = SignalInput(samples, "correct_channel()")
    x = inp.samples
    if isinstance(how, ChannelEstimate):
        if center_tap is not None:
            raise ValueError(
                "correct_channel(): center_tap is for an impulse response; a "
                "ChannelEstimate carries its own."
            )
        channel_estimate, center_tap = how.taps, how.center_tap
    else:
        channel_estimate = how

    noise_var = require_finite(
        noise_var, "correct_channel(): noise_var", low=0.0, inclusive=True
    )
    logger.info(
        "correct_channel: %s, noise_var=%.2e",
        "MMSE" if noise_var > 0 else "ZF",
        noise_var,
    )
    x, xp, _ = dispatch(x)
    # In the samples' precision before the FFT: a float64 estimate would
    # promote the whole overlap-save spectrum and its inverse to complex128.
    channel_estimate = xp.asarray(channel_estimate).astype(
        xp.result_type(x.dtype, xp.complex64), copy=False
    )

    # Capture BEFORE any reshape so the later frequency-domain branch
    # can still take the efficient scalar path for SISO channels.
    siso_channel = channel_estimate.ndim == 1

    x, was_1d = as_2d(x, name="samples")
    num_ch, N = x.shape
    if siso_channel:
        if channel_estimate.size == 0:
            raise ValueError("correct_channel(): channel taps must be non-empty.")
        if was_1d:
            channel_estimate = channel_estimate[None, None, :]
    elif channel_estimate.ndim != 3 or channel_estimate.shape[:2] != (num_ch, num_ch):
        raise ValueError(
            f"correct_channel(): channel taps must have shape (L,) or "
            f"({num_ch}, {num_ch}, L), got {channel_estimate.shape}."
        )
    elif channel_estimate.shape[-1] == 0:
        raise ValueError("correct_channel(): channel taps must be non-empty.")

    L = channel_estimate.shape[-1]
    center = 0 if center_tap is None else operator.index(center_tap)
    if not 0 <= center < L:
        raise ValueError(
            f"correct_channel(): center_tap must lie in [0, {L}) for a {L}-tap "
            f"channel, got {center}"
        )
    reg = noise_var if noise_var > 0 else 1e-12

    # N_fft must satisfy discard = N_fft//4 >= L so the IIR filter's causal
    # and anti-causal tails both decay within the discarded guard regions.
    N_fft = max(1024, _next_pow2(max(1, 4 * L)))

    logger.debug(
        "ZF/MMSE internals: N=%s, L=%s, num_ch=%s, N_fft=%s", N, L, num_ch, N_fft
    )

    if center:
        # Move lag 0 to index 0 circularly, so taps before `center` wrap to the
        # end of the FFT window as negative lags.
        pad = [(0, 0)] * (channel_estimate.ndim - 1) + [(0, N_fft - L)]
        channel_estimate = xp.roll(xp.pad(channel_estimate, pad), -center, axis=-1)

    if siso_channel:
        # SISO: scalar frequency-domain ZF/MMSE inversion.
        # channel_estimate was reshaped to (1,1,L) for was_1d; flatten to 1D.
        H = _fft.fft(channel_estimate.reshape(-1), n=N_fft)
        W = xp.conj(H) / (xp.abs(H) ** 2 + reg)

        def process(Y: ArrayType) -> ArrayType:
            """Each bin times its scalar inverse."""
            return Y * W

    else:
        # MIMO: per-bin (CxC) matrix inversion - cannot reduce to scalar multiply.
        H_f = _fft.fft(channel_estimate, n=N_fft, axis=-1)
        Hk = xp.transpose(H_f, (2, 0, 1))  # (N_fft, C_rx, C_tx)
        Hk_H = xp.conj(xp.transpose(Hk, (0, 2, 1)))  # (N_fft, C_tx, C_rx)
        HHh = Hk @ Hk_H  # (N_fft, C_rx, C_rx)

        eye = xp.eye(num_ch, dtype=x.dtype)[None, :, :]
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

        def process(Y: ArrayType) -> ArrayType:
            """Each bin's (C, C) inverse times the bin's channel vector.

            Batched GEMM over the bins rather than einsum, for GPU use.
            """
            Y_t = xp.transpose(Y, (2, 0, 1))  # (N_fft, num_ch, blocks)
            return xp.transpose(Wk @ Y_t, (1, 2, 0))  # (num_ch, blocks, N_fft)

    out = ols_apply(x, N_fft, process)

    return inp.wrap(restore_1d(was_1d, out))


def apply_taps(
    samples: S,
    weights: ArrayType,
    *,
    sps: int | None = None,
    normalize: bool = True,
    input_norm_factor: float | np.ndarray | None = None,
    samples_prefix: ArrayType | None = None,
    pad_mode: str = "zeros",
) -> S:
    """Apply frozen equalizer taps: the butterfly filter without adaptation.

    Reuses the weights of an earlier equalizer run on new samples, one
    output symbol per ``sps`` input samples::

        y[i, n] = sum_j sum_t conj(W[i, j, t]) * x[j, n*sps - T_c + t]

    with the center tap ``T_c = num_taps // 2``: the filter of the adaptive
    equalizers, applied by overlap-save on the input's device.

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
        Divide each channel of ``samples`` by its RMS first (unit mean
        power), as the adaptive equalizers do.  ``False`` when the caller has already scaled the
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
    inp = SignalInput(samples, "apply_taps()")
    x = inp.samples
    metadata: dict[str, Any] = {}
    sps = inp.integer_sps(sps)
    if inp.signal is not None:
        metadata["sampling_rate"] = inp.symbol_rate()

    x, xp, _ = dispatch(x)
    weights = xp.asarray(weights)

    # Promote SISO -> MIMO shapes for uniform butterfly code path
    x, was_1d = as_2d(x, name="samples")
    if weights.ndim == 1:
        weights = weights[None, None, :]  # (T,) -> (1, 1, T)

    C, N = x.shape
    num_taps = weights.shape[-1]
    n_sym = N // sps

    _at_eq_norm = None
    if normalize:
        x, _at_eq_norm = _normalize_inputs(x, input_norm_factor=input_norm_factor)

    if weights.ndim != 3 or weights.shape[:2] != (C, C):
        raise ValueError(
            f"weights must be (num_taps,) or ({C}, {C}, num_taps) for {C} "
            f"channel(s), got shape {tuple(weights.shape)}."
        )
    dtype = np.result_type(x.dtype, weights.dtype, np.complex64)

    # y[i, n] = sum_j sum_t conj(W[i, j, t]) * p[j, n*sps + t] on the record
    # p padded by the centre tap on the left: the correlation of p with the
    # taps at the full rate, kept every sps-th sample.  It runs by
    # overlap-save as the convolution with h[k] = conj(W[-k]) (k = 1 - T ..
    # 0), shifted by the centre tap when the padding is zeros, which the
    # overlap-save extension supplies: p is then x itself.
    c_tap = num_taps // 2
    if samples_prefix is None and pad_mode == "zeros":
        shift = c_tap
        padded = x
    else:
        # _build_padded_samples is backend-generic: pad on the input device
        # instead of round-tripping the full signal through the host.
        shift = 0
        pad_right = max(0, (n_sym - 1) * sps + num_taps - (c_tap + N))
        padded = _build_padded_samples(
            x.astype(dtype, copy=False),
            c_tap,
            pad_right,
            samples_prefix,
            pad_mode,
            _at_eq_norm,
            sps,
        )
    N_fft = max(1024, _next_pow2(4 * num_taps))
    # h[k] at index k mod N_fft, so H = conj(FFT of W placed at -shift + t).
    kernel = xp.zeros((C, C, N_fft), dtype=dtype)
    kernel[..., :num_taps] = weights
    H = xp.conj(_fft.fft(xp.roll(kernel, -shift, axis=-1), axis=-1))

    def process(Y: ArrayType) -> ArrayType:
        """Each bin's (C, C) response times the bin's channel vector."""
        out = H[:, 0, None, :] * Y[0]
        for j in range(1, C):
            out += H[:, j, None, :] * Y[j]
        return out

    # The kernel spans lags shift - T + 1 .. shift, so a guard of T - 1
    # holds both transients; a quarter block would discard half of each.
    y = ols_apply(padded.astype(dtype, copy=False), N_fft, process, guard=num_taps - 1)
    y = y[:, : n_sym * sps : sps]

    return inp.wrap(restore_1d(was_1d, y), **metadata)


# -----------------------------------------------------------------------------
# DATA-AIDED CHANNEL ESTIMATION (Welch H1 transfer function; Signal-aware)
# -----------------------------------------------------------------------------


def estimate_channel(
    samples: ArrayType | Signal,
    reference: ArrayType | Signal,
    *,
    nfft: int = 512,
    regularization: float = 1e-2,
    max_windows: int = 4096,
    num_taps: int | None = None,
) -> ChannelEstimate:
    r"""Estimate a (MIMO) channel from a known reference by Welch's H1 method.

    Estimates the frequency response
    :math:`B(f) \approx S_{yx}(f)\,S_{xx}(f)^{-1}` from a known ``reference``
    ``x`` to the measured ``samples`` ``y``, where :math:`S_{xx}` and
    :math:`S_{yx}` are the Welch auto- and cross-spectra (averaged
    Hann-windowed periodograms).  This is the H1 channel-sounding estimator,
    extended to the full ``(C, C)`` cross-spectral matrix, so the MIMO
    response comes out of one pass: the reference is transformed once and
    every channel pair is formed in one product.

    Invert the estimated channel with :func:`correct_channel`::

        est = estimate_channel(rx, tx, num_taps=L)
        tx_hat = correct_channel(rx, est)  # aligned with tx, no delay

    The taps are a channel response, not equalizer weights, so they are not
    input for :func:`apply_taps`.

    Parameters
    ----------
    samples, reference : array_like or Signal
        Channel output and input, time-aligned at the same rate, ``(N,)`` or
        ``(C, N)``; the measured data first and the known reference second,
        as in the metrics.  A Signal contributes its samples.
    nfft : int, default 512
        Welch segment length (frequency resolution ``fs / nfft``).
    regularization : float, default 1e-2
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
        Length of the impulse response, at most ``nfft``: the response is
        centred on lag 0 (``center_tap = num_taps // 2``), Hann-tapered and
        truncated.  ``None`` takes ``nfft``.

    Returns
    -------
    ChannelEstimate
        The taps, their centre tap and the frequency response ``B(f)``, on
        the input's device.
    """
    received = SignalInput(samples, "estimate_channel()").samples
    ref_raw = SignalInput(reference, "estimate_channel()").samples
    y, xp, _ = dispatch(received)
    x = to_device(ref_raw, "cpu" if xp is np else "gpu")
    if x.shape != y.shape:
        raise ValueError(
            f"estimate_channel(): samples shape {y.shape} does not match "
            f"reference shape {x.shape}."
        )
    nfft = require_int(nfft, "estimate_channel(): nfft", low=2)
    max_windows = require_int(max_windows, "estimate_channel(): max_windows", low=1)
    regularization = require_finite(
        regularization, "estimate_channel(): regularization", low=0.0, inclusive=True
    )
    if num_taps is not None:
        num_taps = require_int(num_taps, "estimate_channel(): num_taps", low=1)
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
    nfft = int(min(nfft, N))
    win = xp.hanning(nfft).astype(np.empty(0, cdtype).real.dtype)

    # Welch auto/cross spectra, vectorized over windows (capped). One batched FFT
    # of the reference and the received, then every (C, C) cross term via einsum -
    # no per-window Python loop, no redundant re-FFT of shared channels.
    step = max(nfft // 2, (N - nfft) // max(max_windows - 1, 1))
    starts = xp.arange(0, N - nfft + 1, step)
    idx = starts[:, None] + xp.arange(nfft)[None, :]  # (n_win, nfft)
    X = _fft.fft(x[:, idx] * win, axis=-1)  # (C, n_win, nfft)
    Y = _fft.fft(y[:, idx] * win, axis=-1)
    Sxx = xp.einsum("cwf,dwf->fcd", X, X.conj()).astype(xp.complex128)  # (nfft, C, C)
    Syx = xp.einsum("cwf,dwf->fcd", Y, X.conj()).astype(xp.complex128)

    eye = xp.eye(C, dtype=xp.complex128)
    reg_xx = regularization * float(xp.einsum("fcc->f", Sxx).real.mean() / C)
    B = Syx @ xp.linalg.inv(Sxx + reg_xx * eye)  # (nfft, C, C), H1 estimate

    logger.info(
        "estimate_channel (Welch H1): nfft=%d, n_win=%d, regularization=%.2e "
        "[C=%d, N=%d]",
        nfft,
        int(starts.shape[0]),
        regularization,
        C,
        N,
    )

    # Centered, Hann-tapered impulse response, truncated to num_taps: lag 0 sits
    # at index num_taps // 2 for odd and even lengths alike.
    n_taps = nfft if num_taps is None else int(min(num_taps, nfft))
    b_t = xp.fft.fftshift(_fft.ifft(B, axis=0), axes=0)  # (nfft, C, C), lag axis 0
    center = nfft // 2
    half = n_taps // 2
    taps = b_t[center - half : center - half + n_taps]  # (L, C, C)
    # Symmetric Hann about lag 0 (unit weight there); an even length drops the
    # last sample of the (2*half + 1)-point window.
    window = np.hanning(2 * half + 1)[:n_taps]
    taps = taps * xp.asarray(window)[:, None, None]
    if single:
        return ChannelEstimate(taps[:, 0, 0], half, B[:, 0, 0])  # (L,), (nfft,)
    # (C, C, L), the layout of correct_channel
    return ChannelEstimate(xp.transpose(taps, (1, 2, 0)), half, B)


# -----------------------------------------------------------------------------
# DATA-AIDED GAIN (Signal-aware)
# -----------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class GainEstimate(_Record):
    """The data-aided amplitude gain of symbols against a reference, from
    :func:`estimate_gain`, removed by :func:`correct_gain`.

    The constructor takes the same fields.

    Attributes
    ----------
    value : ndarray
        Gain ``g = |<r s*>| / <|s|^2>`` per channel, float64: 0-d for
        ``(N,)`` input, ``(C,)`` for ``(C, N)``, on the input's device.
    """

    value: ArrayType


def estimate_gain(
    symbols: ArrayType | Signal,
    reference: ArrayType | None = None,
    *,
    num_skip_symbols: int = 0,
) -> GainEstimate:
    """Measure the data-aided amplitude gain of symbols against a reference.

    ``g = |<r s*>| / <|s|^2>`` per channel: the amplitude at which the
    reference ``s`` appears in the received symbols ``r``.  AWGN is
    uncorrelated with ``s``, so ``g`` is unbiased at any SNR, unlike the
    gain of a power normalization, which counts the noise as signal and
    shrinks the symbols by ``1/sqrt(1 + 1/SNR)``.  Only the magnitude is
    measured: a rotation is left to carrier recovery.  It is the gain
    :func:`~commkit.metrics.evm`, :func:`~commkit.metrics.snr`,
    :func:`~commkit.metrics.noise_variance` and the information rates of a
    Signal divide by.

    Parameters
    ----------
    symbols : array_like or Signal
        Received symbols at one sample per symbol, ``(N,)`` or ``(C, N)``.  A
        Signal must be at one sample per symbol; its ``reference.symbols`` is
        the default reference.
    reference : array_like, optional
        Transmitted symbols, ``(N,)`` (shared by every channel) or the shape
        of ``symbols``.  Required for arrays.
    num_skip_symbols : int, default 0
        Leading symbols left out of the estimate (for example equalizer
        training or convergence).

    Returns
    -------
    GainEstimate
        The gain per channel, on the input's device.

    Raises
    ------
    ValueError
        Without a reference for an array, on a shape mismatch, if
        ``num_skip_symbols`` is negative or leaves no symbol, or for a frame
        Signal or one not at one sample per symbol.
    TypeError
        If ``num_skip_symbols`` is not an integer.

    See Also
    --------
    correct_gain : Divide the symbols by the gain.

    Examples
    --------
    >>> y = correct_gain(y, estimate_gain(y, tx))
    >>> llrs = compute_llr(y, constellation=c, noise_var=noise_variance(y, tx))
    """
    name = "estimate_gain()"
    skip = require_int(num_skip_symbols, f"{name}: num_skip_symbols", low=0)
    inp = SignalInput(symbols, name)
    y, xp, _ = dispatch(inp.symbols())
    ref = inp.reference_symbols(reference, required=True)
    y2, was_1d = as_2d(y, name="symbols")
    d2 = broadcast_channels(
        to_device(ref, "cpu" if xp is np else "gpu"), y2.shape[0], xp, name="reference"
    )
    if d2.shape[-1] != y2.shape[-1]:
        raise ValueError(
            f"{name}: shape mismatch: symbols have {y2.shape[-1]} symbols, the "
            f"reference {d2.shape[-1]}."
        )
    if skip >= y2.shape[-1]:
        raise ValueError(
            f"{name}: num_skip_symbols={skip} leaves no symbol of {y2.shape[-1]}."
        )
    y2, d2 = y2[:, skip:], d2[:, skip:]
    g = _data_aided_gain(y2, d2, xp).astype(xp.float64)[:, 0]
    if logger.isEnabledFor(logging.INFO):
        logger.info("estimate_gain: g = %s", np.round(to_device(g, "cpu"), 4).tolist())
    return GainEstimate(value=g[0] if was_1d else g)


def correct_gain(samples: S, how: GainEstimate) -> S:
    """Divide samples by a data-aided gain, putting them on the reference's scale.

    Parameters
    ----------
    samples : array_like or Signal
        Samples, ``(N,)`` or ``(C, N)``, at any rate: the gain of the
        symbols applies to the waveform they came from too.
    how : GainEstimate
        From :func:`estimate_gain`; its value is 0-d (one gain for every
        channel) or ``(C,)`` for ``C`` channels of ``samples``.

    Returns
    -------
    ndarray or Signal
        ``samples / g``, in the precision of ``samples``, on their device.

    Raises
    ------
    TypeError
        If ``how`` is not a :class:`GainEstimate`.
    ValueError
        If the gain has another number of channels than ``samples``, or a
        value that is not finite and ``> 0``.
    """
    name = "correct_gain()"
    if not isinstance(how, GainEstimate):
        raise TypeError(
            f"{name}: how must be a GainEstimate, got {type(how).__name__}."
        )
    inp = SignalInput(samples, name)
    x, xp, _ = dispatch(inp.samples)
    x2, was_1d = as_2d(x, name="samples")
    g = xp.asarray(to_device(how.value, "cpu" if xp is np else "gpu"), dtype=xp.float64)
    if g.ndim > 1 or (g.ndim == 1 and g.shape[0] != x2.shape[0]):
        raise ValueError(
            f"{name}: the gain has shape {g.shape}; samples have {x2.shape[0]} "
            "channel(s)."
        )
    if not bool(xp.all(xp.isfinite(g) & (g > 0))):
        raise ValueError(f"{name}: the gain must be finite and > 0.")
    real = x2.real.dtype if x2.dtype.kind in "fc" else xp.dtype(xp.float32)
    inverse = (1.0 / xp.reshape(g, (-1, 1))).astype(real)
    logger.debug("correct_gain: %d channel(s)", x2.shape[0])
    return inp.wrap(restore_1d(was_1d, x2 * inverse))
