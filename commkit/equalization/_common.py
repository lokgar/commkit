"""Shared helpers of the equalization package (private).

Normalization, padding, weight initialization and validation, used by the
sequential and the block equalizers.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .._array import restore_1d
from ..backend import dispatch, to_device
from ..core._signal_adapter import require_integer_sps
from ..logger import logger
from .result import EqualizerResult, EqualizerState

# -----------------------------------------------------------------------------
# SHARED HELPERS
# -----------------------------------------------------------------------------


def _normalize_inputs(
    samples: Any, training_symbols: Any, sps: int, input_norm_factor: Any = None
) -> tuple[Any, Any, Any]:
    """Scale samples to unit symbol power; training symbols pass through.

    At more than one sample per symbol the timing phase is unknown, and the
    strided power ``samples[..., ::sps]`` can land on zero crossings of the
    Nyquist pulse, underestimating the power and destabilizing the
    adaptation.  The power of all samples over the whole record is used
    instead; it does not depend on whether training symbols are given.

    Parameters
    ----------
    samples : array_like
        Complex samples, ``(N,)`` or ``(C, N)``, NumPy or CuPy.
    training_symbols : array_like or None
        Training symbols at one sample per symbol, ``(K,)`` or ``(C, K)``.
    sps : int
        Samples per symbol.
    input_norm_factor : float or numpy.ndarray, optional
        Factor to divide by instead of measuring the RMS: the
        ``EqualizerResult.input_norm_factor`` of an earlier call, to keep
        successive records on one power scale.  A ``float`` for SISO, ``(C,)``
        for MIMO.

    Returns
    -------
    samples : ndarray
        The samples at unit symbol power, same shape and device.
    training_symbols : ndarray or None
        Unchanged: known symbols are already on the constellation's scale.
    input_norm_factor : float or numpy.ndarray
        Per-channel factor ``rms(channel)·sqrt(sps)`` the samples were divided
        by: a ``float`` for SISO, ``(C,)`` for MIMO.  Stored in
        ``EqualizerResult.input_norm_factor``, so that another capture (vacuum
        noise) passed through the same frozen taps can be put on the same
        absolute power scale.
    """
    if input_norm_factor is not None:
        # Caller supplies scale - skip RMS and just apply it.
        nf = input_norm_factor
        if samples.ndim == 1:
            samples = samples / float(nf)
        else:
            # Keep the stored dtype: a cold start divides by the float32 RMS,
            # and a resumed run must round identically.
            nf_arr = np.asarray(nf).ravel()
            if nf_arr.shape[0] != samples.shape[0]:
                raise ValueError(
                    f"input_norm_factor shape {nf_arr.shape} does not match "
                    f"samples channel count {samples.shape[0]}."
                )
            _, xp_loc, _ = dispatch(samples)
            nf_dev = xp_loc.asarray(nf_arr)[..., None]  # (C, 1) on same device
            samples = samples / nf_dev
        return samples, training_symbols, input_norm_factor

    from commkit.math import rms as _rms

    ref_samples = samples

    norm_vec = _rms(ref_samples, axis=-1) * (sps**0.5)
    if samples.ndim == 1:
        input_norm_factor = float(norm_vec)
        samples = samples / float(norm_vec)
    else:
        input_norm_factor = to_device(norm_vec, "cpu")
        # Broadcast (C,) divisor over last axis
        samples = samples / norm_vec[..., None]

    return samples, training_symbols, input_norm_factor


def _build_padded_samples(
    samples_np: Any,
    pad_left: int,
    pad_right: int,
    samples_prefix: Any,
    pad_mode: str,
    eq_norm: Any,
    sps: int,
) -> Any:
    """Construct the padded input array for the equalizer.

    When ``samples_prefix`` is supplied its last ``pad_left`` samples replace the
    leading zero-pad, eliminating the warm-start transient.  Otherwise the
    leading edge is filled according to ``pad_mode``.

    Device-generic: runs on whichever device ``samples_np`` lives on (NumPy
    callers see unchanged behavior; ``apply_taps`` can pad GPU-resident
    signals without a host round trip).
    """
    samples_arr, xp, _ = dispatch(samples_np)
    if samples_prefix is not None:
        # Normalize prefix by the same factor used for the main block.
        prefix = xp.ascontiguousarray(
            to_device(samples_prefix, "cpu" if xp is np else "gpu"),
            dtype=xp.complex64,
        )
        if prefix.ndim == 1:
            prefix = prefix[None, :]
        if prefix.shape[-1] < pad_left:
            raise ValueError(
                f"samples_prefix last axis length {prefix.shape[-1]} is less than "
                f"pad_left={pad_left}. Provide at least pad_left samples."
            )
        if eq_norm is not None:
            nf_arr = np.asarray(to_device(eq_norm, "cpu"), dtype=np.float64).ravel()
            if prefix.shape[0] == 1:
                prefix = prefix / float(nf_arr[0])
            else:
                prefix = prefix / xp.asarray(nf_arr)[:, None]
        left_pad = prefix[:, -pad_left:]
        if samples_arr.ndim == 1:
            samples_2d = samples_arr[None, :]
        else:
            samples_2d = samples_arr
        right_zero = xp.zeros((samples_2d.shape[0], pad_right), dtype=xp.complex64)
        padded = xp.concatenate([left_pad, samples_2d, right_zero], axis=-1)
        return padded
    if pad_mode == "zeros":
        if samples_arr.ndim == 1:
            return xp.pad(samples_arr, (pad_left, pad_right))[None, :]
        return xp.pad(samples_arr, ((0, 0), (pad_left, pad_right)))
    if pad_mode in ("edge", "reflect"):
        if samples_arr.ndim == 1:
            return xp.pad(samples_arr, (pad_left, pad_right), mode=pad_mode)[None, :]
        return xp.pad(samples_arr, ((0, 0), (pad_left, pad_right)), mode=pad_mode)
    raise ValueError(
        f"pad_mode must be 'zeros', 'edge', or 'reflect'. Got {pad_mode!r}."
    )


# Samples per channel from which the cold start estimates the input's
# autocovariance and its output power: a rough spectral shape is enough.
_COLD_START_SAMPLES = 1 << 12


def _cold_start_weights(
    samples: Any, num_taps: int, center_tap: int | None = None
) -> np.ndarray:
    """Build the cold-start butterfly weights: a zero-phase matched filter.

    Each diagonal filter has the zero-phase response ``sqrt(S_i(f))``, the
    square root of the power spectral density of its normalized input
    channel, scaled so that its output has unit mean power, the power of the
    decision constellation.  For a record shaped by a root-Nyquist pulse
    this is the matched filter up to its phase (``sqrt(RC) = RRC``), so the
    first outputs are nearly free of ISI, and it has little gain beyond the
    roll-off, where only noise is.  The output power ``wᴴ R w`` follows from
    the covariance ``R`` of the input over the taps (autocovariance plus the
    squared mean); averaged over the timing phase, it is within ``β/4`` of
    the on-symbol power for a raised-cosine output of roll-off ``β``.

    The PSD is the Fourier transform of the autocovariance over the lags
    ``|m| < num_taps`` under a Bartlett window (non-negative), estimated
    from the first 4096 samples with the mean removed (a unipolar
    constellation would put its power at DC).  At one sample per symbol of
    white symbols the filter is the unit tap.

    Why not the unit centre tap.  The all-sample normalization leaves the
    on-symbol samples of a pulse-shaped record at 2 sps with a gain near
    0.72, and a decision-directed loop that starts below the margin of the
    outer decision regions (0.83 for unipolar PAM-4, 0.86 for 64-QAM) locks
    to a shrunk solution.  A centre tap scaled to unit output power avoids
    that, but lifts the empty band beyond the roll-off too, which the
    adaptation drains only at the rate of the noise eigenvalues (about 1e5
    symbols at a step of 1e-3).  A filter shaped like the PSD itself has no
    such band, but squares the channel's response and slows the blind
    equalizers.

    Falls back to the unit centre tap for a channel whose spectrum or output
    power is not positive, and limits the centre tap to 2.

    Parameters
    ----------
    samples : ndarray
        Normalized samples (mean power ``1/sps``), ``(N,)`` or ``(C, N)``,
        NumPy or CuPy, without padding.  At most the first 4096 samples per
        channel are used.
    num_taps : int
        FIR length T.
    center_tap : int, optional
        Index of the centre tap; defaults to ``num_taps // 2``.

    Returns
    -------
    numpy.ndarray
        complex64 weights, ``(C, C, num_taps)``.
    """
    from .. import _fft

    x, xp, _ = dispatch(samples)
    x2 = x[None, :] if x.ndim == 1 else x
    num_ch, n = x2.shape
    center = center_tap if center_tap is not None else num_taps // 2
    W = _init_butterfly_weights_numpy(num_ch, num_taps, center)
    length = min(n, _COLD_START_SAMPLES)
    if length < num_taps:
        return W
    xs = x2[:, :length].astype(xp.complex64, copy=False)
    mean = xp.mean(xs, axis=-1, keepdims=True)
    # Autocovariance r[m] = E[xz[n+m] xz[n]*] for |m| < T, linear (zero-padded):
    # the only pass over the samples, on their device.  The rest works on
    # (C, 2T - 1) values on the host, after one small copy.
    nfft = 1 << (length + num_taps - 1).bit_length()
    spec = _fft.fft(xs - mean, n=nfft)
    acov = _fft.ifft(spec * xp.conj(spec))
    lags = np.arange(-(num_taps - 1), num_taps)
    small = xp.concatenate([acov[:, xp.asarray(lags % nfft)], mean], axis=-1)
    small_h = np.asarray(to_device(small, "cpu"), dtype=np.complex64)
    r = small_h[:, :-1] / np.float32(length)
    mu2 = np.abs(small_h[:, -1]) ** 2
    # PSD on a grid eight times finer than the taps (Bartlett window
    # 1 - |m|/T), then its square root back to a zero-phase impulse response
    # g[m]; W[i, i, c + m] = g_i[m].  Single precision is ample for a start.
    grid = 1 << (8 * num_taps).bit_length()
    lagged = np.zeros((num_ch, grid), dtype=np.complex64)
    window = (1.0 - np.abs(lags) / num_taps).astype(np.float32)
    lagged[:, lags % grid] = r * window
    psd = np.maximum(_fft.fft(lagged).real, np.float32(0.0))
    response = _fft.ifft(np.sqrt(psd).astype(np.complex64))
    taps = response[:, np.arange(-center, num_taps - center) % grid]
    # Covariance of the regressor, R[t, t'] = r[t - t'] + |μ|², and the
    # output power wᴴ R w of y = Σ_t conj(w_t) x_t.
    diff = np.arange(num_taps)[:, None] - np.arange(num_taps)[None, :]
    for i in range(num_ch):
        cov = r[i, diff + num_taps - 1] + mu2[i]
        power = float(np.real(np.conj(taps[i]) @ cov @ taps[i]))
        g0 = float(taps[i, center].real)
        if not (g0 > 0.0 and power > 0.0):
            continue
        scale = min(1.0 / np.sqrt(power), 2.0 / g0)
        W[i, i] = (taps[i] * np.float32(scale)).astype(np.complex64)
    return W


def _init_butterfly_weights_numpy(
    num_ch: int, num_taps: int, center_tap: int | None = None
) -> np.ndarray:
    """Build the identity butterfly weights: unit center taps, zeros elsewhere.

    ``W[i, i, center_tap] = 1`` for every channel ``i``, so at time 0 the
    equalizer passes each channel straight through.

    Parameters
    ----------
    num_ch : int
        Number of input and output channels C.
    num_taps : int
        FIR length T.
    center_tap : int, optional
        Index of the unit tap; defaults to ``num_taps // 2``.

    Returns
    -------
    numpy.ndarray
        complex64 weights, ``(C, C, num_taps)``.
    """
    W = np.zeros((num_ch, num_ch, num_taps), dtype=np.complex64)
    center = center_tap if center_tap is not None else num_taps // 2
    for i in range(num_ch):
        W[i, i, center] = 1.0 + 0j
    return W


def _validate_w_init(w: np.ndarray, num_ch: int, num_taps: int) -> np.ndarray:
    """Check ``initial_taps`` and return it in the butterfly layout ``(C, C, T)``.

    ``EqualizerResult.weights`` squeezes SISO weights from ``(1, 1, T)`` to
    ``(T,)``, so the weights of one SISO stage passed as ``initial_taps`` to
    the next arrive as ``(T,)`` (or ``(1, T)``); both are accepted.

    Parameters
    ----------
    w : numpy.ndarray
        Candidate initial taps, already on the host.
    num_ch : int
        Expected number of channels C.
    num_taps : int
        Expected number of taps T.

    Returns
    -------
    numpy.ndarray
        ``w`` reshaped to ``(num_ch, num_ch, num_taps)`` where needed.

    Raises
    ------
    ValueError
        If the shape cannot be mapped to ``(num_ch, num_ch, num_taps)``.
    """
    expected = (num_ch, num_ch, num_taps)
    if w.shape == expected:
        return w
    # SISO: accept the squeezed shapes emitted by _unpack_result_*:
    #   (T,)    - both channel dims collapsed  (_unpack: W[0, 0])
    #   (1, T)  - one channel dim collapsed
    if num_ch == 1 and w.shape in ((num_taps,), (1, num_taps)):
        return w.reshape(1, 1, num_taps)
    raise ValueError(
        f"initial_taps shape {tuple(w.shape)} does not match expected "
        f"(num_ch={num_ch}, num_ch={num_ch}, num_taps={num_taps}) = {expected}."
    )


def _prepare_training_numpy(
    training_symbols: Any,
    num_ch: int,
    n_sym: int,
) -> tuple[np.ndarray, int]:
    """Build the training array of the Numba scan kernels.

    NumPy only: the caller moves ``training_symbols`` to the host first
    (``to_device(training_symbols, "cpu")``).

    Parameters
    ----------
    training_symbols : array_like or None
        Training symbols, ``(K,)`` (shared by all channels) or ``(C, K)``.
    num_ch : int
        Number of channels C.
    n_sym : int
        Number of output symbols, the columns of the result.

    Returns
    -------
    train_full : numpy.ndarray
        complex64 training symbols, ``(C, n_train_aligned)``, C-contiguous;
        the kernels read symbol ``idx`` only while ``idx < n_train_aligned``.
    n_train_aligned : int
        Number of data-aided symbols, ``min(K, n_sym)``.
    """
    if training_symbols is None:
        return np.zeros((num_ch, 0), dtype=np.complex64), 0
    train_arr = np.asarray(training_symbols, dtype=np.complex64)
    if train_arr.ndim == 1:
        train_arr = train_arr[None, :]
    n_train_aligned = max(0, min(train_arr.shape[1], n_sym))
    train_full = np.empty((num_ch, n_train_aligned), dtype=np.complex64)
    train_full[:] = train_arr[:, :n_train_aligned]
    return train_full, n_train_aligned


def _unpack_result_numpy(
    y_out: np.ndarray,
    e_out: np.ndarray,
    W_final: np.ndarray,
    w_hist: np.ndarray,
    was_1d: bool,
    store_weights: bool,
    n_sym: int | None = None,
    xp: Any = np,
    num_train_symbols: int = 0,
    input_norm_factor: Any = 1.0,
    phase_trajectory: Any = None,
    state: EqualizerState | None = None,
    tail_trim: int = 0,
) -> EqualizerResult:
    """Convert the Numba kernels' host outputs into an ``EqualizerResult``.

    The outputs are moved to ``xp``, the array module of the input.

    Parameters
    ----------
    y_out : numpy.ndarray
        Equalized symbols, complex64, ``(N_sym, C)``.
    e_out : numpy.ndarray
        Errors, complex64, ``(N_sym, C)``.
    W_final : numpy.ndarray
        Final weights, complex64, ``(C, C, num_taps)``.
    w_hist : numpy.ndarray
        Weight history, ``(N_sym, C, C, num_taps)``, or a ``(1, ...)``
        placeholder without ``store_weights``.
    was_1d : bool
        Squeeze the channel axes for SISO input.
    store_weights : bool
        Return ``w_hist`` as ``weights_history``; otherwise it is ``None``.
    n_sym : int, optional
        Number of symbols to keep; by default all.
    xp : module, default numpy
        Array module of the outputs (``numpy`` or ``cupy``).
    num_train_symbols : int, default 0
        Training symbols used, stored in the result.
    input_norm_factor : float or numpy.ndarray, default 1.0
        Input normalization factor, stored in the result.
    phase_trajectory : ndarray, optional
        Phase of the inline CPR, already on ``xp`` and in the output layout.
    state : EqualizerState, optional
        Continuation state, stored in the result.
    tail_trim : int, default 0
        Symbols trimmed from the tail (RLS), stored in the result.

    Returns
    -------
    EqualizerResult
    """
    y_hat = xp.asarray(y_out.T)  # (N_sym, C) -> (C, N_sym)
    errors = xp.asarray(e_out.T)
    W = xp.asarray(W_final)

    if n_sym is not None:
        y_hat = y_hat[..., :n_sym]
        errors = errors[..., :n_sym]

    if was_1d:
        y_hat, errors = restore_1d(was_1d, y_hat, errors)
        W = W[0, 0]

    w_history = None
    if store_weights:
        w_history = xp.asarray(w_hist if n_sym is None else w_hist[:n_sym])
        if was_1d:
            w_history = w_history[:, 0, 0, :]

    return EqualizerResult(
        y_hat=y_hat,
        weights=W,
        error=errors,
        weights_history=w_history,
        num_train_symbols=num_train_symbols,
        input_norm_factor=input_norm_factor,
        phase_trajectory=phase_trajectory,
        state=state,
        tail_trim=tail_trim,
    )


def _cpr_symmetry(constellation: Any) -> int:
    """Rotational symmetry used by the inline CPR.

    The BPS searches ``[0, 2π/symmetry)`` and cycle slips are multiples of
    ``2π/symmetry``.

    The constellation's own symmetry, as in ``recovery.BPS``; 4 without one.
    """
    if constellation is None:
        return 4
    return int(constellation.rotational_symmetry)


def _validate_sps(sps: Any, num_taps: int) -> None:
    """Validate sps; warn about unusual values, check tap count minimum."""
    sps = require_integer_sps(sps, "equalizer")
    if sps == 1:
        logger.info(
            "sps=1: symbol-spaced equalizer mode. "
            "No fractional-spacing benefit; suitable for residual ISI correction "
            "after a prior FSE stage + FOE + CPR."
        )
    elif sps > 2:
        logger.warning(
            "sps=%s: non-standard oversampling ratio for adaptive "
            "equalization. T/2-spaced (sps=2) is the industry standard. "
            "Higher values give marginal benefit but require "
            "proportionally more taps.",
            sps,
        )
    if sps > 1 and num_taps < 2 * sps:  # fractionally spaced only
        logger.warning(
            "num_taps=%s is small for sps=%s. Recommend num_taps >= %s "
            "for fractionally-spaced equalization.",
            num_taps,
            sps,
            4 * sps + 1,
        )


def _godard_radius(constellation: Any) -> float:
    """Godard dispersion radius ``R2 = E[|c|^4] / E[|c|^2]``.

    Over the constellation's prior (pmf-weighted for a shaped one); 1, the
    unit circle, without a constellation.
    """
    if constellation is None:
        return 1.0
    pts = np.asarray(constellation.points)
    if constellation.pmf is None:
        return float(np.mean(np.abs(pts) ** 4) / np.mean(np.abs(pts) ** 2))
    pmf = np.asarray(constellation.pmf, dtype=np.float64)
    e = float(np.dot(pmf, np.abs(pts) ** 2))
    return float(np.dot(pmf, np.abs(pts) ** 4)) / (e**2)


def _rde_ring_radii(constellation: Any) -> np.ndarray:
    """Unique ring radii (float32) of the unit-power constellation."""
    if constellation is None:
        return np.array([1.0], dtype=np.float32)
    raw = np.abs(np.asarray(constellation.points)).astype(np.float32)
    # Round to 6 decimals to merge numerically identical radii.
    return np.unique(np.round(raw, 6))


def _rde_rings(radii: np.ndarray) -> np.ndarray:
    """Ring table of the RDE decision, float32 ``(2, K)``.

    Row 0 holds the squared radii (sorted ascending), row 1 the squared
    midpoint to the next ring, then the largest float32 after the last ring
    (finite, because the kernels compile with fast math).  ``|y|²`` decides
    the ring of the first bound it does not exceed, the nearest radius
    without a square root; a tie at a midpoint goes to the inner ring.
    """
    r = np.asarray(radii, dtype=np.float32)
    rings = np.empty((2, r.size), dtype=np.float32)
    rings[0] = r * r
    mid = 0.5 * (r[:-1].astype(np.float64) + r[1:])
    rings[1, :-1] = mid * mid
    rings[1, -1] = np.finfo(np.float32).max
    return rings
