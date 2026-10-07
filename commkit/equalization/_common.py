"""Shared helpers for the equalization package (normalization, padding, weight init, validation)."""

from __future__ import annotations

from typing import Any

import numpy as np

from .._array import restore_1d
from ..backend import dispatch, to_device
from ..core._signal_adapter import require_integer_sps
from ..logger import logger
from .result import EqualizerResult

# -----------------------------------------------------------------------------
# SHARED HELPERS
# -----------------------------------------------------------------------------


def _normalize_inputs(samples, training_symbols, sps, input_norm_factor=None):
    """Scale samples to unit symbol power (training symbols pass through).

    For fractionally-spaced equalization (sps > 1) the fractional timing phase
    is unknown.  Strided power measurement ``samples[..., ::sps]`` is unsafe
    because it can land on zero-crossings of the Nyquist pulse, severely
    underestimating signal power and destabilising adaptation.

    Instead the *wideband* (all-sample) power over the full signal is used.
    This is the most accurate power estimate and is independent of whether
    training symbols are provided.

    Parameters
    ----------
    samples          : (C, N) or (N,)  complex, any backend (NumPy / CuPy)
    training_symbols : (C, K) or (K,)  or None - always at 1 sps
    sps              : int - samples per symbol
    input_norm_factor : float or np.ndarray, optional
        When supplied, skip the RMS computation and divide samples by this
        factor instead.  Pass ``EqualizerResult.input_norm_factor`` from a
        previous call to keep successive blocks on the same power scale.
        ``float`` for SISO, ``(C,)`` array for MIMO.

    Returns
    -------
    samples          : unit symbol-power, same shape/backend
    training_symbols : unchanged (known symbols are on the constellation scale)
    input_norm_factor : float or np.ndarray
        Per-channel normalization factor(s) ``rms(ch) * sqrt(sps)`` applied
        to *samples* before this function returned.  ``float`` for SISO,
        ``np.ndarray`` of shape ``(C,)`` for MIMO.  Stored in
        ``EqualizerResult.input_norm_factor`` so callers can reconstruct
        the physical power scale of a different capture (e.g. vacuum noise)
        passed through the same frozen taps.  See ``EqualizerResult`` docs.
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
    samples_np, pad_left, pad_right, samples_prefix, pad_mode, eq_norm, sps
):
    """Construct the padded input array for the equalizer.

    When ``samples_prefix`` is supplied its last ``pad_left`` samples replace the
    leading zero-pad, eliminating the warm-start transient.  Otherwise the
    leading edge is filled according to ``pad_mode``.

    Backend-generic: runs on whichever device ``samples_np`` lives on (NumPy
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


def _init_butterfly_weights_numpy(num_ch, num_taps, center_tap=None):
    """Build center-tap identity butterfly weight matrix as a NumPy array.

    ``W[i, i, center] = 1+0j`` for each channel ``i``, all other entries zero:
    at time 0 the equalizer passes each channel straight through.

    Parameters
    ----------
    num_ch     : int - number of input/output channels C
    num_taps   : int - FIR filter length T
    center_tap : int or None - tap index for unit initialization;
                 defaults to ``num_taps // 2``

    Returns
    -------
    W : (C, C, num_taps) complex64 NumPy array
    """
    W = np.zeros((num_ch, num_ch, num_taps), dtype=np.complex64)
    center = center_tap if center_tap is not None else num_taps // 2
    for i in range(num_ch):
        W[i, i, center] = 1.0 + 0j
    return W


def _validate_w_init(w: np.ndarray, num_ch: int, num_taps: int) -> np.ndarray:
    """Validate initial_taps shape and return it in butterfly layout ``(C, C, T)``.

    The library's unpack helpers squeeze SISO weights from ``(1, 1, T)`` to
    ``(T,)`` in ``EqualizerResult.weights`` for user convenience.  This means
    a weight array produced by one SISO equalizer stage and passed as ``initial_taps``
    to the next stage arrives here as ``(T,)``; that shape must be accepted.

    Parameters
    ----------
    w        : np.ndarray - candidate initial_taps array (already cast to NumPy)
    num_ch   : int        - expected number of channels C
    num_taps : int        - expected number of FIR taps T

    Returns
    -------
    np.ndarray - w reshaped to ``(num_ch, num_ch, num_taps)`` if needed.

    Raises
    ------
    ValueError if the shape cannot be mapped to ``(num_ch, num_ch, num_taps)``.
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
    training_symbols,
    num_ch,
    n_sym,
):
    """Build the zero-padded training array for the Numba scan kernels.

    Pure NumPy implementation - no CuPy or ``dispatch`` dependencies.
    The caller must ensure ``training_symbols`` is already a NumPy array
    (use ``to_device(training_symbols, "cpu")`` before calling).

    Parameters
    ----------
    training_symbols : (K,) or (C, K) complex64 NumPy array, or None
    num_ch           : int - C
    n_sym            : int - symbol count (columns of output array)

    Returns
    -------
    train_full      : (C, n_sym) complex64 NumPy array
    n_train_aligned : int - effective number of data-aided symbols
    """
    if training_symbols is not None:
        train_arr = np.asarray(training_symbols, dtype=np.complex64)
        if train_arr.ndim == 1:
            train_arr = (
                np.tile(train_arr[None, :], (num_ch, 1))
                if num_ch > 1
                else train_arr[None, :]
            )
        n_raw = train_arr.shape[1]
        n_train_aligned = max(0, min(n_raw, n_sym))

        train_full = np.zeros((num_ch, n_sym), dtype=np.complex64)
        if n_train_aligned > 0:
            train_full[:, :n_train_aligned] = train_arr[:, :n_train_aligned]
    else:
        n_train_aligned = 0
        train_full = np.zeros((num_ch, n_sym), dtype=np.complex64)

    return train_full, n_train_aligned


def _unpack_result_numpy(
    y_out,
    e_out,
    W_final,
    w_hist,
    was_1d,
    store_weights,
    n_sym=None,
    xp=np,
    num_train_symbols=0,
    input_norm_factor=1.0,
):
    """Convert Numba kernel outputs (plain NumPy) into an ``EqualizerResult``.

    All inputs are NumPy arrays produced by the Numba kernels; outputs are
    placed on ``xp`` (the input's array module).

    Parameters
    ----------
    y_out           : (N_sym, C) complex64 NumPy array - equalized symbols
    e_out           : (N_sym, C) complex64 NumPy array - complex errors
    W_final         : (C, C, num_taps) complex64 NumPy array - final weights
    w_hist          : (N_sym or 1, C, C, num_taps) NumPy - weight history
    was_1d          : bool - squeeze C=1 dimension for SISO inputs
    store_weights   : bool - if False, ``weights_history`` is None
    n_sym           : int or None - truncation length (None = no truncation)
    xp              : output array module (np or cp)
    num_train_symbols: int - stored in result for caller reference

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
        w_history = xp.asarray(w_hist)
        if was_1d:
            w_history = w_history[:, 0, 0, :]

    return EqualizerResult(
        y_hat=y_hat,
        weights=W,
        error=errors,
        weights_history=w_history,
        num_train_symbols=num_train_symbols,
        input_norm_factor=input_norm_factor,
    )


def _cpr_symmetry(constellation: Any) -> int:
    """Rotational symmetry of the inline CPR: the BPS searches
    ``[0, 2π/symmetry)`` and cycle slips are multiples of ``2π/symmetry``.

    The constellation's own symmetry, as in ``recovery.BPS``; 4 without one.
    """
    if constellation is None:
        return 4
    return int(constellation.rotational_symmetry)


def _validate_sps(sps, num_taps):
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
    if num_taps < 2 * sps:
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
