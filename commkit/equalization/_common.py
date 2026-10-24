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
from .result import EqualizerResult

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
    """Build the zero-padded training array of the Numba scan kernels.

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
        complex64 training symbols followed by zeros, ``(C, n_sym)``.
    n_train_aligned : int
        Number of data-aided symbols, ``min(K, n_sym)``.
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
