"""Shared helpers for the equalization package (normalization, padding, weight init, validation)."""

from __future__ import annotations

import numpy as np

from ..backend import dispatch, from_jax, to_device
from ..helpers import restore_1d
from ..logger import logger
from .result import CPRState, EqualizerResult


def _sq_qam_slicer_params(constellation_np):
    """Return (sq_side, lev_min_f32, d_grid_f32) for O(1) square-QAM slicing.

    For a square M-QAM constellation the nearest symbol can be found by
    independently snapping the real and imaginary components to the nearest
    level - no O(M) search required.  Returns sq_side=0 for non-square
    constellations (PSK, PAM, shaped QAM) to indicate the O(M) fallback
    should be used.
    """
    M = len(constellation_np)
    sq_side = int(M**0.5)
    if sq_side * sq_side != M:
        return 0, np.float32(0.0), np.float32(1.0)
    levels = np.unique(np.round(constellation_np.real, 6))
    if len(levels) != sq_side:
        return 0, np.float32(0.0), np.float32(1.0)
    lev_min = np.float32(levels[0])
    d_grid = np.float32(levels[1] - levels[0]) if sq_side > 1 else np.float32(1.0)
    return sq_side, lev_min, d_grid


def _cpr_state_to_jax_inits(state: CPRState, num_ch: int, KB: int, H: int):
    """Extract CPR carry init arrays from a CPRState for JAX warm-start.

    Returns CPU NumPy arrays with correct dtypes/shapes for the JAX carry.
    Missing fields are zero-initialised.  Caller converts to JAX arrays via
    ``to_jax(..., device=platform)``.

    Returns (pll_phi, pll_freq, bps_buf, bps_buf_ptr, bps_prev4,
             cs_buf_x, cs_buf_y, cs_buf_ptr) - all NumPy.
    """

    def _get(val, shape, dtype):
        return (
            np.asarray(val, dtype=dtype)
            if val is not None
            else np.zeros(shape, dtype=dtype)
        )

    pll_phi = _get(state.pll_phi, (num_ch,), np.float64)
    pll_freq = _get(state.pll_freq, (num_ch,), np.float64)
    bps_buf = _get(state.jax_bps_buf, (KB, num_ch), np.complex64)
    bps_buf_ptr = np.int32(
        state.jax_bps_buf_ptr if state.jax_bps_buf_ptr is not None else 0
    )
    bps_prev4 = _get(state.bps_prev4, (num_ch,), np.float64)
    cs_buf_x = _get(state.cs_buf_x, (num_ch, H), np.float64)
    cs_buf_y = _get(state.cs_buf_y, (num_ch, H), np.float64)
    cs_buf_ptr = _get(state.cs_buf_ptr, (num_ch,), np.int32)
    return (
        pll_phi,
        pll_freq,
        bps_buf,
        bps_buf_ptr,
        bps_prev4,
        cs_buf_x,
        cs_buf_y,
        cs_buf_ptr,
    )


# -----------------------------------------------------------------------------
# SHARED HELPERS
# -----------------------------------------------------------------------------


def _normalize_inputs(samples, training_symbols, sps, input_norm_factor=None):
    """Scale samples and training symbols to a common unit symbol-power reference.

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
    training_symbols : unit average-power, same shape/backend (or None)
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
            nf_arr = np.asarray(nf, dtype=np.float64).ravel()
            if nf_arr.shape[0] != samples.shape[0]:
                raise ValueError(
                    f"input_norm_factor shape {nf_arr.shape} does not match "
                    f"samples channel count {samples.shape[0]}."
                )
            _, xp_loc, _ = dispatch(samples)
            nf_dev = xp_loc.asarray(nf_arr)[..., None]  # (C, 1) on same device
            samples = samples / nf_dev
        if training_symbols is not None:
            from commkit.helpers import normalize as c_normalize

            training_symbols = c_normalize(training_symbols, "average_power", axis=-1)
        return samples, training_symbols, input_norm_factor

    from commkit.helpers import rms as _rms

    ref_samples = samples

    norm_vec = _rms(ref_samples, axis=-1) * (sps**0.5)
    if samples.ndim == 1:
        input_norm_factor = float(norm_vec)
        samples = samples / float(norm_vec)
    else:
        input_norm_factor = to_device(norm_vec, "cpu")
        # Broadcast (C,) divisor over last axis
        samples = samples / norm_vec[..., None]

    if training_symbols is not None:
        from commkit.helpers import normalize as c_normalize

        # Training symbols are at 1 sps; "average_power" == "symbol_power" at sps=1.
        training_symbols = c_normalize(training_symbols, "average_power", axis=-1)

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


def _init_butterfly_weights_jax(num_ch, num_taps, jnp, center_tap=None):
    """Build center-tap identity butterfly weight matrix as a JAX array.

    Initializes a ``(C, C, num_taps)`` complex64 array where
    ``W[i, i, center] = 1+0j`` for each channel ``i`` and all other entries
    are zero.  This is the canonical identity starting point: at time 0 the
    equalizer passes each channel straight through with unit gain and zero
    delay relative to the center tap.

    Parameters
    ----------
    num_ch     : int - number of input/output channels C
    num_taps   : int - FIR filter length T
    jnp        : JAX numpy module (passed as argument to avoid importing at
                 module level when JAX is unavailable)
    center_tap : int or None - tap index for unit initialization;
                 defaults to ``num_taps // 2``

    Returns
    -------
    W : (C, C, num_taps) complex64 JAX array
    """
    W = jnp.zeros((num_ch, num_ch, num_taps), dtype="complex64")
    center = center_tap if center_tap is not None else num_taps // 2
    W = W.at[jnp.arange(num_ch), jnp.arange(num_ch), center].set(1.0 + 0j)
    return W


def _init_butterfly_weights_numpy(num_ch, num_taps, center_tap=None):
    """Build center-tap identity butterfly weight matrix as a NumPy array.

    NumPy counterpart of ``_init_butterfly_weights_jax`` for use with the
    Numba backend.  Same semantics and output shape; no JAX dependency.

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
    """Validate w_init shape and return it in butterfly layout ``(C, C, T)``.

    The library's unpack helpers squeeze SISO weights from ``(1, 1, T)`` to
    ``(T,)`` in ``EqualizerResult.weights`` for user convenience.  This means
    a weight array produced by one SISO equalizer stage and passed as ``w_init``
    to the next stage arrives here as ``(T,)``; that shape must be accepted.

    Parameters
    ----------
    w        : np.ndarray - candidate w_init array (already cast to NumPy)
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
        f"w_init shape {tuple(w.shape)} does not match expected "
        f"(num_ch={num_ch}, num_ch={num_ch}, num_taps={num_taps}) = {expected}."
    )


def _prepare_training_jax(
    training_symbols,
    num_ch,
    n_sym,
):
    """Build the zero-padded training array expected by the JAX scan kernels.

    The kernels index ``training_padded[:, sym_idx]`` at every symbol,
    conditioned on ``sym_idx < n_train``.  Symbols beyond ``n_train_aligned``
    are zero - the kernel ignores them (DD slicer is used instead).

    If ``training_symbols`` is 1-D it is broadcast to all ``num_ch`` channels.
    Any extra training symbols beyond ``n_sym`` are silently clamped.
    The array is kept on the same device as ``training_symbols`` to avoid
    unnecessary CPU round-trips before the ``to_jax()`` transfer.

    Parameters
    ----------
    training_symbols : array or None - (K,) or (C, K), any backend
    num_ch           : int - C
    n_sym            : int - padded symbol count (columns of output array)

    Returns
    -------
    train_full      : (C, n_sym) complex64 on same backend as input (or NumPy)
    n_train_aligned : int - effective number of data-aided symbols
    """
    if training_symbols is not None:
        # Keep training data on its original device
        train_arr, xp, _ = dispatch(training_symbols)
        train_arr = train_arr.astype("complex64")
        if train_arr.ndim == 1:
            train_arr = (
                xp.tile(train_arr[None, :], (num_ch, 1))
                if num_ch > 1
                else train_arr[None, :]
            )
        n_raw = train_arr.shape[1]
        n_train_aligned = max(0, min(n_raw, n_sym))

        train_full = xp.zeros((num_ch, n_sym), dtype="complex64")
        if n_train_aligned > 0:
            train_full[:, :n_train_aligned] = train_arr[:, :n_train_aligned]
    else:
        n_train_aligned = 0
        train_full = np.zeros((num_ch, n_sym), dtype="complex64")

    return train_full, n_train_aligned


def _prepare_training_numpy(
    training_symbols,
    num_ch,
    n_sym,
):
    """Build the zero-padded training array for the Numba scan kernels.

    Pure NumPy implementation - no JAX, CuPy, or ``dispatch`` dependencies.
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


def _unpack_result_jax(
    y_hat_jax,
    errors_jax,
    W_final_jax,
    w_hist_jax,
    was_1d,
    store_weights,
    n_sym=None,
    xp=np,
    num_train_symbols=0,
    input_norm_factor=1.0,
):
    """Convert JAX scan outputs into an ``EqualizerResult``.

    Transfers JAX device arrays back to NumPy/CuPy via ``from_jax``, then:
      1. Transposes from ``(N_sym, C)`` scan layout to ``(C, N_sym)`` convention.
      2. Truncates to ``n_sym`` when provided (e.g. RLS early-halt boundary).
      3. Squeezes the channel dimension for 1-D SISO inputs (``was_1d=True``).
      4. Optionally keeps the weight-trajectory array.

    Parameters
    ----------
    y_hat_jax       : (N_sym, C) JAX array - equalized symbols
    errors_jax      : (N_sym, C) JAX array - complex errors
    W_final_jax     : (C, C, num_taps) JAX array - final weights
    w_hist_jax      : (N_sym, C, C, num_taps) JAX array - weight history
    was_1d          : bool - squeeze C=1 dimension for SISO inputs
    store_weights   : bool - if False, ``weights_history`` is None
    n_sym           : int or None - truncation length (None = no truncation)
    xp              : output array module (np or cp)
    num_train_symbols: int - stored in the result for caller reference

    Returns
    -------
    EqualizerResult
    """
    # ``from_jax`` follows the JAX *compute* device (CuPy if the scan ran on
    # GPU, NumPy on CPU), which may differ from the input's module ``xp`` - e.g.
    # NumPy input with ``device='gpu'``.  Coerce to the input's device so the
    # result honours the "output on the input's device" convention (same
    # pattern as the Point 8 CPR-state unpacking).
    _tgt = "cpu" if xp is np else "gpu"
    y_hat = xp.asarray(to_device(from_jax(y_hat_jax), _tgt).T)  # (N,C) -> (C,N)
    errors = xp.asarray(to_device(from_jax(errors_jax), _tgt).T)
    W_final = xp.asarray(to_device(from_jax(W_final_jax), _tgt))

    if n_sym is not None:
        y_hat = y_hat[..., :n_sym]
        errors = errors[..., :n_sym]

    if was_1d:
        y_hat, errors = restore_1d(was_1d, y_hat, errors)
        W_final = W_final[0, 0]

    w_history = None
    if store_weights:
        w_history = xp.asarray(to_device(from_jax(w_hist_jax), _tgt))
        if was_1d:
            w_history = w_history[:, 0, 0, :]

    return EqualizerResult(
        y_hat=y_hat,
        weights=W_final,
        error=errors,
        weights_history=w_history,
        num_train_symbols=num_train_symbols,
        input_norm_factor=input_norm_factor,
    )


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

    No ``from_jax`` calls - all inputs are already NumPy arrays produced by
    the Numba kernels.  Same post-processing as ``_unpack_result_jax`` but
    operates directly on NumPy memory without any device transfer overhead.

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


def _cpr_symmetry(modulation: str | None, order: int | None) -> int:
    """Return the rotational symmetry order used for cycle-slip correction.

    QAM and most practical CPR algorithms exploit 4-fold (π/2) symmetry.
    BPSK is the only exception (2-fold).

    Parameters
    ----------
    modulation : str or None
    order : int or None

    Returns
    -------
    int - 4 (default/QAM/PSK M≥4) or 2 (BPSK/PAM)
    """
    if modulation is None:
        return 4
    m = modulation.lower().strip()
    if m in ("pam",):
        return 2
    if m in ("psk", "bpsk") and order == 2:
        return 2
    return 4


def _validate_sps(sps, num_taps):
    """Validate sps; warn about unusual values, check tap count minimum."""
    if sps < 1:
        raise ValueError(f"sps must be >= 1. Got sps={sps}.")
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


def _godard_radius(modulation, order, unipolar, pmf):
    """Godard dispersion radius R2 and PS-QAM pilot scale (mirrors ``cma``)."""
    if modulation is None or order is None:
        return 1.0, None
    from ..mapping import Constellation

    c = Constellation.gray(modulation, order, unipolar=unipolar, pmf=pmf)
    const = c.points
    if pmf is not None:
        _pmf = np.asarray(pmf, dtype=np.float64)
        _e_ps = c.power()  # Σ P(s_m)|s_m|² - single source of truth for E_PS
        r2 = float(np.dot(_pmf, np.abs(const) ** 4)) / (_e_ps**2)
        c_ps = np.float32(1.0 / np.sqrt(_e_ps)) if _e_ps < 1.0 - 1e-6 else None
        return r2, c_ps
    r2 = float(np.mean(np.abs(const) ** 4) / np.mean(np.abs(const) ** 2))
    return r2, None


def _rde_ring_radii(modulation, order, unipolar, pmf):
    """Unique constellation ring radii and PS-QAM pilot scale (mirrors ``rde``)."""
    if modulation is None or order is None:
        return np.array([1.0], dtype=np.float32), None
    from ..mapping import Constellation

    c = Constellation.gray(modulation, order, unipolar=unipolar, pmf=pmf)
    raw = np.abs(c.points).astype(np.float32)
    c_ps = None
    if pmf is not None:
        _e_ps = c.power()  # Σ P(s_m)|s_m|² - single source of truth for E_PS
        if _e_ps < 1.0 - 1e-6:
            c_ps = np.float32(1.0 / np.sqrt(_e_ps))
            raw = (raw * c_ps).astype(np.float32)
    return np.unique(np.round(raw, 6)), c_ps
