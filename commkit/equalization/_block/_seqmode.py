"""Block-execution-mode backend for sequential.py's lms/cma/rde (update_mode='block').

Shared by the "block" update mode of the time-domain sequential equalizers
(``lms``, ``cma``, ``rde`` in the ``sequential`` subpackage) - not to be
confused with ``block_lms`` (the standalone frequency-domain FDAF engine in
``_dd.py``) or the FDAF blind engine in ``_blind.py``, neither of which use
any of the functions here.
"""

from __future__ import annotations

import numpy as np

from ...backend import _get_jax, dispatch, to_device, to_jax
from .._common import (
    _build_padded_samples,
    _init_butterfly_weights_numpy,
    _normalize_inputs,
    _unpack_result_jax,
    _unpack_result_numpy,
    _validate_w_init,
)
from .._kernels_jax import _get_jax_cma_block, _get_jax_lms_block, _get_jax_rde_block
from ..result import _log_equalizer_exit


def _block_eq_xp(
    kind,
    x_padded,
    W,
    mu,
    D,
    stride,
    *,
    constellation=None,
    n_train=0,
    training=None,
    r2=1.0,
    radii=None,
    pref=None,
    pmask=None,
    sq_side=0,
    sq_lev_min=0.0,
    sq_d_grid=1.0,
):
    """Array-native (NumPy/CuPy) block-update butterfly equalizer.

    A plain Python loop over ``ceil(n_sym/D)`` chunks: the per-chunk work is
    matmul-sized so interpreter overhead is amortised, and the same code runs
    on NumPy (CPU) and CuPy (GPU) via the array module of ``x_padded``.  The
    forward/gradient einsums promote to ``complex128`` per CLAUDE.md, with
    ``complex64`` weight storage.

    Parameters mirror the per-symbol kernels; ``kind`` is ``'lms'``/``'cma'``/
    ``'rde'`` and pilots (``pref``/``pmask``) invert the error to ``y - pref``.

    Returns ``(y_out (n_sym, C), e_out (n_sym, C), W)`` on the input's backend.
    """
    x, xp, _ = dispatch(x_padded)
    num_ch = W.shape[0]
    num_taps = W.shape[2]
    n_pad_samples = x.shape[1]
    n_sym = (n_pad_samples - num_taps) // stride + 1

    y_out = xp.empty((n_sym, num_ch), dtype=xp.complex64)
    e_out = xp.empty((n_sym, num_ch), dtype=xp.complex64)

    has_pilots = pref is not None and pmask is not None
    if radii is not None:
        radii = xp.asarray(radii).astype(xp.float64)
    if constellation is not None:
        constellation = xp.asarray(constellation).astype(xp.complex64)

    for start in range(0, n_sym, D):
        stop = min(start + D, n_sym)
        d = stop - start  # actual chunk length (D, or shorter for the tail)
        # Window gather for this chunk -> (d, C, T)
        base = (start + xp.arange(d))[:, None] * stride + xp.arange(num_taps)[None, :]
        X_chunk = xp.transpose(x[:, base], (1, 0, 2))  # (d, C, T)
        X64 = X_chunk.astype(xp.complex128)
        Y = xp.einsum("ijt,djt->di", xp.conj(W).astype(xp.complex128), X64)  # (d, C)
        Y = Y.astype(xp.complex64)

        if kind == "lms":
            dd = _slice_block_xp(Y, xp, constellation, sq_side, sq_lev_min, sq_d_grid)
            sym_idx = start + xp.arange(d)
            use_train = (sym_idx < n_train)[:, None]
            tr = training[:, start:stop].T if training is not None else xp.zeros_like(Y)
            dsym = xp.where(use_train, tr, dd)
            E = Y - dsym
        elif kind == "cma":
            E = Y * (xp.real(Y * xp.conj(Y)) - xp.float32(r2))
        elif kind == "rde":
            abs_y2 = xp.real(Y * xp.conj(Y))
            abs_y = xp.sqrt(abs_y2)
            rd = radii[
                xp.argmin(xp.abs(abs_y[..., None] - radii[None, None, :]), axis=-1)
            ]
            E = Y * (abs_y2 - rd.astype(xp.float32) ** 2)
        else:
            raise ValueError(f"unknown block kind {kind!r}")

        if has_pilots:
            pm = pmask[start:stop].astype(bool)[:, None]
            pr = pref[:, start:stop].T
            E = xp.where(pm, Y - pr, E)

        grad = xp.einsum(
            "di,djt->ijt", xp.conj(E).astype(xp.complex128), X64
        )  # (C, C, T)
        W = (W - (xp.complex128(mu) * grad)).astype(xp.complex64)

        y_out[start:stop] = Y
        e_out[start:stop] = E

    return y_out, e_out, W


def _slice_block_xp(Y, xp, constellation, sq_side, sq_lev_min, sq_d_grid):
    """Vectorised nearest-constellation slicer for the array-native block path."""
    if sq_side > 0:
        ir = xp.clip(
            xp.round((Y.real - sq_lev_min) / sq_d_grid).astype(xp.int32),
            0,
            sq_side - 1,
        )
        ii = xp.clip(
            xp.round((Y.imag - sq_lev_min) / sq_d_grid).astype(xp.int32),
            0,
            sq_side - 1,
        )
        nr = sq_lev_min + ir.astype(xp.float32) * xp.float32(sq_d_grid)
        ni = sq_lev_min + ii.astype(xp.float32) * xp.float32(sq_d_grid)
        return (nr + 1j * ni).astype(xp.complex64)
    d2 = xp.abs(Y[..., None] - constellation) ** 2  # (d, C, M)
    return constellation[xp.argmin(d2, axis=-1)]


def _validate_block_mode(
    update_mode, block_len, backend, *, cpr_type=None, store_weights=False
):
    """Validate ``update_mode``/``block_len`` and the block-mode constraints.

    No-op for ``update_mode='sequential'``.  For ``'block'`` it enforces the
    following: ``backend in {'jax', 'xp'}`` (``'numba'`` is a pointless
    combination), a positive ``block_len``, and no ``cpr_type``/``store_weights``
    (unsupported in v1).
    """
    if update_mode not in ("sequential", "block"):
        raise ValueError(
            f"update_mode must be 'sequential' or 'block'. Got {update_mode!r}."
        )
    if update_mode == "sequential":
        return
    if block_len < 1:
        raise ValueError(f"block_len must be >= 1. Got {block_len}.")
    if backend not in ("jax", "xp"):
        raise ValueError(
            "update_mode='block' requires backend='jax' (chunked scan) or "
            f"backend='xp' (array-native NumPy/CuPy). Got backend={backend!r}; "
            "backend='numba' is a pointless combination for block updates."
        )
    if cpr_type is not None:
        raise ValueError(
            "cpr_type is not supported with update_mode='block' (v1). Run CPR "
            "as a separate stage, or use update_mode='sequential'."
        )
    if store_weights:
        raise ValueError(
            "store_weights is not supported with update_mode='block' (v1)."
        )


def _resolve_jax_platform(x_jax, device):
    """Resolve the JAX placement platform string for a transferred input."""
    try:
        if device is not None:
            return device.lower()
        if hasattr(x_jax, "device"):
            return x_jax.device.platform
        return list(x_jax.devices())[0].platform
    except Exception:
        return "cpu"


def _build_slicer_constellation(modulation, order, unipolar, training_np, pmf):
    """Build the NumPy slicer constellation for the DD/block paths.

    Mirrors the per-symbol kernels: a Gray constellation when ``modulation``
    and ``order`` are given (PS-QAM scaled to unit power when ``pmf`` is
    supplied), otherwise the unique rounded training symbols.
    """
    if modulation is not None and order is not None:
        from ...mapping import constellation_power, gray_constellation

        reference_constellation = gray_constellation(
            modulation, order, unipolar=unipolar
        )
        constellation_np = (
            to_device(reference_constellation, "cpu").flatten().astype(np.complex64)
        )
    elif training_np is not None:
        constellation_np = np.unique(np.round(training_np.reshape(-1), decimals=8))
    else:
        raise ValueError("modulation and order must be provided for DD mode.")

    if pmf is not None and modulation is not None and order is not None:
        _e_ps = constellation_power(constellation_np, pmf)
        if _e_ps < 1.0 - 1e-6:
            constellation_np = (
                constellation_np * np.float32(1.0 / np.sqrt(_e_ps))
            ).astype(np.complex64)
    return constellation_np


def _prep_blind_block_inputs(
    samples,
    *,
    sps,
    stride,
    pad_left,
    pad_right,
    samples_prefix,
    pad_mode,
    input_norm_factor,
    use_pilots,
    pilot_gain_db,
    pilot_mask,
    pilot_ref,
    c_ps,
    num_ch,
    num_taps,
    center_tap,
    w_init,
):
    """Normalise/pad inputs and build pilot arrays for blind block CMA/RDE.

    Mirrors the per-symbol ``cma``/``rde`` numba prep: pilot deboost before the
    global RMS normalisation, overlap padding, initial butterfly weights, and
    the dense pilot reference/mask (PS-QAM-scaled).  Returns NumPy arrays for
    the backend-agnostic block dispatch.
    """
    samples_np = np.ascontiguousarray(to_device(samples, "cpu"), dtype=np.complex64)
    if use_pilots and pilot_gain_db != 0.0:
        _amp = np.float32(10.0 ** (pilot_gain_db / 20.0))
        _smask = np.repeat(pilot_mask.astype(bool), stride)
        samples_np[..., _smask] /= _amp
    samples_np, _, eq_norm = _normalize_inputs(
        samples_np, None, sps, input_norm_factor=input_norm_factor
    )
    samples_padded_np = _build_padded_samples(
        samples_np, pad_left, pad_right, samples_prefix, pad_mode, eq_norm, sps
    )
    if w_init is not None:
        w_arr = _validate_w_init(
            np.ascontiguousarray(to_device(w_init, "cpu"), dtype=np.complex64),
            num_ch,
            num_taps,
        )
    else:
        w_arr = _init_butterfly_weights_numpy(num_ch, num_taps, center_tap=center_tap)
    pref_np = pmask_np = None
    if use_pilots:
        pref_np = np.ascontiguousarray(to_device(pilot_ref, "cpu"), dtype=np.complex64)
        if c_ps is not None:
            pref_np = (pref_np * c_ps).astype(np.complex64)
        pmask_np = np.ascontiguousarray(pilot_mask, dtype=np.uint8)
    return samples_padded_np, w_arr, eq_norm, pref_np, pmask_np


def _run_block_equalizer(
    kind,
    *,
    samples_padded_np,
    w_arr,
    num_ch,
    num_taps,
    n_sym,
    stride,
    block_len,
    step_size,
    backend,
    device,
    was_1d,
    xp,
    eq_norm,
    name,
    debug_plot=False,
    plot_smoothing=50,
    check_convergence=False,
    constellation_np=None,
    train_full=None,
    n_train_aligned=0,
    sq_side=0,
    sq_lev_min=0.0,
    sq_d_grid=1.0,
    r2=1.0,
    radii_np=None,
    pref_np=None,
    pmask_np=None,
):
    """Backend dispatch shared by the ``update_mode='block'`` path of the
    time-domain equalizers (``lms``/``cma``/``rde``).

    ``backend='xp'`` runs the array-native loop on the input's module
    (NumPy/CuPy); ``backend='jax'`` runs the chunked ``lax.scan``.  Returns a
    finalised ``EqualizerResult`` (via ``_log_equalizer_exit``).
    """
    D = int(block_len)
    has_pilots = pref_np is not None and pmask_np is not None

    if backend == "xp":
        x_pad = xp.asarray(samples_padded_np)
        W0 = xp.asarray(w_arr)
        kwargs = {}
        if kind == "lms":
            kwargs.update(
                constellation=xp.asarray(constellation_np),
                n_train=int(n_train_aligned),
                training=xp.asarray(train_full),
                sq_side=int(sq_side),
                sq_lev_min=float(sq_lev_min),
                sq_d_grid=float(sq_d_grid),
            )
        elif kind == "cma":
            kwargs.update(r2=float(r2))
        elif kind == "rde":
            kwargs.update(radii=xp.asarray(radii_np))
        if has_pilots:
            kwargs.update(pref=xp.asarray(pref_np), pmask=xp.asarray(pmask_np))
        y_out, e_out, W_final = _block_eq_xp(
            kind, x_pad, W0, float(step_size), D, stride, **kwargs
        )
        result = _unpack_result_numpy(
            y_out,
            e_out,
            W_final,
            np.empty((1, num_ch, num_ch, num_taps), dtype=np.complex64),
            was_1d,
            False,
            n_sym=None,
            xp=xp,
            num_train_symbols=int(n_train_aligned),
            input_norm_factor=eq_norm,
        )
        return _log_equalizer_exit(
            result,
            name=name,
            debug_plot=debug_plot,
            check_convergence=check_convergence,
            plot_smoothing=plot_smoothing,
        )

    # backend == "jax"
    jax, jnp, _ = _get_jax()
    if jax is None or jnp is None:
        raise ImportError("JAX is required for backend='jax'.")
    x_jax = to_jax(samples_padded_np, device=device)  # (C, N_pad)
    platform = _resolve_jax_platform(x_jax, device)
    W_jax = to_jax(w_arr, device=platform)
    mu_jax = to_jax(jnp.float32(step_size), device=platform)

    if kind == "lms":
        const_jax = to_jax(constellation_np, device=platform)
        train_jax = to_jax(train_full, device=platform)
        n_train_jax = to_jax(jnp.int32(n_train_aligned), device=platform)
        run = _get_jax_lms_block(
            num_taps,
            stride,
            len(constellation_np),
            num_ch,
            n_sym,
            D,
            int(sq_side),
            float(sq_lev_min),
            float(sq_d_grid),
        )
        y_jax, e_jax, W_out, _ = run(
            x_jax, train_jax, const_jax, W_jax, mu_jax, n_train_jax
        )
    else:
        if has_pilots:
            pref_jax = to_jax(pref_np, device=platform)  # (C, n_sym)
            pmask_jax = to_jax(pmask_np.astype(bool), device=platform)  # (n_sym,)
        else:
            pref_jax = to_jax(np.zeros((num_ch, n_sym), np.complex64), device=platform)
            pmask_jax = to_jax(np.zeros((n_sym,), bool), device=platform)
        if kind == "cma":
            r2_jax = to_jax(jnp.float32(r2), device=platform)
            run = _get_jax_cma_block(num_taps, stride, num_ch, n_sym, D, has_pilots)
            y_jax, e_jax, W_out, _ = run(
                x_jax, W_jax, mu_jax, r2_jax, pref_jax, pmask_jax
            )
        else:  # rde
            radii_jax = to_jax(np.asarray(radii_np, np.float32), device=platform)
            run = _get_jax_rde_block(
                num_taps, stride, len(radii_np), num_ch, n_sym, D, has_pilots
            )
            y_jax, e_jax, W_out, _ = run(
                x_jax, W_jax, mu_jax, radii_jax, pref_jax, pmask_jax
            )

    result = _unpack_result_jax(
        y_jax,
        e_jax,
        W_out,
        None,
        was_1d,
        False,
        n_sym=None,
        xp=xp,
        num_train_symbols=int(n_train_aligned),
        input_norm_factor=eq_norm,
    )
    return _log_equalizer_exit(
        result,
        name=name,
        debug_plot=debug_plot,
        check_convergence=check_convergence,
        plot_smoothing=plot_smoothing,
    )
