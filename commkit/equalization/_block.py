"""Block / frequency-domain equalizer engine (block_lms, FDAF)."""

from __future__ import annotations

import contextlib
from typing import Any

import numpy as np

from ..backend import ArrayType, _get_jax, dispatch, to_device, to_jax
from ..core.signal import Signal
from ..helpers import as_2d, rewrap_signal, unwrap_signal
from ..logger import logger
from ._common import (
    _build_padded_samples,
    _cpr_symmetry,
    _init_butterfly_weights_numpy,
    _normalize_inputs,
    _sq_qam_slicer_params,
    _unpack_result_jax,
    _unpack_result_numpy,
    _validate_sps,
    _validate_w_init,
)
from ._kernels_jax import _get_jax_cma_block, _get_jax_lms_block, _get_jax_rde_block
from ._kernels_numba import _get_numba_cs_block
from .result import CPRState, EqualizerResult, _log_equalizer_exit


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
        from ..mapping import constellation_power, gray_constellation

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


def block_lms(
    samples: ArrayType | Signal,
    training_symbols: ArrayType | None = None,
    num_taps: int = 21,
    sps: int | None = None,
    step_size: float = 2e-4,
    block_size: int = 256,
    modulation: str | None = None,
    order: int | None = None,
    unipolar: bool = False,
    store_weights: bool = False,
    w_init: ArrayType | None = None,
    pmf: Any | None = None,
    cpr_type: str | None = None,
    cpr_bps_test_phases: int = 64,
    cpr_bps_block_size: int = 32,
    cpr_joint_channels: bool = False,
    cpr_cycle_slip_correction: bool = False,
    cpr_cycle_slip_history: int = 100,
    cpr_cycle_slip_threshold: float = np.pi / 4,
    debug_plot: bool = False,
    plot_smoothing: int = 50,
    cpr_state: CPRState | None = None,
    input_norm_factor: float | np.ndarray | None = None,
    samples_prefix: ArrayType | None = None,
    pad_mode: str = "zeros",
    cuda_graph: bool = True,
) -> EqualizerResult:
    """Block LMS equalizer with frequency-domain gradient accumulation.

    Processes the signal in fixed-size blocks of ``block_size`` symbols.
    Within each block the filter is held frozen, all ``block_size`` errors are
    accumulated into a single frequency-domain gradient, and the weights are
    updated once per block.  This amortises the FFT overhead over many symbols,
    making it significantly more efficient than per-symbol LMS on GPU for large
    MIMO configurations (C ≥ 4) or long sequences.

    The primary target is **GPU** via CuPy.  On CPU, per-symbol LMS with the
    Numba backend (``lms(..., backend='numba')``) is typically faster because
    the block-FFT overhead outweighs the gradient-accumulation saving for
    small channel counts.

    ``block_lms`` is the **trained / decision-directed** frequency-domain
    equalizer.  Its blind siblings share the same overlap-save engine but use a
    phase-blind error: :func:`block_cma` (Godard constant-modulus) and
    :func:`block_rde` (ring-directed, for multi-ring QAM).  For a time-domain
    block update with shorter adaptation lag (fast dynamics) use
    ``lms(..., update_mode='block')``.

    Algorithm (per block b)
    -----------------------
    1. **Forward pass** - frequency-domain butterfly filter::

           Y_fd[i] = sum_j conj(H_fd[i,j]) * X_fd[j]

       where ``H_fd = FFT(h, n=F)`` and ``X_fd = FFT(x_block, n=F)``.
       Output symbols are extracted at decimated positions ``y[n] = y_time[n*sps]``.

    2. **BPS phase recovery** (if ``cpr_type='bps'``) - for each symbol in the
       block, averages the min-distance metric over a causal trailing window of
       ``cpr_bps_block_size`` symbols and picks the minimum-metric candidate
       rotation.  This produces one phase estimate ``phi_n`` per symbol
       (not one per block), so ``cpr_bps_block_size`` and ``block_size`` are
       independent parameters: ``block_size`` controls FFT/gradient efficiency
       while ``cpr_bps_block_size`` controls phase noise suppression.  The raw
       ``[0, pi/2)`` argmin is converted to full-range radians by a causal 4-fold
       unwrap, and stored in a float64 accumulator in ``phase_trajectory``.

    3. **Cycle-slip correction** (if ``cpr_cycle_slip_correction=True``) - for
       each symbol of the per-symbol BPS phase tensor ``phi_n`` (shape
       ``(C, B)``) the phase is compared to a linear-regression prediction
       built from a circular buffer of ``cpr_cycle_slip_history`` past
       corrected phases (identical algorithm to ``lms`` with
       ``cpr_type='bps'``).  If ``|phi_n - phi_pred| > cpr_cycle_slip_threshold``
       the nearest ``2*pi/symmetry`` quantum is subtracted and the corrected
       value is stored in the history buffer.  On GPU the detector runs as a
       small sequential CUDA kernel on device-resident buffers (no host
       round-trip); when that kernel is unavailable the ``(C, B)`` float64
       block is transferred device->host, corrected by the Numba/Python
       detector, and written back - one D->H + H->D round-trip per block.

    4. **Error** - training or DD slicer on CPR-corrected output; back-rotated
       to the tap plane using the block-average phase ``phi_b``::

           e_taps[n] = e_clean[n] * exp(+j*phi_b)

    5. **Gradient** - scatter ``e_taps`` to sample positions, then::

           dH_fd[i,j]  = conj(E_fd[i]) * X_fd[j]
           h          += mu * IFFT(dH_fd)[...:T]

       ``mu`` is applied to the **summed** block gradient (all B per-symbol
       contributions).  This sum is exactly what a frozen-weight per-symbol LMS
       would accumulate over the same B symbols, so ``step_size`` is on the
       **same scale as** ``lms``: the same ``mu`` yields the same
       convergence and steady-state MSE (see ``step_size`` below).  Only the
       *stability ceiling* is Bx lower - the operating step that matches
       ``lms`` is unchanged.

    Parameters
    ----------
    samples : array_like or Signal
        Input signal samples.  Shape: ``(N_samples,)`` for SISO or
        ``(C, N_samples)`` for MIMO butterfly equalization.
        Typically at 2 samples/symbol for fractionally-spaced equalization.
        A :class:`Signal` returns an :class:`EqualizerResult` whose ``y_hat``
        is a new :class:`Signal` at the symbol rate (``sampling_rate =
        symbol_rate``); ``sps`` defaults to the signal's ``sps`` when not
        given explicitly.
    training_symbols : array_like, optional
        Known transmitted symbols at 1 SPS.
        Shape: ``(N_train,)`` for SISO or ``(C, N_train)`` for MIMO.
    num_taps : int, default 21
        Number of taps per FIR filter (tap count in samples).
    sps : int, optional, default 2
        Samples per symbol.  ``sps=2`` (T/2-spaced) is the default.  Defaults
        to the signal's ``sps`` for :class:`Signal` input.
    step_size : float, default 2e-4
        LMS step size μ, on the **same scale as** ``lms``.  Use the same
        value you would use for ``lms``: because the block update is the
        summed gradient over all B symbols - exactly what a frozen-weight
        per-symbol LMS accumulates over those symbols - the same μ produces the
        same convergence speed and steady-state MSE, independent of
        ``block_size``.  **Do not** divide by ``block_size``; doing so
        under-adapts the filter by that factor.

        The only ``block_size`` dependence is the *stability ceiling*: because
        the weights are frozen across the block, the maximum stable μ is
        ``2/(B·C·T·P_x)`` - roughly ``block_size`` times lower than
        ``lms``.  Reduce μ below your ``lms`` value **only if** it
        exceeds this ceiling (i.e. the run raises the divergence error); the
        default ``2e-4`` is conservative and safe for ``block_size`` up to a
        few thousand.
    block_size : int, default 256
        Number of output symbols per LMS gradient accumulation block.  Larger
        values increase GPU efficiency but reduce adaptation speed.  Independent
        of the BPS averaging window (see ``cpr_bps_block_size``).
    modulation : str, optional
        Modulation scheme (e.g., ``'qam'``, ``'psk'``).  Required when
        ``training_symbols`` is ``None``.
    order : int, optional
        Modulation order (e.g., 16, 64).
    unipolar : bool, default False
        Unipolar PAM flag.
    store_weights : bool, default False
        If ``True``, stores the weight tensor at every block start in
        ``EqualizerResult.weights_history``.
    w_init : array_like, optional
        Initial tap weights, shape ``(C, C, T)`` or SISO short-hands.
    pmf : array_like, optional
        Probability mass function for PS-QAM constellation scaling.
    cpr_type : {'bps', None}, default None
        Inline carrier phase recovery.  Only ``'bps'`` is supported; PLL is
        not available because its per-symbol PI integration does not fit the
        block gradient model.

        **BPS dual-path design:** within each equalizer block, the wrapped
        float32 phase estimate rotates the pre-CPR symbol ``y_raw`` to
        produce the weight-update error, while the unwrapped float64
        accumulator is written to ``phase_trajectory``.  This separation
        prevents float32 rounding errors from accumulating over long signals.

        **4-fold causal unwrap:** the BPS ``argmin`` is in ``[0, π/2)``.
        A per-symbol causal tracker adds or subtracts ``π/2`` multiples to
        maintain continuity, then scales to full-range radians.
    cpr_bps_test_phases : int, default 64
        Number of BPS candidate angles in ``[0, π/2)``.  32-64 is
        sufficient for ≤ 16-QAM; use 64-128 for 64-QAM.
    cpr_bps_block_size : int, default 32
        Trailing-window length (symbols) summed before the BPS ``argmin``.
        This is evaluated per symbol (not per equalizer block), so it is
        independent of ``block_size``.  Larger values reduce phase-noise
        variance at the cost of increased tracking latency.
        ``cpr_bps_block_size=1`` gives single-symbol BPS.
    cpr_joint_channels : bool, default False
        For MIMO inputs (C > 1): if ``True``, the BPS distance metrics are
        summed across all C channels before ``argmin``, producing one shared
        phase estimate broadcast to all channels.  Reduces estimation
        variance by ~√C for shared-LO transmitters.  When ``False``, each
        channel estimates its phase independently.  Ignored for SISO inputs.
    cpr_cycle_slip_correction : bool, default False
        Enable per-symbol cycle-slip detection and correction using the same
        algorithm as ``lms``: after each BPS block every symbol phase is
        compared to a regression prediction, corrected if a slip is detected,
        and added to the circular history buffer.  On GPU the detector runs
        on-device (custom CUDA kernel); without it each block costs one
        ``(C, B)`` float64 D->H + H->D round-trip through the CPU detector.
    cpr_cycle_slip_history : int, default 100
        Length of the per-symbol phase history buffer used for the linear
        regression predictor.  Same semantics as in ``lms``: one entry
        per symbol, so ``100`` means 100 past corrected symbol phases.
        Ignored when ``cpr_cycle_slip_correction=False``.
    cpr_cycle_slip_threshold : float, default π/4
        Maximum phase step (radians) between adjacent symbols before a
        cycle slip is declared.  Set to half the constellation's angular
        symmetry quantum (``π/4`` for QPSK/QAM).  Ignored when
        ``cpr_cycle_slip_correction=False``.
    debug_plot : bool, default False
        Show a convergence + phase diagnostic plot on exit.
    plot_smoothing : int, default 50
        Moving-average window for the MSE curve in the debug plot.
    cpr_state : CPRState, optional
        Warm-start BPS CPR state from a previous ``block_lms()`` call.
        When provided, the BPS 4-fold unwrap accumulators (``bps_prev4``,
        ``bps_offset4``) and the block-distance history matrix
        (``bps_d2_hist``, shape ``(B, C, K-1)``) are restored from the
        previous block boundary.  This prevents the BPS from re-converging
        its phase estimate at each block boundary, which otherwise causes
        a ~``cpr_bps_block_size``-symbol transient of increased phase error.
        Pass ``None`` (default) to cold-start.  Only BPS state is used;
        PLL/cycle-slip fields are ignored.
    input_norm_factor : float or ndarray, optional
        Pre-computed RMS normalization factor.  See ``lms()`` for the full
        description; behaviour is identical.
    samples_prefix : array_like, optional
        Signal history from the end of the previous block.  See ``lms()``
        for the full description; behaviour is identical.
    pad_mode : {'zeros', 'edge'}, default 'zeros'
        Padding strategy when ``samples_prefix`` is ``None``.  See
        ``lms()`` for the full description; behaviour is identical.
    cuda_graph : bool, default True
        On the GPU (CuPy) backend, capture the per-block compute into a CUDA
        graph and replay it once per block, collapsing the ~30-50 per-block
        kernel launches into a single launch.  This removes the launch-overhead
        floor that dominates small/medium ``block_size`` runs.  Only full blocks
        in the decision-directed region are captured; the training and final
        partial blocks always run eagerly.  Has no effect on CPU, when
        ``store_weights=True``, or when cycle-slip correction is enabled without
        the ``cs_block`` CUDA kernel; capture failures fall back to the eager
        loop with a warning, so output is unaffected either way.  Set ``False``
        to force the eager loop (e.g. for debugging or profiling).

    Returns
    -------
    EqualizerResult
        Same fields as ``lms``, plus:

        * ``input_norm_factor`` - RMS factor used to normalize inputs.
        * ``cpr_state`` - ``CPRState`` with BPS accumulators after the last
          block.  ``None`` when ``cpr_type=None``.

        ``phase_trajectory`` is populated when ``cpr_type='bps'``; shape
        ``(N_sym,)`` SISO or ``(C, N_sym)`` MIMO, one estimate per symbol.

        When ``samples`` is a :class:`Signal`, ``y_hat`` is a new
        :class:`Signal` at the symbol rate (``sampling_rate = symbol_rate``).

    Warnings
    --------
    **GPU throughput - use large block_size:** On GPU (CuPy) each Python
    loop iteration launches ~10-20 CUDA kernels (FFT, einsum, IFFT, BPS
    rotations, ...).  At ``block_size=64`` and 100k symbols that is ~1 500
    blocks x kernel-launch overhead; at ``block_size=2048`` it drops to
    ~49 blocks.  Throughput improves markedly once the cuFFT/cuBLAS work
    per block dominates the Python overhead.  On GPU prefer
    ``block_size`` ≥ 512, ideally 1024-4096.

    **BPS cycle-slip correction (``cpr_cycle_slip_correction=True``):**
    On GPU the detector runs as a sequential CUDA kernel on device-resident
    history buffers, so enabling it adds one extra kernel launch per block
    and no host synchronization.  Only when that kernel is unavailable
    (no custom-kernel support) does each block fall back to a synchronous
    ``(C, block_size)`` float64 device->host round-trip through the CPU
    detector, which serialises the GPU pipeline.

    **CPU (NumPy) backend:** even slower than GPU because the Python loop
    dominates at any practical block size.  Use ``lms(backend='numba')``
    or ``lms(backend='jax')`` for CPU workloads instead.

    **Stability / overflow:** ``step_size`` is applied to the **summed**
    gradient over all ``block_size`` symbols (not averaged).  This keeps μ on
    the same scale as ``lms`` (same μ -> same convergence and steady-state
    MSE), but it also means the *stability ceiling* -
    ``0 < μ < 2/(block_size·C·T·P_x)`` - is roughly ``block_size`` times lower
    than per-symbol LMS, because the weights are frozen across the block.  So
    start from the **same** ``step_size`` you use for ``lms``; if a large
    ``block_size`` pushes that value above the ceiling the run diverges (NaN
    weights, detected at end of run), in which case reduce μ until stable -
    do **not** routinely divide by ``block_size`` (that under-adapts the
    filter by the same factor).
    """
    x, sig = unwrap_signal(samples)
    if sig is not None:
        result = block_lms(
            x,
            training_symbols=training_symbols,
            num_taps=num_taps,
            sps=sps if sps is not None else int(sig.sps),
            step_size=step_size,
            block_size=block_size,
            modulation=modulation,
            order=order,
            unipolar=unipolar,
            store_weights=store_weights,
            w_init=w_init,
            pmf=pmf,
            cpr_type=cpr_type,
            cpr_bps_test_phases=cpr_bps_test_phases,
            cpr_bps_block_size=cpr_bps_block_size,
            cpr_joint_channels=cpr_joint_channels,
            cpr_cycle_slip_correction=cpr_cycle_slip_correction,
            cpr_cycle_slip_history=cpr_cycle_slip_history,
            cpr_cycle_slip_threshold=cpr_cycle_slip_threshold,
            debug_plot=debug_plot,
            plot_smoothing=plot_smoothing,
            cpr_state=cpr_state,
            input_norm_factor=input_norm_factor,
            samples_prefix=samples_prefix,
            pad_mode=pad_mode,
            cuda_graph=cuda_graph,
        )
        result.y_hat = rewrap_signal(sig, result.y_hat, sampling_rate=sig.symbol_rate)
        return result

    if sps is None:
        sps = 2

    if cpr_type is not None and cpr_type != "bps":
        raise ValueError(
            f"block_lms only supports cpr_type='bps' or None. Got {cpr_type!r}. "
            "PLL is not available for block processing."
        )

    num_taps = int(num_taps)
    sps = int(sps)
    block_size = int(block_size)

    _validate_sps(sps, num_taps)

    samples, xp, _ = dispatch(samples)
    if xp is np:
        logger.warning(
            "block_lms is running on CPU (NumPy). "
            "For CPU workloads lms(..., backend='numba') is typically 2-10x faster. "
            "Move samples to GPU (CuPy) to benefit from block-FFT acceleration."
        )

    samples, was_1d = as_2d(samples, name="samples")

    C = samples.shape[0]
    N = samples.shape[1]
    n_sym = N // sps

    if training_symbols is not None:
        training_symbols, _, _ = dispatch(training_symbols)
        if training_symbols.ndim == 1:
            training_symbols = training_symbols[np.newaxis, :]

    samples, training_symbols, eq_norm = _normalize_inputs(
        samples, training_symbols, sps, input_norm_factor=input_norm_factor
    )

    # -- Constellation ---------------------------------------------------------
    if modulation is not None and order is not None:
        from ..mapping import gray_constellation

        reference_constellation = gray_constellation(
            modulation, order, unipolar=unipolar
        )
        constellation_np = (
            to_device(reference_constellation, "cpu").flatten().astype(np.complex64)
        )
    elif training_symbols is not None:
        train_flat = to_device(training_symbols, "cpu").reshape(-1)
        constellation_np = np.unique(np.round(train_flat, decimals=8)).astype(
            np.complex64
        )
    else:
        raise ValueError("Provide modulation+order or training_symbols for DD slicer.")

    if pmf is not None and modulation is not None and order is not None:
        _pmf_arr = np.asarray(pmf, dtype=np.float64)
        _e_ps = float(
            np.dot(_pmf_arr, np.abs(constellation_np).astype(np.float64) ** 2)
        )
        if _e_ps < 1.0 - 1e-6:
            constellation_np = (constellation_np / np.sqrt(_e_ps)).astype(np.complex64)

    constellation = xp.asarray(constellation_np)  # (M,) on device
    _sq_side, _sq_lev_min_f, _sq_d_grid_f = _sq_qam_slicer_params(constellation_np)
    _sq_lev_min = float(_sq_lev_min_f)
    _sq_d_grid = float(_sq_d_grid_f)
    _sq_m1 = _sq_side - 1  # clip upper bound (0 when sq_side==0 - never used)

    # -- Training alignment ----------------------------------------------------
    if training_symbols is not None:
        n_train_aligned = min(int(training_symbols.shape[-1]), n_sym)
    else:
        n_train_aligned = 0

    # -- Weight initialisation -------------------------------------------------
    if w_init is not None:
        w_arr = np.ascontiguousarray(to_device(w_init, "cpu"), dtype=np.complex64)
        w_arr = _validate_w_init(w_arr, C, num_taps)
        h = xp.asarray(w_arr.copy())
    else:
        h = xp.asarray(_init_butterfly_weights_numpy(C, num_taps))  # (C, C, T)

    # -- BPS setup -------------------------------------------------------------
    if cpr_type == "bps":
        symmetry = _cpr_symmetry(modulation, order)
        P = int(cpr_bps_test_phases)
        bps_angles_np = np.linspace(
            0.0, np.pi / 2.0, P, endpoint=False, dtype=np.float32
        )
        bps_phases_neg = xp.asarray(
            np.exp(-1j * bps_angles_np).astype(np.complex64)
        )  # (P,)
        bps_angles = xp.asarray(bps_angles_np)  # (P,)
        quantum = np.float64(2.0 * np.pi / symmetry)
        _two_pi = xp.float64(2.0 * np.pi)
        # Cross-block 4-fold unwrap state (one float64 per channel)
        _cs_H = min(int(cpr_cycle_slip_history), n_sym)
        _bps_K = int(cpr_bps_block_size)
        _bps_hist_len = max(0, _bps_K - 1)
        _st = cpr_state
        _st_ok = (
            _st is not None
            and _st.cpr_type == cpr_type
            and _st.num_ch == C
            and _st.cs_H == _cs_H
            and _st.bps_P == P
            and _st.bps_K == _bps_K
            and _st.bps_prev4 is not None
        )
        if _st_ok:
            assert _st is not None
            assert _st.bps_prev4 is not None
            assert _st.bps_offset4 is not None
            assert _st.cs_buf_x is not None
            assert _st.cs_buf_y is not None
            assert _st.cs_buf_ptr is not None
            assert _st.cs_buf_n is not None
            assert _st.cs_stats is not None
            bps_prev4 = _st.bps_prev4.copy()
            bps_offset4 = _st.bps_offset4.copy()
            cs_buf_x = _st.cs_buf_x.copy()
            cs_buf_y = _st.cs_buf_y.copy()
            cs_buf_ptr = _st.cs_buf_ptr.copy()
            cs_buf_n = _st.cs_buf_n.copy()
            cs_stats = _st.cs_stats.copy()
            # xp.array (not asarray): the history buffer is updated in place
            # inside the block loop, so it must never alias the caller's state.
            bps_d2_hist = (
                xp.array(_st.bps_d2_hist, dtype=xp.float32)
                if _st.bps_d2_hist is not None
                else xp.zeros((P, C, _bps_hist_len), dtype=xp.float32)
            )
        else:
            bps_prev4 = np.zeros(C, dtype=np.float64)
            bps_offset4 = np.zeros(C, dtype=np.float64)
            cs_buf_x = np.zeros((C, _cs_H), dtype=np.float64)
            cs_buf_y = np.zeros((C, _cs_H), dtype=np.float64)
            cs_buf_ptr = np.zeros(C, dtype=np.int64)
            cs_buf_n = np.zeros(C, dtype=np.int64)
            cs_stats = np.zeros((C, 4), dtype=np.float64)
            bps_d2_hist = xp.zeros((P, C, _bps_hist_len), dtype=xp.float32)
        # Promote the unwrap carries to the active device once - every
        # per-block update then runs on-device, with no per-block H2D/D2H
        # sync.  The CPRState CPU-NumPy contract is honoured at the API
        # boundary only: ingested above, exported back via to_device() at
        # state export.  Cycle-slip buffers stay CPU-resident because the
        # slip kernel itself still runs on the host.
        bps_prev4 = xp.asarray(bps_prev4)
        bps_offset4 = xp.asarray(bps_offset4)

    # -- Fused CUDA kernels (CuPy only; None => xp fallback) -------------------
    # _k_bps: per-block inline-BPS min-distance metric (GRID for square QAM,
    # TABLE otherwise).  _k_dd: nearest-point search for the non-square DD
    # slicer (TABLE + argmin, called with a single unit phasor).  _k_cs: the
    # sequential cycle-slip detector (one thread per channel) operating on
    # device-resident history buffers.
    _k_bps = None
    _k_dd = None
    _k_cs = None
    _dd_phasor = None
    if xp is not np:
        from .. import _cuda

        _M_const = int(constellation_np.size)
        if cpr_type == "bps" and P <= 128:
            if _sq_side > 0:
                _k_bps = _cuda.get_kernel("bps_min_d2", mode="grid")
            elif _M_const <= 1024:
                _k_bps = _cuda.get_kernel("bps_min_d2", mode="table")
        if n_train_aligned < n_sym and _sq_side == 0 and _M_const <= 1024:
            _k_dd = _cuda.get_kernel("bps_min_d2", mode="table", return_argmin=True)
            if _k_dd is not None:
                _dd_phasor = xp.ones(1, dtype=xp.complex64)
        if cpr_type == "bps" and cpr_cycle_slip_correction and C <= 1024:
            _k_cs = _cuda.get_kernel("cs_block")
    if _k_cs is not None:
        # Cycle-slip state lives in device memory for the whole block loop;
        # the kernel mutates it in place.  Exported back to CPU NumPy at
        # state export, preserving the CPRState contract.  cs_buf_x is
        # unused by the detector and stays on CPU.
        cs_buf_y = xp.asarray(cs_buf_y)
        cs_buf_ptr = xp.asarray(cs_buf_ptr)
        cs_buf_n = xp.asarray(cs_buf_n)
        cs_stats = xp.asarray(cs_stats)

    # -- OLS block size --------------------------------------------------------
    # fftsize must be >= block_size * sps + num_taps - 1 (linear OLS condition)
    _ols_min = block_size * sps + num_taps - 1
    fftsize = 1 << (_ols_min - 1).bit_length()  # next power of 2

    _cpr_info = ""
    if cpr_type == "bps":
        _cs = (
            f", cs_corr=True(thr={cpr_cycle_slip_threshold:.3f})"
            if cpr_cycle_slip_correction
            else ", cs_corr=False"
        )
        _joint = ", joint" if cpr_joint_channels and C > 1 else ""
        _cpr_info = (
            f", cpr=bps(P={cpr_bps_test_phases}, K={cpr_bps_block_size}{_joint}{_cs})"
        )
    logger.info(
        "Block-LMS: C=%s, num_taps=%s, sps=%s, block_size=%s, fftsize=%s, "
        "mu=%s, n_sym=%s%s",
        C,
        num_taps,
        sps,
        block_size,
        fftsize,
        step_size,
        n_sym,
        _cpr_info,
    )

    # -- Padding - matches lms() convention -----------------------------------
    c_tap = num_taps // 2
    pad_total = max(0, n_sym * sps - N + num_taps - 1)
    pad_left = min(c_tap, pad_total)
    pad_right = pad_total - pad_left
    if samples_prefix is not None or xp is np or pad_mode != "zeros":
        _samp_cpu_blms = to_device(samples, "cpu").astype(np.complex64)
        x_padded = xp.asarray(
            _build_padded_samples(
                _samp_cpu_blms,
                pad_left,
                pad_right,
                samples_prefix,
                pad_mode,
                eq_norm,
                sps,
            )
        )  # (C, N_pad)
    else:
        # Fast on-device path: avoid D->H->D round-trip when there is no prefix.
        # _normalize_inputs already normalized samples; eq_norm is only needed
        # by _build_padded_samples to normalize the samples_prefix, which is
        # handled by the branch above.
        _samp_f32 = (
            samples if samples.dtype == xp.complex64 else samples.astype(xp.complex64)
        )
        _left = xp.zeros((C, pad_left), dtype=xp.complex64)
        _right = (
            xp.zeros((C, pad_right), dtype=xp.complex64)
            if pad_right > 0
            else xp.empty((C, 0), dtype=xp.complex64)
        )
        x_padded = xp.concatenate([_left, _samp_f32, _right], axis=1)  # (C, N_pad)
    N_padded = x_padded.shape[1]

    # -- Output buffers --------------------------------------------------------
    y_all = xp.empty((C, n_sym), dtype=xp.complex64)
    e_all = xp.empty((C, n_sym), dtype=xp.complex64)
    w_hist = (
        xp.empty((n_sym, C, C, num_taps), dtype=xp.complex64) if store_weights else None
    )
    phi_all = xp.zeros((C, n_sym), dtype=xp.float32) if cpr_type == "bps" else None

    # Pre-allocate scratch buffers - reused every block to avoid per-block heap
    # pressure, and (on GPU) to give the CUDA-graph capture stable pointers.
    x_win = xp.zeros((C, fftsize), dtype=xp.complex64)
    e_scatter = xp.zeros((C, fftsize), dtype=xp.complex64)
    # Fixed-width output workspaces (block_size columns).  The block body always
    # writes its per-symbol outputs here; the driver copies the valid [:, :B]
    # slice into the full-length result arrays at the right offset.  Routing
    # through fixed buffers (rather than writing y_all[:, b_start:b_end]
    # directly) is what lets the body be captured once and replayed: the only
    # thing that varies per block is the eager input fill and output copy.
    y_rot_ws = xp.empty((C, block_size), dtype=xp.complex64)
    e_clean_ws = xp.empty((C, block_size), dtype=xp.complex64)
    phi_ws = xp.empty((C, block_size), dtype=xp.float32) if cpr_type == "bps" else None

    n_blocks = (n_sym + block_size - 1) // block_size
    # On-device divergence flag - accumulates with |= inside the loop, zero D->H
    # syncs during iteration.  Checked once with bool() after the loop exits.
    _div_flag = xp.zeros(1, dtype=xp.bool_)

    def _run_block(B, b_start, n_train_blk):
        """Compute one block: forward filter, CPR, error, gradient, weight update.

        Reads the input window from ``x_win`` (filled by the caller), reads and
        updates the persistent filter/CPR state in place, and writes the
        per-symbol outputs into the fixed workspaces ``y_rot_ws``/``e_clean_ws``/
        ``phi_ws`` (columns ``[:, :B]``).  Issues no host synchronization, so on
        GPU it is safe to capture into a CUDA graph and replay.
        """
        nonlocal h, bps_prev4, bps_offset4, _div_flag

        # -- Forward pass (frequency-domain butterfly) ---------------------
        X_fd = xp.fft.fft(x_win, axis=-1)  # (C, F)
        H_fd = xp.fft.fft(h, n=fftsize, axis=-1)  # (C, C, F)
        # Butterfly contraction Y[i,k] = Σ_j conj(H[i,j,k]) X[j,k].  Written as a
        # broadcast-multiply + reduction (not einsum) for two reasons: einsum
        # dispatches to cuBLAS, which cannot be called inside a CUDA-graph stream
        # capture; and the reduction is accumulated in complex128 before the
        # complex64 downcast, per the CLAUDE.md filter-dot-product precision rule.
        Y_fd = (
            (xp.conj(H_fd).astype(xp.complex128) * X_fd.astype(xp.complex128)[None])
            .sum(axis=1)
            .astype(xp.complex64)
        )  # (C, F)
        y_time = xp.fft.ifft(Y_fd, axis=-1)  # (C, F)
        y_block = y_time[:, : B * sps : sps].astype(xp.complex64)  # (C, B)

        # -- BPS phase recovery --------------------------------------------
        if cpr_type == "bps":
            # min_d2: (P, C, B) - min squared distance to constellation over
            # all candidate rotations of all block symbols.
            if _k_bps is not None:
                # Fused kernel: single pass over y_block, no (P, C, B[, M])
                # rotated/distance intermediates.
                if _sq_side > 0:
                    min_d2 = _k_bps(
                        y_block,
                        bps_phases_neg,
                        lev_min=_sq_lev_min,
                        d_grid=_sq_d_grid,
                        side=_sq_side,
                    )
                else:
                    min_d2 = _k_bps(
                        y_block, bps_phases_neg, constellation=constellation
                    )
            else:
                # rotated: (P, C, B) - all candidate rotations for all block symbols
                rotated = bps_phases_neg[:, None, None] * y_block[None, :, :]
                # O(1) square-QAM: snap I/Q independently to nearest level grid point.
                if _sq_side > 0:
                    _nr = (
                        _sq_lev_min
                        + xp.clip(
                            xp.round((rotated.real - _sq_lev_min) / _sq_d_grid),
                            0,
                            _sq_m1,
                        )
                        * _sq_d_grid
                    )
                    _ni = (
                        _sq_lev_min
                        + xp.clip(
                            xp.round((rotated.imag - _sq_lev_min) / _sq_d_grid),
                            0,
                            _sq_m1,
                        )
                        * _sq_d_grid
                    )
                    min_d2 = (
                        (rotated.real - _nr) ** 2 + (rotated.imag - _ni) ** 2
                    ).astype(xp.float32)
                else:
                    d2_all = (
                        xp.abs(rotated[..., None] - constellation[None, None, None, :])
                        ** 2
                    ).real
                    min_d2 = xp.min(d2_all, axis=-1).astype(xp.float32)

            # Causal sliding-window average of width K along the B (symbol) axis.
            # win_sum[:,:,n] = sum of min_d2[:,:, n-K+1..n] (with K-1 samples from
            # previous block as prefix so the window is full from symbol 0).
            K = min(_bps_K, B)
            hist_prefix = (
                bps_d2_hist[:, :, -(K - 1) :]
                if K > 1
                else xp.empty((P, C, 0), dtype=xp.float32)
            )
            cat_d2 = xp.concatenate([hist_prefix, min_d2], axis=2)  # (P, C, K-1+B)
            cs_d2 = xp.concatenate(
                [xp.zeros((P, C, 1), dtype=xp.float32), cat_d2.cumsum(axis=2)], axis=2
            )  # (P, C, K+B)
            win_sum = cs_d2[:, :, K:] - cs_d2[:, :, :-K]  # (P, C, B)
            metric = win_sum / xp.float32(K)  # (P, C, B) - always full K-sample window

            # Per-symbol argmin over P phases -> raw (C, B) in [0, π/2)
            if cpr_joint_channels and C > 1:
                best_k = xp.argmin(metric.sum(axis=1), axis=0)  # (B,)
                phi_raw = xp.broadcast_to(
                    bps_angles[best_k][None, :], (C, B)
                ).copy()  # (C, B)
            else:
                best_k = xp.argmin(metric, axis=0)  # (C, B)
                phi_raw = bps_angles[best_k]  # (C, B)

            # Slide BPS distance history across block boundary (for next block's
            # prefix).  In-place write into the persistent buffer (not a rebind)
            # so the captured graph reads/writes the same address every replay.
            if _bps_hist_len > 0:
                combined_hist = xp.concatenate([bps_d2_hist, min_d2], axis=2)
                bps_d2_hist[...] = combined_hist[:, :, -_bps_hist_len:]

            # 4-fold causal unwrap: equivalent to np.unwrap(phi*4)/4 via
            # diff->wrap[-π,π]->cumsum.  CPU: np.unwrap directly (no transfer cost).
            # GPU: fully on-device - the carries are device arrays, so the
            # block loop issues no host sync for the unwrap state.
            if xp is np:
                raw4 = phi_raw.astype(np.float64) * 4.0  # (C, B)
                extended = np.concatenate(
                    [bps_prev4[:, np.newaxis], raw4], axis=1
                )  # (C, B+1)
                unwrapped_ext = np.unwrap(extended, axis=1)  # (C, B+1)
                _cumul_cpu = unwrapped_ext[:, 1:] - unwrapped_ext[:, 0:1]  # (C, B)
                _phi_f64 = (bps_offset4[:, np.newaxis] + _cumul_cpu) / 4.0  # (C, B)
                bps_prev4[:] = unwrapped_ext[:, -1]
                bps_offset4 += _cumul_cpu[:, -1]
            else:
                raw4_dev = phi_raw.astype(xp.float64) * xp.float64(4.0)  # (C, B)
                ext_dev = xp.concatenate(
                    [bps_prev4[:, None], raw4_dev], axis=1
                )  # (C, B+1)
                _two_pi_d = xp.float64(2.0 * np.pi)
                d4 = ext_dev[:, 1:] - ext_dev[:, :-1]  # (C, B)
                d4 -= xp.round(d4 / _two_pi_d) * _two_pi_d  # wrap to [-π, π]
                _cumul_dev = xp.cumsum(d4, axis=1)  # (C, B)
                _phi_f64 = (bps_offset4[:, None] + _cumul_dev) / xp.float64(4.0)
                bps_prev4 += _cumul_dev[:, -1]
                bps_offset4 += _cumul_dev[:, -1]
            # -- Per-symbol cycle-slip correction --------------------------
            # Identical algorithm to per-symbol lms: for each symbol in the
            # block, compare its BPS phase to the regression prediction, snap
            # to the nearest quantum if |diff| > threshold, then add the
            # corrected (x, y) pair to the circular history buffer.
            # GPU with the cs_block CUDA kernel: one launch per block on the
            # device-resident history buffers - zero host syncs.  Fallback
            # (CPU, or kernel unavailable): D->H transfer of the full (C, B)
            # float64 phase block, CPU detector (Numba or Python), H->D
            # write-back; cost is O(C·B) per block.
            if cpr_cycle_slip_correction:
                if _k_cs is not None:
                    _phi_corr_dev = xp.empty_like(_phi_f64)
                    _k_cs(
                        _phi_f64,
                        _phi_corr_dev,
                        cs_buf_y,
                        cs_buf_ptr,
                        cs_buf_n,
                        cs_stats,
                        float(quantum),
                        float(cpr_cycle_slip_threshold),
                        _cs_H,
                    )
                else:
                    phi_blk_np = to_device(_phi_f64, "cpu").astype(np.float64)  # (C, B)
                    phi_corr_np = phi_blk_np.copy()

                    _cs_kernel = _get_numba_cs_block()
                    if _cs_kernel is not None:
                        _cs_kernel(
                            phi_blk_np,
                            phi_corr_np,
                            cs_buf_x,
                            cs_buf_y,
                            cs_buf_ptr,
                            cs_buf_n,
                            cs_stats,
                            b_start,
                            float(quantum),
                            float(cpr_cycle_slip_threshold),
                            _cs_H,
                        )
                    else:
                        _H_f = float(_cs_H)
                        for ci in range(C):
                            for i in range(B):
                                y_b = phi_blk_np[ci, i]
                                n_b = int(cs_buf_n[ci])
                                ptr = int(cs_buf_ptr[ci])

                                if n_b == 0:
                                    phi_expected = y_b
                                elif n_b < 10:
                                    last_pos = (ptr - 1 + _cs_H) % _cs_H
                                    phi_expected = cs_buf_y[ci, last_pos]
                                else:
                                    sy = cs_stats[ci, 0]
                                    sxy = cs_stats[ci, 1]
                                    n_f = float(n_b)
                                    if n_b < _cs_H:
                                        Sx_c = n_f * (n_f - 1.0) / 2.0
                                        Sxx_c = (
                                            n_f * (n_f - 1.0) * (2.0 * n_f - 1.0) / 6.0
                                        )
                                        denom = n_f * Sxx_c - Sx_c * Sx_c
                                    else:
                                        Sx_c = _H_f * (_H_f - 1.0) / 2.0
                                        Sxx_c = (
                                            _H_f
                                            * (_H_f - 1.0)
                                            * (2.0 * _H_f - 1.0)
                                            / 6.0
                                        )
                                        denom = _H_f * Sxx_c - Sx_c * Sx_c
                                    if abs(denom) > 1e-30:
                                        slope = (n_f * sxy - Sx_c * sy) / denom
                                        intercept = (sy - slope * Sx_c) / n_f
                                    else:
                                        slope = 0.0
                                        intercept = sy / n_f
                                    phi_expected = slope * n_f + intercept

                                diff = y_b - phi_expected
                                k_slip = int(round(diff / quantum))
                                if (
                                    abs(diff) > float(cpr_cycle_slip_threshold)
                                    and k_slip != 0
                                ):
                                    y_b -= float(k_slip) * quantum
                                phi_corr_np[ci, i] = y_b

                                # Update circular buffer - relative coords, only y needed
                                write_pos = ptr % _cs_H
                                if n_b == _cs_H:
                                    old_y = cs_buf_y[ci, write_pos]
                                    old_sy = cs_stats[ci, 0]
                                    # Sxy_new = Sxy_old - Sy_old + y_old + (H-1)*y_new
                                    cs_stats[ci, 1] = (
                                        cs_stats[ci, 1]
                                        - old_sy
                                        + old_y
                                        + (_H_f - 1.0) * y_b
                                    )
                                    cs_stats[ci, 0] = old_sy - old_y + y_b
                                else:
                                    cs_stats[ci, 1] += float(n_b) * y_b
                                    cs_stats[ci, 0] += y_b
                                cs_buf_y[ci, write_pos] = y_b
                                cs_buf_ptr[ci] = ptr + 1
                                if n_b < _cs_H:
                                    cs_buf_n[ci] = n_b + 1

                    _phi_corr_dev = xp.asarray(phi_corr_np)

                # Carry the net slip correction into bps_offset4 so that
                # subsequent blocks do not re-detect the same slip.
                # bps_offset4 tracks the 4x-domain accumulated phase; the slip
                # quantum is π/2 -> 2π in 4x, which is a multiple of 2π and
                # therefore transparent to the 4-fold unwrap of bps_prev4.
                # Vectorized on-device op - a zero net slip adds 0.0 for free,
                # so no per-channel host-side != 0.0 test is needed.
                bps_offset4 += (_phi_corr_dev[:, -1] - _phi_f64[:, -1]) * 4.0
                _phi_f64 = _phi_corr_dev

            # Wrap unbounded float64 phase to [-π, π] before float32 cast so that
            # the GPU exp() argument is bounded (dual-path: wrapped for rotation,
            # unwrapped for trajectory storage).
            phi_c_dev = (_phi_f64 - xp.round(_phi_f64 / _two_pi) * _two_pi).astype(
                xp.float32
            )
            phi_c_traj = _phi_f64.astype(xp.float32)  # unwrapped, for output trajectory

            phi_c = phi_c_dev  # wrapped float32, for rotation
            y_rot = y_block * xp.exp(-1j * phi_c.astype(xp.complex64))  # (C, B)
            assert phi_ws is not None
            phi_ws[:, :B] = phi_c_traj  # unwrapped float32, for trajectory
        else:
            y_rot = y_block

        # -- Error computation (training or DD slicer) ---------------------
        e_clean = xp.empty((C, B), dtype=xp.complex64)

        if n_train_blk > 0:
            d_train = training_symbols[:, b_start : b_start + n_train_blk]
            e_clean[:, :n_train_blk] = d_train - y_rot[:, :n_train_blk]

        if n_train_blk < B:
            y_dd = y_rot[:, n_train_blk:]
            if _sq_side > 0:
                _dd_r = (
                    _sq_lev_min
                    + xp.clip(
                        xp.round((y_dd.real - _sq_lev_min) / _sq_d_grid), 0, _sq_m1
                    )
                    * _sq_d_grid
                )
                _dd_i = (
                    _sq_lev_min
                    + xp.clip(
                        xp.round((y_dd.imag - _sq_lev_min) / _sq_d_grid), 0, _sq_m1
                    )
                    * _sq_d_grid
                )
                d_dd = xp.empty(y_dd.shape, dtype=xp.complex64)
                d_dd.real[:] = _dd_r
                d_dd.imag[:] = _dd_i
            elif _k_dd is not None:
                # Fused nearest-point search: (1, C, B_dd) argmin indices with
                # a unit phasor, replacing the (C, B_dd, M) distance tensor.
                _, _dd_idx = _k_dd(y_dd, _dd_phasor, constellation=constellation)
                d_dd = constellation[_dd_idx[0]]
            else:
                d2_sl = (
                    xp.abs(y_dd[:, :, None] - constellation[None, None, :]) ** 2
                ).real
                d_dd = constellation[xp.argmin(d2_sl, axis=-1)]
            e_clean[:, n_train_blk:] = d_dd - y_dd

        # -- Store per-symbol outputs into the fixed workspaces ------------
        y_rot_ws[:, :B] = y_rot
        e_clean_ws[:, :B] = e_clean
        if store_weights:
            assert w_hist is not None
            w_hist[b_start : b_start + B] = h[None, :, :, :]

        # -- Back-rotate error to tap plane and compute gradient -----------
        if cpr_type == "bps":
            e_taps = e_clean * xp.exp(1j * phi_c.astype(xp.complex64))  # (C, B)
        else:
            e_taps = e_clean

        # Scatter e_taps to sample positions within the block window
        e_scatter.fill(0)
        e_scatter[:, : B * sps : sps] = e_taps

        # Frequency-domain gradient: dH_fd[i,j,k] = conj(E_fd[i,k]) * X_fd[j,k].
        # Outer product over the channel axes - no contracted index, hence no
        # accumulation (complex64 is exact enough) and no cuBLAS, so the
        # broadcast form is both capture-safe and replaces the einsum directly.
        E_fd = xp.fft.fft(e_scatter, axis=-1)  # (C, F)
        dH_fd = xp.conj(E_fd)[:, None, :] * X_fd[None, :, :]  # (C, C, F)
        dh = xp.fft.ifft(dH_fd, axis=-1)[:, :, :num_taps]  # (C, C, T)

        # In-place weight update so the captured graph reads/writes one buffer.
        h += xp.float32(step_size) * dh

        # Accumulate divergence flag on-device - no D->H sync here.
        _div_flag |= ~xp.isfinite(h).all()

    # CUDA-graph invariant: _fill_x_win and _store_outputs run eagerly *between*
    # graph replays, so they MUST NOT allocate from the device memory pool.  The
    # captured graph's intermediates were freed back to the pool after capture;
    # any allocation here could hand those exact blocks out and clobber the
    # pointers the replay reads/writes (intermittent garbage / divergence).  Both
    # helpers touch only pre-allocated buffers (x_win, y_all/e_all/phi_all,
    # *_ws) via fill/slice-assign - keep them allocation-free.
    def _fill_x_win(b_start):
        """Eager (per-block, varying offset) load of the input window into x_win.

        Kept outside ``_run_block`` because the source offset changes every
        block - a varying-pointer copy cannot live inside the captured graph.
        """
        x_start = b_start * sps
        x_win.fill(0)
        available = min(fftsize, N_padded - x_start)
        if available > 0:
            x_win[:, :available] = x_padded[:, x_start : x_start + available]

    def _store_outputs(b_start, b_end, B):
        """Eager copy of the fixed workspaces into the result arrays."""
        y_all[:, b_start:b_end] = y_rot_ws[:, :B]
        e_all[:, b_start:b_end] = e_clean_ws[:, :B]
        if cpr_type == "bps":
            assert phi_all is not None and phi_ws is not None
            phi_all[:, b_start:b_end] = phi_ws[:, :B]

    # -- CUDA-graph eligibility ------------------------------------------------
    # The block-loop body is host-sync-free, so on GPU it can
    # be captured once and replayed per block, collapsing ~30-50 kernel launches
    # into one.  Only full blocks (B == block_size) lying entirely in the
    # decision-directed region (n_train_blk == 0) share one fixed control flow
    # and shape, so only those are captured; the training/straddle blocks and
    # the final partial block always run eagerly.  Cycle-slip correction is
    # capturable only via the cs_block CUDA kernel (the CPU fallback syncs).
    _first_dd_full = ((n_train_aligned + block_size - 1) // block_size) * block_size
    _n_dd_full = max(0, (n_sym - _first_dd_full) // block_size)
    _use_graph = (
        cuda_graph
        and xp is not np
        and not store_weights
        and (not cpr_cycle_slip_correction or _k_cs is not None)
        and _n_dd_full >= 2  # need ≥1 warmup block + ≥1 captured block to pay off
    )
    try:
        import cupy as _cp_graph

        _graph_stream = _cp_graph.cuda.Stream(non_blocking=True) if _use_graph else None
    except Exception:
        _use_graph = False
        _graph_stream = None

    # All loop work - eager blocks, input fills, output copies, and graph
    # capture/replay - runs on a single stream so the shared state buffers
    # (h, bps_*, cs_*) stay ordered across the eager<->replay boundary.  On the
    # eager (CPU or graph-disabled) path this is a no-op context.
    _loop_stream_ctx: Any
    if _use_graph:
        assert _graph_stream is not None  # set together with _use_graph above
        # _graph_stream is non-blocking, so it does NOT implicitly serialize
        # with the default stream that produced x_padded, h, the scratch
        # buffers, and the BPS/cs constants above.  Without an explicit join the
        # first block (and the graph capture) races those still-in-flight setup
        # writes - intermittently reading/capturing garbage, which surfaces as
        # divergence or bad convergence that "fixes itself" on rerun (the race
        # is timing-dependent).  Make the loop stream wait for that setup work.
        _setup_done = _cp_graph.cuda.Event()
        _setup_done.record()  # records on the current (default) stream
        _graph_stream.wait_event(_setup_done)
        _loop_stream_ctx = _graph_stream
    else:
        _loop_stream_ctx = contextlib.nullcontext()

    # -- Block loop ------------------------------------------------------------
    _graph = None  # captured CUDA graph, built lazily on the 2nd full DD block
    _graph_warmed = False  # True once one full DD block has primed the mem pool
    with _loop_stream_ctx:
        for b in range(n_blocks):
            b_start = b * block_size
            b_end = min(b_start + block_size, n_sym)
            B = b_end - b_start  # symbols this block (may be < block_size for last)
            n_train_blk = max(0, min(n_train_aligned - b_start, B))
            _capturable = _use_graph and block_size == B and n_train_blk == 0

            _fill_x_win(b_start)

            if not _capturable:
                _run_block(B, b_start, n_train_blk)  # eager
            elif _graph is not None:
                _graph.launch()  # replay (current stream == _graph_stream)
            elif not _graph_warmed:
                _run_block(B, b_start, 0)  # eager warmup - primes the memory pool
                _graph_warmed = True
            else:
                # Capture the body once.  Stream capture records the kernel
                # sequence without executing it (pool allocations reuse the
                # blocks the warmup freed, so no cudaMalloc occurs); launch()
                # then executes this block.  On any capture failure, fall back
                # to running the remaining blocks eagerly.
                assert (
                    _graph_stream is not None
                )  # _capturable => _use_graph => stream set
                try:
                    _graph_stream.begin_capture()
                    _run_block(B, b_start, 0)
                    _graph = _graph_stream.end_capture()
                    _graph.launch()
                except Exception as exc:  # pragma: no cover - hw/version dependent
                    with contextlib.suppress(Exception):
                        _graph_stream.end_capture()
                    _graph = None
                    _use_graph = False
                    logger.warning(
                        "block_lms CUDA-graph capture failed (%s); "
                        "continuing with the eager block loop.",
                        exc,
                    )
                    _run_block(B, b_start, 0)  # ensure this block runs once

            _store_outputs(b_start, b_end, B)

    if _graph_stream is not None:
        _graph_stream.synchronize()

    # Single D->H sync after the full loop to check for divergence.
    if bool(_div_flag[0]):
        raise RuntimeError(
            f"block_lms diverged (step_size={step_size}, block_size={block_size}). "
            f"step_size is on the same scale as lms(), but because the weights are "
            f"frozen across the block the stability ceiling is ~{block_size}x lower "
            f"than per-symbol LMS. Reduce step_size until stable (e.g. try "
            f"{step_size / 2:.2e}, then keep halving) rather than dividing by "
            f"block_size, which would under-adapt the filter by that factor."
        )

    # -- Pack result -----------------------------------------------------------
    if was_1d:
        y_out = y_all[0]
        e_out = e_all[0]
        W_out = h[0, 0]
        if store_weights and w_hist is not None:
            w_history = w_hist[:, 0, 0, :]
        else:
            w_history = None

        if cpr_type == "bps" and phi_all is not None:
            phase_traj = phi_all[0]
        else:
            phase_traj = None
    else:
        y_out = y_all
        e_out = e_all
        W_out = h
        w_history = w_hist if store_weights else None
        phase_traj = phi_all if cpr_type == "bps" else None

    result = EqualizerResult(
        y_hat=y_out,
        weights=W_out,
        error=e_out,
        weights_history=w_history,
        num_train_symbols=n_train_aligned,
        input_norm_factor=eq_norm,
        phase_trajectory=phase_traj,
    )
    if cpr_type == "bps":
        result.cpr_state = CPRState(
            bps_prev4=to_device(bps_prev4, "cpu").copy(),
            bps_offset4=to_device(bps_offset4, "cpu").copy(),
            bps_d2_hist=to_device(bps_d2_hist, "cpu"),
            cs_buf_x=cs_buf_x.copy(),
            cs_buf_y=to_device(cs_buf_y, "cpu").copy(),
            cs_buf_ptr=to_device(cs_buf_ptr, "cpu").copy(),
            cs_buf_n=to_device(cs_buf_n, "cpu").copy(),
            cs_stats=to_device(cs_stats, "cpu").copy(),
            cpr_type=cpr_type,
            num_ch=C,
            symmetry=_cpr_symmetry(modulation, order),
            bps_P=P,
            bps_K=_bps_K,
            cs_H=_cs_H,
        )
    return _log_equalizer_exit(
        result, name="Block-LMS", debug_plot=debug_plot, plot_smoothing=plot_smoothing
    )


# -----------------------------------------------------------------------------
# BLIND FREQUENCY-DOMAIN EQUALIZERS  (block_cma / block_rde)
# -----------------------------------------------------------------------------
#
# block_cma / block_rde are the blind, phase-directed siblings of block_lms:
# the same overlap-save frequency-domain adaptive filter (FDAF), but with the
# Godard / ring-radius error instead of the trained/DD slicer.  They share the
# two FDAF primitives below (forward butterfly + frequency-domain gradient);
# block_lms keeps its own CPR/cycle-slip/CUDA-graph-optimised body.
#
# All three use the same additive update ``h += mu * conj(E_fd) * X_fd`` with
# the gradient correlation in the frequency domain, so the blind error is cast
# into the trained convention's sign: CMA ``E = y*(R2 - |y|^2)`` and RDE
# ``E = y*(R_d^2 - |y|^2)`` (the negated Godard/ring gradient), and pilot
# positions use the LMS residual ``E = pilot_ref - y`` directly.


def _fdaf_forward(h, x_win, fftsize, sps, B, xp):
    """One overlap-save FDAF forward block.

    Returns ``(y_block (C, B), X_fd (C, F))`` - the decimated output symbols and
    the input spectrum (reused by the gradient).  The butterfly contraction is a
    complex128-accumulated broadcast-reduce (CLAUDE.md filter-dot-product rule),
    matching ``block_lms``'s capture-safe form.
    """
    X_fd = xp.fft.fft(x_win, axis=-1)  # (C, F)
    H_fd = xp.fft.fft(h, n=fftsize, axis=-1)  # (C, C, F)
    Y_fd = (
        (xp.conj(H_fd).astype(xp.complex128) * X_fd.astype(xp.complex128)[None])
        .sum(axis=1)
        .astype(xp.complex64)
    )  # (C, F)
    y_time = xp.fft.ifft(Y_fd, axis=-1)
    y_block = y_time[:, : B * sps : sps].astype(xp.complex64)  # (C, B)
    return y_block, X_fd


def _fdaf_gradient_update(h, X_fd, e_block, e_scatter, sps, B, num_taps, mu, xp):
    """Scatter ``e_block`` to sample positions, form the frequency-domain
    gradient ``dH[i,j] = conj(E_fd[i]) * X_fd[j]``, and apply the in-place
    update ``h += mu * IFFT(dH)[..., :num_taps]``."""
    e_scatter.fill(0)
    e_scatter[:, : B * sps : sps] = e_block
    E_fd = xp.fft.fft(e_scatter, axis=-1)  # (C, F)
    dH_fd = xp.conj(E_fd)[:, None, :] * X_fd[None, :, :]  # (C, C, F)
    dh = xp.fft.ifft(dH_fd, axis=-1)[:, :, :num_taps]  # (C, C, T)
    h += xp.float32(mu) * dh


def _block_fdaf_blind(
    kind,
    samples,
    *,
    num_taps,
    sps,
    step_size,
    block_size,
    r2,
    radii_np,
    w_init,
    input_norm_factor,
    samples_prefix,
    pad_mode,
    pilot_ref,
    pilot_mask,
    pilot_gain_db,
    c_ps,
    cuda_graph,
    debug_plot,
    plot_smoothing,
    name,
):
    """Shared overlap-save FDAF engine for ``block_cma``/``block_rde``.

    Blind, phase-directed adaptation with no CPR - the per-block error is the
    Godard (``kind='cma'``) or nearest-ring (``kind='rde'``) gradient, with
    pilot positions overridden by the LMS residual.  ``r2`` is the Godard radius
    (CMA) and ``radii_np`` the unique ring radii (RDE).
    """
    num_taps = int(num_taps)
    sps = int(sps)
    block_size = int(block_size)
    _validate_sps(sps, num_taps)

    samples, xp, _ = dispatch(samples)
    if xp is np:
        logger.warning(
            "%s is running on CPU (NumPy). For CPU workloads "
            "%s(..., backend='numba') is typically faster. Move samples to "
            "GPU (CuPy) to benefit from block-FFT acceleration.",
            name,
            kind,
        )

    samples, was_1d = as_2d(samples, name="samples")
    C = samples.shape[0]
    N = samples.shape[1]
    n_sym = N // sps

    use_pilots = pilot_ref is not None and pilot_mask is not None
    # Deboost pilot positions before normalisation so boosted pilots don't
    # inflate the RMS and bias the Godard target (matches cma/rde).
    if use_pilots and pilot_gain_db != 0.0:
        amp = xp.float32(10.0 ** (pilot_gain_db / 20.0))
        smask = xp.asarray(np.repeat(np.asarray(pilot_mask).astype(bool), sps))
        samples = samples.copy()
        samples[..., smask] /= amp
    samples, _, eq_norm = _normalize_inputs(
        samples, None, sps, input_norm_factor=input_norm_factor
    )

    # -- OLS block size + padding (matches block_lms) -----------------------
    _ols_min = block_size * sps + num_taps - 1
    fftsize = 1 << (_ols_min - 1).bit_length()
    logger.info(
        "%s: C=%s, num_taps=%s, sps=%s, block_size=%s, fftsize=%s, mu=%s, "
        "n_sym=%s, pilot_aided=%s",
        name,
        C,
        num_taps,
        sps,
        block_size,
        fftsize,
        step_size,
        n_sym,
        use_pilots,
    )
    c_tap = num_taps // 2
    pad_total = max(0, n_sym * sps - N + num_taps - 1)
    pad_left = min(c_tap, pad_total)
    pad_right = pad_total - pad_left
    if samples_prefix is not None or xp is np or pad_mode != "zeros":
        _cpu = to_device(samples, "cpu").astype(np.complex64)
        x_padded = xp.asarray(
            _build_padded_samples(
                _cpu, pad_left, pad_right, samples_prefix, pad_mode, eq_norm, sps
            )
        )
    else:
        _f32 = (
            samples if samples.dtype == xp.complex64 else samples.astype(xp.complex64)
        )
        _l = xp.zeros((C, pad_left), dtype=xp.complex64)
        _r = (
            xp.zeros((C, pad_right), dtype=xp.complex64)
            if pad_right > 0
            else xp.empty((C, 0), dtype=xp.complex64)
        )
        x_padded = xp.concatenate([_l, _f32, _r], axis=1)
    N_padded = x_padded.shape[1]

    # -- Weight initialisation ----------------------------------------------
    if w_init is not None:
        w_arr = _validate_w_init(
            np.ascontiguousarray(to_device(w_init, "cpu"), dtype=np.complex64),
            C,
            num_taps,
        )
        h = xp.asarray(w_arr.copy())
    else:
        h = xp.asarray(_init_butterfly_weights_numpy(C, num_taps))  # (C, C, T)

    # -- Pilot / radii device arrays ----------------------------------------
    if use_pilots:
        pref = xp.asarray(
            np.ascontiguousarray(to_device(pilot_ref, "cpu"), dtype=np.complex64)
        )
        if pref.ndim == 1:
            pref = xp.tile(pref[None, :], (C, 1))
        if c_ps is not None:
            pref = (pref * xp.complex64(c_ps)).astype(xp.complex64)
        pmask_dev = xp.asarray(np.asarray(pilot_mask).astype(bool))
    if kind == "rde":
        radii = xp.asarray(np.asarray(radii_np, dtype=np.float64))

    # -- Scratch + output buffers -------------------------------------------
    # Pre-allocated and reused every block: avoids per-block heap pressure and
    # (on GPU) gives the CUDA-graph capture stable read/write pointers.
    x_win = xp.zeros((C, fftsize), dtype=xp.complex64)
    e_scatter = xp.zeros((C, fftsize), dtype=xp.complex64)
    y_all = xp.empty((C, n_sym), dtype=xp.complex64)
    e_all = xp.empty((C, n_sym), dtype=xp.complex64)
    # Fixed-width (block_size) output workspaces.  The capturable body always
    # writes its per-symbol outputs here; the eager driver copies the valid
    # [:, :B] slice into the full-length arrays at the right offset.  Routing
    # through fixed buffers (rather than writing y_all[:, b_start:b_end]
    # directly) is what lets the body be captured once and replayed: only the
    # input fill and output copy vary per block.
    y_ws = xp.empty((C, block_size), dtype=xp.complex64)
    e_ws = xp.empty((C, block_size), dtype=xp.complex64)
    n_blocks = (n_sym + block_size - 1) // block_size

    def _run_block(B, b_start):
        """Compute one block: forward FDAF, blind error, gradient + in-place
        weight update.  Reads the window from ``x_win`` (filled by the caller),
        updates ``h`` in place, and writes outputs into ``y_ws``/``e_ws`` (cols
        ``[:, :B]``).  Issues no host sync, so on GPU it is capturable.  The
        ``b_start``-varying pilot slice is only touched on the (non-captured)
        pilot-aided path."""
        nonlocal h
        y_block, X_fd = _fdaf_forward(h, x_win, fftsize, sps, B, xp)

        abs2 = xp.real(y_block * xp.conj(y_block))  # (C, B) strict-real |y|^2
        if kind == "cma":
            e = y_block * (xp.float32(r2) - abs2)
        else:  # rde - nearest ring radius per symbol
            abs_y = xp.sqrt(abs2)
            rd = radii[
                xp.argmin(xp.abs(abs_y[:, :, None] - radii[None, None, :]), axis=-1)
            ]
            e = y_block * (rd.astype(xp.float32) ** 2 - abs2)
        if use_pilots:
            pm = pmask_dev[b_start : b_start + B][None, :]
            e = xp.where(pm, pref[:, b_start : b_start + B] - y_block, e)

        y_ws[:, :B] = y_block
        e_ws[:, :B] = e
        _fdaf_gradient_update(h, X_fd, e, e_scatter, sps, B, num_taps, step_size, xp)

    # CUDA-graph invariant: _fill_x_win / _store_outputs run eagerly *between*
    # replays and MUST NOT allocate from the device memory pool (a fresh
    # allocation could hand out the blocks the captured graph's freed
    # intermediates still reference).  Both touch only pre-allocated buffers.
    def _fill_x_win(b_start):
        """Eager (per-block, varying offset) load of the input window."""
        x_start = b_start * sps
        x_win.fill(0)
        avail = min(fftsize, N_padded - x_start)
        if avail > 0:
            x_win[:, :avail] = x_padded[:, x_start : x_start + avail]

    def _store_outputs(b_start, b_end, B):
        """Eager copy of the fixed workspaces into the result arrays."""
        y_all[:, b_start:b_end] = y_ws[:, :B]
        e_all[:, b_start:b_end] = e_ws[:, :B]

    # -- CUDA-graph eligibility ----------------------------------------------
    # The block body is host-sync-free, so on GPU it can be captured once and
    # replayed per block, collapsing the per-block kernel launches into one.
    # Only full blocks (B == block_size) share a fixed shape and control flow,
    # so only those are captured; the final partial block runs eagerly.  The
    # pilot-aided path reads b_start-varying pilot slices inside the body and is
    # therefore not capturable - it always runs the eager loop.
    n_full = n_sym // block_size
    _use_graph = (
        cuda_graph
        and xp is not np
        and not use_pilots
        and n_full >= 2  # need ≥1 warmup block + ≥1 captured block to pay off
    )
    try:
        import cupy as _cp_graph

        _graph_stream = _cp_graph.cuda.Stream(non_blocking=True) if _use_graph else None
    except Exception:
        _use_graph = False
        _graph_stream = None

    # Run the whole loop on one stream so the shared state (h) stays ordered
    # across the eager<->replay boundary.  The non-blocking graph stream does not
    # implicitly serialize with the default stream that produced x_padded/h/the
    # scratch buffers, so join it to that setup work before the first block to
    # avoid capturing still-in-flight garbage.
    _loop_stream_ctx: Any
    if _use_graph:
        assert _graph_stream is not None
        _setup_done = _cp_graph.cuda.Event()
        _setup_done.record()
        _graph_stream.wait_event(_setup_done)
        _loop_stream_ctx = _graph_stream
    else:
        _loop_stream_ctx = contextlib.nullcontext()

    # -- Block loop ----------------------------------------------------------
    _graph = None  # captured CUDA graph, built lazily on the 2nd full block
    _graph_warmed = False  # True once one full block has primed the mem pool
    with _loop_stream_ctx:
        for b in range(n_blocks):
            b_start = b * block_size
            b_end = min(b_start + block_size, n_sym)
            B = b_end - b_start
            _capturable = _use_graph and block_size == B

            _fill_x_win(b_start)

            if not _capturable:
                _run_block(B, b_start)  # eager (partial block, or graph off)
            elif _graph is not None:
                _graph.launch()  # replay (current stream == _graph_stream)
            elif not _graph_warmed:
                _run_block(B, b_start)  # eager warmup - primes the memory pool
                _graph_warmed = True
            else:
                # Capture the body once; pool allocations reuse the blocks the
                # warmup freed, so no cudaMalloc occurs during capture.  On any
                # failure, fall back to the eager loop for the rest of the run.
                assert _graph_stream is not None
                try:
                    _graph_stream.begin_capture()
                    _run_block(B, b_start)
                    _graph = _graph_stream.end_capture()
                    _graph.launch()
                except Exception as exc:  # pragma: no cover - hw/version dependent
                    with contextlib.suppress(Exception):
                        _graph_stream.end_capture()
                    _graph = None
                    _use_graph = False
                    logger.warning(
                        "%s CUDA-graph capture failed (%s); "
                        "continuing with the eager block loop.",
                        name,
                        exc,
                    )
                    _run_block(B, b_start)  # ensure this block runs once

            _store_outputs(b_start, b_end, B)

    if _graph_stream is not None:
        _graph_stream.synchronize()

    # Single D->H sync to surface divergence after the whole loop.
    if not bool(xp.isfinite(h).all()):
        raise RuntimeError(
            f"{name} diverged (step_size={step_size}, block_size={block_size}). "
            f"step_size is on the same scale as {kind}(); because the weights are "
            f"frozen across the block the stability ceiling is ~{block_size}x lower. "
            f"Reduce step_size (e.g. {step_size / 2:.2e}, then keep halving)."
        )

    if was_1d:
        y_out, e_out, W_out = y_all[0], e_all[0], h[0, 0]
    else:
        y_out, e_out, W_out = y_all, e_all, h
    result = EqualizerResult(
        y_hat=y_out,
        weights=W_out,
        error=e_out,
        weights_history=None,
        num_train_symbols=0,
        input_norm_factor=eq_norm,
    )
    return _log_equalizer_exit(
        result,
        name=name,
        debug_plot=debug_plot,
        check_convergence=True,
        plot_smoothing=plot_smoothing,
    )
