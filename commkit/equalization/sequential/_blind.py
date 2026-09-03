"""Blind sequential adaptive equalizers: cma, rde."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...backend import ArrayType, _get_jax, dispatch, to_device, to_jax
from ...core.signal import Signal
from ...helpers import rewrap_signal, unwrap_signal
from ...logger import logger
from .._block import (
    _prep_blind_block_inputs,
    _run_block_equalizer,
    _validate_block_mode,
)
from .._common import (
    _build_padded_samples,
    _init_butterfly_weights_jax,
    _init_butterfly_weights_numpy,
    _normalize_inputs,
    _unpack_result_jax,
    _unpack_result_numpy,
    _validate_sps,
    _validate_w_init,
)
from .._kernels_jax import _get_jax_cma, _get_jax_pa_cma, _get_jax_pa_rde, _get_jax_rde
from .._kernels_numba import (
    _get_numba,
    _get_numba_cma,
    _get_numba_pa_cma,
    _get_numba_pa_rde,
    _get_numba_rde,
)
from ..result import EqualizerResult, _log_equalizer_exit

# -----------------------------------------------------------------------------
# BLIND equalization
# -----------------------------------------------------------------------------


def cma(
    samples: ArrayType | Signal,
    num_taps: int = 21,
    sps: int | None = None,
    step_size: float = 1e-3,
    modulation: str | None = None,
    order: int | None = None,
    unipolar: bool = False,
    store_weights: bool = False,
    device: str | None = "cpu",
    center_tap: int | None = None,
    backend: str = "numba",
    w_init: ArrayType | None = None,
    pilot_ref: ArrayType | None = None,
    pilot_mask: np.ndarray | None = None,
    pilot_gain_db: float = 0.0,
    pmf: Any | None = None,
    debug_plot: bool = False,
    plot_smoothing: int = 50,
    input_norm_factor: float | np.ndarray | None = None,
    samples_prefix: ArrayType | None = None,
    pad_mode: str = "zeros",
    update_mode: str = "sequential",
    block_len: int = 16,
) -> EqualizerResult:
    """
    Constant Modulus Algorithm blind equalizer with butterfly MIMO support.

    ``update_mode='block'`` (``block_len`` 8-32) freezes the weights over
    ``block_len`` symbols and applies one aggregated Godard gradient per chunk,
    turning the per-symbol update into a matrix product the GPU can occupy.  It
    requires ``backend='jax'`` (chunked ``lax.scan``) or ``backend='xp'``
    (array-native NumPy/CuPy); ``backend='numba'`` and ``store_weights`` are not
    supported.  ``step_size`` is on the **same scale as** sequential mode (the
    aggregated gradient is the sum over the chunk): the same ``mu`` gives the
    same floor - only the stability ceiling is ~``block_len``x lower, so reduce
    ``mu`` only if the run diverges.  Pilot-aided masking carries over unchanged.

    CMA minimizes the Godard dispersion criterion and requires no training
    symbols. It is the standard blind equalizer for constant-modulus signals
    (PSK) and near-constant-modulus signals (low-order QAM).

    CMA recovers the signal up to a phase ambiguity. A phase recovery step
    (e.g. Viterbi-Viterbi, pilot-aided) is typically needed after CMA.

    When ``pilot_ref`` and ``pilot_mask`` are both supplied the equalizer
    switches to a **pilot-aided hybrid** mode: the standard Godard CMA error
    is used at data positions while an LMS residual error
    (``pilot_ref - y``) is used at every pilot position.  This resolves the
    phase ambiguity at pilot locations while preserving blind adaptation
    elsewhere.  Build the dense arrays with ``build_pilot_ref``.

    Algorithm (per symbol n)
    ------------------------
    Steps 1 and 2 are identical to ``lms`` (sliding input window and
    butterfly filter output ``y_raw[n]``).  There is **no CPR step** -
    CMA's cost surface is phase-invariant; no radial error can drive a
    phase rotator (see Notes below).

    3. **Godard error** - third-order radial gradient of the dispersion
       cost ``J = E[(|y|^2 - R^2)^2]``::

           e[n] = (|y[n]|^2 - R^2) * y[n]

       The Godard radius ``R^2 = E[|s|^4] / E[|s|^2]`` is computed once
       from the normalised constellation (defaults to 1 if ``modulation``
       is not given).  The error is purely radial: any constant phase
       rotation of ``y`` leaves ``|y|^2`` and therefore ``e`` unchanged
       up to the same rotation, so CMA cannot resolve the phase ambiguity
       it introduces.

    4. **Weight update** - steepest descent on the Godard criterion (note
       the minus sign, opposite to LMS)::

           w_{c,c'} -= mu * conj(e_c[n]) * x_{c',n}

    **Pilot-aided hybrid** (when ``pilot_ref`` and ``pilot_mask`` are set):
    at pilot positions the Godard error is replaced by the LMS pilot error
    ``e_p[n] = pilot_ref[n] - y[n]``, and the weight update sign flips to
    ``+mu`` (standard LMS gradient ascent toward the reference).  This
    resolves the phase ambiguity at pilot locations while CMA handles data
    positions blindly.

    Notes
    -----
    **Why joint CMA + CPR is not supported:**
    PLL requires a phase-coherent decision ``d[n]`` (nearest constellation
    point) to form the cross-product error ``Im(y * conj(d))``; but CMA
    output has an unknown phase rotation, so the decision is unreliable.
    BPS is blind, but CMA weights converge to one of four equally-valid
    90° rotations and slowly drift between them - BPS would track that
    drift, but the next CMA gradient step would fight the correction.  Use
    the sequential pipeline instead: CMA ->
    ``correct_carrier_phase`` (BPS or
    Viterbi-Viterbi) -> optional ``lms`` fine-tune.

    Parameters
    ----------
    samples : array_like or Signal
        Input signal samples. Shape: ``(N_samples,)`` or ``(C, N_samples)``.
        Typically at 2 samples/symbol for fractionally-spaced equalization.
        A :class:`Signal` returns an :class:`EqualizerResult` whose ``y_hat``
        is a new :class:`Signal` at the symbol rate (``sampling_rate =
        symbol_rate``); ``sps`` defaults to the signal's ``sps`` when not
        given explicitly.
    num_taps : int, default 21
        Number of equalizer taps per FIR filter.
    sps : int, optional, default 2
        Samples per symbol at the input.  Use ``sps=2`` (T/2-spaced, default)
        for the standard first-stage blind equalization.  ``sps=1`` enables
        symbol-spaced CMA, useful when input is already decimated but phase
        ambiguity resolution is still needed.  Ignored for :class:`Signal`
        input, which always uses the signal's own ``sps``.
    step_size : float, default 1e-3
        CMA step size (mu). Unlike LMS, CMA's cost surface is non-convex and
        higher-order, so input-power normalization distorts the gradient geometry.
        Use a fixed step size in the range 1e-5 to 1e-3 for stability.
    modulation : str, optional
        Modulation type for auto-computing Godard radius R2 (e.g. ``"psk"``, ``"qam"``).
        If None, defaults to R2=1.0.
    order : int, optional
        Modulation order for auto-computing R2.
    unipolar : bool, default False
        Use unipolar constellation for auto-computing R2.
    store_weights : bool, default False
        If True, stores weight trajectory.
    device : str, optional
        Target device for JAX computations (e.g., 'cpu', 'gpu', 'tpu').
        Default is 'cpu'. Ignored when ``backend='numba'``.
    center_tap : int, optional
        Index of the center tap. If None, defaults to ``num_taps // 2``.
    backend : str, default 'numba'
        Execution backend. ``'numba'`` uses Numba ``@njit``; LLVM-compiled,
        typically fastest on CPU. ``'jax'`` uses ``jax.lax.scan``
        (XLA-compiled, GPU-capable).  ``'xp'`` is valid only with
        ``update_mode='block'`` - the array-native NumPy/CuPy block loop.
    w_init : array_like, optional
        Initial tap weights. Shape: ``(C, C, num_taps)`` complex64, or the
        SISO short-hand ``(num_taps,)`` / ``(1, num_taps)`` as returned by
        ``EqualizerResult.weights`` for single-channel equalizers.
        Warm-starts blind equalization from pre-converged weights (e.g. from
        a prior ``lms()`` call on the preamble). Raises ``ValueError`` on
        shape mismatch.
    pilot_ref : (C, N_sym) complex64 array, optional
        Dense pilot reference array - zeros at data positions, known symbols
        at pilot positions.  Build with ``build_pilot_ref``.
        Must be provided together with ``pilot_mask``.
    pilot_mask : (N_sym,) uint8 array, optional
        Pilot position mask - ``1`` at pilot positions, ``0`` elsewhere.
        Build with ``build_pilot_ref``.
    pilot_gain_db : float, default 0.0
        Pilot boosting in dB relative to payload power, matching
        ``SingleCarrierFrame.pilot_gain_db``.  When non-zero, the received
        signal at pilot positions is attenuated by the inverse of the boost
        factor before the global RMS normalisation.  This prevents boosted
        pilots from inflating the RMS estimate and biasing the Godard
        convergence target at data positions.  Set to ``0.0`` when pilots
        are not boosted.
    pmf : array_like of float, optional
        Probability mass function for PS-QAM.  When provided with ``modulation``
        and ``order``, the Godard R2 is computed for the unit-power PS
        distribution ``{s_m/sqrt(E_PS)}``:
        ``R2 = E_PS[|s_m|^4] / E_PS^2``.  Pilot references are also scaled
        by ``1/sqrt(E_PS)`` so pilot-aided and blind sections converge to the
        same unit-power target.
    input_norm_factor : float or ndarray, optional
        Pre-computed RMS normalization factor from a previous call.  See
        ``lms()`` for the full description; behaviour is identical.
    samples_prefix : array_like, optional
        Signal history from the end of the previous block.  See ``lms()``
        for the full description; behaviour is identical.
    pad_mode : {'zeros', 'edge'}, default 'zeros'
        Padding strategy when ``samples_prefix`` is ``None``.  See
        ``lms()`` for the full description; behaviour is identical.
    update_mode : {'sequential', 'block'}, default 'sequential'
        Weight-update cadence.  ``'sequential'`` updates every symbol (the
        default; the only mode for ``backend='numba'``).  ``'block'`` freezes
        the weights over ``block_len`` symbols and applies one aggregated Godard
        gradient per chunk (a GPU-occupying matrix product); requires
        ``backend='jax'`` or ``backend='xp'`` and is incompatible with
        ``store_weights``.  Pilot-aided masking carries over unchanged.
    block_len : int, default 16
        Symbols per frozen-weight chunk when ``update_mode='block'`` (typically
        8-32; ignored otherwise).  ``step_size`` stays on the same scale as
        sequential mode; only the stability ceiling is ~``block_len``x lower.

    Returns
    -------
    EqualizerResult
        Equalized symbols, final weights, CMA error history, and optionally
        weight trajectory.  ``input_norm_factor`` field is populated.  When
        ``samples`` is a :class:`Signal`, ``y_hat`` is a new :class:`Signal`
        at the symbol rate (``sampling_rate = symbol_rate``).

    Warnings
    --------
    **JAX GPU mode is typically slower than CPU for adaptive equalization.**
    CMA is inherently sequential: each weight update depends on the previous
    weights, so ``lax.scan`` serializes execution even on GPU.  Use
    ``device='cpu'`` for typical SISO sequences, or ``backend='numba'`` for
    CPU-optimal throughput.
    """
    x, sig = unwrap_signal(samples)
    if sig is not None:
        # sig.sps is always populated (derived from the required sampling_rate
        # / symbol_rate fields), so the Signal's own value always wins over a
        # supplied sps - see CLAUDE.md, "Signal-Awareness".
        if sps is not None:
            logger.warning(
                "cma(): ignoring supplied sps=%r for Signal input; using the "
                "signal's own sps=%r instead.",
                sps,
                sig.sps,
            )
        result = cma(
            x,
            num_taps=num_taps,
            sps=int(sig.sps),
            step_size=step_size,
            modulation=modulation,
            order=order,
            unipolar=unipolar,
            store_weights=store_weights,
            device=device,
            center_tap=center_tap,
            backend=backend,
            w_init=w_init,
            pilot_ref=pilot_ref,
            pilot_mask=pilot_mask,
            pilot_gain_db=pilot_gain_db,
            pmf=pmf,
            debug_plot=debug_plot,
            plot_smoothing=plot_smoothing,
            input_norm_factor=input_norm_factor,
            samples_prefix=samples_prefix,
            pad_mode=pad_mode,
            update_mode=update_mode,
            block_len=block_len,
        )
        result.y_hat = rewrap_signal(sig, result.y_hat, sampling_rate=sig.symbol_rate)
        return result

    if sps is None:
        sps = 2

    use_pilots = pilot_ref is not None and pilot_mask is not None
    _validate_block_mode(update_mode, block_len, backend, store_weights=store_weights)
    logger.info(
        "CMA equalizer: num_taps=%s, mu=%s, sps=%s, backend=%s, "
        "pilot_aided=%s, pilot_gain_db=%s",
        num_taps,
        step_size,
        sps,
        backend,
        use_pilots,
        pilot_gain_db,
    )
    if sps > 1:
        logger.warning(
            "CMA output y_hat is at 1 SPS (symbol rate). "
            "Update sampling_rate = symbol_rate after applying this equalizer."
        )

    samples, xp, _ = dispatch(samples)
    stride = int(sps)
    _validate_sps(sps, num_taps)

    was_1d = samples.ndim == 1
    if was_1d:
        num_ch = 1
        n_samples = samples.shape[0]
    else:
        num_ch, n_samples = samples.shape

    # Compute R2 and PS-QAM scale factor from the Godard constellation.
    _c_ps = None  # 1/sqrt(E_PS) scale factor; None for uniform modulation
    if modulation is not None and order is not None:
        from ...mapping import gray_constellation

        const = gray_constellation(modulation, order, unipolar=unipolar)
        if pmf is not None:
            # PS-QAM: R2 for the unit-power distribution {s_m/sqrt(E_PS)}:
            #   R2 = E_PS[|s_m/sqrt(E_PS)|^4] / E_PS[|s_m/sqrt(E_PS)|^2]
            #      = (E_PS[|s_m|^4] / E_PS^2) / 1
            #      = E_PS[|s_m|^4] / E_PS^2
            _pmf_arr = np.asarray(pmf, dtype=np.float64)
            _abs2 = np.abs(const) ** 2
            _e_ps = float(np.dot(_pmf_arr, _abs2))
            r2 = float(np.dot(_pmf_arr, np.abs(const) ** 4)) / (_e_ps**2)
            if _e_ps < 1.0 - 1e-6:
                _c_ps = np.float32(1.0 / np.sqrt(_e_ps))
            logger.debug(
                "CMA R2 (PS-QAM pmf-weighted, %s-%s): %.4f",
                modulation.upper(),
                order,
                r2,
            )
        else:
            r2 = float(np.mean(np.abs(const) ** 4) / np.mean(np.abs(const) ** 2))
            logger.debug("CMA R2 from %s-%s: %.4f", modulation.upper(), order, r2)
    else:
        r2 = 1.0

    n_sym = n_samples // stride

    c_tap = center_tap if center_tap is not None else num_taps // 2
    pad_total = max(0, n_sym * stride - n_samples + num_taps - 1)
    pad_left = min(c_tap, pad_total)
    pad_right = pad_total - pad_left

    if update_mode == "block":
        samples_padded_np, w_arr, eq_norm, pref_np, pmask_np = _prep_blind_block_inputs(
            samples,
            sps=sps,
            stride=stride,
            pad_left=pad_left,
            pad_right=pad_right,
            samples_prefix=samples_prefix,
            pad_mode=pad_mode,
            input_norm_factor=input_norm_factor,
            use_pilots=use_pilots,
            pilot_gain_db=pilot_gain_db,
            pilot_mask=pilot_mask,
            pilot_ref=pilot_ref,
            c_ps=_c_ps,
            num_ch=num_ch,
            num_taps=num_taps,
            center_tap=center_tap,
            w_init=w_init,
        )
        return _run_block_equalizer(
            "cma",
            samples_padded_np=samples_padded_np,
            w_arr=w_arr,
            num_ch=num_ch,
            num_taps=num_taps,
            n_sym=n_sym,
            stride=stride,
            block_len=block_len,
            step_size=step_size,
            backend=backend,
            device=device,
            was_1d=was_1d,
            xp=xp,
            eq_norm=eq_norm,
            name="CMA(block)" if not use_pilots else "CMA(PA,block)",
            debug_plot=debug_plot,
            plot_smoothing=plot_smoothing,
            check_convergence=True,
            r2=r2,
            pref_np=pref_np,
            pmask_np=pmask_np,
        )

    if backend == "numba":
        numba = _get_numba()
        if numba is None:
            raise ImportError("Numba is required for backend='numba'.")

        samples_np = np.ascontiguousarray(to_device(samples, "cpu"), dtype=np.complex64)
        # Deboost pilot positions before global normalisation so boosted pilots
        # don't inflate the RMS estimate and bias the Godard convergence target.
        if use_pilots and pilot_gain_db != 0.0:
            assert pilot_mask is not None
            _amp = np.float32(10.0 ** (pilot_gain_db / 20.0))
            _smask = np.repeat(pilot_mask.astype(bool), stride)  # (N_samples,)
            samples_np[..., _smask] /= _amp
        # RMS-normalize samples to unit symbol-rate power (CMA has no training)
        samples_np, _, eq_norm = _normalize_inputs(
            samples_np, None, sps, input_norm_factor=input_norm_factor
        )

        x_np = _build_padded_samples(
            samples_np, pad_left, pad_right, samples_prefix, pad_mode, eq_norm, sps
        )
        x_np = np.ascontiguousarray(x_np)

        if w_init is not None:
            w_arr = np.ascontiguousarray(to_device(w_init, "cpu"), dtype=np.complex64)
            w_arr = _validate_w_init(w_arr, num_ch, num_taps)
            W = w_arr.copy()
        else:
            W = _init_butterfly_weights_numpy(num_ch, num_taps, center_tap=center_tap)
        y_out = np.empty((n_sym, num_ch), dtype=np.complex64)
        e_out = np.empty((n_sym, num_ch), dtype=np.complex64)
        w_hist_buf = (
            np.empty((n_sym, num_ch, num_ch, num_taps), dtype=np.complex64)
            if store_weights
            else np.empty((1, num_ch, num_ch, num_taps), dtype=np.complex64)
        )
        if use_pilots:
            pref = np.ascontiguousarray(to_device(pilot_ref, "cpu"), dtype=np.complex64)
            if _c_ps is not None:
                pref = (pref * _c_ps).astype(np.complex64)
            pmask = np.ascontiguousarray(pilot_mask, dtype=np.uint8)
            _get_numba_pa_cma()(
                x_np,
                W,
                np.float32(step_size),
                np.float32(r2),
                stride,
                store_weights,
                y_out,
                e_out,
                w_hist_buf,
                pref,
                pmask,
            )
        else:
            _get_numba_cma()(
                x_np,
                W,
                np.float32(step_size),
                np.float32(r2),
                stride,
                store_weights,
                y_out,
                e_out,
                w_hist_buf,
            )
        return _log_equalizer_exit(
            _unpack_result_numpy(
                y_out,
                e_out,
                W,
                w_hist_buf,
                was_1d,
                store_weights,
                n_sym=None,
                xp=xp,
                input_norm_factor=eq_norm,
            ),
            name="CMA" if not use_pilots else "CMA(PA)",
            debug_plot=debug_plot,
            check_convergence=True,
            plot_smoothing=plot_smoothing,
        )

    # JAX backend
    jax, jnp, _ = _get_jax()
    if jax is None or jnp is None:
        raise ImportError("JAX is required for backend='jax'.")

    # Deboost pilot positions before global normalisation so boosted pilots
    # don't inflate the RMS estimate and bias the Godard convergence target.
    if use_pilots and pilot_gain_db != 0.0:
        assert pilot_mask is not None
        _amp_jax = float(10.0 ** (pilot_gain_db / 20.0))
        _smask_jax = xp.asarray(np.repeat(pilot_mask.astype(bool), stride))
        samples = samples.copy()
        samples[..., _smask_jax] /= xp.float32(_amp_jax)
    # RMS-normalize samples to unit symbol-rate power (CMA has no training)
    samples, _, eq_norm = _normalize_inputs(
        samples, None, sps, input_norm_factor=input_norm_factor
    )

    _samp_cpu_cma = to_device(samples, "cpu").astype(np.complex64)
    samples_padded_np_cma = _build_padded_samples(
        _samp_cpu_cma, pad_left, pad_right, samples_prefix, pad_mode, eq_norm, sps
    )
    samples_padded = (
        xp.asarray(samples_padded_np_cma)
        if not was_1d
        else xp.asarray(samples_padded_np_cma[0])
    )

    x_jax = to_jax(samples_padded, device=device)
    if was_1d:
        x_jax = x_jax[None, :]

    try:
        platform = (
            device.lower()
            if device is not None
            else (
                x_jax.device.platform
                if hasattr(x_jax, "device")
                else list(x_jax.devices())[0].platform
            )
        )
    except Exception:
        platform = "cpu"

    if w_init is not None:
        w_arr = np.ascontiguousarray(to_device(w_init, "cpu"), dtype=np.complex64)
        w_arr = _validate_w_init(w_arr, num_ch, num_taps)
        W_jax = to_jax(w_arr, device=platform)
    else:
        W_jax = _init_butterfly_weights_jax(
            num_ch, num_taps, jnp, center_tap=center_tap
        )
        W_jax = to_jax(W_jax, device=platform)
    mu_jax = to_jax(jnp.float32(step_size), device=platform)
    r2_jax = to_jax(jnp.float32(r2), device=platform)

    if use_pilots:
        pref_np = np.ascontiguousarray(to_device(pilot_ref, "cpu"), dtype=np.complex64)
        if _c_ps is not None:
            pref_np = (pref_np * _c_ps).astype(np.complex64)
        pmask_np = np.ascontiguousarray(pilot_mask, dtype=np.uint8)
        # pilot_ref: (C, n_sym) -> (n_sym, C) for scan xs
        pref_jax = to_jax(pref_np.T, device=platform)
        pmask_jax = to_jax(pmask_np.astype(bool), device=platform)
        scan_fn = _get_jax_pa_cma(num_taps, stride, num_ch)
        y_jax, e_jax, W_jax, wh_jax = scan_fn(
            x_jax, W_jax, mu_jax, r2_jax, pref_jax, pmask_jax, n_sym
        )
    else:
        scan_fn = _get_jax_cma(num_taps, stride, num_ch)
        y_jax, e_jax, W_jax, wh_jax = scan_fn(x_jax, W_jax, mu_jax, r2_jax, n_sym)
    return _log_equalizer_exit(
        _unpack_result_jax(
            y_jax,
            e_jax,
            W_jax,
            wh_jax,
            was_1d,
            store_weights,
            n_sym=None,
            xp=xp,
            input_norm_factor=eq_norm,
        ),
        name="CMA" if not use_pilots else "CMA(PA)",
        debug_plot=debug_plot,
        check_convergence=True,
    )


def rde(
    samples: ArrayType | Signal,
    num_taps: int = 21,
    sps: int | None = None,
    step_size: float = 1e-3,
    modulation: str | None = None,
    order: int | None = None,
    unipolar: bool = False,
    store_weights: bool = False,
    device: str | None = "cpu",
    center_tap: int | None = None,
    backend: str = "numba",
    w_init: ArrayType | None = None,
    pilot_ref: ArrayType | None = None,
    pilot_mask: np.ndarray | None = None,
    pilot_gain_db: float = 0.0,
    pmf: Any | None = None,
    debug_plot: bool = False,
    plot_smoothing: int = 50,
    input_norm_factor: float | np.ndarray | None = None,
    samples_prefix: ArrayType | None = None,
    pad_mode: str = "zeros",
    update_mode: str = "sequential",
    block_len: int = 16,
) -> EqualizerResult:
    """
    Radius Directed Equalizer (RDE) - blind equalizer for multi-ring constellations.

    ``update_mode='block'`` (``block_len`` 8-32) freezes the weights over
    ``block_len`` symbols and applies one aggregated ring-directed gradient per
    chunk.  It requires ``backend='jax'`` (chunked ``lax.scan``) or
    ``backend='xp'`` (array-native NumPy/CuPy); ``backend='numba'`` and
    ``store_weights`` are not supported.  ``step_size`` is on the **same scale
    as** sequential mode (the aggregated gradient is the sum over the chunk):
    the same ``mu`` gives the same floor - only the stability ceiling is
    ~``block_len``x lower, so reduce ``mu`` only if the run diverges.
    Pilot-aided masking carries over to block mode unchanged.

    RDE is a CMA variant that replaces the single Godard dispersion radius with
    per-symbol radius selection from the set of unique constellation ring radii.
    This corrects CMA's fundamental weakness on higher-order QAM: CMA forces
    all symbols toward a single average circle, severely degrading convergence
    when the constellation spans multiple rings (e.g. inner, middle, outer rings
    of 16-QAM).  RDE instead drives each symbol toward its *nearest* ring,
    producing a gradient surface that matches the true constellation geometry.

    Like CMA, RDE is fully blind (no training symbols) and recovers the channel
    up to a **phase ambiguity**.  A carrier-phase recovery step is needed after
    convergence; see ``cma`` Notes for why joint CPR is not supported.

    Algorithm (per symbol n)
    ------------------------
    Steps 1 and 2 are identical to ``lms`` (sliding input window and
    butterfly filter output ``y[n]``).  Like ``cma``, there is no
    CPR step.

    3. **Ring selection** - choose the constellation ring radius closest to
       the current output magnitude::

           R_d[n] = argmin_{r in R_set} |r - |y[n]||
           R_set  = {|c| : c in constellation}

       ``R_set`` is the set of unique ring radii extracted once from the
       normalised Gray constellation.  For 16-QAM this yields three radii
       rather than the single CMA average, eliminating the inward/outward
       pull that degrades CMA convergence on higher-order QAM.

    4. **RDE error** - same third-order form as ``cma`` but using the
       per-symbol ring radius::

           e[n] = (|y[n]|^2 - R_d[n]^2) * y[n]

    5. **Weight update** - steepest descent (same sign convention as CMA)::

           w_{c,c'} -= mu * conj(e_c[n]) * x_{c',n}

    **Pilot-aided hybrid** (when ``pilot_ref`` and ``pilot_mask`` are set):
    identical to ``cma`` - at pilot positions the RDE error is replaced
    by ``e_p[n] = pilot_ref[n] - y[n]`` and the sign flips to ``+mu``,
    resolving the phase ambiguity at those locations.

    Parameters
    ----------
    samples : array_like or Signal
        Input signal samples. Shape: ``(N_samples,)`` or ``(C, N_samples)``.
        Typically at 2 samples/symbol for fractionally-spaced equalization.
        A :class:`Signal` returns an :class:`EqualizerResult` whose ``y_hat``
        is a new :class:`Signal` at the symbol rate (``sampling_rate =
        symbol_rate``); ``sps`` defaults to the signal's ``sps`` when not
        given explicitly.
    num_taps : int, default 21
        Number of equalizer taps per FIR filter.
    sps : int, optional, default 2
        Samples per symbol at the input.  Use ``sps=2`` (T/2-spaced, default)
        for standard blind equalization.  ``sps=1`` is accepted.  Ignored for
        :class:`Signal` input, which always uses the signal's own ``sps``.
    step_size : float, default 1e-3
        RDE step size (mu). Same non-convex gradient geometry as CMA; use a
        fixed step in the range 1e-5 to 1e-3 for stability.
    modulation : str, optional
        Modulation type for constellation construction (``"psk"``, ``"qam"``).
        Required to extract unique ring radii.  If ``None``, falls back to a
        single unit radius (identical to CMA with ``R²=1``).
    order : int, optional
        Modulation order (e.g. 4, 16, 64).
    unipolar : bool, default False
        Use unipolar constellation for radius extraction.
    store_weights : bool, default False
        If True, stores weight trajectory in ``result.weights_history``.
    device : str, optional
        Target JAX device (``'cpu'``, ``'gpu'``). Ignored for ``backend='numba'``.
    center_tap : int, optional
        Index of the center tap. Defaults to ``num_taps // 2``.
    backend : str, default 'numba'
        ``'numba'`` uses Numba ``@njit``; ``'jax'`` uses ``jax.lax.scan``.
        ``'xp'`` is valid only with ``update_mode='block'`` - the array-native
        NumPy/CuPy block loop.
    w_init : array_like, optional
        Initial tap weights. Shape: ``(C, C, num_taps)`` complex64, or the
        SISO short-hand ``(num_taps,)`` / ``(1, num_taps)`` as returned by
        ``EqualizerResult.weights`` for single-channel equalizers.
        Warm-starts blind equalization from pre-converged weights (e.g. from
        a prior ``lms()`` or ``cma()`` call). Raises ``ValueError`` on shape
        mismatch.
    pilot_ref : (C, N_sym) complex64 array, optional
        Dense pilot reference array - zeros at data positions, known symbols
        at pilot positions.  Build with ``build_pilot_ref``.
        Must be provided together with ``pilot_mask``.
    pilot_mask : (N_sym,) uint8 array, optional
        Pilot position mask - ``1`` at pilot positions, ``0`` elsewhere.
        Build with ``build_pilot_ref``.
    pilot_gain_db : float, default 0.0
        Pilot boosting in dB relative to payload power, matching
        ``SingleCarrierFrame.pilot_gain_db``.  When non-zero, the received
        signal at pilot positions is attenuated by the inverse of the boost
        factor before the global RMS normalisation.  This prevents boosted
        pilots from inflating the RMS estimate and biasing the ring-radius
        convergence targets at data positions.  Set to ``0.0`` when pilots
        are not boosted.
    pmf : array_like of float, optional
        Probability mass function for PS-QAM.  When provided with ``modulation``
        and ``order``, the ring radii are scaled by ``1/sqrt(E_PS)`` to target
        the unit-power constellation ``{|s_m|/sqrt(E_PS)}``.  Pilot references
        are also scaled accordingly.  Requires ``modulation`` and ``order``.
    input_norm_factor : float or ndarray, optional
        Pre-computed RMS normalization factor from a previous call.  See
        ``lms()`` for the full description; behaviour is identical.
    samples_prefix : array_like, optional
        Signal history from the end of the previous block.  See ``lms()``
        for the full description; behaviour is identical.
    pad_mode : {'zeros', 'edge'}, default 'zeros'
        Padding strategy when ``samples_prefix`` is ``None``.  See
        ``lms()`` for the full description; behaviour is identical.
    update_mode : {'sequential', 'block'}, default 'sequential'
        Weight-update cadence.  ``'sequential'`` updates every symbol (the
        default; the only mode for ``backend='numba'``).  ``'block'`` freezes
        the weights over ``block_len`` symbols and applies one aggregated
        ring-directed gradient per chunk (a GPU-occupying matrix product);
        requires ``backend='jax'`` or ``backend='xp'`` and is incompatible with
        ``store_weights``.  Pilot-aided masking carries over unchanged.
    block_len : int, default 16
        Symbols per frozen-weight chunk when ``update_mode='block'`` (typically
        8-32; ignored otherwise).  ``step_size`` stays on the same scale as
        sequential mode; only the stability ceiling is ~``block_len``x lower.

    Returns
    -------
    EqualizerResult
        Equalized symbols, final weights, RDE error history, and optionally
        weight trajectory.  ``input_norm_factor`` field is populated.  When
        ``samples`` is a :class:`Signal`, ``y_hat`` is a new :class:`Signal`
        at the symbol rate (``sampling_rate = symbol_rate``).

    Notes
    -----
    **Why RDE outperforms CMA on high-order QAM:**

    For 16-QAM the Godard radius ``R² = E[|s|⁴]/E[|s|²] ≈ 1.32`` (normalized).
    This single target is a poor proxy for the three distinct rings at
    ``|c| ≈ {0.45, 1.00, 1.34}`` (normalized unit-average-power 16-QAM).
    CMA pulls inner-ring symbols outward and outer-ring symbols inward,
    creating a persistent gradient that opposes correct convergence.
    RDE eliminates this bias entirely: each symbol is only attracted to its
    own ring, so the steady-state gradient vanishes at the correct solution.

    **Phase ambiguity:** Both CMA and RDE share the same 90°-symmetric cost
    surface for QAM/PSK.  Use a phase recovery algorithm after blind equalization.

    **GPU note:** RDE is inherently sequential (each weight update depends on
    previous weights), so ``lax.scan`` serializes execution even on GPU.
    Use ``device='cpu'`` for typical SISO sequences, or ``backend='numba'``
    for CPU-optimal throughput.
    """
    x, sig = unwrap_signal(samples)
    if sig is not None:
        # sig.sps is always populated (derived from the required sampling_rate
        # / symbol_rate fields), so the Signal's own value always wins over a
        # supplied sps - see CLAUDE.md, "Signal-Awareness".
        if sps is not None:
            logger.warning(
                "rde(): ignoring supplied sps=%r for Signal input; using the "
                "signal's own sps=%r instead.",
                sps,
                sig.sps,
            )
        result = rde(
            x,
            num_taps=num_taps,
            sps=int(sig.sps),
            step_size=step_size,
            modulation=modulation,
            order=order,
            unipolar=unipolar,
            store_weights=store_weights,
            device=device,
            center_tap=center_tap,
            backend=backend,
            w_init=w_init,
            pilot_ref=pilot_ref,
            pilot_mask=pilot_mask,
            pilot_gain_db=pilot_gain_db,
            pmf=pmf,
            debug_plot=debug_plot,
            plot_smoothing=plot_smoothing,
            input_norm_factor=input_norm_factor,
            samples_prefix=samples_prefix,
            pad_mode=pad_mode,
            update_mode=update_mode,
            block_len=block_len,
        )
        result.y_hat = rewrap_signal(sig, result.y_hat, sampling_rate=sig.symbol_rate)
        return result

    if sps is None:
        sps = 2

    use_pilots = pilot_ref is not None and pilot_mask is not None
    _validate_block_mode(update_mode, block_len, backend, store_weights=store_weights)
    logger.info(
        "RDE equalizer: num_taps=%s, mu=%s, sps=%s, backend=%s, "
        "pilot_aided=%s, pilot_gain_db=%s",
        num_taps,
        step_size,
        sps,
        backend,
        use_pilots,
        pilot_gain_db,
    )
    if sps > 1:
        logger.warning(
            "RDE output y_hat is at 1 SPS (symbol rate). "
            "Update sampling_rate = symbol_rate after applying this equalizer."
        )

    samples, xp, _ = dispatch(samples)
    stride = int(sps)
    _validate_sps(sps, num_taps)

    was_1d = samples.ndim == 1
    if was_1d:
        num_ch = 1
        n_samples = samples.shape[0]
    else:
        num_ch, n_samples = samples.shape

    # Compute unique ring radii from constellation.
    # For constant-modulus signals (PSK) this degenerates to a single radius,
    # making RDE identical to CMA.
    _c_ps = None  # 1/sqrt(E_PS) scale factor; None for uniform modulation
    if modulation is not None and order is not None:
        from ...mapping import gray_constellation

        const = gray_constellation(modulation, order, unipolar=unipolar)
        raw_radii = np.abs(const).astype(np.float32)
        if pmf is not None:
            # PS-QAM: scale radii to unit-power targets {|s_m|/sqrt(E_PS)}
            _pmf_arr = np.asarray(pmf, dtype=np.float64)
            _e_ps = float(np.dot(_pmf_arr, raw_radii.astype(np.float64) ** 2))
            if _e_ps < 1.0 - 1e-6:
                _c_ps = np.float32(1.0 / np.sqrt(_e_ps))
                raw_radii = (raw_radii * _c_ps).astype(np.float32)
        # Round to 6 significant digits to merge numerically identical radii
        radii = np.unique(np.round(raw_radii, 6))
        logger.debug(
            "RDE radii from %s-%s: %s",
            modulation.upper(),
            order,
            ", ".join(f"{r:.4f}" for r in radii),
        )
    else:
        radii = np.array([1.0], dtype=np.float32)
        logger.debug("RDE: no modulation provided, using single unit radius (≡ CMA)")

    n_sym = n_samples // stride
    num_radii = len(radii)

    c_tap = center_tap if center_tap is not None else num_taps // 2
    pad_total = max(0, n_sym * stride - n_samples + num_taps - 1)
    pad_left = min(c_tap, pad_total)
    pad_right = pad_total - pad_left

    if update_mode == "block":
        samples_padded_np, w_arr, eq_norm, pref_np, pmask_np = _prep_blind_block_inputs(
            samples,
            sps=sps,
            stride=stride,
            pad_left=pad_left,
            pad_right=pad_right,
            samples_prefix=samples_prefix,
            pad_mode=pad_mode,
            input_norm_factor=input_norm_factor,
            use_pilots=use_pilots,
            pilot_gain_db=pilot_gain_db,
            pilot_mask=pilot_mask,
            pilot_ref=pilot_ref,
            c_ps=_c_ps,
            num_ch=num_ch,
            num_taps=num_taps,
            center_tap=center_tap,
            w_init=w_init,
        )
        return _run_block_equalizer(
            "rde",
            samples_padded_np=samples_padded_np,
            w_arr=w_arr,
            num_ch=num_ch,
            num_taps=num_taps,
            n_sym=n_sym,
            stride=stride,
            block_len=block_len,
            step_size=step_size,
            backend=backend,
            device=device,
            was_1d=was_1d,
            xp=xp,
            eq_norm=eq_norm,
            name="RDE(block)" if not use_pilots else "RDE(PA,block)",
            debug_plot=debug_plot,
            plot_smoothing=plot_smoothing,
            check_convergence=True,
            radii_np=radii,
            pref_np=pref_np,
            pmask_np=pmask_np,
        )

    if backend == "numba":
        numba = _get_numba()
        if numba is None:
            raise ImportError("Numba is required for backend='numba'.")

        samples_np = np.ascontiguousarray(to_device(samples, "cpu"), dtype=np.complex64)
        # Deboost pilot positions before global normalisation so boosted pilots
        # don't inflate the RMS estimate and bias the ring-radius convergence targets.
        if use_pilots and pilot_gain_db != 0.0:
            assert pilot_mask is not None
            _amp = np.float32(10.0 ** (pilot_gain_db / 20.0))
            _smask = np.repeat(pilot_mask.astype(bool), stride)  # (N_samples,)
            samples_np[..., _smask] /= _amp
        samples_np, _, eq_norm = _normalize_inputs(
            samples_np, None, sps, input_norm_factor=input_norm_factor
        )

        x_np = _build_padded_samples(
            samples_np, pad_left, pad_right, samples_prefix, pad_mode, eq_norm, sps
        )
        x_np = np.ascontiguousarray(x_np)

        # Normalize radii to match the unit-power-normalized samples
        # (constellation is unit-average-power after gray_constellation)
        radii_np = np.ascontiguousarray(radii, dtype=np.float32)

        if w_init is not None:
            w_arr = np.ascontiguousarray(to_device(w_init, "cpu"), dtype=np.complex64)
            w_arr = _validate_w_init(w_arr, num_ch, num_taps)
            W = w_arr.copy()
        else:
            W = _init_butterfly_weights_numpy(num_ch, num_taps, center_tap=center_tap)
        y_out = np.empty((n_sym, num_ch), dtype=np.complex64)
        e_out = np.empty((n_sym, num_ch), dtype=np.complex64)
        w_hist_buf = (
            np.empty((n_sym, num_ch, num_ch, num_taps), dtype=np.complex64)
            if store_weights
            else np.empty((1, num_ch, num_ch, num_taps), dtype=np.complex64)
        )
        if use_pilots:
            pref = np.ascontiguousarray(to_device(pilot_ref, "cpu"), dtype=np.complex64)
            if _c_ps is not None:
                pref = (pref * _c_ps).astype(np.complex64)
            pmask = np.ascontiguousarray(pilot_mask, dtype=np.uint8)
            _get_numba_pa_rde()(
                x_np,
                W,
                np.float32(step_size),
                radii_np,
                stride,
                store_weights,
                y_out,
                e_out,
                w_hist_buf,
                pref,
                pmask,
            )
        else:
            _get_numba_rde()(
                x_np,
                W,
                np.float32(step_size),
                radii_np,
                stride,
                store_weights,
                y_out,
                e_out,
                w_hist_buf,
            )
        return _log_equalizer_exit(
            _unpack_result_numpy(
                y_out,
                e_out,
                W,
                w_hist_buf,
                was_1d,
                store_weights,
                n_sym=None,
                xp=xp,
                input_norm_factor=eq_norm,
            ),
            name="RDE" if not use_pilots else "RDE(PA)",
            debug_plot=debug_plot,
            check_convergence=True,
            plot_smoothing=plot_smoothing,
        )

    # JAX backend
    jax, jnp, _ = _get_jax()
    if jax is None or jnp is None:
        raise ImportError("JAX is required for backend='jax'.")

    # Deboost pilot positions before global normalisation so boosted pilots
    # don't inflate the RMS estimate and bias the ring-radius convergence targets.
    if use_pilots and pilot_gain_db != 0.0:
        assert pilot_mask is not None
        _amp_jax = float(10.0 ** (pilot_gain_db / 20.0))
        _smask_jax = xp.asarray(np.repeat(pilot_mask.astype(bool), stride))
        samples = samples.copy()
        samples[..., _smask_jax] /= xp.float32(_amp_jax)
    samples, _, eq_norm = _normalize_inputs(
        samples, None, sps, input_norm_factor=input_norm_factor
    )

    _samp_cpu_rde = to_device(samples, "cpu").astype(np.complex64)
    samples_padded_np_rde = _build_padded_samples(
        _samp_cpu_rde, pad_left, pad_right, samples_prefix, pad_mode, eq_norm, sps
    )
    samples_padded = (
        xp.asarray(samples_padded_np_rde)
        if not was_1d
        else xp.asarray(samples_padded_np_rde[0])
    )

    x_jax = to_jax(samples_padded, device=device)
    if was_1d:
        x_jax = x_jax[None, :]

    try:
        platform = (
            device.lower()
            if device is not None
            else (
                x_jax.device.platform
                if hasattr(x_jax, "device")
                else list(x_jax.devices())[0].platform
            )
        )
    except Exception:
        platform = "cpu"

    if w_init is not None:
        w_arr = np.ascontiguousarray(to_device(w_init, "cpu"), dtype=np.complex64)
        w_arr = _validate_w_init(w_arr, num_ch, num_taps)
        W_jax = to_jax(w_arr, device=platform)
    else:
        W_jax = _init_butterfly_weights_jax(
            num_ch, num_taps, jnp, center_tap=center_tap
        )
        W_jax = to_jax(W_jax, device=platform)
    mu_jax = to_jax(jnp.float32(step_size), device=platform)
    radii_jax = to_jax(jnp.asarray(radii, dtype=jnp.float32), device=platform)

    if use_pilots:
        pref_np = np.ascontiguousarray(to_device(pilot_ref, "cpu"), dtype=np.complex64)
        if _c_ps is not None:
            pref_np = (pref_np * _c_ps).astype(np.complex64)
        pmask_np = np.ascontiguousarray(pilot_mask, dtype=np.uint8)
        # pilot_ref: (C, n_sym) -> (n_sym, C) for scan xs
        pref_jax = to_jax(pref_np.T, device=platform)
        pmask_jax = to_jax(pmask_np.astype(bool), device=platform)
        scan_fn = _get_jax_pa_rde(num_taps, stride, num_radii, num_ch)
        y_jax, e_jax, W_jax, wh_jax = scan_fn(
            x_jax, W_jax, mu_jax, radii_jax, pref_jax, pmask_jax, n_sym
        )
    else:
        scan_fn = _get_jax_rde(num_taps, stride, num_radii, num_ch)
        y_jax, e_jax, W_jax, wh_jax = scan_fn(x_jax, W_jax, mu_jax, radii_jax, n_sym)
    return _log_equalizer_exit(
        _unpack_result_jax(
            y_jax,
            e_jax,
            W_jax,
            wh_jax,
            was_1d,
            store_weights,
            n_sym=None,
            xp=xp,
            input_norm_factor=eq_norm,
        ),
        name="RDE" if not use_pilots else "RDE(PA)",
        debug_plot=debug_plot,
        check_convergence=True,
    )
