"""Blind sequential adaptive equalizers: cma, rde."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...backend import ArrayType, to_device
from ...core._signal_adapter import adapt_signal, require_integer_sps
from ...core.signal import Signal
from ...logger import logger
from .._kernels_numba import (
    _get_numba_cma,
    _get_numba_pa_cma,
    _get_numba_pa_rde,
    _get_numba_rde,
)
from ..result import EqualizerResult, _attach_equalized_signal, _log_equalizer_exit
from ._setup import _assemble_sequential, _prepare_sequential

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
    center_tap: int | None = None,
    w_init: ArrayType | None = None,
    pilot_ref: ArrayType | None = None,
    pilot_mask: np.ndarray | None = None,
    pilot_gain_db: float = 0.0,
    pmf: Any | None = None,
    input_norm_factor: float | np.ndarray | None = None,
    samples_prefix: ArrayType | None = None,
    pad_mode: str = "zeros",
) -> EqualizerResult:
    """
    Constant Modulus Algorithm blind equalizer with butterfly MIMO support.

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
    center_tap : int, optional
        Index of the center tap. If None, defaults to ``num_taps // 2``.
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

    Returns
    -------
    EqualizerResult
        Equalized symbols, final weights, CMA error history, and optionally
        weight trajectory.  ``input_norm_factor`` field is populated.  When
        ``samples`` is a :class:`Signal`, ``y_hat`` is a new :class:`Signal`
        at the symbol rate (``sampling_rate = symbol_rate``).

    Notes
    -----
    The adaptation is inherently sequential (each update depends on the
    previous weights), so it always runs as a compiled Numba loop on the CPU.
    CuPy input is copied to the host once and the outputs are copied back
    once; the result arrays live on the input's device.
    """
    signal_adapter = adapt_signal(samples, function_name="cma()")
    samples = signal_adapter.array
    sig = signal_adapter.signal
    if sig is not None:
        sps = require_integer_sps(signal_adapter.resolve_required("sps", sps), "cma()")
    if sps is None:
        sps = 2

    use_pilots = pilot_ref is not None and pilot_mask is not None
    logger.info(
        "CMA equalizer: num_taps=%s, mu=%s, sps=%s, pilot_aided=%s, pilot_gain_db=%s",
        num_taps,
        step_size,
        sps,
        use_pilots,
        pilot_gain_db,
    )
    if sps > 1:
        logger.warning(
            "CMA output y_hat is at 1 SPS (symbol rate). "
            "Update sampling_rate = symbol_rate after applying this equalizer."
        )

    # Compute R2 and PS-QAM scale factor from the Godard constellation.
    _c_ps = None  # 1/sqrt(E_PS) scale factor; None for uniform modulation
    if modulation is not None and order is not None:
        from ...mapping.gray import _gray_points

        const = _gray_points(modulation, order, unipolar=unipolar)
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

    # RMS-normalize samples to unit symbol-rate power (CMA has no training)
    run = _prepare_sequential(
        samples,
        sps=sps,
        num_taps=num_taps,
        center_tap=center_tap,
        w_init=w_init,
        store_weights=store_weights,
        input_norm_factor=input_norm_factor,
        samples_prefix=samples_prefix,
        pad_mode=pad_mode,
        pilot_mask=pilot_mask if use_pilots else None,
        pilot_gain_db=pilot_gain_db,
    )
    args = (
        run.x,
        run.W,
        np.float32(step_size),
        np.float32(r2),
        run.stride,
        store_weights,
        run.y_out,
        run.e_out,
        run.w_hist,
    )
    if use_pilots:
        pref = np.ascontiguousarray(to_device(pilot_ref, "cpu"), dtype=np.complex64)
        if _c_ps is not None:
            pref = (pref * _c_ps).astype(np.complex64)
        pmask = np.ascontiguousarray(pilot_mask, dtype=np.uint8)
        _get_numba_pa_cma()(*args, pref, pmask)
    else:
        _get_numba_cma()(*args)
    result = _log_equalizer_exit(
        _assemble_sequential(run),
        name="CMA" if not use_pilots else "CMA(PA)",
        check_convergence=True,
    )
    return _attach_equalized_signal(result, sig)


def rde(
    samples: ArrayType | Signal,
    num_taps: int = 21,
    sps: int | None = None,
    step_size: float = 1e-3,
    modulation: str | None = None,
    order: int | None = None,
    unipolar: bool = False,
    store_weights: bool = False,
    center_tap: int | None = None,
    w_init: ArrayType | None = None,
    pilot_ref: ArrayType | None = None,
    pilot_mask: np.ndarray | None = None,
    pilot_gain_db: float = 0.0,
    pmf: Any | None = None,
    input_norm_factor: float | np.ndarray | None = None,
    samples_prefix: ArrayType | None = None,
    pad_mode: str = "zeros",
) -> EqualizerResult:
    """
    Radius Directed Equalizer (RDE) - blind equalizer for multi-ring constellations.

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
    center_tap : int, optional
        Index of the center tap. Defaults to ``num_taps // 2``.
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

    **Execution:** the adaptation is inherently sequential (each update
    depends on the previous weights), so it always runs as a compiled Numba
    loop on the CPU.
    CuPy input is copied to the host once and the outputs are copied back
    once; the result arrays live on the input's device.
    """
    signal_adapter = adapt_signal(samples, function_name="rde()")
    samples = signal_adapter.array
    sig = signal_adapter.signal
    if sig is not None:
        sps = require_integer_sps(signal_adapter.resolve_required("sps", sps), "rde()")
    if sps is None:
        sps = 2

    use_pilots = pilot_ref is not None and pilot_mask is not None
    logger.info(
        "RDE equalizer: num_taps=%s, mu=%s, sps=%s, pilot_aided=%s, pilot_gain_db=%s",
        num_taps,
        step_size,
        sps,
        use_pilots,
        pilot_gain_db,
    )
    if sps > 1:
        logger.warning(
            "RDE output y_hat is at 1 SPS (symbol rate). "
            "Update sampling_rate = symbol_rate after applying this equalizer."
        )

    # Compute unique ring radii from constellation.
    # For constant-modulus signals (PSK) this degenerates to a single radius,
    # making RDE identical to CMA.
    _c_ps = None  # 1/sqrt(E_PS) scale factor; None for uniform modulation
    if modulation is not None and order is not None:
        from ...mapping.gray import _gray_points

        const = _gray_points(modulation, order, unipolar=unipolar)
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

    # RMS-normalize samples to unit symbol-rate power (RDE has no training)
    run = _prepare_sequential(
        samples,
        sps=sps,
        num_taps=num_taps,
        center_tap=center_tap,
        w_init=w_init,
        store_weights=store_weights,
        input_norm_factor=input_norm_factor,
        samples_prefix=samples_prefix,
        pad_mode=pad_mode,
        pilot_mask=pilot_mask if use_pilots else None,
        pilot_gain_db=pilot_gain_db,
    )
    args = (
        run.x,
        run.W,
        np.float32(step_size),
        np.ascontiguousarray(radii, dtype=np.float32),
        run.stride,
        store_weights,
        run.y_out,
        run.e_out,
        run.w_hist,
    )
    if use_pilots:
        pref = np.ascontiguousarray(to_device(pilot_ref, "cpu"), dtype=np.complex64)
        if _c_ps is not None:
            pref = (pref * _c_ps).astype(np.complex64)
        pmask = np.ascontiguousarray(pilot_mask, dtype=np.uint8)
        _get_numba_pa_rde()(*args, pref, pmask)
    else:
        _get_numba_rde()(*args)
    result = _log_equalizer_exit(
        _assemble_sequential(run),
        name="RDE" if not use_pilots else "RDE(PA)",
        check_convergence=True,
    )
    return _attach_equalized_signal(result, sig)
