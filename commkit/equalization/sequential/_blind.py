"""Blind sequential adaptive equalizers: cma, rde."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...backend import ArrayType, to_device
from ...core._signal_adapter import adapt_signal, require_integer_sps
from ...core.signal import Signal
from ...logger import logger
from .._common import _godard_radius, _rde_ring_radii
from .._kernels_numba import (
    _get_numba_cma,
    _get_numba_pa_cma,
    _get_numba_pa_rde,
    _get_numba_rde,
)
from ..result import (
    EqualizerResult,
    EqualizerState,
    _attach_equalized_signal,
    _log_equalizer_exit,
)
from ._setup import (
    _assemble_sequential,
    _prepare_sequential,
    _run_resumable,
    _Snapshot,
)


def _check_pilots(
    samples: Any,
    sps: int,
    pilot_ref: Any,
    pilot_mask: Any,
    function_name: str,
    state: Any = None,
) -> tuple[Any, Any]:
    """Validate the pilot reference and mask before any work is done.

    ``pilot_ref`` is ``(C, N_sym)`` (``(N_sym,)`` for SISO), ``pilot_mask``
    ``(N_sym,)``, for the output symbols of this call (a ``state`` adds its
    pending input); one without the other raises.
    """
    if (pilot_ref is None) != (pilot_mask is None):
        raise ValueError(
            f"{function_name}: pilot_ref and pilot_mask must be given together."
        )
    if pilot_ref is None:
        return None, None
    num_ch = 1 if samples.ndim == 1 else samples.shape[0]
    offset = 0 if state is None else state.pending.shape[-1] - state.lead
    n_sym = (offset + samples.shape[-1]) // sps
    if pilot_ref.ndim == 1:
        pilot_ref = pilot_ref[None, :]
    if tuple(pilot_ref.shape) != (num_ch, n_sym):
        raise ValueError(
            f"{function_name}: pilot_ref must have shape ({num_ch}, {n_sym}) "
            f"(channels, symbols), got {tuple(pilot_ref.shape)}."
        )
    if tuple(np.shape(pilot_mask)) != (n_sym,):
        raise ValueError(
            f"{function_name}: pilot_mask must have shape ({n_sym},), got "
            f"{tuple(np.shape(pilot_mask))}."
        )
    return pilot_ref, pilot_mask


# -----------------------------------------------------------------------------
# BLIND equalization
# -----------------------------------------------------------------------------


def cma(
    samples: ArrayType | Signal,
    *,
    num_taps: int = 21,
    sps: int | None = None,
    step_size: float = 1e-3,
    constellation: Any = None,
    store_weights: bool = False,
    center_tap: int | None = None,
    initial_taps: ArrayType | None = None,
    pilot_ref: ArrayType | None = None,
    pilot_mask: np.ndarray | None = None,
    pilot_gain_db: float = 0.0,
    pad_mode: str = "zeros",
    state: EqualizerState | None = None,
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
       from the unit-power constellation (1 without a constellation).  The error is purely radial: any constant phase
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
        A :class:`Signal` supplies ``sps`` and ``constellation``; the result
        then also carries ``signal``, the 1-SPS output Signal.
    num_taps : int, default 21
        Number of equalizer taps per FIR filter.
    sps : int, optional
        Samples per symbol at the input, an integer.  Taken from the Signal;
        required for array input.  A value that disagrees with the Signal
        raises.
    step_size : float, default 1e-3
        CMA step size (mu). Unlike LMS, CMA's cost surface is non-convex and
        higher-order, so input-power normalization distorts the gradient geometry.
        Use a fixed step size in the range 1e-5 to 1e-3 for stability.
    constellation : Constellation, optional
        Sets the Godard radius ``R2 = E[|c|^4]/E[|c|^2]`` (pmf-weighted for a
        shaped constellation).  Defaults to the Signal's ``constellation``;
        without one the target is the unit circle.
    store_weights : bool, default False
        If True, stores weight trajectory.
    center_tap : int, optional
        Index of the center tap. If None, defaults to ``num_taps // 2``.
    initial_taps : array_like, optional
        Initial tap weights instead of the center-tap identity, e.g. the
        ``weights`` of a previous stage.  Shape ``(C, C, num_taps)``, or
        ``(num_taps,)`` / ``(1, num_taps)`` for SISO.  Only for a cold start:
        a ``state`` carries its own weights.
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
    pad_mode : {'zeros', 'edge'}, default 'zeros'
        Left padding of a cold start; see :func:`lms`.
    state : EqualizerState, optional
        Continue from ``result.state`` of a previous call with the same
        configuration; see :func:`lms`.  Pilots start at this call's first
        output symbol.

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
    sps = require_integer_sps(signal_adapter.resolve_fact("sps", sps), "cma()")
    constellation = signal_adapter.resolve_choice("constellation", constellation)

    pilot_ref, pilot_mask = _check_pilots(
        signal_adapter.array, sps, pilot_ref, pilot_mask, "cma()", state
    )
    use_pilots = pilot_ref is not None
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

    r2, _c_ps = _godard_radius(constellation)
    logger.debug("CMA R2: %.4f", r2)

    # RMS-normalize samples to unit symbol-rate power (CMA has no training)
    run = _prepare_sequential(
        samples,
        equalizer="cma",
        sps=sps,
        num_taps=num_taps,
        center_tap=center_tap,
        initial_taps=initial_taps,
        state=state,
        store_weights=store_weights,
        pad_mode=pad_mode,
        pilot_mask=pilot_mask if use_pilots else None,
        pilot_gain_db=pilot_gain_db,
    )
    target = np.float32(r2)
    mu = np.float32(step_size)
    if use_pilots:
        pref = np.ascontiguousarray(to_device(pilot_ref, "cpu"), dtype=np.complex64)
        if _c_ps is not None:
            pref = (pref * _c_ps).astype(np.complex64)
        pmask = np.ascontiguousarray(pilot_mask, dtype=np.uint8)

    def segment(start: int, stop: int) -> None:
        x, _, _, y_out, e_out, w_hist = run.segment(start, stop)
        args = (x, run.W, mu, target, run.stride, store_weights, y_out, e_out, w_hist)
        if use_pilots:
            _get_numba_pa_cma()(
                *args,
                np.ascontiguousarray(pref[:, start:]),
                np.ascontiguousarray(pmask[start:]),
            )
        else:
            _get_numba_cma()(*args)

    resume = run.n_done
    snap = _run_resumable(run, resume, segment, lambda: _Snapshot(weights=run.W.copy()))
    result = _log_equalizer_exit(
        _assemble_sequential(
            run, equalizer="cma", cpr=None, resume=resume, snapshot=snap
        ),
        name="CMA" if not use_pilots else "CMA(PA)",
        check_convergence=True,
    )
    return _attach_equalized_signal(result, sig, state)


def rde(
    samples: ArrayType | Signal,
    *,
    num_taps: int = 21,
    sps: int | None = None,
    step_size: float = 1e-3,
    constellation: Any = None,
    store_weights: bool = False,
    center_tap: int | None = None,
    initial_taps: ArrayType | None = None,
    pilot_ref: ArrayType | None = None,
    pilot_mask: np.ndarray | None = None,
    pilot_gain_db: float = 0.0,
    pad_mode: str = "zeros",
    state: EqualizerState | None = None,
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
        A :class:`Signal` supplies ``sps`` and ``constellation``; the result
        then also carries ``signal``, the 1-SPS output Signal.
    num_taps : int, default 21
        Number of equalizer taps per FIR filter.
    sps : int, optional
        Samples per symbol at the input, an integer.  Taken from the Signal;
        required for array input.  A value that disagrees with the Signal
        raises.
    step_size : float, default 1e-3
        RDE step size (mu). Same non-convex gradient geometry as CMA; use a
        fixed step in the range 1e-5 to 1e-3 for stability.
    constellation : Constellation, optional
        Sets the ring radii (one ring for PSK, where RDE equals CMA).
        Defaults to the Signal's ``constellation``; without one the single
        ring is the unit circle.
    store_weights : bool, default False
        If True, stores weight trajectory in ``result.weights_history``.
    center_tap : int, optional
        Index of the center tap. Defaults to ``num_taps // 2``.
    initial_taps : array_like, optional
        Initial tap weights instead of the center-tap identity, e.g. the
        ``weights`` of a previous stage.  Shape ``(C, C, num_taps)``, or
        ``(num_taps,)`` / ``(1, num_taps)`` for SISO.  Only for a cold start:
        a ``state`` carries its own weights.
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
    pad_mode : {'zeros', 'edge'}, default 'zeros'
        Left padding of a cold start; see :func:`lms`.
    state : EqualizerState, optional
        Continue from ``result.state`` of a previous call with the same
        configuration; see :func:`lms`.  Pilots start at this call's first
        output symbol.

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
    sps = require_integer_sps(signal_adapter.resolve_fact("sps", sps), "rde()")
    constellation = signal_adapter.resolve_choice("constellation", constellation)

    pilot_ref, pilot_mask = _check_pilots(
        signal_adapter.array, sps, pilot_ref, pilot_mask, "rde()", state
    )
    use_pilots = pilot_ref is not None
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

    radii, _c_ps = _rde_ring_radii(constellation)
    logger.debug("RDE radii: %s", ", ".join(f"{r:.4f}" for r in radii))

    # RMS-normalize samples to unit symbol-rate power (RDE has no training)
    run = _prepare_sequential(
        samples,
        equalizer="rde",
        sps=sps,
        num_taps=num_taps,
        center_tap=center_tap,
        initial_taps=initial_taps,
        state=state,
        store_weights=store_weights,
        pad_mode=pad_mode,
        pilot_mask=pilot_mask if use_pilots else None,
        pilot_gain_db=pilot_gain_db,
    )
    target = np.ascontiguousarray(radii, dtype=np.float32)
    mu = np.float32(step_size)
    if use_pilots:
        pref = np.ascontiguousarray(to_device(pilot_ref, "cpu"), dtype=np.complex64)
        if _c_ps is not None:
            pref = (pref * _c_ps).astype(np.complex64)
        pmask = np.ascontiguousarray(pilot_mask, dtype=np.uint8)

    def segment(start: int, stop: int) -> None:
        x, _, _, y_out, e_out, w_hist = run.segment(start, stop)
        args = (x, run.W, mu, target, run.stride, store_weights, y_out, e_out, w_hist)
        if use_pilots:
            _get_numba_pa_rde()(
                *args,
                np.ascontiguousarray(pref[:, start:]),
                np.ascontiguousarray(pmask[start:]),
            )
        else:
            _get_numba_rde()(*args)

    resume = run.n_done
    snap = _run_resumable(run, resume, segment, lambda: _Snapshot(weights=run.W.copy()))
    result = _log_equalizer_exit(
        _assemble_sequential(
            run, equalizer="rde", cpr=None, resume=resume, snapshot=snap
        ),
        name="RDE" if not use_pilots else "RDE(PA)",
        check_convergence=True,
    )
    return _attach_equalized_signal(result, sig, state)
