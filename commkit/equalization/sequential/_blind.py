"""Blind sequential adaptive equalizers (private).

:func:`cma` and :func:`rde`, re-exported by :mod:`commkit.equalization`.
"""

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
    """Equalize blindly with the constant-modulus algorithm (CMA), SISO or butterfly MIMO.

    CMA minimizes the Godard dispersion criterion and needs no training
    symbols: the standard blind equalizer for constant-modulus signals (PSK)
    and nearly constant-modulus ones (low-order QAM).  It recovers the signal
    up to a phase, so a carrier-phase recovery stage (e.g. Viterbi-Viterbi or
    pilot-aided) usually follows.

    With ``pilot_ref`` and ``pilot_mask`` the equalizer runs pilot-aided: the
    Godard error at the data positions, the LMS error ``pilot_ref - y`` at
    the pilot positions, which resolves the phase there while the data
    positions stay blind.  :func:`build_pilot_ref` builds the two arrays.

    Parameters
    ----------
    samples : array_like or Signal
        Input samples, ``(N,)`` or ``(C, N)``, typically at 2 samples per
        symbol (fractionally spaced).  A :class:`Signal` supplies ``sps`` and
        ``constellation``; the result then also carries ``signal``, the
        1-SPS output Signal.
    num_taps : int, default 21
        Taps per FIR filter.
    sps : int, optional
        Samples per symbol at the input, an integer.  Taken from the Signal;
        required for array input.  A value that disagrees with the Signal
        raises.
    step_size : float, default 1e-3
        Step size ``μ``.  The CMA cost is non-convex and of higher order
        than the LMS cost, so normalizing the step by the input power would
        distort the gradient: use a fixed step between 1e-5 and 1e-3.
    constellation : Constellation, optional
        Sets the Godard radius ``R² = E[|c|⁴]/E[|c|²]`` (weighted by the pmf
        of a shaped constellation).  Defaults to the Signal's
        ``constellation``; without one the target is the unit circle.
    store_weights : bool, default False
        Record the weights of every symbol in ``weights_history``.
    center_tap : int, optional
        Index of the center tap; defaults to ``num_taps // 2``.
    initial_taps : array_like, optional
        Initial tap weights instead of the center-tap identity, e.g. the
        ``weights`` of a previous stage.  Shape ``(C, C, num_taps)``, or
        ``(num_taps,)`` / ``(1, num_taps)`` for SISO.  Only for a cold start:
        a ``state`` carries its own weights.
    pilot_ref : array_like, optional
        Dense pilot reference, ``(C, N_sym)``: zeros at the data positions,
        the known symbols (on the scale of the unit-power ``constellation``,
        used as given) at the pilot positions.  Give it with ``pilot_mask``;
        :func:`build_pilot_ref` builds both.
    pilot_mask : array_like, optional
        Pilot positions, ``(N_sym,)``: 1 at a pilot, 0 elsewhere.
    pilot_gain_db : float, default 0.0
        Pilot boost in dB relative to the payload, as
        ``SingleCarrierFrame.pilot_gain_db``.  When non-zero, the received
        samples at the pilot positions are scaled down by the boost before
        the RMS normalization, so boosted pilots do not inflate the RMS and
        bias the Godard target of the data positions.  0.0 for unboosted
        pilots.
    pad_mode : {'zeros', 'edge'}, default 'zeros'
        Left padding of a cold start; see :func:`lms`.
    state : EqualizerState, optional
        Continue from ``result.state`` of a previous call with the same
        configuration; see :func:`lms`.  Pilots start at this call's first
        output symbol.

    Returns
    -------
    EqualizerResult
        ``y_hat`` (always an array), the final ``weights``, the CMA ``error``,
        ``input_norm_factor``, ``state``, ``weights_history`` with
        ``store_weights=True``, and for Signal input ``signal``, the 1-SPS
        output Signal (see :class:`EqualizerResult`).

    Notes
    -----
    **Why CMA has no inline CPR.**
    PLL requires a phase-coherent decision ``d[n]`` (nearest constellation
    point) to form the cross-product error ``Im(y * conj(d))``; but CMA
    output has an unknown phase rotation, so the decision is unreliable.
    BPS is blind, but the CMA cost does not depend on the output phase, so
    the weights drift freely in phase: BPS would track that drift, and the
    next CMA gradient step would fight the correction.  Use the stages in
    sequence instead: CMA, then ``correct_carrier_phase`` (BPS or
    Viterbi-Viterbi), then optionally ``lms`` to fine-tune.

    **Algorithm (per symbol n).**

    Steps 1 and 2 are identical to ``lms`` (sliding input window and
    butterfly filter output ``y_raw[n]``).  There is **no CPR step** -
    CMA's cost surface is phase-invariant; no radial error can drive a
    phase rotator (see above).

    3. **Godard error** - third-order radial gradient of the dispersion
       cost ``J = E[(|y|^2 - R^2)^2]``::

           e[n] = (|y[n]|^2 - R^2) * y[n]

       The Godard radius ``R^2 = E[|s|^4] / E[|s|^2]`` is computed once
       from the unit-power constellation (1 without a constellation).
       The error is purely radial: any constant phase
       rotation of ``y`` leaves ``|y|^2`` and therefore ``e`` unchanged
       up to the same rotation, so CMA cannot resolve the phase ambiguity
       it introduces.

    4. **Weight update** - steepest descent on the Godard criterion (note
       the minus sign, opposite to LMS)::

           w_{c,c'} -= mu * conj(e_c[n]) * x_{c',n}

    **Pilot-aided hybrid** (when ``pilot_ref`` and ``pilot_mask`` are set):
    at pilot positions the Godard error is replaced by the LMS pilot error
    ``e_p[n] = pilot_ref[n] - y[n]``, and the weight update takes the LMS
    sign, ``+mu``, towards the reference.  This
    resolves the phase ambiguity at pilot locations while CMA handles data
    positions blindly.

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

    r2 = _godard_radius(constellation)
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
    """Equalize blindly with the radius-directed equalizer (RDE), for multi-ring constellations.

    RDE is the CMA variant that replaces the single Godard radius by the
    nearest of the constellation's ring radii, symbol by symbol.  CMA pulls
    every symbol towards one average circle, which degrades its convergence
    when the constellation has several rings (the inner, middle and outer
    rings of 16-QAM); RDE drives each symbol towards its nearest ring, a
    gradient that matches the constellation's geometry.

    Like CMA, it is blind and recovers the channel up to a phase, so a
    carrier-phase recovery stage follows (see the Notes of :func:`cma` for
    why there is no inline CPR).

    Parameters
    ----------
    samples : array_like or Signal
        Input samples, ``(N,)`` or ``(C, N)``, typically at 2 samples per
        symbol.  A :class:`Signal` supplies ``sps`` and ``constellation``;
        the result then also carries ``signal``, the 1-SPS output Signal.
    num_taps : int, default 21
        Taps per FIR filter.
    sps : int, optional
        Samples per symbol at the input, an integer.  Taken from the Signal;
        required for array input.  A value that disagrees with the Signal
        raises.
    step_size : float, default 1e-3
        Step size ``μ``; the cost is non-convex as for :func:`cma`, so use a
        fixed step between 1e-5 and 1e-3.
    constellation : Constellation, optional
        Sets the ring radii (one ring for PSK, where RDE equals CMA).
        Defaults to the Signal's ``constellation``; without one the single
        ring is the unit circle.
    store_weights : bool, default False
        Record the weights of every symbol in ``weights_history``.
    center_tap : int, optional
        Index of the center tap; defaults to ``num_taps // 2``.
    initial_taps : array_like, optional
        Initial tap weights, as for :func:`cma`; only for a cold start.
    pilot_ref, pilot_mask : array_like, optional
        Dense pilot reference and mask, as for :func:`cma`.
    pilot_gain_db : float, default 0.0
        Pilot boost in dB, as for :func:`cma` (here it keeps boosted pilots
        from biasing the ring radii).
    pad_mode : {'zeros', 'edge'}, default 'zeros'
        Left padding of a cold start; see :func:`lms`.
    state : EqualizerState, optional
        Continue from ``result.state`` of a previous call with the same
        configuration; see :func:`lms`.  The state of a :func:`cma` call is
        accepted too: CMA pre-convergence, then RDE on the rest of the record,
        avoids RDE's slow cold start on multi-ring QAM.  Pilots start at this
        call's first output symbol.

    Returns
    -------
    EqualizerResult
        ``y_hat`` (always an array), the final ``weights``, the RDE ``error``,
        ``input_norm_factor``, ``state``, ``weights_history`` with
        ``store_weights=True``, and for Signal input ``signal``, the 1-SPS
        output Signal (see :class:`EqualizerResult`).

    Notes
    -----
    **Algorithm (per symbol n).**

    Steps 1 and 2 are identical to ``lms`` (sliding input window and
    butterfly filter output ``y[n]``).  Like ``cma``, there is no
    CPR step.

    3. **Ring selection** - choose the constellation ring radius closest to
       the current output magnitude::

           R_d[n] = argmin_{r in R_set} |r - |y[n]||
           R_set  = {|c| : c in constellation}

       ``R_set`` is the set of distinct ring radii of the unit-power
       constellation, computed once.  For 16-QAM this yields three radii
       rather than the single CMA average, eliminating the inward/outward
       pull that degrades CMA convergence on higher-order QAM.

    4. **RDE error** - same third-order form as ``cma`` but using the
       per-symbol ring radius::

           e[n] = (|y[n]|^2 - R_d[n]^2) * y[n]

    5. **Weight update** - steepest descent (same sign convention as CMA)::

           w_{c,c'} -= mu * conj(e_c[n]) * x_{c',n}

    **Pilot-aided** (with ``pilot_ref`` and ``pilot_mask``): as for
    :func:`cma`, the RDE error at a pilot position is replaced by
    ``e_p[n] = pilot_ref[n] - y[n]`` with the LMS sign ``+mu``, which
    resolves the phase there.

    **Why RDE beats CMA on high-order QAM.**  For unit-power 16-QAM the
    Godard radius is ``R² = E[|s|⁴]/E[|s|²] ≈ 1.32``, a poor stand-in for the
    three rings at ``|c| ≈ 0.45, 1.00, 1.34``: CMA pulls the inner ring
    outwards and the outer ring inwards, a lasting gradient against the
    right solution.  In RDE each symbol is attracted only to its own ring, so
    the gradient vanishes at the right solution.

    **Phase.**  Both costs depend only on ``|y|``, so any rotation of the
    output is a solution: run carrier-phase recovery after blind
    equalization.

    **Execution.**  The adaptation is sequential (each update depends on the
    previous weights), so it always runs as a compiled Numba loop on the
    CPU.  CuPy input is copied to the host once and the outputs back once;
    the result's arrays are on the input's device.
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

    radii = _rde_ring_radii(constellation)
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
    )
    return _attach_equalized_signal(result, sig, state)
