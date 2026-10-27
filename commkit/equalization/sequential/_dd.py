"""Trained and decision-directed sequential adaptive equalizers (private).

:func:`lms` and :func:`rls`, re-exported by :mod:`commkit.equalization`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...backend import ArrayType
from ...core._signal_adapter import adapt_signal, require_integer_sps
from ...core.signal import Signal
from ...logger import logger
from ...mapping.constellation import _slicer_params
from ...recovery import BPS, PLL
from .._kernels_numba import _get_numba_cpr, _get_numba_lms, _get_numba_rls
from ..result import (
    EqualizerResult,
    EqualizerState,
    _attach_equalized_signal,
    _log_equalizer_exit,
)
from ._setup import (
    _assemble_sequential,
    _dd_constellation,
    _inline_cpr,
    _prepare_sequential,
    _run_resumable,
    _Snapshot,
)

# -----------------------------------------------------------------------------
# ADAPTIVE equalization
# -----------------------------------------------------------------------------


_NO_P = np.empty((0, 0), dtype=np.complex128)  # the RLS matrix, unused by LMS


def lms(
    samples: ArrayType | Signal,
    training_symbols: ArrayType | None = None,
    *,
    num_taps: int = 21,
    sps: int | None = None,
    step_size: float = 0.01,
    constellation: Any = None,
    store_weights: bool = False,
    center_tap: int | None = None,
    initial_taps: ArrayType | None = None,
    cpr: PLL | BPS | None = None,
    state: EqualizerState | None = None,
    pad_mode: str = "zeros",
) -> EqualizerResult:
    """Equalize with the least-mean-squares (LMS) algorithm, SISO or butterfly MIMO.

    The weights adapt at every symbol (Numba, on the CPU).  For high
    throughput on slow or static channels, the frequency-domain
    :func:`block_lms` adapts once per block instead.

    With ``training_symbols`` the equalizer adapts to them first, then
    switches to decision-directed (DD) adaptation on decisions against
    ``constellation``; without them it is decision-directed throughout.
    For ``(C, N)`` input the filter is a ``(C, C, num_taps)`` butterfly:
    every output is a weighted sum of all input streams, which cancels the
    crosstalk between them.

    Parameters
    ----------
    samples : array_like or Signal
        Input samples, ``(N,)`` for SISO or ``(C, N)`` for the MIMO
        butterfly, typically at 2 samples per symbol (fractionally spaced).
        A :class:`Signal` supplies ``sps`` and ``constellation``; the result
        then also carries ``signal``, the 1-SPS output Signal.
    training_symbols : array_like, optional
        Known transmitted symbols at one sample per symbol, ``(N_train,)``
        or ``(C, N_train)``, on the scale of the unit-power
        ``constellation`` (used as given, not renormalized).
    num_taps : int, default 21
        Taps per FIR filter.  For fractionally spaced equalization
        (``sps > 1``) use at least ``4 * sps``.
    sps : int, optional
        Samples per symbol at the input, an integer: 2 (T/2-spaced) for a
        first stage, 1 for a symbol-spaced stage after frequency and phase
        recovery (with 3-11 taps); larger values work but are uncommon.  The
        output has one symbol per ``sps`` samples.  Taken from the Signal;
        required for array input.  A value that disagrees with the Signal
        raises.
    step_size : float, default 0.01
        Step size ``μ`` of plain LMS: the gradient is applied without
        normalizing by the input power, as in Haykin's *Adaptive Filter
        Theory* and most of the literature.  Stability requires
        ``0 < μ < 2 / (C · num_taps · P_x)``, with ``P_x`` the mean input
        power per tap; the inputs are normalized to unit symbol power, so
        ``1e-4`` to ``1e-2`` is a safe range to start from.  Steps near the
        bound converge faster but leave a larger steady-state misadjustment.
    constellation : Constellation, optional
        Decision constellation, at unit power (a shaped constellation brings
        its pmf).  Defaults to the Signal's ``constellation``; required for
        array input.
    store_weights : bool, default False
        Record the weights of every symbol in ``weights_history``.
    center_tap : int, optional
        Index of the center tap; defaults to ``num_taps // 2``.
    initial_taps : array_like, optional
        Initial tap weights instead of the center-tap identity, e.g. the
        ``weights`` of an earlier stage (preamble LMS, then payload RLS):
        ``(C, C, num_taps)``, or ``(num_taps,)`` / ``(1, num_taps)`` for SISO.
        Only for a cold start: a ``state`` carries its own weights.
    cpr : PLL or BPS, optional
        Inline carrier-phase recovery (from :mod:`commkit.recovery`), run
        jointly with the weight updates at every symbol; ``None`` for none.

        * ``PLL(bandwidth=, mu=, beta=, phase_init=, joint_channels=)``: a
          decision-directed loop in which the cross-product detector
          ``Im(y · conj(d))`` drives a PI filter (gains from ``bandwidth``,
          or the raw ``mu`` and ``beta``); ``phase_init`` seeds a cold start.
          It has a low noise floor and is recommended for QPSK to 64-QAM.
        * ``BPS(test_phases=, block_size=, joint_channels=)``: blind phase
          search over ``test_phases`` angles in ``[0, 2π/S)``, ``S`` the
          constellation's rotational symmetry (an interval of π/2 for QAM),
          averaged over a causal window of the last ``block_size`` symbols.
          Preferred for bursts, where the pull-in of a PLL is impractical.
          The wrapped float32 estimate rotates the output, and the causally
          ``S``-fold-unwrapped float64 accumulator is the
          ``phase_trajectory``, so float32 rounding never accumulates.

        ``joint_channels`` shares one estimate across MIMO channels (summed
        BPS metrics, or the PLL error averaged before the integrator).  A
        nested ``CycleSlip(history=, threshold=)`` predicts each phase from a
        linear fit through the last ``history`` values and removes the
        nearest multiple of ``2π/S`` from a deviation beyond ``threshold``.
    state : EqualizerState, optional
        Continue from ``result.state`` of an earlier call of this equalizer
        with the same configuration: its weights, normalization, inline CPR
        state and the input from where it stopped.  The earlier result's last
        ``state.overlap`` symbols are computed again here, so the outputs
        join into exactly one uninterrupted run.  Training symbols start at
        this call's first output symbol.
    pad_mode : {'zeros', 'edge', 'reflect'}, default 'zeros'
        Padding of a cold start before the first ``center_tap`` positions:
        ``'zeros'`` prepends zeros, ``'edge'`` repeats the first sample and
        ``'reflect'`` mirrors the record about its first sample.  The last two
        can soften the initial amplitude jump.

    Returns
    -------
    EqualizerResult
        With the fields:

        * ``y_hat``: the equalized symbols at one sample per symbol,
          ``(N_sym,)`` or ``(C, N_sym)``; always an array.
        * ``signal``: for Signal input, the output as a 1-SPS Signal whose
          reference is cut to the output symbols.
        * ``weights``: the final taps, ``(num_taps,)`` or
          ``(C, C, num_taps)``.
        * ``error``: the complex error ``e[n] = d[n] - y[n]``, shaped as
          ``y_hat``.
        * ``weights_history``: the taps at every symbol with
          ``store_weights=True``, ``(N_sym, num_taps)`` or
          ``(N_sym, C, C, num_taps)``; ``None`` otherwise.
        * ``phase_trajectory``: the per-symbol phase of the inline CPR,
          ``(N_sym,)`` or ``(C, N_sym)``: for BPS the causally
          ``S``-fold-unwrapped float64 phase, for the PLL its integrator
          state; ``None`` without ``cpr``.
        * ``num_train_symbols``: the training symbols consumed.
        * ``input_norm_factor``: the RMS factor that normalized the input.
        * ``state``: an :class:`EqualizerState`; pass it as ``state=`` to the
          next call to continue without converging again.

        The arrays are on the input's device.

    Notes
    -----
    **Algorithm (per symbol n).**

    1. **Sliding input window**: the length-T window of input channel c'
       starts at the strided sample position of the padded input::

           x_{c',n} = [x_{c'}[n*sps - T_c], ..., x_{c'}[n*sps - T_c + T - 1]]

       with ``T_c`` = ``center_tap`` (default ``T // 2``).  The delay ``T_c``
       sits inside the window, so the filter covers both pre- and
       post-cursor ISI.

    2. **Butterfly filter**: the conjugate weights applied to the windows of
       all C input channels::

           y_c_raw[n] = sum_{c'} w_{c,c'}^H * x_{c',n}

    3. **Carrier phase recovery** (if ``cpr`` is set):

       * **PLL**: the cross-product detector
         ``phi_err = Im(y_raw * conj(d_prev))`` drives a PI filter with
         gains ``mu`` and ``beta``; the accumulated phase ``phi_n`` is
         applied as ``y[n] = y_raw * exp(-j*phi_n)``.
       * **BPS**: of ``B`` candidate rotations ``exp(-j*2*pi*k/(S*B))``, ``S``
         the constellation's rotational symmetry, the one with the smallest
         summed distance to the nearest points over the last
         ``K`` = ``cpr.block_size`` symbols is chosen.  A causal ``S``-fold
         unwrap turns the argmin in ``[0, 2*pi/S)`` into the full-range
         ``phi_n``, kept in a float64 accumulator.

    4. **Decision**: the training symbol ``d[n]`` while
       ``n < len(training_symbols)``, then the nearest constellation point
       to ``y[n]``.

    5. **Error, rotated back to the tap plane**::

           e_clean[n] = d[n] - y[n]
           e_taps[n]  = e_clean[n] * exp(+j*phi_n)

       The back-rotation undoes the CPR correction, so the gradient acts in
       the taps' own phase reference.

    6. **LMS weight update** (plain, not normalized by the input power)::

           w_{c,c'} += mu * conj(e_taps_c[n]) * x_{c',n}

       Stability bound: ``0 < mu < 2 / (C * T * P_x)`` with ``P_x`` the mean
       input power per tap.  The equalizer normalizes the input to unit
       symbol power before adapting, so ``P_x ≈ 1``.

    7. **Cycle-slip correction** (with ``cpr.cycle_slip``): each channel keeps
       a circular buffer of its last ``history`` phase values, and a
       running least-squares line through the buffer predicts
       ``phi_pred_n``.  If
       ``|phi_n - phi_pred_n| > threshold``, the nearest multiple of
       ``2*pi/S`` (``S`` the rotational symmetry, 4 for QAM) is removed from
       ``phi_n``; the corrected value replaces ``phi_n`` in steps 5 and 6
       and is written into the history buffer.

    The adaptation is sequential (each update depends on the previous
    weights), so it always runs as a compiled Numba loop on the CPU.  CuPy
    input is copied to the host once and the outputs back once; the result's
    arrays are on the input's device.
    """
    signal_adapter = adapt_signal(samples, function_name="lms()")
    samples = signal_adapter.array
    sig = signal_adapter.signal
    sps = require_integer_sps(signal_adapter.resolve_fact("sps", sps), "lms()")
    constellation = signal_adapter.resolve_choice("constellation", constellation)

    inline = _inline_cpr(cpr, constellation, "lms()")

    n_train_log = training_symbols.shape[-1] if training_symbols is not None else 0
    logger.info(
        "LMS equalizer: num_taps=%s, mu=%s, sps=%s, n_train=%s%s",
        num_taps,
        step_size,
        sps,
        n_train_log,
        f", cpr={type(cpr).__name__}" if cpr is not None else "",
    )

    run = _prepare_sequential(
        samples,
        equalizer="lms",
        sps=sps,
        num_taps=num_taps,
        center_tap=center_tap,
        initial_taps=initial_taps,
        state=state,
        store_weights=store_weights,
        pad_mode=pad_mode,
        cpr=cpr,
        training_symbols=training_symbols,
    )
    constellation_np = _dd_constellation(
        constellation, "lms()", decisions=run.n_train < run.n_sym
    )
    sq_side, sq_lev_min, sq_d_grid = _slicer_params(constellation)
    slicer = (sq_lev_min, sq_d_grid, np.int32(sq_side))
    mu = np.float32(step_size)

    phase_out = None
    if inline is None:

        def segment(start: int, stop: int) -> None:
            x, train, n_train, y_out, e_out, w_hist = run.segment(start, stop)
            _get_numba_lms()(
                x,
                train,
                constellation_np,
                run.W,
                mu,
                n_train,
                run.stride,
                store_weights,
                y_out,
                e_out,
                w_hist,
                *slicer,
            )

        def snapshot() -> _Snapshot:
            return _Snapshot(weights=run.W.copy())

    else:
        cpr_args = inline
        carrier = (
            cpr_args.cold_carrier(run.num_ch)
            if state is None or state.carrier is None
            else state.carrier.copy()
        )
        phase = np.empty((run.n_sym, run.num_ch), dtype=np.float64)
        phase_out = phase

        def segment(start: int, stop: int) -> None:
            x, train, n_train, y_out, e_out, w_hist = run.segment(start, stop)
            _get_numba_cpr()(
                x,
                train,
                constellation_np,
                *cpr_args.bps_args(),
                run.W,
                False,  # LMS: no RLS state
                mu,
                _NO_P,
                np.float32(1.0),
                np.float32(0.0),
                n_train,
                np.int32(0),
                run.stride,
                store_weights,
                *cpr_args.loop_args(),
                *carrier.args(),
                y_out,
                e_out,
                phase[start:stop],
                w_hist,
                *slicer,
            )

        def snapshot() -> _Snapshot:
            return _Snapshot(weights=run.W.copy(), carrier=carrier.copy())

    resume = run.n_done
    snap = _run_resumable(run, resume, segment, snapshot)
    result = _assemble_sequential(
        run,
        equalizer="lms",
        cpr=cpr,
        resume=resume,
        snapshot=snap,
        phase_out=phase_out,
    )
    result = _log_equalizer_exit(result, name="LMS")
    return _attach_equalized_signal(result, sig, state)


def _check_rls_divergence(
    weights: Any, xp: Any, forgetting_factor: float, delta: float
) -> None:
    """Raise if RLS produced non-finite weights (silent divergence guard).

    Mirrors the ``_div_flag`` check in ``block_lms``: a single device->host sync on
    the assembled weights catches loss of positive-definiteness in the inverse
    correlation matrix P (which surfaces as NaN/Inf taps) and converts it into an
    actionable error instead of returning garbage weights.
    """
    if not bool(xp.isfinite(weights).all()):
        raise RuntimeError(
            f"RLS equalizer diverged (forgetting_factor={forgetting_factor}, "
            f"delta={delta}). RLS requires a positive-definite correlation matrix. "
            "Try increasing regularization 'delta', reducing 'forgetting_factor', "
            "or adding 'leakage' (e.g. 1e-4) to stabilize fractionally-spaced inputs."
        )


def rls(
    samples: ArrayType | Signal,
    training_symbols: ArrayType | None = None,
    *,
    num_taps: int = 21,
    sps: int | None = None,
    forgetting_factor: float = 0.99,
    delta: float = 0.01,
    leakage: float = 0.0,
    constellation: Any = None,
    store_weights: bool = False,
    center_tap: int | None = None,
    initial_taps: ArrayType | None = None,
    cpr: PLL | BPS | None = None,
    state: EqualizerState | None = None,
    pad_mode: str = "zeros",
) -> EqualizerResult:
    """Equalize with the recursive-least-squares (RLS) algorithm, SISO or butterfly MIMO.

    RLS converges faster than LMS at a higher cost per symbol:
    ``O((C·num_taps)²)`` for the rank-1 Riccati update against
    ``O(C·num_taps)`` for LMS.  It keeps the inverse correlation matrix ``P``
    of the stacked input windows.

    Parameters
    ----------
    samples : array_like or Signal
        Input samples, ``(N,)`` or ``(C, N)``.  A :class:`Signal` supplies
        ``sps`` and ``constellation``; the result then also carries
        ``signal``, the 1-SPS output Signal.
    training_symbols : array_like, optional
        Known symbols at one sample per symbol for the trained start, on the
        scale of the unit-power ``constellation`` (used as given).
    num_taps : int, default 21
        Taps per FIR filter.
    sps : int, optional
        Samples per symbol at the input; 1 is the well-conditioned case (see
        Warnings).  Taken from the Signal; required for array input.  A value
        that disagrees with the Signal raises.
    forgetting_factor : float, default 0.99
        Forgetting factor ``λ`` in ``(0, 1]``; values close to 1 give a
        longer memory.
    delta : float, default 0.01
        Regularization that seeds the inverse correlation matrix,
        ``P₀ = (1/delta) · I``: a Tikhonov (ridge) prior of ``delta`` per
        tap, since ``P`` must start positive definite before any data.  The
        input is normalized to unit symbol power, so ``delta`` is on the
        scale of the noise variance, whatever the raw amplitude.

        * A large ``delta`` (e.g. 1.0) gives a small ``P₀`` and small first
          gains ``k = P x / (λ + xᴴ P x)``: the equalizer starts slowly over
          the first tens of symbols, and the bias then disappears.  Safe
          with a long training sequence when robustness matters most.
        * A small ``delta`` (e.g. 1e-4) gives large first gains: the
          equalizer converges in few symbols, but those updates are
          dominated by the noise of the first observations and can take
          longer to settle.  Below about 1e-5, ``P`` can overflow before the
          Riccati update contracts it.

        With training symbols the default works for most symbol rates and
        SNRs: raise it towards 1.0 if the taps swing wildly during the first
        training symbols, lower it towards 1e-3 if a short training block
        converges too slowly.  Decision-directed only, prefer ``delta=1.0``
        and let the forgetting factor drive convergence.  For fractionally
        spaced input (``sps=2``, with ``leakage > 0``), 0.1 to 1.0 counters
        the growth of ``P``'s eigenvalues in the null subspace; pair it with
        ``leakage=1e-4``.
    leakage : float, default 0.0
        Weight decay ``γ`` applied to the taps at every step::

            W <- (1 - γ)·W + k·ē         # leaky weight update
            P <- (P - k·x̄ᴴP) / λ         # Riccati update, unchanged

        It damps the taps in the null subspace (the noise-only bands of a
        T/2-spaced input) without touching ``P``.  0.0 is standard RLS; for
        fractionally spaced equalization start at 1e-4 and increase it if the
        steady-state EVM stays high.
    constellation : Constellation, optional
        Decision constellation, at unit power (a shaped constellation brings
        its pmf).  Defaults to the Signal's ``constellation``; required for
        array input.
    store_weights : bool, default False
        Record the weights of every symbol in ``weights_history``.
    center_tap : int, optional
        Index of the center tap; defaults to ``num_taps // 2``.
    initial_taps : array_like, optional
        Initial tap weights instead of the center-tap identity, e.g. the
        ``weights`` of an earlier stage (preamble LMS, then payload RLS):
        ``(C, C, num_taps)``, or ``(num_taps,)`` / ``(1, num_taps)`` for SISO.
        Only for a cold start: a ``state`` carries its own weights.
    cpr : PLL or BPS, optional
        Inline carrier-phase recovery, as for :func:`lms`; ``None`` for none.
    state : EqualizerState, optional
        Continue from ``result.state`` of an earlier call with the same
        configuration, as for :func:`lms`; the state also carries ``P``.
    pad_mode : {'zeros', 'edge', 'reflect'}, default 'zeros'
        Padding of a cold start, as for :func:`lms`.

    Returns
    -------
    EqualizerResult
        The fields described for :func:`lms`.  ``y_hat`` lacks the last
        ``num_taps // 2`` symbols (``tail_trim``), and ``signal`` carries the
        reference cut to match.

    Warnings
    --------
    RLS is not recommended for fractionally spaced input (``sps > 1``).  The
    signal then fills only part of the Nyquist band, the other bands carry
    noise alone, and the input correlation matrix is nearly singular: RLS
    inverts its near-zero eigenvalues, which amplifies the high-frequency
    noise and inflates the taps.  :func:`lms` is the stable alternative.

    Notes
    -----
    ``initial_taps`` seeds the tap weights only: ``P`` starts at
    ``(1/delta) · I``.  ``state`` continues both.

    **Algorithm (per symbol n).**

    Steps 1-5 and 7 are those of :func:`lms` (input window, butterfly
    filter, carrier-phase recovery, decision, error rotated back to the tap
    plane, cycle-slip correction).  Step 6 replaces the LMS gradient with a
    rank-1 Riccati update:

    6. **RLS weight update**: with the regressor ``x_n`` that stacks the
       windows of all C inputs (``C*T`` entries) and its inverse
       auto-correlation matrix ``P`` of shape ``(C*T, C*T)``, shared by the
       C outputs::

           k   = (P @ x_n) / (lambda + x_n^H @ P @ x_n)
           P   = (P - k @ x_n^H @ P) / lambda
           w_c += k * conj(e_taps_c[n])

       where ``lambda`` = ``forgetting_factor`` and ``w_c`` stacks the taps
       of output ``c``.  ``P`` starts at ``(1/delta) * I``.  With a
       ``leakage`` ``gamma > 0`` the update becomes
       ``w = (1 - gamma) * w + k * conj(e_taps)``, which damps the taps in
       the null subspace, where ``P``'s eigenvalues grow for fractionally
       spaced (``sps > 1``) input.
    """
    signal_adapter = adapt_signal(samples, function_name="rls()")
    samples = signal_adapter.array
    sig = signal_adapter.signal
    sps = require_integer_sps(signal_adapter.resolve_fact("sps", sps), "rls()")
    constellation = signal_adapter.resolve_choice("constellation", constellation)
    if sps > 1:
        logger.warning(
            "RLS is mathematically ill-conditioned for fractionally-spaced "
            "signals (sps=%s). The noise-only null-subspace creates a "
            "singular correlation matrix, causing tap bloat. Use LMS for "
            "fractionally-spaced equalization unless heavy Tikhonov "
            "regularization is applied.",
            sps,
        )

    inline = _inline_cpr(cpr, constellation, "rls()")

    n_train_log = training_symbols.shape[-1] if training_symbols is not None else 0
    logger.info(
        "RLS equalizer: num_taps=%s, forgetting_factor=%s, delta=%.2e, "
        "leakage=%.2e, sps=%s, n_train=%s%s",
        num_taps,
        forgetting_factor,
        delta,
        leakage,
        sps,
        n_train_log,
        f", cpr={type(cpr).__name__}" if cpr is not None else "",
    )

    run = _prepare_sequential(
        samples,
        equalizer="rls",
        sps=sps,
        num_taps=num_taps,
        center_tap=center_tap,
        initial_taps=initial_taps,
        state=state,
        store_weights=store_weights,
        pad_mode=pad_mode,
        cpr=cpr,
        training_symbols=training_symbols,
    )
    # Early-halt boundary: freeze W and P once the sliding window reaches the
    # right zero-padding (last num_taps//2 symbols have contaminated windows).
    n_update_halt = max(0, run.n_sym - num_taps // 2)
    tail_trim = num_taps // 2
    if tail_trim > 0:
        logger.warning(
            "RLS tail trim: last %s symbols removed from y_hat "
            "(zero-padding contamination zone). result.signal carries the "
            "trimmed reference; trim reference arrays to match: "
            "symbols[..., :-result.tail_trim], "
            "bits[..., :-result.tail_trim * bits_per_symbol].",
            tail_trim,
        )
    constellation_np = _dd_constellation(
        constellation, "rls()", decisions=run.n_train < run.n_sym
    )
    sq_side, sq_lev_min, sq_d_grid = _slicer_params(constellation)
    slicer = (sq_lev_min, sq_d_grid, np.int32(sq_side))
    # Inverse correlation matrix: complex128 throughout (single precision loses
    # the Hermitian positive-definite property and the filter diverges).
    if state is not None and state.inverse_correlation is not None:
        P = state.inverse_correlation.copy()
    else:
        P = np.eye(run.num_ch * num_taps, dtype=np.complex128) / np.float64(delta)
    lam, leak = np.float32(forgetting_factor), np.float32(leakage)

    phase_out = None
    if inline is None:

        def segment(start: int, stop: int) -> None:
            x, train, n_train, y_out, e_out, w_hist = run.segment(start, stop)
            _get_numba_rls()(
                x,
                train,
                constellation_np,
                run.W,
                P,
                lam,
                leak,
                n_train,
                np.int32(n_update_halt - start),
                run.stride,
                store_weights,
                y_out,
                e_out,
                w_hist,
                *slicer,
            )

        def snapshot() -> _Snapshot:
            return _Snapshot(weights=run.W.copy(), inverse_correlation=P.copy())

    else:
        cpr_args = inline
        carrier = (
            cpr_args.cold_carrier(run.num_ch)
            if state is None or state.carrier is None
            else state.carrier.copy()
        )
        phase = np.empty((run.n_sym, run.num_ch), dtype=np.float64)
        phase_out = phase

        def segment(start: int, stop: int) -> None:
            x, train, n_train, y_out, e_out, w_hist = run.segment(start, stop)
            _get_numba_cpr()(
                x,
                train,
                constellation_np,
                *cpr_args.bps_args(),
                run.W,
                True,  # RLS
                np.float32(0.0),
                P,
                lam,
                leak,
                n_train,
                np.int32(n_update_halt - start),
                run.stride,
                store_weights,
                *cpr_args.loop_args(),
                *carrier.args(),
                y_out,
                e_out,
                phase[start:stop],
                w_hist,
                *slicer,
            )

        def snapshot() -> _Snapshot:
            return _Snapshot(
                weights=run.W.copy(),
                carrier=carrier.copy(),
                inverse_correlation=P.copy(),
            )

    # W and P freeze at n_update_halt, so the state resumes there at the latest.
    resume = min(run.n_done, n_update_halt)
    snap = _run_resumable(run, resume, segment, snapshot)
    result = _assemble_sequential(
        run,
        equalizer="rls",
        cpr=cpr,
        resume=resume,
        snapshot=snap,
        n_sym=n_update_halt,
        phase_out=phase_out,
        tail_trim=tail_trim,
    )
    result = _log_equalizer_exit(result, name="RLS")
    _check_rls_divergence(result.weights, run.xp, forgetting_factor, delta)
    return _attach_equalized_signal(result, sig, state)
