"""Decision-directed sequential adaptive equalizers: lms, rls."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...backend import ArrayType
from ...core._signal_adapter import adapt_signal, require_integer_sps
from ...core.signal import Signal
from ...logger import logger
from ...mapping.gray import _square_qam_slicer_params
from ...recovery import BPS, PLL
from .._kernels_numba import (
    _get_numba_lms,
    _get_numba_lms_cpr,
    _get_numba_rls,
    _get_numba_rls_cpr,
)
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
    """
    Least Mean Squares adaptive equalizer with butterfly MIMO support.

    The weights adapt every symbol (CPU, Numba).  For high throughput on slow
    or static channels, the frequency-domain :func:`block_lms` adapts once per
    block instead.

    Supports data-aided (training) and decision-directed (DD) modes.
    When ``training_symbols`` are provided, the equalizer uses them for the
    initial convergence phase, then switches to DD mode, slicing to
    ``constellation``.

    For MIMO inputs ``(C, N)``, a butterfly ``(C, C, num_taps)`` filter
    structure is used so each output is a weighted sum of all input streams,
    enabling cross-channel interference cancellation.

    Algorithm (per symbol n)
    ------------------------
    1. **Sliding input window** - the length-T tap vector for output channel c
       is drawn from the padded input at the strided sample position::

           x_{c',n} = [x_{c'}[n*sps - T_c], ..., x_{c'}[n*sps - T_c + T - 1]]

       where ``T_c`` = ``center_tap`` (default ``T // 2``).  The causal delay
       ``T_c`` is absorbed into the tap vector so the filter can model both
       pre- and post-cursor ISI.

    2. **Butterfly filter** - cross-correlate conjugate weights with the input
       across all C input channels::

           y_c_raw[n] = sum_{c'} w_{c,c'}^H * x_{c',n}

    3. **Carrier phase recovery** (if ``cpr`` is set):

       * **PLL** - cross-product phase detector
         ``phi_err = Im(y_raw * conj(d_prev))``
         drives a PI integrator with gains ``K_p``, ``K_i``; accumulated
         phase ``phi_n`` is applied as ``y[n] = y_raw * exp(-j*phi_n)``.
       * **BPS** - ``B`` candidate rotations ``exp(-j*2*pi*k/(S*B))`` are
         tested, ``S`` the constellation's rotational symmetry; the one
         minimising the summed nearest-constellation distance over the
         trailing ``K`` = ``cpr.block_size`` symbols is chosen.  A causal
         ``S``-fold unwrap converts the ``[0, 2*pi/S)`` argmin to full-range
         ``phi_n`` stored in a float64 accumulator.

    4. **Decision** - training symbol ``d[n]`` (DA phase, while
       ``n < len(training_symbols)``) or nearest-constellation hard decision on
       ``y[n]`` (DD phase thereafter).

    5. **Error and tap-plane back-rotation**::

           e_clean[n] = d[n] - y[n]
           e_taps[n]  = e_clean[n] * exp(+j*phi_n)

       The back-rotation undoes the CPR correction so the gradient operates in
       the original tap space.

    6. **LMS weight update** (plain, no input-power normalisation)::

           w_{c,c'} += mu * conj(e_taps_c[n]) * x_{c',n}

       Stability bound: ``0 < mu < 2 / (C * T * P_x)`` where ``P_x`` is the
       mean per-tap input power.  The equalizer normalises inputs to unit
       symbol-rate power before adaptation, so ``P_x ≈ 1``.

    7. **Cycle-slip correction** (if ``cpr.cycle_slip`` is set) - a
       circular buffer of ``history`` past phase values is
       maintained per channel.  An online least-squares linear fit over the
       buffer predicts ``phi_pred_n``.  If
       ``|phi_n - phi_pred_n| > threshold``,
       ``phi_n`` is snapped to the nearest ``2*pi/sym``
       multiple (``sym`` = constellation symmetry order, 4 for QAM/QPSK); the
       corrected value replaces ``phi_n`` in steps 5 and 6 and is written into
       the history buffer.

    Parameters
    ----------
    samples : array_like or Signal
        Input signal samples. Shape: ``(N_samples,)`` for SISO or
        ``(C, N_samples)`` for MIMO butterfly equalization.
        Typically at 2 samples/symbol for fractionally-spaced equalization.
        A :class:`Signal` supplies ``sps`` and ``constellation``; the result
        then also carries ``signal``, the 1-SPS output Signal.
    training_symbols : array_like, optional
        Known transmitted symbols (at symbol rate, 1 SPS), on the scale of
        the unit-power ``constellation`` (used as given, not renormalized).
        Shape: ``(N_train,)`` for SISO or ``(C, N_train)`` for MIMO.
        Without them the equalizer is decision-directed throughout.
    num_taps : int, default 21
        Number of equalizer taps per FIR filter. For fractionally-spaced
        equalization (sps > 1), use at least ``4 * sps`` taps.
    sps : int, optional
        Samples per symbol at the input, an integer: 2 (T/2-spaced) for a
        first stage, 1 for a symbol-spaced stage after FOE and CPR (use 3-11
        taps), more is accepted but uncommon.  The output is one symbol per
        ``sps`` samples.  Taken from the Signal; required for array input.
        A value that disagrees with the Signal raises.
    step_size : float, default 0.01
        Plain LMS step size (mu). The gradient is applied directly without
        input-power normalization, matching the convention in Haykin's
        *Adaptive Filter Theory* and most published papers.  Stability
        requires ``0 < mu < 2 / (C * num_taps * P_x)`` where ``P_x`` is the
        mean per-tap input power.  Because inputs are normalized to unit
        symbol-rate power by default, a safe starting range for typical
        settings is ``1e-4`` to ``1e-2``.  Values closer to the upper bound
        converge faster but produce higher steady-state misadjustment.
    constellation : Constellation, optional
        Decision constellation for the slicer, unit power (a shaped
        constellation carries its pmf).  Defaults to the Signal's
        ``constellation``; required for array input.
    store_weights : bool, default False
        If True, stores weight trajectory in ``weights_history``.
    center_tap : int, optional
        Index of the center tap. If None, defaults to ``num_taps // 2``.
    initial_taps : array_like, optional
        Initial tap weights instead of the center-tap identity, e.g. the
        ``weights`` of a previous stage (preamble LMS -> payload RLS).  Shape
        ``(C, C, num_taps)``, or ``(num_taps,)`` / ``(1, num_taps)`` for SISO.
        Only for a cold start: a ``state`` carries its own weights.
    cpr : PLL or BPS, optional
        Inline carrier phase recovery (``commkit.recovery``), run jointly
        with the weight updates at every symbol; ``None`` disables it.

        * ``PLL(bandwidth=, mu=, beta=, phase_init=, joint_channels=)`` -
          decision-directed loop: the cross-product detector
          ``Im(y · conj(d))`` drives a PI loop (gains from ``bandwidth`` or
          raw ``mu`` / ``beta``).  ``phase_init`` seeds a cold start.  Low
          noise floor; recommended for QPSK through 64-QAM.
        * ``BPS(test_phases=, block_size=, joint_channels=)`` - blind phase
          search over ``test_phases`` angles in ``[0, 2π/S)``, ``S`` the
          constellation's rotational symmetry (π/2 for QAM), averaged over a causal window of the last
          ``block_size`` symbols.  Preferred for bursts where PLL pull-in is
          impractical.  The wrapped float32 estimate rotates the output; the
          causally ``S``-fold-unwrapped float64 accumulator is the
          ``phase_trajectory``, so float32 rounding never accumulates.

        ``joint_channels`` shares one estimate across MIMO channels (summed
        BPS metrics, or the PLL error averaged before the integrator).  A
        nested ``CycleSlip(history=, threshold=)`` predicts each phase from a
        linear fit through the last ``history`` values and snaps a deviation
        beyond ``threshold`` to the nearest ``2π/4`` multiple (``π`` for
        2-fold constellations).
    state : EqualizerState, optional
        Continue from ``result.state`` of a previous call of this equalizer
        with the same configuration: weights, normalization, inline CPR state
        and the input from where the previous call left off.  The previous
        result's last ``state.overlap`` symbols are recomputed here, so the
        outputs stitch to exactly one uninterrupted run.  Training symbols
        start at this call's first output symbol.
    pad_mode : {'zeros', 'edge'}, default 'zeros'
        Left padding of a cold start: ``'zeros'`` (default) prepends
        ``center_tap`` zeros, ``'edge'`` replicates the first sample, which
        can soften the initial amplitude jump.

    Returns
    -------
    EqualizerResult
        Result container with the following fields:

        * ``y_hat`` - equalized symbol estimates, shape ``(N_sym,)`` SISO
          or ``(C, N_sym)`` MIMO, at 1 SPS (symbol rate); always an array.
        * ``signal`` - for Signal input, the output as a 1-SPS Signal with
          its reference cut to the output symbols.
        * ``weights`` - final tap-weight tensor, shape ``(num_taps,)`` SISO
          or ``(C, C, num_taps)`` MIMO.
        * ``error`` - complex error signal ``e[n] = d[n] - y[n]``, same
          shape as ``y_hat``.
        * ``weights_history`` - tap weights recorded at each symbol (only
          when ``store_weights=True``); shape ``(N_sym, num_taps)`` SISO or
          ``(N_sym, C, C, num_taps)`` MIMO.  ``None`` otherwise.
        * ``phase_trajectory`` - accumulated per-symbol phase estimates,
          shape ``(N_sym,)`` SISO or ``(C, N_sym)`` MIMO.  For BPS, this
          is the causal ``S``-fold-unwrapped float64 phase.  For PLL, it is
          the PI integrator state accumulated over all symbols.  ``None``
          when ``cpr=None``.
        * ``num_train_symbols`` - number of training symbols consumed
          (data-aided phase).
        * ``input_norm_factor`` - the RMS factor used to normalize inputs.
        * ``state`` - :class:`EqualizerState`; pass as ``state=`` to the
          next call to continue without a re-convergence transient.

        Arrays reside on the same device as the input (NumPy CPU or CuPy
        GPU).

    Notes
    -----
    The adaptation is inherently sequential (each update depends on the
    previous weights), so it always runs as a compiled Numba loop on the CPU.
    CuPy input is copied to the host once and the outputs are copied back
    once; the result arrays live on the input's device.
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
    sq_side, sq_lev_min, sq_d_grid = _square_qam_slicer_params(constellation_np)
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
            _get_numba_lms_cpr()(
                x,
                train,
                constellation_np,
                *cpr_args.bps_args(),
                run.W,
                mu,
                n_train,
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


def _check_rls_divergence(weights, xp, forgetting_factor, delta):
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
    """
    Recursive Least Squares adaptive equalizer with butterfly MIMO support.

    RLS converges faster than LMS at the cost of higher per-symbol
    complexity (O(num_taps²) for the rank-1 Riccati update vs O(num_taps)
    for LMS).  It maintains an inverse correlation matrix P per output stream.

    Algorithm (per symbol n)
    ------------------------
    Steps 1-5 and 7 are identical to ``lms`` (input windowing, butterfly
    filter output, carrier phase recovery, decision, error + tap-plane
    back-rotation, and cycle-slip correction).  Step 6 replaces the plain LMS
    gradient with a rank-1 Riccati update:

    6. **RLS weight update** - for each output channel c, maintaining the
       inverse input auto-correlation matrix ``P_c`` of shape ``(T, T)``::

           k_c        = (P_c @ x_{c,n}) / (lambda + x_{c,n}^H @ P_c @ x_{c,n})
           P_c        = (P_c - k_c @ x_{c,n}^H @ P_c) / lambda
           w_{c,c'}  += k_c * conj(e_taps_c[n])

       where ``lambda`` = ``forgetting_factor``.  ``P_c`` is initialised to
       ``(1/delta) * I`` (``delta`` parameter).  With ``leakage`` ``gamma > 0``
       the weight update becomes
       ``w = (1 - gamma) * w + k * conj(e_taps)``, which exponentially
       suppresses tap energy in frequency-null subspaces and prevents the
       eigenvalue blow-up that afflicts ``P`` for fractionally-spaced
       (sps > 1) inputs.

    Parameters
    ----------
    samples : array_like or Signal
        Input signal samples. Shape: ``(N_samples,)`` or ``(C, N_samples)``.
        A :class:`Signal` supplies ``sps`` and ``constellation``; the result
        then also carries ``signal``, the 1-SPS output Signal.
    training_symbols : array_like, optional
        Known symbols for data-aided adaptation (at symbol rate, 1 SPS), on
        the scale of the unit-power ``constellation`` (used as given).
    num_taps : int, default 21
        Number of equalizer taps per FIR filter.
    sps : int, optional
        Samples per symbol at the input (1 is the well-conditioned case).
        Taken from the Signal; required for array input.  A value that
        disagrees with the Signal raises.
    forgetting_factor : float, default 0.99
        RLS forgetting factor (lambda). Range: (0, 1].
        Values close to 1 give longer memory.
    delta : float, default 0.01
        Tikhonov regularisation coefficient that seeds the inverse correlation
        matrix as ``P₀ = (1/delta) · I``.

        **Physical interpretation.**  RLS recursively refines a running estimate
        of ``Rxx⁻¹``, where ``Rxx = E[x xᴴ]`` is the input auto-correlation
        matrix.  Before any data have been observed, ``P`` must be initialised
        to some positive-definite matrix.  Choosing ``P₀ = (1/delta) · I``
        is equivalent to assuming a fictitious prior with `delta` units of
        regularisation energy per tap - a textbook Tikhonov (ridge) prior on
        the tap vector with regularisation parameter ``delta``.

        **Effect on convergence.**

        * **Large delta** (e.g. 1.0): ``P₀`` is small -> the first Kalman gain
          vectors ``k = P x / (λ + xᴴ P x)`` are small -> the equalizer adapts
          sluggishly over the first tens of symbols.  Once enough data are seen
          the bias disappears, so this is safe when a long training sequence is
          available and numerical robustness is the priority.
        * **Small delta** (e.g. 1e-4): ``P₀ = (1/delta) · I`` is a large matrix
          -> ``k`` is large for the first symbols -> the equalizer converges in
          very few symbols but the weight update is dominated by noise on those
          first few observations, potentially requiring more symbols to settle.
          Extremely small values (< 1e-5) risk numerical overflow of ``P``
          before the Riccati update can contract it.

        **Sensitivity to signal power.**  The code normalises input samples to
        unit symbol-rate power before running the Riccati recursion, so
        ``delta`` is expressed in normalised units (≈ noise variance scale) and
        is not sensitive to the raw signal amplitude.

        **Practical guidelines.**

        * **Training-aided mode** (``training_symbols`` provided): the default
          ``delta=0.01`` works well for most symbol rates and SNR regimes.
          Increase toward 1.0 if tap weights oscillate wildly during the first
          training symbols; decrease toward 1e-3 if convergence is slow and
          your training block is short.
        * **Decision-directed (DD) only**: prefer ``delta=1.0`` and rely on the
          forgetting factor to drive convergence, keeping ``P`` bounded.
        * **Fractionally-spaced signals** (``sps=2``, with ``leakage > 0``):
          larger ``delta`` (0.1-1.0) helps counteract the positive-feedback
          tendency of the unbounded ``P`` eigenvalues in the null sub-space.
          Pair with ``leakage=1e-4`` for structural stability.
    leakage : float, default 0.0
        Weight-decay coefficient (γ) applied to the tap vector at every step::

            W <- (1 - γ)·W + k·ē         # leaky weight update
            P <- (P - k·x̄ᴴP) / λ         # standard Riccati (unchanged)

        Weight decay exponentially suppresses tap weights in the null subspace -
        noise-only frequency bands that arise in T/2-spaced (sps=2) signals -
        without inflating ``P``'s eigenvalues.
        A value of ``0.0`` (default) gives standard RLS.
        For fractionally-spaced equalization start with ``leakage=1e-4`` and
        increase if steady-state EVM remains high.
    constellation : Constellation, optional
        Decision constellation for the slicer, unit power (a shaped
        constellation carries its pmf).  Defaults to the Signal's
        ``constellation``; required for array input.
    store_weights : bool, default False
        If True, stores weight trajectory.
    center_tap : int, optional
        Index of the center tap. If None, defaults to ``num_taps // 2``.
    initial_taps : array_like, optional
        Initial tap weights instead of the center-tap identity, e.g. the
        ``weights`` of a previous stage (preamble LMS -> payload RLS).  Shape
        ``(C, C, num_taps)``, or ``(num_taps,)`` / ``(1, num_taps)`` for SISO.
        Only for a cold start: a ``state`` carries its own weights.
    cpr : PLL or BPS, optional
        Inline carrier phase recovery (``commkit.recovery``), run jointly
        with the weight updates at every symbol; ``None`` disables it.

        * ``PLL(bandwidth=, mu=, beta=, phase_init=, joint_channels=)`` -
          decision-directed loop: the cross-product detector
          ``Im(y · conj(d))`` drives a PI loop (gains from ``bandwidth`` or
          raw ``mu`` / ``beta``).  ``phase_init`` seeds a cold start.  Low
          noise floor; recommended for QPSK through 64-QAM.
        * ``BPS(test_phases=, block_size=, joint_channels=)`` - blind phase
          search over ``test_phases`` angles in ``[0, 2π/S)``, ``S`` the
          constellation's rotational symmetry (π/2 for QAM), averaged over a causal window of the last
          ``block_size`` symbols.  Preferred for bursts where PLL pull-in is
          impractical.  The wrapped float32 estimate rotates the output; the
          causally ``S``-fold-unwrapped float64 accumulator is the
          ``phase_trajectory``, so float32 rounding never accumulates.

        ``joint_channels`` shares one estimate across MIMO channels (summed
        BPS metrics, or the PLL error averaged before the integrator).  A
        nested ``CycleSlip(history=, threshold=)`` predicts each phase from a
        linear fit through the last ``history`` values and snaps a deviation
        beyond ``threshold`` to the nearest ``2π/4`` multiple (``π`` for
        2-fold constellations).
    state : EqualizerState, optional
        Continue from ``result.state`` of a previous call of this equalizer
        with the same configuration: weights, normalization, inline CPR state
        and the input from where the previous call left off.  The previous
        result's last ``state.overlap`` symbols are recomputed here, so the
        outputs stitch to exactly one uninterrupted run.  Training symbols
        start at this call's first output symbol.
    pad_mode : {'zeros', 'edge'}, default 'zeros'
        Left padding of a cold start: ``'zeros'`` (default) prepends
        ``center_tap`` zeros, ``'edge'`` replicates the first sample, which
        can soften the initial amplitude jump.

    Returns
    -------
    EqualizerResult
        Result container with the following fields:

        * ``y_hat`` - equalized symbol estimates, shape ``(N_sym,)`` SISO
          or ``(C, N_sym)`` MIMO, at 1 SPS; always an array.
        * ``signal`` - for Signal input, the output as a 1-SPS Signal with
          its reference cut to the output symbols (RLS drops its tail).
        * ``weights`` - final tap-weight tensor, shape ``(num_taps,)`` SISO
          or ``(C, C, num_taps)`` MIMO.
        * ``error`` - complex error signal ``e[n] = d[n] - y[n]``, same
          shape as ``y_hat``.
        * ``weights_history`` - tap weights at each symbol when
          ``store_weights=True``; ``None`` otherwise.
        * ``phase_trajectory`` - per-symbol phase estimates, shape
          ``(N_sym,)`` SISO or ``(C, N_sym)`` MIMO.  BPS: causal
          ``S``-fold-unwrapped float64.  PLL: PI integrator state.  ``None``
          when ``cpr=None``.
        * ``num_train_symbols`` - number of data-aided training symbols.
        * ``input_norm_factor`` - RMS factor used to normalize inputs.
        * ``state`` - :class:`EqualizerState` for ``state=`` continuation;
          it also carries the inverse correlation matrix ``P``.

    Warnings
    --------
    **Fractional Spacing Singularity:**
    Applying RLS to fractionally-spaced signals (sps > 1) is not recommended.
    Fractional spacing bounds the signal energy within a subset of the Nyquist
    bandwidth. The unexcited frequency bands contain strictly thermal noise,
    rendering the input correlation matrix mathematically singular. RLS
    attempts to invert these near-zero eigenvalues, exponentially amplifying
    high-frequency noise and causing severe tap weight bloat.  Normalized LMS
    is the structurally stable alternative.

    ``initial_taps`` seeds the tap weights only: ``P`` starts at
    ``(1/delta) · I``.  ``state`` continues both.
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
    sq_side, sq_lev_min, sq_d_grid = _square_qam_slicer_params(constellation_np)
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
            _get_numba_rls_cpr()(
                x,
                train,
                constellation_np,
                *cpr_args.bps_args(),
                run.W,
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
    )
    result = _log_equalizer_exit(result, name="RLS")
    result.tail_trim = tail_trim
    _check_rls_divergence(result.weights, run.xp, forgetting_factor, delta)
    return _attach_equalized_signal(result, sig, state)
