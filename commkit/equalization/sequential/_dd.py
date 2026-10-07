"""Decision-directed sequential adaptive equalizers: lms, rls."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...backend import ArrayType
from ...core._signal_adapter import adapt_signal, require_integer_sps
from ...core.signal import Signal
from ...logger import logger
from ...mapping.gray import _square_qam_slicer_params
from ...recovery._common import _resolve_pll_gains
from .._common import _cpr_symmetry
from .._kernels_numba import (
    _get_numba_lms,
    _get_numba_lms_cpr,
    _get_numba_rls,
    _get_numba_rls_cpr,
)
from ..result import (
    CPRState,
    EqualizerResult,
    _attach_equalized_signal,
    _log_equalizer_exit,
)
from ._setup import (
    _assemble_sequential,
    _bps_phases,
    _carrier_args,
    _carrier_arrays,
    _dd_constellation,
    _prepare_sequential,
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
    w_init: ArrayType | None = None,
    cpr_type: str | None = None,
    cpr_pll_bandwidth: float = 1e-3,
    cpr_pll_mu: float | None = None,
    cpr_pll_beta: float | None = None,
    cpr_bps_test_phases: int = 64,
    cpr_bps_block_size: int = 32,
    cpr_joint_channels: bool = False,
    cpr_cycle_slip_correction: bool = False,
    cpr_cycle_slip_history: int = 100,
    cpr_cycle_slip_threshold: float = np.pi / 4,
    cpr_state: CPRState | None = None,
    input_norm_factor: float | np.ndarray | None = None,
    samples_prefix: ArrayType | None = None,
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

    3. **Carrier phase recovery** (if ``cpr_type`` is set):

       * **PLL** - cross-product phase detector
         ``phi_err = Im(y_raw * conj(d_prev))``
         drives a PI integrator with gains ``K_p``, ``K_i``; accumulated
         phase ``phi_n`` is applied as ``y[n] = y_raw * exp(-j*phi_n)``.
       * **BPS** - ``B`` candidate rotations ``exp(-j*k*pi/(2*B))`` are
         tested; the one minimising the summed nearest-constellation distance
         over the trailing ``K`` = ``cpr_bps_block_size`` symbols is chosen.
         A causal 4-fold unwrap converts the ``[0, pi/2)`` argmin to full-range
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

    7. **Cycle-slip correction** (if ``cpr_cycle_slip_correction=True``) - a
       circular buffer of ``cpr_cycle_slip_history`` past phase values is
       maintained per channel.  An online least-squares linear fit over the
       buffer predicts ``phi_pred_n``.  If
       ``|phi_n - phi_pred_n| > cpr_cycle_slip_threshold``,
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
        A :class:`Signal` returns an :class:`EqualizerResult` whose ``y_hat``
        is a new :class:`Signal` at the symbol rate (``sampling_rate =
        symbol_rate``); ``sps`` defaults to the signal's ``sps`` when not
        given explicitly.
    training_symbols : array_like, optional
        Known transmitted symbols (at symbol rate, 1 SPS).
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
    w_init : array_like, optional
        Initial tap weights. Shape: ``(C, C, num_taps)`` complex64, or the
        SISO short-hand ``(num_taps,)`` / ``(1, num_taps)`` as returned by
        ``EqualizerResult.weights`` for single-channel equalizers.
        If provided, the equalizer warm-starts from these weights instead of
        the default center-tap identity matrix.  Useful for weight handoff from
        a prior stage (e.g. preamble LMS -> payload LMS).
        Raises ``ValueError`` if the shape does not match.
    cpr_type : {'pll', 'bps', None}, default None
        Inline carrier phase recovery algorithm applied jointly with weight
        updates at every symbol.  ``None`` disables CPR (default, bit-exact
        with the legacy behaviour).

        * ``'pll'`` - 2nd-order decision-directed phase-locked loop.  The
          cross-product phase detector ``Im(y · conj(d))`` drives a PI loop
          with gains derived from ``cpr_pll_bandwidth``.  Low noise floor;
          recommended for QPSK through 64-QAM.
        * ``'bps'`` - Blind Phase Search over ``cpr_bps_test_phases`` candidate
          angles in ``[0, π/2)`` (exploiting 4-fold QAM symmetry), averaged
          over a causal window of ``cpr_bps_block_size`` past y_raw samples.
          Preferred for burst/packet modes where PLL pull-in is impractical.

          **Dual-path output:** the wrapped float32 phase estimate rotates
          ``y_raw`` for the weight update, while the unwrapped float64
          accumulator is stored in ``phase_trajectory``.  The two paths are
          kept separate to prevent float32 rounding errors from accumulating
          in the trajectory over thousands of symbols.

          **4-fold causal unwrap:** the raw BPS ``argmin`` lives in
          ``[0, π/2)``.  A causal unwrap tracks the argmin evolution symbol
          by symbol, adding or subtracting ``π/2`` multiples as needed to
          keep the estimate continuous, then converting to full-range radians.
    cpr_pll_bandwidth : float, default 1e-3
        Normalised loop bandwidth ``B_L · T_s`` for the PLL.  Gains are
        computed as ``K_p = 4 B_L``, ``K_i = 4 B_L²`` (critically-damped
        approximation, ζ=1).  Typical range: ``5e-4`` (low phase noise) to
        ``5e-3`` (high phase noise / fast drift).  Used only when
        ``cpr_pll_mu is None`` (the bandwidth shortcut) and ``cpr_type == 'pll'``.
    cpr_pll_mu : float, optional
        Raw proportional PLL gain ``μ``.  If given, overrides
        ``cpr_pll_bandwidth`` and uses raw PI gains directly; ``cpr_pll_beta``
        then defaults to ``0.0`` (a 1st-order loop).  Leave ``None`` to derive
        critically-damped gains from ``cpr_pll_bandwidth``.  Interchangeable
        with the ``mu`` of ``recovery.PLL``.
    cpr_pll_beta : float, optional
        Raw integral PLL gain ``β``.  ``β=0`` => 1st-order loop (no frequency
        integrator); ``β>0`` => 2nd-order loop.  Requires ``cpr_pll_mu`` to be
        set (passing ``cpr_pll_beta`` alone raises ``ValueError``).  The
        ``(μ,β) <-> (B_L,ζ)`` mapping is ``ωₙT = √β``, ``ζ = μ/(2√β)``.
    cpr_bps_test_phases : int, default 64
        Number of candidate phase angles for the BPS search in ``[0, π/2)``.
        Higher values improve phase resolution at the cost of ``B`` extra
        distance evaluations per symbol.  32-64 is sufficient for ≤ 16-QAM;
        use 64-128 for 64-QAM.  Ignored when ``cpr_type != 'bps'``.
    cpr_bps_block_size : int, default 32
        Number of past y_raw samples whose min-distance metrics are summed
        before the BPS ``argmin``.  Larger values reduce noise on the phase
        estimate at the cost of increased latency (``K-1`` symbols).
        ``cpr_bps_block_size=1`` recovers the degenerate single-symbol BPS.
        Ignored when ``cpr_type != 'bps'``.
    cpr_joint_channels : bool, default False
        For MIMO inputs (C > 1): if ``True``, the per-symbol phase estimate
        is computed jointly across all C channels before being applied.

        * **BPS** - the per-candidate distance metrics are summed across
          channels before ``argmin``, giving one shared estimate broadcast
          to all channels.  Reduces estimation variance by ~√C for
          shared-LO transmitter setups.
        * **PLL** - the cross-product phase-error signal
          ``Im(y · conj(d))`` is averaged across channels before the PI
          integrator, so all channels share one phase trajectory.

        When ``False``, each channel runs its own independent estimator.
        Ignored for SISO inputs (C == 1).
    cpr_cycle_slip_correction : bool, default False
        Enable causal cycle-slip detection and correction.  A circular
        buffer of ``cpr_cycle_slip_history`` past phase estimates is
        maintained per channel.  Before each symbol, an online linear
        regression over the buffer predicts the current phase; if the new
        estimate deviates from the prediction by more than
        ``cpr_cycle_slip_threshold``, the estimate is snapped to the
        nearest ``2π/symmetry`` quantum (e.g. π/2 for QPSK/QAM) and the
        buffer is updated with the corrected value.  Disable for
        deterministic parity checks or when the channel is known to be
        slip-free.
    cpr_cycle_slip_history : int, default 100
        Length of the phase-history buffer used for cycle-slip prediction.
        Longer buffers produce a more stable linear-trend fit but are
        slower to adapt when the true carrier frequency drifts.  Ignored
        when ``cpr_cycle_slip_correction=False``.
    cpr_cycle_slip_threshold : float, default π/4
        Maximum tolerated deviation (radians) between the predicted and
        observed phase before a slip is declared.  Should be set to half
        the constellation's angular symmetry quantum (``π/4`` for
        QPSK/QAM).  Ignored when ``cpr_cycle_slip_correction=False``.
    cpr_state : CPRState, optional
        Warm-start CPR state from a previous ``lms()`` call (obtained via
        ``EqualizerResult.cpr_state``).  When provided and the CPR type and
        channel count match, the PLL integrators, BPS unwrap accumulators,
        and cycle-slip buffers are pre-loaded rather than zero-initialized.
        This eliminates the ~5-10 k symbol CPR convergence transient that
        occurs at every block boundary in streaming pipelines.  Pass
        ``None`` (default) to cold-start the CPR from zero.  Ignored when
        ``cpr_type=None`` or when the stored state is incompatible (mismatched
        ``cpr_type``, channel count, or history depth), in which case the
        equalizer falls back to cold-start silently.
    input_norm_factor : float or ndarray, optional
        Pre-computed RMS normalization factor from a previous call (obtained
        via ``EqualizerResult.input_norm_factor``).  When provided, the
        ``_normalize_inputs`` step is skipped and this value is used directly
        to scale the input samples and training symbols.  This ensures that
        warm-started weight vectors see the same amplitude regime as the
        block on which they were trained, preventing a gradient scale mismatch
        when signal power drifts slowly between blocks.
        Pass ``None`` (default) to recompute the RMS from the current block.
    samples_prefix : array_like, optional
        Signal history from the end of the previous block, used to eliminate
        the zero-padded leading transient at each block boundary.  Shape:
        ``(≥ pad_left,)`` SISO or ``(C, ≥ pad_left)`` MIMO, where
        ``pad_left = min(center_tap, max(0, num_taps - 1))``.  The last
        ``pad_left`` samples of ``samples_prefix`` replace the leading zeros
        in the tap window so that the first output symbol sees a fully
        populated, real-signal tap vector.  The prefix is normalized by the
        same ``input_norm_factor`` as the main block before being prepended.
        Pass ``None`` (default) for standard zero-padding.  Raises
        ``ValueError`` if the prefix length is less than ``pad_left``.
    pad_mode : {'zeros', 'edge'}, default 'zeros'
        Padding strategy for the leading tap window when ``samples_prefix``
        is ``None``.  ``'zeros'`` (default) prepends ``pad_left`` complex
        zeros, which is the standard causal initialisation.  ``'edge'``
        replicates the first sample of the current block, which can reduce
        the initial amplitude jump at cold start.  Has no effect when
        ``samples_prefix`` is provided.

    Returns
    -------
    EqualizerResult
        Result container with the following fields:

        * ``y_hat`` - equalized symbol estimates, shape ``(N_sym,)`` SISO
          or ``(C, N_sym)`` MIMO, at 1 SPS (symbol rate).  A new
          :class:`Signal` (``sampling_rate = symbol_rate``) when ``samples``
          was a :class:`Signal`.
        * ``weights`` - final tap-weight tensor, shape ``(num_taps,)`` SISO
          or ``(C, C, num_taps)`` MIMO.
        * ``error`` - complex error signal ``e[n] = d[n] - y[n]``, same
          shape as ``y_hat``.
        * ``weights_history`` - tap weights recorded at each symbol (only
          when ``store_weights=True``); shape ``(N_sym, num_taps)`` SISO or
          ``(N_sym, C, C, num_taps)`` MIMO.  ``None`` otherwise.
        * ``phase_trajectory`` - accumulated per-symbol phase estimates,
          shape ``(N_sym,)`` SISO or ``(C, N_sym)`` MIMO.  For BPS, this
          is the causal 4-fold-unwrapped float64 phase.  For PLL, it is
          the PI integrator state accumulated over all symbols.  ``None``
          when ``cpr_type=None``.
        * ``num_train_symbols`` - number of training symbols consumed
          (data-aided phase).
        * ``input_norm_factor`` - the RMS factor used to normalize inputs
          (float).  Store and pass as ``input_norm_factor`` on the next call
          to keep weight magnitudes consistent across block boundaries.
        * ``cpr_state`` - ``CPRState`` snapshot of PLL/BPS/cycle-slip
          integrators after the last symbol.  Pass as ``cpr_state`` on the
          next call to resume CPR without a re-convergence transient.
          ``None`` when ``cpr_type=None``.

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

    if cpr_type is not None and cpr_type not in ("pll", "bps"):
        raise ValueError(f"cpr_type must be 'pll', 'bps', or None. Got {cpr_type!r}.")

    n_train_log = training_symbols.shape[-1] if training_symbols is not None else 0
    logger.info(
        "LMS equalizer: num_taps=%s, mu=%s, sps=%s, n_train=%s%s",
        num_taps,
        step_size,
        sps,
        n_train_log,
        f", cpr={cpr_type}" if cpr_type else "",
    )
    if sps > 1:
        logger.warning(
            "LMS output y_hat is at 1 SPS (symbol rate). "
            "Update sampling_rate = symbol_rate after applying this equalizer."
        )

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
        training_symbols=training_symbols,
    )
    constellation_np = _dd_constellation(
        constellation, "lms()", decisions=run.n_train < run.n_sym
    )
    sq_side, sq_lev_min, sq_d_grid = _square_qam_slicer_params(constellation_np)
    slicer = (sq_lev_min, sq_d_grid, np.int32(sq_side))

    if cpr_type is None:
        _get_numba_lms()(
            run.x,
            run.train_full,
            constellation_np,
            run.W,
            np.float32(step_size),
            np.int32(run.n_train),
            run.stride,
            store_weights,
            run.y_out,
            run.e_out,
            run.w_hist,
            *slicer,
        )
        result = _assemble_sequential(run)
    else:
        pll_mu, pll_beta = _resolve_pll_gains(
            cpr_pll_bandwidth, cpr_pll_mu, cpr_pll_beta
        )
        symmetry = _cpr_symmetry(constellation)
        bps_angles, bps_phases_neg = _bps_phases(cpr_bps_test_phases)
        history = int(cpr_cycle_slip_history)
        carrier = _carrier_arrays(cpr_state, cpr_type, run.num_ch, history)
        phase_out = np.empty((run.n_sym, run.num_ch), dtype=np.float64)
        _get_numba_lms_cpr()(
            run.x,
            run.train_full,
            constellation_np,
            bps_phases_neg,
            bps_angles,
            np.int32(cpr_bps_block_size),
            bool(cpr_joint_channels),
            run.W,
            np.float32(step_size),
            np.int32(run.n_train),
            run.stride,
            store_weights,
            np.int32(1 if cpr_type == "pll" else 2),
            pll_mu,
            pll_beta,
            np.int32(symmetry),
            bool(cpr_cycle_slip_correction),
            np.float32(cpr_cycle_slip_threshold),
            *_carrier_args(carrier),
            run.y_out,
            run.e_out,
            phase_out,
            run.w_hist,
            *slicer,
        )
        result = _assemble_sequential(
            run,
            phase_out=phase_out,
            carrier=carrier,
            cpr_state_tags=dict(
                cpr_type=cpr_type,
                num_ch=run.num_ch,
                symmetry=symmetry,
                bps_P=len(bps_angles),
                bps_K=int(cpr_bps_block_size),
                cs_H=history,
            ),
        )
    result = _log_equalizer_exit(result, name="LMS")
    return _attach_equalized_signal(result, sig)


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
    w_init: ArrayType | None = None,
    cpr_type: str | None = None,
    cpr_pll_bandwidth: float = 1e-3,
    cpr_pll_mu: float | None = None,
    cpr_pll_beta: float | None = None,
    cpr_bps_test_phases: int = 64,
    cpr_bps_block_size: int = 32,
    cpr_joint_channels: bool = False,
    cpr_cycle_slip_correction: bool = False,
    cpr_cycle_slip_history: int = 100,
    cpr_cycle_slip_threshold: float = np.pi / 4,
    cpr_state: CPRState | None = None,
    input_norm_factor: float | np.ndarray | None = None,
    samples_prefix: ArrayType | None = None,
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
        A :class:`Signal` returns an :class:`EqualizerResult` whose ``y_hat``
        is a new :class:`Signal` at the symbol rate (``sampling_rate =
        symbol_rate``); ``sps`` defaults to the signal's ``sps`` when not
        given explicitly.
    training_symbols : array_like, optional
        Known symbols for data-aided adaptation (at symbol rate, 1 SPS).
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
    cpr_type : {'pll', 'bps', None}, default None
        Inline carrier phase recovery applied jointly with weight updates at
        every symbol.  ``None`` disables CPR (default).

        * ``'pll'`` - 2nd-order decision-directed PLL.  The cross-product
          detector ``Im(y · conj(d))`` drives a PI loop with gains from
          ``cpr_pll_bandwidth``.  Low noise floor; suited to QPSK-64-QAM.
        * ``'bps'`` - Blind Phase Search over ``cpr_bps_test_phases``
          candidate angles in ``[0, π/2)``, averaged over a causal window
          of ``cpr_bps_block_size`` samples.  Uses a **dual-path** design:
          the wrapped float32 estimate rotates ``y_raw`` for the Riccati
          update; the unwrapped float64 accumulator populates
          ``phase_trajectory``.  A **causal 4-fold unwrap** converts the
          ``[0, π/2)`` argmin to full-range radians symbol by symbol.
    cpr_pll_bandwidth : float, default 1e-3
        Normalised loop bandwidth ``B_L · T_s`` for the PLL.  Gains are
        ``K_p = 4 B_L``, ``K_i = 4 B_L²`` (critically-damped, ζ=1).  Typical
        range: ``5e-4`` (low phase noise) to ``5e-3`` (high drift).
        Used only when ``cpr_pll_mu is None`` and ``cpr_type == 'pll'``.
    cpr_pll_mu : float, optional
        Raw proportional PLL gain ``μ``.  If given, overrides
        ``cpr_pll_bandwidth``; ``cpr_pll_beta`` then defaults to ``0.0``
        (1st-order).  Leave ``None`` for critically-damped gains from
        ``cpr_pll_bandwidth``.  Interchangeable with the ``mu`` of
        ``recovery.PLL``.
    cpr_pll_beta : float, optional
        Raw integral PLL gain ``β``.  ``β=0`` => 1st-order, ``β>0`` => 2nd-order.
        Requires ``cpr_pll_mu`` to be set.  Mapping: ``ωₙT = √β``,
        ``ζ = μ/(2√β)``.
    cpr_bps_test_phases : int, default 64
        Number of BPS candidate phase angles in ``[0, π/2)``.  32-64 is
        sufficient for ≤ 16-QAM; use 64-128 for 64-QAM.  Ignored when
        ``cpr_type != 'bps'``.
    cpr_bps_block_size : int, default 32
        Trailing-window length (symbols) summed before the BPS ``argmin``.
        Larger values reduce phase-noise variance at the cost of latency.
        ``cpr_bps_block_size=1`` gives single-symbol BPS.  Ignored when
        ``cpr_type != 'bps'``.
    cpr_joint_channels : bool, default False
        For MIMO inputs (C > 1): share the phase estimate across channels.

        * **BPS** - distance metrics are summed across channels before
          ``argmin``.  Reduces estimation variance by ~√C for shared-LO
          systems.
        * **PLL** - the phase-error signal is averaged across channels
          before the PI integrator.

        When ``False``, each channel has an independent estimator.
        Ignored for SISO inputs.
    cpr_cycle_slip_correction : bool, default False
        Enable causal cycle-slip detection and correction.  A circular
        buffer of ``cpr_cycle_slip_history`` past phase estimates is kept
        per channel.  An online linear regression predicts the next phase;
        if the new estimate deviates by more than
        ``cpr_cycle_slip_threshold``, it is snapped to the nearest
        ``2π/symmetry`` quantum and the buffer is updated.  Disable for
        parity checks or slip-free channels.
    cpr_cycle_slip_history : int, default 100
        Phase-history buffer length for slip prediction.  Longer buffers
        give a more stable trend estimate but adapt more slowly to genuine
        frequency steps.  Ignored when ``cpr_cycle_slip_correction=False``.
    cpr_cycle_slip_threshold : float, default π/4
        Maximum tolerated deviation (radians) before a slip is declared.
        Set to half the constellation's angular quantum (``π/4`` for
        QPSK/QAM).  Ignored when ``cpr_cycle_slip_correction=False``.
    cpr_state : CPRState, optional
        Warm-start CPR state from a previous ``rls()`` call.  See
        ``lms()`` for the full description; behaviour is identical.
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
        Result container with the following fields:

        * ``y_hat`` - equalized symbol estimates, shape ``(N_sym,)`` SISO
          or ``(C, N_sym)`` MIMO, at 1 SPS.  A new :class:`Signal`
          (``sampling_rate = symbol_rate``) when ``samples`` was a
          :class:`Signal`.
        * ``weights`` - final tap-weight tensor, shape ``(num_taps,)`` SISO
          or ``(C, C, num_taps)`` MIMO.
        * ``error`` - complex error signal ``e[n] = d[n] - y[n]``, same
          shape as ``y_hat``.
        * ``weights_history`` - tap weights at each symbol when
          ``store_weights=True``; ``None`` otherwise.
        * ``phase_trajectory`` - per-symbol phase estimates, shape
          ``(N_sym,)`` SISO or ``(C, N_sym)`` MIMO.  BPS: causal
          4-fold-unwrapped float64.  PLL: PI integrator state.  ``None``
          when ``cpr_type=None``.
        * ``num_train_symbols`` - number of data-aided training symbols.
        * ``input_norm_factor`` - RMS factor used to normalize inputs.
        * ``cpr_state`` - CPRState snapshot after the last symbol; ``None``
          when ``cpr_type=None``.

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

    ``w_init`` warms-start the tap weights; the inverse correlation matrix ``P``
    always begins at ``(1/delta) · I`` regardless of ``w_init``.
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

    if cpr_type is not None and cpr_type not in ("pll", "bps"):
        raise ValueError(f"cpr_type must be 'pll', 'bps', or None. Got {cpr_type!r}.")

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
        f", cpr={cpr_type}" if cpr_type else "",
    )
    if sps > 1:
        logger.warning(
            "RLS output y_hat is at 1 SPS (symbol rate). "
            "Update sampling_rate = symbol_rate after applying this equalizer."
        )

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
        training_symbols=training_symbols,
    )
    # Early-halt boundary: freeze W and P once the sliding window reaches the
    # right zero-padding (last num_taps//2 symbols have contaminated windows).
    n_update_halt = max(0, run.n_sym - num_taps // 2)
    tail_trim = num_taps // 2
    if tail_trim > 0:
        logger.warning(
            "RLS tail trim: last %s symbols removed from y_hat "
            "(zero-padding contamination zone). Trim reference arrays to "
            "match: source_symbols = source_symbols[..., :-result.tail_trim], "
            "source_bits = source_bits[..., "
            ":-result.tail_trim * bits_per_symbol].",
            tail_trim,
        )
    constellation_np = _dd_constellation(
        constellation, "rls()", decisions=run.n_train < run.n_sym
    )
    sq_side, sq_lev_min, sq_d_grid = _square_qam_slicer_params(constellation_np)
    slicer = (sq_lev_min, sq_d_grid, np.int32(sq_side))
    # Inverse correlation matrix: complex128 throughout (single precision loses
    # the Hermitian positive-definite property and the filter diverges).
    P = np.eye(run.num_ch * num_taps, dtype=np.complex128) / np.float64(delta)

    if cpr_type is None:
        _get_numba_rls()(
            run.x,
            run.train_full,
            constellation_np,
            run.W,
            P,
            np.float32(forgetting_factor),
            np.float32(leakage),
            np.int32(run.n_train),
            np.int32(n_update_halt),
            run.stride,
            store_weights,
            run.y_out,
            run.e_out,
            run.w_hist,
            *slicer,
        )
        result = _assemble_sequential(run, n_sym=n_update_halt)
    else:
        pll_mu, pll_beta = _resolve_pll_gains(
            cpr_pll_bandwidth, cpr_pll_mu, cpr_pll_beta
        )
        symmetry = _cpr_symmetry(constellation)
        bps_angles, bps_phases_neg = _bps_phases(cpr_bps_test_phases)
        history = int(cpr_cycle_slip_history)
        carrier = _carrier_arrays(cpr_state, cpr_type, run.num_ch, history)
        phase_out = np.empty((run.n_sym, run.num_ch), dtype=np.float64)
        _get_numba_rls_cpr()(
            run.x,
            run.train_full,
            constellation_np,
            bps_phases_neg,
            bps_angles,
            np.int32(cpr_bps_block_size),
            bool(cpr_joint_channels),
            run.W,
            P,
            np.float32(forgetting_factor),
            np.float32(leakage),
            np.int32(run.n_train),
            np.int32(n_update_halt),
            run.stride,
            store_weights,
            np.int32(1 if cpr_type == "pll" else 2),
            pll_mu,
            pll_beta,
            np.int32(symmetry),
            bool(cpr_cycle_slip_correction),
            np.float32(cpr_cycle_slip_threshold),
            *_carrier_args(carrier),
            run.y_out,
            run.e_out,
            phase_out,
            run.w_hist,
            *slicer,
        )
        result = _assemble_sequential(
            run,
            n_sym=n_update_halt,
            phase_out=phase_out,
            carrier=carrier,
            cpr_state_tags=dict(
                cpr_type=cpr_type,
                num_ch=run.num_ch,
                symmetry=symmetry,
                bps_P=len(bps_angles),
                bps_K=int(cpr_bps_block_size),
                cs_H=history,
            ),
        )
    result = _log_equalizer_exit(result, name="RLS")
    result.tail_trim = tail_trim
    _check_rls_divergence(result.weights, run.xp, forgetting_factor, delta)
    return _attach_equalized_signal(result, sig)
