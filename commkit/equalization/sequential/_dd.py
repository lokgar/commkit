"""Decision-directed sequential adaptive equalizers: lms, rls."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...backend import ArrayType, _get_jax, dispatch, from_jax, to_device, to_jax
from ...core.signal import Signal
from ...helpers import (
    _coerce_integer_sps,
    resolve_pll_gains,
    restore_1d,
    rewrap_signal,
    unwrap_signal,
)
from ...logger import logger
from ...mapping.gray import square_qam_slicer_params
from .._block import (
    _build_slicer_constellation,
    _run_block_equalizer,
    _validate_block_mode,
)
from .._common import (
    _build_padded_samples,
    _cpr_state_to_jax_inits,
    _cpr_symmetry,
    _init_butterfly_weights_jax,
    _init_butterfly_weights_numpy,
    _normalize_inputs,
    _prepare_training_jax,
    _prepare_training_numpy,
    _unpack_result_jax,
    _unpack_result_numpy,
    _validate_sps,
    _validate_w_init,
)
from .._kernels_jax import (
    _get_jax_lms,
    _get_jax_lms_cpr,
    _get_jax_rls,
    _get_jax_rls_cpr,
)
from .._kernels_numba import (
    _get_numba,
    _get_numba_lms,
    _get_numba_lms_cpr,
    _get_numba_rls,
    _get_numba_rls_cpr,
)
from ..result import CPRState, EqualizerResult, _log_equalizer_exit

# -----------------------------------------------------------------------------
# ADAPTIVE equalization
# -----------------------------------------------------------------------------


def lms(
    samples: ArrayType | Signal,
    training_symbols: ArrayType | None = None,
    num_taps: int = 21,
    sps: int | None = None,
    step_size: float = 0.01,
    modulation: str | None = None,
    order: int | None = None,
    unipolar: bool = False,
    store_weights: bool = False,
    device: str | None = "cpu",
    center_tap: int | None = None,
    backend: str = "numba",
    w_init: ArrayType | None = None,
    pmf: Any | None = None,
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
    debug_plot: bool = False,
    plot_smoothing: int = 50,
    cpr_state: CPRState | None = None,
    input_norm_factor: float | np.ndarray | None = None,
    samples_prefix: ArrayType | None = None,
    pad_mode: str = "zeros",
    update_mode: str = "sequential",
    block_len: int = 16,
) -> EqualizerResult:
    """
    Least Mean Squares adaptive equalizer with butterfly MIMO support.

    Update modes
    ------------
    The weight-update cadence is selected by ``update_mode`` - the equalizer
    math, regressor windows, slicer, and error are otherwise identical:

    +-----------------------+--------+----------------+----------------------+
    | Mode                  | Domain | Adaptation lag | Use case             |
    +=======================+========+================+======================+
    | ``'sequential'``      | time   | 1 symbol       | fastest dynamics,    |
    | (default)             |        |                | CPU (Numba)          |
    +-----------------------+--------+----------------+----------------------+
    | ``'block'``,          | time   | ``block_len``  | fast dynamics on GPU |
    | ``block_len`` 8-32    |        | symbols        | (JAX/CuPy)           |
    +-----------------------+--------+----------------+----------------------+
    | ``block_lms``         | freq.  | ``block_size`` | throughput king,     |
    | (separate function)   |        | (~256)         | slow/static channels |
    +-----------------------+--------+----------------+----------------------+

    With ``update_mode='block'`` the weights are frozen over ``block_len``
    symbols and one aggregated gradient - exactly the sum of what a
    frozen-weight per-symbol LMS would accumulate over those symbols - is
    applied per chunk, turning ``block_len`` rank-1 updates into a single matrix
    product the GPU can occupy.  ``step_size`` is therefore on the **same scale
    as** ``update_mode='sequential'``: the same ``mu`` yields the same
    convergence and steady-state floor (do **not** divide by ``block_len``).
    Only the *stability ceiling* is ~``block_len``x lower (the weights are
    frozen across the chunk); reduce ``mu`` below your sequential value only if
    the run diverges.  Block mode requires ``backend='jax'`` (chunked
    ``lax.scan``) or ``backend='xp'`` (array-native NumPy/CuPy loop);
    ``backend='numba'`` and ``cpr_type``/``store_weights`` are not supported.

    Supports data-aided (training) and decision-directed (DD) modes.
    When ``training_symbols`` are provided, the equalizer uses them for the
    initial convergence phase, then switches to DD mode using the
    ``modulation`` and ``order`` parameters, or a constellation auto-inferred
    from the training symbols for hard-decision slicing.

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
        If None, pure DD mode (requires ``modulation`` and ``order``).
    num_taps : int, default 21
        Number of equalizer taps per FIR filter. For fractionally-spaced
        equalization (sps > 1), use at least ``4 * sps`` taps.
    sps : int, optional, default 2
        Samples per symbol at the input.  Use ``sps=2`` (T/2-spaced, default)
        for the first equalization stage.  ``sps=1`` is valid for a second
        symbol-spaced stage after FOE + CPR; use a short filter (3-11 taps)
        for residual ISI cleanup.  ``sps > 2`` is accepted but uncommon.
        The equalizer decimates by ``sps`` to produce one output symbol per
        input stride.  Ignored for :class:`Signal` input, which always uses
        the signal's own ``sps``.
    step_size : float, default 0.01
        Plain LMS step size (mu). The gradient is applied directly without
        input-power normalization, matching the convention in Haykin's
        *Adaptive Filter Theory* and most published papers.  Stability
        requires ``0 < mu < 2 / (C * num_taps * P_x)`` where ``P_x`` is the
        mean per-tap input power.  Because inputs are normalized to unit
        symbol-rate power by default, a safe starting range for typical
        settings is ``1e-4`` to ``1e-2``.  Values closer to the upper bound
        converge faster but produce higher steady-state misadjustment.
    modulation : str, optional
        Modulation scheme (e.g., 'psk', 'qam', 'pam') for DD slicing.
        Required if ``training_symbols`` is None.
    order : int, optional
        Modulation order (e.g., 4, 16).
    unipolar : bool, default False
        If True, indicates the modulation is unipolar (e.g., unipolar PAM).
    store_weights : bool, default False
        If True, stores weight trajectory in ``weights_history``.
    device : str, optional
        Target device for JAX computations (e.g., 'cpu', 'gpu', 'tpu').
        Default is 'cpu'. Ignored when ``backend='numba'``.
    center_tap : int, optional
        Index of the center tap. If None, defaults to ``num_taps // 2``.
    backend : str, default 'numba'
        Execution backend. ``'numba'`` compiles the sequential loop with LLVM
        via Numba ``@njit``; typically 2-5x faster than JAX on CPU for
        SISO/small-MIMO signals (no scan serialization overhead).
        ``'jax'`` uses ``jax.lax.scan`` and supports GPU placement and
        automatic differentiation through the equalizer.
        ``'xp'`` is valid only with ``update_mode='block'``: it runs the
        array-native NumPy/CuPy block loop on the input's device (no JAX).
    w_init : array_like, optional
        Initial tap weights. Shape: ``(C, C, num_taps)`` complex64, or the
        SISO short-hand ``(num_taps,)`` / ``(1, num_taps)`` as returned by
        ``EqualizerResult.weights`` for single-channel equalizers.
        If provided, the equalizer warm-starts from these weights instead of
        the default center-tap identity matrix.  Useful for weight handoff from
        a prior stage (e.g. preamble LMS -> payload LMS).
        Raises ``ValueError`` if the shape does not match.
    pmf : array_like of float, optional
        Probability mass function over the constellation for probabilistically
        shaped QAM (PS-QAM).  When provided together with ``modulation`` and
        ``order``, the DD slicer constellation is scaled by ``1/sqrt(E_PS)``
        (where ``E_PS = sum_m P(s_m)|s_m|^2`` on the normalised grid) so it
        matches the unit-power normalised equaliser input.  Training symbols
        are left untouched - ``_normalize_inputs`` already brings them to unit
        average power.  Has no effect for uniform modulations.
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
        with the ``mu`` of ``recover_carrier_phase_pll``.
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
    debug_plot : bool, default False
        If True, display a convergence + tap-weight diagnostic plot on exit.
    plot_smoothing : int, default 50
        Moving-average window (symbols) for the MSE convergence curve in the
        debug plot.
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

        Note: JAX backend: ``cpr_state`` warm-start is not yet supported;
            passing a non-``None`` value raises ``NotImplementedError``.
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
    update_mode : {'sequential', 'block'}, default 'sequential'
        Weight-update cadence (see the *Update modes* section above).
        ``'sequential'`` updates the weights every symbol (the default; the
        only mode supported by ``backend='numba'``).  ``'block'`` freezes the
        weights over ``block_len`` symbols and applies one aggregated gradient
        per chunk - a matrix product that occupies the GPU - and requires
        ``backend='jax'`` (chunked ``lax.scan``) or ``backend='xp'``
        (array-native NumPy/CuPy loop).  ``cpr_type`` and ``store_weights`` are
        not supported with ``'block'`` (both raise).
    block_len : int, default 16
        Number of symbols per frozen-weight update chunk when
        ``update_mode='block'`` (typically 8-32; ignored otherwise).  Larger
        values amortise launch overhead further but reduce the tracking
        bandwidth proportionally.  ``step_size`` stays on the same scale as
        sequential mode (same ``mu`` => same steady-state floor); only the
        stability ceiling is ~``block_len``x lower.

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

    Warnings
    --------
    **JAX GPU mode is typically slower than CPU for adaptive equalization.**
    LMS is inherently sequential: each symbol's weight update depends on the
    previous weights, so ``lax.scan`` serializes execution even on GPU.
    The per-step arithmetic (a ``num_taps``-length dot product) is far too
    small to saturate GPU compute units, while kernel-launch and
    device-memory-transfer overhead dominate.  In practice, ``device='cpu'``
    is 2-10x faster for typical SISO sequences up to ~100 k symbols.  Use
    ``device='gpu'`` only when the number of MIMO channels (``num_ch``) is
    large enough to amortize GPU launch costs, or when batching many
    independent signals externally.  For CPU-optimal throughput use
    ``backend='numba'`` (the default).
    """
    x, sig = unwrap_signal(samples)
    if sig is not None:
        # sig.sps is always populated (derived from the required sampling_rate
        # / symbol_rate fields), so the Signal's own value always wins over a
        # supplied sps - see CLAUDE.md, "Signal-Awareness".
        if sps is not None:
            logger.warning(
                "lms(): ignoring supplied sps=%r for Signal input; using the "
                "signal's own sps=%r instead.",
                sps,
                sig.sps,
            )
        result = lms(
            x,
            training_symbols=training_symbols,
            num_taps=num_taps,
            sps=_coerce_integer_sps(sig.sps, caller="lms()"),
            step_size=step_size,
            modulation=modulation,
            order=order,
            unipolar=unipolar,
            store_weights=store_weights,
            device=device,
            center_tap=center_tap,
            backend=backend,
            w_init=w_init,
            pmf=pmf,
            cpr_type=cpr_type,
            cpr_pll_bandwidth=cpr_pll_bandwidth,
            cpr_pll_mu=cpr_pll_mu,
            cpr_pll_beta=cpr_pll_beta,
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
            update_mode=update_mode,
            block_len=block_len,
        )
        result.y_hat = rewrap_signal(sig, result.y_hat, sampling_rate=sig.symbol_rate)
        return result

    if sps is None:
        sps = 2

    if cpr_type is not None and cpr_type not in ("pll", "bps"):
        raise ValueError(f"cpr_type must be 'pll', 'bps', or None. Got {cpr_type!r}.")
    _validate_block_mode(
        update_mode,
        block_len,
        backend,
        cpr_type=cpr_type,
        store_weights=store_weights,
    )

    n_train_log = training_symbols.shape[-1] if training_symbols is not None else 0
    logger.info(
        "LMS equalizer: num_taps=%s, mu=%s, sps=%s, backend=%s, n_train=%s%s",
        num_taps,
        step_size,
        sps,
        backend,
        n_train_log,
        f", cpr={cpr_type}" if cpr_type else "",
    )
    if sps > 1:
        logger.warning(
            "LMS output y_hat is at 1 SPS (symbol rate). "
            "Update sampling_rate = symbol_rate after applying this equalizer."
        )

    samples, xp, _ = dispatch(samples)
    _validate_sps(sps, num_taps)
    stride = int(sps)

    if training_symbols is not None:
        training_symbols, _, _ = dispatch(training_symbols)

    # Shape calcs - independent of normalization
    was_1d = samples.ndim == 1
    num_ch = 1 if was_1d else samples.shape[0]
    n_samples = samples.shape[0] if was_1d else samples.shape[1]
    n_sym = n_samples // stride

    if training_symbols is not None and training_symbols.shape[-1] > n_sym:
        logger.warning(
            "training_symbols length (%s) exceeds available symbol count "
            "(%s); excess training symbols will be ignored.",
            training_symbols.shape[-1],
            n_sym,
        )

    c_tap = center_tap if center_tap is not None else num_taps // 2
    pad_total = max(0, n_sym * stride - n_samples + num_taps - 1)
    pad_left = min(c_tap, pad_total)
    pad_right = pad_total - pad_left

    if update_mode == "block":
        samples_np = np.ascontiguousarray(to_device(samples, "cpu"), dtype=np.complex64)
        training_np = (
            to_device(training_symbols, "cpu").astype(np.complex64)
            if training_symbols is not None
            else None
        )
        samples_np, training_np, eq_norm = _normalize_inputs(
            samples_np, training_np, sps, input_norm_factor=input_norm_factor
        )
        samples_padded_np = _build_padded_samples(
            samples_np, pad_left, pad_right, samples_prefix, pad_mode, eq_norm, sps
        )
        constellation_np = _build_slicer_constellation(
            modulation, order, unipolar, training_np, pmf
        )
        train_full, n_train_aligned = _prepare_training_numpy(
            training_np, num_ch, n_sym
        )
        _sq_side, _sq_lev_min, _sq_d_grid = square_qam_slicer_params(constellation_np)
        if w_init is not None:
            w_arr = _validate_w_init(
                np.ascontiguousarray(to_device(w_init, "cpu"), dtype=np.complex64),
                num_ch,
                num_taps,
            )
        else:
            w_arr = _init_butterfly_weights_numpy(
                num_ch, num_taps, center_tap=center_tap
            )
        return _run_block_equalizer(
            "lms",
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
            name="LMS(block)",
            debug_plot=debug_plot,
            plot_smoothing=plot_smoothing,
            constellation_np=constellation_np,
            train_full=train_full,
            n_train_aligned=n_train_aligned,
            sq_side=_sq_side,
            sq_lev_min=_sq_lev_min,
            sq_d_grid=_sq_d_grid,
        )

    if backend == "numba":
        # Convert to plain NumPy (no-op for CPU NumPy; downloads for CuPy)
        samples_np = np.ascontiguousarray(to_device(samples, "cpu"), dtype=np.complex64)
        training_np = (
            to_device(training_symbols, "cpu").astype(np.complex64)
            if training_symbols is not None
            else None
        )
        samples_np, training_np, eq_norm = _normalize_inputs(
            samples_np, training_np, sps, input_norm_factor=input_norm_factor
        )
        # Pad (NumPy)
        samples_padded = _build_padded_samples(
            samples_np, pad_left, pad_right, samples_prefix, pad_mode, eq_norm, sps
        )
        # Constellation (NumPy)
        if modulation is not None and order is not None:
            from ...mapping import gray_constellation

            reference_constellation = gray_constellation(
                modulation, order, unipolar=unipolar
            )
            constellation_np = (
                to_device(reference_constellation, "cpu").flatten().astype(np.complex64)
            )
        elif training_np is not None:
            train_flat = training_np.reshape(-1)
            constellation_np = np.unique(np.round(train_flat, decimals=8))
        else:
            raise ValueError("modulation and order must be provided for DD mode.")
        # PS-QAM: scale slicer constellation to unit-power {s_m/sqrt(E_PS)} so it
        # matches the normalised equaliser input.  Training is already at unit power
        # after _normalize_inputs - only the constellation reference needs scaling.
        if pmf is not None and modulation is not None and order is not None:
            _pmf_arr = np.asarray(pmf, dtype=np.float64)
            _e_ps = float(
                np.dot(_pmf_arr, np.abs(constellation_np).astype(np.float64) ** 2)
            )
            if _e_ps < 1.0 - 1e-6:
                _c_ps = np.float32(1.0 / np.sqrt(_e_ps))
                constellation_np = (constellation_np * _c_ps).astype(np.complex64)
        train_full, n_train_aligned = _prepare_training_numpy(
            training_np,
            num_ch,
            n_sym,
        )
        _sq_side, _sq_lev_min, _sq_d_grid = square_qam_slicer_params(constellation_np)
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
        if cpr_type is None:
            _get_numba_lms()(
                samples_padded,
                train_full,
                constellation_np,
                W,
                np.float32(step_size),
                np.int32(n_train_aligned),
                stride,
                store_weights,
                y_out,
                e_out,
                w_hist_buf,
                _sq_lev_min,
                _sq_d_grid,
                np.int32(_sq_side),
            )
            result = _unpack_result_numpy(
                y_out,
                e_out,
                W,
                w_hist_buf,
                was_1d,
                store_weights,
                n_sym=None,
                xp=xp,
                num_train_symbols=int(n_train_aligned),
                input_norm_factor=eq_norm,
            )
        else:
            pll_mu, pll_beta = resolve_pll_gains(
                cpr_pll_bandwidth, cpr_pll_mu, cpr_pll_beta
            )
            symmetry = _cpr_symmetry(modulation, order)
            B = int(cpr_bps_test_phases)
            bps_angles_np = np.linspace(
                0.0, np.pi / 2.0, B, endpoint=False, dtype=np.float32
            )
            bps_phases_neg_np = np.exp(-1j * bps_angles_np).astype(np.complex64)
            H = int(cpr_cycle_slip_history)
            _st = cpr_state
            _st_ok = (
                _st is not None
                and _st.cpr_type == cpr_type
                and _st.num_ch == num_ch
                and _st.cs_H == H
                and _st.pll_phi is not None
            )
            if _st_ok:
                assert _st is not None
                assert _st.pll_phi is not None
                assert _st.pll_freq is not None
                assert _st.cs_buf_x is not None
                assert _st.cs_buf_y is not None
                assert _st.cs_buf_ptr is not None
                assert _st.cs_buf_n is not None
                assert _st.cs_stats is not None
                pll_phi = _st.pll_phi.copy()
                pll_freq = _st.pll_freq.copy()
                cs_buf_x = _st.cs_buf_x.copy()
                cs_buf_y = _st.cs_buf_y.copy()
                cs_buf_ptr = _st.cs_buf_ptr.copy()
                cs_buf_n = _st.cs_buf_n.copy()
                cs_stats = _st.cs_stats.copy()
                bps_prev4 = (
                    _st.bps_prev4.copy()
                    if _st.bps_prev4 is not None
                    else np.zeros(num_ch, dtype=np.float64)
                )
            else:
                pll_phi = np.zeros(num_ch, dtype=np.float64)
                pll_freq = np.zeros(num_ch, dtype=np.float64)
                cs_buf_x = np.zeros((num_ch, H), dtype=np.float64)
                cs_buf_y = np.zeros((num_ch, H), dtype=np.float64)
                cs_buf_ptr = np.zeros(num_ch, dtype=np.int64)
                cs_buf_n = np.zeros(num_ch, dtype=np.int64)
                cs_stats = np.zeros((num_ch, 4), dtype=np.float64)
                bps_prev4 = np.zeros(num_ch, dtype=np.float64)
            phase_out = np.empty((n_sym, num_ch), dtype=np.float64)
            cpr_mode_int = np.int32(1 if cpr_type == "pll" else 2)
            _get_numba_lms_cpr()(
                samples_padded,
                train_full,
                constellation_np,
                bps_phases_neg_np,
                bps_angles_np,
                np.int32(cpr_bps_block_size),
                bool(cpr_joint_channels),
                W,
                np.float32(step_size),
                np.int32(n_train_aligned),
                stride,
                store_weights,
                cpr_mode_int,
                pll_mu,
                pll_beta,
                np.int32(symmetry),
                bool(cpr_cycle_slip_correction),
                np.float32(cpr_cycle_slip_threshold),
                pll_phi,
                pll_freq,
                cs_buf_x,
                cs_buf_y,
                cs_buf_ptr,
                cs_buf_n,
                cs_stats,
                bps_prev4,
                y_out,
                e_out,
                phase_out,
                w_hist_buf,
                _sq_lev_min,
                _sq_d_grid,
                np.int32(_sq_side),
            )
            result = _unpack_result_numpy(
                y_out,
                e_out,
                W,
                w_hist_buf,
                was_1d,
                store_weights,
                n_sym=None,
                xp=xp,
                num_train_symbols=int(n_train_aligned),
                input_norm_factor=eq_norm,
            )
            phi_t = xp.asarray(phase_out.T)  # (C, N_sym)
            result.phase_trajectory = restore_1d(was_1d, phi_t)
            result.cpr_state = CPRState(
                pll_phi=pll_phi.copy(),
                pll_freq=pll_freq.copy(),
                bps_prev4=bps_prev4.copy(),
                cs_buf_x=cs_buf_x.copy(),
                cs_buf_y=cs_buf_y.copy(),
                cs_buf_ptr=cs_buf_ptr.copy(),
                cs_buf_n=cs_buf_n.copy(),
                cs_stats=cs_stats.copy(),
                cpr_type=cpr_type,
                num_ch=num_ch,
                symmetry=symmetry,
                bps_P=B,
                bps_K=int(cpr_bps_block_size),
                cs_H=H,
            )
        return _log_equalizer_exit(
            result, name="LMS", debug_plot=debug_plot, plot_smoothing=plot_smoothing
        )

    # JAX backend
    jax, jnp, _ = _get_jax()
    if jax is None or jnp is None:
        raise ImportError("JAX is required for backend='jax'.")

    samples, training_symbols, eq_norm = _normalize_inputs(
        samples, training_symbols, sps, input_norm_factor=input_norm_factor
    )
    # Pad - use _build_padded_samples (returns CPU NumPy); convert to xp array after
    _samp_cpu = to_device(samples, "cpu").astype(np.complex64)
    samples_padded_np = _build_padded_samples(
        _samp_cpu, pad_left, pad_right, samples_prefix, pad_mode, eq_norm, sps
    )
    samples_padded = (
        xp.asarray(samples_padded_np)
        if not was_1d
        else xp.asarray(samples_padded_np[0])
    )
    # Constellation
    if modulation is not None and order is not None:
        from ...mapping import gray_constellation

        reference_constellation = gray_constellation(
            modulation, order, unipolar=unipolar
        )
    elif training_symbols is not None:
        _, _xp, _ = dispatch(training_symbols)
        train_flat = _xp.reshape(training_symbols, (-1,))
        reference_constellation = _xp.unique(_xp.round(train_flat, decimals=8))
    else:
        raise ValueError("modulation and order must be provided for DD mode.")
    constellation_np = (
        to_device(reference_constellation, "cpu").flatten().astype("complex64")
    )
    # PS-QAM: scale slicer constellation to unit-power {s_m/sqrt(E_PS)}.
    if pmf is not None and modulation is not None and order is not None:
        _pmf_arr = np.asarray(pmf, dtype=np.float64)
        _e_ps = float(
            np.dot(_pmf_arr, np.abs(constellation_np).astype(np.float64) ** 2)
        )
        if _e_ps < 1.0 - 1e-6:
            constellation_np = (
                constellation_np * np.float32(1.0 / np.sqrt(_e_ps))
            ).astype(np.complex64)
    train_full, n_train_aligned = _prepare_training_jax(
        training_symbols,
        num_ch,
        n_sym,
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

    train_jax = to_jax(train_full, device=platform)
    const_jax = to_jax(constellation_np, device=platform)
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
    n_train_jax = to_jax(jnp.int32(n_train_aligned), device=platform)

    if cpr_type is None:
        _sq_side_j, _sq_lev_min_j, _sq_d_grid_j = square_qam_slicer_params(
            constellation_np
        )
        scan_fn = _get_jax_lms(
            num_taps,
            stride,
            len(constellation_np),
            num_ch,
            int(_sq_side_j),
            float(_sq_lev_min_j),
            float(_sq_d_grid_j),
        )
        y_jax, e_jax, W_jax, wh_jax = scan_fn(
            x_jax, train_jax, const_jax, W_jax, mu_jax, n_train_jax
        )
        result = _unpack_result_jax(
            y_jax,
            e_jax,
            W_jax,
            wh_jax,
            was_1d,
            store_weights,
            n_sym=None,
            xp=xp,
            num_train_symbols=int(n_train_aligned),
            input_norm_factor=eq_norm,
        )
    else:
        x64_enabled = (
            jax.config.jax_enable_x64
            if hasattr(jax.config, "jax_enable_x64")
            else jax.config.read("jax_enable_x64")
        )
        if not x64_enabled:
            raise RuntimeError(
                "JAX x64 mode must be enabled for CPR phase tracking: "
                "call jax.config.update('jax_enable_x64', True) before using "
                "backend='jax' with cpr_type set."
            )
        pll_mu, pll_beta = resolve_pll_gains(
            cpr_pll_bandwidth, cpr_pll_mu, cpr_pll_beta
        )
        symmetry = _cpr_symmetry(modulation, order)
        B = int(cpr_bps_test_phases)
        H = int(cpr_cycle_slip_history)
        bps_angles_np = np.linspace(
            0.0, np.pi / 2.0, B, endpoint=False, dtype=np.float32
        )
        bps_phases_neg_np = np.exp(-1j * bps_angles_np).astype(np.complex64)
        bps_pn_jax = to_jax(bps_phases_neg_np, device=platform)
        bps_ang_jax = to_jax(bps_angles_np, device=platform)
        _sq_side_j, _sq_lev_min_j, _sq_d_grid_j = square_qam_slicer_params(
            constellation_np
        )
        scan_fn = _get_jax_lms_cpr(
            num_taps,
            stride,
            len(constellation_np),
            num_ch,
            cpr_type,
            B,
            int(cpr_bps_block_size),
            bool(cpr_joint_channels),
            H,
            int(symmetry),
            int(_sq_side_j),
            float(_sq_lev_min_j),
            float(_sq_d_grid_j),
        )
        KB = int(cpr_bps_block_size)
        if cpr_state is not None:
            _pi, _pf, _bb, _bbp, _bp4, _cx, _cy, _cp = _cpr_state_to_jax_inits(
                cpr_state, num_ch, KB, H
            )
        else:
            _pi = np.zeros(num_ch, dtype=np.float64)
            _pf = np.zeros(num_ch, dtype=np.float64)
            _bb = np.zeros((KB, num_ch), dtype=np.complex64)
            _bbp = np.int32(0)
            _bp4 = np.zeros(num_ch, dtype=np.float64)
            _cx = np.zeros((num_ch, H), dtype=np.float64)
            _cy = np.zeros((num_ch, H), dtype=np.float64)
            _cp = np.zeros(num_ch, dtype=np.int32)
        (
            y_jax,
            e_jax,
            W_jax,
            wh_jax,
            phi_jax,
            pll_phi_f,
            pll_freq_f,
            bps_buf_f,
            bps_buf_ptr_f,
            bps_prev4_f,
            cs_buf_x_f,
            cs_buf_y_f,
            cs_buf_ptr_f,
        ) = scan_fn(
            x_jax,
            train_jax,
            const_jax,
            bps_pn_jax,
            bps_ang_jax,
            W_jax,
            mu_jax,
            n_train_jax,
            to_jax(jnp.float64(pll_mu), device=platform),
            to_jax(jnp.float64(pll_beta), device=platform),
            to_jax(jnp.float64(cpr_cycle_slip_threshold), device=platform),
            to_jax(jnp.bool_(cpr_cycle_slip_correction), device=platform),
            to_jax(_pi, device=platform),
            to_jax(_pf, device=platform),
            to_jax(_bb, device=platform),
            to_jax(_bbp, device=platform),
            to_jax(_bp4, device=platform),
            to_jax(_cx, device=platform),
            to_jax(_cy, device=platform),
            to_jax(_cp, device=platform),
        )
        result = _unpack_result_jax(
            y_jax,
            e_jax,
            W_jax,
            wh_jax,
            was_1d,
            store_weights,
            n_sym=None,
            xp=xp,
            num_train_symbols=int(n_train_aligned),
            input_norm_factor=eq_norm,
        )
        # from_jax returns CuPy for a GPU-resident JAX array; coerce to host
        # NumPy via to_device (CuPy.get / NumPy passthrough) so CPRState stays
        # host-side and np.asarray never sees a CuPy array.
        phi_np = to_device(from_jax(phi_jax), "cpu")  # (N_sym, C)
        phi_t = xp.asarray(phi_np.T)  # (C, N_sym)
        result.phase_trajectory = restore_1d(was_1d, phi_t)
        result.cpr_state = CPRState(
            pll_phi=to_device(from_jax(pll_phi_f), "cpu"),
            pll_freq=to_device(from_jax(pll_freq_f), "cpu"),
            bps_prev4=to_device(from_jax(bps_prev4_f), "cpu"),
            jax_bps_buf=to_device(from_jax(bps_buf_f), "cpu"),
            jax_bps_buf_ptr=int(to_device(from_jax(bps_buf_ptr_f), "cpu")),
            cs_buf_x=to_device(from_jax(cs_buf_x_f), "cpu"),
            cs_buf_y=to_device(from_jax(cs_buf_y_f), "cpu"),
            cs_buf_ptr=to_device(from_jax(cs_buf_ptr_f), "cpu"),
            cpr_type=cpr_type,
            num_ch=num_ch,
            symmetry=symmetry,
            bps_P=B,
            bps_K=KB,
            cs_H=H,
        )
    return _log_equalizer_exit(result, name="LMS", debug_plot=debug_plot)


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
    num_taps: int = 21,
    sps: int | None = None,
    forgetting_factor: float = 0.99,
    delta: float = 0.01,
    leakage: float = 0.0,
    modulation: str | None = None,
    order: int | None = None,
    unipolar: bool = False,
    store_weights: bool = False,
    device: str | None = "cpu",
    center_tap: int | None = None,
    backend: str = "numba",
    w_init: ArrayType | None = None,
    pmf: Any | None = None,
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
    debug_plot: bool = False,
    plot_smoothing: int = 50,
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
    sps : int, optional, default 1
        Samples per symbol at the input.  Ignored for :class:`Signal` input,
        which always uses the signal's own ``sps``.
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
    modulation : str, optional
        Modulation scheme (e.g., 'psk', 'qam', 'pam') for DD slicing.
        Required if ``training_symbols`` is None.
    order : int, optional
        Modulation order (e.g., 4, 16).
    unipolar : bool, default False
        If True, indicates the modulation is unipolar (e.g., unipolar PAM).
    store_weights : bool, default False
        If True, stores weight trajectory.
    device : str, optional
        Target device for JAX computations (e.g., 'cpu', 'gpu', 'tpu').
        Default is 'cpu'. Ignored when ``backend='numba'``.
    center_tap : int, optional
        Index of the center tap. If None, defaults to ``num_taps // 2``.
    backend : str, default 'numba'
        Execution backend. ``'numba'`` uses Numba ``@njit``; LLVM-compiled,
        typically fastest on CPU, particularly for the O(num_taps²) Riccati
        update. ``'jax'`` uses ``jax.lax.scan`` (XLA-compiled, GPU-capable).
    pmf : array_like of float, optional
        Probability mass function for PS-QAM.  Scales the DD slicer
        constellation by ``1/sqrt(E_PS)`` to match the unit-power normalised
        equaliser input.  Requires ``modulation`` and ``order``.
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
        ``recover_carrier_phase_pll``.
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
    debug_plot : bool, default False
        Display convergence + tap-weight diagnostic plot on exit.
    plot_smoothing : int, default 50
        MSE moving-average window for the debug plot.
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
    x, sig = unwrap_signal(samples)
    if sig is not None:
        # sig.sps is always populated (derived from the required sampling_rate
        # / symbol_rate fields), so the Signal's own value always wins over a
        # supplied sps - see CLAUDE.md, "Signal-Awareness".
        if sps is not None:
            logger.warning(
                "rls(): ignoring supplied sps=%r for Signal input; using the "
                "signal's own sps=%r instead.",
                sps,
                sig.sps,
            )
        result = rls(
            x,
            training_symbols=training_symbols,
            num_taps=num_taps,
            sps=_coerce_integer_sps(sig.sps, caller="rls()"),
            forgetting_factor=forgetting_factor,
            delta=delta,
            leakage=leakage,
            modulation=modulation,
            order=order,
            unipolar=unipolar,
            store_weights=store_weights,
            device=device,
            center_tap=center_tap,
            backend=backend,
            w_init=w_init,
            pmf=pmf,
            cpr_type=cpr_type,
            cpr_pll_bandwidth=cpr_pll_bandwidth,
            cpr_pll_mu=cpr_pll_mu,
            cpr_pll_beta=cpr_pll_beta,
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
        )
        result.y_hat = rewrap_signal(sig, result.y_hat, sampling_rate=sig.symbol_rate)
        return result

    if sps is None:
        sps = 1

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
        "leakage=%.2e, sps=%s, backend=%s, n_train=%s%s",
        num_taps,
        forgetting_factor,
        delta,
        leakage,
        sps,
        backend,
        n_train_log,
        f", cpr={cpr_type}" if cpr_type else "",
    )
    if sps > 1:
        logger.warning(
            "RLS output y_hat is at 1 SPS (symbol rate). "
            "Update sampling_rate = symbol_rate after applying this equalizer."
        )

    samples, xp, _ = dispatch(samples)
    _validate_sps(sps, num_taps)
    stride = int(sps)

    if training_symbols is not None:
        training_symbols, _, _ = dispatch(training_symbols)

    was_1d = samples.ndim == 1
    if was_1d:
        num_ch = 1
        n_samples = samples.shape[0]
    else:
        num_ch, n_samples = samples.shape

    n_sym = n_samples // stride

    if training_symbols is not None and training_symbols.shape[-1] > n_sym:
        logger.warning(
            "training_symbols length (%s) exceeds available symbol count "
            "(%s); excess training symbols will be ignored.",
            training_symbols.shape[-1],
            n_sym,
        )

    # Early-halt boundary: freeze W and P once the sliding window reaches the
    # right zero-padding (last num_taps//2 symbols have contaminated windows).
    n_update_halt = max(0, n_sym - num_taps // 2)
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

    c_tap = center_tap if center_tap is not None else num_taps // 2
    pad_total = max(0, n_sym * stride - n_samples + num_taps - 1)
    pad_left = min(c_tap, pad_total)
    pad_right = pad_total - pad_left

    if backend == "numba":
        numba = _get_numba()
        if numba is None:
            raise ImportError("Numba is required for backend='numba'.")

        samples_np = np.ascontiguousarray(to_device(samples, "cpu"), dtype=np.complex64)
        training_np = (
            to_device(training_symbols, "cpu").astype(np.complex64)
            if training_symbols is not None
            else None
        )
        samples_np, training_np, eq_norm = _normalize_inputs(
            samples_np, training_np, sps, input_norm_factor=input_norm_factor
        )

        x_np = _build_padded_samples(
            samples_np, pad_left, pad_right, samples_prefix, pad_mode, eq_norm, sps
        )
        x_np = np.ascontiguousarray(x_np)

        if modulation is not None and order is not None:
            from ...mapping import gray_constellation

            reference_constellation = gray_constellation(
                modulation, order, unipolar=unipolar
            )
            constellation_np = (
                to_device(reference_constellation, "cpu").flatten().astype(np.complex64)
            )
        elif training_np is not None:
            train_flat = training_np.reshape(-1)
            constellation_np = np.unique(np.round(train_flat, decimals=8)).astype(
                "complex64"
            )
        else:
            raise ValueError("modulation and order must be provided for DD mode.")
        # PS-QAM: scale slicer constellation to unit-power {s_m/sqrt(E_PS)}.
        if pmf is not None and modulation is not None and order is not None:
            _pmf_arr = np.asarray(pmf, dtype=np.float64)
            _e_ps = float(
                np.dot(_pmf_arr, np.abs(constellation_np).astype(np.float64) ** 2)
            )
            if _e_ps < 1.0 - 1e-6:
                _c_ps = np.float32(1.0 / np.sqrt(_e_ps))
                constellation_np = (constellation_np * _c_ps).astype(np.complex64)

        train_full, n_train_aligned = _prepare_training_numpy(
            training_np,
            num_ch,
            n_sym,
        )
        _sq_side, _sq_lev_min, _sq_d_grid = square_qam_slicer_params(constellation_np)
        if w_init is not None:
            w_arr = np.ascontiguousarray(to_device(w_init, "cpu"), dtype=np.complex64)
            w_arr = _validate_w_init(w_arr, num_ch, num_taps)
            W = w_arr.copy()
        else:
            W = _init_butterfly_weights_numpy(num_ch, num_taps, center_tap=center_tap)
        regressor_dim = num_ch * num_taps
        P = np.eye(regressor_dim, dtype=np.complex128) / np.float64(delta)
        y_out = np.empty((n_sym, num_ch), dtype=np.complex64)
        e_out = np.empty((n_sym, num_ch), dtype=np.complex64)
        w_hist_buf = (
            np.empty((n_sym, num_ch, num_ch, num_taps), dtype=np.complex64)
            if store_weights
            else np.empty((1, num_ch, num_ch, num_taps), dtype=np.complex64)
        )
        if cpr_type is None:
            _get_numba_rls()(
                x_np,
                train_full,
                constellation_np,
                W,
                P,
                np.float32(forgetting_factor),
                np.float32(leakage),
                np.int32(n_train_aligned),
                np.int32(n_update_halt),
                stride,
                store_weights,
                y_out,
                e_out,
                w_hist_buf,
                _sq_lev_min,
                _sq_d_grid,
                np.int32(_sq_side),
            )
            result = _unpack_result_numpy(
                y_out,
                e_out,
                W,
                w_hist_buf,
                was_1d,
                store_weights,
                n_sym=n_update_halt,
                xp=xp,
                num_train_symbols=int(n_train_aligned),
                input_norm_factor=eq_norm,
            )
        else:
            pll_mu, pll_beta = resolve_pll_gains(
                cpr_pll_bandwidth, cpr_pll_mu, cpr_pll_beta
            )
            symmetry = _cpr_symmetry(modulation, order)
            B = int(cpr_bps_test_phases)
            bps_angles_np = np.linspace(
                0.0, np.pi / 2.0, B, endpoint=False, dtype=np.float32
            )
            bps_phases_neg_np = np.exp(-1j * bps_angles_np).astype(np.complex64)
            H = int(cpr_cycle_slip_history)
            _st = cpr_state
            _st_ok = (
                _st is not None
                and _st.cpr_type == cpr_type
                and _st.num_ch == num_ch
                and _st.cs_H == H
                and _st.pll_phi is not None
            )
            if _st_ok:
                assert _st is not None
                assert _st.pll_phi is not None
                assert _st.pll_freq is not None
                assert _st.cs_buf_x is not None
                assert _st.cs_buf_y is not None
                assert _st.cs_buf_ptr is not None
                assert _st.cs_buf_n is not None
                assert _st.cs_stats is not None
                pll_phi = _st.pll_phi.copy()
                pll_freq = _st.pll_freq.copy()
                cs_buf_x = _st.cs_buf_x.copy()
                cs_buf_y = _st.cs_buf_y.copy()
                cs_buf_ptr = _st.cs_buf_ptr.copy()
                cs_buf_n = _st.cs_buf_n.copy()
                cs_stats = _st.cs_stats.copy()
                bps_prev4 = (
                    _st.bps_prev4.copy()
                    if _st.bps_prev4 is not None
                    else np.zeros(num_ch, dtype=np.float64)
                )
            else:
                pll_phi = np.zeros(num_ch, dtype=np.float64)
                pll_freq = np.zeros(num_ch, dtype=np.float64)
                cs_buf_x = np.zeros((num_ch, H), dtype=np.float64)
                cs_buf_y = np.zeros((num_ch, H), dtype=np.float64)
                cs_buf_ptr = np.zeros(num_ch, dtype=np.int64)
                cs_buf_n = np.zeros(num_ch, dtype=np.int64)
                cs_stats = np.zeros((num_ch, 4), dtype=np.float64)
                bps_prev4 = np.zeros(num_ch, dtype=np.float64)
            phase_out = np.empty((n_sym, num_ch), dtype=np.float64)
            cpr_mode_int = np.int32(1 if cpr_type == "pll" else 2)
            _get_numba_rls_cpr()(
                x_np,
                train_full,
                constellation_np,
                bps_phases_neg_np,
                bps_angles_np,
                np.int32(cpr_bps_block_size),
                bool(cpr_joint_channels),
                W,
                P,
                np.float32(forgetting_factor),
                np.float32(leakage),
                np.int32(n_train_aligned),
                np.int32(n_update_halt),
                stride,
                store_weights,
                cpr_mode_int,
                pll_mu,
                pll_beta,
                np.int32(symmetry),
                bool(cpr_cycle_slip_correction),
                np.float32(cpr_cycle_slip_threshold),
                pll_phi,
                pll_freq,
                cs_buf_x,
                cs_buf_y,
                cs_buf_ptr,
                cs_buf_n,
                cs_stats,
                bps_prev4,
                y_out,
                e_out,
                phase_out,
                w_hist_buf,
                _sq_lev_min,
                _sq_d_grid,
                np.int32(_sq_side),
            )
            result = _unpack_result_numpy(
                y_out,
                e_out,
                W,
                w_hist_buf,
                was_1d,
                store_weights,
                n_sym=n_update_halt,
                xp=xp,
                num_train_symbols=int(n_train_aligned),
                input_norm_factor=eq_norm,
            )
            phi_t = xp.asarray(phase_out[:n_update_halt].T)  # (C, n_update_halt)
            result.phase_trajectory = restore_1d(was_1d, phi_t)
            result.cpr_state = CPRState(
                pll_phi=pll_phi.copy(),
                pll_freq=pll_freq.copy(),
                bps_prev4=bps_prev4.copy(),
                cs_buf_x=cs_buf_x.copy(),
                cs_buf_y=cs_buf_y.copy(),
                cs_buf_ptr=cs_buf_ptr.copy(),
                cs_buf_n=cs_buf_n.copy(),
                cs_stats=cs_stats.copy(),
                cpr_type=cpr_type,
                num_ch=num_ch,
                symmetry=symmetry,
                bps_P=B,
                bps_K=int(cpr_bps_block_size),
                cs_H=H,
            )
        # Truncate last num_taps//2 symbols (zero-padding contamination).
        result = _log_equalizer_exit(
            result, name="RLS", debug_plot=debug_plot, plot_smoothing=plot_smoothing
        )
        result.tail_trim = tail_trim
        _check_rls_divergence(result.weights, xp, forgetting_factor, delta)
        return result

    # JAX backend
    jax, jnp, _ = _get_jax()
    if jax is None or jnp is None:
        raise ImportError("JAX is required for backend='jax'.")
    x64_enabled = (
        jax.config.jax_enable_x64
        if hasattr(jax.config, "jax_enable_x64")
        else jax.config.read("jax_enable_x64")
    )
    if not x64_enabled:
        raise RuntimeError(
            "JAX x64 mode must be enabled for RLS: the P (Riccati) matrix requires "
            "complex128 precision to remain positive-definite. "
            "Call jax.config.update('jax_enable_x64', True) before using backend='jax'."
        )

    samples, training_symbols, eq_norm = _normalize_inputs(
        samples, training_symbols, sps, input_norm_factor=input_norm_factor
    )

    _samp_cpu_rls = to_device(samples, "cpu").astype(np.complex64)
    samples_padded_np_rls = _build_padded_samples(
        _samp_cpu_rls, pad_left, pad_right, samples_prefix, pad_mode, eq_norm, sps
    )
    samples_padded = (
        xp.asarray(samples_padded_np_rls)
        if not was_1d
        else xp.asarray(samples_padded_np_rls[0])
    )

    if modulation is not None and order is not None:
        from ...mapping import gray_constellation

        reference_constellation = gray_constellation(
            modulation, order, unipolar=unipolar
        )
    elif training_symbols is not None:
        _, _xp, _ = dispatch(training_symbols)
        train_flat = _xp.reshape(training_symbols, (-1,))
        reference_constellation = _xp.unique(_xp.round(train_flat, decimals=8))
    else:
        raise ValueError("modulation and order must be provided for DD mode.")

    constellation_np = (
        to_device(reference_constellation, "cpu").flatten().astype("complex64")
    )
    # PS-QAM: scale slicer constellation to unit-power {s_m/sqrt(E_PS)}.
    if pmf is not None and modulation is not None and order is not None:
        _pmf_arr = np.asarray(pmf, dtype=np.float64)
        _e_ps = float(
            np.dot(_pmf_arr, np.abs(constellation_np).astype(np.float64) ** 2)
        )
        if _e_ps < 1.0 - 1e-6:
            constellation_np = (
                constellation_np * np.float32(1.0 / np.sqrt(_e_ps))
            ).astype(np.complex64)

    logger.debug(
        "RLS internals: n_sym=%s, n_train=%s, n_update_halt=%s, "
        "leakage=%.2e, delta=%.2e",
        n_sym,
        n_train_log,
        n_update_halt,
        leakage,
        delta,
    )

    train_full, n_train_aligned = _prepare_training_jax(
        training_symbols,
        num_ch,
        n_sym,
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

    train_jax = to_jax(train_full, device=platform)
    const_jax = to_jax(constellation_np, device=platform)
    if w_init is not None:
        w_arr = np.ascontiguousarray(to_device(w_init, "cpu"), dtype=np.complex64)
        w_arr = _validate_w_init(w_arr, num_ch, num_taps)
        W_jax = to_jax(w_arr, device=platform)
    else:
        W_jax = _init_butterfly_weights_jax(
            num_ch, num_taps, jnp, center_tap=center_tap
        )
        W_jax = to_jax(W_jax, device=platform)

    regressor_dim = num_ch * num_taps
    P_init = jnp.eye(regressor_dim, dtype="complex128") / delta
    P_init = to_jax(P_init, device=platform)
    lam_jax = to_jax(jnp.float32(forgetting_factor), device=platform)
    n_train_jax = to_jax(jnp.int32(n_train_aligned), device=platform)
    leakage_jax = to_jax(jnp.float32(leakage), device=platform)
    n_update_halt_jax = to_jax(jnp.int32(n_update_halt), device=platform)

    if cpr_type is None:
        _sq_side_j, _sq_lev_min_j, _sq_d_grid_j = square_qam_slicer_params(
            constellation_np
        )
        scan_fn = _get_jax_rls(
            num_taps,
            stride,
            len(constellation_np),
            num_ch,
            int(_sq_side_j),
            float(_sq_lev_min_j),
            float(_sq_d_grid_j),
        )
        y_jax, e_jax, W_jax, wh_jax = scan_fn(
            x_jax,
            train_jax,
            const_jax,
            W_jax,
            P_init,
            lam_jax,
            n_train_jax,
            leakage_jax,
            n_update_halt_jax,
        )
        result = _unpack_result_jax(
            y_jax,
            e_jax,
            W_jax,
            wh_jax,
            was_1d,
            store_weights,
            n_sym=n_update_halt,
            xp=xp,
            num_train_symbols=int(n_train_aligned),
            input_norm_factor=eq_norm,
        )
    else:
        pll_mu, pll_beta = resolve_pll_gains(
            cpr_pll_bandwidth, cpr_pll_mu, cpr_pll_beta
        )
        symmetry = _cpr_symmetry(modulation, order)
        B = int(cpr_bps_test_phases)
        H = int(cpr_cycle_slip_history)
        bps_angles_np = np.linspace(
            0.0, np.pi / 2.0, B, endpoint=False, dtype=np.float32
        )
        bps_phases_neg_np = np.exp(-1j * bps_angles_np).astype(np.complex64)
        bps_pn_jax = to_jax(bps_phases_neg_np, device=platform)
        bps_ang_jax = to_jax(bps_angles_np, device=platform)
        _sq_side_j, _sq_lev_min_j, _sq_d_grid_j = square_qam_slicer_params(
            constellation_np
        )
        scan_fn = _get_jax_rls_cpr(
            num_taps,
            stride,
            len(constellation_np),
            num_ch,
            cpr_type,
            B,
            int(cpr_bps_block_size),
            bool(cpr_joint_channels),
            H,
            int(symmetry),
            int(_sq_side_j),
            float(_sq_lev_min_j),
            float(_sq_d_grid_j),
        )
        KB = int(cpr_bps_block_size)
        if cpr_state is not None:
            _pi, _pf, _bb, _bbp, _bp4, _cx, _cy, _cp = _cpr_state_to_jax_inits(
                cpr_state, num_ch, KB, H
            )
        else:
            _pi = np.zeros(num_ch, dtype=np.float64)
            _pf = np.zeros(num_ch, dtype=np.float64)
            _bb = np.zeros((KB, num_ch), dtype=np.complex64)
            _bbp = np.int32(0)
            _bp4 = np.zeros(num_ch, dtype=np.float64)
            _cx = np.zeros((num_ch, H), dtype=np.float64)
            _cy = np.zeros((num_ch, H), dtype=np.float64)
            _cp = np.zeros(num_ch, dtype=np.int32)
        (
            y_jax,
            e_jax,
            W_jax,
            wh_jax,
            phi_jax,
            pll_phi_f,
            pll_freq_f,
            bps_buf_f,
            bps_buf_ptr_f,
            bps_prev4_f,
            cs_buf_x_f,
            cs_buf_y_f,
            cs_buf_ptr_f,
        ) = scan_fn(
            x_jax,
            train_jax,
            const_jax,
            bps_pn_jax,
            bps_ang_jax,
            W_jax,
            P_init,
            lam_jax,
            n_train_jax,
            leakage_jax,
            n_update_halt_jax,
            to_jax(jnp.float64(pll_mu), device=platform),
            to_jax(jnp.float64(pll_beta), device=platform),
            to_jax(jnp.float64(cpr_cycle_slip_threshold), device=platform),
            to_jax(jnp.bool_(cpr_cycle_slip_correction), device=platform),
            to_jax(_pi, device=platform),
            to_jax(_pf, device=platform),
            to_jax(_bb, device=platform),
            to_jax(_bbp, device=platform),
            to_jax(_bp4, device=platform),
            to_jax(_cx, device=platform),
            to_jax(_cy, device=platform),
            to_jax(_cp, device=platform),
        )
        result = _unpack_result_jax(
            y_jax,
            e_jax,
            W_jax,
            wh_jax,
            was_1d,
            store_weights,
            n_sym=n_update_halt,
            xp=xp,
            num_train_symbols=int(n_train_aligned),
            input_norm_factor=eq_norm,
        )
        # from_jax returns CuPy for a GPU-resident JAX array; coerce to host
        # NumPy via to_device (CuPy.get / NumPy passthrough) so CPRState stays
        # host-side and np.asarray never sees a CuPy array.
        phi_np = to_device(from_jax(phi_jax), "cpu")  # (N_sym, C)
        phi_t = xp.asarray(phi_np[:n_update_halt].T)  # (C, n_update_halt)
        result.phase_trajectory = restore_1d(was_1d, phi_t)
        result.cpr_state = CPRState(
            pll_phi=to_device(from_jax(pll_phi_f), "cpu"),
            pll_freq=to_device(from_jax(pll_freq_f), "cpu"),
            bps_prev4=to_device(from_jax(bps_prev4_f), "cpu"),
            jax_bps_buf=to_device(from_jax(bps_buf_f), "cpu"),
            jax_bps_buf_ptr=int(to_device(from_jax(bps_buf_ptr_f), "cpu")),
            cs_buf_x=to_device(from_jax(cs_buf_x_f), "cpu"),
            cs_buf_y=to_device(from_jax(cs_buf_y_f), "cpu"),
            cs_buf_ptr=to_device(from_jax(cs_buf_ptr_f), "cpu"),
            cpr_type=cpr_type,
            num_ch=num_ch,
            symmetry=symmetry,
            bps_P=B,
            bps_K=KB,
            cs_H=H,
        )
    # Truncate last num_taps//2 symbols (zero-padding contamination).
    result = _log_equalizer_exit(result, name="RLS", debug_plot=debug_plot)
    result.tail_trim = tail_trim
    _check_rls_divergence(result.weights, xp, forgetting_factor, delta)
    return result
