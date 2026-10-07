"""Decision-directed frequency-domain block equalizer: block_lms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ...backend import ArrayType, to_device
from ...core._signal_adapter import adapt_signal, require_integer_sps
from ...core.signal import Signal
from ...logger import logger
from ...mapping.gray import _square_qam_slicer_params
from .._common import _cpr_symmetry
from .._kernels_numba import _get_numba_cs_block
from ..result import (
    CPRState,
    EqualizerResult,
    _attach_equalized_signal,
    _log_equalizer_exit,
)
from ._engine import (
    _Block,
    _block_loop,
    _fdaf_forward,
    _fdaf_gradient_update,
    _prepare_block,
)


@dataclass
class _Slicer:
    """Hard-decision slicer: per-axis rounding for square QAM, else a table."""

    constellation: ArrayType  # (M,) complex64 on the device
    side: int  # points per axis of a square lattice, 0 otherwise
    lev_min: float
    d_grid: float
    kernel: Any = None  # CUDA argmin kernel for the table path, or None
    phasor: ArrayType | None = None  # unit rotation fed to that kernel

    def decide(self, y: ArrayType, xp: Any) -> ArrayType:
        if self.side > 0:
            m1 = self.side - 1
            d_r = (
                self.lev_min
                + xp.clip(xp.round((y.real - self.lev_min) / self.d_grid), 0, m1)
                * self.d_grid
            )
            d_i = (
                self.lev_min
                + xp.clip(xp.round((y.imag - self.lev_min) / self.d_grid), 0, m1)
                * self.d_grid
            )
            d = xp.empty(y.shape, dtype=xp.complex64)
            d.real[:] = d_r
            d.imag[:] = d_i
            return d
        if self.kernel is not None:
            _, idx = self.kernel(y, self.phasor, constellation=self.constellation)
            return self.constellation[idx[0]]
        d2 = (xp.abs(y[:, :, None] - self.constellation[None, None, :]) ** 2).real
        return self.constellation[xp.argmin(d2, axis=-1)]

    def min_d2(self, rotated: ArrayType, xp: Any) -> ArrayType:
        """Squared distance to the nearest point, for rotated ``(P, C, B)``."""
        if self.side > 0:
            m1 = self.side - 1
            nr = (
                self.lev_min
                + xp.clip(xp.round((rotated.real - self.lev_min) / self.d_grid), 0, m1)
                * self.d_grid
            )
            ni = (
                self.lev_min
                + xp.clip(xp.round((rotated.imag - self.lev_min) / self.d_grid), 0, m1)
                * self.d_grid
            )
            return ((rotated.real - nr) ** 2 + (rotated.imag - ni) ** 2).astype(
                xp.float32
            )
        d2_all = (
            xp.abs(rotated[..., None] - self.constellation[None, None, None, :]) ** 2
        ).real
        return xp.min(d2_all, axis=-1).astype(xp.float32)


@dataclass
class _BlockBps:
    """Blind phase search across blocks: candidates, window history, 4-fold
    unwrap and cycle-slip state (device arrays; host for the Numba slip path).
    """

    P: int
    K: int  # BPS averaging window
    hist_len: int
    joint: bool
    angles: ArrayType  # (P,) float32
    phases_neg: ArrayType  # (P,) complex64, exp(-j*angle)
    prev4: ArrayType  # (C,) float64, last 4x angle
    offset4: ArrayType  # (C,) float64, accumulated unwrapped 4x phase
    d2_hist: ArrayType  # (P, C, K-1) float32 trailing metrics
    cs: bool
    cs_H: int
    quantum: float
    threshold: float
    cs_buf_x: np.ndarray
    cs_buf_y: Any
    cs_buf_ptr: Any
    cs_buf_n: Any
    cs_stats: Any
    kernel: Any = None  # CUDA min-distance kernel, or None
    cs_kernel: Any = None  # CUDA cycle-slip kernel, or None

    def phase(
        self, y_block: ArrayType, slicer: _Slicer, b_start: int, xp: Any
    ) -> tuple[ArrayType, ArrayType]:
        """Wrapped float32 phase to rotate by, and the unwrapped trajectory."""
        C, B = y_block.shape
        P = self.P
        if self.kernel is not None:
            if slicer.side > 0:
                min_d2 = self.kernel(
                    y_block,
                    self.phases_neg,
                    lev_min=slicer.lev_min,
                    d_grid=slicer.d_grid,
                    side=slicer.side,
                )
            else:
                min_d2 = self.kernel(
                    y_block, self.phases_neg, constellation=slicer.constellation
                )
        else:
            rotated = self.phases_neg[:, None, None] * y_block[None, :, :]
            min_d2 = slicer.min_d2(rotated, xp)

        # Causal K-sample window that continues across blocks: the last K-1
        # metrics of the previous block are prepended.
        K = min(self.K, B)
        hist_prefix = (
            self.d2_hist[:, :, -(K - 1) :]
            if K > 1
            else xp.empty((P, C, 0), dtype=xp.float32)
        )
        cat_d2 = xp.concatenate([hist_prefix, min_d2], axis=2)  # (P, C, K-1+B)
        cs_d2 = xp.concatenate(
            [xp.zeros((P, C, 1), dtype=xp.float32), cat_d2.cumsum(axis=2)], axis=2
        )  # (P, C, K+B)
        win_sum = cs_d2[:, :, K:] - cs_d2[:, :, :-K]  # (P, C, B)
        metric = win_sum / xp.float32(K)  # (P, C, B) - always full K-sample window
        if self.joint and C > 1:
            best_k = xp.argmin(metric.sum(axis=1), axis=0)  # (B,)
            phi_raw = xp.broadcast_to(self.angles[best_k][None, :], (C, B)).copy()
        else:
            best_k = xp.argmin(metric, axis=0)  # (C, B)
            phi_raw = self.angles[best_k]  # (C, B)
        if self.hist_len > 0:
            combined_hist = xp.concatenate([self.d2_hist, min_d2], axis=2)
            self.d2_hist[...] = combined_hist[:, :, -self.hist_len :]

        # 4-fold unwrap continuing from the previous block.
        if xp is np:
            raw4 = phi_raw.astype(np.float64) * 4.0  # (C, B)
            extended = np.concatenate([self.prev4[:, np.newaxis], raw4], axis=1)
            unwrapped_ext = np.unwrap(extended, axis=1)  # (C, B+1)
            cumul = unwrapped_ext[:, 1:] - unwrapped_ext[:, 0:1]  # (C, B)
            phi_f64 = (self.offset4[:, np.newaxis] + cumul) / 4.0  # (C, B)
            self.prev4[:] = unwrapped_ext[:, -1]
            self.offset4 += cumul[:, -1]
        else:
            raw4_dev = phi_raw.astype(xp.float64) * xp.float64(4.0)  # (C, B)
            ext_dev = xp.concatenate([self.prev4[:, None], raw4_dev], axis=1)
            two_pi = xp.float64(2.0 * np.pi)
            d4 = ext_dev[:, 1:] - ext_dev[:, :-1]  # (C, B)
            d4 -= xp.round(d4 / two_pi) * two_pi  # wrap to [-π, π]
            cumul_dev = xp.cumsum(d4, axis=1)  # (C, B)
            phi_f64 = (self.offset4[:, None] + cumul_dev) / xp.float64(4.0)
            self.prev4 += cumul_dev[:, -1]
            self.offset4 += cumul_dev[:, -1]

        if self.cs:
            if self.cs_kernel is not None:
                phi_corr = xp.empty_like(phi_f64)
                self.cs_kernel(
                    phi_f64,
                    phi_corr,
                    self.cs_buf_y,
                    self.cs_buf_ptr,
                    self.cs_buf_n,
                    self.cs_stats,
                    float(self.quantum),
                    float(self.threshold),
                    self.cs_H,
                )
            else:
                phi_blk_np = to_device(phi_f64, "cpu").astype(np.float64)  # (C, B)
                phi_corr_np = phi_blk_np.copy()
                _get_numba_cs_block()(
                    phi_blk_np,
                    phi_corr_np,
                    self.cs_buf_x,
                    self.cs_buf_y,
                    self.cs_buf_ptr,
                    self.cs_buf_n,
                    self.cs_stats,
                    b_start,
                    float(self.quantum),
                    float(self.threshold),
                    self.cs_H,
                )
                phi_corr = xp.asarray(phi_corr_np)
            # Carry the slip correction into the unwrap accumulator.
            self.offset4 += (phi_corr[:, -1] - phi_f64[:, -1]) * 4.0
            phi_f64 = phi_corr

        two_pi = xp.float64(2.0 * np.pi)
        phi_wrapped = (phi_f64 - xp.round(phi_f64 / two_pi) * two_pi).astype(xp.float32)
        return phi_wrapped, phi_f64.astype(xp.float32)


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
    :func:`block_rde` (ring-directed, for multi-ring QAM).  For fast channel
    dynamics, the per-symbol :func:`lms` adapts with a one-symbol lag.

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
        Samples per symbol.  ``sps=2`` (T/2-spaced) is the default.  Ignored
        for :class:`Signal` input, which always uses the signal's own
        ``sps``.
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
    dominates at any practical block size.  Use the per-symbol :func:`lms`
    (Numba) for CPU workloads instead.

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
    signal_adapter = adapt_signal(samples, function_name="block_lms()")
    samples = signal_adapter.array
    sig = signal_adapter.signal
    if sig is not None:
        sps = require_integer_sps(
            signal_adapter.resolve_required("sps", sps), "block_lms()"
        )
    if sps is None:
        sps = 2

    if cpr_type is not None and cpr_type != "bps":
        raise ValueError(
            f"block_lms only supports cpr_type='bps' or None. Got {cpr_type!r}. "
            "PLL is not available for block processing."
        )

    run = _prepare_block(
        samples,
        sps=sps,
        num_taps=num_taps,
        block_size=block_size,
        w_init=w_init,
        input_norm_factor=input_norm_factor,
        samples_prefix=samples_prefix,
        pad_mode=pad_mode,
        name="block_lms",
        cpu_hint="lms()",
        training_symbols=training_symbols,
    )
    xp, C, n_sym, sps, block_size = run.xp, run.C, run.n_sym, run.sps, run.block_size
    num_taps = run.num_taps
    training = run.training

    if modulation is not None and order is not None:
        from ...mapping.gray import _gray_points

        reference_constellation = _gray_points(modulation, order, unipolar=unipolar)
        constellation_np = (
            to_device(reference_constellation, "cpu").flatten().astype(np.complex64)
        )
    elif training is not None:
        train_flat = to_device(training, "cpu").reshape(-1)
        constellation_np = np.unique(np.round(train_flat, decimals=8)).astype(
            np.complex64
        )
    else:
        raise ValueError("Provide modulation+order or training_symbols for DD slicer.")
    if pmf is not None and modulation is not None and order is not None:
        pmf_arr = np.asarray(pmf, dtype=np.float64)
        e_ps = float(np.dot(pmf_arr, np.abs(constellation_np).astype(np.float64) ** 2))
        if e_ps < 1.0 - 1e-6:
            constellation_np = (constellation_np / np.sqrt(e_ps)).astype(np.complex64)
    sq_side, sq_lev_min, sq_d_grid = _square_qam_slicer_params(constellation_np)
    slicer = _Slicer(
        constellation=xp.asarray(constellation_np),
        side=sq_side,
        lev_min=float(sq_lev_min),
        d_grid=float(sq_d_grid),
    )

    n_train = min(int(training.shape[-1]), n_sym) if training is not None else 0

    bps = None
    if cpr_type == "bps":
        bps = _block_bps(
            xp,
            C,
            n_sym,
            test_phases=int(cpr_bps_test_phases),
            window=int(cpr_bps_block_size),
            joint=bool(cpr_joint_channels),
            cycle_slip=bool(cpr_cycle_slip_correction),
            history=int(cpr_cycle_slip_history),
            threshold=float(cpr_cycle_slip_threshold),
            symmetry=_cpr_symmetry(modulation, order),
            cpr_state=cpr_state,
        )

    if xp is not np:
        from ... import _cuda

        M_const = int(constellation_np.size)
        if bps is not None and bps.P <= 128:
            if slicer.side > 0:
                bps.kernel = _cuda.get_kernel("bps_min_d2", mode="grid")
            elif M_const <= 1024:
                bps.kernel = _cuda.get_kernel("bps_min_d2", mode="table")
        if n_train < n_sym and slicer.side == 0 and M_const <= 1024:
            slicer.kernel = _cuda.get_kernel(
                "bps_min_d2", mode="table", return_argmin=True
            )
            if slicer.kernel is not None:
                slicer.phasor = xp.ones(1, dtype=xp.complex64)
        if bps is not None and bps.cs and C <= 1024:
            bps.cs_kernel = _cuda.get_kernel("cs_block")
            if bps.cs_kernel is not None:
                bps.cs_buf_y = xp.asarray(bps.cs_buf_y)
                bps.cs_buf_ptr = xp.asarray(bps.cs_buf_ptr)
                bps.cs_buf_n = xp.asarray(bps.cs_buf_n)
                bps.cs_stats = xp.asarray(bps.cs_stats)

    cpr_info = ""
    if bps is not None:
        cs_info = (
            f", cs_corr=True(thr={cpr_cycle_slip_threshold:.3f})"
            if bps.cs
            else ", cs_corr=False"
        )
        joint_info = ", joint" if cpr_joint_channels and C > 1 else ""
        cpr_info = (
            f", cpr=bps(P={cpr_bps_test_phases}, K={cpr_bps_block_size}"
            f"{joint_info}{cs_info})"
        )
    logger.info(
        "Block-LMS: C=%s, num_taps=%s, sps=%s, block_size=%s, fftsize=%s, "
        "mu=%s, n_sym=%s%s",
        C,
        num_taps,
        sps,
        block_size,
        run.fftsize,
        step_size,
        n_sym,
        cpr_info,
    )

    y_all = xp.empty((C, n_sym), dtype=xp.complex64)
    e_all = xp.empty((C, n_sym), dtype=xp.complex64)
    w_hist = (
        xp.empty((n_sym, C, C, num_taps), dtype=xp.complex64) if store_weights else None
    )
    phi_all = xp.zeros((C, n_sym), dtype=xp.float32) if bps is not None else None
    e_scatter = xp.zeros((C, run.fftsize), dtype=xp.complex64)
    y_rot_ws = xp.empty((C, block_size), dtype=xp.complex64)
    e_clean_ws = xp.empty((C, block_size), dtype=xp.complex64)
    phi_ws = xp.empty((C, block_size), dtype=xp.float32) if bps is not None else None
    div_flag = xp.zeros(1, dtype=xp.bool_)

    def run_block(B: int, b_start: int) -> None:
        """One block: filter, CPR, slicer/training error, update.

        Writes only persistent buffers, so a full decision-directed block can
        be captured into a CUDA graph.
        """
        nonlocal div_flag
        n_train_blk = max(0, min(n_train - b_start, B))
        y_block, X_fd = _fdaf_forward(run.h, run.x_win, run.fftsize, sps, B, xp)

        if bps is not None:
            phi_c, phi_traj = bps.phase(y_block, slicer, b_start, xp)
            y_rot = y_block * xp.exp(-1j * phi_c.astype(xp.complex64))  # (C, B)
            assert phi_ws is not None
            phi_ws[:, :B] = phi_traj  # unwrapped float32, for trajectory
        else:
            y_rot = y_block

        e_clean = xp.empty((C, B), dtype=xp.complex64)
        if n_train_blk > 0:
            assert training is not None
            d_train = training[:, b_start : b_start + n_train_blk]
            e_clean[:, :n_train_blk] = d_train - y_rot[:, :n_train_blk]
        if n_train_blk < B:
            y_dd = y_rot[:, n_train_blk:]
            e_clean[:, n_train_blk:] = slicer.decide(y_dd, xp) - y_dd

        y_rot_ws[:, :B] = y_rot
        e_clean_ws[:, :B] = e_clean
        if store_weights:
            assert w_hist is not None
            w_hist[b_start : b_start + B] = run.h[None, :, :, :]

        # The update works in the tap plane: undo the CPR rotation.
        if bps is not None:
            e_taps = e_clean * xp.exp(1j * phi_c.astype(xp.complex64))  # (C, B)
        else:
            e_taps = e_clean
        _fdaf_gradient_update(
            run.h, X_fd, e_taps, e_scatter, sps, B, num_taps, step_size, xp
        )
        div_flag |= ~xp.isfinite(run.h).all()

    def store(b_start: int, b_end: int, B: int) -> None:
        y_all[:, b_start:b_end] = y_rot_ws[:, :B]
        e_all[:, b_start:b_end] = e_clean_ws[:, :B]
        if bps is not None:
            assert phi_all is not None and phi_ws is not None
            phi_all[:, b_start:b_end] = phi_ws[:, :B]

    first_dd_full = ((n_train + block_size - 1) // block_size) * block_size
    n_dd_full = max(0, (n_sym - first_dd_full) // block_size)
    _block_loop(
        run,
        run_block=run_block,
        store=store,
        capturable=lambda B, b_start: B == block_size and n_train - b_start <= 0,
        use_graph=(
            cuda_graph
            and xp is not np
            and not store_weights
            and (bps is None or not bps.cs or bps.cs_kernel is not None)
            and n_dd_full >= 2  # need >= 1 warmup block + >= 1 captured block
        ),
        name="block_lms",
    )

    if bool(div_flag[0]):
        raise RuntimeError(
            f"block_lms diverged (step_size={step_size}, block_size={block_size}). "
            f"step_size is on the same scale as lms(), but because the weights are "
            f"frozen across the block the stability ceiling is ~{block_size}x lower "
            f"than per-symbol LMS. Reduce step_size until stable (e.g. try "
            f"{step_size / 2:.2e}, then keep halving) rather than dividing by "
            f"block_size, which would under-adapt the filter by that factor."
        )
    result = _assemble_block(run, y_all, e_all, w_hist, phi_all, n_train)
    if bps is not None:
        result.cpr_state = CPRState(
            bps_prev4=to_device(bps.prev4, "cpu").copy(),
            bps_offset4=to_device(bps.offset4, "cpu").copy(),
            bps_d2_hist=to_device(bps.d2_hist, "cpu"),
            cs_buf_x=bps.cs_buf_x.copy(),
            cs_buf_y=to_device(bps.cs_buf_y, "cpu").copy(),
            cs_buf_ptr=to_device(bps.cs_buf_ptr, "cpu").copy(),
            cs_buf_n=to_device(bps.cs_buf_n, "cpu").copy(),
            cs_stats=to_device(bps.cs_stats, "cpu").copy(),
            cpr_type=cpr_type,
            num_ch=C,
            symmetry=_cpr_symmetry(modulation, order),
            bps_P=bps.P,
            bps_K=bps.K,
            cs_H=bps.cs_H,
        )
    return _attach_equalized_signal(_log_equalizer_exit(result, name="Block-LMS"), sig)


def _block_bps(
    xp: Any,
    C: int,
    n_sym: int,
    *,
    test_phases: int,
    window: int,
    joint: bool,
    cycle_slip: bool,
    history: int,
    threshold: float,
    symmetry: int,
    cpr_state: CPRState | None,
) -> _BlockBps:
    """BPS state for ``block_lms``: warm start from a compatible ``cpr_state``."""
    P = test_phases
    angles_np = np.linspace(0.0, np.pi / 2.0, P, endpoint=False, dtype=np.float32)
    cs_H = min(history, n_sym)
    hist_len = max(0, window - 1)
    st = cpr_state
    if (
        st is not None
        and st.cpr_type == "bps"
        and st.num_ch == C
        and st.cs_H == cs_H
        and st.bps_P == P
        and st.bps_K == window
        and st.bps_prev4 is not None
    ):
        assert st.bps_offset4 is not None
        assert st.cs_buf_x is not None
        assert st.cs_buf_y is not None
        assert st.cs_buf_ptr is not None
        assert st.cs_buf_n is not None
        assert st.cs_stats is not None
        prev4 = st.bps_prev4.copy()
        offset4 = st.bps_offset4.copy()
        cs_buf_x = st.cs_buf_x.copy()
        cs_buf_y = st.cs_buf_y.copy()
        cs_buf_ptr = st.cs_buf_ptr.copy()
        cs_buf_n = st.cs_buf_n.copy()
        cs_stats = st.cs_stats.copy()
        d2_hist = (
            xp.array(st.bps_d2_hist, dtype=xp.float32)
            if st.bps_d2_hist is not None
            else xp.zeros((P, C, hist_len), dtype=xp.float32)
        )
    else:
        prev4 = np.zeros(C, dtype=np.float64)
        offset4 = np.zeros(C, dtype=np.float64)
        cs_buf_x = np.zeros((C, cs_H), dtype=np.float64)
        cs_buf_y = np.zeros((C, cs_H), dtype=np.float64)
        cs_buf_ptr = np.zeros(C, dtype=np.int64)
        cs_buf_n = np.zeros(C, dtype=np.int64)
        cs_stats = np.zeros((C, 4), dtype=np.float64)
        d2_hist = xp.zeros((P, C, hist_len), dtype=xp.float32)
    return _BlockBps(
        P=P,
        K=window,
        hist_len=hist_len,
        joint=joint,
        angles=xp.asarray(angles_np),
        phases_neg=xp.asarray(np.exp(-1j * angles_np).astype(np.complex64)),
        prev4=xp.asarray(prev4),
        offset4=xp.asarray(offset4),
        d2_hist=d2_hist,
        cs=cycle_slip,
        cs_H=cs_H,
        quantum=float(np.float64(2.0 * np.pi / symmetry)),
        threshold=threshold,
        cs_buf_x=cs_buf_x,
        cs_buf_y=cs_buf_y,
        cs_buf_ptr=cs_buf_ptr,
        cs_buf_n=cs_buf_n,
        cs_stats=cs_stats,
    )


def _assemble_block(
    run: _Block,
    y_all: ArrayType,
    e_all: ArrayType,
    w_hist: ArrayType | None,
    phi_all: ArrayType | None,
    n_train: int,
) -> EqualizerResult:
    """Device buffers -> ``EqualizerResult`` (SISO squeezed)."""
    if run.was_1d:
        return EqualizerResult(
            y_hat=y_all[0],
            weights=run.h[0, 0],
            error=e_all[0],
            weights_history=None if w_hist is None else w_hist[:, 0, 0, :],
            num_train_symbols=n_train,
            input_norm_factor=run.eq_norm,
            phase_trajectory=None if phi_all is None else phi_all[0],
        )
    return EqualizerResult(
        y_hat=y_all,
        weights=run.h,
        error=e_all,
        weights_history=w_hist,
        num_train_symbols=n_train,
        input_norm_factor=run.eq_norm,
        phase_trajectory=phi_all,
    )
