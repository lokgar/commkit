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
from ...recovery import BPS, CycleSlip
from .._common import _cpr_symmetry
from .._kernels_numba import _get_numba_cs_block
from ..result import (
    EqualizerResult,
    EqualizerState,
    _attach_equalized_signal,
    _log_equalizer_exit,
)
from ._engine import (
    _Block,
    _block_loop,
    _block_state,
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


@dataclass(frozen=True)
class _BlockCarrier:
    """Host copy of the cross-block BPS state, for ``EqualizerState``."""

    prev4: np.ndarray
    offset4: np.ndarray
    d2_hist: np.ndarray
    cs_buf_x: np.ndarray
    cs_buf_y: np.ndarray
    cs_buf_ptr: np.ndarray
    cs_buf_n: np.ndarray
    cs_stats: np.ndarray
    cs_H: int


@dataclass
class _BlockBps:
    """Blind phase search across blocks: candidates, window history, S-fold
    unwrap and cycle-slip state (device arrays; host for the Numba slip path).
    """

    P: int
    K: int  # BPS averaging window
    hist_len: int
    joint: bool
    angles: ArrayType  # (P,) float32
    phases_neg: ArrayType  # (P,) complex64, exp(-j*angle)
    prev4: ArrayType  # (C,) float64, last S x angle
    offset4: ArrayType  # (C,) float64, accumulated unwrapped S x phase
    d2_hist: ArrayType  # (P, C, K-1) float32 trailing metrics
    cs: bool
    cs_H: int
    symmetry: int  # S: candidates over [0, 2π/S), S-fold unwrap
    quantum: float
    threshold: float
    cs_buf_x: np.ndarray
    cs_buf_y: Any
    cs_buf_ptr: Any
    cs_buf_n: Any
    cs_stats: Any
    da_hist: Any = None  # (C, K-1) trailing y·conj(d) (complex64 on the kernel path)
    anchor_kernel: Any = None  # CUDA data-aided anchor kernel, or None
    kernel: Any = None  # CUDA min-distance kernel, or None
    cs_kernel: Any = None  # CUDA cycle-slip kernel, or None

    def snapshot(self) -> _BlockCarrier:
        """Host copies of the state, at a block boundary."""
        return _BlockCarrier(
            prev4=to_device(self.prev4, "cpu").copy(),
            offset4=to_device(self.offset4, "cpu").copy(),
            d2_hist=to_device(self.d2_hist, "cpu").copy(),
            cs_buf_x=self.cs_buf_x.copy(),
            cs_buf_y=to_device(self.cs_buf_y, "cpu").copy(),
            cs_buf_ptr=to_device(self.cs_buf_ptr, "cpu").copy(),
            cs_buf_n=to_device(self.cs_buf_n, "cpu").copy(),
            cs_stats=to_device(self.cs_stats, "cpu").copy(),
            cs_H=self.cs_H,
        )

    def _anchor(self, y: ArrayType, d: ArrayType, xp: Any) -> ArrayType:
        """Data-aided phase over the training columns, ``(C, n)`` float64.

        The angle of ``sum(y conj(d))`` over a causal ``K``-symbol window
        (continued across blocks), unwrapped from the current phase over the
        full 2π.  The unwrap state then continues from it, so blind BPS keeps
        its branch once decisions take over.
        """
        C = y.shape[0]
        K = self.K
        S = float(self.symmetry)
        if self.anchor_kernel is not None:
            # One launch per block and no host synchronization (the NumPy
            # path below is its reference).
            if self.da_hist is None:
                self.da_hist = xp.zeros((C, max(0, K - 1)), dtype=xp.complex64)
            phi, hist = self.anchor_kernel(
                xp.ascontiguousarray(y, dtype=xp.complex64),
                xp.ascontiguousarray(d, dtype=xp.complex64),
                self.da_hist,
                self.offset4,
                self.prev4,
                symmetry=self.symmetry,
                joint=self.joint and C > 1,
            )
            # In place: a captured graph keeps reading this buffer.
            self.da_hist[...] = hist
            return phi
        # Host NumPy: a training block's (C, n) products are tiny, so the
        # CuPy fallback brings them over once instead of a dozen launches.
        if self.da_hist is None:
            self.da_hist = np.zeros((C, max(0, K - 1)), dtype=np.complex128)
        y_h = np.asarray(to_device(y, "cpu"), dtype=np.complex128)
        d_h = np.asarray(to_device(d, "cpu"), dtype=np.complex128)
        cat = np.concatenate([self.da_hist, y_h * np.conj(d_h)], axis=1)
        cs = np.concatenate(
            [np.zeros((C, 1), dtype=np.complex128), np.cumsum(cat, axis=1)], axis=1
        )
        win = cs[:, K:] - cs[:, :-K]  # (C, n), causal K-symbol window
        if self.joint and C > 1:
            win = np.broadcast_to(win.sum(axis=0, keepdims=True), win.shape)
        offset = np.asarray(to_device(self.offset4, "cpu"), dtype=np.float64)
        ext = np.concatenate([(offset / S)[:, None], np.angle(win)], axis=1)
        phi = np.unwrap(ext, axis=1)[:, 1:]
        if K > 1:
            self.da_hist = cat[:, -(K - 1) :].copy()
        self.offset4[...] = xp.asarray(phi[:, -1] * S)
        self.prev4[...] = xp.asarray(phi[:, -1] * S)
        return xp.asarray(phi)

    def phase(
        self,
        y_block: ArrayType,
        slicer: _Slicer,
        b_start: int,
        xp: Any,
        d_train: ArrayType | None = None,
    ) -> tuple[ArrayType, ArrayType]:
        """Wrapped float32 phase to rotate by, and the unwrapped trajectory.

        The first ``d_train.shape[-1]`` columns are training symbols: their
        phase is data-aided (see ``_anchor``) and blind BPS covers the rest.
        """
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

        n_da = 0 if d_train is None else int(d_train.shape[-1])
        phi_da = self._anchor(y_block[:, :n_da], d_train, xp) if n_da else None
        phi_raw = phi_raw[:, n_da:]

        # S-fold unwrap continuing from the previous block (or the anchor).
        S = float(self.symmetry)
        if phi_raw.shape[1] == 0:
            phi_f64 = xp.empty((C, 0), dtype=xp.float64)
        elif xp is np:
            raw4 = phi_raw.astype(np.float64) * S  # (C, B)
            extended = np.concatenate([self.prev4[:, np.newaxis], raw4], axis=1)
            unwrapped_ext = np.unwrap(extended, axis=1)  # (C, B+1)
            cumul = unwrapped_ext[:, 1:] - unwrapped_ext[:, 0:1]  # (C, B)
            phi_f64 = (self.offset4[:, np.newaxis] + cumul) / S  # (C, B)
            self.prev4[:] = unwrapped_ext[:, -1]
            self.offset4 += cumul[:, -1]
        else:
            raw4_dev = phi_raw.astype(xp.float64) * xp.float64(S)  # (C, B)
            ext_dev = xp.concatenate([self.prev4[:, None], raw4_dev], axis=1)
            two_pi = xp.float64(2.0 * np.pi)
            d4 = ext_dev[:, 1:] - ext_dev[:, :-1]  # (C, B)
            d4 -= xp.round(d4 / two_pi) * two_pi  # wrap to [-π, π]
            cumul_dev = xp.cumsum(d4, axis=1)  # (C, B)
            phi_f64 = (self.offset4[:, None] + cumul_dev) / xp.float64(S)
            self.prev4 += cumul_dev[:, -1]
            self.offset4 += cumul_dev[:, -1]
        if phi_da is not None:
            phi_f64 = xp.concatenate([phi_da, phi_f64], axis=1)

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
            self.offset4 += (phi_corr[:, -1] - phi_f64[:, -1]) * S
            phi_f64 = phi_corr

        two_pi = xp.float64(2.0 * np.pi)
        phi_wrapped = (phi_f64 - xp.round(phi_f64 / two_pi) * two_pi).astype(xp.float32)
        return phi_wrapped, phi_f64.astype(xp.float32)


def block_lms(
    samples: ArrayType | Signal,
    training_symbols: ArrayType | None = None,
    *,
    num_taps: int = 21,
    sps: int | None = None,
    step_size: float = 2e-4,
    block_size: int = 256,
    constellation: Any = None,
    store_weights: bool = False,
    initial_taps: ArrayType | None = None,
    cpr: BPS | None = None,
    pad_mode: str = "zeros",
    state: EqualizerState | None = None,
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

    2. **BPS phase recovery** (if ``cpr`` is set) - for each symbol in the
       block, averages the min-distance metric over a causal trailing window of
       ``cpr.block_size`` symbols and picks the minimum-metric candidate
       rotation.  This produces one phase estimate ``phi_n`` per symbol
       (not one per block), so ``cpr.block_size`` and ``block_size`` are
       independent parameters: ``block_size`` controls FFT/gradient efficiency
       while ``cpr.block_size`` controls phase noise suppression.  The raw
       ``[0, 2*pi/S)`` argmin (``S`` the constellation's rotational symmetry)
       is converted to full-range radians by a causal ``S``-fold unwrap, and stored in a float64 accumulator in ``phase_trajectory``.

    3. **Cycle-slip correction** (if ``cpr.cycle_slip`` is set) - for
       each symbol of the per-symbol BPS phase tensor ``phi_n`` (shape
       ``(C, B)``) the phase is compared to a linear-regression prediction
       built from a circular buffer of ``history`` past
       corrected phases (identical algorithm to ``lms`` with
       ``cpr`` set).  If ``|phi_n - phi_pred| > threshold``
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
        A :class:`Signal` supplies ``sps`` and ``constellation``; the result
        then also carries ``signal``, the 1-SPS output Signal.
    training_symbols : array_like, optional
        Known transmitted symbols at 1 SPS, on the scale of the unit-power
        ``constellation`` (used as given).
        Shape: ``(N_train,)`` for SISO or ``(C, N_train)`` for MIMO.
    num_taps : int, default 21
        Number of taps per FIR filter (tap count in samples).
    sps : int, optional
        Samples per symbol at the input, an integer (2 is T/2-spaced).
        Taken from the Signal; required for array input.  A value that
        disagrees with the Signal raises.
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
        of the BPS averaging window (``cpr.block_size``).
    constellation : Constellation, optional
        Decision constellation for the slicer, unit power (a shaped
        constellation carries its pmf).  Defaults to the Signal's
        ``constellation``; required for array input.
    store_weights : bool, default False
        If ``True``, stores the weight tensor at every block start in
        ``EqualizerResult.weights_history``.
    initial_taps : array_like, optional
        Initial tap weights instead of the center-tap identity, shape
        ``(C, C, T)`` or the SISO short-hands.  Only for a cold start.
    cpr : BPS, optional
        Inline blind phase search (``commkit.recovery.BPS``): ``test_phases``
        candidates in ``[0, 2π/S)`` (``S`` the constellation's rotational
        symmetry), a causal window of the last
        ``block_size`` symbols per output symbol (independent of the
        equalizer's ``block_size``), ``joint_channels`` to sum the metrics
        across MIMO channels, and an optional nested ``CycleSlip``.  The PLL
        is not available: its per-symbol integration does not fit the block
        gradient.
    pad_mode : {'zeros', 'edge'}, default 'zeros'
        Left padding of a cold start; see :func:`lms`.
    state : EqualizerState, optional
        Continue from ``result.state`` of a previous call with the same
        configuration; see :func:`lms`.  The state is taken at the last block
        boundary before the zero-padded tail.
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
        * ``state`` - :class:`EqualizerState` for ``state=`` continuation,
          including the BPS accumulators.

        ``phase_trajectory`` is populated with ``cpr``; shape
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

    **BPS cycle-slip correction (``cpr.cycle_slip``):**
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
    sps = require_integer_sps(signal_adapter.resolve_fact("sps", sps), "block_lms()")
    constellation = signal_adapter.resolve_choice("constellation", constellation)

    if cpr is not None and not isinstance(cpr, BPS):
        raise TypeError(
            "block_lms(): cpr must be a recovery.BPS (the PLL's per-symbol "
            f"integration does not fit the block gradient), got {type(cpr).__name__}."
        )

    run = _prepare_block(
        samples,
        equalizer="block_lms",
        sps=sps,
        num_taps=num_taps,
        block_size=block_size,
        initial_taps=initial_taps,
        state=state,
        pad_mode=pad_mode,
        cpr=cpr,
        name="block_lms",
        cpu_hint="lms()",
        training_symbols=training_symbols,
    )
    xp, C, n_sym, sps, block_size = run.xp, run.C, run.n_sym, run.sps, run.block_size
    num_taps = run.num_taps
    training = run.training

    if constellation is None:
        raise ValueError(
            "block_lms() needs a constellation for its decisions (pass "
            "constellation= or a Signal that has one)."
        )
    constellation_np = np.ascontiguousarray(constellation.points, dtype=np.complex64)
    sq_side, sq_lev_min, sq_d_grid = _square_qam_slicer_params(constellation_np)
    slicer = _Slicer(
        constellation=xp.asarray(constellation_np),
        side=sq_side,
        lev_min=float(sq_lev_min),
        d_grid=float(sq_d_grid),
    )

    n_train = min(int(training.shape[-1]), n_sym) if training is not None else 0

    bps = None
    if cpr is not None:
        slip = cpr.cycle_slip if cpr.cycle_slip is not None else CycleSlip()
        bps = _block_bps(
            xp,
            C,
            n_sym,
            test_phases=int(cpr.test_phases),
            window=int(cpr.block_size),
            joint=bool(cpr.joint_channels),
            cycle_slip=cpr.cycle_slip is not None,
            history=int(slip.history),
            threshold=float(slip.threshold),
            symmetry=_cpr_symmetry(constellation),
            carrier=None if state is None else state.carrier,
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
        if bps is not None and n_train > 0:
            bps.anchor_kernel = _cuda.get_kernel("bps_anchor")
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
            f", cs_corr=True(thr={bps.threshold:.3f})" if bps.cs else ", cs_corr=False"
        )
        joint_info = ", joint" if bps.joint and C > 1 else ""
        cpr_info = f", cpr=bps(P={bps.P}, K={bps.K}{joint_info}{cs_info})"
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
    # Training symbols of a fully trained block, staged by prepare() so that
    # a captured graph reads them from a fixed address.
    d_ws = xp.empty((C, block_size), dtype=xp.complex64) if n_train else None
    div_flag = xp.zeros(1, dtype=xp.bool_)

    def prepare(B: int, b_start: int) -> None:
        if d_ws is not None and n_train - b_start >= B:
            assert training is not None
            d_ws[:, :B] = training[:, b_start : b_start + B]

    def run_block(B: int, b_start: int) -> None:
        """One block: filter, CPR, slicer/training error, update.

        Writes only persistent buffers and reads training symbols from the
        staged d_ws, so a full decision-directed block and a full training
        block can each be captured into a CUDA graph.
        """
        nonlocal div_flag
        n_train_blk = max(0, min(n_train - b_start, B))
        y_block, X_fd = _fdaf_forward(run.h, run.x_win, run.fftsize, sps, B, xp)
        d_train = None
        if n_train_blk == B:
            assert d_ws is not None
            d_train = d_ws[:, :B]
        elif n_train_blk > 0:  # the one partially trained block, eager
            assert training is not None
            d_train = training[:, b_start : b_start + n_train_blk]

        if bps is not None:
            phi_c, phi_traj = bps.phase(y_block, slicer, b_start, xp, d_train)
            y_rot = y_block * xp.exp(-1j * phi_c.astype(xp.complex64))  # (C, B)
            assert phi_ws is not None
            phi_ws[:, :B] = phi_traj  # unwrapped float32, for trajectory
        else:
            y_rot = y_block

        e_clean = xp.empty((C, B), dtype=xp.complex64)
        if d_train is not None:
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

    snap: list[tuple[np.ndarray, Any]] = []

    def at_resume() -> None:
        weights = to_device(run.h, "cpu").copy()
        snap.append((weights, None if bps is None else bps.snapshot()))

    first_dd_full = ((n_train + block_size - 1) // block_size) * block_size
    n_dd_full = max(0, (n_sym - first_dd_full) // block_size)
    n_train_full = min(n_train, n_sym) // block_size
    # A training block needs the device anchor (the NumPy one syncs).
    train_graph = bps is None or bps.anchor_kernel is not None

    def capturable(B: int, b_start: int) -> str | None:
        if B != block_size:
            return None
        if n_train - b_start <= 0:
            return "dd"
        if train_graph and n_train - b_start >= B:
            return "train"
        return None

    _block_loop(
        run,
        run_block=run_block,
        store=store,
        capturable=capturable,
        use_graph=(
            cuda_graph
            and xp is not np
            and not store_weights
            and (bps is None or not bps.cs or bps.cs_kernel is not None)
            # need >= 1 warmup block + >= 1 captured block of a kind
            and (n_dd_full >= 2 or (train_graph and n_train_full >= 2))
        ),
        name="block_lms",
        at_resume=at_resume,
        prepare=prepare,
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
    weights, carrier = snap[0]
    result.state = _block_state(
        run, equalizer="block_lms", cpr=cpr, weights=weights, carrier=carrier
    )
    return _attach_equalized_signal(
        _log_equalizer_exit(result, name="Block-LMS"), sig, state
    )


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
    carrier: _BlockCarrier | None,
) -> _BlockBps:
    """BPS state for ``block_lms``: continued from ``carrier``, or cold."""
    P = test_phases
    angles_np = np.linspace(
        0.0, 2.0 * np.pi / symmetry, P, endpoint=False, dtype=np.float32
    )
    hist_len = max(0, window - 1)
    if carrier is not None:
        cs_H = carrier.cs_H
        prev4 = carrier.prev4.copy()
        offset4 = carrier.offset4.copy()
        cs_buf_x = carrier.cs_buf_x.copy()
        cs_buf_y = carrier.cs_buf_y.copy()
        cs_buf_ptr = carrier.cs_buf_ptr.copy()
        cs_buf_n = carrier.cs_buf_n.copy()
        cs_stats = carrier.cs_stats.copy()
        d2_hist = xp.array(carrier.d2_hist, dtype=xp.float32)
    else:
        cs_H = min(history, n_sym)
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
        symmetry=symmetry,
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
