"""Shared front end, resumable kernel runs and result assembly of the
sequential equalizers.

``lms``, ``rls``, ``cma`` and ``rde`` validate their own parameters, call
:func:`_prepare_sequential` for the host-side setup, run their Numba kernel
through :func:`_run_resumable` and hand the buffers to
:func:`_assemble_sequential`.

Continuation (``state=``): the kernel runs in two segments split at the
``resume`` symbol, the last one whose filter window lies inside the data.
All loop state lives in the arrays passed to the kernel (weights, ``P``, the
CPR arrays), so the split changes nothing numerically; the snapshot taken
between the segments, plus the input from ``resume`` on, is the
:class:`EqualizerState` that a later call continues from.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..._array import restore_1d
from ...backend import ArrayType, dispatch, to_device
from ...logger import logger
from .._common import (
    _build_padded_samples,
    _cpr_symmetry,
    _init_butterfly_weights_numpy,
    _normalize_inputs,
    _prepare_training_numpy,
    _unpack_result_numpy,
    _validate_sps,
    _validate_w_init,
)
from ..result import EqualizerResult, EqualizerState, _check_state


@dataclass
class _Sequential:
    """Host buffers and geometry of one sequential equalizer run."""

    xp: Any  # array module of the input (results go back there)
    was_1d: bool
    num_ch: int
    n_sym: int
    stride: int
    num_taps: int
    x: np.ndarray  # (C, N_pad) complex64, normalized and padded
    lead: int  # samples of x before symbol 0's nominal position
    real_end: int  # x[:, :real_end] is data (or left padding), the rest zeros
    n_done: int  # symbols whose filter window lies inside x[:, :real_end]
    training: np.ndarray | None  # normalized training symbols (host)
    train_full: np.ndarray  # (C, n_sym) complex64, zero after the training
    n_train: int
    eq_norm: Any  # input_norm_factor actually applied
    W: np.ndarray  # (C, C, T) complex64, updated in place by the kernel
    y_out: np.ndarray  # (n_sym, C) complex64
    e_out: np.ndarray  # (n_sym, C) complex64
    w_hist: np.ndarray  # (n_sym or 1, C, C, T) complex64
    store_weights: bool

    def segment(
        self, start: int, stop: int
    ) -> tuple[np.ndarray, np.ndarray, np.int32, np.ndarray, np.ndarray, np.ndarray]:
        """Kernel inputs and output views for symbols ``[start, stop)``.

        Returns ``(x, training, n_train, y_out, e_out, w_hist)`` with indices
        relative to ``start`` (contiguous copies of the inputs past the first
        segment, so the kernels keep their compiled C layout).
        """
        if start == 0:
            x, train = self.x, self.train_full
        else:
            x = np.ascontiguousarray(self.x[:, start * self.stride :])
            train = np.ascontiguousarray(self.train_full[:, start:])
        w_hist = self.w_hist[start:stop] if self.store_weights else self.w_hist
        return (
            x,
            train,
            np.int32(self.n_train - start),
            self.y_out[start:stop],
            self.e_out[start:stop],
            w_hist,
        )


def _prepare_sequential(
    samples: ArrayType,
    *,
    equalizer: str,
    sps: int,
    num_taps: int,
    center_tap: int | None,
    initial_taps: ArrayType | None,
    state: EqualizerState | None,
    store_weights: bool,
    pad_mode: str,
    cpr: Any = None,
    training_symbols: ArrayType | None = None,
    pilot_mask: np.ndarray | None = None,
    pilot_gain_db: float = 0.0,
) -> _Sequential:
    """Validate, move to the host, normalize, pad and allocate.

    A cold start pads the record per ``pad_mode`` around ``center_tap``; a
    ``state`` prepends its pending input instead and reuses its weights and
    normalization.  ``pilot_mask`` with a non-zero ``pilot_gain_db``
    de-boosts the pilot samples before the normalization, so boosted pilots
    do not inflate the RMS estimate (blind equalizers).
    """
    samples, xp, _ = dispatch(samples)
    _validate_sps(sps, num_taps)
    stride = int(sps)

    if training_symbols is not None:
        training_symbols, _, _ = dispatch(training_symbols)

    was_1d = samples.ndim == 1
    num_ch = 1 if was_1d else samples.shape[0]
    n_new = samples.shape[-1]
    _check_state(
        state,
        equalizer=equalizer,
        num_taps=num_taps,
        sps=stride,
        num_ch=num_ch,
        cpr=cpr,
        initial_taps=initial_taps,
        center_tap=center_tap,
    )

    if state is None:
        n_sym = n_new // stride
        c_tap = center_tap if center_tap is not None else num_taps // 2
        pad_total = max(0, n_sym * stride - n_new + num_taps - 1)
        lead = min(c_tap, pad_total)
        pad_right = pad_total - lead
        offset = 0  # position of the new samples relative to symbol 0
    else:
        lead = state.lead
        offset = state.pending.shape[-1] - lead
        n_sym = (offset + n_new) // stride
        pad_right = max(0, n_sym * stride + num_taps - 1 - (lead + offset + n_new))

    if training_symbols is not None and training_symbols.shape[-1] > n_sym:
        logger.warning(
            "training_symbols length (%s) exceeds available symbol count "
            "(%s); excess training symbols will be ignored.",
            training_symbols.shape[-1],
            n_sym,
        )

    # Plain NumPy (no-op for CPU NumPy; downloads CuPy)
    samples_np = np.ascontiguousarray(to_device(samples, "cpu"), dtype=np.complex64)
    if pilot_mask is not None and pilot_gain_db != 0.0:
        # A copy: for NumPy complex64 input samples_np is the caller's array.
        samples_np = samples_np.copy()
        amp = np.float32(10.0 ** (pilot_gain_db / 20.0))
        smask = np.repeat(pilot_mask.astype(bool), stride)  # per sample of x
        if state is None:
            samples_np[..., smask] /= amp
        else:  # the pending samples were de-boosted by the previous call
            new = np.zeros(offset + n_new, dtype=bool)
            new[: smask.size] = smask[: offset + n_new]
            samples_np[..., new[offset:]] /= amp
    training_np = (
        to_device(training_symbols, "cpu").astype(np.complex64)
        if training_symbols is not None
        else None
    )
    samples_np, training_np, eq_norm = _normalize_inputs(
        samples_np,
        training_np,
        sps,
        input_norm_factor=None if state is None else state.input_norm_factor,
    )
    if state is None:
        x = np.ascontiguousarray(
            _build_padded_samples(
                samples_np, lead, pad_right, None, pad_mode, None, sps
            )
        )
        real_end = lead + n_new
    else:
        x = np.ascontiguousarray(
            np.concatenate(
                [
                    state.pending,
                    samples_np.reshape(num_ch, n_new),
                    np.zeros((num_ch, pad_right), dtype=np.complex64),
                ],
                axis=-1,
            ),
            dtype=np.complex64,
        )
        real_end = state.pending.shape[-1] + n_new
    # Symbol k's window is x[:, k*stride : k*stride + T]; it lies inside the
    # data while k*stride + T <= real_end.
    n_done = min(n_sym, max(0, (real_end - num_taps) // stride + 1))
    train_full, n_train = _prepare_training_numpy(training_np, num_ch, n_sym)

    if state is not None:
        W = state.weights.copy()
    elif initial_taps is not None:
        w_arr = np.ascontiguousarray(to_device(initial_taps, "cpu"), dtype=np.complex64)
        W = _validate_w_init(w_arr, num_ch, num_taps).copy()
    else:
        W = _init_butterfly_weights_numpy(num_ch, num_taps, center_tap=center_tap)

    y_out = np.empty((n_sym, num_ch), dtype=np.complex64)
    e_out = np.empty((n_sym, num_ch), dtype=np.complex64)
    w_hist = np.empty(
        (n_sym if store_weights else 1, num_ch, num_ch, num_taps), dtype=np.complex64
    )
    return _Sequential(
        xp=xp,
        was_1d=was_1d,
        num_ch=num_ch,
        n_sym=n_sym,
        stride=stride,
        num_taps=num_taps,
        x=x,
        lead=lead,
        real_end=real_end,
        n_done=n_done,
        training=training_np,
        train_full=train_full,
        n_train=int(n_train),
        eq_norm=eq_norm,
        W=W,
        y_out=y_out,
        e_out=e_out,
        w_hist=w_hist,
        store_weights=store_weights,
    )


def _run_resumable(
    run: _Sequential,
    resume: int,
    segment: Callable[[int, int], None],
    snapshot: Callable[[], Any],
) -> Any:
    """Run ``segment(0, resume)``, take ``snapshot()``, run the rest.

    Returns the snapshot: the loop state at symbol ``resume``.
    """
    if resume > 0:
        segment(0, resume)
    snap = snapshot()
    if resume < run.n_sym:
        segment(resume, run.n_sym)
    return snap


def _dd_constellation(
    constellation: Any, function_name: str, *, decisions: bool = True
) -> np.ndarray:
    """Decision constellation (complex64, unit power) for the DD slicer.

    A run without ``decisions`` (training covers every symbol) needs none;
    the kernel then gets a placeholder point it never reads.
    """
    if constellation is None and not decisions:
        return np.zeros(1, dtype=np.complex64)
    if constellation is None:
        raise ValueError(
            f"{function_name} needs a constellation for its decisions (pass "
            "constellation= or a Signal that has one)."
        )
    return np.ascontiguousarray(constellation.points, dtype=np.complex64)


@dataclass
class _CarrierArrays:
    """In-place CPR state of the inline LMS/RLS kernels (host)."""

    pll_phi: np.ndarray  # (C,) float64
    pll_freq: np.ndarray  # (C,) float64
    cs_buf_x: np.ndarray  # (C, H) float64
    cs_buf_y: np.ndarray  # (C, H) float64
    cs_buf_ptr: np.ndarray  # (C,) int64
    cs_buf_n: np.ndarray  # (C,) int64
    cs_stats: np.ndarray  # (C, 4) float64
    bps_prev4: np.ndarray  # (C,) float64
    bps_dist_buf: np.ndarray  # (C, K, B) float32, BPS window metrics
    bps_running_sum: np.ndarray  # (C, B) float64
    bps_dist_ptr: np.ndarray  # (1,) int64, window write position

    def args(self) -> tuple[np.ndarray, ...]:
        """The kernels' in-place CPR arguments, in their order."""
        return (
            self.pll_phi,
            self.pll_freq,
            self.cs_buf_x,
            self.cs_buf_y,
            self.cs_buf_ptr,
            self.cs_buf_n,
            self.cs_stats,
            self.bps_prev4,
            self.bps_dist_buf,
            self.bps_running_sum,
            self.bps_dist_ptr,
        )

    def copy(self) -> _CarrierArrays:
        return _CarrierArrays(*(a.copy() for a in self.args()))


@dataclass(frozen=True)
class _InlineCpr:
    """Kernel arguments of the inline CPR, resolved from a ``PLL``/``BPS``."""

    kind: str  # "pll" or "bps"
    pll_mu: Any
    pll_beta: Any
    phase_init: float
    angles: np.ndarray  # (B,) float32 BPS candidates over [0, 2π/symmetry)
    phases_neg: np.ndarray  # (B,) complex64, exp(-j*angle)
    window: int  # BPS averaging window
    joint: bool
    symmetry: int  # BPS search range and cycle-slip quantum, 2π/symmetry
    cycle_slip: bool
    history: int
    threshold: float

    def bps_args(self) -> tuple[Any, ...]:
        return (self.phases_neg, self.angles, np.int32(self.window), bool(self.joint))

    def loop_args(self) -> tuple[Any, ...]:
        return (
            np.int32(1 if self.kind == "pll" else 2),
            self.pll_mu,
            self.pll_beta,
            np.int32(self.symmetry),
            bool(self.cycle_slip),
            np.float32(self.threshold),
        )

    def cold_carrier(self, num_ch: int) -> _CarrierArrays:
        """Zero CPR state, the PLL at ``phase_init``."""
        return _CarrierArrays(
            pll_phi=np.full(num_ch, self.phase_init, dtype=np.float64),
            pll_freq=np.zeros(num_ch, dtype=np.float64),
            cs_buf_x=np.zeros((num_ch, self.history), dtype=np.float64),
            cs_buf_y=np.zeros((num_ch, self.history), dtype=np.float64),
            cs_buf_ptr=np.zeros(num_ch, dtype=np.int64),
            cs_buf_n=np.zeros(num_ch, dtype=np.int64),
            cs_stats=np.zeros((num_ch, 4), dtype=np.float64),
            bps_prev4=np.zeros(num_ch, dtype=np.float64),
            bps_dist_buf=np.zeros((num_ch, self.window, len(self.angles)), np.float32),
            bps_running_sum=np.zeros((num_ch, len(self.angles)), dtype=np.float64),
            bps_dist_ptr=np.zeros(1, dtype=np.int64),
        )


def _inline_cpr(cpr: Any, constellation: Any, function_name: str) -> _InlineCpr | None:
    """Resolve ``cpr=`` (``PLL``, ``BPS`` or ``None``) for the LMS/RLS kernels."""
    from ...recovery import BPS, PLL, CycleSlip

    if cpr is None:
        return None
    if not isinstance(cpr, PLL | BPS):
        raise TypeError(
            f"{function_name}: cpr must be a recovery.PLL or recovery.BPS, got "
            f"{type(cpr).__name__}."
        )
    cycle_slip = cpr.cycle_slip
    slip = cycle_slip if cycle_slip is not None else CycleSlip()
    if isinstance(cpr, PLL):
        mu, beta = cpr.gains
        test_phases, window, phase_init = 64, 32, float(cpr.phase_init)
    else:
        mu, beta = PLL().gains  # unused by the BPS path
        test_phases, window, phase_init = cpr.test_phases, cpr.block_size, 0.0
    symmetry = _cpr_symmetry(constellation)
    angles = np.linspace(
        0.0, 2.0 * np.pi / symmetry, int(test_phases), endpoint=False, dtype=np.float32
    )
    return _InlineCpr(
        kind="pll" if isinstance(cpr, PLL) else "bps",
        pll_mu=mu,
        pll_beta=beta,
        phase_init=phase_init,
        angles=angles,
        phases_neg=np.exp(-1j * angles).astype(np.complex64),
        window=int(window),
        joint=bool(cpr.joint_channels),
        symmetry=symmetry,
        cycle_slip=cycle_slip is not None,
        history=int(slip.history),
        threshold=float(slip.threshold),
    )


@dataclass(frozen=True)
class _Snapshot:
    """Loop state at the resume symbol (host copies)."""

    weights: np.ndarray
    carrier: _CarrierArrays | None = None
    inverse_correlation: np.ndarray | None = None


def _assemble_sequential(
    run: _Sequential,
    *,
    equalizer: str,
    cpr: Any,
    resume: int,
    snapshot: _Snapshot,
    n_sym: int | None = None,
    phase_out: np.ndarray | None = None,
) -> EqualizerResult:
    """Kernel buffers -> ``EqualizerResult`` on the input's device.

    ``n_sym`` truncates the outputs (RLS drops its zero-padded tail).
    ``phase_out`` attaches the inline CPR trajectory.  The state continues
    at symbol ``resume`` from ``snapshot``.
    """
    result = _unpack_result_numpy(
        run.y_out,
        run.e_out,
        run.W,
        run.w_hist,
        run.was_1d,
        run.store_weights,
        n_sym=n_sym,
        xp=run.xp,
        num_train_symbols=run.n_train,
        input_norm_factor=run.eq_norm,
    )
    if phase_out is not None:
        phi = phase_out if n_sym is None else phase_out[:n_sym]
        result.phase_trajectory = restore_1d(run.was_1d, run.xp.asarray(phi.T))
    n_out = run.n_sym if n_sym is None else n_sym
    result.state = EqualizerState(
        equalizer=equalizer,
        num_taps=run.num_taps,
        sps=run.stride,
        block_size=0,
        cpr=cpr,
        weights=snapshot.weights,
        input_norm_factor=run.eq_norm,
        pending=run.x[:, resume * run.stride : run.real_end].copy(),
        lead=run.lead,
        overlap=max(0, n_out - resume),
        inverse_correlation=snapshot.inverse_correlation,
        carrier=snapshot.carrier,
    )
    return result
