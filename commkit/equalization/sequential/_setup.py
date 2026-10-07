"""Shared front end and result assembly of the sequential equalizers.

``lms``, ``rls``, ``cma`` and ``rde`` validate their own parameters, call
:func:`_prepare_sequential` for the host-side setup, run their Numba kernel
and hand the buffers to :func:`_assemble_sequential`.
"""

from __future__ import annotations

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
from .._kernels_numba import _get_numba
from ..result import CPRState, EqualizerResult


@dataclass
class _Sequential:
    """Host buffers and geometry of one sequential equalizer run."""

    xp: Any  # array module of the input (results go back there)
    was_1d: bool
    num_ch: int
    n_sym: int
    stride: int
    x: np.ndarray  # (C, N_pad) complex64, normalized and padded
    training: np.ndarray | None  # normalized training symbols (host)
    train_full: np.ndarray  # (C, n_sym) complex64, zero after the training
    n_train: int
    eq_norm: Any  # input_norm_factor actually applied
    W: np.ndarray  # (C, C, T) complex64, updated in place by the kernel
    y_out: np.ndarray  # (n_sym, C) complex64
    e_out: np.ndarray  # (n_sym, C) complex64
    w_hist: np.ndarray  # (n_sym or 1, C, C, T) complex64
    store_weights: bool


def _prepare_sequential(
    samples: ArrayType,
    *,
    sps: int,
    num_taps: int,
    center_tap: int | None,
    w_init: ArrayType | None,
    store_weights: bool,
    input_norm_factor: float | np.ndarray | None,
    samples_prefix: ArrayType | None,
    pad_mode: str,
    training_symbols: ArrayType | None = None,
    pilot_mask: np.ndarray | None = None,
    pilot_gain_db: float = 0.0,
) -> _Sequential:
    """Validate, move to the host, normalize, pad and allocate.

    ``pilot_mask`` with a non-zero ``pilot_gain_db`` de-boosts the pilot
    samples before the normalization, so boosted pilots do not inflate the
    RMS estimate (blind equalizers).
    """
    samples, xp, _ = dispatch(samples)
    _validate_sps(sps, num_taps)
    stride = int(sps)

    if training_symbols is not None:
        training_symbols, _, _ = dispatch(training_symbols)

    was_1d = samples.ndim == 1
    num_ch = 1 if was_1d else samples.shape[0]
    n_samples = samples.shape[-1]
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

    if _get_numba() is None:
        raise ImportError("Numba is required for the sequential equalizers.")

    # Plain NumPy (no-op for CPU NumPy; downloads CuPy)
    samples_np = np.ascontiguousarray(to_device(samples, "cpu"), dtype=np.complex64)
    if pilot_mask is not None and pilot_gain_db != 0.0:
        # A copy: for NumPy complex64 input samples_np is the caller's array.
        samples_np = samples_np.copy()
        amp = np.float32(10.0 ** (pilot_gain_db / 20.0))
        smask = np.repeat(pilot_mask.astype(bool), stride)  # (N_samples,)
        samples_np[..., smask] /= amp
    training_np = (
        to_device(training_symbols, "cpu").astype(np.complex64)
        if training_symbols is not None
        else None
    )
    samples_np, training_np, eq_norm = _normalize_inputs(
        samples_np, training_np, sps, input_norm_factor=input_norm_factor
    )
    x = np.ascontiguousarray(
        _build_padded_samples(
            samples_np, pad_left, pad_right, samples_prefix, pad_mode, eq_norm, sps
        )
    )
    train_full, n_train = _prepare_training_numpy(training_np, num_ch, n_sym)

    if w_init is not None:
        w_arr = np.ascontiguousarray(to_device(w_init, "cpu"), dtype=np.complex64)
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
        x=x,
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


@dataclass(frozen=True)
class _InlineCpr:
    """Kernel arguments of the inline CPR, resolved from a ``PLL``/``BPS``."""

    kind: str  # "pll" or "bps" (tag of the CPRState)
    pll_mu: Any
    pll_beta: Any
    phase_init: float
    angles: np.ndarray  # (B,) float32 BPS candidates over [0, π/2)
    phases_neg: np.ndarray  # (B,) complex64, exp(-j*angle)
    window: int  # BPS averaging window
    joint: bool
    symmetry: int  # cycle-slip quantum 2π/symmetry
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

    def state_tags(self, num_ch: int) -> dict[str, Any]:
        return dict(
            cpr_type=self.kind,
            num_ch=num_ch,
            symmetry=self.symmetry,
            bps_P=len(self.angles),
            bps_K=self.window,
            cs_H=self.history,
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
    angles = np.linspace(
        0.0, np.pi / 2.0, int(test_phases), endpoint=False, dtype=np.float32
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
        symmetry=_cpr_symmetry(constellation),
        cycle_slip=cycle_slip is not None,
        history=int(slip.history),
        threshold=float(slip.threshold),
    )


def _carrier_arrays(
    cpr_state: CPRState | None, inline: _InlineCpr, num_ch: int
) -> _CarrierArrays:
    """Warm-start CPR arrays from a compatible ``cpr_state``, else a cold start
    (zeros, the PLL at ``phase_init``)."""
    st = cpr_state
    history = inline.history
    if (
        st is not None
        and st.cpr_type == inline.kind
        and st.num_ch == num_ch
        and st.cs_H == history
        and st.pll_phi is not None
    ):
        assert st.pll_freq is not None
        assert st.cs_buf_x is not None
        assert st.cs_buf_y is not None
        assert st.cs_buf_ptr is not None
        assert st.cs_buf_n is not None
        assert st.cs_stats is not None
        return _CarrierArrays(
            pll_phi=st.pll_phi.copy(),
            pll_freq=st.pll_freq.copy(),
            cs_buf_x=st.cs_buf_x.copy(),
            cs_buf_y=st.cs_buf_y.copy(),
            cs_buf_ptr=st.cs_buf_ptr.copy(),
            cs_buf_n=st.cs_buf_n.copy(),
            cs_stats=st.cs_stats.copy(),
            bps_prev4=(
                st.bps_prev4.copy()
                if st.bps_prev4 is not None
                else np.zeros(num_ch, dtype=np.float64)
            ),
        )
    return _CarrierArrays(
        pll_phi=np.full(num_ch, inline.phase_init, dtype=np.float64),
        pll_freq=np.zeros(num_ch, dtype=np.float64),
        cs_buf_x=np.zeros((num_ch, history), dtype=np.float64),
        cs_buf_y=np.zeros((num_ch, history), dtype=np.float64),
        cs_buf_ptr=np.zeros(num_ch, dtype=np.int64),
        cs_buf_n=np.zeros(num_ch, dtype=np.int64),
        cs_stats=np.zeros((num_ch, 4), dtype=np.float64),
        bps_prev4=np.zeros(num_ch, dtype=np.float64),
    )


def _carrier_args(c: _CarrierArrays) -> tuple[np.ndarray, ...]:
    """The kernels' in-place CPR arguments, in their order."""
    return (
        c.pll_phi,
        c.pll_freq,
        c.cs_buf_x,
        c.cs_buf_y,
        c.cs_buf_ptr,
        c.cs_buf_n,
        c.cs_stats,
        c.bps_prev4,
    )


def _assemble_sequential(
    run: _Sequential,
    *,
    n_sym: int | None = None,
    phase_out: np.ndarray | None = None,
    carrier: _CarrierArrays | None = None,
    cpr_state_tags: dict[str, Any] | None = None,
) -> EqualizerResult:
    """Kernel buffers -> ``EqualizerResult`` on the input's device.

    ``n_sym`` truncates the outputs (RLS drops its zero-padded tail).
    ``phase_out`` and ``carrier`` attach the inline CPR trajectory and state.
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
    if carrier is not None:
        result.cpr_state = CPRState(
            pll_phi=carrier.pll_phi.copy(),
            pll_freq=carrier.pll_freq.copy(),
            bps_prev4=carrier.bps_prev4.copy(),
            cs_buf_x=carrier.cs_buf_x.copy(),
            cs_buf_y=carrier.cs_buf_y.copy(),
            cs_buf_ptr=carrier.cs_buf_ptr.copy(),
            cs_buf_n=carrier.cs_buf_n.copy(),
            cs_stats=carrier.cs_stats.copy(),
            **(cpr_state_tags or {}),
        )
    return result
