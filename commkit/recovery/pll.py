"""Decision-directed PLL carrier-phase recovery: :class:`PLL`."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..backend import ArrayType, dispatch, to_device
from ..mapping._lattice import square_grid
from ._common import _Context, _Phase, _resolve_pll_gains
from .corrections import (
    _HISTORY_SPAN,
    CycleSlip,
    _log_phase_summary,
    _repair_slips,
)

__all__ = ["PLL"]


def _get_numba_dd_pll() -> Any:
    """The PLL kernel: one loop per channel, or one shared by the channels."""
    from . import _numba_loops

    return _numba_loops.dd_pll


@dataclass(frozen=True)
class PLL:
    r"""Decision-directed phase-locked loop.

    Tracks the carrier phase symbol by symbol from hard decisions:
    derotate by ``φ̂[n]``, decide, take the cross-product error
    ``e[n] = Im(y[n]·d̂*[n])``, then ``φ̂[n+1] = φ̂[n] + μ·e[n] + ν[n]`` and
    ``ν[n+1] = ν[n] + β·e[n]``.  A first-order loop (``β = 0``) tracks the phase;
    ``β > 0`` also tracks a residual frequency offset.  The same object sets
    the inline PLL of the equalizers.

    Parameters
    ----------
    normalized_bandwidth : float, default 1e-3
        Nominal one-sided loop bandwidth ``B·T`` in ``(0, 0.5)``, a fraction
        of the symbol rate (not Hz: ``1e-3`` at 32 GBd is 32 MHz).  Gives
        critically damped gains ``μ = 4BT``, ``β = 4(BT)²``: ``ζ = 1`` and
        ``ω_n T = 2BT``, whose noise bandwidth is ``1.25·B·T`` (``B·T`` is
        exact for a first-order loop only).  Used when ``proportional_gain``
        is not given.
    proportional_gain : float, optional
        Raw proportional gain ``μ``; overrides ``normalized_bandwidth``.
        Typical values ``1e-3`` (high-order QAM, high SNR) to ``5e-2``
        (QPSK).
    integral_gain : float, optional
        Raw integral gain ``β`` (``β ≈ μ²/4`` is critically damped);
        requires ``proportional_gain``.  Defaults to 0 (a first-order loop)
        with ``proportional_gain``.
    phase_init : float, default 0.0
        Initial phase in radians, e.g. the last value of a BPS estimate.
    joint_channels : bool, default False
        MIMO: average the phase error across channels at each symbol and
        drive one shared loop (shared LO).
    cycle_slip : CycleSlip, optional
        Repair slips of ``2π/M`` (M the rotational symmetry, 4 for QAM)
        in the per-symbol trajectory after the loop.  Its default history
        is ``8/μ`` symbols (between 10 and 640).  It does not undo the
        loop's own slips, which drift over about ``1/(B·T)`` symbols (see
        :class:`CycleSlip`).

    Notes
    -----
    The loop needs reliable decisions; a cold start converges over about
    ``1/μ`` symbols.  Numba-compiled on the CPU; GPU input makes one host
    round trip.  A global M-fold ambiguity remains.
    """

    normalized_bandwidth: float = 1e-3
    proportional_gain: float | None = None
    integral_gain: float | None = None
    phase_init: float = 0.0
    joint_channels: bool = False
    cycle_slip: CycleSlip | None = None

    def __post_init__(self) -> None:
        bt = self.normalized_bandwidth
        if not (0.0 < bt < 0.5):
            raise ValueError(
                f"PLL: normalized_bandwidth must be in (0, 0.5) (a fraction of the "
                f"symbol rate), got {bt}."
            )
        # validates integral_gain
        _resolve_pll_gains(bt, self.proportional_gain, self.integral_gain)

    @property
    def gains(self) -> tuple[Any, Any]:
        """The loop gains ``(μ, β)``: proportional and integral."""
        return _resolve_pll_gains(
            self.normalized_bandwidth, self.proportional_gain, self.integral_gain
        )

    @property
    def _slip_history(self) -> int:
        """Default cycle-slip history: twice the loop memory, ``8/μ`` symbols.

        At least 10 and at most the general 640-symbol span (also for a
        frozen loop, ``proportional_gain=0``).
        """
        mu = float(self.gains[0])
        if mu <= 0.0:
            return _HISTORY_SPAN
        return max(10, min(_HISTORY_SPAN, round(8.0 / mu)))


def _pll(symbols: ArrayType, method: PLL, ctx: _Context) -> _Phase:
    """Decision-directed PLL phase of ``(C, N)`` symbols."""
    from ..math import normalize

    constellation = ctx.need_constellation(method)
    mu, beta = method.gains
    phase_init = method.phase_init

    symbols, xp, _ = dispatch(symbols)
    C, N = symbols.shape

    # Normalize to unit average power so the effective loop gain is mu regardless
    # of input amplitude.  The error signal is e[n] = Im(y[n]*d_hat*), which
    # scales with signal amplitude; without this, the effective gain is mu*A
    # (where A is the RMS amplitude), making loop bandwidth input-dependent.
    symbols = normalize(symbols, mode="average_power", axis=-1)

    # Constellation on CPU (decisions are scalar operations in the loop)
    const_np = np.asarray(constellation.points, dtype=np.complex128)
    const_r = const_np.real.copy()
    const_i = const_np.imag.copy()

    # Square-QAM O(1) decision parameters.  For a square lattice the nearest
    # point is found by rounding to the closest level per axis.  The levels
    # are float64, like the loop: the decided point is exactly a
    # constellation point.
    grid = square_grid(constellation)
    _is_sq_qam = grid is not None
    if grid is not None:
        _lev_min, _d_grid, _side, _ = grid
        _levels = _lev_min + np.arange(_side) * _d_grid
    else:
        _lev_min, _d_grid, _side = 0.0, 1.0, 0
        _levels = np.empty(0, dtype=np.float64)

    # Sequential loop on the host, in the samples' own precision: the kernels
    # widen each symbol to float64 as they read it.
    symbols_cpu = np.ascontiguousarray(to_device(symbols, "cpu"))
    loop_args = (
        const_r,
        const_i,
        float(mu),
        float(beta),
        float(phase_init),
        0.0,
        _is_sq_qam,
        _levels,
        _d_grid,
        _lev_min,
        _side,
    )

    joint = method.joint_channels and C > 1
    phi = _get_numba_dd_pll()(symbols_cpu, joint, *loop_args)  # (1 or C, N)
    phi = _repair_slips(
        phi,
        method.cycle_slip,
        int(constellation.rotational_symmetry),
        1,
        method._slip_history,
    )
    if joint:
        phi = np.broadcast_to(phi, (C, N)).copy()
    phi_full = xp.asarray(phi)

    _log_phase_summary(
        phi_full,
        "CPR (DD-PLL, %s)",
        (f"PI {'2nd' if beta > 0.0 else '1st'}-order, mu={mu}, beta={beta}",),
        "[C=%s]",
        (C,),
    )
    return _Phase(phase=phi_full)
