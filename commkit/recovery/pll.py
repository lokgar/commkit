"""Decision-directed PLL carrier phase recovery."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..backend import ArrayType, dispatch, to_device
from ._common import _Context, _Phase, _resolve_pll_gains
from .corrections import CycleSlip, _log_phase_summary, _repair_slips

_NUMBA_PLL: dict = {}


def _get_numba_dd_pll():
    """JIT-compile and cache the Numba DD-PLL sample-wise loop kernel.

    Returns
    -------
    callable
        Numba-compiled ``_dd_pll_loop``.
    """
    if "dd_pll" not in _NUMBA_PLL:
        import numba

        @numba.njit(cache=True, fastmath=True, nogil=True)
        def _dd_pll_loop(
            sym_r,
            sym_i,
            const_r,
            const_i,
            mu,
            beta,
            phi0,
            freq0,
            is_sq_qam,
            levels,
            d_grid,
            lev_min,
            side,
        ):
            """Inner DD-PLL loop compiled to machine code by Numba.

            Parameters
            ----------
            sym_r, sym_i : (N,) float64
                Real and imaginary parts of received symbols.
            const_r, const_i : (M,) float64
                Real and imaginary parts of reference constellation.
            mu : float64
                Proportional (phase) gain - corrects the instantaneous phase error.
            beta : float64
                Integral (frequency) gain - tracks residual frequency drift.
                Set to 0.0 for a 1st-order loop.
            phi0 : float64
                Initial phase state in radians.
            freq0 : float64
                Initial frequency correction state in radians/symbol.
            is_sq_qam : bool
                True when the constellation is a square QAM grid.  Enables the
                O(1) rounding decision path instead of the O(M) linear search.
            levels : (side,) float64
                Sorted unique axis levels for square QAM (ignored when not sq_qam).
            d_grid : float64
                Grid spacing (levels[1] - levels[0]).
            lev_min : float64
                Minimum level value (levels[0]).
            side : int
                Number of points per axis (sqrt of constellation order).

            Returns
            -------
            phase_est : (N,) float64
                Per-symbol phase trajectory φ[n].
            """
            N = len(sym_r)
            M = len(const_r)
            phase_est = np.empty(N, dtype=np.float64)
            phi = phi0
            freq = freq0

            for n in range(N):
                # Rotate received symbol by current phase estimate:
                # y[n] = s[n] · exp(-jφ[n])
                cos_phi = np.cos(phi)
                sin_phi = np.sin(phi)
                yr = sym_r[n] * cos_phi + sym_i[n] * sin_phi
                yi = -sym_r[n] * sin_phi + sym_i[n] * cos_phi

                # Hard decision: argmin_{c ∈ C} |y - c|²
                if is_sq_qam:
                    # O(1) grid rounding for square QAM
                    r_idx = int(round((yr - lev_min) / d_grid))
                    if r_idx < 0:
                        r_idx = 0
                    elif r_idx >= side:
                        r_idx = side - 1
                    d_r = levels[r_idx]
                    i_idx = int(round((yi - lev_min) / d_grid))
                    if i_idx < 0:
                        i_idx = 0
                    elif i_idx >= side:
                        i_idx = side - 1
                    d_i = levels[i_idx]
                else:
                    min_d2 = (yr - const_r[0]) ** 2 + (yi - const_i[0]) ** 2
                    d_r = const_r[0]
                    d_i = const_i[0]
                    for k in range(1, M):
                        d2 = (yr - const_r[k]) ** 2 + (yi - const_i[k]) ** 2
                        if d2 < min_d2:
                            min_d2 = d2
                            d_r = const_r[k]
                            d_i = const_i[k]

                # Cross-product phase error:  e = Im(y · d*) = yi·d_r - yr·d_i
                e = yi * d_r - yr * d_i

                # Record the phase used to derotate symbol n - before the update.
                phase_est[n] = phi

                # 2nd-order loop filter (reduces to 1st order when beta=0):
                #   φ[n+1] = φ[n] + μ·e[n] + ν[n]
                #   ν[n]   = ν[n-1] + β·e[n]
                phi = phi + mu * e + freq
                freq = freq + beta * e

            return phase_est

        _NUMBA_PLL["dd_pll"] = _dd_pll_loop

    return _NUMBA_PLL["dd_pll"]


def _get_numba_dd_pll_joint():
    """JIT-compile and cache the joint-channel DD-PLL PI kernel.

    Averages the cross-product phase error across C channels at each symbol
    before updating the single shared phase/frequency state.  This is the
    MVUE joint estimator for shared-LO systems.

    Returns
    -------
    callable
        Numba-compiled ``_dd_pll_joint_loop``.
    """
    if "dd_pll_joint" not in _NUMBA_PLL:
        import numba

        @numba.njit(cache=True, fastmath=True, nogil=True)
        def _dd_pll_joint_loop(
            sym_r,
            sym_i,
            const_r,
            const_i,
            mu,
            beta,
            phi0,
            freq0,
            is_sq_qam,
            levels,
            d_grid,
            lev_min,
            side,
        ):
            """Joint-channel DD-PLL with PI loop filter.

            Parameters
            ----------
            sym_r, sym_i : (C, N) float64
                Real and imaginary parts of received symbols, all channels.
            const_r, const_i : (M,) float64
                Reference constellation.
            mu, beta, phi0, freq0 : float64
                Loop parameters - same semantics as ``_dd_pll_loop``.
            is_sq_qam : bool
                Enables O(1) rounding decision for square QAM grids.
            levels : (side,) float64
            d_grid, lev_min : float64
            side : int

            Returns
            -------
            phase_est : (N,) float64
                Single shared phase trajectory (broadcast to all channels by caller).
            """
            C = sym_r.shape[0]
            N = sym_r.shape[1]
            M = len(const_r)
            phase_est = np.empty(N, dtype=np.float64)
            phi = phi0
            freq = freq0

            for n in range(N):
                cos_phi = np.cos(phi)
                sin_phi = np.sin(phi)
                e_sum = 0.0
                for c in range(C):
                    yr = sym_r[c, n] * cos_phi + sym_i[c, n] * sin_phi
                    yi = -sym_r[c, n] * sin_phi + sym_i[c, n] * cos_phi
                    if is_sq_qam:
                        r_idx = int(round((yr - lev_min) / d_grid))
                        if r_idx < 0:
                            r_idx = 0
                        elif r_idx >= side:
                            r_idx = side - 1
                        d_r = levels[r_idx]
                        i_idx = int(round((yi - lev_min) / d_grid))
                        if i_idx < 0:
                            i_idx = 0
                        elif i_idx >= side:
                            i_idx = side - 1
                        d_i = levels[i_idx]
                    else:
                        min_d2 = (yr - const_r[0]) ** 2 + (yi - const_i[0]) ** 2
                        d_r = const_r[0]
                        d_i = const_i[0]
                        for k in range(1, M):
                            d2 = (yr - const_r[k]) ** 2 + (yi - const_i[k]) ** 2
                            if d2 < min_d2:
                                min_d2 = d2
                                d_r = const_r[k]
                                d_i = const_i[k]
                    e_sum += yi * d_r - yr * d_i
                # Average error across channels - MVUE for shared LO
                e = e_sum / float(C)
                phase_est[n] = phi
                phi = phi + mu * e + freq
                freq = freq + beta * e

            return phase_est

        _NUMBA_PLL["dd_pll_joint"] = _dd_pll_joint_loop

    return _NUMBA_PLL["dd_pll_joint"]


@dataclass(frozen=True)
class PLL:
    r"""
    Decision-directed phase-locked loop.

    Tracks the carrier phase symbol by symbol from hard decisions:
    derotate by ``φ̂[n]``, decide, take the cross-product error
    ``e[n] = Im(y[n]·d̂*[n])``, then ``φ̂[n+1] = φ̂[n] + μ·e[n] + ν[n]`` and
    ``ν[n+1] = ν[n] + β·e[n]``.  A 1st-order loop (``β = 0``) tracks phase;
    ``β > 0`` also tracks a residual frequency offset.  The same object sets
    the inline PLL of the equalizers.

    Parameters
    ----------
    bandwidth : float, default 1e-3
        Normalised one-sided loop bandwidth in ``(0, 0.5)``, as a fraction of
        the symbol rate.  Gives critically damped gains ``μ = 4B``,
        ``β = 4B²``.  Used when ``mu`` is not given.
    mu : float, optional
        Raw proportional gain; overrides ``bandwidth``.  Typical values
        ``1e-3`` (high-order QAM, high SNR) to ``5e-2`` (QPSK).
    beta : float, optional
        Raw integral gain (``β ≈ μ²/4`` is critically damped); requires
        ``mu``.  Defaults to 0 (1st-order) with ``mu``.
    phase_init : float, default 0.0
        Initial phase in radians, e.g. the last value of a BPS estimate.
    joint_channels : bool, default False
        MIMO: average the phase error across channels at each symbol and
        drive one shared loop (shared LO).
    cycle_slip : CycleSlip, optional
        Repair ``π/2`` slips in the per-symbol trajectory after the loop.

    Notes
    -----
    The loop needs reliable decisions; a cold start converges over about
    ``1/μ`` symbols.  Numba-compiled on the CPU; GPU input makes one host
    round trip.  A global M-fold ambiguity remains.
    """

    bandwidth: float = 1e-3
    mu: float | None = None
    beta: float | None = None
    phase_init: float = 0.0
    joint_channels: bool = False
    cycle_slip: CycleSlip | None = None

    def __post_init__(self) -> None:
        if not (0.0 < self.bandwidth < 0.5):
            raise ValueError(f"bandwidth must be in (0, 0.5), got {self.bandwidth}.")
        _resolve_pll_gains(self.bandwidth, self.mu, self.beta)  # validates beta

    @property
    def gains(self) -> tuple[Any, Any]:
        """The loop gains ``(mu, beta)``."""
        return _resolve_pll_gains(self.bandwidth, self.mu, self.beta)


def _pll(symbols: ArrayType, method: PLL, ctx: _Context) -> _Phase:
    """Decision-directed PLL phase of ``(C, N)`` symbols."""
    from ..math import normalize

    constellation = ctx.need_constellation(method)
    mu, beta = method.gains
    phase_init = method.phase_init

    symbols, xp, _ = dispatch(symbols)
    C, N = symbols.shape

    # Normalise to unit average power so the effective loop gain is mu regardless
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
    grid = constellation._square_grid
    _is_sq_qam = grid is not None
    if grid is not None:
        _lev_min, _d_grid, _side, _ = grid
        _levels = _lev_min + np.arange(_side) * _d_grid
    else:
        _lev_min, _d_grid, _side = 0.0, 1.0, 0
        _levels = np.empty(0, dtype=np.float64)

    # Move to CPU for sequential processing
    symbols_cpu = to_device(symbols, "cpu").astype(np.complex128)
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
    if joint:
        sym_r_all = np.ascontiguousarray(symbols_cpu.real)  # (C, N) float64
        sym_i_all = np.ascontiguousarray(symbols_cpu.imag)
        phi = _get_numba_dd_pll_joint()(sym_r_all, sym_i_all, *loop_args)[None, :]
    else:
        kernel = _get_numba_dd_pll()
        phi = np.stack(
            [
                kernel(
                    symbols_cpu[ch].real.copy(), symbols_cpu[ch].imag.copy(), *loop_args
                )
                for ch in range(C)
            ]
        )
    phi = _repair_slips(phi, np, method.cycle_slip, 4)
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
