"""MAP Tikhonov carrier phase recovery with RTS/SSKF smoothers."""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..backend import ArrayType, dispatch, to_device
from ._common import (
    _check_blocks,
    _Context,
    _mth_power_geometry,
    _Phase,
    _vv_block_phase,
)
from .corrections import CycleSlip, _log_phase_summary, _repair_slips
from .viterbi_viterbi import _warn_small_qam_block

_NUMBA_RTS: dict = {}


def _get_numba_rts_smoother():
    """JIT-compile and cache the Numba RTS-smoother kernel.

    Returns
    -------
    callable
        Numba-compiled ``_rts_loop``.
    """
    if "rts" not in _NUMBA_RTS:
        import numba

        @numba.njit(cache=True, fastmath=True, nogil=True)
        def _rts_loop(phi_obs, sigma_p2, sigma_v2):
            """Rauch-Tung-Striebel smoother - Numba inner kernel.

            Parameters
            ----------
            phi_obs : (B,) float64
            sigma_p2 : float64
            sigma_v2 : float64

            Returns
            -------
            (B,) float64
            """
            B = len(phi_obs)
            x_filt = np.empty(B, dtype=np.float64)
            P_filt = np.empty(B, dtype=np.float64)
            x_pred = np.empty(B, dtype=np.float64)
            P_pred = np.empty(B, dtype=np.float64)

            x_filt[0] = phi_obs[0]
            P_filt[0] = sigma_v2

            for k in range(1, B):
                x_pred[k] = x_filt[k - 1]
                P_pred[k] = P_filt[k - 1] + sigma_p2
                K = P_pred[k] / (P_pred[k] + sigma_v2)
                x_filt[k] = x_pred[k] + K * (phi_obs[k] - x_pred[k])
                P_filt[k] = (1.0 - K) * P_pred[k]

            x_smooth = x_filt.copy()
            for k in range(B - 2, -1, -1):
                G = P_filt[k] / P_pred[k + 1]
                x_smooth[k] = x_filt[k] + G * (x_smooth[k + 1] - x_pred[k + 1])

            return x_smooth

        _NUMBA_RTS["rts"] = _rts_loop

    return _NUMBA_RTS["rts"]


def _rts_smoother_1d(
    phi_obs: np.ndarray,
    sigma_p2: float,
    sigma_v2: float,
) -> np.ndarray:
    """Rauch-Tung-Striebel (RTS) Kalman smoother for a 1-D random-walk state.

    Uses the Numba-compiled kernel (``_get_numba_rts_smoother``) when
    available; falls back to a pure-Python loop otherwise.  Always runs on
    CPU - call with a NumPy array; the caller is responsible for
    ``to_device`` conversion.

    State model  : x[k+1] = x[k] + w[k],   w ~ N(0, sigma_p2)
    Observation  : y[k]   = x[k] + v[k],   v ~ N(0, sigma_v2)

    Parameters
    ----------
    phi_obs : (B,) float64
        Noisy block-phase observations in radians (e.g. from VV).
    sigma_p2 : float
        Process noise variance per block (Wiener phase noise increment).
    sigma_v2 : float
        Observation noise variance (VV estimator variance per block).

    Returns
    -------
    (B,) float64
        MAP-smoothed phase trajectory.
    """
    return _get_numba_rts_smoother()(phi_obs, float(sigma_p2), float(sigma_v2))


def _sskf_smoother_1d(
    phi_obs: ArrayType,
    sigma_p2: float,
    sigma_v2: float,
    sp,
    xp,
) -> ArrayType:
    """Steady-state Kalman smoother via zero-phase IIR filter (filtfilt).

    Approximates the RTS smoother by replacing the sequential Kalman
    recurrence with a 1st-order IIR filter whose gain is solved analytically
    from the discrete algebraic Riccati equation.  The bidirectional
    ``filtfilt`` call makes it equivalent to the RTS smoother in steady state.

    Backend-aware: uses ``sp.signal.filtfilt`` where ``sp`` is
    ``scipy`` (CPU) or ``cupyx.scipy`` (GPU) as returned by
    ``dispatch``.

    The approximation is excellent when ``B >> 1/K_∞``
    (typically ``B > 20``).  For ``B < 7`` (``filtfilt`` minimum), falls
    back to the exact ``_rts_smoother_1d`` on CPU.

    Parameters
    ----------
    phi_obs : (B,) float64, on the target device
        Noisy block-phase observations in radians.
    sigma_p2, sigma_v2 : float
        Process and observation noise variances per block.
    sp : module
        ``scipy`` or ``cupyx.scipy``, from ``dispatch``.
    xp : module
        ``numpy`` or ``cupy``, from ``dispatch``.

    Returns
    -------
    (B,) float64, same device as ``phi_obs``.
    """
    # filtfilt requires at least padlen * 2 + 1 samples; padlen = 3 * max(len(b), len(a)) = 6
    if len(phi_obs) < 7:
        phi_np = to_device(phi_obs, "cpu")
        return xp.asarray(_rts_smoother_1d(phi_np, sigma_p2, sigma_v2))

    # Steady-state prediction error covariance from discrete Riccati equation:
    #   p² - σ_p²·p - σ_p²·σ_v² = 0  ->  p = (σ_p² + √(σ_p⁴ + 4σ_p²σ_v²)) / 2
    p_ss = (sigma_p2 + float(np.sqrt(sigma_p2**2 + 4.0 * sigma_p2 * sigma_v2))) / 2.0
    K_ss = p_ss / (p_ss + sigma_v2)

    # Forward IIR:  y[k] = (1-K)·y[k-1] + K·x[k]
    #   H(z) = K / (1 - (1-K)·z⁻¹)
    # filtfilt applies forward + backward  ->  zero-phase, ≡ RTS smoother at
    # steady state.
    b = [K_ss]
    a = [1.0, -(1.0 - K_ss)]
    return sp.signal.filtfilt(b, a, phi_obs)


@dataclass(frozen=True)
class Tikhonov:
    r"""
    MAP phase estimation with a Wiener (Tikhonov) phase-noise prior.

    Viterbi-Viterbi block phases are smoothed by a Kalman smoother matched
    to the laser phase noise, with process variance
    ``σ_p² = 2π·Δν·T_s·N_b`` and observation variance
    ``σ_v² ≈ 1/(M²·SNR·N_b)``, then interpolated to per-symbol resolution.

    Parameters
    ----------
    linewidth_symbol_periods : float
        Combined linewidth-symbol-time product ``Δν·T_s``; typical values
        ``1e-5`` (narrow laser, 32 GBd) to ``5e-4``.
    snr_db : float
        Operating SNR per symbol in dB; sets ``σ_v²``.
    block_size : int, default 32
        Symbols per Viterbi-Viterbi block (same trade-off as
        :class:`ViterbiViterbi`).
    smoother : {"rts", "steady_state"}, default "rts"
        ``"rts"``: the exact Rauch-Tung-Striebel smoother (Numba, CPU; GPU
        input makes one host round trip of the block phases).
        ``"steady_state"``: the steady-state Kalman gain as a zero-phase IIR
        (``filtfilt``) on the input's device; accurate for 20 or more blocks,
        and the exact smoother below 7 blocks.
    joint_channels : bool, default False
        MIMO: one joint trajectory from the summed block phasors (shared LO).
    cycle_slip : CycleSlip, optional
        Repair cycle slips after smoothing, before interpolation.

    Notes
    -----
    A residual ``2π/M`` ambiguity remains.
    """

    linewidth_symbol_periods: float
    snr_db: float
    block_size: int = 32
    smoother: Literal["rts", "steady_state"] = "rts"
    joint_channels: bool = False
    cycle_slip: CycleSlip | None = None

    def __post_init__(self) -> None:
        if self.smoother not in ("rts", "steady_state"):
            raise ValueError(
                f"smoother must be 'rts' or 'steady_state', got {self.smoother!r}."
            )
        if not self.linewidth_symbol_periods > 0:
            raise ValueError(
                "linewidth_symbol_periods must be > 0, got "
                f"{self.linewidth_symbol_periods}."
            )
        if self.block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {self.block_size}.")


def _tikhonov(symbols: ArrayType, method: Tikhonov, ctx: _Context) -> _Phase:
    """Tikhonov-smoothed Viterbi-Viterbi phase of ``(C, N)`` symbols."""
    constellation = ctx.need_constellation(method)
    symbols, xp, sp = dispatch(symbols)
    C, N = symbols.shape
    block_size = method.block_size
    M, project, bias = _mth_power_geometry(constellation)
    N_blocks = _check_blocks(N, block_size)
    # Same data-residual constraint as Viterbi-Viterbi: block phase variance
    # can exceed π/M before smoothing, causing slips the smoother cannot fix.
    _warn_small_qam_block("Tikhonov", block_size, project, constellation.order)

    snr_lin = 10.0 ** (method.snr_db / 10.0)
    sigma_p2 = float(2.0 * np.pi * method.linewidth_symbol_periods * block_size)
    sigma_v2 = float(1.0 / (M**2 * snr_lin * block_size))

    joint = method.joint_channels and C > 1
    phi_u, block_centers, all_positions = _vv_block_phase(
        symbols, xp, M, project, bias, block_size, method.joint_channels
    )
    if joint:  # rows are copies of the joint trajectory
        phi_u = phi_u[:1]

    if method.smoother == "rts":
        phi_u_np = to_device(phi_u, "cpu")  # (R, N_blocks) float64
        phi_smooth = xp.asarray(
            np.stack([_rts_smoother_1d(row, sigma_p2, sigma_v2) for row in phi_u_np])
        )
    else:
        phi_smooth = xp.stack(
            [_sskf_smoother_1d(row, sigma_p2, sigma_v2, sp, xp) for row in phi_u]
        )
    phi_smooth = _repair_slips(phi_smooth, xp, method.cycle_slip, M)
    phi_full = xp.stack(
        [xp.interp(all_positions, block_centers, row) for row in phi_smooth]
    )
    if joint:
        phi_full = xp.broadcast_to(phi_full, (C, N)).copy()
        phi_smooth = xp.broadcast_to(phi_smooth, (C, N_blocks)).copy()

    _log_phase_summary(
        phi_full,
        "CPR (Tikhonov-%s, M=%s, %s)",
        (method.smoother, M, "joint" if joint else "independent"),
        "[%s blocks x %s, σ_p²=%.2e, σ_v²=%.2e, C=%s, cycle_slip=%s]",
        (N_blocks, block_size, sigma_p2, sigma_v2, C, method.cycle_slip is not None),
    )
    return _Phase(
        phase=phi_full,
        block_centers=np.arange(N_blocks, dtype=np.float64) * block_size
        + block_size / 2,
        block_phase=phi_smooth,
    )
