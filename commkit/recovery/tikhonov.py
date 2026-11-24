"""MAP carrier-phase recovery with a Tikhonov prior: :class:`Tikhonov`.

Viterbi-Viterbi block phases smoothed by a Kalman smoother matched to the
laser phase noise: the exact Rauch-Tung-Striebel (RTS) smoother or its
steady-state form.
"""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ..backend import ArrayType, dispatch, to_device
from ._common import (
    _Context,
    _interp_block_phases,
    _mth_power_geometry,
    _num_blocks,
    _numba,
    _vv_block_phase,
)
from .corrections import CycleSlip, _log_phase_summary, _repair_slips
from .estimates import BlockPhaseEstimate
from .viterbi_viterbi import _warn_small_qam_block

__all__ = ["Tikhonov"]


def _sskf_smoother(
    phi_obs: ArrayType,
    sigma_p2: float,
    sigma_v2: float,
    sp: Any,
    xp: Any,
) -> ArrayType:
    """Approximate the RTS smoother by a steady-state Kalman gain and ``filtfilt``.

    The sequential Kalman recursion is replaced by a first-order IIR filter
    whose gain solves the discrete algebraic Riccati equation; running it
    forward and backward (``filtfilt``) equals the RTS smoother in steady
    state.  It runs on the input's device (``sp.signal.filtfilt``).

    The approximation is close when ``B >> 1/K_∞`` (typically ``B > 20``).
    Below 7 blocks (the ``filtfilt`` minimum) the exact RTS kernel runs on
    the CPU instead.

    Parameters
    ----------
    phi_obs : ndarray
        Noisy block phases in radians, float64, ``(R, B)``.
    sigma_p2, sigma_v2 : float
        Process and observation noise variances per block.
    sp : module
        ``scipy`` or ``cupyx.scipy``, from ``dispatch``.
    xp : module
        ``numpy`` or ``cupy``, from ``dispatch``.

    Returns
    -------
    ndarray
        Smoothed phases, float64, ``(R, B)``, on the device of ``phi_obs``.
    """
    # filtfilt requires at least padlen * 2 + 1 samples; padlen = 3 * max(len(b), len(a)) = 6
    if phi_obs.shape[-1] < 7:
        phi_np = to_device(phi_obs, "cpu")
        rts = _numba().rts_smoother
        return xp.asarray(
            np.stack([rts(row, float(sigma_p2), float(sigma_v2)) for row in phi_np])
        )

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
    return sp.signal.filtfilt(b, a, phi_obs, axis=-1)


@dataclass(frozen=True)
class Tikhonov:
    r"""MAP phase estimation with a Wiener (Tikhonov) phase-noise prior.

    Viterbi-Viterbi block phases are smoothed by a Kalman smoother matched
    to the laser phase noise, with process variance
    ``σ_p² = 2π·Δν·T_s·N_b`` and the observation variance ``σ_v²`` of a
    Viterbi-Viterbi block phase, computed from the constellation and the SNR:
    the noise through the M-th power (``1/(2·SNR·N_b)`` for PSK at high SNR)
    plus the data self-noise of QAM.  The smoothed block phases are then
    interpolated to per-symbol resolution.

    Parameters
    ----------
    linewidth : float
        Combined linewidth ``Δν`` of the transmitter laser and the LO in Hz,
        as in :func:`commkit.impairments.apply_phase_noise`.  The symbol
        period ``T_s`` comes from the sampling rate of the one-sample-per-
        symbol input (the Signal's, or ``sampling_rate=`` of the verb);
        ``Δν·T_s`` is typically ``1e-5`` (narrow laser, 32 GBd) to ``5e-4``.
    snr_db : float
        Operating SNR per symbol in dB; sets ``σ_v²``.
    block_size : int, default 32
        Symbols per Viterbi-Viterbi block (same trade-off as
        :class:`ViterbiViterbi`).
    smoother : {"rts", "steady_state"}, default "rts"
        ``"rts"``: the exact Rauch-Tung-Striebel smoother (Numba, CPU; GPU
        input makes one host round trip of the block phases).
        ``"steady_state"``: the steady-state Kalman gain as a zero-phase IIR
        (``filtfilt``) on the input's device, accurate from about 20 blocks;
        below 7 blocks the exact smoother runs instead.
    joint_channels : bool, default False
        MIMO: one joint trajectory from the summed block phasors (shared LO).
    cycle_slip : CycleSlip, optional
        Repair cycle slips after smoothing, before interpolation.

    Notes
    -----
    A residual ``2π/M`` ambiguity remains.

    ``σ_p²`` is the long-run increment of the block means of a Wiener phase,
    ``σ²·N_b`` per block for ``σ² = 2π·Δν·T_s``.  Adjacent block means
    differ by less, ``σ²(2N_b² + 1)/(3N_b)``, but those differences are
    positively correlated (by about 1/4), and the smoother's gain follows
    the long-run behaviour: on QPSK with a Wiener phase (``Δν·T_s`` from
    ``1e-5`` to ``1e-4``, 8 to 20 dB, blocks of 8 to 64) the lag-one
    variance gave a per-symbol phase error 1 to 1.5 % larger.
    """

    linewidth: float
    snr_db: float
    block_size: int = 32
    smoother: Literal["rts", "steady_state"] = "rts"
    joint_channels: bool = False
    cycle_slip: CycleSlip | None = None

    def __post_init__(self) -> None:
        if self.smoother not in ("rts", "steady_state"):
            raise ValueError(
                f"Tikhonov: smoother must be 'rts' or 'steady_state', got {self.smoother!r}."
            )
        if not self.linewidth > 0:
            raise ValueError(
                f"Tikhonov: linewidth must be > 0 Hz, got {self.linewidth}."
            )
        if self.block_size < 1:
            raise ValueError(
                f"Tikhonov: block_size must be >= 1, got {self.block_size}."
            )


def _vv_observation_variance(
    constellation: Any, M: int, project: bool, snr_lin: float, block_size: int
) -> float:
    """Variance of one Viterbi-Viterbi block phase, in rad².

    The block phase is ``arg(Σ z)/M`` with ``z = r^M`` (``(r/|r|)^M`` when
    ``project``) and ``r = s + n``, ``n ~ CN(0, 1/SNR)``.  To first order its
    error is ``Im(Σ u)/(M N |μ|)`` with ``μ = E[z]`` and ``u = z e^{-j∠μ}``,
    so ``σ_v² = (E|z|² - Re E[u²]) / (2 N M² |μ|²)``: the AWGN through the
    power (``1/(2 SNR N)`` for PSK at high SNR) plus the data self-noise of
    non-constant-modulus constellations.  The expectations run over the
    points (pmf-weighted) and the noise, by 2-D Gauss-Hermite quadrature.
    """
    pts = np.asarray(constellation.points, dtype=np.complex128)
    pmf = constellation.pmf
    w_pts = np.full(pts.size, 1.0 / pts.size) if pmf is None else np.asarray(pmf)
    pts = pts / np.sqrt(float(np.dot(w_pts, np.abs(pts) ** 2)))
    t, w = np.polynomial.hermite.hermgauss(24)
    n = np.sqrt(1.0 / snr_lin) * (t[:, None] + 1j * t[None, :]).ravel()
    w_n = (w[:, None] * w[None, :]).ravel() / np.pi
    r = pts[:, None] + n[None, :]  # (points, nodes)
    if project:
        r = r / np.maximum(np.abs(r), 1e-300)
    z = r**M
    weight = w_pts[:, None] * w_n[None, :]
    mu = complex(np.sum(weight * z))
    second = complex(np.sum(weight * z * z)) * np.exp(-2j * np.angle(mu))
    power = float(np.sum(weight * np.abs(z) ** 2))
    return (power - second.real) / (2.0 * block_size * M**2 * abs(mu) ** 2)


def _tikhonov(
    symbols: ArrayType, method: Tikhonov, ctx: _Context
) -> BlockPhaseEstimate:
    """Tikhonov-smoothed Viterbi-Viterbi phase of ``(C, N)`` symbols."""
    constellation = ctx.need_constellation(method)
    symbols, xp, sp = dispatch(symbols)
    C, N = symbols.shape
    block_size = method.block_size
    M, project, bias = _mth_power_geometry(constellation)
    N_blocks = _num_blocks(N, block_size)
    # Same data-residual constraint as Viterbi-Viterbi: block phase variance
    # can exceed π/M before smoothing, causing slips the smoother cannot fix.
    _warn_small_qam_block("Tikhonov", block_size, project, constellation.order)

    snr_lin = 10.0 ** (method.snr_db / 10.0)
    linewidth_symbol_periods = method.linewidth / ctx.need_sampling_rate(method)
    joint = method.joint_channels and C > 1
    # The long-run increment of the block means, not the variance of adjacent
    # differences (2/3 of it, correlated); see the class Notes.
    sigma_p2 = float(2.0 * np.pi * linewidth_symbol_periods * block_size)
    eff_block = C * block_size if joint else block_size
    sigma_v2 = _vv_observation_variance(constellation, M, project, snr_lin, eff_block)

    phi_u = _vv_block_phase(
        symbols, xp, M, project, bias, block_size, method.joint_channels
    )
    if joint:  # rows are copies of the joint trajectory
        phi_u = phi_u[:1]

    if method.smoother == "rts":
        phi_u_np = to_device(phi_u, "cpu")  # (R, N_blocks) float64
        rts = _numba().rts_smoother
        phi_smooth = xp.asarray(
            np.stack([rts(row, float(sigma_p2), float(sigma_v2)) for row in phi_u_np])
        )
    else:
        phi_smooth = _sskf_smoother(phi_u, sigma_p2, sigma_v2, sp, xp)
    phi_smooth = _repair_slips(phi_smooth, method.cycle_slip, M, block_size)
    phi_full = _interp_block_phases(phi_smooth, N, block_size, xp)
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
    return BlockPhaseEstimate(
        value=phi_full,
        block_centers=np.arange(N_blocks, dtype=np.float64) * block_size
        + block_size / 2,
        block_phase=phi_smooth,
    )
