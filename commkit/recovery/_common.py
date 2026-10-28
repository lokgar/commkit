"""Shared helpers of the recovery package (private).

PLL loop gains and block-phase estimation, used by the method modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..backend import ArrayType, to_device


def _pll_gains(bandwidth: float) -> tuple[np.float32, np.float32]:
    """Convert normalized loop bandwidth to PI gains (mu, beta).

    Uses the usual second-order loop approximation for a critically damped
    (ζ = 1) PI loop:  μ ≈ 4·B_L,  β ≈ 4·B_L².  (With ``ωₙT = √β = 2B`` and
    ``ζ = μ/(2√β) = 1``.)

    Parameters
    ----------
    bandwidth : float
        Normalized one-sided loop bandwidth as a fraction of the symbol rate,
        e.g. ``1e-3`` for a narrow loop.

    Returns
    -------
    mu, beta : float32
    """
    mu = np.float32(4.0 * bandwidth)
    beta = np.float32(4.0 * bandwidth**2)
    return mu, beta


def _resolve_pll_gains(
    bandwidth: float, mu: float | None, beta: float | None
) -> tuple[Any, Any]:
    """Resolve decision-directed PLL PI gains from a raw/bandwidth parameterization.

    Shared by the inline equalizer PLL (``lms``/``rls`` with ``cpr=PLL(...)``)
    and the standalone PLL, so the bandwidth->gain mapping is defined in
    exactly one place.

    Precedence:

    * ``mu`` given: raw PI gains; ``beta`` defaults to ``0.0`` (a first-order loop).
    * ``mu`` is ``None``: critically damped (ζ = 1) gains ``μ = 4B, β = 4B²``
      from ``bandwidth`` via ``_pll_gains``.

    ``beta`` without ``mu`` is ambiguous and raises ``ValueError``.

    Returns
    -------
    mu, beta : float, or float32 from the bandwidth
    """
    if mu is not None:
        return float(mu), float(beta if beta is not None else 0.0)
    if beta is not None:  # beta without mu is ambiguous
        raise ValueError("beta requires mu to be set (or use the bandwidth shortcut).")
    return _pll_gains(bandwidth)


@dataclass(frozen=True)
class _Context:
    """What an estimator may need beyond the samples, resolved by the verb."""

    constellation: Any  # Constellation | None
    sampling_rate: float | None
    reference: ArrayType | None  # known symbols (DataAided)

    def need_constellation(self, method: object) -> Any:
        if self.constellation is None:
            raise ValueError(
                f"estimate_carrier_phase(): {type(method).__name__} needs a "
                "constellation (pass constellation= or a Signal that has one)."
            )
        return self.constellation

    def need_sampling_rate(self, method: object) -> float:
        if self.sampling_rate is None:
            raise ValueError(
                f"estimate_carrier_phase(): {type(method).__name__} requires "
                "sampling_rate for array input."
            )
        return self.sampling_rate


@dataclass(frozen=True)
class _Phase:
    """An estimator's (C, N) trajectory and diagnostics, before ``restore_1d``.

    Field meanings are those of ``CarrierPhaseEstimate``; per-channel fields
    keep their leading channel axis here.
    """

    phase: ArrayType
    block_centers: np.ndarray | None = None
    block_phase: ArrayType | None = None
    pilot_indices: np.ndarray | None = None
    pilot_phase: ArrayType | None = None
    tone_frequencies: np.ndarray | None = None
    tone_snr_db: np.ndarray | None = None
    differential_phase: ArrayType | None = None
    reference_tone: int | None = None
    used_tones: tuple[int, ...] | None = None


def _check_blocks(N: int, block_size: int) -> int:
    """Number of whole blocks; raises when there is none."""
    N_blocks = N // block_size
    if N_blocks == 0:
        raise ValueError(
            f"Signal length {N} is shorter than block_size={block_size}. "
            "Reduce block_size or use a longer symbol sequence."
        )
    return N_blocks


def _interp_block_phases(
    phi_blocks: ArrayType, N: int, block_size: int, xp: Any
) -> ArrayType:
    """Interpolate block phases to every symbol.

    Linear between the block centers ``b·block_size + block_size/2`` and held
    at the end values outside them, as ``numpy.interp``.

    Parameters
    ----------
    phi_blocks : ndarray
        Phase per block, float64, ``(R, N_blocks)``.
    N : int
        Number of symbols.
    block_size : int
        Symbols per block.
    xp : module
        Array module of ``phi_blocks``.

    Returns
    -------
    ndarray
        Phase per symbol, float64, ``(R, N)``.
    """
    n_blocks = phi_blocks.shape[1]
    centers = xp.arange(n_blocks, dtype=xp.float64) * block_size + block_size / 2
    positions = xp.arange(N, dtype=xp.float64)
    # interp is 1-D only; per row it is one pass, 1.6x faster on the CPU than
    # a gather with weights and as fast on the GPU.
    return xp.stack([xp.interp(positions, centers, row) for row in phi_blocks])


def _mth_power_geometry(constellation: Any) -> tuple[int, bool, float]:
    """Exponent, unit-circle projection and bias of the M-th power estimator.

    ``M`` is the rotational symmetry.  Points that are not constant-modulus
    (QAM, multi-level PAM) are projected onto the unit circle before the
    power, so outer rings do not dominate.  The bias is the angle of the
    pmf-weighted mean of ``(c/|c|)^M`` divided by ``M``, reduced to
    ``[0, 2π/M)``: the estimator returns ``φ + bias`` for a noiseless phase
    ``φ``.  It is ``π/4`` for every square QAM and 0 for PSK.
    """
    M = int(constellation.rotational_symmetry)
    pts = np.asarray(constellation.points, dtype=np.complex128)
    mag = np.abs(pts)
    project = bool(np.ptp(mag) > 1e-9 * float(np.max(mag)))
    pmf = constellation.pmf
    weights = np.full(pts.size, 1.0 / pts.size) if pmf is None else np.asarray(pmf)
    z = complex(np.sum(weights * (pts / mag) ** M))
    quantum = 2.0 * np.pi / M
    bias = (float(np.angle(z)) / M) % quantum
    if quantum - bias < 1e-9:  # -0 rounds up to a full quantum
        bias = 0.0
    return M, project, bias


def _vv_block_phase(
    symbols2d: ArrayType,
    xp: Any,
    M: int,
    project: bool,
    bias: float,
    block_size: int,
    joint_channels: bool,
) -> ArrayType:
    """Viterbi-Viterbi (M-th power) block phases.

    Splits the symbols into blocks, projects them onto the unit circle
    (``project``, see ``_mth_power_geometry``) or else scales each channel
    to unit power, sums the M-th power per block, unwraps the block phases
    M-fold and removes the constellation ``bias``.  For MIMO without
    ``joint_channels`` every channel's M-fold branch is aligned to channel 0.

    The shared core of ``ViterbiViterbi`` and ``Tikhonov`` (which smooths
    before the cycle-slip repair and the interpolation): both start from
    these raw block phases.

    Parameters
    ----------
    symbols2d : ndarray
        Symbols at one sample per symbol, ``(C, N)``.
    xp : module
        Array module of ``symbols2d``.
    M : int
        Exponent, the constellation's rotational symmetry.
    project : bool
        Project each symbol onto the unit circle before the power.
    bias : float
        Constellation bias subtracted from the block phase.
    block_size : int
        Symbols per block; the caller has checked ``N // block_size > 0``.
    joint_channels : bool
        For ``C > 1``, sum the block phasors over the channels before the
        angle and the unwrap, and copy the one trajectory to every row.

    Returns
    -------
    ndarray
        Raw block phases, float64, ``(C, N_blocks)``, on the device of
        ``symbols2d``; equal rows with ``joint_channels``.
    """
    C, N = symbols2d.shape
    N_trunc = (N // block_size) * block_size
    N_blocks = N_trunc // block_size

    # Reshape for block processing: (C, N_blocks, block_size), in the
    # symbols' precision.  complex128 moved no block phase by more than 6e-6
    # rad and flipped no unwrap decision (QPSK to 64-QAM, 2.6e6 symbols): a
    # block that close to the ±π/M boundary is decided by its noise, whose
    # phase spread is orders of magnitude larger, in either precision.
    blocks_c = symbols2d[:, :N_trunc].reshape(C, N_blocks, block_size)

    # Project onto the unit circle before the M-th power (normalized VV).
    # This removes outer-ring amplitude dominance and makes the bias
    # correction exact (by the rotational symmetry of the constellation).
    if project:
        mag = xp.abs(blocks_c)
        blocks_c = blocks_c / xp.maximum(mag, 1e-15 * xp.max(mag))
    else:
        # Unit average power per channel, as BPS and the PLL: joint channels
        # then weigh equally instead of by their amplitude^M.
        power = xp.mean(xp.abs(blocks_c) ** 2, axis=(-2, -1), keepdims=True)
        blocks_c = blocks_c / xp.sqrt(xp.maximum(power, 1e-30))

    S_b = xp.sum(blocks_c**M, axis=-1)  # (C, N_blocks)

    if joint_channels and C > 1:
        # Sum M-th-power phasors across channels -> single block-phase trajectory
        S_b_joint = xp.sum(S_b, axis=0)  # (N_blocks,)
        phi_raw_joint = xp.angle(S_b_joint) / M
        phi_u_joint = xp.unwrap((phi_raw_joint * M).astype(xp.float64)) / M
        if bias:
            phi_u_joint = phi_u_joint - bias
        phi_u = xp.broadcast_to(phi_u_joint, (C, N_blocks)).copy()
    else:
        # Raw block phase in [-π/M, π/M)
        phi_raw = xp.angle(S_b) / M  # (C, N_blocks)

        # M-fold unwrap: scale into 2π domain, unwrap, re-scale back.
        # Cast to float64 before unwrap - cp.unwrap preserves input dtype so float32
        # would lose precision during the discontinuity test (diff vs 2π threshold).
        phi_u = (
            xp.unwrap((phi_raw * M).astype(xp.float64), axis=-1) / M
        )  # (C, N_blocks)

        if bias:
            phi_u = phi_u - bias

        # MIMO M-fold alignment: align every channel to channel 0's branch.
        # Skipped in joint mode (all channels share the same trajectory).
        if C > 1:
            # All per-channel means on device, one batched D2H, vectorized shift
            # (instead of one float() sync + one rounding per channel).
            diffs_np = to_device(xp.mean(phi_u[1:] - phi_u[0:1], axis=-1), "cpu")
            k_np = np.round(diffs_np * M / (2 * np.pi))
            phi_u[1:] = phi_u[1:] - xp.asarray(k_np)[:, None] * (2 * np.pi / M)

    return phi_u
