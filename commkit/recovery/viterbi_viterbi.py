"""Viterbi-Viterbi (V&V) carrier phase recovery."""

from dataclasses import dataclass

import numpy as np

from ..backend import ArrayType, dispatch
from ..logger import logger
from ._common import (
    _check_blocks,
    _Context,
    _mth_power_geometry,
    _Phase,
    _vv_block_phase,
)
from .corrections import CycleSlip, _log_phase_summary, _repair_slips


@dataclass(frozen=True)
class ViterbiViterbi:
    """
    Viterbi-Viterbi (M-th power) block phase estimation.

    Each block of symbols is raised to the constellation's rotational
    symmetry ``M`` to remove the modulation, ``S_b = Σ s[n]^M``,
    ``φ_b = ∠S_b / M``.  Block phases are M-fold unwrapped, the constellation
    bias is removed, and the result is interpolated linearly to per-symbol
    resolution.  Points that are not constant-modulus (QAM) are projected
    onto the unit circle first.

    Parameters
    ----------
    block_size : int, default 32
        Symbols per block.  Larger blocks reduce variance but slow tracking.
        PSK cancels exactly per symbol (``block_size`` can be 1); for QAM
        the minimum reliable size grows as about ``4·ceil(√order)``.
    joint_channels : bool, default False
        MIMO: sum the block phasors ``S_b`` across channels before the angle
        and give every channel the one trajectory (shared LO).
    cycle_slip : CycleSlip, optional
        Repair cycle slips in the block phases before interpolation.

    Notes
    -----
    A global ``2π/M`` ambiguity remains.  For strong phase noise prefer
    :class:`BPS`, which needs no M-fold unwrap.
    """

    block_size: int = 32
    joint_channels: bool = False
    cycle_slip: CycleSlip | None = None

    def __post_init__(self) -> None:
        if self.block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {self.block_size}.")


def _warn_small_qam_block(
    name: str, block_size: int, project: bool, order: int
) -> None:
    """Warn when QAM block averaging is too short for the M-fold unwrap.

    For QAM with order > 4 the M-th power of individual symbols does not
    cancel the data modulation; block-phase variance above the ``π/M``
    unwrap threshold produces persistent ``2π/M`` slips.
    """
    if not project or order <= 4:
        return
    min_bs = max(8, 4 * int(np.ceil(order**0.5)))
    if block_size < min_bs:
        logger.warning(
            "CPR (%s): block_size=%s is too small for this %s-point "
            "constellation. Individual symbols' M-th powers do not cancel the "
            "data modulation; insufficient averaging causes block-phase "
            "variance that exceeds the π/M unwrap threshold, producing "
            "persistent 2π/M phase slips. Recommended minimum: block_size ≥ %s.",
            name,
            block_size,
            order,
            min_bs,
        )


def _viterbi_viterbi(
    symbols: ArrayType, method: ViterbiViterbi, ctx: _Context
) -> _Phase:
    """Viterbi-Viterbi phase of ``(C, N)`` symbols."""
    constellation = ctx.need_constellation(method)
    symbols, xp, _ = dispatch(symbols)
    C, N = symbols.shape
    block_size = method.block_size
    M, project, bias = _mth_power_geometry(constellation)
    N_blocks = _check_blocks(N, block_size)
    _warn_small_qam_block("VV", block_size, project, constellation.order)

    joint = method.joint_channels and C > 1
    phi_u, block_centers, all_positions = _vv_block_phase(
        symbols, xp, M, project, bias, block_size, method.joint_channels
    )
    if joint:  # rows are copies of the joint trajectory
        phi_u = phi_u[:1]
    phi_u = _repair_slips(phi_u, xp, method.cycle_slip, M)
    # xp.interp is 1D-only; loop over the rows.
    phi_full = xp.stack([xp.interp(all_positions, block_centers, row) for row in phi_u])
    if joint:
        phi_full = xp.broadcast_to(phi_full, (C, N)).copy()
        phi_u = xp.broadcast_to(phi_u, (C, N_blocks)).copy()

    _log_phase_summary(
        phi_full,
        "CPR (Viterbi-Viterbi, M=%s, %s)",
        (M, "joint" if joint else "independent"),
        "[%s blocks x %s symbols, C=%s, cycle_slip=%s]",
        (N_blocks, block_size, C, method.cycle_slip is not None),
    )
    return _Phase(
        phase=phi_full,
        block_centers=np.arange(N_blocks, dtype=np.float64) * block_size
        + block_size / 2,
        block_phase=phi_u,
    )
