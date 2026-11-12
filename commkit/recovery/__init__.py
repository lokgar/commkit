"""Carrier-phase recovery.

:func:`estimate_carrier_phase` and :func:`correct_carrier_phase`, with one
method object per algorithm: decision-directed :class:`PLL`, block-based
:class:`ViterbiViterbi`, blind phase search :class:`BPS`, MAP
:class:`Tikhonov`, pilot symbols :class:`PilotInterpolation`, pilot tones
:class:`PilotTone` / :class:`PilotTones` and the static :class:`ConstantPhase`
rotation.  The block methods, the PLL and ``PilotInterpolation`` repair cycle
slips with a nested :class:`CycleSlip`.

Also :func:`correct_cycle_slips` and :func:`smooth_phase` on phase
trajectories, and the ambiguity resolution against the reference,
:func:`resolve_phase_ambiguity` and :func:`resolve_channel_permutation`.
"""

from __future__ import annotations

# Re-exported so ``patch("commkit.recovery.logger...")`` and similar
# attribute access on the package namespace keep working.
from ..logger import logger
from .bps import BPS
from .carrier_phase import (
    BlockPhaseEstimate,
    CarrierPhaseEstimate,
    PilotInterpolationEstimate,
    PilotToneEstimate,
    PilotTonesEstimate,
    correct_carrier_phase,
    estimate_carrier_phase,
)
from .corrections import (
    ConstantPhase,
    CycleSlip,
    correct_cycle_slips,
    resolve_channel_permutation,
    resolve_phase_ambiguity,
    smooth_phase,
)
from .pilots import PilotInterpolation, PilotTone, PilotTones
from .pll import PLL
from .tikhonov import Tikhonov
from .viterbi_viterbi import ViterbiViterbi

__all__ = [
    "BPS",
    "BlockPhaseEstimate",
    "CarrierPhaseEstimate",
    "ConstantPhase",
    "CycleSlip",
    "PilotTonesEstimate",
    "PLL",
    "PilotInterpolation",
    "PilotInterpolationEstimate",
    "PilotTone",
    "PilotTones",
    "Tikhonov",
    "PilotToneEstimate",
    "ViterbiViterbi",
    "correct_carrier_phase",
    "correct_cycle_slips",
    "estimate_carrier_phase",
    "resolve_channel_permutation",
    "resolve_phase_ambiguity",
    "smooth_phase",
]
