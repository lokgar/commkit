"""
Carrier phase recovery.

``estimate_carrier_phase`` / ``correct_carrier_phase`` with one method object
per algorithm: decision-directed ``PLL``, block-based ``ViterbiViterbi``,
``BPS`` (blind phase search), MAP ``Tikhonov``, ``PilotAided`` symbols,
``PilotTone`` / ``PilotTones`` and the static ``DataAided`` rotation.  Block
methods repair cycle slips with a nested ``CycleSlip``.  Also: cycle-slip
repair and Wiener smoothing of phase trajectories, and phase/channel
ambiguity resolution against the reference.
"""

from __future__ import annotations

# Re-exported so ``patch("commkit.recovery.logger...")`` and similar
# attribute access on the package namespace keep working.
from ..logger import logger
from .bps import BPS
from .carrier_phase import (
    CarrierPhaseEstimate,
    correct_carrier_phase,
    estimate_carrier_phase,
)
from .corrections import (
    CycleSlip,
    DataAided,
    correct_cycle_slips,
    resolve_channel_permutation,
    resolve_phase_ambiguity,
    smooth_phase_wiener,
)
from .pilots import PilotAided, PilotTone, PilotTones
from .pll import PLL
from .tikhonov import Tikhonov
from .viterbi_viterbi import ViterbiViterbi

__all__ = [
    "BPS",
    "PLL",
    "CarrierPhaseEstimate",
    "CycleSlip",
    "DataAided",
    "PilotAided",
    "PilotTone",
    "PilotTones",
    "Tikhonov",
    "ViterbiViterbi",
    "correct_carrier_phase",
    "correct_cycle_slips",
    "estimate_carrier_phase",
    "resolve_channel_permutation",
    "resolve_phase_ambiguity",
    "smooth_phase_wiener",
]
