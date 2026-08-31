"""
Carrier phase recovery utilities.

This package provides routines for carrier phase recovery (CPR), including
streaming decision-directed PLL, block-based Viterbi-Viterbi, Blind Phase
Search, MAP Tikhonov-RTS, and pilot-aided methods, along with cycle-slip
correction and phase ambiguity resolution.

The public API is unchanged from when this was a single module:
``from commkit.recovery import recover_carrier_phase_bps, ...`` continues to
work.
"""

from __future__ import annotations

# Re-exported so ``patch("commkit.recovery.logger...")`` and similar
# attribute access on the package namespace keep working.
from ..logger import logger
from .bps import recover_carrier_phase_bps
from .corrections import (
    correct_carrier_phase,
    correct_cycle_slips,
    correct_phase_rotation,
    resolve_channel_permutation,
    resolve_phase_ambiguity,
    smooth_phase_wiener,
)
from .pilots import (
    recover_carrier_phase_pilot_symbols,
    recover_carrier_phase_pilot_tone,
    recover_carrier_phase_pilot_tones,
)
from .pll import recover_carrier_phase_pll
from .tikhonov import recover_carrier_phase_tikhonov
from .viterbi_viterbi import recover_carrier_phase_viterbi_viterbi

__all__ = [
    "correct_carrier_phase",
    "correct_cycle_slips",
    "correct_phase_rotation",
    "recover_carrier_phase_bps",
    "recover_carrier_phase_pilot_tone",
    "recover_carrier_phase_pilot_tones",
    "recover_carrier_phase_pilot_symbols",
    "recover_carrier_phase_pll",
    "recover_carrier_phase_tikhonov",
    "recover_carrier_phase_viterbi_viterbi",
    "resolve_channel_permutation",
    "resolve_phase_ambiguity",
    "smooth_phase_wiener",
]
