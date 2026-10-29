"""Adaptive and block channel equalization.

- Sequential (symbol-by-symbol) equalizers :func:`lms` and :func:`rls`
  (trained, then decision-directed) and :func:`cma` and :func:`rde` (blind),
  as Numba loops on the CPU; :class:`EqualizerPilots` runs the blind ones
  pilot-aided.
- Frequency-domain block equalizers :func:`block_lms`, :func:`block_cma` and
  :func:`block_rde`, on the input's device (CUDA graphs on the GPU).
- Linear block equalization: :func:`estimate_transfer_function` (channel
  estimate), :func:`zf_equalizer` (its ZF / MMSE inverse) and
  :func:`apply_taps` (frozen taps).
- Pilot-tone polarization demultiplexing:
  :func:`estimate_polarization_tones` (a :class:`JonesTrack`) and
  :func:`correct_polarization`.

The adaptive equalizers take ``(N,)`` or butterfly-MIMO ``(C, N)`` input,
run optional inline carrier-phase recovery (``cpr=PLL()`` or ``BPS()``) and
return an :class:`EqualizerResult`, whose ``state`` (an
:class:`EqualizerState`) continues a later call with ``state=``.
"""

from __future__ import annotations

from ._block import block_lms
from .blind import block_cma, block_rde
from .linear import apply_taps, estimate_transfer_function, zf_equalizer
from .pilots import EqualizerPilots
from .polarization import (
    JonesTrack,
    correct_polarization,
    estimate_polarization_tones,
)
from .result import EqualizerResult, EqualizerState
from .sequential import cma, lms, rde, rls

__all__ = [
    "EqualizerPilots",
    "EqualizerResult",
    "EqualizerState",
    "JonesTrack",
    "apply_taps",
    "block_cma",
    "block_lms",
    "block_rde",
    "cma",
    "correct_polarization",
    "estimate_polarization_tones",
    "estimate_transfer_function",
    "lms",
    "rde",
    "rls",
    "zf_equalizer",
]
