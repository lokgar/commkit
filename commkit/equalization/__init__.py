"""Adaptive and block channel equalization.

- Sequential (symbol-by-symbol) equalizers :func:`lms` and :func:`rls`
  (trained, then decision-directed) and :func:`cma` and :func:`rde` (blind),
  as Numba loops on the CPU; :func:`build_pilot_ref` builds the pilot
  reference of pilot-aided ``cma`` / ``rde``.
- Frequency-domain block equalizers :func:`block_lms`, :func:`block_cma` and
  :func:`block_rde`, on the input's device (CUDA graphs on the GPU).
- Linear block equalization: :func:`estimate_transfer_function` (channel
  estimate), :func:`zf_equalizer` (its ZF / MMSE inverse) and
  :func:`apply_taps` (frozen taps).
- Pilot-tone polarization demultiplexing:
  :func:`demultiplex_polarization_tones_static`,
  :func:`demultiplex_polarization_tones_dynamic` (with its
  :class:`JonesTrack`) and :func:`apply_interpolated_matrix`.

The adaptive equalizers take ``(N,)`` or butterfly-MIMO ``(C, N)`` input,
run optional inline carrier-phase recovery (``cpr=PLL()`` or ``BPS()``) and
return an :class:`EqualizerResult`, whose ``state`` (an
:class:`EqualizerState`) continues a later call with ``state=``.
"""

from __future__ import annotations

from ._block import block_lms
from .blind import block_cma, block_rde, build_pilot_ref
from .linear import apply_taps, estimate_transfer_function, zf_equalizer
from .polarization import (
    JonesTrack,
    apply_interpolated_matrix,
    demultiplex_polarization_tones_dynamic,
    demultiplex_polarization_tones_static,
)
from .result import EqualizerResult, EqualizerState
from .sequential import cma, lms, rde, rls

__all__ = [
    "EqualizerResult",
    "EqualizerState",
    "JonesTrack",
    "apply_interpolated_matrix",
    "apply_taps",
    "block_cma",
    "block_lms",
    "block_rde",
    "build_pilot_ref",
    "cma",
    "demultiplex_polarization_tones_dynamic",
    "demultiplex_polarization_tones_static",
    "estimate_transfer_function",
    "lms",
    "rde",
    "rls",
    "zf_equalizer",
]
