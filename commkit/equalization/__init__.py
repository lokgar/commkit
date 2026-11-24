"""Equalization: the inverses of channel and front-end responses.

- Sequential (symbol-by-symbol) equalizers :func:`lms` and :func:`rls`
  (trained, then decision-directed) and :func:`cma` and :func:`rde` (blind),
  as Numba loops on the CPU; :class:`EqualizerPilots` runs the blind ones
  pilot-aided.
- Frequency-domain block equalizers :func:`block_lms`, :func:`block_cma` and
  :func:`block_rde`, on the input's device (CUDA graphs on the GPU).
- Linear block equalization: :func:`estimate_channel` (a
  :class:`ChannelEstimate` from a known reference), :func:`correct_channel`
  (its ZF / MMSE inverse) and :func:`apply_taps` (frozen taps).
- Gain: :func:`estimate_gain` (the data-aided amplitude gain against a
  reference, a :class:`GainEstimate`) and :func:`correct_gain`, which puts
  symbols on the reference's scale before LLRs.
- Polarization demultiplexing: :func:`estimate_polarization` (a
  :class:`PolarizationEstimate`) with :class:`PolarizationTones` (a
  :class:`PolarizationTonesEstimate`), and
  :func:`correct_polarization`.
- Fiber: :func:`estimate_chromatic_dispersion` (blind
  :class:`ClockToneScan` and :class:`FourthMomentScan`, giving a
  :class:`DispersionScanEstimate`, and data-aided :class:`QuadraticPhaseFit`,
  giving a :class:`QuadraticPhaseFitEstimate`) and
  :func:`correct_chromatic_dispersion`, the inverse of
  :func:`commkit.impairments.apply_chromatic_dispersion` (its FIR filter
  designed by :func:`dispersion_compensation_taps`), and digital
  backpropagation :func:`correct_fiber_propagation`, the inverse of
  :func:`commkit.impairments.apply_fiber_propagation`.
- IQ imbalance: :func:`estimate_iq_imbalance` (an
  :class:`IQImbalanceEstimate`) and :func:`correct_iq_imbalance` with an
  estimate, :class:`Lowdin` or :class:`GramSchmidt`, the inverse of
  :func:`commkit.impairments.apply_iq_imbalance`.

The adaptive equalizers take ``(N,)`` or butterfly-MIMO ``(C, N)`` input,
run optional inline carrier-phase recovery (``cpr=PLL()`` or ``BPS()``) and
return an :class:`EqualizerResult`, whose ``state`` (an
:class:`EqualizerState`) continues a later call with ``state=``.
"""

from __future__ import annotations

from ._block import block_cma, block_lms, block_rde
from ._sequential import cma, lms, rde, rls
from .fiber import (
    ChromaticDispersionEstimate,
    ClockToneScan,
    DispersionScanEstimate,
    FourthMomentScan,
    QuadraticPhaseFit,
    QuadraticPhaseFitEstimate,
    correct_chromatic_dispersion,
    correct_fiber_propagation,
    dispersion_compensation_taps,
    estimate_chromatic_dispersion,
)
from .iq import (
    GramSchmidt,
    IQImbalanceEstimate,
    Lowdin,
    correct_iq_imbalance,
    estimate_iq_imbalance,
)
from .linear import (
    ChannelEstimate,
    GainEstimate,
    apply_taps,
    correct_channel,
    correct_gain,
    estimate_channel,
    estimate_gain,
)
from .pilots import EqualizerPilots
from .polarization import (
    PolarizationEstimate,
    PolarizationTones,
    PolarizationTonesEstimate,
    correct_polarization,
    estimate_polarization,
)
from .result import EqualizerResult, EqualizerState

__all__ = [
    "ChannelEstimate",
    "ChromaticDispersionEstimate",
    "ClockToneScan",
    "DispersionScanEstimate",
    "EqualizerPilots",
    "EqualizerResult",
    "EqualizerState",
    "FourthMomentScan",
    "GainEstimate",
    "GramSchmidt",
    "IQImbalanceEstimate",
    "Lowdin",
    "PolarizationEstimate",
    "PolarizationTones",
    "PolarizationTonesEstimate",
    "QuadraticPhaseFit",
    "QuadraticPhaseFitEstimate",
    "apply_taps",
    "block_cma",
    "block_lms",
    "block_rde",
    "cma",
    "correct_channel",
    "correct_chromatic_dispersion",
    "correct_fiber_propagation",
    "correct_gain",
    "correct_iq_imbalance",
    "correct_polarization",
    "dispersion_compensation_taps",
    "estimate_channel",
    "estimate_chromatic_dispersion",
    "estimate_gain",
    "estimate_iq_imbalance",
    "estimate_polarization",
    "lms",
    "rde",
    "rls",
]
