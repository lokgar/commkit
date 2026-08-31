"""
Channel impairments and signal degradation models.

This package provides routines for simulating physical layer impairments,
enabling the evaluation of receiver performance under realistic channel
conditions.  The impairments are grouped by where in the link the effect
originates:

- :mod:`~commkit.impairments.noise` - additive measurement noise (AWGN).
- :mod:`~commkit.impairments.source` - laser/oscillator phase noise.
- :mod:`~commkit.impairments.frontend` - transceiver IQ imbalance
  (application + blind compensation).
- :mod:`~commkit.impairments.channel` - fiber-channel effects (linear:
  chromatic dispersion, PMD, polarization mixing; nonlinear: placeholder).

The public import surface is stable: ``from commkit.impairments import
apply_awgn`` (and every other ``apply_*`` / ``compensate_*`` name) is unchanged.
"""

from .channel import (
    apply_chromatic_dispersion,
    apply_pmd,
    apply_polarization_mixing,
)
from .frontend import (
    apply_iq_imbalance,
    compensate_iq_imbalance_gram_schmidt,
    compensate_iq_imbalance_lowdin,
)
from .noise import apply_awgn
from .source import apply_phase_noise, generate_phase_noise

__all__ = [
    "apply_awgn",
    "apply_chromatic_dispersion",
    "apply_iq_imbalance",
    "apply_phase_noise",
    "apply_pmd",
    "apply_polarization_mixing",
    "compensate_iq_imbalance_gram_schmidt",
    "compensate_iq_imbalance_lowdin",
    "generate_phase_noise",
]
