"""
Channel impairments and signal degradation models.

This package provides routines for simulating physical layer impairments,
enabling the evaluation of receiver performance under realistic channel
conditions.  The impairments are grouped by where in the link the effect
originates:

- :mod:`~commkit.impairments.noise` - additive measurement noise (AWGN).
- :mod:`~commkit.impairments.source` - laser/oscillator phase noise.
- :mod:`~commkit.impairments.frontend` - transceiver IQ imbalance
  (``apply_iq_imbalance`` and the blind ``correct_iq_imbalance`` with
  ``Lowdin()`` / ``GramSchmidt()``).
- :mod:`~commkit.impairments.channel` - fiber-channel effects (linear:
  chromatic dispersion, PMD, polarization mixing; nonlinear: placeholder).

"""

from .channel import (
    apply_chromatic_dispersion,
    apply_pmd,
    apply_polarization_mixing,
)
from .frontend import GramSchmidt, Lowdin, apply_iq_imbalance, correct_iq_imbalance
from .noise import apply_awgn
from .source import apply_phase_noise, generate_phase_noise

__all__ = [
    "GramSchmidt",
    "Lowdin",
    "apply_awgn",
    "apply_chromatic_dispersion",
    "apply_iq_imbalance",
    "apply_phase_noise",
    "apply_pmd",
    "apply_polarization_mixing",
    "correct_iq_imbalance",
    "generate_phase_noise",
]
