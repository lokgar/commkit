"""Channel impairments, grouped by where in the link they originate.

- :mod:`~commkit.impairments.noise`: additive noise (:func:`apply_awgn`).
- :mod:`~commkit.impairments.source`: laser and oscillator phase noise
  (:func:`generate_phase_noise`, :func:`apply_phase_noise`).
- :mod:`~commkit.impairments.frontend`: transceiver IQ imbalance
  (:func:`apply_iq_imbalance`, and the blind :func:`correct_iq_imbalance`
  with :class:`Lowdin` or :class:`GramSchmidt`).
- :mod:`~commkit.impairments.channel`: the fiber channel, linear
  (:func:`apply_chromatic_dispersion`, :func:`apply_pmd`,
  :func:`apply_polarization_mixing`) and nonlinear
  (:func:`apply_fiber_propagation` over amplified spans of a :class:`Fiber`).
"""

from .channel import (
    Fiber,
    apply_chromatic_dispersion,
    apply_fiber_propagation,
    apply_pmd,
    apply_polarization_mixing,
)
from .frontend import GramSchmidt, Lowdin, apply_iq_imbalance, correct_iq_imbalance
from .noise import apply_awgn
from .source import apply_phase_noise, generate_phase_noise

__all__ = [
    "Fiber",
    "GramSchmidt",
    "Lowdin",
    "apply_awgn",
    "apply_chromatic_dispersion",
    "apply_fiber_propagation",
    "apply_iq_imbalance",
    "apply_phase_noise",
    "apply_pmd",
    "apply_polarization_mixing",
    "correct_iq_imbalance",
    "generate_phase_noise",
]
