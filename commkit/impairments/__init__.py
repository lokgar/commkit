"""Channel impairments, grouped by where in the link they originate.

- :mod:`~commkit.impairments.noise`: additive noise (:func:`apply_awgn`),
  amplified spontaneous emission at an OSNR (:func:`apply_ase`) and the
  noise of coherent and square-law detection at absolute powers
  (:func:`apply_coherent_detection`, :func:`apply_photodetection`).
- :mod:`~commkit.impairments.source`: laser and oscillator phase noise
  (:func:`generate_phase_noise`, :func:`apply_phase_noise`), intensity
  noise (:func:`apply_intensity_noise`) and the carrier frequency offset
  (:func:`apply_frequency_offset`).
- :mod:`~commkit.impairments.frontend`: transceiver IQ imbalance
  (:func:`apply_iq_imbalance`), and the ADC
  (:func:`apply_sampling_clock_offset`, :func:`apply_sampling_jitter`,
  :func:`apply_quantization`).
- :mod:`~commkit.impairments.fiber_channel`: the fiber channel, linear
  (:func:`apply_chromatic_dispersion`, :func:`apply_pmd`,
  :func:`apply_polarization_mixing`) and nonlinear
  (:func:`apply_fiber_propagation` over amplified spans of a :class:`Fiber`).

These are forward models only.  Each effect's estimation and correction
lives in the receiver module of its quantity: :mod:`commkit.equalization`
(IQ imbalance, chromatic dispersion, digital backpropagation,
polarization), :mod:`commkit.frequency`, :mod:`commkit.timing` (sampling
clock) and :mod:`commkit.recovery` (carrier phase).
"""

from .fiber_channel import (
    Fiber,
    apply_chromatic_dispersion,
    apply_fiber_propagation,
    apply_pmd,
    apply_polarization_mixing,
)
from .frontend import (
    apply_iq_imbalance,
    apply_quantization,
    apply_sampling_clock_offset,
    apply_sampling_jitter,
)
from .noise import (
    apply_ase,
    apply_awgn,
    apply_coherent_detection,
    apply_photodetection,
)
from .source import (
    apply_frequency_offset,
    apply_intensity_noise,
    apply_phase_noise,
    generate_phase_noise,
)

__all__ = [
    "Fiber",
    "apply_ase",
    "apply_awgn",
    "apply_chromatic_dispersion",
    "apply_coherent_detection",
    "apply_fiber_propagation",
    "apply_frequency_offset",
    "apply_intensity_noise",
    "apply_iq_imbalance",
    "apply_quantization",
    "apply_sampling_clock_offset",
    "apply_sampling_jitter",
    "apply_phase_noise",
    "apply_photodetection",
    "apply_pmd",
    "apply_polarization_mixing",
    "generate_phase_noise",
]
