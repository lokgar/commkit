"""Signal-chain filters, pulse shapes and fiber compensation.

- Pulses: the value objects :class:`RRC`, :class:`RC`, :class:`Gaussian`,
  :class:`Rect` and :class:`SmoothRect`; ``pulse.taps(sps)`` gives the taps.
  Wherever a pulse is accepted, a raw taps array is accepted too.
- Design: :func:`fir_taps` (window method) and the IIR designs in
  second-order sections, :func:`butterworth_sos`, :func:`chebyshev1_sos`,
  :func:`chebyshev2_sos`, :func:`elliptic_sos` and :func:`bessel_sos`.
- Application: :func:`fir_filter` (FFT convolution, optionally circular),
  :func:`ols_fir_filter` (overlap-save, for long filters), :func:`iir_filter`
  and :func:`matched_filter`.
- Fiber compensation: :func:`correct_chromatic_dispersion`, the inverse of
  :func:`commkit.impairments.apply_chromatic_dispersion`, and digital
  backpropagation :func:`correct_fiber_propagation`, the inverse of
  :func:`commkit.impairments.apply_fiber_propagation`.

Design functions build coefficients from parameters alone (host NumPy
arrays); application functions run on the device of the samples.  Display
and estimation smoothers are in :mod:`commkit.smoothing`.
"""

from .fiber import correct_chromatic_dispersion, correct_fiber_propagation
from .linear import (
    bessel_sos,
    butterworth_sos,
    chebyshev1_sos,
    chebyshev2_sos,
    elliptic_sos,
    fir_filter,
    fir_taps,
    iir_filter,
    matched_filter,
    ols_fir_filter,
)
from .pulses import (
    RC,
    RRC,
    Gaussian,
    Pulse,
    Rect,
    SmoothRect,
)

__all__ = [
    "Gaussian",
    "Pulse",
    "RC",
    "RRC",
    "Rect",
    "SmoothRect",
    "bessel_sos",
    "butterworth_sos",
    "chebyshev1_sos",
    "chebyshev2_sos",
    "correct_chromatic_dispersion",
    "correct_fiber_propagation",
    "elliptic_sos",
    "fir_filter",
    "fir_taps",
    "iir_filter",
    "matched_filter",
    "ols_fir_filter",
]
