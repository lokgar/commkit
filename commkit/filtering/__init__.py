"""Signal-chain filters and pulse shapes.

- Pulses: the value objects :class:`RRC`, :class:`RC`, :class:`Gaussian`,
  :class:`Rect` and :class:`SmoothRect`; ``pulse.taps(sps)`` gives the taps.
  Wherever a pulse is accepted, a raw taps array is accepted too.
- Design: :func:`fir_taps` (window method) and the IIR designs in
  second-order sections, :func:`butterworth_sos`, :func:`chebyshev1_sos`,
  :func:`chebyshev2_sos`, :func:`elliptic_sos` and :func:`bessel_sos`.
- Responses: :func:`filter_response` (a :class:`FilterResponse`),
  :func:`impulse_response` and :func:`step_response`, computed on the
  coefficients' device.
- Application: :func:`fir_filter` (direct, FFT or overlap-save convolution
  after the sizes, optionally circular), :func:`iir_filter` and
  :func:`matched_filter`.
Design functions build coefficients from parameters alone (host NumPy
arrays); application functions run on the device of the samples.  Filters
here are generic; the compensation of a particular impairment (chromatic
dispersion, a channel) is in :mod:`commkit.equalization`, display and
estimation smoothers are in :mod:`commkit.smoothing`.
"""

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
)
from .pulses import (
    RC,
    RRC,
    Gaussian,
    Pulse,
    Rect,
    SmoothRect,
)
from .response import FilterResponse, filter_response, impulse_response, step_response

__all__ = [
    "FilterResponse",
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
    "elliptic_sos",
    "filter_response",
    "fir_filter",
    "fir_taps",
    "iir_filter",
    "impulse_response",
    "matched_filter",
    "step_response",
]
