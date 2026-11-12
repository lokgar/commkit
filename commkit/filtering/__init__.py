"""Signal-chain filters, pulse shapes and fiber compensation.

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
- Fiber compensation: :func:`estimate_chromatic_dispersion` (blind
  :class:`ClockToneScan` and :class:`FourthMomentScan`, giving a
  :class:`DispersionScanEstimate`, and data-aided :class:`QuadraticPhaseFit`,
  giving a :class:`QuadraticPhaseFitEstimate`) and :func:`correct_chromatic_dispersion`, the inverse
  of :func:`commkit.impairments.apply_chromatic_dispersion` (its FIR filter
  designed by :func:`dispersion_compensation_taps`), and digital
  backpropagation :func:`correct_fiber_propagation`, the inverse of
  :func:`commkit.impairments.apply_fiber_propagation`.

Design functions build coefficients from parameters alone (host NumPy
arrays); application functions run on the device of the samples.  Display
and estimation smoothers are in :mod:`commkit.smoothing`.
"""

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
    "ChromaticDispersionEstimate",
    "ClockToneScan",
    "QuadraticPhaseFitEstimate",
    "DispersionScanEstimate",
    "FilterResponse",
    "FourthMomentScan",
    "Gaussian",
    "Pulse",
    "QuadraticPhaseFit",
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
    "dispersion_compensation_taps",
    "elliptic_sos",
    "estimate_chromatic_dispersion",
    "filter_response",
    "fir_filter",
    "fir_taps",
    "iir_filter",
    "impulse_response",
    "matched_filter",
    "step_response",
]
