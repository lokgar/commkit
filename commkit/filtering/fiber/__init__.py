"""Fiber: chromatic dispersion estimation and compensation, and backpropagation.

:mod:`.estimation` measures the accumulated dispersion
(:func:`estimate_chromatic_dispersion`); :mod:`.compensation` removes it
(:func:`correct_chromatic_dispersion`, with the FIR design
:func:`dispersion_compensation_taps`) and inverts the whole fiber channel
by digital backpropagation (:func:`correct_fiber_propagation`).  Users
import these names from :mod:`commkit.filtering`.
"""

from .compensation import (
    correct_chromatic_dispersion,
    correct_fiber_propagation,
    dispersion_compensation_taps,
)
from .estimation import (
    ChromaticDispersionEstimate,
    ClockToneScan,
    FourthMomentScan,
    QuadraticPhaseFit,
    estimate_chromatic_dispersion,
)

__all__ = [
    "ChromaticDispersionEstimate",
    "ClockToneScan",
    "FourthMomentScan",
    "QuadraticPhaseFit",
    "correct_chromatic_dispersion",
    "correct_fiber_propagation",
    "dispersion_compensation_taps",
    "estimate_chromatic_dispersion",
]
