"""Fiber-channel impairments.

Linear effects (chromatic dispersion, PMD, polarization mixing) are in
:mod:`~commkit.impairments.channel.linear`; nonlinear propagation over
amplified spans (split-step Kerr effect with dispersion, loss, PMD and ASE)
is in :mod:`~commkit.impairments.channel.nonlinear`.
"""

from .linear import (
    apply_chromatic_dispersion,
    apply_pmd,
    apply_polarization_mixing,
)
from .nonlinear import Fiber, apply_fiber_propagation

__all__ = [
    "Fiber",
    "apply_chromatic_dispersion",
    "apply_fiber_propagation",
    "apply_pmd",
    "apply_polarization_mixing",
]
