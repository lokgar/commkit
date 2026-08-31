"""
Fiber-channel impairments.

Linear effects (chromatic dispersion, PMD, polarization mixing) live in
:mod:`~commkit.impairments.channel.linear`; nonlinear propagation
(split-step Kerr) is reserved for
:mod:`~commkit.impairments.channel.nonlinear` (placeholder).
"""

from .linear import (
    apply_chromatic_dispersion,
    apply_pmd,
    apply_polarization_mixing,
)

__all__ = [
    "apply_chromatic_dispersion",
    "apply_pmd",
    "apply_polarization_mixing",
]
