"""
Core data structures and signal factories for commkit.

Re-exports the primary containers and generation factories so existing imports
(``from commkit.core import Signal, Preamble, SingleCarrierFrame``) keep
working after the split of the former monolithic ``core.py`` into a package.

``signal`` is the thin container (no leaf-module dependencies); ``frame`` and
``generation`` build on it.  The generation factories are also re-exported at
the package top level (``commkit.generate_qam(...)`` etc.).
"""

from .frame import Preamble, SingleCarrierFrame
from .generation import (
    generate,
    generate_pam,
    generate_psk,
    generate_psqam,
    generate_qam,
)
from .signal import Signal

__all__ = [
    "Preamble",
    "Signal",
    "SingleCarrierFrame",
    "generate",
    "generate_pam",
    "generate_psk",
    "generate_psqam",
    "generate_qam",
]
