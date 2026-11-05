"""Frequency-domain (FDAF) block equalizers (private).

- ``_engine.py``: the block loop, filter and gradient update shared by all.
- ``_dd.py``: :func:`block_lms`, the trained and decision-directed block
  equalizer.
- ``_blind.py``: :func:`block_cma` and :func:`block_rde`, the blind block
  equalizers, and their FDAF loop.

The public names are re-exported by :mod:`commkit.equalization`.
"""

from __future__ import annotations

from ._blind import block_cma, block_rde
from ._dd import block_lms

__all__ = [
    "block_cma",
    "block_lms",
    "block_rde",
]
