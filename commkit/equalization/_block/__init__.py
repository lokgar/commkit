"""Frequency-domain (FDAF) block equalizers (private).

- ``_engine.py``: the block loop, filter and gradient update shared by all.
- ``_dd.py``: :func:`block_lms`, the decision-directed block equalizer.
- ``_blind.py``: the blind engine behind :func:`commkit.equalization.block_cma`
  and :func:`commkit.equalization.block_rde` (``equalization/blind.py``).
"""

from __future__ import annotations

from ._blind import _block_fdaf_blind
from ._dd import block_lms

__all__ = [
    "_block_fdaf_blind",
    "block_lms",
]
