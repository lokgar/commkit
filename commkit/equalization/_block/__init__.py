"""
Block / frequency-domain equalizer engine (block_lms, FDAF).

Split into two internal concerns:

- ``_dd.py`` - ``block_lms``, the standalone decision-directed
  frequency-domain (FDAF) block equalizer.
- ``_blind.py`` - the blind FDAF engine backing ``blind.py``'s
  ``block_cma``/``block_rde``.

The public import surface is unchanged: ``from commkit.equalization import
block_lms`` and ``from commkit.equalization._block import ...`` (used
internally by ``blind``) continue to work.
"""

from __future__ import annotations

from ._blind import _block_fdaf_blind
from ._dd import block_lms

__all__ = [
    "_block_fdaf_blind",
    "block_lms",
]
