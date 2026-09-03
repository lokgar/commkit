"""
Block / frequency-domain equalizer engine (block_lms, FDAF).

Split into three internal concerns:

- ``_seqmode.py`` - the "block" execution-mode backend shared by
  ``sequential``'s ``lms``/``cma``/``rde`` (``update_mode='block'``).
- ``_dd.py`` - ``block_lms``, the standalone decision-directed
  frequency-domain (FDAF) block equalizer.
- ``_blind.py`` - the blind FDAF engine backing ``blind.py``'s
  ``block_cma``/``block_rde``.

The public import surface is unchanged: ``from commkit.equalization import
block_lms`` and ``from commkit.equalization._block import ...`` (used
internally by ``sequential`` and ``blind``) continue to work.
"""

from __future__ import annotations

from ._blind import _block_fdaf_blind
from ._dd import block_lms
from ._seqmode import (
    _build_slicer_constellation,
    _prep_blind_block_inputs,
    _run_block_equalizer,
    _validate_block_mode,
)

__all__ = [
    "_block_fdaf_blind",
    "_build_slicer_constellation",
    "_prep_blind_block_inputs",
    "_run_block_equalizer",
    "_validate_block_mode",
    "block_lms",
]
