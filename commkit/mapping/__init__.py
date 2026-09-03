"""
Symbol mapping, demapping, and constellation management.

This package provides high-performance routines for the transition between
digital bits and physical IQ symbols.  It is organised by mathematical concern:

- :mod:`~commkit.mapping.gray` - constellation geometry and Gray labelling.
- :mod:`~commkit.mapping.bits` - hard bit mapping / demapping.
- :mod:`~commkit.mapping.llr` - soft-decision (LLR) demapping.
- :mod:`~commkit.mapping.shaping` - probabilistic shaping (PS-QAM).
- :mod:`~commkit.mapping.constellation` - the :class:`Constellation` value
  object bundling points + Gray labels + optional shaping pmf.

The public import surface is stable: every name previously importable from the
flat ``commkit.mapping`` module is re-exported here.  ``Constellation`` is an
additive convenience over the existing loose-array free functions.

Note: codes and constellations are generated using NumPy (host-side).
"""

from .bits import demap_symbols_hard, map_bits
from .constellation import Constellation
from .gray import (
    gray_code,
    gray_constellation,
    gray_to_binary,
    nearest_constellation_index,
    square_qam_slicer_params,
    unpack_bits,
)
from .llr import compute_llr
from .shaping import (
    constellation_power,
    maxwell_boltzmann,
    optimal_nu,
    ps_entropy,
    rescale_ps_symbols,
    sample_ps_symbols,
)

__all__ = [
    "Constellation",
    "compute_llr",
    "constellation_power",
    "demap_symbols_hard",
    "gray_code",
    "gray_constellation",
    "gray_to_binary",
    "map_bits",
    "maxwell_boltzmann",
    "nearest_constellation_index",
    "optimal_nu",
    "ps_entropy",
    "rescale_ps_symbols",
    "sample_ps_symbols",
    "square_qam_slicer_params",
    "unpack_bits",
]
