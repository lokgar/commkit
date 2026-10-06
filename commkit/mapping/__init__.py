"""
Symbol mapping, demapping, and constellation management.

This package provides high-performance routines for the transition between
digital bits and physical IQ symbols.  It is organised by mathematical concern:

- :mod:`~commkit.mapping.gray` - constellation geometry and Gray labelling.
- :mod:`~commkit.mapping.bits` - hard bit mapping / demapping.
- :mod:`~commkit.mapping.llr` - soft-decision (LLR) demapping.
- :mod:`~commkit.mapping.shaping` - probabilistic shaping (PS-QAM).
- :mod:`~commkit.mapping.constellation` - the :class:`Constellation` value
  object (points, bit labels, optional shaping pmf), the 2.0 way to describe a
  modulation.

The public import surface is stable: every name previously importable from the
flat ``commkit.mapping`` module is re-exported here.  The string-based free
functions take a ``Constellation`` from module pass 3.2 on.

Note: codes and constellations are generated using NumPy (host-side).
"""

from .bits import demap_symbols_hard, map_bits
from .constellation import Constellation
from .gray import gray_code, gray_to_binary
from .llr import compute_llr
from .shaping import maxwell_boltzmann, optimal_nu, ps_entropy

__all__ = [
    "Constellation",
    "compute_llr",
    "demap_symbols_hard",
    "gray_code",
    "gray_to_binary",
    "map_bits",
    "maxwell_boltzmann",
    "optimal_nu",
    "ps_entropy",
]
