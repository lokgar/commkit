"""
Symbol mapping, demapping and constellations.

A modulation is described by a :class:`Constellation` (points, bit labels and
an optional shaping prior); every function here takes one, never a
``modulation`` string plus an ``order``:

- :class:`Constellation` - factories ``.qam/.psk/.pam``, ``.shaped()``, and
  the array-level ``map`` / ``demap`` / ``llr``.
- :func:`map_bits`, :func:`demap_symbols_hard`, :func:`compute_llr` - the same
  operations as functions; demapping and LLRs also accept a Signal.
- :func:`maxwell_boltzmann`, :func:`optimal_nu` - probabilistic shaping.
- :func:`gray_code`, :func:`gray_to_binary` - Gray code sequences.

Constellations live on the host; operations run on the data's device.
"""

from .bits import demap_symbols_hard, map_bits
from .constellation import Constellation
from .gray import gray_code, gray_to_binary
from .llr import compute_llr
from .shaping import maxwell_boltzmann, optimal_nu

__all__ = [
    "Constellation",
    "compute_llr",
    "demap_symbols_hard",
    "gray_code",
    "gray_to_binary",
    "map_bits",
    "maxwell_boltzmann",
    "optimal_nu",
]
