"""
Channel coding / forward error correction (FEC).

Scaffold only - no algorithms are implemented yet.  This package reserves the
public layout for the bits-layer neighbour of :mod:`commkit.mapping`:
encoders turn information bits -> coded bits, which ``mapping.map_bits`` maps to
symbols; soft decoders consume LLRs produced by ``mapping.compute_llr`` /
``metrics`` (see :mod:`commkit.coding.base` for the shared hard/soft
interface conventions).

Each module below is an importable placeholder carrying only its scope
docstring.  The package is intentionally **absent from the top-level
``commkit`` public surface** until at least one real encode/decode entry
point exists; do not add ``from . import coding`` to ``commkit/__init__.py``
before then.

Promotion path (apply the §7.4 size+cohesion trigger as real code lands):
``ldpc`` and ``polar`` graduate to ``{construction,decode}`` subpackages, and
the algebraic block codes (``hamming``/``bch``/``reed_solomon``) collect under a
``block/`` subpackage once they share enough ``galois`` machinery.
"""

__all__: list[str] = []
