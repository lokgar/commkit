"""Channel coding and forward error correction (FEC): planned, not implemented.

No algorithms exist yet.  This package reserves the public layout of the
bit-level neighbor of :mod:`commkit.mapping`: encoders turn information bits
into coded bits, which ``mapping.map_bits`` maps to symbols; soft decoders
consume the LLRs of ``mapping.compute_llr`` (see :mod:`commkit.coding.base`
for the shared hard and soft interface conventions).

Each module below is an importable placeholder carrying only its scope
docstring.  The package is intentionally **absent from the top-level
``commkit`` public surface** until at least one real encode/decode entry
point exists; do not add ``from . import coding`` to ``commkit/__init__.py``
before then.

Promotion path, applying the module-splitting rule of AGENTS.md as real code
lands: ``ldpc`` and ``polar`` become ``{construction,decode}`` subpackages,
and the algebraic block codes (``hamming``, ``bch``, ``reed_solomon``) move
into a ``block/`` subpackage once they share enough ``galois`` machinery.
"""

__all__: list[str] = []
