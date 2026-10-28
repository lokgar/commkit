"""Channel coding: forward error correction (FEC) codes and their decoders.

The bit-level neighbor of :mod:`commkit.mapping`: :func:`encode` turns
information bits into the transmitted bits of a code, which
``Constellation.map`` maps to symbols; :func:`decode` takes the LLRs of
:func:`commkit.mapping.compute_llr` (positive means bit 0) or hard
decisions back to information bits, as a :class:`DecodeResult`.  Both
dispatch on the code object, and :func:`decode` on an optional decoder
object::

    code = LinearBlockCode.hamming(7, extended=True)
    res = decode(compute_llr(rx, noise_var=nv), code=code, method=Syndrome())

Codes: :class:`LinearBlockCode` (Hamming), :class:`LDPCCode` (DVB-S2, PON).
Decoders: :class:`Syndrome`, :class:`MaximumLikelihood`, :class:`MinSum`,
:class:`SumProduct`.

The modules ``bch``, ``convolutional``, ``crc``, ``galois``,
``interleaving``, ``polar``, ``ratematch``, ``reed_solomon`` and ``turbo``
are placeholders for planned codes, carrying only their scope
docstring.
"""

from .base import DecodeResult, decode, encode
from .ldpc import LDPCCode, MinSum, SumProduct
from .linear import LinearBlockCode, MaximumLikelihood, Syndrome

__all__ = [
    "DecodeResult",
    "LDPCCode",
    "LinearBlockCode",
    "MaximumLikelihood",
    "MinSum",
    "SumProduct",
    "Syndrome",
    "decode",
    "encode",
]
