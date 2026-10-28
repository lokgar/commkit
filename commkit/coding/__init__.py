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

Codes: :class:`LinearBlockCode` (Hamming), :class:`LDPCCode` (DVB-S2, PON),
:class:`ReedSolomonCode` (G.709, KR4, KP4, PON), :class:`BCHCode`,
:class:`CRC` (error detection), and the rate-matching wrappers
:class:`Punctured` and :class:`Shortened`.  Distribution matching:
:class:`CCDM` and probabilistic amplitude shaping, :class:`PAS`.  Decoders: :class:`Syndrome`,
:class:`MaximumLikelihood`, :class:`MinSum`, :class:`SumProduct`,
:class:`BerlekampMassey`.  Bit
permutations: :func:`interleave` and :func:`deinterleave` with an
:class:`Interleaver`.

The modules ``convolutional``, ``polar`` and ``turbo`` are placeholders for planned codes, carrying only their scope
docstring.
"""

from .algebraic import BCHCode, BerlekampMassey, ReedSolomonCode
from .base import DecodeResult, decode, encode
from .crc import CRC
from .interleaving import Interleaver, deinterleave, interleave
from .ldpc import LDPCCode, MinSum, SumProduct
from .linear import LinearBlockCode, MaximumLikelihood, Syndrome
from .ratematch import Punctured, Shortened
from .shaping import CCDM, PAS

__all__ = [
    "BCHCode",
    "BerlekampMassey",
    "CCDM",
    "CRC",
    "DecodeResult",
    "Interleaver",
    "LDPCCode",
    "LinearBlockCode",
    "MaximumLikelihood",
    "MinSum",
    "PAS",
    "Punctured",
    "ReedSolomonCode",
    "Shortened",
    "SumProduct",
    "Syndrome",
    "decode",
    "deinterleave",
    "encode",
    "interleave",
]
