"""Cyclic redundancy checks: :class:`CRC`.

A CRC is a code object: :func:`~commkit.coding.encode` appends the check
bits to each block of ``k`` information bits (``n = k + r``), and
:func:`~commkit.coding.decode` returns the hard decisions of the
information bits with ``valid`` telling whether the check matched.  CRCs
detect errors; they correct none.  Concatenated with a code, they select
among list-decoding candidates (CRC-aided polar decoding) or flag frames.

The register follows the Rocksoft model (polynomial, initial value, input
and output reflection, final XOR), so standard check values reproduce.  A
CRC is affine in the message bits, ``crc(u) = u G + c`` (mod 2), so
computing it is one :func:`~commkit.coding._gf2.matmul_mod2` per call on
the data's device, with ``G`` (``k x r``) and ``c`` built once per object.

Named polynomials: :meth:`CRC.nr` (3GPP TS 38.212, clause 5.1: CRC6,
CRC11, CRC16, CRC24A, CRC24B, CRC24C; LTE's CRC24A and CRC24B are the
same) and :meth:`CRC.ethernet` (IEEE 802.3 clause 3.2.9, CRC-32).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from ..backend import ArrayType, _device_tables, dispatch
from ._gf2 import matmul_mod2
from .base import _Code, _Codec, _Decoded, _hard, _register

__all__ = ["CRC"]

# 3GPP TS 38.212 clause 5.1: generator polynomials without the leading term.
_NR = {
    "6": (6, 0x21),
    "11": (11, 0x621),
    "16": (16, 0x1021),
    "24A": (24, 0x864CFB),
    "24B": (24, 0x800063),
    "24C": (24, 0xB2B117),
}


@dataclass(frozen=True, eq=False)
class CRC(_Code):
    """Cyclic redundancy check of ``degree`` bits over blocks of ``info_bits`` bits.

    Parameters
    ----------
    polynomial : int
        Generator polynomial without its leading ``x^degree`` term, MSB
        first (CRC-32: ``0x04C11DB7``).
    degree : int
        Check bits ``r``, 1 to 64.
    info_bits : int
        Information bits ``k`` per block.
    init : int, default 0
        Initial register value.
    reflect_in : bool, default False
        Reverse the bit order of each input byte (``info_bits`` must be a
        multiple of 8).
    reflect_out : bool, default False
        Reverse the check bits before the final XOR.
    xor_out : int, default 0
        Final XOR of the check bits.

    Notes
    -----
    Blocks are bit sequences, first bit first; for byte data the first bit
    is the MSB of the first byte (``reflect_in`` reverses that).  The check
    bits follow the information bits, MSB of the register first.
    """

    polynomial: int
    degree: int
    info_bits: int
    init: int = 0
    reflect_in: bool = False
    reflect_out: bool = False
    xor_out: int = 0

    def __post_init__(self) -> None:
        r, k = self.degree, self.info_bits
        if isinstance(r, bool) or not isinstance(r, int) or not 1 <= r <= 64:
            raise ValueError(f"degree must be an integer in [1, 64], got {r!r}.")
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError(f"info_bits must be an integer >= 1, got {k!r}.")
        for name in ("polynomial", "init", "xor_out"):
            value = getattr(self, name)
            if not isinstance(value, int) or not 0 <= value < (1 << r):
                raise ValueError(f"{name} must be an integer in [0, 2^{r}).")
        if self.reflect_in and k % 8:
            raise ValueError("reflect_in needs info_bits that is a multiple of 8.")

    @property
    def n(self) -> int:
        """Bits per block with the check: ``k + degree``."""
        return self.info_bits + self.degree

    @property
    def k(self) -> int:
        """Information bits per block."""
        return self.info_bits

    def __repr__(self) -> str:
        return f"CRC(0x{self.polynomial:X}, degree={self.degree}, k={self.k})"

    @classmethod
    def nr(cls, name: str, info_bits: int) -> CRC:
        """A CRC of 3GPP TS 38.212 clause 5.1 (5G NR; LTE uses 24A and 24B).

        Parameters
        ----------
        name : {"6", "11", "16", "24A", "24B", "24C"}
            The polynomial ``g_CRC<name>``.
        info_bits : int
            Information bits per block.

        Returns
        -------
        CRC
            Zero initial value, no reflection, no final XOR.
        """
        if name not in _NR:
            raise ValueError(f"name must be one of {sorted(_NR)}, got {name!r}.")
        degree, poly = _NR[name]
        return cls(poly, degree, info_bits)

    @classmethod
    def ethernet(cls, info_bits: int) -> CRC:
        """The CRC-32 of IEEE 802.3 (frame check sequence), over whole bytes.

        Parameters
        ----------
        info_bits : int
            Information bits per block, a multiple of 8.

        Returns
        -------
        CRC
            Polynomial ``0x04C11DB7``, initial value and final XOR all ones,
            reflected input and output: the check value of the ASCII bytes
            ``"123456789"`` is ``0xCBF43926``.
        """
        ones = (1 << 32) - 1
        return cls(0x04C11DB7, 32, info_bits, ones, True, True, ones)

    @cached_property
    def _tables(self) -> tuple[np.ndarray, np.ndarray]:
        """``(G, c)``: float32 ``(k, r)`` and int8 ``(r,)``, ``crc = u G + c``."""
        r, k = self.degree, self.info_bits
        g, c = _remainders(self.polynomial, r, k, self.init)
        if self.reflect_in:
            g = g.reshape(-1, 8, r)[:, ::-1].reshape(k, r)
        if self.reflect_out:
            g, c = g[:, ::-1], c[::-1]
        c = c ^ _bits(self.xor_out, r)
        return g.astype(np.float32), c.astype(np.int8)

    def checksum(self, bits: ArrayType) -> ArrayType:
        """The check bits of blocks of information bits.

        Parameters
        ----------
        bits : array_like
            Information bits, ``(..., k)``.

        Returns
        -------
        ndarray
            The check bits, int8, ``(..., degree)``, on the input's device.
        """
        _, xp, _ = dispatch(bits)
        g, c = _device_tables(self, "crc", xp, lambda: self._tables)
        return matmul_mod2(bits, g) ^ c


def _bits(value: int, width: int) -> np.ndarray:
    """``width`` bits of ``value``, MSB first, int8."""
    return np.array(
        [(value >> (width - 1 - j)) & 1 for j in range(width)], dtype=np.int8
    )


def _remainders(
    polynomial: int, degree: int, k: int, init: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Systematic parity of a binary cyclic code (or CRC) with generator ``g``.

    ``g(x) = x^degree + polynomial`` (MSB first).  Returns ``G`` ``(k,
    degree)`` int8, row ``i`` the remainder ``x^(degree + k - 1 - i) mod
    g`` of message bit ``i``, and ``c`` ``(degree,)``, the remainder of an
    initial register ``init``, ``init x^k mod g``: the parity of ``u`` is
    ``u G + c`` (mod 2).  Python integers, so any degree.  Shared by
    :class:`CRC` and the BCH codes.
    """
    mask = (1 << degree) - 1
    top = 1 << (degree - 1)

    def times_x(v: int) -> int:
        """``v x mod g``."""
        return ((v << 1) & mask) ^ (polynomial if v & top else 0)

    rows = [0] * k
    v = polynomial  # x^degree mod g
    for i in range(k - 1, -1, -1):
        rows[i] = v
        v = times_x(v)
    c = init
    for _ in range(k):
        c = times_x(c)
    g = np.array([_bits(row, degree) for row in rows], dtype=np.int8).reshape(k, degree)
    return g, _bits(c, degree)


def _encode(code: CRC, bits: ArrayType) -> ArrayType:
    _, xp, _ = dispatch(bits)
    return xp.concatenate([bits, code.checksum(bits)], axis=1)


def _decode(code: CRC, llrs: ArrayType, method: object) -> _Decoded:
    _, xp, _ = dispatch(llrs)
    hard = _hard(llrs)
    info = hard[:, : code.k]
    valid = xp.all(code.checksum(info) == hard[:, code.k :], axis=1)
    return _Decoded(info, None, valid, xp.ones(llrs.shape[0], dtype=np.int32))


@dataclass(frozen=True)
class _Check:
    """The only decoder of a CRC: compare the recomputed check bits."""


_register(
    CRC,
    _Codec(
        encode=_encode,
        decode=_decode,
        default_method=lambda code: _Check(),
        methods=(_Check,),
        info_positions=lambda code: np.arange(code.k),
    ),
)
