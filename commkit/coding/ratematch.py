"""Rate matching: :class:`Punctured` and :class:`Shortened` codes.

Both wrap a code and are codes themselves, so :func:`~commkit.coding.encode`
and :func:`~commkit.coding.decode` (with the inner code's decoders) take
them like any other:

- :class:`Punctured` leaves chosen codeword bits untransmitted; the decoder
  sees them as erasures (zero LLRs).  Rate up.
- :class:`Shortened` fixes the last information bits of a systematic code
  to zero and does not transmit them; the decoder knows them (a large
  positive LLR).  Rate down, the code's correction power kept.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np

from ..backend import ArrayType, _device_tables, dispatch
from .base import (
    _Code,
    _Codec,
    _codec,
    _decode_blocks,
    _Decoded,
    _encode_blocks,
    _info_positions,
    _register,
    _resolve_method,
)

__all__ = ["Punctured", "Shortened"]

# LLR of a bit known to be zero: far above any channel LLR, finite.
_KNOWN_LLR = 1.0e4


@dataclass(frozen=True, eq=False)
class Punctured(_Code):
    """A code whose codeword bits at ``positions`` are not transmitted.

    Parameters
    ----------
    code : code object
        The mother code.
    positions : array_like
        Distinct codeword positions in ``[0, code.n)`` to drop.  For a
        systematic code, puncture parity bits only to keep the information
        recoverable.

    Notes
    -----
    ``n = code.n - len(positions)``, ``k = code.k``.  Decoding inserts zero
    LLRs at the punctured positions and decodes the mother code.
    """

    code: Any
    positions: np.ndarray

    def __post_init__(self) -> None:
        _codec(self.code, "Punctured()")
        p = np.unique(np.asarray(self.positions))
        if p.size != np.size(self.positions) or p.size == 0:
            raise ValueError("positions must be distinct and not empty.")
        if p.dtype.kind not in "iu" or p[0] < 0 or p[-1] >= self.code.n:
            raise ValueError(f"positions must be integers in [0, {self.code.n}).")
        if p.size >= self.code.n:
            raise ValueError("positions cannot cover the whole codeword.")
        p = p.astype(np.int64)
        p.setflags(write=False)
        object.__setattr__(self, "positions", p)

    @property
    def n(self) -> int:
        """Transmitted bits per block."""
        return int(self.code.n - self.positions.size)

    @property
    def k(self) -> int:
        """Information bits per block."""
        return int(self.code.k)

    def __repr__(self) -> str:
        return f"Punctured({self.code!r}, {self.positions.size} bits)"

    @cached_property
    def _kept(self) -> np.ndarray:
        return np.setdiff1d(np.arange(self.code.n), self.positions)


@dataclass(frozen=True, eq=False)
class Shortened(_Code):
    """A systematic code with its last ``num_bits`` information bits fixed to zero.

    Parameters
    ----------
    code : code object
        A systematic code (linear block codes in systematic form, LDPC
        codes, CRCs, and wrappers of them).
    num_bits : int
        Information bits fixed to zero and not transmitted, ``1 <= num_bits
        < code.k``: the last ones, as IEEE 802.3ca appends its zero padding.

    Notes
    -----
    ``n = code.n - num_bits``, ``k = code.k - num_bits``.
    """

    code: Any
    num_bits: int

    def __post_init__(self) -> None:
        _info_positions(self.code, "Shortened()")
        s = self.num_bits
        if isinstance(s, bool) or not isinstance(s, int) or not 1 <= s < self.code.k:
            raise ValueError(
                f"num_bits must be an integer in [1, {self.code.k}), got {s!r}."
            )

    @property
    def n(self) -> int:
        """Transmitted bits per block."""
        return int(self.code.n - self.num_bits)

    @property
    def k(self) -> int:
        """Information bits per block."""
        return int(self.code.k - self.num_bits)

    def __repr__(self) -> str:
        return f"Shortened({self.code!r}, {self.num_bits} bits)"

    @cached_property
    def _layout(self) -> tuple[np.ndarray, np.ndarray]:
        """``(kept, zeros)``: kept codeword positions and the shortened ones."""
        zeros = np.sort(_info_positions(self.code, "Shortened()")[-self.num_bits :])
        return np.setdiff1d(np.arange(self.code.n), zeros), zeros


def _expand(llrs: ArrayType, kept: Any, n: int, fill: float) -> ArrayType:
    """``(R, len(kept))`` LLRs into ``(R, n)`` with ``fill`` elsewhere."""
    _, xp, _ = dispatch(llrs)
    full = xp.full((llrs.shape[0], n), fill, dtype=xp.float32)
    full[:, kept] = llrs
    return full


def _encode_punctured(code: Punctured, bits: ArrayType) -> ArrayType:
    _, xp, _ = dispatch(bits)
    (kept,) = _device_tables(code, "kept", xp, lambda: (code._kept,))
    return _encode_blocks(code.code, bits)[:, kept]


def _decode_punctured(code: Punctured, llrs: ArrayType, method: Any) -> _Decoded:
    _, xp, _ = dispatch(llrs)
    (kept,) = _device_tables(code, "kept", xp, lambda: (code._kept,))
    res = _decode_blocks(code.code, _expand(llrs, kept, code.code.n, 0.0), method)
    return res._replace(llrs=None if res.llrs is None else res.llrs[:, kept])


def _encode_shortened(code: Shortened, bits: ArrayType) -> ArrayType:
    _, xp, _ = dispatch(bits)
    kept, _ = _device_tables(code, "layout", xp, lambda: code._layout)
    padded = xp.zeros((bits.shape[0], code.code.k), dtype=np.int8)
    padded[:, : code.k] = bits
    return _encode_blocks(code.code, padded)[:, kept]


def _decode_shortened(code: Shortened, llrs: ArrayType, method: Any) -> _Decoded:
    _, xp, _ = dispatch(llrs)
    kept, _ = _device_tables(code, "layout", xp, lambda: code._layout)
    res = _decode_blocks(
        code.code, _expand(llrs, kept, code.code.n, _KNOWN_LLR), method
    )
    return res._replace(
        bits=res.bits[:, : code.k],
        llrs=None if res.llrs is None else res.llrs[:, kept],
    )


def _punctured_info(code: Punctured) -> np.ndarray | None:
    info = _codec(code.code, "Punctured()").info_positions
    inner = None if info is None else info(code.code)
    if inner is None or np.isin(inner, code.positions).any():
        return None
    return np.searchsorted(code._kept, inner)


def _shortened_info(code: Shortened) -> np.ndarray:
    kept, _ = code._layout
    inner = _info_positions(code.code, "Shortened()")[: code.k]
    return np.searchsorted(kept, inner)


for _cls, _enc, _dec, _info in (
    (Punctured, _encode_punctured, _decode_punctured, _punctured_info),
    (Shortened, _encode_shortened, _decode_shortened, _shortened_info),
):
    _register(
        _cls,
        _Codec(
            encode=_enc,
            decode=_dec,
            # The inner code's decoders, resolved (and checked) on the inner code.
            default_method=lambda code: _resolve_method(code.code, None, "decode()"),
            methods=(object,),
            info_positions=_info,
        ),
    )
