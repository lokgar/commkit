"""The coding verbs :func:`encode` and :func:`decode`, and :class:`DecodeResult`.

Every code object (a FEC code, a distribution matcher or their
composition) maps blocks of ``k`` information bits to blocks of ``n``
transmitted bits.  :func:`encode` and :func:`decode` dispatch on the code's
type through a private table that each family registers into, and on the
decoder ``method`` object, which the family validates.

Soft (LLR) convention, as :func:`commkit.mapping.compute_llr`:
``LLR = log P(b = 0 | y) / P(b = 1 | y)``, so a positive LLR means bit 0.
Soft-output decoders return a-posteriori LLRs in the same convention.

Layout: bits and LLRs are ``(N,)`` or ``(C, N)``, blocks concatenated
along the last axis; a block never spans channels.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np

from .._array import as_2d, restore_1d, validate_array, warn_if_double
from ..backend import ArrayType, get_array_module
from ..logger import logger

__all__ = ["DecodeResult", "decode", "encode"]

# Magnitude of the LLR given to an integer hard decision.
_HARD_LLR = 1.0


@dataclass(frozen=True)
class DecodeResult:
    """Output of :func:`decode`, arrays on the input's device.

    Attributes
    ----------
    bits : ndarray
        Decoded information bits, int8, ``(..., B * k)``.
    llrs : ndarray or None
        A-posteriori LLRs of the transmitted bits, float32, ``(..., B * n)``,
        from soft-output decoders; ``None`` otherwise.
    valid : ndarray
        Per block, ``(..., B)`` bool: the decoder's own success check
        (syndrome zero, CRC passed).  Detected success, not true success.
    iterations : ndarray
        Iterations used per block, ``(..., B)`` int32 (1 for one-shot
        decoders).
    """

    bits: Any
    llrs: Any | None
    valid: Any
    iterations: Any


class _Decoded(NamedTuple):
    """A family decoder's output per block row: ``(R, k)``, ``(R, n)``, ``(R,)``."""

    bits: ArrayType
    llrs: ArrayType | None
    valid: ArrayType
    iterations: ArrayType


class _Value:
    """Value equality and hashing of frozen dataclasses with host arrays.

    Equality compares the fields (arrays by value), so equal objects share
    their cached device tables.
    """

    def _key(self) -> tuple[Any, ...]:
        return tuple(
            (v.shape, v.dtype.str, v.tobytes()) if isinstance(v, np.ndarray) else v
            for f in dataclasses.fields(self)  # type: ignore[arg-type]
            if f.compare
            for v in (getattr(self, f.name),)
        )

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return self._key() == other._key()  # type: ignore[attr-defined]

    def __hash__(self) -> int:
        # Cached: the device-table cache hashes the object on every call, and
        # a large code's arrays are megabytes.
        h: int | None = self.__dict__.get("_hash")
        if h is None:
            h = hash((type(self), self._key()))
            self.__dict__["_hash"] = h
        return h


class _Code(_Value):
    """Base of the code value objects: ``rate`` and value semantics.

    Subclasses are frozen dataclasses with read-only host arrays and define
    ``n`` and ``k``.
    """

    @property
    def rate(self) -> float:
        """Code rate ``k / n``."""
        return float(self.k / self.n)  # type: ignore[attr-defined]


@dataclass(frozen=True)
class _Codec:
    """A family's entry in the dispatch table."""

    encode: Callable[[Any, ArrayType], ArrayType]  # (code, (R, k)) -> (R, n)
    decode: Callable[[Any, ArrayType, Any], _Decoded]  # (code, (R, n), method)
    default_method: Callable[[Any], Any]
    methods: tuple[type, ...]
    # Codeword positions of the information bits, for systematic codes.
    info_positions: Callable[[Any], np.ndarray | None] | None = None
    # Sequential on the host (distribution matchers): GPU input round-trips,
    # and generate() encodes before moving the payload.
    host: bool = False


_CODECS: dict[type, _Codec] = {}


def _register(cls: type, codec: _Codec) -> None:
    _CODECS[cls] = codec


def _codec(code: Any, function_name: str) -> _Codec:
    codec = _CODECS.get(type(code))
    if codec is None:
        raise TypeError(
            f"{function_name}: code must be a code object (LinearBlockCode, ...), "
            f"got {type(code).__name__}."
        )
    return codec


def _resolve_method(code: Any, method: Any, function_name: str) -> Any:
    codec = _codec(code, function_name)
    if method is None:
        return codec.default_method(code)
    if not isinstance(method, codec.methods):
        names = ", ".join(m.__name__ for m in codec.methods)
        raise TypeError(
            f"{function_name}: method {type(method).__name__} does not decode "
            f"{type(code).__name__}; use one of {names}."
        )
    return method


def _blocks(x: ArrayType, size: int, name: str, function_name: str) -> ArrayType:
    """``(C, B * size)`` to ``(C * B, size)``; raises on a partial block."""
    if x.shape[-1] % size:
        raise ValueError(
            f"{function_name}: the last axis of {name} ({x.shape[-1]}) must be a "
            f"multiple of the block length {size}."
        )
    return x.reshape(-1, size)


def _info_positions(code: Any, function_name: str) -> np.ndarray:
    """Positions of the information bits in a systematic code's codewords."""
    positions = _codec(code, function_name).info_positions
    found = None if positions is None else positions(code)
    if found is None:
        raise TypeError(f"{function_name}: {code!r} is not systematic.")
    return found


def _encode_blocks(code: Any, bits: ArrayType) -> ArrayType:
    """Encode ``(R, k)`` int8 rows into ``(R, n)`` int8 rows."""
    if bits.shape[0] == 0:
        return get_array_module(bits).empty((0, code.n), dtype=np.int8)
    out: ArrayType = _codec(code, "encode()").encode(code, bits)
    return out


def _decode_blocks(code: Any, llrs: ArrayType, method: Any = None) -> _Decoded:
    """Decode ``(R, n)`` float32 LLR rows; for the families that nest codes."""
    method = _resolve_method(code, method, "decode()")
    if llrs.shape[0] == 0:
        xp = get_array_module(llrs)
        return _Decoded(
            xp.empty((0, code.k), dtype=np.int8),
            None,
            xp.empty(0, dtype=bool),
            xp.empty(0, dtype=np.int32),
        )
    return _codec(code, "decode()").decode(code, llrs, method)


def _hard(llrs: ArrayType) -> ArrayType:
    """Hard decisions of LLRs: int8 1 where the LLR is negative."""
    return (llrs < 0).astype(np.int8)


def _as_llrs(llrs: Any, function_name: str) -> ArrayType:
    """float32 LLRs; integer or bool input is read as hard decisions."""
    x = validate_array(llrs, name="llrs")
    xp = get_array_module(x)
    if x.dtype.kind == "c":
        raise TypeError(f"{function_name}: llrs must be real, got {x.dtype}.")
    if x.dtype.kind in "biu":
        return xp.where(x != 0, -_HARD_LLR, _HARD_LLR).astype(xp.float32)
    warn_if_double(x, function_name, "float32")
    return x.astype(xp.float32, copy=False)


def encode(bits: ArrayType, *, code: Any) -> ArrayType:
    """Encode information bits into the transmitted bits of a code.

    Parameters
    ----------
    bits : array_like
        Information bits, values 0 and 1, ``(N,)`` or ``(C, N)`` with ``N`` a
        multiple of ``code.k``.
    code : code object
        The code: :class:`LinearBlockCode`, ...

    Returns
    -------
    ndarray
        Transmitted bits, int8, ``(..., N / k * n)``, on the input's device.

    Raises
    ------
    TypeError
        If ``code`` is not a code object.
    ValueError
        If ``N`` is not a multiple of ``code.k``.

    See Also
    --------
    decode : The inverse, from LLRs.
    """
    _codec(code, "encode()")
    x = validate_array(bits, name="bits")
    if x.dtype.kind not in "biu":
        raise TypeError(f"encode(): bits must be integers or bool, got {x.dtype}.")
    x2, was_1d = as_2d(x, name="bits")
    blocks = _blocks(x2, code.k, "bits", "encode()").astype(np.int8, copy=False)
    logger.debug("Encoding %d block(s) with %r.", blocks.shape[0], code)
    out = _encode_blocks(code, blocks)
    num_blocks = x2.shape[-1] // code.k
    return restore_1d(was_1d, out.reshape(x2.shape[0], num_blocks * code.n))


def decode(llrs: ArrayType, *, code: Any, method: Any = None) -> DecodeResult:
    """Decode LLRs (or hard decisions) of a code's transmitted bits.

    Parameters
    ----------
    llrs : array_like
        LLRs of the transmitted bits, ``(N,)`` or ``(C, N)`` with ``N`` a
        multiple of ``code.n``; positive means bit 0, as
        :func:`commkit.mapping.compute_llr` returns.  float32 (float64 is
        reduced to float32 with a :class:`~commkit.PrecisionWarning`).
        Integer or bool input is read as hard decisions (1 for bit 1).
    code : code object
        The code: :class:`LinearBlockCode`, ...
    method : decoder object, optional
        The decoding algorithm (:class:`Syndrome`, :class:`MaximumLikelihood`,
        ...).  ``None`` takes the code's default.

    Returns
    -------
    DecodeResult
        Decoded information bits, a-posteriori LLRs (soft-output decoders),
        and per block the success check and the iterations used.

    Raises
    ------
    TypeError
        If ``code`` is not a code object or ``method`` does not decode it.
    ValueError
        If ``N`` is not a multiple of ``code.n``.

    Warns
    -----
    PrecisionWarning
        For float64 LLRs.

    See Also
    --------
    encode : The inverse.
    """
    method = _resolve_method(code, method, "decode()")
    x = _as_llrs(llrs, "decode()")
    x2, was_1d = as_2d(x, name="llrs")
    blocks = _blocks(x2, code.n, "llrs", "decode()")
    logger.debug("Decoding %d block(s) of %r with %r.", blocks.shape[0], code, method)
    res = _decode_blocks(code, blocks, method)
    num_blocks = x2.shape[-1] // code.n

    def shaped(a: ArrayType | None, size: int) -> ArrayType | None:
        """Per-block rows back to the input's layout, ``size`` per block."""
        if a is None:
            return None
        return restore_1d(was_1d, a.reshape(x2.shape[0], num_blocks * size))

    return DecodeResult(
        bits=shaped(res.bits, code.k),
        llrs=shaped(res.llrs, code.n),
        valid=shaped(res.valid, 1),
        iterations=shaped(res.iterations, 1),
    )
