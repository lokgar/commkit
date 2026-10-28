"""Staircase and concatenated codes: :class:`StaircaseCode`, :class:`Concatenated`.

:class:`StaircaseCode` is the 512 x 510 staircase code of ITU-T G.709.2
Annex A (OTU4 long reach, and the outer code of OIF 400ZR): blocks of 512
rows and 510 columns, every row of block ``B_i`` a codeword of a
double-extended BCH(1022, 990) code (``t = 3`` plus two extra parity
checks) whose first 512 bits are a column of ``B_(i-1)``, permuted by the
standard's ``Pi_d``.  A frame of ``num_blocks`` blocks starts from a zero
block and ends with a terminating block whose information is zero, so the
last block is protected like the others; only the terminating block's
parity is transmitted.  :class:`HardIterations` decodes it: each
iteration corrects the rows of every block in order, each block's rows
in parallel (bounded-distance BCH decoding, kept only when all 32 parity
checks then hold).  The standard's decoder slides a window of at least
five blocks over the stream; iterating over the whole frame is the same
decoder with the window covering the frame.

:class:`Concatenated` puts an outer code (hard decision, such as a
Reed-Solomon or staircase code) around an inner one (soft decision, such
as a Hamming code with :class:`~commkit.coding.Chase`), with an optional
interleaver between them and zero padding to whole inner codewords:
:meth:`Concatenated.cfec` is the C-FEC of OIF 400ZR and
:meth:`Concatenated.ieee_802_3dj` the KP4 + Hamming(128, 120) code of IEEE
802.3dj, both at code level (the standards' framers, scramblers and
interleaver details are left out).  :class:`Stages` chooses the two
decoders.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np

from ..backend import ArrayType, _device_tables, dispatch
from ._galois import _Field
from ._gf2 import matmul_mod2
from .algebraic import _Algebraic, _correct
from .base import (
    _Code,
    _Codec,
    _codec,
    _decode_blocks,
    _Decoded,
    _encode_blocks,
    _hard,
    _info_positions,
    _register,
    _resolve_method,
)
from .interleaving import Interleaver, deinterleave, interleave
from .linear import LinearBlockCode

__all__ = ["Concatenated", "HardIterations", "Stages", "StaircaseCode"]

_ROWS = 512
_COLUMNS = 510
_INFO = 478
_PARITY = 32


def _pi_d() -> np.ndarray:
    """``Pi_d`` of ITU-T G.709.2 Table A.2: column of ``B_(i-1)`` for row ``j + 2``."""
    ranges = [
        (0, 7, 478), (8, 8, 0), (9, 11, 486), (12, 12, 1), (13, 13, 489), (14, 16, 2),
        (17, 19, 490), (20, 20, 5), (21, 21, 493), (22, 24, 6), (25, 25, 494), (26, 32, 9),
        (33, 35, 495), (36, 36, 16), (37, 37, 498), (38, 40, 17), (41, 41, 499), (42, 48, 20),
        (49, 49, 500), (50, 64, 27), (65, 67, 501), (68, 68, 42), (69, 69, 504), (70, 72, 43),
        (73, 73, 505), (74, 80, 46), (81, 81, 506), (82, 128, 53), (129, 129, 507),
        (130, 130, 100), (131, 131, 508), (132, 256, 101), (257, 257, 509), (258, 509, 226),
    ]  # fmt: skip
    pi = np.empty(_COLUMNS, dtype=np.int64)
    for start, stop, value in ranges:
        pi[start : stop + 1] = value + np.arange(stop - start + 1)
    return pi


@dataclass(frozen=True, eq=False)
class _StaircaseComponent(_Algebraic):
    """The double-extended BCH(1022, 990) of G.709.2 clause A.7 (private).

    Position ``p`` has the locator whose binary representation is the
    integer of the standard's column ``f(i)``: ``1021, 1022, 1 ... 510``
    for the 512 left bits, ``511 + Pi_d^-1(k)`` for right column ``k``.
    Rows ``beta, beta^3, beta^5`` make it a BCH code with ``t = 3``; two
    more rows, ``F(beta)`` and its complement, a function of the three
    low bits, extend it.
    """

    @property
    def n(self) -> int:
        """Bits per component codeword, 1022."""
        return _ROWS + _COLUMNS

    @property
    def k(self) -> int:
        """Information bits per component codeword, 990."""
        return self.n - _PARITY

    @property
    def length(self) -> int:  # type: ignore[override]
        """Transmitted bits per codeword."""
        return self.n

    @property
    def t(self) -> int:  # type: ignore[override]
        """Correctable errors of the BCH part."""
        return 3

    @property
    def first_root(self) -> int:  # type: ignore[override]
        """``b = 1``: roots ``beta, beta^3, beta^5``."""
        return 1

    @property
    def binary(self) -> bool:  # type: ignore[override]
        """True: the symbols are bits."""
        return True

    @cached_property
    def field(self) -> _Field:  # type: ignore[override]
        """GF(2^10) of the locators."""
        return _Field(10, 0x409)  # p(x) = 1 + x^3 + x^10, clause A.6

    @cached_property
    def locators(self) -> np.ndarray:
        """The locator ``X_p`` of each bit, a GF(2^10) element as an integer, ``(1022,)``."""
        inverse = np.argsort(_pi_d())
        left = np.concatenate([[1021, 1022], np.arange(1, 511)])
        return np.concatenate([left, 511 + inverse])

    @cached_property
    def _locator_logs(self) -> np.ndarray:
        return self.field._tables[1][self.locators]

    @cached_property
    def parity_check(self) -> np.ndarray:
        """``H``, ``(32, 1022)``: the bits of ``beta, beta^3, beta^5``, ``F``, ``not F``."""
        f = self.field
        logs = self._locator_logs
        rows = [f.bits(f._tables[0][(power * logs) % 1023]).T for power in (1, 3, 5)]
        b = (self.locators[None, :] >> np.arange(3)[:, None]) & 1  # b0, b1, b2
        b0, b1, b2 = b.astype(bool)
        extra = (b2 & ~b1 & ~b0) | (~b2 & b1) | (~b2 & ~b1 & b0)
        return np.concatenate([*rows, extra[None], ~extra[None]]).astype(np.int8)

    def correct(self, words: ArrayType) -> tuple[ArrayType, ArrayType]:
        """Bounded-distance correction of ``(R, 1022)`` words.

        A correction is kept only where all 32 checks then hold (the
        extension rejects most miscorrections).
        """
        _, xp, _ = dispatch(words)
        fixed, ok = _correct(self, words)
        (h_t,) = _device_tables(
            self, "checks", xp, lambda: (self.parity_check.T.astype(np.float32),)
        )
        ok = ok & ~matmul_mod2(fixed, h_t).any(axis=1)
        return xp.where(ok[:, None], fixed, words), ok

    @cached_property
    def _parity_matrix(self) -> np.ndarray:
        """``P`` of clause A.7.3, float32 ``(990, 32)``: ``H`` reduced with the
        identity on the right."""
        g = LinearBlockCode.from_parity_check(self.parity_check).generator
        return g[:, self.k :].astype(np.float32)


@dataclass(frozen=True)
class HardIterations:
    """Iterative hard-decision decoding of a staircase frame.

    Parameters
    ----------
    iterations : int, default 8
        Passes over the frame; each corrects the rows of every block in
        order.  Stops early once a pass changes nothing.
    """

    iterations: int = 8

    def __post_init__(self) -> None:
        it = self.iterations
        if isinstance(it, bool) or not isinstance(it, int) or it < 1:
            raise ValueError(f"iterations must be an integer >= 1, got {it!r}.")


@dataclass(frozen=True, eq=False)
class StaircaseCode(_Code):
    """The 512 x 510 staircase code of ITU-T G.709.2 Annex A, in frames.

    Parameters
    ----------
    num_blocks : int, default 4
        Staircase blocks of information per frame, each 512 x 478
        information bits; the frame adds a terminating block of 512 x 32
        parity bits.

    Notes
    -----
    Bits: the blocks in order, each row by row (510 bits per row, the
    information columns first), then the terminating block's parity row by
    row.  Rate ``478 / 510`` per block (6.7% overhead), slightly less per
    frame for the terminating parity.  The standard's error decorrelator
    and its mapping to OTU frames are not included.
    """

    num_blocks: int = 4

    def __post_init__(self) -> None:
        b = self.num_blocks
        if isinstance(b, bool) or not isinstance(b, int) or b < 1:
            raise ValueError(f"num_blocks must be an integer >= 1, got {b!r}.")

    @property
    def n(self) -> int:
        """Transmitted bits per frame."""
        return self.num_blocks * _ROWS * _COLUMNS + _ROWS * _PARITY

    @property
    def k(self) -> int:
        """Information bits per frame."""
        return self.num_blocks * _ROWS * _INFO

    def __repr__(self) -> str:
        return f"StaircaseCode(num_blocks={self.num_blocks})"

    @classmethod
    def g709_2(cls, num_blocks: int = 4) -> StaircaseCode:
        """The staircase code of ITU-T G.709.2 (also the outer code of OIF 400ZR).

        Its measured threshold (Table A.1): an input BER of 4.5e-3 gives an
        output BER of 1e-15.

        Parameters
        ----------
        num_blocks : int, default 4
            Staircase blocks of information per frame.

        Returns
        -------
        StaircaseCode
        """
        return cls(num_blocks)

    @cached_property
    def _left_columns(self) -> np.ndarray:
        """Column of ``B_(i-1)`` holding the left half of row ``j`` (``j >= 2``)."""
        return _pi_d()


_COMPONENT = _StaircaseComponent()


def _left(previous: ArrayType, columns: ArrayType) -> ArrayType:
    """Left halves of the rows of a block, ``(R, 512, 512)``: rows 0 and 1
    zero (shortened), row ``j`` the column ``Pi_d(j - 2)`` of the previous
    block."""
    _, xp, _ = dispatch(previous)
    left = xp.zeros((previous.shape[0], _ROWS, _ROWS), dtype=np.int8)
    left[:, 2:, :] = previous.transpose(0, 2, 1)[:, columns, :]
    return left


def _encode_staircase(code: StaircaseCode, bits: ArrayType) -> ArrayType:
    _, xp, _ = dispatch(bits)
    rows = bits.shape[0]
    (p,) = _device_tables(
        _COMPONENT, "parity", xp, lambda: (_COMPONENT._parity_matrix,)
    )
    (columns,) = _device_tables(code, "columns", xp, lambda: (code._left_columns,))
    info = bits.reshape(rows, code.num_blocks, _ROWS, _INFO)
    previous = xp.zeros((rows, _ROWS, _COLUMNS), dtype=np.int8)
    out = []
    for i in range(code.num_blocks + 1):
        block_info = info[:, i] if i < code.num_blocks else xp.zeros_like(info[:, 0])
        word = xp.concatenate([_left(previous, columns), block_info], axis=2)
        previous = xp.concatenate([block_info, matmul_mod2(word, p)], axis=2)
        out.append(
            previous.reshape(rows, -1)
            if i < code.num_blocks
            else previous[:, :, _INFO:]
        )
    return xp.concatenate([o.reshape(rows, -1) for o in out], axis=1)


def _decode_staircase(
    code: StaircaseCode, llrs: ArrayType, method: HardIterations
) -> _Decoded:
    _, xp, _ = dispatch(llrs)
    rows, num = llrs.shape[0], code.num_blocks
    (columns,) = _device_tables(code, "columns", xp, lambda: (code._left_columns,))
    hard = _hard(llrs)
    blocks = xp.zeros((rows, num + 1, _ROWS, _COLUMNS), dtype=np.int8)
    blocks[:, :num] = hard[:, : num * _ROWS * _COLUMNS].reshape(
        rows, num, _ROWS, _COLUMNS
    )
    blocks[:, num, :, _INFO:] = hard[:, num * _ROWS * _COLUMNS :].reshape(
        rows, _ROWS, _PARITY
    )
    zero = xp.zeros((rows, _ROWS, _COLUMNS), dtype=np.int8)
    valid = xp.zeros((rows, num + 1), dtype=bool)
    passes = 0
    for _ in range(method.iterations):
        passes += 1
        changed = False
        for i in range(num + 1):
            previous = blocks[:, i - 1] if i > 0 else zero
            word = xp.concatenate([_left(previous, columns), blocks[:, i]], axis=2)
            fixed, ok = _COMPONENT.correct(word.reshape(-1, _COMPONENT.n))
            fixed = fixed.reshape(rows, _ROWS, _COMPONENT.n)
            ok = ok.reshape(rows, _ROWS)
            valid[:, i] = ok.all(axis=1)
            changed = changed or bool((fixed != word).any())
            blocks[:, i] = fixed[:, :, _ROWS:]
            if i > 0:
                blocks[:, i - 1, :, columns] = fixed[:, 2:, :_ROWS].transpose(1, 0, 2)
            blocks[:, num, :, :_INFO] = (
                0  # the terminating block's information is known
            )
        if not changed:
            break
    bits = blocks[:, :num, :, :_INFO].reshape(rows, -1)
    return _Decoded(
        bits, None, valid.all(axis=1), xp.full(rows, passes, dtype=np.int32)
    )


@dataclass(frozen=True)
class Stages:
    """Decoders of the two stages of a :class:`Concatenated` code.

    Parameters
    ----------
    inner, outer : decoder object, optional
        ``None`` takes each code's default.
    """

    inner: Any = None
    outer: Any = None


@dataclass(frozen=True, eq=False)
class Concatenated(_Code):
    """An outer code around an inner code.

    Parameters
    ----------
    outer : code object
        Encoded first; decoded last.
    inner : code object
        A systematic code encoding the (interleaved) outer codeword bits,
        padded with zeros to whole inner codewords.
    interleaver : Interleaver, optional
        Permutation of each outer codeword (length ``outer.n``) before the
        inner encoder.

    Notes
    -----
    ``k = outer.k`` and ``n = ceil(outer.n / inner.k) inner.n``.  Decoding
    runs the inner decoder, passes its a-posteriori LLRs of the
    information bits on (hard decisions as +-1 for hard-output decoders),
    and runs the outer decoder.
    """

    outer: Any
    inner: Any
    interleaver: Interleaver | None = None

    def __post_init__(self) -> None:
        _codec(self.outer, "Concatenated()")
        _info_positions(self.inner, "Concatenated()")
        if self.interleaver is not None and (
            not isinstance(self.interleaver, Interleaver)
            or self.interleaver.length != self.outer.n
        ):
            raise ValueError(
                f"interleaver must be an Interleaver of length outer.n = {self.outer.n}."
            )

    @property
    def num_inner(self) -> int:
        """Inner codewords per outer codeword."""
        return -(-self.outer.n // self.inner.k)

    @property
    def n(self) -> int:
        """Transmitted bits per outer codeword."""
        return int(self.num_inner * self.inner.n)

    @property
    def k(self) -> int:
        """Information bits per outer codeword."""
        return int(self.outer.k)

    def __repr__(self) -> str:
        return f"Concatenated({self.outer!r}, {self.inner!r})"

    @classmethod
    def cfec(cls, num_blocks: int = 4) -> Concatenated:
        """The C-FEC of OIF 400ZR (OIF-400ZR-01.0 clause 10), at code level.

        The G.709.2 staircase code outside the double-extended
        Hamming(128, 119) (:meth:`LinearBlockCode.oif400zr`); decode the
        inner code softly, e.g. ``Stages(inner=Chase(hard=Syndrome(max_weight=1)))``.
        A fixed pseudo-random interleaver stands in for the standard's error
        decorrelator and convolutional interleaver, which spread each inner
        codeword's errors over the staircase rows.  OIF gives an input BER
        of 1.25e-2 for an output BER of 1e-15.

        Parameters
        ----------
        num_blocks : int, default 4
            Staircase blocks per frame, as for :meth:`StaircaseCode.g709_2`.

        Returns
        -------
        Concatenated
        """
        outer = StaircaseCode.g709_2(num_blocks)
        return cls(
            outer, LinearBlockCode.oif400zr(), Interleaver.random(outer.n, rng=400)
        )

    @classmethod
    def ieee_802_3dj(cls) -> Concatenated:
        """KP4 RS(544, 514) outside an extended Hamming(128, 120), at code level.

        The concatenated FEC of the IEEE 802.3dj 200G-per-lane PMDs, with a
        block interleaver (120 rows of 136 columns over three KP4
        codewords' worth of bits) in place of the standard's interleaver:
        each Hamming codeword draws bits from 120 different RS symbols.
        """
        from .algebraic import ReedSolomonCode

        kp4 = ReedSolomonCode.kp4()
        triple = _Repeat(kp4, 3)
        return cls(
            triple,
            LinearBlockCode.hamming(7, extended=True),
            Interleaver.block(120, 136),
        )

    @cached_property
    def _inner_info(self) -> np.ndarray:
        return _info_positions(self.inner, "Concatenated()")


@dataclass(frozen=True, eq=False)
class _Repeat(_Code):
    """``count`` codewords of a code side by side, as one code (private)."""

    code: Any
    count: int

    @property
    def n(self) -> int:
        """Transmitted bits of the ``count`` codewords."""
        return int(self.code.n * self.count)

    @property
    def k(self) -> int:
        """Information bits of the ``count`` codewords."""
        return int(self.code.k * self.count)

    def __repr__(self) -> str:
        return f"{self.count} x {self.code!r}"


def _encode_concatenated(code: Concatenated, bits: ArrayType) -> ArrayType:
    _, xp, _ = dispatch(bits)
    rows = bits.shape[0]
    words = _encode_blocks(code.outer, bits)
    if code.interleaver is not None:
        words = interleave(words, interleaver=code.interleaver)
    padded = xp.zeros((rows, code.num_inner * code.inner.k), dtype=np.int8)
    padded[:, : code.outer.n] = words
    inner = _encode_blocks(code.inner, padded.reshape(-1, code.inner.k))
    return inner.reshape(rows, -1)


def _decode_concatenated(
    code: Concatenated, llrs: ArrayType, method: Stages
) -> _Decoded:
    _, xp, _ = dispatch(llrs)
    rows = llrs.shape[0]
    res = _decode_blocks(code.inner, llrs.reshape(-1, code.inner.n), method.inner)
    if res.llrs is None:
        soft = (1.0 - 2.0 * res.bits).astype(xp.float32)
    else:
        (info,) = _device_tables(code, "inner_info", xp, lambda: (code._inner_info,))
        soft = res.llrs[:, info]
    soft = soft.reshape(rows, -1)[:, : code.outer.n]
    if code.interleaver is not None:
        soft = deinterleave(soft, interleaver=code.interleaver)
    out = _decode_blocks(code.outer, soft, method.outer)
    return out._replace(llrs=None)


def _encode_repeat(code: _Repeat, bits: ArrayType) -> ArrayType:
    rows = bits.shape[0]
    return _encode_blocks(code.code, bits.reshape(-1, code.code.k)).reshape(rows, -1)


def _decode_repeat(code: _Repeat, llrs: ArrayType, method: Any) -> _Decoded:
    rows = llrs.shape[0]
    res = _decode_blocks(code.code, llrs.reshape(-1, code.code.n), method)
    return _Decoded(
        res.bits.reshape(rows, -1),
        None if res.llrs is None else res.llrs.reshape(rows, -1),
        res.valid.reshape(rows, -1).all(axis=1),
        res.iterations.reshape(rows, -1).max(axis=1),
    )


_register(
    StaircaseCode,
    _Codec(
        encode=_encode_staircase,
        decode=_decode_staircase,
        default_method=lambda code: HardIterations(),
        methods=(HardIterations,),
    ),
)
_register(
    Concatenated,
    _Codec(
        encode=_encode_concatenated,
        decode=_decode_concatenated,
        default_method=lambda code: Stages(),
        methods=(Stages,),
    ),
)
_register(
    _Repeat,
    _Codec(
        encode=_encode_repeat,
        decode=_decode_repeat,
        default_method=lambda code: _resolve_method(code.code, None, "decode()"),
        methods=(object,),
    ),
)
