"""Binary linear block codes: :class:`LinearBlockCode` and its decoders.

A code is given by its generator matrix ``G`` (``k x n``): information bits
``u`` encode to ``c = u G`` (mod 2).  :meth:`LinearBlockCode.from_parity_check`
builds a systematic generator from a parity-check matrix, and
:meth:`LinearBlockCode.hamming` the Hamming codes (extended and shortened:
the inner codes of 400ZR, Hamming(128,119), and of IEEE 802.3dj,
Hamming(128,120)).

Decoders: :class:`Syndrome` (hard decisions, coset leaders from a table),
:class:`MaximumLikelihood` (soft, exhaustive over the ``2^k`` codewords,
for small codes and as a test oracle) and :class:`Chase` (soft, Chase-II
over any hard decoder, with Pyndiah's soft output; also for BCH codes).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from itertools import combinations
from math import comb
from typing import Any

import numpy as np

from ..backend import ArrayType, _device_tables, dispatch
from ..mapping.gray import _pack, _unpack_bits
from ._gf2 import matmul_mod2, null_space, row_reduce
from .base import (
    _Code,
    _Codec,
    _decode_blocks,
    _Decoded,
    _encode_blocks,
    _hard,
    _register,
)

__all__ = ["Chase", "LinearBlockCode", "MaximumLikelihood", "Syndrome"]

# Largest syndrome table of Syndrome (2^(n-k) coset leaders of n bits).
_MAX_TABLE_BITS = 1 << 26
# Most error patterns Syndrome enumerates per weight.
_MAX_PATTERNS = 1 << 22
# Largest k of MaximumLikelihood.
_MAX_ML_K = 20
# Elements of one (rows, codewords) correlation chunk of MaximumLikelihood.
_ML_CHUNK = 1 << 24
# Elements of one (rows, test patterns, n) chunk of Chase.
_CHASE_CHUNK = 1 << 23


@dataclass(frozen=True, eq=False)
class LinearBlockCode(_Code):
    """Binary linear block code ``(n, k)`` given by its generator matrix.

    Parameters
    ----------
    generator : array_like
        Generator matrix ``G``, ``(k, n)``, values 0 and 1, full row rank
        over GF(2).  Information bits ``u`` (``k`` per block) encode to
        ``u G`` mod 2.

    Notes
    -----
    Decoders find a codeword ``c`` and return the information bits ``u``
    with ``u G = c``: ``c`` at the generator's pivot columns, unmixed by a
    ``k x k`` matrix unless ``G`` is systematic there.
    """

    generator: np.ndarray

    def __post_init__(self) -> None:
        g = np.array(self.generator, copy=True)
        if g.ndim != 2 or g.shape[0] < 1 or g.shape[1] < g.shape[0]:
            raise ValueError(
                f"generator must have shape (k, n) with 1 <= k <= n, got {g.shape}."
            )
        if not np.all((g == 0) | (g == 1)):
            raise ValueError("generator must contain only 0 and 1.")
        g = g.astype(np.int8)
        if row_reduce(g)[1].size != g.shape[0]:
            raise ValueError("generator must have full row rank over GF(2).")
        g.setflags(write=False)
        object.__setattr__(self, "generator", g)

    @property
    def n(self) -> int:
        """Transmitted bits per block."""
        return int(self.generator.shape[1])

    @property
    def k(self) -> int:
        """Information bits per block."""
        return int(self.generator.shape[0])

    @cached_property
    def parity_check(self) -> np.ndarray:
        """Parity-check matrix ``H``, ``(n - k, n)``: ``G H^T = 0`` mod 2."""
        h = null_space(self.generator)[0]
        h.setflags(write=False)
        return h

    @cached_property
    def _unmix(self) -> tuple[np.ndarray, np.ndarray | None]:
        """``(pivots, A)``: ``u = c[pivots] A``; ``A = None`` for the identity."""
        k, n = self.generator.shape
        rref, pivots = row_reduce(
            np.concatenate([self.generator, np.eye(k, dtype=np.int8)], axis=1)
        )
        a = rref[:, n:]
        return pivots, None if np.array_equal(a, np.eye(k)) else a

    def __repr__(self) -> str:
        return f"LinearBlockCode(n={self.n}, k={self.k})"

    @classmethod
    def from_parity_check(cls, parity_check: object) -> LinearBlockCode:
        """The code with parity-check matrix ``H``, systematic generator.

        Parameters
        ----------
        parity_check : array_like
            ``H``, ``(m, n)``, values 0 and 1; rows may be dependent
            (``k = n - rank H``).

        Returns
        -------
        LinearBlockCode
            Generator in systematic form: the information bits occupy the
            leftmost positions that are independent of the parity checks
            (the first ``k`` when ``H = [A | I]``).
        """
        h = np.asarray(parity_check)
        if h.ndim != 2 or not np.all((h == 0) | (h == 1)):
            raise ValueError("parity_check must be a binary matrix (m, n).")
        g = null_space(h.astype(np.int8), reverse=True)[0]
        if g.shape[0] == 0:
            raise ValueError("parity_check has full column rank: the code is empty.")
        return cls(generator=g)

    @classmethod
    def hamming(
        cls, r: int, *, extended: bool = False, length: int | None = None
    ) -> LinearBlockCode:
        """Hamming code ``(2^r - 1, 2^r - 1 - r)``, minimum distance 3.

        Parameters
        ----------
        r : int
            Parity bits, ``r >= 2``.
        extended : bool, default False
            Add an overall parity bit: ``(2^r, 2^r - 1 - r)``, minimum
            distance 4 (single-error correcting, double-error detecting).
        length : int, optional
            Shorten to ``n = length`` by dropping leading information bits;
            the minimum distance does not decrease.  Hamming(128,119), the
            inner code of 400ZR, is ``hamming(8, extended=True,
            length=128)``; Hamming(128,120) of IEEE 802.3dj is
            ``hamming(7, extended=True)``.

        Returns
        -------
        LinearBlockCode
            Systematic: information bits first, then parity.

        Notes
        -----
        The parity-check columns are the nonzero ``r``-bit vectors: those of
        weight 2 or more in increasing order (information bits), then the
        identity (parity bits).  Extension appends a row of ones.  The
        standards fix their own column orders; any order gives an
        equivalent code with the same distance properties.
        """
        if isinstance(r, bool) or not isinstance(r, int | np.integer) or r < 2:
            raise ValueError(f"r must be an integer >= 2, got {r!r}.")
        values = np.arange(1, 1 << r)
        info = values[(values & (values - 1)) != 0]
        h = np.concatenate([_unpack_bits(info, r).T, np.eye(r, dtype=np.int8)], axis=1)
        if extended:
            h = np.concatenate([h, np.zeros((r, 1), dtype=np.int8)], axis=1)
            h = np.concatenate([h, np.ones((1, h.shape[1]), dtype=np.int8)], axis=0)
        if length is not None:
            parity = h.shape[0]
            if not parity < length <= h.shape[1]:
                raise ValueError(
                    f"length must be in ({parity}, {h.shape[1]}], got {length}."
                )
            h = h[:, h.shape[1] - length :]
        return cls.from_parity_check(h)

    @classmethod
    def oif400zr(cls) -> LinearBlockCode:
        """The double-extended Hamming(128, 119) inner code of OIF 400ZR (C-FEC).

        OIF-400ZR-01.0, clause 10.5: the parity-check matrix column by
        column from ``g(i) = [s_0 ... s_6, s_7, 1]`` (``s`` the bits of
        ``i``, ``s_7`` a function of ``s_0, s_1, s_2``) in the clause's
        order; systematic, ``c = [b, b P^T]``.  Minimum distance 4.

        Returns
        -------
        LinearBlockCode
        """
        i = np.arange(128)
        s = (i[None, :] >> np.arange(7)[:, None]) & 1  # s_0 ... s_6
        s0, s1, s2 = s[0].astype(bool), s[1].astype(bool), s[2].astype(bool)
        s7 = (s0 & s2) | (~s0 & ~s1 & ~s2) | (s0 & s1 & ~s2)
        g = np.concatenate([s, s7[None, :], np.ones((1, 128), dtype=s.dtype)]).astype(
            np.int8
        )
        info = [
            *range(0, 63),
            *range(64, 95),
            *range(96, 111),
            *range(112, 119),
            120,
            122,
            124,
        ]
        parity = [63, 95, 111, 119, 121, 123, 125, 126, 127]
        return cls.from_parity_check(g[:, info + parity])


@dataclass(frozen=True)
class Chase:
    """Chase-II soft decoding around a hard decoder, with Pyndiah's soft output.

    The ``test_positions`` least reliable bits of each word are flipped in
    all ``2^test_positions`` combinations; each test word goes through the
    hard decoder, and the candidate codeword of largest correlation
    ``sum_i (1 - 2 c_i) LLR_i`` is the decision.  The soft output of bit
    ``i`` is half the metric difference between the best candidates with
    ``c_i = 0`` and ``c_i = 1`` (max-log), or, when no candidate disagrees
    with the decision there, ``beta`` with the decision's sign (Pyndiah,
    1998).  For codes with a hard decoder: linear block codes and BCH codes.

    Parameters
    ----------
    test_positions : int, default 4
        Least reliable positions flipped, 1 to 12.
    hard : decoder object, optional
        The hard decoder; ``None`` takes the code's default
        (``Syndrome()``, ``BerlekampMassey()``).  For an extended Hamming
        code, ``Syndrome(max_weight=1)`` rejects test words with two errors.
    beta : float, optional
        Soft-output magnitude of bits without a competitor; ``None`` takes
        the largest competitor magnitude of the same word.
    """

    test_positions: int = 4
    hard: Any = None
    beta: float | None = None

    def __post_init__(self) -> None:
        p = self.test_positions
        if isinstance(p, bool) or not isinstance(p, int) or not 1 <= p <= 12:
            raise ValueError(
                f"test_positions must be an integer in [1, 12], got {p!r}."
            )
        if self.beta is not None and self.beta <= 0:
            raise ValueError(f"beta must be positive, got {self.beta}.")


@dataclass(frozen=True)
class Syndrome:
    """Hard-decision syndrome decoding with a coset-leader table.

    Each syndrome maps to a minimum-weight error pattern that produces it
    (the lowest positions among equal weights), which is minimum-distance
    decoding of the hard decisions.

    Parameters
    ----------
    max_weight : int, optional
        Correct error patterns of at most this weight; other syndromes
        leave the hard decisions unchanged and report ``valid=False``.
        ``None`` corrects every syndrome (complete decoding).  An extended
        Hamming code corrects one error and detects two with
        ``max_weight=1``.

    Notes
    -----
    The table holds ``2^(n-k)`` patterns of ``n`` bits, built once per code
    on the host and kept on each device.
    """

    max_weight: int | None = None

    def __post_init__(self) -> None:
        w = self.max_weight
        if w is not None and (isinstance(w, bool) or not isinstance(w, int) or w < 0):
            raise ValueError(f"max_weight must be an integer >= 0 or None, got {w!r}.")


@dataclass(frozen=True)
class MaximumLikelihood:
    """Soft maximum-likelihood decoding by exhaustive search (``k <= 20``).

    The codeword maximizing ``sum_i (1 - 2 c_i) LLR_i`` (minimum Hamming
    distance for hard decisions), over all ``2^k`` codewords in chunks of
    one GEMM each.  For small codes and as a test oracle.
    """


def _encode(code: LinearBlockCode, bits: ArrayType) -> ArrayType:
    _, xp, _ = dispatch(bits)
    (g,) = _device_tables(
        code, "generator", xp, lambda: (code.generator.astype(np.float32),)
    )
    return matmul_mod2(bits, g)


def _information(code: LinearBlockCode, words: ArrayType) -> ArrayType:
    """Information bits ``u`` of codewords ``c = u G``, ``(R, n)`` to ``(R, k)``."""
    _, xp, _ = dispatch(words)
    pivots, a = code._unmix
    (idx,) = _device_tables(code, "pivots", xp, lambda: (pivots,))
    u = words[:, idx]
    if a is None:
        return u
    (a_dev,) = _device_tables(code, "unmix", xp, lambda: (a.astype(np.float32),))
    return matmul_mod2(u, a_dev)


def _syndrome_table(
    code: LinearBlockCode, max_weight: int | None
) -> tuple[np.ndarray, ...]:
    """``(H^T, leaders, found)``: float32 ``(n, r)``, int8 ``(2^r, n)``, bool."""
    h = code.parity_check
    r, n = h.shape
    if (n << r) > _MAX_TABLE_BITS:
        raise ValueError(
            f"Syndrome(): the table of {code!r} would hold 2^{r} patterns of {n} "
            "bits; use another decoder."
        )
    column_syndromes = _pack(h.T)  # syndrome of a single error at each position
    leaders = np.zeros((1 << r, n), dtype=np.int8)
    found = np.zeros(1 << r, dtype=bool)
    found[0] = True
    top = n if max_weight is None else min(max_weight, n)
    for w in range(1, top + 1):
        if found.all():
            break
        if comb(n, w) > _MAX_PATTERNS:
            raise ValueError(
                f"Syndrome(): {comb(n, w)} patterns of weight {w} for {code!r}; "
                "pass a smaller max_weight."
            )
        pos = np.fromiter(
            (p for c in combinations(range(n), w) for p in c), dtype=np.int64
        ).reshape(-1, w)
        syn = np.bitwise_xor.reduce(column_syndromes[pos], axis=1)
        new = ~found[syn]
        values, first = np.unique(syn[new], return_index=True)
        leaders[values[:, None], pos[new][first]] = 1
        found[values] = True
    return h.T.astype(np.float32), leaders, found


def _decode_syndrome(
    code: LinearBlockCode, llrs: ArrayType, method: Syndrome
) -> _Decoded:
    _, xp, _ = dispatch(llrs)
    ht, leaders, found = _device_tables(
        code,
        f"syndrome:{method.max_weight}",
        xp,
        lambda: _syndrome_table(code, method.max_weight),
    )
    hard = _hard(llrs)
    s = _pack(matmul_mod2(hard, ht))
    words = hard ^ leaders[s]
    return _Decoded(
        _information(code, words),
        None,
        found[s],
        xp.ones(llrs.shape[0], dtype=np.int32),
    )


def _decode_ml(
    code: LinearBlockCode, llrs: ArrayType, method: MaximumLikelihood
) -> _Decoded:
    _, xp, _ = dispatch(llrs)
    k = code.k
    if k > _MAX_ML_K:
        raise ValueError(
            f"MaximumLikelihood(): k = {k} exceeds {_MAX_ML_K} (2^k codewords)."
        )
    (g,) = _device_tables(
        code, "generator", xp, lambda: (code.generator.astype(np.float32),)
    )
    rows = llrs.shape[0]
    num_words = 1 << k
    words_chunk = min(num_words, max(1, _ML_CHUNK // code.n))
    rows_chunk = max(1, _ML_CHUNK // words_chunk)
    best = xp.empty(rows, dtype=xp.int64)
    for r0 in range(0, rows, rows_chunk):
        x = llrs[r0 : r0 + rows_chunk]
        best_metric = xp.full(x.shape[0], -np.inf, dtype=xp.float32)
        best_index = xp.zeros(x.shape[0], dtype=xp.int64)
        for w0 in range(0, num_words, words_chunk):
            u = _unpack_bits(xp.arange(w0, min(w0 + words_chunk, num_words)), k)
            signs = 1.0 - 2.0 * matmul_mod2(u, g).astype(xp.float32)
            metric = x @ signs.T
            arg = xp.argmax(metric, axis=1)
            value = xp.take_along_axis(metric, arg[:, None], axis=1)[:, 0]
            better = value > best_metric
            best_metric = xp.where(better, value, best_metric)
            best_index = xp.where(better, arg + w0, best_index)
        best[r0 : r0 + rows_chunk] = best_index
    return _Decoded(
        _unpack_bits(best, k),
        None,
        xp.ones(rows, dtype=bool),
        xp.ones(rows, dtype=np.int32),
    )


def _decode_chase(code: Any, llrs: ArrayType, method: Chase) -> _Decoded:
    """Chase-II over ``code``'s hard decoder; shared with the BCH codes."""
    _, xp, _ = dispatch(llrs)
    rows, n = llrs.shape
    p = min(method.test_positions, n)
    tests = 1 << p
    patterns = _unpack_bits(xp.arange(tests), p).astype(bool)  # (T, p)
    chunk = max(1, _CHASE_CHUNK // (tests * n * p))
    bits_out = xp.empty((rows, code.k), dtype=np.int8)
    llrs_out = xp.empty((rows, n), dtype=xp.float32)
    valid_out = xp.empty(rows, dtype=bool)
    positions = xp.arange(n)
    for r0 in range(0, rows, chunk):
        x = llrs[r0 : r0 + chunk]
        r = x.shape[0]
        least = xp.argsort(xp.abs(x), axis=1)[:, :p]  # (r, p)
        flips = (
            (least[:, None, :, None] == positions) & patterns[None, :, :, None]
        ).any(axis=2)
        test_words = _hard(x)[:, None, :] ^ flips.astype(np.int8)  # (r, T, n)
        res = _decode_blocks(
            code,
            (1.0 - 2.0 * test_words.reshape(-1, n)).astype(xp.float32),
            method.hard,
        )
        words = _encode_blocks(code, res.bits).reshape(r, tests, n)
        valid = res.valid.reshape(r, tests)
        metric = xp.einsum("rtn,rn->rt", (1.0 - 2.0 * words).astype(xp.float32), x)
        metric = xp.where(valid, metric, -xp.inf)
        best = xp.argmax(metric, axis=1)
        pick = xp.arange(r)
        bits_out[r0 : r0 + r] = res.bits.reshape(r, tests, -1)[pick, best]
        valid_out[r0 : r0 + r] = valid.any(axis=1)
        # Max-log soft output from the candidates of each value of each bit.
        m = metric[:, :, None]
        m0 = xp.where(words == 0, m, -xp.inf).max(axis=1)
        m1 = xp.where(words == 1, m, -xp.inf).max(axis=1)
        with np.errstate(invalid="ignore"):
            soft = 0.5 * (m0 - m1)
        decided = words[pick, best]
        known = xp.isfinite(soft)
        if method.beta is None:
            beta = xp.where(known, xp.abs(soft), 0).max(axis=1, keepdims=True)
            beta = xp.where(beta > 0, beta, xp.abs(x).max(axis=1, keepdims=True))
        else:
            beta = xp.float32(method.beta)
        fallback = beta * (1.0 - 2.0 * decided)
        llrs_out[r0 : r0 + r] = xp.where(known, soft, fallback).astype(xp.float32)
    return _Decoded(bits_out, llrs_out, valid_out, xp.ones(rows, dtype=np.int32))


def _decode(code: LinearBlockCode, llrs: ArrayType, method: object) -> _Decoded:
    if isinstance(method, Chase):
        return _decode_chase(code, llrs, method)
    if isinstance(method, MaximumLikelihood):
        return _decode_ml(code, llrs, method)
    assert isinstance(method, Syndrome)
    return _decode_syndrome(code, llrs, method)


_register(
    LinearBlockCode,
    _Codec(
        encode=_encode,
        decode=_decode,
        default_method=lambda code: Syndrome(),
        methods=(Syndrome, MaximumLikelihood, Chase),
        info_positions=lambda code: code._unmix[0] if code._unmix[1] is None else None,
    ),
)
