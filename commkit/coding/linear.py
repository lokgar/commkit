"""Binary linear block codes: :class:`LinearBlockCode` and its decoders.

A code is given by its generator matrix ``G`` (``k x n``): information bits
``u`` encode to ``c = u G`` (mod 2).  :meth:`LinearBlockCode.from_parity_check`
builds a systematic generator from a parity-check matrix, and
:meth:`LinearBlockCode.hamming` the Hamming codes (extended and shortened:
the inner codes of 400ZR, Hamming(128,119), and of IEEE 802.3dj,
Hamming(128,120)).

Decoders: :class:`Syndrome` (hard decisions, coset leaders from a table)
and :class:`MaximumLikelihood` (soft, exhaustive over the ``2^k``
codewords, for small codes and as a test oracle).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from itertools import combinations
from math import comb

import numpy as np

from ..backend import ArrayType, _device_tables, dispatch
from ..mapping.gray import _pack, _unpack_bits
from ._gf2 import matmul_mod2, null_space, row_reduce
from .base import _Code, _Codec, _Decoded, _hard, _register

__all__ = ["LinearBlockCode", "MaximumLikelihood", "Syndrome"]

# Largest syndrome table of Syndrome (2^(n-k) coset leaders of n bits).
_MAX_TABLE_BITS = 1 << 26
# Most error patterns Syndrome enumerates per weight.
_MAX_PATTERNS = 1 << 22
# Largest k of MaximumLikelihood.
_MAX_ML_K = 20
# Elements of one (rows, codewords) correlation chunk of MaximumLikelihood.
_ML_CHUNK = 1 << 24


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


def _decode(code: LinearBlockCode, llrs: ArrayType, method: object) -> _Decoded:
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
        methods=(Syndrome, MaximumLikelihood),
    ),
)
