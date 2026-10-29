"""Algebraic block codes: :class:`BCHCode`, :class:`ReedSolomonCode` and :class:`BerlekampMassey`.

Both families are cyclic codes over GF(2^m), shortened by leaving out
leading information symbols, and decoded by one hard-decision decoder
(:class:`BerlekampMassey`): syndromes, the error locator by the
Berlekamp-Massey algorithm, its roots by a Chien search, and for
Reed-Solomon the error values by Forney's formula.  Up to ``t`` symbol
errors per codeword are corrected; a codeword whose locator does not have
as many roots among the code's positions as its degree is reported
``valid=False`` and left as received.

Everything except Berlekamp-Massey is GF(2)-linear in bits, so it is one
:func:`~commkit.coding._gf2.matmul_mod2` with a matrix built once per
code: the systematic encoder, the syndromes, and the evaluation of the
locator, its derivative and the error evaluator at every position.
Berlekamp-Massey runs its ``2t`` steps in lockstep over all codewords,
small array operations on either device.

Bits: a Reed-Solomon symbol is ``m`` bits, MSB first, the first transmitted
symbol the highest-degree coefficient; information first, then parity.
Presets: :meth:`ReedSolomonCode.g709` (ITU-T G.709 GFEC, RS(255,239)),
:meth:`~ReedSolomonCode.kr4` and :meth:`~ReedSolomonCode.kp4` (IEEE 802.3
clauses 91 and 119, RS(528,514) and RS(544,514) over GF(2^10)),
:meth:`~ReedSolomonCode.epon10g` (IEEE 802.3 clause 76, RS(255,223)) and
:meth:`~ReedSolomonCode.xgspon` (ITU-T G.9807.1, RS(248,216)).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np

from ..backend import ArrayType, _device_tables, dispatch
from ..mapping.gray import _pack, _unpack_bits
from ._galois import PRIMITIVE, _Field
from ._gf2 import matmul_mod2
from .base import _Code, _Codec, _Decoded, _hard, _register, _resolve_method
from .crc import _remainders
from .linear import Chase, _decode_chase, _HardWords

__all__ = ["BCHCode", "BerlekampMassey", "ReedSolomonCode"]


@dataclass(frozen=True)
class BerlekampMassey:
    """Hard-decision algebraic decoding: Berlekamp-Massey, Chien, Forney.

    Corrects up to ``t`` symbol errors (bit errors for BCH) per codeword.
    """


class _Algebraic(_Code):
    """What the shared decoder needs from a code.

    Subclasses provide ``field`` (a :class:`_Field`), ``t``, ``first_root``
    (``b``), ``length`` (transmitted symbols, bits for BCH) and ``binary``.
    """

    field: Any
    t: Any
    first_root: Any
    length: Any
    binary: Any

    @cached_property
    def _locator_logs(self) -> np.ndarray:
        """``log_alpha`` of each position's locator ``X_p``.

        For a cyclic code the polynomial degree of each transmitted symbol,
        ``n - 1 ... 0``; codes with another position map override it.
        """
        return np.arange(self.length - 1, -1, -1)

    @cached_property
    def _decoder_tables(self) -> tuple[np.ndarray, ...]:
        """float32 GF(2) matrices: syndromes, locator and evaluator evaluation.

        ``syndromes``: transmitted bits to the bits of ``S_j = c(alpha^(b +
        j))``, ``j < 2t``.  ``evaluate``: bits of ``t + 1`` coefficients
        to the bits of the polynomial at ``X_p^-1 = alpha^-deg(p)`` for
        every position ``p``.
        """
        f, t, b = self.field, self.t, self.first_root
        roots = (b + np.arange(2 * t))[:, None] * self._locator_logs[None, :]  # (2t, n)
        syn = f.evaluation_matrix(roots.T)  # rows: symbol p's bits; cols: S_j bits
        if self.binary:
            syn = syn[self.field.m - 1 :: self.field.m]  # a bit is the element 0 or 1
        powers = -np.arange(t + 1)[:, None] * self._locator_logs[None, :]
        evaluate = f.evaluation_matrix(powers)
        return syn.astype(np.float32), evaluate.astype(np.float32)

    @cached_property
    def _forney_tables(self) -> tuple[np.ndarray, np.ndarray]:
        """Field elements per position: ``X_p^(1-b)`` ``(n,)`` and ``X_p^-k``,
        ``k < t``, ``(n, t)``."""
        exp, q1 = self.field._tables[0], self.field.order - 1
        logs = self._locator_logs
        xinv = exp[(-np.arange(self.t)[None, :] * logs[:, None]) % q1]
        return exp[((1 - self.first_root) * logs) % q1], xinv


@dataclass(frozen=True, eq=False)
class ReedSolomonCode(_Algebraic):
    """Reed-Solomon code ``(n, k)`` over GF(2^m), ``t = (n - k) / 2``.

    Parameters
    ----------
    symbols, info_symbols : int
        Symbols per codeword and information symbols, ``symbols -
        info_symbols`` even, ``symbols <= 2^m - 1`` (shorter codes are
        shortened).
    m : int
        Bits per symbol.
    primitive : int, optional
        Field polynomial with its ``x^m`` term (``0x11D`` for
        ``x^8 + x^4 + x^3 + x^2 + 1``); ``None`` takes the common primitive
        polynomial of degree ``m``.
    first_root : int, default 0
        ``b`` of the generator ``g(x) = prod_{j<2t} (x - alpha^(b + j))``.

    Notes
    -----
    In bits, ``n m`` transmitted and ``k m`` information bits per codeword.
    """

    symbols: int
    info_symbols: int
    m: int
    primitive: int | None = None
    first_root: int = 0

    def __post_init__(self) -> None:
        n, k, m = self.symbols, self.info_symbols, self.m
        prim = PRIMITIVE.get(m) if self.primitive is None else self.primitive
        if prim is None:
            raise ValueError(f"no default primitive polynomial for m = {m}.")
        object.__setattr__(self, "primitive", prim)
        if not 1 <= k < n <= (1 << m) - 1 or (n - k) % 2:
            raise ValueError(
                f"need 1 <= k < n <= 2^m - 1 with n - k even, got n={n}, k={k}, m={m}."
            )

    @property
    def n(self) -> int:
        """Transmitted bits per codeword, ``n m``."""
        return self.symbols * self.m

    @property
    def k(self) -> int:
        """Information bits per codeword, ``k m``."""
        return self.info_symbols * self.m

    @property
    def t(self) -> int:
        """Correctable symbol errors per codeword."""
        return (self.symbols - self.info_symbols) // 2

    @property
    def length(self) -> int:
        """Transmitted symbols per codeword."""
        return self.symbols

    @property
    def binary(self) -> bool:
        """False: the symbols are elements of GF(2^m)."""
        return False

    @cached_property
    def field(self) -> _Field:
        """The symbol field GF(2^m)."""
        assert self.primitive is not None
        return _Field(self.m, self.primitive)

    def __repr__(self) -> str:
        return f"ReedSolomonCode({self.symbols}, {self.info_symbols}, m={self.m})"

    @cached_property
    def generator_polynomial(self) -> np.ndarray:
        """``g(x)`` coefficients, highest degree first, ``(2t + 1,)``."""
        f = self.field
        g = [1]
        for j in range(2 * self.t):
            root = f.alpha(self.first_root + j)
            # g(x) (x + root)
            g = [
                a ^ f.mul_scalar(root, c) for a, c in zip([*g, 0], [0, *g], strict=True)
            ]
        out = np.asarray(g, dtype=np.int64)
        out.setflags(write=False)
        return out

    @cached_property
    def _parity_matrix(self) -> np.ndarray:
        """float32 ``(k m, 2t m)``: information bits to parity bits."""
        f, k, r = self.field, self.info_symbols, 2 * self.t
        g = self.generator_polynomial[1:]  # x^2t = sum g_i x^(2t - 1 - i) mod g
        rows = np.zeros((k, r), dtype=np.int64)
        v = g.copy()  # x^2t mod g
        for i in range(k - 1, -1, -1):  # symbol i has degree 2t + k - 1 - i
            rows[i] = v
            lead = int(v[0])
            v = np.append(v[1:], 0) ^ np.array([f.mul_scalar(lead, int(c)) for c in g])
        # Row (i, bit) = the parity of the element 2^(m-1-bit) at symbol i.
        basis = 1 << np.arange(self.m - 1, -1, -1)
        exp, log = f._tables
        nz = rows != 0
        logs = np.where(nz, log[rows], 0)
        prod = np.where(
            nz[:, None, :], exp[logs[:, None, :] + log[basis][None, :, None]], 0
        )  # (k, m, r)
        return f.bits(prod).reshape(k * self.m, r * self.m).astype(np.float32)

    # -- presets -------------------------------------------------------------

    @classmethod
    def g709(cls) -> ReedSolomonCode:
        """RS(255, 239) over GF(2^8), the GFEC of ITU-T G.709 Annex A (t = 8)."""
        return cls(255, 239, m=8, primitive=0x11D)

    @classmethod
    def kr4(cls) -> ReedSolomonCode:
        """RS(528, 514) over GF(2^10), IEEE 802.3 clause 91 (100GBASE-KR4, t = 7)."""
        return cls(528, 514, m=10, primitive=0x409)

    @classmethod
    def kp4(cls) -> ReedSolomonCode:
        """RS(544, 514) over GF(2^10), IEEE 802.3 clauses 91 and 119 (KP4, t = 15).

        The FEC of 200G, 400G and 800G Ethernet PMDs; its pre-FEC BER
        threshold is about 2.4e-4.
        """
        return cls(544, 514, m=10, primitive=0x409)

    @classmethod
    def epon10g(cls) -> ReedSolomonCode:
        """RS(255, 223) over GF(2^8), IEEE 802.3 clause 76 (10G-EPON, t = 16)."""
        return cls(255, 223, m=8, primitive=0x11D)

    @classmethod
    def xgspon(cls) -> ReedSolomonCode:
        """RS(248, 216), ITU-T G.9807.1 (XGS-PON): RS(255, 223) shortened by 7."""
        return cls(248, 216, m=8, primitive=0x11D)


@dataclass(frozen=True, eq=False)
class BCHCode(_Algebraic):
    """Binary narrow-sense BCH code over GF(2^m) correcting ``t`` errors.

    Parameters
    ----------
    m : int
        Field degree: the full code length is ``2^m - 1``.
    t : int
        Correctable bit errors per codeword.
    length : int, optional
        Shorten to ``n = length`` by leaving out leading information bits.
    primitive : int, optional
        Field polynomial with its ``x^m`` term; ``None`` takes the common
        one of degree ``m``.

    Notes
    -----
    The generator is the least common multiple of the minimal polynomials
    of ``alpha ... alpha^(2t)``; ``k = n - deg g``.  Systematic: information
    bits first.
    """

    m: int
    t: int
    length: int | None = None
    primitive: int | None = None

    def __post_init__(self) -> None:
        prim = PRIMITIVE.get(self.m) if self.primitive is None else self.primitive
        if prim is None:
            raise ValueError(f"no default primitive polynomial for m = {self.m}.")
        object.__setattr__(self, "primitive", prim)
        full = (1 << self.m) - 1
        if not isinstance(self.t, int) or self.t < 1:
            raise ValueError(f"t must be an integer >= 1, got {self.t!r}.")
        if self.length is None:
            object.__setattr__(self, "length", full)
        degree = self._generator.bit_length() - 1
        if degree >= full:
            raise ValueError(
                f"t = {self.t} leaves no information bits for m = {self.m}."
            )
        assert self.length is not None
        if not degree < self.length <= full:
            raise ValueError(
                f"length must be in ({degree}, {full}], got {self.length}."
            )

    @property
    def n(self) -> int:
        """Transmitted bits per codeword."""
        assert self.length is not None
        return self.length

    @property
    def k(self) -> int:
        """Information bits per codeword."""
        return self.n - (self._generator.bit_length() - 1)

    @property
    def first_root(self) -> int:
        """``b = 1``: narrow-sense, roots ``alpha^1 .. alpha^(2t)``."""
        return 1

    @property
    def binary(self) -> bool:
        """True: the symbols are bits."""
        return True

    @cached_property
    def field(self) -> _Field:
        """The locator field GF(2^m)."""
        assert self.primitive is not None
        return _Field(self.m, self.primitive)

    def __repr__(self) -> str:
        return f"BCHCode(n={self.n}, k={self.k}, t={self.t})"

    @cached_property
    def _generator(self) -> int:
        """``g(x)`` as a bit mask (bit ``i`` the coefficient of ``x^i``)."""
        f = self.field
        q1 = f.order - 1
        seen: set[int] = set()
        g = 1
        for j in range(1, 2 * self.t + 1):
            if j % q1 in seen:
                continue
            # Minimal polynomial of alpha^j: product over its conjugates.
            coset, e = [], j % q1
            while e not in coset:
                coset.append(e)
                e = (2 * e) % q1
            seen.update(coset)
            poly = [1]  # GF(2^m) coefficients, lowest degree first
            for e in coset:
                root = f.alpha(e)
                poly = [
                    a ^ f.mul_scalar(root, c)
                    for a, c in zip([0, *poly], [*poly, 0], strict=True)
                ]
            minimal = sum(
                int(c) << i for i, c in enumerate(poly)
            )  # coefficients are 0/1
            g = _clmul(g, minimal)
        return g

    @cached_property
    def generator_polynomial(self) -> np.ndarray:
        """``g(x)`` coefficients, highest degree first, int8."""
        degree = self._generator.bit_length() - 1
        out = np.array(
            [(self._generator >> (degree - i)) & 1 for i in range(degree + 1)], np.int8
        )
        out.setflags(write=False)
        return out

    @cached_property
    def _parity_matrix(self) -> np.ndarray:
        degree = self._generator.bit_length() - 1
        g, _ = _remainders(self._generator ^ (1 << degree), degree, self.k)
        return g.astype(np.float32)


def _clmul(a: int, b: int) -> int:
    """Product of two GF(2) polynomials given as bit masks."""
    out = 0
    while b:
        if b & 1:
            out ^= a
        a <<= 1
        b >>= 1
    return out


def _encode(code: Any, bits: ArrayType) -> ArrayType:
    _, xp, _ = dispatch(bits)
    (p,) = _device_tables(code, "parity", xp, lambda: (code._parity_matrix,))
    return xp.concatenate([bits, matmul_mod2(bits, p)], axis=1)


_XOR_KERNEL: dict[str, Any] = {}


def _xor_reduce(x: ArrayType, axis: int) -> ArrayType:
    """XOR of field elements along ``axis`` (the field sum)."""
    xp = dispatch(x)[1]
    if xp is np:
        return np.bitwise_xor.reduce(x, axis=axis)
    # CuPy's bitwise_xor has no reduce.
    if "k" not in _XOR_KERNEL:
        _XOR_KERNEL["k"] = xp.ReductionKernel(
            "T x", "T y", "x", "a ^ b", "y = a", "0", "commkit_xor_reduce"
        )
    return _XOR_KERNEL["k"](x, axis=axis)


def _decode(code: Any, llrs: ArrayType, method: BerlekampMassey) -> _Decoded:
    _, xp, _ = dispatch(llrs)
    words, valid = _correct(code, _hard(llrs))
    return _Decoded(
        words[:, : code.k], None, valid, xp.ones(llrs.shape[0], dtype=np.int32)
    )


def _correct(code: Any, hard: ArrayType) -> tuple[ArrayType, ArrayType]:
    """Bounded-distance correction of hard words ``(R, n bits)``.

    Returns the corrected words (the received ones where decoding failed)
    and the per-word success.
    """
    _, xp, _ = dispatch(hard)
    f, t, m = code.field, code.t, code.field.m
    rows, n = hard.shape[0], code.length
    syn_t, evaluate = _device_tables(code, "decoder", xp, lambda: code._decoder_tables)
    # Syndromes as field elements, (R, 2t).
    s = _pack(matmul_mod2(hard, syn_t).reshape(rows, 2 * t, m))
    locator, length = _berlekamp_massey(f, s, t)  # (R, 2t + 1), lowest degree first
    degree = (locator != 0).astype(xp.int64)
    degree = (degree * xp.arange(2 * t + 1)).max(axis=1)

    # The locator at every X_p^-1: a root where all m bits of the value are 0.
    # Only Lambda_0 .. Lambda_t: a locator of higher degree fails anyway.
    bits = _unpack_bits(locator[:, : t + 1].reshape(-1), m).reshape(rows, -1)
    roots = ~matmul_mod2(bits, evaluate).reshape(rows, n, m).any(axis=2)
    # A locator of the length Berlekamp-Massey found (its top coefficients
    # may vanish beyond t errors), with that many roots among the positions.
    valid = (roots.sum(axis=1) == degree) & (degree == length) & (length <= t)
    errors = roots & valid[:, None]
    if code.binary:
        words = hard ^ errors.astype(np.int8)
    else:
        # Forney: e_p = X_p^(1-b) Omega(X_p^-1) / Lambda'(X_p^-1), with
        # Omega = S Lambda mod x^2t (characteristic 2: no signs).  For a
        # decodable word both have degree < t: Omega by the key equation,
        # Lambda' = sum over odd j of Lambda_j x^(j-1).
        omega = _xor_reduce(
            f.mul(locator[:, None, :t], _toeplitz(s[:, :t], xp)), axis=2
        )
        derivative = xp.zeros_like(omega)
        derivative[:, ::2] = locator[:, 1 : t + 1 : 2]
        xpow, xinv = _device_tables(code, "forney", xp, lambda: code._forney_tables)
        # Evaluated only at the at most t error positions of each row, (R, t).
        pos = xp.argsort(~errors, axis=1)[:, :t]
        hit = xp.take_along_axis(errors, pos, axis=1)
        powers = xinv[pos]  # (R, t, t): X_p^-k at those positions
        values = f.mul(
            xpow[pos],
            f.div(
                _xor_reduce(f.mul(omega[:, None, :], powers), axis=2),
                _xor_reduce(f.mul(derivative[:, None, :], powers), axis=2),
            ),
        )
        values = xp.where(hit, values, 0)
        valid = valid & ~xp.any(hit & (values == 0), axis=1)
        values = xp.where(valid[:, None], values, 0)
        words = hard.copy()
        flips = _unpack_bits(values.reshape(-1), m).reshape(rows, t, m)
        words.reshape(rows, n, m)[xp.arange(rows)[:, None], pos] ^= flips
    return words, valid


def _toeplitz(s: ArrayType, xp: Any) -> ArrayType:
    """``T[r, k, i] = S_(k - i)`` for ``i <= k`` (else 0): Omega_k = sum_i L_i T."""
    rows, n2 = s.shape
    k = xp.arange(n2)[:, None]
    i = xp.arange(n2)[None, :]
    idx = k - i
    return xp.where(idx >= 0, s[:, xp.clip(idx, 0, None)], 0)


def _berlekamp_massey(f: _Field, s: ArrayType, t: int) -> tuple[ArrayType, ArrayType]:
    """Error locators of all rows in lockstep, ``(R, 2t + 1)``, lowest degree
    first, and their linear complexities ``L``, ``(R,)``."""
    xp = dispatch(s)[1]
    rows = s.shape[0]
    width = 2 * t + 1
    locator = xp.zeros((rows, width), dtype=xp.int64)
    locator[:, 0] = 1
    previous = locator.copy()
    length = xp.zeros(rows, dtype=xp.int64)
    shift = xp.ones(rows, dtype=xp.int64)
    last = xp.ones(rows, dtype=xp.int64)
    cols = xp.arange(width)
    for r in range(2 * t):
        # Discrepancy d = sum_{i <= r} Lambda_i S_(r - i).
        terms = f.mul(locator[:, : r + 1], s[:, r::-1])
        d = _xor_reduce(terms, axis=1)
        nonzero = d != 0
        grow = nonzero & (2 * length <= r)
        src = cols[None, :] - shift[:, None]
        shifted = xp.where(
            src >= 0, xp.take_along_axis(previous, xp.clip(src, 0, None), axis=1), 0
        )
        updated = locator ^ f.mul(f.div(d, last)[:, None], shifted)
        previous = xp.where(grow[:, None], locator, previous)
        locator = xp.where(nonzero[:, None], updated, locator)
        last = xp.where(grow, d, last)
        length = xp.where(grow, r + 1 - length, length)
        shift = xp.where(grow, 1, shift + 1)
    return locator, length


def _decode_bch(code: BCHCode, llrs: ArrayType, method: Any) -> _Decoded:
    if isinstance(method, Chase):
        hard = _resolve_method(code, method.hard, "decode()")
        if not isinstance(hard, BerlekampMassey):
            return _decode_chase(code, llrs, method)
        return _decode_chase(
            code,
            llrs,
            method,
            _HardWords(lambda h: _correct(code, h), lambda w: w[:, : code.k]),
        )
    return _decode(code, llrs, method)


for _cls, _dec, _methods in (
    (ReedSolomonCode, _decode, (BerlekampMassey,)),
    (BCHCode, _decode_bch, (BerlekampMassey, Chase)),
):
    _register(
        _cls,
        _Codec(
            encode=_encode,
            decode=_dec,
            default_method=lambda code: BerlekampMassey(),
            methods=_methods,
            info_positions=lambda code: np.arange(code.k),
        ),
    )
