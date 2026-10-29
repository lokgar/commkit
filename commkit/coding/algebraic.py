"""Algebraic block codes: :class:`BCHCode`, :class:`ReedSolomonCode` and :class:`BerlekampMassey`.

Both families are cyclic codes over GF(2^m), shortened by leaving out
leading information symbols, and decoded by one hard-decision decoder
(:class:`BerlekampMassey`): syndromes, the error locator by the
Berlekamp-Massey algorithm, its roots by a Chien search, and for
Reed-Solomon the error values by Forney's formula.  Up to ``t`` symbol
errors per codeword are corrected; a codeword whose locator does not have
as many roots among the code's positions as its degree is reported
``valid=False`` and left as received.

The systematic encoder and the syndromes are GF(2)-linear in bits, so
each is one :func:`~commkit.coding._gf2.matmul_mod2` with a matrix built
once per code.  From the syndromes, each codeword is decoded on its own
(Berlekamp-Massey, a Chien search that stops at the locator's degree,
Forney at the roots): a Numba kernel parallel over codewords on the CPU,
a CUDA kernel with one thread per codeword on the GPU, the same steps on
both.

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

from .. import _cuda
from ..backend import ArrayType, _device_tables, dispatch
from ..mapping.gray import _pack
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
    (``b``) and ``binary``.
    """

    field: Any
    t: Any
    first_root: Any
    binary: Any
    n: Any

    @property
    def _positions(self) -> int:
        """Transmitted code symbols per codeword (bits for a binary code)."""
        return self.n if self.binary else self.n // self.field.m

    @cached_property
    def _locator_logs(self) -> np.ndarray:
        """``log_alpha`` of each position's locator ``X_p``.

        For a cyclic code the polynomial degree of each transmitted symbol,
        ``n - 1 ... 0``; codes with another position map override it.
        """
        return np.arange(self._positions - 1, -1, -1)

    @cached_property
    def _syndrome_matrix(self) -> np.ndarray:
        """float32 GF(2) matrix from the transmitted bits to the bits of the
        syndromes ``S_j = c(alpha^(b + j))``, ``j < 2t``."""
        f, t, b = self.field, self.t, self.first_root
        roots = (b + np.arange(2 * t))[:, None] * self._locator_logs[None, :]  # (2t, n)
        syn = f.evaluation_matrix(roots.T)  # rows: symbol p's bits; cols: S_j bits
        if self.binary:
            syn = syn[self.field.m - 1 :: self.field.m]  # a bit is the element 0 or 1
        return syn.astype(np.float32)


@dataclass(frozen=True, eq=False)
class ReedSolomonCode(_Algebraic):
    """Reed-Solomon code over GF(2^m), correcting ``t`` symbol errors.

    The standards name an RS code by its sizes in *code symbols*, elements
    of GF(2^m) of ``m`` bits each (not constellation symbols): RS(544, 514)
    over GF(2^10), KP4, has codewords of 544 code symbols, 514 of them
    information, and corrects ``t = (544 - 514) / 2 = 15`` wrong code
    symbols, whatever the bit errors inside each.  Like every code object,
    :attr:`n` and :attr:`k` count *bits*: 5440 and 5140 for KP4.  The code
    symbol counts are the fields ``symbols`` and ``info_symbols``.

    Parameters
    ----------
    symbols, info_symbols : int
        Code symbols per codeword and information code symbols (the two
        numbers of RS(544, 514)), ``symbols - info_symbols`` even,
        ``symbols <= 2^m - 1`` (shorter codes are shortened).
    m : int
        Bits per code symbol.
    primitive : int, optional
        Field polynomial with its ``x^m`` term (``0x11D`` for
        ``x^8 + x^4 + x^3 + x^2 + 1``); ``None`` takes the common primitive
        polynomial of degree ``m``.
    first_root : int, default 0
        ``b`` of the generator ``g(x) = prod_{j<2t} (x - alpha^(b + j))``.

    Notes
    -----
    ``n = symbols * m`` transmitted and ``k = info_symbols * m``
    information bits per codeword, each code symbol MSB first; a codeword
    is one block of ``generate(..., num_blocks=)``.  KP4 on PAM4 (2 bits
    per constellation symbol) fills 2720 constellation symbols per
    codeword.
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
        """Transmitted bits per codeword, ``symbols * m``."""
        return self.symbols * self.m

    @property
    def k(self) -> int:
        """Information bits per codeword, ``info_symbols * m``."""
        return self.info_symbols * self.m

    @property
    def t(self) -> int:
        """Correctable code-symbol errors per codeword."""
        return (self.symbols - self.info_symbols) // 2

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

    Named by the field and the errors it corrects; the code's sizes follow:
    ``n = 2^m - 1`` bits (or ``length`` when shortened) and ``k = n - deg
    g``, at most ``m t`` parity bits.  ``BCHCode(6, 3)`` is BCH(63, 45),
    ``BCHCode(10, 8)`` BCH(1023, 943): read :attr:`n` and :attr:`k`.

    Parameters
    ----------
    m : int
        Field degree: the full code length is ``2^m - 1`` bits.
    t : int
        Correctable bit errors per codeword.
    length : int, optional
        Shorten to ``n = length`` bits by leaving out leading information
        bits (``k`` drops by the same number).
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


def _decode(code: Any, llrs: ArrayType, method: BerlekampMassey) -> _Decoded:
    _, xp, _ = dispatch(llrs)
    words, valid = _correct(code, _hard(llrs))
    return _Decoded(
        words[:, : code.k], None, valid, xp.ones(llrs.shape[0], dtype=np.int32)
    )


def _correct(code: Any, hard: ArrayType) -> tuple[ArrayType, ArrayType]:
    """Bounded-distance correction of hard words ``(R, n bits)``.

    Returns the corrected words (the received ones where decoding failed)
    and the per-word success.  The syndromes are one GEMM; the rest runs
    per word, in Numba on the CPU and in the CUDA kernel
    ``algebraic_correct`` on the GPU.
    """
    _, xp, _ = dispatch(hard)
    f, t, m = code.field, code.t, code.field.m
    rows = hard.shape[0]
    bits = 1 if code.binary else m  # per position
    (syn_t,) = _device_tables(code, "decoder", xp, lambda: (code._syndrome_matrix,))
    # Syndromes as field elements, (R, 2t).
    s = _pack(matmul_mod2(hard, syn_t).reshape(rows, 2 * t, m))
    if xp is np:
        words = np.array(hard, dtype=np.int8, order="C")
        valid = np.empty(rows, dtype=np.bool_)
        exp, log = f._tables
        _get_numba_algebraic()(
            s, exp, log, code._locator_logs, t, code.first_root, bits, code.binary,
            words, valid,
        )  # fmt: skip
        return words, valid
    words = xp.array(hard, dtype=np.int8, order="C")
    exp, log = f.tables(xp)
    (logs,) = _device_tables(
        code, "locators", xp, lambda: (np.asarray(code._locator_logs, np.int64),)
    )
    launch = _cuda.get_kernel("algebraic_correct")
    valid = launch(
        s, exp, log, logs, words, t=t, first_root=code.first_root, m=bits,
        binary=code.binary,
    )  # fmt: skip
    return words, valid


def _get_numba_algebraic() -> Any:
    """Numba decoder from the syndromes for the CPU, parallel over words.

    Per word: Berlekamp-Massey, a Chien search over the positions that
    stops at the locator's degree, and for Reed-Solomon Forney's formula
    at the roots; the word is corrected in place, or left as received
    where decoding fails.  The CPU reference of ``algebraic.cu``.
    """
    from . import _numba_loops

    return _numba_loops.algebraic_correct


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
