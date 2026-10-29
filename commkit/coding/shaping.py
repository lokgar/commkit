"""Distribution matching: :class:`CCDM` and probabilistic amplitude shaping, :class:`PAS`.

A distribution matcher turns uniform information bits into symbols with a
target distribution, invertibly.  Both classes are code objects, so
:func:`~commkit.coding.encode`, :func:`~commkit.coding.decode` and
``generate(..., code=)`` take them like any code.  The pmf design itself
(Maxwell-Boltzmann priors, ``Constellation.shaped``) is in
:mod:`commkit.mapping.shaping`.

:class:`CCDM` (constant-composition distribution matching, Schulte and
Böcherer, 2016) maps ``k`` bits to ``block_length`` points whose counts are
fixed: the composition that best approximates the constellation's pmf
(Böcherer and Geiger, 2016).  Each block of bits selects a point of
``[0, 1)``; the sequences of that composition partition ``[0, 1)`` by
arithmetic coding over the remaining counts, and the matcher outputs the
sequence whose interval holds the point.  The coder is a 64-bit range coder:
the interval is split in proportion to the remaining counts with floor
rounding, the remainder going to the most frequent point, so matcher and
dematcher compute the same partition exactly.  ``k`` is chosen so that no
sequence interval exceeds ``2^-k`` even with that rounding, which makes the
map injective (the dematcher returns the one grid point in the interval);
the rate loss of the rounding is under two bits per block.  The matcher is a
sequential recursion per block: Numba over blocks on the CPU; GPU input to
:func:`~commkit.coding.decode` makes one round trip, and ``generate`` runs
it on the host before moving the payload.

:class:`PAS` (probabilistic amplitude shaping, Böcherer, Steiner and
Schulte, 2015) combines a CCDM over the amplitudes of one real dimension
with a systematic FEC code, whose parity bits supply the signs.  It needs
labels with a sign bit per real dimension and the same amplitude label for
``±a``: Gray QAM and bipolar PAM as the factories build them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np

from ..backend import ArrayType, dispatch, to_device
from ..mapping import Constellation
from ..mapping.gray import _pack
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

__all__ = ["CCDM", "PAS"]

# Range-coder precision: the range is renormalized to [2^31, 2^32), so a
# split range * count stays below 2^63 for blocks below 2^31 points.
_RANGE_BITS = 32
_MAX_BLOCK = 1 << 30


@dataclass(frozen=True, eq=False)
class CCDM(_Code):
    """Constant-composition distribution matcher onto a constellation.

    Parameters
    ----------
    constellation : Constellation
        The target: its ``pmf`` (uniform if none), points and labels.
    block_length : int
        Points per block, ``1 <= block_length <= 2^30``.

    Notes
    -----
    ``k`` information bits per block map to ``block_length`` points, whose
    labels are the transmitted bits: ``n = block_length * bits_per_symbol``.
    Every block holds exactly :attr:`composition` of each point, so the
    transmitted statistics are the composition, which differs from the pmf
    by its quantization to ``1 / block_length``.  The rate ``k /
    block_length`` approaches the entropy of the pmf as ``O(log(n) / n)``.
    """

    constellation: Constellation
    block_length: int

    def __post_init__(self) -> None:
        if not isinstance(self.constellation, Constellation):
            raise TypeError(
                "CCDM(): constellation must be a Constellation, got "
                f"{type(self.constellation).__name__}."
            )
        n = self.block_length
        if (
            isinstance(n, bool)
            or not isinstance(n, int | np.integer)
            or not 1 <= n <= _MAX_BLOCK
        ):
            raise ValueError(
                f"block_length must be an integer in [1, 2^30], got {n!r}."
            )
        object.__setattr__(self, "block_length", int(n))
        if self.k < 1:
            raise ValueError(
                f"CCDM(): a block of {n} points carries no information bit; use a "
                "longer block."
            )

    @property
    def n(self) -> int:
        """Transmitted bits per block: ``block_length * bits_per_symbol``."""
        return self.block_length * self.constellation.bits_per_symbol

    @property
    def k(self) -> int:
        """Information bits per block."""
        return self._k

    def __repr__(self) -> str:
        return f"CCDM({self.constellation!r}, block_length={self.block_length})"

    @cached_property
    def composition(self) -> np.ndarray:
        """Points of each kind per block, ``(M,)`` int64 summing to ``block_length``.

        The quantization of the pmf that minimizes the divergence
        ``D(composition / block_length || pmf)``: ``floor(n p)`` raised one
        point at a time where the divergence grows least.
        """
        c = _composition(self._pmf, self.block_length)
        c.setflags(write=False)
        return c

    @cached_property
    def _pmf(self) -> np.ndarray:
        pmf = self.constellation.pmf
        order = self.constellation.order
        return np.full(order, 1.0 / order) if pmf is None else pmf

    @cached_property
    def _log2_sequences(self) -> float:
        """``log2`` of the number of sequences of the composition."""
        counts = self.composition
        n = self.block_length
        lg = math.lgamma(n + 1) - sum(math.lgamma(c + 1) for c in counts)
        return lg / math.log(2)

    @cached_property
    def _k(self) -> int:
        log2_m = self._log2_sequences
        a = int(np.count_nonzero(self.composition))
        # Worst-case growth of a sequence interval from the floor rounding:
        # the most frequent point's share grows by at most (a - 1) / range.
        slack = self.block_length * math.log2(
            1.0 + a * (a - 1) / 2.0 ** (_RANGE_BITS - 1)
        )
        # lgamma is good to a few ulp; the margin covers it many times over.
        margin = 1e-9 * max(1.0, log2_m)
        return max(0, math.floor(log2_m - slack - margin))

    @cached_property
    def _buffer_bits(self) -> int:
        """Fractional bits of the dematcher's interval bound, with margin."""
        n = self.block_length
        shrink = -n * math.log2(1.0 - n / 2.0 ** (_RANGE_BITS - 1))
        return int(math.ceil(self._log2_sequences + shrink)) + 2 * _RANGE_BITS + 8

    @cached_property
    def _label_index(self) -> np.ndarray:
        """Point index of each packed label, ``(M,)`` int64."""
        lut = np.empty(self.constellation.order, dtype=np.int64)
        lut[_pack(self.constellation.bit_labels)] = np.arange(self.constellation.order)
        return lut

    def _match(self, bits: np.ndarray) -> np.ndarray:
        """Host ``(R, k)`` bits to ``(R, block_length)`` point indices."""
        out = np.empty((bits.shape[0], self.block_length), dtype=np.int64)
        _get_numba_ccdm("match")(
            np.ascontiguousarray(bits, dtype=np.int8), self.composition, out
        )
        return out

    def _dematch(self, symbols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Host ``(R, block_length)`` indices to ``(R, k)`` bits and validity."""
        rows = symbols.shape[0]
        bits = np.empty((rows, self.k), dtype=np.int8)
        valid = np.empty(rows, dtype=np.bool_)
        _get_numba_ccdm("dematch")(
            np.ascontiguousarray(symbols, dtype=np.int64),
            self.composition,
            self._buffer_bits,
            bits,
            valid,
        )
        return bits, valid


def _composition(pmf: np.ndarray, n: int) -> np.ndarray:
    """KL-optimal counts ``c`` (sum ``n``) for ``pmf`` (Böcherer and Geiger)."""
    p = np.asarray(pmf, dtype=np.float64)
    c = np.floor(n * p).astype(np.int64)
    for _ in range(n - int(c.sum())):
        # Divergence increase of one more point i: (c+1) log((c+1)/np) - c log(c/np).
        with np.errstate(divide="ignore", invalid="ignore"):
            cost = (c + 1) * np.log((c + 1) / (n * p)) - np.where(
                c > 0, c * np.log(c / (n * p)), 0.0
            )
        cost[p == 0] = np.inf
        c[int(np.argmin(cost))] += 1
    return c


def _get_numba_ccdm(name: str) -> Any:
    """Numba matcher and dematcher, parallel over blocks.

    The range coder of the module docstring.  The interval of the current
    prefix is ``[low, low + range 2^-S)``; ``range`` is kept in
    ``[2^31, 2^32)`` by doubling it (``S += 1``).  Splitting gives point
    ``i`` the share ``floor(range c_i / n_rem)``, plus the remainder for the
    most frequent point (lowest index among equals).  The matcher tracks
    ``D = floor((v - low) 2^S)`` for its input point ``v``; the dematcher
    accumulates ``low`` exactly in a bit array and rounds it up to ``k``
    bits.
    """
    from . import _numba_loops

    return getattr(_numba_loops, f"ccdm_{name}")


def _encode_ccdm(code: CCDM, bits: ArrayType) -> ArrayType:
    _, xp, _ = dispatch(bits)
    symbols = code._match(to_device(bits, "cpu"))
    labels = code.constellation.bit_labels[symbols].reshape(bits.shape[0], -1)
    return to_device(labels, "cpu" if xp is np else "gpu")


def _decode_ccdm(code: CCDM, llrs: ArrayType, method: object) -> _Decoded:
    return _dematch_labels(code, _hard(llrs))


def _dematch_labels(code: CCDM, labels: ArrayType) -> _Decoded:
    """Dematch hard label bits ``(R, block_length m)``, on their device."""
    _, xp, _ = dispatch(labels)
    m = code.constellation.bits_per_symbol
    hard = to_device(labels, "cpu").reshape(labels.shape[0], code.block_length, m)
    symbols = code._label_index[_pack(hard)]
    bits, valid = code._dematch(symbols)
    device = "cpu" if xp is np else "gpu"
    return _Decoded(
        to_device(bits, device),
        None,
        to_device(valid, device),
        xp.ones(labels.shape[0], dtype=np.int32),
    )


@dataclass(frozen=True)
class _Dematch:
    """The only decoder of a matcher: invert the hard decisions."""


@dataclass(frozen=True, eq=False)
class PAS(_Code):
    """Probabilistic amplitude shaping: a CCDM on the amplitudes, a FEC code for signs.

    Parameters
    ----------
    code : code object
        A systematic binary code (LDPC, linear block codes, ...) of rate
        ``R_c >= (m - 1) / m`` for ``m`` bits per real dimension, with ``n``
        a multiple of ``2 m`` (QAM) or ``m`` (PAM).
    constellation : Constellation
        Gray square QAM or bipolar PAM with a shaping pmf symmetric in each
        real dimension (``Constellation.qam(64).shaped(entropy=5)``).

    Notes
    -----
    Per FEC codeword of ``n_c`` bits there are ``N = n_c / m`` real symbols.
    A CCDM of ``N`` amplitudes maps ``k_dm`` bits to ``(m - 1) N``
    amplitude bits; the FEC encodes them together with ``gamma N = k_c -
    (m - 1) N`` uniform information bits, and the ``gamma N`` uniform bits
    and the ``(1 - gamma) N`` parity bits are the signs.  So ``n = n_c`` and
    ``k = k_dm + gamma N``.  A fixed permutation puts the bits on the labels
    (sign, then amplitude bits, per real dimension; QAM pairs consecutive
    dimensions as I and Q).  Decoding runs the FEC decoder on the
    de-permuted LLRs, which must include the prior (as
    :func:`~commkit.mapping.compute_llr` does), then the dematcher.
    """

    code: Any
    constellation: Constellation

    def __post_init__(self) -> None:
        _codec(self.code, "PAS()")
        if not isinstance(self.constellation, Constellation):
            raise TypeError("PAS(): constellation must be a Constellation.")
        m = self._layout[0]
        dims = 2 if self.constellation.is_complex else 1
        if self.code.n % (dims * m):
            raise ValueError(
                f"PAS(): the code length {self.code.n} must be a multiple of "
                f"{dims * m} ({dims} dimension(s) of {m} bits)."
            )
        if self._sign_info < 0:
            raise ValueError(
                f"PAS(): the code rate {self.code.rate:.4f} is below (m - 1) / m = "
                f"{(m - 1) / m:.4f}: its parity cannot all be signs."
            )
        _info_positions(self.code, "PAS()")

    @property
    def n(self) -> int:
        """Transmitted bits per FEC codeword."""
        return int(self.code.n)

    @property
    def k(self) -> int:
        """Information bits per FEC codeword."""
        return int(self.matcher.k + self._sign_info)

    def __repr__(self) -> str:
        return f"PAS({self.code!r}, {self.constellation!r})"

    @cached_property
    def _layout(self) -> tuple[int, Constellation]:
        """``(m, amplitudes)``: bits per dimension and the amplitude alphabet."""
        c = self.constellation
        if c.is_complex:
            axes = c._axes
            m = c.bits_per_symbol // 2
            if (
                axes is None
                or c.bits_per_symbol % 2
                or not np.array_equal(axes[0].bits, np.arange(m))
                or not np.array_equal(axes[1].bits, np.arange(m, 2 * m))
                or not np.allclose(axes[0].levels, axes[1].levels)
                or not np.allclose(
                    np.zeros(1) if axes[0].pmf is None else axes[0].pmf,
                    np.zeros(1) if axes[1].pmf is None else axes[1].pmf,
                )
            ):
                raise ValueError(
                    "PAS(): needs square QAM with the I bits first and the same "
                    "levels and prior on both axes, as Constellation.qam builds."
                )
            levels, labels, pmf = axes[0].levels, axes[0].labels, axes[0].pmf
        else:
            order = np.argsort(c.points)
            levels, labels, pmf = c.points[order], c.bit_labels[order], None
            pmf = None if c.pmf is None else c.pmf[order]
            m = c.bits_per_symbol
        size = levels.size
        mirror = labels[::-1]
        if (
            m < 2
            or not np.allclose(levels, -levels[::-1])
            or not np.array_equal(labels[:, 0], (levels > 0).astype(labels.dtype))
            or not np.array_equal(labels[:, 1:], mirror[:, 1:])
            or (pmf is not None and not np.allclose(pmf, pmf[::-1]))
        ):
            raise ValueError(
                "PAS(): needs at least two bits per real dimension, a sign bit "
                "first, the same amplitude label for +a and -a and a symmetric "
                "prior (Gray QAM or bipolar PAM; unipolar PAM has no sign)."
            )
        positive = slice(size // 2, size)
        amp_pmf = None if pmf is None else 2 * pmf[positive]
        amplitudes = Constellation(
            levels[positive], bit_labels=labels[positive, 1:], pmf=amp_pmf
        )
        return m, amplitudes

    @cached_property
    def matcher(self) -> CCDM:
        """The CCDM of the amplitudes of one FEC codeword."""
        m, amplitudes = self._layout
        return CCDM(amplitudes, block_length=self.code.n // m)

    @cached_property
    def _sign_info(self) -> int:
        """``gamma N``: uniform information bits that become signs."""
        m = self._layout[0]
        return int(self.code.k - (m - 1) * (self.code.n // m))

    @cached_property
    def _interleaver(self) -> Interleaver:
        """Codeword bit order to label order."""
        m = self._layout[0]
        dims = self.code.n // m
        info = _info_positions(self.code, "PAS()")
        parity = np.setdiff1d(np.arange(self.code.n), info)
        amplitude = info[: (m - 1) * dims].reshape(dims, m - 1)
        sign = np.concatenate([info[(m - 1) * dims :], parity])
        return Interleaver(np.concatenate([sign[:, None], amplitude], axis=1).ravel())


def _encode_pas(code: PAS, bits: ArrayType) -> ArrayType:
    _, xp, _ = dispatch(bits)
    ccdm = code.matcher
    amplitude_bits = _encode_ccdm(ccdm, bits[:, : ccdm.k])
    info = xp.concatenate([amplitude_bits, bits[:, ccdm.k :]], axis=1)
    return interleave(_encode_blocks(code.code, info), interleaver=code._interleaver)


def _decode_pas(code: PAS, llrs: ArrayType, method: Any) -> _Decoded:
    _, xp, _ = dispatch(llrs)
    ccdm = code.matcher
    res = _decode_blocks(
        code.code, deinterleave(llrs, interleaver=code._interleaver), method
    )
    amplitude_bits = res.bits[:, : ccdm.n]
    dm = _dematch_labels(ccdm, amplitude_bits)
    return _Decoded(
        xp.concatenate([dm.bits, res.bits[:, ccdm.n :]], axis=1),
        None
        if res.llrs is None
        else interleave(res.llrs, interleaver=code._interleaver),
        res.valid & dm.valid,
        res.iterations,
    )


_register(
    CCDM,
    _Codec(
        encode=_encode_ccdm,
        decode=_decode_ccdm,
        default_method=lambda code: _Dematch(),
        methods=(_Dematch,),
        host=True,
    ),
)
_register(
    PAS,
    _Codec(
        encode=_encode_pas,
        decode=_decode_pas,
        default_method=lambda code: _resolve_method(code.code, None, "decode()"),
        methods=(object,),
        host=True,
    ),
)
