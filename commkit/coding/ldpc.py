"""LDPC codes: :class:`LDPCCode` and the layered decoders :class:`MinSum` and :class:`SumProduct`.

A code is its sparse parity-check matrix ``H`` (``m x N``), optionally with
the last ``punctured`` bits of each codeword not transmitted.  Build one
from ``H`` (:meth:`LDPCCode.from_parity_check`), from a quasi-cyclic base
matrix (:meth:`LDPCCode.quasi_cyclic`), or take a standard code:
:meth:`LDPCCode.dvbs2` (ETSI EN 302 307, the common proxy for the
proprietary SD-FEC of coherent modems) and :meth:`LDPCCode.pon`
(LDPC(17280, 14592) of IEEE 802.3ca and ITU-T G.9804.2).

Encoding is systematic, information bits first, by the cheapest form the
parity part of ``H`` allows: a dual-diagonal (accumulator) parity part, as
in DVB-S2, is a sparse product and a cumulative XOR; an invertible parity
part is a sparse product and one GEMM with its inverse; otherwise a dense
systematic generator from row reduction (moderate sizes only).

Decoding is layered belief propagation over a row-layer partition built
once per code (greedy first-fit coloring of the rows: a layer's rows share
no variable): Numba over codewords on the CPU, one CUDA thread block per
codeword on the GPU, the same recursion in the same order.  Early stopping
is per codeword, on a satisfied syndrome.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache, cached_property
from typing import Any, NamedTuple

import numpy as np

from .. import _cuda
from ..backend import ArrayType, _device_tables, dispatch
from ..logger import logger
from ._gf2 import inverse, matmul_mod2, rank, row_reduce
from ._ldpc_tables import DVBS2, PON_BASE
from .base import _Code, _Codec, _Decoded, _register

__all__ = ["LDPCCode", "MinSum", "SumProduct"]

# Largest parity part inverted densely (m x m bits, kept on each device).
_MAX_INVERSE = 8192
# Largest m * N elimination of the dense-generator fallback.
_MAX_DENSE = 1 << 27
# float32 elements of the GPU message buffer per launch (512 MiB).
_GPU_MESSAGES = 1 << 27


@dataclass(frozen=True, eq=False)
class LDPCCode(_Code):
    """LDPC code given by the edges of its sparse parity-check matrix.

    Parameters
    ----------
    checks, variables : array_like
        Row and column of each nonzero entry of ``H``, ``(E,)`` integers,
        without duplicates.  Every row has at least two entries.
    num_checks : int
        Rows ``m`` of ``H``.
    length : int
        Columns ``N`` of ``H``: the mother codeword length.
    punctured : int, default 0
        Trailing codeword bits that are not transmitted (parity bits), so
        ``n = N - punctured``; their LLRs are zero at the decoder.

    Notes
    -----
    Prefer the constructors.  ``k = N - rank(H)``; the information bits are
    the first ``k`` codeword bits for codes whose parity part (the last
    ``m`` columns) is dual-diagonal or invertible.
    """

    checks: np.ndarray
    variables: np.ndarray
    num_checks: int
    length: int
    punctured: int = 0

    def __post_init__(self) -> None:
        rows = np.asarray(self.checks)
        cols = np.asarray(self.variables)
        for name in ("num_checks", "length", "punctured"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int | np.integer):
                raise TypeError(f"{name} must be an integer, got {v!r}.")
        m, n = int(self.num_checks), int(self.length)
        if rows.ndim != 1 or rows.shape != cols.shape or rows.size == 0:
            raise ValueError("checks and variables must be equal-length 1-D arrays.")
        if rows.dtype.kind not in "iu" or cols.dtype.kind not in "iu":
            raise TypeError(
                f"checks and variables must be integers, got {rows.dtype} and "
                f"{cols.dtype}."
            )
        if m < 1 or n <= m:
            raise ValueError(f"need 1 <= num_checks < length, got {m} and {n}.")
        if rows.min() < 0 or rows.max() >= m or cols.min() < 0 or cols.max() >= n:
            raise ValueError("checks or variables out of range.")
        order = np.lexsort((cols, rows))
        rows, cols = rows[order].astype(np.int32), cols[order].astype(np.int32)
        flat = rows.astype(np.int64) * n + cols
        if np.any(flat[1:] == flat[:-1]):
            raise ValueError("H has duplicate entries.")
        if np.bincount(rows, minlength=m).min() < 2:
            raise ValueError("every parity check needs at least two bits.")
        p = int(self.punctured)
        if p < 0:
            raise ValueError(f"punctured must be >= 0, got {p}.")
        for name, value in (("checks", rows), ("variables", cols)):
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "num_checks", m)
        object.__setattr__(self, "length", n)
        object.__setattr__(self, "punctured", p)
        if self._encoder.info.max(initial=-1) >= n - p:
            raise ValueError("punctured bits must be parity bits.")

    @property
    def n(self) -> int:
        """Transmitted bits per codeword."""
        return self.length - self.punctured

    @property
    def k(self) -> int:
        """Information bits per codeword."""
        return int(self._encoder.info.size)

    def __repr__(self) -> str:
        return f"LDPCCode(n={self.n}, k={self.k})"

    def parity_check(self) -> Any:
        """The parity-check matrix ``H``.

        Returns
        -------
        scipy.sparse.csr_array
            ``(num_checks, length)`` int8 on the host, over the mother
            codeword (punctured positions included).
        """
        import scipy.sparse

        return scipy.sparse.csr_array(
            (np.ones(self.checks.size, np.int8), (self.checks, self.variables)),
            shape=(self.num_checks, self.length),
        )

    # -- constructors ----------------------------------------------------------

    @classmethod
    def from_parity_check(cls, parity_check: Any, *, punctured: int = 0) -> LDPCCode:
        """The code of a parity-check matrix.

        Parameters
        ----------
        parity_check : array_like or scipy.sparse array or matrix
            ``H``, ``(m, N)``, values 0 and 1, on the host.
        punctured : int, default 0
            Trailing codeword bits not transmitted.

        Returns
        -------
        LDPCCode
            Logs a warning when the Tanner graph has 4-cycles (two checks
            sharing two bits), which degrade belief propagation.
        """
        import scipy.sparse

        h = scipy.sparse.coo_array(parity_check)
        h.sum_duplicates()
        h.eliminate_zeros()
        if np.any(h.data != 1):
            raise ValueError("parity_check must contain only 0 and 1.")
        code = cls(h.row, h.col, h.shape[0], h.shape[1], punctured)
        _warn_if_4_cycles(code)
        return code

    @classmethod
    def quasi_cyclic(cls, base: Any, lifting: int, *, punctured: int = 0) -> LDPCCode:
        """Quasi-cyclic code from a base matrix of circulant shifts.

        Parameters
        ----------
        base : array_like
            ``(mb, nb)`` integers: ``s >= 0`` places the ``lifting x lifting``
            identity shifted right by ``s`` columns (row ``r`` has its one
            at column ``(r + s) mod lifting``), ``-1`` a zero block.
        lifting : int
            Circulant size ``Z``: ``H`` is ``(mb Z, nb Z)``.
        punctured : int, default 0
            Trailing codeword bits not transmitted.

        Returns
        -------
        LDPCCode
            Logs a warning on 4-cycles, as :meth:`from_parity_check`.
        """
        b = np.asarray(base)
        z = int(lifting)
        if b.ndim != 2 or z < 1:
            raise ValueError("base must be a matrix and lifting >= 1.")
        if b.min() < -1:
            raise ValueError("base entries must be shifts >= 0 or -1.")
        bi, bj = np.nonzero(b >= 0)
        r = np.arange(z)
        rows = (bi[:, None] * z + r).ravel()
        cols = (bj[:, None] * z + (r + b[bi, bj][:, None]) % z).ravel()
        code = cls(rows, cols, b.shape[0] * z, b.shape[1] * z, punctured)
        _warn_if_4_cycles(code)
        return code

    @classmethod
    def dvbs2(cls, rate: str, frame: str = "normal") -> LDPCCode:
        """DVB-S2 LDPC code (ETSI EN 302 307-1, clause 5.3.2, Annexes B and C).

        Parameters
        ----------
        rate : str
            Code rate: ``"1/4"``, ``"1/3"``, ``"2/5"``, ``"1/2"``, ``"3/5"``,
            ``"2/3"``, ``"3/4"``, ``"4/5"``, ``"5/6"``, ``"8/9"``, ``"9/10"``
            (normal frames only).
        frame : {"normal", "short"}, default "normal"
            ``n = 64800`` or ``16200``.  Short frames have the standard's
            nominal rates (``k = 7200`` for ``"1/2"``).

        Returns
        -------
        LDPCCode
            The LDPC code alone, without the outer BCH code of the standard;
            parity part dual-diagonal (accumulator encoding).
        """
        return _dvbs2(rate, frame)

    @classmethod
    def pon(cls) -> LDPCCode:
        """LDPC(17280, 14592) of IEEE 802.3ca and ITU-T G.9804.2 (Annex B.1).

        The 12 x 69 quasi-cyclic mother code with circulants of 256
        (``N = 17664``), the last 384 parity bits punctured: rate 0.844, the
        code of 25G/50G EPON and of 50G-PON downstream.

        Returns
        -------
        LDPCCode
        """
        return _pon()

    # -- cached structure ----------------------------------------------------

    @cached_property
    def _schedule(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(row_ptr, cols, layer_ptr)``: CSR of H with rows in layer order."""
        m = self.num_checks
        ptr = np.concatenate([[0], np.cumsum(np.bincount(self.checks, minlength=m))])
        color = _first_fit(ptr, self.variables, self.length)
        order = np.argsort(color, kind="stable")
        degree = np.diff(ptr)[order]
        row_ptr = np.concatenate([[0], np.cumsum(degree)]).astype(np.int32)
        cols = np.concatenate([self.variables[ptr[r] : ptr[r + 1]] for r in order])
        counts = np.bincount(color)
        layer_ptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)
        logger.debug("%r: %d layers.", self, counts.size)
        return row_ptr, cols.astype(np.int32), layer_ptr

    @cached_property
    def _encoder(self) -> _Encoder:
        return _build_encoder(self)


class _Encoder(NamedTuple):
    """How a code encodes: information and parity positions and the tables.

    ``kind`` is ``"accumulate"`` (``tables`` = the CSR of ``H`` over the
    information columns), ``"inverse"`` (that CSR and ``(H_p^-1)^T``) or
    ``"dense"`` (the parity part of a systematic generator, ``(k, m')``).
    """

    kind: str
    info: np.ndarray
    parity: np.ndarray
    tables: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class MinSum:
    """Layered normalized (and offset) min-sum decoding.

    The check-to-bit message is ``sign * max(scale * min - offset, 0)``
    over the other bits of the check.

    Parameters
    ----------
    scale : float, default 0.875
        Normalization, in ``(0, 1]``; 1 is plain min-sum.  0.875 decoded
        DVB-S2 rate 1/2 within 0.1 dB of sum-product in the layered
        schedule; 0.75, a common choice for flooding, lost 0.4 dB there.
    offset : float, default 0.0
        Offset, ``>= 0``.
    max_iterations : int, default 50
        Iterations (passes over all layers) per codeword.
    early_stop : bool, default True
        Stop a codeword once its syndrome is zero (checked before each
        iteration).
    """

    scale: float = 0.875
    offset: float = 0.0
    max_iterations: int = 50
    early_stop: bool = True

    def __post_init__(self) -> None:
        if not 0.0 < self.scale <= 1.0:
            raise ValueError(f"scale must be in (0, 1], got {self.scale}.")
        if self.offset < 0:
            raise ValueError(f"offset must be >= 0, got {self.offset}.")
        _check_iterations(self.max_iterations)


@dataclass(frozen=True)
class SumProduct:
    """Layered sum-product (belief-propagation) decoding.

    The check-to-bit message is ``sign * phi(sum phi(|Q|))`` over the other
    bits, with ``phi(x) = -log tanh(x / 2)`` evaluated in float32 on
    ``|x|`` clipped to ``[1e-7, 30]``.

    Parameters
    ----------
    max_iterations : int, default 50
        Iterations per codeword.
    early_stop : bool, default True
        Stop a codeword once its syndrome is zero.
    """

    max_iterations: int = 50
    early_stop: bool = True

    def __post_init__(self) -> None:
        _check_iterations(self.max_iterations)


def _check_iterations(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"max_iterations must be an integer >= 1, got {value!r}.")


# -----------------------------------------------------------------------------
# Construction
# -----------------------------------------------------------------------------


@cache
def _dvbs2(rate: str, frame: str) -> LDPCCode:
    n = {"normal": 64800, "short": 16200}.get(frame)
    if n is None:
        raise ValueError(f"frame must be 'normal' or 'short', got {frame!r}.")
    table = DVBS2.get((frame, rate))
    if table is None:
        rates = [r for f, r in DVBS2 if f == frame]
        raise ValueError(
            f"rate must be one of {rates} for {frame} frames, got {rate!r}."
        )
    k = 360 * len(table)
    m = n - k
    q = m // 360
    j = np.arange(360)
    rows, cols = [], []
    for g, addresses in enumerate(table):
        x = np.asarray(addresses)
        rows.append(((x[:, None] + j * q) % m).ravel())
        cols.append(np.broadcast_to(360 * g + j, (x.size, 360)).ravel())
    # Accumulator: parity bit i checks rows i and i + 1.
    p = np.arange(m)
    rows += [p, p[:-1] + 1]
    cols += [k + p, k + p[:-1]]
    return LDPCCode(np.concatenate(rows), np.concatenate(cols), m, n)


@cache
def _pon() -> LDPCCode:
    return LDPCCode.quasi_cyclic(np.asarray(PON_BASE), 256, punctured=384)


def _warn_if_4_cycles(code: LDPCCode) -> None:
    h = code.parity_check().astype(np.int32)
    overlap = (h @ h.T).tocoo()
    if np.any((overlap.data > 1) & (overlap.row != overlap.col)):
        logger.warning("%r has 4-cycles; belief propagation degrades on them.", code)


def _first_fit(ptr: np.ndarray, variables: np.ndarray, n: int) -> np.ndarray:
    """Layer of each row: the first layer none of whose rows shares a bit.

    Rows in order; each bit keeps the set of layers using it as a bit mask
    (a Python integer).
    """
    masks = [0] * n
    cols = variables.tolist()
    bounds = ptr.tolist()
    color = np.empty(len(bounds) - 1, dtype=np.int64)
    for r in range(len(bounds) - 1):
        row = cols[bounds[r] : bounds[r + 1]]
        used = 0
        for v in row:
            used |= masks[v]
        c = (~used & (used + 1)).bit_length() - 1
        bit = 1 << c
        for v in row:
            masks[v] |= bit
        color[r] = c
    return color


def _build_encoder(code: LDPCCode) -> _Encoder:
    m, n = code.num_checks, code.length
    rows, cols = code.checks, code.variables
    k = n - m
    in_parity = cols >= k
    p_rows, p_cols = rows[in_parity], cols[in_parity] - k
    info_csr = _csr(rows[~in_parity], cols[~in_parity], (m, k))
    # Dual-diagonal parity part: column i has rows i and i + 1 (the last only i).
    expected = np.sort(
        np.concatenate([np.arange(m) * (m + 1), np.arange(m - 1) * (m + 1) + m])
    )
    if p_rows.size == 2 * m - 1 and np.array_equal(
        np.sort(p_rows.astype(np.int64) * m + p_cols), expected
    ):
        return _Encoder("accumulate", np.arange(k), np.arange(k, n), info_csr)
    if m <= _MAX_INVERSE:
        hp = np.zeros((m, m), dtype=np.int8)
        hp[p_rows, p_cols] = 1
        if rank(hp) == m:
            inv_t = inverse(hp).T.astype(np.float32)
            return _Encoder(
                "inverse", np.arange(k), np.arange(k, n), (*info_csr, inv_t)
            )
    if m * n > _MAX_DENSE:
        raise ValueError(
            f"LDPCCode ({m} x {n}): the parity part is neither dual-diagonal nor "
            "invertible, and H is too large for a dense systematic generator."
        )
    h = np.zeros((m, n), dtype=np.int8)
    h[rows, cols] = 1
    rref, pivots = row_reduce(h, reverse=True)
    free = np.setdiff1d(np.arange(n), pivots)
    # c[pivots[i]] = sum over free f of rref[i, f] c[f].
    return _Encoder("dense", free, pivots, (rref[:, free].T.astype(np.float32),))


def _csr(
    rows: np.ndarray, cols: np.ndarray, shape: tuple[int, int]
) -> tuple[np.ndarray, ...]:
    """``(data, indices, indptr)`` of a binary CSR matrix, float32 data."""
    order = np.lexsort((cols, rows))
    indptr = np.concatenate([[0], np.cumsum(np.bincount(rows, minlength=shape[0]))])
    return (
        np.ones(rows.size, np.float32),
        cols[order].astype(np.int32),
        indptr.astype(np.int32),
    )


def _sparse(xp: Any, tables: tuple[Any, ...], shape: tuple[int, int]) -> Any:
    """A CSR matrix on ``xp``'s device from cached ``(data, indices, indptr)``.

    SciPy's sparse array on the CPU; CuPy has only the matrix interface.
    """
    if xp is np:
        import scipy.sparse

        return scipy.sparse.csr_array(tuple(tables), shape=shape)
    import cupyx.scipy.sparse

    return cupyx.scipy.sparse.csr_matrix(tuple(tables), shape=shape)


# -----------------------------------------------------------------------------
# Encoding and decoding
# -----------------------------------------------------------------------------


def _encode(code: LDPCCode, bits: ArrayType) -> ArrayType:
    _, xp, _ = dispatch(bits)
    enc = code._encoder
    m, k = code.num_checks, code.k
    tables = _device_tables(code, "encoder", xp, lambda: enc.tables)
    if enc.kind == "dense":
        parity = matmul_mod2(bits, tables[0])
    else:
        # s = H_info u (mod 2), one sparse product for all codewords.
        s = _sparse(xp, tables[:3], (m, k)) @ bits.T.astype(xp.float32)
        s = s.astype(xp.int32) & 1
        if enc.kind == "accumulate":
            parity = (xp.cumsum(s, axis=0) & 1).astype(xp.int8).T
        else:
            parity = matmul_mod2(s.T, tables[3])
    words = xp.empty((bits.shape[0], code.length), dtype=xp.int8)
    if enc.kind == "dense":
        info_dev, parity_dev = _device_tables(
            code, "positions", xp, lambda: (enc.info, enc.parity)
        )
        words[:, info_dev] = bits
        words[:, parity_dev] = parity
    else:
        words[:, :k] = bits
        words[:, k:] = parity
    return words[:, : code.n]


def _decode(code: LDPCCode, llrs: ArrayType, method: MinSum | SumProduct) -> _Decoded:
    _, xp, _ = dispatch(llrs)
    row_ptr, cols, layer_ptr = _device_tables(
        code, "schedule", xp, lambda: code._schedule
    )
    rows = llrs.shape[0]
    app = xp.zeros((rows, code.length), dtype=xp.float32)
    app[:, : code.n] = llrs
    sum_product = isinstance(method, SumProduct)
    scale = 1.0 if sum_product else method.scale  # type: ignore[union-attr]
    offset = 0.0 if sum_product else method.offset  # type: ignore[union-attr]
    if xp is np:
        iters = np.empty(rows, dtype=np.int32)
        valid = np.empty(rows, dtype=np.bool_)
        _get_numba_ldpc()(
            app,
            row_ptr,
            cols,
            sum_product,
            np.float32(scale),
            np.float32(offset),
            method.max_iterations,
            method.early_stop,
            iters,
            valid,
        )
    else:
        launch = _cuda.get_kernel("ldpc_layered", sum_product=sum_product)
        chunk = max(1, _GPU_MESSAGES // cols.size)
        iters = xp.empty(rows, dtype=xp.int32)
        valid = xp.empty(rows, dtype=bool)
        for r0 in range(0, rows, chunk):
            part = app[r0 : r0 + chunk]
            iters[r0 : r0 + chunk], valid[r0 : r0 + chunk] = launch(
                part,
                row_ptr,
                cols,
                layer_ptr,
                scale=scale,
                offset=offset,
                max_iterations=method.max_iterations,
                early_stop=method.early_stop,
            )
    info = code._encoder.info
    if info.size == info[-1] + 1:
        bits = (app[:, : code.k] < 0).astype(np.int8)
    else:
        (info_dev,) = _device_tables(code, "info", xp, lambda: (info,))
        bits = (app[:, info_dev] < 0).astype(np.int8)
    return _Decoded(bits, app[:, : code.n], valid, iters)


def _get_numba_ldpc() -> Any:
    """Numba layered decoder for the CPU, parallel over codewords.

    Rows run in the schedule's order, each in the two passes of the CUDA
    kernel (``ldpc.cu``), compiled without fast-math so min-sum matches the
    kernel bit for bit.  ``app`` holds the channel LLRs on entry and the
    a-posteriori LLRs on exit.
    """
    from . import _numba_loops

    return _numba_loops.ldpc_layered


_register(
    LDPCCode,
    _Codec(
        encode=_encode,
        decode=_decode,
        default_method=lambda code: MinSum(),
        methods=(MinSum, SumProduct),
        info_positions=lambda code: code._encoder.info,
    ),
)
