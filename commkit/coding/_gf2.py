"""Linear algebra over GF(2) (private).

Host routines on small binary matrices: :func:`row_reduce` (reduced row
echelon form), :func:`rank`, :func:`null_space` and :func:`inverse`, on
rows packed into 64-bit words, so an elimination step is one vectorized
XOR of the affected rows.  And :func:`matmul_mod2`, the product of bit
arrays modulo 2 on the data's device as one float32 GEMM (BLAS or cuBLAS)
followed by ``& 1``.  Used by every binary linear code in
:mod:`commkit.coding` (encoding, syndromes, parity checks).
"""

from typing import Any

import numpy as np

from ..backend import ArrayType, dispatch

# Elements of one float32 operand chunk of matmul_mod2 (64 MiB).
_CHUNK_ELEMENTS = 1 << 24
# float32 represents every integer below 2^24 exactly: the largest inner
# dimension whose sums stay exact.
_MAX_INNER = 1 << 24


def _packed(m: np.ndarray) -> np.ndarray:
    """Bit rows ``(r, n)`` to little-endian uint64 words ``(r, ceil(n/64))``.

    Column ``j`` is bit ``j % 64`` of word ``j // 64``.
    """
    r, n = m.shape
    words = (n + 63) // 64
    by = np.packbits(m.astype(np.uint8), axis=1, bitorder="little")
    padded = np.zeros((r, words * 8), dtype=np.uint8)
    padded[:, : by.shape[1]] = by
    return padded.view("<u8")


def _unpacked(p: np.ndarray, n: int) -> np.ndarray:
    """Inverse of :func:`_packed`: ``(r, n)`` int8 bits."""
    by = np.ascontiguousarray(p).view(np.uint8)
    return np.unpackbits(by, axis=1, count=n, bitorder="little").astype(np.int8)


def row_reduce(m: Any, *, reverse: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Reduced row echelon form over GF(2).

    Parameters
    ----------
    m : array_like
        Binary matrix ``(r, n)`` on the host.
    reverse : bool, default False
        Choose pivots from the last column backwards, so the pivot columns
        are the rightmost independent ones (parity bits at the end of a
        codeword).

    Returns
    -------
    rref : numpy.ndarray
        int8 ``(rank, n)``: the nonzero rows, ``rref[i, pivots[i]] = 1`` and
        every other row zero in that column.
    pivots : numpy.ndarray
        int64 pivot column of each row, ``(rank,)``.
    """
    a = np.asarray(m)
    if a.ndim != 2:
        raise ValueError(f"expected a matrix, got shape {a.shape}.")
    r, n = a.shape
    p = _packed(a & 1)
    pivots: list[int] = []
    row = 0
    for col in range(n - 1, -1, -1) if reverse else range(n):
        if row == r:
            break
        w, b = divmod(col, 64)
        bit = np.uint64(1 << b)
        hits = np.flatnonzero(p[row:, w] & bit)
        if hits.size == 0:
            continue
        pivot = row + int(hits[0])
        if pivot != row:
            p[[row, pivot]] = p[[pivot, row]]
        others = np.flatnonzero(p[:, w] & bit)
        others = others[others != row]
        p[others] ^= p[row]
        pivots.append(col)
        row += 1
    return _unpacked(p[:row], n), np.asarray(pivots, dtype=np.int64)


def rank(m: Any) -> int:
    """Rank of a binary matrix over GF(2)."""
    return int(row_reduce(m)[1].size)


def null_space(m: Any, *, reverse: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Basis of the null space ``{x : m x = 0}`` over GF(2).

    Parameters
    ----------
    m : array_like
        Binary matrix ``(r, n)`` on the host.
    reverse : bool, default False
        As in :func:`row_reduce`: the free columns (where the basis is the
        identity) are then the leftmost ones.

    Returns
    -------
    basis : numpy.ndarray
        int8 ``(n - rank, n)``, rows independent, ``m @ basis.T = 0`` mod 2.
    free : numpy.ndarray
        int64 free columns, ``(n - rank,)``: ``basis[:, free]`` is the
        identity.
    """
    rref, pivots = row_reduce(m, reverse=reverse)
    n = rref.shape[1]
    free = np.setdiff1d(np.arange(n), pivots)
    basis = np.zeros((free.size, n), dtype=np.int8)
    basis[np.arange(free.size), free] = 1
    # x_pivot[i] = sum over free f of rref[i, f] x_f.
    basis[:, pivots] = rref[:, free].T
    return basis, free


def inverse(m: Any) -> np.ndarray:
    """Inverse of a square binary matrix over GF(2); raises if singular."""
    a = np.asarray(m)
    k = a.shape[0]
    if a.shape != (k, k):
        raise ValueError(f"expected a square matrix, got shape {a.shape}.")
    rref, pivots = row_reduce(np.concatenate([a, np.eye(k, dtype=a.dtype)], axis=1))
    if pivots.size < k or pivots[k - 1] != k - 1:
        raise ValueError("matrix is singular over GF(2).")
    return rref[:, k:]


def matmul_mod2(a: ArrayType, b: ArrayType) -> ArrayType:
    """Product of bit arrays modulo 2, ``(a @ b) & 1``, on ``a``'s device.

    One float32 GEMM per chunk of rows of ``a`` (BLAS on the CPU, cuBLAS on
    the GPU), exact while the inner dimension is below 2^24.

    Parameters
    ----------
    a : array_like
        Bits ``(..., r)``, values 0 and 1 (any integer or float dtype).
    b : array_like
        Bits ``(r, c)`` on ``a``'s device; float32 avoids a cast per call.

    Returns
    -------
    ndarray
        int8 ``(..., c)`` on ``a``'s device.
    """
    a, xp, _ = dispatch(a)
    b = xp.asarray(b)
    inner = a.shape[-1]
    if b.ndim != 2 or b.shape[0] != inner:
        raise ValueError(f"cannot multiply {a.shape} by {b.shape}.")
    if inner >= _MAX_INNER:
        raise ValueError(f"inner dimension {inner} exceeds the float32 limit 2^24.")
    if b.dtype != xp.float32:
        b = b.astype(xp.float32)
    lead = a.shape[:-1]
    a2 = a.reshape(-1, inner)
    out = xp.empty((a2.shape[0], b.shape[1]), dtype=xp.int8)
    rows = max(1, _CHUNK_ELEMENTS // max(inner, b.shape[1], 1))
    for r0 in range(0, a2.shape[0], rows):
        prod = a2[r0 : r0 + rows].astype(xp.float32) @ b
        out[r0 : r0 + rows] = prod.astype(xp.int32) & 1
    return out.reshape(*lead, b.shape[1])
