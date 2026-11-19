"""Launch wrappers of the bit-LLR kernels of :mod:`commkit.mapping._demap` (private).

``llr`` (all points), ``llr_separable`` (per axis of a square lattice) and
``llr_separable_maxlog`` (its closed-form max-log); at most ``LLR_MAX_K``
bits per symbol.
"""

from collections.abc import Callable
from typing import Any

# LLR_MAX_K and LLR_THREADS in llr.cu.
LLR_MAX_K = 16
LLR_THREADS = 128


def _check_llr_arrays(arrays: tuple[tuple[str, Any, Any, tuple], ...]) -> None:
    """Check the dtype, shape and contiguity of LLR kernel arguments."""
    for label, arr, dtype, shape in arrays:
        if arr.dtype != dtype:
            raise TypeError(f"{label} must be {dtype}, got {arr.dtype}")
        if arr.shape != shape:
            raise ValueError(f"{label} must have shape {shape}, got {arr.shape}")
        if not arr.flags.c_contiguous:
            raise ValueError(f"{label} must be C-contiguous")


def _llr_launch(cp: Any, kern: Any, n: int, k: int, head: tuple, tail: tuple) -> Any:
    """Launch an LLR kernel over ``n`` symbols; returns ``(n, k)`` float32.

    The kernel's arguments are ``head``, the output, ``n``, then ``tail``.
    """
    if n >= 2**31:
        raise ValueError(f"at most 2**31 - 1 symbols per call, got {n}")
    out = cp.empty((n, k), dtype=cp.float32)
    if n == 0:
        return out
    blocks = min((n + LLR_THREADS - 1) // LLR_THREADS, 65535)
    kern(
        (blocks,),
        (LLR_THREADS,),
        (*head, out, cp.int32(n), *tail),
        shared_mem=LLR_THREADS * (k | 1) * 4,  # llr_stride(k) in llr.cu
    )
    return out


def _llr_factory() -> Callable:
    """Wrapper factory for the bit-LLR kernel (``mapping._demap``).

    One thread per symbol walks the constellation and keeps the per-bit
    maxima (and, for the exact method, the sums of one exponential per
    point) in registers, so no ``(N, k, M/2)`` intermediate is formed.
    Returns ``(n, k)`` float32 LLRs.
    """
    import cupy as cp

    from . import compiler

    kern = compiler.get_raw_kernel("llr", "llr")

    def launch(
        x: Any,
        points: Any,
        log_pmf: Any,
        labels: Any,
        *,
        k: int,
        inv_s2: float,
        exact: bool,
    ) -> Any:
        """LLRs of the ``(n,)`` complex64 symbols ``x``, shape ``(n, k)``."""
        if not 1 <= k <= LLR_MAX_K:
            raise ValueError(f"bits per symbol must be in [1, {LLR_MAX_K}], got {k}")
        n = x.shape[0]
        M = points.shape[0]
        _check_llr_arrays(
            (
                ("x", x, cp.complex64, (n,)),
                ("points", points, cp.complex64, (M,)),
                ("log_pmf", log_pmf, cp.float32, (M,)),
                ("labels", labels, cp.int32, (M,)),
            ),
        )
        return _llr_launch(
            cp,
            kern,
            n,
            k,
            (x, points, log_pmf, labels),
            (cp.int32(M), cp.int32(k), cp.float32(inv_s2), cp.int32(1 if exact else 0)),
        )

    return launch


def _llr_separable_factory() -> Callable:
    """Wrapper factory for the per-axis bit-LLR kernel (``mapping._demap``).

    The LLRs of a separable square lattice from the levels of each axis:
    2·sqrt(M) metrics (and exponentials) per symbol instead of M.
    Returns ``(n, k)`` float32 LLRs.
    """
    import cupy as cp

    from . import compiler

    kern = compiler.get_raw_kernel("llr", "llr_separable")

    def launch(
        x: Any,
        re: tuple[Any, Any, Any, Any],
        im: tuple[Any, Any, Any, Any],
        *,
        inv_s2: float,
        exact: bool,
    ) -> Any:
        """LLRs of the ``(n,)`` complex64 symbols ``x``, shape ``(n, k)``.

        ``re`` and ``im`` are each ``(levels, log_prior, labels, positions)``
        of one axis: float32, float32 and int32 ``(L,)``, int32 ``(k_axis,)``.
        """
        n = x.shape[0]
        checks = [("x", x, cp.complex64, (n,))]
        axis_args: list[Any] = []
        for name, (lev, logp, lab, pos) in (("re", re), ("im", im)):
            L, kb = lev.shape[0], pos.shape[0]
            if not 1 <= kb <= LLR_MAX_K:
                raise ValueError(
                    f"bits of the {name} axis must be in [1, {LLR_MAX_K}], got {kb}"
                )
            checks += [
                (f"{name} levels", lev, cp.float32, (L,)),
                (f"{name} log prior", logp, cp.float32, (L,)),
                (f"{name} labels", lab, cp.int32, (L,)),
                (f"{name} positions", pos, cp.int32, (kb,)),
            ]
            axis_args += [lev, logp, lab, pos, cp.int32(L), cp.int32(kb)]
        _check_llr_arrays(tuple(checks))
        k = re[3].shape[0] + im[3].shape[0]
        if k > LLR_MAX_K:
            raise ValueError(f"bits per symbol must be at most {LLR_MAX_K}, got {k}")
        return _llr_launch(
            cp,
            kern,
            n,
            k,
            (x, *axis_args),
            (cp.int32(k), cp.float32(inv_s2), cp.int32(1 if exact else 0)),
        )

    return launch


def _llr_separable_maxlog_factory() -> Callable:
    """Wrapper factory for the closed-form max-log LLR kernel (``mapping._demap``).

    Max-log LLRs of a uniform separable square lattice: per axis the snapped
    level's metric against the two levels just outside its run of equal
    bits, 1 + 2k metrics instead of sqrt(M).  Returns ``(n, k)`` float32.
    """
    import cupy as cp

    from . import compiler

    kern = compiler.get_raw_kernel("llr", "llr_separable_maxlog")

    def launch(
        x: Any,
        re: tuple[Any, Any, Any, Any, Any],
        im: tuple[Any, Any, Any, Any, Any],
        *,
        inv_s2: float,
    ) -> Any:
        """LLRs of the ``(n,)`` complex64 symbols ``x``, shape ``(n, k)``.

        ``re`` and ``im`` are each ``(levels, labels, lo, hi, positions)`` of
        one axis: float32 and int32 ``(L,)``, int32 ``(k_axis, L)`` twice,
        int32 ``(k_axis,)``; ``L >= 2`` equally spaced ascending levels.
        """
        n = x.shape[0]
        checks: list[tuple[str, Any, Any, tuple[int, ...]]] = [
            ("x", x, cp.complex64, (n,))
        ]
        axis_args: list[Any] = []
        for name, (lev, lab, lo, hi, pos) in (("re", re), ("im", im)):
            L, kb = lev.shape[0], pos.shape[0]
            if not 1 <= kb <= LLR_MAX_K:
                raise ValueError(
                    f"bits of the {name} axis must be in [1, {LLR_MAX_K}], got {kb}"
                )
            if L < 2:
                raise ValueError(f"the {name} axis needs at least 2 levels, got {L}")
            checks += [
                (f"{name} levels", lev, cp.float32, (L,)),
                (f"{name} labels", lab, cp.int32, (L,)),
                (f"{name} lo", lo, cp.int32, (kb, L)),
                (f"{name} hi", hi, cp.int32, (kb, L)),
                (f"{name} positions", pos, cp.int32, (kb,)),
            ]
            axis_args += [lev, lab, lo, hi, pos, cp.int32(L), cp.int32(kb)]
        _check_llr_arrays(tuple(checks))
        k = re[4].shape[0] + im[4].shape[0]
        if k > LLR_MAX_K:
            raise ValueError(f"bits per symbol must be at most {LLR_MAX_K}, got {k}")
        return _llr_launch(
            cp, kern, n, k, (x, *axis_args), (cp.int32(k), cp.float32(inv_s2))
        )

    return launch
