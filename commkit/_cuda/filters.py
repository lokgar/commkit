"""Launch wrappers of the filter and interpolation kernels (private).

``iir`` (the block-parallel SOS filter of :mod:`commkit.filtering.linear`)
and ``interpolate`` (the windowed sinc of :mod:`commkit._interpolation`).
"""

from collections.abc import Callable
from typing import Any

# IIR_MAX_S in iir.cu, and the threads per block of its launches.
_IIR_MAX_S = 16
_IIR_THREADS = 128
# Sample dtype -> (sample type, state type) of the iir.cu templates.
_IIR_TYPES = {
    "float32": ("float", "double"),
    "float64": ("double", "double"),
    "complex64": ("complex<float>", "complex<double>"),
    "complex128": ("complex<double>", "complex<double>"),
}


def _iir_factory() -> Callable:
    """Wrapper factory for the block-parallel SOS filter (``filtering.linear``).

    Filters the rows of a C-contiguous ``(C, N)`` record through the
    sections in double precision, reading and writing the samples in their
    own dtype (see ``iir.cu``).
    """
    import cupy as cp
    import numpy as np

    from . import compiler

    kernels: dict[tuple[str, str, str], Any] = {}

    def kernel(name: str, t: str, u: str) -> Any:
        """The compiled ``name`` kernel, cached per specialization.

        ``t`` is the sample type and ``u`` the state type of the template.
        """
        key = (name, t, u)
        if key not in kernels:
            carry = name in ("iir_carry", "iir_group_carry")
            expr = f"{name}<{u}>" if carry else f"{name}<{t}, {u}>"
            kernels[key] = compiler.get_raw_kernel("iir", expr)
        return kernels[key]

    def launch(
        sos: Any, x: Any, zi: Any, block: int, group: int, transitions: Any
    ) -> Any:
        """``y`` of the causal filter of ``x`` from the initial state ``zi``.

        ``sos`` is ``(S, 6)`` float64 on the host, ``x`` ``(C, N)`` float32,
        float64, complex64 or complex128, ``zi`` ``(C, S, 2)`` float64 (real
        ``x``) or complex128 (complex ``x``).  ``transitions`` is the host
        float64 ``(group + 1, 2S, 2S)`` stack of the zero-input state
        transitions of the cascade over ``j·block`` samples, ``j = 0 ..
        group`` (state index ``2k + i``).
        """
        sos = np.ascontiguousarray(sos, dtype=np.float64)
        if sos.ndim != 2 or sos.shape[1] != 6 or not 1 <= sos.shape[0] <= _IIR_MAX_S:
            raise ValueError(
                f"sos must be (S, 6) with 1 <= S <= {_IIR_MAX_S}, got {sos.shape}"
            )
        if x.dtype.name not in _IIR_TYPES:
            raise TypeError(f"x must be float32/64 or complex64/128, got {x.dtype}")
        if x.ndim != 2 or not x.flags.c_contiguous:
            raise ValueError(f"x must be a C-contiguous (C, N) array, got {x.shape}")
        S = sos.shape[0]
        C, N = x.shape
        state = cp.complex128 if x.dtype.kind == "c" else cp.float64
        if zi.dtype != state or zi.shape != (C, S, 2) or not zi.flags.c_contiguous:
            raise ValueError(
                f"zi must be a C-contiguous ({C}, {S}, 2) {np.dtype(state)} array"
            )
        D = 2 * S
        if block < 1 or group < 1:
            raise ValueError(f"block and group must be >= 1, got {block}, {group}")
        if np.shape(transitions) != (group + 1, D, D):
            raise ValueError(f"transitions must be ({group + 1}, {D}, {D})")
        if x.size == 0:  # a zero-size grid is a launch error
            return cp.empty_like(x)
        t, u = _IIR_TYPES[x.dtype.name]
        L, G = int(block), int(group)
        nb = -(-N // L)
        ng = -(-nb // G)
        tr = cp.asarray(np.ascontiguousarray(transitions, dtype=np.float64))
        sos_dev = cp.asarray(sos)
        zend = cp.empty((C, nb, S, 2), dtype=state)
        zloc = cp.empty_like(zend)
        zgend = cp.empty((C, ng, S, 2), dtype=state)
        zgin = cp.empty_like(zgend)
        y = cp.empty_like(x)

        def grid(threads: int) -> tuple[int]:
            """Blocks of ``_IIR_THREADS`` covering ``threads`` threads."""
            return (-(-threads // _IIR_THREADS),)

        th = (_IIR_THREADS,)
        common = (sos_dev, cp.int32(S), x, cp.int64(N), cp.int32(L), cp.int32(nb))
        kernel("iir_block_state", t, u)(grid(C * nb), th, (*common, cp.int32(C), zend))
        kernel("iir_group_carry", t, u)(
            grid(C * ng),
            th,
            (tr[1], cp.int32(S), zend, cp.int32(nb), cp.int32(G), cp.int32(ng),
             cp.int32(C), zloc, zgend),
        )  # fmt: skip
        kernel("iir_carry", t, u)(
            grid(C),
            th,
            (tr[G], cp.int32(S), zi, zgend, cp.int32(ng), cp.int32(C), zgin),
        )
        kernel("iir_block_out", t, u)(
            grid(C * nb),
            th,
            (*common, cp.int32(C), tr[:G], cp.int32(G), cp.int32(ng), zloc, zgin, y),
        )
        return y

    return launch


#: Sample dtype -> (sample type, real type) of the ``interpolate`` kernel.
_INTERP_TYPES = {
    "float32": ("float", "float"),
    "float64": ("double", "double"),
    "complex64": ("complex<float>", "float"),
    "complex128": ("complex<double>", "double"),
}


def _interpolate_factory(half: int = 24) -> Callable:
    """Wrapper factory for the windowed-sinc interpolator (``_interpolation``).

    ``x`` ``(C, N)`` at arbitrary instants, ``2·half`` taps, one thread per
    instant (see ``interpolate.cu``).  The wrapper returned takes
    ``(x, base=, frac=)`` (an index and a fraction per instant, shared by the
    channels) or ``(x, knots=, sub=)`` (instants linear between float64
    knots, one row or one per channel).
    """
    import cupy as cp

    from . import compiler

    if half < 1:
        raise ValueError(f"half must be >= 1, got {half}")
    kernels: dict[tuple[str, int], Any] = {}
    block = 256

    def kernel(dtype: str, mode: int) -> Any:
        """The compiled kernel, cached per sample dtype and mode."""
        key = (dtype, mode)
        if key not in kernels:
            t, r = _INTERP_TYPES[dtype]
            kernels[key] = compiler.get_raw_kernel(
                "interpolate", f"interpolate<{t}, {r}, {half}, {mode}>"
            )
        return kernels[key]

    def launch(
        x: Any,
        *,
        base: Any = None,
        frac: Any = None,
        knots: Any = None,
        sub: int = 0,
    ) -> Any:
        """``(C, M)`` in ``x``'s dtype: ``M`` the length of ``base`` and
        ``frac`` (int64 and ``x``'s real dtype), or ``sub · (K - 1)`` for
        ``knots`` ``(rows, K)`` float64, ``rows`` 1 or ``C``."""
        if x.dtype.name not in _INTERP_TYPES:
            raise TypeError(f"x must be float32/64 or complex64/128, got {x.dtype}")
        if x.ndim != 2 or not x.flags.c_contiguous:
            raise ValueError(f"x must be a C-contiguous (C, N) array, got {x.shape}")
        C, N = x.shape
        real = x.real.dtype
        null_i64 = cp.empty(0, dtype=cp.int64)
        null_r = cp.empty(0, dtype=real)
        null_f64 = cp.empty(0, dtype=cp.float64)
        if knots is None:
            if base is None or frac is None:
                raise ValueError("give base and frac, or knots and sub")
            base = cp.ascontiguousarray(base)
            frac = cp.ascontiguousarray(frac)
            if base.dtype != cp.int64 or frac.dtype != real:
                raise TypeError(
                    f"base must be int64 and frac {real}, got {base.dtype}/{frac.dtype}"
                )
            if base.ndim != 1 or base.shape != frac.shape:
                raise ValueError(
                    f"base and frac must be (M,) alike, got {base.shape}/{frac.shape}"
                )
            M, mode, rows, K = base.shape[0], 0, 0, 0
            args = (base, frac, null_f64)
        else:
            knots = cp.ascontiguousarray(knots)
            if knots.dtype != cp.float64 or knots.ndim != 2:
                raise TypeError(f"knots must be 2-D float64, got {knots.dtype}")
            rows, K = knots.shape
            if rows not in (1, C) or K < 1 or sub < 1:
                raise ValueError(
                    f"knots must be (1, K) or ({C}, K) and sub >= 1, "
                    f"got {knots.shape}, sub={sub}"
                )
            M, mode = int(sub) * (K - 1), 1
            args = (null_i64, null_r, knots)
        y = cp.empty((C, M), dtype=x.dtype)
        if y.size == 0:  # a zero-size grid is a launch error
            return y
        grid = (M + block - 1) // block
        kernel(x.dtype.name, mode)(
            (grid,),
            (block,),
            (x, y, cp.int64(C), cp.int64(N), cp.int64(M), *args,
             cp.int64(rows), cp.int64(K), cp.int64(sub)),
        )  # fmt: skip
        return y

    return launch
