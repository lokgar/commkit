"""Hand-written CUDA kernels for CuPy input (private).

Raw CUDA C++ kernels (under ``src/``) and their lazy compilation through
:class:`cupy.RawModule`.  :func:`get_kernel` is the one entry point: it
returns a launchable wrapper for a kernel, and callers use it only on CuPy
input, so CuPy already works.  A kernel that fails to compile or load
raises: there is no slower path to fall back to, since the kernels are
tested on a GPU before every release.

Importing this package (or ``commkit``) never touches NVRTC: kernel sources
are read and compiled on the first ``get_kernel()`` call for a given
specialization and cached in-process afterwards.  Cross-process reuse comes
from CuPy's on-disk kernel cache (``~/.cupy/kernel_cache``).
"""

from collections.abc import Callable
from typing import Any


def get_kernel(name: str, **spec: Any) -> Callable:
    """Return a launchable kernel wrapper.

    Parameters
    ----------
    name : str
        Registered kernel name (see ``_KERNEL_FACTORIES``).
    **spec
        Kernel-specific specialization options (e.g. mode or dtype),
        forwarded to the kernel's wrapper factory.

    Returns
    -------
    callable
        A wrapper that handles grid/block computation and input
        validation. Call sites pass CuPy arrays and plain Python scalars
        only.

    Raises
    ------
    KeyError
        If `name` is not a registered kernel.
    ValueError
        If ``spec`` is not a valid specialization of the kernel.
    """
    factory = _KERNEL_FACTORIES.get(name)
    if factory is None:
        raise KeyError(
            f"Unknown CUDA kernel {name!r}; registered: {sorted(_KERNEL_FACTORIES)}"
        )
    return factory(**spec)


def _selftest_scale_factory(dtype: str = "float32") -> Callable:
    """Wrapper factory for the infrastructure self-test kernel.

    Exercises the whole path of compiling, specializing and launching; no
    DSP code uses it.
    """
    import cupy as cp

    from . import compiler

    ctype = {"float32": "float", "float64": "double"}[dtype]
    kern = compiler.get_raw_kernel("selftest", f"selftest_scale<{ctype}>")
    np_dtype = cp.dtype(dtype)
    block = 256

    def launch(x: Any, alpha: float) -> Any:
        x = cp.ascontiguousarray(x)
        if x.dtype != np_dtype:
            raise TypeError(f"expected {np_dtype} input, got {x.dtype}")
        y = cp.empty_like(x)
        n = x.size
        grid = (n + block - 1) // block
        kern((grid,), (block,), (x, y, np_dtype.type(alpha), cp.int64(n)))
        return y

    return launch


def _bps_min_d2_factory(mode: str = "table", return_argmin: bool = False) -> Callable:
    """Wrapper factory for the fused BPS minimum-distance kernel.

    Computes ``min_d2[p, c, n] = min_m |x[c, n] * phasor[p] - const[m]|**2``
    in one pass. ``mode="table"`` searches an explicit constellation table;
    ``mode="grid"`` snaps to the uniform square-QAM level grid. With
    ``return_argmin=True`` (TABLE only) the nearest-point indices are
    returned alongside the distances.
    """
    import cupy as cp

    from . import compiler

    mode_id = {"table": 0, "grid": 1}[mode]
    if mode_id == 1 and return_argmin:
        raise ValueError("return_argmin is only supported in TABLE mode")
    kern = compiler.get_raw_kernel(
        "bps_min_d2",
        f"bps_min_d2<{mode_id}, {'true' if return_argmin else 'false'}>",
    )
    block_n = 128
    _empty_i32 = cp.empty(0, dtype=cp.int32)

    def launch(
        x: Any,
        phasor: Any,
        constellation: Any = None,
        lev_min: float = 0.0,
        d_grid: float = 1.0,
        side: int = 0,
    ) -> Any:
        """Return min_d2 (P, C, N) float32; with argmin, an (min_d2, idx) tuple.

        `x` is (C, N) complex64 with time on the last axis; `phasor` is
        (P,) complex64, P <= 128. TABLE mode requires `constellation`
        ((M,) complex64, M <= 1024); GRID mode requires the level-grid
        scalars `lev_min`, `d_grid`, `side`.
        """
        x = cp.ascontiguousarray(x)
        phasor = cp.ascontiguousarray(phasor)
        if x.ndim != 2:
            raise ValueError(f"x must be 2-D (C, N), got shape {x.shape}")
        if x.dtype != cp.complex64 or phasor.dtype != cp.complex64:
            raise TypeError(
                f"x and phasor must be complex64, got {x.dtype}/{phasor.dtype}"
            )
        C, N = x.shape
        P = int(phasor.size)
        if P < 1 or P > 128:
            raise ValueError(f"phasor count must be in [1, 128], got {P}")

        if mode_id == 0:
            constellation = cp.ascontiguousarray(constellation)
            if constellation.dtype != cp.complex64:
                raise TypeError(
                    f"constellation must be complex64, got {constellation.dtype}"
                )
            M = int(constellation.size)
            if M < 1 or M > 1024:
                raise ValueError(f"constellation size must be in [1, 1024], got {M}")
            shared_mem = M * 8
        else:
            if side < 2:
                raise ValueError(f"GRID mode requires side >= 2, got {side}")
            constellation = _empty_i32  # never dereferenced (M == 0)
            M = 0
            shared_mem = 0

        min_d2 = cp.empty((P, C, N), dtype=cp.float32)
        argmin = cp.empty((P, C, N), dtype=cp.int32) if return_argmin else _empty_i32
        grid = ((N + block_n - 1) // block_n, P, C)
        kern(
            grid,
            (block_n,),
            (
                x,
                phasor,
                constellation,
                cp.float32(lev_min),
                cp.float32(d_grid),
                cp.int32(side),
                cp.int32(M),
                cp.int64(N),
                min_d2,
                argmin,
            ),
            shared_mem=shared_mem,
        )
        if return_argmin:
            return min_d2, argmin
        return min_d2

    return launch


def _cs_block_factory() -> Callable:
    """Wrapper factory for the block_lms cycle-slip correction kernel.

    Per-channel slip detector (one block of threads per channel, decisions
    speculated in parallel; see ``cs_block.cu``) operating in place on
    device-resident state buffers. One launch processes one equalizer block,
    instead of a round trip per block through the CPU path (device to host,
    the Numba kernel, host to device).

    The wrapper binds one run: it checks the state buffers (float64 or
    int64, changed in place, so non-contiguous ones are rejected instead of
    copied silently) and allocates the kernel's scratch for the largest
    block once, and returns the per-block launch, which checks only the
    block's phases.
    """
    import cupy as cp

    from . import compiler

    kern = compiler.get_raw_kernel("cs_block", "cs_block")

    def bind(
        cs_buf_y: Any,
        cs_buf_ptr: Any,
        cs_buf_n: Any,
        cs_stats: Any,
        *,
        quantum: float,
        threshold: float,
        cs_H: int,
        max_block: int,
    ) -> Callable[[Any, Any], Any]:
        """Bind the state buffers of one run, ``C`` channels and blocks of at
        most ``max_block`` symbols."""
        C = cs_buf_ptr.shape[0] if cs_buf_ptr.ndim == 1 else -1
        cs_H = int(cs_H)
        for label, arr, dtype, shape in (
            ("cs_buf_y", cs_buf_y, cp.float64, (C, cs_H)),
            ("cs_buf_ptr", cs_buf_ptr, cp.int64, (C,)),
            ("cs_buf_n", cs_buf_n, cp.int64, (C,)),
            ("cs_stats", cs_stats, cp.float64, (C, 4)),
        ):
            if arr.dtype != dtype:
                raise TypeError(f"{label} must be {dtype}, got {arr.dtype}")
            if arr.shape != shape:
                raise ValueError(f"{label} must have shape {shape}, got {arr.shape}")
            if not arr.flags.c_contiguous:
                raise ValueError(f"{label} must be C-contiguous (mutated in place)")
        if max_block < 1:
            raise ValueError(f"max_block must be >= 1, got {max_block}")
        # Scratch for the largest block: the kernel strides by the actual B.
        L = cs_H + max_block
        z = cp.empty(C * L, dtype=cp.float64)
        p1 = cp.empty(C * (L + 1), dtype=cp.float64)
        p2 = cp.empty(C * (L + 1), dtype=cp.float64)
        guess = cp.empty(C * max_block, dtype=cp.int32)
        decided = cp.empty(C * max_block, dtype=cp.int32)
        scalars = (cp.float64(quantum), cp.float64(threshold), cp.int32(cs_H))

        def launch(phi_blk: Any, phi_corr: Any) -> Any:
            """Repair the cycle slips of ``phi_blk`` into ``phi_corr``.

            Both are ``(C, B)`` float64, ``1 <= B <= max_block``; the bound
            history buffers are updated in place.
            """
            shape = phi_blk.shape
            if (
                phi_blk.dtype != cp.float64
                or len(shape) != 2
                or shape[0] != C
                or not 1 <= shape[1] <= max_block
                or not phi_blk.flags.c_contiguous
            ):
                raise ValueError(
                    f"phi_blk must be C-contiguous float64 (C={C}, B<={max_block}), "
                    f"got {phi_blk.dtype} {shape}"
                )
            if (
                phi_corr.dtype != cp.float64
                or phi_corr.shape != shape
                or not phi_corr.flags.c_contiguous
            ):
                raise ValueError(
                    f"phi_corr must be C-contiguous float64 {shape}, got "
                    f"{phi_corr.dtype} {phi_corr.shape}"
                )
            kern(
                (C,),
                (256,),  # CS_THREADS in cs_block.cu
                (
                    phi_blk, phi_corr, cs_buf_y, cs_buf_ptr, cs_buf_n, cs_stats,
                    z, p1, p2, guess, decided, *scalars, cp.int32(shape[1]),
                ),
            )  # fmt: skip
            return phi_corr

        return launch

    return bind


def _bps_anchor_factory() -> Callable:
    """Wrapper factory for the block_lms data-aided BPS anchor kernel.

    One launch scans the training columns of one equalizer block (one block
    of 256 threads), keeping the block loop free of the host round
    trip of the NumPy reference (``_BlockBps._anchor``).  ``offset4`` and ``prev4``
    are updated in place; the anchored phase and the new trailing products
    (complex64) are returned.  The window sums are float32 and the unwrap
    float64.
    """
    import cupy as cp

    from . import compiler

    kern = compiler.get_raw_kernel("bps_anchor", "bps_anchor")

    def launch(
        y: Any,
        d: Any,
        hist: Any,
        offset4: Any,
        prev4: Any,
        *,
        symmetry: int,
        joint: bool,
    ) -> tuple[Any, Any]:
        """Anchored phase ``(C, n)`` float64 and the new ``(C, K-1)`` history."""
        if y.ndim != 2:
            raise ValueError(f"y must be 2-D (C, n), got shape {y.shape}")
        C, n = y.shape
        if n < 1:
            raise ValueError("y must have at least one training column")
        H = hist.shape[-1]
        for label, arr, dtype, shape in (
            ("y", y, cp.complex64, (C, n)),
            ("d", d, cp.complex64, (C, n)),
            ("hist", hist, cp.complex64, (C, H)),
            ("offset4", offset4, cp.float64, (C,)),
            ("prev4", prev4, cp.float64, (C,)),
        ):
            if arr.dtype != dtype:
                raise TypeError(f"{label} must be {dtype}, got {arr.dtype}")
            if arr.shape != shape:
                raise ValueError(f"{label} must have shape {shape}, got {arr.shape}")
            if not arr.flags.c_contiguous:
                raise ValueError(f"{label} must be C-contiguous")
        phi = cp.empty((C, n), dtype=cp.float64)
        hist_out = cp.empty((C, H), dtype=cp.complex64)
        total = cp.zeros(n if joint else 0, dtype=cp.complex64)
        kern(
            (1,),
            (256,),
            (
                y,
                d,
                hist,
                hist_out,
                total,
                offset4,
                prev4,
                phi,
                cp.int32(C),
                cp.int32(n),
                cp.int32(H + 1),
                cp.int32(1 if joint else 0),
                cp.float64(symmetry),
            ),
        )
        return phi, hist_out

    return launch


# LLR_MAX_K and LLR_THREADS in llr.cu.
_LLR_MAX_K = 16
_LLR_THREADS = 128


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
    out = cp.empty((n, k), dtype=cp.float32)
    if n == 0:
        return out
    blocks = min((n + _LLR_THREADS - 1) // _LLR_THREADS, 65535)
    kern(
        (blocks,),
        (_LLR_THREADS,),
        (*head, out, cp.int32(n), *tail),
        shared_mem=_LLR_THREADS * (k | 1) * 4,  # llr_stride(k) in llr.cu
    )
    return out


def _llr_factory() -> Callable:
    """Wrapper factory for the bit-LLR kernel (``mapping.llr``).

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
        if not 1 <= k <= _LLR_MAX_K:
            raise ValueError(f"bits per symbol must be in [1, {_LLR_MAX_K}], got {k}")
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
    """Wrapper factory for the per-axis bit-LLR kernel (``mapping.llr``).

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
            if not 1 <= kb <= _LLR_MAX_K:
                raise ValueError(
                    f"bits of the {name} axis must be in [1, {_LLR_MAX_K}], got {kb}"
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
        if k > _LLR_MAX_K:
            raise ValueError(f"bits per symbol must be at most {_LLR_MAX_K}, got {k}")
        return _llr_launch(
            cp,
            kern,
            n,
            k,
            (x, *axis_args),
            (cp.int32(k), cp.float32(inv_s2), cp.int32(1 if exact else 0)),
        )

    return launch


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
        if np.shape(transitions) != (group + 1, D, D):
            raise ValueError(f"transitions must be ({group + 1}, {D}, {D})")
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


# name -> wrapper factory.
_KERNEL_FACTORIES: dict[str, Callable[..., Callable]] = {
    "selftest_scale": _selftest_scale_factory,
    "bps_min_d2": _bps_min_d2_factory,
    "cs_block": _cs_block_factory,
    "bps_anchor": _bps_anchor_factory,
    "llr": _llr_factory,
    "llr_separable": _llr_separable_factory,
    "iir": _iir_factory,
}
