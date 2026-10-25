"""Hand-written CUDA kernels for CuPy input (private).

Raw CUDA C++ kernels (under ``src/``) and their lazy compilation through
:class:`cupy.RawModule`.  Two functions are used by the rest of the library:

is_available :
    True when CuPy is functional and a CUDA device with compute
    capability >= 7.0 is present.
get_kernel :
    A launchable kernel wrapper, or ``None``, in which case the caller
    **must** fall back to its ``xp`` implementation.

Importing this package (or ``commkit``) never touches NVRTC: kernel sources
are read and compiled on the first ``get_kernel()`` call for a given
specialization and cached in-process afterwards.  Cross-process reuse comes
from CuPy's on-disk kernel cache (``~/.cupy/kernel_cache``).
"""

from collections.abc import Callable
from functools import lru_cache
from typing import Any

from ..backend import is_cupy_available
from ..logger import logger

# Volta and newer. Older parts lack the independent-thread-scheduling and
# shared-memory sizes the kernels in src/ are written against.
_MIN_COMPUTE_CAPABILITY = 70

# Kernel names that have already produced a compile/load warning; the
# fallback contract promises at most one warning per process per kernel.
_warned_kernels: set[str] = set()


@lru_cache(maxsize=1)
def _device_supported() -> bool:
    """Hardware probe, cached for the process lifetime.

    Checks that at least one CUDA device is present and that the current
    device has compute capability >= 7.0. Only called when CuPy itself is
    importable and functional.
    """
    import cupy as cp

    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            return False
        cc = int(cp.cuda.Device().compute_capability)
    except Exception:
        return False
    return cc >= _MIN_COMPUTE_CAPABILITY


def is_available() -> bool:
    """Check whether custom CUDA kernels can be compiled and launched.

    Returns
    -------
    bool
        True if CuPy is installed and functional, at least one CUDA device is
        present, and the current device has compute capability >= 7.0.
    """
    if not is_cupy_available():
        return False
    return _device_supported()


def get_kernel(name: str, **spec: Any) -> Callable | None:
    """Return a launchable kernel wrapper, or ``None`` if unavailable.

    With ``None`` the caller falls back to its ``xp`` implementation.  A
    compile or load failure logs one warning per kernel name and process and
    returns ``None``.

    Parameters
    ----------
    name : str
        Registered kernel name (see ``_KERNEL_FACTORIES``).
    **spec
        Kernel-specific specialization options (e.g. mode or dtype),
        forwarded to the kernel's wrapper factory.

    Returns
    -------
    callable or None
        A wrapper that handles grid/block computation and input
        validation. Call sites pass CuPy arrays and plain Python scalars
        only. ``None`` when no usable GPU is present or compilation fails.

    Raises
    ------
    KeyError
        If `name` is not a registered kernel. Unknown names are
        programming errors, not runtime fallback conditions.
    """
    factory = _KERNEL_FACTORIES.get(name)
    if factory is None:
        raise KeyError(
            f"Unknown CUDA kernel {name!r}; registered: {sorted(_KERNEL_FACTORIES)}"
        )
    if not is_available():
        return None
    try:
        return factory(**spec)
    except Exception as exc:
        if name not in _warned_kernels:
            _warned_kernels.add(name)
            logger.warning(
                "CUDA kernel %r failed to compile/load (%s); "
                "falling back to the array-module implementation.",
                name,
                exc,
            )
        return None


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
    replacing the fallback's round trip per block (device to host, the Numba
    kernel, host to device).  All state arrays are float64 or int64 and are
    changed in place, so the wrapper rejects non-contiguous inputs instead
    of copying them silently.
    """
    import cupy as cp

    from . import compiler

    kern = compiler.get_raw_kernel("cs_block", "cs_block")

    def launch(
        phi_blk: Any,
        phi_corr: Any,
        cs_buf_y: Any,
        cs_buf_ptr: Any,
        cs_buf_n: Any,
        cs_stats: Any,
        quantum: float,
        threshold: float,
        cs_H: int,
    ) -> Any:
        """Repair the cycle slips of ``phi_blk`` into ``phi_corr``.

        Both are ``(C, B)`` float64; the per-channel circular history
        buffers are updated in place.
        """
        if phi_blk.ndim != 2:
            raise ValueError(f"phi_blk must be 2-D (C, B), got shape {phi_blk.shape}")
        C, B = phi_blk.shape
        if B < 1:
            raise ValueError("phi_blk must have at least one column")
        cs_H = int(cs_H)
        for label, arr, dtype, shape in (
            ("phi_blk", phi_blk, cp.float64, (C, B)),
            ("phi_corr", phi_corr, cp.float64, (C, B)),
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
        L = cs_H + B
        kern(
            (C,),
            (256,),  # CS_THREADS in cs_block.cu
            (
                phi_blk,
                phi_corr,
                cs_buf_y,
                cs_buf_ptr,
                cs_buf_n,
                cs_stats,
                cp.empty((C, L), dtype=cp.float64),
                cp.empty((C, L + 1), dtype=cp.float64),
                cp.empty((C, L + 1), dtype=cp.float64),
                cp.empty((C, B), dtype=cp.int32),
                cp.empty((C, B), dtype=cp.int32),
                cp.float64(quantum),
                cp.float64(threshold),
                cp.int32(cs_H),
                cp.int32(B),
            ),
        )
        return phi_corr

    return launch


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


# name -> wrapper factory. Factories may raise; get_kernel translates any
# failure into the warn-once-and-return-None fallback contract.
_KERNEL_FACTORIES: dict[str, Callable[..., Callable]] = {
    "selftest_scale": _selftest_scale_factory,
    "bps_min_d2": _bps_min_d2_factory,
    "cs_block": _cs_block_factory,
    "bps_anchor": _bps_anchor_factory,
    "llr": _llr_factory,
    "llr_separable": _llr_separable_factory,
}
