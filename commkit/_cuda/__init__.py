"""
Hand-written CUDA kernel infrastructure for the CuPy backend.

This subpackage hosts raw CUDA C++ kernels (under ``src/``) together with the
lazy compilation machinery built on :class:`cupy.RawModule`. Its public
surface is two functions:

is_available :
    True when CuPy is functional and a CUDA device with compute
    capability >= 7.0 is present.
get_kernel :
    Returns a launchable kernel wrapper, or ``None`` - in which case the
    caller **must** fall back to its existing ``xp`` implementation.

Importing this package (or ``commkit``) never touches NVRTC: kernel
sources are read and compiled on the first ``get_kernel()`` call for a
given specialization, and cached in-process afterwards. Cross-process
reuse comes from CuPy's on-disk kernel cache (``~/.cupy/kernel_cache``).
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
    """Checks whether custom CUDA kernels can be compiled and launched.

    Returns
    -------
    bool
        True if CuPy is installed and functional (and not disabled via
        ``backend.use_cpu_only``), at least one CUDA device is present,
        and the current device has compute capability >= 7.0.
    """
    # The CuPy check is evaluated fresh on every call so that
    # backend.use_cpu_only() is honored; only the hardware probe is cached.
    if not is_cupy_available():
        return False
    return _device_supported()


def get_kernel(name: str, **spec: Any) -> Callable | None:
    """Returns a launchable kernel wrapper, or ``None`` if unavailable.

    ``None`` means the caller **must** fall back to the existing ``xp``
    implementation. Any compile/load failure logs one warning per kernel
    name per process and returns ``None``.

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

    Validates the full compile -> specialize -> launch path; not used by
    any DSP code.
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
        """Returns min_d2 (P, C, N) float32; with argmin, an (min_d2, idx) tuple.

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

    Sequential per-channel slip detector (one block, one thread per channel)
    operating in-place on device-resident state buffers. One launch processes
    one equalizer block, replacing the per-block D2H -> CPU Numba -> H2D
    round trip of the fallback path. All state arrays are float64/int64 and
    are mutated in place - the wrapper therefore rejects non-contiguous
    inputs instead of silently copying them.
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
        """Corrects phi_blk into phi_corr ((C, B) float64), updating the
        per-channel circular history buffers in place."""
        if phi_blk.ndim != 2:
            raise ValueError(f"phi_blk must be 2-D (C, B), got shape {phi_blk.shape}")
        C, B = phi_blk.shape
        if C > 1024:
            raise ValueError(f"channel count must be <= 1024, got {C}")
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
        kern(
            (1,),
            (C,),
            (
                phi_blk,
                phi_corr,
                cs_buf_y,
                cs_buf_ptr,
                cs_buf_n,
                cs_stats,
                cp.float64(quantum),
                cp.float64(threshold),
                cp.int32(cs_H),
                cp.int32(B),
            ),
        )
        return phi_corr

    return launch


# name -> wrapper factory. Factories may raise; get_kernel translates any
# failure into the warn-once-and-return-None fallback contract.
_KERNEL_FACTORIES: dict[str, Callable[..., Callable]] = {
    "selftest_scale": _selftest_scale_factory,
    "bps_min_d2": _bps_min_d2_factory,
    "cs_block": _cs_block_factory,
}
