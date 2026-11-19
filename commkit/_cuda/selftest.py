"""Self-test kernel of the CUDA infrastructure (private).

``selftest_scale`` exercises compiling, specializing and launching; no DSP
code uses it.
"""

from collections.abc import Callable
from typing import Any


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
        """``alpha * x`` elementwise, in the factory's dtype."""
        x = cp.ascontiguousarray(x)
        if x.dtype != np_dtype:
            raise TypeError(f"expected {np_dtype} input, got {x.dtype}")
        y = cp.empty_like(x)
        n = x.size
        grid = (n + block - 1) // block
        kern((grid,), (block,), (x, y, np_dtype.type(alpha), cp.int64(n)))
        return y

    return launch
