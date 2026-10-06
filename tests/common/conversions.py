"""Array conversion and precision helpers for tests."""

from typing import Any

import numpy as np

try:
    import cupy as cp

    _CUPY_AVAILABLE = True
except ImportError:
    cp = None
    _CUPY_AVAILABLE = False


def to_numpy(arr: Any) -> np.ndarray:
    """Convert any array (NumPy or CuPy) to a host numpy.ndarray.

    Parameters
    ----------
    arr : Any
        Array-like or scalar object.

    Returns
    -------
    np.ndarray
        NumPy array on CPU.
    """
    if arr is None:
        return None  # type: ignore[return-value]
    if _CUPY_AVAILABLE and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    if hasattr(arr, "get") and callable(arr.get):
        try:
            return np.asarray(arr.get())
        except Exception:
            pass
    return np.asarray(arr)


def device_of(xp: Any) -> str:
    """Device name for an array module: ``"gpu"`` for CuPy, else ``"cpu"``.

    Library generators return host arrays; tests move them explicitly with
    ``sig.to(device_of(xp))`` or ``to_device(arr, device_of(xp))``.
    """
    return "gpu" if xp is not np else "cpu"
