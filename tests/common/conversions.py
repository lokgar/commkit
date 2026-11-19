"""Array conversion and precision helpers for tests."""

from typing import Any

import numpy as np

from commkit.backend import to_device


def to_numpy(arr: Any) -> np.ndarray:
    """``arr`` (NumPy, CuPy, a list or a scalar) as a host NumPy array.

    ``None`` passes through, for optional result fields.
    """
    if arr is None:
        return None  # type: ignore[return-value]
    return np.asarray(to_device(arr, "cpu"))


def device_of(xp: Any) -> str:
    """Device name for an array module: ``"gpu"`` for CuPy, else ``"cpu"``.

    Library generators return host arrays; tests move them explicitly with
    ``sig.to(device_of(xp))`` or ``to_device(arr, device_of(xp))``.
    """
    return "gpu" if xp is not np else "cpu"
