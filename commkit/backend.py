"""
Computational backend management and device orchestration.

This module provides backend-agnostic execution on CPU (NumPy) and GPU (CuPy).
The device follows the data: :func:`dispatch` returns the array module of the
input, and data moves only through an explicit :func:`to_device` (or
``Signal.to``).  Arrays from other frameworks are rejected with ``TypeError``;
exchange data with them explicitly through DLPack.
"""

import types
import warnings
from functools import cache
from typing import Any, cast

import numpy as np

from .logger import logger

__all__ = [
    "ArrayType",
    "dispatch",
    "get_array_module",
    "get_scipy_module",
    "is_cupy_available",
    "to_device",
]


@cache
def _cupy() -> types.ModuleType | None:
    """Import CuPy and check it works, once, on the first GPU-related call.

    Importing ``commkit`` never imports CuPy or touches the GPU.  The probe
    allocates and runs one kernel, which catches installations whose CUDA
    libraries (nvrtc, cublas, driver) are missing or broken.
    """
    try:
        import cupy
    except ImportError:
        logger.debug("CuPy is not installed; GPU support is unavailable.")
        return None
    try:
        cupy.arange(1)
    except Exception as exc:
        logger.warning("CuPy is installed but not functional (%s); GPU disabled.", exc)
        return None
    return cast(types.ModuleType, cupy)


def _is_cupy_array(data: Any) -> bool:
    """True for a CuPy array.

    Checks the type's module, so it never imports CuPy: a CuPy array can only
    exist if CuPy is already loaded.
    """
    return type(data).__module__ == "cupy"


# Any for CuPy array to avoid a hard dependency in the type hint if not installed
ArrayType = np.ndarray | Any


def is_cupy_available() -> bool:
    """
    Checks if NVIDIA GPU acceleration is functional via CuPy.

    Data placement never depends on this: arrays stay where the caller put
    them, and only an explicit ``to_device(x, "gpu")`` or ``Signal.to("gpu")``
    moves data to the GPU.

    Returns
    -------
    bool
        True if CuPy is installed and functional.  The first call imports CuPy
        and runs a small probe kernel; the result is cached.
    """
    return _cupy() is not None


def get_array_module(data: Any) -> types.ModuleType:
    """
    Infers the array module (NumPy or CuPy) for the given data.

    The decision is made by inspecting the **actual type of the data**: the
    device follows the data.

    Parameters
    ----------
    data : array_like or list
        The input data to inspect.

    Returns
    -------
    module
        `cupy` if the data is a CuPy device array, otherwise `numpy`
        (CPU arrays, lists, and scalars).
    """
    if _is_cupy_array(data):
        import cupy

        return cast(types.ModuleType, cupy)
    return np


@cache
def get_scipy_module(xp: types.ModuleType) -> types.ModuleType:
    """
    Returns the signal processing library compatible with the given array module.

    Parameters
    ----------
    xp : module
        The array module (typically `numpy` or `cupy`).

    Returns
    -------
    sp : module
        The corresponding signal processing module (`scipy` or `cupyx.scipy`).
    """
    # Match sp to the actual array module so dispatch() returns a consistent
    # (xp, sp) pair.
    if xp.__name__ == "cupy":
        # cupyx.scipy.signal imports CuPy's experimental JIT, which emits a
        # FutureWarning on import; it is CuPy-internal and not actionable for
        # users, so it is silenced here only (no global warning filter).
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*cupyx.jit.rawkernel is experimental.*"
            )
            import cupyx.scipy
            import cupyx.scipy.ndimage
            import cupyx.scipy.signal
            import cupyx.scipy.special

        return cast(types.ModuleType, cupyx.scipy)

    import scipy
    import scipy.ndimage
    import scipy.signal
    import scipy.special

    return cast(types.ModuleType, scipy)


def to_device(data: Any, device: str) -> ArrayType:
    """
    Moves data between CPU and GPU devices.

    Parameters
    ----------
    data : array_like
        The data to move.
    device : {"CPU", "GPU"}
        Target device name (case-insensitive).

    Returns
    -------
    array_like
        The data residing on the target device.

    Raises
    ------
    ImportError
        If "GPU" is requested but CuPy is not available.
    ValueError
        If an unsupported device name is provided.

    Notes
    -----
    If the data is already on the target device, this operation
    typically returns a view or the original array to avoid
    unnecessary copies.
    """
    logger.debug("Moving data to %s.", device.upper())
    device = device.lower()
    if device == "cpu":
        # Dispatch by the *actual array type* (mirrors get_array_module): an
        # array already on the GPU is always brought to the host.
        if _is_cupy_array(data):
            return data.get()
        if isinstance(data, np.ndarray):
            return data
        return np.asarray(data)

    elif device == "gpu":
        cp = _cupy()
        if cp is None:
            raise ImportError("CuPy is not available.")
        if isinstance(data, cp.ndarray):
            return data
        return cp.asarray(data)

    else:
        raise ValueError(f"Unknown device: {device.upper()}")


def dispatch(
    data: Any,
) -> tuple[ArrayType, types.ModuleType, types.ModuleType]:
    """
    Inspects data and returns appropriate backend modules.

    This helper is used throughout the library to implement backend-agnostic
    functional logic.

    Parameters
    ----------
    data : array_like
        The input data to analyze.

    Returns
    -------
    data_array : array_like
        The input data forced to an array on its current device.
    xp : module
        The array module (`numpy` or `cupy`).
    sp : module
        The signal processing module (`scipy` or `cupyx.scipy`).

    Raises
    ------
    TypeError
        If ``data`` is an array from another framework (JAX, PyTorch, ...).

    Notes
    -----
    Dispatch accepts NumPy arrays and scalars, CuPy arrays, and plain Python
    numbers and sequences (converted with ``np.asarray``).  Arrays from other
    frameworks are rejected rather than silently copied: convert them
    explicitly, e.g. ``np.from_dlpack(x)`` or ``cupy.from_dlpack(x)``.
    """
    if not (isinstance(data, np.ndarray | np.generic) or _is_cupy_array(data)):
        _reject_foreign_array(data)
    xp = get_array_module(data)
    sp = get_scipy_module(xp)

    if not (isinstance(data, np.ndarray) or _is_cupy_array(data)):
        data = xp.asarray(data)

    return data, xp, sp


_ARRAY_PROTOCOLS = (
    "__array__",
    "__array_interface__",
    "__dlpack__",
    "__cuda_array_interface__",
)


def _reject_foreign_array(data: Any) -> None:
    """Raise ``TypeError`` for array objects that are not NumPy or CuPy."""
    if any(hasattr(data, attr) for attr in _ARRAY_PROTOCOLS):
        kind = f"{type(data).__module__}.{type(data).__qualname__}"
        raise TypeError(
            f"Unsupported array type {kind}: commkit works on NumPy and CuPy "
            "arrays. Convert explicitly, e.g. np.from_dlpack(x) for host data "
            "or cupy.from_dlpack(x) for GPU data."
        )
