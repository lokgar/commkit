"""Array modules, devices and data movement.

:func:`dispatch` returns an input as an array together with its array module
(NumPy or CuPy) and SciPy module (``scipy`` or ``cupyx.scipy``), so the
device follows the data; :func:`get_array_module` and
:func:`get_scipy_module` are its two halves.  Data moves only through an
explicit :func:`to_device` (or ``Signal.to``), and :func:`is_cupy_available`
reports whether the GPU can be used.  Arrays from other frameworks are
rejected with ``TypeError``; exchange data with them explicitly through
DLPack.

Also :class:`PrecisionWarning`, emitted when double-precision input reaches a
block that computes in single precision.
"""

import types
import warnings
import weakref
from collections.abc import Callable
from functools import cache
from typing import Any, cast

import numpy as np

from .logger import logger

__all__ = [
    "ArrayType",
    "PrecisionWarning",
    "dispatch",
    "get_array_module",
    "get_scipy_module",
    "is_cupy_available",
    "to_device",
]


class PrecisionWarning(UserWarning):
    """Double-precision input reduced to a block's single working precision.

    The equalizers and the soft demapper compute in complex64 or float32 (the
    GPU's fast path) whatever their input; transforms keep the input's
    precision.  Shown once per call site by Python's default filter; turn it
    into an error for strict precision with
    ``warnings.simplefilter("error", commkit.PrecisionWarning)``.
    """


@cache
def _cupy_probe() -> tuple[types.ModuleType | None, Exception | None]:
    """Import CuPy and check it works, once, on the first GPU-related call.

    Importing ``commkit`` never imports CuPy or touches the GPU.  The probe
    allocates and runs one kernel, which catches installations whose CUDA
    libraries (nvrtc, cublas, driver) are missing or broken.  Returns the
    module, or ``None`` and the reason.
    """
    try:
        import cupy

        cupy.arange(1)
    except Exception as exc:
        return None, exc
    return cast(types.ModuleType, cupy), None


def _cupy() -> types.ModuleType:
    """The CuPy module; raises ``ImportError`` with the probe's reason."""
    cp, exc = _cupy_probe()
    if cp is None:
        raise ImportError(f"CuPy is not available: {exc}") from exc
    return cp


def _is_cupy_array(data: Any) -> bool:
    """True for a CuPy array.

    Checks the type's module, so it never imports CuPy: a CuPy array can only
    exist if CuPy is already loaded.
    """
    return type(data).__module__ == "cupy"


# Any for CuPy array to avoid a hard dependency in the type hint if not installed
ArrayType = np.ndarray | Any


def is_cupy_available() -> bool:
    """Report whether CuPy is installed and can run on a GPU.

    Data placement never depends on this: arrays stay where the caller put
    them, and only an explicit ``to_device(x, "gpu")`` or ``Signal.to("gpu")``
    moves data to the GPU.

    Returns
    -------
    bool
        True if CuPy is installed and functional.  The first call imports CuPy
        and runs a small probe kernel; the result is cached.
    """
    return _cupy_probe()[0] is not None


def _module_for(device: str) -> types.ModuleType:
    """The array module for a validated ``"cpu"`` or ``"gpu"`` (factories)."""
    if device == "cpu":
        return np
    return _cupy()


def get_array_module(data: Any) -> types.ModuleType:
    """Return the array module (NumPy or CuPy) of the data.

    Decided by the type of the data itself, so the device follows the data.
    Never imports CuPy for host data.

    Parameters
    ----------
    data : array_like
        The data to inspect (an array, a list or a scalar).

    Returns
    -------
    module
        ``cupy`` for a CuPy array, otherwise ``numpy`` (host arrays, lists
        and scalars).
    """
    if _is_cupy_array(data):
        import cupy

        return cast(types.ModuleType, cupy)
    return np


@cache
def get_scipy_module(xp: types.ModuleType) -> types.ModuleType:
    """Return the SciPy module that matches an array module.

    Parameters
    ----------
    xp : module
        ``numpy`` or ``cupy``.

    Returns
    -------
    sp : module
        ``scipy`` for NumPy, ``cupyx.scipy`` for CuPy, with ``signal``,
        ``special`` and ``ndimage`` imported.
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
    """Move data between the host (CPU) and the GPU.

    Parameters
    ----------
    data : array_like
        The data to move (an array, a list or a scalar).
    device : {"cpu", "gpu"}
        Target device, case-insensitive.

    Returns
    -------
    ndarray
        The data on the target device: a NumPy array for ``"cpu"``, a CuPy
        array for ``"gpu"``.  An array already on the target device is
        returned as is, not copied.

    Raises
    ------
    ImportError
        If ``"gpu"`` is requested but CuPy is not available.
    ValueError
        On an unknown device name.
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
        if isinstance(data, cp.ndarray):
            return data
        return cp.asarray(data)

    else:
        raise ValueError(f"Unknown device: {device.upper()}")


# Device copies of the host tables of value objects (a constellation's
# points and labels, a code's parity checks), per owner and then per (table
# kind, CUDA device id): built on the first call instead of on every one.
# Weak keys, so a dropped object frees its tables; equal objects share them.
_DEVICE_TABLES: weakref.WeakKeyDictionary[
    Any, dict[tuple[str, int], tuple[Any, ...]]
] = weakref.WeakKeyDictionary()


def _device_tables(
    owner: Any,
    kind: str,
    xp: types.ModuleType,
    build: Callable[[], tuple[np.ndarray, ...]],
) -> tuple[Any, ...]:
    """``build()``'s host tables on ``xp``'s device, cached per ``owner``.

    On the current CUDA device for CuPy; for NumPy the host tables
    themselves, so a table derived from the owner's fields is also built
    once.  ``owner`` is a hashable, weakly referenceable value object (a
    frozen dataclass); ``None`` (nothing to cache on) builds the tables
    every call.  Used by :mod:`commkit.mapping.llr` and
    :mod:`commkit.coding`.
    """
    if owner is None:
        return tuple(xp.asarray(a) for a in build())
    per_owner = _DEVICE_TABLES.setdefault(owner, {})
    key = (kind, -1 if xp is np else int(xp.cuda.Device().id))
    tables = per_owner.get(key)
    if tables is None:
        tables = per_owner[key] = tuple(xp.asarray(a) for a in build())
    return tables


def dispatch(
    data: Any,
) -> tuple[ArrayType, types.ModuleType, types.ModuleType]:
    """Return the data as an array with its array and SciPy modules.

    The entry point of every numerical function: the device follows the
    data.

    Parameters
    ----------
    data : array_like
        The input data.

    Returns
    -------
    data_array : ndarray
        The input as an array on its current device (NumPy and CuPy arrays
        unchanged).
    xp : module
        ``numpy`` or ``cupy``.
    sp : module
        ``scipy`` or ``cupyx.scipy``.

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
