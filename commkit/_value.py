"""Equality of the frozen dataclasses that hold arrays (private).

Every frozen dataclass of the library compares by value.  Those without
arrays use the dataclass's own ``__eq__`` and ``__hash__``; those with
arrays derive from one of two bases here:

- :class:`_Value`: descriptions and host-only results whose arrays are
  read-only NumPy arrays (codes, interleavers, algorithm objects, frame
  maps, EXIT and error-rate curves).  Equal and hashable by their fields,
  arrays by shape, dtype and bytes, so equal objects share cached device
  tables (``backend._device_tables``).
- :class:`_Record`: data and results whose arrays may live on the GPU and
  are not frozen (``Signal``, ``Reference``, the estimates and results).
  Equal by their fields (arrays of one type, shape and dtype with equal
  elements, NaN equal to NaN), and unhashable, as arrays are: a hash of
  writable or device data could not stay consistent with equality.

Arrays on different devices never compare equal: equality makes no
transfer.  Used by the value objects and result classes across the package.
"""

from __future__ import annotations

import dataclasses
import numbers
from typing import Any, ClassVar

import numpy as np

from .backend import get_array_module


def _is_array(value: Any) -> bool:
    return isinstance(value, np.ndarray) or (
        type(value).__module__.startswith("cupy") and hasattr(value, "shape")
    )


def _equal(a: Any, b: Any) -> bool:
    """Value equality of two field values (arrays, sequences, scalars)."""
    if a is b:
        return True
    if _is_array(a) or _is_array(b):
        if type(a) is not type(b) or a.shape != b.shape or a.dtype != b.dtype:
            return False
        xp = get_array_module(a)
        return bool(xp.array_equal(a, b, equal_nan=a.dtype.kind in "fc"))
    if isinstance(a, tuple | list):
        return (
            type(a) is type(b)
            and len(a) == len(b)
            and all(_equal(x, y) for x, y in zip(a, b, strict=True))
        )
    if isinstance(a, numbers.Number) and isinstance(b, numbers.Number):
        if a != a and b != b:  # NaN equals NaN, as in the arrays
            return True
    result = a == b
    return bool(result) if isinstance(result, bool | np.bool_) else result is True


def _compared(obj: Any) -> tuple[Any, ...]:
    return tuple(getattr(obj, f.name) for f in dataclasses.fields(obj) if f.compare)


class _Value:
    """Value equality and hashing of frozen dataclasses with host arrays.

    Equality compares the fields (arrays by shape, dtype and bytes), so equal
    objects share their cached device tables.
    """

    def _key(self) -> tuple[Any, ...]:
        return tuple(
            (v.shape, v.dtype.str, v.tobytes()) if isinstance(v, np.ndarray) else v
            for v in _compared(self)
        )

    def __eq__(self, other: object) -> bool:
        # The device-table cache compares an object with itself on every
        # lookup (through a weak reference); _key() copies each array to bytes.
        if self is other:
            return True
        if type(other) is not type(self):
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self) -> int:
        # Cached: the device-table cache hashes the object on every call, and
        # a large code's arrays are megabytes.
        h: int | None = self.__dict__.get("_hash")
        if h is None:
            h = hash((type(self), self._key()))
            self.__dict__["_hash"] = h
        return h

    def __getstate__(self) -> dict[str, Any]:
        # Not the cached hash: hashes of bytes differ between processes.
        state = self.__dict__.copy()
        state.pop("_hash", None)
        return state


class _Record:
    """Value equality, without a hash, of frozen dataclasses with device arrays.

    ``a == b`` holds for objects of one class whose compared fields are
    equal: arrays of one type (NumPy or CuPy), shape and dtype with equal
    elements (NaN equal to NaN), sequences element by element, nested
    records and value objects by their own equality.  Comparing device
    arrays reads one boolean per array back to the host.
    """

    __hash__: ClassVar[None] = None  # type: ignore[assignment]

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if type(other) is not type(self):
            return NotImplemented
        return all(
            _equal(a, b) for a, b in zip(_compared(self), _compared(other), strict=True)
        )
