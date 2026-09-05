"""Boundary helpers for functions accepting arrays or :class:`Signal` objects.

This module deliberately contains container adaptation only.  Numerical DSP
implementations should receive arrays and fully resolved scalar metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np

from ..logger import logger

if TYPE_CHECKING:
    from ..backend import ArrayType
    from .signal import Signal


@dataclass(frozen=True)
class SignalInput:
    """An unwrapped input together with its optional container context."""

    array: ArrayType | None
    signal: Signal | None
    function_name: str

    def required(self, field: str, supplied: Any = None) -> Any:
        """Resolve required metadata, with Signal metadata taking precedence."""
        if self.signal is None:
            if supplied is None:
                raise ValueError(
                    f"{self.function_name} requires {field} for array input."
                )
            return supplied

        value = getattr(self.signal, field)
        if value is None:
            raise ValueError(f"{self.function_name}: Signal has no {field} metadata.")
        if supplied is not None:
            logger.warning(
                "%s: ignoring supplied %s=%r for Signal input; using the "
                "signal's own %s=%r instead.",
                self.function_name,
                field,
                supplied,
                field,
                value,
            )
        return value

    def optional(self, field: str, supplied: Any = None) -> Any:
        """Resolve optional metadata, falling back only when Signal lacks it."""
        if self.signal is None:
            return supplied
        value = getattr(self.signal, field)
        if value is not None:
            return value
        if supplied is not None:
            logger.warning(
                "%s: Signal has no %s set; falling back to supplied %s=%r.",
                self.function_name,
                field,
                field,
                supplied,
            )
        return supplied

    @overload
    def return_value(self, array: ArrayType, /, **metadata: Any) -> ArrayType: ...

    @overload
    def return_value(self, array: None, /, **metadata: Any) -> None: ...

    def return_value(
        self, array: ArrayType | None, /, **metadata: Any
    ) -> ArrayType | Signal | None:
        """Rewrap a transformed sample array when the input was a Signal."""
        if self.signal is None:
            return array
        if array is None:
            raise ValueError(f"{self.function_name}: input Signal field is empty.")
        return self.signal.replace_samples(array, **metadata)

    def replace_field(
        self,
        field: Literal["resolved_symbols", "resolved_bits"],
        value: ArrayType,
    ) -> Signal:
        """Return a Signal with an explicitly replaced derived-data field."""
        if self.signal is None:
            raise TypeError("replace_field() requires Signal input.")
        result = self.signal._shallow_clone()
        setattr(result, field, value)
        if field == "resolved_symbols":
            result.resolved_bits = None
        return result


def prepare_signal_input(
    value: ArrayType | Signal,
    *,
    function_name: str,
    field: str = "samples",
) -> SignalInput:
    """Unwrap an array/Signal input once at the public API boundary."""
    from .signal import Signal

    if isinstance(value, Signal):
        return SignalInput(getattr(value, field), value, function_name)
    return SignalInput(value, None, function_name)


def require_integer_sps(value: float, function_name: str) -> int:
    """Validate positive integral SPS before converting it to ``int``."""
    if not np.isfinite(value) or value < 1 or value % 1 != 0:
        raise ValueError(
            f"{function_name} requires sps to be a positive integer; got {value!r}."
        )
    return int(value)
