"""Boundary helpers for functions accepting arrays or :class:`Signal` objects.

This module deliberately contains container adaptation only.  Numerical DSP
implementations should receive arrays and fully resolved scalar metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from ..logger import logger

if TYPE_CHECKING:
    from ..backend import ArrayType
    from .signal import Signal


@dataclass(frozen=True)
class SignalAdapter:
    """Unwrapped array data and the original Signal, if the caller supplied one.

    Use ``signal_adapter = adapt_signal(...)`` at DSP boundaries. Resolve
    metadata through this object, process ``signal_adapter.array``, then wrap
    waveform results with ``signal_adapter.wrap_samples(...)``. Estimates and
    plots return their own result types without wrapping.
    """

    array: ArrayType | None
    signal: Signal | None
    function_name: str

    def resolve_required(self, field: str, supplied: Any = None) -> Any:
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

    def resolve_optional(self, field: str, supplied: Any = None) -> Any:
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

    def wrap_samples[SamplesT](
        self, samples: SamplesT, /, **metadata: Any
    ) -> SamplesT | Signal:
        """Return samples directly for array input, or a new Signal for Signal input.

        A new Signal shares unchanged metadata and provenance with the input,
        applies validated metadata overrides, and invalidates resolved symbols
        and bits. The input Signal is not modified. Replacement sample buffers
        are not unconditionally copied; see ``Signal.replace_samples()``.

        For array input, samples (including None) pass through unchanged and
        metadata overrides are unused. None is invalid for Signal output.
        """
        if self.signal is None:
            return samples
        if samples is None:
            raise ValueError(f"{self.function_name}: input Signal field is empty.")
        return self.signal.replace_samples(samples, **metadata)

    def replace_signal_field(
        self,
        field: Literal["resolved_symbols", "resolved_bits"],
        value: ArrayType,
    ) -> Signal:
        """Return a new Signal with resolved symbols or bits replaced.

        Requires Signal input and leaves its waveform unchanged. Replacing
        resolved symbols invalidates resolved bits; replacing bits preserves
        resolved symbols. The original Signal is not modified.
        """
        if self.signal is None:
            raise TypeError("replace_signal_field() requires Signal input.")
        result = self.signal._shallow_clone()
        setattr(result, field, value)
        if field == "resolved_symbols":
            result.resolved_bits = None
        return result


def adapt_signal(
    value: ArrayType | Signal,
    *,
    function_name: str,
    field: str = "samples",
) -> SignalAdapter:
    """Unwrap an array/Signal input once at the public API boundary."""
    from .signal import Signal

    if isinstance(value, Signal):
        return SignalAdapter(getattr(value, field), value, function_name)
    return SignalAdapter(value, None, function_name)


def require_integer_sps(value: float, function_name: str) -> int:
    """Validate positive integral SPS before converting it to ``int``."""
    if not np.isfinite(value) or value < 1 or value % 1 != 0:
        raise ValueError(
            f"{function_name} requires sps to be a positive integer; got {value!r}."
        )
    return int(value)
