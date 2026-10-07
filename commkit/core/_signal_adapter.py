"""Boundary helpers for functions accepting arrays or :class:`Signal` objects.

This module deliberately contains container adaptation only.  Numerical DSP
implementations should receive arrays and fully resolved scalar metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

import numpy as np

if TYPE_CHECKING:
    from ..backend import ArrayType
    from .signal import Signal

#: Type of a transform's waveform argument and result: ``def f(x: S) -> S``
#: returns a Signal for a Signal and an array for an array.  CuPy has no type
#: stubs, so a CuPy array is ``Any`` to the checker and gives ``Any`` back.
S = TypeVar("S", bound="np.ndarray | Signal")


@dataclass(frozen=True)
class SignalAdapter(Generic[S]):
    """Unwrapped array data and the original Signal, if the caller supplied one.

    Use ``signal_adapter = adapt_signal(...)`` at DSP boundaries. Resolve
    metadata through this object, process ``signal_adapter.array``, then wrap
    waveform results with ``signal_adapter.wrap_samples(...)``. Estimates and
    plots return their own result types without wrapping.
    """

    array: ArrayType | None
    signal: Signal | None
    function_name: str

    def resolve_fact(self, field: str, supplied: Any = None) -> Any:
        """Resolve a fact about the samples (``sampling_rate``, ``sps``, ...).

        For Signal input the Signal's value is used; a supplied value must
        agree with it (to a relative 1e-9) or ``ValueError`` is raised.  For
        array input the supplied value is required.
        """
        if self.signal is None:
            if supplied is None:
                raise ValueError(
                    f"{self.function_name} requires {field} for array input."
                )
            return supplied
        value = getattr(self.signal, field)
        if value is None:
            raise ValueError(f"{self.function_name}: Signal has no {field}.")
        if supplied is not None and not _same_fact(supplied, value):
            raise ValueError(
                f"{self.function_name}: {field}={supplied!r} conflicts with the "
                f"Signal's {field}={value!r}. Omit the argument for Signal input."
            )
        return value

    def resolve_choice(self, field: str, supplied: Any = None) -> Any:
        """Resolve a processing choice (``constellation``, ``pulse``, ...).

        An explicit argument wins; otherwise the Signal's value (or ``None``
        for array input) is used.  The caller decides whether ``None`` is
        acceptable.
        """
        if supplied is not None or self.signal is None:
            return supplied
        return getattr(self.signal, field)

    def symbol_array(self) -> ArrayType:
        """The symbols: the array, or a Signal's samples at one sample per symbol.

        A frame Signal (preamble, pilots and payload) or one not at one sample
        per symbol raises ``ValueError``.
        """
        sig = self.signal
        if sig is not None:
            if sig.frame is not None:
                raise ValueError(
                    f"{self.function_name}: the Signal is a frame (preamble, pilots "
                    "and payload); take its payload with extract_payload(sig)."
                )
            if not _same_fact(sig.sps, 1):
                raise ValueError(
                    f"{self.function_name} needs one sample per symbol, got "
                    f"sps={sig.sps}; decimate to the symbol rate first."
                )
        return self.array

    def wrap_samples(self, samples: Any, /, **metadata: Any) -> S:
        """Return samples directly for array input, or a new Signal for Signal input.

        A new Signal shares unchanged metadata and provenance with the input
        and applies validated metadata overrides; the input Signal is not
        modified.  Replacement sample buffers are not copied.

        For array input, samples (including None) pass through unchanged and
        metadata overrides are unused. None is invalid for Signal output.
        """
        if self.signal is None:
            return cast(S, samples)
        if samples is None:
            raise ValueError(f"{self.function_name}: input Signal field is empty.")
        return cast(S, self.signal.replace(samples=samples, **metadata))


def adapt_signal(
    value: S,
    *,
    function_name: str,
    field: str = "samples",
) -> SignalAdapter[S]:
    """Unwrap an array/Signal input once at the public API boundary."""
    from .signal import Signal

    if isinstance(value, Signal):
        return SignalAdapter(getattr(value, field), value, function_name)
    return SignalAdapter(value, None, function_name)


# Relative tolerance for facts derived by floating-point arithmetic, e.g.
# sps = sampling_rate / symbol_rate = 3.0000000000000004.
_FACT_RTOL = 1e-9


def _same_fact(a: Any, b: Any) -> bool:
    try:
        return bool(np.isclose(float(a), float(b), rtol=_FACT_RTOL, atol=0.0))
    except (TypeError, ValueError):
        return bool(a == b)


def require_device(device: str, function_name: str) -> str:
    """Validate a factory's ``device`` and return it lower-cased.

    Factories have no input data whose device they could follow, so they are
    the only functions that take ``device=``.
    """
    if isinstance(device, str) and device.lower() in ("cpu", "gpu"):
        return device.lower()
    raise ValueError(
        f'{function_name} requires device to be "cpu" or "gpu"; got {device!r}.'
    )


def require_integer_sps(value: float, function_name: str) -> int:
    """Validate a positive integral SPS and return it as ``int``.

    Values within a relative 1e-9 of an integer are accepted, since an SPS
    computed as ``sampling_rate / symbol_rate`` carries rounding error.  A
    genuinely fractional SPS (1.5) raises; it is never truncated.
    """
    if not np.isfinite(value) or value < 1:
        raise ValueError(
            f"{function_name} requires sps to be a positive integer; got {value!r}."
        )
    nearest = round(float(value))
    if abs(value - nearest) > _FACT_RTOL * nearest:
        raise ValueError(
            f"{function_name} requires sps to be a positive integer; got {value!r}."
        )
    return int(nearest)
