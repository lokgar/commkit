"""Boundary helpers for functions that accept an array or a Signal (private).

:func:`adapt_signal` unwraps the argument once into a :class:`SignalAdapter`,
which resolves facts (``resolve_fact``: a conflicting argument raises) and
choices (``resolve_choice``: an explicit argument wins), and returns results
through ``wrap_samples``.  :data:`S` is the TypeVar of a transform's
argument and result.  Also :func:`require_choice` (a choice that must be
there), :func:`require_device` and :func:`require_integer_sps`.

This module holds container adaptation only: numerical code receives arrays
and fully resolved scalar metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, cast

from ._signal_input import S, require_device, require_integer_sps
from ._signal_input import _same_value as _same_fact
from ._signal_input import require_value as require_choice

if TYPE_CHECKING:
    from ..backend import ArrayType
    from .signal import Signal

__all__ = [
    "S",
    "SignalAdapter",
    "adapt_signal",
    "require_choice",
    "require_device",
    "require_integer_sps",
]


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

    def resolve_wavelength_nm(self, supplied: float | None) -> float:
        """Resolve the carrier wavelength in nm of an optical process.

        An optical Signal (``domain="OPT"``) gives ``c / center_frequency``;
        a supplied value must agree with it to 0.01 % (0.15 nm in the C
        band) or ``ValueError`` is raised.  Arrays and Signals of another
        domain need the supplied value.
        """
        sig = self.signal
        carrier = None if sig is None else sig.center_wavelength_nm
        if sig is None or carrier is None:
            if supplied is None:
                raise ValueError(
                    f"{self.function_name} requires center_wavelength_nm for "
                    "array input or a Signal that is not optical with a carrier "
                    "(domain='OPT', center_frequency > 0)."
                )
            return float(supplied)
        if supplied is not None and abs(supplied - carrier) > 1e-4 * carrier:
            raise ValueError(
                f"{self.function_name}: center_wavelength_nm={supplied!r} conflicts "
                f"with the optical Signal's carrier, {carrier:.3f} nm at "
                f"{sig.center_frequency / 1e12:.4f} THz.  Omit the "
                "argument for an optical Signal."
            )
        return carrier

    def resolve_choice(self, field: str, supplied: Any = None) -> Any:
        """Resolve a processing choice (``constellation``, ``pulse``, ...).

        An explicit argument wins; otherwise the Signal's value (or ``None``
        for array input) is used.  The caller decides whether ``None`` is
        acceptable.
        """
        if supplied is not None or self.signal is None:
            return supplied
        return getattr(self.signal, field)

    def require_choice(
        self, field: str, supplied: Any = None, *, purpose: str = "", kind: Any = None
    ) -> Any:
        """Resolve a choice as :meth:`resolve_choice` does, raising when there is
        none (see :func:`require_choice`)."""
        return require_choice(
            self.resolve_choice(field, supplied),
            field,
            self.function_name,
            purpose=purpose,
            kind=kind,
        )

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
