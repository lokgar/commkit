"""The array-or-Signal argument of a Signal-aware function (private).

:class:`SignalInput` unwraps the argument once, resolves what the function
needs from the Signal through one typed method per field, and returns
results through :meth:`SignalInput.wrap`.  :data:`S` is the TypeVar of a
transform's argument and result.  Also :func:`reference_of` (the
:class:`Reference` of a Signal passed as a reference argument),
:func:`require_value` (a value that must be there), :func:`require_device`
and :func:`require_integer_sps`.

Two rules, the same for every field.  A **property** of the samples
(``sampling_rate``, ``symbol_rate``, ``sps``, ``domain``, the carrier) comes
from a Signal: an argument that disagrees with it raises ``ValueError``, an
equal one is accepted, and array input needs the argument.  A **default**
(``constellation``, ``pulse``, the reference symbols) comes from a Signal
too, but an explicit argument wins.  Neither case warns.

This module holds container adaptation only: numerical code receives arrays
and fully resolved scalar metadata.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar, cast, overload

import numpy as np

if TYPE_CHECKING:
    from ..backend import ArrayType
    from ..filtering.pulses import Pulse
    from ..mapping import Constellation
    from .signal import Signal

#: Type of a transform's waveform argument and result: ``def f(x: S) -> S``
#: returns a Signal for a Signal and an array for an array.  CuPy has no type
#: stubs, so a CuPy array is ``Any`` to the checker and gives ``Any`` back.
S = TypeVar("S", bound="np.ndarray | Signal")

# Relative tolerance for properties derived by floating-point arithmetic,
# e.g. sps = sampling_rate / symbol_rate = 3.0000000000000004.
_PROPERTY_RTOL = 1e-9


class SignalInput(Generic[S]):
    """The array-or-Signal argument of a Signal-aware function, unwrapped once.

    Built at the top of the function, it gives the samples to process, then
    resolves each value the function needs from the Signal by a method named
    after the field, and wraps the result for return.

    Parameters
    ----------
    samples : array_like or Signal
        The function's waveform argument.
    function_name : str
        The function, as ``"name()"``, for error messages.

    Notes
    -----
    A **property** of the samples (:meth:`sampling_rate`,
    :meth:`symbol_rate`, :meth:`sps`, :meth:`center_frequency`,
    :meth:`domain`, :meth:`wavelength_nm`)
    comes from a Signal: a supplied argument must agree with it (to a
    relative 1e-9, the wavelength to 0.01 %) or ``ValueError`` is raised,
    and array input needs the argument.  A **default**
    (:meth:`constellation`, :meth:`pulse`, :meth:`reference_symbols`) is
    the supplied argument when given, else the Signal's value, else
    ``None``, which ``required=True`` turns into a ``ValueError``.  Called
    without an argument, each method returns the Signal's value.
    """

    __slots__ = ("_samples", "_signal", "function_name")

    def __init__(self, samples: S, function_name: str) -> None:
        from .signal import Signal

        self.function_name = function_name
        if isinstance(samples, Signal):
            self._signal: Signal | None = samples
            self._samples: ArrayType = samples.samples
        else:
            self._signal = None
            self._samples = samples

    @property
    def samples(self) -> ArrayType:
        """The samples: the array, or the Signal's ``samples``."""
        return self._samples

    @property
    def signal(self) -> Signal | None:
        """The Signal, or ``None`` for array input."""
        return self._signal

    # -- properties of the samples: a conflicting argument raises ----------

    def _property(self, field: str, supplied: Any) -> Any:
        if self._signal is None:
            if supplied is None:
                raise ValueError(
                    f"{self.function_name} requires {field} for array input."
                )
            return supplied
        value = getattr(self._signal, field)
        if value is None:
            raise ValueError(f"{self.function_name}: Signal has no {field}.")
        if supplied is not None and not _same_value(supplied, value):
            raise ValueError(
                f"{self.function_name}: {field}={supplied!r} conflicts with the "
                f"Signal's {field}={value!r}. Omit the argument for Signal input."
            )
        return value

    def sampling_rate(self, supplied: float | None = None) -> float:
        """The sampling rate in Hz (a property of the samples)."""
        return float(self._property("sampling_rate", supplied))

    def symbol_rate(self, supplied: float | None = None) -> float:
        """The symbol rate in Hz (a property of the samples)."""
        return float(self._property("symbol_rate", supplied))

    def sps(self, supplied: float | None = None) -> float:
        """The samples per symbol (a property of the samples)."""
        return float(self._property("sps", supplied))

    def integer_sps(self, supplied: float | None = None) -> int:
        """The samples per symbol as a positive ``int``, see
        :func:`require_integer_sps`; a fractional value raises."""
        return require_integer_sps(self.sps(supplied), self.function_name)

    def center_frequency(self, supplied: float | None = None) -> float:
        """The carrier frequency in Hz, 0 for baseband (a property)."""
        return float(self._property("center_frequency", supplied))

    def domain(self, supplied: str | None = None) -> str:
        """The domain, ``"DIG"``, ``"RF"`` or ``"OPT"`` (a property)."""
        return cast(str, self._property("domain", supplied))

    def wavelength_nm(self, supplied: float | None = None) -> float:
        """The carrier wavelength in nm of an optical process (a property).

        An optical Signal (``domain="OPT"``) gives ``c / center_frequency``;
        a supplied value must agree with it to 0.01 % (0.15 nm in the C
        band) or ``ValueError`` is raised.  Arrays and Signals of another
        domain need the supplied value.
        """
        sig = self._signal
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

    # -- defaults: an explicit argument wins --------------------------------

    def _default(self, field: str, supplied: Any) -> Any:
        if supplied is not None or self._signal is None:
            return supplied
        return getattr(self._signal, field)

    @overload
    def constellation(
        self,
        supplied: Constellation | None = None,
        *,
        required: Literal[True],
        purpose: str = "",
    ) -> Constellation: ...
    @overload
    def constellation(
        self,
        supplied: Constellation | None = None,
        *,
        required: bool = False,
        purpose: str = "",
    ) -> Constellation | None: ...
    def constellation(
        self,
        supplied: Constellation | None = None,
        *,
        required: bool = False,
        purpose: str = "",
    ) -> Constellation | None:
        """The decision constellation (a default).

        ``required=True`` raises ``ValueError`` without one, naming what it
        is needed for (``purpose``, such as ``" for its decisions"``).  A
        value that is not a :class:`~commkit.mapping.Constellation` raises
        ``TypeError``.
        """
        from ..mapping import Constellation

        value = self._default("constellation", supplied)
        if value is None and not required:
            return None
        return cast(
            Constellation,
            require_value(
                value,
                "constellation",
                self.function_name,
                purpose=purpose,
                kind=Constellation,
            ),
        )

    @overload
    def pulse(
        self,
        supplied: Pulse | ArrayType | None = None,
        *,
        required: Literal[True],
        purpose: str = "",
    ) -> Pulse | ArrayType: ...
    @overload
    def pulse(
        self,
        supplied: Pulse | ArrayType | None = None,
        *,
        required: bool = False,
        purpose: str = "",
    ) -> Pulse | ArrayType | None: ...
    def pulse(
        self,
        supplied: Pulse | ArrayType | None = None,
        *,
        required: bool = False,
        purpose: str = "",
    ) -> Pulse | ArrayType | None:
        """The pulse, a Pulse object or raw taps (a default).

        ``required=True`` raises ``ValueError`` without one, naming what it
        is needed for (``purpose``).
        """
        value = self._default("pulse", supplied)
        if value is None and not required:
            return None
        return require_value(value, "pulse", self.function_name, purpose=purpose)

    @overload
    def reference_symbols(
        self, supplied: ArrayType | None = None, *, required: Literal[True]
    ) -> ArrayType: ...
    @overload
    def reference_symbols(
        self, supplied: ArrayType | None = None, *, required: bool = False
    ) -> ArrayType | None: ...
    def reference_symbols(
        self, supplied: ArrayType | None = None, *, required: bool = False
    ) -> ArrayType | None:
        """The known transmitted symbols (a default): the supplied array,
        else the Signal's ``reference.symbols``.

        ``required=True`` raises ``ValueError`` without them.
        """
        if supplied is not None:
            return supplied
        sig = self._signal
        if sig is not None and sig.reference is not None:
            return sig.reference.symbols
        if required:
            raise ValueError(
                f"{self.function_name} needs reference symbols: pass reference= "
                "or a Signal that has a reference."
            )
        return None

    # -- the samples in and out ---------------------------------------------

    def symbols(self) -> ArrayType:
        """The symbols: the array, or a Signal's samples at one sample per symbol.

        A frame Signal (preamble, pilots and payload) or one not at one sample
        per symbol raises ``ValueError``.
        """
        sig = self._signal
        if sig is not None:
            if sig.frame is not None:
                raise ValueError(
                    f"{self.function_name}: the Signal is a frame (preamble, pilots "
                    "and payload); take its payload with extract_payload(sig)."
                )
            if not _same_value(sig.sps, 1):
                raise ValueError(
                    f"{self.function_name} needs one sample per symbol, got "
                    f"sps={sig.sps}; decimate to the symbol rate first."
                )
        return self._samples

    def wrap(
        self, samples: Any, /, *, num_symbols: int | None = None, **metadata: Any
    ) -> S:
        """Return the samples for array input, or a new Signal for Signal input.

        The new Signal shares the input's other fields and takes the
        ``metadata`` overrides (``sampling_rate=``, ...); the input is not
        modified and ``samples`` is not copied.  ``num_symbols`` keeps the
        first that many symbols of the reference, for a function that drops
        trailing symbol periods (the alignment invariant); for array input
        both are unused.  ``None`` raises ``ValueError``.
        """
        if samples is None:
            raise ValueError(f"{self.function_name}: no samples to return.")
        sig = self._signal
        if sig is None:
            return cast(S, samples)
        if num_symbols is not None and sig.reference is not None:
            metadata["reference"] = sig.reference.head(num_symbols)
        return cast(S, sig.replace(samples=samples, **metadata))


def reference_of(value: Any) -> Any:
    """The :class:`Reference` of a Signal passed as a reference argument;
    any other value as it is."""
    from .signal import Signal

    return value.reference if isinstance(value, Signal) else value


def _same_value(a: Any, b: Any) -> bool:
    try:
        return bool(np.isclose(float(a), float(b), rtol=_PROPERTY_RTOL, atol=0.0))
    except (TypeError, ValueError):
        return bool(a == b)


def require_value(
    value: Any, name: str, function_name: str, *, purpose: str = "", kind: Any = None
) -> Any:
    """``value``, which must be there: ``None`` raises ``ValueError``.

    The message names the function, the parameter ``name`` and what it is
    needed for (``purpose``, such as ``" for its decisions"``).  With
    ``kind`` a value of another type raises ``TypeError``.
    """
    if value is None:
        raise ValueError(
            f"{function_name} needs a {name}{purpose}: pass {name}= or a Signal "
            "that has one."
        )
    if kind is not None and not isinstance(value, kind):
        raise TypeError(
            f"{function_name}: {name} must be a {kind.__name__}, got "
            f"{type(value).__name__}."
        )
    return value


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
    if abs(value - nearest) > _PROPERTY_RTOL * nearest:
        raise ValueError(
            f"{function_name} requires sps to be a positive integer; got {value!r}."
        )
    return int(nearest)
