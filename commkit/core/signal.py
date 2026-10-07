"""
The :class:`Signal` container and its :class:`Reference` ground truth.

A Signal is a waveform plus the facts and descriptions needed to process it:
sampling and symbol rate, the constellation and pulse it was built with, and
the transmitted reference.  It never holds pipeline results.

Containers are frozen dataclasses: every update returns a new object through
:meth:`Signal.replace`, which validates the changed fields.
"""

from __future__ import annotations

import copy
import dataclasses
import types
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from .._array import validate_array
from ..backend import (
    ArrayType,
    get_array_module,
    get_scipy_module,
    to_device,
)

__all__ = ["Reference", "Signal"]

if TYPE_CHECKING:
    from ..filtering import Pulse
    from ..mapping import Constellation


@dataclass(frozen=True, eq=False)
class Reference:
    """Ground truth of a Signal: the transmitted symbols and bits.

    Parameters
    ----------
    symbols : array_like
        Transmitted symbols at one sample per symbol, ``(N,)`` or ``(C, N)``,
        NumPy or CuPy.  Symbol *k* corresponds to symbol period *k* of the
        Signal's samples.
    bits : array_like, optional
        Transmitted bits, ``(N * k,)`` or ``(C, N * k)``, on the same device
        as ``symbols``.
    """

    symbols: Any
    bits: Any | None = None

    def __post_init__(self) -> None:
        symbols = validate_array(self.symbols, name="symbols")
        if symbols.ndim not in (1, 2):
            raise ValueError(
                f"symbols must have shape (N,) or (C, N), got {symbols.shape}."
            )
        object.__setattr__(self, "symbols", symbols)
        if self.bits is not None:
            bits = validate_array(self.bits, name="bits")
            if bits.ndim != symbols.ndim or bits.shape[:-1] != symbols.shape[:-1]:
                raise ValueError(
                    f"bits shape {bits.shape} does not match symbols shape "
                    f"{symbols.shape} on the channel axis."
                )
            if get_array_module(bits) is not get_array_module(symbols):
                raise ValueError("bits and symbols must be on the same device.")
            object.__setattr__(self, "bits", bits)

    def to(self, device: str) -> Reference:
        """Return a copy with its arrays on ``device`` ("cpu" or "gpu")."""
        bits = None if self.bits is None else to_device(self.bits, device)
        return Reference(symbols=to_device(self.symbols, device), bits=bits)

    def head(self, num_symbols: int) -> Reference:
        """The first ``num_symbols`` symbols and their bits."""
        if num_symbols >= self.symbols.shape[-1]:
            return self
        bits = self.bits
        if bits is not None:
            k = bits.shape[-1] // self.symbols.shape[-1]
            bits = bits[..., : num_symbols * k]
        return Reference(symbols=self.symbols[..., :num_symbols], bits=bits)


@dataclass(frozen=True, eq=False, kw_only=True)
class Signal:
    """
    A sampled waveform with its facts, description and ground truth.

    Parameters
    ----------
    samples : array_like
        IQ (or real) samples, NumPy or CuPy, ``(N,)`` for one channel or
        ``(C, N)`` for C channels.  Time is always the last axis.
    sampling_rate : float
        Sampling rate in Hz (a fact).
    symbol_rate : float
        Symbol rate in Hz (a fact).  ``sps`` is derived from the two rates.
    constellation : Constellation, optional
        The constellation the symbols were drawn from.  Functions use it as
        their default decision constellation; an explicit argument wins.
    pulse : Pulse, optional
        The transmit pulse (``RRC``, ``RC``, ``Gaussian``, ``Rect``,
        ``SmoothRect``).  Used as the default for matched filtering.
    reference : Reference, optional
        The transmitted symbols and bits.  Symbol *k* corresponds to symbol
        period *k* of ``samples``; functions that drop or shift symbol periods
        slice the reference to match.
    frame : SingleCarrierFrame, optional
        The frame that generated the samples (a layout snapshot).
    center_frequency : float, default 0.0
        Carrier frequency in Hz (a fact), used to label spectra.

    Notes
    -----
    Construction does no hidden work: no bit-to-symbol mapping, no
    normalization, no device move and no shape guessing.  The factories
    (``generate``, ``SingleCarrierFrame.to_signal``) do that work.
    """

    samples: Any
    sampling_rate: float
    symbol_rate: float

    constellation: Constellation | None = None
    pulse: Pulse | None = None
    reference: Reference | None = None
    frame: Any | None = field(default=None, repr=False)
    center_frequency: float = 0.0

    def __post_init__(self) -> None:
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            object.__setattr__(self, f.name, _validate_field(f.name, value))

    def replace(self, **changes: Any) -> Signal:
        """Return a new Signal with ``changes`` applied.

        Changed fields are validated; unchanged arrays and the frame are shared
        with this Signal, not copied (use :meth:`clone` for that).

        Raises
        ------
        TypeError
            If a name is not a Signal field.
        ValueError
            If a value is invalid.
        """
        unknown = set(changes) - _FIELD_NAMES
        if unknown:
            raise TypeError(f"Signal has no field(s) {sorted(unknown)}.")
        new = copy.copy(self)
        for name, value in changes.items():
            object.__setattr__(new, name, _validate_field(name, value))
        return new

    # -------------------------------------------------------------------------
    # Convenience
    # -------------------------------------------------------------------------

    def time_axis(self) -> ArrayType:
        """Time of each sample in seconds, starting at 0, shape ``(N,)``."""
        n_samples = self.samples.shape[-1]
        return self.xp.arange(0, n_samples) / self.sampling_rate

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    def _info_rows(self) -> list[tuple[str, str]]:
        """Summary rows rendered by ``str(sig)`` and ``_repr_html_``."""

        def _yn(v: Any) -> str:
            return "yes" if v is not None else "no"

        rows: list[tuple[str, str]] = [
            (
                "Constellation",
                repr(self.constellation) if self.constellation else "None",
            ),
            ("Pulse", repr(self.pulse) if self.pulse else "None"),
            ("Symbol rate", _format_si(self.symbol_rate, "Baud")),
        ]
        if self.constellation is not None:
            rows.append(
                (
                    "Bit rate",
                    _format_si(
                        self.symbol_rate * self.constellation.bits_per_symbol, "bps"
                    ),
                )
            )
        rows += [
            ("Sampling rate", _format_si(self.sampling_rate, "Hz")),
            ("Samples per symbol", f"{self.sps:.2f}"),
            ("Duration", _format_si(self.duration, "s")),
            ("Center frequency", _format_si(self.center_frequency, "Hz")),
            ("Backend", self.backend.upper()),
            (
                "Configuration",
                "SISO" if self.num_streams == 1 else f"MIMO ({self.num_streams}x)",
            ),
            ("Samples shape", str(self.samples.shape)),
        ]

        frame = self.frame
        if frame is not None:
            rows.append(("--- Frame structure", ""))
            if frame.preamble is not None:
                p = frame.preamble
                preamble_str = f"{p.sequence_type.upper()}  len={p.length}"
                if p.sequence_type == "zc":
                    preamble_str += f"  root={p.root}"
                rows.append(("Preamble", preamble_str))
            else:
                rows.append(("Preamble", "none"))
            rows.append(("Payload length", f"{frame.payload_len} symbols"))
            if frame.pilot_pattern != "none":
                mask, _ = frame._generate_pilot_mask()
                pilot_str = (
                    f"{frame.pilot_pattern}  count={int(np.sum(mask))}"
                    + (f"  period={frame.pilot_period}" if frame.pilot_period else "")
                    + f"  gain={frame.pilot_gain_db} dB"
                )
                rows.append(("Pilots", pilot_str))
            else:
                rows.append(("Pilots", "none"))
            if frame.guard_len:
                rows.append(("Guard", f"{frame.guard_type}  len={frame.guard_len}"))
            else:
                rows.append(("Guard", "none"))

        rows.append(("--- Reference data", ""))
        ref = self.reference
        rows.append(("reference symbols", _yn(ref and ref.symbols)))
        rows.append(("reference bits", _yn(ref and ref.bits)))
        rows.append(("frame attached", _yn(frame)))
        return rows

    def __str__(self) -> str:
        rows = self._info_rows()
        width = max(len(prop) for prop, _ in rows)
        return "\n".join(f"{prop.ljust(width)}  {val}" for prop, val in rows)

    def _repr_html_(self) -> str:
        body = "".join(
            f"<tr><td><b>{prop}</b></td><td>{val}</td></tr>"
            for prop, val in self._info_rows()
        )
        return f"<table>{body}</table>"

    # -------------------------------------------------------------------------
    # Copies and device moves
    # -------------------------------------------------------------------------

    def clone(self) -> Signal:
        """Return a fully independent (deep) copy, including all arrays."""
        return copy.deepcopy(self)

    def to(self, device: str) -> Signal:
        """
        Return a copy of this signal with its arrays on ``device``.

        Moves the samples and the reference.  Host
        metadata (constellation, pulse, frame) stays where it is.  The
        original signal is unchanged; arrays already on ``device`` are shared.

        Parameters
        ----------
        device : {"cpu", "gpu"}
            The target device. Case-insensitive.

        Raises
        ------
        ImportError
            If GPU is requested but CuPy is not installed/functional.
        """
        moved: dict[str, Any] = {}
        if self.reference is not None:
            moved["reference"] = self.reference.to(device)
        return self.replace(samples=to_device(self.samples, device), **moved)

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def xp(self) -> types.ModuleType:
        """The array module of the samples (``numpy`` or ``cupy``)."""
        return get_array_module(self.samples)

    @property
    def sp(self) -> types.ModuleType:
        """The SciPy module matching the samples (``scipy`` or ``cupyx.scipy``)."""
        return get_scipy_module(self.xp)

    @property
    def backend(self) -> str:
        """``"CPU"`` or ``"GPU"``: where the samples live."""
        return "CPU" if self.xp is np else "GPU"

    @property
    def num_streams(self) -> int:
        """Number of channels: 1 for ``(N,)`` samples, C for ``(C, N)``."""
        if self.samples.ndim == 1:
            return 1
        return int(self.samples.shape[0])

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return float(self.samples.shape[-1] / self.sampling_rate)

    @property
    def sps(self) -> float:
        """Samples per symbol, ``sampling_rate / symbol_rate``."""
        return self.sampling_rate / self.symbol_rate

    @property
    def bits_per_symbol(self) -> int | None:
        """Bits per symbol of the constellation, or ``None`` without one."""
        if self.constellation is None:
            return None
        return self.constellation.bits_per_symbol


# -----------------------------------------------------------------------------
# Field validation (shared by construction and Signal.replace)
# -----------------------------------------------------------------------------

_FIELD_NAMES = frozenset(f.name for f in dataclasses.fields(Signal))


def _validate_samples(v: Any) -> Any:
    """Coerce samples to a NumPy/CuPy array with time on the last axis."""
    arr = validate_array(v, name="samples")
    if arr.ndim > 2:
        raise ValueError(
            f"Samples array has {arr.ndim} dimensions. "
            "Only 1D (SISO) or 2D (MIMO/Dual-Pol) arrays are supported."
        )
    if arr.ndim == 2:
        s0, s1 = arr.shape
        if s0 > s1 and s0 > 32:
            raise ValueError(
                f"samples has shape {arr.shape}, which looks like (N, C). "
                "Time is the last axis: pass (C, N), e.g. samples.T."
            )
    return arr


def _number(
    name: str,
    value: Any,
    kind: type,
    *,
    low: float | None = None,
    low_open: bool = False,
) -> Any:
    if isinstance(value, bool) or not isinstance(value, int | float | np.number):
        raise ValueError(f"{name} must be a number, got {value!r}.")
    value = kind(value)
    if low is not None and (value <= low if low_open else value < low):
        bound = ">" if low_open else ">="
        raise ValueError(f"{name} must be {bound} {low}, got {value}.")
    return value


def _validate_field(name: str, value: Any) -> Any:
    """Validate and coerce one Signal field."""
    if name == "samples":
        return _validate_samples(value)
    if name in ("sampling_rate", "symbol_rate"):
        return _number(name, value, float, low=0, low_open=True)
    if name == "center_frequency":
        return _number(name, value, float, low=0)
    if value is None:
        return None
    if name == "constellation":
        from ..mapping import Constellation

        if not isinstance(value, Constellation):
            raise ValueError(
                f"constellation must be a Constellation, got {type(value).__name__}."
            )
    elif name == "pulse":
        from ..filtering import Pulse

        if not isinstance(value, Pulse):
            raise ValueError(f"pulse must be a Pulse, got {type(value).__name__}.")
    elif name == "reference":
        if not isinstance(value, Reference):
            raise ValueError(
                f"reference must be a Reference, got {type(value).__name__}."
            )
    return value


# -----------------------------------------------------------------------------
# Display
# -----------------------------------------------------------------------------


def _format_si(value: float | None, unit: str = "Hz") -> str:
    """
    Formats a numeric value into a human-readable string with SI prefixes.

    Automatically selects the appropriate SI prefix (e.g., k, M, G, m, u, n)
    based on the magnitude of the value. Supports a wide range from
    femto (10^-15) to Peta (10^15).

    Parameters
    ----------
    value : float or None
        The numeric value to format. If `None`, returns "None".
    unit : str, default "Hz"
        The unit suffix to append (e.g., 'Hz', 'Baud', 's', 'W').

    Returns
    -------
    str
        The formatted string (e.g., '10.00 MHz', '50.00 ns').
    """
    if value is None:
        return "None"

    if abs(value) == 0:
        return f"0.00 {unit}"

    # Standard SI prefixes
    si_units = {
        -5: "f",
        -4: "p",
        -3: "n",
        -2: "µ",
        -1: "m",
        0: "",
        1: "k",
        2: "M",
        3: "G",
        4: "T",
        5: "P",
    }

    rank = int(np.floor(np.log10(abs(value)) / 3))
    # clamp to supported range
    rank = max(min(si_units.keys()), min(rank, max(si_units.keys())))

    scaled = value / (1000.0**rank)
    return f"{scaled:.2f} {si_units[rank]}{unit}"
