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

from .. import helpers
from ..backend import (
    ArrayType,
    get_array_module,
    get_scipy_module,
    to_device,
)

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
        symbols = helpers.validate_array(self.symbols, name="symbols")
        if symbols.ndim not in (1, 2):
            raise ValueError(
                f"symbols must have shape (N,) or (C, N), got {symbols.shape}."
            )
        object.__setattr__(self, "symbols", symbols)
        if self.bits is not None:
            bits = helpers.validate_array(self.bits, name="bits")
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
    resolved_symbols, resolved_bits : array_like, optional
        Bridge-only caches written by ``resolve_symbols`` and
        ``demap_symbols_hard``; removed in module pass 3.8.

    Notes
    -----
    Construction does no hidden work: no bit-to-symbol mapping, no
    normalization, no device move and no shape guessing.  The factories
    (``generate``, ``SingleCarrierFrame.to_signal``) do that work.

    The 1.x attributes (``mod_scheme``, ``mod_order``, ``source_symbols``,
    ``pulse_shape``, ...) remain as read-only properties derived from the new
    fields until every module is migrated.
    """

    samples: Any
    sampling_rate: float
    symbol_rate: float

    constellation: Constellation | None = None
    pulse: Pulse | None = None
    reference: Reference | None = None
    frame: Any | None = field(default=None, repr=False)
    center_frequency: float = 0.0

    # Bridge-only pipeline caches (removed in 3.8).
    resolved_symbols: Any | None = field(default=None, repr=False)
    resolved_bits: Any | None = field(default=None, repr=False)

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
            ("Symbol rate", helpers.format_si(self.symbol_rate, "Baud")),
        ]
        if self.constellation is not None:
            rows.append(
                (
                    "Bit rate",
                    helpers.format_si(
                        self.symbol_rate * self.constellation.bits_per_symbol, "bps"
                    ),
                )
            )
        rows += [
            ("Sampling rate", helpers.format_si(self.sampling_rate, "Hz")),
            ("Samples per symbol", f"{self.sps:.2f}"),
            ("Duration", helpers.format_si(self.duration, "s")),
            ("Center frequency", helpers.format_si(self.center_frequency, "Hz")),
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

    def replace_samples(
        self,
        samples: Any,
        *,
        _preserve_resolved: bool = False,
        **metadata: Any,
    ) -> Signal:
        """Bridge: :meth:`replace` with new samples that also drops ``resolved_*``.

        Removed with the ``resolved_*`` caches in module pass 3.8.
        """
        if not _preserve_resolved:
            metadata = {"resolved_symbols": None, "resolved_bits": None, **metadata}
        return self.replace(samples=samples, **metadata)

    def to(self, device: str) -> Signal:
        """
        Return a copy of this signal with its arrays on ``device``.

        Moves the samples, the reference and the bridge caches.  Host
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
        moved: dict[str, Any] = {
            name: to_device(value, device)
            for name in ("resolved_symbols", "resolved_bits")
            if (value := getattr(self, name)) is not None
        }
        if self.reference is not None:
            moved["reference"] = self.reference.to(device)
        return self.replace(samples=to_device(self.samples, device), **moved)

    def time_axis(self) -> ArrayType:
        """Time of each sample in seconds, starting at 0, shape ``(N,)``."""
        n_samples = self.samples.shape[-1]
        return self.xp.arange(0, n_samples) / self.sampling_rate

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
        return self.samples.shape[0]

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return self.samples.shape[-1] / self.sampling_rate

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

    # -------------------------------------------------------------------------
    # 1.x bridge (read-only; removed as modules migrate, gone in 4.1)
    # -------------------------------------------------------------------------

    @property
    def source_symbols(self) -> Any:
        return None if self.reference is None else self.reference.symbols

    @property
    def source_bits(self) -> Any:
        return None if self.reference is None else self.reference.bits

    @property
    def mod_scheme(self) -> str | None:
        c = self.constellation
        if c is None or c.family is None:
            return None
        return "PS-QAM" if c.pmf is not None else c.family.upper()

    @property
    def mod_order(self) -> int | None:
        return None if self.constellation is None else self.constellation.order

    @property
    def mod_unipolar(self) -> bool | None:
        return None if self.constellation is None else self.constellation.unipolar

    @property
    def ps_pmf(self) -> Any:
        if self.constellation is not None:
            return self.constellation.pmf
        if self.frame is not None:
            return self.frame.payload_constellation.pmf
        return None

    @property
    def signal_type(self) -> str | None:
        return "Single-Carrier Frame" if self.frame is not None else None

    @property
    def pulse_shape(self) -> str | None:
        return None if self.pulse is None else _PULSE_NAMES[type(self.pulse).__name__]

    @property
    def filter_span(self) -> int:
        return getattr(self.pulse, "span", 10)

    @property
    def rrc_rolloff(self) -> float:
        return self.pulse.rolloff if self.pulse_shape == "rrc" else 0.35  # type: ignore[union-attr]

    @property
    def rc_rolloff(self) -> float:
        return self.pulse.rolloff if self.pulse_shape == "rc" else 0.35  # type: ignore[union-attr]

    @property
    def duty_cycle(self) -> float:
        if self.pulse_shape == "gaussian":
            return self.pulse.fwhm  # type: ignore[union-attr]
        return getattr(self.pulse, "duty_cycle", 1.0)

    @property
    def rise_time(self) -> float:
        return getattr(self.pulse, "rise_time", 0.0)

    @property
    def mod_rz(self) -> bool:
        return self.pulse_shape in ("rect", "smoothrect") and self.duty_cycle < 1.0


_PULSE_NAMES = {
    "RRC": "rrc",
    "RC": "rc",
    "Gaussian": "gaussian",
    "Rect": "rect",
    "SmoothRect": "smoothrect",
}


# -----------------------------------------------------------------------------
# Field validation (shared by construction and Signal.replace)
# -----------------------------------------------------------------------------

_FIELD_NAMES = frozenset(f.name for f in dataclasses.fields(Signal))


def _validate_samples(v: Any) -> Any:
    """Coerce samples to a NumPy/CuPy array with time on the last axis."""
    arr = helpers.validate_array(v, name="samples")
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
