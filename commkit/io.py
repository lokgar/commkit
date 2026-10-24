"""Signal storage: :func:`save_npz` and :func:`load_npz`.

A Signal is stored as a NumPy ``.npz`` archive with these entries:

``samples``
    The samples (always present).
``reference_symbols``, ``reference_bits``
    The reference (omitted without one; bits omitted when ``None``).
``constellation_points``, ``constellation_bit_labels``, ``constellation_pmf``
    The constellation (omitted when ``None``; the pmf only when shaped).
``__metadata__``
    A 0-d unicode array holding a JSON string with the rates, the center
    frequency, the constellation family and the pulse (type name and fields).
``__frame_metadata__``
    A 0-d unicode array holding a JSON string with the fields of the
    Signal's frame (omitted without a frame).
``frame_payload_symbols``, ``frame_pilot_symbols``, ``frame_payload_bits``
    The frame's generated symbols and bits (omitted without a frame, or
    without pilots).
``frame_payload_constellation_*``, ``frame_pilot_constellation_*``
    The frame's constellations, stored as the Signal's.

The archive contains only numeric and unicode arrays, so it is read with
``allow_pickle=False``: loading a file never executes code from it.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from . import backend as _backend

__all__ = ["load_npz", "save_npz"]

if TYPE_CHECKING:
    from .core import Signal

# -----------------------------------------------------------------------------
# Internal constants
# -----------------------------------------------------------------------------

# Scalar metadata fields round-tripped through JSON.
_META_FIELDS: tuple[str, ...] = ("sampling_rate", "symbol_rate", "center_frequency")

# Pulse classes that may be reconstructed from an archive.
_PULSE_TYPES: tuple[str, ...] = ("RRC", "RC", "Gaussian", "Rect", "SmoothRect")


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def save_npz(
    signal: Signal,
    path: str | Path,
    *,
    compressed: bool = True,
) -> None:
    """Save a :class:`Signal` to a NumPy archive (``.npz``).

    The layout is described in the module docstring of :mod:`commkit.io`.
    Samples on the GPU are copied to the host first.

    Parameters
    ----------
    signal : Signal
        The signal to persist.
    path : str or Path
        Destination path; ``.npz`` is appended when missing.
    compressed : bool, default True
        Write with ``savez_compressed`` (zlib).  ``False`` writes
        uncompressed with ``savez``: faster, but a larger file.

    Notes
    -----
    When the Signal has a ``frame`` (one built by
    :meth:`SingleCarrierFrame.to_signal`), the frame is stored too
    (``__frame_metadata__``, ``frame_payload_symbols``, ...).
    :func:`load_npz` rebuilds it, so ``estimate_timing``, ``extract_payload``
    and the frame-aware equalizer steps work on a loaded Signal as on a
    freshly generated one.

    Examples
    --------
    >>> save_npz(sig, "capture.npz")
    >>> save_npz(sig, "capture", compressed=False)
    """
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")

    # -------------------------------------------------------------------------
    # Collect arrays
    # -------------------------------------------------------------------------
    arrays: dict[str, Any] = {"samples": _backend.to_device(signal.samples, "CPU")}

    meta: dict = {f: getattr(signal, f) for f in _META_FIELDS}

    ref = signal.reference
    if ref is not None:
        arrays["reference_symbols"] = _backend.to_device(ref.symbols, "CPU")
        if ref.bits is not None:
            arrays["reference_bits"] = _backend.to_device(ref.bits, "CPU")

    _put_constellation(signal.constellation, "constellation", arrays, meta)

    pulse = signal.pulse
    meta["pulse"] = (
        None
        if pulse is None
        else {"type": type(pulse).__name__, **dataclasses.asdict(pulse)}
    )

    # -------------------------------------------------------------------------
    # Serialize originating SingleCarrierFrame (if present)
    # -------------------------------------------------------------------------
    # The frame holds the complete transmit-side description (structure map,
    # pilot masks, preamble sequence, generated payload/pilot symbols).
    # Saving it ensures that correct_timing() and pilot extraction all work
    # after a save/load round-trip without requiring the caller to keep a
    # reference to the original frame object.
    frame = signal.frame
    if frame is not None:
        # All public fields are JSON-serializable primitives; the nested
        # Preamble dataclass becomes a dict.
        # _frame_type stores the class name so load_npz can reconstruct the
        # correct type when multiple frame classes exist (SingleCarrierFrame,
        # future OFDMFrame, etc.) without hardcoding the class.
        frame_dict = _init_fields(frame)
        for name in ("payload_constellation", "pilot_constellation"):
            _put_constellation(
                frame_dict.pop(name), f"frame_{name}", arrays, frame_dict
            )
        if frame_dict.get("preamble") is not None:
            frame_dict["preamble"] = _init_fields(frame_dict["preamble"])
        frame_dict["_frame_type"] = type(frame).__name__
        arrays["__frame_metadata__"] = _json_array(frame_dict)

        # Save the generated symbol/bit arrays from the frame's cache, which the
        # field dict above does not include.  Payload symbols and bits are
        # random; pilot symbols are deterministic but cheap to cache anyway.
        for npz_key, frame_attr in (
            ("frame_payload_symbols", "payload_symbols"),
            ("frame_pilot_symbols", "pilot_symbols"),
            ("frame_payload_bits", "payload_bits"),
        ):
            arr = getattr(frame, frame_attr, None)
            if arr is not None:
                arrays[npz_key] = _backend.to_device(arr, "CPU")

    # -------------------------------------------------------------------------
    # Build metadata dict and serialize to JSON
    # -------------------------------------------------------------------------
    arrays["__metadata__"] = _json_array(meta)

    # -------------------------------------------------------------------------
    # Write
    # -------------------------------------------------------------------------
    if compressed:
        np.savez_compressed(path, **arrays)  # type: ignore[arg-type]
    else:
        np.savez(path, **arrays)  # type: ignore[arg-type]


def _put_constellation(c: Any, key: str, arrays: dict, meta: dict) -> None:
    """Store constellation ``c`` (or ``None``) under ``key``."""
    meta[f"{key}_family"] = None if c is None else c.family
    if c is not None:
        arrays[f"{key}_points"] = c.points
        arrays[f"{key}_bit_labels"] = c.bit_labels
        if c.pmf is not None:
            arrays[f"{key}_pmf"] = c.pmf


def _get_constellation(key: str, data: Any, meta: dict) -> Any:
    """The constellation stored under ``key``, or ``None``."""
    from .mapping import Constellation

    if f"{key}_points" not in data:
        return None
    return Constellation(
        data[f"{key}_points"],
        bit_labels=data[f"{key}_bit_labels"],
        pmf=data[f"{key}_pmf"] if f"{key}_pmf" in data else None,
        family=meta[f"{key}_family"],
    )


def _init_fields(obj: Any) -> dict:
    """The constructor arguments of a dataclass instance (no derived fields)."""
    return {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj) if f.init}


def load_npz(
    path: str | Path,
    *,
    device: str = "cpu",
) -> Signal:
    """Load a :class:`Signal` from a ``.npz`` archive written by :func:`save_npz`.

    Parameters
    ----------
    path : str or Path
        Path to the archive; ``.npz`` is appended when missing.
    device : {"cpu", "gpu"}, default "cpu"
        Device of the loaded Signal's arrays.

    Returns
    -------
    Signal
        The stored samples (in their stored dtype), facts, constellation,
        pulse, reference and frame.

    Raises
    ------
    ValueError
        On an unknown device, pulse type or frame type, or an archive whose
        metadata is not JSON (written by commkit < 2.0, which pickled it).

    Notes
    -----
    If the archive contains ``__frame_metadata__``, the frame is rebuilt
    (its class named by the ``_frame_type`` key of the metadata) with its
    stored payload and pilots, and attached as ``sig.frame``.

    Examples
    --------
    >>> sig = load_npz("capture.npz")                   # on the CPU
    >>> sig_gpu = load_npz("capture.npz", device="gpu")
    """
    from .core import Signal
    from .core._signal_adapter import require_device

    device = require_device(device, "load_npz()")
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")

    # Metadata is stored as plain unicode arrays, so pickle is never needed:
    # loading an untrusted archive cannot execute code.
    data = np.load(path, allow_pickle=False)

    # -------------------------------------------------------------------------
    # Parse JSON metadata
    # -------------------------------------------------------------------------
    meta: dict = _read_json(data, "__metadata__")

    # Build Signal constructor kwargs
    # -------------------------------------------------------------------------
    from . import filtering
    from .core import Reference

    kwargs: dict = {f: meta[f] for f in _META_FIELDS}
    kwargs["samples"] = data["samples"]

    if "reference_symbols" in data:
        kwargs["reference"] = Reference(
            symbols=data["reference_symbols"],
            bits=data["reference_bits"] if "reference_bits" in data else None,
        )
    kwargs["constellation"] = _get_constellation("constellation", data, meta)
    if meta["pulse"] is not None:
        pulse_fields = dict(meta["pulse"])
        pulse_type = pulse_fields.pop("type")
        if pulse_type not in _PULSE_TYPES:
            raise ValueError(f"Unknown pulse type {pulse_type!r} in {path}.")
        kwargs["pulse"] = getattr(filtering, pulse_type)(**pulse_fields)

    # 1.x archives may hold resolved_symbols/resolved_bits caches; they are
    # derived data and not restored.
    sig = Signal(**kwargs)

    # -------------------------------------------------------------------------
    # Reconstruct originating frame (if serialized)
    # -------------------------------------------------------------------------
    if "__frame_metadata__" in data:
        from . import core as _core

        frame_dict = _read_json(data, "__frame_metadata__")
        frame_type_name = frame_dict.pop("_frame_type", "SingleCarrierFrame")

        # Registry of known frame classes - extend here as new frame types land.
        _FRAME_CLASSES = {
            "SingleCarrierFrame": _core.SingleCarrierFrame,
        }
        frame_cls = _FRAME_CLASSES.get(frame_type_name)
        if frame_cls is None:
            raise ValueError(
                f"Cannot reconstruct frame of type {frame_type_name!r}: "
                "unknown frame class. Extend _FRAME_CLASSES in io.py."
            )
        for name in ("payload_constellation", "pilot_constellation"):
            frame_dict[name] = _get_constellation(f"frame_{name}", data, frame_dict)
            frame_dict.pop(f"frame_{name}_family")
        if frame_dict.get("preamble") is not None:
            frame_dict["preamble"] = _core.Preamble(**frame_dict["preamble"])
        frame = frame_cls(**frame_dict)

        # Refill the frame's cache so that frame.payload_symbols,
        # frame.pilot_symbols and frame.payload_bits return the original
        # generated data without re-randomizing.
        for npz_key, cache_key in (
            ("frame_payload_symbols", "payload_symbols"),
            ("frame_pilot_symbols", "pilot_symbols"),
            ("frame_payload_bits", "payload_bits"),
        ):
            if npz_key in data:
                frame._cache[cache_key] = data[npz_key]

        sig = sig.replace(frame=frame)

    # -------------------------------------------------------------------------
    # Move to target device
    # -------------------------------------------------------------------------
    return sig.to(device)


# -----------------------------------------------------------------------------
# JSON metadata helpers
# -----------------------------------------------------------------------------


def _json_default(value: Any) -> Any:
    """Convert NumPy scalars (e.g. ``np.float64``) to Python values for JSON."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot store {type(value).__name__} in npz metadata.")


def _json_array(obj: dict) -> np.ndarray:
    """A 0-d unicode array holding ``obj`` as JSON (no pickling needed)."""
    return np.array(json.dumps(obj, default=_json_default))


def _read_json(data: Any, key: str) -> dict:
    try:
        return json.loads(str(data[key]))
    except ValueError as exc:
        raise ValueError(
            f"{key!r} in this archive is not JSON. It was probably written by "
            "commkit < 2.0, which stored YAML in a pickled object array; such "
            "files are not loaded because unpickling can execute code."
        ) from exc
