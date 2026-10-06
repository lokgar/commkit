"""
Hard bit mapping and demapping.

``map_bits`` maps bits to the points of a :class:`Constellation`;
``demap_symbols_hard`` returns the bits of the nearest point (minimum
Euclidean distance).  Both run on the data's device.
"""

from typing import Any

from ..backend import ArrayType
from ..core._signal_adapter import adapt_signal
from ..core.signal import Signal
from ..logger import logger
from .constellation import Constellation

__all__ = ["demap_symbols_hard", "map_bits"]


def map_bits(bits: ArrayType, *, constellation: Constellation) -> ArrayType:
    """
    Map bits to constellation points.

    Groups of ``k = constellation.bits_per_symbol`` bits (MSB first) become
    one symbol, following ``constellation.bit_labels``.

    Parameters
    ----------
    bits : array_like
        Bits (0/1), shape ``(..., n)`` with ``n`` a multiple of ``k``.
    constellation : Constellation
        Points and bit labels to map onto.

    Returns
    -------
    array_like
        Symbols of shape ``(..., n / k)`` on the bits' device: ``complex64``,
        or ``float32`` for a real constellation.
    """
    _check_constellation(constellation, "map_bits()")
    return constellation.map(bits)


def demap_symbols_hard(
    symbols: ArrayType | Signal, *, constellation: Constellation | None = None
) -> ArrayType | Signal:
    """
    Hard decisions: the bits of the nearest constellation point.

    Symbols are compared with ``constellation.points`` as they are, so they
    must be on the constellation's scale (unit average power for the
    factories, including shaped constellations).

    Parameters
    ----------
    symbols : array_like or Signal
        Received symbols at one sample per symbol, shape ``(..., N)``.  A
        :class:`Signal` supplies its ``resolved_symbols``.
    constellation : Constellation, optional
        Decision constellation.  Defaults to the Signal's ``constellation``;
        required for array input.

    Returns
    -------
    array_like or Signal
        ``int8`` bits of shape ``(..., N * k)`` on the symbols' device.  For
        Signal input, a new Signal with ``resolved_bits`` set (bridge until
        module pass 3.8).
    """
    signal_adapter = adapt_signal(
        symbols, function_name="demap_symbols_hard()", field="resolved_symbols"
    )
    sig = signal_adapter.signal
    if sig is not None and sig.signal_type is not None:
        logger.warning(
            "demap_symbols_hard() called on a frame-generated signal - skipping. "
            "Extract the payload segment via frame.get_structure_map() and build "
            "a plain Signal before demapping."
        )
        return sig.replace()
    constellation = signal_adapter.resolve_choice("constellation", constellation)
    _check_constellation(constellation, "demap_symbols_hard()")
    if sig is None:
        return constellation.demap(signal_adapter.array)
    if signal_adapter.array is None:
        raise ValueError(
            "No resolved symbols available. Call resolve_symbols(sig) first."
        )
    bits = constellation.demap(signal_adapter.array)
    return signal_adapter.replace_signal_field("resolved_bits", bits)


def _check_constellation(constellation: Any, function_name: str) -> None:
    if constellation is None:
        raise ValueError(
            f"{function_name} needs a constellation: pass constellation= or "
            "a Signal that has one."
        )
    if not isinstance(constellation, Constellation):
        raise TypeError(
            f"{function_name}: constellation must be a Constellation, got "
            f"{type(constellation).__name__}; use e.g. Constellation.qam(16)."
        )
