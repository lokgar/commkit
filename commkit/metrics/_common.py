"""Input resolution and logging shared by the metrics (private).

The reference, constellation and data-aided gain lookups and the
per-channel log lines of :mod:`commkit.metrics.quality`,
:mod:`commkit.metrics.error_rates` and :mod:`commkit.metrics.information`.
"""

import logging
from typing import Any

import numpy as np

from ..backend import ArrayType, to_device
from ..core._signal_adapter import SignalAdapter
from ..core.signal import Reference, Signal
from ..logger import logger
from ..mapping import Constellation
from ..math import normalize


def _unit_reference(tx: ArrayType, xp: Any) -> tuple[ArrayType, ArrayType]:
    """``tx`` at unit average power per channel, and its power ``(..., 1)``.

    A reference already at unit power (within 1e-3) is kept as it is; the
    power is measured once and remeasured only after normalizing.
    """
    pwr = xp.mean(xp.abs(tx) ** 2, axis=-1, keepdims=True)
    if not xp.allclose(pwr, 1.0, atol=1e-3):
        tx = normalize(tx, axis=-1, mode="average_power")
        pwr = xp.mean(xp.abs(tx) ** 2, axis=-1, keepdims=True)
    return tx, pwr


def _log_per_channel(fmt: str, *arrays: ArrayType, extra: tuple = ()) -> None:
    """Gate per-channel INFO logging behind one D2H transfer per array.

    Emits, per channel index ``ch``: ``logger.info(fmt, ch, *values, *extra)``
    where ``values`` are each array's per-channel scalar (Python ``int`` for
    integer dtypes, ``float`` otherwise) and ``extra`` are constants appended
    after the per-channel values (e.g. a fixed ``total`` symbol/bit count).
    No device transfer happens at all when INFO logging is disabled.
    """
    if not logger.isEnabledFor(logging.INFO):
        return
    arrays_np = [to_device(a, "cpu") for a in arrays]
    n_ch = arrays_np[0].shape[0]
    for ch in range(n_ch):
        values = [
            int(a[ch]) if np.issubdtype(a.dtype, np.integer) else float(a[ch])
            for a in arrays_np
        ]
        logger.info(fmt, ch, *values, *extra)


# -----------------------------------------------------------------------------
# Input resolution shared by the metrics
# -----------------------------------------------------------------------------


def _reference(
    adapter: SignalAdapter, reference: Any, field: str, name: str, hint: str = ""
) -> Any:
    """The explicit reference, else the Signal's ``reference.<field>``."""
    if reference is not None:
        return reference
    ref = None if adapter.signal is None else adapter.signal.reference
    value = None if ref is None else getattr(ref, field)
    if value is None:
        raise ValueError(
            f"{name} needs reference {field}: pass reference= or a Signal whose "
            f"reference has {field}{hint}."
        )
    return value


def _decoded(bits: Any, reference: Any, name: str) -> tuple[Any, Any, Any] | None:
    """For a :class:`~commkit.coding.DecodeResult`: its bits, the reference's
    information bits and the reference's code (``None`` for an array
    reference); ``None`` for any other ``bits``."""
    from ..coding import DecodeResult

    if not isinstance(bits, DecodeResult):
        return None
    ref = reference.reference if isinstance(reference, Signal) else reference
    if isinstance(ref, Reference):
        info = ref.info_bits if ref.info_bits is not None else ref.bits
        if info is None:
            raise ValueError(f"{name}: the reference has no bits.")
        return bits.bits, info, ref.code
    if ref is None:
        raise ValueError(
            f"{name}: a DecodeResult needs the transmitted Signal, its Reference "
            "or the information bits as reference."
        )
    return bits.bits, ref, None


def _constellation(adapter: SignalAdapter, constellation: Any, name: str) -> Any:
    """The decision constellation: the argument, else the Signal's."""
    constellation = adapter.resolve_choice("constellation", constellation)
    if constellation is None:
        raise ValueError(
            f"{name} needs a constellation: pass constellation= or a Signal that "
            "has one."
        )
    if not isinstance(constellation, Constellation):
        raise TypeError(
            f"{name}: constellation must be a Constellation, got "
            f"{type(constellation).__name__}; use e.g. Constellation.qam(16)."
        )
    return constellation


def _paired(
    rx: ArrayType, reference: Any, xp: Any, name: str
) -> tuple[ArrayType, ArrayType]:
    """Move the reference to ``rx``'s device and precision.

    The precision is matched when both are floating point, so a complex128
    reference does not promote complex64 data.  The shapes must agree.
    """
    ref = to_device(reference, "cpu" if xp is np else "gpu")
    if rx.shape != ref.shape:
        raise ValueError(
            f"{name}: shape mismatch: received {rx.shape} != reference {ref.shape}."
        )
    if ref.dtype.kind in "fc" and rx.dtype.kind in "fc":
        base = np.complex64 if ref.dtype.kind == "c" else np.float32
        ref = ref.astype(np.result_type(base, rx.real.dtype), copy=False)
    return rx, ref


def _gain(
    rx: ArrayType, tx: ArrayType, xp: Any, tx_power: ArrayType | None = None
) -> ArrayType:
    """The data-aided gain ``g = |<r s*>| / <|s|^2>`` per channel, ``(..., 1)``.

    ``g`` is the amplitude of the symbols in ``rx`` (the transmittance), so
    ``rx / g`` is on the scale of ``tx`` whatever the noise; normalizing to
    unit total power instead shrinks the symbols by ``1/sqrt(1 + 1/SNR)``.
    Only the magnitude is used: a rotation, ``pi`` included, stays visible.
    A channel without reference power or correlation gets ``g = 1``.
    ``tx_power`` is ``<|s|^2>`` ``(..., 1)`` when the caller has it.
    """
    num = xp.abs(xp.mean(rx * xp.conj(tx), axis=-1, keepdims=True))
    den = (
        xp.mean(xp.abs(tx) ** 2, axis=-1, keepdims=True)
        if tx_power is None
        else tx_power
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        g = num / den
    return xp.where((den > 0) & (g > 0), g, 1.0).astype(rx.real.dtype)


def _reference_scaled(
    adapter: SignalAdapter, rx: ArrayType, xp: Any, name: str, num_skip: int
) -> ArrayType:
    """Divide a Signal's symbols by their data-aided gain.

    The gain is estimated against the reference symbols after the first
    ``num_skip``.  Arrays and Signals without a reference are returned as
    given.
    """
    sig = adapter.signal
    if sig is None or sig.reference is None:
        return rx
    _, ref = _paired(rx, sig.reference.symbols, xp, name)
    measured = _skip(rx, num_skip, name)
    return rx / _gain(measured, _skip(ref, num_skip, name), xp)


def _skip(x: ArrayType, n: int, name: str) -> ArrayType:
    """Drop the first ``n`` entries of the last axis; nothing left raises."""
    if n < 0:
        raise ValueError(f"{name}: num_skip_symbols must be >= 0, got {n}.")
    x = x[..., n:]
    if x.shape[-1] == 0:
        raise ValueError(f"{name}: nothing to measure (no symbols after skipping).")
    return x
