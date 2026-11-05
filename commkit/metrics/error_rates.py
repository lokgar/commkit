"""Error rates of decisions: :func:`ber`, :func:`fer` and :func:`ser`.

The received symbols are decided as given, against the constellation's
points.  A :class:`~commkit.coding.DecodeResult` gives the post-FEC rates
against the reference's ``info_bits``, in blocks of the code's ``k`` bits.
"""

from typing import Any

import numpy as np

from .._array import to_report_scalar
from ..backend import ArrayType, dispatch, to_device
from ..core._signal_adapter import adapt_signal
from ..core.signal import Reference, Signal
from ..logger import logger
from ..mapping import Constellation
from ._common import (
    _constellation,
    _decoded,
    _log_per_channel,
    _paired,
    _reference,
    _skip,
)

__all__ = ["ber", "fer", "ser"]


def ber(
    bits: ArrayType | Signal | Any,
    reference: ArrayType | Signal | Reference | None = None,
    *,
    constellation: Constellation | None = None,
    num_skip_symbols: int = 0,
) -> float | np.ndarray:
    """Bit error rate.

    Parameters
    ----------
    bits : array_like, Signal or DecodeResult
        Received bits (0 and 1), ``(N_bits,)`` or ``(C, N_bits)``.  A Signal holds
        received *symbols* at one sample per symbol: they are hard-decided
        against the constellation and compared with ``reference.bits``, the
        bits on the air (the pre-FEC BER of a coded payload).  A
        :class:`~commkit.coding.DecodeResult` gives the post-FEC BER: its
        bits against the reference's ``info_bits``.
    reference : array_like, Signal or Reference, optional
        Transmitted bits, same shape as the received bits.  Required for
        arrays.  With a DecodeResult, the transmitted Signal or its
        :class:`~commkit.core.Reference` (or the information bits):
        ``ber(decode(llrs, code=sig.reference.code), sig)``.
    constellation : Constellation, optional
        Decision constellation for a Signal (defaults to the Signal's).  For
        bit arrays, needed only to convert ``num_skip_symbols`` to bits.
    num_skip_symbols : int, default 0
        Leading symbols left out, ``k = constellation.bits_per_symbol`` bits
        each.

    Returns
    -------
    float or ndarray
        BER in ``[0, 1]``: a float for 1-D input, ``(C,)`` for 2-D input.

    Raises
    ------
    ValueError
        On a missing reference or constellation, a shape mismatch, an empty
        selection, or a frame Signal or one not at one sample per symbol.
    """
    name = "ber()"
    post = _decoded(bits, reference, name)
    if post is not None:
        if num_skip_symbols:
            raise ValueError(
                f"{name}: num_skip_symbols does not apply to decoded bits."
            )
        bits, reference, _ = post
    adapter = adapt_signal(bits, function_name=name)
    received = adapter.symbol_array()
    if adapter.signal is not None or num_skip_symbols:
        c = _constellation(adapter, constellation, name)
        k = c.bits_per_symbol
        if adapter.signal is not None:
            received = c.demap(received)
    rx, xp, _ = dispatch(received)
    rx, tx = _paired(rx, _reference(adapter, reference, "bits", name), xp, name)
    if num_skip_symbols:
        rx = _skip(rx, num_skip_symbols * k, name)
        tx = _skip(tx, num_skip_symbols * k, name)
    elif rx.shape[-1] == 0:
        raise ValueError(f"{name}: nothing to measure (no bits).")

    errors = to_device(xp.sum(rx != tx, axis=-1), "cpu")  # one transfer, logged too
    total = rx.shape[-1]
    out = to_report_scalar(errors / total)
    if isinstance(out, float):
        logger.info("BER: %.2e (%s/%s errors)", out, int(errors), total)
    else:
        _log_per_channel("BER Ch%s: %.2e (%s/%s errors)", out, errors, extra=(total,))
    return out


def fer(
    bits: ArrayType | Any,
    reference: ArrayType | Signal | Reference,
    *,
    info_bits: int | None = None,
) -> float | np.ndarray:
    """Block (frame) error rate: the fraction of blocks with any bit in error.

    The post-FEC companion of :func:`ber`: compare the decoded information
    bits of :func:`commkit.coding.decode` with the reference's
    ``info_bits``, in blocks of the code's ``k`` bits:
    ``fer(decode(llrs, code=sig.reference.code), sig)``.

    Parameters
    ----------
    bits : array_like or DecodeResult
        Received bits (0 and 1), ``(N,)`` or ``(C, N)``, with ``N`` a multiple
        of ``info_bits``, or a :class:`~commkit.coding.DecodeResult`.
    reference : array_like, Signal or Reference
        Transmitted bits, same shape.  With a DecodeResult, the transmitted
        Signal or its :class:`~commkit.core.Reference`, whose ``info_bits``
        and code (``k`` bits per block) are used.
    info_bits : int, optional
        Information bits per block, the code's ``k``.  Required for arrays;
        with a Signal or Reference it defaults to the code's ``k`` and must
        agree with it.

    Returns
    -------
    float or ndarray
        FER in ``[0, 1]``: a float for 1-D input, ``(C,)`` for 2-D input.

    Raises
    ------
    ValueError
        On a shape mismatch, a length that is not a multiple of
        ``info_bits``, or no bits.
    """
    name = "fer()"
    post = _decoded(bits, reference, name)
    if post is not None:
        bits, reference, code = post
        k = None if code is None else int(code.k)
        if info_bits is None:
            info_bits = k
        elif k is not None and info_bits != k:
            raise ValueError(
                f"{name}: info_bits={info_bits} disagrees with the "
                f"reference's code (k = {k})."
            )
    if info_bits is None:
        raise ValueError(
            f"{name}: give info_bits, or a DecodeResult and the Signal or "
            "Reference that holds the code."
        )
    if isinstance(info_bits, bool) or not isinstance(info_bits, int | np.integer):
        raise ValueError(f"{name}: info_bits must be an integer, got {info_bits!r}.")
    if info_bits < 1:
        raise ValueError(f"{name}: info_bits must be >= 1, got {info_bits}.")
    rx, xp, _ = dispatch(bits)
    rx, tx = _paired(rx, reference, xp, name)
    total = rx.shape[-1]
    if total == 0:
        raise ValueError(f"{name}: nothing to measure (no bits).")
    if total % info_bits:
        raise ValueError(
            f"{name}: the last axis ({total}) is not a multiple of info_bits "
            f"({info_bits})."
        )
    wrong = (rx != tx).reshape(*rx.shape[:-1], -1, info_bits).any(axis=-1)
    errors = to_device(xp.sum(wrong, axis=-1), "cpu")  # one transfer, logged too
    blocks = total // info_bits
    out = to_report_scalar(errors / blocks)
    if isinstance(out, float):
        logger.info("FER: %.2e (%s/%s blocks)", out, int(errors), blocks)
    else:
        _log_per_channel("FER Ch%s: %.2e (%s/%s blocks)", out, errors, extra=(blocks,))
    return out


def ser(
    symbols: ArrayType | Signal,
    reference: ArrayType | None = None,
    *,
    constellation: Constellation | None = None,
    num_skip_symbols: int = 0,
) -> float | np.ndarray:
    """Symbol error rate of minimum-distance hard decisions.

    The received and the reference symbols are both decided against the
    constellation, so rounding in the reference does not count.  SER does
    not depend on the bit labeling.

    Parameters
    ----------
    symbols : array_like or Signal
        Received symbols at one sample per symbol, ``(N,)`` or ``(C, N)``, on
        the constellation's scale.  A Signal must be at one sample per
        symbol; its ``reference.symbols`` is the default reference.
    reference : array_like, optional
        Transmitted symbols, same shape as ``symbols``.  Required for arrays.
    constellation : Constellation, optional
        Decision constellation.  Defaults to the Signal's; required for
        arrays.
    num_skip_symbols : int, default 0
        Leading symbols left out (for example equalizer training).

    Returns
    -------
    float or ndarray
        SER in ``[0, 1]``: a float for ``(N,)`` input, ``(C,)`` for
        ``(C, N)``.

    Raises
    ------
    ValueError
        On a missing reference or constellation, a shape mismatch, an empty
        selection, or a frame Signal or one not at one sample per symbol.
    """
    name = "ser()"
    adapter = adapt_signal(symbols, function_name=name)
    rx, xp, _ = dispatch(adapter.symbol_array())
    c = _constellation(adapter, constellation, name)
    rx, tx = _paired(rx, _reference(adapter, reference, "symbols", name), xp, name)
    rx = _skip(rx, num_skip_symbols, name)
    tx = _skip(tx, num_skip_symbols, name)

    dec_rx = c._nearest(rx.reshape(-1)).reshape(rx.shape)
    dec_tx = c._nearest(tx.reshape(-1)).reshape(tx.shape)
    errors = to_device(xp.sum(dec_rx != dec_tx, axis=-1), "cpu")  # logged too
    total = rx.shape[-1]
    out = to_report_scalar(errors / total)
    if isinstance(out, float):
        logger.info("SER: %.2e (%s/%s errors)", out, int(errors), total)
    else:
        _log_per_channel("SER Ch%s: %.2e (%s/%s errors)", out, errors, extra=(total,))
    return out
