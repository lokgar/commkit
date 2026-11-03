"""Signal quality against the reference: :func:`evm`, :func:`snr` and :func:`estimate_noise_variance`.

All three divide the received symbols by the data-aided gain ``g = |<r s*>|
/ <|s|^2>`` against the reference normalized to unit average power, so
they are unbiased at any SNR and a rotation stays an error.
"""

from typing import Any

import numpy as np

from .._array import to_report_scalar
from ..backend import ArrayType, dispatch
from ..core._signal_adapter import adapt_signal
from ..core.signal import Signal
from ..logger import logger
from ..mapping import Constellation
from ..math import linear_to_db
from ._common import (
    _constellation,
    _gain,
    _log_per_channel,
    _paired,
    _reference,
    _skip,
    _unit_reference,
)

__all__ = ["estimate_noise_variance", "evm", "snr"]


def evm(
    symbols: ArrayType | Signal,
    reference: ArrayType | None = None,
    *,
    constellation: Constellation | None = None,
    blind: bool = False,
    num_skip_symbols: int = 0,
) -> float | np.ndarray:
    """Error vector magnitude (RMS), in percent.

    Data-aided by default: the error is measured against the known
    transmitted symbols.  With ``blind=True`` it is measured against the
    nearest constellation point instead, as a vector signal analyzer does;
    no reference is needed, but at low SNR wrong decisions make the estimate
    optimistic.

    Parameters
    ----------
    symbols : array_like or Signal
        Received symbols at one sample per symbol, ``(N,)`` or ``(C, N)``.  A
        Signal must be at one sample per symbol; its ``reference.symbols`` is
        the default reference.
    reference : array_like, optional
        Transmitted symbols, same shape as ``symbols``.  Required for
        data-aided EVM of an array.
    constellation : Constellation, optional
        Decision constellation for ``blind=True``.  Defaults to the Signal's.
        The symbols are compared with its points as they are, so they must be
        on its scale (unit power).
    blind : bool, default False
        Measure against hard decisions instead of the reference.
    num_skip_symbols : int, default 0
        Leading symbols left out (for example equalizer training).

    Returns
    -------
    float or ndarray
        EVM in percent: a float for ``(N,)`` input, ``(C,)`` for ``(C, N)``.
        ``inf`` where the reference has no power.  In dB:
        ``20 * log10(evm / 100)``.

    Raises
    ------
    ValueError
        On a missing reference or constellation, a shape mismatch, an empty
        selection, or a frame Signal or one not at one sample per symbol.

    Notes
    -----
    Data-aided EVM normalizes the reference to unit average power per
    channel and divides the received symbols by the data-aided gain
    ``|<r s*>| / <|s|^2>``, so a common gain does not count as error at any
    SNR (normalizing to unit total power would shrink the symbols by
    ``1/sqrt(1 + 1/SNR)``).  No rotation is applied.
    """
    name = "evm()"
    adapter = adapt_signal(symbols, function_name=name)
    rx, xp, _ = dispatch(adapter.symbol_array())
    axis = -1

    if blind:
        if reference is not None:
            raise ValueError(
                f"{name}: blind=True measures against hard decisions; drop reference=."
            )
        c = _constellation(adapter, constellation, name)
        rx = _skip(rx, num_skip_symbols, name)
        # No gain correction: the caller passes symbols at the constellation's
        # scale.
        points = c.points.astype(np.result_type(np.complex64, rx.real.dtype))
        tx = xp.asarray(points)[c._nearest(rx.reshape(-1))].reshape(rx.shape)
        ref_pwr = xp.mean(xp.abs(tx) ** 2, axis=axis)
    else:
        ref = _reference(adapter, reference, "symbols", name, hint=", or blind=True")
        rx, tx = _paired(rx, ref, xp, name)
        rx = _skip(rx, num_skip_symbols, name)
        tx = _skip(tx, num_skip_symbols, name)
        tx, tx_pwr = _unit_reference(tx, xp)
        rx = rx / _gain(rx, tx, xp, tx_pwr)
        ref_pwr = tx_pwr[..., 0]

    error_power = xp.mean(xp.abs(rx - tx) ** 2, axis=axis, dtype=xp.float64)
    # The reference power is 1 after normalization.
    evm_percent = xp.sqrt(error_power) * 100.0
    evm_percent = xp.where(ref_pwr < 1e-20, xp.inf, evm_percent)
    out = to_report_scalar(evm_percent)
    if np.any(np.isinf(out)):
        logger.warning("Reference signal power near zero in one or more channels.")
    if isinstance(out, float):
        logger.info("EVM: %.2f%%", out)
    else:
        _log_per_channel("EVM Ch%s: %.2f%%", out)
    return out


def _error_power(
    symbols: Any, reference: Any, num_skip_symbols: int, name: str
) -> tuple[ArrayType, ArrayType, Any]:
    """``E[|r / g - s|^2]`` per channel (float64, on the device) with the
    reference at unit power, the reference's power, and the array module."""
    adapter = adapt_signal(symbols, function_name=name)
    rx, xp, _ = dispatch(adapter.symbol_array())
    rx, tx = _paired(rx, _reference(adapter, reference, "symbols", name), xp, name)
    rx = _skip(rx, num_skip_symbols, name)
    tx = _skip(tx, num_skip_symbols, name)
    tx, tx_pwr = _unit_reference(tx, xp)
    rx = rx / _gain(rx, tx, xp, tx_pwr)
    noise_power = xp.mean(xp.abs(rx - tx) ** 2, axis=-1, dtype=xp.float64)
    return noise_power, tx_pwr[..., 0], xp


def estimate_noise_variance(
    symbols: ArrayType | Signal,
    reference: ArrayType | None = None,
    *,
    num_skip_symbols: int = 0,
) -> float | np.ndarray:
    """Data-aided variance of the noise in the symbols, on the constellation's scale.

    ``σ² = E[|r / g - s|^2]`` with the reference normalized to unit average
    power and the data-aided gain ``g`` of :func:`snr` (so
    ``σ² = 10 ** (-snr / 10)``): the ``noise_var`` that
    :func:`~commkit.mapping.compute_llr`, :func:`gmi`, :func:`ngmi` and
    :func:`mi` take for symbols on the constellation's scale, complex or
    real: each uses the likelihood of the noise those symbols carry.

    Parameters
    ----------
    symbols : array_like or Signal
        Received symbols at one sample per symbol, ``(N,)`` or ``(C, N)``.  A
        Signal must be at one sample per symbol; its ``reference.symbols`` is
        the default reference.
    reference : array_like, optional
        Transmitted symbols, same shape as ``symbols``.  Required for arrays.
    num_skip_symbols : int, default 0
        Leading symbols left out (for example equalizer training).

    Returns
    -------
    float or ndarray
        ``σ²``: a float for ``(N,)`` input, ``(C,)`` for ``(C, N)``.

    Raises
    ------
    ValueError
        As for :func:`snr`.

    Examples
    --------
    >>> noise_var = float(np.mean(estimate_noise_variance(y)))
    >>> llrs = compute_llr(y, noise_var=noise_var)
    """
    noise_power, _, _ = _error_power(
        symbols, reference, num_skip_symbols, "estimate_noise_variance()"
    )
    out = to_report_scalar(noise_power)
    if isinstance(out, float):
        logger.info("Noise variance: %.3e", out)
    else:
        _log_per_channel("Noise variance Ch%s: %.3e", out)
    return out


def snr(
    symbols: ArrayType | Signal,
    reference: ArrayType | None = None,
    *,
    num_skip_symbols: int = 0,
) -> float | np.ndarray:
    """Data-aided SNR in dB: signal power over error power.

    ``SNR = 1 / E[|r / g - s|^2]`` with the reference ``s`` normalized to
    unit average power per channel and the data-aided gain
    ``g = |<r s*>| / <|s|^2>``, which is unbiased at any SNR (normalizing
    ``r`` to unit total power is not).  No rotation is applied.  A power
    ratio in the symbols, as ``apply_awgn(snr_db=)`` sets: for complex
    symbols it equals the Es/N0 of AWGN, for real symbols it is 3.01 dB
    above it (they carry the in-phase half of the noise).

    Parameters
    ----------
    symbols : array_like or Signal
        Received symbols at one sample per symbol, ``(N,)`` or ``(C, N)``.  A
        Signal must be at one sample per symbol; its ``reference.symbols`` is
        the default reference.
    reference : array_like, optional
        Transmitted symbols, same shape as ``symbols``.  Required for arrays.
    num_skip_symbols : int, default 0
        Leading symbols left out (for example equalizer training).

    Returns
    -------
    float or ndarray
        SNR in dB: a float for ``(N,)`` input, ``(C,)`` for ``(C, N)``.
        ``inf`` without error, ``-inf`` where the reference has no power.

    Raises
    ------
    ValueError
        On a missing reference, a shape mismatch, an empty selection, or a
        frame Signal or one not at one sample per symbol.
    """
    noise_power, ref_pwr, xp = _error_power(
        symbols, reference, num_skip_symbols, "snr()"
    )
    with np.errstate(divide="ignore"):
        snr_db = linear_to_db(1.0 / noise_power, power=True)
    # No error: +inf even for a zero reference; error but no reference: -inf.
    zero_noise = noise_power < 1e-20
    snr_db = xp.where(zero_noise, xp.inf, snr_db)
    snr_db = xp.where((ref_pwr < 1e-20) & ~zero_noise, -xp.inf, snr_db)
    out = to_report_scalar(snr_db)
    if isinstance(out, float):
        logger.info("SNR: %.2f dB", out)
    else:
        _log_per_channel("SNR Ch%s: %.2f dB", out)
    return out
