"""Signal quality against the reference: :func:`evm`, :func:`snr` and :func:`noise_variance`.

All three measure the relative error power ``E[|r / g - s|^2] / E[|s|^2]``
per channel, with the data-aided gain ``g = |<r s*>| / <|s|^2>``: it does
not depend on the scale of either the symbols or the reference, is
unbiased at any SNR, and keeps a rotation as an error.
"""

from typing import Any

import numpy as np

from .._array import to_report_scalar
from ..backend import ArrayType, dispatch
from ..core._signal_input import SignalInput
from ..core.signal import Signal
from ..logger import logger
from ..mapping import Constellation
from ..math import _data_aided_gain, linear_to_db
from ._common import (
    _log_per_channel,
    _paired,
    _reference,
    _require_points,
    _skip,
)

# Below this reference power a channel counts as empty (EVM inf, SNR -inf).
_NO_POWER = 1e-20

__all__ = ["noise_variance", "evm", "snr"]


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
        on its scale; the error is relative to the mean power of the decided
        points.  Data-aided, a known constellation (this or the Signal's)
        checks that the reference symbols are its points.
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
        On a missing reference or constellation, reference symbols that are
        not points of a known constellation, a shape mismatch, an empty
        selection, or a frame Signal or one not at one sample per symbol.

    Notes
    -----
    Data-aided EVM is ``100·sqrt(E[|r/g - s|^2] / E[|s|^2])`` per channel
    with the data-aided gain ``g = |<r s*>| / <|s|^2>``, so a common gain
    does not count as error at any SNR (normalizing to unit total power
    would shrink the symbols by ``1/sqrt(1 + 1/SNR)``), and neither the
    symbols' nor the reference's scale matters.  Blind EVM fits no gain and
    divides the error power by the mean power of the decided points.  No
    rotation is applied.
    """
    name = "evm()"
    inp = SignalInput(symbols, name)
    rx, xp, _ = dispatch(inp.symbols())
    axis = -1

    if blind:
        if reference is not None:
            raise ValueError(
                f"{name}: blind=True measures against hard decisions; drop reference=."
            )
        c = inp.constellation(constellation, required=True)
        rx = _skip(rx, num_skip_symbols, name)
        # No gain correction: the caller passes symbols at the constellation's
        # scale.  The error is relative to the power of the decided points.
        points = c.points.astype(np.result_type(np.complex64, rx.real.dtype))
        tx = xp.asarray(points)[c._nearest(rx.reshape(-1))].reshape(rx.shape)
        ref_pwr = xp.mean(xp.abs(tx) ** 2, axis=axis, dtype=xp.float64)
        error_power = xp.mean(xp.abs(rx - tx) ** 2, axis=axis, dtype=xp.float64)
        error_power = error_power / xp.where(ref_pwr < _NO_POWER, 1.0, ref_pwr)
    else:
        ref = _reference(inp, reference, "symbols", name, hint=", or blind=True")
        rx, tx = _paired(rx, ref, xp, name)
        rx = _skip(rx, num_skip_symbols, name)
        tx = _skip(tx, num_skip_symbols, name)
        known = inp.constellation(constellation)
        if known is not None:
            _require_points(tx, known, xp, name)
        error_power, ref_pwr = _relative_error(rx, tx, xp)

    evm_percent = xp.sqrt(error_power) * 100.0
    evm_percent = xp.where(ref_pwr < _NO_POWER, xp.inf, evm_percent)
    out = to_report_scalar(evm_percent)
    if np.any(np.isinf(out)):
        logger.warning("Reference signal power near zero in one or more channels.")
    if isinstance(out, float):
        logger.info("EVM: %.2f%%", out)
    else:
        _log_per_channel("EVM", "%.2f%%", out)
    return out


def _relative_error(
    rx: ArrayType, tx: ArrayType, xp: Any
) -> tuple[ArrayType, ArrayType]:
    """``E[|r/g - s|^2] / E[|s|^2]`` and ``E[|s|^2]`` per channel (float64,
    on the device).

    Invariant under any scaling of ``r`` or ``s``: ``g`` scales with ``r``
    and inversely with ``s``, and the ratio cancels what remains.  A channel
    without reference power keeps its unscaled error power.
    """
    ref_pwr = xp.mean(xp.abs(tx) ** 2, axis=-1, keepdims=True, dtype=xp.float64)
    rx = rx / _data_aided_gain(rx, tx, xp, ref_pwr)
    error = xp.mean(xp.abs(rx - tx) ** 2, axis=-1, keepdims=True, dtype=xp.float64)
    error = error / xp.where(ref_pwr < _NO_POWER, 1.0, ref_pwr)
    return error[..., 0], ref_pwr[..., 0]


def _error_power(
    symbols: Any, reference: Any, num_skip_symbols: int, name: str
) -> tuple[ArrayType, ArrayType, Any]:
    """The relative error power and reference power of
    :func:`_relative_error`, for a Signal or arrays, and the array module."""
    inp = SignalInput(symbols, name)
    rx, xp, _ = dispatch(inp.symbols())
    rx, tx = _paired(rx, _reference(inp, reference, "symbols", name), xp, name)
    rx = _skip(rx, num_skip_symbols, name)
    tx = _skip(tx, num_skip_symbols, name)
    error, ref_pwr = _relative_error(rx, tx, xp)
    return error, ref_pwr, xp


def noise_variance(
    symbols: ArrayType | Signal,
    reference: ArrayType | None = None,
    *,
    num_skip_symbols: int = 0,
) -> float | np.ndarray:
    """Data-aided variance of the noise in the symbols, on the reference's unit-power scale.

    ``σ² = E[|r / g - s|^2] / E[|s|^2]`` with the data-aided gain ``g`` of
    :func:`snr` (so ``σ² = 10 ** (-snr / 10)``): the noise variance of the
    symbols once :func:`~commkit.equalization.correct_gain` has put them on
    the scale of a unit-power reference, such as the points of a factory
    constellation.  That is the ``noise_var`` that
    :func:`~commkit.mapping.compute_llr`, :func:`gmi`, :func:`ngmi` and
    :func:`mi` take for those symbols, complex or real: each uses the
    likelihood of the noise they carry.  For symbols at another scale
    (after :func:`~commkit.math.normalize`, whose unit total power includes
    the noise), the LLRs would be mis-scaled: correct the gain first.

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

    See Also
    --------
    commkit.equalization.estimate_gain : The gain ``g``, as an estimate.
    commkit.equalization.correct_gain : Divide symbols by it.

    Examples
    --------
    >>> y = correct_gain(y, estimate_gain(y, tx))
    >>> noise_var = float(np.mean(noise_variance(y, tx)))
    >>> llrs = compute_llr(y, constellation=c, noise_var=noise_var)
    """
    noise_power, _, _ = _error_power(
        symbols, reference, num_skip_symbols, "noise_variance()"
    )
    out = to_report_scalar(noise_power)
    if isinstance(out, float):
        logger.info("Noise variance: %.3e", out)
    else:
        _log_per_channel("Noise variance", "%.3e", out)
    return out


def snr(
    symbols: ArrayType | Signal,
    reference: ArrayType | None = None,
    *,
    num_skip_symbols: int = 0,
) -> float | np.ndarray:
    """Data-aided SNR in dB: signal power over error power.

    ``SNR = E[|s|^2] / E[|r / g - s|^2]`` per channel with the data-aided
    gain ``g = |<r s*>| / <|s|^2>``, which is unbiased at any SNR
    (normalizing ``r`` to unit total power is not) and independent of the
    scale of ``r`` and ``s``.  No rotation is applied.  A power
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
    with np.errstate(divide="ignore", invalid="ignore"):
        snr_db = linear_to_db(1.0 / noise_power, power=True)
    # No error: +inf even for a zero reference; error but no reference: -inf.
    zero_noise = noise_power < _NO_POWER
    snr_db = xp.where(zero_noise, xp.inf, snr_db)
    snr_db = xp.where((ref_pwr < _NO_POWER) & ~zero_noise, -xp.inf, snr_db)
    out = to_report_scalar(snr_db)
    if isinstance(out, float):
        logger.info("SNR: %.2f dB", out)
    else:
        _log_per_channel("SNR", "%.2f dB", out)
    return out
