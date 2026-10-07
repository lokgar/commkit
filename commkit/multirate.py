"""
Multirate signal processing and resampling.

This module provides high-performance implementations of multirate
operations, including interpolation, decimation, and rational rate
conversion using polyphase filter banks.

Notes on power scaling
----------------------
All rate-changing functions in this module delegate to
``scipy.signal.resample_poly``, which is designed with **unity DC gain**:
a constant-amplitude input produces a constant-amplitude output.

**Bandlimited (pulse-shaped) signals**

For a signal whose bandwidth fits within the new Nyquist band
(i.e. ``signal_bandwidth < fs_out / 2``), the polyphase filter is
effectively transparent - it passes all signal energy and the average
sample power is preserved:

    E[|x_out[n]|^2] ≈ E[|x_in[n]|^2]

This holds regardless of the resampling ratio, rolloff factor, or
signal length (verified for RRC-shaped signals with rolloff 0.01-0.99,
sps_in down to 1.5, and block sizes as short as 64 symbols).

**Consequence for the ``"symbol_power"`` convention**

The ``Signal`` class uses the ``"symbol_power"`` normalization
(``E[|x|^2] = 1/sps``) so that symbol energy ``Es = E[|x|^2] * sps = 1``
is independent of the oversampling factor.  Because ``resample_poly``
preserves sample power while ``sps`` changes, the convention is broken
after any rate change: the actual sample power remains ``1/sps_old``
instead of ``1/sps_new``.

``upsample``, ``decimate`` and ``resample`` correct for this by applying a
deterministic amplitude gain of ``sqrt(sps_old / sps_new)`` when
``correct_power=True`` (the default for Signal input).  This correction is **exact** for pulse-shaped
signals because the power-preserving behaviour of ``resample_poly`` is
guaranteed (not statistical).

**Non-bandlimited signals (white noise, arbitrary arrays)**

For a flat-PSD (white-noise) signal, decimation removes the
out-of-band spectral power together with the aliased bandwidth, so
sample power scales as ``up / down``:

    E[|x_out[n]|^2] ≈ (up / down) * E[|x_in[n]|^2]

Upsampling preserves sample power for non-bandlimited signals too
(the anti-imaging filter passes the baseband content unchanged).

If you are passing raw noise or an unfiltered wideband array through
these functions and need to maintain a specific power level, apply
``correct_power=False`` in the ``Signal`` methods and rescale manually,
or use ``commkit.math.normalize`` after the fact.
"""

from fractions import Fraction
from typing import Any

from .backend import ArrayType, dispatch
from .core._signal_adapter import S, adapt_signal, require_integer_sps
from .logger import logger
from .math import normalize as _normalize


def decimate_to_symbol_rate(
    samples: S,
    *,
    sps: int | None = None,
    offset: int = 0,
    normalize: bool | None = None,
) -> S:
    """
    Decimate an oversampled signal to the symbol rate by direct slicing.

    Use this **after** matched filtering: it keeps every ``sps``-th sample
    without further filtering, which is correct because the matched filter
    has already done the optimal noise suppression.

    Parameters
    ----------
    samples : array_like or Signal
        Matched-filtered samples, ``(N,)`` or ``(C, N)``.  A :class:`Signal`
        returns a new :class:`Signal` at the symbol rate.
    sps : int, optional
        Samples per symbol (the decimation factor).  Taken from the Signal;
        required for array input.  A value that disagrees with the Signal
        raises.
    offset : int, default 0
        Sampling phase in samples, ``0 <= offset < sps``: choose the eye
        centre.
    normalize : bool, optional
        Rescale the output to unit average power ``mean(|x|²) = 1``.
        Defaults to ``True`` for Signal input and ``False`` for arrays.

    Returns
    -------
    array_like or Signal
        Symbols at one sample per symbol, ``(..., ceil((N - offset) / sps))``.
    """
    signal_adapter = adapt_signal(samples, function_name="decimate_to_symbol_rate()")
    sps_int = require_integer_sps(
        signal_adapter.resolve_fact("sps", sps), "decimate_to_symbol_rate()"
    )
    _check_offset(offset, sps_int, "decimate_to_symbol_rate()")
    if signal_adapter.signal is None:
        return signal_adapter.wrap_samples(
            _decimate_to_symbol_rate_array(
                signal_adapter.array, sps_int, offset, bool(normalize)
            )
        )
    do_norm = True if normalize is None else normalize
    result = _decimate_to_symbol_rate_array(
        signal_adapter.array, sps_int, offset, do_norm
    )
    return signal_adapter.wrap_samples(
        result, sampling_rate=signal_adapter.signal.symbol_rate
    )


def _check_offset(offset: int, sps: int, function_name: str) -> None:
    if not 0 <= offset < sps:
        raise ValueError(
            f"{function_name}: offset must be in [0, sps) = [0, {sps}), got {offset}."
        )


def _decimate_to_symbol_rate_array(
    samples: ArrayType, sps: int, offset: int, normalize: bool
) -> ArrayType:
    """Array-only direct symbol-rate decimation along the last axis."""
    logger.debug("Downsampling to symbols: sps=%s, offset=%s", sps, offset)
    arr, _, _ = dispatch(samples)
    out = arr[..., offset::sps]
    if normalize:
        out = _normalize(out, mode="average_power", axis=-1)
    return out


def upsample(
    samples: S,
    *,
    factor: int,
    correct_power: bool | None = None,
) -> S:
    """
    Increase the sampling rate by an integer factor (polyphase interpolation).

    Zero insertion followed by an anti-imaging filter
    (``scipy.signal.resample_poly``).

    Parameters
    ----------
    samples : array_like or Signal
        Input samples, ``(N,)`` or ``(C, N)``.  A :class:`Signal` returns a
        new :class:`Signal` with ``sampling_rate`` multiplied by ``factor``.
    factor : int
        Interpolation factor, ``>= 1``.
    correct_power : bool, optional
        Multiply by ``factor**-0.5`` so that ``E[|x|²] = 1/sps`` still holds.
        Defaults to ``True`` for Signal input and ``False`` for arrays.

    Returns
    -------
    array_like or Signal
        Upsampled samples, ``(..., N * factor)``.
    """
    factor = _check_factor(factor, "upsample()")
    signal_adapter = adapt_signal(samples, function_name="upsample()")
    metadata: dict[str, Any] = {}
    if signal_adapter.signal is not None:
        correct_power = True if correct_power is None else correct_power
        metadata["sampling_rate"] = signal_adapter.signal.sampling_rate * factor

    logger.debug("Upsampling by factor %s (polyphase).", factor)
    arr, _, sp = dispatch(signal_adapter.array)
    out = sp.signal.resample_poly(arr, factor, 1, axis=-1)
    if correct_power:
        out = out * (factor**-0.5)
    return signal_adapter.wrap_samples(out, **metadata)


def decimate(
    samples: S,
    *,
    factor: int,
    method: str = "decimate",
    correct_power: bool | None = None,
    zero_phase: bool = True,
    ftype: str = "fir",
) -> S:
    """
    Reduce the sampling rate by an integer factor with anti-aliasing.

    Parameters
    ----------
    samples : array_like or Signal
        Input samples, ``(N,)`` or ``(C, N)``.  A :class:`Signal` returns a
        new :class:`Signal` with ``sampling_rate`` divided by ``factor``.
    factor : int
        Decimation factor, ``>= 1``.
    method : {"decimate", "polyphase"}, default "decimate"
        ``"decimate"`` uses ``scipy.signal.decimate`` (FIR or Chebyshev I
        anti-aliasing filter, see ``ftype``); ``"polyphase"`` uses
        ``resample_poly``.
    correct_power : bool, optional
        Multiply by ``factor**0.5`` so that ``E[|x|²] = 1/sps`` still holds.
        Defaults to ``True`` for Signal input and ``False`` for arrays.
    zero_phase : bool, default True
        ``method="decimate"`` only: filter forward and backward.
    ftype : {"fir", "iir"}, default "fir"
        ``method="decimate"`` only: anti-aliasing filter type.

    Returns
    -------
    array_like or Signal
        Decimated samples, ``(..., ceil(N / factor))``.

    Notes
    -----
    Do not use this for symbol extraction after a matched filter: the extra
    anti-aliasing filter degrades the signal.  Use
    :func:`decimate_to_symbol_rate` instead.
    """
    factor = _check_factor(factor, "decimate()")
    if method not in ("decimate", "polyphase"):
        raise ValueError(
            f"Unknown decimation method: {method!r}. Use 'decimate' or 'polyphase'."
        )
    signal_adapter = adapt_signal(samples, function_name="decimate()")
    metadata: dict[str, Any] = {}
    if signal_adapter.signal is not None:
        correct_power = True if correct_power is None else correct_power
        metadata["sampling_rate"] = signal_adapter.signal.sampling_rate / factor

    logger.debug("Decimating by factor %s (method: %s).", factor, method)
    arr, _, sp = dispatch(signal_adapter.array)
    if method == "decimate":
        out = sp.signal.decimate(
            arr, factor, ftype=ftype, axis=-1, zero_phase=zero_phase
        )
    else:
        out = sp.signal.resample_poly(arr, 1, factor, axis=-1)

    if correct_power:
        out = out * (factor**0.5)
    return signal_adapter.wrap_samples(out, **metadata)


def resample(
    samples: S,
    *,
    up: int | None = None,
    down: int | None = None,
    sps_in: float | None = None,
    sps_out: float | None = None,
    correct_power: bool | None = None,
) -> S:
    """
    Rational resampling by ``up / down`` (polyphase).

    Give either the integer factors ``up`` and ``down``, or the target
    ``sps_out`` (with ``sps_in``, which a Signal supplies).

    Parameters
    ----------
    samples : array_like or Signal
        Input samples, ``(N,)`` or ``(C, N)``.  A :class:`Signal` returns a
        new :class:`Signal` with the new ``sampling_rate``.
    up, down : int, optional
        Integer interpolation and decimation factors.
    sps_in : float, optional
        Input samples per symbol.  Taken from the Signal; required with
        ``sps_out`` for array input.  A value that disagrees with the Signal
        raises.
    sps_out : float, optional
        Target samples per symbol; the ratio ``sps_out / sps_in`` is
        approximated by a fraction.
    correct_power : bool, optional
        Multiply by ``sqrt(down / up)`` so that ``E[|x|²] = 1/sps`` still
        holds.  Defaults to ``True`` for Signal input and ``False`` for arrays.

    Returns
    -------
    array_like or Signal
        Resampled samples, about ``N * up / down`` long.

    Raises
    ------
    ValueError
        If both or neither of ``(up, down)`` and ``sps_out`` are given.
    """
    signal_adapter = adapt_signal(samples, function_name="resample()")
    by_factors = up is not None or down is not None
    if by_factors and (sps_in is not None or sps_out is not None):
        raise ValueError("resample(): give either (up, down) or sps_out, not both.")
    if by_factors:
        if up is None or down is None:
            raise ValueError("resample(): up and down must be given together.")
        up = _check_factor(up, "resample()")
        down = _check_factor(down, "resample()")
    elif sps_out is not None:
        if signal_adapter.signal is None and sps_in is None:
            raise ValueError("resample() requires sps_in with sps_out for array input.")
        sps_in = signal_adapter.resolve_fact("sps", sps_in)
        ratio = Fraction(sps_out / sps_in).limit_denominator()
        up, down = ratio.numerator, ratio.denominator
    else:
        raise ValueError("resample(): give either (up, down) or sps_out.")

    metadata: dict[str, Any] = {}
    sig = signal_adapter.signal
    if sig is not None:
        correct_power = True if correct_power is None else correct_power
        metadata["sampling_rate"] = (
            sig.sampling_rate * up / down
            if sps_out is None
            else sps_out * sig.symbol_rate
        )

    logger.debug("Resampling by rational factor %s/%s (polyphase).", up, down)
    arr, _, sp = dispatch(signal_adapter.array)
    out = sp.signal.resample_poly(arr, up, down, axis=-1)
    if correct_power:
        # sps_after / sps_before = up / down  ->  gain = sqrt(down/up).
        out = out * (down / up) ** 0.5
    return signal_adapter.wrap_samples(out, **metadata)


def _check_factor(factor: int, function_name: str) -> int:
    if isinstance(factor, bool) or int(factor) != factor or factor < 1:
        raise ValueError(
            f"{function_name}: factors must be positive integers, got {factor!r}."
        )
    return int(factor)
