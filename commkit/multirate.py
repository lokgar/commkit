"""Sampling-rate conversion.

:func:`upsample`, :func:`decimate` and :func:`resample` change the
sampling rate through ``scipy.signal.resample_poly`` (polyphase),
``scipy.signal.decimate`` (``decimate(method="decimate")``) or, with
``periodic=True``, Fourier interpolation (``scipy.signal.resample``);
:func:`decimate_to_symbol_rate` slices a matched-filtered signal to one
sample per symbol.

Power scaling.  These filters have unity DC gain: a signal whose band fits
in the output Nyquist band keeps its average sample power,
``E[|x_out|²] ≈ E[|x_in|²]``, whatever the ratio.  A Signal is normalized to
``E[|x|²] = 1/sps``, so a rate change alone would leave the power at
``1/sps_old``; with ``correct_power=True`` (the default for Signal input)
the output is scaled by ``sqrt(sps_old / sps_new)``, which is exact for
band-limited signals.  White noise and other wideband arrays do not follow
this rule (decimation removes the out-of-band power, scaling it by
``up / down``): pass ``correct_power=False`` and rescale them, e.g. with
:func:`commkit.math.normalize`.
"""

import math
from fractions import Fraction
from typing import Any

from .backend import ArrayType, dispatch
from .core._signal_adapter import S, adapt_signal, require_integer_sps
from .logger import logger
from .math import normalize as _normalize

__all__ = ["decimate", "decimate_to_symbol_rate", "resample", "upsample"]


def decimate_to_symbol_rate(
    samples: S,
    *,
    sps: int | None = None,
    offset: int = 0,
    normalize: bool | None = None,
) -> S:
    """Decimate an oversampled signal to the symbol rate by direct slicing.

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
        center.
    normalize : bool, optional
        Rescale the output to unit average power ``mean(|x|²) = 1``.
        Defaults to ``True`` for Signal input and ``False`` for arrays.

    Returns
    -------
    ndarray or Signal
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
    periodic: bool = False,
) -> S:
    """Increase the sampling rate by an integer factor (polyphase interpolation).

    Zero insertion followed by an anti-imaging filter
    (``scipy.signal.resample_poly``); with ``periodic=True``, Fourier
    interpolation (``scipy.signal.resample``).

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
    periodic : bool, default False
        Treat the input as one period of a periodic sequence: the output is
        one period at the new rate, loops seamlessly (e.g. on a streaming
        AWG) and has no filter transients at its ends.

    Returns
    -------
    ndarray or Signal
        Upsampled samples, ``(..., N * factor)``.
    """
    factor = _check_factor(factor, "upsample()")
    signal_adapter = adapt_signal(samples, function_name="upsample()")
    metadata: dict[str, Any] = {}
    if signal_adapter.signal is not None:
        correct_power = True if correct_power is None else correct_power
        metadata["sampling_rate"] = signal_adapter.signal.sampling_rate * factor

    arr, _, sp = dispatch(signal_adapter.array)
    if periodic:
        out = _resample_periodic(arr, factor, 1, sp, "upsample()")
    else:
        logger.debug("Upsampling by factor %s (polyphase).", factor)
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
    """Reduce the sampling rate by an integer factor with anti-aliasing.

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
    ndarray or Signal
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
    periodic: bool = False,
) -> S:
    """Rational resampling by ``up / down`` (polyphase, or Fourier when
    ``periodic``).

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
    periodic : bool, default False
        Treat the input as one period of a periodic sequence and resample it
        by Fourier interpolation (``scipy.signal.resample``): the output is
        one period at the new rate, loops seamlessly (e.g. on a streaming
        AWG) and has no filter transients at its ends.  Exact for a
        band-limited periodic input; downsampling discards everything above
        the new Nyquist frequency.  ``N * up / down`` must be an integer.

    Returns
    -------
    ndarray or Signal
        Resampled samples, about ``N * up / down`` long (exactly, when
        ``periodic``).

    Raises
    ------
    ValueError
        If both or neither of ``(up, down)`` and ``sps_out`` are given, or
        if ``periodic`` and ``N * up / down`` is not an integer.
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

    arr, _, sp = dispatch(signal_adapter.array)
    if periodic:
        out = _resample_periodic(arr, up, down, sp, "resample()")
    else:
        logger.debug("Resampling by rational factor %s/%s (polyphase).", up, down)
        out = sp.signal.resample_poly(arr, up, down, axis=-1)
    if correct_power:
        # sps_after / sps_before = up / down  ->  gain = sqrt(down/up).
        out = out * (down / up) ** 0.5
    return signal_adapter.wrap_samples(out, **metadata)


def _resample_periodic(
    arr: ArrayType, up: int, down: int, sp: Any, function_name: str
) -> ArrayType:
    """One period resampled by ``up / down`` with Fourier interpolation."""
    n = arr.shape[-1]
    step = down // math.gcd(up, down)
    if n % step:
        raise ValueError(
            f"{function_name}: periodic=True needs N * up / down to be an integer "
            f"(N={n}, up={up}, down={down}); use a record length that is a "
            f"multiple of {step}, e.g. {n - n % step or step}."
        )
    logger.debug("Resampling one period by %s/%s (Fourier).", up, down)
    out = sp.signal.resample(arr, n * up // down, axis=-1)
    return out.astype(arr.dtype, copy=False)


def _check_factor(factor: int, function_name: str) -> int:
    if isinstance(factor, bool) or int(factor) != factor or factor < 1:
        raise ValueError(
            f"{function_name}: factors must be positive integers, got {factor!r}."
        )
    return int(factor)
