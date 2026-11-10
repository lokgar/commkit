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

import numpy as np

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

    Use it after matched filtering: it keeps every ``sps``-th sample without
    further filtering, because the matched filter has already removed the
    out-of-band noise.

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
        Sampling phase in samples, ``0 <= offset < sps``, chosen at the eye
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
    """Resample by a rational factor ``up / down``.

    Polyphase filtering (``scipy.signal.resample_poly``), or Fourier
    interpolation with ``periodic=True``.  Give either the integer factors ``up`` and ``down``, or the target
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


# -----------------------------------------------------------------------------
# INTERPOLATION AT ARBITRARY INSTANTS
# -----------------------------------------------------------------------------

# Half-width of the windowed-sinc interpolator (2·K taps) and its
# Blackman-Harris window.  Measured error of a complex tone, float32 arithmetic:
# below -119 dB up to 0.4 cycles per sample (80 % of the Nyquist band),
# -36 dB at 0.45.
_INTERP_HALF_WIDTH = 24
_BLACKMAN_HARRIS = (0.35875, 0.48829, 0.14128, 0.01168)
_INTERP_CHUNK = 1 << 14


def _interpolate(
    x: ArrayType, base: ArrayType, frac: ArrayType, *, margin: int = 0
) -> ArrayType:
    """Band-limited interpolation of ``x`` at the instants ``base + frac``.

    ``x`` is ``(C, N)``, ``base`` the integer sample index left of each
    instant (``(M,)``, int64) and ``frac`` its fraction in ``[0, 1)`` (``(M,)``,
    in the real precision of ``x``).  A Blackman-Harris windowed sinc of
    ``2 · _INTERP_HALF_WIDTH`` taps, computed in the precision of ``x`` and
    chunked over the instants; the record is zero-extended, so instants
    within the half-width of either end lose accuracy.  ``base`` may reach
    ``margin`` samples beyond either end.  Returns ``(C, M)`` in the dtype of
    ``x`` on its device.
    """
    x, xp, _ = dispatch(x)
    K = _INTERP_HALF_WIDTH
    real = xp.float32 if x.dtype in (xp.float32, xp.complex64) else xp.float64
    C, _n = x.shape
    m = base.shape[0]
    pad = K + margin
    padded = xp.pad(x, ((0, 0), (pad, pad)))
    # Row r of windows[c] is padded[c, r : r + 2K], a strided view: gathering
    # whole rows is about twice as fast as indexing every tap.
    windows = [_sliding_window_view(xp, padded[c], 2 * K) for c in range(C)]
    offsets = xp.arange(-K + 1, K + 1)
    k = offsets.astype(real)
    # sin(π(k - μ)) = -(-1)^k sin(πμ): one sine per instant, and the window's
    # cos 2φ, cos 3φ from cos φ (Chebyshev), one cosine per tap.
    neg_alt = -((-1.0) ** offsets).astype(real)
    a0, a1, a2, a3 = _BLACKMAN_HARRIS
    out = xp.empty((C, m), dtype=x.dtype)
    for start in range(0, m, _INTERP_CHUNK):
        stop = min(start + _INTERP_CHUNK, m)
        mu = frac[start:stop, None].astype(real, copy=False)
        u = k[None, :] - mu
        # sin(πμ) = sin(π(1 - μ)), on the nearer side: 1 - μ is exact, while
        # the rounded π times a μ near 1 loses the small result (2 % at
        # μ = 1 - 1e-6 in float32).
        sin_pi_mu = xp.sin(math.pi * xp.minimum(mu, 1.0 - mu))
        c1 = xp.cos((u / K + 1.0) * math.pi)
        window = (a0 - a2) - (a1 - 3.0 * a3) * c1 + c1 * c1 * (2.0 * a2 - 4.0 * a3 * c1)
        # u = 0 only at μ = 0, where the instant is a sample: tap 1 there
        on_sample = u == 0
        sinc = xp.where(
            on_sample,
            1.0,
            neg_alt[None, :] * sin_pi_mu / (math.pi * xp.where(on_sample, 1.0, u)),
        )
        taps = (sinc * window).astype(real, copy=False)
        rows = base[start:stop] + (pad - K + 1)  # window start per instant
        for c in range(C):
            out[c, start:stop] = xp.sum(windows[c][rows] * taps, axis=-1)
    return out


def _stretched_instants(
    num_out: int, rate: float, real: Any, xp: Any
) -> tuple[ArrayType, ArrayType]:
    """The instants ``n · rate``, ``n = 0 .. num_out - 1``, as the integer
    index left of each (int64) and its fraction in ``[0, 1)`` (``real``).

    Written ``n - d`` with ``d = n · (1 - rate)`` in ``real`` precision, so
    no sample-rate float64 is made: the rounding of ``d`` (6e-8 relative in
    float32) is 6e-6 samples at ``d = 100``.
    """
    n = xp.arange(num_out)
    d = n.astype(real) * (1.0 - rate)
    whole = xp.floor(d)
    r = d - whole
    on_sample = r == 0
    base = n - whole.astype(xp.int64) - 1 + on_sample
    frac = xp.where(on_sample, 0.0, 1.0 - r).astype(real, copy=False)
    return base, frac


def _sliding_window_view(xp: Any, a: ArrayType, width: int) -> ArrayType:
    if xp is np:
        return np.lib.stride_tricks.sliding_window_view(a, width)
    from cupy.lib.stride_tricks import sliding_window_view

    return sliding_window_view(a, width)
