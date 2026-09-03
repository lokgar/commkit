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

``Signal.upsample``, ``Signal.decimate``, and ``Signal.resample`` correct
for this by applying a deterministic amplitude gain of
``sqrt(sps_old / sps_new)`` when their ``correct_power=True`` parameter
is set (the default).  This correction is **exact** for pulse-shaped
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
or use ``helpers.normalize`` after the fact.
"""

from fractions import Fraction
from typing import Any

from . import helpers
from .backend import ArrayType, dispatch
from .core.signal import Signal
from .logger import logger


def decimate_to_symbol_rate(
    samples: ArrayType | Signal,
    sps: int | None = None,
    offset: int = 0,
    normalize: bool | None = None,
    axis: int = -1,
) -> ArrayType | Signal:
    """
    Decimates an oversampled signal to symbol-rate by direct slicing.

    This function should be used **after** matched filtering to extract
    pulse-shaped symbols at 1 sps. It does not apply additional
    filtering, which is correct since the matched filter has already
    performed optimal noise suppression.

    Parameters
    ----------
    samples : array_like or Signal
        Input matched-filtered signal. Shape: (..., N_samples).  When a
        :class:`Signal` is passed, ``sps`` defaults to the signal's integer
        ``sps`` and a new :class:`Signal` at the symbol rate is returned.
    sps : int, optional
        Input samples per symbol (decimation factor).  Required for array
        input; derived from the Signal otherwise.
    offset : int, default 0
        Sampling phase offset in samples [0, sps-1]. Adjust this to
        sample at the peak of the impulse response (center of the eye).
    normalize : bool, optional
        Normalize the output to unit average power (``mean(|x|²)=1``) after
        slicing.  Absorbs channel/equalizer/filter gain uncertainty.  Defaults
        to ``True`` for :class:`Signal` input (the canonical receive path) and
        ``False`` for raw arrays (the unmodified slicing primitive).
    axis : int, default -1
        The axis along which to downsample.

    Returns
    -------
    array_like or Signal
        Symbols at 1 sps. Shape: (..., N_samples / sps).
    """
    x, sig = helpers.unwrap_signal(samples)
    if sig is not None:
        do_norm = True if normalize is None else normalize
        sps_int = int(sig.sps)
        meta: dict[str, Any] = {}
        if sps_int <= 1:
            logger.info("Signal already at 1 sps, no downsampling needed.")
            result = x
        else:
            result = decimate_to_symbol_rate(
                x, sps=sps_int, offset=offset, normalize=False, axis=-1
            )
            meta["sampling_rate"] = sig.symbol_rate
        if do_norm:
            result = helpers.normalize(result, "average_power", axis=-1)
        return helpers.rewrap_signal(sig, result, **meta)

    if sps is None:
        raise ValueError("decimate_to_symbol_rate() requires sps for array input.")
    logger.debug("Downsampling to symbols: sps=%s, offset=%s", sps, offset)
    arr, xp, _ = dispatch(samples)

    # Build slicing for arbitrary axis
    slices = [slice(None)] * arr.ndim
    slices[axis] = slice(offset, None, sps)
    out = arr[tuple(slices)]
    if normalize:
        out = helpers.normalize(out, "average_power", axis=axis)
    return out


def upsample(
    samples: ArrayType | Signal,
    factor: int,
    correct_power: bool | None = None,
    axis: int = -1,
) -> ArrayType | Signal:
    """
    Increases the sampling rate by an integer factor with filtering.

    This is a convenience wrapper around `resample_poly` that performs
    both zero-insertion (expansion) and anti-imaging filtering to suppress
    spectral replicas.

    Parameters
    ----------
    samples : array_like or Signal
        Input signal samples. Shape: (..., N_samples).  A :class:`Signal`
        returns a new :class:`Signal` with ``sampling_rate`` scaled by *factor*.
    factor : int
        The interpolation factor.
    correct_power : bool, optional
        Apply the deterministic amplitude gain ``factor**-0.5`` to preserve the
        ``"symbol_power"`` invariant (``E[|x|²]=1/sps``).  Defaults to ``True``
        for :class:`Signal` input, ``False`` for raw arrays.
    axis : int, default -1
        The axis along which to perform upsampling.

    Returns
    -------
    array_like or Signal
        The upsampled signal. Shape: (..., N_samples * factor).
    """
    x, sig = helpers.unwrap_signal(samples)
    if sig is not None:
        result = upsample(
            x,
            factor,
            correct_power=True if correct_power is None else correct_power,
            axis=-1,
        )
        return helpers.rewrap_signal(
            sig, result, sampling_rate=sig.sampling_rate * factor
        )

    logger.debug("Upsampling by factor %s (polyphase, axis=%s).", factor, axis)
    arr, xp, sp = dispatch(samples)
    out = sp.signal.resample_poly(arr, factor, 1, axis=axis)
    if correct_power:
        out = out * (factor**-0.5)
    return out


def decimate(
    samples: ArrayType | Signal,
    factor: int,
    method: str = "decimate",
    correct_power: bool | None = None,
    axis: int = -1,
    **kwargs: Any,
) -> ArrayType | Signal:
    """
    Reduces the sampling rate with anti-aliasing filtering.

    Decimation combines lowpass filtering (to prevent aliasing) with
    downsampling (keeping every Nth sample).

    Parameters
    ----------
    samples : array_like or Signal
        Input signal samples. Shape: (..., N_samples).  A :class:`Signal`
        returns a new :class:`Signal` with ``sampling_rate`` divided by *factor*.
    factor : int
        The decimation factor.
    method : {"decimate", "polyphase"}, default "decimate"
        The implementation strategy:
        - "decimate": Uses `scipy.signal.decimate` (Chebyshev I or FIR).
        - "polyphase": Uses `resample_poly` for filter-and-sample.
    correct_power : bool, optional
        Apply the deterministic amplitude gain ``factor**0.5`` to preserve the
        ``"symbol_power"`` invariant.  Defaults to ``True`` for :class:`Signal`
        input, ``False`` for raw arrays.
    axis : int, default -1
        The axis along which to perform decimation.
    **kwargs : Any
        Additional parameters passed to the underlying filter design, such
        as `zero_phase` or `ftype`.

    Returns
    -------
    array_like or Signal
        The decimated signal. Shape: (..., N_samples / factor).

    Notes
    -----
    Do NOT use this function for symbol extraction after a matched filter.
    Matched filters already perform optimal noise suppression and
    anti-aliasing; adding an extra decimation filter will degrade the
    signal. Use `decimate_to_symbol_rate` instead.
    """
    x, sig = helpers.unwrap_signal(samples)
    if sig is not None:
        result = decimate(
            x,
            factor,
            method=method,
            correct_power=True if correct_power is None else correct_power,
            axis=-1,
            **kwargs,
        )
        return helpers.rewrap_signal(
            sig, result, sampling_rate=sig.sampling_rate / factor
        )

    logger.debug("Decimating by factor %s (method: %s).", factor, method)
    arr, _, sp = dispatch(samples)

    if method == "decimate":
        # scipy.signal.decimate (includes antialiasing)
        zero_phase = kwargs.get("zero_phase", True)
        ftype = kwargs.get("ftype", "fir")
        out = sp.signal.decimate(
            arr, int(factor), ftype=ftype, axis=axis, zero_phase=zero_phase
        )
    elif method == "polyphase":
        # resample_poly with up=1
        out = sp.signal.resample_poly(arr, 1, int(factor), axis=axis)
    else:
        raise ValueError(f"Unknown decimation method: {method}")

    if correct_power:
        out = out * (factor**0.5)
    return out


def resample(
    samples: ArrayType | Signal,
    up: int | None = None,
    down: int | None = None,
    sps_in: float | None = None,
    sps_out: float | None = None,
    correct_power: bool | None = None,
    axis: int = -1,
) -> ArrayType | Signal:
    """
    Performs rational resampling of a signal.

    Changes the sampling rate of the input by a rational factor. The rate
    can be specified either as direct integer factors (`up`, `down`) or
    relative to symbols (`sps_in`, `sps_out`).

    Parameters
    ----------
    samples : array_like or Signal
        Input signal samples. Shape: (..., N_samples).  A :class:`Signal`
        returns a new :class:`Signal` with updated ``sampling_rate``; ``sps_in``
        is taken from the signal when ``sps_out`` is given.
    up : int, optional
        Integer upsampling factor.
    down : int, optional
        Integer downsampling factor.
    sps_in : float, optional
        Input samples per symbol.
    sps_out : float, optional
        Target samples per symbol.
    correct_power : bool, optional
        Apply the deterministic amplitude gain ``sqrt(down/up)`` (=
        ``sqrt(sps_before/sps_after)``) to preserve the ``"symbol_power"``
        invariant.  Defaults to ``True`` for :class:`Signal` input, ``False``
        for raw arrays.
    axis : int, default -1
        The axis along which to perform resampling.

    Returns
    -------
    array_like or Signal
        The resampled signal. Shape: (..., N_samples * Ratio).

    Raises
    ------
    ValueError
        If parameters are insufficient or contradictory.
    """
    x, sig = helpers.unwrap_signal(samples)
    if sig is not None:
        # When sps_out is given, the input sps comes from the signal itself.
        sig_sps_in = sig.sps if sps_out is not None else None
        result = resample(
            x,
            up=up,
            down=down,
            sps_in=sig_sps_in,
            sps_out=sps_out,
            correct_power=True if correct_power is None else correct_power,
            axis=-1,
        )
        meta: dict[str, Any] = {}
        if sps_out is not None:
            meta["sampling_rate"] = sps_out * sig.symbol_rate
        elif up is not None and down is not None:
            meta["sampling_rate"] = sig.sampling_rate * up / down
        return helpers.rewrap_signal(sig, result, **meta)

    if (up is not None or down is not None) and (
        sps_in is not None or sps_out is not None
    ):
        raise ValueError("Cannot specify both (up, down) and (sps_in, sps_out).")

    if sps_in is not None and sps_out is not None:
        ratio = Fraction(sps_out / sps_in).limit_denominator()
        up = ratio.numerator
        down = ratio.denominator
    elif up is None or down is None:
        raise ValueError("Must specify either (up, down) or (sps_in, sps_out).")

    logger.debug(
        "Resampling by rational factor %s/%s (polyphase, axis=%s).", up, down, axis
    )
    arr, xp, sp = dispatch(samples)
    out = sp.signal.resample_poly(arr, int(up), int(down), axis=axis)
    if correct_power:
        # sps_after / sps_before = up / down  ->  gain = sqrt(down/up).
        out = out * (down / up) ** 0.5
    return out


def resolve_symbols(
    samples: ArrayType | Signal,
    sps: int | None = None,
    offset: int = 0,
) -> ArrayType | Signal:
    """
    Decimate to symbol rate (1 sps) and normalize to unit average power.

    This is the canonical receive-side symbol-extraction step: it slices the
    matched-filtered signal to one sample per symbol and renormalizes to
    ``E[|x|²]=1`` (absorbing channel/equalizer/filter gain).

    Parameters
    ----------
    samples : array_like or Signal
        Oversampled (matched-filtered) signal, or a :class:`Signal`.  For a
        :class:`Signal`, a new :class:`Signal` is returned with
        ``resolved_symbols`` populated (the samples themselves are unchanged);
        ``sps`` is taken from the signal and frame-generated signals are
        skipped with a warning.
    sps : int, optional
        Input samples per symbol.  Required for array input; derived from the
        signal otherwise.
    offset : int, default 0
        Integer sampling-phase offset applied before decimation.

    Returns
    -------
    array_like or Signal
        Unit-average-power symbols at 1 sps (array), or a new :class:`Signal`
        with ``resolved_symbols`` set.

    Raises
    ------
    ValueError
        If ``sps`` is missing/invalid (array), or the signal's ``sps`` is not a
        positive integer.
    """
    x, sig = helpers.unwrap_signal(samples)
    if sig is not None:
        if sig.signal_type is not None:
            logger.warning(
                "resolve_symbols() called on a frame-generated signal - skipping. "
                "Frame signals mix preamble, pilots, and payload segments that may "
                "have different modulations or gains. Extract the desired segment "
                "via frame.get_structure_map(), build a plain Signal, then call "
                "resolve_symbols() on that."
            )
            return sig.copy()
        s = sig.sps
        if s is None:
            raise ValueError("Symbol rate or sampling rate missing.")
        if s < 1:
            raise ValueError("Symbol rate must be >= 1.")
        if s % 1 != 0:
            raise ValueError("Symbol rate must be an integer.")
        # Output field (resolved_symbols) differs from the input field
        # (samples), so this can't go through rewrap_signal (which always
        # sets .samples) - copy and assign the target field by hand.
        new = sig.copy()
        new.resolved_symbols = resolve_symbols(x, sps=int(s), offset=offset)
        return new

    if sps is None:
        raise ValueError("resolve_symbols() requires sps for array input.")
    # decimate_to_symbol_rate slices [offset::sps] (identity when sps==1) then
    # normalizes to unit average power.
    return decimate_to_symbol_rate(
        samples, sps=int(sps), offset=int(offset), normalize=True
    )
