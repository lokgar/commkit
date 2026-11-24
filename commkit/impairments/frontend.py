"""Transceiver front-end impairments: IQ imbalance and the ADC.

:func:`apply_iq_imbalance` mixes I and Q widely linearly, as a mismatched
IQ modulator or receiver; its blind measurement and compensation are in
:mod:`commkit.equalization.iq`.  A time skew between the branches, the
frequency-dependent part of the model, is applied here but lies outside
what the memoryless estimate and corrections describe.
:func:`apply_sampling_clock_offset` resamples a signal as an ADC whose clock
is off its nominal rate, :func:`apply_sampling_jitter` as one whose
sampling instants jitter, and :func:`apply_quantization` rounds it as the
ADC's quantizer.
"""

import logging
import math
from types import ModuleType

import numpy as np

from .. import _fft
from .._array import as_2d, require_finite, require_int, restore_1d
from .._interpolation import interpolate_at, interpolate_piecewise
from .._random import RNG, as_generator, complex_normal, standard_normal
from ..backend import ArrayType, dispatch
from ..core._signal_input import S, SignalInput
from ..logger import logger
from ..math import _derotate, db_to_linear

__all__ = [
    "apply_iq_imbalance",
    "apply_quantization",
    "apply_sampling_clock_offset",
    "apply_sampling_jitter",
]


def apply_iq_imbalance(
    samples: S,
    *,
    amplitude_imbalance_db: float,
    phase_imbalance_deg: float,
    skew: float = 0.0,
    sampling_rate: float | None = None,
) -> S:
    """Apply IQ imbalance to a complex baseband signal.

    The widely linear mixing of I and Q branches with mismatched gain and
    non-orthogonal phase, ``r = I + j·g·e^{jφ}·Q`` for ``s = I + jQ``:

        r[n] = K1 · s[n] + K2 · conj(s[n]),
        K1 = (1 + g e^{jφ}) / 2,   K2 = (1 - g e^{jφ}) / 2,

    with the Q-to-I amplitude ratio ``g = 10^(A/20)`` and the phase error
    ``φ`` in radians.  A time ``skew`` delays the Q branch against I before
    the mixing, ``Q(t) → Q(t - τ)``: unlike the gain and phase errors it is
    frequency dependent, and a tone at ``f`` leaves an image of power
    ``tan²(π f τ)`` relative to the tone.

    Parameters
    ----------
    samples : array_like or Signal
        Complex baseband samples, ``(N,)`` or ``(C, N)``.
    amplitude_imbalance_db : float
        Amplitude imbalance ``A`` between the branches in dB; positive gives
        Q the higher gain, ``0.0`` none.
    phase_imbalance_deg : float
        Phase error ``φ`` of the Q branch in degrees; ``0.0`` for none.
    skew : float, default 0.0
        Delay ``τ`` of the Q branch against I in seconds; negative advances
        Q.  Applied as a linear phase on the spectrum of Q (ideal band-limited
        interpolation), so the record is treated as periodic, as in
        :func:`~commkit.timing.apply_delay`.
    sampling_rate : float, optional
        Sampling rate in Hz.  Taken from the Signal; required for array input
        when ``skew`` is not zero.  A value that disagrees with the Signal
        raises.

    Returns
    -------
    ndarray or Signal
        The imbalanced samples, same shape, dtype and device as the input.

    Raises
    ------
    ValueError
        If the samples are real, or if ``skew`` is not zero and the sampling
        rate is missing or conflicts with the Signal's.

    Notes
    -----
    The blind estimate (:func:`estimate_iq_imbalance`) and the
    orthogonalizations (:class:`Lowdin`, :class:`GramSchmidt`) are memoryless:
    they model and remove the gain and phase errors only.  A skew needs a
    filter per branch, as a widely linear equalizer has.

    Examples
    --------
    >>> r = apply_iq_imbalance(s, amplitude_imbalance_db=1.0, phase_imbalance_deg=3.0)
    >>> r = apply_iq_imbalance(
    ...     s, amplitude_imbalance_db=0.0, phase_imbalance_deg=0.0, skew=2e-12
    ... )
    """
    inp = SignalInput(samples, "apply_iq_imbalance()")
    if skew != 0.0:
        sampling_rate = inp.sampling_rate(sampling_rate)
    elif inp.signal is not None and sampling_rate is not None:
        inp.sampling_rate(sampling_rate)

    logger.info(
        "Applying IQ imbalance (amplitude=%.2f dB, phase=%.2f deg, skew=%.3g s).",
        amplitude_imbalance_db,
        phase_imbalance_deg,
        skew,
    )

    x, xp, _ = dispatch(inp.samples)
    _require_complex(x, xp, "apply_iq_imbalance()")
    if skew != 0.0:
        assert sampling_rate is not None
        x = _delay_quadrature(x, xp, skew * sampling_rate)

    g = db_to_linear(amplitude_imbalance_db, power=False)
    phi = math.radians(phase_imbalance_deg)

    # Mixing coefficients: r = K1*s + K2*conj(s)
    K1 = complex(0.5 * (1.0 + g * math.cos(phi)), 0.5 * g * math.sin(phi))
    K2 = complex(0.5 * (1.0 - g * math.cos(phi)), -0.5 * g * math.sin(phi))

    result = K1 * x + K2 * xp.conj(x)

    if result.dtype != x.dtype:
        result = result.astype(x.dtype)

    return inp.wrap(result)


def _require_complex(x: ArrayType, xp: ModuleType, function_name: str) -> None:
    """Raise unless ``x`` is complex: real samples have no Q branch."""
    if not xp.iscomplexobj(x):
        raise ValueError(
            f"{function_name}: samples must be complex baseband (an I and a Q "
            f"branch), got {x.dtype}."
        )


def _delay_quadrature(x: ArrayType, xp: ModuleType, delay: float) -> ArrayType:
    """``I + j·Q(n - delay)``: the Q branch delayed by ``delay`` samples
    through a linear phase on its real spectrum, in the precision of ``x``
    (the phase in float64)."""
    n = x.shape[-1]
    q = x.imag
    spec = _fft.rfft(q, axis=-1)
    freqs = xp.fft.rfftfreq(n, d=1.0)
    q_delayed = _fft.irfft(_derotate(spec, 2.0 * math.pi * delay * freqs), n=n, axis=-1)
    return (x.real + 1j * q_delayed).astype(x.dtype, copy=False)


def apply_sampling_clock_offset(samples: S, *, offset_ppm: float) -> S:
    """Resample a signal as an ADC whose clock is off its nominal rate.

    The receiver's clock runs at ``fs · (1 + ε)``, ``ε = offset_ppm · 1e-6``,
    while the samples are still labelled with the nominal rate ``fs``: output
    sample ``n`` is the input waveform at ``n / (1 + ε)`` input samples.  A
    positive offset takes more samples per symbol than nominal, so the
    symbols drift later in the record by ``ε`` samples per sample; a
    negative one, earlier.
    :func:`~commkit.timing.correct_sampling_clock_offset` undoes it.

    The waveform between samples is the band-limited interpolant, through a
    48-tap Blackman-Harris windowed sinc, returned in the input's precision:
    its error stays below -119 dB for content up to 0.4 cycles per sample
    (80 % of the Nyquist band, 1.6 samples per symbol for a 0.1-rolloff
    pulse at 2 sps), at any record length and offset, as each instant is
    formed in double precision.  The record is zero-extended, so the first
    and last 24 samples lose that accuracy.  Numba-compiled on the CPU.

    Parameters
    ----------
    samples : array_like or Signal
        Input samples, ``(N,)`` or ``(C, N)``; all channels share the clock.
    offset_ppm : float
        Clock frequency error ``ε`` in parts per million, more than -1e6.

    Returns
    -------
    ndarray or Signal
        ``floor((N - 1) · (1 + ε)) + 1`` samples covering the same waveform,
        same dtype and device (a copy of the input for a zero offset).  A Signal keeps its ``sampling_rate`` (the
        nominal one, which the receiver believes) and its reference: the
        same symbols were sent, and their alignment to the symbol periods
        is what the offset breaks.

    Raises
    ------
    ValueError
        If ``offset_ppm`` is not above -1e6.

    Examples
    --------
    >>> rx = apply_sampling_clock_offset(tx, offset_ppm=20.0)
    """
    offset_ppm = require_finite(
        offset_ppm, "apply_sampling_clock_offset(): offset_ppm", low=-1e6
    )
    inp = SignalInput(samples, "apply_sampling_clock_offset()")
    x, xp, _ = dispatch(inp.samples)
    x, was_1d = as_2d(x, name="samples")
    n_in = x.shape[-1]
    eps = offset_ppm * 1e-6
    if eps == 0.0:
        return inp.wrap(restore_1d(was_1d, x.copy()))
    n_out = math.floor((n_in - 1) * (1.0 + eps)) + 1 if n_in else 0
    logger.info(
        "Applying a sampling-clock offset of %.3f ppm (%d -> %d samples).",
        offset_ppm,
        n_in,
        n_out,
    )
    # output n at input n / (1 + ε), formed in float64 per instant
    y = interpolate_piecewise(x, np.array([[0.0, n_out / (1.0 + eps)]]), n_out)
    return inp.wrap(restore_1d(was_1d, y))


def apply_quantization(
    samples: S,
    *,
    num_bits: int,
    full_scale: float,
    enob: float | None = None,
    rng: RNG = None,
) -> S:
    """Quantize a signal as an ADC: clip, round, and add the converter's noise.

    Each real component (I and Q separately for complex input) is clipped to
    ``±full_scale`` and rounded by a uniform mid-rise quantizer with
    ``2^num_bits`` levels at ``(k + 1/2)·Δ``, ``Δ = 2·full_scale / 2^num_bits``.
    Its error, uniform in ``±Δ/2`` while the input stays in range, has the
    variance ``Δ²/12``; a full-scale sine then has the SQNR
    ``6.02·num_bits + 1.76`` dB.  With ``enob`` the converter's other noise
    (thermal, aperture, nonlinearity, lumped) is added as white Gaussian noise
    of variance ``Δ²/12 · (2^{2(num_bits - enob)} - 1)`` per component, so
    the total is that of an ideal ``enob``-bit quantizer.

    Parameters
    ----------
    samples : array_like or Signal
        Input samples, ``(N,)`` or ``(C, N)``, real or complex.
    num_bits : int
        Resolution, at least 1.
    full_scale : float
        Clipping level of each component, in the units of the samples
        (positive).  A Signal at ``E[|x|²] = 1/sps`` has components of rms
        ``1/sqrt(2·sps)``; the ratio of ``full_scale`` to that rms is the
        loading.
    enob : float, optional
        Effective number of bits, in ``(0, num_bits]``.  ``None`` adds no
        noise beyond the rounding.
    rng : int, numpy.random.Generator or None
        Seed or generator for the ``enob`` noise.

    Returns
    -------
    ndarray or Signal
        The quantized samples, same shape, dtype and device as the input.

    Raises
    ------
    ValueError
        If ``num_bits``, ``full_scale`` or ``enob`` is out of range.

    Examples
    --------
    >>> y = apply_quantization(x, num_bits=8, full_scale=0.8, enob=6.0, rng=1)
    """
    num_bits = require_int(num_bits, "apply_quantization(): num_bits", low=1)
    full_scale = require_finite(full_scale, "apply_quantization(): full_scale", low=0.0)
    if enob is not None and not (
        0 < require_finite(enob, "apply_quantization(): enob") <= num_bits
    ):
        raise ValueError(
            f"apply_quantization(): enob must be in (0, num_bits={num_bits}], got {enob!r}."
        )
    inp = SignalInput(samples, "apply_quantization()")
    x, xp, _ = dispatch(inp.samples)
    step = 2.0 * full_scale / 2 ** int(num_bits)
    top = full_scale - step / 2

    def quantize(v: ArrayType) -> ArrayType:
        return xp.clip((xp.floor(v / step) + 0.5) * step, -top, top)

    if xp.iscomplexobj(x):
        y = (quantize(x.real) + 1j * quantize(x.imag)).astype(x.dtype, copy=False)
    else:
        y = quantize(x).astype(x.dtype, copy=False)
    clipped_fraction = None
    if logger.isEnabledFor(logging.INFO):
        comps = (x.real, x.imag) if xp.iscomplexobj(x) else (x,)
        clipped = sum(float(xp.count_nonzero(xp.abs(c) > full_scale)) for c in comps)
        clipped_fraction = clipped / max(1, x.size * len(comps))
    if enob is not None and enob < num_bits:
        var = step**2 / 12 * (2 ** (2 * (num_bits - enob)) - 1)
        gen = as_generator(rng)
        real = x.real.dtype
        if xp.iscomplexobj(x):
            y = y + complex_normal(gen, x.shape, std=math.sqrt(var), dtype=real, xp=xp)
        else:
            y = y + math.sqrt(var) * standard_normal(gen, x.shape, dtype=real, xp=xp)
    if clipped_fraction is not None:
        logger.info(
            "Quantized to %d bits (full scale %.4g%s): %.3g %% of the components clipped.",
            num_bits,
            full_scale,
            "" if enob is None else f", ENOB {enob:g}",
            100 * clipped_fraction,
        )
    return inp.wrap(y)


def apply_sampling_jitter(
    samples: S,
    *,
    jitter: float,
    sampling_rate: float | None = None,
    rng: RNG = None,
) -> S:
    """Sample a signal at instants that jitter around the clock's grid.

    Sample ``n`` is the band-limited waveform at ``n/fs + δ_n``, with
    independent Gaussian ``δ_n`` of rms ``jitter`` seconds, shared by all
    channels (one clock).  A tone at ``f`` then carries the noise of
    relative power ``(2π f σ)²``: the SNR ``-20 log10(2π f σ)`` of a sampled
    sine.  The waveform between samples comes from the 48-tap windowed-sinc
    interpolator of :func:`apply_sampling_clock_offset` (error below
    -119 dB up to 0.4 cycles per sample, the first and last 24 samples
    excepted).  The ``δ_n`` are drawn on the data's device.

    Parameters
    ----------
    samples : array_like or Signal
        Input samples, ``(N,)`` or ``(C, N)``.
    jitter : float
        RMS sampling-time error ``σ`` in seconds, non-negative.
    sampling_rate : float, optional
        Sampling rate in Hz.  Taken from the Signal; required for array
        input.  A value that disagrees with the Signal raises.
    rng : int, numpy.random.Generator or None
        Seed or generator for the jitter.

    Returns
    -------
    ndarray or Signal
        The jittered samples, same shape, dtype and device (a copy of the
        input for zero jitter).

    Raises
    ------
    ValueError
        If ``jitter`` is negative, or the sampling rate is missing or
        conflicts with the Signal's.

    Examples
    --------
    >>> y = apply_sampling_jitter(x, jitter=100e-15, rng=1)
    """
    jitter = require_finite(
        jitter, "apply_sampling_jitter(): jitter", low=0.0, inclusive=True
    )
    inp = SignalInput(samples, "apply_sampling_jitter()")
    fs = inp.sampling_rate(sampling_rate)
    x, xp, _ = dispatch(inp.samples)
    x, was_1d = as_2d(x, name="samples")
    logger.info(
        "Applying %.3g s rms sampling jitter (%.3g sample periods).",
        jitter,
        jitter * fs,
    )
    if jitter == 0:
        return inp.wrap(restore_1d(was_1d, x.copy()))
    real = x.real.dtype
    n = x.shape[-1]
    d = (jitter * fs) * standard_normal(as_generator(rng), (n,), dtype=real, xp=xp)
    whole = xp.floor(d)
    base = xp.arange(n, dtype=xp.int64) + whole.astype(xp.int64)
    y = interpolate_at(x, base, (d - whole).astype(real, copy=False))
    return inp.wrap(restore_1d(was_1d, y))
