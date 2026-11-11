"""Transceiver front-end impairments: IQ imbalance and the ADC.

:func:`apply_iq_imbalance`, its blind measurement
:func:`estimate_iq_imbalance` and its compensation
:func:`correct_iq_imbalance` (:class:`Lowdin` or :class:`GramSchmidt`
orthogonalization, or the inverse of an :class:`IQImbalanceEstimate`) are
kept together because they are one device model, the widely linear mixing
of I and Q.  A time skew between the branches, the frequency-dependent part
of the model, is applied by :func:`apply_iq_imbalance` but lies outside what
the memoryless estimate and corrections describe.
:func:`apply_sampling_clock_offset` resamples a signal as an ADC whose clock
is off its nominal rate, :func:`apply_sampling_jitter` as one whose
sampling instants jitter, and :func:`apply_quantization` rounds it as the
ADC's quantizer.
"""

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType

from .. import _fft
from .._array import as_2d, restore_1d
from .._random import RNG, as_generator, complex_normal, standard_normal
from ..backend import ArrayType, dispatch, to_device
from ..core._signal_adapter import S, adapt_signal
from ..core.signal import Signal
from ..logger import logger
from ..math import _derotate, db_to_linear
from ..multirate import _interpolate, _stretched_instants

__all__ = [
    "GramSchmidt",
    "IQImbalanceEstimate",
    "Lowdin",
    "apply_iq_imbalance",
    "apply_quantization",
    "apply_sampling_clock_offset",
    "apply_sampling_jitter",
    "correct_iq_imbalance",
    "estimate_iq_imbalance",
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
        :func:`~commkit.timing.fft_fractional_delay`.
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
    signal_adapter = adapt_signal(samples, function_name="apply_iq_imbalance()")
    if skew != 0.0:
        sampling_rate = signal_adapter.resolve_fact("sampling_rate", sampling_rate)
    elif signal_adapter.signal is not None and sampling_rate is not None:
        signal_adapter.resolve_fact("sampling_rate", sampling_rate)

    logger.info(
        "Applying IQ imbalance (amplitude=%.2f dB, phase=%.2f deg, skew=%.3g s).",
        amplitude_imbalance_db,
        phase_imbalance_deg,
        skew,
    )

    x, xp, _ = dispatch(signal_adapter.array)
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

    return signal_adapter.wrap_samples(result)


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
    48-tap Blackman-Harris windowed sinc in the input's precision: its error
    stays below -119 dB for content up to 0.4 cycles per sample (80 % of the
    Nyquist band, 1.6 samples per symbol for a 0.1-rolloff pulse at 2 sps).
    The record is zero-extended, so the first and last 24 samples lose that
    accuracy.

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
    if not offset_ppm > -1e6:
        raise ValueError(
            f"apply_sampling_clock_offset(): offset_ppm must be above -1e6, got {offset_ppm!r}."
        )
    signal_adapter = adapt_signal(
        samples, function_name="apply_sampling_clock_offset()"
    )
    x, xp, _ = dispatch(signal_adapter.array)
    x, was_1d = as_2d(x, name="samples")
    n_in = x.shape[-1]
    eps = offset_ppm * 1e-6
    if eps == 0.0:
        return signal_adapter.wrap_samples(restore_1d(was_1d, x.copy()))
    n_out = math.floor((n_in - 1) * (1.0 + eps)) + 1 if n_in else 0
    logger.info(
        "Applying a sampling-clock offset of %.3f ppm (%d -> %d samples).",
        offset_ppm,
        n_in,
        n_out,
    )
    real = x.real.dtype
    base, frac = _stretched_instants(n_out, 1.0 / (1.0 + eps), real, xp)
    y = _interpolate(x, base, frac)
    return signal_adapter.wrap_samples(restore_1d(was_1d, y))


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
    if isinstance(num_bits, bool) or int(num_bits) != num_bits or num_bits < 1:
        raise ValueError(
            f"apply_quantization(): num_bits must be a positive integer, got {num_bits!r}."
        )
    if not full_scale > 0:
        raise ValueError(
            f"apply_quantization(): full_scale must be positive, got {full_scale!r}."
        )
    if enob is not None and not (0 < enob <= num_bits):
        raise ValueError(
            f"apply_quantization(): enob must be in (0, num_bits={num_bits}], got {enob!r}."
        )
    signal_adapter = adapt_signal(samples, function_name="apply_quantization()")
    x, xp, _ = dispatch(signal_adapter.array)
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
    return signal_adapter.wrap_samples(y)


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
    if not jitter >= 0:
        raise ValueError(
            f"apply_sampling_jitter(): jitter must be non-negative, got {jitter!r}."
        )
    signal_adapter = adapt_signal(samples, function_name="apply_sampling_jitter()")
    fs = signal_adapter.resolve_fact("sampling_rate", sampling_rate)
    x, xp, _ = dispatch(signal_adapter.array)
    x, was_1d = as_2d(x, name="samples")
    logger.info(
        "Applying %.3g s rms sampling jitter (%.3g sample periods).",
        jitter,
        jitter * fs,
    )
    if jitter == 0:
        return signal_adapter.wrap_samples(restore_1d(was_1d, x.copy()))
    real = x.real.dtype
    n = x.shape[-1]
    d = (jitter * fs) * standard_normal(as_generator(rng), (n,), dtype=real, xp=xp)
    whole = xp.floor(d)
    base = xp.arange(n) + whole.astype(xp.int64)
    margin = int(math.ceil(10 * jitter * fs)) + 1  # 10 σ
    base = xp.clip(base, -margin, n - 1 + margin)
    y = _interpolate(x, base, (d - whole).astype(real, copy=False), margin=margin)
    return signal_adapter.wrap_samples(restore_1d(was_1d, y))


# -----------------------------------------------------------------------------
# IQ-IMBALANCE ESTIMATION
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class IQImbalanceEstimate:
    """Result of :func:`estimate_iq_imbalance`, in the units of
    :func:`apply_iq_imbalance`.

    Per-channel fields follow the rank rule (0-d for ``(N,)`` input, ``(C,)``
    for ``(C, N)``) and stay on the input's device.

    Attributes
    ----------
    amplitude_imbalance_db : ndarray
        Q-to-I amplitude ratio ``A`` in dB; positive when Q has the higher
        gain.
    phase_imbalance_deg : ndarray
        Phase error ``φ`` of the Q branch in degrees.
    """

    amplitude_imbalance_db: ArrayType
    phase_imbalance_deg: ArrayType


def estimate_iq_imbalance(samples: ArrayType | Signal) -> IQImbalanceEstimate:
    """Measure IQ imbalance blindly, as the parameters of
    :func:`apply_iq_imbalance`.

    Assumes the undistorted signal is circular, its I and Q of equal power
    and uncorrelated, as square QAM, PSK and Gaussian noise are.  The
    imbalanced branches ``r_I = I - g sin φ · Q`` and ``r_Q = g cos φ · Q``
    then have the second moments

        E[r_Q²] = σ² g² cos² φ,   E[r_I r_Q] = -σ² g² sin φ cos φ,
        E[r_I²] - E[r_I r_Q]² / E[r_Q²] = σ²,

    which give ``tan φ = -E[r_I r_Q] / E[r_Q²]`` and ``g`` in closed form.

    Parameters
    ----------
    samples : array_like or Signal
        Complex baseband samples, ``(N,)`` or ``(C, N)``.  A carrier
        frequency offset does no harm (it rotates the signal, which stays
        circular), but the imbalance must sit after it, at the receiver.

    Returns
    -------
    IQImbalanceEstimate
        Per channel, on the input's device.

    Raises
    ------
    ValueError
        If the samples are real.

    Notes
    -----
    The moments are estimated from ``N`` samples, so the spread of the
    estimate falls as ``1/sqrt(N)``: about ``57°/sqrt(N)`` in phase for
    QPSK, 0.06° at ``N = 10^6``.

    Examples
    --------
    >>> r = apply_iq_imbalance(s, amplitude_imbalance_db=0.5, phase_imbalance_deg=3.0)
    >>> est = estimate_iq_imbalance(r)  # ≈ 0.5 dB, 3.0°
    >>> s_hat = correct_iq_imbalance(r, est)
    """
    signal_adapter = adapt_signal(samples, function_name="estimate_iq_imbalance()")
    x, xp, _ = dispatch(signal_adapter.array)
    _require_complex(x, xp, "estimate_iq_imbalance()")
    x2, was_1d = as_2d(x, name="samples")
    sii, sqq, siq = _branch_moments(x2, xp)
    g2 = (sqq * sqq + siq * siq) / (sii * sqq - siq * siq)
    amplitude_db = 10.0 * xp.log10(g2)
    phase_deg = xp.degrees(xp.arctan2(-siq, sqq))
    if was_1d:
        amplitude_db, phase_deg = amplitude_db[0], phase_deg[0]
    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "Estimated IQ imbalance: amplitude=%s dB, phase=%s deg.",
            to_device(amplitude_db, "cpu").round(2),
            to_device(phase_deg, "cpu").round(2),
        )
    return IQImbalanceEstimate(amplitude_db, phase_deg)


def _branch_moments(
    samples: ArrayType, xp: ModuleType
) -> tuple[ArrayType, ArrayType, ArrayType]:
    """``E[I²]``, ``E[Q²]``, ``E[IQ]`` per channel of ``(C, N)`` samples,
    float64 ``(C,)``.

    Means in the samples' precision (pairwise on NumPy, a tree on CuPy,
    whose float64-accumulating reduction over few long rows is 10x slower).
    """
    i_branch, q_branch = samples.real, samples.imag
    sii = xp.mean(i_branch * i_branch, axis=-1).astype(xp.float64)
    sqq = xp.mean(q_branch * q_branch, axis=-1).astype(xp.float64)
    siq = xp.mean(i_branch * q_branch, axis=-1).astype(xp.float64)
    return sii, sqq, siq


# -----------------------------------------------------------------------------
# BLIND IQ-IMBALANCE CORRECTION
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Lowdin:
    """Löwdin symmetric orthogonalization of the I and Q branches.

    Whitens ``X = [I; Q]`` with ``W = M^(-1/2)``, ``M = X Xᵀ / N`` (symmetric
    eigendecomposition), so the corrected branches have equal power and zero
    correlation.  The transform is symmetric: both branches are adjusted
    equally, which minimizes the total distortion.
    """


@dataclass(frozen=True)
class GramSchmidt:
    """Gram-Schmidt orthogonalization with I as the reference branch.

    Normalizes I to unit RMS, removes its projection from Q, then normalizes
    Q (the classical GSOP front-end calibration).
    """


# The correctors map the per-channel branch statistics ``E[I²]``, ``E[Q²]``
# and ``E[IQ]`` (float64, ``(C,)``) to the 2x2 transform ``[I'; Q'] = A [I; Q]``
# of each channel, as the four entries ``(a_ii, a_iq, a_qi, a_qq)``.
_Coefficients = tuple[ArrayType, ArrayType, ArrayType, ArrayType]


def _lowdin(
    sii: ArrayType, sqq: ArrayType, siq: ArrayType, xp: ModuleType
) -> _Coefficients:
    # M^(-1/2) of M = [[sii, siq], [siq, sqq]] in closed form: the square root
    # of a 2x2 SPD matrix is (M + s·1) / t, with s = sqrt(det M) and
    # t = sqrt(tr M + 2s), so M^(-1/2) = t · (M + s·1)^(-1).
    det_s = xp.sqrt(sii * sqq - siq * siq)
    t = xp.sqrt(sii + sqq + 2.0 * det_s)
    a, d = sii + det_s, sqq + det_s
    scale = t / (a * d - siq * siq)
    return d * scale, -siq * scale, -siq * scale, a * scale


def _gram_schmidt(
    sii: ArrayType, sqq: ArrayType, siq: ArrayType, xp: ModuleType
) -> _Coefficients:
    # I / rms(I); Q minus its projection on I, siq/sii · I, over its rms,
    # sqrt(sqq - siq²/sii).
    inv_i = 1.0 / xp.sqrt(sii)
    inv_q = 1.0 / xp.sqrt(sqq - siq * siq / sii)
    return inv_i, xp.zeros_like(sii), -(siq / sii) * inv_q, inv_q


_IQ_CORRECTORS: dict[
    type, Callable[[ArrayType, ArrayType, ArrayType, ModuleType], _Coefficients]
] = {Lowdin: _lowdin, GramSchmidt: _gram_schmidt}


def correct_iq_imbalance(
    samples: S, how: Lowdin | GramSchmidt | IQImbalanceEstimate
) -> S:
    """Correct IQ imbalance, blindly or from an estimate.

    With :class:`Lowdin` or :class:`GramSchmidt`, orthogonalizes the I and Q
    branches of each channel and restores the channel's input power; this
    undoes :func:`apply_iq_imbalance` up to a common gain and rotation.  With
    an :class:`IQImbalanceEstimate` (from :func:`estimate_iq_imbalance`,
    possibly of an earlier capture), inverts the widely linear model exactly,
    ``s = (conj(K1) · r - K2 · conj(r)) / (|K1|² - |K2|²)``.

    Parameters
    ----------
    samples : array_like or Signal
        Complex baseband samples, ``(N,)`` or ``(C, N)``.
    how : Lowdin, GramSchmidt or IQImbalanceEstimate
        Orthogonalization algorithm, or the imbalance to invert (0-d, or
        ``(C,)`` per channel).

    Returns
    -------
    ndarray or Signal
        Corrected samples, same shape, dtype and device as the input.

    Raises
    ------
    TypeError
        If ``how`` is not one of the accepted types.
    ValueError
        If the samples are real.

    Examples
    --------
    >>> r = apply_iq_imbalance(s, amplitude_imbalance_db=1.5, phase_imbalance_deg=4.0)
    >>> s_hat = correct_iq_imbalance(r, Lowdin())
    >>> s_hat = correct_iq_imbalance(r, estimate_iq_imbalance(r))
    """
    correct_fn = _IQ_CORRECTORS.get(type(how))
    if correct_fn is None and not isinstance(how, IQImbalanceEstimate):
        raise TypeError(
            "correct_iq_imbalance(): how must be Lowdin(), GramSchmidt() or an "
            f"IQImbalanceEstimate, got {type(how).__name__}."
        )
    signal_adapter = adapt_signal(samples, function_name="correct_iq_imbalance()")
    x, xp, _ = dispatch(signal_adapter.array)
    _require_complex(x, xp, "correct_iq_imbalance()")
    logger.info("Correcting IQ imbalance (%s).", type(how).__name__)
    if correct_fn is None:
        assert isinstance(how, IQImbalanceEstimate)
        return signal_adapter.wrap_samples(_invert_iq_imbalance(x, xp, how))
    return signal_adapter.wrap_samples(_apply_iq_correction(x, xp, correct_fn))


def _invert_iq_imbalance(
    samples: ArrayType, xp: ModuleType, how: IQImbalanceEstimate
) -> ArrayType:
    """The exact inverse of :func:`apply_iq_imbalance`, with common or
    per-channel parameters."""
    samples, was_1d = as_2d(samples, name="samples")
    num_channels = samples.shape[0]

    def column(value: ArrayType, name: str) -> ArrayType:
        arr = xp.asarray(value, dtype=xp.float64).reshape(-1)
        if arr.size not in (1, num_channels):
            raise ValueError(
                f"correct_iq_imbalance(): how.{name} has {arr.size} channels "
                f"for {num_channels}."
            )
        return arr[:, None]

    g = 10.0 ** (column(how.amplitude_imbalance_db, "amplitude_imbalance_db") / 20.0)
    phi = xp.radians(column(how.phase_imbalance_deg, "phase_imbalance_deg"))
    ge = g * xp.exp(1j * phi)
    k1, k2 = (1.0 + ge) / 2.0, (1.0 - ge) / 2.0
    det = g * xp.cos(phi)  # |K1|² - |K2|²
    a = (xp.conj(k1) / det).astype(samples.dtype)
    b = (-k2 / det).astype(samples.dtype)
    return restore_1d(was_1d, a * samples + b * xp.conj(samples))


def _apply_iq_correction(
    samples: ArrayType,
    xp: ModuleType,
    correct_fn: Callable[[ArrayType, ArrayType, ArrayType, ModuleType], _Coefficients],
) -> ArrayType:
    """Shared scaffold for the blind IQ-imbalance compensators, all channels
    at once.

    Measures the branch statistics, takes the 2x2 transform of
    ``correct_fn`` (Löwdin symmetric whitening or Gram-Schmidt
    orthogonalization; both give unit-power, uncorrelated branches), folds in
    the gain that restores each channel's input power, and writes the
    corrected real and imaginary parts in the samples' precision.
    """
    samples, was_1d = as_2d(samples, name="samples")
    i_branch, q_branch = samples.real, samples.imag
    # The 2x2 algebra in float64.
    sii, sqq, siq = _branch_moments(samples, xp)

    a_ii, a_iq, a_qi, a_qq = correct_fn(sii, sqq, siq, xp)
    # Restore input power: E[|s_hat|^2] = E[|r|^2], each branch at half.
    gain = xp.sqrt((sii + sqq) / 2.0)
    real = i_branch.dtype

    def coef(a: ArrayType) -> ArrayType:
        """A per-channel coefficient, scaled by ``gain``, as a ``(C, 1)`` column."""
        return (a * gain).astype(real)[:, None]

    result = xp.empty_like(samples)
    result.real[...] = coef(a_ii) * i_branch + coef(a_iq) * q_branch
    result.imag[...] = coef(a_qi) * i_branch + coef(a_qq) * q_branch
    return restore_1d(was_1d, result)
