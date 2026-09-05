"""
Digital filtering and pulse shaping.

This module provides routines for design and application of digital filters
commonly used in communication systems. It supports both standard FIR filters
and specialized pulse-shaping filters, with high-performance execution on
both CPU and GPU backends.
"""

import numpy as np
import scipy

from .backend import ArrayType, dispatch, to_device
from .core._signal_adapter import prepare_signal_input, require_integer_sps
from .core.signal import Signal
from .helpers import (
    _cd_beta2_length,
    as_2d,
    normalize,
    restore_1d,
)
from .logger import logger

# -----------------------------------------------------------------------------
# FILTER DESIGN - TAP GENERATORS (array-only)
# -----------------------------------------------------------------------------
# rect_taps:     Hard or trapezoidal rectangular pulse taps
# gaussian_taps: Gaussian filter taps
# smoothrect_taps: Gaussian-smoothed rectangular pulse taps
# rrc_taps:      Root Raised Cosine filter taps
# rc_taps:       Raised Cosine filter taps
# fir_taps:      Lowpass/highpass/bandpass/bandstop FIR filters (btype=)
#
# All build taps from parameters alone - no signal input, so none are
# Signal-aware (same category as gray_code/barker_sequence).


def rect_taps(sps: int, duty_cycle: float = 1.0, rise_time: float = 0.0) -> np.ndarray:
    """
    Generates rectangular or trapezoidal pulse-shaping filter taps.

    With ``rise_time=0`` (default) the output is a hard rectangular pulse of
    width ``duty_cycle`` symbol periods.  With ``rise_time > 0`` the leading
    and trailing edges are replaced by linear ramps, producing an isosceles
    trapezoidal pulse that models a slew-rate-limited driver or modulator.

    Parameters
    ----------
    sps : int
        Samples per symbol.
    duty_cycle : float, default 1.0
        Total pulse width in symbol periods, including both ramps.
        Must be in the range ``(0, 1]``.
    rise_time : float, default 0.0
        Duration of each linear ramp (10%->90% of amplitude is the full ramp
        here; the ramp spans the full ``rise_time``) in symbol periods.
        Must satisfy ``rise_time <= duty_cycle / 2``; otherwise the ramps
        overlap and no flat top exists.

    Returns
    -------
    ndarray
        Pulse taps (unnormalized). Shape: ``(N_taps,)``.

    Raises
    ------
    ValueError
        If ``rise_time > duty_cycle / 2``.
    """
    if rise_time > duty_cycle / 2:
        raise ValueError(
            f"rise_time ({rise_time}) must be <= duty_cycle / 2 ({duty_cycle / 2:.3f}); "
            "ramps would overlap with no flat top."
        )

    sps = require_integer_sps(sps, "rect_taps()")
    n_total = int(round(sps * duty_cycle))
    if n_total < 1:
        n_total = 1

    if rise_time == 0.0:
        h = np.ones(n_total)
    else:
        n_ramp = int(round(sps * rise_time))
        n_flat = n_total - 2 * n_ramp
        if n_flat < 0:
            n_flat = 0
        ramp_up = np.linspace(0.0, 1.0, n_ramp, endpoint=False)
        ramp_dn = np.linspace(1.0, 0.0, n_ramp, endpoint=False)
        flat = np.ones(n_flat)
        h = np.concatenate([ramp_up, flat, ramp_dn])

    logger.debug(
        "Generating Rect taps: sps=%s, duty_cycle=%s, rise_time=%s, n_taps=%s",
        sps,
        duty_cycle,
        rise_time,
        len(h),
    )
    return h


def gaussian_taps(sps: float, span: int = 4, duty_cycle: float = 1.0) -> np.ndarray:
    """
    Generates Gaussian pulse-shaping filter taps.

    The Gaussian filter is typically used in GMSK/GFSK modulation to minimize
    occupied bandwidth while introducing controlled Inter-Symbol Interference (ISI).

    Parameters
    ----------
    sps : float
        Samples per symbol.
    span : int, default 4
        Total filter span in symbols. The number of taps will be ``span * sps + 1``
        to ensure symmetry.
    duty_cycle : float, default 1.0
        Full-Width at Half-Maximum (FWHM) of the Gaussian pulse in symbol periods.
        Smaller values produce a narrower pulse (lower ISI but wider bandwidth).
        The Bandwidth-Time product is derived internally as
        ``bt = √2·ln(2) / (π·duty_cycle)``.

    Returns
    -------
    ndarray
        Gaussian filter taps normalized to unit energy.
        Shape: (N_taps,).
    """
    # Convert duty_cycle (FWHM in symbol periods) to BT product.
    # FWHM of h(t) = exp(-(π·t/α)²) is α·√(ln2)/π = √(ln2/2)/B·√(ln2)/π = ln2/(π·B).
    # Wait - using the standard relation:
    #   FWHM = √(2·ln2) · σ_freq,  where B = 1/(2π·σ_freq)  ->  BT = √(ln2/2)/π
    # More directly: FWHM_time = √(ln2/2) / (π·B) which gives BT = √(ln2/2)/π·(1/FWHM)
    # Rearranged: bt = √2·ln(2) / (π·duty_cycle)
    bt = np.sqrt(2) * np.log(2) / (np.pi * duty_cycle)
    logger.debug(
        "Generating Gaussian taps: sps=%s, span=%s, duty_cycle=%s (bt=%.4f)",
        sps,
        span,
        duty_cycle,
        bt,
    )
    # Ensure odd number of taps to have a center peak
    num_taps = int(span * sps)
    if num_taps % 2 == 0:
        num_taps += 1

    t = np.linspace(-span / 2, span / 2, num_taps)

    # Gaussian function
    # h(t) = (sqrt(pi)/alpha) * exp(-(pi*t/alpha)^2)
    # where alpha = sqrt(ln(2)/2)/B
    alpha = np.sqrt(np.log(2) / 2) / bt
    h = (np.sqrt(np.pi) / alpha) * np.exp(-((np.pi * t / alpha) ** 2))

    return normalize(h, "unit_energy")


def smoothrect_taps(
    sps: int, span: int, rise_time: float = 0.22, duty_cycle: float = 1.0
) -> ArrayType:
    """
    Generates a perfectly centered Gaussian-smoothed rectangular pulse.

    This method uses the analytical closed-form solution (Error Function)
    to avoid the 0.5 sample shift artifact typically caused by convolving
    odd/even discrete arrays.

    Parameters
    ----------
    sps : int
        Samples per symbol.
    span : int
        Filter span in symbols. The number of taps will be approximately ``span * sps``.
    rise_time : float, default 0.22
        10%-90% edge transition duration in symbol periods. Smaller values produce
        sharper edges (closer to a hard rect); larger values yield softer transitions
        (approaching a Gaussian pulse). Converted internally to the Gaussian sigma via
        ``σ = rise_time / (2·√2·erfinv(0.8))``.
    duty_cycle : float, default 1.0
        Width of the underlying rectangular pulse in symbol periods. Use 1.0 for NRZ
        and 0.5 for RZ signaling.

    Returns
    -------
    ndarray
        Gaussian-smoothed rectangular pulse taps normalized to unit energy.
        Shape: (N_taps,).
    """
    logger.debug(
        "Generating SmoothRect taps: sps=%s, span=%s, rise_time=%s, duty_cycle=%s",
        sps,
        span,
        rise_time,
        duty_cycle,
    )
    # Ensure odd number of taps to have a center peak
    sps = require_integer_sps(sps, "smoothrect_taps()")
    num_taps = int(span * sps)
    if num_taps % 2 == 0:
        num_taps += 1

    t = np.linspace(-span / 2, span / 2, num_taps)

    # Convert rise_time to Gaussian sigma.
    # rise_time (10%-90%) = 2·√2·erfinv(0.8)·σ  ->  σ = rise_time / (2·√2·erfinv(0.8))
    sigma = rise_time / (2 * np.sqrt(2) * float(scipy.special.erfinv(0.8)))

    # Analytical Formula (Convolved Rect and Gaussian)
    # The underlying rect spans [-duty_cycle/2, +duty_cycle/2].
    # Convolution of rect with Gaussian = difference of error functions.
    w_half = duty_cycle / 2.0
    h = 0.5 * (
        scipy.special.erf((t + w_half) / (sigma * np.sqrt(2)))
        - scipy.special.erf((t - w_half) / (sigma * np.sqrt(2)))
    )

    return normalize(h, "unit_energy")


def rrc_taps(sps: float, rolloff: float = 0.35, span: int = 8) -> np.ndarray:
    """
    Generates Root Raised Cosine (RRC) filter taps.

    RRC filters are used at both the transmitter (pulse shaping) and
    receiver (matched filtering) to satisfy the Nyquist ISI criterion.

    Parameters
    ----------
    sps : float
        Samples per symbol.
    rolloff : float, default 0.35
        Roll-off factor (alpha), range [0, 1].
    span : int, default 8
        Filter span in symbols.

    Returns
    -------
    ndarray
        RRC filter taps normalized to unit energy.
        Shape: (N_taps,).
    """
    logger.debug("Generating RRC taps: sps=%s, rolloff=%s, span=%s", sps, rolloff, span)
    # Ensure odd number of taps
    num_taps = int(span * sps)
    if num_taps % 2 == 0:
        num_taps += 1

    t = np.linspace(-span / 2, span / 2, num_taps)

    # Avoid division by zero
    # 1. t = 0
    # 2. t = +/- 1/(4*rolloff)

    # Initialize array
    h = np.zeros_like(t)

    # Case 1: t = 0
    idx_0 = np.isclose(t, 0)
    h = np.where(idx_0, 1.0 - rolloff + (4 * rolloff / np.pi), h)

    # Case 2: t = +/- 1/(4*rolloff)
    if rolloff > 0:
        idx_singularity = np.isclose(np.abs(t), 1 / (4 * rolloff))
        h = np.where(
            idx_singularity,
            (rolloff / np.sqrt(2))
            * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * rolloff))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * rolloff))
            ),
            h,
        )
    else:
        idx_singularity = np.zeros_like(t, dtype=bool)

    # Case 3: General case
    idx_general = ~(idx_0 | idx_singularity)

    numer = np.sin(np.pi * t * (1 - rolloff)) + 4 * rolloff * t * np.cos(
        np.pi * t * (1 + rolloff)
    )
    denom = np.pi * t * (1 - (4 * rolloff * t) ** 2)

    # Avoid invalid value warning by making den safe
    denom_safe = np.where(idx_general, denom, 1.0)
    h = np.where(idx_general, numer / denom_safe, h)

    return normalize(h, "unit_energy")


def rc_taps(sps: float, rolloff: float = 0.35, span: int = 8) -> ArrayType:
    """
    Generates Raised Cosine (RC) filter taps.

    Parameters
    ----------
    sps : float
        Samples per symbol.
    rolloff : float, default 0.35
        Roll-off factor (alpha), range [0, 1].
    span : int, default 8
        Filter span in symbols.

    Returns
    -------
    ndarray
        RC filter taps normalized to unit energy.
        Shape: (N_taps,).
    """
    logger.debug("Generating RC taps: sps=%s, rolloff=%s, span=%s", sps, rolloff, span)
    # Ensure odd number of taps
    num_taps = int(span * sps)
    if num_taps % 2 == 0:
        num_taps += 1

    t = np.linspace(-span / 2, span / 2, num_taps)

    # Avoid division by zero
    # Singularities at t = +/- 1 / (2 * rolloff)

    # Initialize array
    h = np.zeros_like(t)

    # General case mask
    # Denominator: 1 - (2 * rolloff * t)**2
    # Singularity when 2 * rolloff * |t| = 1 => |t| = 1 / (2 * rolloff)

    if rolloff > 0:
        idx_singularity = np.isclose(np.abs(t), 1 / (2 * rolloff))
        # Value at singularity: (pi / 4) * sinc(1 / (2 * rolloff))
        # sinc(x) = sin(pi * x) / (pi * x)
        # arg = 1 / (2 * rolloff)
        # val = (pi / 4) * sin(pi * arg) / (pi * arg)
        #     = (pi / 4) * sin(pi / (2 * rolloff)) * (2 * rolloff / pi)
        #     = (rolloff / 2) * sin(pi / (2 * rolloff))
        val_singularity = (rolloff / 2) * np.sin(np.pi / (2 * rolloff))
        h = np.where(idx_singularity, val_singularity, h)
    else:
        idx_singularity = np.zeros_like(t, dtype=bool)

    idx_general = ~idx_singularity

    # h(t) = sinc(t) * cos(pi * alpha * t) / (1 - (2 * alpha * t)^2)
    # sinc(t) = sin(pi * t) / (pi * t) (normalized sinc)

    # To avoid t=0 in sinc division, use np.sinc which handles 0 safely
    sinc_t = np.sinc(t)
    cos_t = np.cos(np.pi * rolloff * t)
    denom = 1 - (2 * rolloff * t) ** 2

    # We masked out where denom is 0, so safe to divide where idx_general is true
    # However we compute everywhere then mask, so denom should not be 0 to avoid warning/NaN if backend evals strict
    # backend.where usually evals both branches
    # So we set denom to 1 where it is 0
    denom_safe = np.where(idx_singularity, 1.0, denom)

    res = sinc_t * cos_t / denom_safe
    h = np.where(idx_general, res, h)

    return normalize(h, "unit_energy")


def fir_taps(
    sampling_rate: float,
    num_taps: int,
    cutoff: float | tuple[float, float],
    btype: str = "low",
    window: str = "hamming",
) -> ArrayType:
    """
    Design an FIR filter using the window method.

    Parameters
    ----------
    sampling_rate : float
        The sampling rate of the signal in Hz.
    num_taps : int
        Number of filter coefficients.  For ``btype in {"high", "bandstop"}``
        this should typically be odd to avoid a zero at the Nyquist frequency.
    cutoff : float or (float, float)
        Cutoff frequency in Hz.  A scalar for ``btype in {"low", "high"}``;
        a ``(low, high)`` pair for ``btype in {"band", "bandstop"}``.
    btype : {"low", "high", "band", "bandstop"}, default "low"
        Filter shape - same convention as :func:`butterworth_sos` and the
        other IIR SOS generators below.
    window : str, default "hamming"
        Type of window function to apply (e.g., 'hamming', 'blackman').

    Returns
    -------
    ndarray
        Filter taps with 0 dB passband gain.
        Shape: (num_taps,).
    """
    pass_zero = btype in ("low", "bandstop")
    logger.debug(
        "Designing FIR: btype=%s, cutoff=%s Hz, taps=%s.", btype, cutoff, num_taps
    )
    h = scipy.signal.firwin(
        num_taps, cutoff, window=window, fs=sampling_rate, pass_zero=pass_zero
    )
    return h


# -----------------------------------------------------------------------------
# FILTER DESIGN - IIR SOS GENERATORS (array-only)
# -----------------------------------------------------------------------------
# butterworth_sos, chebyshev1_sos, chebyshev2_sos, elliptic_sos, bessel_sos:
#   Classic IIR filter families in second-order-sections (SOS) form - the
#   IIR-design counterpart to fir_taps() above, one function per filter
#   *family* (since each is a genuinely different algorithm) rather than
#   per shape.  ``btype`` selects the shape (as in scipy's own
#   butter/cheby1/cheby2/ellip/bessel and fir_taps() above), matching how
#   these designs are parameterized in practice rather than splitting one
#   function per shape.
#
# Like the FIR tap generators, these build coefficients from parameters
# alone - no signal input, so none are Signal-aware.  Apply the resulting
# ``sos`` array to a signal with the generic iir_filter() below, the same
# way fir_filter() applies any of the FIR tap generators above.


def _iir_wn(
    cutoff: float | tuple[float, float], btype: str, nyq: float
) -> float | tuple[float, float]:
    """Normalize cutoff(s) in Hz to scipy's ``Wn`` convention (frac. of Nyquist)."""
    if btype in ("band", "bandstop"):
        low, high = cutoff  # type: ignore[misc]
        return (float(low) / nyq, float(high) / nyq)
    return float(cutoff) / nyq  # type: ignore[arg-type]


def butterworth_sos(
    sampling_rate: float,
    cutoff: float | tuple[float, float],
    order: int = 4,
    btype: str = "low",
) -> np.ndarray:
    """
    Design a Butterworth IIR filter in second-order-sections (SOS) form.

    Maximally flat passband, monotonic roll-off - the standard general-purpose
    IIR design.

    Parameters
    ----------
    sampling_rate : float
        Sampling rate of the signal in Hz.
    cutoff : float or (float, float)
        Cutoff frequency in Hz.  A scalar for ``btype in {"low", "high"}``;
        a ``(low, high)`` pair for ``btype in {"band", "bandstop"}``.
    order : int, default 4
        Filter order.
    btype : {"low", "high", "band", "bandstop"}, default "low"
        Filter shape.

    Returns
    -------
    ndarray
        SOS coefficients. Shape: ``(n_sections, 6)``.
    """
    nyq = 0.5 * float(sampling_rate)
    Wn = _iir_wn(cutoff, btype, nyq)
    logger.debug(
        "Designing Butterworth SOS: btype=%s, cutoff=%s Hz, order=%s.",
        btype,
        cutoff,
        order,
    )
    return scipy.signal.butter(order, Wn, btype=btype, output="sos")


def chebyshev1_sos(
    sampling_rate: float,
    cutoff: float | tuple[float, float],
    order: int = 4,
    btype: str = "low",
    ripple: float = 1.0,
) -> np.ndarray:
    """
    Design a Chebyshev Type I IIR filter in SOS form.

    Sharper roll-off than Butterworth for the same order, at the cost of
    passband ripple.

    Parameters
    ----------
    sampling_rate : float
        Sampling rate of the signal in Hz.
    cutoff : float or (float, float)
        Cutoff frequency in Hz, see :func:`butterworth_sos`.
    order : int, default 4
        Filter order.
    btype : {"low", "high", "band", "bandstop"}, default "low"
        Filter shape.
    ripple : float, default 1.0
        Maximum passband ripple, in dB.

    Returns
    -------
    ndarray
        SOS coefficients. Shape: ``(n_sections, 6)``.
    """
    nyq = 0.5 * float(sampling_rate)
    Wn = _iir_wn(cutoff, btype, nyq)
    logger.debug(
        "Designing Chebyshev-I SOS: btype=%s, cutoff=%s Hz, order=%s, ripple=%s dB.",
        btype,
        cutoff,
        order,
        ripple,
    )
    return scipy.signal.cheby1(order, ripple, Wn, btype=btype, output="sos")


def chebyshev2_sos(
    sampling_rate: float,
    cutoff: float | tuple[float, float],
    order: int = 4,
    btype: str = "low",
    attenuation: float = 40.0,
) -> np.ndarray:
    """
    Design a Chebyshev Type II (inverse Chebyshev) IIR filter in SOS form.

    Monotonic passband (no ripple), equiripple stopband - trades a Type I's
    passband ripple for stopband ripple instead.

    Parameters
    ----------
    sampling_rate : float
        Sampling rate of the signal in Hz.
    cutoff : float or (float, float)
        Cutoff frequency in Hz, see :func:`butterworth_sos`.
    order : int, default 4
        Filter order.
    btype : {"low", "high", "band", "bandstop"}, default "low"
        Filter shape.
    attenuation : float, default 40.0
        Minimum stopband attenuation, in dB.

    Returns
    -------
    ndarray
        SOS coefficients. Shape: ``(n_sections, 6)``.
    """
    nyq = 0.5 * float(sampling_rate)
    Wn = _iir_wn(cutoff, btype, nyq)
    logger.debug(
        "Designing Chebyshev-II SOS: btype=%s, cutoff=%s Hz, order=%s, attenuation=%s dB.",
        btype,
        cutoff,
        order,
        attenuation,
    )
    return scipy.signal.cheby2(order, attenuation, Wn, btype=btype, output="sos")


def elliptic_sos(
    sampling_rate: float,
    cutoff: float | tuple[float, float],
    order: int = 4,
    btype: str = "low",
    ripple: float = 1.0,
    attenuation: float = 40.0,
) -> np.ndarray:
    """
    Design an Elliptic (Cauer) IIR filter in SOS form.

    Sharpest roll-off per order of the classic families, at the cost of
    ripple in both passband and stopband.

    Parameters
    ----------
    sampling_rate : float
        Sampling rate of the signal in Hz.
    cutoff : float or (float, float)
        Cutoff frequency in Hz, see :func:`butterworth_sos`.
    order : int, default 4
        Filter order.
    btype : {"low", "high", "band", "bandstop"}, default "low"
        Filter shape.
    ripple : float, default 1.0
        Maximum passband ripple, in dB.
    attenuation : float, default 40.0
        Minimum stopband attenuation, in dB.

    Returns
    -------
    ndarray
        SOS coefficients. Shape: ``(n_sections, 6)``.
    """
    nyq = 0.5 * float(sampling_rate)
    Wn = _iir_wn(cutoff, btype, nyq)
    logger.debug(
        "Designing Elliptic SOS: btype=%s, cutoff=%s Hz, order=%s, ripple=%s dB, "
        "attenuation=%s dB.",
        btype,
        cutoff,
        order,
        ripple,
        attenuation,
    )
    return scipy.signal.ellip(order, ripple, attenuation, Wn, btype=btype, output="sos")


def bessel_sos(
    sampling_rate: float,
    cutoff: float | tuple[float, float],
    order: int = 4,
    btype: str = "low",
    norm: str = "phase",
) -> np.ndarray:
    """
    Design a Bessel/Thomson IIR filter in SOS form.

    Maximally flat group delay (linear phase in the passband) rather than a
    sharp magnitude roll-off - the IIR analogue of a linear-phase FIR design,
    useful when waveform shape (not stopband rejection) matters most.

    Parameters
    ----------
    sampling_rate : float
        Sampling rate of the signal in Hz.
    cutoff : float or (float, float)
        Cutoff frequency in Hz, see :func:`butterworth_sos`.
    order : int, default 4
        Filter order.
    btype : {"low", "high", "band", "bandstop"}, default "low"
        Filter shape.
    norm : {"phase", "delay", "mag"}, default "phase"
        Critical frequency normalization, passed to ``scipy.signal.bessel``.

    Returns
    -------
    ndarray
        SOS coefficients. Shape: ``(n_sections, 6)``.
    """
    nyq = 0.5 * float(sampling_rate)
    Wn = _iir_wn(cutoff, btype, nyq)
    logger.debug(
        "Designing Bessel SOS: btype=%s, cutoff=%s Hz, order=%s, norm=%s.",
        btype,
        cutoff,
        order,
        norm,
    )
    return scipy.signal.bessel(order, Wn, btype=btype, output="sos", norm=norm)


# -----------------------------------------------------------------------------
# FILTERING OPERATIONS (Signal-aware)
# -----------------------------------------------------------------------------
# _ols_forward:  OLS block windowing + batch FFT (shared scaffold)
# _ols_backward: OLS batch IFFT + symmetric discard + reshape (shared scaffold)
# ols_fir_filter: Public OLS FIR convolution (long-tap / memory-bounded)
# shaping_filter_taps: Reconstruct pulse-shaping taps from a Signal's own
#   metadata (pulse_shape/sps/rolloff) - takes a Signal, returns taps, used
#   by matched_filter below to derive default taps for Signal input.
# fir_filter: Generic FIR filtering operation (short-to-medium taps) - applies
#   any of the FIR tap generators above.
# matched_filter: Apply matched filter (time-reversed conjugate of pulse shape)
# iir_filter: Generic IIR filtering operation (SOS form, causal or zero-phase)
#   - applies any of the IIR SOS generators above, the IIR sibling of
#   fir_filter().
#
# shape_pulse (TX symbol -> waveform synthesis) lives in core/generation.py,
# not here: it is a signal-construction primitive, not a transform on an
# existing Signal's samples (see CLAUDE.md, "Signal-Awareness").


def _ols_forward(samples: ArrayType, N_fft: int):
    """
    Overlap-and-save forward pass: block windowing and batch FFT.

    This is the shared OLS scaffolding used by both ``ols_fir_filter`` (SISO
    scalar convolution) and ``zf_equalizer`` (MIMO per-bin matrix multiply).
    It should be called on samples that have already been dispatched to the
    correct backend and shaped as ``(num_ch, N)``.

    Parameters
    ----------
    samples : array_like
        Input samples. Shape: ``(num_ch, N)``. Must be 2-D.
    N_fft : int
        FFT block size. Must be a power of 2 and satisfy
        ``N_fft // 4 >= filter_length`` so the causal/anti-causal guard
        regions fully contain the filter transients.

    Returns
    -------
    Y : array_like
        Batch FFT of all OLS windows. Shape: ``(num_ch, num_blocks, N_fft)``.
    meta : dict
        Scaffold parameters required by ``_ols_backward``:
        ``{'N': int, 'B': int, 'discard': int, 'num_blocks': int}``.
    """
    _, xp, _ = dispatch(samples)
    num_ch, N = samples.shape
    B = N_fft // 2  # 50 % hop - maximises block reuse
    discard = N_fft // 4  # symmetric guard: absorbs causal & anti-causal transients
    num_blocks = (N + B - 1) // B

    # Pre-pad by discard so the first valid output aligns with sample 0.
    # Post-pad to fill the last block window completely.
    pad_left = discard
    pad_right = num_blocks * B - N + discard
    samples_padded = xp.pad(samples, ((0, 0), (pad_left, pad_right)))

    # Zero-copy window extraction via as_strided (view, not copy).
    stride = samples_padded.strides
    windows = xp.lib.stride_tricks.as_strided(
        samples_padded,
        shape=(num_ch, num_blocks, N_fft),
        strides=(stride[0], B * stride[1], stride[1]),
    )

    Y = xp.fft.fft(windows, n=N_fft, axis=-1)  # (num_ch, num_blocks, N_fft)
    meta = {"N": N, "B": B, "discard": discard, "num_blocks": num_blocks}
    return Y, meta


def _ols_backward(X_hat_f: ArrayType, meta: dict) -> ArrayType:
    """
    Overlap-and-save backward pass: batch IFFT, symmetric discard, reshape.

    Parameters
    ----------
    X_hat_f : array_like
        Frequency-domain blocks after per-bin processing.
        Shape: ``(num_ch, num_blocks, N_fft)``.
    meta : dict
        Scaffold parameters returned by ``_ols_forward``.

    Returns
    -------
    array_like
        Time-domain output trimmed to the original signal length ``N``.
        Shape: ``(num_ch, N)``.
    """
    _, xp, _ = dispatch(X_hat_f)
    N = meta["N"]
    B = meta["B"]
    discard = meta["discard"]
    N_fft = X_hat_f.shape[-1]
    num_ch = X_hat_f.shape[0]

    x_hat = xp.fft.ifft(X_hat_f, n=N_fft, axis=-1)
    # Keep the center B samples of each block (symmetric discard of guard regions).
    valid = x_hat[:, :, discard : discard + B]
    out = valid.reshape(num_ch, -1)[:, :N]
    return out


def ols_fir_filter(
    samples: ArrayType | Signal,
    taps: ArrayType,
    N_fft: int | None = None,
    center: bool = True,
) -> ArrayType | Signal:
    """
    Overlap-and-save FIR filter for long-tap or large-signal convolution.

    Implements the overlap-and-save (OLS) block-processing algorithm, which
    processes the signal in fixed-size FFT blocks. This makes it suitable
    for filters with long impulse responses (e.g., chromatic dispersion
    compensation, group-delay equalizers) where a single full-signal FFT
    would be memory-prohibitive on GPU.

    For short filters on moderate-length signals, ``fir_filter`` (which
    uses scipy's FFT convolution) is equally efficient and simpler.

    Parameters
    ----------
    samples : array_like or Signal
        Input signal. Shape: ``(N,)`` for SISO or ``(C, N)`` for
        multi-channel.  A :class:`Signal` returns a new filtered
        :class:`Signal`.
    taps : array_like
        FIR filter coefficients. Shape: ``(L,)``.
    N_fft : int, optional
        FFT block size. Must be a power of 2. Defaults to
        ``max(1024, next_power_of_2(4 * L))`` so that the 25 % guard
        region is at least ``L`` samples long.
    center : bool, default True
        When ``True`` (default), the output alignment matches
        ``fir_filter`` (scipy ``mode='same'``, center-aligned at tap
        ``L // 2``).  The output at position ``n`` equals
        ``sum_k x[n + L//2 - k] * taps[k]``, which is correct for
        pulse-shaped signals where the filter group delay must be
        compensated before symbol sampling.

        When ``False``, the output is the causal linear convolution
        ``y[n] = sum_k x[n-k] * taps[k]`` (equivalent to
        ``numpy.convolve(x, taps, mode='full')[:N]``).  Use this when
        you need the raw causal impulse response (e.g. measuring filter
        step response) or when writing CD/dispersion compensation where
        the two-sided inverse filter alignment is handled externally.

    Returns
    -------
    array_like
        Filtered signal, same shape as ``samples``.

    Notes
    -----
    A symmetric guard of ``N_fft // 4`` samples is discarded from each
    block edge, so ``N_fft // 4 >= len(taps)`` must hold.

    The ``center=True`` path post-pads the input by ``L // 2`` zeros
    before OLS processing and trims the same number of leading output
    samples - a zero-copy shift that costs one extra OLS block at most.
    """
    context = prepare_signal_input(samples, function_name="ols_fir_filter()")
    samples = context.array

    samples, xp, _ = dispatch(samples)
    taps = xp.asarray(taps)
    is_real = not xp.iscomplexobj(samples) and not xp.iscomplexobj(taps)
    out_dtype = samples.dtype  # capture before any reshape

    # Signal drives precision: cast taps to match signal so float64 tap
    # generators do not silently upcast complex64 signals via FFT multiply.
    target_tap_dtype = (
        samples.real.dtype if not xp.iscomplexobj(taps) else samples.dtype
    )
    if taps.dtype != target_tap_dtype:
        taps = taps.astype(target_tap_dtype)

    L = len(taps)
    half = L // 2

    samples, was_1d = as_2d(samples, name="samples")

    N = samples.shape[-1]

    if N_fft is None:
        N_fft = max(1024, 1 << (max(1, 4 * L) - 1).bit_length())

    logger.debug(
        "ols_fir_filter: L=%s, N=%s, N_fft=%s, num_ch=%s, center=%s",
        L,
        N,
        N_fft,
        samples.shape[0],
        center,
    )

    H = xp.fft.fft(taps, n=N_fft)  # frequency response of the filter

    if center:
        # Post-pad by half so the OLS can compute full_conv[half : half+N].
        # This matches scipy's mode='same' (center-aligned, group-delay compensated),
        # which is required for correct eye-opening after pulse-shaped filtering.
        samples_ext = xp.pad(samples, ((0, 0), (0, half)))
        Y, meta = _ols_forward(samples_ext, N_fft)
        X_hat_f = Y * H
        out_ext = _ols_backward(X_hat_f, meta)  # shape: (num_ch, N + half)
        out = out_ext[:, half:]  # trim leading half -> shape: (num_ch, N)
    else:
        Y, meta = _ols_forward(samples, N_fft)
        X_hat_f = Y * H
        out = _ols_backward(X_hat_f, meta)

    if is_real:
        out = out.real  # strip IFFT imaginary noise for real inputs
    elif out.dtype != out_dtype:
        out = out.astype(
            out_dtype
        )  # guard complex inputs (e.g. complex64 -> complex128)
    return context.return_value(restore_1d(was_1d, out))


def shaping_filter_taps(sig: Signal) -> ArrayType:
    """
    Compute pulse-shaping filter taps from a :class:`Signal`'s metadata.

    Reconstructs the transmit pulse-shaping taps from ``pulse_shape`` and the
    associated parameters (``sps``, ``filter_span``, roll-offs, ``duty_cycle``,
    ``rise_time``) stored on the signal.  The taps are returned on the signal's
    current backend.

    Parameters
    ----------
    sig : Signal
        Signal carrying valid ``pulse_shape`` metadata.

    Returns
    -------
    array_like
        Generated filter taps on the signal's device.

    Raises
    ------
    ValueError
        If ``pulse_shape`` is missing or unsupported.
    """
    if not sig.pulse_shape or sig.pulse_shape == "none":
        raise ValueError("No pulse shape defined for this signal.")
    logger.info("Generating shaping filter taps (shape: %s).", sig.pulse_shape)

    # Use stored duty_cycle for RZ; NRZ always uses the full symbol period.
    duty_cycle = sig.duty_cycle if sig.mod_rz else 1.0

    if sig.pulse_shape == "rect":
        taps = rect_taps(
            require_integer_sps(sig.sps, "shaping_filter_taps()"),
            duty_cycle=duty_cycle,
            rise_time=sig.rise_time,
        )
    elif sig.pulse_shape == "smoothrect":
        taps = smoothrect_taps(
            sps=require_integer_sps(sig.sps, "shaping_filter_taps()"),
            span=sig.filter_span,
            rise_time=sig.rise_time,
            duty_cycle=duty_cycle,
        )
    elif sig.pulse_shape == "gaussian":
        taps = gaussian_taps(
            sps=sig.sps, span=sig.filter_span, duty_cycle=sig.duty_cycle
        )
    elif sig.pulse_shape == "rrc":
        taps = rrc_taps(sps=sig.sps, span=sig.filter_span, rolloff=sig.rrc_rolloff)
    elif sig.pulse_shape == "rc":
        taps = rc_taps(sps=sig.sps, span=sig.filter_span, rolloff=sig.rc_rolloff)
    else:
        raise ValueError(f"Unknown pulse shape: {sig.pulse_shape}")

    return to_device(taps, sig.backend)


def fir_filter(
    samples: ArrayType | Signal, taps: ArrayType, axis: int = -1
) -> ArrayType | Signal:
    """
    Apply a Finite Impulse Response (FIR) filter to signal samples.

    The filter is applied via FFT-based convolution for high throughput,
    efficiently handling both CPU and GPU backends.

    Parameters
    ----------
    samples : array_like or Signal
        Input signal samples. Shape: (..., N_samples).  A :class:`Signal`
        returns a new filtered :class:`Signal`.
    taps : array_like
        FIR filter coefficients (impulse response). Shape: (N_taps,).
    axis : int, default -1
        The axis along which the filter is applied (typically the Time axis).

    Returns
    -------
    array_like or Signal
        Filtered samples with the same shape as `samples` (mode='same').
    """
    context = prepare_signal_input(samples, function_name="fir_filter()")
    samples = context.array
    if context.signal is not None:
        axis = -1

    logger.debug(
        "Applying FIR filter via convolution (%s taps, axis=%s).", len(taps), axis
    )
    samples, xp, sp = dispatch(samples)

    # Ensure taps are on the correct backend
    taps = xp.asarray(taps)

    # Signal drives precision: cast taps to match signal dtype so numpy/scipy
    # type-promotion rules do not silently upcast float32/complex64 signals.
    target_tap_dtype = (
        samples.real.dtype if not xp.iscomplexobj(taps) else samples.dtype
    )
    if taps.dtype != target_tap_dtype:
        taps = taps.astype(target_tap_dtype)

    if samples.ndim > 1:
        # Ensure axis is positive
        axis = axis % samples.ndim

        new_shape = [1] * samples.ndim
        new_shape[axis] = len(taps)
        taps_nd = taps.reshape(new_shape)

        result = sp.signal.convolve(samples, taps_nd, mode="same", method="fft")
    else:
        # 1D case
        result = sp.signal.convolve(samples, taps, mode="same", method="fft")

    # Belt-and-suspenders: scipy may still promote internally (version-dependent)
    if result.dtype != samples.dtype:
        result = result.astype(samples.dtype)
    return context.return_value(result)


def matched_filter(
    samples: ArrayType | Signal,
    pulse_taps: ArrayType | None = None,
    taps_normalization: str = "unit_energy",
    axis: int = -1,
) -> ArrayType | Signal:
    """
    Applies a matched filter to the received signal.

    The matched filter is the time-reversed complex conjugate of the pulse
    shaping filter. It maximizes the Signal-to-Noise Ratio (SNR) in the
    presence of AWGN.

    Parameters
    ----------
    samples : array_like or Signal
        Input received samples. Shape: (..., N_samples).  A :class:`Signal`
        returns a new matched-filtered :class:`Signal`; when ``pulse_taps`` is
        omitted, the taps are derived from the signal's ``pulse_shape`` metadata
        via :func:`shaping_filter_taps`.
    pulse_taps : array_like, optional
        Taps of the pulse-shaping filter used at the transmitter.
        Shape: (N_taps,).  Required for raw-array input.
    taps_normalization : {"unit_energy", "unity_gain"}, default "unit_energy"
        Designates how the matched filter taps are normalized.
    axis : int, default -1
        The axis along which to apply the filter.

    Returns
    -------
    array_like or Signal
        Matched filtered samples. Shape: (..., N_samples).
    """
    context = prepare_signal_input(samples, function_name="matched_filter()")
    samples = context.array
    if context.signal is not None:
        sig = context.signal
        taps = pulse_taps
        if taps is None:
            try:
                taps = shaping_filter_taps(sig)
            except ValueError as e:
                logger.error("Cannot apply matched filter: %s", e)
                return sig._shallow_clone()
        pulse_taps = taps
        axis = -1

    if pulse_taps is None:
        raise ValueError("matched_filter() requires pulse_taps for array input.")

    logger.debug("Applying Matched Filter (taps length=%s).", len(pulse_taps))
    samples, xp, _ = dispatch(samples)

    # Matched filter is conjugate and time-reversed version of pulse
    # Ensure pulse_taps is on correct backend
    pulse_taps = xp.asarray(pulse_taps)
    matched_taps = xp.conj(pulse_taps[::-1])

    if taps_normalization == "unity_gain":
        matched_taps = normalize(matched_taps, mode="unity_gain")
    elif taps_normalization == "unit_energy":
        matched_taps = normalize(matched_taps, mode="unit_energy")
    else:
        raise ValueError(
            f"Unknown taps_normalization: {taps_normalization!r}. "
            "Use 'unity_gain' or 'unit_energy'."
        )

    return context.return_value(fir_filter(samples, matched_taps, axis=axis))


def iir_filter(
    samples: ArrayType | Signal,
    sos: ArrayType,
    *,
    axis: int = -1,
    zero_phase: bool = True,
) -> ArrayType | Signal:
    """
    Apply an Infinite Impulse Response (IIR) filter, in SOS form, to signal samples.

    The IIR sibling of :func:`fir_filter`: takes coefficients designed
    separately (:func:`butterworth_sos`, :func:`chebyshev1_sos`,
    :func:`chebyshev2_sos`, :func:`elliptic_sos`, :func:`bessel_sos`, or any
    other second-order-sections design) and applies them, rather than coupling
    a specific filter family to the application step.

    Parameters
    ----------
    samples : array_like or Signal
        Input signal samples. Shape: ``(N,)`` or ``(C, N)``.  A
        :class:`Signal` returns a new filtered :class:`Signal`.
    sos : array_like
        Second-order-sections filter coefficients. Shape: ``(n_sections, 6)``.
    axis : int, default -1
        The axis along which the filter is applied.
    zero_phase : bool, default True
        ``True`` - forward-backward (``sosfiltfilt``): zero phase distortion
        (no group delay), at the cost of needing the whole record up front
        (non-causal, offline use).
        ``False`` - causal (``sosfilt``): the filter's real, frequency-
        dependent group delay is present in the output, as it would be in a
        streamed/real-time application.

    Returns
    -------
    array_like or Signal
        Filtered samples, same shape as ``samples``.

    Notes
    -----
    Internally promotes to ``float64``/``complex128`` for the filtering call
    and casts back to the input dtype on return: at very low normalized
    cutoffs (e.g. phase-drift extraction), SOS poles bunch near ``z=1`` and
    single precision is not numerically safe (see ``CLAUDE.md``, "Phase
    Unwrapping & Kalman Smoothers").
    """
    context = prepare_signal_input(samples, function_name="iir_filter()")
    samples = context.array

    samples, xp, sp = dispatch(samples)
    sos = xp.asarray(sos)

    logger.debug(
        "Applying IIR filter (%s SOS sections, zero_phase=%s, axis=%s).",
        sos.shape[0],
        zero_phase,
        axis,
    )

    in_dtype = samples.dtype
    work_dtype = xp.complex128 if xp.iscomplexobj(samples) else xp.float64
    x_work = samples.astype(work_dtype)
    if zero_phase:
        result = sp.signal.sosfiltfilt(sos, x_work, axis=axis)
    else:
        result = sp.signal.sosfilt(sos, x_work, axis=axis)
    return context.return_value(result.astype(in_dtype, copy=False))


# -----------------------------------------------------------------------------
# CHROMATIC DISPERSION (Signal-aware)
# -----------------------------------------------------------------------------


def compensate_chromatic_dispersion(
    samples: ArrayType | Signal,
    sampling_rate: float | None = None,
    dispersion_ps_nm_km: float | None = None,
    fiber_length_km: float | None = None,
    center_wavelength_nm: float | None = None,
) -> ArrayType | Signal:
    """
    Electronic dispersion compensation (EDC) for chromatic dispersion.

    Applies the inverse of the CD frequency-domain transfer function to remove
    chromatic dispersion accumulated over a fiber link:

        H_EDC(f) = exp(j/2 * beta_2 * (2*pi*f)^2 * L)

    where

        beta_2 = -D * lambda^2 / (2*pi*c)

    and D is the dispersion parameter, lambda is the center wavelength,
    c is the speed of light, and L is the fiber length.

    Parameters
    ----------
    samples : array_like or Signal
        Complex baseband signal. Shape: ``(N,)`` (SISO) or ``(C, N)`` (MIMO).
    sampling_rate : float, optional
        Sampling rate in Hz.  Required for array input; ignored for
        :class:`Signal` input, which always uses the signal's own
        ``sampling_rate``.
    dispersion_ps_nm_km : float
        Fiber dispersion parameter D in ps / (nm * km).
        Standard SMF-28: 17 ps/(nm*km) at 1550 nm.
    fiber_length_km : float
        Fiber span length in km.
    center_wavelength_nm : float
        Center wavelength in nm (e.g. 1550 for C-band).

    Returns
    -------
    array_like or Signal
        CD-compensated signal, same shape, dtype, and backend as input.  A
        :class:`Signal` returns a new compensated :class:`Signal`.

    See Also
    --------
    commkit.impairments.apply_chromatic_dispersion :
        Apply the forward CD impairment (use before this function in simulation).

    Examples
    --------
    >>> cd_free = compensate_chromatic_dispersion(
    ...     received, dispersion_ps_nm_km=17.0, fiber_length_km=80.0,
    ...     center_wavelength_nm=1550.0, sampling_rate=fs)
    """
    context = prepare_signal_input(
        samples, function_name="compensate_chromatic_dispersion()"
    )
    samples = context.array
    sampling_rate = context.required("sampling_rate", sampling_rate)
    if (
        dispersion_ps_nm_km is None
        or fiber_length_km is None
        or center_wavelength_nm is None
    ):
        raise ValueError(
            "compensate_chromatic_dispersion() requires dispersion_ps_nm_km, "
            "fiber_length_km, and center_wavelength_nm."
        )

    logger.info(
        "Compensating CD (D=%s ps/nm/km, L=%s km, λ=%s nm).",
        dispersion_ps_nm_km,
        fiber_length_km,
        center_wavelength_nm,
    )

    samples, xp, _ = dispatch(samples)
    samples, was_1d = as_2d(samples, name="samples")
    C, N = samples.shape

    beta2 = _cd_beta2_length(
        dispersion_ps_nm_km, fiber_length_km, center_wavelength_nm
    )  # s²  (β₂·L product)

    omega = 2.0 * np.pi * xp.fft.fftfreq(N, d=1.0 / sampling_rate)
    H = xp.exp(1j * (beta2 / 2.0) * omega**2)

    S_F = xp.fft.fft(samples, axis=-1)
    out_F = S_F * H[None, :]
    result = xp.fft.ifft(out_F, axis=-1)

    if result.dtype != samples.dtype:
        result = result.astype(samples.dtype)

    return context.return_value(restore_1d(was_1d, result))
