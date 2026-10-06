"""
Digital filtering and pulse shaping.

This module provides routines for design and application of digital filters
commonly used in communication systems. It supports both standard FIR filters
and specialized pulse-shaping filters, with high-performance execution on
both CPU and GPU backends.
"""

from dataclasses import dataclass

import numpy as np
import scipy

from ._array import as_2d, restore_1d
from ._dispersion import apply_dispersion
from ._overlap_save import ols_backward, ols_forward
from .backend import ArrayType, dispatch
from .core._signal_adapter import S, adapt_signal, require_integer_sps
from .logger import logger
from .math import normalize

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


def rect_taps(
    *, sps: int, duty_cycle: float = 1.0, rise_time: float = 0.0
) -> np.ndarray:
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


def gaussian_taps(*, sps: float, fwhm: float = 1.0, span: int = 4) -> np.ndarray:
    """
    Generates Gaussian pulse-shaping filter taps.

    The Gaussian filter is typically used in GMSK/GFSK modulation to minimize
    occupied bandwidth while introducing controlled Inter-Symbol Interference (ISI).

    Parameters
    ----------
    sps : float
        Samples per symbol.
    fwhm : float, default 1.0
        Full width at half maximum of the pulse in symbol periods.  Smaller
        values give a narrower pulse (less ISI, wider bandwidth).  The
        bandwidth-time product is ``BT = √2·ln(2) / (π·fwhm)``.
    span : int, default 4
        Total filter span in symbols. The number of taps will be ``span * sps + 1``
        to ensure symmetry.

    Returns
    -------
    ndarray
        Gaussian filter taps normalized to unit energy.
        Shape: (N_taps,).
    """
    # h(t) = exp(-(π t / α)²) with α = √(ln2 / 2) / BT has its half-maximum
    # at |t| = α √ln2 / π, so FWHM = √2 ln2 / (π BT)  ->  BT = √2 ln2 / (π fwhm).
    bt = np.sqrt(2) * np.log(2) / (np.pi * fwhm)
    logger.debug(
        "Generating Gaussian taps: sps=%s, span=%s, fwhm=%s (bt=%.4f)",
        sps,
        span,
        fwhm,
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

    return normalize(h, mode="unit_energy")


def smoothrect_taps(
    *, sps: int, span: int, rise_time: float = 0.22, duty_cycle: float = 1.0
) -> np.ndarray:
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

    return normalize(h, mode="unit_energy")


def rrc_taps(*, sps: float, rolloff: float = 0.35, span: int = 8) -> np.ndarray:
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

    return normalize(h, mode="unit_energy")


def rc_taps(*, sps: float, rolloff: float = 0.35, span: int = 8) -> np.ndarray:
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

    return normalize(h, mode="unit_energy")


# -----------------------------------------------------------------------------
# PULSE VALUE OBJECTS
# -----------------------------------------------------------------------------
# A pulse describes a transmit pulse shape independently of the sampling rate;
# ``pulse.taps(sps)`` builds the taps.  Wherever a pulse is accepted, a raw
# taps array is accepted too.


@dataclass(frozen=True)
class Pulse:
    """Base class of the pulse value objects (``RRC``, ``RC``, ``Gaussian``,
    ``Rect``, ``SmoothRect``)."""

    def taps(self, sps: float) -> np.ndarray:
        """Pulse taps at ``sps`` samples per symbol (host ``float64``)."""
        raise NotImplementedError


def _check_span(span: int) -> None:
    if not isinstance(span, int | np.integer) or span < 1:
        raise ValueError(f"span must be a positive integer, got {span!r}.")


def _check_rolloff(rolloff: float) -> None:
    if not 0.0 <= rolloff <= 1.0:
        raise ValueError(f"rolloff must be in [0, 1], got {rolloff}.")


def _check_duty_cycle(duty_cycle: float) -> None:
    if not 0.0 < duty_cycle <= 1.0:
        raise ValueError(f"duty_cycle must be in (0, 1], got {duty_cycle}.")


@dataclass(frozen=True)
class RRC(Pulse):
    """Root-raised-cosine pulse.

    Parameters
    ----------
    rolloff : float
        Roll-off factor in ``[0, 1]``.
    span : int, default 10
        Length in symbols; the taps have ``span * sps`` samples, rounded up to
        an odd count.  Unit energy.
    """

    rolloff: float
    span: int = 10

    def __post_init__(self) -> None:
        _check_rolloff(self.rolloff)
        _check_span(self.span)

    def taps(self, sps: float) -> np.ndarray:
        return rrc_taps(sps=sps, rolloff=self.rolloff, span=self.span)


@dataclass(frozen=True)
class RC(Pulse):
    """Raised-cosine (Nyquist) pulse; zero ISI at the symbol instants.

    Parameters are those of :class:`RRC`.
    """

    rolloff: float
    span: int = 10

    def __post_init__(self) -> None:
        _check_rolloff(self.rolloff)
        _check_span(self.span)

    def taps(self, sps: float) -> np.ndarray:
        return rc_taps(sps=sps, rolloff=self.rolloff, span=self.span)


@dataclass(frozen=True)
class Gaussian(Pulse):
    """Gaussian pulse.

    Parameters
    ----------
    fwhm : float, default 1.0
        Full width at half maximum in symbol periods.  The bandwidth-time
        product is ``BT = sqrt(2) ln(2) / (pi fwhm)``.
    span : int, default 10
        Length in symbols.  Unit energy.
    """

    fwhm: float = 1.0
    span: int = 10

    def __post_init__(self) -> None:
        if not self.fwhm > 0:
            raise ValueError(f"fwhm must be > 0, got {self.fwhm}.")
        _check_span(self.span)

    def taps(self, sps: float) -> np.ndarray:
        return gaussian_taps(sps=sps, fwhm=self.fwhm, span=self.span)


@dataclass(frozen=True)
class Rect(Pulse):
    """Rectangular or trapezoidal pulse (integer ``sps`` only).

    Parameters
    ----------
    duty_cycle : float, default 1.0
        Total width in symbol periods, in ``(0, 1]``: 1.0 is NRZ, 0.5 is RZ.
    rise_time : float, default 0.0
        Length of each linear edge in symbol periods, at most
        ``duty_cycle / 2``.  The taps are not normalized (unit height).
    """

    duty_cycle: float = 1.0
    rise_time: float = 0.0

    def __post_init__(self) -> None:
        _check_duty_cycle(self.duty_cycle)
        if not 0.0 <= self.rise_time <= self.duty_cycle / 2:
            raise ValueError(
                f"rise_time must be in [0, duty_cycle / 2], got {self.rise_time}."
            )

    def taps(self, sps: float) -> np.ndarray:
        """Taps at integer ``sps``; ``duty_cycle * sps`` and ``rise_time * sps``
        must be whole samples, so the pulse is never silently rounded."""
        sps = require_integer_sps(sps, "Rect.taps()")
        for name, value in (
            ("duty_cycle", self.duty_cycle),
            ("rise_time", self.rise_time),
        ):
            n = value * sps
            if abs(n - round(n)) > 1e-9:
                raise ValueError(
                    f"Rect: {name} * sps = {value} * {sps} = {n:g} is not a whole "
                    "number of samples; choose sps accordingly."
                )
        return rect_taps(sps=sps, duty_cycle=self.duty_cycle, rise_time=self.rise_time)


@dataclass(frozen=True)
class SmoothRect(Pulse):
    """Rectangle convolved with a Gaussian (integer ``sps`` only).

    Parameters
    ----------
    rise_time : float, default 0.22
        10%-90% edge time in symbol periods.
    duty_cycle : float, default 1.0
        Width of the underlying rectangle in symbol periods: 1.0 is NRZ,
        0.5 is RZ.
    span : int, default 10
        Length in symbols.  Unit energy.
    """

    rise_time: float = 0.22
    duty_cycle: float = 1.0
    span: int = 10

    def __post_init__(self) -> None:
        if not self.rise_time > 0:
            raise ValueError(f"rise_time must be > 0, got {self.rise_time}.")
        _check_duty_cycle(self.duty_cycle)
        _check_span(self.span)

    def taps(self, sps: float) -> np.ndarray:
        return smoothrect_taps(
            sps=require_integer_sps(sps, "SmoothRect.taps()"),
            span=self.span,
            rise_time=self.rise_time,
            duty_cycle=self.duty_cycle,
        )


def fir_taps(
    *,
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
    *,
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
    return np.asarray(scipy.signal.butter(order, Wn, btype=btype, output="sos"))


def chebyshev1_sos(
    *,
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
    return np.asarray(scipy.signal.cheby1(order, ripple, Wn, btype=btype, output="sos"))


def chebyshev2_sos(
    *,
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
    return np.asarray(
        scipy.signal.cheby2(order, attenuation, Wn, btype=btype, output="sos")
    )


def elliptic_sos(
    *,
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
    return np.asarray(
        scipy.signal.ellip(order, ripple, attenuation, Wn, btype=btype, output="sos")
    )


def bessel_sos(
    *,
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
    return np.asarray(
        scipy.signal.bessel(order, Wn, btype=btype, output="sos", norm=norm)
    )


# -----------------------------------------------------------------------------
# FILTERING OPERATIONS (Signal-aware)
# -----------------------------------------------------------------------------
# ols_fir_filter: Public OLS FIR convolution (long-tap / memory-bounded)
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


def ols_fir_filter(
    samples: S,
    taps: ArrayType,
    *,
    fft_size: int | None = None,
    center: bool = True,
) -> S:
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
    fft_size : int, optional
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
    A symmetric guard of ``fft_size // 4`` samples is discarded from each
    block edge, so ``fft_size // 4 >= len(taps)`` must hold.

    The ``center=True`` path post-pads the input by ``L // 2`` zeros
    before OLS processing and trims the same number of leading output
    samples - a zero-copy shift that costs one extra OLS block at most.
    """
    signal_adapter = adapt_signal(samples, function_name="ols_fir_filter()")
    x = signal_adapter.array

    x, xp, _ = dispatch(x)
    taps = xp.asarray(taps)
    is_real = not xp.iscomplexobj(x) and not xp.iscomplexobj(taps)
    out_dtype = x.dtype  # capture before any reshape

    # Signal drives precision: cast taps to match signal so float64 tap
    # generators do not silently upcast complex64 signals via FFT multiply.
    target_tap_dtype = x.real.dtype if not xp.iscomplexobj(taps) else x.dtype
    if taps.dtype != target_tap_dtype:
        taps = taps.astype(target_tap_dtype)

    L = len(taps)
    half = L // 2

    x, was_1d = as_2d(x, name="samples")

    N = x.shape[-1]

    N_fft = fft_size
    if N_fft is None:
        N_fft = max(1024, 1 << (max(1, 4 * L) - 1).bit_length())
    elif N_fft & (N_fft - 1) or N_fft // 4 < L:
        raise ValueError(
            f"fft_size must be a power of 2 with fft_size // 4 >= len(taps) "
            f"({L}), got {fft_size}."
        )

    logger.debug(
        "ols_fir_filter: L=%s, N=%s, N_fft=%s, num_ch=%s, center=%s",
        L,
        N,
        N_fft,
        x.shape[0],
        center,
    )

    H = xp.fft.fft(taps, n=N_fft)  # frequency response of the filter

    if center:
        # Post-pad by half so the OLS can compute full_conv[half : half+N].
        # This matches scipy's mode='same' (center-aligned, group-delay compensated),
        # which is required for correct eye-opening after pulse-shaped filtering.
        samples_ext = xp.pad(x, ((0, 0), (0, half)))
        Y, meta = ols_forward(samples_ext, N_fft)
        X_hat_f = Y * H
        out_ext = ols_backward(X_hat_f, meta)  # shape: (num_ch, N + half)
        out = out_ext[:, half:]  # trim leading half -> shape: (num_ch, N)
    else:
        Y, meta = ols_forward(x, N_fft)
        X_hat_f = Y * H
        out = ols_backward(X_hat_f, meta)

    if is_real:
        out = out.real  # strip IFFT imaginary noise for real inputs
    elif out.dtype != out_dtype:
        out = out.astype(
            out_dtype
        )  # guard complex inputs (e.g. complex64 -> complex128)
    return signal_adapter.wrap_samples(restore_1d(was_1d, out))


def fir_filter(samples: S, taps: ArrayType) -> S:
    """
    Apply an FIR filter along the time (last) axis.

    FFT convolution, ``mode="same"``: the output is centred on tap
    ``len(taps) // 2``, so a symmetric filter adds no delay.

    Parameters
    ----------
    samples : array_like or Signal
        Input samples, ``(N,)`` or ``(C, N)``.  A :class:`Signal` returns a
        new filtered :class:`Signal`.
    taps : array_like
        Filter coefficients (impulse response), ``(L,)``.  Cast to the
        precision of ``samples``.

    Returns
    -------
    array_like or Signal
        Filtered samples, same shape and dtype as ``samples``.
    """
    signal_adapter = adapt_signal(samples, function_name="fir_filter()")
    x, xp, sp = dispatch(signal_adapter.array)
    taps = xp.asarray(taps)
    if taps.ndim != 1:
        raise ValueError(f"taps must be 1-D, got shape {taps.shape}.")
    logger.debug("Applying FIR filter via convolution (%s taps).", taps.size)

    # Signal drives precision: cast taps to match signal dtype so numpy/scipy
    # type-promotion rules do not silently upcast float32/complex64 signals.
    target_tap_dtype = x.real.dtype if not xp.iscomplexobj(taps) else x.dtype
    if taps.dtype != target_tap_dtype:
        taps = taps.astype(target_tap_dtype)

    taps_nd = taps.reshape((1,) * (x.ndim - 1) + (-1,))
    result = sp.signal.convolve(x, taps_nd, mode="same", method="fft")

    # Belt-and-suspenders: scipy may still promote internally (version-dependent)
    if result.dtype != x.dtype:
        result = result.astype(x.dtype)
    return signal_adapter.wrap_samples(result)


def matched_filter(
    samples: S,
    *,
    pulse: Pulse | ArrayType | None = None,
    taps_normalization: str = "unit_energy",
) -> S:
    """
    Matched filter: convolve with the time-reversed conjugate of the pulse.

    Maximizes the SNR at the symbol instants in AWGN.

    Parameters
    ----------
    samples : array_like or Signal
        Received samples, ``(N,)`` or ``(C, N)``.  A :class:`Signal` returns
        a new filtered :class:`Signal`.
    pulse : Pulse or array_like, optional
        Transmit pulse, as a pulse object (``RRC(0.35)``) or its taps.
        Defaults to the Signal's ``pulse``; required for array input.  A
        pulse object needs the samples per symbol, so array input takes taps.
    taps_normalization : {"unit_energy", "unity_gain"}, default "unit_energy"
        Normalization of the matched-filter taps.

    Returns
    -------
    array_like or Signal
        Filtered samples, same shape as ``samples``.
    """
    signal_adapter = adapt_signal(samples, function_name="matched_filter()")
    pulse = signal_adapter.resolve_choice("pulse", pulse)
    if pulse is None:
        raise ValueError(
            "matched_filter() needs a pulse: pass pulse= (a Pulse or taps) or a "
            "Signal that has one."
        )
    if isinstance(pulse, Pulse):
        sig = signal_adapter.signal
        if sig is None:
            raise ValueError(
                "matched_filter(): a Pulse needs the x per symbol; pass "
                "pulse.taps(sps) for array input."
            )
        pulse = pulse.taps(sig.sps)
    if taps_normalization not in ("unit_energy", "unity_gain"):
        raise ValueError(
            f"Unknown taps_normalization: {taps_normalization!r}. "
            "Use 'unity_gain' or 'unit_energy'."
        )

    x, xp, _ = dispatch(signal_adapter.array)
    pulse_taps = xp.asarray(pulse)
    logger.debug("Applying matched filter (%s taps).", pulse_taps.size)
    matched_taps = normalize(xp.conj(pulse_taps[::-1]), mode=taps_normalization)
    return signal_adapter.wrap_samples(fir_filter(x, matched_taps))


def iir_filter(
    samples: S,
    sos: ArrayType,
    *,
    zero_phase: bool = True,
) -> S:
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
    signal_adapter = adapt_signal(samples, function_name="iir_filter()")
    x = signal_adapter.array

    x, xp, sp = dispatch(x)
    sos = xp.asarray(sos)

    logger.debug(
        "Applying IIR filter (%s SOS sections, zero_phase=%s).",
        sos.shape[0],
        zero_phase,
    )

    in_dtype = x.dtype
    work_dtype = xp.complex128 if xp.iscomplexobj(x) else xp.float64
    x_work = x.astype(work_dtype)
    if zero_phase:
        result = sp.signal.sosfiltfilt(sos, x_work, axis=-1)
    else:
        result = sp.signal.sosfilt(sos, x_work, axis=-1)
    return signal_adapter.wrap_samples(result.astype(in_dtype, copy=False))


# -----------------------------------------------------------------------------
# CHROMATIC DISPERSION (Signal-aware)
# -----------------------------------------------------------------------------


def correct_chromatic_dispersion(
    samples: S,
    *,
    dispersion_ps_nm_km: float,
    fiber_length_km: float,
    center_wavelength_nm: float,
    sampling_rate: float | None = None,
) -> S:
    """
    Electronic dispersion compensation: the inverse fiber response.

    Multiplies the spectrum by ``H(ω) = exp(+j β₂ L ω² / 2)`` with
    ``β₂ = -D λ² / (2π c)``, which undoes
    :func:`commkit.impairments.apply_chromatic_dispersion` exactly.

    Parameters
    ----------
    samples : array_like or Signal
        Complex baseband samples, ``(N,)`` or ``(C, N)``.
    dispersion_ps_nm_km : float
        Fiber dispersion parameter D in ps / (nm km) (SMF-28: 17 at 1550 nm).
    fiber_length_km : float
        Fiber length in km.
    center_wavelength_nm : float
        Carrier wavelength in nm.
    sampling_rate : float, optional
        Sampling rate in Hz.  Taken from the Signal; required for array
        input.  A value that disagrees with the Signal raises.

    Returns
    -------
    array_like or Signal
        Compensated samples, same shape, dtype and device as the input.

    Examples
    --------
    >>> sig = correct_chromatic_dispersion(
    ...     sig, dispersion_ps_nm_km=17.0, fiber_length_km=80.0,
    ...     center_wavelength_nm=1550.0)
    """
    signal_adapter = adapt_signal(
        samples, function_name="correct_chromatic_dispersion()"
    )
    sampling_rate = signal_adapter.resolve_fact("sampling_rate", sampling_rate)
    logger.info(
        "Correcting CD (D=%s ps/nm/km, L=%s km, λ=%s nm).",
        dispersion_ps_nm_km,
        fiber_length_km,
        center_wavelength_nm,
    )
    result = apply_dispersion(
        signal_adapter.array,
        sampling_rate=sampling_rate,
        dispersion_ps_nm_km=dispersion_ps_nm_km,
        fiber_length_km=fiber_length_km,
        center_wavelength_nm=center_wavelength_nm,
        inverse=True,
    )
    return signal_adapter.wrap_samples(result)
