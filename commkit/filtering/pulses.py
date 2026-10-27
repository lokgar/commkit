"""Pulse shapes: the value objects and their tap generators.

The pulses :class:`RRC`, :class:`RC`, :class:`Gaussian`, :class:`Rect` and
:class:`SmoothRect` (``pulse.taps(sps)``, subclasses of :class:`Pulse`) and
their tap generators :func:`rrc_taps`, :func:`rc_taps`,
:func:`gaussian_taps`, :func:`rect_taps` and :func:`smoothrect_taps`.  Taps
are built from parameters alone, as host NumPy arrays.
"""

from dataclasses import dataclass

import numpy as np
import scipy

from ..core._signal_adapter import require_integer_sps
from ..logger import logger
from ..math import normalize

__all__ = [
    "Gaussian",
    "Pulse",
    "RC",
    "RRC",
    "Rect",
    "SmoothRect",
    "gaussian_taps",
    "rc_taps",
    "rect_taps",
    "rrc_taps",
    "smoothrect_taps",
]

# -----------------------------------------------------------------------------
# PULSE TAP GENERATORS (array-only)
# -----------------------------------------------------------------------------
# rect_taps:     Hard or trapezoidal rectangular pulse taps
# gaussian_taps: Gaussian filter taps
# smoothrect_taps: Gaussian-smoothed rectangular pulse taps
# rrc_taps:      Root Raised Cosine filter taps
# rc_taps:       Raised Cosine filter taps
#
# All build taps from parameters alone - no signal input, so none are
# Signal-aware (same category as gray_code/barker_sequence).


def _symbol_grid(sps: float, span: float) -> np.ndarray:
    """Tap times of a pulse, in symbols.

    An odd count, centered on 0, spaced exactly ``1/sps``, covering at most
    ``span`` symbols.

    ``int(span * sps)`` taps, plus one when that is even, so the taps sit on
    the sample grid of the signal (a symmetric pulse has its peak on a
    sample and its Nyquist zeros on the other symbols' samples).
    """
    num_taps = int(span * sps)
    num_taps += 1 - num_taps % 2
    return (np.arange(num_taps) - num_taps // 2) / sps


def rect_taps(
    *, sps: int, duty_cycle: float = 1.0, rise_time: float = 0.0
) -> np.ndarray:
    """Generate rectangular or trapezoidal pulse taps.

    With ``rise_time=0`` (default) the output is a hard rectangular pulse of
    width ``duty_cycle`` symbol periods.  With ``rise_time > 0`` the leading
    and trailing edges are linear ramps from 0 towards 1, an isosceles
    trapezoid that models a slew-rate-limited driver or modulator.

    Parameters
    ----------
    sps : int
        Samples per symbol; must be an integer value.
    duty_cycle : float, default 1.0
        Total pulse width in symbol periods, including both ramps, in
        ``(0, 1]``: 1.0 is NRZ, 0.5 is RZ.  Rounded to whole samples.
    rise_time : float, default 0.0
        Length of each linear ramp (0 to 100 % of the height) in symbol
        periods, at most ``duty_cycle / 2`` so that the ramps do not
        overlap.  Rounded to whole samples.

    Returns
    -------
    ndarray
        Host ``float64`` taps of unit height (not normalized),
        ``(round(duty_cycle * sps),)``.

    Raises
    ------
    ValueError
        If ``rise_time > duty_cycle / 2`` or ``sps`` is not an integer.

    See Also
    --------
    Rect : the pulse object, which refuses widths that are not whole samples.
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


def gaussian_taps(*, sps: float, fwhm: float = 1.0, span: int = 10) -> np.ndarray:
    """Generate Gaussian pulse taps.

    The pulse shape of GMSK/GFSK: a narrow occupied bandwidth at the price of
    controlled inter-symbol interference.

    Parameters
    ----------
    sps : float
        Samples per symbol.
    fwhm : float, default 1.0
        Full width at half maximum of the pulse in symbol periods.  Smaller
        values give a narrower pulse (less ISI, wider bandwidth).  The
        bandwidth-time product is ``BT = √2·ln(2) / (π·fwhm)``.
    span : int, default 10
        Filter span in symbols: ``int(span * sps)`` taps, made odd, spaced
        ``1/sps`` and centered on a sample.

    Returns
    -------
    ndarray
        Host ``float64`` taps of unit energy.
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
    t = _symbol_grid(sps, span)

    # Gaussian function
    # h(t) = (sqrt(pi)/alpha) * exp(-(pi*t/alpha)^2)
    # where alpha = sqrt(ln(2)/2)/B
    alpha = np.sqrt(np.log(2) / 2) / bt
    h = (np.sqrt(np.pi) / alpha) * np.exp(-((np.pi * t / alpha) ** 2))

    return normalize(h, mode="unit_energy")


def smoothrect_taps(
    *, sps: int, span: int, rise_time: float = 0.22, duty_cycle: float = 1.0
) -> np.ndarray:
    """Generate the taps of a rectangle smoothed by a Gaussian.

    Evaluates the closed form of the convolution (a difference of error
    functions) on the sample grid, so the pulse is centered on a sample,
    without the half-sample shift of convolving two discrete arrays.

    Parameters
    ----------
    sps : int
        Samples per symbol; must be an integer value.
    span : int
        Filter span in symbols: ``int(span * sps)`` taps, made odd, spaced
        ``1/sps``.
    rise_time : float, default 0.22
        10 %-90 % edge time in symbol periods, ``> 0``.  Smaller values give
        sharper edges (closer to a hard rectangle), larger ones softer edges
        (towards a Gaussian).  The Gaussian's standard deviation is
        ``σ = rise_time / (2·√2·erfinv(0.8))``.
    duty_cycle : float, default 1.0
        Width of the underlying rectangle in symbol periods: 1.0 for NRZ,
        0.5 for RZ.

    Returns
    -------
    ndarray
        Host ``float64`` taps of unit energy.

    Raises
    ------
    ValueError
        If ``sps`` is not an integer.
    """
    logger.debug(
        "Generating SmoothRect taps: sps=%s, span=%s, rise_time=%s, duty_cycle=%s",
        sps,
        span,
        rise_time,
        duty_cycle,
    )
    sps = require_integer_sps(sps, "smoothrect_taps()")
    t = _symbol_grid(sps, span)

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


def rrc_taps(*, sps: float, rolloff: float = 0.35, span: int = 32) -> np.ndarray:
    """Generate root-raised-cosine (RRC) pulse taps.

    An RRC at the transmitter (pulse shaping) and at the receiver (matched
    filter) together make a raised cosine, which meets the Nyquist ISI
    criterion.

    Parameters
    ----------
    sps : float
        Samples per symbol.
    rolloff : float, default 0.35
        Roll-off factor in ``[0, 1]``.
    span : int, default 32
        Filter span in symbols: ``int(span * sps)`` taps, made odd, spaced
        ``1/sps`` (see :class:`RRC` for the SNR floor of truncation).

    Returns
    -------
    ndarray
        Host ``float64`` taps of unit energy.
    """
    logger.debug("Generating RRC taps: sps=%s, rolloff=%s, span=%s", sps, rolloff, span)
    # Ensure odd number of taps
    t = _symbol_grid(sps, span)

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


def rc_taps(*, sps: float, rolloff: float = 0.35, span: int = 32) -> np.ndarray:
    """Generate raised-cosine (RC) pulse taps.

    Parameters
    ----------
    sps : float
        Samples per symbol.
    rolloff : float, default 0.35
        Roll-off factor in ``[0, 1]``.
    span : int, default 32
        Filter span in symbols: ``int(span * sps)`` taps, made odd, spaced
        ``1/sps`` (see :class:`RRC` for the SNR floor of truncation).

    Returns
    -------
    ndarray
        Host ``float64`` taps of unit energy.
    """
    logger.debug("Generating RC taps: sps=%s, rolloff=%s, span=%s", sps, rolloff, span)
    # Ensure odd number of taps
    t = _symbol_grid(sps, span)

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
    span : int, default 32
        Length in symbols; ``int(span * sps)`` taps made odd, spaced
        ``1/sps``.  Unit energy.  Truncation limits the matched-filter SNR
        (RRC on both ends, measured): at 32 symbols about 37 dB for
        roll-off 0.05, 44 dB for 0.1, 52 dB for 0.2; at 10 symbols only
        20, 25 and 44 dB.  Roll-offs below 0.05 need a longer span.
    """

    rolloff: float
    span: int = 32

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
    span: int = 32

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
