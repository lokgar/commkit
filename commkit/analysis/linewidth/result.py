"""The results of :func:`estimate_linewidth`: :class:`LinewidthEstimate` and
one subclass per method.

Also the parameter checks that the phase-trajectory and the beat method
objects share.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "BetaSeparationEstimate",
    "DshFmFitEstimate",
    "DshIncrementEstimate",
    "DshLorentzianEstimate",
    "FmNoiseFitEstimate",
    "IncrementEstimate",
    "IncrementSlopeEstimate",
    "LinewidthEstimate",
]


# -----------------------------------------------------------------------------
# Result
# -----------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class LinewidthEstimate:
    """Result of :func:`estimate_linewidth`, the fields every method gives.

    The constructor takes the same fields.  Each method returns a subclass
    with its diagnostics:

    - FM-noise PSD fits, :class:`FmNoiseFitEstimate`:
      :class:`BetaSeparationEstimate` and :class:`DshFmFitEstimate`;
    - phase increments, :class:`IncrementEstimate` (``IncrementSubtract``):
      :class:`IncrementSlopeEstimate` and its :class:`DshIncrementEstimate`;
    - the beat lineshape, :class:`DshLorentzianEstimate`.

    Scalars are floats (SISO) or ``(C,)`` arrays (MIMO); arrays are host
    NumPy arrays.

    Attributes
    ----------
    value : float or numpy.ndarray
        Linewidth in Hz: the Lorentzian (white-FM) linewidth, except for
        ``BetaSeparation`` (the β-separation FWHM) and ``DshLorentzian``
        (the deep-width estimate).
    method : object
        The method object that produced the estimate.
    """

    value: float | np.ndarray
    method: Any


@dataclass(frozen=True, eq=False)
class FmNoiseFitEstimate(LinewidthEstimate):
    """A linewidth from a fit to the FM-noise PSD (``BetaSeparation``,
    ``DshFmFit``), the fields both give.

    The constructor takes the same fields.

    Attributes
    ----------
    value : float or numpy.ndarray
        Linewidth in Hz, as in :class:`LinewidthEstimate`.
    method : object
        The method object that produced the estimate.
    frequencies : numpy.ndarray
        Frequency axis of ``fm_psd`` in Hz.
    fm_psd : numpy.ndarray
        FM-noise PSD :math:`S_f` in Hz²/Hz (for ``DshFmFit`` the laser PSD,
        the beat PSD divided by the interferometer response, NaN at masked
        bins).
    fm_psd_model : numpy.ndarray
        The fitted model in the units of ``fm_psd``: white FM, flicker FM and
        the detection-noise term.
    used : numpy.ndarray
        Boolean mask of the bins the white-FM fit ran over, spurs removed.
    band : tuple of float
        ``BetaSeparation``: the ``(f_min, f_max)`` fence applied.
        ``DshFmFit``: the frequency extent of ``used``.
    num_segments : int
        Welch segment count ``K`` behind every bin.
    flicker : float or numpy.ndarray
        Fitted flicker-FM coefficient ``h_-1`` in Hz² (``S_f = h_-1/f``).
    angle_noise_var : float or numpy.ndarray
        Fitted angle-noise variance per sample in rad².
    """

    frequencies: np.ndarray
    fm_psd: np.ndarray
    fm_psd_model: np.ndarray
    used: np.ndarray
    band: tuple[float, float]
    num_segments: int
    flicker: float | np.ndarray
    angle_noise_var: float | np.ndarray


@dataclass(frozen=True, eq=False)
class BetaSeparationEstimate(FmNoiseFitEstimate):
    """Result of ``BetaSeparation``.

    The constructor takes the same fields: those of
    :class:`FmNoiseFitEstimate`, then the attributes below.

    Attributes
    ----------
    linewidth_floor : float or numpy.ndarray
        The white-FM (Lorentzian) linewidth ``π·S_white`` of the fitted
        model, in Hz.
    area_hz2 : float or numpy.ndarray
        Integrated FM-noise area above the β-line, Hz².
    beta_line : numpy.ndarray
        The β-line ``8 ln2 f / π²``.
    above : numpy.ndarray
        Boolean mask of the integrated bins within ``band``: where
        ``fm_psd_model`` exceeds the β-line, or ``fm_psd`` does at bins
        beyond the fit's spur screen; in general a union of disjoint
        intervals.
    """

    linewidth_floor: float | np.ndarray
    area_hz2: float | np.ndarray
    beta_line: np.ndarray
    above: np.ndarray


@dataclass(frozen=True, eq=False)
class DshFmFitEstimate(FmNoiseFitEstimate):
    """Result of ``DshFmFit``.

    The constructor takes the same fields: those of
    :class:`FmNoiseFitEstimate`, then the attributes below.

    Attributes
    ----------
    valid : numpy.ndarray
        Bins away from the interferometer notches and, for a real
        (single-photodiode) beat, within the beat's support,
        ``f <= min(f_shift, f_s/2 - f_shift)``; ``fm_psd`` and
        ``fm_psd_model`` are NaN elsewhere.
    f_shift : float or numpy.ndarray
        The removed beat carrier in Hz.
    """

    valid: np.ndarray
    f_shift: float | np.ndarray


@dataclass(frozen=True, eq=False)
class IncrementEstimate(LinewidthEstimate):
    """A linewidth from the variance of phase increments, the result of
    ``IncrementSubtract`` and the fields every increment method gives.

    The constructor takes the same fields.

    Attributes
    ----------
    value : float or numpy.ndarray
        Linewidth in Hz, as in :class:`LinewidthEstimate`.
    method : object
        The method object that produced the estimate.
    lag_s : numpy.ndarray
        Increment lags in seconds, ``(n_lags,)``.
    increment_var : numpy.ndarray
        Measured increment variances in rad², ``(C, n_lags)``.
    angle_noise_var : float or numpy.ndarray
        Additive-noise term in rad²: the fitted intercept (``IncrementSlope``,
        ``DshIncrement``) or the subtracted term (``IncrementSubtract``).
    delta_phi_var : float or numpy.ndarray
        ``IncrementSlope``, ``IncrementSubtract``: lag-1 increment variance in
        rad².  ``DshIncrement``: total variance of the differential phase.
    """

    lag_s: np.ndarray
    increment_var: np.ndarray
    angle_noise_var: float | np.ndarray
    delta_phi_var: float | np.ndarray


@dataclass(frozen=True, eq=False)
class IncrementSlopeEstimate(IncrementEstimate):
    """Result of ``IncrementSlope``: a line through the increment variances.

    The constructor takes the same fields: those of
    :class:`IncrementEstimate`, then the attributes below.

    Attributes
    ----------
    slope, intercept : numpy.ndarray
        The fit ``increment_var = slope * lag_s + intercept`` per channel,
        rad²/s and rad².
    """

    slope: np.ndarray
    intercept: np.ndarray


@dataclass(frozen=True, eq=False)
class DshIncrementEstimate(IncrementSlopeEstimate):
    """Result of ``DshIncrement``: the increment fit on a self-heterodyne
    beat.

    The constructor takes the same fields: those of
    :class:`IncrementSlopeEstimate`, then the attributes below.

    Attributes
    ----------
    lags : numpy.ndarray
        Increment lags in samples.
    f_shift : float or numpy.ndarray
        The removed beat carrier in Hz.
    """

    lags: np.ndarray
    f_shift: float | np.ndarray


@dataclass(frozen=True, eq=False)
class DshLorentzianEstimate(LinewidthEstimate):
    """Result of ``DshLorentzian``: widths of the self-heterodyne beat.

    The constructor takes the same fields.

    Attributes
    ----------
    value : float or numpy.ndarray
        The deep-width linewidth estimate in Hz.
    method : object
        The method object that produced the estimate.
    frequencies : numpy.ndarray
        Frequency axis of ``psd`` in Hz, two-sided.
    psd : numpy.ndarray
        The beat PSD, ``(nfreq,)`` or ``(C, nfreq)``.
    peak_frequency : float or numpy.ndarray
        Beat peak frequency in Hz.
    linewidth_3db : float or numpy.ndarray
        Half-power width / 2 (the effective linewidth).
    lineshape_ratio : float or numpy.ndarray
        ``W_L/W_3`` (about 9.95 Lorentzian, 2.6 Gaussian at -20 dB).
    coherence_factor : float or numpy.ndarray
        ``τ_d/τ_c``.
    """

    frequencies: np.ndarray
    psd: np.ndarray
    peak_frequency: float | np.ndarray
    linewidth_3db: float | np.ndarray
    lineshape_ratio: float | np.ndarray
    coherence_factor: float | np.ndarray


def _check_nperseg(nperseg: int | None) -> None:
    if nperseg is not None and (int(nperseg) != nperseg or nperseg < 2):
        raise ValueError(f"nperseg must be an integer >= 2, got {nperseg}.")


def _check_fence(f_min: float | None, f_max: float | None) -> None:
    if f_min is not None and f_max is not None and not f_min < f_max:
        raise ValueError(f"f_min={f_min} must be below f_max={f_max}.")
