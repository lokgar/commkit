"""The result of :func:`estimate_linewidth`: :class:`LinewidthEstimate`.

Also the parameter checks that the phase-trajectory and the beat method
objects share.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "LinewidthEstimate",
]


# -----------------------------------------------------------------------------
# Result
# -----------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class LinewidthEstimate:
    """Result of :func:`estimate_linewidth`.

    ``value`` and ``method`` are always set; the other fields belong to one
    or more methods and are ``None`` for the rest.  Scalars are floats (SISO)
    or ``(C,)`` arrays (MIMO); arrays are host NumPy arrays.

    Attributes
    ----------
    value : float or numpy.ndarray
        Linewidth in Hz: the Lorentzian (white-FM) linewidth, except for
        ``BetaSeparation`` (the β-separation FWHM) and ``DshLorentzian``
        (the deep-width estimate).
    method : object
        The method object that produced the estimate.
    f : numpy.ndarray or None
        Frequency axis in Hz: of ``S_f`` (``BetaSeparation``, ``DshFmPsd``)
        or of ``psd`` (``DshLorentzian``, two-sided).
    S_f : numpy.ndarray or None
        FM-noise PSD in Hz²/Hz (``BetaSeparation``; for ``DshFmPsd`` the
        laser PSD, the beat PSD divided by the interferometer response, NaN at
        masked bins).
    S_f_model : numpy.ndarray or None
        ``DshFmPsd``: the fitted model in the units of ``S_f`` - white FM,
        flicker FM and the detection-noise term.
    used : numpy.ndarray or None
        Boolean mask of the bins the estimate ran over: the white-FM floor
        median (``BetaSeparation``) or the fit, spurs removed (``DshFmPsd``).
    band : tuple of float or None
        ``BetaSeparation``: the ``(f_min, f_max)`` fence applied.
        ``DshFmPsd``: the frequency extent of ``used``.
    n_segments : int or None
        Welch segment count ``K`` behind every bin (``BetaSeparation``,
        ``DshFmPsd``).
    flicker : float or numpy.ndarray or None
        ``DshFmPsd``: fitted flicker-FM coefficient ``h_-1`` in Hz²
        (``S_f = h_-1/f``).
    linewidth_floor : float or numpy.ndarray or None
        ``BetaSeparation``: the white-FM floor cross-check ``π·median(S_f)``
        in Hz.
    area_hz2 : float or numpy.ndarray or None
        ``BetaSeparation``: integrated FM-noise area above the β-line, Hz².
    beta_line : numpy.ndarray or None
        ``BetaSeparation``: the β-line ``8 ln2 f / π²``.
    above : numpy.ndarray or None
        ``BetaSeparation``: boolean mask of the integrated bins
        (``S_f > β`` within ``band``), in general a union of disjoint
        intervals.
    valid : numpy.ndarray or None
        ``DshFmPsd``: bins away from the interferometer notches.
    f_shift : float or numpy.ndarray or None
        ``DshFmPsd``, ``DshIncrement``: the removed beat carrier in Hz.
    lags : numpy.ndarray or None
        ``DshIncrement``: increment lags in samples.
    lag_s : numpy.ndarray or None
        Increment lags in seconds, ``(n_lags,)`` (increment methods).
    var : numpy.ndarray or None
        Measured increment variances in rad², ``(C, n_lags)`` (increment
        methods).
    slope, intercept : numpy.ndarray or None
        The fit ``var = slope * lag_s + intercept`` per channel, rad²/s and
        rad² (``IncrementSlope``, ``DshIncrement``).
    awgn_var : float or numpy.ndarray or None
        Additive-noise term in rad²: the fitted intercept (``IncrementSlope``,
        ``DshIncrement``), the subtracted term (``IncrementSubtract``) or the
        fitted angle-noise variance per sample (``DshFmPsd``).
    dphi_var : float or numpy.ndarray or None
        ``IncrementSlope``, ``IncrementSubtract``: lag-1 increment variance in
        rad².  ``DshIncrement``: total variance of the differential phase.
    linewidth_3db : float or numpy.ndarray or None
        ``DshLorentzian``: half-power width / 2 (the effective linewidth).
    lineshape_ratio : float or numpy.ndarray or None
        ``DshLorentzian``: ``W_L/W_3`` (about 9.95 Lorentzian, 2.6 Gaussian at
        -20 dB).
    coherence_factor : float or numpy.ndarray or None
        ``DshLorentzian``: ``τ_d/τ_c``.
    psd : numpy.ndarray or None
        ``DshLorentzian``: the beat PSD, ``(nfreq,)`` or ``(C, nfreq)``.
    f_peak : float or numpy.ndarray or None
        ``DshLorentzian``: beat peak frequency in Hz.
    """

    value: float | np.ndarray
    method: Any
    f: np.ndarray | None = None
    S_f: np.ndarray | None = None
    S_f_model: np.ndarray | None = None
    used: np.ndarray | None = None
    band: tuple[float, float] | None = None
    n_segments: int | None = None
    flicker: float | np.ndarray | None = None
    linewidth_floor: float | np.ndarray | None = None
    area_hz2: float | np.ndarray | None = None
    beta_line: np.ndarray | None = None
    above: np.ndarray | None = None
    valid: np.ndarray | None = None
    f_shift: float | np.ndarray | None = None
    lags: np.ndarray | None = None
    lag_s: np.ndarray | None = None
    var: np.ndarray | None = None
    slope: np.ndarray | None = None
    intercept: np.ndarray | None = None
    awgn_var: float | np.ndarray | None = None
    dphi_var: float | np.ndarray | None = None
    linewidth_3db: float | np.ndarray | None = None
    lineshape_ratio: float | np.ndarray | None = None
    coherence_factor: float | np.ndarray | None = None
    psd: np.ndarray | None = None
    f_peak: float | np.ndarray | None = None


def _check_nperseg(nperseg: int | None) -> None:
    if nperseg is not None and (int(nperseg) != nperseg or nperseg < 2):
        raise ValueError(f"nperseg must be an integer >= 2, got {nperseg}.")


def _check_fence(f_min: float | None, f_max: float | None) -> None:
    if f_min is not None and f_max is not None and not f_min < f_max:
        raise ValueError(f"f_min={f_min} must be below f_max={f_max}.")
