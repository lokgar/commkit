r"""Laser and carrier linewidth: ``estimate_linewidth`` and its methods.

The method object chooses the estimator and the kind of record it reads:

* phase trajectories (radians, real-valued) - ``IncrementSlope``,
  ``IncrementSubtract`` and ``BetaSeparation``, e.g. the output of
  ``carrier_phase_trajectory`` or ``separate_drift_phase_noise``;
* delayed self-heterodyne / self-homodyne beat records (real photocurrent or
  complex IQ, array or Signal) - ``DshFmPsd``, ``DshIncrement`` and
  ``DshLorentzian`` (see :mod:`commkit.analysis.interferometry`).

Every method returns a :class:`LinewidthEstimate`: the linewidth in Hz plus
that method's diagnostics, ready for the ``plotting.analysis`` plots.  The
values are host floats (SISO) or ``(C,)`` arrays (MIMO), like the metrics:
the fits run host-side on plot-sized reductions.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from .._array import as_2d, broadcast_channels, restore_1d, to_report_scalar
from ..backend import ArrayType, dispatch, to_device
from ..core._signal_adapter import adapt_signal
from ..core.signal import Signal
from ..logger import logger
from ..smoothing import moving_average
from ..spectral import welch_psd
from ._common import (
    _BETA_SLOPE,
    _FWHM_FROM_AREA,
    _floor_levels,
    _increment_variance_fit,
    _resolve_nperseg,
    _welch_floor_bias,
)
from .fm_noise import dsh_fm_noise_psd, fm_noise_psd
from .interferometry import _analytic_beat, dsh_phase

__all__ = [
    "BetaSeparation",
    "DshFmPsd",
    "DshIncrement",
    "DshLorentzian",
    "IncrementSlope",
    "IncrementSubtract",
    "LinewidthEstimate",
    "estimate_linewidth",
]


# -----------------------------------------------------------------------------
# Result
# -----------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class LinewidthEstimate:
    """Result of :func:`estimate_linewidth`.

    ``value`` and ``method`` are always set; the other fields belong to one
    or more methods and are ``None`` for the rest.  Scalars are floats (SISO)
    or ``(C,)`` arrays (MIMO); arrays are host NumPy.

    Attributes
    ----------
    value : float or np.ndarray
        Linewidth in Hz: the Lorentzian (white-FM) linewidth, except for
        ``BetaSeparation`` (the β-separation FWHM) and ``DshLorentzian``
        (the deep-width estimate).
    method : object
        The method object that produced the estimate.
    f : np.ndarray or None
        Frequency axis in Hz: of ``S_f`` (``BetaSeparation``, ``DshFmPsd``)
        or of ``psd`` (``DshLorentzian``, two-sided).
    S_f : np.ndarray or None
        FM-noise PSD in Hz²/Hz (``BetaSeparation``; for ``DshFmPsd`` the
        deconvolved laser PSD, NaN at masked bins).
    used : np.ndarray or None
        Boolean mask of the bins the white-FM floor median ran over
        (``BetaSeparation``, ``DshFmPsd``).
    band : tuple of float or None
        ``BetaSeparation``: the ``(f_min, f_max)`` fence applied.
        ``DshFmPsd``: the frequency extent of ``used``.
    n_segments : int or None
        Welch segment count ``K`` behind every bin (``BetaSeparation``,
        ``DshFmPsd``).
    linewidth_floor : float or np.ndarray or None
        ``BetaSeparation``: the white-FM floor cross-check ``π·median(S_f)``
        in Hz.
    area_hz2 : float or np.ndarray or None
        ``BetaSeparation``: integrated FM-noise area above the β-line, Hz².
    beta_line : np.ndarray or None
        ``BetaSeparation``: the β-line ``8 ln2 f / π²``.
    above : np.ndarray or None
        ``BetaSeparation``: boolean mask of the integrated bins
        (``S_f > β`` within ``band``), in general a union of disjoint
        intervals.
    valid : np.ndarray or None
        ``DshFmPsd``: bins away from the interferometer notches.
    f_shift : float or np.ndarray or None
        ``DshFmPsd``, ``DshIncrement``: the removed beat carrier in Hz.
    lags : np.ndarray or None
        ``DshIncrement``: increment lags in samples.
    lag_s : np.ndarray or None
        Increment lags in seconds, ``(n_lags,)`` (increment methods).
    var : np.ndarray or None
        Measured increment variances in rad², ``(C, n_lags)`` (increment
        methods).
    slope, intercept : np.ndarray or None
        The fit ``var = slope * lag_s + intercept`` per channel, rad²/s and
        rad² (``IncrementSlope``, ``DshIncrement``).
    awgn_var : float or np.ndarray or None
        Additive-noise term in rad²: the fitted intercept (``IncrementSlope``,
        ``DshIncrement``) or the subtracted term (``IncrementSubtract``).
    dphi_var : float or np.ndarray or None
        ``IncrementSlope``, ``IncrementSubtract``: lag-1 increment variance in
        rad².  ``DshIncrement``: total variance of the differential phase.
    linewidth_3db : float or np.ndarray or None
        ``DshLorentzian``: half-power width / 2 (the effective linewidth).
    lineshape_ratio : float or np.ndarray or None
        ``DshLorentzian``: ``W_L/W_3`` (about 9.95 Lorentzian, 2.6 Gaussian at
        -20 dB).
    coherence_factor : float or np.ndarray or None
        ``DshLorentzian``: ``τ_d/τ_c``.
    psd : np.ndarray or None
        ``DshLorentzian``: the beat PSD, ``(nfreq,)`` or ``(C, nfreq)``.
    f_peak : float or np.ndarray or None
        ``DshLorentzian``: beat peak frequency in Hz.
    """

    value: float | np.ndarray
    method: Any
    f: np.ndarray | None = None
    S_f: np.ndarray | None = None
    used: np.ndarray | None = None
    band: tuple[float, float] | None = None
    n_segments: int | None = None
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


# -----------------------------------------------------------------------------
# Methods on phase trajectories
# -----------------------------------------------------------------------------


def _check_edge_trim(edge_trim: int) -> None:
    if int(edge_trim) != edge_trim or edge_trim < 0:
        raise ValueError(f"edge_trim must be a non-negative integer, got {edge_trim}.")


def _check_nperseg(nperseg: int | None) -> None:
    if nperseg is not None and (int(nperseg) != nperseg or nperseg < 2):
        raise ValueError(f"nperseg must be an integer >= 2, got {nperseg}.")


def _check_fence(f_min: float | None, f_max: float | None) -> None:
    if f_min is not None and f_max is not None and not f_min < f_max:
        raise ValueError(f"f_min={f_min} must be below f_max={f_max}.")


@dataclass(frozen=True)
class IncrementSlope:
    r"""Wiener linewidth from the lag-slope of the phase-increment variance.

    For a Wiener phase plus AWGN angle noise, the variance of the lag-``k``
    increment ``Δφ_k = φ(n) - φ(n-k)`` is **linear in ``k``**:

        Var(Δφ_k) = slope · k·T + intercept,
        slope = 2π·Δν,  intercept = 2σ_φ²,

    because the random-walk variance accumulates with ``k`` while the
    uncorrelated angle noise contributes a fixed ``2σ_φ²``.  A least-squares
    fit over ``lags`` gives ``Δν = slope/(2π)``: the additive noise (AWGN
    *and* residual white error from imperfect equalization) cancels into the
    intercept, so **no noise estimate is needed**.  The recommended method.

    Parameters
    ----------
    lags : tuple of int, default (1, 2, 3, 4, 5)
        Increment lags ``k`` in samples; at least two distinct lags ``>= 1``
        (smaller entries are ignored).
    edge_trim : int, default 0
        Samples discarded from each end before differencing (filter edge
        transients of ``separate_drift_phase_noise``).

    Notes
    -----
    **Limitations.**

    * The linearity of ``Var(Δφ_k)`` in ``k`` holds for **white-FM (Wiener)**
      phase noise only.  Flicker (1/f) FM noise makes the variance grow
      *faster* than linear, biasing the fitted slope - and hence Δν - high;
      what is reported is then an *effective* linewidth at the lag timescale,
      not the intrinsic Lorentzian linewidth.
    * The variance subtracts the per-lag mean, so a **constant** residual
      frequency offset does not bias the fit, but a frequency *ramp*
      (nonlinear drift) adds a ``k²`` term.  Detrend first
      (``separate_drift_phase_noise``) and keep the largest lag ``k·T`` well
      inside the drift timescale.
    * Larger lags raise the phase-noise term above the AWGN intercept but
      admit more drift/flicker contamination; the default 1-5 symbol lags
      suit multi-MHz linewidths at GBaud rates.  For sub-100-kHz linewidths at
      high symbol rates the per-lag walk variance may sit orders of magnitude
      below ``2σ_φ²`` - prefer ``BetaSeparation`` there.
    """

    lags: tuple[int, ...] = (1, 2, 3, 4, 5)
    edge_trim: int = 0

    def __post_init__(self) -> None:
        lags = tuple(sorted({int(k) for k in self.lags if k >= 1}))
        if len(lags) < 2:
            raise ValueError("IncrementSlope needs at least two distinct lags >= 1.")
        _check_edge_trim(self.edge_trim)
        object.__setattr__(self, "lags", lags)


@dataclass(frozen=True, eq=False)
class IncrementSubtract:
    r"""Lag-1 phase-increment variance minus a known AWGN term.

    ``Δν = (Var(Δφ_1) - σ²) / (2π·T)``.  With unit-power symbols the angle
    noise of the increment is ``σ_n²`` (complex noise variance): the flat
    correction subtracts ``σ_n²`` (exact for PSK, *under*-corrects QAM),
    while passing ``reference`` applies the amplitude-aware
    ``σ_n²·E[1/|d|²]`` (rigorous for QAM, since inner-ring symbols carry
    larger angle noise).

    Needs the *additive*-noise variance only: ``metrics.snr`` reports the
    total residual (noise, phase noise and ISI) and over-subtracts.  Prefer
    ``IncrementSlope``, which needs no noise estimate.

    Parameters
    ----------
    noise_var : float or array_like, optional
        Complex noise variance ``σ_n²`` of unit-power symbols, one value or
        one per channel ``(C,)``.  For an SNR in dB: ``10 ** (-snr_db / 10)``.
        ``None`` subtracts nothing.
    reference : array_like, optional
        Transmitted symbols aligned with the phase, ``(N,)`` or ``(C, N)``,
        for the amplitude-aware correction.
    edge_trim : int, default 0
        Samples discarded from each end before differencing.
    """

    noise_var: float | np.ndarray | None = None
    reference: np.ndarray | None = None
    edge_trim: int = 0

    def __post_init__(self) -> None:
        if self.noise_var is not None:
            nv = np.atleast_1d(np.asarray(to_device(self.noise_var, "cpu"), float))
            if nv.ndim != 1 or np.any(nv < 0):
                raise ValueError(
                    "noise_var must be a non-negative scalar or (C,) array, "
                    f"got {self.noise_var!r}."
                )
            nv.setflags(write=False)
            object.__setattr__(self, "noise_var", nv)
        if self.reference is not None:
            ref = np.array(to_device(self.reference, "cpu"))
            if ref.ndim not in (1, 2):
                raise ValueError(
                    f"reference must have shape (N,) or (C, N), got {ref.shape}."
                )
            ref.setflags(write=False)
            object.__setattr__(self, "reference", ref)
        _check_edge_trim(self.edge_trim)


@dataclass(frozen=True)
class BetaSeparation:
    r"""Linewidth by the Di Domenico β-separation line.

    Integrates the frequency-noise PSD ``S_f(f)`` (``fm_noise_psd``) only
    over the **region where it lies above** the β-separation line
    ``S_f = (8 ln2 / π²)·f`` - in general a union of disjoint intervals; the
    estimate's ``above`` mask is the exact region used.  The FWHM linewidth
    is ``sqrt(8 ln2 · A)`` for the integrated area ``A`` (Hz²).  The
    ``[f_min, f_max]`` window is only an outer *fence* on that region: it
    excludes the unresolved DC bin and - with an appropriate ``f_max`` - the
    high-frequency AWGN ``f²`` tail (which eventually climbs back above the
    line and would otherwise be integrated as fake linewidth).

    A white-FM-floor cross-check ``linewidth_floor = π·median(S_f)`` is also
    returned.  Without a fence the floor's median band is auto-detected as
    the PSD's minimum-level (plateau) region - octave-band-median floor, all
    bins within 3x of it - so a low-frequency drift/flicker rise and the AWGN
    ``f²`` tail are excluded without manual fencing.  An explicit ``f_min``
    or ``f_max`` switches the floor back to a literal-band median (the β-area
    integral always uses the literal fence).  The floor is corrected for the
    χ²-median bias of Welch bins (a raw median reads ``≈ 1 - 1/(3K)`` below
    the true level for ``K`` averaged segments).

    Parameters
    ----------
    nperseg : int, optional
        Welch segment length (see ``fm_noise_psd``).
    f_min : float, optional
        Lower fence in Hz (drops the residual-FOE DC region).  Defaults to
        the first non-zero Welch bin (``sampling_rate/nperseg``).  This is a
        *resolution* floor, not the canonical ``1/T_obs`` of the method:
        FM-noise area between ``1/T_obs`` and the first bin is unresolved and
        excluded, so for drift/flicker-dominated sources the result depends
        on ``nperseg``.
    f_max : float, optional
        Upper fence in Hz.  **Set it below the AWGN ``f²`` knee**; defaults
        to the Nyquist bin.

    Notes
    -----
    **Limitations.**

    * The β-separation FWHM is an *approximation* (about 10 % for lineshapes
      dominated by slow FM noise, exact for pure white FM - the line is
      constructed so a flat ``S_f = Δν/π`` integrates back to ``Δν``).
    * The result is **observation-time dependent** for flicker/drift-dominated
      sources: lowering ``f_min`` (longer capture) adds low-frequency area and
      grows ``Δν``.  Always quote ``f_min`` with the number.
    * The AWGN ``f²`` tail crosses back above the β-line: set ``f_max`` below
      the knee where the plateau ``Δν/π`` meets the tail ``2σ_φ²T·f²``, i.e.
      ``f_knee = (Δν/(2π σ_φ² T))^{1/2}``.  Inspect it with
      ``plotting.plot_frequency_noise_psd``.
    * ``linewidth_floor`` is the more robust estimate when a clean white-FM
      plateau exists in the band; the two should agree within tens of
      percent, otherwise inspect the PSD.
    """

    nperseg: int | None = None
    f_min: float | None = None
    f_max: float | None = None

    def __post_init__(self) -> None:
        _check_nperseg(self.nperseg)
        _check_fence(self.f_min, self.f_max)


# -----------------------------------------------------------------------------
# Methods on delayed self-heterodyne beat records
# -----------------------------------------------------------------------------


def _check_delay(delay: float) -> None:
    if not float(delay) > 0.0:
        raise ValueError(f"delay={delay} must be positive (seconds).")


@dataclass(frozen=True)
class DshFmPsd:
    r"""White-FM floor of the deconvolved DSH FM-noise PSD.

    ``dsh_phase`` -> ``dsh_fm_noise_psd`` -> ``Δν = π·median(S_f,laser)``
    over the usable band.  Works for **any** delay (coherent or incoherent
    regime); the notch structure, not the regime, sets the usable band.

    Parameters
    ----------
    delay : float
        Interferometer differential delay ``τ_d`` in seconds (≈ 4.9 µs per km
        of SMF).
    f_shift : float, optional
        Known AOM/beat carrier in Hz; estimated if None (see ``dsh_phase``).
    nperseg : int, optional
        Welch segment length.
    f_min, f_max : float, optional
        Manual analysis fence.  When **both** are None the plateau is
        **auto-detected**: the white-FM floor is located as the minimum of
        octave-band medians over all valid bins and the median runs over
        every bin within 3x that floor - spanning as many interferometer
        lobes as the detection-noise knee allows, and excluding a rising
        low-frequency (drift/flicker) region.  Passing either bound switches
        to the literal fence (``f_min`` -> 0, ``f_max`` -> the first notch
        ``1/τ_d`` when the other is omitted), with the median over *all*
        valid bins inside.
    notch_guard : float, default 0.1
        Notch mask threshold (see ``dsh_fm_noise_psd``).

    Notes
    -----
    The estimate reports the white-FM *floor*.  For **real** captures the
    eligible band is additionally capped at the receiver's FM detection
    bandwidth ``min(f_shift, f_s/2 - f_shift)`` - beyond it the beat carries
    no sidebands and the deconvolved PSD reads fake-low.  It falls back to the
    first lobe (with a warning) when no plateau is found.  If the *entire*
    usable band is 1/f-dominated (very long ``τ_d``, quiet laser), the floor
    is the lowest resolved noise level - inspect the PSD before quoting it as
    Δν.  The χ²-median bias of the Welch bins is divided back out (keep the
    record ≳ 5·nperseg so ``K ≳ 10``); the plotted log-binned median curve is
    *not* corrected, so at small ``K`` the ``Δν/π`` guide sits visibly above
    it.
    """

    delay: float
    f_shift: float | None = None
    nperseg: int | None = None
    f_min: float | None = None
    f_max: float | None = None
    notch_guard: float = 0.1

    def __post_init__(self) -> None:
        _check_delay(self.delay)
        _check_nperseg(self.nperseg)
        _check_fence(self.f_min, self.f_max)


@dataclass(frozen=True)
class DshIncrement:
    r"""Wiener linewidth from the DSH differential-phase increment variance.

    For lag ``a = ℓ/f_s ≤ τ_d`` the two Wiener increments of the measured
    differential phase are disjoint, so

        Var[Δφ(t) - Δφ(t-a)] = 4π·Δν·a + 2σ_w²,

    and a straight-line fit against ``a`` gives ``Δν = slope/(4π)``, with the
    beat angle noise cancelling into the intercept - no SNR estimate needed
    (as for ``IncrementSlope``).  White-FM (Wiener) assumption: flicker bends
    ``Var(a)`` super-linear and biases Δν high.  Slow drift is harmless up to
    a linear frequency chirp.

    Parameters
    ----------
    delay : float
        Interferometer differential delay ``τ_d`` in seconds.
    f_shift : float, optional
        Known AOM/beat carrier in Hz; estimated if None.
    lags : tuple of int, optional
        Increment lags in *samples*.  Default: five lags up to
        ``a_max = min(0.5·τ_d·f_s, N/2000)`` - inside the disjoint window
        ``ℓ ≤ τ_d·f_s`` *and* small enough that the variance estimator keeps
        many independent averages when the delay is long (decoherence
        spools).
    """

    delay: float
    f_shift: float | None = None
    lags: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        _check_delay(self.delay)
        if self.lags is not None:
            lags = tuple(int(lag) for lag in self.lags)
            if any(lag < 1 for lag in lags):
                raise ValueError("lags must be positive sample counts.")
            if len(set(lags)) < 2:
                raise ValueError("DshIncrement needs at least two distinct lags.")
            object.__setattr__(self, "lags", lags)


@dataclass(frozen=True)
class DshLorentzian:
    r"""Spectral width of the DSH beat line (the textbook method).

    In the incoherent regime (``τ_d ≫ τ_c = 1/(πΔν)``) the beat line is
    Lorentzian with FWHM ``2Δν``.  The width is measured ``level_db`` below
    the peak and converted via ``Δν = W_L / (2·√(10^{L/10} - 1))``
    (``W₂₀/(2√99)`` for the customary -20 dB width, which suppresses the
    Gaussian 1/f-noise core that contaminates the -3 dB width).

    Parameters
    ----------
    delay : float
        Interferometer differential delay ``τ_d`` in seconds (for the
        coherence check).
    nperseg : int, optional
        Welch segment length of the beat PSD.
    level_db : float, default 20.0
        Depth below the peak at which the width is measured.

    Notes
    -----
    Requires ``coherence_factor ≳ 6``: below that the spectrum develops a
    coherent carrier spike plus fringes at ``1/τ_d`` spacing and the width no
    longer reads ``2Δν`` (a warning is logged).  It also needs the
    ``-level_db`` contour above the noise floor (peak dynamic range ≳
    ``level_db`` + 10 dB) and ``≳ 10`` Welch bins across the line.  With a
    homodyne IQ capture, calibrate photodiode DC offsets out first: a DC spur
    mid-line hijacks the peak and narrows the widths.
    """

    delay: float
    nperseg: int | None = None
    level_db: float = 20.0

    def __post_init__(self) -> None:
        _check_delay(self.delay)
        _check_nperseg(self.nperseg)
        if not float(self.level_db) > 0.0:
            raise ValueError(f"level_db must be positive, got {self.level_db}.")


# -----------------------------------------------------------------------------
# Phase-trajectory estimators
# -----------------------------------------------------------------------------


def _trimmed(p: ArrayType, edge_trim: int, xp: Any) -> tuple[ArrayType, slice, int]:
    """``(C, N)`` float64 phase with ``edge_trim`` samples dropped per end."""
    p2, _ = as_2d(p, name="phase")
    n_full = p2.shape[-1]
    sl = slice(edge_trim, n_full - edge_trim) if edge_trim > 0 else slice(None)
    return p2[:, sl].astype(xp.float64), sl, n_full


def _increment_slope(
    p: ArrayType, method: IncrementSlope, fs: float
) -> LinewidthEstimate:
    _, xp, _ = dispatch(p)
    p2, _, _ = _trimmed(p, method.edge_trim, xp)
    var1 = xp.var(p2[:, 1:] - p2[:, :-1], axis=-1)
    # Shared with DshIncrement: the fit is in seconds, so ``slope`` is
    # rad²/s and only the constant differs.
    slope, intercept, var_k_cpu, lag_sec = _increment_variance_fit(
        p2, np.asarray(method.lags), 1.0 / fs, xp
    )
    return LinewidthEstimate(
        value=to_report_scalar(np.maximum(slope, 0.0) / (2.0 * np.pi)),
        method=method,
        dphi_var=to_report_scalar(to_device(var1, "cpu")),
        awgn_var=to_report_scalar(intercept),
        lag_s=lag_sec,
        var=var_k_cpu.T,
        slope=slope,
        intercept=intercept,
    )


def _increment_subtract(
    p: ArrayType, method: IncrementSubtract, fs: float
) -> LinewidthEstimate:
    _, xp, _ = dispatch(p)
    p2, sl, n_full = _trimmed(p, method.edge_trim, xp)
    c = p2.shape[0]
    t_sym = 1.0 / fs
    var1 = xp.var(p2[:, 1:] - p2[:, :-1], axis=-1)

    # The noise variance is a host value: resolve it host-side and upload
    # once, so the branch below needs no device sync.
    nv = None if method.noise_var is None else np.atleast_1d(method.noise_var)
    if nv is None:
        sigma_host = np.zeros(c, dtype=np.float64)
    elif nv.size == 1:
        sigma_host = np.full(c, float(nv[0]))
    elif nv.size == c:
        sigma_host = np.asarray(nv, dtype=np.float64)
    else:
        raise ValueError(f"noise_var has {nv.size} values for {c} channels.")
    sigma_n2 = xp.asarray(sigma_host)

    if method.reference is not None and bool(np.any(sigma_host)):
        from ..math import normalize

        d2 = broadcast_channels(xp.asarray(method.reference), c, xp, name="reference")
        d2 = d2[:, :n_full][:, sl]
        d2 = normalize(d2, mode="average_power", axis=-1)
        inv = 1.0 / xp.maximum(xp.abs(d2) ** 2, 1e-12)
        pair_mean = 0.5 * (inv[:, 1:] + inv[:, :-1])
        awgn_var = sigma_n2 * xp.mean(pair_mean, axis=-1)
    else:
        awgn_var = sigma_n2.copy()
    linewidth = xp.maximum(var1 - awgn_var, 0.0) / (2.0 * np.pi * t_sym)

    var1_cpu = to_device(var1, "cpu")
    return LinewidthEstimate(
        value=to_report_scalar(to_device(linewidth, "cpu")),
        method=method,
        dphi_var=to_report_scalar(var1_cpu),
        awgn_var=to_report_scalar(to_device(awgn_var, "cpu")),
        lag_s=np.array([t_sym]),
        var=np.atleast_1d(var1_cpu)[:, None],
    )


def _beta_separation(
    phi: ArrayType, method: BetaSeparation, fs: float
) -> LinewidthEstimate:
    n_phi = phi.shape[-1]
    f, S_f = fm_noise_psd(phi, symbol_rate=fs, nperseg=method.nperseg)
    _, xp, _ = dispatch(f)

    # Transfer the plot-sized spectrum once, up front: every fence, mask and
    # median below is host-side arithmetic on these arrays, so reading f[1] /
    # f[-1] off the device beforehand would only add two syncs.
    f_cpu = np.asarray(to_device(f, "cpu"), dtype=np.float64)
    S_cpu = np.asarray(to_device(S_f, "cpu"), dtype=np.float64)
    S2c = S_cpu[None, :] if S_cpu.ndim == 1 else S_cpu

    # One-sided Welch axis: f[0] = 0, f[1] is the first non-zero bin.
    fmin = float(f_cpu[1]) if method.f_min is None else float(method.f_min)
    fmax = float(f_cpu[-1]) if method.f_max is None else float(method.f_max)
    beta_cpu = _BETA_SLOPE * f_cpu
    band_cpu = (f_cpu >= fmin) & (f_cpu <= fmax)
    # The reported/plotted mask is rebuilt from the transferred spectrum -
    # same expression, same float64 values - instead of being transferred.
    above_cpu = band_cpu[None, :] & (S2c > beta_cpu[None, :])

    # The β-area integral itself stays on the input backend (sample-rate work),
    # with the fence rebuilt there from the host scalars.
    S2 = S_f[None, :] if S_f.ndim == 1 else S_f
    beta = _BETA_SLOPE * f
    band = (f >= fmin) & (f <= fmax)
    # Vectorized over channels: (C, nfreq) masks instead of a per-channel loop.
    above = band[None, :] & (S2 > beta[None, :])
    integrand = xp.where(above, S2, 0.0)
    area = xp.trapezoid(integrand, f, axis=-1)
    lw = xp.sqrt(_FWHM_FROM_AREA * area)

    # Pack the two (C,) metrics into one D2H transfer; the floor median runs
    # host-side so the plateau auto-detection is shared with DshFmPsd.
    lw_cpu, area_cpu = to_device(xp.stack([lw, area]), "cpu")

    base2 = np.zeros(S2c.shape, dtype=bool)
    base2[:] = band_cpu & (f_cpu > 0)
    base2 &= np.isfinite(S2c)

    # Welch bins are χ²-distributed: the median-based floor reads the true
    # level low by median(χ²_ν)/ν ≈ 1 - 1/(3K); divided back out below.  The
    # segment length is reproduced from the resolver rather than read back off
    # the device frequency axis (f[1] = fs/nperseg by construction).
    nps_used = _resolve_nperseg(n_phi - 1, method.nperseg, cap=4096)
    m_med, k_seg = _welch_floor_bias(n_phi - 1, nps_used, label="BetaSeparation")

    used2, levels = _floor_levels(
        f_cpu,
        S2c,
        base2,
        auto=(method.f_min is None and method.f_max is None),
        label="BetaSeparation",
        fallback_desc="the full band",
    )
    # A band with no eligible bin reports a zero floor rather than NaN.
    lw_floor_cpu = np.pi * np.where(np.isfinite(levels), levels, 0.0) / m_med
    used_cpu = used2[0] if S_cpu.ndim == 1 else used2
    if S_cpu.ndim == 1:
        above_cpu = above_cpu[0]

    return LinewidthEstimate(
        value=to_report_scalar(lw_cpu),
        method=method,
        linewidth_floor=to_report_scalar(lw_floor_cpu),
        n_segments=k_seg,
        area_hz2=to_report_scalar(area_cpu),
        f=f_cpu,
        S_f=S_cpu,
        beta_line=beta_cpu,
        above=above_cpu,
        used=used_cpu,
        band=(fmin, fmax),
    )


# -----------------------------------------------------------------------------
# DSH estimators
# -----------------------------------------------------------------------------


def _delay_samples(delay: float, fs: float) -> float:
    """``τ_d·f_s``; a delay below one sample raises."""
    m_samp = float(delay) * fs
    if m_samp < 1.0:
        raise ValueError(
            f"delay·sampling_rate = {m_samp:.3g} < 1 sample - the differential "
            "delay is unresolvable at this sampling rate."
        )
    return m_samp


def _lorentzian_widths(f, p, level_lin):
    """Full width of a spectral line ``level_lin`` (linear ratio) below its peak.

    Walks outward from the peak bin to the nearest below-threshold bin on each
    side and interpolates the crossing in log-power.  Returns NaN when the
    threshold is never crossed (e.g. it sits below the noise floor).

    Host-side NumPy by design: the caller hands it an ``nperseg``-sized Welch
    spectrum already brought to the host with a single transfer, and the
    crossing walk is scalar, data-dependent branching that a GPU cannot help
    with.
    """
    i_pk = int(np.argmax(p))
    thr = p[i_pk] / level_lin

    left = np.nonzero(p[:i_pk] < thr)[0]
    right = np.nonzero(p[i_pk + 1 :] < thr)[0]
    if left.size == 0 or right.size == 0:
        return np.nan

    i0 = left[-1]  # crossing between i0 and i0+1
    i1 = i_pk + 1 + right[0]  # crossing between i1-1 and i1
    l0, l1 = np.log(p[i0]), np.log(p[i0 + 1])
    f_lo = f[i0] + (f[i0 + 1] - f[i0]) * (np.log(thr) - l0) / (l1 - l0)
    r0, r1 = np.log(p[i1 - 1]), np.log(p[i1])
    f_hi = f[i1 - 1] + (f[i1] - f[i1 - 1]) * (np.log(thr) - r0) / (r1 - r0)
    return float(f_hi - f_lo)


def _dsh_fm_psd(samples: ArrayType, method: DshFmPsd, fs: float) -> LinewidthEstimate:
    td = float(method.delay)
    _delay_samples(td, fs)
    dphi, f_hat = dsh_phase(samples, sampling_rate=fs, f_shift=method.f_shift)
    psd = dsh_fm_noise_psd(
        dphi,
        sampling_rate=fs,
        delay=td,
        nperseg=method.nperseg,
        notch_guard=method.notch_guard,
    )
    # Summary layer: one transfer, host-side plateau search + median
    # (plot-sized spectra - see the package backend policy).
    f_cpu = np.asarray(to_device(psd.f, "cpu"), dtype=np.float64)
    S_cpu = np.asarray(to_device(psd.S_f, "cpu"), dtype=np.float64)
    valid_cpu = np.asarray(to_device(psd.valid, "cpu"), dtype=bool)
    S2 = S_cpu[None, :] if S_cpu.ndim == 1 else S_cpu

    # Welch bins are χ²-distributed, so every median-based floor below reads
    # the true level low by median(χ²_ν)/ν ≈ 1 - 1/(3K); the factor is
    # divided back out of the linewidth at the end.  The segment length is
    # reproduced from the resolver (f[1] = f_s/nperseg) rather than read back
    # off the device.
    nps_used = _resolve_nperseg(dphi.shape[-1] - 1, method.nperseg, cap=4096)
    m_med, k_seg = _welch_floor_bias(
        dphi.shape[-1] - 1, nps_used, label="estimate_linewidth(DshFmPsd)"
    )

    if method.f_min is None and method.f_max is None:
        base = valid_cpu & (f_cpu > 0)
        # Real (single-photodiode) captures carry FM sidebands only out to
        # the beat carrier's distance from the band edges: offsets beyond
        # min(f_shift, f_nyq - f_shift) have no physical support after the
        # analytic-signal step and read as fake-low PSD - cap the eligible
        # band there.  Complex IQ captures have no such limit (full ±Nyquist).
        x_in, xp_in, _ = dispatch(samples)
        if not xp_in.iscomplexobj(x_in):
            fh = float(np.min(np.asarray(f_hat)))
            base = base & (f_cpu <= min(fh, fs / 2.0 - fh))
        base2 = np.broadcast_to(base, S2.shape)
        # Channels where no plateau is found fall back to the first lobe.
        first_lobe = np.broadcast_to(
            valid_cpu & (f_cpu > 0) & (f_cpu <= 1.0 / td), S2.shape
        )
        used2, levels = _floor_levels(
            f_cpu,
            S2,
            base2,
            auto=True,
            label="estimate_linewidth(DshFmPsd)",
            fallback2=first_lobe,
            fallback_desc="the first-lobe band [0, 1/τ_d]",
        )
    else:
        fmin = 0.0 if method.f_min is None else float(method.f_min)
        fmax = (1.0 / td) if method.f_max is None else float(method.f_max)
        base2 = np.broadcast_to(valid_cpu & (f_cpu >= fmin) & (f_cpu <= fmax), S2.shape)
        used2, levels = _floor_levels(
            f_cpu, S2, base2, auto=False, label="estimate_linewidth(DshFmPsd)"
        )

    if not used2.any(axis=-1).all():
        raise ValueError(
            "No valid FM-PSD bins in the analysis band - widen "
            "[f_min, f_max], lower notch_guard, or increase nperseg."
        )
    lw = np.pi * levels / m_med
    f_used = f_cpu[used2.any(axis=0)]

    return LinewidthEstimate(
        value=to_report_scalar(lw),
        method=method,
        f_shift=f_hat,
        f=f_cpu,
        S_f=S_cpu,
        valid=valid_cpu,
        used=used2[0] if S_cpu.ndim == 1 else used2,
        band=(float(f_used[0]), float(f_used[-1])),
        n_segments=k_seg,
    )


def _dsh_increment(
    samples: ArrayType, method: DshIncrement, fs: float
) -> LinewidthEstimate:
    m_samp = _delay_samples(method.delay, fs)
    dphi, f_hat = dsh_phase(samples, sampling_rate=fs, f_shift=method.f_shift)
    d2, _ = as_2d(dphi, name="delta_phi")
    _, xp, _ = dispatch(d2)

    m_int = int(round(m_samp))
    if method.lags is None:
        # Largest default lag: half the delay (disjoint-increment bound),
        # further capped by the record length - the Var estimator's
        # correlation support scales with the lag, so for long decoherence
        # spools small lags give far more independent averages (measured:
        # ~3x lower spread at τ_d·f_s = 6·10⁴, N = 2·10⁶) while the AWGN
        # intercept still cancels in the fit.
        a_max = max(1.0, min(0.5 * m_samp, d2.shape[-1] / 2000.0))
        ls = np.unique(
            np.maximum(
                1, np.round(a_max * np.array([0.2, 0.4, 0.6, 0.8, 1.0])).astype(int)
            )
        )
    else:
        ls = np.unique(np.asarray(method.lags))
    if ls.size < 2:
        raise ValueError(
            f"DshIncrement needs >= 2 distinct lags (delay spans only {m_int} "
            "samples - increase the sampling rate or pass lags)."
        )
    if int(ls[-1]) > m_int:
        logger.warning(
            "estimate_linewidth: max lag %d exceeds the delay (%d samples); the "
            "Wiener increments overlap and Var(a) is no longer linear - the fit "
            "will be biased low.",
            int(ls[-1]),
            m_int,
        )

    dphi_var = xp.var(d2, axis=-1)
    # Shared with IncrementSlope: same Var-vs-lag least squares, only the
    # constant differs (a DSH differential phase accumulates the walk twice,
    # hence 4π rather than 2π).
    slope, intercept, var_cpu, a_sec = _increment_variance_fit(d2, ls, 1.0 / fs, xp)

    return LinewidthEstimate(
        value=to_report_scalar(np.maximum(slope, 0.0) / (4.0 * np.pi)),
        method=method,
        f_shift=f_hat,
        awgn_var=to_report_scalar(intercept),
        dphi_var=to_report_scalar(to_device(dphi_var, "cpu")),
        lags=ls,
        lag_s=a_sec,
        var=var_cpu.T,
        slope=slope,
        intercept=intercept,
    )


def _dsh_lorentzian(
    samples: ArrayType, method: DshLorentzian, fs: float
) -> LinewidthEstimate:
    td = float(method.delay)
    _delay_samples(td, fs)
    level_db = float(method.level_db)
    z2, was_1d, xp = _analytic_beat(samples)
    npseg = _resolve_nperseg(z2.shape[-1], method.nperseg, cap=1 << 14)
    f, P = welch_psd(z2, sampling_rate=fs, nperseg=npseg, return_onesided=False)
    # From here on the work is scalar peak/width searching on an
    # nperseg-sized spectrum - host-side NumPy on purpose.
    f_cpu = np.asarray(to_device(f, "cpu"), dtype=np.float64)
    P_cpu = np.asarray(to_device(P, "cpu"), dtype=np.float64)
    P2 = P_cpu[None, :] if P_cpu.ndim == 1 else P_cpu
    c = P2.shape[0]

    r_deep = 10.0 ** (level_db / 10.0)
    dnu_deep = np.full(c, np.nan)
    dnu_3db = np.full(c, np.nan)
    ratio = np.full(c, np.nan)
    f_peak = np.full(c, np.nan)
    bin_hz = f_cpu[1] - f_cpu[0]
    for ch in range(c):
        p = P2[ch]
        # Pass 1 - rough half-power width on the raw spectrum, only to size
        # the smoothing window.  The raw argmax bin rides the upward Welch
        # fluctuations (max over many ±1/√K bins), which biases the peak high
        # and every width low; smoothing over ≈ FWHM/5 removes that bias at
        # < 3 % lineshape droop.
        w3_rough = _lorentzian_widths(f_cpu, p, 2.0)
        if np.isfinite(w3_rough):
            w_bins = min(int(w3_rough / (5.0 * bin_hz)) | 1, 101)
            if w_bins >= 3:
                p = moving_average(p, window=w_bins, mode="same")

        i_pk = int(np.argmax(p))
        f_peak[ch] = f_cpu[i_pk]
        dyn_db = 10.0 * np.log10(p[i_pk] / np.median(p))
        if dyn_db < level_db + 10.0:
            logger.warning(
                "estimate_linewidth: beat peak only %.1f dB above the PSD floor "
                "(channel %d) - the -%g dB contour is noise-limited; increase "
                "averaging or lower level_db.",
                dyn_db,
                ch,
                level_db,
            )
        w3 = _lorentzian_widths(f_cpu, p, 2.0)  # half-power width
        w_deep = _lorentzian_widths(f_cpu, p, r_deep)
        if np.isfinite(w3) and w3 < 6.0 * bin_hz:
            logger.warning(
                "estimate_linewidth: half-power width spans < 6 Welch bins "
                "(channel %d) - increase nperseg for a resolved line.",
                ch,
            )
        dnu_3db[ch] = w3 / 2.0
        dnu_deep[ch] = w_deep / (2.0 * np.sqrt(r_deep - 1.0))
        if np.isfinite(w3) and np.isfinite(w_deep) and w3 > 0.0:
            ratio[ch] = w_deep / w3

    coh = np.pi * dnu_deep * td  # τ_d / τ_c
    if np.any(np.isfinite(coh) & (coh < 6.0)):
        logger.warning(
            "estimate_linewidth: τ_d/τ_c = %s < 6 - coherent-regime fringes; the "
            "Lorentzian width is unreliable. Use DshFmPsd or DshIncrement.",
            np.array2string(coh, precision=2),
        )

    return LinewidthEstimate(
        value=to_report_scalar(dnu_deep),
        method=method,
        linewidth_3db=to_report_scalar(dnu_3db),
        lineshape_ratio=to_report_scalar(ratio),
        coherence_factor=to_report_scalar(coh),
        f=f_cpu,
        psd=restore_1d(was_1d, P2),
        f_peak=to_report_scalar(f_peak),
    )


# -----------------------------------------------------------------------------
# Verb
# -----------------------------------------------------------------------------

_PHASE_METHODS: dict[type, Any] = {
    IncrementSlope: _increment_slope,
    IncrementSubtract: _increment_subtract,
    BetaSeparation: _beta_separation,
}
_BEAT_METHODS: dict[type, Any] = {
    DshFmPsd: _dsh_fm_psd,
    DshIncrement: _dsh_increment,
    DshLorentzian: _dsh_lorentzian,
}


def estimate_linewidth(
    samples: ArrayType | Signal,
    method: Any,
    *,
    sampling_rate: float | None = None,
) -> LinewidthEstimate:
    r"""Laser or carrier linewidth by the given method.

    Parameters
    ----------
    samples : array_like or Signal
        The record the method reads, ``(N,)`` or ``(C, N)`` (channels are
        independent records):

        * ``IncrementSlope``, ``IncrementSubtract``, ``BetaSeparation``: a
          real phase trajectory in radians (``carrier_phase_trajectory``,
          ``separate_drift_phase_noise``, ``dsh_phase``).  Complex input
          raises.
        * ``DshFmPsd``, ``DshIncrement``, ``DshLorentzian``: a delayed
          self-heterodyne beat record, real (single photodetector) or complex
          (IQ front end), as an array or a Signal.
    method : IncrementSlope, IncrementSubtract, BetaSeparation, DshFmPsd, DshIncrement or DshLorentzian
        The estimator and its parameters.
    sampling_rate : float, optional
        Sampling rate of the record in Hz (a fact: the symbol rate for a
        per-symbol phase trajectory).  Taken from a Signal; required for
        arrays.  A value that disagrees with the Signal raises.

    Returns
    -------
    LinewidthEstimate
        ``value`` (Hz) plus the method's diagnostics, host-side.

    Raises
    ------
    TypeError
        If ``method`` is not one of the methods above.
    ValueError
        On complex input to a phase method, a missing ``sampling_rate`` or a
        delay below one sample.
    """
    name = "estimate_linewidth()"
    adapter = adapt_signal(samples, function_name=name)
    fs = float(adapter.resolve_fact("sampling_rate", sampling_rate))
    x, xp, _ = dispatch(adapter.array)
    kind = type(method)
    if kind in _PHASE_METHODS:
        if xp.iscomplexobj(x):
            raise ValueError(
                f"{name}: {kind.__name__} reads a real phase trajectory in "
                "radians, got complex samples (extract it with "
                "carrier_phase_trajectory or dsh_phase, or use a Dsh* method "
                "for a beat record)."
            )
        return _PHASE_METHODS[kind](x, method, fs)
    if kind in _BEAT_METHODS:
        return _BEAT_METHODS[kind](x, method, fs)
    raise TypeError(
        f"{name}: method must be IncrementSlope, IncrementSubtract, "
        "BetaSeparation, DshFmPsd, DshIncrement or DshLorentzian, got "
        f"{kind.__name__}."
    )
