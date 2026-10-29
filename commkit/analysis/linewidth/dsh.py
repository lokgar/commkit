r"""Linewidth from delayed self-heterodyne beats: :class:`DshFmPsd`,
:class:`DshIncrement` and :class:`DshLorentzian`, and their estimators.

They read a beat record, real (single photodetector) or complex (IQ front
end), array or Signal; the front end is :mod:`commkit.analysis.interferometry`.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from ... import _fft
from ..._array import as_2d, restore_1d, to_report_scalar
from ...backend import ArrayType, dispatch, to_device
from ...logger import logger
from ...smoothing import moving_average
from ...spectral import welch_psd
from .._common import (
    _increment_variance_fit,
    _resolve_nperseg,
    _welch_median_bias,
    _whittle_fit,
)
from ..fm_noise import _dsh_nperseg, _interferometer_response, fm_noise_psd
from ..interferometry import _analytic_beat, dsh_phase
from .result import LinewidthEstimate, _check_fence, _check_nperseg

__all__ = [
    "DshFmPsd",
    "DshIncrement",
    "DshLorentzian",
]


# -----------------------------------------------------------------------------
# Methods on delayed self-heterodyne beat records
# -----------------------------------------------------------------------------


def _check_delay(delay: float) -> None:
    if not float(delay) > 0.0:
        raise ValueError(f"delay={delay} must be positive (seconds).")


@dataclass(frozen=True)
class DshFmPsd:
    r"""Linewidth from a model fit to the DSH beat FM-noise PSD.

    The beat's differential phase (:func:`dsh_phase`) gives the FM-noise PSD
    of the beat, to which a Whittle fit of white FM, flicker FM and detection
    noise is applied: ``Δν = π·S_white``.  Works for any delay, in the
    coherent and the incoherent regime.

    Parameters
    ----------
    delay : float
        Differential delay ``τ_d`` of the interferometer in seconds (about
        4.9 µs per km of standard fiber).  With the default segments it only
        has to be known well enough to choose them (see Notes); it does not
        scale the result.
    f_shift : float, optional
        Known beat carrier (AOM frequency) in Hz; estimated when ``None``
        (see :func:`dsh_phase`).
    nperseg : int, optional
        Welch segment length.  Defaults to the longest power of two within the
        delay, clipped to ``[256, 16384]``.  Longer segments resolve the
        interferometer fringes and lower frequencies, but make the fit depend
        on the exact delay.
    f_min, f_max : float, optional
        Fit band in Hz.  Defaults to every bin above DC, capped for a real
        (single-photodiode) capture at its FM detection bandwidth
        ``min(f_shift, f_s/2 - f_shift)``.
    notch_guard : float, default 0.1
        Masks the bins near the notches of the reported ``S_f`` when the
        segments resolve the fringes (see :func:`dsh_fm_noise_psd`); the fit
        uses every bin either way.
    combine_polarizations : bool, default False
        Treat the ``C ≥ 2`` channels as the polarization outputs of one beat
        and combine them before estimating (maximum-ratio, immune to
        polarization fading; see :mod:`commkit.analysis.interferometry`).
        The estimate is then a single value.

    Notes
    -----
    **The measurement.**  The beat carries ``Δφ(t) = φ(t) - φ(t - τ_d)``, so
    its instantaneous frequency is the laser frequency minus its own delayed
    copy, ``f_beat(t) = f(t) - f(t - τ_d)``.  A white-FM laser of Lorentzian
    linewidth ``Δν`` has a flat one-sided FM-noise PSD ``S_f = Δν/π``; the
    task is to read that level from ``f_beat``.

    **The interferometer seen through a Welch window.**  ``f(t) - f(t-τ_d)``
    filters ``S_f`` by ``2 - 2 cos 2πfτ_d``.  A Welch bin averages over the
    window's spectral kernel, which damps the oscillating term by the window
    autocorrelation ``ρ_w`` at the delay, so the expected beat PSD is

        E[S_beat(f)] = (2 - 2ρ_w(τ_d)·cos 2πfτ_d) · S_laser(f) + S_noise(f).

    Segments longer than the delay (``ρ_w → 1``) resolve the ``4 sin²``
    fringes and their notches.  With the default segments, no longer than
    the delay, ``ρ_w = 0``: inside one segment the two copies of the laser
    frequency are disjoint stretches of the same process and add their
    power, so ``E[S_beat] = 2·S_laser + S_noise``.  The fit then needs no
    deconvolution and no exact delay.  A delay error no longer moves notches
    across the band (with resolved fringes, notch ``k`` moves by ``k·Δτ/τ_d``
    of a lobe, so a 0.1 % error spoils the fit beyond lobe ~100), and a long
    spool still gives many segments.

    **The model.**  The mean beat PSD is fitted as

        M(f) = g(f)·(S_white + h_-1/f) + σ²·(2f_s/π²)·sin²(πf/f_s),

    with ``g(f) = 2 - 2ρ_w(τ_d)·cos 2πfτ_d``:

    * ``S_white = Δν/π``: white FM, the Lorentzian linewidth;
    * ``h_-1/f``: flicker FM (technical noise), so a rising low-frequency
      region does not bias the floor (reported as ``flicker``);
    * the last term: additive detection noise.  Noise on the beat adds a
      white angle error of ``σ²`` rad² per sample, whose first difference
      has this ``≈ f²`` spectrum; it bypasses the interferometer, so ``g``
      does not shape it (reported as ``angle_noise_var``).

    The PSD is computed without the ``sinc²`` correction of
    :func:`fm_noise_psd`: sampled white FM has white increments, so its PSD is
    flat to Nyquist as measured.

    **The fit.**  A Welch bin is ``M·χ²_ν/ν``-distributed with ``ν ≈ 2K``
    for ``K`` segments.  The coefficients minimize the Whittle negative
    log-likelihood ``Σ_k [ln M_k + P_k/M_k]``, whose score has zero mean for
    any ``K``: unlike a median of the bins, which reads ``χ²_ν``'s median and
    is biased low at small ``K``, the fit stays unbiased down to a single
    segment.  Bins far above the model (spurs) are screened out by a χ²
    test and the fit repeated; ``used`` shows the bins that remained.

    **Limitations.**

    * The model covers white FM, flicker FM and white detection noise.
      Other structure in the band (random-walk FM ``∝ 1/f²`` at the lowest
      bins, fiber acoustic lines, a colored receiver noise) biases the fit;
      fence it out with ``f_min`` and ``f_max``, and compare ``S_f`` with
      ``S_f_model`` on :func:`commkit.plotting.plot_frequency_noise_psd`.
    * The detection-noise term assumes additive Gaussian angle noise, that is a
      beat SNR well above the unwrap threshold (≳ 10 dB in the record's
      bandwidth).  Below it phase slips add a white-FM-like floor.
    * ``flicker`` is only as good as the band's coverage of the flicker
      region.  The default segments put the first bin at about ``1/τ_d``; a
      flicker corner below it barely shapes the fitted band, and ``h_-1``
      comes out low (by about 2x for a corner a quarter of the first bin).
      ``Δν`` is unaffected.  To measure ``h_-1``, pass segments long enough
      to resolve the corner (then the delay must be accurate, see above).
    * Everything the interferometer adds (acoustic and thermal noise of the
      delay fiber, noise of the AOM's synthesizer) cannot be told apart from
      laser noise.
    """

    delay: float
    f_shift: float | None = None
    nperseg: int | None = None
    f_min: float | None = None
    f_max: float | None = None
    notch_guard: float = 0.1
    combine_polarizations: bool = False

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
    beat's angle noise in the intercept, so no SNR estimate is needed (as for
    ``IncrementSlope``).  It assumes white FM (Wiener phase noise): flicker
    bends ``Var(a)`` upwards and biases ``Δν`` high.  Slow drift does no harm
    up to a linear frequency chirp.

    Parameters
    ----------
    delay : float
        Differential delay ``τ_d`` of the interferometer in seconds.
    f_shift : float, optional
        Known beat carrier (AOM frequency) in Hz; estimated when ``None``.
    lags : tuple of int, optional
        Increment lags in samples.  Defaults to five lags up to
        ``a_max = min(0.5·τ_d·f_s, N/2000)``: inside the disjoint window
        ``ℓ ≤ τ_d·f_s``, and small enough that the variance estimate keeps
        many independent averages for a long delay (decoherence spools).
    combine_polarizations : bool, default False
        Treat the ``C ≥ 2`` channels as the polarization outputs of one beat
        and combine them before estimating (maximum-ratio, immune to
        polarization fading; see :mod:`commkit.analysis.interferometry`).
        The estimate is then a single value.
    """

    delay: float
    f_shift: float | None = None
    lags: tuple[int, ...] | None = None
    combine_polarizations: bool = False

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
    (``W₂₀/(2√99)`` for the usual -20 dB width, which avoids the Gaussian
    core of the 1/f noise that distorts the -3 dB width).

    Parameters
    ----------
    delay : float
        Differential delay ``τ_d`` of the interferometer in seconds, for the
        coherence check.
    nperseg : int, optional
        Welch segment length of the beat PSD.
    level_db : float, default 20.0
        Depth in dB below the peak at which the width is measured.
    combine_polarizations : bool, default False
        Treat the ``C ≥ 2`` channels as the polarization outputs of one beat
        and combine them before estimating (maximum-ratio, immune to
        polarization fading; see :mod:`commkit.analysis.interferometry`).
        The estimate is then a single value.

    Notes
    -----
    Requires ``coherence_factor ≳ 6``: below that the spectrum has a
    coherent carrier spike and fringes spaced ``1/τ_d``, and the width no
    longer reads ``2Δν`` (a warning is logged).  It also needs the
    ``-level_db`` contour above the noise floor (a peak dynamic range of
    ``level_db`` + 10 dB or more) and about 10 or more Welch bins across the
    line.  For a homodyne IQ capture, calibrate the photodiode DC offsets out
    first: a DC spur in the middle of the line takes the peak and narrows
    the widths.
    """

    delay: float
    nperseg: int | None = None
    level_db: float = 20.0
    combine_polarizations: bool = False

    def __post_init__(self) -> None:
        _check_delay(self.delay)
        _check_nperseg(self.nperseg)
        if not float(self.level_db) > 0.0:
            raise ValueError(f"level_db must be positive, got {self.level_db}.")


# -----------------------------------------------------------------------------
# DSH estimators
# -----------------------------------------------------------------------------


# The beat-line test of ``_not_a_beat``: segment length of the averaged
# periodogram, and how far a beat line must rise above the median PSD on
# both sides of it, in the bands at 0.4-0.6 and 1.4-1.6 times its frequency.
# A drift-removed phase residual may rise steeply below its peak (the
# high-pass), but above it falls only as 1/f^2 (about 2x from f to 1.5 f),
# and white noise is flat on both sides.
_LINE_NPERSEG = 4096
_LINE_CONTRAST = 10.0


def _not_a_beat(x: ArrayType, f_shift: float | None, fs: float) -> np.ndarray:
    """Per channel: real input that is not a DSH beat record, ``(C,)`` bool.

    A real beat sits away from DC as a spectral line.  A channel fails when
    (a) more than half of its mean-removed power lies below half its carrier
    (``fs/400`` when the carrier is unknown): a raw phase trajectory, or
    (b) its spectral peak (within ``0.8-1.2·f_shift`` when the carrier is
    known) does not rise ``_LINE_CONTRAST`` above the median PSD on both
    sides of it: a drift-removed residual, white or not, or a beat whose
    line is too weak or too wide for its carrier to measure.  Complex input
    is never a phase trajectory.
    """
    _, xp, _ = dispatch(x)
    x2, _ = as_2d(x, name="samples")
    if xp.iscomplexobj(x2):
        return np.zeros(x2.shape[0], dtype=bool)
    # The first 2^16 samples keep the check at a few percent of an estimate.
    x2 = x2[:, : 1 << 16].astype(xp.float64)
    x2 = x2 - xp.mean(x2, axis=-1, keepdims=True)
    n = x2.shape[-1]
    power = xp.abs(_fft.rfft(x2, axis=-1)) ** 2
    f_full = xp.fft.rfftfreq(n, d=1.0 / fs)
    f_lo = abs(float(f_shift)) / 2.0 if f_shift else fs / 400.0
    low = xp.sum(xp.where(f_full < f_lo, power, 0.0), axis=-1)
    seg = min(_LINE_NPERSEG, n)
    k_seg = n // seg
    segs = x2[:, : k_seg * seg].reshape(x2.shape[0], k_seg, seg)
    psd = xp.mean(xp.abs(_fft.rfft(segs * xp.hanning(seg), axis=-1)) ** 2, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.asarray(to_device(low / xp.sum(power, axis=-1), "cpu"))
    psd_cpu = np.asarray(to_device(psd, "cpu"))  # (C, seg/2 + 1), one transfer

    f = np.fft.rfftfreq(seg, d=1.0 / fs)
    if f_shift:
        search = np.abs(f / abs(float(f_shift)) - 1.0) <= 0.2
    else:
        search = f >= fs / 400.0
    flags = frac > 0.5
    for c, p in enumerate(psd_cpu):
        if flags[c] or not search.any():
            continue
        k = int(np.flatnonzero(search)[np.argmax(p[search])])
        sides = [
            p[(f >= a * f[k]) & (f <= b * f[k])] for a, b in ((0.4, 0.6), (1.4, 1.6))
        ]
        levels = [float(np.median(side)) for side in sides if side.size]
        flags[c] = not levels or p[k] < _LINE_CONTRAST * max(levels)
    return flags


def _warn_if_phase_trajectory(x: ArrayType, f_shift: float | None, fs: float) -> None:
    """Warn when real input to a DSH method is not a beat (``_not_a_beat``)."""
    channels = np.flatnonzero(_not_a_beat(x, f_shift, fs))
    if channels.size:
        logger.warning(
            "estimate_linewidth: channel(s) %s show no DSH beat line - this "
            "looks like a phase trajectory (raw or drift-removed), not a beat "
            "record. Phase trajectories take IncrementSlope, IncrementSubtract "
            "or BetaSeparation. A real beat needs its line 10 dB above the "
            "spectrum at half and 1.5 times its carrier; one with a carrier "
            "below %.3g Hz also needs f_shift=.",
            channels.tolist(),
            fs / 400.0,
        )


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
    m_samp = _delay_samples(td, fs)
    dphi, f_hat = dsh_phase(
        samples,
        sampling_rate=fs,
        f_shift=method.f_shift,
        combine_polarizations=method.combine_polarizations,
    )
    n_diff = dphi.shape[-1] - 1
    nps = _dsh_nperseg(n_diff, m_samp, method.nperseg)
    f, S_beat = fm_noise_psd(dphi, sampling_rate=fs, nperseg=nps)
    # Summary layer: one transfer, host-side fit on the plot-sized spectrum.
    f_cpu = np.asarray(to_device(f, "cpu"), dtype=np.float64)
    S_cpu = np.asarray(to_device(S_beat, "cpu"), dtype=np.float64)
    S2 = S_cpu[None, :] if S_cpu.ndim == 1 else S_cpu
    n_ch = S2.shape[0]

    g = np.asarray(
        _interferometer_response(f_cpu, delay=td, sampling_rate=fs, nperseg=nps)
    )
    if nps > m_samp:
        logger.info(
            "estimate_linewidth(DshFmPsd): nperseg=%d exceeds the delay (%.0f "
            "samples) - the fit models the interferometer fringes and depends "
            "on the exact delay.",
            nps,
            m_samp,
        )

    _, k_seg = _welch_median_bias(n_diff, nps)
    dof = 2.0 * (k_seg / 1.056 if k_seg > 1 else 1.0)

    fmin = 0.0 if method.f_min is None else float(method.f_min)
    fmax = fs / 2.0 if method.f_max is None else float(method.f_max)
    x_in, xp_in, _ = dispatch(samples)
    if method.f_max is None and not xp_in.iscomplexobj(x_in):
        # A real (single-photodiode) beat carries FM sidebands only out to the
        # carrier's distance from DC and Nyquist; beyond it the analytic
        # signal has no support and the PSD reads fake-low.
        fh = float(np.min(np.abs(np.asarray(f_hat))))
        fmax = min(fmax, fh, fs / 2.0 - fh)
    band = (f_cpu > 0) & (f_cpu >= fmin) & (f_cpu <= fmax)
    if int(band.sum()) < 8:
        raise ValueError(
            f"Only {int(band.sum())} FM-PSD bins in the fit band - widen "
            "[f_min, f_max] or lengthen nperseg."
        )

    fb = f_cpu[band]
    X = np.stack(
        [g[band], g[band] / fb, 2.0 * fs / np.pi**2 * np.sin(np.pi * fb / fs) ** 2],
        axis=1,
    )
    theta = np.empty((n_ch, 3))
    used2 = np.zeros(S2.shape, dtype=bool)
    for c in range(n_ch):
        theta[c], kept = _whittle_fit(S2[c, band], X, dof=dof)
        used2[c, np.flatnonzero(band)[kept]] = True

    # Display: the beat PSD divided by the interferometer response, and the
    # fitted model in the same units.
    valid = g >= 4.0 * float(method.notch_guard)
    with np.errstate(divide="ignore", invalid="ignore"):
        S_laser = np.where(valid, S2 / g, np.nan)
        noise = theta[:, 2:3] * (2.0 * fs / np.pi**2) * np.sin(np.pi * f_cpu / fs) ** 2
        model = np.where(
            valid & (f_cpu > 0),
            theta[:, 0:1] + theta[:, 1:2] / f_cpu + noise / g,
            np.nan,
        )
    f_used = f_cpu[used2.any(axis=0)]

    def per_channel(a: np.ndarray) -> Any:
        """The channel axis dropped for 1-D input."""
        return a[0] if S_cpu.ndim == 1 else a

    was_1d = dphi.ndim == 1
    return LinewidthEstimate(
        value=to_report_scalar(np.pi * theta[:, 0], was_1d=was_1d),
        method=method,
        f_shift=f_hat,
        f=f_cpu,
        S_f=per_channel(S_laser),
        S_f_model=per_channel(model),
        valid=valid,
        used=per_channel(used2),
        band=(float(f_used[0]), float(f_used[-1])),
        n_segments=k_seg,
        flicker=to_report_scalar(theta[:, 1], was_1d=was_1d),
        angle_noise_var=to_report_scalar(theta[:, 2], was_1d=was_1d),
    )


def _dsh_increment(
    samples: ArrayType, method: DshIncrement, fs: float
) -> LinewidthEstimate:
    m_samp = _delay_samples(method.delay, fs)
    dphi, f_hat = dsh_phase(
        samples,
        sampling_rate=fs,
        f_shift=method.f_shift,
        combine_polarizations=method.combine_polarizations,
    )
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

    was_1d = dphi.ndim == 1
    return LinewidthEstimate(
        value=to_report_scalar(np.maximum(slope, 0.0) / (4.0 * np.pi), was_1d=was_1d),
        method=method,
        f_shift=f_hat,
        angle_noise_var=to_report_scalar(intercept, was_1d=was_1d),
        dphi_var=to_report_scalar(to_device(dphi_var, "cpu"), was_1d=was_1d),
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
    z2, was_1d, xp = _analytic_beat(samples, method.combine_polarizations, fs)
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
        value=to_report_scalar(dnu_deep, was_1d=was_1d),
        method=method,
        linewidth_3db=to_report_scalar(dnu_3db, was_1d=was_1d),
        lineshape_ratio=to_report_scalar(ratio, was_1d=was_1d),
        coherence_factor=to_report_scalar(coh, was_1d=was_1d),
        f=f_cpu,
        psd=restore_1d(was_1d, P2),
        f_peak=to_report_scalar(f_peak, was_1d=was_1d),
    )
