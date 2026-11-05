r"""Linewidth from phase trajectories: :class:`IncrementSlope`,
:class:`IncrementSubtract` and :class:`BetaSeparation`, and their estimators.

They read a real phase trajectory in radians, e.g. the output of
``carrier_phase_trajectory`` or ``separate_drift_phase_noise``.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..._array import as_2d, broadcast_channels, to_report_scalar
from ...backend import ArrayType, dispatch, to_device
from ...logger import logger
from .._common import (
    _BETA_SLOPE,
    _FWHM_FROM_AREA,
    _increment_variance_fit,
    _plateau_mask,
    _welch_median_bias,
    _whittle_fit,
)
from ..fm_noise import fm_noise_psd
from .result import LinewidthEstimate, _check_fence, _check_nperseg

__all__ = [
    "BetaSeparation",
    "IncrementSlope",
    "IncrementSubtract",
]


# -----------------------------------------------------------------------------
# Methods on phase trajectories
# -----------------------------------------------------------------------------


def _check_edge_trim(edge_trim: int) -> None:
    if int(edge_trim) != edge_trim or edge_trim < 0:
        raise ValueError(f"edge_trim must be a non-negative integer, got {edge_trim}.")


@dataclass(frozen=True)
class IncrementSlope:
    r"""Wiener linewidth from the lag-slope of the phase-increment variance.

    For a Wiener phase plus AWGN angle noise, the variance of the lag-``k``
    increment ``Δφ_k = φ(n) - φ(n-k)`` is linear in ``k``:

        Var(Δφ_k) = slope · k·T + intercept,
        slope = 2π·Δν,  intercept = 2σ_φ²,

    because the random-walk variance accumulates with ``k`` while the
    uncorrelated angle noise contributes a fixed ``2σ_φ²``.  A least-squares
    fit over ``lags`` gives ``Δν = slope/(2π)``: the additive noise (the AWGN
    and the residual white error of the equalizer) goes into the intercept,
    so no noise estimate is needed.  The recommended method.

    Parameters
    ----------
    lags : tuple of int, default (1, 2, 3, 4, 5)
        Increment lags ``k`` in samples; at least two distinct lags ``>= 1``
        (smaller entries are ignored).
    edge_trim : int, default 0
        Samples dropped at each end before differencing (the filter
        transients of :func:`separate_drift_phase_noise`).

    Notes
    -----
    **Limitations.**

    * ``Var(Δφ_k)`` is linear in ``k`` for white-FM (Wiener) phase noise
      only.  Flicker (1/f) FM noise makes the variance grow faster than
      linearly, which biases the fitted slope, and so ``Δν``, high: the result
      is then an effective linewidth at the time scale of the lags, not the
      intrinsic Lorentzian linewidth.
    * The variance subtracts the mean of each lag, so a constant residual
      frequency offset does not bias the fit, but a frequency ramp
      (nonlinear drift) adds a ``k²`` term.  Detrend first
      (:func:`separate_drift_phase_noise`) and keep the largest lag ``k·T``
      well inside the drift's time scale.
    * Larger lags lift the phase-noise term above the AWGN intercept but let
      in more drift and flicker; the default lags of 1 to 5 symbols suit
      linewidths of several MHz at GBd rates.  For linewidths below 100 kHz
      at high symbol rates the walk variance per lag can sit orders of
      magnitude below ``2σ_φ²``: prefer ``BetaSeparation`` there.
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
    noise of the increment is ``σ_n²`` (the complex noise variance): the
    flat correction subtracts ``σ_n²`` (exact for PSK, too little for QAM),
    and passing ``reference`` applies the amplitude-aware ``σ_n²·E[1/|d|²]``
    (exact for QAM too, whose inner-ring symbols carry more angle noise).

    It needs the additive-noise variance alone: :func:`commkit.metrics.snr`
    measures the total residual (noise, phase noise and ISI), which
    subtracts too much.  Prefer ``IncrementSlope``, which needs no noise
    estimate.

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
        Samples dropped at each end before differencing.
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

    Integrates the frequency-noise PSD ``S_f(f)`` (:func:`fm_noise_psd`) only
    where it lies above the β-separation line ``S_f = (8 ln2 / π²)·f``, in
    general a union of disjoint intervals; the estimate's ``above`` mask is
    the region used.  The FWHM linewidth is ``sqrt(8 ln2 · A)`` for the
    integrated area ``A`` in Hz².  The ``[f_min, f_max]`` window only fences
    that region: it excludes the unresolved DC bin and, with a suitable
    ``f_max``, the AWGN's ``f²`` tail at high frequencies, which climbs back
    above the line and would otherwise be integrated as a false linewidth.

    The white-FM floor ``linewidth_floor = π·S_white`` is returned too, from
    a Whittle-likelihood fit of white FM, flicker FM and the white angle
    noise of the detection (the ``f²`` tail),
    ``S_f = S_white + h_-1/f + σ²·(2 f_s/π²)·sin²(πf/f_s)``, the model of
    :class:`DshFmFit`; ``flicker``, ``angle_noise_var`` and ``fm_psd_model``
    carry the other terms.  Without a fence the fit runs from the lower edge
    of the PSD's lowest-level (plateau) region, which excludes a drift rise
    at low frequencies, to Nyquist, so the tail is modelled rather than cut
    away by hand.  An explicit ``f_min`` or ``f_max`` makes the fit band the
    fence instead (the β-area integral always uses the fence as given).

    Parameters
    ----------
    nperseg : int, optional
        Welch segment length.  Defaults to the largest power of two within an
        eighth of the record (at least 256 samples): the β-area needs bins
        well below the white-FM crossover ``f_c = πΔν/(8 ln 2)``, about
        ``0.57·Δν``, which a coarse segment cannot resolve (a warning says
        so).
    f_min : float, optional
        Lower fence in Hz (it drops the DC region of a residual frequency
        offset).  Defaults to the first non-zero Welch bin,
        ``sampling_rate/nperseg``.  That is a resolution limit, not the
        method's usual ``1/T_obs``: the FM-noise area between ``1/T_obs`` and
        the first bin is unresolved and left out, so for sources dominated by
        drift or flicker the result depends on ``nperseg``.
    f_max : float, optional
        Upper fence in Hz; set it below the knee of the AWGN's ``f²`` tail.
        Defaults to the Nyquist bin.

    Notes
    -----
    **Limitations.**

    * The β-separation FWHM is an approximation: within about 10 % for
      lineshapes dominated by slow FM noise, and exact for pure white FM (the
      line is built so that a flat ``S_f = Δν/π`` integrates back to
      ``Δν``).
    * For sources dominated by flicker or drift the result depends on the
      observation time: a lower ``f_min`` (a longer capture) adds
      low-frequency area and a larger ``Δν``.  Quote ``f_min`` with the
      number.
    * The AWGN's ``f²`` tail crosses back above the β-line: set ``f_max``
      below the knee where the plateau ``Δν/π`` meets the tail
      ``2σ_φ²T·f²``, ``f_knee = (Δν/(2π σ_φ² T))^{1/2}``.  Inspect it with
      :func:`commkit.plotting.plot_frequency_noise_psd`.
    * For pure white FM the β-area FWHM equals ``linewidth_floor`` once
      ``f_c`` is resolved.  Drift or flicker adds area at low frequencies,
      so the β-area FWHM then exceeds the floor: it is the width of the line
      observed over the record, the floor the intrinsic (Lorentzian) part.
    """

    nperseg: int | None = None
    f_min: float | None = None
    f_max: float | None = None

    def __post_init__(self) -> None:
        _check_nperseg(self.nperseg)
        _check_fence(self.f_min, self.f_max)


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
    was_1d = p.ndim == 1
    return LinewidthEstimate(
        value=to_report_scalar(np.maximum(slope, 0.0) / (2.0 * np.pi), was_1d=was_1d),
        method=method,
        delta_phi_var=to_report_scalar(to_device(var1, "cpu"), was_1d=was_1d),
        angle_noise_var=to_report_scalar(intercept, was_1d=was_1d),
        lag_s=lag_sec,
        increment_var=var_k_cpu.T,
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
        angle_noise_var = sigma_n2 * xp.mean(pair_mean, axis=-1)
    else:
        angle_noise_var = sigma_n2.copy()
    linewidth = xp.maximum(var1 - angle_noise_var, 0.0) / (2.0 * np.pi * t_sym)

    var1_cpu = to_device(var1, "cpu")
    was_1d = p.ndim == 1
    return LinewidthEstimate(
        value=to_report_scalar(to_device(linewidth, "cpu"), was_1d=was_1d),
        method=method,
        delta_phi_var=to_report_scalar(var1_cpu, was_1d=was_1d),
        angle_noise_var=to_report_scalar(
            to_device(angle_noise_var, "cpu"), was_1d=was_1d
        ),
        lag_s=np.array([t_sym]),
        increment_var=np.atleast_1d(var1_cpu)[:, None],
    )


def _beta_separation(
    phi: ArrayType, method: BetaSeparation, fs: float
) -> LinewidthEstimate:
    n_phi = phi.shape[-1]
    nps = min(_beta_nperseg(n_phi, method.nperseg), n_phi - 1)
    f, S_f = fm_noise_psd(phi, sampling_rate=fs, nperseg=nps)
    _, xp, _ = dispatch(f)

    # Transfer the plot-sized spectrum once, up front: every fence, mask and
    # fit below is host-side arithmetic on these arrays.
    f_cpu = np.asarray(to_device(f, "cpu"), dtype=np.float64)
    S_cpu = np.asarray(to_device(S_f, "cpu"), dtype=np.float64)
    S2c = S_cpu[None, :] if S_cpu.ndim == 1 else S_cpu
    n_ch = S2c.shape[0]

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
    above = band[None, :] & (S2 > beta[None, :])
    integrand = xp.where(above, S2, 0.0)
    area = xp.trapezoid(integrand, f, axis=-1)
    lw = xp.sqrt(_FWHM_FROM_AREA * area)
    lw_cpu, area_cpu = to_device(xp.stack([lw, area]), "cpu")

    # White-FM floor: a Whittle fit of white FM, flicker FM and the white
    # angle noise (the f² tail) - the model of DshFmFit - so the tail is
    # modelled rather than cut away.  Without a fence the fit runs from the
    # lower edge of the detected plateau (above any drift rise) to Nyquist.
    eligible = band_cpu & (f_cpu > 0)
    base2 = eligible[None, :] & np.isfinite(S2c)
    auto = method.f_min is None and method.f_max is None
    fit2 = base2.copy()
    if auto:
        plateau2, _ = _plateau_mask(f_cpu, S2c, base2)
        for c in range(n_ch):
            if plateau2[c].any():
                fit2[c] &= f_cpu >= f_cpu[plateau2[c]].min()
            else:
                logger.warning(
                    "BetaSeparation: no white-FM plateau detected (channel %d) - "
                    "the floor is fitted over the full band. Inspect the PSD "
                    "before quoting Δν.",
                    c,
                )
    _, k_seg = _welch_median_bias(n_phi - 1, nps)
    dof = 2.0 * (k_seg / 1.056 if k_seg > 1 else 1.0)
    pos = f_cpu > 0
    noise_col = np.zeros_like(f_cpu)
    noise_col[pos] = 2.0 * fs / np.pi**2 * np.sin(np.pi * f_cpu[pos] / fs) ** 2
    with np.errstate(divide="ignore"):
        X_all = np.stack([np.ones_like(f_cpu), 1.0 / f_cpu, noise_col], axis=1)
    theta = np.zeros((n_ch, 3))
    used2 = np.zeros(S2c.shape, dtype=bool)
    for c in range(n_ch):
        idx = np.flatnonzero(fit2[c])
        if idx.size < 3:
            raise ValueError(
                f"BetaSeparation: only {idx.size} FM-PSD bins in the floor fit "
                "band - widen [f_min, f_max] or lengthen nperseg."
            )
        theta[c], kept = _whittle_fit(S2c[c, idx], X_all[idx], dof=dof)
        used2[c, idx[kept]] = True
    model = np.full(S2c.shape, np.nan)
    model[:, pos] = theta @ X_all[pos].T
    floor_cpu = np.pi * theta[:, 0]

    # The β-area needs bins below the crossover of the white-FM plateau with
    # the line, f_c = π Δν / (8 ln 2) ≈ 0.57 Δν; at the first bin or above
    # it, the area is truncated or empty.
    f_first = float(f_cpu[np.argmax(eligible)]) if eligible.any() else np.inf
    for c in range(n_ch):
        f_cross = np.pi * floor_cpu[c] / (8.0 * np.log(2.0))
        if f_cross < 3.0 * f_first:
            logger.warning(
                "BetaSeparation: the white-FM crossover with the β-line "
                "(f_c ≈ %.3g Hz, channel %d) is not resolved by the band, which "
                "starts at %.3g Hz - the β-area misses the white-FM part; "
                "lengthen nperseg to %d or more (and leave f_min below f_c/3).",
                f_cross,
                c,
                f_first,
                int(2 ** np.ceil(np.log2(3.0 * fs / max(f_cross, 1e-300)))),
            )

    def per_channel(a: np.ndarray) -> Any:
        """The channel axis dropped for 1-D input."""
        return a[0] if S_cpu.ndim == 1 else a

    was_1d = phi.ndim == 1
    return LinewidthEstimate(
        value=to_report_scalar(lw_cpu, was_1d=was_1d),
        method=method,
        linewidth_floor=to_report_scalar(floor_cpu, was_1d=was_1d),
        flicker=to_report_scalar(theta[:, 1], was_1d=was_1d),
        angle_noise_var=to_report_scalar(theta[:, 2], was_1d=was_1d),
        num_segments=k_seg,
        area_hz2=to_report_scalar(area_cpu, was_1d=was_1d),
        frequencies=f_cpu,
        fm_psd=S_cpu,
        fm_psd_model=per_channel(model),
        beta_line=beta_cpu,
        above=per_channel(above_cpu),
        used=per_channel(used2),
        band=(fmin, fmax),
    )


def _beta_nperseg(n: int, nperseg: int | None) -> int:
    """Welch segment of ``BetaSeparation``: the largest power of two within
    ``n/8`` by default (eight segments or more, at least 256 samples), since
    the β-area needs bins far below the white-FM crossover and the Whittle
    floor fit stays unbiased with few segments."""
    if nperseg is not None:
        return int(min(int(nperseg), n))
    return int(min(max(1 << max(int(n // 8).bit_length() - 1, 0), 256), n))
