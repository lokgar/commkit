"""Linewidth estimators: phase-increment slope, FM-noise PSD, β-separation."""

import numpy as np

from ..backend import ArrayType, dispatch, to_device
from ..logger import logger
from ..spectral import welch_psd
from ._common import (
    _BETA_SLOPE,
    _FWHM_FROM_AREA,
    _as_2d,
    _plateau_mask,
    _scalar_or_array,
    _welch_median_bias,
)

__all__ = ["fm_noise_psd", "linewidth_beta_separation", "linewidth_increment"]


def linewidth_increment(
    pn_phase: ArrayType,
    symbol_rate: float,
    *,
    method: str = "slope",
    lags: tuple[int, ...] = (1, 2, 3, 4, 5),
    noise_var: float | None = None,
    snr_db: float | np.ndarray | None = None,
    ref_symbols: ArrayType | None = None,
    edge_trim: int = 0,
    debug_plot: bool = False,
) -> dict[str, float | np.ndarray | bool]:
    r"""Wiener linewidth from the phase-increment variance.

    For a Wiener phase + AWGN angle noise, the variance of the lag-``k``
    increment ``Δφ_k = φ(n) - φ(n-k)`` is **linear in ``k``**:

    Var(delta_phi_k) = slope * k + intercept
    where slope = 2 * pi * linewidth * T_sym and intercept = 2 * noise_var_phi

    because the random-walk variance accumulates with ``k`` while the
    *uncorrelated* AWGN angle noise contributes a fixed ``2σ_φ²`` regardless of
    ``k``.  Two estimators are provided:

    * ``method="slope"`` (default, **rigorous & AWGN-free**): least-squares fit
      of ``Var(Δφ_k)`` vs ``k`` over ``lags``; ``Δν = slope/(2π·T_sym)``.  The
      additive noise (AWGN *and* any residual white error from imperfect
      equalization) cancels into the intercept, so **no noise estimate is
      needed** - the key advantage over single-lag subtraction.
    * ``method="subtract"``: single-lag (``k=1``) variance minus an explicit
      AWGN term.  With ``d`` at unit power, ``σ_n² = 1/ρ``; the flat correction
      subtracts ``σ_n²`` (exact for QPSK, *under*-corrects QAM), while passing
      ``ref_symbols`` applies the amplitude-aware ``σ_n²·E[1/|d|²]`` (rigorous
      for QAM, since inner-ring symbols carry larger angle noise).

    Note: ``method="subtract"`` needs the additive-noise variance only.
    ``metrics.snr`` reports total residual (noise + phase noise + ISI) and
    over-subtracts.  Prefer ``method="slope"``.

    Parameters
    ----------
    pn_phase : array_like
        Phase-noise residual (radians) - the detrended phase - ``(N,)`` or
        ``(C, N)``.  (Use the drift-removed ``pn`` so a residual frequency ramp
        does not add a spurious ``k²`` term to the slope fit.)
    symbol_rate : float
        Symbol rate in Baud.
    method : {"slope", "subtract"}, default "slope"
        Estimator, as above.
    lags : tuple of int, default (1, 2, 3, 4, 5)
        Increment lags ``k`` for the slope fit (``method="slope"``).
    noise_var, snr_db, ref_symbols : optional
        AWGN-correction inputs for ``method="subtract"`` (see above).
    edge_trim : int, default 0
        Samples discarded from each end before differencing.
    debug_plot : bool, default False
        If True, plot ``Var(Δφ_k)`` vs lag with the fitted line
        (``increment_variance``); points only for ``method="subtract"``.

    Returns
    -------
    dict
        ``{'linewidth', 'dphi_var', 'awgn_var', 'method'}`` - linewidth /
        variances are floats (SISO) or per-channel arrays.  ``dphi_var`` is the
        lag-1 increment variance; ``awgn_var`` is the fitted intercept
        (``slope``) or the subtracted AWGN term (``subtract``).

    Notes
    -----
    **Limitations.**

    * The linearity of ``Var(Δφ_k)`` in ``k`` holds for **white-FM (Wiener)**
      phase noise only.  Flicker (1/f) FM noise makes the variance grow
      *faster* than linear, biasing the fitted slope - and hence Δν - high;
      what is reported is then an *effective* linewidth at the lag timescale,
      not the intrinsic Lorentzian linewidth.
    * ``xp.var`` subtracts the per-lag mean, so a **constant** residual
      frequency offset does not bias the fit, but a frequency *ramp*
      (nonlinear drift) adds a ``k²`` term.  Detrend first
      (``separate_drift_phase_noise``) and keep the largest lag ``k·T_sym``
      well inside the drift timescale.
    * Larger lags raise the phase-noise term above the AWGN intercept but
      admit more drift/flicker contamination; the default 1-5 symbol lags
      suit multi-MHz linewidths at GBaud rates.  For sub-100-kHz linewidths at
      high symbol rates the per-lag walk variance may sit orders of magnitude
      below ``2σ_φ²`` - prefer the β-separation/PSD route there.
    """
    p, xp, _ = dispatch(pn_phase)
    p2, was_1d = _as_2d(p)
    n_full = p2.shape[-1]
    sl = slice(edge_trim, n_full - edge_trim) if edge_trim > 0 else slice(None)
    p2 = p2[:, sl].astype(xp.float64)
    c = p2.shape[0]
    t_sym = 1.0 / float(symbol_rate)

    def _var_lag(k):
        dk = p2[:, k:] - p2[:, :-k]
        return xp.var(dk, axis=-1)

    var1 = _var_lag(1)

    if method == "slope":
        ks = np.asarray(sorted(set(int(k) for k in lags if k >= 1)), dtype=np.float64)
        if ks.size < 2:
            raise ValueError("method='slope' needs at least two distinct lags ≥ 1.")
        var_k = xp.stack([_var_lag(int(k)) for k in ks], axis=0)  # (n_lag, C)
        # The fit input is a tiny (n_lag, C) matrix - one D2H transfer and a
        # host-side polyfit beat launching a device least-squares here.
        var_k_cpu = np.asarray(to_device(var_k, "cpu"), dtype=np.float64)
        coeffs = np.polyfit(ks, var_k_cpu, 1)  # (2, C): [slope, intercept]
        slope, intercept = coeffs[0], coeffs[1]
        linewidth_cpu = np.maximum(slope, 0.0) / (2.0 * np.pi * t_sym)
        awgn_var_cpu = intercept
    elif method == "subtract":
        if noise_var is not None:
            sigma_n2 = xp.full(c, float(noise_var), dtype=xp.float64)
        elif snr_db is not None:
            snr_val = xp.atleast_1d(xp.asarray(snr_db, dtype=xp.float64))
            sigma_n2 = 10.0 ** (-snr_val / 10.0)
            if sigma_n2.size == 1:
                sigma_n2 = xp.full(c, float(sigma_n2[0]))
        else:
            sigma_n2 = xp.zeros(c, dtype=xp.float64)

        if ref_symbols is not None and xp.any(sigma_n2):
            from ..helpers import normalize

            d2, _ = _as_2d(xp.asarray(ref_symbols))
            d2 = d2[:, :n_full][:, sl]
            d2 = normalize(d2, mode="average_power", axis=-1)
            inv = 1.0 / xp.maximum(xp.abs(d2) ** 2, 1e-12)
            pair_mean = 0.5 * (inv[:, 1:] + inv[:, :-1])
            e_inv = xp.mean(pair_mean, axis=-1)
            awgn_var = sigma_n2 * e_inv
        else:
            awgn_var = sigma_n2.copy()
        linewidth = xp.maximum(var1 - awgn_var, 0.0) / (2.0 * np.pi * t_sym)

        linewidth_cpu = to_device(linewidth, "cpu")
        awgn_var_cpu = to_device(awgn_var, "cpu")
    else:
        raise ValueError(f"Unknown method {method!r} (use 'slope' or 'subtract').")

    var1_cpu = to_device(var1, "cpu")

    if debug_plot:
        from .. import plotting as _plotting

        if method == "slope":
            _plotting.plot_increment_variance(
                ks * t_sym,
                var_k_cpu.T,
                slope=coeffs[0] / t_sym,
                intercept=coeffs[1],
                show=True,
            )
        else:
            _plotting.plot_increment_variance(
                np.array([t_sym]), np.atleast_1d(var1_cpu)[:, None], show=True
            )

    return {
        "linewidth": _scalar_or_array(linewidth_cpu),
        "dphi_var": _scalar_or_array(var1_cpu),
        "awgn_var": _scalar_or_array(awgn_var_cpu),
        "method": method,
    }


def fm_noise_psd(
    phi: ArrayType,
    symbol_rate: float,
    *,
    nperseg: int | None = None,
    detrend: str | bool = "constant",
    bias_correction: bool = True,
    debug_plot: bool = False,
) -> tuple[ArrayType, ArrayType]:
    r"""One-sided frequency-noise PSD S_f(f) [Hz²/Hz] from the phase.

    Differentiates the phase to the instantaneous frequency
    ``f_inst = diff(phi)/(2π·T_sym)`` (Hz) and estimates its one-sided PSD via
    Welch's method (``welch_psd``).  Distinct
    impairments occupy distinct regions of S_f(f):

    * **white-FM** (linewidth): flat plateau at ``S_f = Δν/π``,
    * **drift / flicker**: steep ``1/f`` (and steeper) rise at low ``f``,
    * **AWGN** angle noise: white phase noise -> ``S_f ∝ f²`` rise at high ``f``.

    Parameters
    ----------
    phi : array_like
        Unwrapped carrier phase (radians), ``(N,)`` or ``(C, N)``.
    symbol_rate : float
        Symbol rate in Baud (sampling rate of ``phi``).
    nperseg : int, optional
        Welch segment length.  Defaults to ``min(N//8, 4096)`` (clipped ≥ 256).
    detrend : str or bool, default "constant"
        Per-segment detrend passed to Welch; ``"constant"`` removes the mean
        residual frequency offset.
    bias_correction : bool, default True
        Undo the first-difference roll-off (see Notes).
    debug_plot : bool, default False
        If True, plot the PSD (``frequency_noise_psd``).

    Returns
    -------
    f : array_like
        One-sided frequency axis in Hz (length ``nperseg//2 + 1``).
    S_f : array_like
        Frequency-noise PSD in Hz²/Hz, shape ``(nfreq,)`` or ``(C, nfreq)``.
        Both stay on the input backend (compute layer - no host transfer;
        the ``linewidth_*`` summary functions are the reporting layer).

    Notes
    -----
    The first difference is not an ideal differentiator: its magnitude
    response is ``|2 sin(πfT)|`` versus the ideal ``2πfT``, so the raw
    estimate is ``S_f,true(f) · sinc²(fT)`` - a -3.9 dB droop at Nyquist
    (``R/2``).  With ``bias_correction=True`` the PSD is divided by
    ``sinc²(fT)`` so the white-FM plateau and the AWGN ``f²`` tail keep their
    analytic levels all the way to Nyquist.

    **Limitations.**

    * Frequency resolution is ``R/nperseg``; noise processes slower than the
      segment length (drift, flicker below the first bin) alias into the
      lowest bins and are *not* resolved - extend the capture, not
      ``nperseg``, to see them.
    * Welch averaging trades variance for resolution: with ``K`` segments the
      per-bin relative std is ``≈ 1/√K``.  The default ``N//8`` with 50 %
      overlap gives ``K ≈ 15``.
    * ``detrend="constant"`` removes the *mean* frequency per segment; a
      residual frequency ramp within a segment still leaks into the lowest
      bins.
    """
    p, xp, _ = dispatch(phi)
    p2, was_1d = _as_2d(p)
    t_sym = 1.0 / float(symbol_rate)

    f_inst = xp.diff(p2.astype(xp.float64), axis=-1) / (2.0 * np.pi * t_sym)
    n = f_inst.shape[-1]
    if nperseg is None:
        nperseg = int(min(max(n // 8, 256), 4096))
    nperseg = min(nperseg, n)

    f, S_f = welch_psd(
        f_inst,
        sampling_rate=float(symbol_rate),
        nperseg=nperseg,
        detrend=detrend,
        return_onesided=True,
        axis=-1,
    )
    if bias_correction:
        # S_f,est = S_f,true · sinc²(fT); undo the diff-differentiator droop.
        S_f = S_f / (xp.sinc(f * t_sym) ** 2)
    S_out = S_f[0] if was_1d else S_f

    if debug_plot:
        from .. import plotting as _plotting

        _plotting.plot_frequency_noise_psd(f, S_out, show=True)

    return f, S_out


def linewidth_beta_separation(
    phi: ArrayType,
    symbol_rate: float,
    *,
    nperseg: int | None = None,
    f_min: float | None = None,
    f_max: float | None = None,
    debug_plot: bool = False,
) -> dict[str, float | np.ndarray]:
    r"""Linewidth via the Di Domenico β-separation line (canonical method).

    Integrates the frequency-noise PSD S_f(f) only over the **region where it
    lies above** the beta-separation line S_f = (8 * ln(2) / pi^2) * f - the
    Heaviside-gated "surface" of the Di Domenico method, in general a union
    of disjoint intervals rather than a contiguous band (the returned
    ``above`` mask is the exact region used).  The FWHM linewidth is
    linewidth = sqrt(8 * ln(2) * A) with A the integrated area (Hz²).  The
    ``[f_min, f_max]`` window is only an outer *fence* on that region: it
    excludes the unresolved DC bin and - with an appropriate f_max - the
    high-frequency AWGN f^2 tail (which eventually climbs back above the
    line and would otherwise be integrated as fake linewidth).

    A white-FM-floor cross-check ``linewidth_floor = π · median(S_f)`` is
    also returned.  When **no fence is given**, the floor's median band is
    auto-detected as the PSD's minimum-level (plateau) region - octave-band-
    median floor, all bins within 3x of it - so a low-frequency drift/flicker
    rise and the AWGN ``f²`` tail are excluded without manual fencing.
    Explicit ``f_min``/``f_max`` switch the floor back to a literal-band
    median (the β-area integral always uses the literal fence).  The floor is
    corrected for the χ²-median bias of Welch bins - a raw median reads
    ``≈ 1 - 1/(3K)`` below the true level for ``K`` averaged segments (the
    β-area integral is mean-based and needs no such correction).

    Parameters
    ----------
    phi : array_like
        Unwrapped carrier phase (radians), ``(N,)`` or ``(C, N)``.
    symbol_rate : float
        Symbol rate in Baud.
    nperseg : int, optional
        Welch segment length (see ``fm_noise_psd``).
    f_min : float, optional
        Lower fence of the analysis window in Hz (drops the residual-FOE DC
        region).  Defaults to the first non-zero Welch bin
        (``symbol_rate/nperseg``).  Note this is a *resolution* floor, not the
        canonical ``1/T_obs`` of the method: FM-noise area between ``1/T_obs``
        and the first Welch bin is unresolved and silently excluded, so for
        drift/flicker-dominated sources the result depends on ``nperseg`` -
        raise ``nperseg`` (or quote ``f_min``) accordingly.
    f_max : float, optional
        Upper fence of the analysis window in Hz.  **Set this below the AWGN
        ``f²`` knee** - the tail crosses back above the β-line and would be
        integrated as fake linewidth; defaults to the Nyquist bin.
    debug_plot : bool, default False
        If True, plot the PSD with the β-line, white-FM floor, and the actual
        integration region shaded (``plot_frequency_noise_psd``).

    Returns
    -------
    dict
        ``{'linewidth', 'linewidth_floor', 'area_hz2', 'f', 'S_f',
        'beta_line', 'above', 'used', 'band', 'n_segments'}`` - linewidths
        are floats (SISO) / arrays (MIMO); ``f``/``S_f``/``beta_line`` are
        NumPy arrays for plotting.  ``above`` is the boolean mask of the bins
        actually integrated (``S_f > β``-line within ``band``, generally a
        **union of disjoint intervals**, not a contiguous band); ``used`` is
        the mask of bins the floor median ran over (auto-detected plateau, or
        the fenced band); ``band`` is the ``(f_min, f_max)`` fence applied;
        ``n_segments`` is the Welch segment count ``K`` behind every bin.

    Notes
    -----
    **Limitations.**

    * The β-separation FWHM is an *approximation* (accurate to ~10 % for
      lineshapes dominated by slow FM noise, and exact for pure white FM -
      the line is constructed so a flat ``S_f = Δν/π`` integrates back to
      ``Δν``).  It is not a substitute for a full lineshape integral when the
      noise sits near the β-line over a wide band.
    * The result is **observation-time dependent** for flicker/drift-dominated
      sources: lowering ``f_min`` (longer capture) adds low-frequency area and
      grows ``Δν``.  Always quote ``f_min`` (equivalently the measurement
      time) with the number - there is no unique "linewidth" of a non-white
      FM source.
    * The AWGN ``f²`` tail eventually crosses back above the β-line and would
      be integrated as *fake* linewidth: set ``f_max`` below the knee where
      the plateau ``Δν/π`` meets the tail ``2σ_φ²T_sym·f²``, i.e.
      ``f_knee = (Δν/(2π σ_φ² T_sym))^{1/2}``.  Check ``debug_plot=True``.
    * ``linewidth_floor`` (π·median of in-band ``S_f``) is the more robust
      estimate when a clean white-FM plateau exists in the band; the two
      should agree within tens of percent, otherwise inspect the PSD.
    """
    f, S_f = fm_noise_psd(phi, symbol_rate, nperseg=nperseg)
    _, xp, _ = dispatch(f)
    S2 = S_f[None, :] if S_f.ndim == 1 else S_f

    beta = _BETA_SLOPE * f
    # One-sided Welch axis: f[0] = 0, f[1] is the first non-zero bin.
    fmin = float(f[1]) if f_min is None else float(f_min)
    fmax = float(f[-1]) if f_max is None else float(f_max)
    band = (f >= fmin) & (f <= fmax)

    # Vectorized over channels: (C, nfreq) masks instead of a per-channel loop.
    above = band[None, :] & (S2 > beta[None, :])
    integrand = xp.where(above, S2, 0.0)
    area = xp.trapezoid(integrand, f, axis=-1)
    lw = xp.sqrt(_FWHM_FROM_AREA * area)

    # Pack the two (C,) metrics into one D2H transfer; the floor median runs
    # host-side so the plateau auto-detection is shared with linewidth_dsh.
    lw_cpu, area_cpu = to_device(xp.stack([lw, area]), "cpu")
    f_cpu = np.asarray(to_device(f, "cpu"), dtype=np.float64)
    S_cpu = np.asarray(to_device(S_f, "cpu"), dtype=np.float64)
    beta_cpu = np.asarray(to_device(beta, "cpu"), dtype=np.float64)
    above_cpu = np.asarray(to_device(above, "cpu"), dtype=bool)

    S2c = S_cpu[None, :] if S_cpu.ndim == 1 else S_cpu
    n_ch = S2c.shape[0]
    band_cpu = (f_cpu >= fmin) & (f_cpu <= fmax)
    base2 = np.zeros(S2c.shape, dtype=bool)
    base2[:] = band_cpu & (f_cpu > 0)
    base2 &= np.isfinite(S2c)

    # Welch bins are χ²-distributed: the median-based floor reads the true
    # level low by median(χ²_ν)/ν ≈ 1 - 1/(3K); divided back out below.
    nps_used = int(round(float(symbol_rate) / float(f_cpu[1])))
    m_med, k_seg = _welch_median_bias(phi.shape[-1] - 1, nps_used)
    if k_seg < 4:
        logger.warning(
            "linewidth_beta_separation: only %d Welch segment(s) at "
            "nperseg=%d - the floor median is noisy and the χ²-median "
            "correction (÷%.3f) is asymptotic; extend the record (aim for "
            "≥ 5·nperseg samples).",
            k_seg,
            nps_used,
            m_med,
        )

    if f_min is None and f_max is None:
        used2, levels = _plateau_mask(f_cpu, S2c, base2)
        lw_floor_cpu = np.pi * levels
        for c in range(n_ch):
            if not np.isfinite(lw_floor_cpu[c]):
                logger.warning(
                    "linewidth_beta_separation: no plateau detected "
                    "(channel %d) - floor median falls back to the full "
                    "band. Inspect the PSD before quoting it.",
                    c,
                )
                used2[c] = base2[c]
                lw_floor_cpu[c] = (
                    np.pi * np.median(S2c[c][used2[c]]) if used2[c].any() else 0.0
                )
    else:
        used2 = base2
        lw_floor_cpu = np.array(
            [
                np.pi * np.median(S2c[c][used2[c]]) if used2[c].any() else 0.0
                for c in range(n_ch)
            ]
        )
    lw_floor_cpu = lw_floor_cpu / m_med
    used_cpu = used2[0] if S_cpu.ndim == 1 else used2
    if S_cpu.ndim == 1:
        above_cpu = above_cpu[0]

    result = {
        "linewidth": _scalar_or_array(lw_cpu),
        "linewidth_floor": _scalar_or_array(lw_floor_cpu),
        "n_segments": k_seg,
        "area_hz2": _scalar_or_array(area_cpu),
        "f": f_cpu,
        "S_f": S_cpu,
        "beta_line": beta_cpu,
        "above": above_cpu,
        "used": used_cpu,
        "band": (fmin, fmax),
    }

    if debug_plot:
        from .. import plotting as _plotting

        _plotting.plot_frequency_noise_psd(
            f_cpu,
            S_cpu,
            beta_line=beta_cpu,
            floor=result["linewidth_floor"],
            band=(fmin, fmax),
            above=above_cpu,
            used=used_cpu,
            show=True,
        )

    return result
