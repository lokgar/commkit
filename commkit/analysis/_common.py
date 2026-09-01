"""Shared helpers and constants for the carrier-phase analysis package."""

import numpy as np

from ..backend import to_device
from ..logger import logger

__all__ = [
    "_BETA_SLOPE",
    "_FWHM_FROM_AREA",
    "_floor_levels",
    "_increment_variance_fit",
    "_plateau_mask",
    "_resolve_nperseg",
    "_welch_floor_bias",
    "_welch_median_bias",
]

# β-separation-line slope (Di Domenico 2010): S_f(f) = (8 ln2 / π²) · f
_BETA_SLOPE = 8.0 * np.log(2.0) / (np.pi**2)
# FWHM linewidth from the integrated FM-noise area above the β-line:
# Δν = sqrt(8 ln2 · A)
_FWHM_FROM_AREA = 8.0 * np.log(2.0)


def _resolve_nperseg(n: int, nperseg: int | None, *, cap: int) -> int:
    """Welch segment length actually used for a length-``n`` record.

    ``None`` selects the default ``n//8`` with a floor of 256 bins (a short
    record still resolves the FM-noise shape) and a ceiling of ``cap`` (so
    ``K = n/nperseg`` stays large enough for the Welch average, and for the
    χ²-median correction of ``_welch_median_bias`` to be in its asymptotic
    regime).  An explicit value is honoured, clipped to the record length.

    Callers use the returned value both to *drive* Welch and to reproduce its
    bin spacing (``f[1] = f_s / nperseg``) without reading the frequency axis
    back off the device.
    """
    if nperseg is None:
        return int(min(max(n // 8, 256), cap, n))
    return int(min(int(nperseg), n))


def _welch_median_bias(n_samples: int, nperseg: int) -> tuple[float, int]:
    r"""Median/mean ratio of mean-averaged Welch PSD bins (Hann, 50 % overlap).

    Per-bin Welch estimates are ``S·χ²_ν/ν``-distributed with ``ν = 2·K_eff``
    effective degrees of freedom, so every *median*-based floor summary
    (plateau cell medians, fenced-band medians) reads the true level ``S``
    low by ``median(χ²_ν)/ν`` - ``ln 2 ≈ 0.69`` for a single segment,
    ``≈ 1 - 1/(3K)`` for ``K`` segments.  Callers divide their median-based
    floor by the returned factor.

    ``K_eff = K/1.056`` accounts for the ``ρ = (1/6)²`` power correlation of
    adjacent Hann half-overlapped segments (variance factor ``1 + 2ρ``); the
    χ² median uses the Wilson-Hilferty approximation
    ``median(χ²_ν) ≈ ν·(1 - 2/(9ν))³`` (< 1.3 % error even at ``ν = 2``).

    Parameters
    ----------
    n_samples : int
        Length of the sequence handed to Welch (for ``fm_noise_psd`` this is
        the phase length minus one - the first difference).
    nperseg : int
        Welch segment length actually used.

    Returns
    -------
    factor : float
        ``median/mean`` ratio in ``(0, 1]``.
    n_segments : int
        Welch segment count ``K``.
    """
    step = max(nperseg // 2, 1)
    k = max((int(n_samples) - int(nperseg)) // step + 1, 1)
    k_eff = k / 1.056 if k > 1 else float(k)
    nu = 2.0 * k_eff
    return (1.0 - 2.0 / (9.0 * nu)) ** 3, k


def _plateau_mask(
    f,
    s2,
    base2,
    *,
    points_per_octave=24,
    tolerance=1.8,
    min_cells=6,
    max_rel_iqr=2.0,
):
    """Auto-detect the white-FM plateau of FM-noise PSDs (level + region).

    Laser FM-noise PSDs are bathtub-shaped: drift/flicker rise toward DC and
    the detection-noise ``f²`` tail rises toward Nyquist, so the white-FM
    plateau is the **minimum-level region** in between.  Detection works on a
    **log-frequency grid**: linearly spaced Welch bins are median-reduced
    into log-spaced cells, the floor is located as the minimum of the
    median-filtered cell levels, the level is re-centered on the median of
    the near-floor cells, and the plateau is every cell within a **two-sided**
    ``tolerance`` factor of that level.  The log-uniform weighting is
    essential - a plain median over linear bins is dominated by the highest
    decade, so the slowly-rising ``f²`` tail would bias it high even when it
    is barely above the plateau.  The two-sided acceptance keeps the region
    tight: cells already visibly climbing the tail (or dipping below the
    floor) are excluded, not merely down-weighted.

    A per-cell **dispersion screen** (relative IQR) rejects cells whose bins
    are not statistically homogeneous before the floor search.  This guards
    the DSH deconvolution against a slightly wrong delay: at high frequency
    the assumed and true notch combs decorrelate (argument error ``πfΔτ``)
    and the deconvolved PSD develops fake-low/fake-high stripes *within* a
    cell - huge relative IQR - whereas genuine plateau cells carry uniform
    Welch scatter (relative IQR ≲ 1.6 even for a single-average chi²₂).

    Host-side NumPy by design (plot-sized Welch spectra, one prior transfer).

    Parameters
    ----------
    f : ndarray, (nf,)
        One-sided frequency axis.
    s2 : ndarray, (C, nf)
        FM-noise PSD per channel (NaN allowed at masked bins).
    base2 : ndarray of bool, (C, nf)
        Eligibility per bin (validity mask and any user fences).
    points_per_octave : int, default 24
        Log-grid density for the cell reduction.
    tolerance : float, default 1.8
        Two-sided cell-acceptance factor around the re-centered plateau
        level (±2.6 dB): wide enough for Welch scatter of sparse low-f
        cells, tight enough that cells visibly on the 1/f or f² slopes are
        excluded.
    min_cells : int, default 6
        Minimum accepted cells; channels with fewer return an all-False row
        and NaN level (caller falls back and warns).
    max_rel_iqr : float, default 2.0
        Dispersion screen: cells with ``IQR/median`` above this (evaluated
        for cells holding ≥ 5 bins) are excluded from the floor search and
        the plateau.

    Returns
    -------
    used2 : ndarray of bool, (C, nf)
        Linear bins belonging to accepted cells (all-False where detection
        failed) - the display/consistency mask.
    levels : ndarray, (C,)
        Plateau level per channel: median of accepted cell levels (NaN where
        detection failed).  ``Δν = π · level``, after the caller divides out
        the χ²-median bias of Welch bins (``_welch_median_bias``).
    """
    used2 = np.zeros_like(base2, dtype=bool)
    levels = np.full(base2.shape[0], np.nan)

    for c in range(base2.shape[0]):
        el = base2[c] & np.isfinite(s2[c]) & (f > 0)
        idx_el = np.flatnonzero(el)
        if idx_el.size < min_cells:
            continue
        fe, se = f[idx_el], s2[c][idx_el]

        n_cells = max(int(np.ceil(np.log2(fe[-1] / fe[0]) * points_per_octave)), 1)
        edges = np.geomspace(fe[0], fe[-1], n_cells + 1)
        cell_of = np.clip(np.searchsorted(edges, fe, side="right") - 1, 0, n_cells - 1)
        occupied = np.unique(cell_of)

        cell_med = np.empty(occupied.size)
        homogeneous = np.ones(occupied.size, dtype=bool)
        for i, k in enumerate(occupied):
            vals = se[cell_of == k]
            q1, q2, q3 = np.percentile(vals, (25.0, 50.0, 75.0))
            cell_med[i] = q2
            if vals.size >= 5 and q2 > 0.0:
                homogeneous[i] = (q3 - q1) / q2 <= max_rel_iqr

        occ_h, med_h = occupied[homogeneous], cell_med[homogeneous]
        if med_h.size < min_cells or not (med_h > 0.0).all():
            continue

        # Robust floor: min of the median-filtered cell levels, so a single
        # low-outlier cell (e.g. one noisy bin alone in a cell) cannot set it.
        w = min(5, med_h.size)
        filt = np.array(
            [
                np.median(med_h[max(0, i - w // 2) : i + w // 2 + 1])
                for i in range(med_h.size)
            ]
        )
        floor = float(filt.min())

        # Re-center: the min is biased low by scatter; the plateau level is
        # the median of the near-floor cells.
        near = med_h <= 2.0 * floor
        if int(near.sum()) < min_cells:
            continue
        level0 = float(np.median(med_h[near]))

        # Adaptive two-sided acceptance: a cell joins the plateau only if its
        # level is consistent with level0 *given its own standard error*
        # (median SE ≈ 1.2533·σ_bin/√n).  Densely populated cells get a tight
        # gate (floor ±0.6 dB) that cuts the f² tail where it visibly departs
        # from the plateau; sparse low-frequency cells keep a wide gate up to
        # ``tolerance`` so genuine Welch scatter is not speckled out.
        n_bins = np.array([int((cell_of == k).sum()) for k in occ_h])
        rich = near & (n_bins >= 8)
        if rich.any():
            iqr_rel = np.array(
                [
                    float(np.subtract(*np.percentile(se[cell_of == k], (75.0, 25.0))))
                    / med
                    for k, med in zip(occ_h[rich], med_h[rich])
                ]
            )
            sigma_bin = float(np.median(iqr_rel)) / 1.349
        else:
            sigma_bin = 0.5  # sparse everywhere: fall back to wide gates
        sem = 1.2533 * sigma_bin / np.sqrt(n_bins)
        tol_log = np.clip(3.0 * sem, 0.15, np.log(tolerance))
        accepted = np.abs(np.log(med_h / level0)) <= tol_log
        if int(accepted.sum()) < min_cells:
            continue

        # Keep the largest connected accepted region (gaps of up to 6 cells -
        # notch dropouts - are bridged) so isolated outlier cells far from
        # the plateau cannot stretch the reported band.
        idx_acc = np.flatnonzero(accepted)
        splits = np.flatnonzero(np.diff(idx_acc) > 6) + 1
        clusters = np.split(idx_acc, splits)
        best = max(clusters, key=len)
        if best.size < min_cells:
            continue
        final = np.zeros_like(accepted)
        final[best] = True

        levels[c] = float(np.median(med_h[final]))
        used2[c, idx_el[np.isin(cell_of, occ_h[final])]] = True

    return used2, levels


def _welch_floor_bias(n_samples: int, nperseg: int, *, label: str) -> tuple[float, int]:
    """``_welch_median_bias`` plus the shared segment-count diagnostic.

    Every median-based white-FM floor in this package (β-separation and DSH
    alike) divides out the same χ²-median bias and wants the same warning when
    too few Welch segments back it, so the message lives here rather than being
    re-typed per estimator.

    Parameters
    ----------
    n_samples : int
        Length of the sequence handed to Welch (the differentiated phase).
    nperseg : int
        Welch segment length actually used (see ``_resolve_nperseg``).
    label : str
        Caller name used in the log messages.

    Returns
    -------
    factor : float
        ``median/mean`` ratio to divide the median-based floor by.
    n_segments : int
        Welch segment count ``K``.
    """
    m_med, k_seg = _welch_median_bias(n_samples, nperseg)
    if k_seg < 4:
        logger.warning(
            "%s: only %d Welch segment(s) at nperseg=%d - the floor median is "
            "noisy and the χ²-median correction (÷%.3f) is asymptotic; extend "
            "the record (aim for ≥ 5·nperseg samples).",
            label,
            k_seg,
            nperseg,
            m_med,
        )
    else:
        logger.info(
            "%s: %d Welch segments - χ²-median floor bias corrected by ÷%.4f.",
            label,
            k_seg,
            m_med,
        )
    return m_med, k_seg


def _floor_levels(
    f,
    s2,
    base2,
    *,
    auto: bool,
    label: str,
    fallback2=None,
    fallback_desc: str = "the fenced band",
):
    """White-FM floor level per channel, by plateau auto-detection or fence.

    The summary half of both PSD linewidth routes: ``Δν = π · level`` once the
    caller divides out the χ²-median bias.  Host-side NumPy by design (the
    spectrum has already been transferred once).

    Parameters
    ----------
    f : ndarray, (nf,)
        One-sided frequency axis (host).
    s2 : ndarray, (C, nf)
        FM-noise PSD per channel (host; NaN allowed at masked bins).
    base2 : ndarray of bool, (C, nf)
        Eligible bins (validity mask and any user fences).
    auto : bool
        If True, auto-detect the plateau (``_plateau_mask``) and fall back to
        ``fallback2`` per channel where detection fails.  If False, the median
        runs over ``base2`` as given.
    label : str
        Caller name used in the fallback warning.
    fallback2 : ndarray of bool, (C, nf), optional
        Per-channel fallback mask for failed detection; defaults to ``base2``.
    fallback_desc : str
        Human-readable description of that fallback, for the warning.

    Returns
    -------
    used2 : ndarray of bool, (C, nf)
        Bins the median actually ran over.
    levels : ndarray, (C,)
        Raw median level per channel - **not** χ²-corrected - or NaN where no
        eligible bin remained (the caller decides whether that is a zero floor
        or an error).
    """
    n_ch = s2.shape[0]

    def _median(mask_row, ch):
        return float(np.median(s2[ch][mask_row])) if mask_row.any() else np.nan

    if auto:
        used2, levels = _plateau_mask(f, s2, base2)
        fb = base2 if fallback2 is None else np.broadcast_to(fallback2, s2.shape)
        for c in range(n_ch):
            if np.isfinite(levels[c]):
                continue
            logger.warning(
                "%s: no plateau detected (channel %d) - the floor median falls "
                "back to %s. Inspect the PSD before quoting Δν.",
                label,
                c,
                fallback_desc,
            )
            used2[c] = fb[c]
            levels[c] = _median(used2[c], c)
        return used2, levels

    used2 = np.broadcast_to(base2, s2.shape)
    levels = np.array([_median(used2[c], c) for c in range(n_ch)])
    return used2, levels


def _increment_variance_fit(y2, lags, dt: float, xp):
    r"""Least-squares fit of the increment variance ``Var(Δ_a)`` against lag.

    For a Wiener phase process the lag-``a`` increment variance is linear in
    ``a``, with the uncorrelated additive (AWGN angle) noise contributing a
    lag-independent intercept - the property both linewidth-from-increment
    estimators exploit, so the fit itself is defined once here.  The lag axis
    is converted to **seconds**, so the returned slope is in rad²/s and the
    caller only applies its own constant (``Δν = slope/2π`` for a direct phase
    record, ``slope/4π`` for a DSH differential phase).

    The variances are computed on the input backend and the ``(n_lag, C)``
    result is transferred once - a tiny host-side ``polyfit`` beats launching a
    device least-squares.

    Parameters
    ----------
    y2 : array_like, (C, N)
        Phase record on the input backend.
    lags : sequence of int
        Increment lags in samples (each ``≥ 1``).
    dt : float
        Sample interval in seconds.
    xp : module
        Array module of ``y2``.

    Returns
    -------
    slope, intercept : ndarray, (C,)
        Fitted line coefficients (rad²/s and rad²).
    var_cpu : ndarray, (n_lag, C)
        The fitted variances, host-side.
    x_sec : ndarray, (n_lag,)
        Lag axis in seconds.
    """
    var = xp.stack(
        [xp.var(y2[:, int(a) :] - y2[:, : -int(a)], axis=-1) for a in lags], axis=0
    )  # (n_lag, C)
    var_cpu = np.asarray(to_device(var, "cpu"), dtype=np.float64)
    x_sec = np.asarray(lags, dtype=np.float64) * float(dt)
    coeffs = np.polyfit(x_sec, var_cpu, 1)  # (2, C): [slope, intercept]
    return coeffs[0], coeffs[1], var_cpu, x_sec
