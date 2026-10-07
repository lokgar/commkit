"""Laser/carrier characterization plots (drift, Allan, linewidth).

Each plot takes the result it draws: a ``FrequencyDrift``, an
``AllanDeviation`` or a ``LinewidthEstimate`` from ``commkit.analysis``.
"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from ..backend import to_device
from .sync import plot_carrier_phase_decomposition
from .theme import (
    _as_channels,
    _finish,
    _get_axis,
    _grid_figsize,
    _set_eng_formatter,
)


def _require(result: Any, field: str, name: str, what: str) -> Any:
    """``result.<field>``; a result without it raises, naming what is needed."""
    value = getattr(result, field, None)
    if value is None:
        raise ValueError(
            f"{name} needs {what} (got {type(result).__name__} without {field})."
        )
    return value


def plot_frequency_drift(
    drift,
    *,
    sampling_rate: float,
    amp_ref: float | None = None,
    ax=None,
    show: bool = False,
    title: str = "Residual frequency drift",
) -> tuple[Any, Any] | None:
    """
    Plots the instantaneous residual frequency offset vs time.

    Draws the ``df`` of ``analysis.frequency_drift`` - the slope of the
    smoothed (drift) phase.  This is the spin the carrier-phase recovery must
    track.

    Parameters
    ----------
    drift : FrequencyDrift
        From ``analysis.frequency_drift``; ``df`` is ``(M,)`` or ``(C, M)``.
    sampling_rate : float
        Sampling rate of ``df`` in Hz (time axis).
    amp_ref : float, optional
        If given, draws dashed ``±amp_ref`` reference lines (e.g. the
        injected wander amplitude in a simulation).
    ax : Axes, optional
    show : bool, default False
    title : str

    Returns
    -------
    (fig, ax) or None
    """
    df_c = _as_channels(
        _require(drift, "df", "plot_frequency_drift()", "a FrequencyDrift")
    )
    C, M = df_c.shape

    fig, axi = _get_axis(ax)

    t = np.arange(M) / float(sampling_rate)
    for i in range(C):
        axi.plot(t, df_c[i], color=f"C{i}", label=f"Pol {i}" if C > 1 else None)

    if amp_ref is not None:
        axi.axhline(amp_ref, color="white", ls="--", label="±amplitude")
        axi.axhline(-amp_ref, color="white", ls="--")

    _set_eng_formatter(axi, "x", "s")
    _set_eng_formatter(axi, "y", "Hz")
    axi.set_xlabel("Time [s]")
    axi.set_ylabel(r"$\Delta f$ [Hz]")
    axi.set_title(title)
    if C > 1 or amp_ref is not None:
        axi.legend(loc="best")

    return _finish((fig, axi), show)


def _log_cell_median(f_pos, s, sel, points_per_octave=24):
    """Median-reduce a PSD onto a log-frequency grid (NaN for empty cells).

    Returns geometric cell-center frequencies and the per-cell median of the
    ``sel``-selected bins - the readable trace for dense Welch spectra, and
    exactly the reduction the plateau detector medians over.
    """
    if not sel.any():
        return np.array([]), np.array([])
    f_lo, f_hi = float(f_pos[sel][0]), float(f_pos[sel][-1])
    if f_hi <= f_lo:
        return np.array([f_lo]), np.array([float(np.median(s[sel]))])
    n_cells = max(int(np.ceil(np.log2(f_hi / f_lo) * points_per_octave)), 1)
    edges = np.geomspace(f_lo, f_hi, n_cells + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    idx = np.clip(np.searchsorted(edges, f_pos, side="right") - 1, 0, n_cells - 1)
    med = np.full(n_cells, np.nan)
    for k in range(n_cells):
        cell_sel = sel & (idx == k)
        if cell_sel.any():
            med[k] = float(np.median(s[cell_sel]))
    return centers, med


def plot_frequency_noise_psd(
    estimate,
    *,
    ax=None,
    show: bool = False,
    title: str = "Frequency-noise PSD",
) -> tuple[Any, Any] | None:
    """
    Plots the frequency-noise PSD of a linewidth estimate on log-log axes.

    Draws ``S_f`` with the white-FM floor ``Δν/π`` (``linewidth_floor`` for
    ``BetaSeparation``, the estimate itself for ``DshFmPsd``) and the region
    annotations the estimate carries:

    * ``band`` - the analysis window: the ``[f_min, f_max]`` fence of
      ``BetaSeparation`` or the extent of the ``DshFmPsd`` plateau.  It is
      *not* itself the integration region.
    * ``above`` (``BetaSeparation``) - the **actual β-integration region**
      ``{f : S_f(f) > β(f)}`` (in general a union of disjoint intervals): the
      area between the dashed β-line and the PSD is filled wherever it holds.
    * ``used`` - the bins the floor median ran over (the auto-detected
      plateau).  Sparse masks (≤ 400 bins) are drawn as markers on the PSD
      trace, dense masks as a highlighted log-binned median curve.

    Dense spectra are drawn faint with a log-binned median trace on top.
    Every channel's PSD is drawn; the masks are those of channel 0.

    Parameters
    ----------
    estimate : LinewidthEstimate
        From ``analysis.estimate_linewidth`` with ``BetaSeparation`` or
        ``DshFmPsd``.
    ax : Axes, optional
    show : bool, default False
    title : str

    Returns
    -------
    (fig, ax) or None
    """
    name = "plot_frequency_noise_psd()"
    S_f = _require(estimate, "S_f", name, "a BetaSeparation or DshFmPsd estimate")
    f, beta_line, above = estimate.f, estimate.beta_line, estimate.above
    band, used = estimate.band, estimate.used
    floor = estimate.linewidth_floor
    if floor is None:
        floor = estimate.value
    f_c = np.asarray(to_device(f, "cpu"), dtype=np.float64)
    S_c = _as_channels(S_f)
    C = S_c.shape[0]
    pos = f_c > 0
    fp = f_c[pos]
    # Dense Welch spectra (low-averaged DSH deconvolutions especially) are
    # unreadable raw: draw them faint and put a log-binned median curve on
    # top - the same reduction the plateau detector runs on.
    dense = int(pos.sum()) > 4000

    fig, axi = _get_axis(ax)

    for i in range(C):
        s_i = S_c[i, pos]
        lbl = f"$S_f$ pol {i}" if C > 1 else "$S_f(f)$"
        if dense:
            axi.loglog(fp, s_i, color=f"C{i}", alpha=0.4)
            fg, sg = _log_cell_median(fp, s_i, np.isfinite(s_i))
            axi.loglog(fg, sg, color=f"C{i}", label=lbl + " (log-binned)")
        else:
            axi.loglog(fp, s_i, color=f"C{i}", label=lbl)

    b_c = None
    if beta_line is not None:
        b_c = np.asarray(to_device(beta_line, "cpu"), dtype=np.float64)
        axi.loglog(
            f_c[pos],
            b_c[pos],
            color="#ff5555",
            ls="--",
            label=r"$\beta$-separation line",
        )

    if above is not None and b_c is not None:
        a_c = np.asarray(to_device(above, "cpu"), dtype=bool)
        if a_c.ndim > 1:
            a_c = a_c[0]
        axi.fill_between(
            f_c[pos],
            b_c[pos],
            S_c[0, pos],
            where=a_c[pos].tolist(),
            color="#ff5555",
            alpha=0.4,
            lw=0,
            label=r"$\beta$-area (integrated region)",
        )

    if used is not None:
        u_c = np.asarray(to_device(used, "cpu"), dtype=bool)
        if u_c.ndim > 1:
            u_c = u_c[0]
        sel = u_c[pos] & np.isfinite(S_c[0, pos])
        if int(sel.sum()) <= 400:
            axi.loglog(
                fp[sel],
                S_c[0, pos][sel],
                ".",
                color="#06d6a0",
                ms=3.0,
                ls="none",
                label="Plateau bins (median)",
            )
        else:
            # Dense mask: highlight the log-binned median over the accepted
            # region instead of splattering one marker per bin.
            fg, sg = _log_cell_median(fp, S_c[0, pos], sel)
            axi.loglog(fg, sg, color="#06d6a0", label="Plateau (median region)")

    if floor is not None:
        floors = np.atleast_1d(np.asarray(floor, dtype=np.float64))
        floor_mean = float(np.mean(floors))
        axi.axhline(
            floor_mean / np.pi,
            color="#ffd166",
            ls=":",
            label=r"White-FM floor  $\Delta\nu/\pi$",
        )

    if band is not None:
        lo = max(float(band[0]), float(fp[0])) if fp.size else float(band[0])
        if used is None:
            # With a used-region overlay the span is redundant clutter; keep
            # the full shading only for fence-defined (manual) bands.
            axi.axvspan(
                lo,
                band[1],
                color="#06d6a0",
                alpha=0.4,
                label="Analysis band [f_min, f_max]",
            )
        for edge in (lo, float(band[1])):
            axi.axvline(edge, color="#06d6a0")

    _set_eng_formatter(axi, "x", "Hz")
    axi.set_xlabel("Frequency [Hz]")
    axi.set_ylabel("$S_f$ [Hz²/Hz]")
    axi.set_title(title)
    axi.legend(loc="best")
    axi.grid(True, which="both")

    return _finish((fig, axi), show)


def plot_allan_deviation(
    allan,
    *,
    reference_slopes: bool = True,
    ax=None,
    show: bool = False,
    title: str = "Allan deviation",
) -> tuple[Any, Any] | None:
    """
    Plots the (overlapping) Allan deviation vs averaging time on log-log axes.

    The local slope classifies the dominant frequency-noise process:
    white-FM ~ tau^(-1/2), flicker-FM ~ tau^0 (flat), random-walk-FM ~ tau^(+1/2),
    linear drift ~ tau^(+1).

    Parameters
    ----------
    allan : AllanDeviation
        From ``analysis.allan_deviation``; ``adev`` in Hz, ``(n_tau,)`` or
        ``(C, n_tau)``.
    reference_slopes : bool, default True
        If True, overlays a faint ``τ^{-1/2}`` (white-FM) guide line.
    ax : Axes, optional
    show : bool, default False
    title : str

    Returns
    -------
    (fig, ax) or None
    """
    adev = _require(allan, "adev", "plot_allan_deviation()", "an AllanDeviation")
    tau = np.asarray(to_device(allan.tau_s, "cpu"), dtype=np.float64)
    adv = _as_channels(adev)
    C = adv.shape[0]

    fig, axi = _get_axis(ax)

    for i in range(C):
        axi.loglog(
            tau,
            adv[i],
            "o-",
            ms=3,
            color=f"C{i}",
            label=f"Pol {i}" if C > 1 else r"$\sigma_y(\tau)$",
        )

    if reference_slopes:
        good = np.isfinite(adv[0]) & (adv[0] > 0)
        if np.any(good):
            tau0, a0 = tau[good][0], adv[0][good][0]
            guide = a0 * np.sqrt(tau0 / tau)  # τ^{-1/2} anchored at first point
            axi.loglog(
                tau,
                guide,
                color="gray",
                ls=":",
                label=r"$\tau^{-1/2}$ (white-FM)",
            )

    _set_eng_formatter(axi, "x", "s")
    _set_eng_formatter(axi, "y", "Hz")
    axi.set_xlabel(r"Averaging Time $\tau$ [s]")
    axi.set_ylabel(r"Allan Deviation $\sigma_y(\tau)$ [Hz]")
    axi.set_title(title)
    axi.legend(loc="best")
    axi.grid(True, which="both")

    return _finish((fig, axi), show)


def plot_increment_variance(
    estimate,
    *,
    ax=None,
    show: bool = False,
    title: str = "Phase-increment variance",
) -> tuple[Any, Any] | None:
    """
    Plots Var of phase increments vs lag with the fitted linear model.

    The measured points should follow ``Var = slope·lag + intercept`` for
    white-FM (Wiener) phase noise; curvature signals flicker or drift
    contamination.  The fit and its intercept (the additive-noise term
    ``2σ_φ²``) are drawn when the estimate has one (``IncrementSlope``,
    ``DshIncrement``); ``IncrementSubtract`` draws its single lag-1 point.

    Parameters
    ----------
    estimate : LinewidthEstimate
        From ``analysis.estimate_linewidth`` with ``IncrementSlope``,
        ``IncrementSubtract`` or ``DshIncrement``.
    ax : Axes, optional
    show : bool, default False
    title : str

    Returns
    -------
    (fig, ax) or None
    """
    var = _require(
        estimate, "var", "plot_increment_variance()", "an increment-method estimate"
    )
    slope, intercept = estimate.slope, estimate.intercept
    lag = np.asarray(to_device(estimate.lag_s, "cpu"), dtype=np.float64)
    v = _as_channels(var)
    C = v.shape[0]

    fig, axi = _get_axis(ax)

    for i in range(C):
        axi.plot(
            lag,
            v[i],
            "o",
            ms=4,
            color=f"C{i}",
            label=f"Measured pol {i}" if C > 1 else "Measured",
        )
    if slope is not None and intercept is not None:
        sl = np.atleast_1d(np.asarray(to_device(slope, "cpu"), dtype=np.float64))
        ic = np.atleast_1d(np.asarray(to_device(intercept, "cpu"), dtype=np.float64))
        for i in range(C):
            axi.plot(
                lag,
                sl[min(i, sl.size - 1)] * lag + ic[min(i, ic.size - 1)],
                "-",
                color=f"C{i}",
                alpha=0.4,
                label="Fit" if i == 0 else None,
            )
        axi.axhline(
            float(np.mean(ic)),
            color="gray",
            ls=":",
            label=r"Intercept (noise $2\sigma_\phi^2$)",
        )

    _set_eng_formatter(axi, "x", "s")
    axi.set_xlabel("Increment Lag [s]")
    axi.set_ylabel("Variance [rad²]")
    axi.set_title(title)
    axi.legend(loc="best")

    return _finish((fig, axi), show)


def plot_dsh_beat_psd(
    estimate,
    *,
    ax=None,
    show: bool = False,
    title: str = "DSH beat spectrum",
) -> tuple[Any, Any] | None:
    """
    Plots the self-heterodyne beat PSD (dB rel. peak) with width annotations.

    Centres the axis on the beat peak, overlays the half-power and
    ``-level_db`` contours, and shades the full widths implied by the
    linewidth estimates (``FWHM = 2Δν``, ``W_L = 2√(10^{L/10}-1)·Δν``).

    Parameters
    ----------
    estimate : LinewidthEstimate
        From ``analysis.estimate_linewidth`` with ``DshLorentzian``.
    ax : Axes, optional
    show : bool, default False
    title : str

    Returns
    -------
    (fig, ax) or None
    """
    psd = _require(estimate, "psd", "plot_dsh_beat_psd()", "a DshLorentzian estimate")
    f, f_peak = estimate.f, estimate.f_peak
    linewidth, linewidth_3db = estimate.value, estimate.linewidth_3db
    level_db = float(estimate.method.level_db)
    f_c = np.asarray(to_device(f, "cpu"), dtype=np.float64)
    p = _as_channels(psd)
    C = p.shape[0]
    f0 = 0.0
    if f_peak is not None:
        f0 = float(np.mean(np.atleast_1d(np.asarray(f_peak, dtype=np.float64))))

    fig, axi = _get_axis(ax)

    for i in range(C):
        p_db = 10.0 * np.log10(p[i] / p[i].max())
        axi.plot(
            f_c - f0,
            p_db,
            color=f"C{i}",
            label=f"Pol {i}" if C > 1 else "Beat PSD",
        )

    if linewidth_3db is not None:
        w3 = 2.0 * float(np.mean(np.atleast_1d(linewidth_3db)))
        axi.axhline(-3.01, color="C1", ls=":")
        axi.axvspan(
            -w3 / 2, w3 / 2, color="C1", alpha=0.4, label=r"FWHM = $2\Delta\nu$"
        )
    if linewidth is not None:
        r = 10.0 ** (float(level_db) / 10.0)
        w_deep = 2.0 * np.sqrt(r - 1.0) * float(np.mean(np.atleast_1d(linewidth)))
        axi.axhline(-float(level_db), color="C3", ls=":")
        axi.axvspan(
            -w_deep / 2,
            w_deep / 2,
            color="C3",
            alpha=0.4,
            label=f"$W_{{-{level_db:g}\\,dB}}$",
        )

    _set_eng_formatter(axi, "x", "Hz")
    axi.set_xlabel(
        "Frequency Offset from Beat Peak [Hz]"
        if f_peak is not None
        else "Frequency [Hz]"
    )
    axi.set_ylabel("PSD [dB rel. peak]")
    axi.set_title(title)
    axi.legend(loc="best")

    return _finish((fig, axi), show)


def plot_carrier_phase_characterization(
    phi,
    *,
    drift,
    linewidth,
    allan,
    sampling_rate: float,
    drift_phase=None,
    drift_cutoff: float | None = None,
    amp_ref: float | None = None,
    show: bool = False,
    title: str | None = None,
) -> tuple[Any, Any] | None:
    """
    Full 2x2 carrier-phase characterization dashboard.

    Combines ``plot_carrier_phase_decomposition``, ``plot_frequency_drift``,
    ``plot_frequency_noise_psd`` and ``plot_allan_deviation`` in one figure
    (see ``examples/carrier_phase_analysis.py`` for the full chain).

    Parameters
    ----------
    phi : array_like
        Carrier phase trajectory in radians (``carrier_phase_trajectory``).
    drift : FrequencyDrift
        From ``analysis.frequency_drift``.
    linewidth : LinewidthEstimate
        A ``BetaSeparation`` or ``DshFmPsd`` estimate (its PSD is drawn).
    allan : AllanDeviation
        From ``analysis.allan_deviation``.
    sampling_rate : float
        Sampling rate of ``phi`` in Hz.
    drift_phase : array_like, optional
        Smoothed drift phase (``separate_drift_phase_noise``), overlaid on
        ``phi``.
    drift_cutoff : float, optional
        Annotated in the phase-decomposition panel title.
    amp_ref : float, optional
        Injected wander amplitude reference for the drift panel.
    show : bool, default False
    title : str, optional

    Returns
    -------
    (fig, axes) or None
    """
    fig, axes = plt.subplots(2, 2, figsize=_grid_figsize(2, 2))

    lp = f"  (LP {drift_cutoff / 1e6:.1f} MHz)" if drift_cutoff else ""
    plot_carrier_phase_decomposition(
        phi,
        drift_phase,
        symbol_rate=sampling_rate,
        ax=axes[0, 0],
        title=f"Recovered carrier phase{lp}",
    )
    plot_frequency_drift(
        drift, sampling_rate=sampling_rate, amp_ref=amp_ref, ax=axes[0, 1]
    )
    plot_frequency_noise_psd(linewidth, ax=axes[1, 0])
    plot_allan_deviation(allan, ax=axes[1, 1])

    if title:
        fig.suptitle(title)
    return _finish((fig, axes), show)
