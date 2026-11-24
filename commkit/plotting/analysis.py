"""Laser and carrier characterization plots (drift, Allan deviation, linewidth).

Each plot takes the result it draws: a ``FrequencyDrift``
(:func:`plot_frequency_drift`), an ``AllanDeviation``
(:func:`plot_allan_deviation`) or a ``LinewidthEstimate``
(:func:`plot_linewidth_estimate`, which draws the diagnostics of its
method, chosen by the estimate's class), from :mod:`commkit.analysis`, or the
phase and drift split of :func:`commkit.analysis.separate_drift_phase_noise`
(:func:`plot_carrier_phase_decomposition`);
:func:`plot_carrier_phase_characterization` puts four of them in one figure.
"""

from typing import Any

import numpy as np
from matplotlib.ticker import EngFormatter

from ..analysis import (
    AllanDeviation,
    BetaSeparationEstimate,
    DshLorentzianEstimate,
    FmNoiseFitEstimate,
    FrequencyDrift,
    IncrementEstimate,
    IncrementSlopeEstimate,
    LinewidthEstimate,
)
from ..backend import _to_host
from .theme import (
    _as_channels,
    _device_channels,
    _finish,
    _get_axis,
    _panel_grid,
    _reference_line,
    _require,
    _series_label,
    _set_eng_formatter,
    _trace,
)

__all__ = [
    "plot_allan_deviation",
    "plot_carrier_phase_characterization",
    "plot_carrier_phase_decomposition",
    "plot_frequency_drift",
    "plot_linewidth_estimate",
]


def plot_carrier_phase_decomposition(
    phi,
    drift=None,
    *,
    symbol_rate: float,
    num_train_symbols: int = 0,
    ax=None,
    label: str | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    legend: bool | None = None,
    show: bool = False,
    title: str = "Recovered carrier phase",
    max_points: int = 4000,
) -> tuple[Any, Any] | None:
    """Plot the recovered carrier-phase trajectory and its slow drift component.

    The unwrapped phase ``φ(t)`` is drawn faintly with the low-pass drift
    in bold over it: the split of
    :func:`commkit.analysis.separate_drift_phase_noise`.  All channels of
    MIMO input are overlaid.

    Parameters
    ----------
    phi : array_like
        Unwrapped carrier phase in radians, ``(N,)`` or ``(C, N)``.
    drift : array_like, optional
        The low-pass drift, shaped as ``phi``; drawn in bold.
    symbol_rate : float
        Symbol rate in Bd, for the time axis in seconds.
    num_train_symbols : int, default 0
        End of the training symbols, drawn as a dashed line when above 0.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; ``None`` creates a figure.
    label : str, optional
        Label for the trajectory, prefixed to channel names when needed.
    xlim, ylim : tuple of float, optional
        Limits for the displayed axes. ``None`` leaves the limits automatic.
    legend : bool, optional
        Show or hide the legend; ``None`` preserves the default.
    show : bool, default False
        Call ``plt.show()`` and return ``None``.
    title : str, default "Recovered carrier phase"
        Title of the figure.
    max_points : int, default 4000
        Points per trace; longer traces are drawn as their min/max envelope,
        reduced on the data's device.  ``<= 0`` draws every point.

    Returns
    -------
    (fig, ax) or None
        The figure and its axes; ``None`` with ``show=True``, which
        shows the figure instead.
    """
    phi_c = _device_channels(phi)
    C = phi_c.shape[0]
    drift_c = _device_channels(drift) if drift is not None else None

    fig, axi = _get_axis(ax)

    dt = 1.0 / float(symbol_rate)
    for i in range(C):
        clabel = _series_label(label, f"pol {i}" if C > 1 else None, r"$\phi$ total")
        line = axi.plot(*_trace(phi_c[i], max_points, dx=dt), alpha=0.4, label=clabel)[
            0
        ]
        if drift_c is not None:
            drift_label = (
                f"{label} drift"
                if label
                else (f"Drift (pol {i})" if C > 1 else "Drift")
            )
            axi.plot(
                *_trace(drift_c[i], max_points, dx=dt),
                color=line.get_color(),
                label=drift_label,
            )

    if num_train_symbols > 0:
        axi.axvline(
            num_train_symbols / float(symbol_rate),
            label=f"DD start ({num_train_symbols})",
            **_reference_line(),
        )

    _set_eng_formatter(axi, "x")
    axi.set_xlabel("Time [s]")
    axi.set_ylabel("Phase [rad]")
    axi.set_title(title)
    axi.legend(loc="best")

    return _finish((fig, axi), show, xlim=xlim, ylim=ylim, legend=legend)


def plot_frequency_drift(
    drift: FrequencyDrift,
    *,
    reference_amplitude: float | None = None,
    ax=None,
    label: str | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    legend: bool | None = None,
    show: bool = False,
    title: str = "Residual frequency drift",
    max_points: int = 4000,
) -> tuple[Any, Any] | None:
    """Plot the instantaneous residual frequency offset against time.

    Draws the ``df`` of :func:`commkit.analysis.frequency_drift`, the slope
    of the smoothed (drift) phase: the rotation the carrier-phase recovery
    must follow.

    Parameters
    ----------
    drift : FrequencyDrift
        From :func:`commkit.analysis.frequency_drift`; ``df`` is ``(M,)`` or
        ``(C, M)``, its ``sampling_rate`` sets the time axis.
    reference_amplitude : float, optional
        Draws dashed reference lines at ``±reference_amplitude`` (e.g. the wander
        amplitude injected in a simulation).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; ``None`` creates a figure.
    label : str, optional
        Label for the plotted data, prefixed to channel names when needed.
    xlim, ylim : tuple of float, optional
        Limits for the displayed axes. ``None`` leaves the limits automatic.
    legend : bool, optional
        Show or hide the legend; ``None`` preserves the default.
    show : bool, default False
        Call ``plt.show()`` and return ``None``.
    title : str, default "Residual frequency drift"
        Title of the figure.
    max_points : int, default 4000
        Points per trace; longer traces are drawn as their min/max envelope,
        reduced on the data's device.  ``<= 0`` draws every point.

    Returns
    -------
    (fig, ax) or None
        The figure and its axes; ``None`` with ``show=True``, which
        shows the figure instead.
    """
    df_c = _device_channels(
        _require(drift, FrequencyDrift, "plot_frequency_drift()", "a FrequencyDrift").df
    )
    C = df_c.shape[0]

    fig, axi = _get_axis(ax)

    dt = 1.0 / float(drift.sampling_rate)
    for i in range(C):
        axi.plot(
            *_trace(df_c[i], max_points, dx=dt),
            label=_series_label(label, f"Pol {i}" if C > 1 else None),
        )

    if reference_amplitude is not None:
        axi.axhline(reference_amplitude, label="±amplitude", **_reference_line())
        axi.axhline(-reference_amplitude, **_reference_line())

    _set_eng_formatter(axi, "x")
    _set_eng_formatter(axi, "y")
    axi.set_xlabel("Time [s]")
    axi.set_ylabel(r"$\Delta f$ [Hz]")
    axi.set_title(title)
    if C > 1 or reference_amplitude is not None:
        axi.legend(loc="best")

    return _finish((fig, axi), show, xlim=xlim, ylim=ylim, legend=legend)


def _log_cell_median(f_pos, s, sel, points_per_octave=24):
    """Median-reduce a PSD onto a log-frequency grid.

    Returns the geometric cell-center frequencies and the per-cell median of
    the bins selected by ``sel``, for the occupied cells only (at low
    frequencies a cell is narrower than a Welch bin, and an empty cell would
    break the drawn line): a readable trace of a dense Welch spectrum.
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
    occupied = np.isfinite(med)
    return centers[occupied], med[occupied]


def _plot_frequency_noise_psd(
    estimate: FmNoiseFitEstimate,
    *,
    ax=None,
    label: str | None = None,
    annotations: bool = True,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    legend: bool | None = None,
    show: bool = False,
    title: str = "Frequency-noise PSD",
) -> tuple[Any, Any] | None:
    """Plot the frequency-noise PSD of a linewidth estimate on log-log axes.

    Draws ``fm_psd`` (:math:`S_f`) with the white-FM floor ``Δν/π`` (``linewidth_floor`` for
    ``BetaSeparation``, the estimate itself for ``DshFmFit``) and the region
    annotations the estimate carries:

    * ``band``: the analysis window, the ``[f_min, f_max]`` fence of
      ``BetaSeparation`` or the extent of the ``DshFmFit`` fit.  It is not
      the integration region itself.
    * ``above`` (``BetaSeparation``): the β-integration region
      ``{f : S_f(f) > β(f)}``, in general a union of disjoint intervals; the
      area between the dashed β-line and the PSD is filled where it holds.
    * ``used``: the bins the white-FM fit ran over, spurs removed.  Sparse
      masks (400 bins or fewer) are drawn as markers on the PSD, dense ones
      as a highlighted log-binned median curve.
    * ``fm_psd_model``: the fitted model of white FM, flicker FM and
      detection noise, drawn over the PSD.  The white-FM plateau, the fit
      bins where the model stays within 10 % of ``Δν/π``, is shaded.  It
      shows where the floor is pinned down; ``Δν`` itself comes from the
      model fitted over all of ``used``, not from an average over the
      plateau.

    Dense spectra are drawn faint with a log-binned median trace on top, and
    the model through the same reduction (a model divided by the response
    of a fringe-resolving DSH fit swings at every notch).  The β-line is
    drawn only up to ten times the highest PSD value, so it does not set
    the vertical range.
    Every channel's PSD is drawn; the masks are those of channel 0.

    Parameters
    ----------
    estimate : FmNoiseFitEstimate
        From :func:`commkit.analysis.estimate_linewidth` with
        ``BetaSeparation`` or ``DshFmFit``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; ``None`` creates a figure.
    label : str, optional
        Label for measured data, prefixed to channel names when needed.
    annotations : bool, default True
        Draw the estimate's fit, guides and markers; ``False`` leaves only
        the measured curves, for overlays.
    xlim, ylim : tuple of float, optional
        Limits for the displayed axes. ``None`` leaves the limits automatic.
    legend : bool, optional
        Show or hide the legend; ``None`` preserves the default.
    show : bool, default False
        Call ``plt.show()`` and return ``None``.
    title : str, default "Frequency-noise PSD"
        Title of the figure.

    Returns
    -------
    (fig, ax) or None
        The figure and its axes; ``None`` with ``show=True``, which
        shows the figure instead.
    """
    S_f = estimate.fm_psd
    f, band, used = estimate.frequencies, estimate.band, estimate.used
    # BetaSeparation only: the β-line, its region and the white-FM floor.
    if isinstance(estimate, BetaSeparationEstimate):
        beta_line, above = estimate.beta_line, estimate.above
        floor = estimate.linewidth_floor
    else:
        beta_line, above, floor = None, None, estimate.value
    f_c = _to_host(f, np.float64)
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
        lbl = _series_label(label, f"$S_f$ pol {i}" if C > 1 else None, "$S_f(f)$")
        if dense:
            line = axi.loglog(fp, s_i, alpha=0.4)[0]
            fg, sg = _log_cell_median(fp, s_i, np.isfinite(s_i))
            axi.loglog(fg, sg, color=line.get_color(), label=lbl + " (log-binned)")
        else:
            axi.loglog(fp, s_i, label=lbl)

    if annotations:
        b_c = None
        if beta_line is not None:
            b_c = _to_host(beta_line, np.float64)
            top = 10.0 * float(np.nanmax(S_c[:, pos]))
            shown = pos & (b_c <= top)
            axi.loglog(
                f_c[shown],
                b_c[shown],
                color="C5",
                ls="--",
                label=r"$\beta$-separation line",
            )

        if above is not None and b_c is not None:
            a_c = _to_host(above, bool)
            if a_c.ndim > 1:
                a_c = a_c[0]
            axi.fill_between(
                f_c[pos],
                b_c[pos],
                S_c[0, pos],
                where=a_c[pos].tolist(),
                color="C5",
                alpha=0.4,
                lw=0,
                label=r"$\beta$-area (integrated region)",
            )

        model = estimate.fm_psd_model
        m_i = None
        if model is not None:
            m_c = _as_channels(model)
            m_i = m_c[0, pos]
            if dense:
                fm, mm = _log_cell_median(fp, m_i, np.isfinite(m_i))
            else:
                fm, mm = fp, m_i
            axi.loglog(fm, mm, color="C3", label="Fitted model", zorder=3)

        used_label = "Fit bins"
        if used is not None:
            u_c = _to_host(used, bool)
            if u_c.ndim > 1:
                u_c = u_c[0]
            sel = u_c[pos] & np.isfinite(S_c[0, pos])
            if int(sel.sum()) <= 400:
                axi.loglog(
                    fp[sel],
                    S_c[0, pos][sel],
                    ".",
                    color="C2",
                    ls="none",
                    label=used_label,
                )
            else:
                # Dense mask: highlight the log-binned median over the accepted
                # region instead of splattering one marker per bin.
                fg, sg = _log_cell_median(fp, S_c[0, pos], sel)
                axi.loglog(fg, sg, color="C2", label="Fit region")

        if floor is not None:
            floors = np.atleast_1d(np.asarray(floor, dtype=np.float64))
            floor_mean = float(np.mean(floors))
            axi.axhline(
                floor_mean / np.pi,
                color="C4",
                ls=":",
                label=r"White-FM floor  $\Delta\nu/\pi$",
            )
            if m_i is not None:
                # Δν is the white term of a model fitted over the whole fit
                # region, not a plateau average.  Shade where the model stays
                # within 10 % of that term (flicker and the angle-noise tail
                # negligible): the plateau that pins the floor down.
                white = float(floors[0]) / np.pi
                dom = np.isfinite(m_i) & (m_i <= 1.1 * white)
                if used is not None:
                    dom &= u_c[pos]
                if dom.any():
                    axi.axvspan(
                        float(fp[dom].min()),
                        float(fp[dom].max()),
                        color="C4",
                        alpha=0.18,
                        lw=0,
                        label="Plateau (model within 10 % of floor)",
                    )

        if band is not None:
            lo = max(float(band[0]), float(fp[0])) if fp.size else float(band[0])
            if used is None:
                # With a used-region overlay the span is redundant clutter; keep
                # the full shading only for fence-defined (manual) bands.
                axi.axvspan(
                    lo,
                    band[1],
                    color="C2",
                    alpha=0.4,
                    label="Analysis band [f_min, f_max]",
                )
            for edge in (lo, float(band[1])):
                axi.axvline(edge, color="C2")

    _set_eng_formatter(axi, "x")
    axi.set_xlabel("Frequency [Hz]")
    axi.set_ylabel("$S_f$ [Hz²/Hz]")
    axi.set_title(title)
    axi.legend(loc="best")

    return _finish((fig, axi), show, xlim=xlim, ylim=ylim, legend=legend)


def plot_allan_deviation(
    allan,
    *,
    reference_slopes: bool = False,
    ax=None,
    label: str | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    legend: bool | None = None,
    show: bool = False,
    title: str = "Allan deviation",
) -> tuple[Any, Any] | None:
    """Plot the overlapping Allan deviation against averaging time, log-log.

    The local slope identifies the dominant frequency-noise process: white
    FM goes as ``τ^(-1/2)``, flicker FM as ``τ^0`` (flat), random-walk FM as
    ``τ^(1/2)`` and a linear drift as ``τ^1``.

    Parameters
    ----------
    allan : AllanDeviation
        From :func:`commkit.analysis.allan_deviation`; ``adev`` in Hz,
        ``(n_tau,)`` or ``(C, n_tau)``.
    reference_slopes : bool, default False
        Overlay a faint ``τ^(-1/2)`` (white-FM) guide line through the first
        point, a slope to compare against; it is not a fit, and where the
        first point is not white-FM limited (a low-passed or
        discriminator-averaged frequency) its level means nothing.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; ``None`` creates a figure.
    label : str, optional
        Label for the plotted data, prefixed to channel names when needed.
    xlim, ylim : tuple of float, optional
        Limits for the displayed axes. ``None`` leaves the limits automatic.
    legend : bool, optional
        Show or hide the legend; ``None`` preserves the default.
    show : bool, default False
        Call ``plt.show()`` and return ``None``.
    title : str, default "Allan deviation"
        Title of the figure.

    Returns
    -------
    (fig, ax) or None
        The figure and its axes; ``None`` with ``show=True``, which
        shows the figure instead.
    """
    adev = _require(
        allan, AllanDeviation, "plot_allan_deviation()", "an AllanDeviation"
    ).adev
    tau = _to_host(allan.taus, np.float64)
    adv = _as_channels(adev)
    C = adv.shape[0]

    fig, axi = _get_axis(ax)

    for i in range(C):
        axi.loglog(
            tau,
            adv[i],
            "o-",
            label=_series_label(
                label, f"Pol {i}" if C > 1 else None, r"$\sigma_y(\tau)$"
            ),
        )

    if reference_slopes:
        good = np.isfinite(adv[0]) & (adv[0] > 0)
        if np.any(good):
            tau0, a0 = tau[good][0], adv[0][good][0]
            guide = a0 * np.sqrt(tau0 / tau)  # τ^{-1/2} anchored at first point
            axi.loglog(
                tau,
                guide,
                label=r"$\tau^{-1/2}$ (white-FM)",
                **_reference_line(),
            )

    _set_eng_formatter(axi, "x")
    _set_eng_formatter(axi, "y")
    axi.set_xlabel(r"Averaging Time $\tau$ [s]")
    axi.set_ylabel(r"Allan Deviation $\sigma_y(\tau)$ [Hz]")
    axi.set_title(title)
    axi.legend(loc="best")

    return _finish((fig, axi), show, xlim=xlim, ylim=ylim, legend=legend)


def _plot_increment_variance(
    estimate: IncrementEstimate,
    *,
    ax=None,
    label: str | None = None,
    annotations: bool = True,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    legend: bool | None = None,
    show: bool = False,
    title: str = "Phase-increment variance",
) -> tuple[Any, Any] | None:
    """Plot the variance of phase increments against lag, with the fitted line.

    For white-FM (Wiener) phase noise the points follow
    ``Var = slope·lag + intercept``; curvature points to flicker or drift.  The fit and its intercept (the additive-noise term
    ``2σ_φ²``) are drawn when the estimate has one (``IncrementSlope``,
    ``DshIncrement``); ``IncrementSubtract`` draws its single lag-1 point.

    Parameters
    ----------
    estimate : IncrementEstimate
        From :func:`commkit.analysis.estimate_linewidth` with
        ``IncrementSlope``, ``IncrementSubtract`` or ``DshIncrement``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; ``None`` creates a figure.
    label : str, optional
        Label for measured data, prefixed to channel names when needed.
    annotations : bool, default True
        Draw the estimate's fit, guides and markers; ``False`` leaves only
        the measured curves, for overlays.
    xlim, ylim : tuple of float, optional
        Limits for the displayed axes. ``None`` leaves the limits automatic.
    legend : bool, optional
        Show or hide the legend; ``None`` preserves the default.
    show : bool, default False
        Call ``plt.show()`` and return ``None``.
    title : str, default "Phase-increment variance"
        Title of the figure.

    Returns
    -------
    (fig, ax) or None
        The figure and its axes; ``None`` with ``show=True``, which
        shows the figure instead.
    """
    var = estimate.increment_var
    # IncrementSlope and DshIncrement only.
    slope: Any = None
    intercept: Any = None
    if isinstance(estimate, IncrementSlopeEstimate):
        slope, intercept = estimate.slope, estimate.intercept
    lag = _to_host(estimate.lag_s, np.float64)
    v = _as_channels(var)
    C = v.shape[0]

    fig, axi = _get_axis(ax)

    colors = []
    for i in range(C):
        curve_label = _series_label(
            label, f"Measured pol {i}" if C > 1 else None, "Measured"
        )
        line = axi.plot(lag, v[i], "o", label=curve_label)[0]
        colors.append(line.get_color())
    if annotations and slope is not None and intercept is not None:
        sl = np.atleast_1d(_to_host(slope, np.float64))
        ic = np.atleast_1d(_to_host(intercept, np.float64))
        for i in range(C):
            axi.plot(
                lag,
                sl[min(i, sl.size - 1)] * lag + ic[min(i, ic.size - 1)],
                "-",
                color=colors[i],
                alpha=0.4,
                label="Fit" if i == 0 else None,
            )
        axi.axhline(
            float(np.mean(ic)),
            label=r"Intercept (noise $2\sigma_\phi^2$)",
            **_reference_line(),
        )

    _set_eng_formatter(axi, "x")
    axi.set_xlabel("Increment Lag [s]")
    axi.set_ylabel("Variance [rad²]")
    axi.set_title(title)
    axi.legend(loc="best")

    return _finish((fig, axi), show, xlim=xlim, ylim=ylim, legend=legend)


def _plot_dsh_beat_psd(
    estimate: DshLorentzianEstimate,
    *,
    ax=None,
    label: str | None = None,
    annotations: bool = True,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    legend: bool | None = None,
    show: bool = False,
    title: str = "DSH beat spectrum",
) -> tuple[Any, Any] | None:
    """Plot the self-heterodyne beat PSD (dB relative to the peak) with its widths.

    The axis is centered on the beat peak; the half-power and ``-level_db``
    contours are overlaid, and the full widths that the linewidth estimates
    imply are shaded (``FWHM = 2Δν``, ``W_L = 2√(10^{L/10}-1)·Δν``).

    Parameters
    ----------
    estimate : DshLorentzianEstimate
        From :func:`commkit.analysis.estimate_linewidth` with
        ``DshLorentzian``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; ``None`` creates a figure.
    label : str, optional
        Label for measured data, prefixed to channel names when needed.
    annotations : bool, default True
        Draw the estimate's fit, guides and markers; ``False`` leaves only
        the measured curves, for overlays.
    xlim, ylim : tuple of float, optional
        Limits for the displayed axes, the horizontal one in Hz from the
        peak.  ``None`` shows twice the ``-level_db`` width on each side of
        the peak, where the line is (the whole spectrum when the estimate
        has no width).
    legend : bool, optional
        Show or hide the legend; ``None`` preserves the default.
    show : bool, default False
        Call ``plt.show()`` and return ``None``.
    title : str, default "DSH beat spectrum"
        Title of the figure.

    Returns
    -------
    (fig, ax) or None
        The figure and its axes; ``None`` with ``show=True``, which
        shows the figure instead.
    """
    psd = estimate.psd
    f, f_peak = estimate.frequencies, estimate.peak_frequency
    linewidth, linewidth_3db = estimate.value, estimate.linewidth_3db
    level_db = float(estimate.method.level_db)
    f_c = _to_host(f, np.float64)
    p = _as_channels(psd)
    C = p.shape[0]
    f0 = 0.0
    if f_peak is not None:
        f0 = float(np.mean(np.atleast_1d(np.asarray(f_peak, dtype=np.float64))))

    fig, axi = _get_axis(ax)

    for i in range(C):
        peak = float(p[i].max()) if p[i].size else 0.0
        if peak > 0.0:
            p_db = 10.0 * np.log10(np.maximum(p[i] / peak, 1e-20))
        else:
            p_db = np.full_like(p[i], -200.0, dtype=np.float64)
        axi.plot(
            f_c - f0,
            p_db,
            label=_series_label(label, f"Pol {i}" if C > 1 else None, "Beat PSD"),
        )

    if annotations and linewidth_3db is not None:
        w3 = 2.0 * float(np.mean(np.atleast_1d(linewidth_3db)))
        axi.axhline(-3.01, color="C1", ls=":")
        axi.axvspan(
            -w3 / 2, w3 / 2, color="C1", alpha=0.4, label=r"FWHM = $2\Delta\nu$"
        )
    r = 10.0 ** (float(level_db) / 10.0)
    w_deep = np.nan
    if linewidth is not None:
        w_deep = 2.0 * np.sqrt(r - 1.0) * float(np.nanmean(np.atleast_1d(linewidth)))
    if xlim is None and np.isfinite(w_deep) and w_deep > 0:
        xlim = (-2.0 * w_deep, 2.0 * w_deep)
    if annotations and np.isfinite(w_deep):
        axi.axhline(-float(level_db), color="C3", ls=":")
        axi.axvspan(
            -w_deep / 2,
            w_deep / 2,
            color="C3",
            alpha=0.4,
            label=f"$W_{{-{level_db:g}\\,dB}}$",
        )

    _set_eng_formatter(axi, "x")
    axi.set_xlabel(
        "Frequency Offset from Beat Peak [Hz]"
        if f_peak is not None
        else "Frequency [Hz]"
    )
    axi.set_ylabel("PSD [dB rel. peak]")
    axi.set_title(title)
    axi.legend(loc="best")

    return _finish((fig, axi), show, xlim=xlim, ylim=ylim, legend=legend)


def plot_carrier_phase_characterization(
    phi,
    *,
    drift,
    linewidth,
    allan,
    sampling_rate: float,
    drift_phase=None,
    drift_cutoff: float | None = None,
    reference_amplitude: float | None = None,
    ax=None,
    show: bool = False,
    title: str | None = None,
) -> tuple[Any, Any] | None:
    """Plot the four panels of a carrier-phase characterization.

    :func:`plot_carrier_phase_decomposition`, :func:`plot_frequency_drift`,
    :func:`plot_linewidth_estimate` and :func:`plot_allan_deviation` in one
    2x2 figure (``examples/25-carrier-phase-characterization.ipynb`` shows
    the whole chain).

    Parameters
    ----------
    phi : array_like
        Carrier-phase trajectory in radians (from
        :func:`commkit.analysis.carrier_phase_trajectory`).
    drift : FrequencyDrift
        From :func:`commkit.analysis.frequency_drift`.
    linewidth : LinewidthEstimate
        Drawn by :func:`plot_linewidth_estimate`: the frequency-noise PSD of a
        ``BetaSeparation`` or ``DshFmFit`` estimate, the most informative
        panel here.
    allan : AllanDeviation
        From :func:`commkit.analysis.allan_deviation`.
    sampling_rate : float
        Sampling rate of ``phi`` in Hz.
    drift_phase : array_like, optional
        Smoothed drift phase (from
        :func:`commkit.analysis.separate_drift_phase_noise`), overlaid on
        ``phi``.
    drift_cutoff : float, optional
        Low-pass cutoff of the drift in Hz, shown in the title of the
        phase panel.
    reference_amplitude : float, optional
        Reference wander amplitude for the drift panel, as for
        :func:`plot_frequency_drift`.
    ax : array_like of matplotlib.axes.Axes, optional
        Four axes in a 2-by-2 grid. ``None`` creates a figure.
    show : bool, default False
        Call ``plt.show()`` and return ``None``.
    title : str, optional
        Title of the figure.

    Returns
    -------
    (fig, axes) or None
        The figure and its axes; ``None`` with ``show=True``, which
        shows the figure instead.
    """
    fig, axes = _panel_grid(ax, 2, 2)

    lp = (
        f"  (LP {EngFormatter(unit='Hz', places=1).format_data(drift_cutoff)})"
        if drift_cutoff
        else ""
    )
    plot_carrier_phase_decomposition(
        phi,
        drift_phase,
        symbol_rate=sampling_rate,
        ax=axes[0, 0],
        title=f"Recovered carrier phase{lp}",
    )
    plot_frequency_drift(drift, reference_amplitude=reference_amplitude, ax=axes[0, 1])
    plot_linewidth_estimate(linewidth, ax=axes[1, 0])
    plot_allan_deviation(allan, ax=axes[1, 1])

    if title:
        fig.suptitle(title)
    return _finish((fig, axes), show)


_LINEWIDTH_PLOTS: dict[type, tuple[Any, str]] = {
    FmNoiseFitEstimate: (_plot_frequency_noise_psd, "Frequency-noise PSD"),
    IncrementEstimate: (_plot_increment_variance, "Phase-increment variance"),
    DshLorentzianEstimate: (_plot_dsh_beat_psd, "DSH beat spectrum"),
}


def plot_linewidth_estimate(
    estimate: LinewidthEstimate,
    *,
    ax=None,
    label: str | None = None,
    annotations: bool = True,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    legend: bool | None = None,
    show: bool = False,
    title: str | None = None,
) -> tuple[Any, Any] | None:
    """Plot a linewidth estimate with the diagnostics of its method.

    The panel follows the estimate's class:

    - :class:`~commkit.analysis.FmNoiseFitEstimate` (``BetaSeparation``,
      ``DshFmFit``): the frequency-noise PSD :math:`S_f` on log-log axes,
      with the white-FM floor ``Δν/π``, the analysis band, the fit bins
      (spurs removed), the fitted model with its white-FM plateau, and for
      ``BetaSeparation`` the β-line and the integrated region where
      :math:`S_f` exceeds it.  Dense spectra are drawn faint under a
      log-binned median; the masks are those of channel 0.
    - :class:`~commkit.analysis.IncrementEstimate` (``IncrementSlope``,
      ``IncrementSubtract``, ``DshIncrement``): the variance of the phase
      increments against the lag, with the fitted line
      ``slope·lag + intercept`` and its intercept (the additive noise
      ``2σ_φ²``) when the method fits one.
    - :class:`~commkit.analysis.DshLorentzianEstimate`: the beat PSD in dB
      relative to its peak, centered on it, with the half-power and
      ``-level_db`` contours and the full widths the estimate implies
      (``FWHM = 2Δν``, ``W_L = 2√(10^{L/10}-1)·Δν``); by default the axis
      spans twice the ``-level_db`` width on each side.

    Parameters
    ----------
    estimate : LinewidthEstimate
        From :func:`commkit.analysis.estimate_linewidth`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; ``None`` creates a figure.
    label : str, optional
        Label for measured data, prefixed to channel names when needed.
    annotations : bool, default True
        Draw the estimate's fit, guides and markers; ``False`` leaves only
        the measured curves, for overlays.
    xlim, ylim : tuple of float, optional
        Limits for the displayed axes. ``None`` leaves the limits automatic.
    legend : bool, optional
        Show or hide the legend; ``None`` preserves the default.
    show : bool, default False
        Call ``plt.show()`` and return ``None``.
    title : str, optional
        Title of the figure; ``None`` names the panel.

    Returns
    -------
    (fig, ax) or None
        The figure and its axes; ``None`` with ``show=True``, which
        shows the figure instead.

    Raises
    ------
    TypeError
        If ``estimate`` is not a ``LinewidthEstimate``.
    ValueError
        For an estimate of none of the families above.
    """
    name = "plot_linewidth_estimate()"
    if not isinstance(estimate, LinewidthEstimate):
        raise TypeError(
            f"{name}: estimate must be a LinewidthEstimate, got "
            f"{type(estimate).__name__}."
        )
    for kind, (draw, default_title) in _LINEWIDTH_PLOTS.items():
        if isinstance(estimate, kind):
            return draw(
                estimate,
                ax=ax,
                label=label,
                annotations=annotations,
                xlim=xlim,
                ylim=ylim,
                legend=legend,
                show=show,
                title=title or default_title,
            )
    raise ValueError(f"{name}: a {type(estimate).__name__} has no diagnostics to draw.")
