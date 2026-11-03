"""Synchronization plots: timing, frequency offset and carrier phase.

Each plot draws an estimate from :mod:`commkit.timing`,
:mod:`commkit.frequency` or :mod:`commkit.recovery`, or recomputes what it
shows through their public functions.
"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .. import _fft
from ..backend import get_array_module, to_device
from .theme import (
    _as_channels,
    _decimate_minmax,
    _finish,
    _get_axis,
    _grid_figsize,
    _panel_grid,
    _select,
    _series_label,
    _set_eng_formatter,
)

__all__ = [
    "plot_carrier_phase_decomposition",
    "plot_carrier_phase_trajectory",
    "plot_frequency_offset_blockwise_result",
    "plot_frequency_offset_spectrum",
    "plot_mm_autocorrelation",
    "plot_pilot_phase_estimate",
    "plot_pilot_tone_phase_estimate",
    "plot_pilot_tones_phase_estimate",
    "plot_timing_correlation",
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

    Returns
    -------
    (fig, ax) or None
        The figure and its axes; ``None`` with ``show=True``, which
        shows the figure instead.
    """
    phi_c = _as_channels(phi)
    C, N = phi_c.shape
    drift_c = _as_channels(drift) if drift is not None else None

    fig, axi = _get_axis(ax)

    t = np.arange(N) / float(symbol_rate)
    for i in range(C):
        clabel = _series_label(label, f"pol {i}" if C > 1 else None, r"$\phi$ total")
        line = axi.plot(
            t,
            phi_c[i],
            alpha=0.4,
            label=clabel if drift_c is None else None,
        )[0]
        if drift_c is not None:
            drift_label = (
                f"{label} drift"
                if label
                else (f"Drift (pol {i})" if C > 1 else "Drift")
            )
            axi.plot(t, drift_c[i], color=line.get_color(), label=drift_label)

    if num_train_symbols > 0:
        axi.axvline(
            num_train_symbols / float(symbol_rate),
            color="white",
            ls="--",
            label=f"DD start ({num_train_symbols})",
        )

    _set_eng_formatter(axi, "x")
    axi.set_xlabel("Time [s]")
    axi.set_ylabel("Phase [rad]")
    axi.set_title(title)
    axi.legend(loc="best")

    return _finish((fig, axi), show, xlim=xlim, ylim=ylim, legend=legend)


# -----------------------------------------------------------------------------
# Public diagnostics: each takes the estimate whose fields it draws
# -----------------------------------------------------------------------------


def _host(a: Any) -> np.ndarray:
    return np.asarray(to_device(a, "cpu"))


def _require(estimate: Any, field: str, function_name: str, method: str) -> Any:
    value = getattr(estimate, field, None)
    if value is None:
        raise ValueError(
            f"{function_name} needs an estimate with {field} (from {method})."
        )
    return value


def plot_timing_correlation(
    estimate: Any,
    *,
    threshold: float = 3.0,
    quantity: str | None = None,
    label: str | None = None,
    legend: bool | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    ax=None,
    show: bool = False,
    title: str = "Timing Correlation",
) -> tuple[Any, Any] | None:
    """Plot the correlation of a :class:`~commkit.timing.TimingEstimate`.

    Per channel, an overall view of ``|correlation|`` and a zoom on the
    detected start, with the detection threshold ``threshold · mean|R|``
    (the peak-to-mean ``metric`` the estimator compares with it).

    Parameters
    ----------
    estimate : TimingEstimate
        From :func:`~commkit.timing.estimate_timing`.
    threshold : float, default 3.0
        Detection threshold drawn (the ``threshold`` of
        :func:`~commkit.timing.estimate_timing`).
    quantity : {"full", "detail"}, optional
        Select the full correlation or the peak detail; ``None`` draws both.
    label : str, optional
        Label for the correlation on a selected panel.
    legend : bool, optional
        Show or hide legends; ``None`` preserves the default.
    xlim, ylim : tuple of float, optional
        Limits applied to the selected panels.
    ax : matplotlib.axes.Axes or array_like of Axes, optional
        One axis per channel and selected quantity; ``None`` creates a figure.
    show : bool, default False
        Call ``plt.show()`` and return ``None``.
    title : str, default "Timing Correlation"
        Title of the figure.

    Returns
    -------
    (fig, axes) or None
        The figure and its axes; ``None`` with ``show=True``, which
        shows the figure instead.
    """
    panels = _select(quantity, ("full", "detail"), label=label, xlim=xlim, ylim=ylim)
    corr = np.atleast_2d(np.abs(_host(estimate.correlation)))
    peaks = np.atleast_1d(_host(estimate.integer)) - estimate.search_start
    corr_mag = corr
    peak_indices = peaks
    norm_factors = corr.mean(axis=-1)
    offset = estimate.search_start
    corr_mag = to_device(corr_mag, "cpu")
    peak_indices = to_device(peak_indices, "cpu").flatten()
    norm_factors = to_device(norm_factors, "cpu").flatten()

    if corr_mag.ndim == 1:
        corr_mag = corr_mag[None, :]
    C = corr_mag.shape[0]

    fig, axes = _panel_grid(ax, C, len(panels))

    for i in range(C):
        c_ch = corr_mag[i]
        pk_idx = int(peak_indices[i]) if i < len(peak_indices) else 0
        norm_val = float(norm_factors[i]) if i < len(norm_factors) else 1.0
        abs_thresh = threshold * norm_val
        metric_val = float(c_ch[pk_idx] / norm_val) if norm_val > 0 else 0.0
        ch_suffix = f" - Ch {i}" if C > 1 else ""

        if "full" in panels:
            ax1 = axes[i, panels.index("full")]
            x_all = np.arange(len(c_ch)) + offset
            ax1.plot(x_all, c_ch, label=label or f"Ch {i}  metric={metric_val:.2f}")
            ax1.axhline(float(c_ch[pk_idx]), color="r", linestyle="--", label="Peak")
            ax1.axvline(pk_idx + offset, color="r", linestyle="--")
            if abs_thresh > 0:
                ax1.axhline(abs_thresh, color="g", linestyle=":", label="Thresh")
            ax1.set_title(f"{title}{ch_suffix}")
            ax1.set_xlabel("Sample Index")
            ax1.set_ylabel("|R|")
            ax1.legend(loc="upper right")

        if "detail" in panels:
            ax2 = axes[i, panels.index("detail")]
            zoom_w = 40
            s_z = max(0, pk_idx - zoom_w)
            e_z = min(len(c_ch), pk_idx + zoom_w)
            x_zoom = np.arange(s_z, e_z) + offset
            ax2.plot(x_zoom, c_ch[s_z:e_z], label=label or "Peak area")
            ax2.axvline(
                pk_idx + offset,
                color="r",
                linestyle="--",
                label=f"Pk @ {pk_idx + offset}",
            )
            if abs_thresh > 0:
                ax2.axhline(abs_thresh, color="g", linestyle=":", label="Thresh")
            ax2.set_title(f"{title}{ch_suffix} - Detail")
            ax2.set_xlabel("Sample Index")
            ax2.set_ylabel("|R|")
            ax2.legend(loc="upper right")

    output_axes = axes if quantity is None else axes[0, 0] if C == 1 else axes[:, 0]
    return _finish((fig, output_axes), show, xlim=xlim, ylim=ylim, legend=legend)


def plot_mm_autocorrelation(
    estimate: Any,
    *,
    sampling_rate: float,
    quantity: str | None = None,
    label: str | None = None,
    legend: bool | None = None,
    annotations: bool = True,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    ax=None,
    show: bool = False,
    title: str = "FOE - Mengali-Morelli",
) -> tuple[Any, Any] | None:
    """Plot the autocorrelation of a Mengali-Morelli frequency estimate.

    Per channel: ``|R[m]|`` and the wrapped ``angle(R[m])`` against the lag,
    with the ramp ``2π·Δf·M·m/f_s`` the estimate implies.

    Parameters
    ----------
    estimate : FrequencyOffsetEstimate
        From ``estimate_frequency_offset(x, MengaliMorelli())``.
    sampling_rate : float
        Sampling rate in Hz of the estimated samples.
    quantity : {"magnitude", "phase"}, optional
        Select one panel; ``None`` draws both.
    label : str, optional
        Label for the measured values on selected panels.
    legend : bool, optional
        Show or hide legends; ``None`` preserves the default.
    annotations : bool, default True
        Draw the expected phase and the wrap boundaries; ``False`` leaves
        only the measured values, for overlays.
    xlim, ylim : tuple of float, optional
        Limits for the selected panels.
    ax : matplotlib.axes.Axes or array_like of Axes, optional
        One axis per channel and selected panel; ``None`` creates a figure.
    show : bool, default False
        Call ``plt.show()`` and return ``None``.
    title : str, default "FOE - Mengali-Morelli"
        Title of the figure.

    Returns
    -------
    (fig, axes) or None
        The figure and its axes; ``None`` with ``show=True``, which
        shows the figure instead.
    """
    panels = _select(
        quantity, ("magnitude", "phase"), label=label, xlim=xlim, ylim=ylim
    )
    r = _require(
        estimate, "autocorrelation", "plot_mm_autocorrelation()", "MengaliMorelli"
    )
    R_np = _host(r)
    f_est = np.atleast_1d(_host(estimate.value)).tolist()
    M = estimate.power or 1
    if R_np.ndim == 1:
        R_np = R_np[None, :]  # (1, L)
    C, L = R_np.shape

    if hasattr(f_est, "__len__"):
        f_ests = [float(f) for f in f_est]
    else:
        f_ests = [float(f_est)] * C

    lags = np.arange(1, L + 1)

    fig, raw_axes = _panel_grid(ax, C, len(panels))

    for c in range(C):
        amp = np.abs(R_np[c])
        theta = np.angle(R_np[c])
        f_c = f_ests[c]
        ch_suffix = f" - Ch {c}" if C > 1 else ""
        expected_phase = 2.0 * np.pi * f_c * M * lags / sampling_rate

        if "magnitude" in panels:
            ax_amp = raw_axes[c, panels.index("magnitude")]
            ax_amp.plot(lags, amp, label=label)
            ax_amp.set_xlabel("Lag m")
            ax_amp.set_ylabel("|R[m]|")
            ax_amp.set_title(f"{title}{ch_suffix}  ($\\Delta f$={f_c:.2f} Hz, M={M})")

        if "phase" in panels:
            ax_phase = raw_axes[c, panels.index("phase")]
            ax_phase.scatter(
                lags, theta, s=6, label=label or "Angle(R[m])  (wrapped)", zorder=3
            )
            if annotations:
                ax_phase.plot(
                    lags,
                    (expected_phase + np.pi) % (2 * np.pi) - np.pi,
                    color="red",
                    linestyle="--",
                    label=f"Expected $2\\pi \\cdot \\Delta f \\cdot M \\cdot m/f_s$  "
                    f"($\\Delta f$={f_c:.2f} Hz)",
                )
                ax_phase.axhline(
                    np.pi, color="gray", linestyle=":", label=r"$\pm\pi$ wrap boundary"
                )
                ax_phase.axhline(-np.pi, color="gray", linestyle=":")
            ax_phase.set_xlabel("Lag m")
            ax_phase.set_ylabel("Phase [rad]")
            ax_phase.set_ylim(-np.pi - 0.3, np.pi + 0.3)
            ax_phase.legend()

    output_axes = (
        (raw_axes[0] if C == 1 else list(raw_axes))
        if quantity is None
        else (raw_axes[0, 0] if C == 1 else raw_axes[:, 0])
    )
    return _finish((fig, output_axes), show, xlim=xlim, ylim=ylim, legend=legend)


def plot_frequency_offset_spectrum(
    estimate: Any,
    *,
    search_range: tuple[float, float] | None = None,
    label: str | None = None,
    annotations: bool = True,
    ax=None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    legend: bool | None = None,
    show: bool = False,
    title: str = "FOE - M-th Power Spectrum",
) -> tuple[Any, Any] | None:
    """Plot the M-th power spectrum of an ``MthPower`` frequency estimate.

    The axis is mapped back from ``M·Δf`` to ``Δf`` so the detected peak
    lines up with the estimate.

    Parameters
    ----------
    estimate : FrequencyOffsetEstimate
        From ``estimate_frequency_offset(x, MthPower())``.
    search_range : (float, float), optional
        The ``MthPower(search_range=)`` used, shaded.
    label : str, optional
        Label for the measured spectrum.
    annotations : bool, default True
        Draw the detected peak and the search range; ``False`` leaves only
        the measured spectrum, for overlays.
    ax : array_like of matplotlib.axes.Axes, optional
        Axes to draw on, one per channel; ``None`` creates a figure.
    xlim, ylim : tuple of float, optional
        Limits applied to the displayed channels.
    legend : bool, optional
        Show or hide the legend; ``None`` preserves the default.
    show : bool, default False
        Call ``plt.show()`` and return ``None``.
    title : str, default "FOE - M-th Power Spectrum"
        Title of the figure.

    Returns
    -------
    (fig, axes) or None
        The figure and its axes; ``None`` with ``show=True``, which
        shows the figure instead.
    """
    spectrum = _require(
        estimate, "spectrum", "plot_frequency_offset_spectrum()", "MthPower"
    )
    mag_spectrum = _host(spectrum)
    freqs = _host(estimate.spectrum_frequencies)
    M = estimate.power or 1
    f_estimates = np.atleast_1d(_host(estimate.value)).tolist()
    f_estimates = list(f_estimates)

    if mag_spectrum.ndim == 1:
        mag_spectrum = mag_spectrum[None, :]
    C = mag_spectrum.shape[0]

    if ax is None:
        fig, raw_axes = plt.subplots(1, C, figsize=_grid_figsize(1, C), squeeze=False)
        axes_list = list(raw_axes[0])
    else:
        axes_list = list(ax) if hasattr(ax, "__len__") else [ax]
        fig = axes_list[0].figure

    sort_idx = np.argsort(freqs)
    f_sorted = freqs[sort_idx]
    f_delta = f_sorted / M  # map M·Δf -> Δf

    for i in range(C):
        axi = axes_list[i]
        mag = mag_spectrum[i][sort_idx]
        f_est = float(f_estimates[i]) if i < len(f_estimates) else 0.0
        ch_suffix = f" - Ch {i}" if C > 1 else ""

        axi.plot(f_delta, mag, label=label)
        if annotations:
            axi.axvline(
                f_est,
                color="r",
                linestyle="--",
                label=rf"$\hat{{f}}$ = {f_est:.2f} Hz",
            )
            if search_range is not None:
                axi.axvspan(
                    search_range[0],
                    search_range[1],
                    alpha=0.4,
                    color="green",
                    label="Search range",
                )
        axi.set_title(f"{title}{ch_suffix}")
        axi.set_xlabel(f"$\\Delta f$ [Hz]  (÷M={M} applied)")
        axi.set_ylabel(f"|X^{M}(f)|")
        if annotations or label is not None:
            axi.legend()

    return _finish(
        (fig, axes_list[0] if C == 1 else axes_list),
        show,
        xlim=xlim,
        ylim=ylim,
        legend=legend,
    )


def plot_frequency_offset_blockwise_result(
    estimate: Any,
    *,
    num_samples: int,
    sampling_rate: float,
    quantity: str | None = None,
    label: str | None = None,
    legend: bool | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    ax=None,
    show: bool = False,
    title: str = "Block-wise FOE",
    max_points: int = 4000,
) -> tuple[Any, Any] | None:
    """Plot a blockwise frequency estimate and the correction it applies.

    Two panels: the block estimates with the interpolated ``Δf`` trajectory,
    and the integrated phase.  Both are recomputed with
    :func:`~commkit.frequency.correct_frequency_offset` on a unit record of
    ``num_samples`` samples, so they show exactly what correction applies.

    Parameters
    ----------
    estimate : FrequencyOffsetEstimate
        A blockwise estimate (``MthPower(block_size=)`` or
        ``MengaliMorelli(block_size=)``); the first channel is drawn.
    num_samples : int
        Length of the corrected record.
    sampling_rate : float
        Sampling rate in Hz.
    quantity : {"frequency", "phase"}, optional
        Select one panel; ``None`` draws both.
    label : str, optional
        Label for the trajectory on a selected panel.
    legend : bool, optional
        Show or hide legends; ``None`` preserves the default.
    xlim, ylim : tuple of float, optional
        Limits applied to a selected panel.
    ax : matplotlib.axes.Axes or array_like of Axes, optional
        One axis for a selected panel, or two for the dashboard.
    show : bool, default False
        Call ``plt.show()`` and return ``None``.
    title : str, default "Block-wise FOE"
        Title of the figure.
    max_points : int, default 4000
        Points per trace; longer traces are drawn as their min/max envelope.
        ``<= 0`` draws every point.

    Returns
    -------
    (fig, axes) or None
        The figure and its axes; ``None`` with ``show=True``, which
        shows the figure instead.
    """
    from ..frequency import correct_frequency_offset

    panels = _select(
        quantity, ("frequency", "phase"), label=label, xlim=xlim, ylim=ylim
    )

    centers = _require(
        estimate,
        "block_centers",
        "plot_frequency_offset_blockwise_result()",
        "a blockwise method (block_size=)",
    )
    # The probe lives on the estimate's device, as the samples it was made from.
    xp = get_array_module(estimate.block_values)
    rotation = correct_frequency_offset(
        xp.ones(num_samples, dtype=xp.complex128),
        estimate,
        sampling_rate=sampling_rate,
    )
    phase = -np.unwrap(np.angle(np.atleast_2d(_host(rotation))[0]))
    df_dense = np.gradient(phase) * sampling_rate / (2.0 * np.pi)
    block_values = np.atleast_2d(_host(estimate.block_values))[0]
    t_centers = centers
    df_estimates = block_values
    n_grid = np.arange(num_samples)
    phase_trajectory = phase
    t_centers = np.asarray(to_device(t_centers, "cpu"), dtype=np.float64)
    df_estimates = np.asarray(to_device(df_estimates, "cpu"), dtype=np.float64)
    n_grid = np.asarray(to_device(n_grid, "cpu"), dtype=np.float64)
    df_dense = np.asarray(to_device(df_dense, "cpu"), dtype=np.float64)
    phase_trajectory = np.asarray(to_device(phase_trajectory, "cpu"), dtype=np.float64)

    fig, grid = _panel_grid(ax, 1, len(panels))
    axes = grid[0]

    # Decimate the dense per-sample traces for fast rendering (envelope-preserving).
    nf, df_k = _decimate_minmax(n_grid, df_dense * 1e-3, max_points)
    np_, ph_deg = _decimate_minmax(n_grid, np.degrees(phase_trajectory), max_points)

    if "frequency" in panels:
        # Panel 1: frequency trajectory
        ax_f = axes[panels.index("frequency")]
        ax_f.plot(nf, df_k, label=label or r"Interpolated $\Delta f$")
        ax_f.scatter(
            t_centers,
            df_estimates * 1e-3,
            s=50,
            zorder=5,
            color="C1",
            label="Block estimate",
        )
        ax_f.set_xlabel("Sample Index")
        ax_f.set_ylabel(r"$\Delta f$ [kHz]")
        ax_f.set_title(f"{title} - Frequency")
        ax_f.legend()

    if "phase" in panels:
        # Panel 2: phase trajectory
        ax_p = axes[panels.index("phase")]
        ax_p.plot(np_, ph_deg, label=label)
        ax_p.set_xlabel("Sample Index")
        ax_p.set_ylabel("Phase [deg]")
        ax_p.set_title(f"{title} - Integrated Phase")

    output_axes = axes if quantity is None else axes[0]
    return _finish((fig, output_axes), show, xlim=xlim, ylim=ylim, legend=legend)


def plot_pilot_phase_estimate(
    estimate: Any,
    *,
    sampling_rate: float = 1.0,
    quantity: str | None = None,
    label: str | None = None,
    legend: bool | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    ax=None,
    show: bool = False,
    title: str = "Pilot Phase Estimate",
) -> tuple[Any, Any] | None:
    """Plot the pilot phases of a pilot-based estimate with their linear fit.

    For a :class:`~commkit.recovery.CarrierPhaseEstimate` (``PilotAided``) a
    second panel shows the interpolated per-symbol trajectory; for a
    :class:`~commkit.frequency.FrequencyOffsetEstimate` (``PilotSymbols``)
    the fit is labeled with the estimated offset.

    Parameters
    ----------
    estimate : CarrierPhaseEstimate or FrequencyOffsetEstimate
        With ``pilot_indices`` and ``pilot_phase``.
    sampling_rate : float, default 1.0
        Sampling rate in Hz of the pilot indices (labels the fitted slope in
        Hz; 1.0 gives cycles per sample).
    quantity : {"pilots", "trajectory"}, optional
        Select one panel; ``None`` draws all available panels. A trajectory
        is available only for a carrier-phase estimate.
    label : str, optional
        Label for the selected measurement or trajectory.
    legend : bool, optional
        Show or hide legends; ``None`` preserves the default.
    xlim, ylim : tuple of float, optional
        Limits applied to selected panels.
    ax : matplotlib.axes.Axes or array_like of Axes, optional
        One axis per channel and selected panel; ``None`` creates a figure.
    show : bool, default False
        Call ``plt.show()`` and return ``None``.
    title : str, default "Pilot Phase Estimate"
        Title of the figure.

    Returns
    -------
    (fig, axes) or None
        The figure and its axes; ``None`` with ``show=True``, which
        shows the figure instead.
    """
    from ..recovery import CarrierPhaseEstimate

    panels = _select(
        quantity, ("pilots", "trajectory"), label=label, xlim=xlim, ylim=ylim
    )
    name = "plot_pilot_phase_estimate()"
    pilot_phase = _require(estimate, "pilot_phase", name, "a pilot method")
    is_phase = isinstance(estimate, CarrierPhaseEstimate)
    pilot_indices = _host(estimate.pilot_indices)
    phi_pilots_u = _host(pilot_phase)
    phi_full = _host(estimate.value) if is_phase else None
    f_est = None if is_phase else np.atleast_1d(_host(estimate.value))
    pilot_indices = to_device(pilot_indices, "cpu").astype(float)
    if phi_pilots_u.ndim == 1:
        phi_pilots_u = phi_pilots_u[None, :]
    C, P = phi_pilots_u.shape

    f_ests: list[float] | None
    if f_est is None:
        f_ests = None  # label the fitted slope instead
    elif isinstance(f_est, (np.ndarray, list, tuple)):
        f_ests = [float(f) for f in f_est]  # type: ignore[union-attr]
    else:
        f_ests = [float(f_est)] * C  # type: ignore[arg-type]

    has_full = phi_full is not None
    if quantity == "trajectory" and not has_full:
        raise ValueError("quantity='trajectory' requires a carrier-phase estimate")
    if has_full:
        phi_full = to_device(phi_full, "cpu")
        if phi_full.ndim == 1:
            phi_full = phi_full[None, :]
        N = phi_full.shape[1]

    if not has_full:
        panels = tuple(p for p in panels if p != "trajectory")
    fig, axes = _panel_grid(ax, C, len(panels))

    t_pilots = pilot_indices / sampling_rate
    for i in range(C):
        phi_p = phi_pilots_u[i]
        ch_suffix = f" - Ch {i}" if C > 1 else ""

        # Linear fit: φ(t) = 2π·Δf·t + φ₀
        if P > 1:
            t_c = t_pilots - np.mean(t_pilots)
            t_var = float(np.dot(t_c, t_c))
            slope = (
                float(np.dot(t_c, phi_p - np.mean(phi_p)) / t_var) if t_var > 0 else 0.0
            )
            phi_fit = slope * t_pilots + (np.mean(phi_p) - slope * np.mean(t_pilots))
        else:
            slope = 0.0
            phi_fit = phi_p.copy()
        f_label = f_ests[i] if f_ests is not None else slope / (2.0 * np.pi)

        if "pilots" in panels:
            ax1 = axes[i, panels.index("pilots")]
            ax1.scatter(
                pilot_indices,
                np.degrees(phi_p),
                s=14,
                zorder=5,
                label=label or r"Pilot $\phi$ (unwrapped)",
            )
            ax1.plot(
                pilot_indices,
                np.degrees(phi_fit),
                "r--",
                label=f"Fit  $\\Delta f$={f_label:.3f} Hz",
            )
            ax1.set_title(f"{title}{ch_suffix} - Pilots")
            ax1.set_xlabel("Sample Index")
            ax1.set_ylabel("Phase [deg]")
            ax1.legend()

        if "trajectory" in panels:
            assert phi_full is not None
            ax2 = axes[i, panels.index("trajectory")]
            sym_idx = np.arange(N)
            ax2.plot(
                sym_idx,
                np.degrees(phi_full[i]),
                alpha=0.4,
                label=label or r"$\hat{\phi}$ (interpolated)",
            )
            ax2.scatter(
                pilot_indices,
                np.degrees(phi_p),
                s=10,
                color="r",
                zorder=5,
                label="Pilots",
            )
            ax2.set_title(f"{title}{ch_suffix} - Full Trajectory")
            ax2.set_xlabel("Symbol Index")
            ax2.set_ylabel("Phase [deg]")
            ax2.legend()

    output_axes = axes if quantity is None else axes[0, 0] if C == 1 else axes[:, 0]
    return _finish((fig, output_axes), show, xlim=xlim, ylim=ylim, legend=legend)


def plot_pilot_tone_phase_estimate(
    estimate: Any,
    samples: Any,
    *,
    method: Any,
    sampling_rate: float | None = None,
    quantity: str | None = None,
    label: str | None = None,
    annotations: bool = True,
    legend: bool | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    ax=None,
    show: bool = False,
    title: str = "CPR - Pilot Tone",
    max_points: int = 4000,
) -> tuple[Any, Any] | None:
    """Plot a ``PilotTone`` carrier-phase estimate against the tone spectrum.

    Two panels: ``|FFT(samples)|`` (dB) with the extraction passband
    ``f_p ± B`` and the measured tone peaks, and the recovered phase.

    Parameters
    ----------
    estimate : CarrierPhaseEstimate
        From ``estimate_carrier_phase(samples, method)``.
    samples : array_like or Signal
        The samples the estimate was made from.
    method : PilotTone
        The method used (its ``frequency`` and ``bandwidth`` are drawn).
    sampling_rate : float, optional
        Sampling rate in Hz (a fact): taken from a Signal, required for
        arrays.
    quantity : {"spectrum", "phase"}, optional
        Select one panel; ``None`` draws both.
    label : str, optional
        Label for the selected measurement or phase trajectory.
    annotations : bool, default True
        Draw the extraction window and the tone markers; ``False`` leaves
        only the measured spectrum, for overlays.
    legend : bool, optional
        Show or hide the legend; ``None`` preserves the default.
    xlim, ylim : tuple of float, optional
        Limits for a selected panel.
    ax : matplotlib.axes.Axes or array_like of Axes, optional
        One axis for a selected panel, or two for both.
    show : bool, default False
        Call ``plt.show()`` and return ``None``.
    title : str, default "CPR - Pilot Tone"
        Title of the figure.
    max_points : int, default 4000
        Points per trace; longer traces are drawn as their min/max envelope.
        ``<= 0`` draws every point.

    Returns
    -------
    (fig, axes) or None
        The figure and its axes; ``None`` with ``show=True``, which
        shows the figure instead.
    """
    from ..core._signal_adapter import adapt_signal

    _select(quantity, ("spectrum", "phase"), label=label, xlim=xlim, ylim=ylim)
    name = "plot_pilot_tone_phase_estimate()"
    tones = _require(estimate, "tone_frequencies", name, "PilotTone")
    adapter = adapt_signal(samples, function_name=name)
    fs = float(adapter.resolve_fact("sampling_rate", sampling_rate))
    if quantity == "phase":
        theta = np.atleast_2d(_host(estimate.value))
        fig, axis = _get_axis(ax)
        sample_idx = np.arange(theta.shape[1])
        for i, row in enumerate(theta):
            trace_label = _series_label(label, f"Ch {i}" if len(theta) > 1 else None)
            n_d, th_d = _decimate_minmax(sample_idx, np.degrees(row), max_points)
            axis.plot(n_d, th_d, label=trace_label)
        axis.set_title(f"{title} - Recovered phase")
        axis.set_xlabel("Sample Index")
        axis.set_ylabel("Phase [deg]")
        if len(theta) > 1:
            axis.legend()
        return _finish((fig, axis), show, xlim=xlim, ylim=ylim, legend=legend)
    x = np.atleast_2d(_host(adapter.array))
    freqs = np.fft.fftfreq(x.shape[-1], d=1.0 / fs)
    centers = np.atleast_1d(np.asarray(tones, dtype=float))[:, None]
    passband = (np.abs(freqs[None, :] - centers) <= method.bandwidth).astype(float)
    mag_spectrum = np.abs(_fft.fft(x, axis=-1))
    window = np.broadcast_to(passband, x.shape)
    f_tones = tones
    theta = _host(estimate.value)
    tone_frequency = method.frequency
    bandwidth = method.bandwidth
    freqs = np.asarray(to_device(freqs, "cpu"), dtype=float)
    mag_spectrum = to_device(mag_spectrum, "cpu")
    window = to_device(window, "cpu")
    if mag_spectrum.ndim == 1:
        mag_spectrum = mag_spectrum[None, :]
    if window.ndim == 1:
        window = window[None, :]
    if theta.ndim == 1:
        theta = theta[None, :]
    f_tones = np.atleast_1d(np.asarray(to_device(f_tones, "cpu"), dtype=float))
    C, N = mag_spectrum.shape

    if quantity == "spectrum":
        fig, ax_spec = _get_axis(ax)
    else:
        fig, grid = _panel_grid(ax, 1, 2)
        ax_spec, ax_phase = grid[0]

    order = np.argsort(freqs)
    f_sorted = freqs[order]
    eps = 1e-300

    # Panel 1 - spectrum (dB) with extraction window on a twin axis.
    # Envelope-decimate the full-resolution traces: min/max preserves the tone
    # peak and the window edges that plain striding would skip over.
    ax_win = ax_spec.twinx() if annotations else None
    for i in range(C):
        mag_db = 20.0 * np.log10(np.maximum(mag_spectrum[i][order], eps))
        curve_label = _series_label(label, f"Ch {i}" if C > 1 else None, "|X(f)|")
        f_d, mag_d = _decimate_minmax(f_sorted, mag_db, max_points)
        ax_spec.plot(f_d, mag_d, label=curve_label)
        if ax_win is not None:
            f_w, win_d = _decimate_minmax(f_sorted, window[i][order], max_points)
            ax_win.plot(f_w, win_d, color="C3", alpha=0.4)
            ax_spec.axvline(
                float(f_tones[i]),
                color="C2",
                linestyle=":",
                label=("Tone peak" if i == 0 else None),
            )
    if ax_win is not None:
        ax_spec.axvspan(
            tone_frequency - bandwidth,
            tone_frequency + bandwidth,
            color="C3",
            alpha=0.4,
            label=f"Window ±B ({bandwidth:.3g} Hz)",
        )
        ax_spec.axvline(
            tone_frequency,
            color="C1",
            linestyle="--",
            label=f"Nominal f_p ({tone_frequency:.3g} Hz)",
        )
        ax_win.set_ylim(-0.05, 1.35)
        ax_win.set_ylabel("Window W(f)", color="C3")
    ax_spec.set_title(f"{title} - Tone Spectrum")
    ax_spec.set_xlabel("Frequency [Hz]")
    ax_spec.set_ylabel("|X(f)| [dB]")
    ax_spec.legend(loc="upper left")

    if quantity == "spectrum":
        return _finish((fig, ax_spec), show, xlim=xlim, ylim=ylim, legend=legend)

    # Panel 2 - recovered phase trajectory.
    sample_idx = np.arange(theta.shape[1])
    for i in range(C):
        ph_label = f"Ch {i}" if C > 1 else None
        n_d, th_d = _decimate_minmax(sample_idx, np.degrees(theta[i]), max_points)
        ax_phase.plot(n_d, th_d, label=ph_label)
    th_mean = float(np.mean(np.degrees(theta)))
    th_std = float(np.std(np.degrees(theta)))
    ax_phase.set_title(
        f"{title} - Recovered $\\hat{{\\theta}}$  "
        f"[$\\mu$={th_mean:.1f}°, $\\sigma$={th_std:.2f}°]"
    )
    ax_phase.set_xlabel("Sample Index")
    ax_phase.set_ylabel("Phase [deg]")
    if C > 1:
        ax_phase.legend(loc="upper right")

    return _finish((fig, (ax_spec, ax_phase)), show)


def plot_pilot_tones_phase_estimate(
    estimate: Any,
    *,
    quantity: str | None = None,
    label: str | None = None,
    legend: bool | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    ax=None,
    show: bool = False,
    title: str = "CPR - Pilot Tones (MRC)",
    max_points: int = 4000,
) -> tuple[Any, Any] | None:
    """Plot a ``PilotTones`` carrier-phase estimate.

    Two panels: the tracked inter-tone differentials ``δ_k[n]`` (gated
    tones dashed) and the combined phase (the first channel).

    Parameters
    ----------
    estimate : CarrierPhaseEstimate
        From ``estimate_carrier_phase(x, PilotTones(...))``.
    quantity : {"differential", "phase"}, optional
        Select one panel; ``None`` draws both.
    label : str, optional
        Label for the selected trace or prefix for its tone curves.
    legend : bool, optional
        Show or hide legends; ``None`` preserves the default.
    xlim, ylim : tuple of float, optional
        Limits for a selected panel.
    ax : matplotlib.axes.Axes or array_like of Axes, optional
        One axis for a selected panel, or two for both.
    show : bool, default False
        Call ``plt.show()`` and return ``None``.
    title : str, default "CPR - Pilot Tones (MRC)"
        Title of the figure.
    max_points : int, default 4000
        Points per trace; longer traces are drawn as their min/max envelope.
        ``<= 0`` draws every point.

    Returns
    -------
    (fig, axes) or None
        The figure and its axes; ``None`` with ``show=True``, which
        shows the figure instead.
    """
    panels = _select(
        quantity, ("differential", "phase"), label=label, xlim=xlim, ylim=ylim
    )
    delta = _require(
        estimate,
        "differential_phase",
        "plot_pilot_tones_phase_estimate()",
        "PilotTones",
    )
    delta = _host(delta)
    phi = np.atleast_2d(_host(estimate.value))[0]
    ref = int(estimate.reference_tone)
    used = tuple(estimate.used_tones)
    delta = [np.asarray(to_device(d, "cpu"), dtype=float) for d in delta]
    phi = np.asarray(to_device(phi, "cpu"), dtype=float)
    K = len(delta)
    used_set = {int(u) for u in used}

    fig, grid = _panel_grid(ax, 1, len(panels))
    axes = list(grid[0])

    if "differential" in panels:
        ax_delta = axes[panels.index("differential")]
        # Panel 1 - per-tone slow differential (skip the reference, whose δ ≡ 0).
        n_plotted = 0
        for k in range(K):
            if k == ref:
                continue
            gated = k not in used_set
            idx = np.arange(delta[k].shape[-1])
            n_d, d_d = _decimate_minmax(idx, np.degrees(delta[k]), max_points)
            ax_delta.plot(
                n_d,
                d_d,
                linestyle="--" if gated else "-",
                label=_series_label(
                    label, f"$\\delta$ tone {k}" + (" (gated)" if gated else "")
                ),
            )
            n_plotted += 1
        ax_delta.set_title(f"{title} - Inter-tone $\\delta$  (ref = tone {ref})")
        ax_delta.set_xlabel("Sample Index")
        ax_delta.set_ylabel("Differential phase [deg]")
        if n_plotted:
            ax_delta.legend(loc="upper right")

    if "phase" in panels:
        ax_phase = axes[panels.index("phase")]
        # Panel 2 - combined common-phase track.
        idx = np.arange(phi.shape[-1])
        n_d, p_d = _decimate_minmax(idx, np.degrees(phi), max_points)
        ax_phase.plot(n_d, p_d, label=label)
        ph_std = float(np.std(np.degrees(phi)))
        ax_phase.set_title(
            f"{title} - Combined $\\hat{{\\phi}}$  [$\\sigma$={ph_std:.2f}°, used={list(used)}]"
        )
        ax_phase.set_xlabel("Sample Index")
        ax_phase.set_ylabel("Phase [deg]")

    output_axes = tuple(axes) if quantity is None else axes[0]
    return _finish((fig, output_axes), show, xlim=xlim, ylim=ylim, legend=legend)


def plot_carrier_phase_trajectory(
    phase: Any,
    *,
    num_train_symbols: int = 0,
    ax=None,
    label: str | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    legend: bool | None = None,
    show: bool = False,
    title: str = "Carrier Phase Trajectory",
) -> tuple[Any, Any] | None:
    """Plot a per-symbol carrier-phase trajectory, all channels overlaid.

    Parameters
    ----------
    phase : CarrierPhaseEstimate or array_like
        An estimate (its block centers, if any, are marked) or a phase array
        in radians, ``(N,)`` or ``(C, N)``, e.g. an equalizer's
        ``phase_trajectory``.
    num_train_symbols : int, default 0
        End of the training symbols, drawn as a dashed line.
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
    title : str, default "Carrier Phase Trajectory"
        Title of the figure.

    Returns
    -------
    (fig, ax) or None
        The figure and its axes; ``None`` with ``show=True``, which
        shows the figure instead.
    """
    from ..recovery import CarrierPhaseEstimate

    if isinstance(phase, CarrierPhaseEstimate):
        phi_full, block_centers = phase.value, phase.block_centers
    else:
        phi_full, block_centers = phase, None
    phi_full = to_device(phi_full, "cpu")
    if phi_full.ndim == 1:
        phi_full = phi_full[None, :]
    C, N = phi_full.shape

    fig, axi = _get_axis(ax)

    sym_idx = np.arange(N)
    for i in range(C):
        phi_deg = np.degrees(phi_full[i])
        curve_label = _series_label(label, f"Ch {i}" if C > 1 else None)
        axi.plot(sym_idx, phi_deg, label=curve_label)

    if block_centers is not None:
        for tc in np.asarray(to_device(block_centers, "cpu")):
            axi.axvline(tc, color="gray", alpha=0.4)

    if num_train_symbols > 0:
        axi.axvline(
            num_train_symbols,
            color="white",
            linestyle="--",
            label=f"DD start ({num_train_symbols})",
        )

    phi_all = np.degrees(phi_full)
    phi_mean = float(np.mean(phi_all))
    phi_std = float(np.std(phi_all))
    axi.set_title(f"{title}  [$\\mu$={phi_mean:.1f}°,  $\\sigma$={phi_std:.2f}°]")
    axi.set_xlabel("Symbol Index")
    axi.set_ylabel("Phase [deg]")
    if C > 1 or num_train_symbols > 0:
        axi.legend(loc="upper right")

    return _finish((fig, axi), show, xlim=xlim, ylim=ylim, legend=legend)
