"""Frequency-offset plots.

:func:`plot_mm_autocorrelation`, :func:`plot_frequency_offset_spectrum` and
:func:`plot_frequency_offset_blockwise_result` draw the fields of a
:class:`~commkit.frequency.FrequencyOffsetEstimate`.
"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from ..backend import get_array_module, to_device
from .theme import (
    _decimate_minmax,
    _device_channels,
    _finish,
    _grid_figsize,
    _host,
    _panel_grid,
    _require,
    _select,
    _set_eng_formatter,
    _trace,
)

__all__ = [
    "plot_frequency_offset_blockwise_result",
    "plot_frequency_offset_spectrum",
    "plot_mm_autocorrelation",
]


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
    max_points: int = 4000,
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
    max_points : int, default 4000
        Lags drawn: ``|R[m]|`` beyond it as its min/max envelope, the phase
        at evenly spaced lags, both reduced on the data's device.  ``<= 0``
        draws every lag.

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
    R = _device_channels(r)
    xp = get_array_module(R)
    f_ests = [float(f) for f in np.atleast_1d(_host(estimate.value))]
    M = estimate.power or 1
    C, L = R.shape
    # Phase at evenly spaced lags: a wrapped angle has no envelope.
    step = 1 if max_points <= 0 or L <= max_points else -(-L // max_points)
    lags = np.arange(1, L + 1, step)
    theta_all = _host(xp.angle(R[:, ::step]))
    amp_all = xp.abs(R)

    fig, raw_axes = _panel_grid(ax, C, len(panels))

    for c in range(C):
        theta = theta_all[c]
        f_c = f_ests[c] if c < len(f_ests) else f_ests[0]
        ch_suffix = f" - Ch {c}" if C > 1 else ""
        expected_phase = 2.0 * np.pi * f_c * M * lags / sampling_rate

        if "magnitude" in panels:
            ax_amp = raw_axes[c, panels.index("magnitude")]
            ax_amp.plot(*_trace(amp_all[c], max_points, x0=1.0), label=label)
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
    max_points: int = 4000,
) -> tuple[Any, Any] | None:
    """Plot the M-th power spectrum of an ``MthPower`` frequency estimate.

    The axis is mapped back from ``M·Δf`` to ``Δf`` so the detected peak
    lines up with the estimate.  The spectrum covers the search range the
    estimate used (all bins without one).

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
    max_points : int, default 4000
        Points per trace; longer traces are drawn as their min/max envelope,
        reduced on the data's device.  ``<= 0`` draws every point.

    Returns
    -------
    (fig, axes) or None
        The figure and its axes; ``None`` with ``show=True``, which
        shows the figure instead.
    """
    spectrum = _require(
        estimate, "spectrum", "plot_frequency_offset_spectrum()", "MthPower"
    )
    mag_spectrum = _device_channels(spectrum)
    freqs = np.asarray(estimate.spectrum_frequencies, dtype=float)
    M = estimate.power or 1
    f_estimates = [float(f) for f in np.atleast_1d(_host(estimate.value))]
    C = mag_spectrum.shape[0]
    # Evenly spaced bins in ascending frequency: the axis is f0 + k·df.
    f0 = float(freqs[0]) / M
    df = float(freqs[1] - freqs[0]) / M if freqs.size > 1 else 0.0

    if ax is None:
        fig, raw_axes = plt.subplots(1, C, figsize=_grid_figsize(1, C), squeeze=False)
        axes_list = list(raw_axes[0])
    else:
        axes_list = list(ax) if hasattr(ax, "__len__") else [ax]
        fig = axes_list[0].figure

    for i in range(C):
        axi = axes_list[i]
        f_est = float(f_estimates[i]) if i < len(f_estimates) else 0.0
        ch_suffix = f" - Ch {i}" if C > 1 else ""

        axi.plot(*_trace(mag_spectrum[i], max_points, x0=f0, dx=df), label=label)
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
        axi.set_xlabel(f"$\\Delta f$ (÷M={M} applied) [Hz]")
        _set_eng_formatter(axi, "x")
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
    nf, df_k = _decimate_minmax(n_grid, df_dense, max_points)
    np_, ph_deg = _decimate_minmax(n_grid, np.degrees(phase_trajectory), max_points)

    if "frequency" in panels:
        # Panel 1: frequency trajectory
        ax_f = axes[panels.index("frequency")]
        ax_f.plot(nf, df_k, label=label or r"Interpolated $\Delta f$")
        ax_f.scatter(
            t_centers,
            df_estimates,
            s=50,
            zorder=5,
            color="C1",
            label="Block estimate",
        )
        ax_f.set_xlabel("Sample Index")
        ax_f.set_ylabel(r"$\Delta f$ [Hz]")
        _set_eng_formatter(ax_f, "y")
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
