"""Timing-synchronization plots: :func:`plot_timing_correlation`.

Draws the correlation of a :class:`~commkit.timing.TimingEstimate`.
"""

from typing import Any

import numpy as np

from ..backend import get_array_module
from .theme import (
    _device_channels,
    _finish,
    _host,
    _panel_grid,
    _select,
    _trace,
)

__all__ = [
    "plot_timing_correlation",
]


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
    max_points: int = 4000,
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
    max_points : int, default 4000
        Points of the full correlation; longer traces are drawn as their min/max envelope,
        reduced on the data's device.  ``<= 0`` draws every point.

    Returns
    -------
    (fig, axes) or None
        The figure and its axes; ``None`` with ``show=True``, which
        shows the figure instead.
    """
    panels = _select(quantity, ("full", "detail"), label=label, xlim=xlim, ylim=ylim)
    corr = _device_channels(estimate.correlation)
    xp = get_array_module(corr)
    corr = xp.abs(corr)  # on the device; only drawn points are transferred
    C = corr.shape[0]
    offset = estimate.search_start
    peak_indices = np.atleast_1d(_host(estimate.integer)).ravel() - offset
    norm_factors = _host(corr.mean(axis=-1)).ravel()

    fig, axes = _panel_grid(ax, C, len(panels))

    zoom_w = 40
    for i in range(C):
        c_ch = corr[i]
        pk_idx = int(peak_indices[i]) if i < len(peak_indices) else 0
        norm_val = float(norm_factors[i]) if i < len(norm_factors) else 1.0
        abs_thresh = threshold * norm_val
        s_z = max(0, pk_idx - zoom_w)
        e_z = min(c_ch.shape[-1], pk_idx + zoom_w)
        zoom = _host(c_ch[s_z:e_z])  # holds the peak
        peak_val = float(zoom[pk_idx - s_z])
        metric_val = peak_val / norm_val if norm_val > 0 else 0.0
        ch_suffix = f" - Ch {i}" if C > 1 else ""

        if "full" in panels:
            ax1 = axes[i, panels.index("full")]
            ax1.plot(
                *_trace(c_ch, max_points, x0=offset),
                label=label or f"Ch {i}  metric={metric_val:.2f}",
            )
            ax1.axhline(peak_val, color="r", linestyle="--", label="Peak")
            ax1.axvline(pk_idx + offset, color="r", linestyle="--")
            if abs_thresh > 0:
                ax1.axhline(abs_thresh, color="g", linestyle=":", label="Thresh")
            ax1.set_title(f"{title}{ch_suffix}")
            ax1.set_xlabel("Sample Index")
            ax1.set_ylabel("|R|")
            ax1.legend(loc="upper right")

        if "detail" in panels:
            ax2 = axes[i, panels.index("detail")]
            x_zoom = np.arange(s_z, e_z) + offset
            ax2.plot(x_zoom, zoom, label=label or "Peak area")
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
