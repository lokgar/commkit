"""Carrier-phase recovery plots.

:func:`plot_carrier_phase_trajectory`, :func:`plot_pilot_phase_estimate`,
:func:`plot_pilot_tone_phase_estimate` and
:func:`plot_pilot_tones_phase_estimate` draw a
:class:`~commkit.recovery.CarrierPhaseEstimate` or a phase trajectory.
"""

from typing import Any

import numpy as np

from .. import _fft
from ..backend import get_array_module, to_device
from .theme import (
    _device_channels,
    _finish,
    _get_axis,
    _host,
    _panel_grid,
    _require,
    _select,
    _series_label,
    _set_eng_formatter,
    _trace,
)

__all__ = [
    "plot_carrier_phase_trajectory",
    "plot_pilot_phase_estimate",
    "plot_pilot_tone_phase_estimate",
    "plot_pilot_tones_phase_estimate",
]


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
    max_points: int = 4000,
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
    max_points : int, default 4000
        Points per trace; longer traces are drawn as their min/max envelope,
        reduced on the data's device.  ``<= 0`` draws every point.

    Returns
    -------
    (fig, ax) or None
        The figure and its axes; ``None`` with ``show=True``, which
        shows the figure instead.
    """
    from ..recovery import CarrierPhaseEstimate

    if isinstance(phase, CarrierPhaseEstimate):
        # Block methods only.
        phi_full = phase.value
        block_centers = getattr(phase, "block_centers", None)
    else:
        phi_full, block_centers = phase, None
    phi_deg = _device_channels(phi_full) * (180.0 / np.pi)
    C = phi_deg.shape[0]

    fig, axi = _get_axis(ax)

    for i in range(C):
        curve_label = _series_label(label, f"Ch {i}" if C > 1 else None)
        axi.plot(*_trace(phi_deg[i], max_points), label=curve_label)

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

    phi_mean = float(phi_deg.mean())
    phi_std = float(phi_deg.std())
    axi.set_title(f"{title}  [$\\mu$={phi_mean:.1f}°,  $\\sigma$={phi_std:.2f}°]")
    axi.set_xlabel("Symbol Index")
    axi.set_ylabel("Phase [deg]")
    if C > 1 or num_train_symbols > 0:
        axi.legend(loc="upper right")

    return _finish((fig, axi), show, xlim=xlim, ylim=ylim, legend=legend)


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
    max_points: int = 4000,
) -> tuple[Any, Any] | None:
    """Plot the pilot phases of a pilot-based estimate with their linear fit.

    For a :class:`~commkit.recovery.PilotPhaseEstimate` (``PilotInterpolation``) a
    second panel shows the interpolated per-symbol trajectory; for a
    :class:`~commkit.frequency.PilotSlopeEstimate` (``PilotSlope``)
    the fit is labeled with the estimated offset.

    Parameters
    ----------
    estimate : PilotPhaseEstimate or PilotSlopeEstimate
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
    max_points : int, default 4000
        Points of the trajectory; a longer one is drawn as its min/max
        envelope, reduced on the data's device.  ``<= 0`` draws every point.

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
    phi_full = _device_channels(estimate.value) if is_phase else None
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
            ax2.plot(
                *_trace(phi_full[i] * (180.0 / np.pi), max_points),
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
    ``f_p ± B`` and the measured tone peaks, and the recovered phase.  The
    spectrum is computed and reduced on the samples' device.

    Parameters
    ----------
    estimate : TonePhaseEstimate
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
        Points per trace; longer traces are drawn as their min/max envelope,
        reduced on the data's device.  ``<= 0`` draws every point.

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
    theta_deg = _device_channels(estimate.value) * (180.0 / np.pi)
    if quantity == "phase":
        fig, axis = _get_axis(ax)
        for i, row in enumerate(theta_deg):
            trace_label = _series_label(
                label, f"Ch {i}" if len(theta_deg) > 1 else None
            )
            axis.plot(*_trace(row, max_points), label=trace_label)
        axis.set_title(f"{title} - Recovered phase")
        axis.set_xlabel("Sample Index")
        axis.set_ylabel("Phase [deg]")
        if len(theta_deg) > 1:
            axis.legend()
        return _finish((fig, axis), show, xlim=xlim, ylim=ylim, legend=legend)
    # The spectrum on the samples' device, in ascending frequency; only the
    # drawn envelope is transferred.
    x = _device_channels(adapter.array)
    xp = get_array_module(x)
    C, N = x.shape
    mag = xp.fft.fftshift(xp.abs(_fft.fft(x, axis=-1)), axes=-1)
    mag_db = 20.0 * xp.log10(xp.maximum(mag, xp.finfo(mag.dtype).tiny))
    f_sorted = xp.fft.fftshift(xp.fft.fftfreq(N, d=1.0 / fs))
    f_tones = np.atleast_1d(_host(tones)).astype(float)
    tone_frequency = method.frequency
    bandwidth = method.bandwidth

    if quantity == "spectrum":
        fig, ax_spec = _get_axis(ax)
    else:
        fig, grid = _panel_grid(ax, 1, 2)
        ax_spec, ax_phase = grid[0]

    # Panel 1 - spectrum (dB) with extraction window on a twin axis.
    # Envelope-decimate the full-resolution traces: min/max preserves the tone
    # peak and the window edges that plain striding would skip over.
    ax_win = ax_spec.twinx() if annotations else None
    for i in range(C):
        curve_label = _series_label(label, f"Ch {i}" if C > 1 else None, "|X(f)|")
        ax_spec.plot(*_trace(mag_db[i], max_points, x=f_sorted), label=curve_label)
        if ax_win is not None:
            center = float(f_tones[min(i, len(f_tones) - 1)])
            window = (xp.abs(f_sorted - center) <= bandwidth).astype(xp.float64)
            ax_win.plot(*_trace(window, max_points, x=f_sorted), color="C3", alpha=0.4)
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
    _set_eng_formatter(ax_spec, "x")
    ax_spec.set_ylabel("|X(f)| [dB]")
    ax_spec.legend(loc="upper left")

    if quantity == "spectrum":
        return _finish((fig, ax_spec), show, xlim=xlim, ylim=ylim, legend=legend)

    # Panel 2 - recovered phase trajectory.
    for i, row in enumerate(theta_deg):
        ph_label = f"Ch {i}" if len(theta_deg) > 1 else None
        ax_phase.plot(*_trace(row, max_points), label=ph_label)
    th_mean = float(theta_deg.mean())
    th_std = float(theta_deg.std())
    ax_phase.set_title(
        f"{title} - Recovered $\\hat{{\\theta}}$  "
        f"[$\\mu$={th_mean:.1f}°, $\\sigma$={th_std:.2f}°]"
    )
    ax_phase.set_xlabel("Sample Index")
    ax_phase.set_ylabel("Phase [deg]")
    if len(theta_deg) > 1:
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
    estimate : MultiTonePhaseEstimate
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
        Points per trace; longer traces are drawn as their min/max envelope,
        reduced on the data's device.  ``<= 0`` draws every point.

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
    phi_deg = _device_channels(estimate.value)[0] * (180.0 / np.pi)
    ref = int(estimate.reference_tone)
    used = tuple(estimate.used_tones)
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
            ax_delta.plot(
                *_trace(delta[k] * (180.0 / np.pi), max_points),
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
        ax_phase.plot(*_trace(phi_deg, max_points), label=label)
        ph_std = float(phi_deg.std())
        ax_phase.set_title(
            f"{title} - Combined $\\hat{{\\phi}}$  [$\\sigma$={ph_std:.2f}°, used={list(used)}]"
        )
        ax_phase.set_xlabel("Sample Index")
        ax_phase.set_ylabel("Phase [deg]")

    output_axes = tuple(axes) if quantity is None else axes[0]
    return _finish((fig, output_axes), show, xlim=xlim, ylim=ylim, legend=legend)
