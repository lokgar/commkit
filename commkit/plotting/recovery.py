"""Carrier-phase recovery plots.

:func:`plot_carrier_phase_estimate` draws a
:class:`~commkit.recovery.CarrierPhaseEstimate` with the diagnostics of its
method, chosen by the estimate's class through a private table;
:func:`plot_carrier_phase_trajectory` draws a phase array (an equalizer's
``phase_trajectory``).
"""

from typing import Any

import numpy as np

from .. import _fft
from ..backend import _to_host, get_array_module, to_device
from ..core._signal_input import SignalInput
from ..frequency import PilotSlopeEstimate
from ..recovery import (
    BlockPhaseEstimate,
    CarrierPhaseEstimate,
    PilotInterpolationEstimate,
    PilotToneEstimate,
    PilotTonesEstimate,
)
from .theme import (
    _device_channels,
    _finish,
    _get_axis,
    _marker_area,
    _panel_grid,
    _reference_line,
    _select,
    _series_label,
    _set_eng_formatter,
    _trace,
)

__all__ = ["plot_carrier_phase_estimate", "plot_carrier_phase_trajectory"]


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
    phase : array_like
        Phase in radians, ``(N,)`` or ``(C, N)``, e.g. an equalizer's
        ``phase_trajectory``.  An estimate is drawn by
        :func:`plot_carrier_phase_estimate`.
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

    Raises
    ------
    TypeError
        For a :class:`~commkit.recovery.CarrierPhaseEstimate`, which
        :func:`plot_carrier_phase_estimate` draws.
    """
    if isinstance(phase, CarrierPhaseEstimate):
        raise TypeError(
            "plot_carrier_phase_trajectory() draws a phase array; draw an "
            "estimate with plot_carrier_phase_estimate()."
        )
    return _plot_trajectory(
        phase,
        None,
        num_train_symbols=num_train_symbols,
        ax=ax,
        label=label,
        xlim=xlim,
        ylim=ylim,
        legend=legend,
        show=show,
        title=title,
        max_points=max_points,
    )


def _plot_trajectory(
    phi_full: Any,
    block_centers: Any,
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
    """The trajectory panel, with the block centers of a block method."""
    phi_deg = _device_channels(phi_full) * (180.0 / np.pi)
    C = phi_deg.shape[0]

    fig, axi = _get_axis(ax)

    for i in range(C):
        curve_label = _series_label(label, f"Ch {i}" if C > 1 else None)
        axi.plot(*_trace(phi_deg[i], max_points), label=curve_label)

    if block_centers is not None:
        for tc in _to_host(block_centers):
            axi.axvline(tc, color="gray", alpha=0.4)

    if num_train_symbols > 0:
        axi.axvline(
            num_train_symbols,
            label=f"DD start ({num_train_symbols})",
            **_reference_line(),
        )

    phi_mean = float(phi_deg.mean())
    phi_std = float(phi_deg.std())
    axi.set_title(f"{title}  [$\\mu$={phi_mean:.1f}°,  $\\sigma$={phi_std:.2f}°]")
    axi.set_xlabel("Symbol Index")
    axi.set_ylabel("Phase [deg]")
    if C > 1 or num_train_symbols > 0:
        axi.legend(loc="upper right")

    return _finish((fig, axi), show, xlim=xlim, ylim=ylim, legend=legend)


def _plot_pilot_phases(
    estimate: PilotInterpolationEstimate | PilotSlopeEstimate,
    *,
    sampling_rate: float | None = None,
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
    """The pilot phases of a pilot-based estimate with their linear fit.

    ``PilotInterpolation`` adds the interpolated trajectory as a second
    panel; ``PilotSlope`` (``sampling_rate`` given) labels the fit with its
    offset in Hz, ``PilotInterpolation`` with its slope per symbol.
    """
    panels = _select(
        quantity, ("pilots", "trajectory"), label=label, xlim=xlim, ylim=ylim
    )
    pilot_phase = estimate.pilot_phase
    is_phase = isinstance(estimate, CarrierPhaseEstimate)
    pilot_indices = _to_host(estimate.pilot_indices)
    phi_pilots_u = _to_host(pilot_phase)
    phi_full = _device_channels(estimate.value) if is_phase else None
    f_est = None if is_phase else np.atleast_1d(_to_host(estimate.value))
    pilot_indices = to_device(pilot_indices, "cpu").astype(float)
    if phi_pilots_u.ndim == 1:
        phi_pilots_u = phi_pilots_u[None, :]
    C, P = phi_pilots_u.shape

    f_ests: list[float] | None
    if f_est is None:
        f_ests = None  # label the fitted slope instead
    elif isinstance(f_est, (np.ndarray, list, tuple)):
        f_ests = [float(f) for f in f_est]
    else:
        f_ests = [float(f_est)] * C

    has_full = phi_full is not None
    if quantity == "trajectory" and not has_full:
        raise ValueError("quantity='trajectory' requires a carrier-phase estimate")
    if not has_full:
        panels = tuple(p for p in panels if p != "trajectory")
    fig, axes = _panel_grid(ax, C, len(panels))

    # Pilot indices are samples of the estimated record: seconds with a
    # sampling rate (PilotSlope), symbols otherwise (PilotInterpolation).
    t_pilots = pilot_indices / (sampling_rate or 1.0)
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
        if f_ests is not None:
            fit_label = f"Fit  $\\Delta f$={f_ests[i]:.3f} Hz"
        elif sampling_rate is not None:
            fit_label = f"Fit  $\\Delta f$={slope / (2.0 * np.pi):.3f} Hz"
        else:
            fit_label = f"Fit  {np.degrees(slope):.3g}°/symbol"

        if "pilots" in panels:
            ax1 = axes[i, panels.index("pilots")]
            ax1.scatter(
                pilot_indices,
                np.degrees(phi_p),
                s=_marker_area(0.62),
                zorder=5,
                label=label or r"Pilot $\phi$ (unwrapped)",
            )
            ax1.plot(
                pilot_indices,
                np.degrees(phi_fit),
                "--",
                color="C3",
                label=fit_label,
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
                s=_marker_area(0.53),
                color="C3",
                zorder=5,
                label="Pilots",
            )
            ax2.set_title(f"{title}{ch_suffix} - Full Trajectory")
            ax2.set_xlabel("Symbol Index")
            ax2.set_ylabel("Phase [deg]")
            ax2.legend()

    output_axes = axes if quantity is None else axes[0, 0] if C == 1 else axes[:, 0]
    return _finish((fig, output_axes), show, xlim=xlim, ylim=ylim, legend=legend)


def _plot_pilot_tone(
    estimate: PilotToneEstimate,
    samples: Any,
    *,
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
    """A ``PilotTone`` estimate: ``|FFT(samples)|`` (dB) with the extraction
    passband ``f_p ± B`` and the measured tone peaks, computed and reduced
    on the samples' device, and the recovered phase.  Without ``samples``
    only the phase is drawn."""
    name = "plot_carrier_phase_estimate()"
    if samples is None:
        if quantity == "spectrum":
            raise ValueError(
                f"{name}: the spectrum of a PilotTone estimate needs the samples "
                "it was made from (samples=)."
            )
        quantity = "phase"
    _select(quantity, ("spectrum", "phase"), label=label, xlim=xlim, ylim=ylim)
    tones = estimate.tone_frequencies
    method = estimate.method
    fs = estimate.sampling_rate
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
    # A Signal must be at the estimate's rate (a conflicting property raises).
    inp = SignalInput(samples, name)
    inp.sampling_rate(fs)
    x = _device_channels(inp.samples)
    xp = get_array_module(x)
    C, N = x.shape
    mag = xp.fft.fftshift(xp.abs(_fft.fft(x, axis=-1)), axes=-1)
    mag_db = 20.0 * xp.log10(xp.maximum(mag, xp.finfo(mag.dtype).tiny))
    f_sorted = xp.fft.fftshift(xp.fft.fftfreq(N, d=1.0 / fs))
    f_tones = np.atleast_1d(_to_host(tones)).astype(float)
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

    return _finish((fig, (ax_spec, ax_phase)), show, legend=legend)


def _plot_pilot_tones(
    estimate: PilotTonesEstimate,
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
    """A ``PilotTones`` estimate: the tracked inter-tone differentials
    ``δ_k[n]`` (gated tones dashed) and the combined phase (first channel)."""
    panels = _select(
        quantity, ("differential", "phase"), label=label, xlim=xlim, ylim=ylim
    )
    delta = estimate.differential_phase
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


_PANELS: dict[type, tuple[str, ...]] = {
    PilotTonesEstimate: ("differential", "phase"),
    PilotToneEstimate: ("spectrum", "phase"),
    PilotInterpolationEstimate: ("pilots", "trajectory"),
    CarrierPhaseEstimate: ("trajectory",),
}


def plot_carrier_phase_estimate(
    estimate: CarrierPhaseEstimate,
    samples: Any = None,
    *,
    quantity: str | None = None,
    label: str | None = None,
    annotations: bool = True,
    legend: bool | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    ax=None,
    show: bool = False,
    title: str | None = None,
    max_points: int = 4000,
) -> tuple[Any, Any] | None:
    """Plot a carrier-phase estimate with the diagnostics of its method.

    The panels follow the estimate's class:

    - :class:`~commkit.recovery.CarrierPhaseEstimate` (``PLL``,
      ``ConstantPhase``) and :class:`~commkit.recovery.BlockPhaseEstimate`
      (BPS, Viterbi-Viterbi, Tikhonov): ``"trajectory"``, the phase of every
      channel, with the block centers of a block method.
    - :class:`~commkit.recovery.PilotInterpolationEstimate`: ``"pilots"``,
      the unwrapped pilot phases with their linear fit, and
      ``"trajectory"``, the interpolated phase with the pilots.
    - :class:`~commkit.recovery.PilotToneEstimate`: ``"spectrum"``,
      ``|FFT(samples)|`` in dB with the extraction window ``f_p ± B`` and the
      measured tone peaks (needs ``samples``), and ``"phase"``.
    - :class:`~commkit.recovery.PilotTonesEstimate`: ``"differential"``, the
      tracked inter-tone phases (gated tones dashed), and ``"phase"``, the
      combined phase.

    Parameters
    ----------
    estimate : CarrierPhaseEstimate
        From :func:`commkit.recovery.estimate_carrier_phase`.
    samples : array_like or Signal, optional
        The samples a ``PilotTone`` estimate was made from, for its spectrum
        (a Signal must have the estimate's sampling rate); ignored for other
        estimates.  Without them a ``PilotTone`` estimate draws its phase.
    quantity : str, optional
        Select one panel of those above; ``None`` draws all available.
    label : str, optional
        Label for the trace of a selected panel (the trajectory of a
        one-panel estimate).
    annotations : bool, default True
        Draw the extraction window and tone markers of a ``PilotTone``
        spectrum; ``False`` leaves the measured spectrum, for overlays.
    legend : bool, optional
        Show or hide legends; ``None`` preserves the default.
    xlim, ylim : tuple of float, optional
        Limits for a selected panel.
    ax : matplotlib.axes.Axes or array_like of Axes, optional
        One axis per drawn panel (per channel and panel for pilot
        interpolation); ``None`` creates a figure.
    show : bool, default False
        Call ``plt.show()`` and return ``None``.
    title : str, optional
        Title of the figure; ``None`` names the method.
    max_points : int, default 4000
        Points per trace; longer traces are drawn as their min/max envelope,
        reduced on the data's device.  ``<= 0`` draws every point.

    Returns
    -------
    (fig, ax) or (fig, axes) or None
        The figure and its axes (an array for several panels); ``None`` with
        ``show=True``, which shows the figure instead.

    Raises
    ------
    TypeError
        If ``estimate`` is not a ``CarrierPhaseEstimate``.
    ValueError
        For a ``quantity`` the estimate does not draw, or the spectrum of a
        ``PilotTone`` estimate without ``samples``.
    """
    if not isinstance(estimate, CarrierPhaseEstimate):
        raise TypeError(
            "plot_carrier_phase_estimate(): estimate must be a CarrierPhaseEstimate, "
            f"got {type(estimate).__name__}."
        )
    kind = next(k for k in _PANELS if isinstance(estimate, k))
    if quantity is not None and quantity not in _PANELS[kind]:
        raise ValueError(
            f"plot_carrier_phase_estimate(): a {type(estimate).__name__} draws "
            f"{', '.join(_PANELS[kind])}, got quantity={quantity!r}."
        )
    common = dict(
        quantity=quantity,
        label=label,
        legend=legend,
        xlim=xlim,
        ylim=ylim,
        ax=ax,
        show=show,
        max_points=max_points,
    )
    if kind is PilotTonesEstimate:
        assert isinstance(estimate, PilotTonesEstimate)
        return _plot_pilot_tones(
            estimate, title=title or "CPR - Pilot Tones (MRC)", **common
        )
    if kind is PilotToneEstimate:
        assert isinstance(estimate, PilotToneEstimate)
        return _plot_pilot_tone(
            estimate,
            samples,
            annotations=annotations,
            title=title or "CPR - Pilot Tone",
            **common,
        )
    if kind is PilotInterpolationEstimate:
        assert isinstance(estimate, PilotInterpolationEstimate)
        return _plot_pilot_phases(
            estimate, title=title or "Pilot Phase Estimate", **common
        )
    centers = (
        estimate.block_centers if isinstance(estimate, BlockPhaseEstimate) else None
    )
    del common["quantity"]
    return _plot_trajectory(
        estimate.value,
        centers,
        title=title or "Carrier Phase Trajectory",
        **common,
    )
