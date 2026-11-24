"""Frequency-offset plots.

:func:`plot_frequency_offset_estimate` draws a
:class:`~commkit.frequency.FrequencyOffsetEstimate` with the diagnostics of
its method, chosen by the estimate's class through a private table.
"""

from typing import Any

import numpy as np

from ..backend import _to_host, get_array_module
from ..frequency import (
    BlockwiseFrequencyEstimate,
    FrequencyOffsetEstimate,
    MengaliMorelliEstimate,
    MthPowerEstimate,
    PilotSlopeEstimate,
    correct_frequency_offset,
)
from .recovery import _plot_pilot_phases
from .theme import (
    _background,
    _device_channels,
    _finish,
    _marker_area,
    _panel_grid,
    _reference_line,
    _select,
    _set_eng_formatter,
    _trace,
)

__all__ = ["plot_frequency_offset_estimate"]


def _plot_mm_autocorrelation(
    estimate: MengaliMorelliEstimate,
    *,
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
    """A Mengali-Morelli estimate: per channel ``|R[m]|`` and the wrapped
    ``angle(R[m])`` against the lag, with the ramp ``2π·Δf·M·m/f_s`` the
    estimate implies."""
    panels = _select(
        quantity, ("magnitude", "phase"), label=label, xlim=xlim, ylim=ylim
    )
    sampling_rate = estimate.sampling_rate
    R = _device_channels(estimate.autocorrelation)
    xp = get_array_module(R)
    f_ests = [float(f) for f in np.atleast_1d(_to_host(estimate.value))]
    M = estimate.power
    C, L = R.shape
    # Phase at evenly spaced lags: a wrapped angle has no envelope.
    step = 1 if max_points <= 0 or L <= max_points else -(-L // max_points)
    lags = np.arange(1, L + 1, step)
    theta_all = _to_host(xp.angle(R[:, ::step]))
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
                lags,
                theta,
                s=_marker_area(0.41),
                label=label or "Angle(R[m])  (wrapped)",
                zorder=3,
            )
            if annotations:
                ax_phase.plot(
                    lags,
                    (expected_phase + np.pi) % (2 * np.pi) - np.pi,
                    color="C3",
                    linestyle="--",
                    label=f"Expected $2\\pi \\cdot \\Delta f \\cdot M \\cdot m/f_s$  "
                    f"($\\Delta f$={f_c:.2f} Hz)",
                )
                ax_phase.axhline(
                    np.pi, label=r"$\pm\pi$ wrap boundary", **_reference_line()
                )
                ax_phase.axhline(-np.pi, **_reference_line())
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


def _plot_mth_power_spectrum(
    estimate: MthPowerEstimate,
    *,
    quantity: str | None = None,
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
    """An ``MthPower`` estimate: the M-th power spectrum over the bins the
    estimate searched, its axis mapped back from ``M·Δf`` to ``Δf`` so the
    detected peak lines up with the estimate."""
    mag_spectrum = _device_channels(estimate.spectrum)
    freqs = np.asarray(estimate.spectrum_frequencies, dtype=float)
    M = estimate.power
    f_estimates = [float(f) for f in np.atleast_1d(_to_host(estimate.value))]
    C = mag_spectrum.shape[0]
    # Evenly spaced bins in ascending frequency: the axis is f0 + k·df.
    f0 = float(freqs[0]) / M
    df = float(freqs[1] - freqs[0]) / M if freqs.size > 1 else 0.0

    fig, grid = _panel_grid(ax, 1, C)
    axes_list = list(grid[0])

    for i in range(C):
        axi = axes_list[i]
        f_est = float(f_estimates[i]) if i < len(f_estimates) else 0.0
        ch_suffix = f" - Ch {i}" if C > 1 else ""

        axi.plot(*_trace(mag_spectrum[i], max_points, x0=f0, dx=df), label=label)
        if annotations:
            axi.axvline(
                f_est,
                color="C3",
                linestyle="--",
                label=rf"$\hat{{f}}$ = {f_est:.2f} Hz",
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


def _plot_blockwise(
    estimate: BlockwiseFrequencyEstimate,
    *,
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
    """A blockwise estimate, every channel on the same axes: the block
    estimates with the interpolated ``Δf`` trajectory, and the integrated
    phase, both recomputed by
    :func:`~commkit.frequency.correct_frequency_offset` on a unit record of
    the estimate's length, so they show exactly the correction it applies."""
    panels = _select(
        quantity, ("frequency", "phase"), label=label, xlim=xlim, ylim=ylim
    )

    num_samples, fs = estimate.num_samples, estimate.sampling_rate
    # The probe lives on the estimate's device, as the samples it was made
    # from; unwrapped, differentiated and decimated there, so only the drawn
    # points cross to the host.
    values = _device_channels(estimate.block_values)  # (C, B)
    xp = get_array_module(values)
    C = values.shape[0]
    rotation = correct_frequency_offset(
        xp.ones((C, num_samples), dtype=xp.complex128),
        estimate,
        sampling_rate=fs,
    )
    phase = -xp.unwrap(xp.angle(rotation), axis=-1)  # (C, N)
    df_dense = xp.gradient(phase, axis=-1) * (fs / (2.0 * np.pi))
    t_centers = _to_host(estimate.block_centers, np.float64) / fs
    df_blocks = _to_host(values, np.float64)

    fig, grid = _panel_grid(ax, 1, len(panels))
    axes = grid[0]
    for c in range(C):
        name = label if C == 1 else f"{label or 'Channel'} {c}"
        if "frequency" in panels:
            ax_f = axes[panels.index("frequency")]
            ax_f.plot(
                *_trace(df_dense[c], max_points, dx=1.0 / fs),
                color=f"C{c}",
                label=name or r"Interpolated $\Delta f$",
            )
            ax_f.scatter(
                t_centers,
                df_blocks[c],
                s=_marker_area(1.18),
                zorder=5,
                color=f"C{c}",
                edgecolors=_background(),
                label="Block estimate" if C == 1 else None,
            )
        if "phase" in panels:
            ax_p = axes[panels.index("phase")]
            ax_p.plot(
                *_trace(phase[c] * (180.0 / np.pi), max_points, dx=1.0 / fs),
                color=f"C{c}",
                label=name,
            )

    if "frequency" in panels:
        ax_f = axes[panels.index("frequency")]
        ax_f.set_xlabel("Time [s]")
        ax_f.set_ylabel(r"$\Delta f$ [Hz]")
        _set_eng_formatter(ax_f, "x")
        _set_eng_formatter(ax_f, "y")
        ax_f.set_title(f"{title} - Frequency")
        ax_f.legend()
    if "phase" in panels:
        ax_p = axes[panels.index("phase")]
        ax_p.set_xlabel("Time [s]")
        ax_p.set_ylabel("Phase [deg]")
        _set_eng_formatter(ax_p, "x")
        ax_p.set_title(f"{title} - Integrated Phase")

    output_axes = axes if quantity is None else axes[0]
    return _finish((fig, output_axes), show, xlim=xlim, ylim=ylim, legend=legend)


_PANELS: dict[type, tuple[str, ...]] = {
    MthPowerEstimate: ("spectrum",),
    MengaliMorelliEstimate: ("magnitude", "phase"),
    PilotSlopeEstimate: ("pilots",),
    BlockwiseFrequencyEstimate: ("frequency", "phase"),
}


def plot_frequency_offset_estimate(
    estimate: FrequencyOffsetEstimate,
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
    """Plot a frequency-offset estimate with the diagnostics of its method.

    The panels follow the estimate's class:

    - :class:`~commkit.frequency.MthPowerEstimate`: ``"spectrum"``, the
      M-th power spectrum over the searched bins, its axis mapped back to
      ``Δf``, with the detected peak; one panel per channel.
    - :class:`~commkit.frequency.MengaliMorelliEstimate`: ``"magnitude"``
      and ``"phase"`` of the autocorrelation ``R[m]`` against the lag, with
      the ramp ``2π·Δf·M·m/f_s`` the estimate implies; per channel.
    - :class:`~commkit.frequency.PilotSlopeEstimate`: ``"pilots"``, the
      unwrapped pilot phases with the fit labeled by the offset; per
      channel.
    - :class:`~commkit.frequency.BlockwiseFrequencyEstimate`:
      ``"frequency"``, the block estimates and the interpolated ``Δf``, and
      ``"phase"``, the integrated phase the correction removes (first
      channel).

    The estimate carries the sampling rate (and a blockwise one the record
    length) its axes need.

    Parameters
    ----------
    estimate : FrequencyOffsetEstimate
        From :func:`commkit.frequency.estimate_frequency_offset`.
    quantity : str, optional
        Select one panel of those above; ``None`` draws all.
    label : str, optional
        Label for the measured trace of a selected panel.
    annotations : bool, default True
        Draw the detected peak or the expected phase ramp and its wrap
        boundaries; ``False`` leaves the measured values, for overlays.
    legend : bool, optional
        Show or hide legends; ``None`` preserves the default.
    xlim, ylim : tuple of float, optional
        Limits for a selected panel (every channel's).
    ax : matplotlib.axes.Axes or array_like of Axes, optional
        One axis per drawn panel; ``None`` creates a figure.
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
        If ``estimate`` is not a ``FrequencyOffsetEstimate``.
    ValueError
        For an estimate without diagnostics (``SpectralPeak``'s, or one
        reduced by :meth:`~commkit.frequency.FrequencyOffsetEstimate.combined`),
        or a ``quantity`` it does not draw.
    """
    name = "plot_frequency_offset_estimate()"
    if not isinstance(estimate, FrequencyOffsetEstimate):
        raise TypeError(
            f"{name}: estimate must be a FrequencyOffsetEstimate, got "
            f"{type(estimate).__name__}."
        )
    kind = next((k for k in _PANELS if isinstance(estimate, k)), None)
    if kind is None:
        raise ValueError(
            f"{name}: a {type(estimate).__name__} holds only the offset, no "
            "diagnostics to draw (SpectralPeak, or a combined() estimate)."
        )
    if quantity is not None and quantity not in _PANELS[kind]:
        raise ValueError(
            f"{name}: a {type(estimate).__name__} draws {', '.join(_PANELS[kind])}, "
            f"got quantity={quantity!r}."
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
    if kind is MthPowerEstimate:
        assert isinstance(estimate, MthPowerEstimate)
        return _plot_mth_power_spectrum(
            estimate,
            annotations=annotations,
            title=title or "FOE - M-th Power Spectrum",
            **common,
        )
    if kind is MengaliMorelliEstimate:
        assert isinstance(estimate, MengaliMorelliEstimate)
        return _plot_mm_autocorrelation(
            estimate,
            annotations=annotations,
            title=title or "FOE - Mengali-Morelli",
            **common,
        )
    if kind is PilotSlopeEstimate:
        assert isinstance(estimate, PilotSlopeEstimate)
        return _plot_pilot_phases(
            estimate,
            sampling_rate=estimate.sampling_rate,
            title=title or "FOE - Pilot Slope",
            **common,
        )
    assert isinstance(estimate, BlockwiseFrequencyEstimate)
    return _plot_blockwise(estimate, title=title or "Block-wise FOE", **common)
