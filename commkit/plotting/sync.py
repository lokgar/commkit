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
        clabel = f"pol {i}" if C > 1 else r"$\phi$ total"
        axi.plot(
            t,
            phi_c[i],
            color=f"C{i}",
            alpha=0.4,
            label=clabel if drift_c is None else None,
        )
        if drift_c is not None:
            axi.plot(
                t,
                drift_c[i],
                color=f"C{i}",
                label=f"Drift (pol {i})" if C > 1 else "Drift",
            )

    if num_train_symbols > 0:
        axi.axvline(
            num_train_symbols / float(symbol_rate),
            color="white",
            ls="--",
            label=f"DD start ({num_train_symbols})",
        )

    _set_eng_formatter(axi, "x", "s")
    axi.set_xlabel("Time [s]")
    axi.set_ylabel("Phase [rad]")
    axi.set_title(title)
    axi.legend(loc="best")

    return _finish((fig, axi), show)


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
    ax : array_like of matplotlib.axes.Axes, optional
        Axes to draw on, ``(C, 2)``: the overall view and the zoom of each
        channel; ``None`` creates a figure.
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

    if ax is None:
        fig, axes = plt.subplots(C, 2, figsize=_grid_figsize(C, 2), squeeze=False)
    else:
        axes = ax
        fig = axes[0][0].figure

    for i in range(C):
        ax1 = axes[i][0]
        ax2 = axes[i][1]

        c_ch = corr_mag[i]
        pk_idx = int(peak_indices[i]) if i < len(peak_indices) else 0
        norm_val = float(norm_factors[i]) if i < len(norm_factors) else 1.0
        abs_thresh = threshold * norm_val
        metric_val = float(c_ch[pk_idx] / norm_val) if norm_val > 0 else 0.0
        ch_suffix = f" - Ch {i}" if C > 1 else ""

        x_all = np.arange(len(c_ch)) + offset
        ax1.plot(x_all, c_ch, label=f"Ch {i}  metric={metric_val:.2f}")
        ax1.axhline(float(c_ch[pk_idx]), color="r", linestyle="--", label="Peak")
        ax1.axvline(pk_idx + offset, color="r", linestyle="--")
        if abs_thresh > 0:
            ax1.axhline(abs_thresh, color="g", linestyle=":", label="Thresh")
        ax1.set_title(f"{title}{ch_suffix}")
        ax1.set_xlabel("Sample Index")
        ax1.set_ylabel("|R|")
        ax1.legend(loc="upper right")

        zoom_w = 40
        s_z = max(0, pk_idx - zoom_w)
        e_z = min(len(c_ch), pk_idx + zoom_w)
        x_zoom = np.arange(s_z, e_z) + offset
        ax2.plot(x_zoom, c_ch[s_z:e_z], label="Peak area")
        ax2.axvline(
            pk_idx + offset, color="r", linestyle="--", label=f"Pk @ {pk_idx + offset}"
        )
        if abs_thresh > 0:
            ax2.axhline(abs_thresh, color="g", linestyle=":", label="Thresh")
        ax2.set_title(f"{title}{ch_suffix} - Detail")
        ax2.set_xlabel("Sample Index")
        ax2.set_ylabel("|R|")
        ax2.legend(loc="upper right")

    return _finish((fig, axes), show)


def plot_mm_autocorrelation(
    estimate: Any,
    *,
    sampling_rate: float,
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
    ax : array_like of matplotlib.axes.Axes, optional
        Axes to draw on, two (magnitude and angle) or ``(C, 2)``; ``None``
        creates a figure.
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

    if ax is None:
        if C == 1:
            fig, raw_axes = plt.subplots(1, 2, figsize=_grid_figsize(1, 2))
            axes_per_ch = [raw_axes]  # list of (ax_amp, ax_phase)
        else:
            fig, raw_axes = plt.subplots(
                C, 2, figsize=_grid_figsize(C, 2), squeeze=False
            )
            axes_per_ch = [(raw_axes[c, 0], raw_axes[c, 1]) for c in range(C)]
    else:
        ax_arr = np.asarray(ax, dtype=object)
        if ax_arr.ndim == 1 and ax_arr.shape[0] == 2:
            axes_per_ch = [ax_arr]
        else:
            axes_per_ch = [ax_arr[c] for c in range(C)]
        fig = axes_per_ch[0][0].figure

    for c in range(C):
        ax_amp, ax_phase = axes_per_ch[c]
        amp = np.abs(R_np[c])
        theta = np.angle(R_np[c])
        f_c = f_ests[c]
        ch_suffix = f" - Ch {c}" if C > 1 else ""
        expected_phase = 2.0 * np.pi * f_c * M * lags / sampling_rate

        ax_amp.plot(lags, amp, color=f"C{c}")
        ax_amp.set_xlabel("Lag m")
        ax_amp.set_ylabel("|R[m]|")
        ax_amp.set_title(f"{title}{ch_suffix}  ($\\Delta f$={f_c:.2f} Hz, M={M})")

        ax_phase.scatter(
            lags, theta, s=6, color=f"C{c}", label="Angle(R[m])  (wrapped)", zorder=3
        )
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

    return _finish((fig, axes_per_ch[0] if C == 1 else axes_per_ch), show)


def plot_frequency_offset_spectrum(
    estimate: Any,
    *,
    search_range: tuple[float, float] | None = None,
    ax=None,
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
    ax : array_like of matplotlib.axes.Axes, optional
        Axes to draw on, one per channel; ``None`` creates a figure.
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

        axi.plot(f_delta, mag, color="C0")
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
        axi.legend()

    return _finish((fig, axes_list[0] if C == 1 else axes_list), show)


def plot_frequency_offset_blockwise_result(
    estimate: Any,
    *,
    num_samples: int,
    sampling_rate: float,
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
    ax : array_like of matplotlib.axes.Axes, optional
        Axes to draw on, two; ``None`` creates a figure.
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

    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=_grid_figsize(1, 2))
    else:
        axes = np.asarray(ax).flatten()[:2]
        fig = axes[0].figure

    # Decimate the dense per-sample traces for fast rendering (envelope-preserving).
    nf, df_k = _decimate_minmax(n_grid, df_dense * 1e-3, max_points)
    np_, ph_deg = _decimate_minmax(n_grid, np.degrees(phase_trajectory), max_points)

    # Panel 1: frequency trajectory
    ax_f = axes[0]
    ax_f.plot(nf, df_k, label=r"Interpolated $\Delta f$")
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

    # Panel 2: phase trajectory
    ax_p = axes[1]
    ax_p.plot(np_, ph_deg)
    ax_p.set_xlabel("Sample Index")
    ax_p.set_ylabel("Phase [deg]")
    ax_p.set_title(f"{title} - Integrated Phase")

    return _finish((fig, axes), show)


def plot_pilot_phase_estimate(
    estimate: Any,
    *,
    sampling_rate: float = 1.0,
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
    ax : array_like of matplotlib.axes.Axes, optional
        Axes to draw on, one per panel; ``None`` creates a figure.
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
    if has_full:
        phi_full = to_device(phi_full, "cpu")
        if phi_full.ndim == 1:
            phi_full = phi_full[None, :]
        N = phi_full.shape[1]

    n_cols = 2 if has_full else 1
    axes: Any = None
    if ax is None:
        fig, axes = plt.subplots(
            C, n_cols, figsize=_grid_figsize(C, n_cols), squeeze=False
        )
    else:
        # ax expected as (C, n_cols) sequence - use list of lists to avoid
        # np.reshape() which fails when ax contains non-array objects
        if hasattr(ax, "__len__") and hasattr(ax[0], "__len__"):
            axes = ax  # already 2-D list
        elif hasattr(ax, "__len__"):
            axes = [[a] for a in ax]  # 1-D list -> wrap each in a row
        else:
            axes = [[ax]]
        fig = axes[0][0].figure

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

        ax1 = axes[i][0]
        ax1.scatter(
            pilot_indices,
            np.degrees(phi_p),
            s=14,
            zorder=5,
            label=r"Pilot $\phi$ (unwrapped)",
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

        if has_full:
            assert phi_full is not None
            ax2 = axes[i][1]
            sym_idx = np.arange(N)
            ax2.plot(
                sym_idx,
                np.degrees(phi_full[i]),
                alpha=0.4,
                label=r"$\hat{\phi}$ (interpolated)",
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

    return _finish((fig, axes), show)


def plot_pilot_tone_phase_estimate(
    estimate: Any,
    samples: Any,
    *,
    method: Any,
    sampling_rate: float | None = None,
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
    ax : array_like of matplotlib.axes.Axes, optional
        Axes to draw on, two; ``None`` creates a figure.
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

    name = "plot_pilot_tone_phase_estimate()"
    tones = _require(estimate, "tone_frequencies", name, "PilotTone")
    adapter = adapt_signal(samples, function_name=name)
    fs = float(adapter.resolve_fact("sampling_rate", sampling_rate))
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

    if ax is None:
        fig, raw_axes = plt.subplots(1, 2, figsize=_grid_figsize(1, 2), squeeze=False)
        ax_spec, ax_phase = raw_axes[0]
    else:
        ax_spec, ax_phase = ax[0], ax[1]
        fig = ax_spec.figure

    order = np.argsort(freqs)
    f_sorted = freqs[order]
    eps = 1e-300

    # Panel 1 - spectrum (dB) with extraction window on a twin axis.
    # Envelope-decimate the full-resolution traces: min/max preserves the tone
    # peak and the window edges that plain striding would skip over.
    ax_win = ax_spec.twinx()
    for i in range(C):
        mag_db = 20.0 * np.log10(np.maximum(mag_spectrum[i][order], eps))
        label = f"Ch {i}" if C > 1 else "|X(f)|"
        f_d, mag_d = _decimate_minmax(f_sorted, mag_db, max_points)
        f_w, win_d = _decimate_minmax(f_sorted, window[i][order], max_points)
        ax_spec.plot(f_d, mag_d, label=label)
        ax_win.plot(f_w, win_d, color="C3", alpha=0.4)
        ax_spec.axvline(
            float(f_tones[i]),
            color="C2",
            linestyle=":",
            label=("Tone peak" if i == 0 else None),
        )
    ax_spec.axvspan(
        tone_frequency - bandwidth,
        tone_frequency + bandwidth,
        color="C3",
        alpha=0.4,
        label=f"Window ±B ({bandwidth:.3g} Hz)",
    )
    ax_spec.axvline(
        tone_frequency,
        color="k",
        linestyle="--",
        label=f"Nominal f_p ({tone_frequency:.3g} Hz)",
    )
    ax_win.set_ylim(-0.05, 1.35)
    ax_win.set_ylabel("Window W(f)", color="C3")
    ax_spec.set_title(f"{title} - Tone Spectrum")
    ax_spec.set_xlabel("Frequency [Hz]")
    ax_spec.set_ylabel("|X(f)| [dB]")
    ax_spec.legend(loc="upper left")

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
    ax : array_like of matplotlib.axes.Axes, optional
        Axes to draw on, two; ``None`` creates a figure.
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

    if ax is None:
        fig, raw_axes = plt.subplots(1, 2, figsize=_grid_figsize(1, 2), squeeze=False)
        ax_delta, ax_phase = raw_axes[0]
    else:
        ax_delta, ax_phase = ax[0], ax[1]
        fig = ax_delta.figure

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
            label=f"$\\delta$ tone {k}" + (" (gated)" if gated else ""),
        )
        n_plotted += 1
    ax_delta.set_title(f"{title} - Inter-tone $\\delta$  (ref = tone {ref})")
    ax_delta.set_xlabel("Sample Index")
    ax_delta.set_ylabel("Differential phase [deg]")
    if n_plotted:
        ax_delta.legend(loc="upper right")

    # Panel 2 - combined common-phase track.
    idx = np.arange(phi.shape[-1])
    n_d, p_d = _decimate_minmax(idx, np.degrees(phi), max_points)
    ax_phase.plot(n_d, p_d)
    ph_std = float(np.std(np.degrees(phi)))
    ax_phase.set_title(
        f"{title} - Combined $\\hat{{\\phi}}$  [$\\sigma$={ph_std:.2f}°, used={list(used)}]"
    )
    ax_phase.set_xlabel("Sample Index")
    ax_phase.set_ylabel("Phase [deg]")

    return _finish((fig, (ax_delta, ax_phase)), show)


def plot_carrier_phase_trajectory(
    phase: Any,
    *,
    num_train_symbols: int = 0,
    ax=None,
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
        label = f"Ch {i}" if C > 1 else None
        axi.plot(sym_idx, phi_deg, label=label)

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

    return _finish((fig, axi), show)
