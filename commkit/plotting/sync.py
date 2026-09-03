"""Synchronization plots (timing, frequency offset, carrier phase)."""

from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from ..backend import to_device
from .theme import (
    _as_channels,
    _decimate_minmax,
    _finish,
    _get_axis,
    _grid_figsize,
    _set_eng_formatter,
)


def plot_timing_correlation(
    corr_mag,
    peak_indices,
    norm_factors,
    threshold: float,
    offset: int = 0,
    ax=None,
    show: bool = False,
    title: str = "Timing Correlation",
) -> tuple[Any, Any] | None:
    """
    Plots cross-correlation magnitude for timing estimation diagnostics.

    For each channel, two panels are drawn: an overall view of the
    correlation and a zoomed view around the detected peak.

    Parameters
    ----------
    corr_mag : array_like
        Correlation magnitude. Shape: ``(C, N)`` or ``(N,)``.
    peak_indices : array_like
        Integer peak positions per channel. Shape: ``(C,)`` or scalar.
    norm_factors : array_like
        Per-channel normalization factors. The displayed threshold line is
        ``threshold * norm_factors[c]``. Shape: ``(C,)``.
    threshold : float
        Detection threshold (normalized 0-1).
    offset : int, default 0
        Search-range start sample added to sample indices for correct labels.
    ax : array_like of Axes, optional
        Pre-existing axes of shape ``(C, 2)`` - overall and zoom per channel.
        If ``None``, a new figure is created.
    show : bool, default False
        If ``True``, calls ``plt.show()`` and returns ``None``.
    title : str, default "Timing Correlation"

    Returns
    -------
    (fig, axes) or None
    """
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
    R_np,
    f_est,
    sampling_rate: float,
    M: int = 1,
    ax=None,
    show: bool = False,
    title: str = "FOE - Mengali-Morelli",
) -> tuple[Any, Any] | None:
    """
    Plots the Mengali-Morelli autocorrelation diagnostics.

    Per-channel two-panel layout:

    * **Left / Top** - Normalised autocorrelation magnitude ``|R[m]|`` vs lag
      ``m``.  Encodes per-lag SNR; used as weight proxy ``w[m] ∝ m²|R[m]|²``.
    * **Right / Bottom** - Wrapped phase ``angle(R[m])`` vs lag, with the
      expected linear ramp overlaid and ``±π`` wrap boundaries marked.

    Parameters
    ----------
    R_np : (L,) or (C, L) complex128
        Normalised per-channel autocorrelation at lags ``m = 1 ... L``.
        A 1-D input is treated as a single channel.
    f_est : float or list of float
        Frequency offset estimate(s) in Hz.  Scalar for a single channel;
        list/array of length ``C`` for multi-channel input.
    sampling_rate : float
        Sampling rate in Hz.
    M : int, default 1
        Modulation pre-processing exponent.
    ax : Axes or array of Axes, optional
        For a single channel: pair ``[ax_amp, ax_phase]``.
        For multiple channels: array of shape ``(C, 2)``.
        A new figure is created when ``None``.
    show : bool, default False
        If ``True``, calls ``plt.show()`` and returns ``None``.
    title : str, default "FOE - Mengali-Morelli"

    Returns
    -------
    (fig, axes) or None
        Single channel: ``axes`` is ``[ax_amp, ax_phase]``.
        Multi-channel: ``axes`` is a list of ``[ax_amp_c, ax_phase_c]`` pairs.
    """
    R_np = np.asarray(R_np)
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
    mag_spectrum,
    freqs,
    M: int,
    k_peaks,
    f_estimates,
    search_range=None,
    ax=None,
    show: bool = False,
    title: str = "FOE - M-th Power Spectrum",
) -> tuple[Any, Any] | None:
    """
    Plots the M-th power spectrum used for blind frequency offset estimation.

    The spectrum has a tone at ``M·Δf``; this function maps the x-axis back
    to ``Δf`` by dividing by ``M`` so the detected peak aligns with the
    reported frequency estimate.

    Parameters
    ----------
    mag_spectrum : array_like
        Magnitude spectrum ``|X^M(f)|``. Shape: ``(C, nfft)`` or ``(nfft,)``.
    freqs : array_like
        Frequency axis in Hz (``np.fft.fftfreq(nfft) * fs``).
        Shape: ``(nfft,)``.
    M : int
        M-th power used to remove modulation.
    k_peaks : array_like
        Peak bin index per channel. Shape: ``(C,)`` or scalar.
    f_estimates : list of float
        Per-channel frequency estimates in Hz (before multi-channel averaging).
    search_range : tuple of float, optional
        ``(f_min, f_max)`` in Hz that restricted the search. Shown as a
        shaded region.
    ax : array_like of Axes, optional
        One Axes per channel. If ``None``, a new figure is created.
    show : bool, default False
    title : str, default "FOE - M-th Power Spectrum"

    Returns
    -------
    (fig, axes) or None
    """
    mag_spectrum = to_device(mag_spectrum, "cpu")
    freqs = to_device(freqs, "cpu")
    k_peaks = to_device(k_peaks, "cpu").flatten()
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


def plot_carrier_phase_trajectory(
    phi_full,
    block_centers=None,
    phi_blocks=None,
    n_train: int = 0,
    ax=None,
    show: bool = False,
    title: str = "Carrier Phase Trajectory",
) -> tuple[Any, Any] | None:
    """
    Plots per-symbol carrier phase trajectory for CPR algorithm diagnostics.

    All channels are overlaid on a single subplot.  Block-based methods may
    pass ``block_centers`` / ``phi_blocks`` to annotate block-phase estimates
    as vertical lines (not scatter markers, consistent with the joint
    equalizer phase panel).

    Parameters
    ----------
    phi_full : array_like
        Per-symbol phase estimate in radians. Shape: ``(C, N)`` or ``(N,)``.
    block_centers : array_like, optional
        Block centre positions in symbols (VV, BPS). Shape: ``(N_blocks,)``.
        If provided, thin vertical lines at each block centre are drawn.
    phi_blocks : array_like, optional
        Kept for backwards compatibility - ignored (block markers removed).
    n_train : int, default 0
        Training/DD boundary symbol index. Draws a dashed vertical line.
    ax : Axes, optional
        Single Axes object to plot into.  If ``None``, a new figure is created.
    show : bool, default False
    title : str, default "Carrier Phase Trajectory"

    Returns
    -------
    (fig, ax) or None
    """
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
        bc = np.asarray(to_device(block_centers, "cpu"))
        for tc in bc:
            axi.axvline(tc, color="gray", alpha=0.4)

    if n_train > 0:
        axi.axvline(
            n_train,
            color="white",
            linestyle="--",
            label=f"DD start ({n_train})",
        )

    phi_all = np.degrees(phi_full)
    phi_mean = float(np.mean(phi_all))
    phi_std = float(np.std(phi_all))
    axi.set_title(f"{title}  [$\\mu$={phi_mean:.1f}°,  $\\sigma$={phi_std:.2f}°]")
    axi.set_xlabel("Symbol Index")
    axi.set_ylabel("Phase [deg]")
    if C > 1 or n_train > 0:
        axi.legend(loc="upper right")

    return _finish((fig, axi), show)


def plot_frequency_offset_blockwise_result(
    t_centers,
    df_estimates,
    n_grid,
    df_dense,
    phase_trajectory,
    ax=None,
    show: bool = False,
    title: str = "Block-wise FOE",
    max_points: int = 4000,
) -> tuple[Any, Any] | None:
    """
    Diagnostic plot for ``frequency.correct_frequency_offset_blockwise``.

    Shows three panels:

    1. Per-block frequency estimates (scatter) and interpolated Δf trajectory (line).
    2. Integrated phase trajectory in degrees.

    Parameters
    ----------
    t_centers : array_like
        Block centre sample indices. Shape: ``(K,)``.
    df_estimates : array_like
        Per-block frequency estimates in Hz. Shape: ``(K,)``.
    n_grid : array_like
        Dense sample index grid. Shape: ``(N,)``.
    df_dense : array_like
        Interpolated frequency at each sample in Hz. Shape: ``(N,)``.
    phase_trajectory : array_like
        Integrated phase in radians. Shape: ``(N,)``.
    ax : list of 2 Axes, optional
        Pre-existing axes ``[ax_freq, ax_phase]``. If ``None``, a new figure
        with 2 panels is created.
    show : bool, default False
    title : str
    max_points : int, default 4000
        Approximate per-trace point budget; the dense per-sample Δf and phase
        traces are envelope-decimated (min/max) before plotting so long records
        render quickly.  Pass ``<= 0`` to plot every point.

    Returns
    -------
    (fig, axes) or None
    """
    t_centers = np.asarray(t_centers, dtype=np.float64)
    df_estimates = np.asarray(df_estimates, dtype=np.float64)
    n_grid = np.asarray(n_grid, dtype=np.float64)
    df_dense = np.asarray(df_dense, dtype=np.float64)
    phase_trajectory = np.asarray(phase_trajectory, dtype=np.float64)

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
    pilot_indices,
    phi_pilots_u,
    phi_full=None,
    f_est: float | Sequence[float] | np.ndarray = 0.0,
    sampling_rate: float = 1.0,
    ax=None,
    show: bool = False,
    title: str = "Pilot Phase Estimate",
) -> tuple[Any, Any] | None:
    """
    Plots pilot phase scatter, linear fit, and the full interpolated trajectory.

    Used as a diagnostic for both pilot-based frequency offset estimation
    (``estimate_frequency_offset_pilot_symbols``) and pilot-aided carrier phase
    recovery (``recover_carrier_phase_pilot_symbols``).

    Parameters
    ----------
    pilot_indices : array_like
        Sample indices of pilot positions. Shape: ``(P,)``.
    phi_pilots_u : array_like
        Unwrapped pilot phases in radians. Shape: ``(C, P)`` or ``(P,)``.
    phi_full : array_like, optional
        Interpolated per-symbol phase in radians. Shape: ``(C, N)`` or ``(N,)``.
        If provided, a second panel shows the full trajectory per channel.
    f_est : float or list of float, default 0.0
        Estimated frequency offset in Hz (annotation on the fit line).
        Scalar applies the same label to all channels; list of length ``C``
        annotates each channel independently.
    sampling_rate : float, default 1.0
        Sampling rate in Hz.
    ax : array_like of Axes, optional
        Shape ``(C, 1)`` when ``phi_full`` is ``None``, else ``(C, 2)``.
        If ``None``, a new figure is created.
    show : bool, default False
    title : str, default "Pilot Phase Estimate"

    Returns
    -------
    (fig, axes) or None
    """
    pilot_indices = to_device(pilot_indices, "cpu").astype(float)
    phi_pilots_u = to_device(phi_pilots_u, "cpu")
    if phi_pilots_u.ndim == 1:
        phi_pilots_u = phi_pilots_u[None, :]
    C, P = phi_pilots_u.shape

    if isinstance(f_est, (np.ndarray, list, tuple)):
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
            phi_fit = phi_p.copy()

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
            label=f"Fit  $\\Delta f$={f_ests[i]:.3f} Hz",
        )
        ax1.set_title(f"{title}{ch_suffix} - Pilots")
        ax1.set_xlabel("Sample Index")
        ax1.set_ylabel("Phase [deg]")
        ax1.legend()

        if has_full:
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
    freqs,
    mag_spectrum,
    window,
    f_tones,
    theta,
    tone_frequency: float,
    bandwidth: float,
    ax=None,
    show: bool = False,
    title: str = "CPR - Pilot Tone",
    max_points: int = 4000,
) -> tuple[Any, Any] | None:
    """
    Diagnostic for ``recover_carrier_phase_pilot_tone``.

    Two panels:

    1. **Tone spectrum** - magnitude spectrum ``|X(f)|`` (dB) with the
       zero-phase extraction window overlaid on a twin axis, the per-channel
       refined tone peak marked, and the nominal tone frequency annotated.
    2. **Recovered phase** - the per-sample phase estimate ``θ̂[n]`` (deg).

    All channels are overlaid on each panel.

    Parameters
    ----------
    freqs : array_like
        FFT frequency axis in Hz (``np.fft.fftfreq(N) * fs``). Shape: ``(N,)``.
    mag_spectrum : array_like
        Magnitude spectrum ``|X(f)|``. Shape: ``(C, N)`` or ``(N,)``.
    window : array_like
        Extraction window ``W(f)`` in ``[0, 1]``. Shape: ``(C, N)`` or ``(N,)``.
    f_tones : array_like
        Per-channel refined tone frequency in Hz. Shape: ``(C,)``.
    theta : array_like
        Recovered per-sample phase in radians. Shape: ``(C, N)`` or ``(N,)``.
    tone_frequency : float
        Nominal pilot-tone frequency in Hz.
    bandwidth : float
        Extraction window half-width in Hz (shaded around the nominal tone).
    ax : array_like of Axes, optional
        Two Axes ``[spectrum, phase]``. If ``None``, a new figure is created.
    show : bool, default False
    title : str, default "CPR - Pilot Tone"
    max_points : int, default 4000
        Approximate per-trace point budget; longer traces are envelope-decimated
        (min/max) before plotting so oversampled records render quickly without
        losing the tone peak or window edges.  Pass ``<= 0`` to plot every point.

    Returns
    -------
    (fig, axes) or None
    """
    freqs = np.asarray(to_device(freqs, "cpu"), dtype=float)
    mag_spectrum = to_device(mag_spectrum, "cpu")
    window = to_device(window, "cpu")
    theta = to_device(theta, "cpu")
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
    sample_idx = np.arange(N)
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
    delta,
    phi,
    ref: int,
    used: Sequence[int],
    ax=None,
    show: bool = False,
    title: str = "CPR - Pilot Tones (MRC)",
    max_points: int = 4000,
) -> tuple[Any, Any] | None:
    """
    Diagnostic for ``recover_carrier_phase_pilot_tones``.

    Two panels:

    1. **Inter-tone differential** - the slow tracked phase ``δ_k[n]`` (deg) for
       each non-reference tone; the line style flags whether the tone was
       combined (solid) or gated out (dashed).
    2. **Combined phase** - the per-sample common estimate ``φ̂[n]`` (deg).

    All channels share the active theme; no colours or line widths are set
    explicitly, so the trace colours come from the rcParams cycle.

    Parameters
    ----------
    delta : sequence of array_like
        Per-tone differential phase ``δ_k[n]`` in radians - a length-``K``
        sequence (or ``(K, N)`` array) of ``(N,)`` traces.  The reference tone's
        entry is ``≈ 0`` and is skipped in panel 1.
    phi : array_like
        Combined per-sample phase estimate ``φ̂[n]`` in radians.  Shape ``(N,)``.
    ref : int
        Reference-tone index (its differential is identically zero).
    used : sequence of int
        Indices of the tones that were combined (the rest were gated out).
    ax : array_like of Axes, optional
        Two Axes ``[differential, phase]``. If ``None``, a new figure is created.
    show : bool, default False
    title : str, default "CPR - Pilot Tones (MRC)"
    max_points : int, default 4000
        Per-trace point budget; longer traces are envelope-decimated (min/max)
        before plotting.  Pass ``<= 0`` to plot every point.

    Returns
    -------
    (fig, axes) or None
    """
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


def plot_carrier_phase_decomposition(
    phi,
    drift=None,
    *,
    symbol_rate: float,
    n_train: int = 0,
    ax=None,
    show: bool = False,
    title: str = "Recovered carrier phase",
) -> tuple[Any, Any] | None:
    """
    Plots the recovered carrier-phase trajectory and its slow drift component.

    The total unwrapped phase phi(t) is drawn faintly with the
    low-pass drift overlaid in bold, visualising the
    ``analysis.separate_drift_phase_noise`` split.  MIMO inputs overlay all
    channels.

    Parameters
    ----------
    phi : array_like
        Unwrapped carrier phase in radians. Shape ``(N,)`` or ``(C, N)``.
    drift : array_like, optional
        Drift (low-pass) component, same shape as ``phi``.  Overlaid in bold.
    symbol_rate : float
        Symbol rate in Baud; sets the (seconds) time axis.
    n_train : int, default 0
        If > 0, draws a dashed training/DD boundary marker.
    ax : Axes, optional
        Target axes; a new figure is created when None.
    show : bool, default False
        If True, calls ``plt.show()`` and returns None.
    title : str

    Returns
    -------
    (fig, ax) or None
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

    if n_train > 0:
        axi.axvline(
            n_train / float(symbol_rate),
            color="white",
            ls="--",
            label=f"DD start ({n_train})",
        )

    _set_eng_formatter(axi, "x", "s")
    axi.set_xlabel("Time [s]")
    axi.set_ylabel("Phase [rad]")
    axi.set_title(title)
    axi.legend(loc="best")

    return _finish((fig, axi), show)
