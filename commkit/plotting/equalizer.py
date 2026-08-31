"""Filter and equalizer response plots."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from ..backend import dispatch, to_device
from ..logger import logger
from .theme import _grid_figsize


def plot_filter_response(
    taps: Any, sps: float = 1.0, ax: Any | None = None, show: bool = False
) -> tuple[Any, tuple[Any, Any, Any]] | None:
    """
    Plots the impulse and frequency response of a filter.

    Provides a 3-panel analysis showing the filter taps in the time domain,
    the magnitude response in dB, and the unwrapped phase response.

    Parameters
    ----------
    taps : array_like
        Filter taps (impulse response).
    sps : float, default 1.0
        Samples per symbol for time-axis normalization.
    ax : array_like, optional
        A list or tuple of 3 axes to plot on.
    show : bool, default False
        If True, calls `plt.show()`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    axes : tuple of matplotlib.axes.Axes
        The (impulse, magnitude, phase) axes.
    """

    import matplotlib.ticker as ticker

    # Dispatch
    taps, xp, sp = dispatch(taps)

    if ax is None:
        logger.debug("Generating filter response plot.")
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=_grid_figsize(3, 1))
    elif isinstance(ax, (list, tuple, np.ndarray)) and len(ax) == 3:
        fig = ax[0].figure
        ax1, ax2, ax3 = ax
    else:
        logger.warning("filter_response requires 3 axes. Creating new figure.")
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=_grid_figsize(1, 3))

    # 1. Impulse Response
    num_taps = len(taps)

    t = (xp.arange(num_taps) - (num_taps - 1) / 2) / sps

    # Move to cpu for plotting
    t_cpu = to_device(t, "cpu")
    taps_cpu = to_device(taps, "cpu")

    if xp.iscomplexobj(taps):
        ax1.plot(t_cpu, taps_cpu.real, label="Real", color="C0")
        ax1.plot(t_cpu, taps_cpu.imag, label="Imag", color="C1")
        ax1.legend()
    else:
        ax1.plot(t_cpu, taps_cpu, color="C0")

    ax1.set_title("Impulse Response")
    ax1.set_xlabel("Time [Symbol Periods]")
    ax1.set_ylabel("Amplitude")

    # Set ticks at integer intervals (1T, 2T, etc.)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))

    def t_formatter(x, pos):
        if np.isclose(x, 0):
            return "0"
        return f"{int(x)}T" if float(x).is_integer() else f"{x}T"

    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(t_formatter))

    # 2. Frequency Response
    # Compute frequency response using backend sp
    w, h = sp.signal.freqz(taps, worN=2048)

    # Normalize to Nyquist (0 to 1)
    # w is in radians/sample (0 to pi)
    freqs = w / (2 * xp.pi)

    # Avoid log(0)
    mag = 20 * xp.log10(xp.abs(h) + 1e-12)
    angles = xp.unwrap(xp.angle(h))

    # Move to cpu
    freqs = to_device(freqs, "cpu")
    mag = to_device(mag, "cpu")
    angles = to_device(angles, "cpu")

    # Magnitude
    ax2.plot(freqs, mag, color="C2")
    ax2.set_ylabel("Magnitude [dB]")
    ax2.set_title("Frequency Response (Magnitude)")
    ax2.set_xlabel("Frequency [Cycles/Sample]")
    ax2.set_xlim(0, 0.5)

    # Phase
    ax3.plot(freqs, angles, color="C3")
    ax3.set_ylabel("Phase [rad]")
    ax3.set_title("Frequency Response (Phase)")
    ax3.set_xlabel("Frequency [Cycles/Sample]")
    ax3.set_xlim(0, 0.5)

    if show:
        plt.show()
        return None
    return fig, (ax1, ax2, ax3)


def plot_equalizer_result(
    result,
    smoothing: int = 50,
    ax=None,
    show: bool = False,
) -> tuple[Any, Any] | None:
    """
    Plots adaptive equalizer diagnostics: convergence curve, tap weights, and
    (when CPR was enabled) the recovered phase trajectory.

    Parameters
    ----------
    result : EqualizerResult
        Equalizer output containing ``error``, ``weights``, and optionally
        ``phase_trajectory`` fields.
    smoothing : int, default 50
        Moving-average window length for the MSE convergence curve.
    ax : list of Axes, optional
        Pre-existing axes.  Pass 2 axes when ``phase_trajectory`` is None, or
        3 axes when CPR was used (``[ax_convergence, ax_taps, ax_phase]``).
        If None, a new figure is created with the appropriate number of panels.
    show : bool, default False
        If True, calls ``plt.show()`` and returns None.

    Returns
    -------
    (fig, axes) or None
        Figure and axes array when ``show=False``, None otherwise.
    """
    error = to_device(result.error, "cpu")
    weights = to_device(result.weights, "cpu")
    phase_traj = getattr(result, "phase_trajectory", None)
    if phase_traj is not None:
        phase_traj = to_device(phase_traj, "cpu")

    is_mimo = error.ndim == 2
    n_panels = 3 if phase_traj is not None else 2

    if ax is None:
        fig, axes = plt.subplots(1, n_panels, figsize=_grid_figsize(1, n_panels))
    else:
        axes = np.asarray(ax).flatten()[:n_panels]
        fig = axes[0].figure

    # --- Panel 1: MSE convergence ---
    ax_conv = axes[0]

    def _smooth_mse(mse, smoothing):
        """Return (x_coords, smoothed_mse_db) in raw-symbol-index space.

        The smoothing window is clipped to at most n//3 so the curve always
        spans at least 2/3 of the data range even for short sequences.
        """
        n = len(mse)
        effective = max(1, min(smoothing, n // 3))
        if effective > 1 and n > effective:
            kernel = np.ones(effective) / effective
            mse_smooth = np.convolve(mse, kernel, mode="valid")
            # Element k of mode="valid" output averages mse[k : k+effective].
            # Place it at the centre of that window so the x-axis is in
            # actual symbol-index space, not smoothed-bin-index space.
            x = np.arange(len(mse_smooth)) + (effective - 1) / 2.0
        else:
            mse_smooth = mse
            x = np.arange(n)
        return x, 10 * np.log10(mse_smooth + 1e-30)

    if is_mimo:
        num_ch = error.shape[0]
        for ch in range(num_ch):
            mse = np.abs(error[ch]) ** 2
            x_smooth, mse_db = _smooth_mse(mse, smoothing)
            ax_conv.plot(x_smooth, mse_db, label=f"Ch {ch}")
        ax_conv.legend()
    else:
        mse = np.abs(error) ** 2
        x_smooth, mse_db = _smooth_mse(mse, smoothing)
        ax_conv.plot(x_smooth, mse_db)

    n_train = getattr(result, "num_train_symbols", 0)
    if n_train and n_train > 0:
        ax_conv.axvline(
            n_train,
            color="white",
            linestyle="--",
            label=f"DD start ({n_train})",
        )
        ax_conv.legend()

    ax_conv.set_xlabel("Symbol Index")
    ax_conv.set_ylabel("MSE [dB]")
    ax_conv.set_title("Convergence")

    # --- Panel 2: Final tap weights ---
    ax_taps = axes[1]

    if is_mimo:
        # Butterfly weights: (C, C, num_taps) - plot magnitude of each row
        num_ch = weights.shape[0]
        num_taps = weights.shape[2]
        tap_idx = np.arange(num_taps) - (num_taps // 2)
        for i in range(num_ch):
            for j in range(num_ch):
                label = f"w[{i},{j}]"
                ax_taps.plot(
                    tap_idx,
                    np.abs(weights[i, j]),
                    marker="o",
                    markersize=3,
                    label=label,
                )
        ax_taps.legend(ncol=2)
    else:
        # SISO: (num_taps,) - stem plot
        num_taps = weights.shape[0]
        tap_idx = np.arange(num_taps) - (num_taps // 2)
        markerline, stemlines, _ = ax_taps.stem(tap_idx, np.abs(weights))
        plt.setp(stemlines)
        plt.setp(markerline, markersize=4)

    ax_taps.set_xlabel("Tap Index")
    ax_taps.set_ylabel("|w|")
    ax_taps.set_title("Tap Weights")

    # --- Panel 3: Phase trajectory (CPR only) ---
    if phase_traj is not None:
        ax_phase = axes[2]
        phi = np.asarray(phase_traj)
        if phi.ndim == 1:
            ax_phase.plot(np.degrees(phi))
        else:
            for ch in range(phi.shape[0]):
                ax_phase.plot(np.degrees(phi[ch]), label=f"Ch {ch}")
            ax_phase.legend()
        ax_phase.set_xlabel("Symbol Index")
        ax_phase.set_ylabel("Phase [deg]")
        ax_phase.set_title("CPR Phase Trajectory")

    if show:
        plt.show()
        return None
    return fig, axes


# -----------------------------------------------------------------------------
# SYNCHRONIZATION DIAGNOSTICS
# -----------------------------------------------------------------------------


def plot_zf_equalizer_response(
    channel_estimate,
    noise_variance: float = 0.0,
    nfft: int = 1024,
    sampling_rate: float = 1.0,
    ax=None,
    show: bool = False,
    title: str = "ZF/MMSE Equalizer Response",
) -> tuple[Any, Any] | None:
    """
    Plots channel, equalizer, and combined frequency responses for ZF/MMSE.

    For each channel path, three panels show:

    * ``|H(f)|`` - channel response in dB.
    * ``|W(f)|`` - equalizer response (``W = H* / (|H|² + σ²)``) in dB.
    * ``|H(f)·W(f)|`` - combined response (≈ 0 dB for ZF, soft mask for MMSE).

    Parameters
    ----------
    channel_estimate : array_like
        Channel impulse response. Shape: ``(L,)`` for SISO or
        ``(C_rx, C_tx, L)`` for MIMO.
    noise_variance : float, default 0.0
        Noise variance ``σ²`` for MMSE regularization. ``0.0`` gives pure ZF.
    nfft : int, default 1024
        FFT size for the frequency response computation.
    sampling_rate : float, default 1.0
        Sampling rate in Hz for frequency-axis scaling.
    ax : array_like of Axes, optional
        For SISO: 3 Axes. For MIMO: ``(C_rx * C_tx, 3)`` Axes.
        If ``None``, a new figure is created.
    show : bool, default False
    title : str, default "ZF/MMSE Equalizer Response"

    Returns
    -------
    (fig, axes) or None
    """
    h = to_device(channel_estimate, "cpu")
    reg = max(noise_variance, 1e-12)
    siso = h.ndim == 1

    freqs = np.fft.fftfreq(nfft, d=1.0 / sampling_rate)
    sort_idx = np.argsort(freqs)
    f_sorted = freqs[sort_idx]

    max_f = float(np.max(np.abs(f_sorted)))
    if max_f >= 1e9:
        scale, unit = 1e9, "GHz"
    elif max_f >= 1e6:
        scale, unit = 1e6, "MHz"
    elif max_f >= 1e3:
        scale, unit = 1e3, "kHz"
    else:
        scale, unit = 1.0, "Hz"
    f_disp = f_sorted / scale

    def _draw_triplet(H_f, ax_row, row_label=""):
        H_s = H_f[sort_idx]
        W_s = np.conj(H_s) / (np.abs(H_s) ** 2 + reg)
        HW_s = H_s * W_s
        for axi, data, lbl, col in zip(
            ax_row,
            [H_s, W_s, HW_s],
            ["Channel |H(f)|", "Equalizer |W(f)|", "Combined |H·W(f)|"],
            ["C0", "C1", "C2"],
        ):
            axi.plot(f_disp, 20 * np.log10(np.abs(data) + 1e-12), color=col)
            prefix = f"{row_label} " if row_label else ""
            axi.set_title(f"{prefix}{lbl}")
            axi.set_xlabel(f"Frequency [{unit}]")
            axi.set_ylabel("Magnitude [dB]")

    if siso:
        H_f = np.fft.fft(h, n=nfft)
        if ax is None:
            fig, axes = plt.subplots(1, 3, figsize=_grid_figsize(1, 3))
        else:
            axes = list(ax) if hasattr(ax, "__len__") else [ax]
            fig = axes[0].figure
        _draw_triplet(H_f, axes)
        fig.suptitle(title)
        if show:
            plt.show()
            return None
        return fig, axes

    C_rx, C_tx = h.shape[0], h.shape[1]
    n_rows = C_rx * C_tx
    if ax is None:
        fig, raw_axes = plt.subplots(
            n_rows, 3, figsize=_grid_figsize(n_rows, 3), squeeze=False
        )
        axes = [list(raw_axes[r]) for r in range(n_rows)]
    else:
        if hasattr(ax, "__len__") and hasattr(ax[0], "__len__"):
            axes = [list(row) for row in ax]
        else:
            flat = list(ax)
            axes = [flat[r * 3 : (r + 1) * 3] for r in range(n_rows)]
        fig = axes[0][0].figure

    for i in range(C_rx):
        for j in range(C_tx):
            row = i * C_tx + j
            H_f = np.fft.fft(h[i, j], n=nfft)
            _draw_triplet(H_f, axes[row], row_label=f"[rx{i}$\\rightarrow$tx{j}]")

    fig.suptitle(title)
    if show:
        plt.show()
        return None
    return fig, axes


# -----------------------------------------------------------------------------
# CARRIER-PHASE CHARACTERIZATION DIAGNOSTICS
# -----------------------------------------------------------------------------
