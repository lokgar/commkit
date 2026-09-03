"""Power spectral density and spectrogram plots."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from ..backend import dispatch, to_device
from ..core.signal import Signal
from ..logger import logger
from .theme import (
    _create_subplot_grid,
    _grid_figsize,
    _set_eng_formatter,
)


def plot_psd(
    samples: Any,
    sampling_rate: float = 1.0,
    nperseg: int = 256,
    detrend: str | bool | None = False,
    average: str | None = "mean",
    window: str | tuple[Any, ...] | Any = "hann",
    noverlap: int | None = None,
    nfft: int | None = None,
    scaling: str = "density",
    center_frequency: float = 0.0,
    domain: str = "RF",
    x_axis: str = "frequency",
    ax: Any | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    title: str | None = "Power Spectral Density",
    show: bool = False,
    **kwargs: Any,
) -> tuple[Any, Any] | None:
    """
    Plots the Power Spectral Density (PSD) of the signal.

    Supports automatic frequency scaling (Hz, MHz, GHz, etc.) or wavelength
    conversion for optical signals. Handles multidimensional (MIMO) signals
    by generating a grid of subplots.

    Parameters
    ----------
    samples : array_like or Signal
        Input signal samples. Shape: (..., N_samples).
    sampling_rate : float, default 1.0
        Sampling rate in Hz.
    nperseg : int, default 256
        Length of each segment for Welch's method. Higher values provide
        better frequency resolution but more noise.
    detrend : str or bool, default False
        Specifies how to detrend each segment (e.g., 'constant', 'linear').
    average : str, default "mean"
        Method to use for averaging segments ('mean' or 'median').
    window : str or tuple or array_like, default "hann"
        Desired window to use. If `window` is a string or tuple, it is
        passed to `scipy.signal.get_window` to generate the window values.
    noverlap : int, optional
        Number of points to overlap between segments. If None,
        `noverlap = nperseg // 2`.
    nfft : int, optional
        Length of the FFT used, if a zero padded FFT is desired. If None,
        the FFT length is `nperseg`.
    scaling : {"density", "spectrum"}, default "density"
        Selects between computing the power spectral density ('density')
        where Pxx has units of V**2/Hz and computing the power spectrum
        ('spectrum') where Pxx has units of V**2.
    center_frequency : float, default 0.0
        Frequency offset to apply to the x-axis in Hz.
    domain : {"RF", "OPT"}, default "RF"
        Signal domain. If "OPT", wavelength scaling is enabled.
    x_axis : {"frequency", "wavelength"}, default "frequency"
        Units for the horizontal axis.
    ax : matplotlib.axes.Axes, optional
        Existing axis to plot on. If `None`, a new figure is created.
    xlim, ylim : tuple of float, optional
        Axis limits for the plot.
    title : str, optional
        Plot title. Defaults to "Power Spectral Density".
    show : bool, default False
        If True, calls `plt.show()` immediately.
    **kwargs : Any
        Additional keyword arguments passed to `ax.plot`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes or ndarray
        The axis or array of axes used for the plot.
    """
    if isinstance(samples, Signal):
        sig = samples
        return plot_psd(
            sig.samples,
            sampling_rate=sig.sampling_rate,
            nperseg=nperseg,
            detrend=detrend,
            average=average,
            window=window,
            noverlap=noverlap,
            nfft=nfft,
            scaling=scaling,
            center_frequency=sig.center_frequency,
            domain=sig.physical_domain or "RF",
            x_axis=x_axis,
            ax=ax,
            xlim=xlim,
            ylim=ylim,
            title=title,
            show=show,
            **kwargs,
        )

    logger.debug("Generating PSD plot (sampling_rate=%s Hz).", sampling_rate)

    samples, xp, _ = dispatch(samples)

    # Handle Multichannel (e.g. Dual-Pol)
    # Convention: (Channels, Time)
    if samples.ndim > 1:
        num_channels = samples.shape[0]

        if ax is None:
            nrows, ncols = _create_subplot_grid(num_channels)
            fig, axes = plt.subplots(
                nrows, ncols, figsize=_grid_figsize(nrows, ncols), squeeze=False
            )
        else:
            if not isinstance(ax, (list, tuple, np.ndarray)):
                # If single axis provided but multiple channels, warn and overlay?
                # Or better, just overlay on the same axis for PSD
                # Actually, the user asked for "side-by-side".
                # But if the user provides a single axis, we must respect it.
                # Let's overlay if single axis provided, or fail.
                logger.warning(
                    "Multiple channels detected but single axis provided. "
                    "Overlaying plots."
                )
                axes = np.array([[ax] * num_channels])
                fig = ax.figure
            else:
                axes = np.atleast_2d(ax)
                fig = axes.flat[0].figure

        for i in range(num_channels):
            # Recursively call psd for each channel
            channel_samples = samples[i]

            # Determine target axis using 2D indexing
            row, col = divmod(i, axes.shape[1])
            target_ax = axes[row, col] if row < axes.shape[0] else axes.flat[-1]

            ch_title = f"{title} (Ch {i})" if title else f"Channel {i}"

            plot_psd(
                channel_samples,
                sampling_rate=sampling_rate,
                nperseg=nperseg,
                detrend=detrend,
                average=average,
                window=window,
                noverlap=noverlap,
                nfft=nfft,
                scaling=scaling,
                center_frequency=center_frequency,
                domain=domain,
                x_axis=x_axis,
                ax=target_ax,
                xlim=xlim,
                ylim=ylim,
                title=ch_title,
                show=False,
                **kwargs,
            )

        if show:
            plt.show()
            return None
        return fig, axes

    # --- 1D Logic Starts Here ---

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    from .. import spectral

    # Calculate PSD
    f, Pxx = spectral.welch_psd(
        samples,
        sampling_rate=sampling_rate,
        nperseg=nperseg,
        detrend=detrend,
        average=average,
        window=window,
        noverlap=noverlap,
        nfft=nfft,
        scaling=scaling,
    )

    # Move to cpu for plotting
    f = to_device(f, "cpu")
    Pxx = to_device(Pxx, "cpu")

    # Apply center frequency shift
    f = f + center_frequency

    xlabel = "Frequency [Hz]"
    x_values = f

    if x_axis == "wavelength":
        if domain != "OPT":
            logger.warning("Wavelength plotting is typically used for optical signals.")

        # c = 299,792,458 m/s
        c = 299792458.0
        # Avoid division by zero
        # Convert frequency to wavelength: lambda = c / f
        # Result in nanometers (1e9)
        valid_indices = f > 0
        x_values = np.zeros_like(f)
        x_values[valid_indices] = (c / f[valid_indices]) * 1e9
        x_values[~valid_indices] = np.nan  # Handle non-positive frequencies

        xlabel = "Wavelength [nm]"
    else:
        xlabel = "Frequency [Hz]"

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    # Add epsilon to avoid log(0) warnings
    ax.plot(x_values, 10 * np.log10(Pxx + 1e-20), **kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("PSD [dB/Hz]")
    if x_axis != "wavelength":
        _set_eng_formatter(ax, "x", "Hz")
    if title is not None:
        ax.set_title(title)

    if show:
        plt.show()
        return None
    return fig, ax


def plot_spectrogram(
    samples: Any,
    sampling_rate: float = 1.0,
    window: str | tuple[Any, ...] | Any = "hann",
    nperseg: int = 256,
    noverlap: int | None = None,
    nfft: int | None = None,
    detrend: str | bool | None = False,
    return_onesided: bool | None = None,
    scaling: str = "density",
    axis: int = -1,
    mode: str = "psd",
    center_frequency: float = 0.0,
    domain: str = "RF",
    ax: Any | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    title: str | None = "Spectrogram",
    cmap: str = "viridis",
    show: bool = False,
    **kwargs: Any,
) -> tuple[Any, Any] | None:
    """
    Plots the spectrogram of a signal.

    Plots with Frequency on the horizontal axis (x-axis) and Time on the
    vertical axis (y-axis). Supports dynamic subplots for multichannel/MIMO
    signals.

    Parameters
    ----------
    samples : array_like or Signal
        Input signal samples. Shape: (..., N_samples).
    sampling_rate : float, default 1.0
        Sampling rate in Hz.
    window : str or tuple or array_like, default "hann"
        Desired window to use.
    nperseg : int, default 256
        Length of each segment.
    noverlap : int, optional
        Number of points to overlap between segments.
    nfft : int, optional
        Length of the FFT used.
    detrend : str or bool, default False
        Specifies how to detrend each segment.
    return_onesided : bool, optional
        If True, returns a one-sided spectrum for real-valued data.
    scaling : {"density", "spectrum"}, default "density"
        Selects between computing power spectral density or power spectrum.
    axis : int, default -1
        The axis along which to compute the spectrogram.
    mode : {"psd", "complex", "magnitude", "angle", "phase"}, default "psd"
        Type of spectrogram to return.
    center_frequency : float, default 0.0
        Frequency offset to apply to the frequency axis in Hz.
    domain : {"RF", "OPT"}, default "RF"
        Signal domain.
    ax : matplotlib.axes.Axes, optional
        Existing axis to plot on.
    xlim : tuple of float, optional
        Frequency limits for the plot (in Hz, after center_frequency offset).
        Used to crop data before plotting for performance.
    ylim : tuple of float, optional
        Time limits for the plot (in seconds).
        Used to crop data before plotting for performance.
    title : str, optional
        Plot title.
    cmap : str, default "viridis"
        Colormap for the spectrogram plot.
    show : bool, default False
        If True, calls `plt.show()` immediately.
    **kwargs : Any
        Additional keyword arguments passed to `ax.pcolormesh`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes or ndarray
        The axis or array of axes used for the plot.
    """
    if isinstance(samples, Signal):
        sig = samples
        return plot_spectrogram(
            sig.samples,
            sampling_rate=sig.sampling_rate,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            detrend=detrend,
            return_onesided=return_onesided,
            scaling=scaling,
            axis=-1,
            mode=mode,
            center_frequency=sig.center_frequency,
            domain=sig.physical_domain or "RF",
            ax=ax,
            xlim=xlim,
            ylim=ylim,
            title=title,
            cmap=cmap,
            show=show,
            **kwargs,
        )

    logger.debug("Generating spectrogram plot (sampling_rate=%s Hz).", sampling_rate)

    samples, xp, _ = dispatch(samples)

    # Handle Multichannel (MIMO)
    # Convention: (Channels, Time)
    if samples.ndim > 1:
        num_channels = samples.shape[0]

        if ax is None:
            nrows, ncols = _create_subplot_grid(num_channels)
            fig, axes = plt.subplots(
                nrows, ncols, figsize=_grid_figsize(nrows, ncols), squeeze=False
            )
        else:
            if not isinstance(ax, (list, tuple, np.ndarray)):
                logger.warning(
                    "Multiple channels detected but single axis provided. "
                    "Overlaying plots."
                )
                axes = np.array([[ax] * num_channels])
                fig = ax.figure
            else:
                axes = np.atleast_2d(ax)
                fig = axes.flat[0].figure

        for i in range(num_channels):
            channel_samples = samples[i]
            row, col = divmod(i, axes.shape[1])
            target_ax = axes[row, col] if row < axes.shape[0] else axes.flat[-1]
            ch_title = f"{title} (Ch {i})" if title else f"Channel {i}"

            plot_spectrogram(
                channel_samples,
                sampling_rate=sampling_rate,
                window=window,
                nperseg=nperseg,
                noverlap=noverlap,
                nfft=nfft,
                detrend=detrend,
                return_onesided=return_onesided,
                scaling=scaling,
                axis=axis,
                mode=mode,
                center_frequency=center_frequency,
                domain=domain,
                ax=target_ax,
                xlim=xlim,
                ylim=ylim,
                title=ch_title,
                cmap=cmap,
                show=False,
                **kwargs,
            )

        if show:
            plt.show()
            return None
        return fig, axes

    # --- 1D Logic ---
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    from .. import spectral

    # Calculate spectrogram
    f, t, Sxx = spectral.spectrogram(
        samples,
        sampling_rate=sampling_rate,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend=detrend,
        return_onesided=return_onesided,
        scaling=scaling,
        axis=axis,
        mode=mode,
    )

    # Move to CPU for plotting
    f = to_device(f, "cpu")
    t = to_device(t, "cpu")
    Sxx = to_device(Sxx, "cpu")

    # Shift frequency axis first
    f_shifted = f + center_frequency

    # Masking frequency axis (xlim corresponds to frequency axis)
    f_start, f_end = 0, len(f_shifted)
    if xlim is not None:
        f_mask = (f_shifted >= xlim[0]) & (f_shifted <= xlim[1])
        f_indices = np.where(f_mask)[0]
        if len(f_indices) > 0:
            f_start, f_end = f_indices[0], f_indices[-1] + 1
        else:
            logger.warning(
                "xlim (frequency) %s does not overlap with frequency range "
                "[%.3f, %.3f]. Plotting whole frequency range.",
                xlim,
                f_shifted[0],
                f_shifted[-1],
            )

    # Masking time axis (ylim corresponds to time axis)
    t_start, t_end = 0, len(t)
    if ylim is not None:
        t_mask = (t >= ylim[0]) & (t <= ylim[1])
        t_indices = np.where(t_mask)[0]
        if len(t_indices) > 0:
            t_start, t_end = t_indices[0], t_indices[-1] + 1
        else:
            logger.warning(
                "ylim (time) %s does not overlap with time range "
                "[%.3f, %.3f]. Plotting whole time axis.",
                ylim,
                t[0],
                t[-1],
            )

    # Slice arrays for plotting performance
    f_plot = f_shifted[f_start:f_end]
    t_plot = t[t_start:t_end]
    Sxx_slice = Sxx[f_start:f_end, t_start:t_end]

    # Convert values based on mode (e.g. dB scale for PSD/magnitude)
    if mode == "psd":
        Sxx_plot = 10 * np.log10(Sxx_slice + 1e-20)
    elif mode in ("complex", "magnitude"):
        Sxx_plot = 10 * np.log10(np.abs(Sxx_slice) ** 2 + 1e-20)
    else:
        # Angle, phase, etc., plot linearly
        Sxx_plot = Sxx_slice

    # Plot spectrogram with frequency on x-axis and time on y-axis
    # Sxx_plot has shape (len(f_plot), len(t_plot)).
    # Transposing Sxx_plot to (len(t_plot), len(f_plot)) matches y-axis (time) and x-axis (frequency).
    mesh = ax.pcolormesh(
        f_plot, t_plot, Sxx_plot.T, cmap=cmap, shading="auto", **kwargs
    )
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Time [s]")
    _set_eng_formatter(ax, "x", "Hz")
    _set_eng_formatter(ax, "y", "s")
    if title is not None:
        ax.set_title(title)

    # Add colorbar
    cbar = fig.colorbar(mesh, ax=ax)
    if mode == "psd":
        cbar.set_label("PSD [dB/Hz]")
    elif mode in ("complex", "magnitude"):
        cbar.set_label("Magnitude [dB]")
    elif mode in ("angle", "phase"):
        cbar.set_label("Phase [rad]")

    if show:
        plt.show()
        return None
    return fig, ax
