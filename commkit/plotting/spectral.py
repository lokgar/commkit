"""Power spectral density and spectrogram plots.

:func:`plot_psd` and :func:`plot_spectrogram`.
"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from ..backend import dispatch, to_device
from ..core._signal_adapter import adapt_signal
from ..logger import logger
from .theme import (
    _channel_axes,
    _set_eng_formatter,
)

__all__ = ["plot_psd", "plot_spectrogram"]


def plot_psd(
    samples: Any,
    *,
    sampling_rate: float | None = None,
    nperseg: int = 256,
    detrend: str | bool | None = False,
    average: str | None = "mean",
    window: str | tuple[Any, ...] | Any = "hann",
    noverlap: int | None = None,
    nfft: int | None = None,
    scaling: str = "density",
    center_frequency: float | None = None,
    domain: str = "RF",
    x_axis: str = "frequency",
    ax: Any | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    title: str | None = "Power Spectral Density",
    show: bool = False,
    **kwargs: Any,
) -> tuple[Any, Any] | None:
    """Plot the power spectral density (PSD) of a signal, in dB.

    The Welch estimate of :func:`commkit.spectral.welch_psd`, against
    frequency (SI prefixes) or, for optical signals, wavelength.  ``(C, N)``
    input gets one panel per channel.

    Parameters
    ----------
    samples : array_like or Signal
        Input samples, ``(N,)`` or ``(C, N)``.
    sampling_rate : float, optional
        Sampling rate in Hz.  Taken from the Signal; required for array
        input.
    nperseg, detrend, average, window, noverlap, nfft, scaling
        As for :func:`commkit.spectral.welch_psd`.
    center_frequency : float, optional
        Frequency added to the axis in Hz (the carrier, for an RF or optical
        axis).  Taken from the Signal (a value that disagrees with it
        raises); 0 for arrays when not given.
    domain : {"RF", "OPT"}, default "RF"
        Signal domain; a wavelength axis for an ``"RF"`` signal logs a
        warning.
    x_axis : {"frequency", "wavelength"}, default "frequency"
        Horizontal axis: frequency in Hz, or wavelength ``c / f`` in nm
        (which needs ``center_frequency``).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; ``None`` creates a figure.
    xlim, ylim : tuple of float, optional
        Axis limits (frequency in Hz or wavelength in nm; PSD in dB).
    title : str or None, default "Power Spectral Density"
        Title of the figure; ``None`` for none.
    show : bool, default False
        Call ``plt.show()`` and return ``None``.
    **kwargs : Any
        Passed to ``ax.plot``.

    Returns
    -------
    (fig, ax) or None
        The figure and the Axes (an array of Axes for several panels);
        ``None`` with ``show=True``, which shows the figure instead.
    """
    signal_adapter = adapt_signal(samples, function_name="plot_psd()")
    x = signal_adapter.array
    sampling_rate = signal_adapter.resolve_fact("sampling_rate", sampling_rate)
    if signal_adapter.signal is not None:
        center_frequency = signal_adapter.resolve_fact(
            "center_frequency", center_frequency
        )
    elif center_frequency is None:
        center_frequency = 0.0
    return _plot_psd(
        x,
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
        ax=ax,
        xlim=xlim,
        ylim=ylim,
        title=title,
        show=show,
        **kwargs,
    )


def _plot_psd(
    samples: Any,
    *,
    sampling_rate: Any,
    nperseg: Any,
    detrend: Any,
    average: Any,
    window: Any,
    noverlap: Any,
    nfft: Any,
    scaling: Any,
    center_frequency: Any,
    domain: Any,
    x_axis: Any,
    ax: Any,
    xlim: Any,
    ylim: Any,
    title: Any,
    show: Any,
    **kwargs: Any,
) -> tuple[Any, Any] | None:
    """Render array data for :func:`plot_psd` (the public boundary resolves the Signal)."""
    logger.debug("Generating PSD plot (sampling_rate=%s Hz).", sampling_rate)

    samples, xp, _ = dispatch(samples)

    # Handle Multichannel (e.g. Dual-Pol)
    # Convention: (Channels, Time)
    if samples.ndim > 1:
        num_channels = samples.shape[0]

        fig, axes, targets = _channel_axes(num_channels, ax)

        for i in range(num_channels):
            # Recursively call psd for each channel
            channel_samples = samples[i]

            target_ax = targets[i]

            ch_title = f"{title} (Ch {i})" if title else f"Channel {i}"

            _plot_psd(
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
    *,
    sampling_rate: float | None = None,
    window: str | tuple[Any, ...] | Any = "hann",
    nperseg: int = 256,
    noverlap: int | None = None,
    nfft: int | None = None,
    detrend: str | bool | None = False,
    return_onesided: bool | None = None,
    scaling: str = "density",
    mode: str = "psd",
    center_frequency: float | None = None,
    domain: str = "RF",
    ax: Any | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    title: str | None = "Spectrogram",
    cmap: str = "viridis",
    show: bool = False,
    **kwargs: Any,
) -> tuple[Any, Any] | None:
    """Plot the spectrogram of a signal, in dB.

    Frequency on the horizontal axis, time on the vertical one; ``(C, N)``
    input gets one panel per channel.

    Parameters
    ----------
    samples : array_like or Signal
        Input samples, ``(N,)`` or ``(C, N)``.
    sampling_rate : float, optional
        Sampling rate in Hz.  Taken from the Signal; required for array
        input.
    window, nperseg, noverlap, nfft, detrend, return_onesided, scaling, mode
        As for :func:`commkit.spectral.spectrogram`.
    center_frequency : float, optional
        Frequency added to the axis in Hz.  Taken from the Signal (a value
        that disagrees with it raises); 0 for arrays when not given.
    domain : {"RF", "OPT"}, default "RF"
        Signal domain; accepted for symmetry with :func:`plot_psd` and
        without effect here.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; ``None`` creates a figure.
    xlim : tuple of float, optional
        Frequency limits in Hz, ``center_frequency`` included; the data is
        cropped to them before drawing.
    ylim : tuple of float, optional
        Time limits in seconds; the data is cropped to them too.
    title : str or None, default "Spectrogram"
        Title of the figure; ``None`` for none.
    cmap : str, default "viridis"
        Colormap.
    show : bool, default False
        Call ``plt.show()`` and return ``None``.
    **kwargs : Any
        Passed to ``ax.imshow`` (``ax.pcolormesh`` for a non-uniform grid).

    Returns
    -------
    (fig, ax) or None
        The figure and the Axes (an array of Axes for several panels);
        ``None`` with ``show=True``, which shows the figure instead.
    """
    signal_adapter = adapt_signal(samples, function_name="plot_spectrogram()")
    x = signal_adapter.array
    sampling_rate = signal_adapter.resolve_fact("sampling_rate", sampling_rate)
    if signal_adapter.signal is not None:
        center_frequency = signal_adapter.resolve_fact(
            "center_frequency", center_frequency
        )
    elif center_frequency is None:
        center_frequency = 0.0
    return _plot_spectrogram(
        x,
        sampling_rate=sampling_rate,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend=detrend,
        return_onesided=return_onesided,
        scaling=scaling,
        mode=mode,
        center_frequency=center_frequency,
        domain=domain,
        ax=ax,
        xlim=xlim,
        ylim=ylim,
        title=title,
        cmap=cmap,
        show=show,
        **kwargs,
    )


def _plot_spectrogram(
    samples: Any,
    *,
    sampling_rate: Any,
    window: Any,
    nperseg: Any,
    noverlap: Any,
    nfft: Any,
    detrend: Any,
    return_onesided: Any,
    scaling: Any,
    mode: Any,
    center_frequency: Any,
    domain: Any,
    ax: Any,
    xlim: Any,
    ylim: Any,
    title: Any,
    cmap: Any,
    show: Any,
    **kwargs: Any,
) -> tuple[Any, Any] | None:
    """Render array data for :func:`plot_spectrogram` (the public boundary resolves the Signal)."""
    logger.debug("Generating spectrogram plot (sampling_rate=%s Hz).", sampling_rate)

    samples, xp, _ = dispatch(samples)

    # Handle Multichannel (MIMO)
    # Convention: (Channels, Time)
    if samples.ndim > 1:
        num_channels = samples.shape[0]

        fig, axes, targets = _channel_axes(num_channels, ax)

        for i in range(num_channels):
            channel_samples = samples[i]
            target_ax = targets[i]
            ch_title = f"{title} (Ch {i})" if title else f"Channel {i}"

            _plot_spectrogram(
                channel_samples,
                sampling_rate=sampling_rate,
                window=window,
                nperseg=nperseg,
                noverlap=noverlap,
                nfft=nfft,
                detrend=detrend,
                return_onesided=return_onesided,
                scaling=scaling,
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
    spec = spectral.spectrogram(
        samples,
        sampling_rate=sampling_rate,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend=detrend,
        return_onesided=return_onesided,
        scaling=scaling,
        mode=mode,
    )

    # The axes are small: bring them over to pick the crop; the values are
    # cropped and converted on the device and transferred once.
    f = np.asarray(to_device(spec.frequencies, "cpu"))
    t = np.asarray(to_device(spec.times, "cpu"))
    Sxx, xp, _ = dispatch(spec.values)

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
        Sxx_plot = 10 * xp.log10(Sxx_slice + 1e-20)
    elif mode in ("complex", "magnitude"):
        Sxx_plot = 10 * xp.log10(xp.abs(Sxx_slice) ** 2 + 1e-20)
    else:
        # Angle, phase, etc., plot linearly
        Sxx_plot = Sxx_slice
    # Time on the y-axis: (len(t_plot), len(f_plot)).
    image = np.asarray(to_device(Sxx_plot.T, "cpu"))

    # The STFT grids are uniform, so an image with the cell edges as its
    # extent draws the same picture as pcolormesh, much faster.
    if _uniform(f_plot) and _uniform(t_plot):
        df = f_plot[1] - f_plot[0] if f_plot.size > 1 else 1.0
        dt = t_plot[1] - t_plot[0] if t_plot.size > 1 else 1.0
        extent = [
            f_plot[0] - df / 2,
            f_plot[-1] + df / 2,
            t_plot[0] - dt / 2,
            t_plot[-1] + dt / 2,
        ]
        imshow_kwargs: dict[str, Any] = {
            "origin": "lower",
            "aspect": "auto",
            "interpolation": "nearest",
            "extent": extent,
            "cmap": cmap,
        }
        imshow_kwargs.update(kwargs)
        mesh = ax.imshow(image, **imshow_kwargs)
    else:
        mesh = ax.pcolormesh(f_plot, t_plot, image, cmap=cmap, shading="auto", **kwargs)
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


def _uniform(axis: np.ndarray) -> bool:
    """True for an increasing, evenly spaced axis (or one with < 3 points)."""
    if axis.size < 3:
        return True
    step = np.diff(axis)
    return bool(np.all(step > 0) and np.allclose(step, step[0], rtol=1e-6))
