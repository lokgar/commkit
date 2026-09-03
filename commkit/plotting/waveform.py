"""Time-domain waveform plots."""

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


def plot_time_domain(
    samples: Any,
    sampling_rate: float = 1.0,
    start_symbol: int = 0,
    num_symbols: int | None = None,
    sps: float | None = None,
    ax: Any | None = None,
    title: str | None = "Waveform",
    show: bool = False,
    **kwargs: Any,
) -> tuple[Any, Any] | None:
    """
    Plots the time-domain representation of the signal.

    For complex signals, both In-Phase (I) and Quadrature (Q) components
    are plotted. Handles SI scaling (s, ms, us, etc.) for the time axis.

    Parameters
    ----------
    samples : array_like or Signal
        Input signal samples. Shape: (..., N_samples).
    sampling_rate : float, default 1.0
        Sampling rate in Hz.
    start_symbol : int, default 0
        The starting symbol to plot.
    num_symbols : int, optional
        Limit plot to a specific number of symbol periods. Requires `sps`.
    sps : float, optional
        Samples per symbol (required if `num_symbols` is used).
    ax : matplotlib.axes.Axes, optional
        Existing axis to plot on.
    title : str, optional
        Plot title. Defaults to "Waveform".
    show : bool, default False
        If True, calls `plt.show()`.
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
        return plot_time_domain(
            sig.samples,
            sampling_rate=sig.sampling_rate,
            start_symbol=start_symbol,
            num_symbols=num_symbols,
            sps=sig.sps,
            ax=ax,
            title=title,
            show=show,
            **kwargs,
        )

    logger.debug("Generating time-domain plot.")

    samples, xp, _ = dispatch(samples)

    # Handle Multichannel
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

            # Determine target axis using 2D indexing
            row, col = divmod(i, axes.shape[1])
            target_ax = axes[row, col] if row < axes.shape[0] else axes.flat[-1]

            ch_title = f"{title} (Ch {i})" if title else f"Channel {i}"

            plot_time_domain(
                channel_samples,
                sampling_rate=sampling_rate,
                start_symbol=start_symbol,
                num_symbols=num_symbols,
                sps=sps,
                ax=target_ax,
                title=ch_title,
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

    samples = to_device(samples, "cpu")

    start_idx = int(start_symbol * sps) if sps is not None else int(start_symbol)

    if num_symbols is not None and sps is not None:
        limit = start_idx + int(num_symbols * sps)
        if limit > len(samples):
            limit = len(samples)
            logger.warning(
                "Limit exceeds number of symbols. Plotting up to last symbol."
            )
        plot_samples = samples[start_idx:limit]
    else:
        plot_samples = samples[start_idx:]

    time_axis = np.arange(len(plot_samples)) / sampling_rate

    if np.iscomplexobj(plot_samples):
        ax.plot(
            time_axis,
            plot_samples.real,
            label="I",
            **kwargs,
        )
        ax.plot(
            time_axis,
            plot_samples.imag,
            label="Q",
            **kwargs,
        )
        ax.legend()
    else:
        ax.plot(
            time_axis,
            plot_samples,
            **kwargs,
        )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    _set_eng_formatter(ax, "x", "s")
    if title is not None:
        ax.set_title(title)

    if show:
        plt.show()
        return None
    return fig, ax
