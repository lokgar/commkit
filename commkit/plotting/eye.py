"""Eye diagram plots."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from ..backend import dispatch, to_device
from ..core._signal_adapter import adapt_signal, require_integer_sps
from ..logger import logger
from ..smoothing import smooth_density_2d
from .theme import _grid_figsize

__all__ = ["plot_eye_diagram"]


def _interp_rows(traces: Any, width: int, xp: Any) -> Any:
    """Each row linearly resampled to ``width`` points over the same span.

    ``numpy.interp`` of every row at once: gather the two neighbours of each
    new position and blend them (a Python loop over rows is one launch per
    trace on the GPU).
    """
    n = traces.shape[-1]
    x_new = xp.linspace(0, n - 1, width, dtype=xp.float64)
    i0 = xp.minimum(xp.floor(x_new).astype(xp.int64), n - 2)
    frac = (x_new - i0).astype(traces.real.dtype)
    return traces[:, i0] * (1 - frac) + traces[:, i0 + 1] * frac


def _plot_eye_traces(
    samples: Any,
    sps: float,
    num_symbols: int,
    ax: Any,
    kind: str,
    title: str | None,
    **kwargs: Any,
) -> None:
    """
    Internal helper to plot eye diagram traces for a single signal component.

    Parameters
    ----------
    samples : array_like
        The component samples (e.g., Real or Imaginary part).
    sps : float
        Samples per symbol.
    num_symbols : int
        Number of symbol periods per window.
    ax : matplotlib.axes.Axes
        The axis to plot on.
    kind : {"line", "hist"}
        Plotting strategy.
    title : str, optional
        Title for the subplot.
    **kwargs : Any
        Additional plotting parameters.
    """
    samples, xp, _ = dispatch(samples)

    # Normalize to max amplitude 1.0
    from ..math import normalize

    samples = normalize(samples, mode="peak")

    # We want to include the endpoint to avoid a gap at the end of the plot
    # So we need one extra sample per trace
    trace_len = int(num_symbols * sps) + 1
    if trace_len > samples.shape[0]:
        raise ValueError("Signal is shorter than the required trace length.")

    # Calculate number of traces
    # We slide by 1 symbol period (sps)
    num_traces = (samples.shape[0] - trace_len) // int(sps) + 1

    if kind == "line":
        # Limit traces for performance/visuals
        max_traces = 5000
        if num_traces > max_traces:
            skip = num_traces // max_traces
            indices = xp.arange(0, num_traces, skip)[:max_traces]
        else:
            indices = xp.arange(num_traces)

        # Vectorized trace extraction
        # indices shape: (num_traces,)
        # offsets shape: (trace_len,)
        # We want to extract samples at [indices[i] * sps + offset[j]]
        start_indices = (indices * int(sps)).astype(int)
        offsets = xp.arange(trace_len, dtype=int)

        # Matrix of indices: (num_traces, 1) + (1, trace_len) -> (num_traces, trace_len)
        gather_indices = start_indices[:, None] + offsets[None, :]

        # Gather samples
        traces = samples[gather_indices]  # (num_traces, trace_len)

        traces = np.asarray(to_device(traces, "cpu"))
        t = np.linspace(0, num_symbols, trace_len, endpoint=True)

        # One LineCollection instead of one Line2D per trace: a single artist
        # draws thousands of traces far faster.
        segments = np.empty((traces.shape[0], trace_len, 2))
        segments[:, :, 0] = t
        segments[:, :, 1] = traces
        line_kwargs: dict[str, Any] = {"alpha": 0.2, "linewidth": 1, "color": "C0"}
        line_kwargs.update(kwargs)
        ax.add_collection(LineCollection(list(segments), **line_kwargs))
        y_lo, y_hi = float(traces.min()), float(traces.max())
        pad = 0.05 * (y_hi - y_lo) or 1.0
        ax.set_ylim(y_lo - pad, y_hi + pad)

    elif kind == "hist":
        max_traces_hist = 20000
        if num_traces > max_traces_hist:
            skip = num_traces // max_traces_hist
            indices = xp.arange(0, num_traces, skip)[:max_traces_hist]
        else:
            indices = xp.arange(num_traces)

        start_indices = (indices * int(sps)).astype(int)
        offsets = xp.arange(trace_len, dtype=int)
        gather_indices = start_indices[:, None] + offsets[None, :]
        traces = samples[gather_indices]  # (num_traces, trace_len)

        # Interpolate traces
        target_width = 500
        if trace_len < target_width:
            traces = _interp_rows(traces, target_width, xp)
            trace_len = target_width

        # Create time matrix
        # Use xp.linspace
        t = xp.linspace(0, num_symbols, trace_len, endpoint=True)
        # Use xp.tile
        t_matrix = xp.tile(t, (traces.shape[0], 1))  # shape: (num_traces, trace_len)

        # Flatten
        t_flat = t_matrix.flatten()
        y_flat = traces.flatten()

        # Compute 2D histogram
        # Bins: Time (x) and Amplitude (y)
        bins_x = trace_len
        bins_y = 500

        # Add padding to Y range
        y_min, y_max = xp.min(y_flat), xp.max(y_flat)
        y_range = y_max - y_min
        if y_range == 0:
            y_range = 1.0
        y_pad = y_range * 0.1
        range_y = [float(y_min - y_pad), float(y_max + y_pad)]

        # Min/max of t_flat
        t_min, t_max = xp.min(t_flat), xp.max(t_flat)

        h, xedges, yedges = xp.histogram2d(
            t_flat,
            y_flat,
            bins=[bins_x, bins_y],
            range=[[float(t_min), float(t_max)], range_y],
        )

        h = h.T
        h = smooth_density_2d(h, sigma=1.0)

        # Normalize
        h_max = xp.max(h)
        if h_max > 0:
            h = h / h_max

        # Move to cpu for plotting
        h = to_device(h, "cpu")
        xedges = to_device(xedges, "cpu")
        yedges = to_device(yedges, "cpu")

        # Plot using imshow
        imshow_kwargs = {
            "origin": "lower",
            "extent": [xedges[0], xedges[-1], yedges[0], yedges[-1]],
            "aspect": "auto",
            "cmap": "inferno",
        }
        imshow_kwargs.update(kwargs)

        ax.imshow(h, **imshow_kwargs)  # type: ignore[arg-type]

    else:
        raise ValueError(f"Unknown kind: {kind}. Supported: 'line', 'hist'")

    ax.set_xlabel("Time [Symbol Periods]")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(0, num_symbols)
    if title is not None:
        ax.set_title(title)


def plot_eye_diagram(
    samples: Any,
    *,
    sps: float | None = None,
    ax: Any | tuple[Any, Any] | None = None,
    num_symbols: int = 2,
    kind: str = "hist",
    title: str | None = "Eye Diagram",
    vmin: float | None = None,
    vmax: float | None = None,
    show: bool = False,
    **kwargs: Any,
) -> tuple[Any, Any] | None:
    """
    Plots the eye diagram of the signal.

    Visualizes signal quality by overlapping segments of the signal
    synchronized to the symbol clock. Supports both fast trace-based
    plotting and high-definition density histograms.

    Parameters
    ----------
    samples : array_like or Signal
        Input signal samples. Usually matched-filtered.
    sps : float, optional
        Samples per symbol (a fact, an integer for windowing): taken from a
        Signal, required for arrays.
    ax : matplotlib.axes.Axes or array_like, optional
        Target axis or list of axes. For complex signals, two axes are
        required per channel (I and Q).
    num_symbols : int, default 2
        Number of symbol periods TO display in each eye window.
    kind : {"hist", "line"}, default "hist"
        Visualization mode:
        - "hist": 2D density histogram (recommended for noisy signals).
        - "line": Vectorized overlapping traces (classic look).
    title : str, optional
        Base title for the plot.
    vmin, vmax : float, optional
         Color scaling limits for "hist" mode.
    show : bool, default False
        If True, calls `plt.show()`.
    **kwargs : Any
        Additional keyword arguments passed to the plotting backend.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes or ndarray
        The axis or array of axes used.

    Notes
    -----
    The signal should typically be synchronized (no CFO or timing offset)
    and matched-filtered before plotting to produce a clear "eye".
    """
    signal_adapter = adapt_signal(samples, function_name="plot_eye_diagram()")
    x = signal_adapter.array
    sps = require_integer_sps(
        signal_adapter.resolve_fact("sps", sps), "plot_eye_diagram()"
    )
    return _plot_eye_diagram(
        x,
        sps=sps,
        ax=ax,
        num_symbols=num_symbols,
        kind=kind,
        title=title,
        vmin=vmin,
        vmax=vmax,
        show=show,
        **kwargs,
    )


def _plot_eye_diagram(
    samples: Any,
    *,
    sps: Any,
    ax: Any,
    num_symbols: Any,
    kind: Any,
    title: Any,
    vmin: Any,
    vmax: Any,
    show: Any,
    **kwargs: Any,
) -> tuple[Any, Any] | None:
    """Render array data for :func:`plot_eye_diagram` (the public boundary resolves the Signal)."""
    logger.debug("Generating eye diagram (%s mode).", kind)

    # Dispatch to check backend
    samples, xp, _ = dispatch(samples)

    # Convention: (Channels, Time)
    if samples.ndim > 1:
        num_channels = samples.shape[0]

        # Complex eye uses 2 axes (I/Q). So for N channels we need 2*N axes.
        is_complex = xp.iscomplexobj(samples)
        axes_per_channel = 2 if is_complex else 1

        if ax is None:
            # Grid: Rows = Channels, Cols = Components
            fig, axes = plt.subplots(
                num_channels,
                axes_per_channel,
                figsize=_grid_figsize(num_channels, axes_per_channel),
                squeeze=False,  # Ensure 2D array
            )
        else:
            # User provided axes. Must be flat list or correct shape
            # We assume user knows what they are doing or we do best effort
            if isinstance(ax, (np.ndarray, list, tuple)):
                # Flatten
                axes_flat = np.array(ax).flatten()
                if len(axes_flat) < num_channels * axes_per_channel:
                    raise ValueError(
                        f"Not enough axes provided. Need {num_channels * axes_per_channel}."
                    )
                # Reshape to (Channels, Components)
                axes = axes_flat[: num_channels * axes_per_channel].reshape(
                    num_channels, axes_per_channel
                )
                fig = axes[0, 0].figure
            else:
                raise ValueError(
                    "For multichannel eye diagram, you must provide a list of axes."
                )

        for i in range(num_channels):
            channel_samples = samples[i]
            ch_axes = axes[i]

            ch_title = f"{title} (Ch {i})" if title else f"Channel {i}"

            # Recursive call with 1D sample
            _plot_eye_diagram(
                channel_samples,
                sps=sps,
                ax=ch_axes,
                num_symbols=num_symbols,
                kind=kind,
                title=ch_title,
                vmin=vmin,
                vmax=vmax,
                show=False,
                **kwargs,
            )

        if show:
            plt.show()
            return None
        return fig, axes

    # --- 1D Logic ---

    is_complex = xp.iscomplexobj(samples)

    if ax is None:
        if is_complex:
            fig, ax = plt.subplots(1, 2, figsize=_grid_figsize(1, 2))
        else:
            fig, ax = plt.subplots(1, 1)
            # Handle the fact that plt.subplots(1, 1) returns a single ax, not a list
    else:
        if isinstance(ax, (list, tuple, np.ndarray)):
            fig = ax[0].figure
        else:
            fig = ax.figure

    if is_complex:
        if not isinstance(ax, (list, tuple, np.ndarray)) or len(ax) < 2:
            raise ValueError(
                "For complex signals, 'ax' must be a list/tuple of at least 2 axes."
            )

        # Plot I
        _plot_eye_traces(
            samples.real,
            sps,
            num_symbols,
            ax[0],
            kind,
            title=f"{title} (I)" if title else "I-Channel",
            **kwargs,
        )

        # Plot Q
        _plot_eye_traces(
            samples.imag,
            sps,
            num_symbols,
            ax[1],
            kind,
            title=f"{title} (Q)" if title else "Q-Channel",
            **kwargs,
        )
    else:
        # If user passed a list of axes for real signal, use the first one
        target_ax = ax
        if isinstance(ax, (list, tuple, np.ndarray)):
            target_ax = ax[0]

        _plot_eye_traces(
            samples,
            sps,
            num_symbols,
            target_ax,
            kind,
            title=title,
            **kwargs,
        )

    if show:
        plt.show()
        return None
    return fig, ax
