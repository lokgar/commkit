"""Eye diagrams: :func:`plot_eye_diagram`."""

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

    ``numpy.interp`` of every row at once: gather the two neighbors of each
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
    vmin: float | None = None,
    vmax: float | None = None,
    **kwargs: Any,
) -> None:
    """Internal helper to plot eye diagram traces for a single signal component.

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
    vmin, vmax : float, optional
        Color limits of the ``"hist"`` image.
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

        # 2-D histogram, time (x) by amplitude (y).  Trace column j is time
        # j * num_symbols / (L - 1), which falls in time bin j of L equal bins
        # over [0, num_symbols]; the amplitude bin is computed per sample, so
        # one bincount of (amplitude bin, column) gives the (bins_y, L) image.
        bins_y = 500
        lo_hi = to_device(xp.stack([xp.min(traces), xp.max(traces)]), "cpu")
        y_min, y_max = float(lo_hi[0]), float(lo_hi[1])
        y_range = (y_max - y_min) or 1.0
        y_lo, y_hi = y_min - 0.1 * y_range, y_max + 0.1 * y_range
        yb = xp.floor((traces - y_lo) * (bins_y / (y_hi - y_lo))).astype(xp.int64)
        yb = xp.clip(yb, 0, bins_y - 1)  # the upper edge closes the last bin
        flat = yb * trace_len + xp.arange(trace_len)[None, :]
        counts = xp.bincount(flat.ravel(), minlength=bins_y * trace_len)
        h = counts.reshape(bins_y, trace_len).astype(xp.float64)
        h = smooth_density_2d(h, sigma=1.0)

        # Normalize on the host, where the image goes anyway.
        h = np.asarray(to_device(h, "cpu"))
        h_max = h.max()
        if h_max > 0:
            h = h / h_max

        # Plot using imshow
        imshow_kwargs = {
            "origin": "lower",
            "extent": [0.0, float(num_symbols), y_lo, y_hi],
            "aspect": "auto",
            "cmap": "inferno",
            "vmin": vmin,
            "vmax": vmax,
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
    """Plot the eye diagram of a signal.

    Overlays segments of ``num_symbols`` symbol periods, aligned to the
    symbol clock, as a 2-D density histogram or as line traces.

    Parameters
    ----------
    samples : array_like or Signal
        Input samples, usually matched-filtered, ``(N,)`` or ``(C, N)``.
    sps : float, optional
        Samples per symbol, an integer value.  Taken from the Signal;
        required for array input.
    ax : matplotlib.axes.Axes or array_like of Axes, optional
        Axes to draw on, two per channel (I and Q) for complex samples;
        ``None`` creates a figure.
    num_symbols : int, default 2
        Symbol periods shown in each eye window.
    kind : {"hist", "line"}, default "hist"
        ``"hist"``: a 2-D density histogram (recommended for noisy
        signals); ``"line"``: overlaid traces.
    title : str or None, default "Eye Diagram"
        Title of the figure; ``None`` for none.
    vmin, vmax : float, optional
        Color limits for ``kind="hist"``.
    show : bool, default False
        Call ``plt.show()`` and return ``None``.
    **kwargs : Any
        Passed to Matplotlib: ``imshow`` for ``"hist"``, the ``LineCollection``
        of traces for ``"line"``.

    Returns
    -------
    (fig, ax) or None
        The figure and the Axes (an array of Axes for several panels);
        ``None`` with ``show=True``, which shows the figure instead.

    Notes
    -----
    For an open eye, synchronize the signal (no frequency or timing offset)
    and matched-filter it first.
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
            vmin=vmin,
            vmax=vmax,
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
            vmin=vmin,
            vmax=vmax,
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
            vmin=vmin,
            vmax=vmax,
            **kwargs,
        )

    if show:
        plt.show()
        return None
    return fig, ax
