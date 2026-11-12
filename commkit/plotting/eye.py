"""Eye diagrams: :func:`plot_eye_diagram`."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from ..backend import dispatch, to_device
from ..core._signal_adapter import adapt_signal, require_integer_sps
from ..logger import logger
from ..smoothing import smooth_density_2d
from .theme import _finish, _grid_figsize

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
        line_kwargs: dict[str, Any] = {"alpha": 0.2, "color": "C0"}
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
    quantity: str | None = None,
    title: str | None = "Eye Diagram",
    vmin: float | None = None,
    vmax: float | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
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
    quantity : {"i", "q"}, optional
        Draw only the in-phase or quadrature component. ``None`` draws both
        for complex input; ``"q"`` requires complex input.
    title : str or None, default "Eye Diagram"
        Title of the figure; ``None`` for none.
    vmin, vmax : float, optional
        Color limits for ``kind="hist"``.
    xlim, ylim : tuple of float, optional
        Limits applied to the displayed axes.
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
    x, _, _ = dispatch(signal_adapter.array)
    if quantity not in (None, "i", "q"):
        raise ValueError("quantity must be i, q, or None")
    if quantity == "q" and x.dtype.kind != "c":
        raise ValueError("quantity='q' requires complex samples")
    if quantity == "i":
        x = x.real
    elif quantity == "q":
        x = x.imag
    sps = require_integer_sps(
        signal_adapter.resolve_fact("sps", sps), "plot_eye_diagram()"
    )
    logger.debug("Generating eye diagram (%s mode).", kind)
    is_complex = x.dtype.kind == "c"
    per_channel = 2 if is_complex else 1  # I and Q panels for complex samples

    def draw(axes_row: Any, row: Any, title: str | None) -> None:
        if is_complex:
            for k, (part, name) in enumerate(((row.real, "I"), (row.imag, "Q"))):
                _plot_eye_traces(
                    part,
                    sps,
                    num_symbols,
                    axes_row[k],
                    kind,
                    title=f"{title} ({name})" if title else f"{name}-Channel",
                    vmin=vmin,
                    vmax=vmax,
                    **kwargs,
                )
        else:
            _plot_eye_traces(
                row, sps, num_symbols, axes_row[0], kind,
                title=title, vmin=vmin, vmax=vmax, **kwargs,
            )  # fmt: skip

    many = isinstance(ax, (list, tuple, np.ndarray))
    if x.ndim > 1:
        num_channels = x.shape[0]
        if ax is None:  # rows: channels, columns: components
            fig, axes = plt.subplots(
                num_channels,
                per_channel,
                figsize=_grid_figsize(num_channels, per_channel),
                squeeze=False,
            )
        elif many:
            axes_flat = np.array(ax).flatten()
            if len(axes_flat) < num_channels * per_channel:
                raise ValueError(
                    f"Not enough axes provided. Need {num_channels * per_channel}."
                )
            axes = axes_flat[: num_channels * per_channel].reshape(
                num_channels, per_channel
            )
            fig = axes[0, 0].figure
        else:
            raise ValueError(
                "For multichannel eye diagram, you must provide a list of axes."
            )
        for i in range(num_channels):
            draw(axes[i], x[i], f"{title} (Ch {i})" if title else f"Channel {i}")
        result: tuple[Any, Any] = (fig, axes)
    else:
        target: Any = ax
        if target is None:
            if is_complex:
                fig, target = plt.subplots(1, 2, figsize=_grid_figsize(1, 2))
            else:
                fig, target = plt.subplots(1, 1)
            many = is_complex
        else:
            fig = (target[0] if many else target).figure
        if is_complex and (not many or len(target) < 2):
            raise ValueError(
                "For complex signals, 'ax' must be a list/tuple of at least 2 axes."
            )
        draw(target if many else [target], x, title)
        result = (fig, target)
    return _finish(result, show, xlim=xlim, ylim=ylim)
