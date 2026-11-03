"""Plot theme and shared layout and formatting helpers.

:func:`apply_default_theme` sets the library's Matplotlib style; the private
helpers (axes and figure layout, SI-prefixed tick labels, min/max
decimation of long traces) are shared by the plot modules.
"""

from typing import Any

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

from ..backend import get_array_module, to_device
from ..logger import logger

__all__ = ["apply_default_theme"]


def apply_default_theme() -> None:
    """Apply the library's Matplotlib style.

    Matplotlib's ``dark_background`` style with Roboto (or the default
    sans-serif font, logged at INFO, when Roboto is not installed)
    for text and math, a 7 x 4.9 inch figure with constrained layout,
    300-dpi saving, a light grid, and inward ticks on all four sides.

    Notes
    -----
    Changes ``matplotlib.rcParams`` globally, so call it once at the start of
    a script or notebook.  The library never calls it on its own.
    """
    logger.debug("Applying default plotting theme.")
    try:
        font_prop = fm.FontProperties(family="Roboto", weight="regular")
        fm.findfont(font_prop, fallback_to_default=False)
        font_name = "Roboto"
    except ValueError:
        font_name = "sans"
        logger.info("Roboto font not found, falling back to default sans-serif.")

    plt.style.use("dark_background")

    mpl.rcParams.update(
        {
            "figure.figsize": (7, 4.9),
            "figure.titlesize": 18,
            "font.family": font_name,
            "font.size": 16,
            "legend.fontsize": 14,
            "lines.linewidth": 2,
            "axes.titlesize": 18,
            "axes.linewidth": 1,
            "axes.grid": True,
            "grid.alpha": 0.4,
            "axes.titleweight": "bold",
            "figure.constrained_layout.use": True,
            "savefig.dpi": 300,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 1,
            "ytick.major.width": 1,
            "xtick.major.size": 8,
            "ytick.major.size": 8,
            "xtick.minor.width": 1,
            "ytick.minor.width": 1,
            "xtick.minor.size": 4,
            "ytick.minor.size": 4,
            "xtick.top": True,
            "ytick.right": True,
        }
    )

    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = font_name
    plt.rcParams["mathtext.it"] = f"{font_name}:italic"
    plt.rcParams["mathtext.bf"] = f"{font_name}:bold"


def _get_axis(ax):
    """Return ``(fig, axi)`` for one ``Axes``, creating them when needed.

    With ``ax=None`` a new 1x1 figure at the theme's default size is
    created; otherwise the given ``Axes`` and its ``Figure`` are reused.
    For single-panel plots only: multi-panel functions build their grid with
    :func:`_create_subplot_grid` and :func:`_grid_figsize`.
    """
    if ax is None:
        fig, axi = plt.subplots(1, 1)
    else:
        axi = ax
        fig = axi.figure
    return fig, axi


def _select(quantity, choices, **single_only) -> tuple[str, ...]:
    """Panels a multi-panel plot draws: all ``choices``, or the one ``quantity``.

    Raises ``ValueError`` for a ``quantity`` not in ``choices``, and for an
    option in ``single_only`` (a label, limits, a title of one panel) given
    while every panel is drawn.
    """
    if quantity is None:
        given = [name for name, value in single_only.items() if value is not None]
        if given:
            raise ValueError(f"{', '.join(given)} require a selected quantity")
        return tuple(choices)
    if quantity not in choices:
        raise ValueError(
            f"quantity must be one of {', '.join(choices)} or None, got {quantity!r}"
        )
    return (quantity,)


def _panel_grid(ax, rows: int, cols: int):
    """Return ``(fig, axes)``, ``axes`` a ``(rows, cols)`` object array.

    With ``ax=None`` a new figure: one Axes at the theme's default size for a
    single panel, otherwise a grid sized by :func:`_grid_figsize`.  Given axes
    are reshaped to ``(rows, cols)``; they must be that many, of one figure.
    """
    if ax is None:
        if rows * cols == 1:
            fig, single = plt.subplots()
            return fig, np.asarray([[single]], dtype=object)
        fig, grid = plt.subplots(
            rows, cols, figsize=_grid_figsize(rows, cols), squeeze=False
        )
        return fig, np.asarray(grid, dtype=object)
    axes = np.asarray(ax, dtype=object)
    if axes.size != rows * cols:
        raise ValueError(f"ax must contain {rows * cols} axes, got {axes.size}")
    axes = axes.reshape(rows, cols)
    fig = axes.flat[0].figure
    if any(axis.figure is not fig for axis in axes.flat):
        raise ValueError("ax must contain axes from one figure")
    return fig, axes


def _series_label(label, part=None, default=None):
    """Legend entry of one curve: ``label`` prefixed to ``part``.

    ``part`` names the curve within the plot (a channel, a bit, a tap) and
    ``default`` stands in when there is neither a label nor a part.
    """
    if label:
        return f"{label} {part}" if part else label
    return part or default


def _finish(result, show: bool, *, xlim=None, ylim=None, legend=None):
    """Show-or-return: call ``plt.show()`` and return ``None``, or return ``result``.

    ``result`` is whatever the caller wants returned when ``show=False``
    (typically ``(fig, axi)`` or ``(fig, axes)``).  ``xlim`` and ``ylim``
    apply to every returned Axes; ``legend=True`` draws a legend on each
    Axes with labeled artists, ``False`` removes them, ``None`` keeps what
    the plot drew.
    """
    if xlim is not None or ylim is not None or legend is not None:
        axes = np.asarray(result[1], dtype=object).reshape(-1)
        for ax in axes:
            if xlim is not None:
                ax.set_xlim(xlim)
            if ylim is not None:
                ax.set_ylim(ylim)
            if legend is True:
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(handles, labels)
            elif legend is False and ax.get_legend() is not None:
                ax.get_legend().remove()
    if show:
        plt.show()
        return None
    return result


def _create_subplot_grid(num_axes: int, max_cols: int = 2) -> tuple[int, int]:
    """Compute a grid layout (rows, cols) for a given number of axes.

    Limits the maximum number of columns to prevent excessively wide figures.

    Parameters
    ----------
    num_axes : int
        Total number of subplots required.
    max_cols : int, default 2
        Maximum allowed number of columns.

    Returns
    -------
    nrows : int
        Number of rows in the grid.
    ncols : int
        Number of columns in the grid.
    """
    if num_axes <= max_cols:
        return 1, num_axes
    ncols = max_cols
    nrows = (num_axes + ncols - 1) // ncols  # Ceiling division
    return nrows, ncols


def _grid_figsize(
    nrows: int, ncols: int, panel: tuple[float, float] | None = None
) -> tuple[float, float]:
    """Figsize for a subplot grid that preserves the default single-panel size.

    Tiles ``panel`` (default: ``rcParams["figure.figsize"]``) across the
    grid, so a single row keeps the default height, a single column keeps
    the default width, and an NxN grid keeps the default size per panel.

    Parameters
    ----------
    nrows, ncols : int
        Grid shape.
    panel : (float, float), optional
        Per-panel ``(width, height)`` in inches. Defaults to the theme's
        ``figure.figsize``.

    Returns
    -------
    (width, height) : tuple of float
    """
    w, h = panel if panel is not None else mpl.rcParams["figure.figsize"]
    return (w * ncols, h * nrows)


def _channel_axes(
    num_channels: int, ax: Any, panel: tuple[float, float] | None = None
) -> tuple[Any, Any, list[Any]]:
    """The figure, its Axes grid and one target Axes per channel.

    ``ax=None`` creates a grid of ``panel``-sized Axes (two columns at most);
    one Axes for several channels overlays them, with a warning; several
    Axes are used in order, the last one taking any surplus channels.
    """
    if ax is None:
        nrows, ncols = _create_subplot_grid(num_channels)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=_grid_figsize(nrows, ncols, panel=panel),
            squeeze=False,
        )
    elif not isinstance(ax, (list, tuple, np.ndarray)):
        logger.warning(
            "Multiple channels detected but single axis provided. Overlaying plots."
        )
        axes = np.array([[ax] * num_channels])
        fig = ax.figure
    else:
        axes = np.atleast_2d(ax)
        fig = axes.flat[0].figure
    targets = []
    for i in range(num_channels):
        row, col = divmod(i, axes.shape[1])
        targets.append(axes[row, col] if row < axes.shape[0] else axes.flat[-1])
    return fig, axes, targets


def _square_figsize(n: int = 1) -> tuple[float, float]:
    """Square panel size derived from the default figure height.

    Used for constellation-style plots that require an equal aspect ratio,
    where the default (non-square) ``figure.figsize`` doesn't apply directly.
    Tiling ``n`` panels preserves the same per-panel size as ``_grid_figsize``.
    """
    _, h = mpl.rcParams["figure.figsize"]
    return (h * n, h * n)


def _decimate_minmax(
    x: np.ndarray, y: np.ndarray, max_points: int = 4000
) -> tuple[np.ndarray, np.ndarray]:
    """Down-sample a line for fast plotting while preserving its envelope.

    Buckets the data and keeps both the minimum and the maximum of each
    bucket (in x order), so sharp features (spectral peaks, window edges,
    glitches) survive, unlike plain striding.  Matplotlib renders every vertex, so for
    long oversampled records (N ≳ 10⁵) plotting the raw trace is the dominant
    cost; reducing to a few thousand points is visually identical but orders of
    magnitude faster.

    Parameters
    ----------
    x, y : ndarray
        1-D arrays of equal length, on one device (NumPy or CuPy): a GPU
        trace is reduced there, so only the plotted points are transferred.
    max_points : int, default 4000
        Approximate cap on the number of plotted points.  Returned length is
        ``≈ max_points`` (``2`` per bucket).  Pass ``<= 0`` to disable.

    Returns
    -------
    x_dec, y_dec : ndarray
        Decimated arrays on the inputs' device, or the inputs unchanged when
        already short enough.
    """
    xp = get_array_module(y)
    n = len(y)
    if max_points <= 0 or n <= max_points:
        return x, y

    n_buckets = max(1, max_points // 2)
    bucket = n // n_buckets
    trimmed = n_buckets * bucket
    yr = y[:trimmed].reshape(n_buckets, bucket)
    xr = x[:trimmed].reshape(n_buckets, bucket)

    rows = xp.arange(n_buckets)
    i_min = yr.argmin(axis=1)
    i_max = yr.argmax(axis=1)
    # Emit the two extrema per bucket in their original x-order so the line
    # does not zig-zag backwards.
    first_is_min = i_min <= i_max
    x_out = xp.empty(n_buckets * 2, dtype=x.dtype)
    y_out = xp.empty(n_buckets * 2, dtype=y.dtype)
    x_out[0::2] = xp.where(first_is_min, xr[rows, i_min], xr[rows, i_max])
    y_out[0::2] = xp.where(first_is_min, yr[rows, i_min], yr[rows, i_max])
    x_out[1::2] = xp.where(first_is_min, xr[rows, i_max], xr[rows, i_min])
    y_out[1::2] = xp.where(first_is_min, yr[rows, i_max], yr[rows, i_min])

    if trimmed < n:  # keep the tail the reshape dropped
        x_out = xp.concatenate([x_out, x[trimmed:]])
        y_out = xp.concatenate([y_out, y[trimmed:]])
    return x_out, y_out


def _set_eng_formatter(ax, which: str) -> None:
    """Apply an engineering (SI-prefix) tick formatter to an axis."""
    fmt = mpl.ticker.EngFormatter(sep=" ")
    if which in ("x", "both"):
        ax.xaxis.set_major_formatter(fmt)
    if which in ("y", "both"):
        ax.yaxis.set_major_formatter(fmt)


def _as_channels(arr) -> np.ndarray:
    """Bring to CPU and promote ``(N,)`` -> ``(1, N)``."""
    arr = np.atleast_2d(np.asarray(to_device(arr, "cpu")))
    return arr
