"""Shared plotting theme, layout, and formatting helpers."""

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

from ..backend import to_device
from ..logger import logger


def apply_default_theme() -> None:
    """
    Applies the library's default visual theme to Matplotlib.

    This theme configures publication-quality defaults, including:
    - Roboto or standard sans-serif typography.
    - Optimized figure dimensions and DPI.
    - Consistent grid styling and axis formatting.
    - Improved LaTeX math rendering.

    Notes
    -----
    This function modifies `matplotlib.rcParams` globally. It is recommended
    to call this at the start of a script or notebook for consistent styling.
    """
    logger.debug("Applying default plotting theme.")
    try:
        font_prop = fm.FontProperties(family="Roboto", weight="regular")
        fm.findfont(font_prop, fallback_to_default=False)
        font_name = "Roboto"
    except ValueError:
        font_name = "sans"
        logger.warning("Roboto font not found, falling back to default sans-serif.")

    plt.style.use("dark_background")

    mpl.rcParams.update(
        {
            "figure.figsize": (7, 4.9),
            "font.family": font_name,
            "font.size": 12,
            "legend.fontsize": 10,
            "lines.linewidth": 2,
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
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "xtick.minor.width": 1,
            "ytick.minor.width": 1,
            "xtick.minor.size": 2,
            "ytick.minor.size": 2,
            "xtick.top": True,
            "ytick.right": True,
        }
    )

    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = font_name
    plt.rcParams["mathtext.it"] = f"{font_name}:italic"
    plt.rcParams["mathtext.bf"] = f"{font_name}:bold"


def _get_axis(ax):
    """Get-or-create a single ``Axes``: returns ``(fig, axi)``.

    If ``ax`` is ``None``, creates a new 1x1 figure/axes pair (theme default
    figsize). Otherwise reuses the given ``Axes`` and its parent ``Figure``.
    For the single-``Axes`` case only - multi-panel functions build their
    own subplot grid via :func:`_create_subplot_grid`/:func:`_grid_figsize`.
    """
    if ax is None:
        fig, axi = plt.subplots(1, 1)
    else:
        axi = ax
        fig = axi.figure
    return fig, axi


def _finish(result, show: bool):
    """Show-or-return: call ``plt.show()`` and return ``None``, or return ``result``.

    ``result`` is whatever the caller wants returned when ``show=False``
    (typically ``(fig, axi)`` or ``(fig, axes)``).
    """
    if show:
        plt.show()
        return None
    return result


def _create_subplot_grid(num_axes: int, max_cols: int = 2) -> tuple[int, int]:
    """
    Computes a grid layout (rows, cols) for a given number of axes.

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
    """
    Figsize for a subplot grid that preserves the default single-panel size.

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


def _square_figsize(n: int = 1) -> tuple[float, float]:
    """
    Square panel size derived from the default figure height.

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

    Buckets the data and keeps the per-bucket min **and** max (emitted in
    x-order), so sharp features - spectral peaks, window edges, glitches -
    survive, unlike plain striding.  Matplotlib renders every vertex, so for
    long oversampled records (N ≳ 10⁵) plotting the raw trace is the dominant
    cost; reducing to a few thousand points is visually identical but orders of
    magnitude faster.

    Parameters
    ----------
    x, y : np.ndarray
        1-D arrays of equal length.
    max_points : int, default 4000
        Approximate cap on the number of plotted points.  Returned length is
        ``≈ max_points`` (``2`` per bucket).  Pass ``<= 0`` to disable.

    Returns
    -------
    (x_dec, y_dec) : tuple of np.ndarray
        Decimated arrays, or the inputs unchanged when already short enough.
    """
    n = len(y)
    if max_points <= 0 or n <= max_points:
        return x, y

    n_buckets = max(1, max_points // 2)
    bucket = n // n_buckets
    trimmed = n_buckets * bucket
    yr = y[:trimmed].reshape(n_buckets, bucket)
    xr = x[:trimmed].reshape(n_buckets, bucket)

    rows = np.arange(n_buckets)
    i_min = yr.argmin(axis=1)
    i_max = yr.argmax(axis=1)
    # Emit the two extrema per bucket in their original x-order so the line
    # does not zig-zag backwards.
    first_is_min = i_min <= i_max
    x_out = np.empty(n_buckets * 2, dtype=x.dtype)
    y_out = np.empty(n_buckets * 2, dtype=y.dtype)
    x_out[0::2] = np.where(first_is_min, xr[rows, i_min], xr[rows, i_max])
    y_out[0::2] = np.where(first_is_min, yr[rows, i_min], yr[rows, i_max])
    x_out[1::2] = np.where(first_is_min, xr[rows, i_max], xr[rows, i_min])
    y_out[1::2] = np.where(first_is_min, yr[rows, i_max], yr[rows, i_min])

    if trimmed < n:  # keep the tail the reshape dropped
        x_out = np.concatenate([x_out, x[trimmed:]])
        y_out = np.concatenate([y_out, y[trimmed:]])
    return x_out, y_out


def _set_eng_formatter(ax, which: str, unit: str) -> None:
    """Apply an engineering (SI-prefix) tick formatter to an axis."""
    fmt = mpl.ticker.EngFormatter(unit=unit, sep=" ")
    if which in ("x", "both"):
        ax.xaxis.set_major_formatter(fmt)
    if which in ("y", "both"):
        ax.yaxis.set_major_formatter(fmt)


def _as_channels(arr) -> np.ndarray:
    """Bring to CPU and promote ``(N,)`` -> ``(1, N)``."""
    arr = np.atleast_2d(np.asarray(to_device(arr, "cpu")))
    return arr
