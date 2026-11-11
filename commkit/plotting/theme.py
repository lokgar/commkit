"""Plot theme and shared layout and formatting helpers.

:func:`apply_default_theme` sets the library's Matplotlib style; the private
helpers (axes and figure layout, SI-prefixed axis units, min/max
decimation of long traces) are shared by the plot modules.
"""

import re
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


def _draw_channels(
    x: Any,
    ax: Any,
    title: str | None,
    draw: Any,
    *,
    panel: tuple[float, float] | None = None,
) -> tuple[Any, Any]:
    """Draw ``(N,)`` or ``(C, N)`` samples, one panel per channel.

    ``draw(axi, row, panel_title)`` draws one channel's ``row``.  Several
    channels get the grid of :func:`_channel_axes`, titled
    ``"{title} (Ch i)"`` (``"Channel i"`` without a title); one channel gets
    ``ax`` or a new figure (of size ``panel`` when given).  Returns
    ``(fig, axes)`` or ``(fig, ax)`` for :func:`_finish`.
    """
    if x.ndim > 1:
        fig, axes, targets = _channel_axes(x.shape[0], ax, panel=panel)
        for i, target in enumerate(targets):
            draw(target, x[i], f"{title} (Ch {i})" if title else f"Channel {i}")
        return fig, axes
    if ax is None:
        fig, axi = plt.subplots(figsize=panel) if panel is not None else plt.subplots()
    else:
        fig, axi = ax.figure, ax
    draw(axi, x, title)
    return fig, axi


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


def _device_channels(a: Any) -> Any:
    """``a`` as a 2-D array on its own device, ``(N,)`` promoted to ``(1, N)``."""
    xp = get_array_module(a)
    a = xp.asarray(a)
    return a[None, :] if a.ndim == 1 else a


def _trace(
    y: Any, max_points: int, *, x0: float = 0.0, dx: float = 1.0, x: Any = None
) -> tuple[np.ndarray, np.ndarray]:
    """Host points of the 1-D trace ``y`` (on any device), ready to draw.

    The abscissa is ``x0 + dx * index``, or ``x`` on ``y``'s device.  A trace
    longer than ``max_points`` is reduced to its min/max envelope
    (:func:`_decimate_minmax`) on its device, so only the drawn points are
    transferred.
    """
    xp = get_array_module(y)
    if x is None:
        x = xp.arange(y.shape[-1], dtype=np.float64) * dx + x0
    x, y = _decimate_minmax(x, y, max_points)
    return _host(x), _host(y)


_SI_PREFIXES = {
    -24: "y", -21: "z", -18: "a", -15: "f", -12: "p", -9: "n", -6: "µ",
    -3: "m", 0: "", 3: "k", 6: "M", 9: "G", 12: "T", 15: "P", 18: "E",
}  # fmt: skip
_UNIT = re.compile(r"\[([^\[\]]+)\](?!.*\[)")  # the last [unit]


class _SIAxisFormatter(mpl.ticker.Formatter):
    """One SI prefix per axis: in the axis label's unit, not on every tick.

    At every draw the prefix is chosen from the largest visible tick (the
    engineering exponent, a multiple of 3), the tick labels are the values
    in that unit and the label's last ``[unit]`` becomes
    ``[<prefix><unit>]``: ``Frequency [Hz]`` reads ``Frequency [GHz]`` with
    ticks ``-10, 0, 10``.  The label is rewritten from the text it had when
    set by the plot (a label set later, by the plot or the user, is the new
    base), so zooming or new limits rechoose the prefix.  A label without a
    ``[unit]`` keeps the prefix on the ticks.  A log axis spans decades, which
    no single prefix fits, so its ticks keep their own prefixes (``1 k``,
    ``1 M``) and its label the base unit.
    """

    def __init__(self) -> None:
        self._eng = mpl.ticker.EngFormatter(sep=" ")
        self._scale = 1.0
        self._decimals = 0
        self._suffix = ""
        self._log = False
        self._written: str | None = None
        self._base = ""

    def set_axis(self, axis: Any) -> None:
        super().set_axis(axis)
        self._eng.set_axis(axis)

    def _label(self) -> Any:
        return getattr(self.axis, "label", None)  # colorbar axes have none

    def _relabel(self, prefix: str) -> bool:
        """Write ``prefix`` into the label's unit; False without a unit."""
        label = self._label()
        if label is None:
            return False
        text = label.get_text()
        if text != self._written:  # set (again) by the plot or the user
            self._base = text
        match = _UNIT.search(self._base)
        if match is None:
            return False
        new = (
            f"{self._base[: match.start()]}[{prefix}{match.group(1)}]"
            f"{self._base[match.end() :]}"
        )
        if new != text:
            label.set_text(new)
        self._written = new
        return True

    def set_locs(self, locs: Any) -> None:
        get_scale = getattr(self.axis, "get_scale", None)
        self._log = get_scale is not None and get_scale() == "log"
        if self._log:
            self._eng.set_locs(locs)
            self._relabel("")
            return
        values = np.asarray(locs, dtype=float)
        if self.axis is not None:
            lo, hi = sorted(self.axis.get_view_interval())
            values = values[(values >= lo - 1e-9 * abs(hi - lo)) & (values <= hi)]
        finite = values[np.isfinite(values) & (values != 0)]
        exponent = 0
        if finite.size:
            exponent = 3 * int(np.floor(np.log10(np.max(np.abs(finite))) / 3))
            exponent = min(max(exponent, -24), 18)
        self._scale = 10.0**exponent
        prefix = _SI_PREFIXES[exponent]
        # Decimals: as many as the tick step needs in the chosen unit.
        self._decimals = 0
        if values.size > 1:
            step = float(np.min(np.diff(np.sort(values)))) / self._scale
            while self._decimals < 6 and abs(
                step * 10**self._decimals - round(step * 10**self._decimals)
            ) > 1e-6 * max(1.0, step * 10**self._decimals):
                self._decimals += 1
        self._suffix = "" if self._relabel(prefix) else (f" {prefix}" if prefix else "")

    def __call__(self, x: float, pos: int | None = None) -> str:
        if self._log:
            return self._eng(x, pos)
        v = x / self._scale
        text = f"{v:.{self._decimals}f}"
        if float(text) == 0.0:
            text = "0"
        return text.replace("-", "\N{MINUS SIGN}") + self._suffix


def _set_eng_formatter(ax, which: str) -> None:
    """Show an axis in SI-prefixed units: the prefix in the axis label's
    ``[unit]``, plain numbers on the ticks (see :class:`_SIAxisFormatter`)."""
    if which in ("x", "both"):
        ax.xaxis.set_major_formatter(_SIAxisFormatter())
    if which in ("y", "both"):
        ax.yaxis.set_major_formatter(_SIAxisFormatter())


def _as_channels(arr) -> np.ndarray:
    """Bring to CPU and promote ``(N,)`` -> ``(1, N)``."""
    arr = np.atleast_2d(np.asarray(to_device(arr, "cpu")))
    return arr


def _host(a: Any) -> np.ndarray:
    """``a`` as a host NumPy array (one transfer for device input)."""
    return np.asarray(to_device(a, "cpu"))


def _require(result: Any, field: str, name: str, what: str) -> Any:
    """``result.<field>``; a result without it raises, naming what is needed."""
    value = getattr(result, field, None)
    if value is None:
        raise ValueError(
            f"{name} needs {what} (got {type(result).__name__} without {field})."
        )
    return value
