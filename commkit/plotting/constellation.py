"""Constellation diagram plots."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from ..backend import dispatch, to_device
from ..core._signal_adapter import adapt_signal
from ..logger import logger
from ..math import rms
from ..smoothing import smooth_density_2d
from .theme import (
    _create_subplot_grid,
    _grid_figsize,
    _square_figsize,
)

__all__ = ["plot_constellation", "plot_ideal_constellation"]


def plot_ideal_constellation(
    constellation: Any,
    *,
    ax: Any | None = None,
    title: str | None = None,
    size: float | None = None,
    show: bool = False,
) -> tuple[Any, Any] | None:
    """
    Plots the points of a constellation with their bit labels.

    Includes concentric rings and center axes for reference.  A shaped
    constellation (one with a ``pmf``) is drawn probability-weighted: each
    marker's colour encodes the symbol probability, and the bit labels are
    left out to keep high orders readable.

    Parameters
    ----------
    constellation : Constellation
        The constellation, e.g. ``Constellation.qam(16)`` or
        ``Constellation.qam(64).shaped(nu=0.05)``.
    ax : matplotlib.axes.Axes, optional
        Target axis.
    title : str, optional
        Plot title; defaults to the constellation's description.
    size : float, optional
        Figure size (square), in inches. Defaults to the theme's square
        panel size (see ``theme._square_figsize``).
    show : bool, default False
        If True, calls `plt.show()`.

    Returns
    -------
    (fig, ax) or None
    """
    from ..mapping import Constellation

    if not isinstance(constellation, Constellation):
        raise TypeError(
            "plot_ideal_constellation(): constellation must be a Constellation, "
            f"got {type(constellation).__name__}; use e.g. Constellation.qam(16)."
        )
    logger.debug("Plotting ideal constellation %r.", constellation)

    if ax is None:
        figsize = (size, size) if size is not None else _square_figsize()
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    const = np.asarray(constellation.points).astype(np.complex128)
    real = const.real
    imag = const.imag
    pmf = constellation.pmf

    if pmf is not None:
        sc = ax.scatter(
            real,
            imag,
            s=100,
            c=np.asarray(pmf, dtype=np.float64),
            cmap="YlOrRd",
            edgecolors="black",
            linewidths=0.5,
            zorder=10,
        )
        plt.colorbar(sc, ax=ax, label="P(sₘ)")
    else:
        ax.scatter(real, imag, s=100, zorder=10)
        for point, bits in zip(const, constellation.bit_labels, strict=True):
            label = "".join(str(int(b)) for b in bits)
            ax.annotate(
                label,
                (point.real, point.imag),
                xytext=(5, 5),
                textcoords="offset points",
            )

    if title is None:
        title = f"Constellation: {constellation!r}"
    ax.set_title(title)
    ax.set_xlabel("In-Phase (I)")
    ax.set_ylabel("Quadrature (Q)")

    # Center lines
    ax.axhline(0, color="white", alpha=0.4, zorder=0)
    ax.axvline(0, color="white", alpha=0.4, zorder=0)

    # Limits and Aspect
    max_range = np.max(np.abs(const))
    limit = max_range * 1.1 if max_range > 0 else 1
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")

    ax.grid(False)

    # Concentric rings at the point magnitudes (the origin excluded).
    radii = np.unique(np.round(np.abs(const), 6))
    radii = radii[radii > 1e-6]
    for r in radii:
        circle = plt.Circle(
            (0, 0),
            r,
            fill=False,
            color="gray",
            linestyle="-",
            alpha=0.4,
            zorder=-5,
        )
        ax.add_artist(circle)

    if show:
        plt.show()
        return None
    return fig, ax


def plot_constellation(
    samples: Any,
    *,
    bins: int = 100,
    cmap: str = "inferno",
    ax: Any | None = None,
    overlay_ideal: bool = False,
    overlay_reference: bool = False,
    constellation: Any | None = None,
    title: str | None = "Constellation",
    vmin: float | None = None,
    vmax: float | None = None,
    show: bool = False,
    **kwargs: Any,
) -> tuple[Any, Any] | None:
    """
    Plots a constellation density diagram from received samples.

    Uses high-definition 2D histograms with Gaussian smoothing to
    visualize noisy or impaired signals. This is significantly more
    informative than scatter plots for large sample sets.

    Parameters
    ----------
    samples : array_like or Signal
        Received complex samples. Shape: ``(N,)`` or ``(C, N)``.
    bins : int, default 100
        Density resolution (bins per axis).
    cmap : str, default "inferno"
        Colormap for the density field.
    ax : matplotlib.axes.Axes, optional
        Target axis.
    overlay_ideal : bool, default False
        Overlay the constellation's points, scaled to the samples' RMS.
    overlay_reference : bool, default False
        Overlay a Signal's ``reference.symbols`` as given.
    constellation : Constellation, optional
        Constellation for ``overlay_ideal``.  Defaults to the Signal's.
    title : str, optional
        Plot title.
    vmin, vmax : float, optional
        Color scaling limits. Defaults to auto-range [0, 1].
    show : bool, default False
        If True, calls `plt.show()`.
    **kwargs : Any
        Additional arguments passed to `ax.imshow`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes or ndarray
        The axis or array of axes used.
    """
    signal_adapter = adapt_signal(samples, function_name="plot_constellation()")
    sig = signal_adapter.signal
    constellation = signal_adapter.resolve_choice("constellation", constellation)
    if overlay_ideal and constellation is None:
        raise ValueError(
            "plot_constellation(): overlay_ideal needs a constellation: pass "
            "constellation= or a Signal that has one."
        )
    result = _plot_constellation_array(
        signal_adapter.array,
        bins=bins,
        cmap=cmap,
        ax=ax,
        overlay_ideal=overlay_ideal,
        constellation=constellation,
        title=title,
        vmin=vmin,
        vmax=vmax,
        show=False,
        **kwargs,
    )

    if overlay_reference:
        if sig is None or sig.reference is None:
            raise ValueError(
                "plot_constellation(): overlay_reference needs a Signal with a "
                "reference."
            )
        assert result is not None
        _, axes = result
        # Each distinct symbol once: a scatter of every reference symbol
        # draws the same few points thousands of times.
        ref, xp_ref, _ = dispatch(sig.reference.symbols)
        src: Any = (
            [to_device(xp_ref.unique(row), "cpu") for row in ref]
            if ref.ndim > 1
            else to_device(xp_ref.unique(ref), "cpu")
        )

        def _scatter_source(axis, symbols):
            axis.scatter(
                symbols.real,
                symbols.imag,
                c="lime",
                edgecolors="dimgray",
                linewidths=1.5,
                s=30,
                zorder=10,
                marker="o",
            )

        if isinstance(src, list):
            ax_list = list(np.asarray(axes).flat)
            for ch in range(min(len(src), len(ax_list))):
                _scatter_source(ax_list[ch], src[ch])
        else:
            _scatter_source(axes, src)

    if show:
        plt.show()
        return None
    return result


def _plot_constellation_array(
    samples: Any,
    bins: int = 100,
    cmap: str = "inferno",
    ax: Any | None = None,
    overlay_ideal: bool = False,
    constellation: Any | None = None,
    title: str | None = "Constellation",
    vmin: float | None = None,
    vmax: float | None = None,
    show: bool = False,
    **kwargs: Any,
) -> tuple[Any, Any] | None:
    """Render array data; the public boundary handles reference overlays."""
    logger.debug("Generating constellation density plot.")

    samples, xp, _ = dispatch(samples)

    # Handle Multichannel (e.g. Dual-Pol)
    # Convention: (Channels, Time)
    if samples.ndim > 1:
        num_channels = samples.shape[0]

        if ax is None:
            nrows, ncols = _create_subplot_grid(num_channels)
            fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=_grid_figsize(nrows, ncols, panel=_square_figsize()),
                squeeze=False,
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

            _plot_constellation_array(
                channel_samples,
                bins=bins,
                cmap=cmap,
                ax=target_ax,
                overlay_ideal=overlay_ideal,
                constellation=constellation,
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

    if ax is None:
        fig, ax = plt.subplots(figsize=_square_figsize())
    else:
        fig = ax.figure

    # Ensure samples are complex
    if not xp.iscomplexobj(samples):
        logger.warning("Constellation plot expects complex samples. Converting.")
        samples = samples.astype(xp.complex64)

    # Bin on the device and transfer only the (bins, bins) counts.  The view
    # spans 2x the RMS (robust to noise outliers).
    i_data = samples.real.ravel()
    q_data = samples.imag.ravel()
    signal_rms = float(rms(samples))
    limit = signal_rms * 2.0
    if limit == 0:
        limit = 1.0  # Default view range for zero signal
    h = _density(i_data, q_data, bins, limit, xp)

    # Transpose for imshow (rows=y, cols=x)
    h = h.T

    # Apply Gaussian smoothing for nicer visuals
    h = smooth_density_2d(h, sigma=1.0)

    # Normalize histogram to [0, 1] for consistent colormap scaling
    h_max = np.max(h)
    if h_max > 0:
        h = h / h_max

    # Plot using imshow
    imshow_kwargs: dict[str, Any] = {
        "origin": "lower",
        "extent": [-limit, limit, -limit, limit],
        "aspect": "equal",
        "cmap": cmap,
        "interpolation": "bilinear",
    }
    if vmin is not None:
        imshow_kwargs["vmin"] = vmin
    if vmax is not None:
        imshow_kwargs["vmax"] = vmax
    imshow_kwargs.update(kwargs)

    ax.imshow(h, **imshow_kwargs)

    # Overlay the constellation, scaled from its unit average power (under
    # its pmf) to the samples' RMS.
    if overlay_ideal and constellation is not None:
        const = np.asarray(constellation.points).astype(np.complex128)
        const_rms = float(np.sqrt(constellation.power()))
        if const_rms > 0:
            const = const * (signal_rms / const_rms)
        ax.scatter(
            const.real,
            const.imag,
            c="lime",
            edgecolors="dimgray",
            linewidths=1.5,
            s=30,
            zorder=10,
            marker="o",
        )

    # Add center lines
    ax.axhline(0, color="white", alpha=0.4, zorder=0)
    ax.axvline(0, color="white", alpha=0.4, zorder=0)

    ax.set_xlabel("In-Phase (I)")
    ax.set_ylabel("Quadrature (Q)")
    if title is not None:
        ax.set_title(title)

    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.grid(False)

    if show:
        plt.show()
        return None
    return fig, ax


def _density(i_data: Any, q_data: Any, bins: int, limit: float, xp: Any) -> np.ndarray:
    """2-D histogram over ``[-limit, limit]^2`` as ``(bins, bins)`` host counts.

    ``bincount`` of the flattened bin index on the input's device, with
    ``numpy.histogram2d``'s edges: equal bins, the last one closed.
    """
    scale = bins / (2.0 * limit)
    ii = xp.floor((i_data + limit) * scale).astype(xp.int64)
    qq = xp.floor((q_data + limit) * scale).astype(xp.int64)
    # The upper edge belongs to the last bin, as in numpy.histogram2d.
    ii = xp.where(i_data == limit, bins - 1, ii)
    qq = xp.where(q_data == limit, bins - 1, qq)
    inside = (ii >= 0) & (ii < bins) & (qq >= 0) & (qq < bins)
    flat = (ii * bins + qq)[inside]
    counts = xp.bincount(flat, minlength=bins * bins)
    return np.asarray(to_device(counts, "cpu"), dtype=np.float64).reshape(bins, bins)


# -----------------------------------------------------------------------------
# EQUALIZER DIAGNOSTICS
# -----------------------------------------------------------------------------
