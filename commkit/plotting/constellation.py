"""Constellation diagram plots."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .. import helpers
from ..backend import dispatch, to_device
from ..core._signal_adapter import adapt_signal
from ..logger import logger
from ..smoothing import smooth_density_2d
from .theme import (
    _create_subplot_grid,
    _grid_figsize,
    _square_figsize,
)


def plot_ideal_constellation(
    modulation: str,
    order: int,
    pmf: Any | None = None,
    nu: float | None = None,
    ax: Any | None = None,
    title: str | None = None,
    size: float | None = None,
    show: bool = False,
    unipolar: bool | None = None,
) -> tuple[Any, Any] | None:
    """
    Plots the ideal constellation diagram for a modulation format.

    Draws theoretical symbol points with their associated Gray-coded bit
    sequences. Includes concentric rings and center axes for reference.

    For PS-QAM, pass either ``pmf`` or ``nu`` (not both) to activate
    probability-weighted rendering: each marker's **area** and **colour**
    encode the symbol probability under the Maxwell-Boltzmann distribution.
    Inner (more probable) points appear larger and warmer.  Bit-label
    annotations are suppressed to keep the plot readable at high orders.

    Parameters
    ----------
    modulation : {"psk", "qam", "ask", "pam"}
        Modulation scheme identifier.
    order : int
        Modulation order (e.g., 4, 16, 64).
    pmf : array-like of float, optional
        Symbol PMF of shape ``(M,)`` for PS-QAM (from ``maxwell_boltzmann``).
        Mutually exclusive with ``nu``.
    nu : float, optional
        Maxwell-Boltzmann shaping parameter nu >= 0 for QAM.  The PMF is
        computed automatically via ``maxwell_boltzmann``.
        nu = 0 gives a uniform distribution (equal-sized markers).
        Mutually exclusive with ``pmf``.
    ax : matplotlib.axes.Axes, optional
        Target axis.
    title : str, optional
        Plot title.
    size : float, optional
        Figure size (square), in inches. Defaults to the theme's square
        panel size (see ``theme._square_figsize``).
    show : bool, default False
        If True, calls `plt.show()`.
    unipolar : bool, default False
        If True, use unipolar plot_constellation (ASK/PAM).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The plotting axis.

    Raises
    ------
    ValueError
        If both ``pmf`` and ``nu`` are provided.
    """
    if pmf is not None and nu is not None:
        raise ValueError("Provide at most one of `pmf` or `nu`, not both.")

    logger.debug("Generating ideal constellation for %s (%s-level).", modulation, order)
    from ..mapping import gray_constellation, maxwell_boltzmann

    if nu is not None:
        pmf = maxwell_boltzmann(order, nu)

    if ax is None:
        figsize = (size, size) if size is not None else _square_figsize()
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    try:
        # Generate constellation on backend (returns NumPy)
        const = gray_constellation(modulation, order, unipolar=unipolar)
    except ValueError as e:
        logger.error("Error generating constellation: %s", e)
        return None

    # Move to cpu for plotting (already NumPy but good practice)
    const = to_device(const, "cpu")

    real = const.real
    imag = const.imag

    if pmf is not None:
        # PS-QAM mode
        pmf_arr = np.asarray(pmf, dtype=np.float64)
        sc = ax.scatter(
            real,
            imag,
            s=100,
            c=pmf_arr,
            cmap="YlOrRd",
            edgecolors="black",
            linewidths=0.5,
            zorder=10,
        )
        plt.colorbar(sc, ax=ax, label="P(sₘ)")
    else:
        # Uniform mode
        ax.scatter(real, imag, s=100, zorder=10)
        n_bits = int(np.log2(order))
        for i, point in enumerate(const):
            x, y = point.real, point.imag
            label = f"{i:0{n_bits}b} ({i})"
            ax.annotate(
                label,
                (x, y),
                xytext=(5, 5),
                textcoords="offset points",
            )

    # Titles and Labels
    if title is None:
        prefix = "PS-" if pmf is not None else ""
        title = f"Constellation: {prefix}{modulation.upper()} {order}"
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

    # Draw concentric circles (rings) at point magnitudes
    # Find unique radii from the constellation points
    radii = np.unique(np.round(np.abs(const), 6))

    # Filter out zero radius (origin)
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
    bins: int = 100,
    cmap: str = "inferno",
    ax: Any | None = None,
    overlay_ideal: bool = False,
    overlay_source: bool = False,
    modulation: str | None = None,
    order: int | None = None,
    unipolar: bool | None = None,
    pmf: Any | None = None,
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
        Received complex samples. Shape: (..., N_symbols).
    bins : int, default 100
        Density resolution (bins per axis).
    cmap : str, default "inferno"
        Colormap for the density field.
    ax : matplotlib.axes.Axes, optional
        Target axis.
    overlay_ideal : bool, default False
        If True, overlays theoretical points and scales them to signal power.
    modulation : str, optional
        Required parameter if `overlay_ideal` is enabled.
    order : int, optional
        Required parameter if `overlay_ideal` is enabled.
    unipolar : bool, optional
        Required parameter if `overlay_ideal` is enabled.
    title : str, optional
        Plot title.
    vmin, vmax : float, optional
        Color scaling limits. Defaults to auto-range [0, 1].
    show : bool, default False
        If True, calls `plt.show()`.
    **kwargs : Any
        Additional theoretical arguments passed to `ax.imshow`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes or ndarray
        The axis or array of axes used.
    """
    signal_adapter = adapt_signal(samples, function_name="plot_constellation()")
    sig = signal_adapter.signal
    result = _plot_constellation_array(
        signal_adapter.array,
        bins=bins,
        cmap=cmap,
        ax=ax,
        overlay_ideal=overlay_ideal,
        modulation=signal_adapter.resolve_optional("mod_scheme", modulation),
        order=signal_adapter.resolve_optional("mod_order", order),
        unipolar=signal_adapter.resolve_optional("mod_unipolar", unipolar),
        pmf=signal_adapter.resolve_optional("ps_pmf", pmf),
        title=title,
        vmin=vmin,
        vmax=vmax,
        show=False,
        **kwargs,
    )

    if (
        sig is not None
        and overlay_source
        and sig.source_symbols is not None
        and result is not None
    ):
        _, axes = result
        src = to_device(sig.source_symbols, "cpu")

        # PS-QAM: source_symbols are on the {s_m} grid (avg power E_PS < 1)
        # but received samples normalise to unit power ({s_m/sqrt(E_PS)}).
        # Scale source symbols to match the received symbol scale.
        if (
            sig.ps_pmf is not None
            and sig.mod_scheme is not None
            and sig.mod_order is not None
        ):
            from ..mapping import gray_constellation as _gc_src

            _const_src = _gc_src(sig.mod_scheme, sig.mod_order)
            _pmf_src = np.asarray(sig.ps_pmf, dtype=np.float64)
            _e_ps = float(np.dot(_pmf_src, np.abs(_const_src) ** 2))
            if 0 < _e_ps < 1.0 - 1e-6:
                src = src / np.sqrt(_e_ps)

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

        if src.ndim > 1:
            ax_list = list(np.asarray(axes).flat)
            for ch in range(min(src.shape[0], len(ax_list))):
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
    modulation: str | None = None,
    order: int | None = None,
    unipolar: bool | None = None,
    pmf: Any | None = None,
    title: str | None = "Constellation",
    vmin: float | None = None,
    vmax: float | None = None,
    show: bool = False,
    **kwargs: Any,
) -> tuple[Any, Any] | None:
    """Render array data; the public boundary handles Signal source overlays."""
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
                modulation=modulation,
                order=order,
                unipolar=unipolar,
                pmf=pmf,
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

    # Extract I and Q
    i_data = samples.real.flatten()
    q_data = samples.imag.flatten()

    # Move to CPU for plotting
    i_data = to_device(i_data, "cpu")
    q_data = to_device(q_data, "cpu")

    # Compute 2D histogram
    # Determine range based on RMS (robust to noise outliers)
    # Using np.sqrt(np.mean(|I|² + |Q|²)) is equivalent to rms(complex_signal)
    signal_rms = float(helpers.rms(i_data + 1j * q_data))
    # Use ~3x RMS as limit (covers most constellation points + noise spread)
    limit = signal_rms * 2.0
    if limit == 0:
        limit = 1.0  # Default view range for zero signal

    h, xedges, yedges = np.histogram2d(
        i_data, q_data, bins=bins, range=[[-limit, limit], [-limit, limit]]
    )

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

    # Overlay ideal constellation if requested
    if overlay_ideal:
        if modulation is None or order is None:
            logger.warning(
                "Modulation and order must be provided to overlay ideal constellation."
            )
        else:
            from ..mapping import constellation_power, gray_constellation

            try:
                const = gray_constellation(modulation, order, unipolar=unipolar)
                const = to_device(const, "cpu")

                # Scale constellation to match signal amplitude.
                # For PS-QAM: use pmf-weighted RMS so ideal points land at
                # {s_m / sqrt(E_PS)}, matching where the received clusters
                # sit after shape_pulse normalises to E_s = 1.
                if pmf is not None:
                    e_ps = constellation_power(const, pmf)
                    const_rms = float(np.sqrt(e_ps)) if e_ps > 0 else helpers.rms(const)
                else:
                    const_rms = helpers.rms(const)
                if const_rms > 0:
                    scale_factor = signal_rms / const_rms
                    const = const * scale_factor

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
            except ValueError as e:
                logger.warning("Could not overlay ideal constellation: %s", e)

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


# -----------------------------------------------------------------------------
# EQUALIZER DIAGNOSTICS
# -----------------------------------------------------------------------------
