"""Display and estimation smoothers.

:func:`moving_average`, :func:`savgol_smooth` and :func:`smooth_density_2d`
are not signal-chain filters: they make no claim of causality or a
meaningful frequency response.  They make a plotted curve, a 2-D density
image or a peak or width search less noisy for a reader or a downstream
estimator.  A routine that removes or passes a frequency band as part of the
signal chain (channel filtering, pulse shaping, phase-noise separation)
belongs in :mod:`commkit.filtering`.
"""

from .backend import ArrayType, dispatch
from .logger import logger

__all__ = ["moving_average", "savgol_smooth", "smooth_density_2d"]

# -----------------------------------------------------------------------------
# SMOOTHERS (array-only - operate on derived/plot-only quantities, never on
# raw Signal samples, so none of these are Signal-aware)
# -----------------------------------------------------------------------------
# moving_average: Boxcar moving average (edge-aware "same", or shrinking "valid")
# savgol_smooth:  Savitzky-Golay polynomial local-regression smoother
# smooth_density_2d: Gaussian blur for 2-D histogram/density plots


def moving_average(
    x: ArrayType, *, window: int, mode: str = "same", axis: int = -1
) -> ArrayType:
    """Moving average (boxcar) along one axis.

    Parameters
    ----------
    x : array_like
        Input array.
    window : int
        Averaging window length, in samples.
    mode : {"same", "valid"}, default "same"
        ``"same"`` - edge-aware boxcar (``nearest``-edge padding), output the
        same length as ``x``. Good for smoothing a signal you still want to
        index/plot against the original axis (e.g. a spectrum before a peak
        search).
        ``"valid"`` - shrinks by ``window - 1`` (``numpy.convolve``-style),
        for callers that realign the output axis themselves (e.g. a
        convergence-curve plot that re-centers the x-axis on the window).
    axis : int, default -1
        Axis along which to average.

    Returns
    -------
    ndarray
        Smoothed array. Same shape as ``x`` for ``mode="same"``; shrunk by
        ``window - 1`` along ``axis`` for ``mode="valid"``.
    """
    x, xp, sp = dispatch(x)
    window = max(1, int(window))
    if window <= 1:
        return x

    if mode == "same":
        return sp.ndimage.uniform_filter1d(x, window, axis=axis, mode="nearest")
    elif mode == "valid":
        axis = axis % x.ndim
        # The kernel in the data's precision (a float64 one promotes it).
        real = xp.empty(0, dtype=x.dtype).real.dtype
        kernel = xp.full(window, 1.0 / window, dtype=real)
        if x.ndim == 1:
            return xp.convolve(x, kernel, mode="valid")
        return xp.apply_along_axis(
            lambda row: xp.convolve(row, kernel, mode="valid"), axis, x
        )
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'same' or 'valid'.")


def savgol_smooth(
    x: ArrayType, *, window: int, polyorder: int, axis: int = -1
) -> ArrayType:
    """Smooth with a Savitzky-Golay filter (``scipy.signal.savgol_filter``).

    A least-squares polynomial fit in a sliding window, parameterized by the
    window and the polynomial order rather than a cutoff frequency: for a
    diagnostic detrend or a curve to read, not for the signal chain.

    Parameters
    ----------
    x : array_like
        Input array.
    window : int
        Window length in samples, greater than ``polyorder``; an odd length
        centers the fit on each sample.
    polyorder : int
        Order of the polynomial fit within each window.
    axis : int, default -1
        Axis along which to smooth.

    Returns
    -------
    ndarray
        Smoothed array, same shape as ``x``.
    """
    _, _, sp = dispatch(x)
    logger.debug(
        "Savitzky-Golay smoothing: window=%s, polyorder=%s.", window, polyorder
    )
    return sp.signal.savgol_filter(x, window, polyorder, axis=axis)


def smooth_density_2d(hist: ArrayType, *, sigma: float = 1.0) -> ArrayType:
    """Blur a 2-D histogram with a Gaussian, for a smoother density image.

    Used by the density plots (constellation, eye diagram); purely cosmetic,
    with no meaning in the signal domain.

    Parameters
    ----------
    hist : array_like
        2-D histogram/density array.
    sigma : float, default 1.0
        Gaussian standard deviation, in bins.

    Returns
    -------
    ndarray
        Blurred array, same shape as ``hist``.
    """
    _, _, sp = dispatch(hist)
    return sp.ndimage.gaussian_filter(hist, sigma=sigma)
