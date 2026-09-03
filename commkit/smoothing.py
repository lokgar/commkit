"""
Diagnostic and analysis smoothers.

These are **not** signal-processing filters: they carry no claim of
causality, linear-time-invariance, or a meaningful frequency response. They
exist purely to make a plotted curve, a 2-D density image, or a peak/width
search less noisy for a human or a downstream estimator to read - the
"just for the plot / just for a robust estimate" counterpart to the real
DSP filters in :mod:`commkit.filtering`. If a routine's job is to remove or
pass a frequency band as part of the signal chain itself (channel filtering,
pulse shaping, phase-noise separation), it belongs in ``filtering.py``
instead, not here.
"""

from .backend import ArrayType, dispatch
from .logger import logger

# -----------------------------------------------------------------------------
# SMOOTHERS (array-only - operate on derived/plot-only quantities, never on
# raw Signal samples, so none of these are Signal-aware; see CLAUDE.md,
# "Signal-Awareness")
# -----------------------------------------------------------------------------
# moving_average: Boxcar moving average (edge-aware "same", or shrinking "valid")
# savgol_smooth:  Savitzky-Golay polynomial local-regression smoother
# smooth_density_2d: Gaussian blur for 2-D histogram/density plots


def moving_average(
    x: ArrayType, window: int, *, mode: str = "same", axis: int = -1
) -> ArrayType:
    """
    Boxcar moving average.

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
    array_like
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
        kernel = xp.ones(window, dtype=xp.float64) / window
        if x.ndim == 1:
            return xp.convolve(x, kernel, mode="valid")
        return xp.apply_along_axis(
            lambda row: xp.convolve(row, kernel, mode="valid"), axis, x
        )
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'same' or 'valid'.")


def savgol_smooth(
    x: ArrayType, window: int, polyorder: int, *, axis: int = -1
) -> ArrayType:
    """
    Savitzky-Golay smoothing: local polynomial regression over a sliding window.

    A curve-fitting smoother, not a linear-time-invariant filter - useful for
    a diagnostic detrend where a designed low-pass cutoff isn't the natural
    parameterization.

    Parameters
    ----------
    x : array_like
        Input array.
    window : int
        Window length, in samples. Must be odd and > ``polyorder``.
    polyorder : int
        Order of the polynomial fit within each window.
    axis : int, default -1
        Axis along which to smooth.

    Returns
    -------
    array_like
        Smoothed array, same shape as ``x``.
    """
    _, _, sp = dispatch(x)
    logger.debug(
        "Savitzky-Golay smoothing: window=%s, polyorder=%s.", window, polyorder
    )
    return sp.signal.savgol_filter(x, window, polyorder, axis=axis)


def smooth_density_2d(hist: ArrayType, sigma: float = 1.0) -> ArrayType:
    """
    Gaussian-blur a 2-D histogram/density image for nicer visual contours.

    Used by density-style plots (constellation, eye diagram) to turn a raw
    2-D histogram into a smoother heatmap; purely cosmetic, no signal-domain
    meaning.

    Parameters
    ----------
    hist : array_like
        2-D histogram/density array.
    sigma : float, default 1.0
        Gaussian standard deviation, in bins.

    Returns
    -------
    array_like
        Blurred array, same shape as ``hist``.
    """
    _, _, sp = dispatch(hist)
    return sp.ndimage.gaussian_filter(hist, sigma=sigma)
