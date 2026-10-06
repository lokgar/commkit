"""Domain helpers awaiting their owning modules (removed in 4.1).

What is left: PLL loop gains (moving to ``recovery`` in 3.6) and the
least-squares linear trend (``analysis``, 3.9).  Shape helpers live in
``commkit._array``, power and dB conversions in ``commkit.math``,
correlation and peak interpolation in ``commkit.timing``, and the
synchronization sequences in ``commkit._sequences``.
"""

from typing import Any

import numpy as np

from .backend import ArrayType, dispatch

# ---------------------------------------------------------------------------
# CPR / PLL loop gains
# ---------------------------------------------------------------------------


def cpr_pll_gains(bandwidth: float):
    """Convert normalised loop bandwidth to PI gains (mu, beta).

    Uses the standard 2nd-order loop approximation for a critically-damped
    (ζ = 1) PI loop:  μ ≈ 4·B_L,  β ≈ 4·B_L².  (With ``ωₙT = √β = 2B`` and
    ``ζ = μ/(2√β) = 1``.)

    Parameters
    ----------
    bandwidth : float
        Normalised one-sided loop bandwidth as a fraction of the symbol rate,
        e.g. ``1e-3`` for a narrow loop.

    Returns
    -------
    mu, beta : float32
    """
    mu = np.float32(4.0 * bandwidth)
    beta = np.float32(4.0 * bandwidth**2)
    return mu, beta


def resolve_pll_gains(bandwidth: float, mu: float | None, beta: float | None):
    """Resolve decision-directed PLL PI gains from a raw/bandwidth parameterization.

    Shared by the inline equalizer PLL (``lms``/``rls`` with ``cpr_type='pll'``)
    and the standalone ``recover_carrier_phase_pll``, so
    the bandwidth->gain mapping is defined in exactly one place.

    Precedence
    ----------
    * ``mu`` given -> raw PI gains; ``beta`` defaults to ``0.0`` (1st-order loop).
    * ``mu`` is ``None`` -> derive critically-damped (ζ=1) gains ``μ=4B, β=4B²``
      from ``bandwidth`` via ``cpr_pll_gains``.

    ``beta`` without ``mu`` is ambiguous and raises ``ValueError``.

    Returns
    -------
    mu, beta : float
    """
    if mu is not None:
        return float(mu), float(beta if beta is not None else 0.0)
    if beta is not None:  # beta without mu is ambiguous
        raise ValueError("beta requires mu to be set (or use the bandwidth shortcut).")
    return cpr_pll_gains(bandwidth)


# ---------------------------------------------------------------------------
# Linear-trend (least-squares slope) helpers
# ---------------------------------------------------------------------------


def _centered_axis(n: int, x: Any, xp: Any) -> tuple[ArrayType, ArrayType]:
    """Mean-removed abscissa and its sum of squares, as device arrays."""
    if x is None:
        # Analytic form for x = arange(n): Σ(i - (n-1)/2)² = n(n²-1)/12, which
        # avoids a reduction and is exact in float64 for realistic record
        # lengths (n < 2⁵², well inside the mantissa).
        xc = xp.arange(n, dtype=xp.float64) - 0.5 * (n - 1)
        denom = xp.asarray(n * (n * n - 1.0) / 12.0, dtype=xp.float64)
    else:
        x_arr = xp.asarray(x, dtype=xp.float64)
        xc = x_arr - xp.mean(x_arr)
        denom = xp.sum(xc * xc)
    # Guard n == 1 (or a degenerate axis) without a host sync on the value.
    return xc, xp.where(denom > 0.0, denom, xp.ones_like(denom))


def linear_trend_slope(y: ArrayType, *, x: Any = None, xp: Any = None) -> ArrayType:
    r"""
    Per-channel least-squares slope of a phase (or any) record.

    Ordinary least squares on the centred normal equations,

    .. math:: \hat{a} = \frac{\sum_k (x_k - \bar{x})(y_k - \bar{y})}
                             {\sum_k (x_k - \bar{x})^2},

    evaluated for every channel in one vectorized pass and returned **on the
    input backend** - no host synchronization, so the caller decides when (and
    whether) to transfer.

    Parameters
    ----------
    y : array_like
        Record to fit, ``(C, N)`` with the fit axis last (promote SISO with
        :func:`as_2d` first).
    x : array_like, optional
        Abscissa, ``(N,)``.  Defaults to the sample index ``arange(N)``, so the
        slope is then in *units of y per sample*.  Pass a time axis in seconds
        to get a slope per second (e.g. for non-uniform pilot positions).
    xp : module, optional
        Array module; inferred from ``y`` when omitted.

    Returns
    -------
    array_like
        Slope per channel, ``(C,)``, ``float64``, on the input backend.
    """
    if xp is None:
        y, xp, _ = dispatch(y)
    n = y.shape[-1]
    xc, denom = _centered_axis(n, x, xp)
    # Subtracting the per-channel mean is mathematically redundant (xc is
    # already centred) but keeps the products small when the record carries a
    # large constant offset - phase trajectories routinely do.
    yc = y - xp.mean(y, axis=-1, keepdims=True)
    return xp.sum(yc * xc, axis=-1) / denom


def remove_linear_trend(y: ArrayType, *, x: Any = None) -> tuple[ArrayType, ArrayType]:
    r"""
    Removes the per-channel least-squares linear trend, preserving the mean.

    On an unwrapped phase record the linear term *is* the mean frequency
    offset, so this is the canonical "strip the residual FOE, keep the phase
    fluctuation" step shared by the pilot-tone recovery and the DSH
    fine-frequency stage.  Only the slope term is subtracted, so the mean phase
    (and hence any constant offset) survives.

    Parameters
    ----------
    y : array_like
        Record to detrend, ``(C, N)`` with time on the last axis.
    x : array_like, optional
        Abscissa, ``(N,)``; defaults to the sample index (slope per sample).

    Returns
    -------
    detrended : array_like
        ``y`` minus the fitted slope term, ``float64``, on the input backend.
    slope : array_like
        Fitted slope per channel, ``(C,)`` - in units of ``y`` per unit ``x``
        (per sample by default).
    """
    y, xp, _ = dispatch(y)
    n = y.shape[-1]
    xc, denom = _centered_axis(n, x, xp)
    yc = y - xp.mean(y, axis=-1, keepdims=True)
    slope = xp.sum(yc * xc, axis=-1) / denom
    return y - slope[..., None] * xc[None, :], slope
