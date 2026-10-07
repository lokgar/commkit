"""Power, normalization and decibel conversions on NumPy/CuPy arrays."""

from typing import Any

import numpy as np

from .backend import ArrayType, dispatch
from .logger import logger

__all__ = ["db_to_linear", "linear_to_db", "normalize", "rms"]

# ---------------------------------------------------------------------------
# Power / normalization
# ---------------------------------------------------------------------------


def rms(x: ArrayType, *, axis: int | None = None, keepdims: bool = False) -> ArrayType:
    """
    Computes the Root-Mean-Square (RMS) value of an array.

    RMS is defined as: sqrt(E[|x|^2]).

    Parameters
    ----------
    x : array_like
        Input array.
    axis : int, optional
        Axis along which to compute the RMS. If None, computes global RMS.
    keepdims : bool, default False
        If True, the reduced axes are left in the result as dimensions with size one.

    Returns
    -------
    array_like or float
        The RMS value of the input.
    """
    x, xp, _ = dispatch(x)
    # RMS = ||x||₂ / √N  ->  linalg.norm routes through BLAS (DZNRM2/SNRM2),
    # eliminating the abs(x)**2 and mean() intermediate allocations.
    n = x.size if axis is None else x.shape[axis]
    # xp.sqrt(Python int) returns float64; cast n to x's real dtype so that
    # float32 norms are not silently promoted to float64.
    return xp.linalg.norm(x, axis=axis, keepdims=keepdims) / xp.sqrt(
        xp.asarray(n, dtype=x.real.dtype)
    )


def normalize(
    x: ArrayType,
    *,
    mode: str = "unity_gain",
    axis: int | None = None,
    sps: float = 1,
) -> ArrayType:
    """
    Normalizes an array according to the specified strategy.

    Parameters
    ----------
    x : array_like
        Input signal or filter taps.
    mode : {"unity_gain", "unit_energy", "peak", "average_power", "symbol_power", "dac_peak"}, default "unity_gain"
        Normalization strategy:
        - "unity_gain": Sum of elements is 1.0 (DC gain normalization).
          Preserves signal levels (e.g., 5V -> 5V). Used for general filters.
        - "unit_energy": L2-norm is 1.0 (sum(|x|^2) = 1).
          Preserves total energy/noise power. Used for pulse shaping and matched filters.
        - "peak": Peak complex envelope is 1.0 (max_n |x[n]| = 1).
          For complex signals this normalizes by the maximum instantaneous magnitude,
          so |x[n]| <= 1 for all n. This bound is invariant under any
          unit-magnitude operation (frequency shifts, phase rotations, equalization),
          making it the correct choice for DSP chains. For real signals the behavior
          is identical: max_n |x[n]| = 1.
        - "average_power": Mean sample power is 1.0 (E[|x|^2] = 1 per sample).
          Normalizes the composite complex signal power at the sample level.
          Used for symbol constellations at 1 sps and for display/plotting.
          **Not suitable for oversampled waveforms**: for a Nyquist pulse with
          unit-energy taps at ``sps`` samples/symbol the natural average sample
          power is ``Es/sps``, so ``"average_power"`` would inflate all samples
          by ``√sps`` and break Es/N0 calibration.
        - "symbol_power": Unit symbol energy regardless of oversampling factor.
          Norm factor is ``rms(x) * √sps``, so the output satisfies
          ``E[|x|²] * sps = 1`` (i.e. average sample power = 1/sps).
          This is the correct mode for pulse-shaped waveforms: all pulse types
          (zero-stuffed, rect, RRC, Gaussian, ...) end up at the same power level
          and ``apply_awgn`` can use ``Es = signal_power * sps = 1`` directly.
          Requires ``sps`` parameter. At ``sps=1`` it is identical to
          ``"average_power"``.
        - "dac_peak": Per-channel ``max(peak_|Re|, peak_|Im|)`` is 1.0.
          Brings the dominant I/Q component to 1.0 while preserving the I/Q
          ratio - unlike ``"peak"`` (complex-envelope ``max(|x[n]|)``), which
          leaves components at ``<= 1/sqrt(2)`` for square QAM/PSK whose
          envelope peak sits at 45°.  Maximises DAC range utilisation for a
          signal section independent of modulation format or constellation
          phase geometry (e.g. a frame's preamble and body, normalised
          separately so each uses the full DAC range).
    axis : int, optional
        The axis along which to compute the normalization factor.
        If `None`, normalizes the entire array globally.
    sps : float, default 1
        Samples per symbol. Only used by the ``"symbol_power"`` mode.

    Returns
    -------
    array_like
        The normalized array.
    """
    logger.debug("Normalizing array (mode: %s, axis=%s, sps=%s).", mode, axis, sps)
    x, xp, _ = dispatch(x)

    # keepdims for proper broadcasting when axis is specified
    keepdims = axis is not None

    if mode == "unity_gain":
        # DC gain = 1: H(0) = sum(h) = 1
        # Use case: filter taps where you want unity passband gain
        norm_factor = xp.sum(x, axis=axis, keepdims=keepdims)

    elif mode == "unit_energy":
        # L2 norm = 1: ||x||₂ = 1
        # Use case: matched filter taps (preserves SNR after correlation)
        # linalg.norm routes through BLAS (DNRM2/DZNRM2 on CPU, cuBLAS on GPU):
        # numerically superior (compensated summation) and avoids intermediate allocations.
        norm_factor = xp.linalg.norm(x, axis=axis, keepdims=keepdims)

    elif mode == "peak":
        # Complex envelope peak: max(|x[n]|) = 1.
        # For complex signals this is the instantaneous magnitude, not the
        # per-component max. The bound is invariant under frequency shifts and
        # phase rotations, unlike per-component (I/Q) normalization which can
        # allow |x[n]| up to sqrt(2) and therefore violate bounds after rotation.
        norm_factor = xp.max(xp.abs(x), axis=axis, keepdims=keepdims)

    elif mode == "average_power":
        # RMS = 1: sqrt(mean(|x|²)) = 1, so mean(|x|²) = 1
        # Use case: 1-sps symbol sequences and constellation normalization.
        norm_factor = rms(x, axis=axis, keepdims=keepdims)

    elif mode == "symbol_power":
        # Symbol-power norm: rms(x) * √sps = 1  ->  mean(|x|²) * sps = 1
        # Equivalent to average_power at 1 sps; at higher sps it accounts for
        # the 1/sps dilution produced by Nyquist pulse shaping with unit-energy
        # taps, leaving Es = 1 per symbol for all pulse shapes.
        # This is the same correction used in the equalizer's _normalize_inputs:
        #   sym_rms = global_rms * √sps
        norm_factor = rms(x, axis=axis, keepdims=keepdims) * xp.asarray(
            sps**0.5, dtype=x.real.dtype
        )

    elif mode == "dac_peak":
        # Per-channel max(peak_|Re|, peak_|Im|) = 1: brings the dominant I/Q
        # component to 1.0, preserving the I/Q ratio.
        norm_factor = xp.maximum(
            xp.max(xp.abs(x.real), axis=axis, keepdims=keepdims),
            xp.max(xp.abs(x.imag), axis=axis, keepdims=keepdims),
        )

    else:
        raise ValueError(f"Unknown normalization mode: {mode}")

    # Handle division by zero safely for both NumPy and CuPy.
    # Avoid control flow based on data values to prevent host-device synchronization.
    # Use ones_like instead of the literal 1.0 (float64) to preserve float32 dtype.
    safe_norm = xp.where(norm_factor == 0, xp.ones_like(norm_factor), norm_factor)
    result = x / safe_norm

    # If norm_factor is 0, the input was all zeros -> output should also be zeros
    return xp.where(norm_factor == 0, xp.zeros(x.shape, dtype=x.dtype), result)


# ---------------------------------------------------------------------------
# dB <-> linear ratio conversion
# ---------------------------------------------------------------------------


def db_to_linear(db: ArrayType | float, *, power: bool = True) -> ArrayType | float:
    """Convert decibels to a linear ratio.

    ``power=True`` (default) uses the power/energy convention
    (``10**(db/10)``) - e.g. Es/N0, SNR.  ``power=False`` uses the
    amplitude/field convention (``10**(db/20)``) - e.g. an I/Q gain
    imbalance or a pilot-tone gain applied directly to complex amplitude
    samples.  Picking the wrong convention silently mis-scales the result
    by a factor of 2 in the exponent, so verify which quantity a given
    ``_db`` parameter actually represents before choosing.

    Parameters
    ----------
    db : array_like or float
        Value(s) in decibels.
    power : bool, default True
        Selects the 10x (power) or 20x (amplitude) convention.

    Returns
    -------
    array_like or float
        The linear ratio, same type as ``db``.
    """
    exponent = 10.0 if power else 20.0
    return 10.0 ** (db / exponent)


def linear_to_db(x: ArrayType, *, power: bool = True) -> ArrayType:
    """Convert a linear ratio to decibels (inverse of :func:`db_to_linear`).

    ``power=True`` (default) uses the power/energy convention
    (``10*log10(x)``) - e.g. an SNR ratio.  ``power=False`` uses the
    amplitude/field convention (``20*log10(x)``) - e.g. an EVM ratio
    (RMS error amplitude / RMS reference amplitude).  ``x <= 0`` maps to
    ``-inf`` (or ``nan`` for ``x < 0``) without raising a RuntimeWarning -
    callers that need a different zero/negative policy should guard before
    calling.

    Parameters
    ----------
    x : array_like
        Linear ratio value(s), any backend (NumPy/CuPy).
    power : bool, default True
        Selects the 10x (power) or 20x (amplitude) convention.

    Returns
    -------
    array_like
        Value(s) in decibels, same shape/backend as ``x``.
    """
    x, xp, _ = dispatch(x)
    factor = 10.0 if power else 20.0
    with np.errstate(divide="ignore", invalid="ignore"):
        return factor * xp.log10(x)


# ---------------------------------------------------------------------------
# Linear trend (least-squares slope), shared by frequency, recovery and analysis
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


def _linear_trend_slope(y: ArrayType, *, x: Any = None, xp: Any = None) -> ArrayType:
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


def _remove_linear_trend(y: ArrayType, *, x: Any = None) -> tuple[ArrayType, ArrayType]:
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
