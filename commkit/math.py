"""Power, normalization and decibel conversions on NumPy/CuPy arrays.

:func:`rms`, :func:`normalize`, :func:`db_to_linear` and
:func:`linear_to_db`.  Also the private helpers shared by the DSP modules:
the linear trend (``_linear_trend_slope``, ``_remove_linear_trend``) and the
phase rotations ``_unit_phasor`` (``exp(j·phase)`` in a chosen precision),
``_phasor`` and ``_derotate`` (``x·exp(-j·phase)`` in the precision of
``x``).
"""

from typing import Any

import numpy as np

from .backend import ArrayType, dispatch
from .logger import logger

__all__ = ["db_to_linear", "linear_to_db", "normalize", "rms"]

# ---------------------------------------------------------------------------
# Power / normalization
# ---------------------------------------------------------------------------


def rms(x: ArrayType, *, axis: int | None = None, keepdims: bool = False) -> ArrayType:
    """Root-mean-square value ``sqrt(E[|x|²])`` of an array.

    Parameters
    ----------
    x : array_like
        Input array.
    axis : int, optional
        Axis to reduce; ``None`` reduces the whole array.
    keepdims : bool, default False
        Keep the reduced axis as a dimension of size one.

    Returns
    -------
    ndarray
        The RMS in the real precision of ``x``, on its device (0-d without
        ``axis``).
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
    """Scale an array to unit gain, energy, peak or power.

    Parameters
    ----------
    x : array_like
        Signal or filter taps.
    mode : {"unity_gain", "unit_energy", "peak", "average_power", "symbol_power", "dac_peak"}, default "unity_gain"
        Quantity set to 1:

        - ``"unity_gain"``: the sum of the elements, the DC gain of filter
          taps.  A filter normalized this way keeps signal levels.
        - ``"unit_energy"``: the energy ``sum(|x|²)``, as for pulse-shaping
          and matched-filter taps, which then keep the noise power.
        - ``"peak"``: the largest magnitude ``max |x[n]|``.  The bound is
          invariant under unit-magnitude operations (frequency shifts, phase
          rotations), so it survives the rest of a DSP chain.
        - ``"average_power"``: the mean sample power ``E[|x|²]``, for symbols
          at one sample per symbol.  Do not use it on an oversampled
          waveform: with unit-energy pulse taps its natural sample power is
          ``1/sps``, and this mode would scale it up by ``√sps``.
        - ``"symbol_power"``: the symbol energy ``sps·E[|x|²]``, the library's
          invariant ``E[|x|²] = 1/sps``.  Every pulse shape then ends at the
          same power, and ``apply_awgn`` can take ``Es = 1``.  Equal to
          ``"average_power"`` at ``sps=1``.
        - ``"dac_peak"``: the larger of the I and Q peaks,
          ``max(max |Re x|, max |Im x|)``.  It keeps the I/Q ratio and uses
          the full DAC range whatever the constellation's orientation; with
          ``"peak"``, square QAM, whose largest points sit at 45°, reaches only
          ``1/√2`` on each axis.  Normalize frame sections (preamble, body)
          separately to give each the full range.
    axis : int, optional
        Axis along which the factor is computed, so each slice is normalized
        separately.  By default the whole array shares one factor.
    sps : float, default 1
        Samples per symbol, used by ``"symbol_power"`` only.

    Returns
    -------
    ndarray
        The normalized array, same dtype and device.  An all-zero slice stays
        zero.
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
    is_zero = norm_factor == 0
    result = x / xp.where(is_zero, xp.ones_like(norm_factor), norm_factor)
    # A zero factor means an all-zero slice, which 0 / 1 keeps at zero, except
    # for "unity_gain", where a non-zero slice can sum to zero.
    if mode == "unity_gain":
        result = xp.where(is_zero, xp.zeros((), dtype=x.dtype), result)
    return result


# ---------------------------------------------------------------------------
# dB <-> linear ratio conversion
# ---------------------------------------------------------------------------


def db_to_linear(db: ArrayType | float, *, power: bool = True) -> ArrayType | float:
    """Convert decibels to a linear ratio.

    ``power=True`` uses the power convention ``10**(db/10)``, for Es/N0 or
    an SNR.  ``power=False`` uses the amplitude convention ``10**(db/20)``,
    for a gain applied to field samples, such as an I/Q gain imbalance or a
    pilot-tone gain.  The wrong convention squares or square-roots the
    result, so check which quantity a ``_db`` parameter describes.

    Parameters
    ----------
    db : array_like or float
        Value in decibels.
    power : bool, default True
        Power (``10**(db/10)``) or amplitude (``10**(db/20)``) convention.

    Returns
    -------
    ndarray or float
        The linear ratio, same type as ``db``.
    """
    exponent = 10.0 if power else 20.0
    return 10.0 ** (db / exponent)


def linear_to_db(x: ArrayType, *, power: bool = True) -> ArrayType:
    """Convert a linear ratio to decibels, the inverse of :func:`db_to_linear`.

    ``power=True`` uses the power convention ``10·log10(x)``, for an SNR.
    ``power=False`` uses the amplitude convention ``20·log10(x)``, for an
    amplitude ratio such as the EVM.  Zero maps to ``-inf`` and negative
    values to ``nan``, without a RuntimeWarning; guard the input if another
    policy is needed.

    Parameters
    ----------
    x : array_like
        Linear ratio.
    power : bool, default True
        Power (10·log10) or amplitude (20·log10) convention.

    Returns
    -------
    ndarray
        Value in decibels, same shape and device as ``x``.
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
    r"""Per-channel least-squares slope of a phase (or any) record.

    Ordinary least squares on the centered normal equations,

    .. math:: \hat{a} = \frac{\sum_k (x_k - \bar{x})(y_k - \bar{y})}
                             {\sum_k (x_k - \bar{x})^2},

    evaluated for every channel in one vectorized pass and returned on the
    input's device, without host synchronization, so the caller decides when
    (and whether) to transfer.

    Parameters
    ----------
    y : array_like
        Record to fit, ``(C, N)`` with the fit axis last (promote SISO with
        :func:`as_2d` first).
    x : array_like, optional
        Abscissa, ``(N,)``.  Defaults to the sample index ``arange(N)``, so the
        slope is then in *units of y per sample*.  Pass a time axis in seconds
        to get a slope per second, for example at non-uniform pilot positions.
    xp : module, optional
        Array module; inferred from ``y`` when omitted.

    Returns
    -------
    ndarray
        Slope per channel, ``(C,)``, ``float64``, on the input's device.
    """
    if xp is None:
        y, xp, _ = dispatch(y)
    n = y.shape[-1]
    xc, denom = _centered_axis(n, x, xp)
    # Subtracting the per-channel mean is mathematically redundant (xc is
    # already centered) but keeps the products small when the record carries a
    # large constant offset - phase trajectories routinely do.
    yc = y - xp.mean(y, axis=-1, keepdims=True)
    return xp.sum(yc * xc, axis=-1) / denom


def _remove_linear_trend(y: ArrayType, *, x: Any = None) -> tuple[ArrayType, ArrayType]:
    r"""Remove the per-channel least-squares linear trend, preserving the mean.

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
    detrended : ndarray
        ``y`` minus the fitted slope term, ``float64``, on the input's device.
    slope : ndarray
        Fitted slope per channel, ``(C,)``, in units of ``y`` per unit of
        ``x`` (per sample by default).
    """
    y, xp, _ = dispatch(y)
    slope = _linear_trend_slope(y, x=x, xp=xp)
    xc, _ = _centered_axis(y.shape[-1], x, xp)
    return y - slope[..., None] * xc[None, :], slope


# ---------------------------------------------------------------------------
# Phase rotation (private, shared)
# ---------------------------------------------------------------------------


def _unit_phasor(phase: ArrayType, dtype: Any, xp: Any) -> ArrayType:
    """``exp(j·phase)`` as ``dtype``, from a real cos and sin.

    Several times faster than a complex exponential on the CPU; on the GPU
    the complex exponential is one kernel and gives the same values.
    ``phase`` keeps its own precision for the cos and sin; wrap large phases
    first if that is float32.
    """
    if xp is not np:
        return xp.exp(1j * phase).astype(dtype, copy=False)
    h = xp.empty(phase.shape, dtype=dtype)
    h.real = xp.cos(phase)
    h.imag = xp.sin(phase)
    return h


def _phasor(phase: ArrayType, dtype: Any, xp: Any) -> ArrayType:
    """``exp(j·phase)`` as complex ``dtype`` for a float64 ``phase`` of any size.

    The phase is wrapped to ``[-π, π]`` in float64, then the cos and sin run
    in the real precision of ``dtype`` (no float64 trigonometry for a
    complex64 result).
    """
    two_pi = 2.0 * np.pi
    wrapped = phase - xp.round(phase / two_pi) * two_pi
    real = np.empty(0, dtype).real.dtype
    return _unit_phasor(wrapped.astype(real, copy=False), dtype, xp)


_DEROTATE_KERNEL: dict[str, Any] = {}


def _cupy_derotate() -> Any:
    """Fused CuPy kernel of ``s · exp(-j·wrap(φ))``.

    The float64 wrap, the float32 sin and cos and the complex product in one
    pass, instead of about 7 ufunc passes.
    """
    if "k" not in _DEROTATE_KERNEL:
        import cupy as cp

        _DEROTATE_KERNEL["k"] = cp.ElementwiseKernel(
            "complex64 s, float64 phi",
            "complex64 out",
            """
            double w = phi - rint(phi * 0.15915494309189535) * 6.283185307179586;
            float sw, cw;
            sincosf((float)w, &sw, &cw);
            out = s * complex<float>(cw, -sw);
            """,
            "commkit_phase_rotate",
        )
    return _DEROTATE_KERNEL["k"]


def _derotate(x: ArrayType, phase: Any) -> ArrayType:
    """``x · exp(-j·phase)`` in the precision of ``x``.

    ``phase`` (radians, broadcasting against ``x``, unbounded) is wrapped to
    ``[-π, π]`` in float64, then the phasor is built in the real precision
    of ``x``: complex64 data is never promoted, complex128 keeps double
    precision.  Real ``x`` gives the matching complex type.
    """
    x, xp, _ = dispatch(x)
    phi = xp.asarray(phase, dtype=xp.float64)
    if xp is not np and x.dtype == xp.complex64:
        return _cupy_derotate()(x, phi)
    return x * _phasor(-phi, np.result_type(x.dtype, np.complex64), xp)
