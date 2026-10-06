"""Domain helpers awaiting their owning modules (removed in 4.1).

Shape and validation helpers live in ``commkit._array``; power and dB
conversions in ``commkit.math``.
"""

from typing import Any

import numpy as np

from ._array import as_2d
from .backend import ArrayType, dispatch

# ---------------------------------------------------------------------------
# Correlation & sequences
# ---------------------------------------------------------------------------


def cross_correlate_fft(
    samples: ArrayType,
    template: ArrayType,
    mode: str = "full",
) -> ArrayType:
    """
    Vectorized FFT-based cross-correlation.

    Computes the cross-correlation of ``samples`` with ``template`` using
    the frequency-domain multiplication approach. Handles 1D and 2D
    (multichannel) inputs natively via ``axis=-1`` broadcasting - no
    Python loops over channels.

    Parameters
    ----------
    samples : array_like
        Input samples. Shape: ``(N,)`` or ``(C, N)``.
    template : array_like
        Reference sequence. Shape: ``(L,)`` or ``(C, L)``.
        If ``(1, L)`` and samples is ``(C, N)``, the template is
        broadcast across all channels.
    mode : {"full", "same", "valid", "positive_lags"}, default "full"
        Output size:
        - ``"full"``: length ``N + L - 1``.
        - ``"same"``: length ``N`` (centered).
        - ``"valid"``: length ``max(N, L) - min(N, L) + 1``.
        - ``"positive_lags"``: length ``N`` (lags 0 ... N-1 only). Returns a
          zero-copy view of the raw circular-correlation output - no
          ``concatenate`` and no reordering. Use this when negative lags are
          not needed (e.g. frame timing search within a bounded window).

    Returns
    -------
    array_like
        Complex cross-correlation with shape matching the input
        dimensionality and the selected ``mode``.
    """
    samples, xp, _ = dispatch(samples)
    template = xp.asarray(template)

    samples, was_1d = as_2d(samples, name="samples")
    if template.ndim == 1:
        template = template[None, :]

    N = samples.shape[-1]
    L = template.shape[-1]
    full_len = N + L - 1

    # Smallest power-of-2 >= full_len for FFT efficiency.
    # `(full_len - 1).bit_length()` is the canonical integer-only formula;
    # `full_len.bit_length()` would round up even when full_len is already a power of 2.
    n_fft = 1 << (full_len - 1).bit_length()

    # FFT-based correlation: R[k] = IFFT(FFT(samples) * conj(FFT(template)))
    # Circular correlation places positive lags at 0..N-1 and negative lags
    # wrap to n_fft-(L-1)..n_fft-1.  Rearrange to match scipy layout:
    # lags [-(L-1), ..., -1, 0, 1, ..., N-1]  (total = N + L - 1).
    SIG = xp.fft.fft(samples, n_fft, axis=-1)
    TPL = xp.fft.fft(template, n_fft, axis=-1)
    corr_circ = xp.fft.ifft(SIG * xp.conj(TPL), axis=-1)

    # Gather negative lags (indices n_fft-(L-1) .. n_fft-1) then positive (0 .. N-1)
    neg_lags = corr_circ[..., n_fft - L + 1 :]  # length L-1
    pos_lags = corr_circ[..., :N]  # length N
    corr = xp.concatenate([neg_lags, pos_lags], axis=-1)  # length N+L-1

    # Apply mode trimming
    if mode == "positive_lags":
        corr = corr_circ[..., :N]  # zero-copy view; lags 0 ... N-1
    elif mode == "same":
        start = (L - 1) // 2
        corr = corr[..., start : start + N]
    elif mode == "valid":
        valid_len = max(N, L) - min(N, L) + 1
        start = min(N, L) - 1
        corr = corr[..., start : start + valid_len]
    # mode == "full": no trimming needed

    if was_1d:
        return corr[0]
    return corr


def zc_mimo_root(stream_idx: int, base_root: int, length: int) -> int:
    """
    Returns the Zadoff-Chu root for TX stream ``stream_idx`` in a MIMO preamble.

    Assigns a deterministic unique root to each TX stream by cycling through
    distinct roots starting from ``base_root``, wrapping in the range
    ``[1, length-1]``.  For prime ``length`` all roots are valid CAZAC
    sequences; any two distinct roots are near-orthogonal with cross-correlation
    magnitude ``1/sqrt(length)`` at every lag.

    Parameters
    ----------
    stream_idx : int
        TX stream index (0-based).
    base_root : int
        ZC root assigned to stream 0.  Must be in ``[1, length-1]``.
    length : int
        Sequence length (should be prime for the CAZAC property).

    Returns
    -------
    int
        ZC root for stream ``stream_idx``, guaranteed in ``[1, length-1]``.

    Examples
    --------
    >>> [zc_mimo_root(k, 1, 13) for k in range(4)]
    [1, 2, 3, 4]
    >>> [zc_mimo_root(k, 10, 13) for k in range(4)]
    [10, 11, 12, 1]
    """
    return ((base_root - 1 + stream_idx) % (length - 1)) + 1


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
# Three-point (log-)parabolic sub-bin/sub-sample peak fit
# ---------------------------------------------------------------------------


def _parabolic_peak_offset(
    y_prev: ArrayType,
    y_curr: ArrayType,
    y_next: ArrayType,
    xp: Any,
    *,
    log: bool = False,
    log_eps: float = 1e-300,
    denom_eps: float | None = None,
) -> ArrayType:
    r"""Three-point (log-)parabolic sub-bin/sub-sample peak-offset fit.

    Fits a parabola through three samples straddling a peak (bins/samples
    k-1, k, k+1) - or through their logs, for the standard log-parabolic
    (Gaussian-equivalent) fit - and returns the offset of the true peak
    relative to the center sample, clipped to ``[-0.5, 0.5]``:

        delta = 0.5 * (y_prev - y_next) / (y_prev - 2*y_curr + y_next)

    Shared by every three-point peak-interpolation site in the library:
    the FOE M-th-power estimator's magnitude-domain fit
    (``frequency.estimate_frequency_offset_mth_power``, ``log=False``), the
    two log-parabolic tone-refinement estimators
    (``frequency.find_bias_tone``, ``frequency._refine_tones_from_spectrum``,
    ``log=True``), and the fractional-delay estimator
    (``timing.estimate_fractional_delay``, either fit) - which first
    phase-rotates a complex peak onto the real axis (a preprocessing step
    outside this function's scope) before calling this with its own
    ``log_eps``/``denom_eps`` tuning.  Inputs may be plain Python floats
    with ``xp=numpy`` (host scalars) or device arrays (NumPy/CuPy) - the
    arithmetic is expressed purely through ``xp``, so both execution models
    are supported by the same implementation.

    Parameters
    ----------
    y_prev, y_curr, y_next : array_like or float
        Samples at bins/positions k-1, k, k+1.
    xp : module
        Array module (``numpy``/``cupy``) providing ``log``, ``maximum``,
        ``abs``, ``where``, ``ones_like``, ``zeros_like``, ``clip``.
    log : bool, default False
        If True, fits to ``log(max(y, log_eps))`` (log-parabolic / Gaussian
        fit - standard for spectral-magnitude tone estimation). If False,
        fits directly to ``y`` (magnitude-domain fit).
    log_eps : float, default 1e-300
        Clamp floor before taking the log. Only used when ``log=True``.
    denom_eps : float, optional
        Threshold below which the fit denominator is treated as degenerate
        (returns ``delta=0`` instead of dividing). Defaults to ``1e-30`` when
        ``log=True``, ``1e-15`` when ``log=False`` - the values already in
        use at every call site except ``timing.estimate_fractional_delay``,
        which passes its own tuning explicitly.

    Returns
    -------
    delta : same type as inputs
        Sub-bin/sub-sample offset in ``[-0.5, 0.5]``.
    """
    if denom_eps is None:
        denom_eps = 1e-30 if log else 1e-15
    if log:
        y_prev = xp.log(xp.maximum(y_prev, log_eps))
        y_curr = xp.log(xp.maximum(y_curr, log_eps))
        y_next = xp.log(xp.maximum(y_next, log_eps))

    denom = y_prev - 2.0 * y_curr + y_next
    valid = xp.abs(denom) > denom_eps
    safe_denom = xp.where(valid, denom, xp.ones_like(denom))
    raw = 0.5 * (y_prev - y_next) / safe_denom
    delta = xp.where(valid, raw, xp.zeros_like(raw))
    return xp.clip(delta, -0.5, 0.5)


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
