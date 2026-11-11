"""Sub-bin peak location (private).

:func:`parabolic_peak_offset`, the three-point (log-)parabolic fit behind
every peak estimate of the library (the frequency estimators, the
fractional delay of a correlation peak), and :func:`refine_tones`, which
refines tone frequencies on a record's spectrum in one batch for the pilot
tones of carrier recovery and the polarization tones of the equalizers.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np

from ._array import as_2d
from .backend import ArrayType, dispatch, to_device


def parabolic_peak_offset(
    y_prev: ArrayType,
    y_curr: ArrayType,
    y_next: ArrayType,
    xp: Any,
    *,
    log: bool = False,
    log_eps: float = 1e-300,
    denom_eps: float | None = None,
) -> ArrayType:
    r"""Offset of a peak from three samples by a (log-)parabolic fit.

    Fits a parabola through the samples at ``k-1``, ``k`` and ``k+1`` around
    a peak, or through their logarithms (the log-parabolic fit, equivalent to
    a Gaussian), and returns the offset of the vertex from ``k``, clipped to
    ``[-0.5, 0.5]``::

        delta = 0.5 * (y_prev - y_next) / (y_prev - 2*y_curr + y_next)

    Every three-point peak fit of the library uses it: ``frequency.MthPower``
    on magnitudes (``log=False``), ``frequency.SpectralPeak`` and
    :func:`refine_tones` (``log=True``), and
    :func:`estimate_fractional_delay` with either fit, which first rotates a
    complex peak onto the real axis and passes its own ``log_eps`` and
    ``denom_eps``.  The arithmetic goes through ``xp`` only, so the inputs
    may be host floats (``xp=numpy``) or device arrays.

    Parameters
    ----------
    y_prev, y_curr, y_next : array_like or float
        Samples at ``k-1``, ``k`` and ``k+1``.
    xp : module
        Array module (``numpy``/``cupy``) providing ``log``, ``maximum``,
        ``abs``, ``where``, ``ones_like``, ``zeros_like``, ``clip``.
    log : bool, default False
        Fit ``log(max(y, log_eps))``, the usual fit for spectral magnitudes,
        instead of ``y``.
    log_eps : float, default 1e-300
        Floor applied before the logarithm, with ``log=True`` only.
    denom_eps : float, optional
        Denominator magnitude below which the fit is degenerate and returns
        0.  Defaults to ``1e-30`` with ``log=True`` and ``1e-15`` without.

    Returns
    -------
    delta : ndarray or float
        Offset from the center sample in ``[-0.5, 0.5]``, the type of the
        inputs.
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


def refine_tones(
    X: ArrayType,
    sampling_rate: float,
    targets: Sequence[float],
    search_band: float,
    rows: Sequence[int] | np.ndarray | None = None,
) -> np.ndarray:
    """Refine tone frequencies on an FFT of the whole record, in one batch.

    The counterpart of ``SpectralPeak`` for callers that already hold the
    record's spectrum: each ``targets[t]`` is refined by an argmax search over
    ``targets[t] ± search_band`` in spectrum row ``rows[t]``, then by the same
    three-bin log-parabolic fit.  There is no zero padding (the fit makes up
    for the coarser bins) and no extra FFT, and the work per tone stays on the
    device: the refined frequencies come back in one host transfer.

    Parameters
    ----------
    X : ndarray
        FFT of the rows to search, in FFT-bin order, ``(R, N)``.
    sampling_rate : float
        Sampling rate in Hz.
    targets : sequence of float
        Nominal tone frequencies to refine, T values.
    search_band : float
        Half-width of the search window of each tone in Hz.
    rows : sequence of int, optional
        Spectrum row of each target; defaults to ``range(T)``, which needs
        ``R == T``.

    Returns
    -------
    numpy.ndarray
        Refined frequencies in Hz, float64, ``(T,)``.
    """
    X, xp, _ = dispatch(X)
    X, _ = as_2d(X, name="X")
    N = X.shape[-1]
    df = sampling_rate / N
    targets = [float(f) for f in targets]
    if rows is None:
        rows = list(range(len(targets)))
    neigh = xp.asarray([-1, 0, 1], dtype=xp.int64)

    refined = []
    for f_t, row in zip(targets, rows):
        # Signed candidate bins covering [f_t - band, f_t + band] on the N-grid.
        b_lo = max(int(np.ceil((f_t - abs(search_band)) / df)), -(N // 2))
        b_hi = min(int(np.floor((f_t + abs(search_band)) / df)), (N - 1) // 2)
        if b_hi < b_lo:
            raise ValueError(
                f"target_frequency={f_t} ± search_band={search_band} produces an "
                f"empty search window for fs={sampling_rate} Hz, N={N}."
            )
        cand = xp.arange(b_lo, b_hi + 1, dtype=xp.int64)  # signed bins
        spec_row = X[int(row)]
        mag2 = xp.abs(spec_row[cand % N]) ** 2
        kb = cand[xp.argmax(mag2)]  # 0-d signed peak bin, stays on device
        # Three-bin log-parabolic fit.
        mags = xp.abs(spec_row[(kb + neigh) % N]).astype(xp.float64)
        delta = parabolic_peak_offset(mags[0], mags[1], mags[2], xp, log=True)
        refined.append((kb.astype(xp.float64) + delta) * df)

    return to_device(xp.stack(refined), "cpu")
