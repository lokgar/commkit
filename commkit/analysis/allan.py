"""Overlapping Allan deviation of an instantaneous-frequency series.

:func:`allan_deviation` returns an :class:`AllanDeviation`.
"""

from dataclasses import dataclass

import numpy as np

from .._array import as_2d, require_finite, require_int, restore_1d
from .._value import _Record
from ..backend import ArrayType, _to_host, dispatch, to_device

__all__ = ["AllanDeviation", "allan_deviation"]


@dataclass(frozen=True, eq=False)
class AllanDeviation(_Record):
    """Overlapping Allan deviation of a frequency record.

    Attributes
    ----------
    taus : numpy.ndarray
        Averaging times in seconds, ``(n_tau,)``.
    adev : numpy.ndarray
        Allan deviation in Hz, ``(n_tau,)`` or ``(C, n_tau)``.
    """

    taus: np.ndarray
    adev: np.ndarray


def allan_deviation(
    df: ArrayType,
    *,
    sampling_rate: float,
    taus: np.ndarray | None = None,
    num_taus: int = 30,
) -> AllanDeviation:
    r"""Overlapping Allan deviation of an instantaneous-frequency series.

    The log-log slope of ``σ_y(τ)`` identifies the dominant noise process at
    each averaging time: white FM goes as ``τ^(-1/2)``, flicker FM as
    ``τ^0``, random-walk FM as ``τ^(1/2)`` and a linear drift as ``τ^1``.

    Parameters
    ----------
    df : array_like
        Instantaneous frequency samples in Hz (e.g. ``frequency_drift(...).df``),
        ``(N,)`` or ``(C, N)``.
    sampling_rate : float
        Sampling rate of ``df`` in Hz (``τ_0 = 1/sampling_rate``).
    taus : array_like, optional
        Averaging times in seconds, rounded to whole multiples ``m·τ_0``;
        those with ``m > N/2``, which the record cannot hold twice, are
        left out.  Defaults to ``num_taus`` values spaced geometrically from
        ``τ_0`` to ``N//4·τ_0``.
    num_taus : int, default 30
        Number of averaging times when ``taus`` is ``None``.

    Returns
    -------
    AllanDeviation
        ``taus`` and ``adev``, ``(n_tau,)`` (SISO) or ``(C, n_tau)`` (MIMO),
        host NumPy arrays.

    Raises
    ------
    ValueError
        If ``sampling_rate`` is not finite and ``> 0``, ``num_taus < 1``,
        ``df`` has fewer than 2 samples, ``taus`` is not a non-empty 1-D
        array of finite values ``> 0``, or none of ``taus`` fits the record.
    TypeError
        If ``sampling_rate`` is not a real number or ``num_taus`` not an
        integer.

    Notes
    -----
    **Limitations.**

    * ``df`` is in Hz, not the dimensionless fractional frequency
      ``y = δf/ν₀`` of clock metrology, so ``adev`` is in Hz too.  Divide by
      the optical carrier frequency (about 193 THz) to compare with
      oscillator specifications quoted as fractions.
    * The overlapping estimator improves the confidence, but the points at
      the largest ``τ`` still average only ``~N/(2m)`` second differences:
      the last decade of the curve is statistically fragile (hence the
      default cap at ``N//4``).  No confidence intervals are computed.
    * ADEV cannot distinguish white-PM from flicker-PM noise (both fall as
      ``~τ⁻¹`` after the frequency differencing); the modified Allan deviation
      would be needed for that.  For lasers this matters at small ``τ``,
      where the angle noise of the AWGN (white PM) dominates.
    * A deterministic sinusoidal wander of period ``T_m`` gives the familiar
      scalloped ADEV with nulls at ``τ = k·T_m``; those dips are not a lower
      noise floor.
    """
    fs = require_finite(sampling_rate, "allan_deviation(): sampling_rate", low=0.0)
    num_taus = require_int(num_taus, "allan_deviation(): num_taus", low=1)
    df_arr, xp, _ = dispatch(df)
    y2, was_1d = as_2d(df_arr, name="df")
    c, n = y2.shape
    if n < 2:
        raise ValueError(f"allan_deviation(): df needs at least 2 samples, got {n}.")
    tau0 = 1.0 / fs

    # Cumulative phase (time error) x_i = Σ y · τ0, in one float64 buffer.
    x = xp.empty((c, n + 1), dtype=xp.float64)
    x[:, 0] = 0.0
    xp.cumsum(y2, axis=-1, dtype=xp.float64, out=x[:, 1:])
    x[:, 1:] *= tau0

    if taus is None:
        m_max = max(1, n // 4)
        ms = np.unique(np.round(np.geomspace(1, m_max, num_taus)).astype(int))
    else:
        taus_cpu = _to_host(taus, np.float64)
        if (
            taus_cpu.ndim != 1
            or taus_cpu.size == 0
            or not np.all(np.isfinite(taus_cpu) & (taus_cpu > 0.0))
        ):
            raise ValueError(
                "allan_deviation(): taus must be a non-empty 1-D array of finite "
                "values > 0."
            )
        ms = np.unique(np.maximum(1, np.round(taus_cpu / tau0).astype(int)))
        ms = ms[ms <= n // 2]
        if ms.size == 0:
            raise ValueError(
                f"allan_deviation(): no tau fits the record of {n} samples "
                f"(the largest is {n // 2}/sampling_rate = {n // 2 * tau0:.3g} s)."
            )

    taus = ms * tau0
    adev = xp.empty((c, ms.size), dtype=xp.float64)
    # One vectorized second-difference per τ across all channels (the C-loop
    # would launch C·n_tau tiny kernels on GPU for no benefit).
    for j, m in enumerate(ms):
        m = int(m)
        d2 = x[:, 2 * m :] - 2.0 * x[:, m:-m] + x[:, : -2 * m]
        avar = xp.mean(d2**2, axis=-1) / (2.0 * (m * tau0) ** 2)
        adev[:, j] = xp.sqrt(avar)

    taus_cpu = np.asarray(taus, dtype=np.float64)
    adev_cpu = to_device(adev, "cpu")
    adev_out = restore_1d(was_1d, adev_cpu)

    return AllanDeviation(taus=taus_cpu, adev=adev_out)
