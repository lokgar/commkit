"""Overlapping Allan deviation of an instantaneous-frequency series."""

import numpy as np

from ..backend import ArrayType, dispatch, to_device
from ._common import _as_2d

__all__ = ["allan_deviation"]


def allan_deviation(
    df: ArrayType,
    symbol_rate: float,
    *,
    taus: np.ndarray | None = None,
    n_taus: int = 30,
    debug_plot: bool = False,
) -> dict[str, np.ndarray]:
    r"""Overlapping Allan deviation of an instantaneous-frequency series.

    The log-log slope of sigma_y(tau) classifies the dominant noise
    process by averaging time: white-FM proportional to tau^(-1/2),
    flicker-FM proportional to tau^0, random-walk-FM proportional to tau^(+1/2),
    linear drift proportional to tau^(+1).

    Parameters
    ----------
    df : array_like
        Instantaneous frequency samples in Hz (e.g. ``frequency_drift_metrics``
        ``df``), ``(N,)`` or ``(C, N)``, sampled at ``symbol_rate``.
    symbol_rate : float
        Sample rate of ``df`` in Hz (``τ_0 = 1/symbol_rate``).
    taus : array_like, optional
        Explicit averaging times in seconds.  Default: ``n_taus`` values
        geometrically spaced from ``τ_0`` to ``N//4·τ_0``.
    n_taus : int, default 30
        Number of log-spaced averaging times when ``taus`` is None.
    debug_plot : bool, default False
        If True, plot the Allan deviation
        (``allan_deviation``).

    Returns
    -------
    dict
        ``{'tau_s', 'adev'}`` where ``adev`` is ``(n_tau,)`` (SISO) or
        ``(C, n_tau)`` (MIMO).  NumPy arrays.

    Notes
    -----
    **Limitations.**

    * ``df`` is in **Hz**, not the dimensionless fractional frequency
      ``y = δf/ν₀`` of clock metrology; ``adev`` is therefore in Hz.  Divide
      by the optical carrier frequency (~193 THz) to compare against
      oscillator specs quoted fractionally.
    * The overlapping estimator improves confidence but the largest-τ points
      still average only ``~N/(2m)`` second differences - the last decade of
      the curve is statistically fragile (hence the ``N//4`` default cap).
      No confidence intervals are computed.
    * ADEV cannot distinguish white-PM from flicker-PM noise (both fall as
      ``~τ⁻¹`` after the frequency differencing); the modified Allan deviation
      would be needed for that.  For laser work this matters at small τ where
      AWGN angle noise (white-PM) dominates.
    * A deterministic sinusoidal wander of period ``T_m`` produces the classic
      scalloped ADEV with nulls at ``τ = k·T_m`` - do not read those dips as
      noise-floor improvements.
    """
    df_arr, xp, _ = dispatch(df)
    y2, was_1d = _as_2d(df_arr)
    c, n = y2.shape
    tau0 = 1.0 / float(symbol_rate)

    # Cumulative phase (time error) x_i = Σ y · τ0.
    zeros_col = xp.zeros((c, 1), dtype=xp.float64)
    x = xp.concatenate(
        [zeros_col, xp.cumsum(y2.astype(xp.float64), axis=-1) * tau0], axis=-1
    )

    if taus is None:
        m_max = max(1, n // 4)
        ms = np.unique(np.round(np.geomspace(1, m_max, n_taus)).astype(int))
    else:
        taus_cpu = np.asarray(to_device(taus, "cpu"), dtype=np.float64)
        ms = np.unique(np.maximum(1, np.round(taus_cpu / tau0).astype(int)))
        ms = ms[ms <= max(1, (x.shape[-1] - 1) // 2)]

    tau_s = ms * tau0
    adev = xp.full((c, ms.size), xp.nan, dtype=xp.float64)
    # One vectorized second-difference per τ across all channels (the C-loop
    # would launch C·n_tau tiny kernels on GPU for no benefit).
    for j, m in enumerate(ms):
        m = int(m)
        if x.shape[-1] - 2 * m < 1:
            continue
        d2 = x[:, 2 * m :] - 2.0 * x[:, m:-m] + x[:, : -2 * m]
        avar = xp.mean(d2**2, axis=-1) / (2.0 * (m * tau0) ** 2)
        adev[:, j] = xp.sqrt(avar)

    tau_s_cpu = np.asarray(tau_s, dtype=np.float64)
    adev_cpu = to_device(adev, "cpu")
    adev_out = adev_cpu[0] if was_1d else adev_cpu

    if debug_plot:
        from .. import plotting as _plotting

        _plotting.plot_allan_deviation(tau_s_cpu, adev_out, show=True)

    return {"tau_s": tau_s_cpu, "adev": adev_out}
