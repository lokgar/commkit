"""Numba loop of the Mengali-Morelli frequency estimator (private).

The iterative bootstrap over the autocorrelation lags behind
:class:`commkit.frequency.MengaliMorelli`.  Imported on first use through
``_get_numba_mm_bootstrap`` in ``frequency.py``, so ``import commkit``
does not load Numba.
"""

import numba
import numpy as np


@numba.njit(cache=True, fastmath=True, nogil=True, parallel=True)
def mm_bootstrap(
    theta: np.ndarray, amp: np.ndarray, M_val: float, fs: float
) -> np.ndarray:
    """Run the Mengali-Morelli bootstrap over the lags of each row (Numba).

    Each lag's phase is predicted from the weighted frequency estimate of the
    previous lags, then folded into the weighted sum.  The dependency between
    lags prevents vectorization; the rows are independent and run in
    parallel.

    Parameters
    ----------
    theta : numpy.ndarray
        Wrapped phase of ``R[m]`` at each lag, float64, ``(R, L)``.
    amp : numpy.ndarray
        Magnitude ``|R[m]|`` at each lag, float64, ``(R, L)``.
    M_val : float
        Power ``M`` of the pre-processing: 1 for data-aided or generic input,
        the order for PSK, 4 for QAM.
    fs : float
        Sampling rate in Hz.

    Returns
    -------
    numpy.ndarray
        Estimated frequency offset of each row in Hz, float64, ``(R,)``.
    """
    two_pi = 2.0 * np.pi
    rows, L = theta.shape
    out = np.empty(rows, dtype=np.float64)

    for r in numba.prange(rows):  # type: ignore[attr-defined,no-untyped-call]
        # Lag 1 initialization (m=1, m²=1)
        Theta_0 = theta[r, 0]
        w0 = amp[r, 0] * amp[r, 0]  # m² · |R|² at m=1
        f_hat = Theta_0 * fs / (two_pi * M_val)
        w_sum = w0 if w0 > 1e-30 else 1e-30
        wf_sum = w_sum * f_hat

        for m_idx in range(1, L):
            m_val = float(m_idx + 1)
            predicted = two_pi * f_hat * M_val * m_val / fs
            diff = predicted - theta[r, m_idx]
            # round() is the C math round - unboxed, no Python overhead
            correction = round(diff / two_pi)
            Theta_m = theta[r, m_idx] + two_pi * correction
            f_m = Theta_m * fs / (two_pi * m_val * M_val)
            w_m = m_val * m_val * amp[r, m_idx] * amp[r, m_idx]
            w_sum += w_m
            wf_sum += w_m * f_m
            f_hat = wf_sum / w_sum

        out[r] = f_hat

    return out
