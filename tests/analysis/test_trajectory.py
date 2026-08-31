"""Tests for data-aided carrier-phase trajectory extraction.

Inputs are built on the active backend via the ``xp`` fixture so CPU and GPU
paths are both exercised.
"""

import numpy as np

from commkit import analysis
from commkit.backend import to_device
from commkit.impairments import generate_phase_noise

R = 32e9  # symbol rate (Baud)
T = 1.0 / R


def _wiener_phase(linewidth, n, seed=0):
    """Discrete Wiener phase walk at the symbol rate (NumPy, float64)."""
    return to_device(generate_phase_noise(n, R, linewidth=linewidth, seed=seed), "cpu")


def _qpsk(n, seed=1):
    rng = np.random.default_rng(seed)
    return rng.choice([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j], n) / np.sqrt(2.0)


def test_carrier_phase_trajectory_recovers_phase(xp, xpt):
    n = 1 << 14
    phi_true = _wiener_phase(1e6, n)
    d = _qpsk(n)
    y = d * np.exp(1j * phi_true)  # noise-free
    phi = analysis.carrier_phase_trajectory(xp.asarray(y), xp.asarray(d))
    # Only the time-variation is meaningful (constant offset is irrelevant).
    dphi = xp.diff(phi)
    xpt.assert_allclose(dphi, xp.asarray(np.diff(phi_true)), atol=1e-6)


def test_carrier_phase_trajectory_auto_pairing(xp):
    n = 1 << 13
    d0, d1 = _qpsk(n, 1), _qpsk(n, 2)
    p0, p1 = _wiener_phase(5e5, n, 3), _wiener_phase(5e5, n, 4)
    y = np.stack([d0 * np.exp(1j * p0), d1 * np.exp(1j * p1)])
    d_swapped = np.stack([d1, d0])  # equalizer mapped pol 0<->1
    phi = analysis.carrier_phase_trajectory(
        xp.asarray(y), xp.asarray(d_swapped), channel_pairing="auto"
    )
    # With the right pairing restored, the phase-error increment variance is
    # tiny (only the phase walk); a wrong pairing would be ~uniform on [-π, π].
    var = float(xp.var(xp.diff(phi, axis=-1)))
    assert var < 0.1
