"""Tests for data-aided carrier-phase trajectory extraction.

Inputs are built on the active backend via the ``xp`` fixture so CPU and GPU
paths are both exercised.
"""

import numpy as np

from commkit import analysis
from commkit.backend import to_device
from commkit.core import Signal
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


def test_carrier_phase_trajectory_auto_pairing_under_frequency_offset(xp):
    """Pairing must survive the residual FOE this stage exists to measure."""
    n = 1 << 13
    d0, d1 = _qpsk(n, 1), _qpsk(n, 2)
    p0, p1 = _wiener_phase(5e5, n, 3), _wiener_phase(5e5, n, 4)
    # 0.005 cycles/symbol of residual carrier left on both channels.
    ramp = 2.0 * np.pi * 0.005 * np.arange(n)
    y = np.stack([d0 * np.exp(1j * (p0 + ramp)), d1 * np.exp(1j * (p1 + ramp))])
    d_swapped = np.stack([d1, d0])
    phi = analysis.carrier_phase_trajectory(
        xp.asarray(y), xp.asarray(d_swapped), channel_pairing="auto"
    )
    # Correct pairing -> increments are the common ramp plus the phase walk.
    dphi = xp.diff(phi, axis=-1) - 2.0 * np.pi * 0.005
    assert float(xp.var(dphi)) < 0.1


def test_carrier_phase_trajectory_auto_pairing_beyond_dual_pol(xp, xpt):
    """Auto pairing generalizes past C == 2 (cyclically permuted 3-stream)."""
    n = 1 << 12
    d = [_qpsk(n, s) for s in (1, 2, 3)]
    p = [_wiener_phase(5e5, n, s) for s in (4, 5, 6)]
    y = np.stack([di * np.exp(1j * pi) for di, pi in zip(d, p)])
    d_rolled = np.stack([d[1], d[2], d[0]])  # equalizer emitted a 3-cycle
    phi = analysis.carrier_phase_trajectory(
        xp.asarray(y), xp.asarray(d_rolled), channel_pairing="auto"
    )
    for c in range(3):
        xpt.assert_allclose(xp.diff(phi[c]), xp.asarray(np.diff(p[c])), atol=1e-6)


def test_carrier_phase_trajectory_signal_input(xp, xpt):
    """Signal input: y_eq is unwrapped to .samples; output stays a raw array."""
    n = 1 << 12
    phi_true = _wiener_phase(1e6, n)
    d = _qpsk(n)
    y = d * np.exp(1j * phi_true)
    sig = Signal(samples=xp.asarray(y), sampling_rate=R, symbol_rate=R)

    phi_sig = analysis.carrier_phase_trajectory(sig, xp.asarray(d))
    phi_arr = analysis.carrier_phase_trajectory(xp.asarray(y), xp.asarray(d))

    assert not isinstance(phi_sig, Signal)
    xpt.assert_allclose(phi_sig, phi_arr)
