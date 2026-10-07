"""Tests for the delayed self-heterodyne front end (dsh_beat, dsh_phase).

Beats are synthesized from a discrete Wiener laser phase of known linewidth
(``Δφ(t) = φ(t) - φ(t-τ_d)`` modulated onto an AOM carrier) so the extraction
is checked against analytic ground truth. Inputs are built on the active
backend via the ``xp`` fixture.
"""

import numpy as np
import pytest

from commkit import analysis
from commkit.backend import to_device
from commkit.core import Signal
from commkit.impairments import generate_phase_noise
from tests.common.signals import make_dsh_beat

FS = 500e6  # beat sampling rate (Hz)


class TestDSHBeatForwardModel:
    """Forward model synthesis and validation for DSH beats."""

    def test_dsh_beat_forward_model(self, xp, xpt):
        """dsh_beat: unit amplitude, exact differential phase, homodyne case."""
        n, m = 1 << 14, 250
        phi = xp.asarray(
            to_device(
                generate_phase_noise(
                    num_samples=n + m, sampling_rate=FS, linewidth=1e6, rng=42
                ),
                "cpu",
            )
        )
        z, dphi = analysis.dsh_beat(phi, sampling_rate=FS, delay=m / FS, f_shift=80e6)
        assert z.shape == (n,)
        assert dphi.shape == (n,)
        xpt.assert_allclose(xp.abs(z), xp.ones(n), atol=1e-12)
        xpt.assert_allclose(dphi, phi[m:] - phi[:-m], atol=0.0)
        # Homodyne (f_shift = 0): the beat is exp(j·Δφ) exactly.
        z0, dphi0 = analysis.dsh_beat(phi, sampling_rate=FS, delay=m / FS)
        xpt.assert_allclose(z0, xp.exp(1j * dphi0), atol=1e-12)

    def test_dsh_beat_rejects_bad_delay(self, xp):
        phi = xp.zeros(100, dtype=xp.float64)
        with pytest.raises(ValueError, match="delay"):
            analysis.dsh_beat(phi, sampling_rate=FS, delay=0.0)
        with pytest.raises(ValueError, match="delay"):
            analysis.dsh_beat(phi, sampling_rate=FS, delay=100 / FS)


class TestDSHPhaseEstimation:
    """Differential carrier phase extraction from DSH beats."""

    def test_dsh_phase_recovers_differential_phase(self, xp, xpt):
        n, m = 1 << 18, 500
        z, dphi = make_dsh_beat(2e6, n, m, 80e6, snr_db=None, seed=1)
        dp, f_used = analysis.dsh_phase(xp.asarray(z), sampling_rate=FS, f_shift=80e6)
        assert f_used == pytest.approx(80e6)
        # Constant offsets are irrelevant; the increments must match exactly.
        xpt.assert_allclose(xp.diff(dp), xp.asarray(np.diff(dphi)), atol=1e-6)

    def test_dsh_phase_estimated_carrier_leaves_no_ramp(self, xp):
        """Kay + LS two-stage carrier removal: Var[Δφ] must match 2πΔν·τ_d."""
        n, m = 1 << 20, 500
        td = m / FS
        z, _ = make_dsh_beat(2e6, n, m, 80e6, snr_db=25, seed=2)
        dp, f_hat = analysis.dsh_phase(xp.asarray(z), sampling_rate=FS)
        assert f_hat == pytest.approx(80e6, abs=5e3)
        # A residual carrier ramp would inflate this variance by an order of
        # magnitude (LS detrend regression guard).
        assert float(xp.var(dp)) == pytest.approx(2.0 * np.pi * 2e6 * td, rel=0.15)

    def test_dsh_phase_real_input_hilbert(self, xp):
        # Narrow line vs f_aom so the beat spectrum is one-sided (Hilbert-exact).
        n, m = 1 << 18, 500
        z, dphi = make_dsh_beat(2e5, n, m, 80e6, snr_db=None, seed=3)
        dp, f_hat = analysis.dsh_phase(
            xp.asarray(z.real), sampling_rate=FS, f_shift=80e6
        )
        edge = 1000  # Hilbert edge transients
        resid = (xp.asarray(dphi) - dp)[edge:-edge]
        err = resid - xp.mean(resid)
        assert float(xp.std(err)) < 0.05

    def test_dsh_phase_real_homodyne_rejected(self, xp):
        x = xp.asarray(np.cos(np.linspace(0.0, 20.0, 1000)))
        with pytest.raises(ValueError, match="cannot be inverted"):
            analysis.dsh_phase(x, sampling_rate=FS, f_shift=0.0)


class TestSignalInputInterferometry:
    """Signal-awareness for dsh_phase."""

    def test_dsh_phase_signal_input(self, xp, xpt):
        n, m = 1 << 16, 500
        z, _ = make_dsh_beat(2e6, n, m, 80e6, snr_db=None, seed=1)
        sig = Signal(samples=xp.asarray(z), sampling_rate=FS, symbol_rate=FS)

        dp_sig, f_sig = analysis.dsh_phase(sig, f_shift=80e6)
        dp_arr, f_arr = analysis.dsh_phase(
            xp.asarray(z), sampling_rate=FS, f_shift=80e6
        )

        assert not isinstance(dp_sig, Signal)
        assert f_sig == pytest.approx(f_arr)
        xpt.assert_allclose(dp_sig, dp_arr)

    def test_conflicting_sampling_rate_raises(self, xp):
        """sampling_rate is a fact: a value that disagrees with the Signal raises."""
        z, _ = make_dsh_beat(2e6, 1 << 12, 50, 80e6, snr_db=None, seed=1)
        sig = Signal(samples=xp.asarray(z), sampling_rate=FS, symbol_rate=FS)
        with pytest.raises(ValueError, match="conflicts"):
            analysis.dsh_phase(sig, sampling_rate=FS / 2)
