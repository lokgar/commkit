"""Tests for transceiver front-end IQ-imbalance application and compensation."""

import pytest

from commkit.core import Signal
from commkit.impairments import (
    GramSchmidt,
    IQImbalanceEstimate,
    Lowdin,
    apply_iq_imbalance,
    correct_iq_imbalance,
    estimate_iq_imbalance,
)


class TestApplyIQImbalance:
    """Tests for apply_iq_imbalance."""

    def test_identity_zero_imbalance(self, xp, xpt):
        """Zero imbalance (0 dB, 0 deg) should leave the signal unchanged."""
        N = 1024
        rng = xp.random.RandomState(42)
        samples = (rng.randn(N) + 1j * rng.randn(N)).astype(xp.complex64)

        out = apply_iq_imbalance(
            samples, amplitude_imbalance_db=0.0, phase_imbalance_deg=0.0
        )

        xpt.assert_allclose(out, samples, atol=1e-5)

    def test_causes_impropriety(self, xp):
        """Imbalance should make a circular signal improper (κ > 0)."""
        N = 4096
        rng = xp.random.RandomState(7)
        samples = (rng.randn(N) + 1j * rng.randn(N)).astype(xp.complex64)

        # κ = |E[r²]| / E[|r|²]: zero for a circular signal
        def kappa(x):
            return float(xp.abs(xp.mean(x**2))) / float(xp.mean(xp.abs(x) ** 2))

        kappa_before = kappa(samples)
        out = apply_iq_imbalance(
            samples, amplitude_imbalance_db=2.0, phase_imbalance_deg=5.0
        )
        kappa_after = kappa(out)

        assert kappa_after > kappa_before + 0.05

    def test_output_dtype_preserved(self, xp):
        """Output dtype should match input dtype."""
        samples = xp.ones(256, dtype=xp.complex64)
        out = apply_iq_imbalance(
            samples, amplitude_imbalance_db=1.0, phase_imbalance_deg=2.0
        )
        assert out.dtype == xp.complex64


class TestIQImbalanceCompensation:
    """Tests for correct_iq_imbalance with Lowdin() and GramSchmidt()."""

    # κ = |E[r²]| / E[|r|²]: zero for a circular signal, positive for improper
    def _kappa(self, xp, x):
        return float(xp.abs(xp.mean(x**2))) / float(xp.mean(xp.abs(x) ** 2))

    def _make_imbalanced(self, xp, N=8192, seed=42):
        rng = xp.random.RandomState(seed)
        s = (rng.randn(N) + 1j * rng.randn(N)).astype(xp.complex64)
        r = apply_iq_imbalance(s, amplitude_imbalance_db=2.0, phase_imbalance_deg=5.0)
        return s, r

    # --- Löwdin ---

    def test_lowdin_restores_circularity(self, xp):
        """Löwdin compensation should drive κ to near zero."""
        _, r = self._make_imbalanced(xp)
        kappa_before = self._kappa(xp, r)
        out = correct_iq_imbalance(r, Lowdin())
        kappa_after = self._kappa(xp, out)
        assert kappa_after < 0.03
        assert kappa_after < kappa_before / 5

    def test_lowdin_iq_balance(self, xp):
        """After Löwdin, I and Q should have equal power and be orthogonal."""
        _, r = self._make_imbalanced(xp)
        out = correct_iq_imbalance(r, Lowdin())
        Iquad, Qquad = out.real, out.imag
        power_ratio = float(xp.mean(Iquad**2)) / float(xp.mean(Qquad**2))
        cross_corr = float(xp.abs(xp.mean(Iquad * Qquad))) / float(
            xp.mean(xp.abs(out) ** 2)
        )
        assert abs(power_ratio - 1.0) < 0.05
        assert cross_corr < 0.02

    def test_lowdin_preserves_power(self, xp):
        """Löwdin output power should equal input power."""
        _, r = self._make_imbalanced(xp)
        P_in = float(xp.mean(xp.abs(r) ** 2))
        out = correct_iq_imbalance(r, Lowdin())
        P_out = float(xp.mean(xp.abs(out) ** 2))
        assert abs(P_out - P_in) / P_in < 0.01

    def test_lowdin_identity_on_balanced_signal(self, xp, xpt):
        """Löwdin applied to a balanced signal should return it unchanged."""
        N = 8192
        rng = xp.random.RandomState(0)
        s = (rng.randn(N) + 1j * rng.randn(N)).astype(xp.complex64)
        out = correct_iq_imbalance(s, Lowdin())
        xpt.assert_allclose(xp.abs(out), xp.abs(s), atol=0.05)

    def test_lowdin_dtype_preserved(self, xp):
        """Löwdin output dtype should match input."""
        _, r = self._make_imbalanced(xp)
        out = correct_iq_imbalance(r, Lowdin())
        assert out.dtype == xp.complex64

    # --- Gram-Schmidt ---

    def test_gram_schmidt_restores_circularity(self, xp):
        """Gram-Schmidt compensation should drive κ to near zero."""
        _, r = self._make_imbalanced(xp)
        kappa_before = self._kappa(xp, r)
        out = correct_iq_imbalance(r, GramSchmidt())
        kappa_after = self._kappa(xp, out)
        assert kappa_after < 0.03
        assert kappa_after < kappa_before / 5

    def test_gram_schmidt_iq_orthogonality(self, xp):
        """After Gram-Schmidt, I and Q should be orthogonal."""
        _, r = self._make_imbalanced(xp)
        out = correct_iq_imbalance(r, GramSchmidt())
        Iquad, Qquad = out.real, out.imag
        cross_corr = float(xp.abs(xp.mean(Iquad * Qquad))) / float(
            xp.mean(xp.abs(out) ** 2)
        )
        assert cross_corr < 0.02

    def test_gram_schmidt_preserves_power(self, xp):
        """Gram-Schmidt output power should equal input power."""
        _, r = self._make_imbalanced(xp)
        P_in = float(xp.mean(xp.abs(r) ** 2))
        out = correct_iq_imbalance(r, GramSchmidt())
        P_out = float(xp.mean(xp.abs(out) ** 2))
        assert abs(P_out - P_in) / P_in < 0.01

    def test_gram_schmidt_dtype_preserved(self, xp):
        """Gram-Schmidt output dtype should match input."""
        _, r = self._make_imbalanced(xp)
        out = correct_iq_imbalance(r, GramSchmidt())
        assert out.dtype == xp.complex64


class TestSignalInputFrontend:
    """Signal-awareness for apply_iq_imbalance and its compensators."""

    def test_apply_iq_imbalance_signal_input(self, xp, xpt):
        """Signal input returns a Signal with the imbalanced samples."""
        rng = xp.random.RandomState(1)
        data = (rng.randn(512) + 1j * rng.randn(512)).astype(xp.complex64)
        sig = Signal(samples=data, sampling_rate=1.0, symbol_rate=1.0)

        out_sig = apply_iq_imbalance(
            sig, amplitude_imbalance_db=1.0, phase_imbalance_deg=3.0
        )
        out_arr = apply_iq_imbalance(
            data, amplitude_imbalance_db=1.0, phase_imbalance_deg=3.0
        )

        assert isinstance(out_sig, Signal)
        xpt.assert_allclose(out_sig.samples, out_arr)

    def test_compensate_lowdin_signal_input(self, xp, xpt):
        """Signal input returns a Signal with the compensated samples."""
        rng = xp.random.RandomState(2)
        data = (rng.randn(256) + 1j * rng.randn(256)).astype(xp.complex64)
        sig = Signal(samples=data, sampling_rate=1.0, symbol_rate=1.0)

        out_sig = correct_iq_imbalance(sig, Lowdin())
        out_arr = correct_iq_imbalance(data, Lowdin())

        assert isinstance(out_sig, Signal)
        xpt.assert_allclose(out_sig.samples, out_arr)

    def test_compensate_gram_schmidt_signal_input(self, xp, xpt):
        """Signal input returns a Signal with the compensated samples."""
        rng = xp.random.RandomState(5)
        data = (rng.randn(256) + 1j * rng.randn(256)).astype(xp.complex64)
        sig = Signal(samples=data, sampling_rate=1.0, symbol_rate=1.0)

        out_sig = correct_iq_imbalance(sig, GramSchmidt())
        out_arr = correct_iq_imbalance(data, GramSchmidt())

        assert isinstance(out_sig, Signal)
        xpt.assert_allclose(out_sig.samples, out_arr)

    def test_unknown_method_raises(self, xp):
        with pytest.raises(TypeError, match="Lowdin"):
            correct_iq_imbalance(xp.ones(8, dtype=xp.complex64), "lowdin")


class TestEstimateIQImbalance:
    """estimate_iq_imbalance measures the parameters of apply_iq_imbalance,
    and correct_iq_imbalance inverts them."""

    @staticmethod
    def _qpsk(xp, shape, seed=0):
        rng = xp.random.RandomState(seed)
        bits = rng.randint(0, 2, size=(2, *shape))
        return ((2 * bits[0] - 1) + 1j * (2 * bits[1] - 1)).astype(xp.complex64) / (
            2**0.5
        )

    @pytest.mark.parametrize("amp_db, phase_deg", [(0.5, 3.0), (-1.5, -7.0), (0, 0)])
    def test_recovers_parameters(self, xp, amp_db, phase_deg):
        s = self._qpsk(xp, (200_000,))
        r = apply_iq_imbalance(
            s, amplitude_imbalance_db=amp_db, phase_imbalance_deg=phase_deg
        )
        est = estimate_iq_imbalance(r)
        assert est.amplitude_imbalance_db.shape == ()
        assert float(est.amplitude_imbalance_db) == pytest.approx(amp_db, abs=0.05)
        assert float(est.phase_imbalance_deg) == pytest.approx(phase_deg, abs=0.5)

    def test_per_channel(self, xp):
        s = self._qpsk(xp, (2, 100_000))
        r = xp.stack(
            [
                apply_iq_imbalance(
                    s[0], amplitude_imbalance_db=1.0, phase_imbalance_deg=2.0
                ),
                apply_iq_imbalance(
                    s[1], amplitude_imbalance_db=-0.5, phase_imbalance_deg=-4.0
                ),
            ]
        )
        est = estimate_iq_imbalance(r)
        assert type(est.amplitude_imbalance_db) is xp.ndarray
        assert est.amplitude_imbalance_db.shape == (2,)
        assert xp.allclose(
            est.amplitude_imbalance_db, xp.asarray([1.0, -0.5]), atol=0.05
        )
        assert xp.allclose(est.phase_imbalance_deg, xp.asarray([2.0, -4.0]), atol=0.5)

    def test_insensitive_to_carrier_rotation(self, xp):
        """A rotating carrier before the imbalance keeps the signal circular."""
        s = self._qpsk(xp, (100_000,))
        n = xp.arange(s.size)
        s = (s * xp.exp(1j * 2 * xp.pi * 1e-3 * n)).astype(xp.complex64)
        r = apply_iq_imbalance(s, amplitude_imbalance_db=0.8, phase_imbalance_deg=5.0)
        est = estimate_iq_imbalance(r)
        assert float(est.amplitude_imbalance_db) == pytest.approx(0.8, abs=0.05)
        assert float(est.phase_imbalance_deg) == pytest.approx(5.0, abs=0.5)

    def test_correct_inverts_exactly(self, xp, xpt):
        """With the true parameters the correction is the exact inverse."""
        s = self._qpsk(xp, (2, 1024))
        r = apply_iq_imbalance(s, amplitude_imbalance_db=1.2, phase_imbalance_deg=6.0)
        how = IQImbalanceEstimate(1.2, 6.0)
        out = correct_iq_imbalance(r, how)
        assert out.dtype == xp.complex64
        xpt.assert_allclose(out, s, atol=1e-5)

    def test_correct_from_estimate(self, xp, xpt):
        s = self._qpsk(xp, (100_000,))
        r = apply_iq_imbalance(s, amplitude_imbalance_db=-1.0, phase_imbalance_deg=4.0)
        out = correct_iq_imbalance(r, estimate_iq_imbalance(r))
        assert float(xp.max(xp.abs(out - s))) < 0.02

    def test_correct_per_channel_size_mismatch_raises(self, xp):
        how = IQImbalanceEstimate(xp.zeros(3), xp.zeros(3))
        with pytest.raises(ValueError, match="3 channels for 2"):
            correct_iq_imbalance(xp.ones((2, 8), dtype=xp.complex64), how)
