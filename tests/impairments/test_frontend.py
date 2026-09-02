"""Tests for transceiver front-end IQ-imbalance application and compensation."""

from commkit.core import Signal
from commkit.impairments import (
    apply_iq_imbalance,
    compensate_iq_imbalance_gram_schmidt,
    compensate_iq_imbalance_lowdin,
)


class TestApplyIQImbalance:
    """Tests for apply_iq_imbalance."""

    def test_identity_zero_imbalance(self, backend_device, xp, xpt):
        """Zero imbalance (0 dB, 0 deg) should leave the signal unchanged."""
        N = 1024
        rng = xp.random.RandomState(42)
        samples = (rng.randn(N) + 1j * rng.randn(N)).astype(xp.complex64)

        out = apply_iq_imbalance(
            samples, amplitude_imbalance_db=0.0, phase_imbalance_deg=0.0
        )

        xpt.assert_allclose(out, samples, atol=1e-5)

    def test_causes_impropriety(self, backend_device, xp):
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

    def test_output_shape_siso(self, backend_device, xp):
        """Output shape should match SISO input."""
        samples = xp.ones(512, dtype=xp.complex64)
        out = apply_iq_imbalance(
            samples, amplitude_imbalance_db=1.0, phase_imbalance_deg=3.0
        )
        assert out.shape == (512,)

    def test_output_shape_mimo(self, backend_device, xp):
        """Output shape should match MIMO input."""
        samples = xp.ones((4, 512), dtype=xp.complex64)
        out = apply_iq_imbalance(
            samples, amplitude_imbalance_db=1.0, phase_imbalance_deg=3.0
        )
        assert out.shape == (4, 512)

    def test_output_dtype_preserved(self, backend_device, xp):
        """Output dtype should match input dtype."""
        samples = xp.ones(256, dtype=xp.complex64)
        out = apply_iq_imbalance(
            samples, amplitude_imbalance_db=1.0, phase_imbalance_deg=2.0
        )
        assert out.dtype == xp.complex64


class TestIQImbalanceCompensation:
    """Tests for compensate_iq_imbalance_lowdin and compensate_iq_imbalance_gram_schmidt."""

    # κ = |E[r²]| / E[|r|²]: zero for a circular signal, positive for improper
    def _kappa(self, xp, x):
        return float(xp.abs(xp.mean(x**2))) / float(xp.mean(xp.abs(x) ** 2))

    def _make_imbalanced(self, xp, N=8192, seed=42):
        rng = xp.random.RandomState(seed)
        s = (rng.randn(N) + 1j * rng.randn(N)).astype(xp.complex64)
        r = apply_iq_imbalance(s, amplitude_imbalance_db=2.0, phase_imbalance_deg=5.0)
        return s, r

    # --- Löwdin ---

    def test_lowdin_restores_circularity(self, backend_device, xp):
        """Löwdin compensation should drive κ to near zero."""
        _, r = self._make_imbalanced(xp)
        kappa_before = self._kappa(xp, r)
        out = compensate_iq_imbalance_lowdin(r)
        kappa_after = self._kappa(xp, out)
        assert kappa_after < 0.03
        assert kappa_after < kappa_before / 5

    def test_lowdin_iq_balance(self, backend_device, xp):
        """After Löwdin, I and Q should have equal power and be orthogonal."""
        _, r = self._make_imbalanced(xp)
        out = compensate_iq_imbalance_lowdin(r)
        Iquad, Qquad = out.real, out.imag
        power_ratio = float(xp.mean(Iquad**2)) / float(xp.mean(Qquad**2))
        cross_corr = float(xp.abs(xp.mean(Iquad * Qquad))) / float(
            xp.mean(xp.abs(out) ** 2)
        )
        assert abs(power_ratio - 1.0) < 0.05
        assert cross_corr < 0.02

    def test_lowdin_preserves_power(self, backend_device, xp):
        """Löwdin output power should equal input power."""
        _, r = self._make_imbalanced(xp)
        P_in = float(xp.mean(xp.abs(r) ** 2))
        out = compensate_iq_imbalance_lowdin(r)
        P_out = float(xp.mean(xp.abs(out) ** 2))
        assert abs(P_out - P_in) / P_in < 0.01

    def test_lowdin_identity_on_balanced_signal(self, backend_device, xp, xpt):
        """Löwdin applied to a balanced signal should return it unchanged."""
        N = 8192
        rng = xp.random.RandomState(0)
        s = (rng.randn(N) + 1j * rng.randn(N)).astype(xp.complex64)
        out = compensate_iq_imbalance_lowdin(s)
        xpt.assert_allclose(xp.abs(out), xp.abs(s), atol=0.05)

    def test_lowdin_siso_shape(self, backend_device, xp):
        """Löwdin: SISO (N,) input should return (N,)."""
        _, r = self._make_imbalanced(xp)
        out = compensate_iq_imbalance_lowdin(r)
        assert out.shape == r.shape

    def test_lowdin_mimo_shape(self, backend_device, xp):
        """Löwdin: MIMO (C, N) input should return (C, N)."""
        _, r = self._make_imbalanced(xp)
        r_mimo = xp.stack([r, r])  # (2, N)
        out = compensate_iq_imbalance_lowdin(r_mimo)
        assert out.shape == r_mimo.shape

    def test_lowdin_dtype_preserved(self, backend_device, xp):
        """Löwdin output dtype should match input."""
        _, r = self._make_imbalanced(xp)
        out = compensate_iq_imbalance_lowdin(r)
        assert out.dtype == xp.complex64

    # --- Gram-Schmidt ---

    def test_gram_schmidt_restores_circularity(self, backend_device, xp):
        """Gram-Schmidt compensation should drive κ to near zero."""
        _, r = self._make_imbalanced(xp)
        kappa_before = self._kappa(xp, r)
        out = compensate_iq_imbalance_gram_schmidt(r)
        kappa_after = self._kappa(xp, out)
        assert kappa_after < 0.03
        assert kappa_after < kappa_before / 5

    def test_gram_schmidt_iq_orthogonality(self, backend_device, xp):
        """After Gram-Schmidt, I and Q should be orthogonal."""
        _, r = self._make_imbalanced(xp)
        out = compensate_iq_imbalance_gram_schmidt(r)
        Iquad, Qquad = out.real, out.imag
        cross_corr = float(xp.abs(xp.mean(Iquad * Qquad))) / float(
            xp.mean(xp.abs(out) ** 2)
        )
        assert cross_corr < 0.02

    def test_gram_schmidt_preserves_power(self, backend_device, xp):
        """Gram-Schmidt output power should equal input power."""
        _, r = self._make_imbalanced(xp)
        P_in = float(xp.mean(xp.abs(r) ** 2))
        out = compensate_iq_imbalance_gram_schmidt(r)
        P_out = float(xp.mean(xp.abs(out) ** 2))
        assert abs(P_out - P_in) / P_in < 0.01

    def test_gram_schmidt_siso_shape(self, backend_device, xp):
        """Gram-Schmidt: SISO (N,) input should return (N,)."""
        _, r = self._make_imbalanced(xp)
        out = compensate_iq_imbalance_gram_schmidt(r)
        assert out.shape == r.shape

    def test_gram_schmidt_mimo_shape(self, backend_device, xp):
        """Gram-Schmidt: MIMO (C, N) input should return (C, N)."""
        _, r = self._make_imbalanced(xp)
        r_mimo = xp.stack([r, r])  # (2, N)
        out = compensate_iq_imbalance_gram_schmidt(r_mimo)
        assert out.shape == r_mimo.shape

    def test_gram_schmidt_dtype_preserved(self, backend_device, xp):
        """Gram-Schmidt output dtype should match input."""
        _, r = self._make_imbalanced(xp)
        out = compensate_iq_imbalance_gram_schmidt(r)
        assert out.dtype == xp.complex64


class TestSignalInputFrontend:
    """Signal-awareness for apply_iq_imbalance and its compensators."""

    def test_apply_iq_imbalance_signal_input(self, backend_device, xp, xpt):
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

    def test_compensate_lowdin_signal_input(self, backend_device, xp, xpt):
        """Signal input returns a Signal with the compensated samples."""
        rng = xp.random.RandomState(2)
        data = (rng.randn(256) + 1j * rng.randn(256)).astype(xp.complex64)
        sig = Signal(samples=data, sampling_rate=1.0, symbol_rate=1.0)

        out_sig = compensate_iq_imbalance_lowdin(sig)
        out_arr = compensate_iq_imbalance_lowdin(data)

        assert isinstance(out_sig, Signal)
        xpt.assert_allclose(out_sig.samples, out_arr)

    def test_compensate_gram_schmidt_signal_input(self, backend_device, xp, xpt):
        """Signal input returns a Signal with the compensated samples."""
        rng = xp.random.RandomState(5)
        data = (rng.randn(256) + 1j * rng.randn(256)).astype(xp.complex64)
        sig = Signal(samples=data, sampling_rate=1.0, symbol_rate=1.0)

        out_sig = compensate_iq_imbalance_gram_schmidt(sig)
        out_arr = compensate_iq_imbalance_gram_schmidt(data)

        assert isinstance(out_sig, Signal)
        xpt.assert_allclose(out_sig.samples, out_arr)
