"""Tests for additive-noise impairments (AWGN)."""

from commkit.core import Signal
from commkit.impairments import apply_awgn


class TestAddAWGN:
    """Tests for apply_awgn."""

    def test_awgn_adds_noise(self, backend_device, xp):
        """Output should differ from input."""
        samples = xp.ones(1000, dtype=xp.complex64)
        noisy = apply_awgn(samples, esn0_db=10, sps=1)

        diff = float(xp.max(xp.abs(noisy - samples)))
        assert diff > 0.001

    def test_awgn_preserves_shape(self, backend_device, xp):
        """Output shape should match input."""
        samples = xp.ones((2, 500), dtype=xp.complex64)
        noisy = apply_awgn(samples, esn0_db=20, sps=2)

        assert noisy.shape == samples.shape

    def test_awgn_noise_power(self, backend_device, xp):
        """Noise power should match 10^(-SNR/10) for a unit-power signal."""
        data = xp.ones(1000, dtype=complex)
        noisy = apply_awgn(data, esn0_db=10.0, sps=1)
        noise = noisy - data
        measured = float(xp.mean(xp.abs(noise) ** 2))
        # Signal power = 1, SNR = 10 dB -> noise power = 10^(-1) = 0.1
        assert 0.08 < measured < 0.12, (
            f"Noise power {measured:.4f} outside expected range"
        )

    def test_awgn_signal_power_override(self, backend_device, xp):
        """Explicit signal_power sets an absolute noise level (dark capture)."""
        dark = xp.zeros(20000, dtype=xp.complex128)
        noisy = apply_awgn(dark, esn0_db=10.0, sps=1, signal_power=1.0, seed=3)
        # Without the override the noise power would be 0; with it, 0.1.
        measured = float(xp.mean(xp.abs(noisy) ** 2))
        assert 0.08 < measured < 0.12, (
            f"Noise power {measured:.4f} outside expected range"
        )

    def test_awgn_real_data(self, backend_device, xp):
        """apply_awgn should handle real-valued input and return a real output."""
        data = xp.ones(1000, dtype="float32")
        noisy = apply_awgn(data, esn0_db=10, sps=1)

        assert xp.isrealobj(noisy)
        assert not xp.allclose(noisy, data)

    def test_awgn_low_snr(self, backend_device, xp):
        """Extremely low SNR should produce very high noise power."""
        data = xp.ones(100, dtype=complex)
        noisy = apply_awgn(data, esn0_db=-300, sps=1)
        measured_power = float(xp.mean(xp.abs(noisy) ** 2))
        assert measured_power > 1e15

    def test_awgn_signal_input_returns_signal(self, backend_device, xp, xpt):
        """Signal input: sps is taken from the signal and a Signal is returned."""
        data = xp.ones(1000, dtype=xp.complex64)
        sig = Signal(samples=data, sampling_rate=4e9, symbol_rate=1e9)  # sps=4

        noisy_sig = apply_awgn(sig, esn0_db=10, seed=1)
        noisy_arr = apply_awgn(data, esn0_db=10, sps=4, seed=1)

        assert isinstance(noisy_sig, Signal)
        xpt.assert_allclose(noisy_sig.samples, noisy_arr)
        xpt.assert_allclose(sig.samples, data)  # original Signal untouched
