"""Tests for additive-noise impairments (AWGN)."""

import numpy as np
import pytest

from commkit.core import Signal
from commkit.impairments import apply_awgn


class TestAddAWGN:
    """Tests for apply_awgn."""

    def test_awgn_adds_noise(self, xp):
        """Output should differ from input."""
        samples = xp.ones(1000, dtype=xp.complex64)
        noisy = apply_awgn(samples, esn0_db=10, sps=1)

        diff = float(xp.max(xp.abs(noisy - samples)))
        assert diff > 0.001

    def test_awgn_preserves_shape(self, xp):
        """Output shape should match input."""
        samples = xp.ones((2, 500), dtype=xp.complex64)
        noisy = apply_awgn(samples, esn0_db=20, sps=2)

        assert noisy.shape == samples.shape

    def test_awgn_noise_power(self, xp):
        """Noise power should match 10^(-SNR/10) for a unit-power signal."""
        data = xp.ones(1000, dtype=complex)
        noisy = apply_awgn(data, esn0_db=10.0, sps=1)
        noise = noisy - data
        measured = float(xp.mean(xp.abs(noise) ** 2))
        # Signal power = 1, SNR = 10 dB -> noise power = 10^(-1) = 0.1
        assert 0.08 < measured < 0.12, (
            f"Noise power {measured:.4f} outside expected range"
        )

    def test_awgn_signal_power_override(self, xp):
        """Explicit signal_power sets an absolute noise level (dark capture)."""
        dark = xp.zeros(20000, dtype=xp.complex128)
        noisy = apply_awgn(dark, esn0_db=10.0, sps=1, signal_power=1.0, rng=3)
        # Without the override the noise power would be 0; with it, 0.1.
        measured = float(xp.mean(xp.abs(noisy) ** 2))
        assert 0.08 < measured < 0.12, (
            f"Noise power {measured:.4f} outside expected range"
        )

    def test_awgn_real_data(self, xp):
        """apply_awgn should handle real-valued input and return a real output."""
        data = xp.ones(1000, dtype="float32")
        noisy = apply_awgn(data, esn0_db=10, sps=1)

        assert xp.isrealobj(noisy)
        assert not xp.allclose(noisy, data)

    def test_awgn_low_snr(self, xp):
        """Extremely low SNR should produce very high noise power."""
        data = xp.ones(100, dtype=complex)
        noisy = apply_awgn(data, esn0_db=-300, sps=1)
        measured_power = float(xp.mean(xp.abs(noisy) ** 2))
        assert measured_power > 1e15

    def test_awgn_signal_input_returns_signal(self, xp, xpt):
        """Signal input: sps is taken from the signal and a Signal is returned."""
        data = xp.ones(1000, dtype=xp.complex64)
        sig = Signal(samples=data, sampling_rate=4e9, symbol_rate=1e9)  # sps=4

        noisy_sig = apply_awgn(sig, esn0_db=10, rng=1)
        noisy_arr = apply_awgn(data, esn0_db=10, sps=4, rng=1)

        assert isinstance(noisy_sig, Signal)
        xpt.assert_allclose(noisy_sig.samples, noisy_arr)
        xpt.assert_allclose(sig.samples, data)  # original Signal untouched


class TestAWGNRandomness:
    """rng: reproducible per device, independent draws, sample precision."""

    def test_same_rng_reproduces_and_generator_advances(self, xp, xpt):
        x = xp.ones(1000, dtype=xp.complex64)
        a = apply_awgn(x, sps=1, esn0_db=10, rng=7)
        xpt.assert_array_equal(a, apply_awgn(x, sps=1, esn0_db=10, rng=7))
        gen = np.random.default_rng(7)
        first = apply_awgn(x, sps=1, esn0_db=10, rng=gen)
        second = apply_awgn(x, sps=1, esn0_db=10, rng=gen)
        assert not bool(xp.allclose(first, second))

    @pytest.mark.parametrize("dtype", ["complex64", "complex128", "float32"])
    def test_noise_keeps_precision_and_variance(self, xp, dtype):
        x = xp.zeros(200_000, dtype=dtype)
        y = apply_awgn(x, sps=2, esn0_db=3.0, signal_power=1.0, rng=1)
        assert y.dtype == x.dtype
        expected = 2 / 10**0.3  # sps * P_signal / (Es/N0)
        assert float(xp.mean(xp.abs(y) ** 2)) == pytest.approx(expected, rel=0.02)
        if y.dtype.kind == "c":
            assert float(xp.mean(y.real**2)) == pytest.approx(expected / 2, rel=0.02)


class TestOSNR:
    """apply_awgn(osnr_db=): ASE in 12.5 GHz over both polarizations."""

    def test_textbook_relation(self, xp):
        """Dual-pol at 32 GBd: Es/N0 = OSNR · 2 B_ref / (p R_s)."""
        n, rs, sps = 1 << 18, 32e9, 2
        x = xp.ones((2, n), dtype=xp.complex64)
        sig = Signal(
            samples=x,
            sampling_rate=sps * rs,
            symbol_rate=rs,
            domain="OPT",
            center_frequency=193.4e12,
        )
        noisy = apply_awgn(sig, osnr_db=20.0, rng=3)
        var = float(xp.mean(xp.abs(noisy.samples - x) ** 2))
        esn0 = sps * 1.0 / var
        assert esn0 == pytest.approx(100 * 2 * 12.5e9 / (2 * rs), rel=0.01)
        # One polarization: the ASE of both counts against half the power.
        single = apply_awgn(sig.replace(samples=x[0]), osnr_db=20.0, rng=3)
        var1 = float(xp.mean(xp.abs(single.samples - x[0]) ** 2))
        assert var1 == pytest.approx(var / 2, rel=0.01)

    def test_needs_one_ratio_and_an_optical_signal(self, xp):
        x = xp.ones(64, dtype=xp.complex64)
        sig = Signal(samples=x, sampling_rate=2.0, symbol_rate=1.0)
        with pytest.raises(ValueError, match="one of esn0_db and osnr_db"):
            apply_awgn(sig)
        with pytest.raises(ValueError, match="one of esn0_db and osnr_db"):
            apply_awgn(sig, esn0_db=10.0, osnr_db=10.0)
        for bad in (x, sig, sig.replace(domain="OPT", samples=x.real)):
            with pytest.raises(ValueError, match="optical Signal"):
                apply_awgn(bad, osnr_db=10.0)
        opt = sig.replace(domain="OPT", center_frequency=193.4e12)
        for bad in (0.0, -12.5e9, float("nan")):
            with pytest.raises(ValueError, match="reference_bandwidth"):
                apply_awgn(opt, osnr_db=10.0, reference_bandwidth=bad)

    @pytest.mark.parametrize("sps", [1, 2, 4])
    def test_independent_of_oversampling(self, xp, sps):
        """The in-band noise, and so Es/N0, does not depend on sps; a wider
        reference bandwidth at the same OSNR means a
        proportionally lower ASE density."""
        n, rs = 1 << 17, 32e9
        x = xp.ones((2, n), dtype=xp.complex64)
        sig = Signal(
            samples=x,
            sampling_rate=sps * rs,
            symbol_rate=rs,
            domain="OPT",
            center_frequency=193.4e12,
        )
        var = float(
            xp.mean(xp.abs(apply_awgn(sig, osnr_db=15.0, rng=1).samples - x) ** 2)
        )
        esn0 = sps / var
        expected = 10**1.5 * 2 * 12.5e9 / (2 * rs)
        assert esn0 == pytest.approx(expected, rel=0.02)
        wide = apply_awgn(sig, osnr_db=15.0, reference_bandwidth=25e9, rng=1)
        var_wide = float(xp.mean(xp.abs(wide.samples - x) ** 2))
        assert var_wide == pytest.approx(var / 2, rel=0.02)
