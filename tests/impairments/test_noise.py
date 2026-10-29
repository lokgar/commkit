"""Tests for additive noise: apply_awgn and apply_ase."""

import numpy as np
import pytest

from commkit import RRC, Constellation, generate
from commkit.core import Preamble, Signal, SingleCarrierFrame
from commkit.impairments import apply_ase, apply_awgn


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
        expected = 2 / 10**0.3
        y = apply_awgn(x, noise_var=expected, rng=1)
        assert y.dtype == x.dtype
        assert float(xp.mean(xp.abs(y) ** 2)) == pytest.approx(expected, rel=0.02)
        if y.dtype.kind == "c":
            assert float(xp.mean(y.real**2)) == pytest.approx(expected / 2, rel=0.02)


def _noise_var(xp, noisy, x):
    """Measured noise variance per channel."""
    return xp.mean(xp.abs(_samples(noisy) - _samples(x)) ** 2, axis=-1)


def _samples(x):
    return x.samples if isinstance(x, Signal) else x


class TestNoiseVar:
    """noise_var: an absolute variance, whatever the signal."""

    def test_absolute_level_on_a_silent_record(self, xp):
        dark = xp.zeros(1 << 16, dtype=xp.complex64)
        assert float(_noise_var(xp, apply_awgn(dark, noise_var=0.1, rng=3), dark)) == (
            pytest.approx(0.1, rel=0.03)
        )

    def test_independent_of_the_signal_scale(self, xp):
        x = xp.ones(1 << 16, dtype=xp.complex64)
        a = _noise_var(xp, apply_awgn(x, noise_var=0.01, rng=1), x)
        b = _noise_var(xp, apply_awgn(10 * x, noise_var=0.01, rng=1), 10 * x)
        assert float(a) == pytest.approx(float(b), rel=1e-4)

    def test_per_channel(self, xp):
        x = xp.zeros((2, 1 << 16), dtype=xp.complex64)
        var = _noise_var(xp, apply_awgn(x, noise_var=[0.2, 0.05], rng=1), x)
        assert xp.allclose(var, xp.asarray([0.2, 0.05]), rtol=0.03)
        with pytest.raises(ValueError, match="one entry per channel"):
            apply_awgn(x, noise_var=[1.0, 1.0, 1.0])

    def test_exactly_one_specification(self, xp):
        x = xp.ones(64, dtype=xp.complex64)
        with pytest.raises(ValueError, match="one of esn0_db and noise_var"):
            apply_awgn(x, sps=1)
        with pytest.raises(ValueError, match="one of esn0_db and noise_var"):
            apply_awgn(x, sps=1, esn0_db=10.0, noise_var=0.1)
        for removed in ({"signal_power": 1.0}, {"osnr_db": 20.0}):
            with pytest.raises(TypeError):
                apply_awgn(x, sps=1, esn0_db=10.0, **removed)


class TestSymbolEnergy:
    """esn0_db measures Es as energy per symbol, so silence changes nothing."""

    def test_es_scales_with_the_signal(self, xp):
        x = xp.ones(1 << 16, dtype=xp.complex64)
        a = _noise_var(xp, apply_awgn(x, esn0_db=10, sps=1, rng=1), x)
        b = _noise_var(xp, apply_awgn(3 * x, esn0_db=10, sps=1, rng=1), 3 * x)
        assert float(b) == pytest.approx(9 * float(a), rel=1e-4)

    def test_zero_padding_keeps_the_noise_level(self, xp):
        tx = generate(
            Constellation.qam(16),
            num_symbols=1 << 14,
            symbol_rate=1e9,
            sps=2,
            pulse=RRC(rolloff=0.1),
            rng=1,
            device="gpu" if xp.__name__ == "cupy" else "cpu",
        )
        pad = xp.zeros(3000, dtype=tx.samples.dtype)
        padded = tx.replace(samples=xp.concatenate([pad, tx.samples, pad]))
        plain = float(_noise_var(xp, apply_awgn(tx, esn0_db=15, rng=1), tx))
        both = float(_noise_var(xp, apply_awgn(padded, esn0_db=15, rng=1), padded))
        assert both == pytest.approx(plain, rel=0.03)
        # Es of unit-energy symbols is 1: σ² = 1 / Es/N0.
        assert plain == pytest.approx(10**-1.5, rel=0.03)
        # An array has no symbol count: the padding counts as signal.
        arr = float(
            _noise_var(
                xp, apply_awgn(padded.samples, esn0_db=15, sps=2, rng=1), padded.samples
            )
        )
        n = tx.samples.shape[-1]
        assert arr == pytest.approx(plain * n / (n + 6000), rel=0.03)

    def test_frame_length_and_zero_guard(self, xp):
        frame = SingleCarrierFrame(
            payload_num_symbols=4096,
            payload_constellation=Constellation.qam(16),
            payload_rng=1,
            preamble=Preamble(sequence_type="zc", num_symbols=127),
            guard_type="zero",
            guard_num_symbols=500,
        )
        device = "gpu" if xp.__name__ == "cupy" else "cpu"
        tx = frame.to_signal(symbol_rate=1e9, device=device)
        pad = xp.zeros(2000, dtype=tx.samples.dtype)
        padded = tx.replace(samples=xp.concatenate([pad, tx.samples]))
        var = float(_noise_var(xp, apply_awgn(padded, esn0_db=10, rng=2), padded))
        active = tx.samples[: frame.num_symbols - 500]
        es = float(xp.mean(xp.abs(active) ** 2))
        assert var == pytest.approx(es / 10, rel=0.03)

    def test_shorter_record_counts_its_own_length(self, xp):
        x = xp.ones(1 << 15, dtype=xp.complex64)
        sig = Signal(samples=x, sampling_rate=1e9, symbol_rate=1e9)
        ref = generate(
            Constellation.qam(4), num_symbols=1 << 16, symbol_rate=1e9, rng=1
        )
        sig = sig.replace(reference=ref.reference.to(sig.backend))
        var = float(_noise_var(xp, apply_awgn(sig, esn0_db=10, rng=1), sig))
        assert var == pytest.approx(0.1, rel=0.03)


class TestPerChannel:
    """per_channel: each channel's own Es, or one floor from the mean."""

    def test_each_channel_gets_its_esn0(self, xp):
        x = xp.ones((2, 1 << 16), dtype=xp.complex64)
        x[1] *= 2.0  # 6 dB stronger
        var = _noise_var(xp, apply_awgn(x, esn0_db=10.0, sps=1, rng=1), x)
        assert xp.allclose(var, xp.asarray([0.1, 0.4]), rtol=0.03)

    def test_shared_floor_from_the_mean(self, xp):
        x = xp.ones((2, 1 << 16), dtype=xp.complex64)
        x[1] *= 2.0
        noisy = apply_awgn(x, esn0_db=10.0, sps=1, per_channel=False, rng=1)
        var = _noise_var(xp, noisy, x)
        assert xp.allclose(var, xp.asarray([0.25, 0.25]), rtol=0.03)


class TestASE:
    """apply_ase: ASE at an OSNR over the guided modes."""

    RS = 32e9

    def _field(self, xp, num_channels, sps=2):
        x = xp.ones((num_channels, 1 << 16), dtype=xp.complex64)
        return Signal(
            samples=x,
            sampling_rate=sps * self.RS,
            symbol_rate=self.RS,
            domain="OPT",
            center_frequency=193.4e12,
        )

    def _esn0(self, xp, sig, osnr_db=20.0, **kwargs):
        noisy = apply_ase(sig, osnr_db=osnr_db, rng=2, **kwargs)
        return sig.sps / float(xp.mean(_noise_var(xp, noisy, sig)))

    @pytest.mark.parametrize("sps", [1, 2, 4])
    def test_textbook_relation_at_any_oversampling(self, xp, sps):
        """Dual-pol: Es/N0 = OSNR · 2 B_ref / (2 R_s)."""
        sig = self._field(xp, 2, sps=sps)
        assert self._esn0(xp, sig, osnr_db=15.0) == pytest.approx(
            10**1.5 * 12.5e9 / self.RS, rel=0.02
        )
        wide = self._esn0(xp, sig, osnr_db=15.0, reference_bandwidth=25e9)
        assert wide == pytest.approx(2 * 10**1.5 * 12.5e9 / self.RS, rel=0.02)

    def test_single_polarization_counts_both(self, xp):
        """One polarization at the same OSNR: half the ASE, all the power."""
        single = self._esn0(xp, self._field(xp, 1))
        dual = self._esn0(xp, self._field(xp, 2))
        assert single == pytest.approx(2 * dual, rel=0.02)

    def test_mode_division_default_counts_every_channel(self, xp):
        """Six channels in a 3-mode fiber: the dual-polarization Es/N0."""
        assert self._esn0(xp, self._field(xp, 6)) == pytest.approx(
            self._esn0(xp, self._field(xp, 2)), rel=0.02
        )

    def test_unexcited_modes_still_carry_ase(self, xp):
        sig = self._field(xp, 4)
        assert self._esn0(xp, sig, noise_modes=6) == pytest.approx(
            1.5 * self._esn0(xp, sig), rel=0.02
        )

    def test_unequal_powers_share_one_ase_level(self, xp):
        sig = self._field(xp, 2)
        x = sig.samples.copy()
        x[1] *= 2.0
        sig = sig.replace(samples=x)
        var = _noise_var(xp, apply_ase(sig, osnr_db=20.0, rng=2), sig)
        assert float(var[1]) == pytest.approx(float(var[0]), rel=0.03)
        # Total power 1 + 4 = 5 over two modes: 5 · f_s / (2 B_ref OSNR).
        expected = 5 * 2 * self.RS / (2 * 12.5e9 * 100)
        assert float(var[0]) == pytest.approx(expected, rel=0.03)

    def test_silence_does_not_lower_the_power(self, xp):
        sig = self._field(xp, 2)
        sig = sig.replace(
            reference=generate(
                Constellation.qam(4),
                num_symbols=1 << 15,
                symbol_rate=self.RS,
                num_channels=2,
                rng=1,
            ).reference.to(sig.backend)
        )
        pad = xp.zeros((2, 4000), dtype=xp.complex64)
        padded = sig.replace(samples=xp.concatenate([pad, sig.samples], axis=-1))
        assert self._esn0(xp, padded) == pytest.approx(self._esn0(xp, sig), rel=0.03)

    def test_validation(self, xp):
        x = xp.ones(64, dtype=xp.complex64)
        sig = Signal(samples=x, sampling_rate=2.0, symbol_rate=1.0)
        for bad in (sig, sig.replace(domain="OPT", samples=x.real)):
            with pytest.raises(ValueError, match="optical Signal"):
                apply_ase(bad, osnr_db=10.0)
        opt = self._field(xp, 4)
        for bad in (0.0, -12.5e9, float("nan")):
            with pytest.raises(ValueError, match="reference_bandwidth"):
                apply_ase(opt, osnr_db=10.0, reference_bandwidth=bad)
        for bad in (3, 0, 4.0, True):
            with pytest.raises(ValueError, match="noise_modes"):
                apply_ase(opt, osnr_db=20.0, noise_modes=bad)
