"""Tests for additive noise: apply_awgn and apply_ase."""

import numpy as np
import pytest

from commkit import RRC, Constellation, generate
from commkit.core import Preamble, Signal, SingleCarrierFrame
from commkit.impairments import (
    apply_ase,
    apply_awgn,
    apply_coherent_detection,
    apply_photodetection,
)
from tests.common.conversions import to_numpy


class TestAddAWGN:
    """Tests for apply_awgn."""

    def test_awgn_adds_noise(self, xp):
        """Output should differ from input."""
        samples = xp.ones(1000, dtype=xp.complex64)
        noisy = apply_awgn(samples, esn0_db=10, sps=1)

        diff = float(xp.max(xp.abs(noisy - samples)))
        assert diff > 0.001

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
        message = "exactly one of esn0_db, snr_db and noise_var"
        with pytest.raises(ValueError, match=message):
            apply_awgn(x, sps=1)
        with pytest.raises(ValueError, match=message):
            apply_awgn(x, sps=1, esn0_db=10.0, noise_var=0.1)
        with pytest.raises(ValueError, match=message):
            apply_awgn(x, esn0_db=10.0, snr_db=10.0)
        for removed in ({"signal_power": 1.0}, {"osnr_db": 20.0}):
            with pytest.raises(TypeError):
                apply_awgn(x, sps=1, esn0_db=10.0, **removed)


class TestRealAndComplex:
    """Es/N0 describes the channel: real samples get the in-phase half of
    the noise, so a constellation's error rates at an Es/N0 do not depend
    on the dtype.  SNR is a power ratio in the samples."""

    def test_esn0_real_samples_get_half_the_variance(self, xp):
        """Unit-energy samples at 10 dB: N0 = 0.1, so 0.1 per complex sample
        and 0.05 per real one (2e5 samples: 0.3 % standard error)."""
        n = 200_000
        for dtype, expected in ((xp.complex64, 0.1), (xp.float32, 0.05)):
            x = xp.ones(n, dtype=dtype)
            var = float(_noise_var(xp, apply_awgn(x, esn0_db=10.0, sps=1, rng=2), x))
            assert var == pytest.approx(expected, rel=0.02), dtype

    def test_snr_is_the_power_ratio_in_the_samples(self, xp):
        """snr_db sets mean(|x|^2) / E|n|^2 for real and complex samples alike,
        per channel, and does not need sps."""
        rng = np.random.default_rng(3)
        for dtype in (np.complex64, np.float32):
            x = rng.standard_normal((2, 100_000)) * np.array([[1.0], [3.0]])
            if dtype is np.complex64:
                x = x + 1j * rng.standard_normal(x.shape)
            x = xp.asarray(x.astype(dtype))
            power = xp.mean(xp.abs(x) ** 2, axis=-1)
            var = _noise_var(xp, apply_awgn(x, snr_db=20.0, rng=4), x)
            assert xp.allclose(power / var, 100.0, rtol=0.02), dtype

    def test_bpsk_ber_does_not_depend_on_the_dtype(self, xp):
        """2-PAM at Es/N0 = 4 dB: BER = Q(sqrt(2 Es/N0)) = 0.0125, whether
        the symbols are stored real or complex (4 sigma of 4e5 bits)."""
        from tests.common.scoring import binomial_margin
        from tests.common.theory import bpsk_ber

        rng = np.random.default_rng(5)
        bits = rng.integers(0, 2, 400_000)
        s = (1.0 - 2.0 * bits).astype(np.float32)
        p = bpsk_ber(10**0.4)
        for x in (s, s.astype(np.complex64)):
            noisy = apply_awgn(xp.asarray(x), esn0_db=4.0, sps=1, rng=6)
            ber = float(xp.mean((noisy.real < 0) != xp.asarray(bits.astype(bool))))
            assert abs(ber - p) < binomial_margin(p, bits.size), (x.dtype, ber, p)


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


class TestCoherentDetection:
    """Noise of a phase-diversity coherent receiver at absolute powers."""

    Q = 1.602176634e-19
    H = 6.62607015e-34
    NU = 299792458.0 / 1550e-9
    RS = 32e9

    def _rx(self, xp, *, num_channels=1, num_symbols=2**15, rng=1):
        return generate(
            Constellation.qam(16),
            num_symbols=num_symbols,
            symbol_rate=self.RS,
            sps=2,
            pulse=RRC(rolloff=0.1),
            num_channels=num_channels,
            rng=rng,
            device="gpu" if xp is not np else "cpu",
        )

    def _snr_db(self, sig):
        from commkit.filtering import matched_filter
        from commkit.metrics import snr
        from commkit.multirate import decimate_to_symbol_rate

        return snr(decimate_to_symbol_rate(matched_filter(sig)))

    def test_shot_noise_limit(self, xp):
        """A strong LO, no thermal noise and a responsivity of q/(hν)
        (quantum efficiency 1): Es/N0 = n_p P_LO / (P_s + P_LO), n_p = P_s T /
        (hν) photons per symbol, counted independently of the noise formula
        (each quadrature of a phase-diversity receiver gets half the photons'
        information).  At -45 dBm and 32 GBd n_p = 24.6 (13.9 dB).  The SNR
        of 2^15 symbols has a standard error of 0.04 dB; bound 0.15 dB."""
        p_s = 1e-3 * 10 ** (-45 / 10)
        out = apply_coherent_detection(
            self._rx(xp),
            received_power_dbm=-45.0,
            lo_power_dbm=13.0,
            responsivity=self.Q / (self.H * self.NU),
            rng=2,
        )
        n_p = p_s / self.RS / (self.H * self.NU)
        p_lo = 1e-3 * 10**1.3
        expected = 10 * np.log10(n_p * p_lo / (p_s + p_lo))
        assert abs(self._snr_db(out) - expected) < 0.15

    def test_thermal_noise_yields_to_the_lo(self, xp):
        """With 20 pA/sqrt(Hz) thermal noise at -30 dBm received: at a weak
        LO the thermal term dominates and the SNR rises 1 dB per dB of LO
        (measured slope from -20 to -10 dBm); at a strong LO it approaches
        the shot limit R P_c T / q within 0.3 dB."""
        sig = self._rx(xp)
        kw = {"received_power_dbm": -30.0, "responsivity": 1.0}
        snr = {}
        for lo in (-20.0, -10.0, 20.0):
            out = apply_coherent_detection(
                sig, lo_power_dbm=lo, thermal_noise_density=20e-12, rng=3, **kw
            )
            snr[lo] = self._snr_db(out)
        assert abs((snr[-10.0] - snr[-20.0]) / 10.0 - 1.0) < 0.05
        shot = 10 * np.log10(1e-6 / self.RS / self.Q)
        assert abs(snr[20.0] - shot) < 0.3

    def test_ber_against_received_power(self, xp):
        """QPSK BER against the AWGN closed form at the Es/N0 the powers
        give: within 5 binomial standard deviations over 2^16 symbols."""
        from commkit.metrics import awgn_ber, ber

        qpsk = generate(
            Constellation.psk(4),
            num_symbols=2**16,
            symbol_rate=self.RS,
            rng=4,
            device="gpu" if xp is not np else "cpu",
        )
        for p_dbm in (-40.0, -37.0):
            out = apply_coherent_detection(
                qpsk,
                received_power_dbm=p_dbm,
                lo_power_dbm=10.0,
                thermal_noise_density=10e-12,
                rng=5,
            )
            p_c, p_lo = 1e-3 * 10 ** (p_dbm / 10), 1e-2
            snr = p_c * p_lo / (self.RS * (self.Q * (p_c + p_lo) + 1e-22))
            p = awgn_ber(Constellation.psk(4), esn0_db=10 * np.log10(snr))
            n = 2 * 2**16
            assert abs(ber(out) - p) < 5 * np.sqrt(p / n) + 1e-5

    def test_channels_share_the_received_power(self, xp):
        """Two polarizations at equal power each get half of it: their SNR
        is that of one channel at 3 dB less received power."""
        both = apply_coherent_detection(
            self._rx(xp, num_channels=2),
            received_power_dbm=-27.0,
            lo_power_dbm=0.0,
            thermal_noise_density=20e-12,
            rng=6,
        )
        one = apply_coherent_detection(
            self._rx(xp),
            received_power_dbm=-30.0,
            lo_power_dbm=0.0,
            thermal_noise_density=20e-12,
            rng=7,
        )
        assert np.all(np.abs(self._snr_db(both) - self._snr_db(one)) < 0.15)

    def test_real_input_becomes_complex(self, xp):
        x = xp.ones(256, dtype=xp.float32)
        y = apply_coherent_detection(
            x, received_power_dbm=-20.0, lo_power_dbm=10.0, sampling_rate=64e9, rng=1
        )
        assert y.dtype == xp.complex64

    def test_validation(self, xp):
        x = xp.ones(64, dtype=xp.complex64)
        kw = {"received_power_dbm": -20.0, "lo_power_dbm": 10.0}
        with pytest.raises(ValueError, match="sampling_rate"):
            apply_coherent_detection(x, **kw)
        with pytest.raises(ValueError, match="responsivity"):
            apply_coherent_detection(x, responsivity=0.0, sampling_rate=1e9, **kw)
        with pytest.raises(ValueError, match="thermal"):
            apply_coherent_detection(
                x, thermal_noise_density=-1.0, sampling_rate=1e9, **kw
            )
        with pytest.raises(ValueError, match="zeros"):
            apply_coherent_detection(
                xp.zeros(64, xp.complex64), sampling_rate=1e9, **kw
            )


class TestDetectionImbalance:
    """Common-mode leak of a coherent receiver: finite CMRR, LO RIN and
    single-ended detection."""

    Q = TestCoherentDetection.Q
    RS = TestCoherentDetection.RS
    FS = 2 * TestCoherentDetection.RS

    def _rx(self, xp, rng=1):
        return TestCoherentDetection()._rx(xp, rng=rng)

    def _snr_db(self, sig):
        return TestCoherentDetection()._snr_db(sig)

    def test_rin_through_finite_cmrr(self, xp):
        """LO RIN of -140 dB/Hz through a 20 dB CMRR, at -20 dBm received
        and 15 dBm LO: the leak κ sqrt(P_LO/P_c) r(t), κ = δ/2,
        δ = 10^(-1), adds white noise of 2 κ² (P_LO/P_c) RIN f_s/2 to the
        shot noise f_s q (P_c + P_LO)/(R P_c P_LO), and Es/N0 = 2 / (sum)
        at 2 samples per symbol (25.95 dB predicted).  Perfect balance
        rejects the RIN: the shot limit, 32.9 dB.  Both with the clean record's
        44.5 dB floor added.  Bound 0.15 dB."""
        p_c, p_lo, rin = 1e-5, 10**1.5 * 1e-3, 1e-14
        kw = {"received_power_dbm": -20.0, "lo_power_dbm": 15.0, "lo_rin_db_hz": -140.0}
        shot = self.FS * self.Q * (p_c + p_lo) / (p_c * p_lo)
        leak = 2 * 0.05**2 * (p_lo / p_c) * rin * self.FS / 2
        sig = self._rx(xp)
        # The matched filter and slicer of the clean record give a 44.5 dB
        # floor (the pulse's truncation), which adds to the noise.
        floor = 10 ** (-self._snr_db(sig) / 10)

        def expected(white):
            return -10 * np.log10(white / 2 + floor)

        with_rin = self._snr_db(
            apply_coherent_detection(sig, cmrr_db=20.0, rng=2, **kw)
        )
        assert abs(with_rin - expected(shot + leak)) < 0.15
        rejected = self._snr_db(apply_coherent_detection(sig, rng=2, **kw))
        assert abs(rejected - expected(shot)) < 0.15

    def test_signal_signal_leak_coefficient(self, xp):
        """The deterministic leak of |s|²: regressing the added part of both
        quadratures on (|x̃|² - 1) sqrt(P) gives κ sqrt(P_c/P_LO) = 0.0281
        (20 dB CMRR, -10 dBm received, -5 dBm LO) within 3 %."""
        sig = self._rx(xp)
        out = apply_coherent_detection(
            sig, received_power_dbm=-10.0, lo_power_dbm=-5.0, cmrr_db=20.0, rng=3
        )
        x = np.asarray(to_numpy(sig.samples), dtype=np.complex128)
        y = np.asarray(to_numpy(out.samples), dtype=np.complex128)
        p = np.mean(np.abs(x) ** 2)
        basis = (np.abs(x) ** 2 / p - 1) * np.sqrt(p)
        d = (y - x).real + (y - x).imag
        coef = np.dot(d, basis) / (2 * np.dot(basis, basis))
        assert coef == pytest.approx(0.05 * np.sqrt(10 ** (-0.5)), rel=0.03)

    def test_single_ended_shot_limit(self, xp):
        """A single diode per quadrature loses half the signal light: the
        shot-limited Es/N0 is n_p/2, 3 dB below the balanced receiver
        (10.9 dB at -45 dBm, quantum efficiency 1; bound 0.15 dB)."""
        h, nu = TestCoherentDetection.H, TestCoherentDetection.NU
        kw = {
            "received_power_dbm": -45.0,
            "lo_power_dbm": 13.0,
            "responsivity": self.Q / (h * nu),
        }
        sig = self._rx(xp)
        single = self._snr_db(
            apply_coherent_detection(sig, balanced=False, rng=4, **kw)
        )
        n_p = 10 ** (-4.5) * 1e-3 / self.RS / (h * nu)
        assert abs(single - 10 * np.log10(n_p / 2)) < 0.15

    def test_signal_signal_beat_falls_with_the_lo(self, xp):
        """Single-ended, no thermal noise or RIN, -10 dBm received: the
        signal-signal beat dominates when the LO is close to the signal and
        fades as the LO-to-signal ratio rises (SNR rising monotonically from
        0 to 20 dB of ratio, by more than 15 dB)."""
        sig = self._rx(xp)
        snr = [
            self._snr_db(
                apply_coherent_detection(
                    sig, received_power_dbm=-10.0, lo_power_dbm=-10.0 + ratio,
                    balanced=False, rng=5,
                )
            )
            for ratio in (0.0, 10.0, 20.0)
        ]  # fmt: skip
        assert snr[0] < snr[1] < snr[2]
        assert snr[2] - snr[0] > 15.0

    def test_ideal_balance_ignores_rin(self, xp, xpt):
        """Baseline: without a CMRR the LO's RIN draws nothing and changes
        nothing, to the bit."""
        sig = self._rx(xp)
        kw = {"received_power_dbm": -20.0, "lo_power_dbm": 10.0, "rng": 6}
        a = apply_coherent_detection(sig, **kw)
        b = apply_coherent_detection(sig, lo_rin_db_hz=-120.0, **kw)
        xpt.assert_allclose(a.samples, b.samples, atol=0)


class TestPhotodetection:
    """apply_photodetection: the square law, shot and thermal noise."""

    Q = TestCoherentDetection.Q
    FS = 64e9

    def test_square_law(self, xp, xpt):
        """Without noise the output is |x|², real in the input's precision,
        its mean the input's mean power."""
        rng = xp.random.default_rng(1)
        x = (rng.standard_normal(1024) + 1j * rng.standard_normal(1024)).astype(
            xp.complex64
        )
        y = apply_photodetection(
            x, received_power_dbm=-10.0, shot_noise=False, sampling_rate=self.FS
        )
        assert y.dtype == xp.float32
        xpt.assert_allclose(y, xp.abs(x) ** 2, rtol=1e-5)

    def test_shot_noise_of_a_constant_field(self, xp):
        """A constant field at -10 dBm, R = 0.8 A/W: I = 80 µA and shot
        noise of variance 2qI · f_s/2 per sample, in output units divided
        by (R k)², k = P / E|x|²; sample variance within 5 relative standard
        errors (sqrt(2/N), 2^16 samples)."""
        n, r = 2**16, 0.8
        x = xp.full(n, 0.5 + 0j, dtype=xp.complex64)  # E|x|² = 0.25
        y = apply_photodetection(
            x, received_power_dbm=-10.0, responsivity=r, sampling_rate=self.FS, rng=2
        )
        p, power = 1e-4, 0.25
        current = r * p
        expected = 2 * self.Q * current * self.FS / 2 / (r * p / power) ** 2
        assert abs(float(xp.var(y)) / expected - 1) < 5 * np.sqrt(2 / n)
        assert float(xp.mean(y)) == pytest.approx(power, rel=1e-3)

    @pytest.mark.parametrize("p_dbm", [-17.0, -15.0])
    def test_thermal_limited_ook(self, xp, p_dbm):
        """On-off keying at 1 sample per symbol with only thermal noise
        (20 pA/sqrt(Hz) at 64 GS/s, σ = 3.58 µA): intensity levels 0 and 2P,
        threshold at P, BER = Q((I1 - I0)/(σ1 + σ0)) = Q(R P / σ) (9e-3 and
        5e-4).  Within 5 binomial standard deviations over 2^18 bits."""
        from commkit.metrics.awgn import _q

        n = 2**18
        bits = np.random.default_rng(3).integers(0, 2, n)
        x = xp.asarray(np.sqrt(2.0 * bits).astype(np.complex64))
        y = apply_photodetection(
            x,
            received_power_dbm=p_dbm,
            thermal_noise_density=20e-12,
            shot_noise=False,
            sampling_rate=self.FS,
            rng=4,
        )
        decided = to_numpy(y) > 1.0
        measured = float(np.mean(decided != bits.astype(bool)))
        sigma = 20e-12 * np.sqrt(self.FS / 2)
        p = float(_q(1e-3 * 10 ** (p_dbm / 10) / sigma))
        assert abs(measured - p) < 5 * np.sqrt(p / n) + 1e-5

    def test_noiseless_ook_is_error_free(self, xp):
        """Baseline: both noises off, the same OOK decides every bit."""
        bits = np.random.default_rng(3).integers(0, 2, 4096)
        x = xp.asarray(np.sqrt(2.0 * bits).astype(np.complex64))
        y = apply_photodetection(
            x, received_power_dbm=-30.0, shot_noise=False, sampling_rate=self.FS
        )
        assert np.array_equal(to_numpy(y) > 1.0, bits.astype(bool))

    def test_dark_current_and_saturation(self, xp):
        """The dark current adds its equivalent I_d / (R k) to the mean; the
        saturation limits the output at I_sat / (R k)."""
        x = xp.asarray(np.array([0.0, 1.0, 2.0, 3.0] * 256, dtype=np.complex64))
        kw = {
            "received_power_dbm": -10.0,
            "shot_noise": False,
            "sampling_rate": self.FS,
        }
        k = 1e-4 / float(np.mean(np.abs(to_numpy(x)) ** 2))
        dark = apply_photodetection(x, dark_current=1e-6, **kw)
        assert float(xp.mean(dark)) == pytest.approx(3.5 + 1e-6 / k, rel=1e-5)
        sat = apply_photodetection(x, saturation_current=4.0 * k, **kw)
        assert float(xp.max(sat)) == pytest.approx(4.0, rel=1e-5)

    def test_sum_channels_and_signal(self, xp):
        """One diode for two polarizations: (N,) output Σ|x_c|²; a Signal
        becomes digital baseband and drops its two-stream reference, while
        per-channel detection keeps it."""
        sig = generate(
            Constellation.qam(4),
            num_symbols=256,
            symbol_rate=1e9,
            num_channels=2,
            rng=5,
            device="gpu" if xp is not np else "cpu",
        ).replace(domain="OPT", center_frequency=193.4e12)
        kw = {"received_power_dbm": 0.0, "shot_noise": False}
        one = apply_photodetection(sig, sum_channels=True, **kw)
        assert one.samples.shape == (256,)
        assert one.domain == "DIG" and one.center_frequency == 0.0
        assert one.reference is None
        x = to_numpy(sig.samples)
        np.testing.assert_allclose(
            to_numpy(one.samples), np.sum(np.abs(x) ** 2, axis=0), rtol=1e-5
        )
        both = apply_photodetection(sig, **kw)
        assert both.samples.shape == (2, 256) and both.reference is sig.reference

    def test_validation(self, xp):
        x = xp.ones(64, dtype=xp.complex64)
        kw = {"received_power_dbm": -10.0, "sampling_rate": self.FS}
        for bad in (
            {"responsivity": 0.0},
            {"dark_current": -1.0},
            {"thermal_noise_density": -1.0},
            {"saturation_current": 0.0},
        ):
            with pytest.raises(ValueError):
                apply_photodetection(x, **kw, **bad)
        with pytest.raises(ValueError, match="zeros"):
            apply_photodetection(xp.zeros(64, xp.complex64), **kw)
