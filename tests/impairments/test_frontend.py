"""Tests for transceiver front-end IQ-imbalance application and compensation."""

import math

import pytest

import commkit as ck
from commkit import RRC, Constellation
from commkit.core import Signal
from commkit.impairments import (
    GramSchmidt,
    IQImbalanceEstimate,
    Lowdin,
    apply_iq_imbalance,
    apply_quantization,
    apply_sampling_clock_offset,
    apply_sampling_jitter,
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


@pytest.mark.parametrize(
    "call",
    [
        lambda x: apply_iq_imbalance(
            x, amplitude_imbalance_db=1.0, phase_imbalance_deg=3.0
        ),
        estimate_iq_imbalance,
        lambda x: correct_iq_imbalance(x, Lowdin()),
        lambda x: correct_iq_imbalance(x, IQImbalanceEstimate(1.0, 3.0)),
    ],
)
def test_iq_functions_reject_real_samples(xp, call):
    """Real samples have no Q branch: each IQ function raises naming them."""
    x = xp.ones(64, dtype=xp.float32)
    with pytest.raises(ValueError, match="samples must be complex"):
        call(x)


class TestIQSkew:
    """The time skew of apply_iq_imbalance: Q delayed against I."""

    FS = 64e9

    @pytest.mark.parametrize("dtype", ["complex64", "complex128"])
    def test_image_ratio_of_tones(self, xp, dtype):
        """A tone at f leaves an image at -f of relative power tan²(π f τ)
        (from I + jQ(t - τ) = e^{jθ}(1 + e^{-jψ})/2 + e^{-jθ}(1 - e^{jψ})/2,
        ψ = 2π f τ).  Tones on FFT bins, so the periodic delay is exact;
        measured agreement 1e-6 relative in complex64."""
        n, tau = 4096, 3e-12
        t = xp.arange(n) / self.FS
        for k in (64, 512, 1500):
            f = k * self.FS / n
            s = xp.exp(2j * xp.pi * f * t).astype(dtype)
            r = apply_iq_imbalance(
                s,
                amplitude_imbalance_db=0.0,
                phase_imbalance_deg=0.0,
                skew=tau,
                sampling_rate=self.FS,
            )
            assert r.dtype == s.dtype
            spec = xp.abs(xp.fft.fft(r.astype(xp.complex128))) ** 2
            measured = float(spec[n - k] / spec[k])
            expected = math.tan(math.pi * f * tau) ** 2
            assert measured == pytest.approx(expected, rel=1e-4)

    def test_delays_q_not_i(self, xp, xpt):
        """A skew of two whole samples rolls Q two samples later and leaves
        I untouched (a sign error rolls Q earlier)."""
        rng = xp.random.default_rng(1)
        s = (rng.standard_normal(1024) + 1j * rng.standard_normal(1024)).astype(
            xp.complex64
        )
        r = apply_iq_imbalance(
            s,
            amplitude_imbalance_db=0.0,
            phase_imbalance_deg=0.0,
            skew=2 / self.FS,
            sampling_rate=self.FS,
        )
        xpt.assert_allclose(r.real, s.real, atol=0)
        xpt.assert_allclose(r.imag, xp.roll(s.imag, 2), atol=1e-5)

    def test_skew_composes_with_gain_and_phase(self, xp, xpt):
        """The skew acts on Q before the mixing: equal to the gain and phase
        errors applied to the skewed signal."""
        rng = xp.random.default_rng(2)
        s = (rng.standard_normal(512) + 1j * rng.standard_normal(512)).astype(
            xp.complex64
        )
        kw = {"amplitude_imbalance_db": 1.0, "phase_imbalance_deg": 4.0}
        skewed = apply_iq_imbalance(
            s,
            amplitude_imbalance_db=0.0,
            phase_imbalance_deg=0.0,
            skew=5e-12,
            sampling_rate=self.FS,
        )
        both = apply_iq_imbalance(s, skew=5e-12, sampling_rate=self.FS, **kw)
        xpt.assert_allclose(both, apply_iq_imbalance(skewed, **kw), atol=1e-5)

    def test_memoryless_correction_leaves_skew(self, xp):
        """Löwdin orthogonalization removes gain and phase errors, not skew:
        on a tone pair at ±f the skew's image (tan²(π f τ) = -19 dB at
        12 GHz and 3 ps) stays above -25 dB after the correction."""
        n, k = 4096, 768
        t = xp.arange(n) / self.FS
        f = k * self.FS / n
        s = (xp.exp(2j * xp.pi * f * t) + 0.5 * xp.exp(-2j * xp.pi * 2 * f * t)).astype(
            xp.complex64
        )
        r = apply_iq_imbalance(
            s,
            amplitude_imbalance_db=0.0,
            phase_imbalance_deg=0.0,
            skew=3e-12,
            sampling_rate=self.FS,
        )
        c = correct_iq_imbalance(r, Lowdin())
        spec = xp.abs(xp.fft.fft(c.astype(xp.complex128))) ** 2
        assert 10 * math.log10(float(spec[n - k] / spec[k])) > -25.0

    def test_sampling_rate_rules(self, xp):
        """Array input needs the sampling rate only for a skew; a Signal
        supplies it, and a conflicting value raises."""
        s = xp.ones(64, dtype=xp.complex64)
        kw = {"amplitude_imbalance_db": 0.0, "phase_imbalance_deg": 0.0}
        apply_iq_imbalance(s, **kw)
        with pytest.raises(ValueError, match="sampling_rate"):
            apply_iq_imbalance(s, skew=1e-12, **kw)
        sig = Signal(samples=s, sampling_rate=self.FS, symbol_rate=self.FS / 2)
        out = apply_iq_imbalance(sig, skew=1e-12, **kw)
        assert isinstance(out, Signal)
        with pytest.raises(ValueError, match="conflicts"):
            apply_iq_imbalance(sig, skew=1e-12, sampling_rate=1e9, **kw)


class TestSamplingClockOffset:
    """apply_sampling_clock_offset samples the band-limited waveform at
    n / (1 + ε)."""

    @pytest.mark.parametrize("dtype", ["complex64", "complex128"])
    @pytest.mark.parametrize("ppm", [150.0, -150.0])
    def test_tone_at_stretched_instants(self, xp, dtype, ppm):
        """A tone at 0.3 cycles per sample comes out as the tone at
        n / (1 + ε): error below -110 dB away from the 24 edge samples
        (measured -132 dB in complex64, the interpolator's design
        -119 dB at 0.4 cycles per sample)."""
        n_in, f = 8192, 0.3
        x = xp.exp(2j * xp.pi * f * xp.arange(n_in)).astype(dtype)
        y = apply_sampling_clock_offset(x, offset_ppm=ppm)
        assert y.dtype == x.dtype
        assert y.shape[0] == math.floor((n_in - 1) * (1 + ppm * 1e-6)) + 1
        t = xp.arange(y.shape[0]) / (1 + ppm * 1e-6)
        err = y - xp.exp(2j * xp.pi * f * t)
        err_db = 10 * math.log10(float(xp.mean(xp.abs(err[30:-30]) ** 2)))
        assert err_db < -110.0

    def test_symbols_drift_later_for_a_fast_clock(self, xp):
        """A band-limited pulse at input sample 8000 peaks near output sample
        8000 · (1 + ε): 8008 at +1000 ppm, 7992 at -1000 ppm."""
        taps = xp.asarray(RRC(rolloff=0.5).taps(4), dtype=xp.complex64)
        x = xp.zeros(16000, dtype=xp.complex64)
        half = taps.shape[0] // 2
        x[8000 - half : 8000 - half + taps.shape[0]] = taps
        for ppm, expected in ((1000.0, 8008), (-1000.0, 7992)):
            y = apply_sampling_clock_offset(x, offset_ppm=ppm)
            assert int(xp.argmax(xp.abs(y))) == expected

    def test_zero_offset_is_identity(self, xp, xpt):
        x = (xp.arange(64) * (1 + 1j)).astype(xp.complex64)
        y = apply_sampling_clock_offset(x, offset_ppm=0.0)
        xpt.assert_allclose(y, x, atol=0)
        assert y is not x

    def test_channels_share_the_clock(self, xp, xpt):
        rng = xp.random.default_rng(3)
        a = (rng.standard_normal(2048) + 1j * rng.standard_normal(2048)).astype(
            xp.complex64
        )
        both = apply_sampling_clock_offset(xp.stack([a, 2 * a]), offset_ppm=40.0)
        xpt.assert_allclose(both[1], 2 * both[0], rtol=1e-5, atol=1e-6)
        xpt.assert_allclose(
            both[0], apply_sampling_clock_offset(a, offset_ppm=40.0), atol=1e-5
        )

    def test_signal_keeps_rate_and_reference(self):
        sig = ck.generate(
            Constellation.qam(16), num_symbols=1000, symbol_rate=1e9, sps=2, rng=1
        )
        out = apply_sampling_clock_offset(sig, offset_ppm=500.0)
        assert out.sampling_rate == sig.sampling_rate
        assert out.samples.shape[-1] == 2000
        assert out.reference is sig.reference

    def test_rejects_offsets_at_or_below_minus_one(self, xp):
        with pytest.raises(ValueError, match="offset_ppm"):
            apply_sampling_clock_offset(xp.ones(8, xp.complex64), offset_ppm=-1e6)


def _sine(xp, n, cycles, amplitude, dtype):
    """A real sine of ``cycles`` periods over ``n`` samples (incommensurate
    with the quantizer's step)."""
    t = xp.arange(n, dtype=xp.float64)
    return (amplitude * xp.sin(2 * xp.pi * cycles * t / n + 0.1)).astype(dtype)


def _sqnr_db(xp, y, x):
    return 10 * math.log10(float(xp.mean(xp.abs(x) ** 2) / xp.mean(xp.abs(y - x) ** 2)))


class TestQuantization:
    @pytest.mark.parametrize("bits", [3, 4, 8])
    def test_error_variance(self, xp, bits):
        """Input spread uniformly over the full scale: the error is uniform in
        ±Δ/2, variance Δ²/12.  Its sample variance over 2^17 values has the
        relative standard deviation sqrt((9/5 - 1)/N) = 0.0025; bound 5 of
        those."""
        n = 2**17
        rng = xp.random.default_rng(7)
        x = rng.uniform(-1.0, 1.0, n).astype(xp.float64)
        y = apply_quantization(x, num_bits=bits, full_scale=1.0)
        step = 2.0 / 2**bits
        ratio = float(xp.mean((y - x) ** 2)) / (step**2 / 12)
        assert abs(ratio - 1) < 5 * math.sqrt(0.8 / n)

    @pytest.mark.parametrize("bits", [10, 12, 14])
    def test_full_scale_sine_sqnr(self, xp, bits):
        """A full-scale sine: 6.02 b + 1.76 dB (Δ²/12 against A²/2).  The
        sine dwells at its peaks, where the error is not uniform, so the
        measured SQNR approaches the formula from below as about 2^(-b/2):
        0.06, 0.03, 0.015 dB short.  Bound 0.1 dB."""
        x = _sine(xp, 2**16, 1001.37, 0.999999, xp.float64)
        y = apply_quantization(x, num_bits=bits, full_scale=1.0)
        assert y.dtype == x.dtype
        assert abs(_sqnr_db(xp, y, x) - (6.02 * bits + 1.76)) < 0.1

    def test_components_of_complex_input(self, xp):
        """I and Q are quantized separately: a complex tone whose components
        are full-scale sines has the SQNR of one sine (0.13 dB short of the
        formula at 8 bits, as for a real sine), and each component equals
        the real quantizer's output."""
        n = 2**16
        t = xp.arange(n, dtype=xp.float64)
        x = (0.999999 * xp.exp(2j * xp.pi * 1001.37 * t / n)).astype(xp.complex64)
        y = apply_quantization(x, num_bits=8, full_scale=1.0)
        assert y.dtype == xp.complex64
        assert abs(_sqnr_db(xp, y, x) - (6.02 * 8 + 1.76)) < 0.2
        xp.testing.assert_array_equal(
            y.imag, apply_quantization(x.imag, num_bits=8, full_scale=1.0)
        )

    @pytest.mark.parametrize("enob", [5.0, 7.3])
    def test_enob_noise(self, xp, enob):
        """With enob the total noise is that of an ideal enob-bit quantizer:
        SQNR 6.02 enob + 1.76 dB.  The noise power over 2^16 samples has a
        relative standard deviation of sqrt(2/N) = 0.006 (0.02 dB); bound
        0.15 dB."""
        x = _sine(xp, 2**16, 1001.37, 0.999999, xp.float32)
        y = apply_quantization(x, num_bits=10, full_scale=1.0, enob=enob, rng=3)
        assert abs(_sqnr_db(xp, y, x) - (6.02 * enob + 1.76)) < 0.15

    def test_clipping_against_gaussian_tail(self, xp):
        """Gaussian components of rms σ against a full scale of 2σ: the
        outputs saturate at the top level ±(A - Δ/2) exactly when the input
        exceeds A - Δ, with probability 2Q((A - Δ)/σ) (0.0503 for 6 bits);
        bound: 5 binomial standard deviations over 2^17 components."""
        from tests.common.theory import q_function

        n, bits, fs = 2**17, 6, 2.0
        rng = xp.random.default_rng(4)
        x = rng.standard_normal(n).astype(xp.float32)
        y = apply_quantization(x, num_bits=bits, full_scale=fs)
        step = 2 * fs / 2**bits
        top = fs - step / 2
        assert float(xp.max(xp.abs(y))) == pytest.approx(top)
        measured = float(xp.mean(xp.abs(y) == xp.float32(top)))
        p = 2 * q_function(fs - step)
        assert abs(measured - p) < 5 * math.sqrt(p * (1 - p) / n)

    def test_levels_are_mid_rise(self, xp, xpt):
        """One bit: the levels ±full_scale/2, no zero level."""
        y = apply_quantization(
            xp.asarray([-3.0, -0.1, 0.0, 0.1, 3.0], dtype=xp.float32),
            num_bits=1,
            full_scale=2.0,
        )
        xpt.assert_allclose(y, xp.asarray([-1.0, -1.0, 1.0, 1.0, 1.0]), atol=0)

    def test_fine_quantizer_is_transparent(self, xp):
        """Baseline: 40 bits leave a double-precision sine untouched to
        200 dB."""
        x = _sine(xp, 4096, 101.3, 0.5, xp.float64)
        y = apply_quantization(x, num_bits=40, full_scale=1.0)
        assert _sqnr_db(xp, y, x) > 200

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"num_bits": 0, "full_scale": 1.0},
            {"num_bits": 8, "full_scale": 0.0},
            {"num_bits": 8, "full_scale": 1.0, "enob": 9.0},
            {"num_bits": 8, "full_scale": 1.0, "enob": 0.0},
        ],
    )
    def test_validation(self, xp, kwargs):
        with pytest.raises(ValueError):
            apply_quantization(xp.ones(8, xp.float32), **kwargs)


class TestSamplingJitter:
    FS = 64e9

    @pytest.mark.parametrize("f_ghz", [1.0, 4.0, 16.0])
    def test_tone_snr(self, xp, f_ghz):
        """A tone at f sampled with jitter σ: y = x·exp(j2πfδ) ≈ x(1 + j2πfδ),
        noise of relative power (2πfσ)², SNR -20 log10(2πfσ) (70.0, 58.0,
        45.9 dB at 50 fs).  The noise power over 2^16 samples has a relative
        standard deviation of 0.006; bound 0.15 dB."""
        n, sigma = 2**16, 50e-15
        f = f_ghz * 1e9
        x = xp.exp(2j * xp.pi * f * xp.arange(n) / self.FS).astype(xp.complex64)
        y = apply_sampling_jitter(x, jitter=sigma, sampling_rate=self.FS, rng=5)
        assert y.dtype == x.dtype
        inner = slice(30, -30)
        snr = _sqnr_db(xp, y[inner], x[inner])
        assert abs(snr + 20 * math.log10(2 * math.pi * f * sigma)) < 0.15

    def test_channels_share_the_clock(self, xp, xpt):
        rng = xp.random.default_rng(6)
        a = (rng.standard_normal(1024) + 1j * rng.standard_normal(1024)).astype(
            xp.complex64
        )
        y = apply_sampling_jitter(
            xp.stack([a, 3 * a]), jitter=2e-12, sampling_rate=self.FS, rng=1
        )
        xpt.assert_allclose(y[1], 3 * y[0], rtol=1e-5, atol=1e-5)

    def test_zero_jitter_is_identity(self, xp, xpt):
        x = xp.arange(32, dtype=xp.float32)
        y = apply_sampling_jitter(x, jitter=0.0, sampling_rate=self.FS)
        xpt.assert_allclose(y, x, atol=0)
        assert y is not x

    def test_sampling_rate_rules(self, xp):
        x = xp.ones(64, dtype=xp.complex64)
        with pytest.raises(ValueError, match="sampling_rate"):
            apply_sampling_jitter(x, jitter=1e-12)
        with pytest.raises(ValueError, match="jitter"):
            apply_sampling_jitter(x, jitter=-1.0, sampling_rate=self.FS)


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
