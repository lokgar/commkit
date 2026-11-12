"""Tests for source impairments (phase noise, intensity noise, frequency offset)."""

import math

import numpy as np
import pytest

from commkit.backend import to_device
from commkit.core import Signal
from commkit.frequency import correct_frequency_offset
from commkit.impairments import (
    apply_frequency_offset,
    apply_intensity_noise,
    apply_phase_noise,
    generate_phase_noise,
)
from tests.common.conversions import device_of


class TestApplyPhaseNoise:
    """Tests for apply_phase_noise."""

    def test_modifies_signal(self, xp):
        """Phase noise should change the signal."""
        samples = xp.ones(1024, dtype=xp.complex64)
        out = apply_phase_noise(samples, linewidth=1e6, sampling_rate=64e9, rng=3)
        diff = float(xp.max(xp.abs(out - samples)))
        assert diff > 1e-4

    def test_preserves_amplitude(self, xp):
        """Phase rotation must not change sample amplitude."""
        rng = xp.random.RandomState(42)
        samples = (rng.randn(2048) + 1j * rng.randn(2048)).astype(xp.complex64)
        out = apply_phase_noise(samples, linewidth=100e3, sampling_rate=64e9, rng=4)
        amp_in = xp.abs(samples)
        amp_out = xp.abs(out)
        assert float(xp.max(xp.abs(amp_out - amp_in))) < 1e-4

    def test_phase_variance_matches_linewidth(self, xp):
        """Incremental phase variance per sample should equal 2π·Δν/fs.

        Extracts phase increments from a long single-channel trajectory and
        measures their variance.  A flat-amplitude input of ones means
        angle(out[n]) = cumulative phase, so diff(angle(out)) = increments.
        """
        import math

        N = 200000
        linewidth = 100e3
        fs = 64e9
        expected_variance = 2.0 * math.pi * linewidth / fs

        # Unit-amplitude input so abs(out)=1 and angle(out) = cumulative phase
        samples = xp.ones(N, dtype=xp.complex128)
        out = apply_phase_noise(samples, linewidth=linewidth, sampling_rate=fs, rng=42)

        # Increments from consecutive phasors: immune to the ±π wrap that
        # diff(angle(out)) takes whenever the walk crosses it.
        increments = xp.angle(out[1:] * xp.conj(out[:-1]))  # (N-1,)
        measured_var = float(xp.var(increments))
        assert abs(measured_var - expected_variance) / expected_variance < 0.05

    def test_shared_lo_all_channels_equal(self, xp):
        """shared_lo=True: all channels receive identical phase trajectory."""
        C, N = 4, 512
        samples = xp.ones((C, N), dtype=xp.complex128)
        out = apply_phase_noise(
            samples, linewidth=1e6, sampling_rate=64e9, rng=7, shared_lo=True
        )
        # All channels should have identical output since same phase is applied
        for c in range(1, C):
            assert float(xp.max(xp.abs(out[c] - out[0]))) < 1e-10

    def test_independent_lo_channels_differ(self, xp):
        """shared_lo=False (default): channels have independent phase trajectories."""
        C, N = 2, 512
        samples = xp.ones((C, N), dtype=xp.complex128)
        out = apply_phase_noise(
            samples, linewidth=10e6, sampling_rate=64e9, rng=5, shared_lo=False
        )
        # Independent trajectories should differ
        diff = float(xp.max(xp.abs(out[0] - out[1])))
        assert diff > 1e-4

    def test_flicker_modifies_signal_preserves_amplitude(self, xp):
        """Flicker-only phase noise rotates without changing amplitude."""
        samples = xp.ones(4096, dtype=xp.complex128)
        out = apply_phase_noise(
            samples, linewidth=0.0, flicker=1e9, sampling_rate=500e6, rng=6
        )
        assert float(xp.max(xp.abs(xp.abs(out) - 1.0))) < 1e-9
        assert float(xp.max(xp.abs(out - samples))) > 1e-4

    def test_signal_input_returns_signal(self, xp, xpt):
        """Signal input: sampling_rate is taken from the signal."""
        fs = 64e9
        data = xp.ones(1024, dtype=xp.complex128)
        sig = Signal(samples=data, sampling_rate=fs, symbol_rate=fs / 4)

        out_sig = apply_phase_noise(sig, linewidth=100e3, rng=1)
        out_arr = apply_phase_noise(data, sampling_rate=fs, linewidth=100e3, rng=1)

        assert isinstance(out_sig, Signal)
        xpt.assert_allclose(out_sig.samples, out_arr)


class TestGeneratePhaseNoise:
    """Tests for generate_phase_noise."""

    def test_shapes_and_dtype(self, xp):
        """SISO (N,) for one stream, (C, N) for several; float64 on ``device``."""
        phi = generate_phase_noise(
            num_samples=256,
            sampling_rate=64e9,
            linewidth=1e6,
            rng=1,
            device=device_of(xp),
        )
        assert isinstance(phi, xp.ndarray)
        assert phi.shape == (256,)
        assert phi.dtype == np.float64
        phi2 = generate_phase_noise(
            num_samples=256, sampling_rate=64e9, linewidth=1e6, num_channels=3, rng=1
        )
        assert phi2.shape == (3, 256)

    def test_seed_reproducible(self, xp):
        """Same seed yields the identical trajectory on one device."""
        kw = dict(num_samples=1024, sampling_rate=64e9, linewidth=1e6, flicker=1e9)
        a = generate_phase_noise(**kw, rng=42, device=device_of(xp))
        b = generate_phase_noise(**kw, rng=42, device=device_of(xp))
        assert float(xp.max(xp.abs(a - b))) == 0.0

    def test_wiener_increment_variance(self, xp):
        """White-FM increments have variance 2π·Δν/fs."""
        fs, dnu = 64e9, 100e3
        phi = generate_phase_noise(
            num_samples=200_000,
            sampling_rate=fs,
            linewidth=dnu,
            rng=7,
            device=device_of(xp),
        )
        expected = 2.0 * math.pi * dnu / fs
        measured = float(xp.var(xp.diff(phi)))
        assert abs(measured - expected) / expected < 0.05

    def test_flicker_fm_psd_level(self, xp):
        """Flicker-only FM PSD follows S_f(f) = h_-1 / f in the shaped band."""
        from scipy import signal as sp_sig

        fs, h_m1 = 500e6, 4e9
        phi = generate_phase_noise(
            num_samples=1 << 20,
            sampling_rate=fs,
            flicker=h_m1,
            rng=9,
            device=device_of(xp),
        )
        df = np.diff(to_device(phi, "cpu")) * fs / (2.0 * np.pi)
        f, s_f = sp_sig.welch(df, fs=fs, nperseg=1 << 15)
        band = (f > 1e5) & (f < 1e7)
        # S_f·f is flat at h_-1 for a 1/f PSD; median over two decades.
        assert abs(np.median(s_f[band] * f[band]) - h_m1) / h_m1 < 0.2

    def test_matches_apply_phase_noise_trajectory(self, xp):
        """apply_phase_noise(ones) reproduces the generate_phase_noise walk."""
        fs, dnu, n = 64e9, 100e3, 8192
        phi = generate_phase_noise(
            num_samples=n, sampling_rate=fs, linewidth=dnu, rng=11, device=device_of(xp)
        )
        out = apply_phase_noise(
            xp.ones(n, dtype=xp.complex128), sampling_rate=fs, linewidth=dnu, rng=11
        )
        # Wrapped comparison: increments are tiny, so unwrap-free diff works.
        err = xp.abs(xp.exp(1j * phi) - out)
        assert float(xp.max(err)) < 1e-9


class TestPeriodicPhaseNoise:
    """periodic=True: the same walk, closed on itself modulo 2π."""

    def test_differs_from_walk_by_a_ramp(self, xp):
        """Every increment, the wrap step included, is shifted by one constant
        of at most π/N, and the increments sum to a multiple of 2π."""
        kw = dict(num_samples=4096, sampling_rate=64e9, linewidth=5e6, flicker=1e12)
        kw.update(num_channels=3, rng=3, device=device_of(xp))
        walk = to_device(generate_phase_noise(**kw), "cpu")
        loop = to_device(generate_phase_noise(periodic=True, **kw), "cpu")
        n = kw["num_samples"]
        shift = np.diff(loop - walk, axis=-1)
        np.testing.assert_allclose(
            shift, shift[:, :1].repeat(n - 1, axis=1), atol=1e-12
        )
        assert np.all(np.abs(shift) <= np.pi / n + 1e-12)
        # Increments around the loop: loop[0] (from the wrap) plus the diffs.
        total = loop[:, -1]
        np.testing.assert_allclose(np.angle(np.exp(1j * total)), np.zeros(3), atol=1e-9)

    def test_wrap_step_is_an_ordinary_increment(self, xp):
        """Many loops: the wrap step has the variance of an interior step."""
        fs, dnu, n = 64e9, 1e6, 256
        phi = to_device(
            generate_phase_noise(
                num_samples=n,
                sampling_rate=fs,
                linewidth=dnu,
                num_channels=4000,
                periodic=True,
                rng=5,
                device=device_of(xp),
            ),
            "cpu",
        )
        wrap = np.angle(np.exp(1j * (phi[:, 0] - phi[:, -1])))
        expected = 2.0 * math.pi * dnu / fs * (1 - 1 / n)
        assert abs(np.var(wrap) - expected) / expected < 0.1
        assert abs(np.var(np.diff(phi, axis=-1)) - expected) / expected < 0.05

    def test_apply_forwards_periodic(self, xp):
        fs, dnu, n = 64e9, 100e3, 1024
        phi = generate_phase_noise(
            num_samples=n,
            sampling_rate=fs,
            linewidth=dnu,
            periodic=True,
            rng=11,
            device=device_of(xp),
        )
        out = apply_phase_noise(
            xp.ones(n, dtype=xp.complex128),
            sampling_rate=fs,
            linewidth=dnu,
            periodic=True,
            rng=11,
        )
        assert float(xp.max(xp.abs(xp.exp(1j * phi) - out))) < 1e-9


class TestIntensityNoise:
    """apply_intensity_noise: |x|² carries the declared RIN."""

    FS = 64e9

    def test_variance_and_spectrum(self, xp):
        """A continuous wave at -130 dB/Hz: the relative intensity r = |y|² - 1
        has variance RIN · f_s / 2 = 3.2e-3 (relative standard error
        sqrt(2/N) = 0.0055 over 2^16 samples, bound 5 of those), and its
        one-sided PSD is flat at RIN (mean over the band within 3 %)."""
        from commkit.spectral import welch_psd

        n = 2**16
        x = xp.ones(n, dtype=xp.complex64)
        y = apply_intensity_noise(x, rin_db_hz=-130.0, sampling_rate=self.FS, rng=1)
        r = xp.abs(y) ** 2 - 1.0
        expected = 1e-13 * self.FS / 2
        assert abs(float(xp.var(r)) / expected - 1) < 5 * np.sqrt(2 / n)
        _, psd = welch_psd(r, sampling_rate=self.FS, nperseg=1024)
        assert float(xp.mean(psd[1:])) == pytest.approx(1e-13, rel=0.03)

    def test_phase_untouched_and_one_laser(self, xp, xpt):
        """The factor is real and shared: every channel's phase is kept and
        both channels scale alike."""
        rng = xp.random.default_rng(2)
        a = (rng.standard_normal(4096) + 1j * rng.standard_normal(4096)).astype(
            xp.complex64
        )
        x = xp.stack([a, 2j * a])
        y = apply_intensity_noise(x, rin_db_hz=-120.0, sampling_rate=self.FS, rng=3)
        ratio = y / x
        xpt.assert_allclose(ratio.imag, 0.0, atol=1e-6)
        xpt.assert_allclose(ratio[0], ratio[1], rtol=1e-6)

    def test_needs_sampling_rate(self, xp):
        with pytest.raises(ValueError, match="sampling_rate"):
            apply_intensity_noise(xp.ones(8, xp.complex64), rin_db_hz=-140.0)


class TestFrequencyOffset:
    """apply_frequency_offset: exact mixing at any offset."""

    FS = 64e9

    def _noise(self, xp, shape, dtype=np.complex64):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        return xp.asarray(x.astype(dtype))

    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    def test_matches_the_double_precision_mixer(self, xp, xpt, dtype):
        """Against x·exp(j2πΔf n/fs) built in float64, over 2^20 samples at
        an offset between FFT bins: the error stays at the precision of the
        data (complex64: rounding of the phasor, about 3e-7 relative, measured
        9e-7 absolute on unit-variance samples; a float32 phase ramp would be
        off by radians by the end)."""
        n = 1 << 20
        fo = 1.234567e9
        x = self._noise(xp, n, dtype)
        y = apply_frequency_offset(x, offset=fo, sampling_rate=self.FS)
        assert y.dtype == dtype
        ref = to_device(x, "cpu").astype(np.complex128) * np.exp(
            2j * np.pi * fo / self.FS * np.arange(n)
        )
        xpt.assert_allclose(
            y, xp.asarray(ref), atol=5e-6 if dtype == np.complex64 else 1e-9
        )

    def test_inverse_of_correct_frequency_offset(self, xp, xpt):
        x = self._noise(xp, (2, 4096))
        y = apply_frequency_offset(x, offset=[3.3e8, -1.1e8], sampling_rate=self.FS)
        back = correct_frequency_offset(
            y, xp.asarray([3.3e8, -1.1e8]), sampling_rate=self.FS
        )
        xpt.assert_allclose(back, x, atol=2e-6)

    def test_per_channel_offsets(self, xp):
        """Each channel turns by its own step 2πΔf_c/fs per sample."""
        x = xp.ones((2, 1000), dtype=xp.complex64)
        fo = np.array([1e8, -3e8])
        y = to_device(
            apply_frequency_offset(x, offset=fo, sampling_rate=self.FS), "cpu"
        )
        step = np.angle(np.sum(y[:, 1:] * np.conj(y[:, :-1]), axis=-1))
        np.testing.assert_allclose(step, 2 * np.pi * fo / self.FS, rtol=1e-5)

    def test_real_input_gives_complex(self, xp):
        x = xp.ones(16, dtype=xp.float32)
        y = apply_frequency_offset(x, offset=1e6, sampling_rate=self.FS)
        assert y.dtype == xp.complex64

    def test_wrong_number_of_offsets_raises(self, xp):
        x = xp.ones((2, 16), dtype=xp.complex64)
        with pytest.raises(ValueError, match="offset"):
            apply_frequency_offset(x, offset=[1.0, 2.0, 3.0], sampling_rate=self.FS)
