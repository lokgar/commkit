"""Tests for spectral analysis routines (Welch PSD, Frequency shifting, Spectrograms)."""

from typing import Any

import numpy as np
import pytest

from commkit import spectral


class TestPowerSpectralDensity:
    """Tests for Welch Power Spectral Density (PSD) estimation."""

    def test_welch_psd_real(self, xp: Any) -> None:
        """Verify Welch PSD estimation for real-valued signals, including one-sided/two-sided modes."""
        fs = 100.0
        t = xp.arange(1000) / fs
        freq = 20.0
        samples = xp.sin(2 * xp.pi * freq * t)

        # 1. Default (one-sided for real)
        f, p = spectral.welch_psd(samples, sampling_rate=fs, nperseg=256)

        assert isinstance(f, xp.ndarray)
        assert isinstance(p, xp.ndarray)
        assert xp.min(f) >= 0
        assert xp.max(f) <= fs / 2 + 1e-6

        # Check peak frequency
        peak_idx = xp.argmax(p)
        peak_freq = f[peak_idx]
        assert xp.abs(peak_freq - freq) < (fs / 256)

        # 2. Force two-sided
        f2, p2 = spectral.welch_psd(
            samples, sampling_rate=fs, nperseg=256, return_onesided=False
        )
        assert xp.min(f2) < 0
        peak_idx_pos = xp.argmax(p2 * (f2 > 0))
        peak_freq_pos = f2[peak_idx_pos]
        assert xp.abs(peak_freq_pos - freq) < (fs / 256)

    def test_welch_psd_complex(self, xp: Any) -> None:
        """Verify Welch PSD estimation for complex-valued signals."""
        fs = 100.0
        t = xp.arange(1000) / fs
        freq = 20.0
        samples = xp.exp(1j * 2 * xp.pi * freq * t)

        # 1. Default (two-sided for complex)
        f, p = spectral.welch_psd(samples, sampling_rate=fs, nperseg=256)
        assert isinstance(f, xp.ndarray)
        assert xp.min(f) < 0

        peak_idx = xp.argmax(p)
        peak_freq = f[peak_idx]
        assert xp.abs(peak_freq - freq) < (fs / 256)

        # 2. One-sided should raise for complex
        with pytest.raises(ValueError, match="Cannot compute one-sided PSD"):
            spectral.welch_psd(samples, sampling_rate=fs, return_onesided=True)

    def test_welch_psd_parameters(self, xp: Any) -> None:
        """Verify Welch PSD estimation with custom window, noverlap, nfft, and scaling."""
        fs = 100.0
        t = xp.arange(1000) / fs
        freq = 20.0
        samples = xp.sin(2 * xp.pi * freq * t)

        f, p = spectral.welch_psd(
            samples,
            sampling_rate=fs,
            nperseg=256,
            window=("kaiser", 8.0),
            noverlap=128,
            nfft=512,
            scaling="spectrum",
        )
        assert len(f) == 257
        assert isinstance(f, xp.ndarray)
        assert isinstance(p, xp.ndarray)


class TestFrequencyShift:
    """Tests for complex frequency shifting and phase modulation."""

    def test_shift_frequency(self, xp: Any) -> None:
        """Verify complex frequency shifting, bin quantization, and energy preservation."""
        fs = 100.0
        N = 100
        t = xp.arange(N) / fs
        s = xp.exp(1j * 2 * xp.pi * 20 * t)

        shifted = spectral.shift_frequency(s, offset=10.0, sampling_rate=fs)
        actual = spectral.grid_frequency(
            10.0, sampling_rate=fs, num_samples=s.shape[-1]
        )
        assert actual == 10.0

        f_axis = xp.fft.fftfreq(N, 1 / fs)
        peak_idx = xp.argmax(xp.abs(xp.fft.fft(shifted)))
        peak_freq = f_axis[peak_idx]
        assert xp.isclose(peak_freq, 30.0)

        # Quantized shift
        shifted_q = spectral.shift_frequency(s, offset=10.5, sampling_rate=fs)
        actual_q = spectral.grid_frequency(
            10.5, sampling_rate=fs, num_samples=s.shape[-1]
        )
        assert actual_q % 1.0 == 0.0
        assert abs(actual_q - 10.5) <= 0.5

        # Energy preservation (unitary)
        energy_in = xp.sum(xp.abs(s) ** 2)
        energy_out = xp.sum(xp.abs(shifted_q) ** 2)
        assert xp.isclose(energy_in, energy_out)

    def test_shift_frequency_preserves_complex64_dtype(self, xp: Any) -> None:
        """shift_frequency: complex64 signal -> complex64 output."""
        rng = np.random.default_rng(20)
        s = xp.asarray(
            (rng.standard_normal(512) + 1j * rng.standard_normal(512)).astype(
                np.complex64
            )
        )
        out = spectral.shift_frequency(s, offset=100.0, sampling_rate=1000.0)
        assert out.dtype == xp.complex64

    def test_shift_frequency_preserves_float32_dtype(self, xp: Any) -> None:
        """shift_frequency: float32 signal -> complex64 output."""
        rng = np.random.default_rng(21)
        s = xp.asarray(rng.standard_normal(512).astype(np.float32))
        out = spectral.shift_frequency(s, offset=100.0, sampling_rate=1000.0)
        assert out.dtype == xp.complex64


class TestSpectrogram:
    """Tests for short-time Fourier transform spectrogram estimation."""

    def test_spectrogram_real(self, xp: Any) -> None:
        """Verify spectrogram calculation for real-valued signals."""
        fs = 100.0
        t_vec = xp.arange(1000) / fs
        freq = 20.0
        samples = xp.sin(2 * xp.pi * freq * t_vec)

        f, t, Sxx = spectral.spectrogram(
            samples, sampling_rate=fs, nperseg=256, noverlap=128
        )

        assert isinstance(f, xp.ndarray)
        assert isinstance(t, xp.ndarray)
        assert isinstance(Sxx, xp.ndarray)
        assert len(f) == 129
        assert Sxx.shape == (129, len(t))
        assert xp.min(f) >= 0
        assert xp.max(f) <= fs / 2 + 1e-6

        for col in range(Sxx.shape[1]):
            peak_idx = xp.argmax(Sxx[:, col])
            peak_freq = f[peak_idx]
            assert xp.abs(peak_freq - freq) < (fs / 256)

    def test_spectrogram_complex_mimo(self, xp: Any) -> None:
        """Verify spectrogram calculation for complex-valued MIMO signals."""
        fs = 100.0
        t_vec = xp.arange(1000) / fs
        freq = 20.0
        samples = xp.exp(1j * 2 * xp.pi * freq * t_vec)
        samples_mimo = xp.stack([samples, samples * 2])

        f, t, Sxx = spectral.spectrogram(
            samples_mimo, sampling_rate=fs, nperseg=256, noverlap=128
        )

        assert isinstance(f, xp.ndarray)
        assert isinstance(t, xp.ndarray)
        assert isinstance(Sxx, xp.ndarray)
        assert len(f) == 256
        assert Sxx.shape == (2, 256, len(t))
        assert xp.min(f) < 0
        assert xp.max(f) > 0

        with pytest.raises(ValueError, match="Cannot compute one-sided spectrogram"):
            spectral.spectrogram(samples_mimo, sampling_rate=fs, return_onesided=True)


class TestAddPilotTone:
    """Tests for spectral.add_pilot_tone (CW pilot-tone injection)."""

    @staticmethod
    def _signal(xp: Any, N: int = 4096, seed: int = 0) -> Any:
        rng = xp.random.RandomState(seed)
        x = (rng.randn(N) + 1j * rng.randn(N)).astype(xp.complex128)
        return x / xp.sqrt(xp.mean(xp.abs(x) ** 2))

    @pytest.mark.parametrize("psr_db", [-20.0, -10.0, 0.0])
    def test_power_ratio(self, xp: Any, psr_db: float) -> None:
        """Added tone power matches the requested pilot-to-signal ratio."""
        fs = 100.0
        x = self._signal(xp)
        p_sig = float(xp.mean(xp.abs(x) ** 2))
        y = spectral.add_pilot_tone(x, fs, 30.0, power_ratio_db=psr_db)
        p_tone = float(xp.mean(xp.abs(y - x) ** 2))
        assert abs(10 * np.log10(p_tone / p_sig) - psr_db) < 0.05

    def test_peak_location(self, xp: Any) -> None:
        """The injected tone shows up as the dominant spectral peak at the snapped f_p."""
        fs = 100.0
        N = 4096
        f_p = 30.0
        x = self._signal(xp, N=N)
        y = spectral.add_pilot_tone(x, fs, f_p, power_ratio_db=10.0)
        f_actual = spectral.grid_frequency(
            f_p, sampling_rate=fs, num_samples=x.shape[-1]
        )
        freqs = xp.fft.fftfreq(N, d=1.0 / fs)
        k = int(xp.argmax(xp.abs(xp.fft.fft(y))))
        assert abs(float(freqs[k]) - f_actual) < fs / N

    def test_snaps_to_grid(self, xp: Any) -> None:
        """Returned frequency lies exactly on the f_s/N grid, near the request."""
        fs = 100.0
        N = 4096
        df = fs / N
        x = self._signal(xp, N=N)
        f_req = 30.0 + 0.4 * df
        spectral.add_pilot_tone(x, fs, f_req)
        f_actual = spectral.grid_frequency(
            f_req, sampling_rate=fs, num_samples=x.shape[-1]
        )
        assert abs(round(f_actual / df) - f_actual / df) < 1e-9
        assert abs(f_actual - f_req) <= df / 2 + 1e-9

    def test_dtype_and_shape_preserved(self, xp: Any) -> None:
        """complex64 stays complex64; SISO/MIMO shapes are preserved."""
        fs = 100.0
        x64 = self._signal(xp).astype(xp.complex64)
        y = spectral.add_pilot_tone(x64, fs, 30.0)
        assert y.dtype == xp.complex64
        assert y.shape == x64.shape

        mimo = xp.stack([self._signal(xp), 2 * self._signal(xp, seed=1)])
        ym = spectral.add_pilot_tone(mimo, fs, 30.0)
        assert ym.shape == mimo.shape

    def test_renormalize_preserves_power(self, xp: Any) -> None:
        """renormalize=True restores each channel's original mean power."""
        fs = 100.0
        mimo = xp.stack([self._signal(xp), 2 * self._signal(xp, seed=1)])
        p_in = xp.mean(xp.abs(mimo) ** 2, axis=-1)
        y = spectral.add_pilot_tone(
            mimo, fs, 30.0, power_ratio_db=-6.0, renormalize=True
        )
        p_out = xp.mean(xp.abs(y) ** 2, axis=-1)
        assert bool(xp.allclose(p_in, p_out, rtol=1e-4))

    def test_invalid_frequency_raises(self, xp: Any) -> None:
        """Tone frequency outside (-fs/2, fs/2) raises ValueError."""
        fs = 100.0
        x = self._signal(xp)
        with pytest.raises(ValueError, match=r"must lie in \(-fs/2, fs/2\)"):
            spectral.add_pilot_tone(x, fs, fs)

    def test_scalar_returns_float(self, xp: Any) -> None:
        """Scalar frequency returns a plain float (back-compat), even for MIMO."""
        fs = 100.0
        mimo = xp.stack([self._signal(xp), self._signal(xp, seed=1)])
        spectral.add_pilot_tone(mimo, fs, 30.0)
        f_actual = spectral.grid_frequency(
            30.0, sampling_rate=fs, num_samples=mimo.shape[-1]
        )
        assert isinstance(f_actual, float)

    def test_per_channel_frequencies(self, xp: Any) -> None:
        """A per-channel list places one distinct tone per channel at its bin."""
        fs = 100.0
        N = 4096
        f_req = [20.0, -35.0]
        mimo = xp.stack([self._signal(xp, N=N), self._signal(xp, N=N, seed=1)])
        y = spectral.add_pilot_tone(mimo, fs, f_req, power_ratio_db=10.0)
        f_actual = spectral.grid_frequency(
            f_req, sampling_rate=fs, num_samples=mimo.shape[-1]
        )
        assert isinstance(f_actual, np.ndarray) and len(f_actual) == 2
        freqs = xp.fft.fftfreq(N, d=1.0 / fs)
        for c in range(2):
            k = int(xp.argmax(xp.abs(xp.fft.fft(y[c]))))
            assert abs(float(freqs[k]) - f_actual[c]) < fs / N
            assert abs(f_actual[c] - f_req[c]) <= fs / N / 2 + 1e-9
        assert f_actual[0] != f_actual[1]

    def test_per_channel_length_mismatch_raises(self, xp: Any) -> None:
        """A per-channel sequence whose length != C raises ValueError."""
        fs = 100.0
        mimo = xp.stack([self._signal(xp), self._signal(xp, seed=1)])
        with pytest.raises(ValueError, match=r"one frequency per channel"):
            spectral.add_pilot_tone(mimo, fs, [20.0, -30.0, 10.0])

    def test_per_channel_invalid_frequency_raises(self, xp: Any) -> None:
        """An out-of-range entry in a per-channel sequence raises ValueError."""
        fs = 100.0
        mimo = xp.stack([self._signal(xp), self._signal(xp, seed=1)])
        with pytest.raises(ValueError, match=r"must lie in \(-fs/2, fs/2\)"):
            spectral.add_pilot_tone(mimo, fs, [20.0, fs])

    def test_per_channel_power_ratio(self, xp: Any) -> None:
        """A per-channel PSR sequence realises a distinct tone power per channel."""
        fs = 100.0
        psr = [-10.0, -20.0]
        mimo = xp.stack([self._signal(xp), self._signal(xp, seed=1)])
        y = spectral.add_pilot_tone(mimo, fs, [20.0, -35.0], power_ratio_db=psr)
        for c in range(2):
            p_sig = float(xp.mean(xp.abs(mimo[c]) ** 2))
            p_tone = float(xp.mean(xp.abs(y[c] - mimo[c]) ** 2))
            assert abs(10 * np.log10(p_tone / p_sig) - psr[c]) < 0.05

    def test_per_channel_power_length_mismatch_raises(self, xp: Any) -> None:
        """A per-channel PSR sequence whose length != C raises ValueError."""
        fs = 100.0
        mimo = xp.stack([self._signal(xp), self._signal(xp, seed=1)])
        with pytest.raises(ValueError, match=r"one PSR per channel"):
            spectral.add_pilot_tone(mimo, fs, [20.0, -30.0], power_ratio_db=[-10.0])


class TestGridFrequency:
    """grid_frequency is the quantization shift_frequency/add_pilot_tone apply."""

    def test_values(self) -> None:
        assert (
            spectral.grid_frequency(1.03e6, sampling_rate=8e6, num_samples=1000)
            == 1.032e6
        )
        np.testing.assert_allclose(
            spectral.grid_frequency(
                [10.4, -10.6], sampling_rate=100.0, num_samples=100
            ),
            [10.0, -11.0],
        )
        with pytest.raises(ValueError, match="num_samples"):
            spectral.grid_frequency(1.0, sampling_rate=1.0, num_samples=0)

    def test_shift_lands_on_grid_frequency(self, xp: Any) -> None:
        fs, n, f = 1000.0, 1000, 123.4
        out = spectral.shift_frequency(xp.ones(n, dtype=xp.complex64), f, fs)
        expected = spectral.grid_frequency(f, sampling_rate=fs, num_samples=n)
        peak = int(xp.argmax(xp.abs(xp.fft.fft(out))))
        assert peak * fs / n == pytest.approx(expected)

    def test_pilot_tone_lands_on_grid_frequency(self, xp: Any) -> None:
        fs, n, f = 1000.0, 1000, -201.6
        y = spectral.add_pilot_tone(xp.zeros(n, dtype=xp.complex64) + 1e-3, fs, f)
        expected = spectral.grid_frequency(f, sampling_rate=fs, num_samples=n)
        spec = xp.abs(xp.fft.fft(y))
        spec[0] = 0
        peak = int(xp.argmax(spec))
        assert (peak - n if peak > n // 2 else peak) * fs / n == pytest.approx(expected)
