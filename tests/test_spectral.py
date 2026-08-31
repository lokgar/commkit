"""Tests for spectral analysis routines (Welch PSD, Frequency shifting)."""

import numpy as np
import pytest

from commkit import spectral


def test_welch_psd_real(backend_device, xp):
    """Verify Welch PSD estimation for real-valued signals, including one-sided/two-sided modes."""
    # Generate a simple sine wave
    fs = 100.0
    t = xp.arange(1000) / fs
    freq = 20.0
    samples = xp.sin(2 * xp.pi * freq * t)

    # 1. Default (one-sided for real)
    f, p = spectral.welch_psd(samples, sampling_rate=fs, nperseg=256)

    assert isinstance(f, xp.ndarray)
    assert isinstance(p, xp.ndarray)

    # Check frequency range: 0 to fs/2
    assert xp.min(f) >= 0
    assert xp.max(f) <= fs / 2 + 1e-6  # small tolerance

    # Check peak frequency
    peak_idx = xp.argmax(p)
    peak_freq = f[peak_idx]
    assert xp.abs(peak_freq - freq) < (fs / 256)  # Resolution check

    # 2. Force two-sided
    f2, p2 = spectral.welch_psd(
        samples, sampling_rate=fs, nperseg=256, return_onesided=False
    )

    # Check frequency range: centered around 0 (fftshifted)
    assert xp.min(f2) < 0

    # Peak should appear at +freq and -freq
    # Tolerant search near positive peak
    peak_idx_pos = xp.argmax(p2 * (f2 > 0))  # mask negative
    peak_freq_pos = f2[peak_idx_pos]
    assert xp.abs(peak_freq_pos - freq) < (fs / 256)


def test_welch_psd_complex(backend_device, xp):
    """Verify Welch PSD estimation for complex-valued signals."""
    # Complex exponential
    fs = 100.0
    t = xp.arange(1000) / fs
    freq = 20.0
    samples = xp.exp(1j * 2 * xp.pi * freq * t)

    # 1. Default (two-sided for complex)
    f, p = spectral.welch_psd(samples, sampling_rate=fs, nperseg=256)

    assert isinstance(f, xp.ndarray)
    assert xp.min(f) < 0  # centered

    peak_idx = xp.argmax(p)
    peak_freq = f[peak_idx]

    assert xp.abs(peak_freq - freq) < (fs / 256)

    # 2. Try force one-sided (should fail for complex)
    with pytest.raises(ValueError, match="Cannot compute one-sided PSD"):
        spectral.welch_psd(samples, sampling_rate=fs, return_onesided=True)


def test_welch_psd_parameters(backend_device, xp):
    """Verify Welch PSD estimation with custom window, noverlap, nfft, and scaling."""
    fs = 100.0
    t = xp.arange(1000) / fs
    freq = 20.0
    samples = xp.sin(2 * xp.pi * freq * t)

    # Test custom window, noverlap, nfft, and scaling
    f, p = spectral.welch_psd(
        samples,
        sampling_rate=fs,
        nperseg=256,
        window=("kaiser", 8.0),
        noverlap=128,
        nfft=512,
        scaling="spectrum",
    )
    assert len(f) == 257  # nfft // 2 + 1 for real
    assert isinstance(f, xp.ndarray)
    assert isinstance(p, xp.ndarray)


def test_shift_frequency(backend_device, xp):
    """Verify complex frequency shifting and energy preservation."""
    # 1. Exact integer shift
    # fs=100, N=100 -> df=1Hz. Shift by 10Hz.
    fs = 100.0
    N = 100
    t = xp.arange(N) / fs
    # Signal at 20 Hz
    s = xp.exp(1j * 2 * xp.pi * 20 * t)

    shifted, actual = spectral.shift_frequency(s, offset=10.0, sampling_rate=fs)
    assert actual == 10.0

    # New frequency should be 30 Hz
    f_axis = xp.fft.fftfreq(N, 1 / fs)
    peak_idx = xp.argmax(xp.abs(xp.fft.fft(shifted)))
    peak_freq = f_axis[peak_idx]
    assert xp.isclose(peak_freq, 30.0)

    # 2. Quantized shift
    # Shift by 10.5 Hz. Should be quantized to integer multiple of df.
    shifted_q, actual_q = spectral.shift_frequency(s, offset=10.5, sampling_rate=fs)
    # Check it is integer multiple of df=1
    assert actual_q % 1.0 == 0.0
    assert abs(actual_q - 10.5) <= 0.5

    # 3. Energy preservation (unitary)
    energy_in = xp.sum(xp.abs(s) ** 2)
    energy_out = xp.sum(xp.abs(shifted_q) ** 2)
    assert xp.isclose(energy_in, energy_out)


def test_shift_frequency_preserves_complex64_dtype(backend_device, xp):
    """shift_frequency: complex64 signal -> complex64 output (no float64 promotion)."""
    import numpy as np

    rng = np.random.default_rng(20)
    s = xp.asarray(
        (rng.standard_normal(512) + 1j * rng.standard_normal(512)).astype(np.complex64)
    )
    out, _ = spectral.shift_frequency(s, offset=100.0, sampling_rate=1000.0)
    assert out.dtype == xp.complex64, f"Expected complex64, got {out.dtype}"


def test_shift_frequency_preserves_float32_dtype(backend_device, xp):
    """shift_frequency: float32 signal -> complex64 output (real signal becomes complex after shift)."""
    import numpy as np

    rng = np.random.default_rng(21)
    s = xp.asarray(rng.standard_normal(512).astype(np.float32))
    out, _ = spectral.shift_frequency(s, offset=100.0, sampling_rate=1000.0)
    assert out.dtype == xp.complex64, f"Expected complex64, got {out.dtype}"


def test_spectrogram_real(backend_device, xp):
    """Verify spectrogram calculation for real-valued signals, including shape and peak frequency."""
    fs = 100.0
    t_vec = xp.arange(1000) / fs
    freq = 20.0
    samples = xp.sin(2 * xp.pi * freq * t_vec)

    # 1. Default (one-sided for real)
    f, t, Sxx = spectral.spectrogram(
        samples, sampling_rate=fs, nperseg=256, noverlap=128
    )

    assert isinstance(f, xp.ndarray)
    assert isinstance(t, xp.ndarray)
    assert isinstance(Sxx, xp.ndarray)

    # Check shapes
    # 256 nperseg -> 129 frequency bins (onesided)
    assert len(f) == 129
    assert Sxx.shape == (129, len(t))

    # Frequency range: 0 to fs/2
    assert xp.min(f) >= 0
    assert xp.max(f) <= fs / 2 + 1e-6

    # Peak frequency should be close to 20Hz
    # We find peak for each time slice
    for col in range(Sxx.shape[1]):
        peak_idx = xp.argmax(Sxx[:, col])
        peak_freq = f[peak_idx]
        assert xp.abs(peak_freq - freq) < (fs / 256)


def test_spectrogram_complex_mimo(backend_device, xp):
    """Verify spectrogram calculation for complex-valued MIMO signals with two-sided spectrum."""
    fs = 100.0
    t_vec = xp.arange(1000) / fs
    freq = 20.0
    samples = xp.exp(1j * 2 * xp.pi * freq * t_vec)

    # Make 2-channel MIMO signal
    samples_mimo = xp.stack([samples, samples * 2])

    # Default is two-sided for complex
    f, t, Sxx = spectral.spectrogram(
        samples_mimo, sampling_rate=fs, nperseg=256, noverlap=128
    )

    assert isinstance(f, xp.ndarray)
    assert isinstance(t, xp.ndarray)
    assert isinstance(Sxx, xp.ndarray)

    # Shapes: (num_channels, len(f), len(t))
    # 256 nperseg -> 256 frequency bins (two-sided)
    assert len(f) == 256
    assert Sxx.shape == (2, 256, len(t))

    # Frequency range should be two-sided (centered around 0 due to fftshift)
    assert xp.min(f) < 0
    assert xp.max(f) > 0

    # Test that error is raised when requesting return_onesided=True for complex data
    with pytest.raises(ValueError, match="Cannot compute one-sided spectrogram"):
        spectral.spectrogram(samples_mimo, sampling_rate=fs, return_onesided=True)


# -----------------------------------------------------------------------------
# Pilot tone injection
# -----------------------------------------------------------------------------


class TestAddPilotTone:
    """Tests for spectral.add_pilot_tone (CW pilot-tone injection)."""

    @staticmethod
    def _signal(xp, N=4096, seed=0):
        rng = xp.random.RandomState(seed)
        x = (rng.randn(N) + 1j * rng.randn(N)).astype(xp.complex128)
        return x / xp.sqrt(xp.mean(xp.abs(x) ** 2))  # unit average power

    @pytest.mark.parametrize("psr_db", [-20.0, -10.0, 0.0])
    def test_power_ratio(self, backend_device, xp, psr_db):
        """Added tone power matches the requested pilot-to-signal ratio."""
        fs = 100.0
        x = self._signal(xp)
        p_sig = float(xp.mean(xp.abs(x) ** 2))
        y, _ = spectral.add_pilot_tone(x, fs, 30.0, power_ratio_db=psr_db)
        # The added tone is exactly y - x; measure its power directly (the
        # signal/tone cross-correlation makes total-minus-signal unreliable at low PSR).
        p_tone = float(xp.mean(xp.abs(y - x) ** 2))
        assert abs(10 * np.log10(p_tone / p_sig) - psr_db) < 0.05

    def test_peak_location(self, backend_device, xp):
        """The injected tone shows up as the dominant spectral peak at the snapped f_p."""
        fs = 100.0
        N = 4096
        f_p = 30.0
        x = self._signal(xp, N=N)
        y, f_actual = spectral.add_pilot_tone(x, fs, f_p, power_ratio_db=10.0)
        freqs = xp.fft.fftfreq(N, d=1.0 / fs)
        k = int(xp.argmax(xp.abs(xp.fft.fft(y))))
        assert abs(float(freqs[k]) - f_actual) < fs / N

    def test_snaps_to_grid(self, backend_device, xp):
        """Returned frequency lies exactly on the f_s/N grid, near the request."""
        fs = 100.0
        N = 4096
        df = fs / N
        x = self._signal(xp, N=N)
        f_req = 30.0 + 0.4 * df  # deliberately off-grid
        _, f_actual = spectral.add_pilot_tone(x, fs, f_req)
        # On-grid: an integer number of bins from DC.
        assert abs(round(f_actual / df) - f_actual / df) < 1e-9
        assert abs(f_actual - f_req) <= df / 2 + 1e-9

    def test_dtype_and_shape_preserved(self, backend_device, xp):
        """complex64 stays complex64; SISO/MIMO shapes are preserved."""
        fs = 100.0
        x64 = self._signal(xp).astype(xp.complex64)
        y, _ = spectral.add_pilot_tone(x64, fs, 30.0)
        assert y.dtype == xp.complex64
        assert y.shape == x64.shape

        mimo = xp.stack([self._signal(xp), 2 * self._signal(xp, seed=1)])
        ym, _ = spectral.add_pilot_tone(mimo, fs, 30.0)
        assert ym.shape == mimo.shape

    def test_renormalize_preserves_power(self, backend_device, xp):
        """renormalize=True restores each channel's original mean power."""
        fs = 100.0
        mimo = xp.stack([self._signal(xp), 2 * self._signal(xp, seed=1)])
        p_in = xp.mean(xp.abs(mimo) ** 2, axis=-1)
        y, _ = spectral.add_pilot_tone(
            mimo, fs, 30.0, power_ratio_db=-6.0, renormalize=True
        )
        p_out = xp.mean(xp.abs(y) ** 2, axis=-1)
        assert bool(xp.allclose(p_in, p_out, rtol=1e-4))

    def test_invalid_frequency_raises(self, backend_device, xp):
        """Tone frequency outside (-fs/2, fs/2) raises ValueError."""
        fs = 100.0
        x = self._signal(xp)
        with pytest.raises(ValueError, match=r"must lie in \(-fs/2, fs/2\)"):
            spectral.add_pilot_tone(x, fs, fs)

    def test_scalar_returns_float(self, backend_device, xp):
        """Scalar frequency returns a plain float (back-compat), even for MIMO."""
        fs = 100.0
        mimo = xp.stack([self._signal(xp), self._signal(xp, seed=1)])
        _, f_actual = spectral.add_pilot_tone(mimo, fs, 30.0)
        assert isinstance(f_actual, float)

    def test_per_channel_frequencies(self, backend_device, xp):
        """A per-channel list places one distinct tone per channel at its bin."""
        fs = 100.0
        N = 4096
        f_req = [20.0, -35.0]
        mimo = xp.stack([self._signal(xp, N=N), self._signal(xp, N=N, seed=1)])
        y, f_actual = spectral.add_pilot_tone(mimo, fs, f_req, power_ratio_db=10.0)
        assert isinstance(f_actual, list) and len(f_actual) == 2
        freqs = xp.fft.fftfreq(N, d=1.0 / fs)
        for c in range(2):
            k = int(xp.argmax(xp.abs(xp.fft.fft(y[c]))))
            assert abs(float(freqs[k]) - f_actual[c]) < fs / N
            assert abs(f_actual[c] - f_req[c]) <= fs / N / 2 + 1e-9
        # Each channel's dominant tone sits at a *different* bin.
        assert f_actual[0] != f_actual[1]

    def test_per_channel_length_mismatch_raises(self, backend_device, xp):
        """A per-channel sequence whose length != C raises ValueError."""
        fs = 100.0
        mimo = xp.stack([self._signal(xp), self._signal(xp, seed=1)])
        with pytest.raises(ValueError, match=r"one frequency per channel"):
            spectral.add_pilot_tone(mimo, fs, [20.0, -30.0, 10.0])

    def test_per_channel_invalid_frequency_raises(self, backend_device, xp):
        """An out-of-range entry in a per-channel sequence raises ValueError."""
        fs = 100.0
        mimo = xp.stack([self._signal(xp), self._signal(xp, seed=1)])
        with pytest.raises(ValueError, match=r"must lie in \(-fs/2, fs/2\)"):
            spectral.add_pilot_tone(mimo, fs, [20.0, fs])

    def test_per_channel_power_ratio(self, backend_device, xp):
        """A per-channel PSR sequence realises a distinct tone power per channel."""
        fs = 100.0
        psr = [-10.0, -20.0]
        mimo = xp.stack([self._signal(xp), self._signal(xp, seed=1)])
        y, _ = spectral.add_pilot_tone(mimo, fs, [20.0, -35.0], power_ratio_db=psr)
        for c in range(2):
            p_sig = float(xp.mean(xp.abs(mimo[c]) ** 2))
            p_tone = float(xp.mean(xp.abs(y[c] - mimo[c]) ** 2))
            assert abs(10 * np.log10(p_tone / p_sig) - psr[c]) < 0.05

    def test_per_channel_power_length_mismatch_raises(self, backend_device, xp):
        """A per-channel PSR sequence whose length != C raises ValueError."""
        fs = 100.0
        mimo = xp.stack([self._signal(xp), self._signal(xp, seed=1)])
        with pytest.raises(ValueError, match=r"one PSR per channel"):
            spectral.add_pilot_tone(mimo, fs, [20.0, -30.0], power_ratio_db=[-10.0])
