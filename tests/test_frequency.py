"""Tests for Frequency Offset Estimation (FOE) algorithms in commkit.frequency."""

import numpy as np
import pytest

from commkit import frequency, generate_psk, generate_qam, spectral
from commkit.impairments import apply_awgn

# -----------------------------------------------------------------------------
# Test helpers
# -----------------------------------------------------------------------------

FS = 1e6  # 1 MHz sampling rate, common to all tests
SNR_DB = 30  # generous SNR so numerical algorithms converge reliably


def _qam_signal(xp, order, n_symbols, fo_hz=0.0, snr_db=SNR_DB, fs=FS, seed=42):
    """Generate a 1-SPS QAM signal with optional frequency offset and AWGN."""
    sig = generate_qam(
        order=order, num_symbols=n_symbols, sps=1, symbol_rate=fs, seed=seed
    )
    sig.samples = apply_awgn(sig.samples, esn0_db=snr_db, sps=1, seed=seed)
    if fo_hz != 0.0:
        sig.samples, _ = spectral.shift_frequency(sig.samples, fo_hz, fs)
    return sig


def _psk_signal(xp, order, n_symbols, fo_hz=0.0, snr_db=SNR_DB, fs=FS, seed=42):
    """Generate a 1-SPS PSK signal with optional frequency offset and AWGN."""
    sig = generate_psk(
        order=order, num_symbols=n_symbols, sps=1, symbol_rate=fs, seed=seed
    )
    sig.samples = apply_awgn(sig.samples, esn0_db=snr_db, sps=1, seed=seed)
    if fo_hz != 0.0:
        sig.samples, _ = spectral.shift_frequency(sig.samples, fo_hz, fs)
    return sig


# -----------------------------------------------------------------------------
# FOE - M-th Power
# -----------------------------------------------------------------------------


class TestFoeMthPower:
    @pytest.mark.parametrize("order", [4, 16, 64])
    @pytest.mark.parametrize("fo_hz", [5_000.0, -8_000.0, 15_000.0])
    def test_accuracy_qam(self, backend_device, xp, order, fo_hz):
        """Estimated offset within 5% of true value for QAM at SNR=30 dB."""
        sig = _qam_signal(xp, order, 4096, fo_hz=fo_hz)
        est = frequency.estimate_frequency_offset_mth_power(
            sig.samples, sampling_rate=FS, modulation="qam", order=order
        )
        assert abs(est - fo_hz) / abs(fo_hz) < 0.05

    @pytest.mark.parametrize("order", [4, 8])
    @pytest.mark.parametrize("fo_hz", [3_000.0, -6_000.0])
    def test_accuracy_psk(self, backend_device, xp, order, fo_hz):
        """Estimated offset within 5% of true value for PSK at SNR=30 dB."""
        sig = _psk_signal(xp, order, 4096, fo_hz=fo_hz)
        est = frequency.estimate_frequency_offset_mth_power(
            sig.samples, sampling_rate=FS, modulation="psk", order=order
        )
        assert abs(est - fo_hz) / abs(fo_hz) < 0.05

    def test_zero_offset_within_lock_range(self, backend_device, xp):
        """With no frequency offset, estimate stays within the lock range [-fs/2M, fs/2M]."""
        sig = _qam_signal(xp, 16, 4096, fo_hz=0.0)
        est = frequency.estimate_frequency_offset_mth_power(
            sig.samples, sampling_rate=FS, modulation="qam", order=16
        )
        # Lock range for QAM with M=4: [-fs/8, fs/8] = ±125 kHz at 1 MHz
        assert abs(est) < FS / (2 * 4)

    def test_search_range_rejects_out_of_range(self, backend_device, xp):
        """Peak outside search_range returns an estimate outside the true value."""
        # True offset is 20 kHz; search range covers only [-5 kHz, 5 kHz]
        sig = _qam_signal(xp, 4, 8192, fo_hz=20_000.0)
        est = frequency.estimate_frequency_offset_mth_power(
            sig.samples,
            sampling_rate=FS,
            modulation="qam",
            order=4,
            search_range=(-5_000.0, 5_000.0),
        )
        # The true offset must not be found - estimate stays within the window
        assert abs(est) <= 5_000.0 + 200.0  # small tolerance for bin quantization

    def test_search_range_empty_raises(self, backend_device, xp):
        """Empty search_range raises ValueError."""
        sig = _qam_signal(xp, 4, 1024, fo_hz=0.0)
        with pytest.raises(ValueError, match="empty search window"):
            frequency.estimate_frequency_offset_mth_power(
                sig.samples,
                sampling_rate=FS,
                modulation="qam",
                order=4,
                search_range=(400_000.0, 500_000.0),
            )

    def test_mimo_returns_per_channel(self, backend_device, xp):
        """MIMO input (C, N) returns ndarray(C,) by default."""
        sig_a = _qam_signal(xp, 4, 2048, fo_hz=5_000.0)
        sig_b = _qam_signal(xp, 4, 2048, fo_hz=5_000.0)
        mimo = xp.stack([sig_a.samples, sig_b.samples], axis=0)  # (2, N)
        est = frequency.estimate_frequency_offset_mth_power(
            mimo, sampling_rate=FS, modulation="qam", order=4
        )
        assert isinstance(est, np.ndarray)
        assert est.shape == (2,)
        assert all(abs(e - 5_000.0) / 5_000.0 < 0.05 for e in est)

    def test_mimo_combine_channels_returns_scalar(self, backend_device, xp):
        """MIMO + combine_channels=True returns a single Python float."""
        sig_a = _qam_signal(xp, 4, 2048, fo_hz=5_000.0)
        sig_b = _qam_signal(xp, 4, 2048, fo_hz=5_000.0)
        mimo = xp.stack([sig_a.samples, sig_b.samples], axis=0)
        est = frequency.estimate_frequency_offset_mth_power(
            mimo, sampling_rate=FS, modulation="qam", order=4, combine_channels=True
        )
        assert isinstance(est, float)
        assert abs(est - 5_000.0) / 5_000.0 < 0.05

    def test_circular_neighbors_near_lock_edge(self, backend_device, xp):
        """Peak near the lock-range boundary uses circular neighbors without clamping bias."""
        # For QPSK (M=4), lock range is ±fs/8. Place tone at 90% of the boundary so
        # the M-th power tone lands near bin nfft-2 (high-frequency edge of spectrum).
        fo_hz = 0.90 * (FS / 8)
        sig = _qam_signal(xp, 4, 4096, fo_hz=fo_hz, snr_db=40)
        est = frequency.estimate_frequency_offset_mth_power(
            sig.samples, sampling_rate=FS, modulation="qam", order=4
        )
        assert abs(est - fo_hz) / abs(fo_hz) < 0.10

    def test_vectorised_subbin_matches_scalar_c2(self, backend_device, xp):
        """Vectorised sub-bin path: C=2 channels with same offset, both estimated accurately.

        MIMO accumulates channels coherently (shared k_safe); the vectorisation changes
        only how mu is computed per-channel, not the shared argmax.  Use different seeds
        so each channel has independent noise while sharing the same carrier offset.
        """
        fo_hz = 8_000.0
        sig_a = _qam_signal(xp, 4, 4096, fo_hz=fo_hz, seed=42)
        sig_b = _qam_signal(xp, 4, 4096, fo_hz=fo_hz, seed=99)
        mimo = xp.stack([sig_a.samples, sig_b.samples], axis=0)
        est = frequency.estimate_frequency_offset_mth_power(
            mimo, sampling_rate=FS, modulation="qam", order=4
        )
        assert abs(est[0] - fo_hz) / abs(fo_hz) < 0.05
        assert abs(est[1] - fo_hz) / abs(fo_hz) < 0.05


# -----------------------------------------------------------------------------
# FOE - Correction functions (frequency half)
# -----------------------------------------------------------------------------


class TestCorrectionFunctions:
    def test_correct_static_frequency_offset_dtype_preserved(self, backend_device, xp):
        """correct_static_frequency_offset: complex64 input -> complex64 output."""
        sig = _qam_signal(xp, 4, 1024)
        assert sig.samples.dtype == xp.complex64
        corrected = frequency.correct_static_frequency_offset(
            sig.samples, offset=5_000.0, sampling_rate=FS
        )
        assert corrected.dtype == xp.complex64

    def test_correct_static_frequency_offset_roundtrip(self, backend_device, xp):
        """Applying +Δf then correcting with -Δf restores the signal."""
        sig = _qam_signal(xp, 4, 1024)
        original = sig.samples.copy()
        # shift_frequency quantizes to the nearest bin; capture actual offset so
        # the correction can cancel it exactly (no residual due to quantization)
        shifted, actual_fo = spectral.shift_frequency(sig.samples, 10_000.0, FS)
        restored = frequency.correct_static_frequency_offset(
            shifted, offset=actual_fo, sampling_rate=FS
        )
        assert float(xp.max(xp.abs(restored - original))) < 1e-4

    def test_correct_static_frequency_offset_size_one_array(self, backend_device, xp):
        """A size-1 (non-0-d) offset array (e.g. a single-channel per-channel
        estimate, shape (1,)) must not crash the scalar float() cast - every
        size-1 shape is equivalent to a plain scalar offset."""
        sig = _qam_signal(xp, 4, 1024)
        scalar = frequency.correct_static_frequency_offset(
            sig.samples, offset=5_000.0, sampling_rate=FS
        )
        one_d = frequency.correct_static_frequency_offset(
            sig.samples, offset=xp.asarray([5_000.0]), sampling_rate=FS
        )
        zero_d = frequency.correct_static_frequency_offset(
            sig.samples, offset=xp.asarray(5_000.0), sampling_rate=FS
        )
        assert float(xp.max(xp.abs(one_d - scalar))) < 1e-9
        assert float(xp.max(xp.abs(zero_d - scalar))) < 1e-9


# -----------------------------------------------------------------------------
# FOE - Scattered Pilots (phase slope)
# -----------------------------------------------------------------------------


class TestFoePilots:
    """Tests for estimate_frequency_offset_pilot_symbols (least-squares phase slope)."""

    N_SAMPLES = 4096
    PILOT_PERIOD = 16  # one pilot every 16 samples -> lock range ±31.25 kHz

    @staticmethod
    def _setup(xp, fo_hz, n_samples=4096, pilot_period=16, snr_db=SNR_DB, fs=FS):
        """
        Build a QPSK-like signal with BPSK (+1) pilots inserted at a regular
        grid, apply an exact frequency offset, and return everything needed
        by the estimator.
        """
        rng = np.random.default_rng(seed=7)
        # Random QPSK-like data (4-point alphabet, unit power)
        symbols = (
            rng.choice([-1, 1], size=n_samples)
            + 1j * rng.choice([-1, 1], size=n_samples)
        ).astype(np.complex64) / np.sqrt(2)

        pilot_indices = np.arange(0, n_samples, pilot_period, dtype=np.intp)
        pilot_values = np.ones(len(pilot_indices), dtype=np.complex64)  # BPSK pilots
        symbols[pilot_indices] = pilot_values  # overwrite data with known pilots

        if snr_db < 100:
            noise_power = 10 ** (-snr_db / 10)
            noise = (
                rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)
            ).astype(np.complex64) * np.sqrt(noise_power / 2)
            symbols = (symbols + noise).astype(np.complex64)

        samples = xp.asarray(symbols)
        # Apply exact frequency offset (no bin quantization)
        t = xp.arange(n_samples, dtype=xp.float64) / fs
        samples = samples * xp.exp(1j * 2 * np.pi * fo_hz * t).astype(xp.complex64)
        return samples, pilot_indices, pilot_values

    @pytest.mark.parametrize("fo_hz", [1_000.0, 5_000.0, -3_000.0])
    def test_accuracy(self, backend_device, xp, fo_hz):
        """Estimated offset within 1 % of true offset at 30 dB SNR."""
        samples, pilot_indices, pilot_values = self._setup(xp, fo_hz)
        est = frequency.estimate_frequency_offset_pilot_symbols(
            samples,
            pilot_indices=pilot_indices,
            pilot_values=pilot_values,
            sampling_rate=FS,
        )
        assert abs(est - fo_hz) < 0.01 * abs(fo_hz) + 1.0

    def test_zero_offset(self, backend_device, xp):
        """Zero frequency offset: estimate is within ±20 Hz."""
        samples, pilot_indices, pilot_values = self._setup(xp, fo_hz=0.0)
        est = frequency.estimate_frequency_offset_pilot_symbols(
            samples,
            pilot_indices=pilot_indices,
            pilot_values=pilot_values,
            sampling_rate=FS,
        )
        assert abs(est) < 20.0

    def test_mimo_returns_per_channel(self, backend_device, xp):
        """MIMO input returns ndarray(C,) by default."""
        fo_hz = 2_000.0
        s0, pilot_indices, pilot_values = self._setup(xp, fo_hz)
        s1, _, _ = self._setup(xp, fo_hz)
        samples_mimo = xp.stack([s0, s1], axis=0)  # (2, N)
        est = frequency.estimate_frequency_offset_pilot_symbols(
            samples_mimo,
            pilot_indices=pilot_indices,
            pilot_values=pilot_values,
            sampling_rate=FS,
        )
        assert isinstance(est, np.ndarray)
        assert est.shape == (2,)
        assert all(abs(e - fo_hz) < 0.01 * fo_hz + 1.0 for e in est)

    def test_mimo_combine_channels_returns_scalar(self, backend_device, xp):
        """MIMO + combine_channels=True returns a single Python float."""
        fo_hz = 2_000.0
        s0, pilot_indices, pilot_values = self._setup(xp, fo_hz)
        s1, _, _ = self._setup(xp, fo_hz)
        samples_mimo = xp.stack([s0, s1], axis=0)
        est = frequency.estimate_frequency_offset_pilot_symbols(
            samples_mimo,
            pilot_indices=pilot_indices,
            pilot_values=pilot_values,
            sampling_rate=FS,
            combine_channels=True,
        )
        assert isinstance(est, float)
        assert abs(est - fo_hz) < 0.01 * fo_hz + 1.0


# -----------------------------------------------------------------------------
# FOE - Mengali-Morelli multi-lag
# -----------------------------------------------------------------------------


class TestFoeMengaliMorelli:
    """Tests for estimate_frequency_offset_mengali_morelli."""

    @pytest.mark.parametrize("fo_hz", [5_000.0, -12_000.0, 30_000.0])
    @pytest.mark.parametrize("order", [4, 16])
    def test_blind_qam_accuracy(self, backend_device, xp, fo_hz, order):
        """Blind QAM mode: estimate within 2 % of true offset at 30 dB SNR."""
        sig = _qam_signal(xp, order, 4096, fo_hz=0.0)  # generate without offset
        # Apply exact frequency offset via direct complex mixing
        n = xp.arange(sig.samples.shape[-1], dtype=xp.float64)
        sig.samples = (sig.samples * xp.exp(1j * 2 * np.pi * fo_hz / FS * n)).astype(
            sig.samples.dtype
        )
        est = frequency.estimate_frequency_offset_mengali_morelli(
            sig.samples, sampling_rate=FS, modulation="qam", order=order
        )
        assert abs(est - fo_hz) / abs(fo_hz) < 0.02

    @pytest.mark.parametrize("fo_hz", [4_000.0, -8_000.0])
    def test_data_aided_accuracy(self, backend_device, xp, fo_hz):
        """Data-aided mode: estimate within 1 % using known reference (exact mixing)."""
        sig = generate_psk(order=4, num_symbols=4096, sps=1, symbol_rate=FS)
        ideal = sig.samples.copy()
        sig.samples = apply_awgn(sig.samples, esn0_db=25, sps=1)
        n = xp.arange(sig.samples.shape[-1], dtype=xp.float64)
        sig.samples = (sig.samples * xp.exp(1j * 2 * np.pi * fo_hz / FS * n)).astype(
            sig.samples.dtype
        )
        est = frequency.estimate_frequency_offset_mengali_morelli(
            sig.samples, sampling_rate=FS, ref_signal=ideal
        )
        assert abs(est - fo_hz) / abs(fo_hz) < 0.01

    def test_large_offset_near_nyquist(self, backend_device, xp):
        """M&M lock range [-fs/2, fs/2]: succeeds at 40 % of Nyquist where Kay wraps."""
        N = 4096
        fo_hz = 0.40 * FS  # 40 % of sampling rate
        n = xp.arange(N, dtype=xp.float64)
        tone = xp.exp(1j * 2 * np.pi * fo_hz / FS * n).astype(xp.complex64)
        # Generic blind mode (no modulation - pure tone)
        est = frequency.estimate_frequency_offset_mengali_morelli(
            tone, sampling_rate=FS
        )
        assert abs(est - fo_hz) < 0.02 * fo_hz

    def test_generic_blind_pure_tone(self, backend_device, xp):
        """Generic mode (no modulation): pure tone estimated accurately."""
        N = 2048
        fo_hz = 7_500.0
        n = xp.arange(N, dtype=xp.float64)
        tone = xp.exp(1j * 2 * np.pi * fo_hz / FS * n).astype(xp.complex64)
        est = frequency.estimate_frequency_offset_mengali_morelli(
            tone, sampling_rate=FS
        )
        assert abs(est - fo_hz) < 500.0

    def test_mimo_returns_per_channel(self, backend_device, xp):
        """MIMO (C, N) input returns ndarray(C,) by default."""
        fo_hz = 6_000.0
        sig_a = _qam_signal(xp, 4, 2048, fo_hz=0.0)
        sig_b = _qam_signal(xp, 4, 2048, fo_hz=0.0)
        n = xp.arange(2048, dtype=xp.float64)
        mixer = xp.exp(1j * 2 * np.pi * fo_hz / FS * n).astype(xp.complex64)
        mimo = xp.stack([sig_a.samples * mixer, sig_b.samples * mixer], axis=0)
        est = frequency.estimate_frequency_offset_mengali_morelli(
            mimo, sampling_rate=FS, modulation="qam", order=4
        )
        assert isinstance(est, np.ndarray)
        assert est.shape == (2,)
        assert all(abs(e - fo_hz) / fo_hz < 0.02 for e in est)

    def test_mimo_combine_channels_returns_scalar(self, backend_device, xp):
        """MIMO + combine_channels=True returns a single Python float."""
        fo_hz = 6_000.0
        sig_a = _qam_signal(xp, 4, 2048, fo_hz=0.0)
        sig_b = _qam_signal(xp, 4, 2048, fo_hz=0.0)
        n = xp.arange(2048, dtype=xp.float64)
        mixer = xp.exp(1j * 2 * np.pi * fo_hz / FS * n).astype(xp.complex64)
        mimo = xp.stack([sig_a.samples * mixer, sig_b.samples * mixer], axis=0)
        est = frequency.estimate_frequency_offset_mengali_morelli(
            mimo, sampling_rate=FS, modulation="qam", order=4, combine_channels=True
        )
        assert isinstance(est, float)
        assert abs(est - fo_hz) / fo_hz < 0.02

    def test_custom_max_lag(self, backend_device, xp):
        """Custom max_lag parameter: still converges to correct estimate (exact mixing)."""
        fo_hz = 3_000.0
        sig = _qam_signal(xp, 4, 2048, fo_hz=0.0)
        n = xp.arange(sig.samples.shape[-1], dtype=xp.float64)
        sig.samples = (sig.samples * xp.exp(1j * 2 * np.pi * fo_hz / FS * n)).astype(
            sig.samples.dtype
        )
        est = frequency.estimate_frequency_offset_mengali_morelli(
            sig.samples, sampling_rate=FS, modulation="qam", order=4, max_lag=16
        )
        assert abs(est - fo_hz) / fo_hz < 0.05


# -----------------------------------------------------------------------------
# FOE - Regression: Jacobsen interpolation, WLSQ pilots
# -----------------------------------------------------------------------------


class TestFoeRegression:
    def test_jacobsen_vs_parabolic_accuracy(self, backend_device, xp):
        """Jacobsen interpolation accuracy is at least as good as parabolic for N=256."""
        fo_hz = 7_777.0  # non-round number to stress sub-bin interpolation
        sig = _qam_signal(xp, 4, 256, fo_hz=fo_hz)
        est_j = frequency.estimate_frequency_offset_mth_power(
            sig.samples,
            sampling_rate=FS,
            modulation="qam",
            order=4,
            interpolation="jacobsen",
        )
        est_p = frequency.estimate_frequency_offset_mth_power(
            sig.samples,
            sampling_rate=FS,
            modulation="qam",
            order=4,
            interpolation="parabolic",
        )
        # Jacobsen error must be ≤ parabolic error (with generous 20 % slack for noise)
        assert abs(est_j - fo_hz) <= abs(est_p - fo_hz) * 1.2 + 100.0

    def test_mth_power_short_signal_raises(self, backend_device, xp):
        """M-th power FOE raises ValueError for signals shorter than 8 samples."""
        short = xp.ones(5, dtype=xp.complex64)
        with pytest.raises(ValueError, match="too short"):
            frequency.estimate_frequency_offset_mth_power(
                short, sampling_rate=FS, modulation="qam", order=4
            )

    def test_pilot_wlsq_vs_ols_at_snr(self, backend_device, xp):
        """WLSQ pilot FOE returns a valid estimate; result within 2% of true offset."""
        fo_hz = 5_000.0
        n_samples = 2048
        pilot_period = 8
        rng = np.random.default_rng(42)
        symbols = (
            rng.choice([-1, 1], size=n_samples)
            + 1j * rng.choice([-1, 1], size=n_samples)
        ).astype(np.complex64) / np.sqrt(2)
        pilot_indices = np.arange(0, n_samples, pilot_period, dtype=np.intp)
        pilot_values = np.ones(len(pilot_indices), dtype=np.complex64)
        symbols[pilot_indices] = pilot_values
        noise = (
            rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)
        ).astype(np.complex64) * np.sqrt(10 ** (-SNR_DB / 10) / 2)
        symbols = (symbols + noise).astype(np.complex64)
        t = np.arange(n_samples, dtype=np.float64) / FS
        samples = xp.asarray(
            symbols * np.exp(1j * 2 * np.pi * fo_hz * t).astype(np.complex64)
        )
        est = frequency.estimate_frequency_offset_pilot_symbols(
            samples,
            pilot_indices=pilot_indices,
            pilot_values=pilot_values,
            sampling_rate=FS,
            snr_weighted=True,
        )
        assert abs(est - fo_hz) / fo_hz < 0.02


# =============================================================================
# CORRECT FREQUENCY OFFSET - REAL AND MIMO BRANCHES
# =============================================================================


class TestCorrectFrequencyOffsetBranches:
    """Branches for real-valued and MIMO input in correct_static_frequency_offset."""

    def test_real_float32_input(self, backend_device, xp):
        """Real float32 input is cast to complex64 and frequency-corrected."""

        N = 256
        t = xp.arange(N, dtype=xp.float32) / 1e6
        # Simple real cosine as stand-in for a real-baseband signal
        sig = xp.cos(t)
        sig = sig.astype(xp.float32)
        out = frequency.correct_static_frequency_offset(
            sig, offset=5000.0, sampling_rate=1e6
        )
        assert out.dtype == xp.complex64
        assert out.shape == sig.shape

    def test_mimo_input_broadcasts_mixer(self, backend_device, xp):
        """MIMO (C, N) input: mixer is broadcast over channels without error."""
        import numpy as np

        C, N = 2, 256
        rng = np.random.default_rng(99)
        sig = xp.asarray(
            (rng.standard_normal((C, N)) + 1j * rng.standard_normal((C, N))).astype(
                np.complex64
            )
        )
        out = frequency.correct_static_frequency_offset(
            sig, offset=3000.0, sampling_rate=1e6
        )
        assert out.shape == (C, N)
        assert out.dtype == xp.complex64


# -----------------------------------------------------------------------------
# FOE - find_bias_tone
# -----------------------------------------------------------------------------


class TestFindBiasTone:
    """Tests for find_bias_tone (log-parabolic CW tone locator)."""

    @pytest.mark.parametrize(
        "fs,tone_hz",
        [
            (1e6, 100e3),  # 100 kHz at 1 MHz  -> bin ~409  (well away from DC/Nyquist)
            (1e9, 250e6),  # 250 MHz at 1 GHz  -> bin ~1024
            (2.5e9, 500e6),  # 500 MHz at 2.5 GHz -> bin ~819
            (2.5e9, -400e6),  # -400 MHz at 2.5 GHz -> bin ~3424 (negative freq)
        ],
    )
    def test_pure_cw_within_one_bin(self, backend_device, xp, fs, tone_hz):
        """Recovered frequency is within 1 FFT bin of the true tone (N=4096)."""
        N = 4096
        nfft = 1 << int(np.ceil(np.log2(N)))
        bin_width = fs / nfft
        n = xp.arange(N, dtype=xp.float64)
        seg = xp.exp(1j * 2 * np.pi * tone_hz / fs * n).astype(xp.complex64)
        est = frequency.find_bias_tone(seg, sampling_rate=fs)
        assert abs(est - tone_hz) < bin_width

    def test_log_parabolic_beats_argmax(self, backend_device, xp):
        """Log-parabolic interpolation is more accurate than argmax-only for a mid-bin tone."""
        fs = 1e6
        N = 4096
        nfft = 1 << int(np.ceil(np.log2(N)))
        bin_width = fs / nfft
        # Place tone at 0.3 bins above an integer bin (maximal argmax bias region)
        tone_hz = 10.0 * bin_width + 0.3 * bin_width
        n_np = np.arange(N, dtype=np.float64)
        seg_np = np.exp(1j * 2 * np.pi * tone_hz / fs * n_np).astype(np.complex64)

        # Argmax-only reference (nearest bin centre, no interpolation)
        X_np = np.fft.fft(seg_np, n=nfft)
        freqs_np = np.fft.fftfreq(nfft, d=1.0 / fs)
        k_peak = int(np.argmax(np.abs(X_np)))
        est_argmax = float(freqs_np[k_peak])

        seg_xp = xp.asarray(seg_np)
        est_interp = frequency.find_bias_tone(seg_xp, sampling_rate=fs)

        assert abs(est_interp - tone_hz) < abs(est_argmax - tone_hz)

    def test_search_window_rejects_stronger_interferer(self, backend_device, xp):
        """target_frequency + search_band rejects a 10x stronger out-of-band tone."""
        fs = 1e9
        N = 4096
        n = xp.arange(N, dtype=xp.float64)
        target = xp.exp(1j * 2 * np.pi * 100e6 / fs * n).astype(xp.complex128)
        interferer = 10.0 * xp.exp(1j * 2 * np.pi * 400e6 / fs * n).astype(
            xp.complex128
        )
        seg = (target + interferer).astype(xp.complex64)
        est = frequency.find_bias_tone(
            seg, sampling_rate=fs, target_frequency=100e6, search_band=50e6
        )
        assert abs(est - 100e6) < 10e6

    def test_returns_python_float(self, backend_device, xp):
        """find_bias_tone always returns a Python float, not an array."""
        N = 512
        n = xp.arange(N, dtype=xp.float64)
        seg = xp.exp(1j * 2 * np.pi * 1e5 / 1e6 * n).astype(xp.complex64)
        result = frequency.find_bias_tone(seg, sampling_rate=1e6)
        assert isinstance(result, float)

    def test_partial_search_params_raises(self, backend_device, xp):
        """Providing only one of target_frequency / search_band raises ValueError."""
        N = 256
        n = xp.arange(N, dtype=xp.float64)
        seg = xp.exp(1j * 2 * np.pi * 1e5 / 1e6 * n).astype(xp.complex64)
        with pytest.raises(ValueError, match="both be provided or both omitted"):
            frequency.find_bias_tone(seg, sampling_rate=1e6, target_frequency=1e5)
        with pytest.raises(ValueError, match="both be provided or both omitted"):
            frequency.find_bias_tone(seg, sampling_rate=1e6, search_band=10e3)

    def test_empty_search_window_raises(self, backend_device, xp):
        """Search window outside the FFT grid raises ValueError."""
        N = 128
        n = xp.arange(N, dtype=xp.float64)
        seg = xp.exp(1j * 2 * np.pi * 1e5 / 1e6 * n).astype(xp.complex64)
        with pytest.raises(ValueError, match="empty search window"):
            frequency.find_bias_tone(
                seg,
                sampling_rate=1e6,
                target_frequency=2e6,  # beyond Nyquist for fs=1 MHz
                search_band=100.0,
            )

    @pytest.mark.parametrize("tone_hz", [50e3, 200e3])
    def test_gpu_matches_cpu_within_1hz(self, backend_device, xp, tone_hz):
        """GPU and CPU backends agree to within 1 Hz on the same segment."""
        fs = 1e6
        N = 4096
        n_np = np.arange(N, dtype=np.float64)
        seg_np = np.exp(1j * 2 * np.pi * tone_hz / fs * n_np).astype(np.complex64)
        est_cpu = frequency.find_bias_tone(seg_np, sampling_rate=fs)
        est_dev = frequency.find_bias_tone(xp.asarray(seg_np), sampling_rate=fs)
        assert abs(est_cpu - est_dev) < 1.0

    def test_circular_neighbors_at_nyquist_edge(self, backend_device, xp):
        """Tone near the Nyquist edge (bin nfft-2) is estimated without clamping bias."""
        fs = 1e6
        N = 512
        nfft = 1 << int(np.ceil(np.log2(N)))
        bin_width = fs / nfft
        # Place tone 2 bins below Nyquist - peak at nfft-2, neighbors wrap to nfft-3 and nfft-1
        tone_hz = -(fs / 2) + 2 * bin_width
        n_np = np.arange(N, dtype=np.float64)
        seg = xp.asarray(
            np.exp(1j * 2 * np.pi * tone_hz / fs * n_np).astype(np.complex64)
        )
        est = frequency.find_bias_tone(seg, sampling_rate=fs)
        assert abs(est - tone_hz) < 2 * bin_width


# -----------------------------------------------------------------------------
# FOE - correct_frequency_offset_blockwise
# -----------------------------------------------------------------------------


class TestCorrectFrequencyOffsetBlockwise:
    """Tests for correct_frequency_offset_blockwise."""

    FS = 1e6

    def test_corrects_constant_offset(self, backend_device, xp):
        """Oracle estimator removes a known constant offset; residual is near zero."""
        fs = self.FS
        fo_hz = 10_000.0
        N = 4096
        sig = _qam_signal(xp, 4, N, fo_hz=fo_hz)
        corrected = frequency.correct_frequency_offset_blockwise(
            sig.samples,
            fs,
            block_size=512,
            overlap=0.5,
            estimator=lambda block, _fs: fo_hz,
        )
        # Residual FOE on corrected signal should be within 500 Hz
        residual = frequency.estimate_frequency_offset_mth_power(
            corrected, sampling_rate=fs, modulation="qam", order=4
        )
        assert abs(residual) < 500.0

    def test_output_on_same_device(self, backend_device, xp):
        """Output array is on the same backend as the input."""
        sig = xp.ones(2048, dtype=xp.complex64)
        out = frequency.correct_frequency_offset_blockwise(
            sig,
            self.FS,
            block_size=512,
            overlap=0.5,
            estimator=lambda b, f: 0.0,
        )
        assert type(out) is type(sig)
        assert out.shape == sig.shape

    def test_callable_called_once_per_block_per_channel(self, backend_device, xp):
        """Estimator is called exactly B times for SISO (once per block)."""
        N = 4096
        block_size = 512
        overlap = 0.5
        step = max(1, round(block_size * (1.0 - overlap)))
        expected_calls = len(list(range(0, N - block_size + 1, step)))
        call_log = []

        def counting_estimator(block, _fs):
            call_log.append(1)
            return 0.0

        frequency.correct_frequency_offset_blockwise(
            xp.zeros(N, dtype=xp.complex64),
            self.FS,
            block_size=block_size,
            overlap=overlap,
            estimator=counting_estimator,
        )
        assert len(call_log) == expected_calls

    def test_pchip_no_overshoot_monotone_input(self, backend_device, xp):
        """Monotone increasing estimates -> output samples are finite (no blow-up from overshoot)."""
        N = 4096
        counter = [0]

        def monotone_estimator(block, _fs):
            v = float(counter[0]) * 500.0
            counter[0] += 1
            return v

        out = frequency.correct_frequency_offset_blockwise(
            xp.ones(N, dtype=xp.complex64),
            self.FS,
            block_size=512,
            overlap=0.5,
            estimator=monotone_estimator,
        )
        out_np = out if xp is np else out.get()
        assert np.all(np.isfinite(out_np))

    def test_overlap_zero(self, backend_device, xp):
        """overlap=0: output has correct shape and is finite."""
        out = frequency.correct_frequency_offset_blockwise(
            xp.ones(4096, dtype=xp.complex64),
            self.FS,
            block_size=512,
            overlap=0.0,
            estimator=lambda b, f: 1000.0,
        )
        out_np = out if xp is np else out.get()
        assert out.shape == (4096,)
        assert np.all(np.isfinite(out_np))

    def test_single_block_fallback(self, backend_device, xp):
        """Signal shorter than block_size: single block, output shape matches input."""
        N = 200
        out = frequency.correct_frequency_offset_blockwise(
            xp.ones(N, dtype=xp.complex64),
            self.FS,
            block_size=512,
            overlap=0.5,
            estimator=lambda b, _f: 3_000.0,
        )
        out_np = out if xp is np else out.get()
        assert out.shape == (N,)
        assert np.all(np.isfinite(out_np))

    def test_mimo_per_channel_output_shape(self, backend_device, xp):
        """(C=2, N) input produces (2, N) output with per-channel correction."""
        C, N = 2, 4096
        fo_a, fo_b = 8_000.0, -5_000.0
        sig_a = _qam_signal(xp, 4, N, fo_hz=fo_a)
        sig_b = _qam_signal(xp, 4, N, fo_hz=fo_b)
        mimo = xp.stack([sig_a.samples, sig_b.samples], axis=0)
        out = frequency.correct_frequency_offset_blockwise(
            mimo,
            self.FS,
            block_size=512,
            overlap=0.5,
            estimator=lambda b, _f: frequency.estimate_frequency_offset_mth_power(
                b, sampling_rate=self.FS, modulation="qam", order=4
            ),
        )
        assert out.shape == (C, N)

    def test_mimo_combine_channels(self, backend_device, xp):
        """combine_channels=True applies a single shared correction to all channels."""
        C, N = 2, 4096
        fo_hz = 7_000.0
        sig_a = _qam_signal(xp, 4, N, fo_hz=fo_hz)
        sig_b = _qam_signal(xp, 4, N, fo_hz=fo_hz)
        mimo = xp.stack([sig_a.samples, sig_b.samples], axis=0)
        out = frequency.correct_frequency_offset_blockwise(
            mimo,
            self.FS,
            block_size=512,
            overlap=0.5,
            estimator=lambda b, _f: fo_hz,
            combine_channels=True,
        )
        assert out.shape == (C, N)
        # Both channels receive identical correction; magnitudes must match because the
        # underlying signal is identical (same seed) and the correction is shared.
        out0_np = out[0] if xp is np else out[0].get()
        out1_np = out[1] if xp is np else out[1].get()
        np.testing.assert_allclose(np.abs(out0_np), np.abs(out1_np), rtol=1e-5)

    def test_functools_partial_with_find_bias_tone(self, backend_device, xp):
        """functools.partial binding of find_bias_tone works as estimator."""
        from functools import partial

        fs = 1e9
        N = 8192
        tone_hz = 100e6
        t = np.arange(N) / fs
        tone = (np.exp(2j * np.pi * tone_hz * t)).astype(np.complex64)
        sig = xp.asarray(tone)

        track = partial(
            frequency.find_bias_tone,
            target_frequency=tone_hz,
            search_band=20e6,
        )
        out = frequency.correct_frequency_offset_blockwise(
            sig, fs, block_size=1024, overlap=0.5, estimator=track
        )
        out_np = out if xp is np else out.get()
        assert out.shape == sig.shape
        assert np.all(np.isfinite(out_np))
