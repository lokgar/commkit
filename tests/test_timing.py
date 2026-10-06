"""Tests for timing synchronization (Barker and Zadoff-Chu sequences, frame detection)."""

from unittest.mock import patch

import numpy as np
import pytest

from commkit import timing
from commkit.core import Preamble, Signal
from commkit.helpers import cross_correlate_fft, zc_mimo_root
from tests.common.conversions import to_numpy


class TestTimingSequences:
    """Generation and mathematical properties of synchronization sequences."""

    def test_barker_sequences(self, xp, xpt):
        """Verify all standard Barker sequence lengths and binary properties."""
        valid_lengths = [2, 3, 4, 5, 7, 11, 13]
        for length in valid_lengths:
            seq = timing.barker_sequence(length)
            assert len(seq) == length
            xpt.assert_array_equal((seq == 1) | (seq == -1), True)

    def test_barker_invalid_length(self, xp):
        """Verify that unsupported Barker lengths raise ValueError."""
        with pytest.raises(ValueError):
            timing.barker_sequence(6)

    def test_barker_autocorrelation(self, xp):
        """Verify that Barker sequences possess optimal autocorrelation properties."""
        seq = xp.asarray(timing.barker_sequence(13))
        acorr = cross_correlate_fft(seq, seq, mode="full")
        peak_idx = len(acorr) // 2
        peak_val = float(xp.abs(acorr[peak_idx]))

        sidelobes = xp.abs(acorr)
        sidelobes_max = float(
            xp.max(xp.concatenate([sidelobes[:peak_idx], sidelobes[peak_idx + 1 :]]))
        )
        assert peak_val == pytest.approx(13.0, abs=1e-4)
        assert sidelobes_max <= 1.0 + 1e-5

    def test_zadoff_chu_cazac(self, xp, xpt):
        """Verify that ZC sequences have constant amplitude (CAZAC property)."""
        zc = xp.asarray(timing.zadoff_chu_sequence(63, root=25))
        magnitudes = xp.abs(zc)
        xpt.assert_allclose(magnitudes, 1.0, atol=1e-5)

    def test_zadoff_chu_length(self, xp, xpt):
        """Verify that ZC sequences are generated with the requested length."""
        for length in [31, 63, 127]:
            zc = timing.zadoff_chu_sequence(length, root=1)
            assert len(zc) == length

        zc_even = xp.asarray(timing.zadoff_chu_sequence(10, root=1))
        assert len(zc_even) == 10
        xpt.assert_allclose(xp.abs(zc_even), 1.0)

    def test_zadoff_chu_errors(self, xp):
        """Verify ZC sequence generator input validation."""
        with pytest.raises(ValueError, match="Length must be positive"):
            timing.zadoff_chu_sequence(0)
        with pytest.raises(ValueError, match="Root must be in"):
            timing.zadoff_chu_sequence(10, root=10)

    def test_preamble_auto_generation(self, xp):
        """Verify automated preamble bit and symbol generation."""
        preamble = Preamble(sequence_type="barker", length=13)
        assert preamble.symbols is not None
        assert len(preamble.symbols) == 13
        # Generated on the host; nothing is moved to the GPU implicitly.
        assert isinstance(preamble.symbols, np.ndarray)

        preamble_zc = Preamble(sequence_type="zc", length=63, root=1)
        assert preamble_zc.symbols is not None
        assert len(preamble_zc.symbols) == 63

        with pytest.raises(ValueError, match="sequence_type"):
            Preamble(sequence_type="invalid", length=13)

        with pytest.raises(TypeError, match="length"):
            Preamble(sequence_type="barker")

    def test_sequences_are_host_arrays(self):
        """Sequence generators return NumPy arrays even when a GPU is present."""
        assert isinstance(timing.barker_sequence(13), np.ndarray)
        assert isinstance(timing.zadoff_chu_sequence(13, root=1), np.ndarray)


class TestCrossCorrelation:
    """Frequency-domain cross-correlation routines and peak detection."""

    def test_correlate_delta(self, xp):
        """Verify correct peak location for delta-like correlation."""
        signal = xp.zeros(100, dtype="float32")
        signal[50] = 1.0
        template = xp.array([1.0], dtype="float32")
        corr = cross_correlate_fft(signal, template, mode="same")
        peak_idx = int(xp.argmax(xp.abs(corr)))
        assert peak_idx == 50

    def test_correlate_shift_detection(self, xp):
        """Verify that correlation correctly identifies the shift of a template."""
        template = xp.array([1.0, 1.0, 1.0, -1.0, -1.0], dtype="float32")
        signal = xp.zeros(50, dtype="float32")
        signal[20:25] = template
        corr = cross_correlate_fft(signal, template, mode="same")
        peak_idx = int(xp.argmax(xp.abs(corr)))
        assert abs(peak_idx - 22) <= 1

    def test_correlate_mimo(self, xp):
        """Verify correlation behavior for multi-stream (MIMO) signals."""
        signal = xp.zeros((2, 50), dtype="float32")
        signal[0, 20] = 1.0
        signal[1, 30] = 1.0
        template = xp.array([1.0], dtype="float32")
        corr = cross_correlate_fft(signal, template, mode="same")
        assert corr.shape == (2, 50)
        peak_0 = int(xp.argmax(xp.abs(corr[0])))
        peak_1 = int(xp.argmax(xp.abs(corr[1])))
        assert peak_0 == 20
        assert peak_1 == 30

    def test_cross_correlate_fft_modes(self, xp):
        """Verify cross_correlate_fft output lengths for each mode."""
        signal = xp.ones(100, dtype="float32")
        template = xp.ones(10, dtype="float32")
        full = cross_correlate_fft(signal, template, mode="full")
        same = cross_correlate_fft(signal, template, mode="same")
        valid = cross_correlate_fft(signal, template, mode="valid")
        assert full.shape[-1] == 100 + 10 - 1
        assert same.shape[-1] == 100
        assert valid.shape[-1] == 91


class TestEstimateTiming:
    """Coarse timing estimation via preamble matched filtering."""

    def test_estimate_timing_advanced_scenarios(self, xp):
        """Verify estimate_timing with raw arrays, MIMO, and search ranges."""
        preamble = Preamble(sequence_type="barker", length=7)
        ref = xp.asarray(preamble.symbols)
        data = xp.zeros(100, dtype="complex64")
        data[20 : 20 + 7] = ref

        integer, _frac = timing.estimate_timing(data, preamble, threshold=2.0, sps=1)
        assert 18 <= integer[0] <= 22

        mimo_data = xp.zeros((2, 100), dtype="complex64")
        mimo_data[0, 30:37] = ref
        mimo_data[1, 30:37] = ref
        integer_mimo, _frac = timing.estimate_timing(mimo_data, ref, threshold=2.0)
        assert 28 <= integer_mimo[0] <= 32
        assert len(integer_mimo) == 2

        integer_range, _frac = timing.estimate_timing(
            data, ref, threshold=2.0, search_range=(10, 50)
        )
        assert 18 <= integer_range[0] <= 22

        with pytest.raises(ValueError, match="No correlation peak above threshold"):
            timing.estimate_timing(data, ref, threshold=100.0)

        zero_data = xp.zeros(100)
        with pytest.raises(ValueError, match="No correlation peak above threshold"):
            timing.estimate_timing(zero_data, ref, threshold=2.0)

    def test_estimate_timing_known_position(self, xp):
        """Verify timing estimation accuracy for a known preamble position."""
        preamble_symbols = xp.asarray(timing.barker_sequence(13))
        signal = xp.zeros(200, dtype="complex64")
        start_pos = 50
        signal[start_pos : start_pos + 13] = preamble_symbols

        integer, _frac = timing.estimate_timing(signal, preamble_symbols, threshold=2.0)
        assert abs(integer[0] - start_pos) <= 1

    def test_estimate_timing_with_preamble_object(self, xp):
        """Verify timing estimation using Preamble objects."""
        preamble = Preamble(sequence_type="barker", length=13)
        signal = xp.zeros(200, dtype="complex64")
        start_pos = 75

        preamble_syms = xp.asarray(to_numpy(preamble.symbols))
        signal[start_pos : start_pos + 13] = preamble_syms

        integer, _frac = timing.estimate_timing(
            signal, preamble, sps=1, pulse_shape="none", threshold=2.0
        )
        assert abs(integer[0] - start_pos) <= 1

    def test_estimate_timing_returns_tuple(self, xp):
        """Verify that estimate_timing returns (integer_offsets, fractional_offsets)."""
        preamble = xp.asarray(timing.barker_sequence(7))
        signal = xp.zeros(100, dtype="complex64")
        signal[30:37] = preamble

        integer, frac = timing.estimate_timing(signal, preamble, threshold=2.0)
        assert len(integer) == 1
        assert len(frac) == 1
        assert abs(float(frac[0])) < 0.5

    def test_estimate_timing_zero_energy(self, xp):
        """Test estimate_timing with zero energy signal."""
        preamble = xp.ones(10)
        sig = xp.zeros(50)
        with pytest.raises(ValueError, match="No correlation peak above threshold"):
            timing.estimate_timing(sig, preamble, threshold=2.0)

    def test_estimate_timing_return_tuple(self, xp):
        """Verify return tuple structure."""
        preamble = xp.ones(4)
        sig = xp.concatenate([xp.zeros(4), preamble, xp.zeros(4)])

        res = timing.estimate_timing(sig, preamble, threshold=2.0)
        assert isinstance(res, tuple)
        assert len(res) == 2
        assert len(res[0]) == 1
        assert len(res[1]) == 1

    def test_estimate_timing_search_range(self, xp):
        """Verify estimate_timing with search_range."""
        preamble = xp.random.randn(10) + 1j * xp.random.randn(10)
        sig = xp.concatenate([xp.zeros(50), preamble, xp.zeros(50)])

        integer, _frac = timing.estimate_timing(
            sig, preamble, search_range=(40, 70), threshold=2.0
        )
        assert integer[0] == 50

    def test_estimate_timing_infer_error(self, xp):
        """Verify error when Preamble object used without sps."""
        pre = Preamble(sequence_type="barker", length=3)
        sig = xp.zeros(20)
        with pytest.raises(ValueError, match="SPS must be provided"):
            timing.estimate_timing(sig, pre)

    def test_estimate_timing_fractional(self, xp):
        """Verify estimate_timing returns fractional offset."""
        preamble = xp.asarray(timing.barker_sequence(13))
        signal = xp.zeros(200, dtype="complex64")
        signal[50:63] = preamble

        integer, frac = timing.estimate_timing(signal, preamble, threshold=2.0)
        assert len(integer) == 1
        assert len(frac) == 1
        assert abs(float(frac[0])) < 0.5

    def test_estimate_timing_no_preamble_error(self, xp):
        """Verify estimate_timing raises when no reference is given."""
        sig = xp.zeros(100, dtype="complex64")
        with pytest.raises(
            ValueError,
            match="A 'reference' sequence must be provided",
        ):
            timing.estimate_timing(sig)

    def test_estimate_timing_with_preamble_object_explicit(self, xp):
        """Verify estimate_timing with explicit Preamble object."""
        barker = xp.asarray(timing.barker_sequence(7))
        samples = xp.zeros(200, dtype="complex64")
        samples[40:47] = barker

        preamble = Preamble(sequence_type="barker", length=7)
        integer, frac = timing.estimate_timing(
            samples, preamble, sps=1, pulse_shape="none", threshold=2.0
        )
        assert abs(int(integer[0]) - 40) <= 1

    def test_estimate_timing_signal_derives_sps_for_preamble(self, xp):
        """A Signal provides the SPS required to reconstruct a Preamble."""
        barker = xp.asarray(timing.barker_sequence(7))
        samples = xp.zeros(200, dtype="complex64")
        samples[40:47] = barker
        sig = Signal(
            samples=samples,
            sampling_rate=1.0,
            symbol_rate=1.0,
        )

        integer, _ = timing.estimate_timing(
            sig, Preamble(sequence_type="barker", length=7), threshold=2.0
        )
        assert abs(int(integer[0]) - 40) <= 1

    def test_estimate_timing_fractional_sps_with_raw_reference(self, xp, xpt):
        """Raw waveform correlation does not require integer samples per symbol."""
        reference = xp.asarray(timing.barker_sequence(7))
        samples = xp.zeros(100, dtype=xp.complex64)
        samples[30:37] = reference
        sig = Signal(samples=samples, sampling_rate=1.5e6, symbol_rate=1e6)
        actual = timing.estimate_timing(sig, reference, threshold=2.0)
        expected = timing.estimate_timing(samples, reference, threshold=2.0)
        for result, baseline in zip(actual, expected):
            xpt.assert_allclose(result, baseline)
        assert int(actual[0][0]) == 30

        with pytest.raises(ValueError, match="sps to be a positive integer"):
            timing.estimate_timing(sig, Preamble(sequence_type="barker", length=7))

    def test_estimate_timing_preamble_kwargs_without_sps(self, xp):
        """Verify estimate_timing raises when preamble is provided but sps is missing."""
        sig = xp.zeros(100, dtype="complex64")
        pre = Preamble(sequence_type="barker", length=7)
        with pytest.raises(ValueError, match="SPS must be provided"):
            timing.estimate_timing(sig, reference=pre)


class TestEstimateTimingMIMO:
    """MIMO unique-root sequence timing synchronization and skew detection."""

    def _make_mimo_signal(self, xp, channel_matrix, preamble_pos=200, skew=0):
        """Helper: build 2x2 MIMO received signal with unique-root ZC preamble."""
        L = 13
        zc0 = xp.asarray(
            timing.zadoff_chu_sequence(L, root=zc_mimo_root(0, 1, L)),
            dtype="complex64",
        )
        zc1 = xp.asarray(
            timing.zadoff_chu_sequence(L, root=zc_mimo_root(1, 1, L)),
            dtype="complex64",
        )

        N = preamble_pos + L + 300
        tx = xp.zeros((2, N), dtype="complex64")
        tx[0, preamble_pos : preamble_pos + L] = zc0
        tx[1, preamble_pos : preamble_pos + L] = zc1

        _rng = np.random.RandomState(42)
        noise = xp.asarray(
            (_rng.randn(2, N) + 1j * _rng.randn(2, N)).astype("complex64") * 0.05
        )
        tx = tx + noise

        H = xp.asarray(channel_matrix, dtype="complex64")
        rx = H @ tx

        if skew != 0:
            rx = xp.stack([rx[0], xp.roll(rx[1], skew)], axis=0)

        preamble = Preamble(sequence_type="zc", length=L, root=1, num_streams=2)
        return rx, preamble, L

    def test_estimate_timing_skew_detection(self, xp):
        """Verify skew warning is emitted when MIMO channels have different preamble positions."""
        barker = xp.asarray(timing.barker_sequence(7))
        sig = xp.zeros((2, 200), dtype="complex64")
        sig[0, 40:47] = barker
        sig[1, 42:49] = barker

        with patch("commkit.timing.logger") as mock_logger:
            integer, frac = timing.estimate_timing(sig, barker, threshold=2.0)
            mock_logger.warning.assert_called()
            call_args = mock_logger.warning.call_args[0][0]
            assert "Skew detected" in call_args

        assert len(integer) == 2
        assert abs(int(integer[0]) - 40) <= 1
        assert abs(int(integer[1]) - 42) <= 1

    def test_estimate_timing_mimo_identity(self, xp):
        """MIMO unique-root ZC: identity channel, both channels align to preamble_pos."""
        preamble_pos = 200
        rx, preamble, L = self._make_mimo_signal(
            xp, [[1.0, 0.0], [0.0, 1.0]], preamble_pos
        )
        integer, frac = timing.estimate_timing(
            rx, preamble, sps=1, pulse_shape="none", threshold=2.0
        )
        for ch in range(2):
            assert abs(int(integer[ch]) - preamble_pos) <= 1

    def test_estimate_timing_mimo_mixed_channel(self, xp):
        """MIMO unique-root ZC: mixed channel (both streams present on each RX)."""
        preamble_pos = 150
        angle = np.radians(40)
        H = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        rx, preamble, L = self._make_mimo_signal(xp, H, preamble_pos)

        integer, frac = timing.estimate_timing(
            rx, preamble, sps=1, pulse_shape="none", threshold=2.0
        )
        for ch in range(2):
            assert abs(int(integer[ch]) - preamble_pos) <= 1

    def test_estimate_timing_mimo_channel_skew(self, xp):
        """MIMO: hardware skew of 5 samples on channel 1 is reflected in per-channel integer offsets."""
        preamble_pos = 200
        skew = 5
        rx, preamble, L = self._make_mimo_signal(
            xp, [[1.0, 0.0], [0.0, 1.0]], preamble_pos, skew=skew
        )

        integer, frac = timing.estimate_timing(
            rx, preamble, sps=1, pulse_shape="none", threshold=2.0
        )
        assert abs(int(integer[0]) - preamble_pos) <= 1
        expected_ch1 = preamble_pos + skew
        assert abs(int(integer[1]) - expected_ch1) <= 1
        assert int(integer[0]) != int(integer[1])

    def test_estimate_timing_mimo_permuted_channel(self, xp):
        """MIMO: pure polarization swap (H = [[0,1],[1,0]]) - RX-0 receives TX-1 and vice versa."""
        preamble_pos = 200
        H = [[0.0, 1.0], [1.0, 0.0]]
        rx, preamble, L = self._make_mimo_signal(xp, H, preamble_pos)

        integer, frac = timing.estimate_timing(
            rx, preamble, sps=1, pulse_shape="none", threshold=2.0
        )
        for ch in range(2):
            assert abs(int(integer[ch]) - preamble_pos) <= 1


class TestFractionalDelay:
    """Fine fractional delay estimation and FFT fractional delay filter."""

    def test_estimate_fractional_delay_known_shift(self, xp):
        """Verify parabolic interpolation recovers a known fractional delay."""
        N = 100
        true_mu = 0.3
        n = np.arange(N, dtype="float64")
        corr = np.exp(-0.5 * ((n - 50.0 - true_mu) / 2.0) ** 2).astype("float32")
        corr = xp.asarray(corr)

        peak_idx = xp.argmax(xp.abs(corr))
        mu = timing.estimate_fractional_delay(corr, peak_idx)
        assert abs(float(mu) - true_mu) < 0.15

    def test_estimate_fractional_delay_edge_peak(self, xp):
        """Verify graceful fallback when peak is at array boundary."""
        corr = xp.zeros(50, dtype="float32")
        corr[0] = 1.0
        mu = timing.estimate_fractional_delay(corr, xp.asarray(0))
        assert float(mu) == 0.0

    def test_estimate_fractional_delay_mimo(self, xp):
        """Verify per-channel fractional delay estimation."""
        N = 100
        corr = xp.zeros((2, N), dtype="float32")
        n = np.arange(N, dtype="float32")
        corr[0] = xp.asarray(
            np.exp(-0.5 * ((n - 40.0 - 0.2) / 2.0) ** 2).astype("float32")
        )
        corr[1] = xp.asarray(
            np.exp(-0.5 * ((n - 60.0 + 0.1) / 2.0) ** 2).astype("float32")
        )

        peaks = xp.array([40, 60])
        mu = timing.estimate_fractional_delay(corr, peaks)
        assert mu.shape == (2,)
        assert abs(float(mu[0]) - 0.2) < 0.15
        assert abs(float(mu[1]) - (-0.1)) < 0.15

    def test_estimate_fractional_delay_methods(self, xp):
        """Verify different fractional delay estimation methods."""
        N = 64
        true_mu = 0.35
        sigma = 2.0
        t = np.arange(N) - N / 2
        gaussian = np.exp(-0.5 * ((t - true_mu) / sigma) ** 2).astype("float32")
        corr_gauss = xp.asarray(gaussian)
        peak_idx = xp.asarray(32)

        est_std = timing.estimate_fractional_delay(
            corr_gauss, peak_idx, method="parabolic"
        )
        est_log = timing.estimate_fractional_delay(
            corr_gauss, peak_idx, method="log-parabolic"
        )
        err_std = abs(float(est_std) - true_mu)
        err_log = abs(float(est_log) - true_mu)
        assert err_log < err_std
        assert err_log < 1e-5

        sinc_val = np.sinc(t - true_mu).astype("float32")
        corr_sinc = xp.asarray(sinc_val)
        est_sinc_1x = timing.estimate_fractional_delay(
            corr_sinc, peak_idx, dft_upsample=1
        )
        err_sinc_1x = abs(float(est_sinc_1x) - true_mu)
        est_sinc_8x = timing.estimate_fractional_delay(
            corr_sinc, peak_idx, dft_upsample=8
        )
        err_sinc_8x = abs(float(est_sinc_8x) - true_mu)
        assert err_sinc_8x < err_sinc_1x
        assert err_sinc_8x < 0.01

    def test_estimate_fractional_delay_dft_edge_fallback(self, xp):
        """Verify DFT upsample with edge peak falls back to standard parabolic estimation."""
        N = 100
        true_mu = 0.25
        n = np.arange(N, dtype="float32")
        corr = np.exp(-0.5 * ((n - 2.0 - true_mu) / 2.0) ** 2).astype("float32")
        corr = xp.asarray(corr)
        peak_idx = xp.asarray(2)
        mu = timing.estimate_fractional_delay(corr, peak_idx, dft_upsample=8)
        assert abs(float(mu)) < 0.5

    def test_fft_fractional_delay_zero_delay(self, xp, xpt):
        """Verify delay=0 is a perfect passthrough (identity operation)."""
        n = np.arange(100, dtype="float32")
        signal = xp.asarray(np.sin(2 * np.pi * 0.05 * n).astype("complex64"))
        out = timing.fft_fractional_delay(signal, 0.0)
        xpt.assert_allclose(out, signal, atol=1e-6)

    def test_fft_fractional_delay_known_sine(self, xp, xpt):
        """Verify fractional delay of a sinusoid against ground truth."""
        f = 0.02
        N = 200
        delay = 0.3
        n = np.arange(N, dtype="float64")
        original = np.exp(2j * np.pi * f * n).astype("complex64")
        truth = np.exp(2j * np.pi * f * (n - delay)).astype("complex64")
        out = timing.fft_fractional_delay(xp.asarray(original), delay)
        xpt.assert_allclose(out, truth, atol=1e-5)

    def test_fft_fractional_delay_mimo(self, xp, xpt):
        """Verify per-channel fractional delays for 2-channel signal."""
        f = 0.02
        N = 200
        n = np.arange(N, dtype="float64")
        sig = np.zeros((2, N), dtype="complex64")
        sig[0] = np.exp(2j * np.pi * f * n).astype("complex64")
        sig[1] = np.exp(2j * np.pi * f * n).astype("complex64")
        delays = xp.asarray([0.3, -0.2], dtype="float32")
        out = timing.fft_fractional_delay(xp.asarray(sig), delays)
        truth0 = np.exp(2j * np.pi * f * (n - 0.3)).astype("complex64")
        truth1 = np.exp(2j * np.pi * f * (n + 0.2)).astype("complex64")
        xpt.assert_allclose(out[0], truth0, atol=1e-5)
        xpt.assert_allclose(out[1], truth1, atol=1e-5)

    def test_fft_fractional_delay_power_conservation(self, xp):
        """Verify FFT-based delay preserves signal power."""
        np.random.seed(42)
        N = 1000
        signal = (np.random.randn(N) + 1j * np.random.randn(N)).astype("complex64")
        signal_xp = xp.asarray(signal)
        delay = 0.3
        delayed = timing.fft_fractional_delay(signal_xp, delay)
        power_in = float(xp.mean(xp.abs(signal_xp) ** 2))
        power_out = float(xp.mean(xp.abs(delayed) ** 2))
        assert abs(power_out / power_in - 1.0) < 1e-5

    def test_fft_fractional_delay_roundtrip(self, xp, xpt):
        """Verify round-trip (delay + undo) recovers original signal."""
        np.random.seed(42)
        N = 1000
        signal = (np.random.randn(N) + 1j * np.random.randn(N)).astype("complex64")
        signal_xp = xp.asarray(signal)
        delay = 0.37
        delayed = timing.fft_fractional_delay(signal_xp, delay)
        recovered = timing.fft_fractional_delay(delayed, -delay)
        xpt.assert_allclose(recovered, signal_xp, atol=1e-5)

    def test_fft_fractional_delay_scalar_ndarray(self, xp, xpt):
        """Verify fft_fractional_delay with 0-d array delay input."""
        f = 0.02
        N = 100
        n = np.arange(N, dtype="float64")
        signal = xp.asarray(np.exp(2j * np.pi * f * n).astype("complex64"))
        delay = xp.asarray(0.3)
        out = timing.fft_fractional_delay(signal, delay)
        assert out.shape == (N,)
        truth = np.exp(2j * np.pi * f * (n - 0.3)).astype("complex64")
        xpt.assert_allclose(out, truth, atol=1e-5)

    def test_fft_fractional_delay_preserves_complex64_dtype(self, xp):
        """fft_fractional_delay: complex64 signal -> complex64 output."""
        n = np.arange(200)
        sig = xp.asarray(np.exp(2j * np.pi * 0.05 * n).astype(np.complex64))
        out = timing.fft_fractional_delay(sig, 0.3)
        assert out.dtype == xp.complex64

    def test_fft_fractional_delay_preserves_float32_dtype(self, xp):
        """fft_fractional_delay: float32 signal -> float32 output."""
        n = np.arange(200, dtype=np.float32)
        sig = xp.asarray(np.sin(2 * np.pi * 0.05 * n))
        out = timing.fft_fractional_delay(sig, 0.3)
        assert out.dtype == xp.float32


class TestCorrectTimingBasic:
    """Basic integer and fractional timing correction applications."""

    def test_correct_timing_integer_only(self, xp):
        """Verify integer-only timing correction via roll."""
        signal = xp.zeros(50, dtype="float32")
        signal[10] = 1.0
        corrected = timing.correct_timing(signal, integer_offset=10)
        assert int(xp.argmax(xp.abs(corrected))) == 0

    def test_correct_timing_combined(self, xp, xpt):
        """Verify integer + fractional timing correction."""
        f = 0.02
        N = 200
        delay = 20.3
        n = np.arange(N, dtype="float64")
        original = np.sin(2 * np.pi * f * n).astype("float32")
        delayed = np.sin(2 * np.pi * f * (n - delay)).astype("float32")
        corrected = timing.correct_timing(
            xp.asarray(delayed), integer_offset=20, fractional_offset=0.3
        )
        xpt.assert_allclose(corrected[25:-25], original[25:-25], atol=0.02)

    def test_correct_timing_per_channel(self, xp):
        """Verify per-channel integer timing correction using an array of offsets."""
        sig = xp.zeros((2, 50), dtype="complex64")
        sig[0, 10] = 1.0
        sig[1, 20] = 1.0
        offsets = xp.array([10, 20])
        corrected = timing.correct_timing(sig, integer_offset=offsets)
        assert corrected.shape == (2, 50)
        assert int(xp.argmax(xp.abs(corrected[0]))) == 0
        assert int(xp.argmax(xp.abs(corrected[1]))) == 0

    def test_correct_timing_fractional_array(self, xp, xpt):
        """Verify fractional offset as array applies per-channel FFT delay and returns 2D output."""
        f = 0.02
        N = 200
        n = np.arange(N, dtype="float64")
        sig = xp.zeros((2, N), dtype="complex64")
        sig[0] = xp.asarray(np.exp(2j * np.pi * f * (n - 0.3)).astype("complex64"))
        sig[1] = xp.asarray(np.exp(2j * np.pi * f * (n + 0.2)).astype("complex64"))

        fractional = xp.array([0.3, -0.2])
        corrected = timing.correct_timing(
            sig, integer_offset=0, fractional_offset=fractional
        )
        assert corrected.ndim == 2
        assert corrected.shape == (2, N)
        truth = np.exp(2j * np.pi * f * n).astype("complex64")
        xpt.assert_allclose(corrected[0], truth, atol=1e-4)
        xpt.assert_allclose(corrected[1], truth, atol=1e-4)


class TestCorrectTiming:
    """Tests for correct_timing scalar zero/slice modes and per-channel vectorized paths."""

    def test_scalar_zero_mode_positive_shift(self, xp):
        """Scalar integer offset with mode='zero': signal shifts left, tail zero-padded."""
        N, shift = 100, 10
        sig = xp.asarray(np.arange(N, dtype=np.complex64))
        out = timing.correct_timing(sig, shift, mode="zero")
        assert out.shape == sig.shape
        assert float(out[0].real) == pytest.approx(float(sig[shift].real))
        assert float(out[-1].real) == pytest.approx(0.0)

    def test_scalar_zero_mode_negative_shift(self, xp):
        """Scalar negative integer offset with mode='zero': signal shifts right, head zero-padded."""
        N, shift = 100, -5
        sig = xp.asarray(np.ones(N, dtype=np.complex64))
        out = timing.correct_timing(sig, shift, mode="zero")
        assert out.shape == sig.shape
        assert float(out[0].real) == pytest.approx(0.0)

    def test_scalar_slice_mode(self, xp):
        """Scalar integer offset with mode='slice': output is shorter by offset."""
        N, shift = 100, 15
        sig = xp.asarray(np.ones(N, dtype=np.complex64))
        out = timing.correct_timing(sig, shift, mode="slice")
        assert out.shape[-1] == N - shift

    def test_per_channel_circular_mode(self, xp):
        """Per-channel array offset with mode='circular': each channel rolled independently."""
        C, N = 2, 64
        rng = np.random.default_rng(20)
        sig = xp.asarray(
            (rng.standard_normal((C, N)) + 1j * rng.standard_normal((C, N))).astype(
                np.complex64
            )
        )
        offsets = xp.asarray(np.array([3, 7], dtype=np.int64))
        out = timing.correct_timing(sig, offsets, mode="circular")
        assert out.shape == (C, N)

    def test_per_channel_zero_mode(self, xp):
        """Per-channel array offset with mode='zero': output same shape, tail zeroed."""
        C, N = 2, 64
        sig = xp.asarray(np.ones((C, N), dtype=np.complex64))
        offsets = xp.asarray(np.array([4, 8], dtype=np.int64))
        out = timing.correct_timing(sig, offsets, mode="zero")
        assert out.shape == (C, N)

    def test_per_channel_slice_mode(self, xp):
        """Per-channel array offset with mode='slice': output length is N - max(offset)."""
        C, N = 2, 64
        offsets = np.array([3, 10])
        sig = xp.asarray(np.ones((C, N), dtype=np.complex64))
        out = timing.correct_timing(sig, xp.asarray(offsets), mode="slice")
        assert out.shape == (C, N - max(offsets))

    def test_slice_mode_fractional_no_edge_wrap(self, xp, xpt):
        """mode='slice' with fractional offset applies the FFT delay on the
        *full pre-slice buffer*, so the new sample 0 is free of circular
        wrap-around from the buffer's trailing edge.
        """
        N, integer, fract = 1024, 200, 0.3
        f0 = 51.0 / N
        n = np.arange(N, dtype=np.float64)
        sig = np.exp(1j * 2 * np.pi * f0 * n).astype(np.complex64)
        sig_xp = xp.asarray(sig)

        out = timing.correct_timing(sig_xp, integer, fract, mode="slice")

        n_out = np.arange(out.shape[-1], dtype=np.float64) + integer + fract
        expected = np.exp(1j * 2 * np.pi * f0 * n_out).astype(np.complex64)
        xpt.assert_allclose(xp.asarray(out)[:20], xp.asarray(expected)[:20], atol=1e-4)

    def test_slice_mode_fractional_matches_delay_then_slice(self, xp, xpt):
        """mode='slice' with fractional offset must be algebraically equivalent
        to (fft_fractional_delay on full buffer) followed by (integer slice).
        """
        rng = np.random.default_rng(7)
        N, integer, fract = 512, 50, -0.27
        sig = (rng.standard_normal(N) + 1j * rng.standard_normal(N)).astype(
            np.complex64
        )
        sig[-10:] += 5.0 + 5.0j
        sig_xp = xp.asarray(sig)

        ref = timing.fft_fractional_delay(sig_xp, -fract)[..., integer:]
        out = timing.correct_timing(sig_xp, integer, fract, mode="slice")
        xpt.assert_allclose(out, ref, atol=1e-5)


class TestCorrectTimingErrors:
    """ValueError for unknown mode - scalar and per-channel paths."""

    def test_scalar_unknown_mode_raises(self, xp):
        """Scalar offset with unsupported mode raises ValueError."""
        sig = xp.asarray(np.ones(64, dtype=np.complex64))
        with pytest.raises(ValueError, match="Unknown mode"):
            timing.correct_timing(sig, 4, mode="wrap")

    def test_per_channel_unknown_mode_raises(self, xp):
        """Per-channel offset with unsupported mode raises ValueError."""
        sig = xp.asarray(np.ones((2, 64), dtype=np.complex64))
        offsets = xp.asarray(np.array([2, 4], dtype=np.int64))
        with pytest.raises(ValueError, match="Unknown mode"):
            timing.correct_timing(sig, offsets, mode="wrap")


class TestSignalInputTiming:
    """Signal-awareness for fft_fractional_delay, estimate_timing, correct_timing."""

    def test_fft_fractional_delay_signal_input(self, xp, xpt):
        """Signal input returns a Signal with the delayed samples."""
        rng = np.random.default_rng(0)
        data = xp.asarray(
            (rng.standard_normal(256) + 1j * rng.standard_normal(256)).astype(
                np.complex64
            )
        )
        sig = Signal(samples=data, sampling_rate=1.0, symbol_rate=1.0)

        out_sig = timing.fft_fractional_delay(sig, 0.3)
        out_arr = timing.fft_fractional_delay(data, 0.3)

        assert isinstance(out_sig, Signal)
        xpt.assert_allclose(out_sig.samples, out_arr)

    def test_estimate_timing_signal_input(self, xp, xpt):
        """Signal input: estimate_timing still returns a raw (int, frac) tuple."""
        preamble_symbols = xp.asarray(timing.barker_sequence(13))
        data = xp.zeros(200, dtype="complex64")
        start_pos = 50
        data[start_pos : start_pos + 13] = preamble_symbols
        sig = Signal(samples=data, sampling_rate=1.0, symbol_rate=1.0)

        int_sig, frac_sig = timing.estimate_timing(sig, preamble_symbols, threshold=2.0)
        int_arr, frac_arr = timing.estimate_timing(
            data, preamble_symbols, threshold=2.0
        )

        assert not isinstance(int_sig, Signal)
        xpt.assert_allclose(int_sig, int_arr)
        xpt.assert_allclose(frac_sig, frac_arr)

    def test_correct_timing_signal_input(self, xp, xpt):
        """Signal input returns a Signal with the timing-corrected samples."""
        data = xp.asarray(np.ones(64, dtype=np.complex64))
        sig = Signal(samples=data, sampling_rate=1.0, symbol_rate=1.0)

        out_sig = timing.correct_timing(sig, 4, mode="slice")
        out_arr = timing.correct_timing(data, 4, mode="slice")

        assert isinstance(out_sig, Signal)
        xpt.assert_allclose(out_sig.samples, out_arr)
