"""Tests for digital filtering and pulse shaping tap generation."""

from typing import Any

import numpy as np
import pytest

from commkit import filtering
from commkit.core import Signal
from tests.common.conversions import to_numpy
from tests.common.kernel_utils import skip_unless_gpu
from tests.common.metrics import calc_freq_response
from tests.common.periodic import middle_period


class TestFilterTapGenerators:
    """Tests for FIR pulse-shaping tap generators and frequency response characteristics."""

    def test_rrc_taps(self) -> None:
        """Verify Root Raised Cosine filter tap length and type."""
        taps = filtering.RRC(0.35, span=10).taps(4)
        assert isinstance(taps, np.ndarray)
        assert len(taps) == 10 * 4 + 1

    @pytest.mark.parametrize(("sps", "span"), [(3, 9), (1, 7), (4, 8), (2, 9)])
    def test_rc_nyquist_zeros_on_the_sample_grid(self, sps: int, span: int) -> None:
        """Taps sit exactly 1/sps apart, so a raised cosine is zero at every
        other symbol's sample, also when span * sps is odd (the grid was
        stretched there: 3.7 % ISI at sps=3, span=9)."""
        h = filtering.RC(0.25, span=span).taps(sps)
        c = len(h) // 2
        assert len(h) % 2 == 1
        others = h[c % sps :: sps]
        others = np.delete(others, c // sps)
        np.testing.assert_allclose(others, 0.0, atol=1e-12 * h[c])

    @pytest.mark.parametrize(("sps", "span"), [(1.5, 10), (2.5, 7)])
    def test_fractional_sps_grid(self, sps: float, span: int) -> None:
        """Non-integer sps: the Gaussian taps are its samples at k/sps."""
        h = filtering.Gaussian(1.0, span=span).taps(sps)
        c = len(h) // 2
        t = (np.arange(len(h)) - c) / sps
        bt = np.sqrt(2) * np.log(2) / np.pi
        alpha = np.sqrt(np.log(2) / 2) / bt
        np.testing.assert_allclose(h / h[c], np.exp(-((np.pi * t / alpha) ** 2)))

    def test_rrc_taps_rolloff_zero(self, xp: Any, xpt: Any) -> None:
        """Verify zero-rolloff RRC taps match a normalised sinc function."""
        taps = filtering.RRC(0, span=8).taps(4)
        assert len(taps) > 0
        t = xp.linspace(-4, 4, len(taps))
        expected = xp.sinc(t)
        expected = expected / xp.sqrt(xp.sum(expected**2))
        xpt.assert_allclose(xp.asarray(taps), expected, atol=1e-3)

    def test_rc_taps(self) -> None:
        """Verify Raised Cosine tap length and unit-energy normalisation."""
        taps = filtering.RC(0.5, span=8).taps(4)
        assert isinstance(taps, np.ndarray)
        assert len(taps) == 8 * 4 + 1
        assert np.isclose(np.sum(np.abs(taps) ** 2), 1.0)

    def test_rc_taps_rolloff_zero(self) -> None:
        """Verify zero-rolloff RC tap unit energy (brick-wall filter)."""
        taps = filtering.RC(0.0, span=8).taps(4)
        assert isinstance(taps, np.ndarray)
        assert np.isclose(np.sum(np.abs(taps) ** 2), 1.0)

    def test_gaussian_taps(self) -> None:
        """Verify Gaussian filter tap length and unit-energy normalisation."""
        taps = filtering.Gaussian(0.5, span=4).taps(4)
        assert isinstance(taps, np.ndarray)
        assert len(taps) == 4 * 4 + 1
        assert np.isclose(np.sum(np.abs(taps) ** 2), 1.0)

    def test_smoothrect_taps(self) -> None:
        """Verify smoothrect tap length and unit-energy normalisation."""
        taps = filtering.SmoothRect(0.22, span=4).taps(8)
        assert isinstance(taps, np.ndarray)
        assert len(taps) == 4 * 8 + 1
        assert np.isclose(np.sum(np.abs(taps) ** 2), 1.0)

    def test_fir_taps_lowpass(self) -> None:
        """Verify lowpass FIR: correct tap count, passband gain, and Nyquist attenuation."""
        taps = filtering.fir_taps(
            num_taps=63, cutoff=0.2, sampling_rate=1.0, btype="low"
        )
        assert len(taps) == 63

        freqs, h = calc_freq_response(taps)
        assert h[0] > 0.99
        nyquist_idx = len(h) // 2
        assert h[nyquist_idx] < 0.05

    def test_fir_taps_highpass(self) -> None:
        """Verify highpass FIR: correct tap count, DC attenuation, and passband gain."""
        taps = filtering.fir_taps(
            num_taps=63, cutoff=0.2, sampling_rate=1.0, btype="high"
        )
        assert len(taps) == 63

        freqs, h = calc_freq_response(taps)
        assert h[0] < 0.05
        nyquist_idx = len(h) // 2
        assert h[nyquist_idx] > 0.95

    def test_fir_taps_bandpass(self) -> None:
        """Verify bandpass FIR: correct tap count, centre gain, and stopband rejection."""
        low, high = 0.15, 0.35
        taps = filtering.fir_taps(
            num_taps=63, cutoff=(low, high), sampling_rate=1.0, btype="band"
        )
        assert len(taps) == 63

        freqs, h = calc_freq_response(taps)
        centre_idx = int(0.25 * len(h))
        assert h[centre_idx] > 0.90
        assert h[0] < 0.05
        nyquist_idx = len(h) // 2
        assert h[nyquist_idx] < 0.05

    def test_fir_taps_bandstop(self) -> None:
        """Verify bandstop FIR: correct tap count, notch rejection, and passband preservation."""
        low, high = 0.15, 0.35
        taps = filtering.fir_taps(
            num_taps=63, cutoff=(low, high), sampling_rate=1.0, btype="bandstop"
        )
        assert len(taps) == 63

        freqs, h = calc_freq_response(taps)
        centre_idx = int(0.25 * len(h))
        assert h[centre_idx] < 0.1
        assert h[0] > 0.95
        nyquist_idx = len(h) // 2
        assert h[nyquist_idx] > 0.95


class TestFilterApplication:
    """Tests for fir_filter, matched_filter, and dtype preservation."""

    def test_fir_filter(self, xp: Any, xpt: Any) -> None:
        """Verify FIR filtering output device, shape, and moving-average correctness."""
        data = xp.ones(100)
        taps = xp.ones(5) / 5.0

        filtered = filtering.fir_filter(data, taps)

        assert isinstance(filtered, xp.ndarray)
        assert len(filtered) == len(data)
        xpt.assert_allclose(filtered[5:-5], xp.ones(90))

    def test_matched_filter_normalization(self, xp: Any) -> None:
        """On a constant input, ten equal taps normalized to unity gain pass
        it unchanged and normalized to unit energy multiply it by
        10/sqrt(10) = sqrt(10) (away from the edges)."""
        samples = xp.ones(100, dtype=xp.complex64)
        pulse = xp.ones(10)
        gain = filtering.matched_filter(
            samples, pulse=pulse, taps_normalization="unity_gain"
        )
        energy = filtering.matched_filter(
            samples, pulse=pulse, taps_normalization="unit_energy"
        )
        np.testing.assert_allclose(to_numpy(gain)[10:-10], 1.0, rtol=1e-6)
        np.testing.assert_allclose(to_numpy(energy)[10:-10], np.sqrt(10), rtol=1e-6)
        with pytest.raises(ValueError, match="Unknown taps_normalization"):
            filtering.matched_filter(samples, pulse=pulse, taps_normalization="magic")

    def test_matched_filter_conjugates_and_reverses(self, xp: Any) -> None:
        """With complex asymmetric taps h the output is the convolution with
        conj(h[::-1]) / ‖h‖, centred: its peak on an isolated copy of h is
        the energy ‖h‖, real, at the copy's centre."""
        h = np.array([0.2 + 0.5j, 1.0 - 0.3j, -0.4 + 0.1j, 0.3j])
        x = np.zeros(64, np.complex64)
        x[30:34] = h
        out = to_numpy(filtering.matched_filter(xp.asarray(x), pulse=xp.asarray(h)))
        g = np.conj(h[::-1]) / np.linalg.norm(h)
        full = np.convolve(x, g)
        expected = full[(len(g) - 1) // 2 : (len(g) - 1) // 2 + x.size]
        np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)
        k = int(np.argmax(np.abs(out)))
        assert out[k] == pytest.approx(np.linalg.norm(h), rel=1e-5)

    def test_fir_filter_preserves_real_dtype(self, xp: Any) -> None:
        """fir_filter: float32 signal + float64 taps -> float32 output."""
        x = xp.ones(512, dtype=xp.float32)
        taps = np.hanning(32)
        out = filtering.fir_filter(x, taps)
        assert out.dtype == xp.float32

    def test_fir_filter_preserves_complex_dtype(self, xp: Any) -> None:
        """fir_filter: complex64 signal + float64 taps -> complex64 output."""
        rng = np.random.default_rng(10)
        x = xp.asarray(
            (rng.standard_normal(512) + 1j * rng.standard_normal(512)).astype(
                np.complex64
            )
        )
        taps = np.hanning(32)
        out = filtering.fir_filter(x, taps)
        assert out.dtype == xp.complex64

    @pytest.mark.parametrize("fn", [filtering.fir_filter, filtering.ols_fir_filter])
    def test_complex_taps_need_complex_samples(self, xp: Any, xpt: Any, fn) -> None:
        """A real output has no room for the imaginary part: [1j] raises on
        real samples instead of returning zeros, and rotates complex ones."""
        taps = xp.asarray([1j], dtype=xp.complex64)
        with pytest.raises(ValueError, match="complex taps need complex samples"):
            fn(xp.ones(256, dtype=xp.float32), taps)
        out = fn(xp.ones(256, dtype=xp.complex64), taps)
        assert out.dtype == xp.complex64
        xpt.assert_allclose(out, xp.full(256, 1j, dtype=xp.complex64), atol=1e-6)

    def test_matched_filter_preserves_dtype(self, xp: Any) -> None:
        """matched_filter with rrc_taps (float64) on complex64 signal -> complex64."""
        rng = np.random.default_rng(11)
        sig = xp.asarray(
            (rng.standard_normal(1000) + 1j * rng.standard_normal(1000)).astype(
                np.complex64
            )
        )
        taps = filtering.RRC(0.35).taps(4)
        out = filtering.matched_filter(sig, pulse=taps)
        assert out.dtype == xp.complex64


class TestPeriodicFirFilter:
    """fir_filter(periodic=True): circular convolution of one period."""

    @pytest.mark.parametrize(
        "num_taps, n, reps, complex_taps",
        [
            (21, 200, 3, False),
            (20, 200, 3, False),
            (21, 64, 3, True),
            (101, 30, 9, False),
        ],
    )
    def test_equals_middle_of_tiled_linear_filter(
        self, xp: Any, num_taps: int, n: int, reps: int, complex_taps: bool
    ) -> None:
        rng = np.random.default_rng(0)
        x = (rng.standard_normal((2, n)) + 1j * rng.standard_normal((2, n))).astype(
            np.complex64
        )
        taps = rng.standard_normal(num_taps)
        if complex_taps:
            taps = taps + 1j * rng.standard_normal(num_taps)
        out = filtering.fir_filter(xp.asarray(x), xp.asarray(taps), periodic=True)
        expected = middle_period(lambda t: filtering.fir_filter(t, taps), x, reps=reps)
        assert out.dtype == xp.complex64
        np.testing.assert_allclose(to_numpy(out), expected, atol=1e-4)

    def test_real_input_stays_real(self, xp: Any) -> None:
        x = xp.asarray(np.random.default_rng(1).standard_normal(64).astype(np.float32))
        out = filtering.fir_filter(x, xp.asarray([0.25, 0.5, 0.25]), periodic=True)
        assert out.dtype == xp.float32
        assert out.shape == x.shape


class TestIirFilter:
    """iir_filter: SciPy's sosfilt/sosfiltfilt in the samples' precision."""

    @staticmethod
    def _record(dtype: Any, shape: tuple[int, ...], seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        x = rng.standard_normal(shape)
        if np.dtype(dtype).kind == "c":
            x = x + 1j * rng.standard_normal(shape)
        return x.astype(dtype)

    @pytest.mark.parametrize("zero_phase", [False, True])
    @pytest.mark.parametrize("dtype", [np.float64, np.complex128])
    @pytest.mark.parametrize("cutoff", [0.2, 1e-3, 1e-5])
    def test_double_input_matches_scipy(
        self,
        backend_device: str,
        xp: Any,
        zero_phase: bool,
        dtype: Any,
        cutoff: float,
    ) -> None:
        """The CPU kernel reproduces SciPy to rounding; the GPU kernel carries
        the state across blocks, which rounds differently, so it gets a looser
        bound (2e-10 measured at ``cutoff=1e-5`` with ``zero_phase``, poles
        nearest z = 1)."""
        import scipy.signal

        x = self._record(dtype, (2, 4000))
        sos = filtering.butterworth_sos(sampling_rate=1.0, cutoff=cutoff, order=4)
        scipy_filter = scipy.signal.sosfiltfilt if zero_phase else scipy.signal.sosfilt
        expected = scipy_filter(sos, x, axis=-1)
        out = filtering.iir_filter(xp.asarray(x), sos, zero_phase=zero_phase)
        assert out.dtype == dtype
        err = np.linalg.norm(to_numpy(out) - expected) / np.linalg.norm(expected)
        assert err < (1e-13 if backend_device == "cpu" else 2e-9)

    @pytest.mark.parametrize("zero_phase", [False, True])
    @pytest.mark.parametrize("dtype", [np.float32, np.complex64])
    def test_single_input_is_filtered_at_double_accuracy(
        self, xp: Any, zero_phase: bool, dtype: Any
    ) -> None:
        """At a cutoff of 1e-4·fs single-precision sections are 17 % off; the
        float64 sections and state keep the single-precision output within
        its rounding of the float64 result."""
        import scipy.signal

        x = self._record(dtype, (8000,), seed=1)
        sos = filtering.butterworth_sos(sampling_rate=1.0, cutoff=1e-4, order=4)
        x64 = x.astype(np.complex128 if np.iscomplexobj(x) else np.float64)
        scipy_filter = scipy.signal.sosfiltfilt if zero_phase else scipy.signal.sosfilt
        expected = scipy_filter(sos, x64)
        out = filtering.iir_filter(xp.asarray(x), sos, zero_phase=zero_phase)
        assert out.dtype == dtype
        out_np = to_numpy(out).astype(x64.dtype)
        err = np.linalg.norm(out_np - expected) / np.linalg.norm(expected)
        assert err < 5e-7

    def test_lowpass_passes_dc_and_removes_a_tone(self, xp: Any, xpt: Any) -> None:
        n = np.arange(4000)
        x = (1.0 + np.cos(2 * np.pi * 0.25 * n)).astype(np.float32)
        sos = filtering.butterworth_sos(sampling_rate=1.0, cutoff=0.05, order=6)
        out = filtering.iir_filter(xp.asarray(x), sos)
        xpt.assert_allclose(out[200:-200], xp.ones(3600, dtype=xp.float32), atol=1e-4)

    def test_default_is_causal(self, xp: Any) -> None:
        """A device model by default: sosfilt, with its group delay."""
        import scipy.signal

        x = self._record(np.float32, (500,), seed=2)
        sos = filtering.butterworth_sos(sampling_rate=1.0, cutoff=0.1, order=4)
        out = to_numpy(filtering.iir_filter(xp.asarray(x), sos))
        np.testing.assert_allclose(out, scipy.signal.sosfilt(sos, x), atol=1e-6)

    def test_bad_sections_raise(self, xp: Any) -> None:
        x = xp.ones(100, dtype=xp.float32)
        with pytest.raises(ValueError, match="sos"):
            filtering.iir_filter(x, np.ones((2, 5)))
        sos = filtering.butterworth_sos(sampling_rate=1.0, cutoff=0.1, order=2)
        sos[:, 3] = 2.0
        with pytest.raises(ValueError, match="a0"):
            filtering.iir_filter(x, sos)

    def test_record_shorter_than_the_padding_raises(self, xp: Any) -> None:
        sos = filtering.butterworth_sos(sampling_rate=1.0, cutoff=0.1, order=4)
        with pytest.raises(ValueError, match="padding"):
            filtering.iir_filter(xp.ones(10, dtype=xp.float32), sos, zero_phase=True)
        out = filtering.iir_filter(xp.ones(20, dtype=xp.float32), sos)
        assert out.shape == (20,)


class TestOverlapSaveFiltering:
    """Tests for overlap-save FFT filtering (ols_fir_filter)."""

    def test_ols_fir_filter_center_matches_fir_filter_siso(
        self, xp: Any, xpt: Any
    ) -> None:
        """ols_fir_filter() by default matches fir_filter."""
        rng = np.random.default_rng(0)
        x_np = rng.standard_normal(512).astype(np.float32)
        taps_np = filtering.RRC(0.35, span=6).taps(4).astype(np.float32)
        x = xp.asarray(x_np)
        taps = xp.asarray(taps_np)

        ref = filtering.fir_filter(x, taps)
        out = filtering.ols_fir_filter(x, taps)

        assert out.shape == ref.shape
        L = len(taps_np)
        xpt.assert_allclose(out[L:-L], ref[L:-L], atol=1e-4)

    def test_ols_fir_filter_center_matches_fir_filter_multichannel(
        self, xp: Any, xpt: Any
    ) -> None:
        """ols_fir_filter() matches fir_filter for 2-channel input."""
        rng = np.random.default_rng(1)
        x_np = rng.standard_normal((2, 512)).astype(np.float32)
        taps_np = filtering.RRC(0.35, span=6).taps(4).astype(np.float32)
        x = xp.asarray(x_np)
        taps = xp.asarray(taps_np)

        ref = filtering.fir_filter(x, taps)
        out = filtering.ols_fir_filter(x, taps)

        assert out.shape == ref.shape
        L = len(taps_np)
        xpt.assert_allclose(out[:, L:-L], ref[:, L:-L], atol=1e-4)

    def test_ols_fir_filter_causal_siso(self, xp: Any, xpt: Any) -> None:
        """ols_fir_filter(causal=True) returns causal convolution."""
        rng = np.random.default_rng(0)
        x_np = rng.standard_normal(512).astype(np.float32)
        taps_np = filtering.RRC(0.35, span=6).taps(4).astype(np.float32)
        x = xp.asarray(x_np)
        taps = xp.asarray(taps_np)

        ref = xp.asarray(np.convolve(x_np, taps_np, mode="full")[: len(x_np)])
        out = filtering.ols_fir_filter(x, taps, causal=True)

        assert out.shape == (len(x_np),)
        L = len(taps_np)
        xpt.assert_allclose(out[L:], ref[L:], atol=1e-4)

    def test_ols_fir_filter_block_size_does_not_change_the_output(
        self, xp: Any, xpt: Any
    ) -> None:
        """The FFT block size is an implementation choice: nfft = 16 (many
        blocks) and the default (one block) give the same samples, and both
        follow the documented ``y[n] = Σ_k x[n + (L-1)//2 - k] taps[k]``."""
        rng = np.random.default_rng(2)
        x = rng.standard_normal(256)
        taps = rng.standard_normal(4)
        expected = np.convolve(x, taps)[1 : 1 + 256]
        for nfft in (16, None):
            out = filtering.ols_fir_filter(xp.asarray(x), xp.asarray(taps), nfft=nfft)
            xpt.assert_allclose(out, xp.asarray(expected), atol=1e-12)

    @pytest.mark.parametrize("num_taps", [4, 5])
    @pytest.mark.parametrize("periodic", [False, True])
    def test_fir_filter_alignment(
        self, xp: Any, xpt: Any, num_taps: int, periodic: bool
    ) -> None:
        """fir_filter follows its documented centre tap ``(L - 1) // 2``
        (odd and even L, linear and circular), and ols_fir_filter, which
        documents fir_filter's alignment, gives the same samples (it centred
        even-length taps one sample later)."""
        rng = np.random.default_rng(num_taps)
        x = rng.standard_normal(256)
        taps = rng.standard_normal(num_taps)
        c = (num_taps - 1) // 2
        src = np.r_[x, x, x] if periodic else x
        start = 256 + c if periodic else c
        expected = np.convolve(src, taps)[start : start + 256]
        out = filtering.fir_filter(xp.asarray(x), xp.asarray(taps), periodic=periodic)
        xpt.assert_allclose(out, xp.asarray(expected), atol=1e-12)
        if not periodic:
            ols = filtering.ols_fir_filter(xp.asarray(x), xp.asarray(taps))
            xpt.assert_allclose(ols, xp.asarray(expected), atol=1e-12)

    @pytest.mark.parametrize("causal", [False, True])
    def test_batches_do_not_change_the_output(
        self, xp: Any, xpt: Any, monkeypatch, causal
    ) -> None:
        """Batches of 3 blocks (a partial last one) give the one-batch output."""
        import commkit._overlap_save as ols

        rng = np.random.default_rng(6)
        x = xp.asarray(
            (
                rng.standard_normal((2, 20_000)) + 1j * rng.standard_normal((2, 20_000))
            ).astype(np.complex64)
        )
        taps = xp.asarray(np.hanning(101).astype(np.float32))
        whole = filtering.ols_fir_filter(x, taps, causal=causal)
        monkeypatch.setattr(ols, "_OLS_ELEMENTS", 2 * 1024 * 3)
        # Batched FFTs round differently in float32.
        xpt.assert_allclose(
            filtering.ols_fir_filter(x, taps, causal=causal),
            whole,
            rtol=1e-5,
            atol=1e-5,
        )

    def test_batches_bound_the_working_memory(self, monkeypatch) -> None:
        """With batches, the peak is the padded input and the output (3.5x the
        input for complex64), not the record's spectra (8x)."""
        import tracemalloc

        import commkit._overlap_save as ols

        x = (np.random.default_rng(7).standard_normal(1 << 20) + 0j).astype(
            np.complex64
        )
        taps = np.hanning(201).astype(np.float32)
        monkeypatch.setattr(ols, "_OLS_ELEMENTS", 1 << 17)
        filtering.ols_fir_filter(x, taps)
        tracemalloc.start()
        try:
            filtering.ols_fir_filter(x, taps)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        assert peak < 4 * x.nbytes

    def test_ols_fir_filter_preserves_real_dtype(self, xp: Any) -> None:
        """ols_fir_filter returns real dtype when both inputs are real."""
        rng = np.random.default_rng(3)
        x = xp.asarray(rng.standard_normal(1024).astype(np.float64))
        taps = xp.asarray(np.hanning(64).astype(np.float64))
        out = filtering.ols_fir_filter(x, taps)
        assert not xp.iscomplexobj(out)

    def test_ols_fir_filter_complex_input_stays_complex(self, xp: Any) -> None:
        """ols_fir_filter returns complex when input is complex."""
        rng = np.random.default_rng(4)
        x = xp.asarray(
            (rng.standard_normal(512) + 1j * rng.standard_normal(512)).astype(
                np.complex128
            )
        )
        taps = xp.asarray(np.hanning(32).astype(np.float64))
        out = filtering.ols_fir_filter(x, taps)
        assert xp.iscomplexobj(out)

    def test_ols_fir_filter_signal_input_returns_signal(
        self, xp: Any, xpt: Any
    ) -> None:
        """Signal input returns a Signal with the filtered samples."""
        rng = np.random.default_rng(5)
        data = xp.asarray(
            (rng.standard_normal(512) + 1j * rng.standard_normal(512)).astype(
                np.complex64
            )
        )
        taps = xp.asarray(np.hanning(32).astype(np.float32))
        sig = Signal(samples=data, sampling_rate=1.0, symbol_rate=1.0)

        out_sig = filtering.ols_fir_filter(sig, taps)
        out_arr = filtering.ols_fir_filter(data, taps)

        assert isinstance(out_sig, Signal)
        xpt.assert_allclose(out_sig.samples, out_arr)

    def test_ols_fir_filter_preserves_complex64_dtype(self, xp: Any) -> None:
        """ols_fir_filter: complex64 signal + float64 taps -> complex64 output."""
        rng = np.random.default_rng(13)
        x = xp.asarray(
            (rng.standard_normal(1024) + 1j * rng.standard_normal(1024)).astype(
                np.complex64
            )
        )
        taps = np.hanning(64)
        out = filtering.ols_fir_filter(x, taps)
        assert out.dtype == xp.complex64


class TestCorrectChromaticDispersion:
    """Tests for correct_chromatic_dispersion (electronic dispersion compensation)."""

    def test_round_trip_siso(self, xp: Any, xpt: Any) -> None:
        """Apply CD then compensate: SISO output should recover input."""
        from commkit.impairments import apply_chromatic_dispersion

        n = 1024
        rng = np.random.default_rng(42)
        samples = xp.asarray(
            (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
        )
        fs = 64e9
        d, l, lam = 17.0, 80.0, 1550.0

        distorted = apply_chromatic_dispersion(
            samples,
            sampling_rate=fs,
            dispersion_ps_nm_km=d,
            fiber_length_km=l,
            center_wavelength_nm=lam,
        )
        recovered = filtering.correct_chromatic_dispersion(
            distorted,
            sampling_rate=fs,
            dispersion_ps_nm=d * l,
            periodic=True,
            center_wavelength_nm=lam,
        )

        xpt.assert_allclose(recovered, samples, atol=1e-3)

    def test_round_trip_mimo(self, xp: Any, xpt: Any) -> None:
        """Apply CD then compensate: MIMO output should recover input."""
        from commkit.impairments import apply_chromatic_dispersion

        n = 1024
        rng = np.random.default_rng(7)
        samples = xp.asarray(
            (rng.standard_normal((2, n)) + 1j * rng.standard_normal((2, n))).astype(
                np.complex64
            )
        )
        fs = 64e9
        d, l, lam = 17.0, 40.0, 1550.0

        distorted = apply_chromatic_dispersion(
            samples,
            sampling_rate=fs,
            dispersion_ps_nm_km=d,
            fiber_length_km=l,
            center_wavelength_nm=lam,
        )
        recovered = filtering.correct_chromatic_dispersion(
            distorted,
            sampling_rate=fs,
            dispersion_ps_nm=d * l,
            periodic=True,
            center_wavelength_nm=lam,
        )

        xpt.assert_allclose(recovered, samples, atol=1e-3)

    def test_zero_dispersion_is_identity(self, xp: Any, xpt: Any) -> None:
        """length=0 should return identical samples."""
        samples = xp.ones(512, dtype=xp.complex64)
        out = filtering.correct_chromatic_dispersion(
            samples,
            dispersion_ps_nm=0.0,
            periodic=True,
            center_wavelength_nm=1550.0,
            sampling_rate=56e9,
        )
        xpt.assert_allclose(out, samples, atol=1e-6)

    def test_output_shape_and_dtype_preserved(self, xp: Any) -> None:
        """Output shape and dtype match input."""
        samples = xp.ones((2, 256), dtype=xp.complex64)
        out = filtering.correct_chromatic_dispersion(
            samples,
            dispersion_ps_nm=170.0,
            periodic=True,
            center_wavelength_nm=1550.0,
            sampling_rate=56e9,
        )
        assert out.shape == (2, 256)
        assert out.dtype == xp.complex64

    def test_signal_input_returns_signal(self, xp: Any, xpt: Any) -> None:
        """Signal in -> Signal out with sampling_rate preserved."""
        samples = xp.ones(512, dtype=xp.complex64)
        sig = Signal(samples=samples, sampling_rate=56e9, symbol_rate=28e9)

        out = filtering.correct_chromatic_dispersion(
            sig,
            dispersion_ps_nm=340.0,
            periodic=True,
            center_wavelength_nm=1550.0,
        )
        assert isinstance(out, Signal)
        assert out.sampling_rate == 56e9
        assert out.symbol_rate == 28e9
        assert out.samples.shape == samples.shape

    def test_signal_missing_fs_raises(self, xp: Any) -> None:
        """Missing sampling_rate on array input raises ValueError."""
        samples = xp.ones(512, dtype=xp.complex64)
        with pytest.raises(ValueError, match="sampling_rate"):
            filtering.correct_chromatic_dispersion(
                samples,
                dispersion_ps_nm=340.0,
                periodic=True,
                center_wavelength_nm=1550.0,
            )

    def test_capture_window_is_compensated(self, xp: Any) -> None:
        """A capture is a window of a longer dispersed transmission.  The
        default FIR recovers the window beyond half the filter from its ends
        (measured -45 dB on a 2-sps RRC signal), and so does the
        whole-record filter of ``periodic=True``, exact away from the ends
        (-68 dB beyond 50 samples).  Neither recovers the first samples,
        whose history lies before the capture (-7 and -2 dB), and no
        compensation at all fails everywhere (the baseline)."""
        from commkit import Constellation, generate
        from commkit.filtering import RRC
        from commkit.impairments import apply_chromatic_dispersion

        tx = generate(
            Constellation.qam(16),
            num_symbols=2**15,
            symbol_rate=32e9,
            sps=2,
            pulse=RRC(rolloff=0.1),
            rng=1,
            device="gpu" if xp.__name__ == "cupy" else "cpu",
        )
        rx = apply_chromatic_dispersion(
            tx.samples,
            dispersion_ps_nm_km=17.0,
            fiber_length_km=80.0,
            center_wavelength_nm=1550.0,
            sampling_rate=64e9,
        )
        start, width = 2**14, 2**14
        window, ref = rx[start : start + width], tx.samples[start : start + width]
        kw = dict(
            dispersion_ps_nm=1360.0, center_wavelength_nm=1550.0, sampling_rate=64e9
        )

        def error_db(z: Any, sl: slice) -> float:
            e = z[sl] - ref[sl]
            return float(
                10 * xp.log10(xp.mean(xp.abs(e) ** 2) / xp.mean(xp.abs(ref[sl]) ** 2))
            )

        fir = filtering.correct_chromatic_dispersion(window, bandwidth=35.2e9, **kw)
        whole = filtering.correct_chromatic_dispersion(window, periodic=True, **kw)
        assert error_db(fir, slice(64, -64)) < -40
        assert error_db(whole, slice(64, -64)) < -60
        assert error_db(fir, slice(0, 8)) > -10
        assert error_db(whole, slice(0, 8)) > -10
        assert error_db(window, slice(64, -64)) > -3

    def test_periodic_takes_no_fir_design(self, xp: Any) -> None:
        with pytest.raises(ValueError, match="periodic"):
            filtering.correct_chromatic_dispersion(
                xp.ones(64, dtype=xp.complex64),
                dispersion_ps_nm=17.0,
                center_wavelength_nm=1550.0,
                sampling_rate=64e9,
                periodic=True,
                num_taps=9,
            )


class TestDispersionCompensationTaps:
    """dispersion_compensation_taps: the FIR form of correct_chromatic_dispersion."""

    FS = 64e9
    KW = {"center_wavelength_nm": 1550.0}

    @staticmethod
    def _beta2_length(km: float) -> float:
        # β₂ = -D λ² / (2π c), D in s/m², λ in m, L in m.
        c = 299_792_458.0
        return -(17.0e-6) * (1550e-9) ** 2 / (2 * np.pi * c) * km * 1e3

    def test_whole_band_is_the_truncated_band_limited_response(self) -> None:
        """Over the whole band the taps are the inverse DTFT of
        exp(+j β₂L ω²/2) on |f| < fs/2, here by a dense inverse FFT (the
        aliasing of the dense grid is below 1e-8)."""
        km, n_taps, m = 80.0, 61, 2**20
        taps = filtering.dispersion_compensation_taps(
            dispersion_ps_nm=17.0 * km,
            sampling_rate=self.FS,
            num_taps=n_taps,
            **self.KW,
        )
        omega = 2 * np.pi * np.fft.fftfreq(m, d=1 / self.FS)
        h = np.fft.ifft(np.exp(0.5j * self._beta2_length(km) * omega**2))
        n = np.arange(n_taps) - (n_taps - 1) // 2
        np.testing.assert_allclose(taps, h[n % m], atol=1e-8)

    def test_default_length_follows_the_spread(self) -> None:
        """1.25 times the group-delay spread 2π|β₂L|·B·fs, odd; over the
        whole band the spread is Savory's D λ² L / (c T²) (45 samples at
        80 km, 64 GS/s, so 57 taps), and a short link keeps 7."""
        savory = 17.0e-6 * (1550e-9) ** 2 * 80e3 / (299_792_458.0 / self.FS**2)
        assert 44 < savory < 45
        full = filtering.dispersion_compensation_taps(
            dispersion_ps_nm=1360.0, sampling_rate=self.FS, **self.KW
        )
        assert len(full) == 57
        band = filtering.dispersion_compensation_taps(
            dispersion_ps_nm=1360.0, sampling_rate=self.FS, bandwidth=32e9, **self.KW
        )
        assert len(band) == 2 * int(np.ceil((1.25 * savory / 2 - 1) / 2)) + 1
        short = filtering.dispersion_compensation_taps(
            dispersion_ps_nm=17.0, sampling_rate=self.FS, **self.KW
        )
        assert len(short) == 7

    def test_zero_dispersion_is_a_delta(self) -> None:
        taps = filtering.dispersion_compensation_taps(
            dispersion_ps_nm=0.0, sampling_rate=self.FS, num_taps=5, **self.KW
        )
        np.testing.assert_array_equal(taps, [0, 0, 1, 0, 0])

    def test_band_fit_beats_truncation_inside_the_band(self) -> None:
        """At one length, the least-squares fit over a band has a smaller
        in-band response error than the truncated response, which spends its
        taps on the whole band (measured: -61 dB against -15 dB)."""
        km, n_taps, band = 80.0, 31, 35.2e9
        f = np.linspace(-band / 2, band / 2, 2001)
        target = np.exp(0.5j * self._beta2_length(km) * (2 * np.pi * f) ** 2)

        def in_band_error_db(taps: np.ndarray) -> float:
            n = np.arange(len(taps)) - (len(taps) - 1) // 2
            resp = np.exp(-2j * np.pi * np.outer(f, n) / self.FS) @ taps
            return float(10 * np.log10(np.mean(np.abs(resp - target) ** 2)))

        common = {
            "dispersion_ps_nm": 17.0 * km,
            "sampling_rate": self.FS,
            "num_taps": n_taps,
        }
        fit = filtering.dispersion_compensation_taps(
            bandwidth=band, **common, **self.KW
        )
        trunc = filtering.dispersion_compensation_taps(**common, **self.KW)
        assert in_band_error_db(fit) < -50
        assert in_band_error_db(trunc) > -25

    @pytest.mark.parametrize("km", [80.0, 1000.0])
    def test_fir_matches_frequency_domain_compensation(
        self, xp: Any, km: float
    ) -> None:
        """Applied by overlap-save to a 2-sps RRC signal, the default taps
        fitted over the occupied band match correct_chromatic_dispersion away
        from the record ends (measured -45 and -50 dB); a filter half the
        spread long, the baseline, does not (-15 and -5 dB)."""
        from commkit import Constellation, generate
        from commkit.filtering import RRC
        from commkit.impairments import apply_chromatic_dispersion

        tx = generate(
            Constellation.qam(16),
            num_symbols=2**14,
            symbol_rate=32e9,
            sps=2,
            pulse=RRC(rolloff=0.1),
            rng=1,
            device="gpu" if xp.__name__ == "cupy" else "cpu",
        )
        kw = {"dispersion_ps_nm": 17.0 * km, **self.KW}
        rx = apply_chromatic_dispersion(
            tx, dispersion_ps_nm_km=17.0, fiber_length_km=km, **self.KW
        )
        ref = to_numpy(
            filtering.correct_chromatic_dispersion(rx, periodic=True, **kw).samples
        )

        def residual_db(taps: np.ndarray) -> float:
            y = to_numpy(filtering.ols_fir_filter(rx, taps).samples)
            k = 2 * len(taps)
            err = y[k:-k] - ref[k:-k]
            return float(
                10 * np.log10(np.mean(np.abs(err) ** 2) / np.mean(np.abs(ref) ** 2))
            )

        taps = filtering.dispersion_compensation_taps(
            sampling_rate=64e9, bandwidth=35.2e9, **kw
        )
        assert residual_db(taps) < -40
        short = filtering.dispersion_compensation_taps(
            sampling_rate=64e9, bandwidth=35.2e9, num_taps=len(taps) // 2 | 1, **kw
        )
        assert residual_db(short) > -30

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"sampling_rate": 0.0}, "sampling_rate"),
            ({"bandwidth": 0.0}, "bandwidth"),
            ({"bandwidth": 70e9}, "bandwidth"),
            ({"num_taps": 0}, "num_taps"),
            ({"num_taps": 2.5}, "num_taps"),
        ],
    )
    def test_invalid_arguments_raise(self, kwargs: dict[str, Any], match: str) -> None:
        args = {
            "dispersion_ps_nm": 1360.0,
            "sampling_rate": self.FS,
            **self.KW,
            **kwargs,
        }
        with pytest.raises(ValueError, match=match):
            filtering.dispersion_compensation_taps(**args)


class TestOlsFftSize:
    def test_invalid_nfft_raises(self, xp: Any) -> None:
        """nfft must be a power of 2 with a guard of at least len(taps)."""
        x = xp.ones(256, dtype=xp.complex64)
        taps = xp.ones(65)
        with pytest.raises(ValueError, match="nfft"):
            filtering.ols_fir_filter(x, taps, nfft=200)
        with pytest.raises(ValueError, match="nfft"):
            filtering.ols_fir_filter(x, taps, nfft=256)


class TestCorrectFiberPropagation:
    """Digital backpropagation: the inverse of apply_fiber_propagation."""

    @staticmethod
    def _field(n=4096, seed=1, dtype=np.complex64):
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((2, n)) + 1j * rng.standard_normal((2, n))
        spectrum = np.fft.fft(x, axis=-1)
        spectrum[:, np.abs(np.fft.fftfreq(n)) > 0.25] = 0.0
        return np.fft.ifft(spectrum, axis=-1).astype(dtype)

    LINK = dict(
        span_length_km=80.0,
        num_spans=4,
        launch_power_dbm=6.0,
        center_wavelength_nm=1550.0,
    )

    def test_inverts_a_noiseless_link(self, xp: Any) -> None:
        """Same steps, ξ = 1: the time-reversible symmetric step makes DBP
        the exact inverse; complex64 keeps a float32 rounding floor."""
        from commkit.impairments import Fiber, apply_fiber_propagation

        for dtype, tol in ((np.complex128, 1e-11), (np.complex64, 1e-4)):
            x = xp.asarray(self._field(dtype=dtype))
            kw = dict(fiber=Fiber(), steps_per_span=40, sampling_rate=64e9, **self.LINK)
            y = apply_fiber_propagation(x, **kw)
            z = filtering.correct_fiber_propagation(y, periodic=True, **kw)
            err = float(xp.sqrt(xp.mean(xp.abs(z - x) ** 2) / xp.mean(xp.abs(x) ** 2)))
            assert err < tol, (dtype, err)
            assert z.dtype == dtype

    def test_without_kerr_it_is_dispersion_compensation(self, xp: Any) -> None:
        """ξ = 0 leaves the dispersion and loss: correct_chromatic_dispersion
        over the whole link (the amplifiers cancel the loss)."""
        from commkit.impairments import Fiber

        x = xp.asarray(self._field())
        z = filtering.correct_fiber_propagation(
            x,
            periodic=True,
            fiber=Fiber(),
            steps_per_span=3,
            nonlinear_scale=0.0,
            sampling_rate=64e9,
            **self.LINK,
        )
        ref = filtering.correct_chromatic_dispersion(
            x,
            dispersion_ps_nm=5440.0,
            periodic=True,
            center_wavelength_nm=1550.0,
            sampling_rate=64e9,
        )
        err = float(xp.sqrt(xp.mean(xp.abs(z - ref) ** 2) / xp.mean(xp.abs(ref) ** 2)))
        assert err < 1e-5

    def test_beats_linear_compensation_on_a_nonlinear_link(self, xp: Any) -> None:
        """6 dBm over 4 x 80 km with ASE (NF 5 dB); distortion after
        dispersion compensation alone is about -16 dB.  Ten DBP steps per
        span reach the ASE floor (about -30 dB, measured); one step per span
        gains about 2 dB, but only with ξ near 0.5 (ξ = 1 overshoots)."""
        from commkit.impairments import Fiber, apply_fiber_propagation

        x = xp.asarray(self._field(n=1 << 14))
        y = apply_fiber_propagation(
            x,
            fiber=Fiber(),
            steps_per_span=50,
            noise_figure_db=5.0,
            rng=3,
            sampling_rate=64e9,
            **self.LINK,
        )

        def error(z: Any) -> float:
            g = xp.sum(xp.conj(x) * z) / xp.sum(xp.abs(x) ** 2)  # common phase
            return float(xp.mean(xp.abs(z - g * x) ** 2) / xp.mean(xp.abs(x) ** 2))

        cdc = error(
            filtering.correct_chromatic_dispersion(
                y,
                dispersion_ps_nm=5440.0,
                periodic=True,
                center_wavelength_nm=1550.0,
                sampling_rate=64e9,
            )
        )
        kw = dict(fiber=Fiber(), sampling_rate=64e9, **self.LINK)
        full = error(
            filtering.correct_fiber_propagation(
                y, steps_per_span=10, periodic=True, **kw
            )
        )
        one = error(
            filtering.correct_fiber_propagation(
                y, steps_per_span=1, nonlinear_scale=0.5, periodic=True, **kw
            )
        )
        assert full < 0.1 * cdc
        assert one < 0.75 * cdc

    def test_capture_window_is_backpropagated_in_blocks(self, xp: Any) -> None:
        """A capture is a window of a longer transmission.  Propagate a long
        record, keep a window from its middle (with its true past and
        future outside), and backpropagate the window alone: the default
        blocks recover the transmitted field away from the window's ends
        (measured -71 dB), as does the whole window.  The first samples,
        whose history lies before the capture, are lost to either (-3 dB
        for the whole window)."""
        from commkit.impairments import Fiber, apply_fiber_propagation

        n, start, width = 1 << 15, 1 << 13, 1 << 14
        x = xp.asarray(self._field(n=n))
        kw = dict(fiber=Fiber(), steps_per_span=10, sampling_rate=64e9, **self.LINK)
        y = apply_fiber_propagation(x, **kw)
        window = y[..., start : start + width]
        # The window's power differs slightly from the record's; the launch
        # power is given, so scale the window as the record was.
        ref = x[..., start : start + width]
        keep = slice(1024, width - 1024)  # beyond the transients (guard ~ 120)

        def error_db(z: Any, sl: slice) -> float:
            e = z[..., sl] - ref[..., sl]
            return float(
                10
                * xp.log10(xp.mean(xp.abs(e) ** 2) / xp.mean(xp.abs(ref[..., sl]) ** 2))
            )

        p_window = float(xp.mean(xp.abs(window) ** 2) / xp.mean(xp.abs(y) ** 2))
        launch = self.LINK["launch_power_dbm"] + 10 * np.log10(p_window)
        link = {**kw, "launch_power_dbm": launch}
        blocks = filtering.correct_fiber_propagation(window, **link)
        whole = filtering.correct_fiber_propagation(window, periodic=True, **link)
        assert error_db(blocks, keep) < -40
        assert error_db(whole, keep) < -40
        assert error_db(blocks, slice(0, 64)) > -20
        assert error_db(whole, slice(0, 64)) > -20

    def test_blocks_match_the_whole_record_inside(self, xp: Any) -> None:
        """On one record, the blocks and the whole-record backpropagation
        agree beyond the guards: the guard covers the link's memory up to
        the band-limited tails it truncates (measured -71 dB, the same in
        complex128, so not rounding)."""
        from commkit.impairments import Fiber

        x = xp.asarray(self._field(n=1 << 14))
        kw = dict(fiber=Fiber(), steps_per_span=5, sampling_rate=64e9, **self.LINK)
        blocks = filtering.correct_fiber_propagation(x, **kw)
        whole = filtering.correct_fiber_propagation(x, periodic=True, **kw)
        sl = slice(1024, -1024)
        e = blocks[..., sl] - whole[..., sl]
        rel = float(xp.mean(xp.abs(e) ** 2) / xp.mean(xp.abs(whole[..., sl]) ** 2))
        assert rel < 1e-6

    def test_bad_nonlinear_scale_raises(self, xp: Any) -> None:
        from commkit.impairments import Fiber

        with pytest.raises(ValueError, match="nonlinear_scale"):
            filtering.correct_fiber_propagation(
                xp.asarray(self._field()),
                fiber=Fiber(),
                nonlinear_scale=float("nan"),
                sampling_rate=64e9,
                **self.LINK,
            )


@pytest.mark.parametrize("shape", [(2, 0), (0, 10), (0,)])
def test_causal_iir_of_empty_input(xp: Any, shape: tuple[int, ...]) -> None:
    """No samples or no channels: an empty result (on the GPU, without a
    zero-size launch)."""
    sos = filtering.butterworth_sos(sampling_rate=1.0, cutoff=0.02, order=4)
    out = filtering.iir_filter(xp.zeros(shape, dtype=xp.float32), sos=sos)
    assert out.shape == shape


@pytest.mark.gpu_only
class TestIIRKernel:
    """The block-parallel CUDA IIR kernel against its CPU reference (Numba)."""

    @pytest.mark.parametrize(
        "dtype", [np.float32, np.float64, np.complex64, np.complex128]
    )
    @pytest.mark.parametrize("n", [1, 5, 1023, 1024, 1025, 3001, 1 << 16])
    @pytest.mark.parametrize("order", [2, 8, 16])
    def test_matches_cpu_from_a_state(
        self, backend_device: str, xp: Any, xpt: Any, dtype: Any, n: int, order: int
    ) -> None:
        """Any length (one block, a block boundary, many blocks), any state."""
        skip_unless_gpu(backend_device)
        from commkit.filtering.linear import _sosfilt

        rng = np.random.default_rng(n + order)
        x = rng.standard_normal((2, n))
        if np.dtype(dtype).kind == "c":
            x = x + 1j * rng.standard_normal((2, n))
        x = x.astype(dtype)
        sos = filtering.butterworth_sos(sampling_rate=1.0, cutoff=0.02, order=order)
        state = np.complex128 if np.dtype(dtype).kind == "c" else np.float64
        zi = (0.1 * rng.standard_normal((2, sos.shape[0], 2))).astype(state)
        expected = _sosfilt(sos, x, zi, np)
        out = _sosfilt(sos, xp.asarray(x), xp.asarray(zi), xp)
        assert out.dtype == dtype
        rtol = 1e-6 if dtype in (np.float32, np.complex64) else 1e-9
        xpt.assert_allclose(out, xp.asarray(expected), rtol=rtol, atol=rtol)
