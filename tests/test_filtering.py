"""Tests for digital filtering and pulse shaping tap generation."""

import logging
from typing import Any

import numpy as np
import pytest

from commkit import filtering
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

    def test_complex_taps_need_complex_samples(self, xp: Any, xpt: Any) -> None:
        """A real output has no room for the imaginary part: [1j] raises on
        real samples instead of returning zeros, and rotates complex ones."""
        taps = xp.asarray([1j], dtype=xp.complex64)
        with pytest.raises(ValueError, match="complex taps need complex samples"):
            filtering.fir_filter(xp.ones(256, dtype=xp.float32), taps)
        out = filtering.fir_filter(xp.ones(256, dtype=xp.complex64), taps)
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


_METHODS = ["direct", "whole", "overlap-save"]


def _force_method(monkeypatch: Any, method: str) -> None:
    """Make fir_filter take ``method`` whatever the sizes."""
    from commkit.filtering import linear

    monkeypatch.setattr(linear, "_fir_method", lambda N, L, device: method)


class TestFirFilterMethods:
    """fir_filter's three methods (direct, one FFT of the record,
    overlap-save) and the size thresholds that pick one."""

    @pytest.mark.parametrize("method", _METHODS)
    @pytest.mark.parametrize("num_taps", [4, 5])
    @pytest.mark.parametrize("periodic", [False, True])
    def test_alignment(
        self, xp: Any, xpt: Any, monkeypatch, method, num_taps: int, periodic: bool
    ) -> None:
        """fir_filter follows its documented centre tap ``(L - 1) // 2``
        (odd and even L, linear and circular), and tap 0 with ``causal``."""
        _force_method(monkeypatch, method)
        rng = np.random.default_rng(num_taps)
        x = rng.standard_normal(256)
        taps = rng.standard_normal(num_taps)
        src = np.r_[x, x, x] if periodic else x
        for causal, c in ((False, (num_taps - 1) // 2), (True, 0)):
            start = 256 + c if periodic else c
            expected = np.convolve(src, taps)[start : start + 256]
            out = filtering.fir_filter(
                xp.asarray(x), xp.asarray(taps), periodic=periodic, causal=causal
            )
            xpt.assert_allclose(out, xp.asarray(expected), atol=1e-12)

    @pytest.mark.parametrize("method", _METHODS)
    @pytest.mark.parametrize("complex_data", [False, True])
    @pytest.mark.parametrize(
        ("shape", "num_taps"),
        [((1,), 1), ((7,), 64), ((2, 1000), 2), ((2, 1000), 65), ((3, 2, 999), 1500)],
    )
    def test_matches_the_full_convolution(
        self, xp: Any, xpt: Any, monkeypatch, shape, num_taps, complex_data, method
    ) -> None:
        """Every method equals ``np.convolve`` at every sample, ends included,
        for records shorter than the filter, even lengths and 3-D input, in
        both alignments.  Real data takes real FFTs (measured 6e-7 of the
        peak in float32, 1e-6 in complex64)."""
        _force_method(monkeypatch, method)
        rng = np.random.default_rng(num_taps)
        x = rng.standard_normal(shape)
        taps = rng.standard_normal(num_taps)
        if complex_data:
            x = x + 1j * rng.standard_normal(shape)
            taps = taps + 1j * rng.standard_normal(num_taps)
        x = x.astype(np.complex64 if complex_data else np.float32)
        full = np.apply_along_axis(np.convolve, -1, x.astype(complex), taps)
        n = shape[-1]
        for causal, start in ((False, (num_taps - 1) // 2), (True, 0)):
            out = filtering.fir_filter(xp.asarray(x), xp.asarray(taps), causal=causal)
            ref = full[..., start : start + n]
            if not complex_data:
                ref = ref.real
            assert out.dtype == x.dtype and out.shape == x.shape
            atol = 3e-6 * np.abs(ref).max()
            xpt.assert_allclose(out, xp.asarray(ref.astype(x.dtype)), atol=atol)

    @pytest.mark.parametrize(
        ("N", "L", "device", "method"),
        [
            # Direct up to 128 taps and N·L = 2^15 (CPU), 2^25 (GPU).
            (256, 128, "cpu", "direct"),
            (256, 129, "cpu", "whole"),
            (1 << 13, 4, "cpu", "direct"),
            (1 << 13, 5, "cpu", "whole"),
            (1 << 18, 128, "gpu", "direct"),
            (1 << 18, 129, "gpu", "whole"),
            (1 << 20, 32, "gpu", "direct"),
            (1 << 20, 33, "gpu", "overlap-save"),
            # One FFT up to 2^13 samples (CPU), 2^18 (GPU).
            (1 << 13, 1501, "cpu", "whole"),
            ((1 << 13) + 1, 1501, "cpu", "overlap-save"),
            (1 << 18, 1501, "gpu", "whole"),
            ((1 << 18) + 1, 1501, "gpu", "overlap-save"),
        ],
    )
    def test_method_thresholds(self, N: int, L: int, device: str, method: str) -> None:
        from commkit.filtering.linear import _fir_method

        assert _fir_method(N, L, device) == method

    @pytest.mark.parametrize(
        ("shape", "num_taps"),
        # Either side of each threshold on the device that runs the test.
        [
            ((2, 256), 128),
            ((2, 256), 129),
            ((2, 8192), 4),
            ((2, 8192), 5),
            ((2, 8193), 201),
            ((1, 1 << 18), 129),
            ((1, (1 << 18) + 1), 129),
            ((1, 1 << 20), 32),
            ((1, 1 << 20), 33),
        ],  # fmt: skip
    )
    def test_sizes_around_the_thresholds(
        self, xp: Any, xpt: Any, shape, num_taps
    ) -> None:
        """The method the sizes pick gives ``np.convolve`` (``mode="same"``)."""
        rng = np.random.default_rng(num_taps)
        x = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(
            np.complex64
        )
        taps = rng.standard_normal(num_taps).astype(np.float32)
        c = (num_taps - 1) // 2
        ref = np.stack([np.convolve(r, taps)[c : c + shape[-1]] for r in x])
        out = filtering.fir_filter(xp.asarray(x), xp.asarray(taps))
        xpt.assert_allclose(out, xp.asarray(ref), atol=3e-6 * np.abs(ref).max())

    @pytest.mark.parametrize("causal", [False, True])
    def test_batches_do_not_change_the_output(
        self, xp: Any, xpt: Any, monkeypatch, causal
    ) -> None:
        """Batches of 3 blocks (a partial last one) give the one-batch output."""
        import commkit._overlap_save as ols

        _force_method(monkeypatch, "overlap-save")
        rng = np.random.default_rng(6)
        x = xp.asarray(
            (
                rng.standard_normal((2, 20_000)) + 1j * rng.standard_normal((2, 20_000))
            ).astype(np.complex64)
        )
        taps = xp.asarray(np.hanning(101).astype(np.float32))
        whole = filtering.fir_filter(x, taps, causal=causal)
        monkeypatch.setattr(ols, "_OLS_ELEMENTS", 2 * 1024 * 3)
        # Batched FFTs round differently in float32.
        xpt.assert_allclose(
            filtering.fir_filter(x, taps, causal=causal),
            whole,
            rtol=1e-5,
            atol=1e-5,
        )

    def test_batches_bound_the_working_memory(self, monkeypatch) -> None:
        """On a long record, the peak is the padded input and the output (3.5x
        the input for complex64), not the record's spectra (8x)."""
        import tracemalloc

        import commkit._overlap_save as ols

        x = (np.random.default_rng(7).standard_normal(1 << 20) + 0j).astype(
            np.complex64
        )
        taps = np.hanning(201).astype(np.float32)
        monkeypatch.setattr(ols, "_OLS_ELEMENTS", 1 << 17)
        filtering.fir_filter(x, taps)
        tracemalloc.start()
        try:
            filtering.fir_filter(x, taps)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        assert peak < 4 * x.nbytes


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

    @pytest.mark.parametrize("zero_phase", [False, True])
    def test_more_than_16_sections_on_gpu_matches_cpu(
        self, backend_device: str, xp: Any, xpt: Any, zero_phase: bool
    ) -> None:
        """Above the 16-section GPU kernel cap, iir_filter routes to CPU and matches."""
        skip_unless_gpu(backend_device)
        rng = np.random.default_rng(17)
        x = (rng.standard_normal((2, 512)) + 1j * rng.standard_normal((2, 512))).astype(
            np.complex64
        )
        # 20 second-order sections (40th-order Butterworth)
        sos = filtering.butterworth_sos(sampling_rate=1.0, cutoff=0.1, order=40)
        assert sos.shape[0] == 20
        expected = filtering.iir_filter(x, sos=sos, zero_phase=zero_phase)
        out = filtering.iir_filter(xp.asarray(x), sos=sos, zero_phase=zero_phase)
        assert type(out) is xp.ndarray
        assert out.dtype == np.complex64
        xpt.assert_allclose(out, xp.asarray(expected), rtol=1e-5, atol=1e-5)


def test_pulse_taps_silent_and_validates_sps(caplog: pytest.LogCaptureFixture) -> None:
    """Pulse value objects do not log and reject non-positive / non-finite / bool sps."""

    pulses = [
        filtering.RRC(rolloff=0.25, span=8),
        filtering.RC(rolloff=0.25, span=8),
        filtering.Gaussian(fwhm=0.5, span=4),
        filtering.Rect(),
        filtering.SmoothRect(rise_time=0.2),
    ]
    with caplog.at_level(logging.DEBUG, logger="commkit"):
        for p in pulses:
            p.taps(4)
    assert not [r for r in caplog.records if r.filename == "pulses.py"]

    for p in pulses[:3]:
        for bad in (0, -2, float("nan"), float("inf"), True):
            error = TypeError if bad is True else ValueError
            with pytest.raises(error, match="sps"):
                p.taps(bad)  # type: ignore[arg-type]
