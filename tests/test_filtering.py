"""Tests for digital filtering and pulse shaping tap generation."""

from typing import Any

import numpy as np
import pytest

from commkit import filtering
from commkit.core import Signal
from tests.common.conversions import to_numpy
from tests.common.metrics import calc_freq_response
from tests.common.periodic import middle_period


class TestFilterTapGenerators:
    """Tests for FIR pulse-shaping tap generators and frequency response characteristics."""

    def test_rrc_taps(self) -> None:
        """Verify Root Raised Cosine filter tap length and type."""
        taps = filtering.rrc_taps(sps=4, span=10, rolloff=0.35)
        assert isinstance(taps, np.ndarray)
        assert len(taps) == 10 * 4 + 1

    def test_rrc_taps_rolloff_zero(self, xp: Any, xpt: Any) -> None:
        """Verify zero-rolloff RRC taps match a normalised sinc function."""
        taps = filtering.rrc_taps(sps=4, rolloff=0, span=8)
        assert len(taps) > 0
        t = xp.linspace(-4, 4, len(taps))
        expected = xp.sinc(t)
        expected = expected / xp.sqrt(xp.sum(expected**2))
        xpt.assert_allclose(xp.asarray(taps), expected, atol=1e-3)

    def test_rc_taps(self) -> None:
        """Verify Raised Cosine tap length and unit-energy normalisation."""
        taps = filtering.rc_taps(sps=4, rolloff=0.5, span=8)
        assert isinstance(taps, np.ndarray)
        assert len(taps) == 8 * 4 + 1
        assert np.isclose(np.sum(np.abs(taps) ** 2), 1.0)

    def test_rc_taps_rolloff_zero(self) -> None:
        """Verify zero-rolloff RC tap unit energy (brick-wall filter)."""
        taps = filtering.rc_taps(sps=4, rolloff=0.0, span=8)
        assert isinstance(taps, np.ndarray)
        assert np.isclose(np.sum(np.abs(taps) ** 2), 1.0)

    def test_gaussian_taps(self) -> None:
        """Verify Gaussian filter tap length and unit-energy normalisation."""
        taps = filtering.gaussian_taps(sps=4, fwhm=0.5, span=4)
        assert isinstance(taps, np.ndarray)
        assert len(taps) == 4 * 4 + 1
        assert np.isclose(np.sum(np.abs(taps) ** 2), 1.0)

    def test_smoothrect_taps(self) -> None:
        """Verify smoothrect tap length and unit-energy normalisation."""
        taps = filtering.smoothrect_taps(sps=8, span=4, rise_time=0.22)
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
        """Verify matched_filter respects taps_normalization."""
        samples = xp.ones(100)
        pulse = xp.ones(10)

        out_gain = filtering.matched_filter(
            samples, pulse=pulse, taps_normalization="unity_gain"
        )
        assert out_gain.shape == (100,)
        assert isinstance(out_gain, xp.ndarray)

        with pytest.raises(ValueError, match="Unknown taps_normalization"):
            filtering.matched_filter(samples, pulse=pulse, taps_normalization="magic")

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

    def test_matched_filter_preserves_dtype(self, xp: Any) -> None:
        """matched_filter with rrc_taps (float64) on complex64 signal -> complex64."""
        rng = np.random.default_rng(11)
        sig = xp.asarray(
            (rng.standard_normal(1000) + 1j * rng.standard_normal(1000)).astype(
                np.complex64
            )
        )
        taps = filtering.rrc_taps(sps=4)
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


class TestOverlapSaveFiltering:
    """Tests for overlap-save FFT filtering (ols_fir_filter)."""

    def test_ols_fir_filter_center_matches_fir_filter_siso(
        self, xp: Any, xpt: Any
    ) -> None:
        """ols_fir_filter(default, center=True) matches fir_filter."""
        rng = np.random.default_rng(0)
        x_np = rng.standard_normal(512).astype(np.float32)
        taps_np = filtering.rrc_taps(sps=4, span=6, rolloff=0.35).astype(np.float32)
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
        """ols_fir_filter(center=True) matches fir_filter for 2-channel input."""
        rng = np.random.default_rng(1)
        x_np = rng.standard_normal((2, 512)).astype(np.float32)
        taps_np = filtering.rrc_taps(sps=4, span=6, rolloff=0.35).astype(np.float32)
        x = xp.asarray(x_np)
        taps = xp.asarray(taps_np)

        ref = filtering.fir_filter(x, taps)
        out = filtering.ols_fir_filter(x, taps)

        assert out.shape == ref.shape
        L = len(taps_np)
        xpt.assert_allclose(out[:, L:-L], ref[:, L:-L], atol=1e-4)

    def test_ols_fir_filter_causal_siso(self, xp: Any, xpt: Any) -> None:
        """ols_fir_filter(center=False) returns causal convolution."""
        rng = np.random.default_rng(0)
        x_np = rng.standard_normal(512).astype(np.float32)
        taps_np = filtering.rrc_taps(sps=4, span=6, rolloff=0.35).astype(np.float32)
        x = xp.asarray(x_np)
        taps = xp.asarray(taps_np)

        ref = xp.asarray(np.convolve(x_np, taps_np, mode="full")[: len(x_np)])
        out = filtering.ols_fir_filter(x, taps, center=False)

        assert out.shape == (len(x_np),)
        L = len(taps_np)
        xpt.assert_allclose(out[L:], ref[L:], atol=1e-4)

    def test_ols_fir_filter_preserves_shape_siso(self, xp: Any) -> None:
        """ols_fir_filter returns 1-D output for 1-D input."""
        x = xp.ones(256, dtype=xp.float32)
        taps = xp.asarray(np.ones(8, dtype=np.float32))
        out = filtering.ols_fir_filter(x, taps)
        assert out.ndim == 1
        assert out.shape == x.shape

    def test_ols_fir_filter_explicit_N_fft(self, xp: Any) -> None:
        """ols_fir_filter accepts an explicit N_fft without error."""
        rng = np.random.default_rng(2)
        x = xp.asarray(rng.standard_normal(256).astype(np.float32))
        taps = xp.asarray(np.ones(4, dtype=np.float32) / 4)
        out = filtering.ols_fir_filter(x, taps, fft_size=1024)
        assert out.shape == x.shape

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
            sampling_rate=d,
            dispersion_ps_nm_km=l,
            fiber_length_km=lam,
            center_wavelength_nm=fs,
        )
        recovered = filtering.correct_chromatic_dispersion(
            distorted,
            sampling_rate=d,
            dispersion_ps_nm_km=l,
            fiber_length_km=lam,
            center_wavelength_nm=fs,
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
            sampling_rate=d,
            dispersion_ps_nm_km=l,
            fiber_length_km=lam,
            center_wavelength_nm=fs,
        )
        recovered = filtering.correct_chromatic_dispersion(
            distorted,
            sampling_rate=d,
            dispersion_ps_nm_km=l,
            fiber_length_km=lam,
            center_wavelength_nm=fs,
        )

        xpt.assert_allclose(recovered, samples, atol=1e-3)

    def test_zero_dispersion_is_identity(self, xp: Any, xpt: Any) -> None:
        """length=0 should return identical samples."""
        samples = xp.ones(512, dtype=xp.complex64)
        out = filtering.correct_chromatic_dispersion(
            samples,
            dispersion_ps_nm_km=17.0,
            fiber_length_km=0.0,
            center_wavelength_nm=1550.0,
            sampling_rate=56e9,
        )
        xpt.assert_allclose(out, samples, atol=1e-6)

    def test_output_shape_and_dtype_preserved(self, xp: Any) -> None:
        """Output shape and dtype match input."""
        samples = xp.ones((2, 256), dtype=xp.complex64)
        out = filtering.correct_chromatic_dispersion(
            samples,
            dispersion_ps_nm_km=17.0,
            fiber_length_km=10.0,
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
            dispersion_ps_nm_km=17.0,
            fiber_length_km=20.0,
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
                dispersion_ps_nm_km=17.0,
                fiber_length_km=20.0,
                center_wavelength_nm=1550.0,
            )


class TestOlsFftSize:
    def test_invalid_fft_size_raises(self, xp: Any) -> None:
        """fft_size must be a power of 2 with a guard of at least len(taps)."""
        x = xp.ones(256, dtype=xp.complex64)
        taps = xp.ones(65)
        with pytest.raises(ValueError, match="fft_size"):
            filtering.ols_fir_filter(x, taps, fft_size=200)
        with pytest.raises(ValueError, match="fft_size"):
            filtering.ols_fir_filter(x, taps, fft_size=256)


class TestCorrectFiberPropagation:
    """Digital backpropagation: the inverse of apply_fiber_propagation."""

    @staticmethod
    def _field(n=4096, seed=1, dtype=np.complex64):
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((2, n)) + 1j * rng.standard_normal((2, n))
        spectrum = np.fft.fft(x, axis=-1)
        spectrum[:, np.abs(np.fft.fftfreq(n)) > 0.25] = 0.0
        return np.fft.ifft(spectrum, axis=-1).astype(dtype)

    LINK = dict(span_length_km=80.0, num_spans=4, launch_power_dbm=6.0)

    def test_inverts_a_noiseless_link(self, xp: Any) -> None:
        """Same steps, ξ = 1: the time-reversible symmetric step makes DBP
        the exact inverse; complex64 keeps a float32 rounding floor."""
        from commkit.impairments import Fiber, apply_fiber_propagation

        for dtype, tol in ((np.complex128, 1e-11), (np.complex64, 1e-4)):
            x = xp.asarray(self._field(dtype=dtype))
            kw = dict(fiber=Fiber(), steps_per_span=40, sampling_rate=64e9, **self.LINK)
            y = apply_fiber_propagation(x, **kw)
            z = filtering.correct_fiber_propagation(y, **kw)
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
            fiber=Fiber(),
            steps_per_span=3,
            nonlinear_scale=0.0,
            sampling_rate=64e9,
            **self.LINK,
        )
        ref = filtering.correct_chromatic_dispersion(
            x,
            dispersion_ps_nm_km=17.0,
            fiber_length_km=320.0,
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
                dispersion_ps_nm_km=17.0,
                fiber_length_km=320.0,
                center_wavelength_nm=1550.0,
                sampling_rate=64e9,
            )
        )
        kw = dict(fiber=Fiber(), sampling_rate=64e9, **self.LINK)
        full = error(filtering.correct_fiber_propagation(y, steps_per_span=10, **kw))
        one = error(
            filtering.correct_fiber_propagation(
                y, steps_per_span=1, nonlinear_scale=0.5, **kw
            )
        )
        assert full < 0.1 * cdc
        assert one < 0.75 * cdc

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
