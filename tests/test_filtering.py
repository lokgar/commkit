"""Tests for digital filtering and pulse shaping tap generation."""

import numpy as np
import pytest

from commkit import filtering
from commkit.core import Signal

# -----------------------------------------------------------------------------
# TAP GENERATOR TESTS - these functions always return NumPy arrays regardless
# of backend, so no backend parametrisation is needed.
# -----------------------------------------------------------------------------


def test_rrc_taps():
    """Verify Root Raised Cosine filter tap length and type."""
    taps = filtering.rrc_taps(sps=4, span=10, rolloff=0.35)
    assert isinstance(taps, np.ndarray)
    assert len(taps) == 10 * 4 + 1


def test_rrc_taps_rolloff_zero(backend_device, xp, xpt):
    """Verify zero-rolloff RRC taps match a normalised sinc function."""
    taps = filtering.rrc_taps(sps=4, rolloff=0, span=8)
    assert len(taps) > 0
    # Compare to normalised sinc on the same device
    t = xp.linspace(-4, 4, len(taps))
    expected = xp.sinc(t)
    expected = expected / xp.sqrt(xp.sum(expected**2))
    xpt.assert_allclose(xp.asarray(taps), expected, atol=1e-3)


def test_rc_taps():
    """Verify Raised Cosine tap length and unit-energy normalisation."""
    taps = filtering.rc_taps(sps=4, rolloff=0.5, span=8)
    assert isinstance(taps, np.ndarray)
    assert len(taps) == 8 * 4 + 1
    assert np.isclose(np.sum(np.abs(taps) ** 2), 1.0)


def test_rc_taps_rolloff_zero():
    """Verify zero-rolloff RC tap unit energy (brick-wall filter)."""
    taps = filtering.rc_taps(sps=4, rolloff=0.0, span=8)
    assert isinstance(taps, np.ndarray)
    assert np.isclose(np.sum(np.abs(taps) ** 2), 1.0)


def test_gaussian_taps():
    """Verify Gaussian filter tap length and unit-energy normalisation."""
    taps = filtering.gaussian_taps(sps=4, duty_cycle=0.5, span=4)
    assert isinstance(taps, np.ndarray)
    assert len(taps) == 4 * 4 + 1
    assert np.isclose(np.sum(np.abs(taps) ** 2), 1.0)


def test_smoothrect_taps():
    """Verify smoothrect tap length and unit-energy normalisation."""
    taps = filtering.smoothrect_taps(sps=8, span=4, rise_time=0.22)
    assert isinstance(taps, np.ndarray)
    assert len(taps) == 4 * 8 + 1
    assert np.isclose(np.sum(np.abs(taps) ** 2), 1.0)


def _freq_response(taps, nfft=1024):
    """Return (normalised_freqs, magnitude_response) for a tap array."""
    H = np.abs(np.fft.fft(taps, nfft))
    freqs = np.fft.fftfreq(nfft)  # cycles per sample, range [-0.5, 0.5)
    return freqs, H


def test_lowpass_taps():
    """Verify lowpass FIR: correct tap count, passband gain, and Nyquist attenuation."""
    taps = filtering.lowpass_taps(num_taps=63, cutoff=0.2, sampling_rate=1.0)
    assert len(taps) == 63

    freqs, H = _freq_response(taps)
    # DC bin should pass (gain ≈ 1)
    assert H[0] > 0.99, f"DC gain too low: {H[0]:.4f}"
    # Nyquist should be heavily attenuated (Hamming window ≥ 40 dB stopband)
    nyquist_idx = len(H) // 2
    assert H[nyquist_idx] < 0.05, f"Nyquist not attenuated: {H[nyquist_idx]:.4f}"


def test_highpass_taps():
    """Verify highpass FIR: correct tap count, DC attenuation, and passband gain."""
    taps = filtering.highpass_taps(num_taps=63, cutoff=0.2, sampling_rate=1.0)
    assert len(taps) == 63

    freqs, H = _freq_response(taps)
    # DC bin should be blocked
    assert H[0] < 0.05, f"DC not attenuated: {H[0]:.4f}"
    # Nyquist bin should pass
    nyquist_idx = len(H) // 2
    assert H[nyquist_idx] > 0.95, f"Nyquist gain too low: {H[nyquist_idx]:.4f}"


def test_bandpass_taps():
    """Verify bandpass FIR: correct tap count, centre gain, and out-of-band attenuation."""
    low, high = 0.15, 0.35
    taps = filtering.bandpass_taps(
        num_taps=63, low_cutoff=low, high_cutoff=high, sampling_rate=1.0
    )
    assert len(taps) == 63

    freqs, H = _freq_response(taps)
    # Centre of passband (normalised freq 0.25)
    centre_idx = int(0.25 * len(H))
    assert H[centre_idx] > 0.90, f"Passband centre gain too low: {H[centre_idx]:.4f}"
    # DC should be blocked
    assert H[0] < 0.05, f"DC not attenuated: {H[0]:.4f}"
    # Nyquist should be blocked
    nyquist_idx = len(H) // 2
    assert H[nyquist_idx] < 0.05, f"Nyquist not attenuated: {H[nyquist_idx]:.4f}"


def test_bandstop_taps():
    """Verify bandstop FIR: correct tap count, notch rejection, and passband preservation."""
    low, high = 0.15, 0.35
    taps = filtering.bandstop_taps(
        num_taps=63, low_cutoff=low, high_cutoff=high, sampling_rate=1.0
    )
    assert len(taps) == 63

    freqs, H = _freq_response(taps)
    # Centre of stopband should be rejected
    centre_idx = int(0.25 * len(H))
    assert H[centre_idx] < 0.1, f"Notch not deep enough: {H[centre_idx]:.4f}"
    # DC should pass
    assert H[0] > 0.95, f"DC gain too low: {H[0]:.4f}"
    # Nyquist should pass
    nyquist_idx = len(H) // 2
    assert H[nyquist_idx] > 0.95, f"Nyquist gain too low: {H[nyquist_idx]:.4f}"


# -----------------------------------------------------------------------------
# DISPATCHING FILTER TESTS - these functions operate on the active backend,
# so they require backend parametrisation.
# -----------------------------------------------------------------------------


def test_fir_filter(backend_device, xp, xpt):
    """Verify FIR filtering output device, shape, and moving-average correctness."""
    data = xp.ones(100)
    taps = xp.ones(5) / 5.0  # Moving average

    filtered = filtering.fir_filter(data, taps)

    assert isinstance(filtered, xp.ndarray)
    assert len(filtered) == len(data)
    # Interior samples of a DC signal through a moving average must stay at 1
    xpt.assert_allclose(filtered[5:-5], xp.ones(90))


def test_matched_filter_normalization(backend_device, xp):
    """Verify matched_filter respects taps_normalization."""
    # matched_filter dispatches on its input, so use xp arrays throughout
    samples = xp.ones(100)
    pulse = xp.ones(10)

    # Unity-gain normalisation
    out_gain = filtering.matched_filter(samples, pulse, taps_normalization="unity_gain")
    assert out_gain.shape == (100,)
    assert isinstance(out_gain, xp.ndarray)

    # Invalid normalisation mode must raise
    with pytest.raises(
        ValueError,
        match="Unknown taps_normalization",
    ):
        filtering.matched_filter(samples, pulse, taps_normalization="magic")


def test_shape_pulse_variants(backend_device, xp):
    """Verify shape_pulse produces correct lengths for RC and sinc shapes."""
    symbols = xp.array([1, -1, 1, -1])

    res_rc = filtering.shape_pulse(symbols, sps=4, pulse_shape="rc")
    assert len(res_rc) == 16

    res_sinc = filtering.shape_pulse(symbols, sps=4, pulse_shape="sinc")
    assert len(res_sinc) == 16

    with pytest.raises(ValueError, match="Not implemented pulse shape"):
        filtering.shape_pulse(symbols, sps=4, pulse_shape="magic")


def test_smoothrect_pulse(backend_device, xp):
    """Verify smoothrect pulse shaping output length."""
    symbols = xp.array([1, 1])
    res = filtering.shape_pulse(
        symbols, sps=8, pulse_shape="smoothrect", filter_span=4, rise_time=0.05
    )
    assert len(res) == 16


def test_shape_pulse_none_with_rz(backend_device, xp):
    """Verify shape_pulse with pulse_shape='none' and rz=True expands using rect."""
    symbols = xp.array([1, -1, 1], dtype=xp.complex64)
    result = filtering.shape_pulse(symbols, sps=4, pulse_shape="none", rz=True)
    assert result is not None
    assert len(result) > 0


# -----------------------------------------------------------------------------
# OLS FIR FILTER TESTS
# -----------------------------------------------------------------------------


def test_ols_fir_filter_center_matches_fir_filter_siso(backend_device, xp, xpt):
    """ols_fir_filter(center=True, default) matches fir_filter (scipy mode='same')."""
    rng = np.random.default_rng(0)
    x_np = rng.standard_normal(512).astype(np.float32)
    taps_np = filtering.rrc_taps(sps=4, span=6, rolloff=0.35).astype(np.float32)
    x = xp.asarray(x_np)
    taps = xp.asarray(taps_np)

    ref = filtering.fir_filter(x, taps)
    out = filtering.ols_fir_filter(x, taps)  # center=True default

    assert out.shape == ref.shape
    L = len(taps_np)
    xpt.assert_allclose(out[L:-L], ref[L:-L], atol=1e-4)


def test_ols_fir_filter_center_matches_fir_filter_multichannel(backend_device, xp, xpt):
    """ols_fir_filter(center=True) matches fir_filter for 2-channel input."""
    rng = np.random.default_rng(1)
    x_np = rng.standard_normal((2, 512)).astype(np.float32)
    taps_np = filtering.rrc_taps(sps=4, span=6, rolloff=0.35).astype(np.float32)
    x = xp.asarray(x_np)
    taps = xp.asarray(taps_np)

    ref = filtering.fir_filter(x, taps)
    out = filtering.ols_fir_filter(x, taps)  # center=True default

    assert out.shape == ref.shape
    L = len(taps_np)
    xpt.assert_allclose(out[:, L:-L], ref[:, L:-L], atol=1e-4)


def test_ols_fir_filter_causal_siso(backend_device, xp, xpt):
    """ols_fir_filter(center=False) returns causal convolution (mode='full'[:N])."""
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


def test_ols_fir_filter_preserves_shape_siso(backend_device, xp):
    """ols_fir_filter returns 1-D output for 1-D input."""
    x = xp.ones(256, dtype=xp.float32)
    taps = xp.asarray(np.ones(8, dtype=np.float32))
    out = filtering.ols_fir_filter(x, taps)
    assert out.ndim == 1
    assert out.shape == x.shape


def test_ols_fir_filter_explicit_N_fft(backend_device, xp):
    """ols_fir_filter accepts an explicit N_fft without error."""
    rng = np.random.default_rng(2)
    x = xp.asarray(rng.standard_normal(256).astype(np.float32))
    taps = xp.asarray(np.ones(4, dtype=np.float32) / 4)
    out = filtering.ols_fir_filter(x, taps, N_fft=1024)
    assert out.shape == x.shape


def test_ols_fir_filter_preserves_real_dtype(backend_device, xp):
    """ols_fir_filter returns real dtype when both inputs are real."""
    rng = np.random.default_rng(3)
    x = xp.asarray(rng.standard_normal(1024).astype(np.float64))
    taps = xp.asarray(np.hanning(64).astype(np.float64))
    out = filtering.ols_fir_filter(x, taps)
    assert not xp.iscomplexobj(out), f"Expected real output, got dtype={out.dtype}"


def test_ols_fir_filter_complex_input_stays_complex(backend_device, xp):
    """ols_fir_filter returns complex when input is complex."""
    rng = np.random.default_rng(4)
    x = xp.asarray(
        (rng.standard_normal(512) + 1j * rng.standard_normal(512)).astype(np.complex128)
    )
    taps = xp.asarray(np.hanning(32).astype(np.float64))
    out = filtering.ols_fir_filter(x, taps)
    assert xp.iscomplexobj(out), f"Expected complex output, got dtype={out.dtype}"


def test_ols_fir_filter_signal_input_returns_signal(backend_device, xp, xpt):
    """Signal input returns a Signal with the filtered samples."""
    rng = np.random.default_rng(5)
    data = xp.asarray(
        (rng.standard_normal(512) + 1j * rng.standard_normal(512)).astype(np.complex64)
    )
    taps = xp.asarray(np.hanning(32).astype(np.float32))
    sig = Signal(samples=data, sampling_rate=1.0, symbol_rate=1.0)

    out_sig = filtering.ols_fir_filter(sig, taps)
    out_arr = filtering.ols_fir_filter(data, taps)

    assert isinstance(out_sig, Signal)
    xpt.assert_allclose(out_sig.samples, out_arr)


# -----------------------------------------------------------------------------
# DTYPE PRESERVATION TESTS
# -----------------------------------------------------------------------------


def test_fir_filter_preserves_real_dtype(backend_device, xp):
    """fir_filter: float32 signal + float64 taps -> float32 output."""
    x = xp.ones(512, dtype=xp.float32)
    taps = np.hanning(32)  # float64
    out = filtering.fir_filter(x, taps)
    assert out.dtype == xp.float32, f"Expected float32, got {out.dtype}"


def test_fir_filter_preserves_complex_dtype(backend_device, xp):
    """fir_filter: complex64 signal + float64 taps -> complex64 output."""
    rng = np.random.default_rng(10)
    x = xp.asarray(
        (rng.standard_normal(512) + 1j * rng.standard_normal(512)).astype(np.complex64)
    )
    taps = np.hanning(32)  # float64
    out = filtering.fir_filter(x, taps)
    assert out.dtype == xp.complex64, f"Expected complex64, got {out.dtype}"


def test_matched_filter_preserves_dtype(backend_device, xp):
    """matched_filter with rrc_taps (float64) on complex64 signal -> complex64."""
    from commkit.filtering import matched_filter, rrc_taps

    rng = np.random.default_rng(11)
    sig = xp.asarray(
        (rng.standard_normal(1000) + 1j * rng.standard_normal(1000)).astype(
            np.complex64
        )
    )
    taps = rrc_taps(4)  # returns float64
    out = matched_filter(sig, taps)
    assert out.dtype == xp.complex64, f"Expected complex64, got {out.dtype}"


def test_shape_pulse_preserves_complex64_dtype(backend_device, xp):
    """shape_pulse: complex64 symbols -> complex64 waveform."""
    rng = np.random.default_rng(12)
    syms = xp.asarray(
        (rng.standard_normal(100) + 1j * rng.standard_normal(100)).astype(np.complex64)
    )
    out = filtering.shape_pulse(syms, sps=4, pulse_shape="rrc")
    assert out.dtype == xp.complex64, f"Expected complex64, got {out.dtype}"


def test_ols_fir_filter_preserves_complex64_dtype(backend_device, xp):
    """ols_fir_filter: complex64 signal + float64 taps -> complex64 output."""
    rng = np.random.default_rng(13)
    x = xp.asarray(
        (rng.standard_normal(1024) + 1j * rng.standard_normal(1024)).astype(
            np.complex64
        )
    )
    taps = np.hanning(64)  # float64
    out = filtering.ols_fir_filter(x, taps)
    assert out.dtype == xp.complex64, f"Expected complex64, got {out.dtype}"


# -----------------------------------------------------------------------------
# CHROMATIC DISPERSION COMPENSATION TESTS
# -----------------------------------------------------------------------------


class TestCompensateChromaticDispersion:
    """Tests for compensate_chromatic_dispersion (EDC)."""

    def test_round_trip_siso(self, backend_device, xp, xpt):
        """Apply CD then compensate: SISO output should recover input."""
        from commkit.impairments import apply_chromatic_dispersion

        N = 1024
        rng = np.random.default_rng(42)
        samples = xp.asarray(
            (rng.standard_normal(N) + 1j * rng.standard_normal(N)).astype(np.complex64)
        )
        fs = 64e9
        D, L, lam = 17.0, 80.0, 1550.0

        distorted = apply_chromatic_dispersion(samples, D, L, lam, fs)
        recovered = filtering.compensate_chromatic_dispersion(distorted, D, L, lam, fs)

        xpt.assert_allclose(recovered, samples, atol=1e-3)

    def test_round_trip_mimo(self, backend_device, xp, xpt):
        """Apply CD then compensate: MIMO output should recover input."""
        from commkit.impairments import apply_chromatic_dispersion

        C, N = 2, 512
        rng = np.random.default_rng(7)
        samples = xp.asarray(
            (rng.standard_normal((C, N)) + 1j * rng.standard_normal((C, N))).astype(
                np.complex64
            )
        )
        fs = 64e9
        D, L, lam = 17.0, 80.0, 1550.0

        distorted = apply_chromatic_dispersion(samples, D, L, lam, fs)
        recovered = filtering.compensate_chromatic_dispersion(distorted, D, L, lam, fs)

        xpt.assert_allclose(recovered, samples, atol=1e-3)

    def test_output_shape_siso(self, backend_device, xp):
        """SISO output shape matches input."""
        samples = xp.ones(512, dtype=xp.complex64)
        out = filtering.compensate_chromatic_dispersion(
            samples, 17.0, 80.0, 1550.0, 64e9
        )
        assert out.shape == (512,)

    def test_output_shape_mimo(self, backend_device, xp):
        """MIMO output shape matches input."""
        samples = xp.ones((2, 512), dtype=xp.complex64)
        out = filtering.compensate_chromatic_dispersion(
            samples, 17.0, 80.0, 1550.0, 64e9
        )
        assert out.shape == (2, 512)

    def test_energy_preserved(self, backend_device, xp, xpt):
        """EDC is an all-pass filter: energy must be preserved."""
        rng = np.random.default_rng(11)
        samples = xp.asarray(
            (rng.standard_normal(1024) + 1j * rng.standard_normal(1024)).astype(
                np.complex64
            )
        )
        out = filtering.compensate_chromatic_dispersion(
            samples, 17.0, 80.0, 1550.0, 64e9
        )
        power_in = float(xp.sum(xp.abs(samples) ** 2))
        power_out = float(xp.sum(xp.abs(out) ** 2))
        xpt.assert_allclose(power_out, power_in, rtol=1e-4)

    def test_signal_input_returns_signal(self, backend_device, xp, xpt):
        """Signal input: sampling_rate is taken from the signal."""
        fs = 64e9
        data = xp.ones(512, dtype=xp.complex64)
        sig = Signal(samples=data, sampling_rate=fs, symbol_rate=fs / 2)

        out_sig = filtering.compensate_chromatic_dispersion(
            sig,
            dispersion_ps_nm_km=17.0,
            fiber_length_km=80.0,
            center_wavelength_nm=1550.0,
        )
        out_arr = filtering.compensate_chromatic_dispersion(
            data,
            sampling_rate=fs,
            dispersion_ps_nm_km=17.0,
            fiber_length_km=80.0,
            center_wavelength_nm=1550.0,
        )

        assert isinstance(out_sig, Signal)
        xpt.assert_allclose(out_sig.samples, out_arr)
