"""Numerical responses of FIR, transfer-function and SOS filters."""

import numpy as np
import pytest
from scipy import signal

from commkit import filtering
from commkit.backend import to_device


def test_fir_response_and_delay(xp, xpt):
    """A symmetric length-five FIR has exactly two samples of delay."""
    taps = xp.asarray([1, 2, 3, 2, 1], dtype=xp.float32)
    result = filtering.filter_response(taps, nfft=128, sampling_rate=1000)
    f, expected = signal.freqz(np.array([1, 2, 3, 2, 1]), worN=128, fs=1000)
    xpt.assert_allclose(result.frequencies, xp.asarray(f), rtol=1e-7, atol=1e-7)
    xpt.assert_allclose(result.response, xp.asarray(expected), rtol=2e-5, atol=2e-5)
    xpt.assert_allclose(result.group_delay[:40], xp.full(40, 0.002), rtol=2e-5)
    assert result.response.dtype == xp.complex64
    assert result.group_delay.dtype == xp.float32


def test_sos_response_and_group_delay(xp, xpt):
    """Group delays add over SOS sections without forming a high-order polynomial."""
    sos = signal.butter(6, 0.2, output="sos")
    result = filtering.filter_response(xp.asarray(sos), nfft=256)
    w, expected = signal.sosfreqz(sos, worN=256)
    expected_delay = sum(signal.group_delay((row[:3], row[3:]), w=w)[1] for row in sos)
    xpt.assert_allclose(result.response, xp.asarray(expected), atol=1e-10, rtol=1e-10)
    xpt.assert_allclose(
        result.group_delay[:200], xp.asarray(expected_delay[:200]), atol=1e-8, rtol=1e-8
    )


def test_impulse_and_step(xp, xpt):
    b, a = xp.asarray([0.5]), xp.asarray([1.0, -0.5])
    impulse = filtering.impulse_response((b, a), num_samples=8)
    step = filtering.step_response((b, a), num_samples=8)
    xpt.assert_allclose(impulse, xp.asarray(0.5 ** np.arange(1, 9)), rtol=1e-6)
    xpt.assert_allclose(step, xp.cumsum(impulse), rtol=1e-6)
    assert type(to_device(impulse, "cpu")) is np.ndarray


def test_response_invalid_arguments():
    with pytest.raises(ValueError, match="nfft"):
        filtering.filter_response(np.ones(3), nfft=0)
    with pytest.raises(ValueError, match="sampling_rate"):
        filtering.filter_response(np.ones(3), sampling_rate=0)
    with pytest.raises(ValueError, match="num_samples"):
        filtering.step_response(np.ones(3), num_samples=0)


def test_zero_response_has_undefined_delay(xp):
    result = filtering.filter_response(xp.asarray([1.0, -1.0]), nfft=32)
    assert bool(xp.isnan(result.group_delay[0]))


def test_long_fir_response_with_folded_coefficients(xp, xpt):
    """An FIR longer than the frequency grid still evaluates at the same bins."""
    taps = np.random.default_rng(7).standard_normal(257).astype(np.float32)
    result = filtering.filter_response(xp.asarray(taps), nfft=32)
    w, expected = signal.freqz(taps, worN=32)
    _, delay = signal.group_delay((taps, [1.0]), w=w)
    xpt.assert_allclose(result.response, xp.asarray(expected), atol=3e-5, rtol=3e-5)
    xpt.assert_allclose(result.group_delay, xp.asarray(delay), atol=5e-3, rtol=2e-4)


def test_custom_frequency_grid(xp, xpt):
    """The user grid supports negative and nonuniform frequencies in Hz."""
    taps = np.array([0.2, 0.7, 0.1], dtype=np.float32)
    frequencies = np.array([-220.0, -37.0, 0.0, 14.0, 200.0])
    result = filtering.filter_response(
        xp.asarray(taps), sampling_rate=1000, frequencies=xp.asarray(frequencies)
    )
    _, expected = signal.freqz(taps, worN=frequencies, fs=1000)
    xpt.assert_allclose(result.frequencies, xp.asarray(frequencies))
    xpt.assert_allclose(result.response, xp.asarray(expected), atol=2e-6)


def test_sos_time_responses(xp, xpt):
    """SOS impulse and step responses agree with an independent reference."""
    sos = signal.butter(4, 0.2, output="sos")
    device_sos = xp.asarray(sos)
    impulse = np.zeros(64)
    impulse[0] = 1
    expected_impulse = signal.sosfilt(sos, impulse)
    expected_step = signal.sosfilt(sos, np.ones(64))
    xpt.assert_allclose(
        filtering.impulse_response(device_sos, num_samples=64),
        xp.asarray(expected_impulse),
        atol=1e-10,
        rtol=1e-10,
    )
    xpt.assert_allclose(
        filtering.step_response(device_sos, num_samples=64),
        xp.asarray(expected_step),
        atol=1e-10,
        rtol=1e-10,
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_custom_frequency_periodicity(xp, xpt, dtype):
    """Frequency response is periodic even for frequencies far beyond one cycle."""
    f = xp.asarray([0.125, 1_000_000_000.125, -999_999_999.875], dtype=xp.float64)
    result = filtering.filter_response(xp.asarray([1, 1], dtype=dtype), frequencies=f)
    expected = xp.asarray([1 + np.exp(-0.25j * np.pi)] * 3, dtype=result.response.dtype)
    xpt.assert_allclose(result.response, expected, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_nyquist_zero_has_undefined_delay(xp, dtype):
    """The FIR 1+z^-1 vanishes at Nyquist, so group delay is undefined."""
    result = filtering.filter_response(
        xp.asarray([1, 1], dtype=dtype), frequencies=xp.asarray([0.5])
    )
    assert bool(xp.isnan(result.group_delay[0]))


def test_low_cutoff_sos_response_uses_double_state(xp, xpt):
    """A low-cutoff float32 SOS must match its float64-state recurrence."""
    sos = signal.butter(4, 1e-3, output="sos").astype(np.float32)
    n = 20_000
    expected = signal.sosfilt(sos.astype(np.float64), np.ones(n))
    actual = filtering.step_response(xp.asarray(sos), num_samples=n)
    assert actual.dtype == xp.float32
    xpt.assert_allclose(
        actual, xp.asarray(expected, dtype=xp.float32), atol=2e-6, rtol=2e-6
    )


def test_low_cutoff_transfer_response_uses_double_state(xp, xpt):
    """A single biquad expressed as (b, a) gets the same precision as SOS."""
    sos = signal.butter(2, 1e-3, output="sos").astype(np.float32)
    b, a = sos[0, :3], sos[0, 3:]
    n = 20_000
    expected = signal.lfilter(b.astype(np.float64), a.astype(np.float64), np.ones(n))
    actual = filtering.step_response((xp.asarray(b), xp.asarray(a)), num_samples=n)
    assert actual.dtype == xp.float32
    xpt.assert_allclose(
        actual, xp.asarray(expected, dtype=xp.float32), atol=2e-6, rtol=2e-6
    )


def test_fir_time_responses_are_the_taps(xp, xpt):
    """An FIR impulse response is its taps, zero-padded or truncated; a step their sum."""
    taps = xp.asarray([0.5, -1.0, 2.0], dtype=xp.float32)
    xpt.assert_allclose(
        filtering.impulse_response(taps, num_samples=5),
        xp.asarray([0.5, -1.0, 2.0, 0.0, 0.0]),
    )
    xpt.assert_allclose(
        filtering.impulse_response(taps, num_samples=2), xp.asarray([0.5, -1.0])
    )
    xpt.assert_allclose(
        filtering.step_response(taps, num_samples=4), xp.asarray([0.5, -0.5, 1.5, 1.5])
    )
    scaled = filtering.impulse_response((taps, xp.asarray([2.0])), num_samples=3)
    xpt.assert_allclose(scaled, xp.asarray([0.25, -0.5, 1.0]))
    assert filtering.impulse_response(taps, num_samples=3).dtype == xp.float32


def test_sos_time_response_normalizes_a0(xp, xpt):
    """Sections with a0 != 1 describe the same filter as the normalized ones."""
    sos = signal.butter(4, 0.1, output="sos")
    scaled = sos * np.array([[2.0], [0.5]])
    xpt.assert_allclose(
        filtering.impulse_response(xp.asarray(scaled), num_samples=64),
        xp.asarray(signal.sosfilt(sos, np.r_[1.0, np.zeros(63)])),
        atol=1e-12,
    )


def test_improper_transfer_function_time_response(xp, xpt):
    """A (b, a) pair with more zeros than poles matches the direct form."""
    b = np.array([0.2, 0.5, 0.3, -0.1])
    a = np.array([1.0, -0.6])
    expected = signal.lfilter(b, a, np.r_[1.0, np.zeros(31)])
    actual = filtering.impulse_response((xp.asarray(b), xp.asarray(a)), num_samples=32)
    xpt.assert_allclose(actual, xp.asarray(expected), atol=1e-12)


def test_complex_transfer_function_time_response(xp, xpt):
    """Complex coefficients are filtered as complex, in their precision."""
    b = np.array([1.0 + 0.5j])
    a = np.array([1.0, -0.5j])
    expected = signal.lfilter(b, a, np.ones(16))
    actual = filtering.step_response((xp.asarray(b), xp.asarray(a)), num_samples=16)
    assert actual.dtype == xp.complex128
    xpt.assert_allclose(actual, xp.asarray(expected), atol=1e-12)
