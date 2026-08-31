"""Tests for multi-stream (MIMO / dual-polarization) Signal support.

Covers factory generation, frame structure, validation, and DSP operations
(upsample, decimate, resample, frequency shift, FIR filtering) on multi-channel signals.
"""

import pytest
from pydantic import ValidationError

from commkit import filtering, generate_qam, multirate, spectral
from commkit.core import Preamble, Signal, SingleCarrierFrame


def test_signal_generate_mimo(backend_device, xp):
    """Verify MIMO signal generation via high-level factories."""
    # Test MIMO generation via factories
    sig = generate_qam(order=4, num_symbols=100, sps=4, symbol_rate=1e6, num_streams=2)

    # Check shape: (num_streams, num_symbols * sps)
    expected_samples = 100 * 4
    assert sig.samples.shape == (2, expected_samples)
    assert sig.num_streams == 2
    assert sig.sps == 4.0

    # Check if streams are not identical (random seed should diverge or be handled)
    assert not xp.allclose(sig.samples[0], sig.samples[1])


def test_frame_mimo_generation(backend_device, xp):
    """Verify basic MIMO frame generation with guard intervals."""
    frame = SingleCarrierFrame(
        payload_len=100,
        symbol_rate=1e6,
        num_streams=2,
        pilot_pattern="none",
        guard_type="zero",
        guard_len=10,
    )

    sig = frame.to_signal(sps=1, pulse_shape="none")
    # Length: 100 payload + 10 guard = 110 symbols
    assert sig.samples.shape == (2, 110)
    assert sig.num_streams == 2

    # Check guard interval (zeros)
    # Last 10 samples (axis=-1)
    assert xp.all(sig.samples[:, -10:] == 0)


def test_frame_mimo_pilots(backend_device, xp):
    """Verify that pilot patterns are correctly applied across all MIMO streams."""
    frame = SingleCarrierFrame(
        payload_len=10,
        symbol_rate=1e6,
        num_streams=2,
        pilot_pattern="comb",
        pilot_period=2,
    )
    # len=10 payload. period=2 -> 1 pilot, 1 data.
    # total len = 10 data -> 10 periods -> 20 symbols.

    sig = frame.to_signal(sps=1, pulse_shape="none")
    assert sig.samples.shape == (2, 20)

    # Check mask and body
    mask, _ = frame._generate_pilot_mask()
    # Mask is 1D (time), applicable to all streams
    assert len(mask) == 20
    assert xp.sum(mask) == 10


def test_frame_mimo_preamble_broadcasting(backend_device, xp, xpt):
    """Verify that a multi-stream Barker preamble tiles identically across all MIMO streams."""
    # Create Barker-13 preamble with num_streams=2 so _symbols is (2, 13)
    preamble = Preamble(sequence_type="barker", length=13, num_streams=2)
    frame = SingleCarrierFrame(
        payload_len=20, symbol_rate=1e6, num_streams=2, preamble=preamble
    )

    sig = frame.to_signal(sps=1, pulse_shape="none")
    # Total: 13 preamble + 20 payload = 33
    assert sig.samples.shape == (2, 33)

    # Check preamble on both streams (Barker tiles identically)
    # (Channels, Time)
    xpt.assert_allclose(sig.samples[0, :13], preamble.symbols[0])
    xpt.assert_allclose(sig.samples[1, :13], preamble.symbols[1])


def test_frame_mimo_waveform(backend_device, xp):
    """Verify MIMO waveform generation with pulse shaping."""
    frame = SingleCarrierFrame(payload_len=10, symbol_rate=1e6, num_streams=2)

    sig = frame.to_signal(sps=4, pulse_shape="rect")
    # 10 symbols * 4 sps = 40 samples
    assert sig.samples.shape == (2, 40)
    assert sig.sps == 4.0


def test_signal_mimo_transpose(backend_device, xp):
    """
    Test the transposition heuristic in Signal.validate_samples.
    If we pass (100, 2), it should detect Time > Channels and transpose to (2, 100).
    """
    data = xp.zeros((100, 2))
    # Note: logger.warning is used, not warnings.warn, so pytest.warns won't catch it.
    # We just verify the transposition logic happened.
    sig = Signal(samples=data, sampling_rate=1.0, symbol_rate=1.0)
    assert sig.samples.shape == (2, 100)


def test_signal_invalid_ndim(backend_device, xp):
    """Test >2D array raises ValueError (wrapped in ValidationError)."""
    data = xp.zeros((2, 2, 2))
    with pytest.raises(ValidationError) as excparams:
        Signal(samples=data, sampling_rate=1.0, symbol_rate=1.0)
    assert "3 dimensions" in str(excparams.value)


# -----------------------------------------------------------------------------
# DSP Operations on Multi-Stream Signals
# -----------------------------------------------------------------------------


def test_dual_pol_initialization(backend_device, xp):
    """Verify initialization of a dual-polarized (2-channel) signal."""
    samples = xp.random.randn(2, 100) + 1j * xp.random.randn(2, 100)
    sig = Signal(samples=samples, sampling_rate=1.0, symbol_rate=1.0)

    assert sig.samples.shape == (2, 100)
    assert sig.num_streams == 2


def test_dual_pol_upsample(backend_device, xp):
    """Verify that integer upsampling is correctly applied to both polarization channels."""
    samples = xp.random.randn(2, 100)
    sig = Signal(samples=samples, sampling_rate=1.0, symbol_rate=1.0)

    sig = multirate.upsample(sig, 2)
    assert sig.samples.shape == (2, 200)
    assert sig.sampling_rate == 2.0


def test_dual_pol_decimate(backend_device, xp):
    """Verify that integer decimation is correctly applied to both polarization channels."""
    samples = xp.random.randn(2, 200)
    sig = Signal(samples=samples, sampling_rate=2.0, symbol_rate=1.0)

    sig = multirate.decimate(sig, 2)
    assert sig.samples.shape == (2, 100)
    assert sig.sampling_rate == 1.0


def test_dual_pol_resample(backend_device, xp):
    """Verify that rational resampling is correctly applied to both polarization channels."""
    samples = xp.random.randn(2, 100)
    sig = Signal(samples=samples, sampling_rate=1.0, symbol_rate=1.0)

    sig = multirate.resample(sig, up=3, down=2)
    assert sig.samples.shape == (2, 150)
    assert sig.sampling_rate == 1.5


def test_dual_pol_frequency_shift(backend_device, xp, xpt):
    """Verify that frequency shifting results in consistent phase rotation across channels."""
    samples = xp.ones((2, 100), dtype=complex)
    sig = Signal(samples=samples, sampling_rate=100.0, symbol_rate=100.0)

    # Shift by 25 Hz (1/4 of Fs) -> exp(j*pi/2 * n) per sample
    sig = spectral.shift_frequency(sig, 25.0)

    expected_sample_1 = xp.exp(1j * xp.pi / 2)

    xpt.assert_allclose(sig.samples[0, 0], 1.0, atol=1e-6)
    xpt.assert_allclose(sig.samples[0, 1], expected_sample_1, atol=1e-6)
    xpt.assert_allclose(sig.samples[1, 1], expected_sample_1, atol=1e-6)


def test_dual_pol_fir_filter(backend_device, xp):
    """Verify that FIR filtering correctly processes multi-stream Signal samples."""
    samples = xp.zeros((2, 10))
    samples[:, 0] = 1.0
    sig = Signal(samples=samples, sampling_rate=1.0, symbol_rate=1.0)

    taps = xp.array([0.5, 0.5])
    sig = filtering.fir_filter(sig, taps)

    assert xp.any(sig.samples != 0)
    assert sig.samples.shape == (2, 10)


def test_siso_fallback(backend_device, xp):
    """Verify that 1D samples are correctly treated as single-polarization (SISO)."""
    samples = xp.random.randn(100)
    sig = Signal(samples=samples, sampling_rate=1.0, symbol_rate=1.0)
    assert sig.num_streams == 1
    assert sig.samples.ndim == 1
