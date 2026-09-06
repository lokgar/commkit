"""Tests for multi-stream (MIMO / dual-polarization) Signal support.

Covers factory generation, frame structure, validation, and DSP operations
(upsample, decimate, resample, frequency shift, FIR filtering) on multi-channel signals.
"""

from typing import Any

import pytest
from pydantic import ValidationError

from commkit import filtering, generate_qam, multirate, spectral
from commkit.core import Preamble, Signal, SingleCarrierFrame


class TestMIMOSignalStructure:
    """Tests for multi-channel sample layout, shapes, and validation."""

    def test_signal_generate_mimo(self, backend_device: str, xp: Any) -> None:
        """Verify MIMO signal generation via high-level factories."""
        sig = generate_qam(
            order=4, num_symbols=100, sps=4, symbol_rate=1e6, num_streams=2
        )

        expected_samples = 100 * 4
        assert sig.samples.shape == (2, expected_samples)
        assert sig.num_streams == 2
        assert sig.sps == 4.0
        assert not xp.allclose(sig.samples[0], sig.samples[1])

    def test_signal_mimo_transpose(self, backend_device: str, xp: Any) -> None:
        """Transposition heuristic: shape (100, 2) is transposed to (2, 100)."""
        data = xp.zeros((100, 2))
        sig = Signal(samples=data, sampling_rate=1.0, symbol_rate=1.0)
        assert sig.samples.shape == (2, 100)

    def test_signal_invalid_ndim(self, backend_device: str, xp: Any) -> None:
        """Arrays with >2 dimensions raise ValidationError."""
        data = xp.zeros((2, 2, 2))
        with pytest.raises(ValidationError) as excparams:
            Signal(samples=data, sampling_rate=1.0, symbol_rate=1.0)
        assert "3 dimensions" in str(excparams.value)

    def test_dual_pol_initialization(self, backend_device: str, xp: Any) -> None:
        """Verify initialization of a dual-polarized (2-channel) signal."""
        samples = xp.zeros((2, 100), dtype=complex)
        sig = Signal(samples=samples, sampling_rate=1.0, symbol_rate=1.0)

        assert sig.samples.shape == (2, 100)
        assert sig.num_streams == 2

    def test_siso_fallback(self, backend_device: str, xp: Any) -> None:
        """Verify that 1D samples are correctly treated as single-polarization (SISO)."""
        samples = xp.zeros(100)
        sig = Signal(samples=samples, sampling_rate=1.0, symbol_rate=1.0)
        assert sig.num_streams == 1
        assert sig.samples.ndim == 1


class TestMIMOFrameIntegration:
    """Tests for SingleCarrierFrame multi-stream generation, pilots, and preambles."""

    def test_frame_mimo_generation(self, backend_device: str, xp: Any) -> None:
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
        assert sig.samples.shape == (2, 110)
        assert sig.num_streams == 2
        assert xp.all(sig.samples[:, -10:] == 0)

    def test_frame_mimo_pilots(self, backend_device: str, xp: Any) -> None:
        """Verify that pilot patterns are correctly applied across all MIMO streams."""
        frame = SingleCarrierFrame(
            payload_len=10,
            symbol_rate=1e6,
            num_streams=2,
            pilot_pattern="comb",
            pilot_period=2,
        )
        sig = frame.to_signal(sps=1, pulse_shape="none")
        assert sig.samples.shape == (2, 20)

        mask, _ = frame._generate_pilot_mask()
        assert len(mask) == 20
        assert xp.sum(mask) == 10

    def test_frame_mimo_preamble_broadcasting(
        self, backend_device: str, xp: Any, xpt: Any
    ) -> None:
        """Verify that a multi-stream Barker preamble tiles across all MIMO streams."""
        preamble = Preamble(sequence_type="barker", length=13, num_streams=2)
        frame = SingleCarrierFrame(
            payload_len=20, symbol_rate=1e6, num_streams=2, preamble=preamble
        )

        sig = frame.to_signal(sps=1, pulse_shape="none")
        assert sig.samples.shape == (2, 33)

        xpt.assert_allclose(sig.samples[0, :13], preamble.symbols[0])
        xpt.assert_allclose(sig.samples[1, :13], preamble.symbols[1])

    def test_frame_mimo_waveform(self, backend_device: str, xp: Any) -> None:
        """Verify MIMO waveform generation with pulse shaping."""
        frame = SingleCarrierFrame(payload_len=10, symbol_rate=1e6, num_streams=2)

        sig = frame.to_signal(sps=4, pulse_shape="rect")
        assert sig.samples.shape == (2, 40)
        assert sig.sps == 4.0


class TestMIMODSPOperations:
    """Tests for multi-stream multirate, spectral, and filtering operations."""

    def test_dual_pol_upsample(self, backend_device: str, xp: Any) -> None:
        """Verify that integer upsampling is applied to both polarization channels."""
        samples = xp.ones((2, 100), dtype=complex)
        sig = Signal(samples=samples, sampling_rate=1.0, symbol_rate=1.0)

        sig = multirate.upsample(sig, 2)
        assert sig.samples.shape == (2, 200)
        assert sig.sampling_rate == 2.0

    def test_dual_pol_decimate(self, backend_device: str, xp: Any) -> None:
        """Verify that integer decimation is applied to both polarization channels."""
        samples = xp.ones((2, 200), dtype=complex)
        sig = Signal(samples=samples, sampling_rate=2.0, symbol_rate=1.0)

        sig = multirate.decimate(sig, 2)
        assert sig.samples.shape == (2, 100)
        assert sig.sampling_rate == 1.0

    def test_dual_pol_resample(self, backend_device: str, xp: Any) -> None:
        """Verify that rational resampling is applied to both polarization channels."""
        samples = xp.ones((2, 100), dtype=complex)
        sig = Signal(samples=samples, sampling_rate=1.0, symbol_rate=1.0)

        sig = multirate.resample(sig, up=3, down=2)
        assert sig.samples.shape == (2, 150)
        assert sig.sampling_rate == 1.5

    def test_dual_pol_frequency_shift(
        self, backend_device: str, xp: Any, xpt: Any
    ) -> None:
        """Verify consistent frequency shift phase rotation across channels."""
        samples = xp.ones((2, 100), dtype=complex)
        sig = Signal(samples=samples, sampling_rate=100.0, symbol_rate=100.0)

        sig = spectral.shift_frequency(sig, 25.0)
        expected_sample_1 = xp.exp(1j * xp.pi / 2)

        xpt.assert_allclose(sig.samples[0, 0], 1.0, atol=1e-6)
        xpt.assert_allclose(sig.samples[0, 1], expected_sample_1, atol=1e-6)
        xpt.assert_allclose(sig.samples[1, 1], expected_sample_1, atol=1e-6)

    def test_dual_pol_fir_filter(self, backend_device: str, xp: Any) -> None:
        """Verify that FIR filtering correctly processes multi-stream Signal samples."""
        samples = xp.zeros((2, 10))
        samples[:, 0] = 1.0
        sig = Signal(samples=samples, sampling_rate=1.0, symbol_rate=1.0)

        taps = xp.array([0.5, 0.5])
        sig = filtering.fir_filter(sig, taps)

        assert xp.any(sig.samples != 0)
        assert sig.samples.shape == (2, 10)
