"""Blind phase search (BPS) carrier phase recovery."""

import numpy as np
import pytest

from commkit import recovery
from commkit.core import Signal
from tests.common.signals import (
    make_test_mimo_samples,
    make_test_qam_signal,
    make_test_symbols,
)

FS = 1e6  # 1 MHz sampling rate, common to all tests
SNR_DB = 30  # generous SNR so numerical algorithms converge reliably


class TestCprBps:
    @pytest.mark.parametrize("order", [16, 64])
    def test_phase_residual(self, xp, order):
        """BPS CPR: RMS phase residual < 0.05 rad for constant phase offset."""
        sig = make_test_qam_signal(
            order=order, num_symbols=1024, sps=1, symbol_rate=FS, xp=xp
        )
        phi_true = 0.2  # radians
        sig = sig.replace(samples=sig.samples * xp.exp(1j * phi_true))

        phase_est = recovery.recover_carrier_phase_bps(
            sig.samples, modulation="qam", order=order
        )
        corrected = recovery.correct_carrier_phase(sig.samples, phase_est)

        phase_resid = recovery.recover_carrier_phase_bps(
            corrected, modulation="qam", order=order
        )
        assert float(xp.sqrt(xp.mean(phase_resid**2))) < 0.05

    def test_output_shape_siso(self, xp):
        """BPS CPR: 1D input -> 1D phase output of same length."""
        sig = make_test_qam_signal(
            order=16, num_symbols=512, sps=1, symbol_rate=FS, xp=xp
        )
        phase = recovery.recover_carrier_phase_bps(
            sig.samples, modulation="qam", order=16
        )
        assert phase.shape == sig.samples.shape

    def test_output_shape_mimo(self, xp):
        """BPS CPR: 2D input (C, N) -> 2D phase output (C, N)."""
        mimo, _ = make_test_mimo_samples(
            num_channels=2, order=16, num_symbols=512, sps=1, xp=xp
        )
        phase = recovery.recover_carrier_phase_bps(mimo, modulation="qam", order=16)
        assert phase.shape == mimo.shape

    def test_too_short_raises(self, xp):
        """BPS CPR: signal shorter than block_size raises ValueError."""
        sig = make_test_qam_signal(
            order=16, num_symbols=20, sps=1, symbol_rate=FS, xp=xp
        )
        with pytest.raises(ValueError, match="shorter than block_size"):
            recovery.recover_carrier_phase_bps(
                sig.samples[:10], modulation="qam", order=16, block_size=32
            )


class TestBPS:
    """Tests for recover_carrier_phase_bps."""

    def _qam16_symbols(self, xp, N=512, seed=2):
        return make_test_symbols(
            scheme="qam", order=16, num_symbols=N, seed=seed, xp=xp
        )

    def _qpsk_symbols(self, xp, N=512, seed=3):
        return make_test_symbols(scheme="psk", order=4, num_symbols=N, seed=seed, xp=xp)

    def test_siso_qam16_output_shape(self, xp):
        """SISO QAM16 (square QAM fast path): output is (N,) float64."""
        syms = self._qam16_symbols(xp)
        phi_est = recovery.recover_carrier_phase_bps(
            syms, "qam", 16, num_test_phases=32, block_size=32
        )
        assert phi_est.shape == syms.shape
        assert phi_est.dtype == xp.float64

    def test_siso_qam16_recovers_static_phase(self, xp):
        """BPS should estimate a static QAM16 phase offset to within π/8 tolerance."""

        phi_true = 0.25
        syms = self._qam16_symbols(xp, N=512)
        rotated = syms * xp.asarray(np.complex64(np.exp(1j * phi_true)))
        phi_est = recovery.recover_carrier_phase_bps(
            rotated, "qam", 16, num_test_phases=64, block_size=32
        )
        phi_mean = float(xp.mean(phi_est))
        # 4-fold ambiguity: allow ±π/8 residual
        residual = (phi_mean - phi_true + np.pi / 4) % (np.pi / 2) - np.pi / 4
        assert abs(residual) < 0.15, (
            f"Residual phase error too large: {residual:.3f} rad"
        )

    def test_siso_qpsk_general_path(self, xp):
        """SISO QPSK (non-square: triggers general distance path): output shape correct."""
        syms = self._qpsk_symbols(xp, N=256)
        phi_est = recovery.recover_carrier_phase_bps(
            syms, "psk", 4, num_test_phases=16, block_size=32
        )
        assert phi_est.shape == syms.shape

    def test_mimo_output_shape(self, xp):
        """MIMO input (C, N): output shape is (C, N)."""

        C, N = 2, 256
        rng = np.random.default_rng(7)
        syms = xp.asarray(
            (rng.standard_normal((C, N)) + 1j * rng.standard_normal((C, N))).astype(
                np.complex64
            )
        )
        phi_est = recovery.recover_carrier_phase_bps(
            syms, "qam", 16, num_test_phases=16, block_size=32
        )
        assert phi_est.shape == (C, N)

    def test_block_size_too_large_raises(self, xp):
        """block_size > N should raise ValueError."""
        syms = self._qam16_symbols(xp, N=16)
        with pytest.raises(ValueError, match="block_size"):
            recovery.recover_carrier_phase_bps(syms, "qam", 16, block_size=64)


class TestSignalInputBpsAndCorrectCarrierPhase:
    """Signal-awareness for recover_carrier_phase_bps and correct_carrier_phase."""

    def test_bps_signal_input_uses_metadata(self, xp, xpt):
        """Signal input: modulation/order come from the signal's metadata."""
        sig = make_test_qam_signal(
            order=16, num_symbols=512, sps=1, symbol_rate=FS, xp=xp
        )

        phi_sig = recovery.recover_carrier_phase_bps(sig)
        phi_arr = recovery.recover_carrier_phase_bps(
            sig.samples, modulation="qam", order=16
        )

        assert not isinstance(phi_sig, Signal)  # phase estimate stays a raw array
        xpt.assert_allclose(phi_sig, phi_arr)

    def test_correct_carrier_phase_signal_input_returns_signal(self, xp, xpt):
        """Signal input returns a Signal with the phase-corrected samples."""
        sig = make_test_qam_signal(
            order=16, num_symbols=256, sps=1, symbol_rate=FS, xp=xp
        )
        phase = xp.full(256, 0.3)

        out_sig = recovery.correct_carrier_phase(sig, phase)
        out_arr = recovery.correct_carrier_phase(sig.samples, phase)

        assert isinstance(out_sig, Signal)
        xpt.assert_allclose(out_sig.samples, out_arr)
