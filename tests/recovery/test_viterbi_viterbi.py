"""Viterbi-Viterbi carrier phase recovery."""

import numpy as np
import pytest

from commkit import generate_psk, generate_qam, recovery, spectral
from commkit.impairments import apply_awgn
from commkit.mapping import gray_constellation

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


class TestCprViterbiViterbi:
    @pytest.mark.parametrize(
        "order,modulation,block_size",
        [
            (4, "psk", 16),
            (4, "psk", 32),
            (4, "psk", 64),
            (16, "qam", 16),
            (16, "qam", 32),
            (16, "qam", 64),
            (64, "qam", 32),
            (64, "qam", 64),
        ],
    )
    def test_phase_residual(self, backend_device, xp, order, modulation, block_size):
        """VV CPR: mean phase estimate within 0.1 rad of true carrier phase (mod M-fold)."""
        sig = (
            _qam_signal(xp, order, 2048)
            if modulation == "qam"
            else _psk_signal(xp, order, 2048)
        )
        phi_true = 0.3  # radians
        sig.samples = sig.samples * xp.exp(1j * phi_true)

        phase_est = recovery.recover_carrier_phase_viterbi_viterbi(
            sig.samples, modulation=modulation, order=order, block_size=block_size
        )

        M = 4 if modulation == "qam" else order
        step = 2 * np.pi / M
        err = float(xp.mean(phase_est)) - phi_true
        err = err - step * round(err / step)  # wrap to [-step/2, step/2)
        assert abs(err) < 0.1

    def test_output_shape_siso(self, backend_device, xp):
        """VV CPR: 1D input -> 1D phase output of same length."""
        sig = _qam_signal(xp, 16, 512)
        phase = recovery.recover_carrier_phase_viterbi_viterbi(
            sig.samples, modulation="qam", order=16
        )
        assert phase.shape == sig.samples.shape

    def test_output_shape_mimo(self, backend_device, xp):
        """VV CPR: 2D input (C, N) -> 2D phase output (C, N)."""
        sig_a = _qam_signal(xp, 4, 512)
        sig_b = _qam_signal(xp, 4, 512)
        mimo = xp.stack([sig_a.samples, sig_b.samples])  # (2, 512)
        phase = recovery.recover_carrier_phase_viterbi_viterbi(
            mimo, modulation="qam", order=4
        )
        assert phase.shape == mimo.shape

    def test_too_short_raises(self, backend_device, xp):
        """VV CPR: signal shorter than block_size raises ValueError."""
        sig = _qam_signal(xp, 4, 20)
        with pytest.raises(ValueError, match="shorter than block_size"):
            recovery.recover_carrier_phase_viterbi_viterbi(
                sig.samples[:10], modulation="qam", order=4, block_size=32
            )


class TestViterbiViterbi:
    """Tests for recover_carrier_phase_viterbi_viterbi."""

    def _qpsk_symbols(self, xp, N=512, seed=0):
        import numpy as np

        rng = np.random.default_rng(seed)
        bits = rng.integers(0, 4, N)
        angles = (2 * np.pi / 4) * bits + np.pi / 4
        return xp.asarray(np.exp(1j * angles).astype(np.complex64))

    def _qam16_symbols(self, xp, N=512, seed=1):
        import numpy as np

        rng = np.random.default_rng(seed)
        const = gray_constellation("qam", 16)
        idx = rng.integers(0, 16, N)
        return xp.asarray(const[idx].astype(np.complex64))

    def test_siso_qpsk_output_shape(self, backend_device, xp):
        """SISO QPSK: output is (N,) float64."""
        syms = self._qpsk_symbols(xp)
        phi_est = recovery.recover_carrier_phase_viterbi_viterbi(
            syms, "psk", 4, block_size=32
        )
        assert phi_est.shape == syms.shape
        assert phi_est.dtype == xp.float64

    def test_siso_qpsk_recovers_static_phase(self, backend_device, xp):
        """VV tracks applied phase: difference between rotated and unrotated estimate equals phi_true mod π/2."""
        import numpy as np

        phi_true = 0.3  # radians
        syms = self._qpsk_symbols(xp, N=512)
        # Baseline (no rotation)
        phi_base = float(
            xp.mean(
                recovery.recover_carrier_phase_viterbi_viterbi(
                    syms, "psk", 4, block_size=32
                )
            )
        )
        # Rotated by phi_true
        rotated = syms * xp.asarray(np.complex64(np.exp(1j * phi_true)))
        phi_rot = float(
            xp.mean(
                recovery.recover_carrier_phase_viterbi_viterbi(
                    rotated, "psk", 4, block_size=32
                )
            )
        )
        # The shift should equal phi_true modulo π/2
        delta = phi_rot - phi_base
        residual = (delta - phi_true + np.pi / 4) % (np.pi / 2) - np.pi / 4
        assert abs(residual) < 0.15, (
            f"Phase tracking error too large: {residual:.3f} rad"
        )

    def test_siso_qam16_output_shape(self, backend_device, xp):
        """SISO QAM16: output shape matches input."""
        syms = self._qam16_symbols(xp)
        phi_est = recovery.recover_carrier_phase_viterbi_viterbi(
            syms, "qam", 16, block_size=32
        )
        assert phi_est.shape == syms.shape

    def test_mimo_output_shape(self, backend_device, xp):
        """MIMO input (C, N): output shape is (C, N)."""
        import numpy as np

        C, N = 2, 256
        syms = xp.asarray(
            np.random.default_rng(5).standard_normal((C, N)).astype(np.float32)
            + 1j * np.random.default_rng(6).standard_normal((C, N)).astype(np.float32)
        )
        phi_est = recovery.recover_carrier_phase_viterbi_viterbi(
            syms, "qam", 16, block_size=32
        )
        assert phi_est.shape == (C, N)

    def test_block_size_too_large_raises(self, backend_device, xp):
        """block_size > N should raise ValueError."""
        syms = self._qpsk_symbols(xp, N=16)
        with pytest.raises(ValueError, match="block_size"):
            recovery.recover_carrier_phase_viterbi_viterbi(
                syms, "psk", 4, block_size=64
            )
