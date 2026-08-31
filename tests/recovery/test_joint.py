"""Joint-channel (MIMO) phase-recovery consistency across algorithms."""

import numpy as np

from commkit import generate_qam, recovery, spectral
from commkit.impairments import apply_awgn

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


class TestJointChannels:
    """joint_channels=True produces identical rows and zero inter-channel spread."""

    N = 4096
    PHASE = 0.3  # rad constant carrier phase offset

    def _make_mimo(self, xp, order=16, phase=PHASE, snr_db=SNR_DB):
        sig_a = _qam_signal(xp, order, self.N, snr_db=snr_db, seed=1)
        sig_b = _qam_signal(xp, order, self.N, snr_db=snr_db, seed=2)
        mimo = xp.stack([sig_a.samples, sig_b.samples], axis=0)
        mimo = mimo * xp.exp(1j * phase).astype(mimo.dtype)
        return mimo

    def test_bps_joint_rows_identical(self, backend_device, xp):
        """joint_channels=True: both phi_full rows are bitwise identical."""
        mimo = self._make_mimo(xp)
        phi = recovery.recover_carrier_phase_bps(
            mimo, "qam", 16, joint_channels=True, cycle_slip_correction=False
        )
        assert phi.shape == (2, self.N)
        phi_np = phi if xp is np else phi.get()
        np.testing.assert_array_equal(phi_np[0], phi_np[1])

    def test_vv_joint_rows_identical(self, backend_device, xp):
        """VV joint_channels=True: both phi_full rows are bitwise identical."""
        mimo = self._make_mimo(xp)
        phi = recovery.recover_carrier_phase_viterbi_viterbi(
            mimo, "qam", 16, joint_channels=True, cycle_slip_correction=False
        )
        assert phi.shape == (2, self.N)
        phi_np = phi if xp is np else phi.get()
        np.testing.assert_array_equal(phi_np[0], phi_np[1])

    def test_tikhonov_joint_rows_identical(self, backend_device, xp):
        """Tikhonov joint_channels=True: both phi_full rows are bitwise identical."""
        mimo = self._make_mimo(xp)
        phi = recovery.recover_carrier_phase_tikhonov(
            mimo,
            "qam",
            16,
            linewidth_symbol_periods=1e-4,
            snr_db=SNR_DB,
            joint_channels=True,
            cycle_slip_correction=False,
        )
        assert phi.shape == (2, self.N)
        phi_np = phi if xp is np else phi.get()
        np.testing.assert_array_equal(phi_np[0], phi_np[1])

    def test_bps_joint_zero_spread(self, backend_device, xp):
        """Joint BPS: inter-channel spread is exactly zero."""
        mimo = self._make_mimo(xp, snr_db=20)
        phi_joint = recovery.recover_carrier_phase_bps(
            mimo, "qam", 16, joint_channels=True, cycle_slip_correction=False
        )
        phi_np = phi_joint if xp is np else phi_joint.get()
        assert float(np.std(phi_np[0] - phi_np[1])) == 0.0

    def test_siso_joint_noop(self, backend_device, xp):
        """joint_channels=True on SISO returns identical result to False."""
        sig = _qam_signal(xp, 16, self.N, seed=7)
        phi_a = recovery.recover_carrier_phase_bps(
            sig.samples, "qam", 16, joint_channels=False, cycle_slip_correction=False
        )
        phi_b = recovery.recover_carrier_phase_bps(
            sig.samples, "qam", 16, joint_channels=True, cycle_slip_correction=False
        )
        phi_a_np = phi_a if xp is np else phi_a.get()
        phi_b_np = phi_b if xp is np else phi_b.get()
        np.testing.assert_allclose(phi_a_np, phi_b_np, atol=1e-10)
