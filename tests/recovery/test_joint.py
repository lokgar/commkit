"""Joint-channel (MIMO) phase-recovery consistency across algorithms."""

from commkit import recovery
from commkit.mapping import Constellation
from tests.common.signals import make_test_mimo_samples, make_test_qam_signal

FS = 1e6  # 1 MHz sampling rate, common to all tests
SNR_DB = 30  # generous SNR so numerical algorithms converge reliably


class TestJointChannels:
    """joint_channels=True produces identical rows and zero inter-channel spread."""

    N = 4096
    PHASE = 0.3  # rad constant carrier phase offset

    def _make_mimo(self, xp, order=16, phase=PHASE, snr_db=SNR_DB):
        mimo, _ = make_test_mimo_samples(
            num_channels=2,
            order=order,
            num_symbols=self.N,
            sps=1,
            snr_db=snr_db,
            seed=1,
            xp=xp,
        )
        return mimo * xp.exp(1j * phase).astype(mimo.dtype)

    def test_bps_joint_rows_identical(self, xp, xpt):
        """joint_channels=True: both phi_full rows are bitwise identical."""
        mimo = self._make_mimo(xp)
        phi = recovery.estimate_carrier_phase(
            mimo, recovery.BPS(joint_channels=True), constellation=Constellation.qam(16)
        ).value
        assert phi.shape == (2, self.N)
        xpt.assert_array_equal(phi[0], phi[1])

    def test_vv_joint_rows_identical(self, xp, xpt):
        """VV joint_channels=True: both phi_full rows are bitwise identical."""
        mimo = self._make_mimo(xp)
        phi = recovery.estimate_carrier_phase(
            mimo,
            recovery.ViterbiViterbi(joint_channels=True),
            constellation=Constellation.qam(16),
        ).value
        assert phi.shape == (2, self.N)
        xpt.assert_array_equal(phi[0], phi[1])

    def test_tikhonov_joint_rows_identical(self, xp, xpt):
        """Tikhonov joint_channels=True: both phi_full rows are bitwise identical."""
        mimo = self._make_mimo(xp)
        phi = recovery.estimate_carrier_phase(
            mimo,
            recovery.Tikhonov(
                linewidth_symbol_periods=1e-4, snr_db=SNR_DB, joint_channels=True
            ),
            constellation=Constellation.qam(16),
        ).value
        assert phi.shape == (2, self.N)
        xpt.assert_array_equal(phi[0], phi[1])

    def test_bps_joint_zero_spread(self, xp, xpt):
        """Joint BPS: inter-channel spread is exactly zero."""
        mimo = self._make_mimo(xp, snr_db=20)
        phi_joint = recovery.estimate_carrier_phase(
            mimo, recovery.BPS(joint_channels=True), constellation=Constellation.qam(16)
        ).value
        xpt.assert_allclose(phi_joint[0], phi_joint[1], atol=1e-12)

    def test_siso_joint_noop(self, xp, xpt):
        """joint_channels=True on SISO returns identical result to False."""
        sig = make_test_qam_signal(
            order=16, num_symbols=self.N, sps=1, snr_db=SNR_DB, seed=7, xp=xp
        )
        phi_a = recovery.estimate_carrier_phase(
            sig.samples,
            recovery.BPS(joint_channels=False),
            constellation=Constellation.qam(16),
        ).value
        phi_b = recovery.estimate_carrier_phase(
            sig.samples,
            recovery.BPS(joint_channels=True),
            constellation=Constellation.qam(16),
        ).value
        xpt.assert_allclose(phi_a, phi_b, atol=1e-10)
