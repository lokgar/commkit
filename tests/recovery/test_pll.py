"""Decision-directed PLL carrier phase recovery."""

import numpy as np
import pytest

from commkit import recovery
from commkit.core import Signal
from commkit.mapping import Constellation
from tests.common.conversions import to_numpy
from tests.common.signals import (
    make_test_mimo_samples,
    make_test_qam_signal,
    make_test_symbols,
)

FS = 1e6  # 1 MHz sampling rate, common to all tests
SNR_DB = 30  # generous SNR so numerical algorithms converge reliably


class TestDDPLLEnhancements:
    """DD-PLL joint_channels and cycle_slip_correction parameters."""

    N = 4096
    PHASE = 0.4  # rad constant phase offset

    def _make_mimo(self, xp, order=16, phase=PHASE, snr_db=SNR_DB):
        mimo, _ = make_test_mimo_samples(
            num_channels=2,
            order=order,
            num_symbols=self.N,
            sps=1,
            snr_db=snr_db,
            seed=10,
            xp=xp,
        )
        return mimo * xp.exp(1j * phase).astype(mimo.dtype)

    def test_joint_rows_identical(self, xp, xpt):
        """PI loop + joint_channels=True: both phi_full rows are bitwise identical."""
        mimo = self._make_mimo(xp)
        phi = recovery.estimate_carrier_phase(
            mimo,
            recovery.PLL(mu=1e-2, joint_channels=True),
            constellation=Constellation.qam(16),
        ).value
        assert phi.shape == (2, self.N)
        xpt.assert_array_equal(phi[0], phi[1])

    def test_siso_joint_noop(self, xp, xpt):
        """joint_channels=True on SISO returns identical result to False."""
        sig = make_test_qam_signal(
            order=16, num_symbols=self.N, sps=1, snr_db=SNR_DB, seed=12, xp=xp
        )
        phi_a = recovery.estimate_carrier_phase(
            sig.samples,
            recovery.PLL(mu=1e-2, joint_channels=False),
            constellation=Constellation.qam(16),
        ).value
        phi_b = recovery.estimate_carrier_phase(
            sig.samples,
            recovery.PLL(mu=1e-2, joint_channels=True),
            constellation=Constellation.qam(16),
        ).value
        xpt.assert_allclose(phi_a, phi_b, atol=1e-10)

    def test_cycle_slip_shape(self, xp):
        """cycle_slip_correction=True (PI loop) returns correct shape."""
        sig = make_test_qam_signal(
            order=16, num_symbols=self.N, sps=1, snr_db=SNR_DB, xp=xp
        )
        phi = recovery.estimate_carrier_phase(
            sig.samples,
            recovery.PLL(mu=1e-2, cycle_slip=recovery.CycleSlip()),
            constellation=Constellation.qam(16),
        ).value
        assert phi.shape == sig.samples.shape

    def test_joint_cycle_slip_mimo_rows_identical(self, xp, xpt):
        """joint_channels=True + cycle_slip_correction=True: rows remain identical."""
        mimo = self._make_mimo(xp)
        phi = recovery.estimate_carrier_phase(
            mimo,
            recovery.PLL(
                mu=1e-2,
                joint_channels=True,
                cycle_slip=recovery.CycleSlip(history=1000),
            ),
            constellation=Constellation.qam(16),
        ).value
        assert phi.shape == (2, self.N)
        xpt.assert_array_equal(phi[0], phi[1])


class TestDDPLL:
    """Tests for recover_carrier_phase_pll."""

    def _qpsk_symbols(self, xp, N=512, seed=10):
        return make_test_symbols(scheme="psk", order=4, num_symbols=N, seed=seed, xp=xp)

    def test_siso_output_shape(self, xp):
        """SISO: output is (N,) float64."""
        syms = self._qpsk_symbols(xp)
        phi = recovery.estimate_carrier_phase(
            syms, recovery.PLL(mu=1e-2), constellation=Constellation.psk(4)
        ).value
        assert phi.shape == syms.shape
        assert phi.dtype == xp.float64

    def test_mimo_output_shape(self, xp):
        """MIMO (C, N): output shape is (C, N)."""
        C, N = 2, 256
        syms, _ = make_test_mimo_samples(
            num_channels=C, order=4, num_symbols=N, sps=1, seed=11, xp=xp
        )
        phi = recovery.estimate_carrier_phase(
            syms, recovery.PLL(mu=1e-2), constellation=Constellation.psk(4)
        ).value
        assert phi.shape == (C, N)

    def test_second_order_loop(self, xp):
        """beta > 0 engages 2nd-order loop without raising."""
        syms = self._qpsk_symbols(xp, N=256)
        phi = recovery.estimate_carrier_phase(
            syms, recovery.PLL(mu=0.02, beta=1e-4), constellation=Constellation.psk(4)
        ).value
        assert phi.shape == syms.shape

    def test_phase_init_applied(self, xp):
        """phase_init shifts the starting phase estimate."""
        syms = self._qpsk_symbols(xp, N=256)
        phi_init = 0.5
        phi = recovery.estimate_carrier_phase(
            syms,
            recovery.PLL(mu=1e-2, phase_init=phi_init),
            constellation=Constellation.psk(4),
        ).value
        assert abs(float(phi[0]) - phi_init) < 0.5

    def test_bandwidth_shortcut_shape(self, xp):
        """mu=None opts into the loop_bandwidth_normalized shortcut; shape/dtype hold."""
        syms = self._qpsk_symbols(xp, N=512)
        phi = recovery.estimate_carrier_phase(
            syms, recovery.PLL(bandwidth=1e-3), constellation=Constellation.psk(4)
        ).value
        assert phi.shape == syms.shape
        assert phi.dtype == xp.float64

    def test_bandwidth_shortcut_equals_raw_gains(self, xp, xpt):
        """mu=None, bandwidth=B is identical to raw mu=4B, beta=4B² (the resolver mapping)."""
        N = 512
        syms = self._qpsk_symbols(xp, N=N, seed=7)
        syms = syms * xp.exp(1j * 0.2).astype(syms.dtype)

        B = 1e-3
        phi_bw = recovery.estimate_carrier_phase(
            syms, recovery.PLL(bandwidth=B), constellation=Constellation.psk(4)
        ).value
        phi_raw = recovery.estimate_carrier_phase(
            syms,
            recovery.PLL(mu=4.0 * B, beta=4.0 * B**2),
            constellation=Constellation.psk(4),
        ).value
        xpt.assert_allclose(phi_bw, phi_raw, rtol=1e-6, atol=1e-9)

    def test_first_vs_second_order_under_frequency_offset(self, xp):
        """Under a frequency offset, a 1st-order loop (beta=0) settles to a constant
        phase lag; a 2nd-order loop (beta>0) nulls it to ~zero steady-state error."""
        N = 4000
        clean = self._qpsk_symbols(xp, N=N, seed=5)
        df = 1e-3
        n = xp.arange(N, dtype=xp.float64)
        ramp = xp.exp(1j * df * n).astype(clean.dtype)
        syms = clean * ramp

        true_phase = to_numpy(df * n)
        tail = slice(N // 2, N)

        phi1 = recovery.estimate_carrier_phase(
            syms, recovery.PLL(mu=0.02, beta=0.0), constellation=Constellation.psk(4)
        ).value
        phi2 = recovery.estimate_carrier_phase(
            syms, recovery.PLL(mu=0.02, beta=2e-4), constellation=Constellation.psk(4)
        ).value
        phi1_np = to_numpy(phi1)
        phi2_np = to_numpy(phi2)

        lag1 = abs(float(np.mean(np.unwrap(phi1_np[tail]) - true_phase[tail])))
        lag2 = abs(float(np.mean(np.unwrap(phi2_np[tail]) - true_phase[tail])))
        assert lag1 > 1e-3
        assert lag2 < lag1 / 100.0

    def test_invalid_bandwidth_raises(self):
        """A bandwidth outside (0, 0.5) raises on construction."""
        with pytest.raises(ValueError, match="bandwidth"):
            recovery.PLL(bandwidth=0.6)

    def test_beta_without_mu_raises(self):
        """Passing beta with mu=None is ambiguous and must raise ValueError."""
        with pytest.raises(ValueError, match="beta requires mu"):
            recovery.PLL(beta=1e-3)


class TestSignalInputPll:
    """Signal-awareness for recover_carrier_phase_pll."""

    def test_signal_input_uses_metadata(self, xp, xpt):
        """Signal input: modulation/order come from the signal's metadata."""
        sig = make_test_qam_signal(order=16, num_symbols=512, sps=1, xp=xp)

        phi_sig = recovery.estimate_carrier_phase(sig, recovery.PLL(mu=1e-2)).value
        phi_arr = recovery.estimate_carrier_phase(
            sig.samples, recovery.PLL(mu=1e-2), constellation=Constellation.qam(16)
        ).value

        assert not isinstance(phi_sig, Signal)  # phase estimate stays a raw array
        xpt.assert_allclose(phi_sig, phi_arr)
