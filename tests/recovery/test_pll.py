"""Decision-directed PLL carrier phase recovery."""

import numpy as np
import pytest

from commkit import generate_qam, recovery, spectral
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


class TestDDPLLEnhancements:
    """DD-PLL joint_channels and cycle_slip_correction parameters."""

    N = 4096
    PHASE = 0.4  # rad constant phase offset

    def _make_mimo(self, xp, order=16, phase=PHASE, snr_db=SNR_DB):
        sig_a = _qam_signal(xp, order, self.N, snr_db=snr_db, seed=10)
        sig_b = _qam_signal(xp, order, self.N, snr_db=snr_db, seed=11)
        mimo = xp.stack([sig_a.samples, sig_b.samples], axis=0)
        return mimo * xp.exp(1j * phase).astype(mimo.dtype)

    def test_joint_rows_identical(self, backend_device, xp):
        """PI loop + joint_channels=True: both phi_full rows are bitwise identical."""
        mimo = self._make_mimo(xp)
        phi = recovery.recover_carrier_phase_pll(
            mimo, "qam", 16, joint_channels=True, cycle_slip_correction=False
        )
        assert phi.shape == (2, self.N)
        phi_np = phi if xp is np else phi.get()
        np.testing.assert_array_equal(phi_np[0], phi_np[1])

    def test_siso_joint_noop(self, backend_device, xp):
        """joint_channels=True on SISO returns identical result to False."""
        sig = _qam_signal(xp, 16, self.N, seed=12)
        phi_a = recovery.recover_carrier_phase_pll(
            sig.samples, "qam", 16, joint_channels=False, cycle_slip_correction=False
        )
        phi_b = recovery.recover_carrier_phase_pll(
            sig.samples, "qam", 16, joint_channels=True, cycle_slip_correction=False
        )
        phi_a_np = phi_a if xp is np else phi_a.get()
        phi_b_np = phi_b if xp is np else phi_b.get()
        np.testing.assert_allclose(phi_a_np, phi_b_np, atol=1e-10)

    def test_cycle_slip_shape(self, backend_device, xp):
        """cycle_slip_correction=True (PI loop) returns correct shape."""
        sig = _qam_signal(xp, 16, self.N, snr_db=SNR_DB)
        phi = recovery.recover_carrier_phase_pll(
            sig.samples, "qam", 16, cycle_slip_correction=True
        )
        assert phi.shape == sig.samples.shape

    def test_joint_cycle_slip_mimo_rows_identical(self, backend_device, xp):
        """joint_channels=True + cycle_slip_correction=True: rows remain identical."""
        mimo = self._make_mimo(xp)
        phi = recovery.recover_carrier_phase_pll(
            mimo,
            "qam",
            16,
            joint_channels=True,
            cycle_slip_correction=True,
            cycle_slip_history=1000,
        )
        assert phi.shape == (2, self.N)
        phi_np = phi if xp is np else phi.get()
        np.testing.assert_array_equal(phi_np[0], phi_np[1])


class TestDDPLL:
    """Tests for recover_carrier_phase_pll."""

    def _qpsk_symbols(self, xp, N=512, seed=10):
        import numpy as np

        rng = np.random.default_rng(seed)
        const = gray_constellation("qpsk", 4)
        return xp.asarray(const[rng.integers(0, 4, N)].astype(np.complex64))

    def test_siso_output_shape(self, backend_device, xp):
        """SISO: output is (N,) float64."""
        syms = self._qpsk_symbols(xp)
        phi = recovery.recover_carrier_phase_pll(syms, "psk", 4)
        assert phi.shape == syms.shape
        assert phi.dtype == xp.float64

    def test_mimo_output_shape(self, backend_device, xp):
        """MIMO (C, N): output shape is (C, N)."""
        import numpy as np

        C, N = 2, 256
        rng = np.random.default_rng(11)

        const = gray_constellation("qpsk", 4)
        syms = xp.asarray(
            const[rng.integers(0, 4, C * N)].reshape(C, N).astype(np.complex64)
        )
        phi = recovery.recover_carrier_phase_pll(syms, "psk", 4)
        assert phi.shape == (C, N)

    def test_second_order_loop(self, backend_device, xp):
        """beta > 0 engages 2nd-order loop without raising."""
        syms = self._qpsk_symbols(xp, N=256)
        phi = recovery.recover_carrier_phase_pll(syms, "psk", 4, mu=0.02, beta=1e-4)
        assert phi.shape == syms.shape

    def test_phase_init_applied(self, backend_device, xp):
        """phase_init shifts the starting phase estimate."""
        syms = self._qpsk_symbols(xp, N=256)
        phi_init = 0.5
        phi = recovery.recover_carrier_phase_pll(syms, "psk", 4, phase_init=phi_init)
        # First estimate should be close to phase_init before any loop correction
        assert abs(float(phi[0]) - phi_init) < 0.5

    def test_bandwidth_shortcut_shape(self, backend_device, xp):
        """mu=None opts into the loop_bandwidth_normalized shortcut; shape/dtype hold."""
        syms = self._qpsk_symbols(xp, N=512)
        phi = recovery.recover_carrier_phase_pll(
            syms, "psk", 4, mu=None, loop_bandwidth_normalized=1e-3
        )
        assert phi.shape == syms.shape
        assert phi.dtype == xp.float64

    def test_bandwidth_shortcut_equals_raw_gains(self, backend_device, xp):
        """mu=None, bandwidth=B is identical to raw mu=4B, beta=4B² (the resolver mapping)."""
        import numpy as np

        N = 512
        const = gray_constellation("qpsk", 4)
        rng = np.random.default_rng(7)
        syms = xp.asarray(const[rng.integers(0, 4, N)].astype(np.complex64))
        syms = syms * np.exp(1j * 0.2)

        B = 1e-3
        phi_bw = recovery.recover_carrier_phase_pll(
            syms, "psk", 4, mu=None, loop_bandwidth_normalized=B
        )
        phi_raw = recovery.recover_carrier_phase_pll(
            syms, "psk", 4, mu=4.0 * B, beta=4.0 * B**2
        )
        phi_bw_np = phi_bw if xp is np else phi_bw.get()
        phi_raw_np = phi_raw if xp is np else phi_raw.get()
        np.testing.assert_allclose(phi_bw_np, phi_raw_np, rtol=1e-6, atol=1e-9)

    def test_first_vs_second_order_under_frequency_offset(self, backend_device, xp):
        """Under a frequency offset, a 1st-order loop (beta=0) settles to a constant
        phase lag; a 2nd-order loop (beta>0) nulls it to ~zero steady-state error."""
        import numpy as np

        N = 4000
        const = gray_constellation("qpsk", 4)
        rng = np.random.default_rng(5)
        clean = const[rng.integers(0, 4, N)].astype(np.complex64)
        # Small constant frequency offset (rad/sample) -> phase ramp the loop tracks.
        df = 1e-3
        ramp = np.exp(1j * df * np.arange(N)).astype(np.complex64)
        syms = xp.asarray(clean * ramp)

        true_phase = df * np.arange(N)
        tail = slice(N // 2, N)

        phi1 = recovery.recover_carrier_phase_pll(syms, "psk", 4, mu=0.02, beta=0.0)
        phi2 = recovery.recover_carrier_phase_pll(syms, "psk", 4, mu=0.02, beta=2e-4)
        phi1_np = phi1 if xp is np else phi1.get()
        phi2_np = phi2 if xp is np else phi2.get()

        # The steady-state error is a constant lag (its std ≈ 0), so measure the mean.
        lag1 = abs(float(np.mean(np.unwrap(phi1_np[tail]) - true_phase[tail])))
        lag2 = abs(float(np.mean(np.unwrap(phi2_np[tail]) - true_phase[tail])))
        assert lag1 > 1e-3, (
            "1st-order loop should exhibit a finite lag under freq offset"
        )
        assert lag2 < lag1 / 100.0, (
            "2nd-order loop should null the frequency-induced lag"
        )

    def test_bandwidth_shortcut_invalid_bandwidth_raises(self, backend_device, xp):
        """On the bandwidth path (mu=None), bandwidth outside (0, 0.5) raises ValueError."""
        syms = self._qpsk_symbols(xp, N=64)
        with pytest.raises(ValueError, match="loop_bandwidth_normalized"):
            recovery.recover_carrier_phase_pll(
                syms, "psk", 4, mu=None, loop_bandwidth_normalized=0.6
            )

    def test_beta_without_mu_raises(self, backend_device, xp):
        """Passing beta with mu=None is ambiguous and must raise ValueError."""
        syms = self._qpsk_symbols(xp, N=64)
        with pytest.raises(ValueError, match="beta requires mu"):
            recovery.recover_carrier_phase_pll(syms, "psk", 4, mu=None, beta=1e-3)
