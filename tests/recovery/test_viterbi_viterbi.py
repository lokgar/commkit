"""Viterbi-Viterbi carrier phase recovery."""

import numpy as np
import pytest

from commkit import recovery
from commkit.core import Signal
from commkit.mapping import Constellation
from tests.common import scoring
from tests.common.signals import (
    make_test_psk_signal,
    make_test_qam_signal,
    make_test_symbols,
)

FS = 1e6  # 1 MHz sampling rate, common to all tests
SNR_DB = 30  # generous SNR so numerical algorithms converge reliably


class TestCprViterbiViterbi:
    @pytest.mark.parametrize("block_size", [16, 32, 64])
    def test_phase_residual(self, xp, block_size):
        """QPSK, static 0.3 rad at Es/N0 30 dB: per-symbol RMS error against
        the truth (one 2π/4 quantum removed) within 2x the M-th power
        estimator's high-SNR standard deviation sqrt(1/(2 N SNR))."""
        sig = make_test_psk_signal(
            order=4, num_symbols=2048, sps=1, symbol_rate=FS, xp=xp
        )
        x = sig.samples * xp.exp(1j * 0.3).astype(sig.samples.dtype)
        est = recovery.estimate_carrier_phase(
            x,
            recovery.ViterbiViterbi(block_size=block_size),
            constellation=Constellation.psk(4),
        ).value
        bound = np.sqrt(1.0 / (2 * block_size * 10 ** (SNR_DB / 10)))
        assert scoring.phase_rms(est, np.full(2048, 0.3), symmetry=4) < 2 * bound

    @pytest.mark.parametrize("order", [16, 64])
    def test_qam_self_noise_averages_with_the_block(self, xp, order):
        """On QAM the normalized 4th power has self-noise (the points are not
        on one circle), which dominates at 30 dB and averages as 1/sqrt(N):
        the error at 64-symbol blocks is 1.6-3x smaller than at 16, and
        stays under 0.1 rad."""
        sig = make_test_qam_signal(
            order=order, num_symbols=2048, sps=1, symbol_rate=FS, xp=xp
        )
        x = sig.samples * xp.exp(1j * 0.3).astype(sig.samples.dtype)
        c = Constellation.qam(order)
        rms = {
            n: scoring.phase_rms(
                recovery.estimate_carrier_phase(
                    x, recovery.ViterbiViterbi(block_size=n), constellation=c
                ).value,
                np.full(2048, 0.3),
                symmetry=4,
            )
            for n in (16, 64)
        }
        assert 1.6 < rms[16] / rms[64] < 3.0, rms
        assert rms[64] < 0.1, rms

    def test_too_short_raises(self, xp):
        """VV CPR: signal shorter than block_size raises ValueError."""
        sig = make_test_qam_signal(
            order=4, num_symbols=20, sps=1, symbol_rate=FS, xp=xp
        )
        with pytest.raises(ValueError, match="shorter than block_size"):
            recovery.estimate_carrier_phase(
                sig.samples[:10],
                recovery.ViterbiViterbi(block_size=32),
                constellation=Constellation.qam(4),
            )


class TestViterbiViterbi:
    """Tests for recover_carrier_phase_viterbi_viterbi."""

    def _qpsk_symbols(self, xp, N=512, seed=0):
        return make_test_symbols(scheme="psk", order=4, num_symbols=N, seed=seed, xp=xp)

    def test_siso_qpsk_output_shape(self, xp):
        """SISO QPSK: output is (N,) float64."""
        syms = self._qpsk_symbols(xp)
        phi_est = recovery.estimate_carrier_phase(
            syms,
            recovery.ViterbiViterbi(block_size=32),
            constellation=Constellation.psk(4),
        ).value
        assert phi_est.shape == syms.shape
        assert phi_est.dtype == xp.float64

    def test_siso_qpsk_recovers_static_phase(self, xp):
        """VV tracks applied phase: difference between rotated and unrotated estimate equals phi_true mod π/2."""
        import numpy as np

        phi_true = 0.3  # radians
        syms = self._qpsk_symbols(xp, N=512)
        # Baseline (no rotation)
        phi_base = float(
            xp.mean(
                recovery.estimate_carrier_phase(
                    syms,
                    recovery.ViterbiViterbi(block_size=32),
                    constellation=Constellation.psk(4),
                ).value
            )
        )
        # Rotated by phi_true
        rotated = syms * xp.asarray(np.complex64(np.exp(1j * phi_true)))
        phi_rot = float(
            xp.mean(
                recovery.estimate_carrier_phase(
                    rotated,
                    recovery.ViterbiViterbi(block_size=32),
                    constellation=Constellation.psk(4),
                ).value
            )
        )
        # The shift should equal phi_true modulo π/2
        delta = phi_rot - phi_base
        residual = (delta - phi_true + np.pi / 4) % (np.pi / 2) - np.pi / 4
        assert abs(residual) < 0.15, (
            f"Phase tracking error too large: {residual:.3f} rad"
        )

    def test_block_size_too_large_raises(self, xp):
        """block_size > N should raise ValueError."""
        syms = self._qpsk_symbols(xp, N=16)
        with pytest.raises(ValueError, match="block_size"):
            recovery.estimate_carrier_phase(
                syms,
                recovery.ViterbiViterbi(block_size=64),
                constellation=Constellation.psk(4),
            )


class TestJointChannelWeighting:
    """Joint channels weigh equally, whatever their power."""

    @pytest.mark.parametrize("method", ["vv", "tikhonov"])
    def test_joint_estimate_ignores_channel_gain(self, xp, xpt, method):
        """Scaling one channel leaves the joint trajectory unchanged.

        With amplitude^M weighting a 10x stronger channel would carry 10^4
        times the weight of the other for QPSK, and the joint estimate would
        follow it alone.
        """
        rng = np.random.default_rng(5)
        n = 4096
        pts = Constellation.psk(4).points
        walk = np.cumsum(rng.normal(0.0, 0.01, n)) + 0.2
        sym = pts[rng.integers(0, 4, (2, n))] * np.exp(1j * walk)
        noise = 0.2 * (rng.standard_normal((2, n)) + 1j * rng.standard_normal((2, n)))
        x = xp.asarray((sym + noise).astype(np.complex64))
        scaled = x * xp.asarray(np.array([[1.0], [10.0]], dtype=np.float32))
        if method == "vv":
            m = recovery.ViterbiViterbi(block_size=32, joint_channels=True)
        else:
            m = recovery.Tikhonov(1e-4, 10, block_size=32, joint_channels=True)
        c = Constellation.psk(4)
        a = recovery.estimate_carrier_phase(
            x, m, constellation=c, sampling_rate=1.0
        ).value
        b = recovery.estimate_carrier_phase(
            scaled, m, constellation=c, sampling_rate=1.0
        ).value
        xpt.assert_allclose(a, b, atol=1e-6)  # float32 rounding of the gain


class TestSignalInputViterbiViterbi:
    """Signal-awareness for recover_carrier_phase_viterbi_viterbi."""

    def test_signal_input_uses_metadata(self, xp, xpt):
        """Signal input: modulation/order come from the signal's metadata."""
        sig = make_test_qam_signal(
            order=16, num_symbols=512, sps=1, symbol_rate=FS, xp=xp
        )

        phi_sig = recovery.estimate_carrier_phase(sig, recovery.ViterbiViterbi()).value
        phi_arr = recovery.estimate_carrier_phase(
            sig.samples, recovery.ViterbiViterbi(), constellation=Constellation.qam(16)
        ).value

        assert not isinstance(phi_sig, Signal)  # phase estimate stays a raw array
        xpt.assert_allclose(phi_sig, phi_arr)
