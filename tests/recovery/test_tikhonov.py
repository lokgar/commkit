"""Tikhonov / Wiener Kalman phase smoothers (RTS, SSKF)."""

import numpy as np
import pytest

from commkit import recovery
from commkit.core import Signal
from commkit.mapping import Constellation
from tests.common.metrics import calc_rms_phase_error
from tests.common.signals import (
    make_test_mimo_samples,
    make_test_psk_signal,
    make_test_qam_signal,
)

FS = 1e6  # 1 MHz sampling rate, common to all tests
SNR_DB = 30  # generous SNR so numerical algorithms converge reliably


class TestWienerPhaseSmoother:
    """Zero-phase Wiener smoother for a random-walk carrier phase."""

    def _random_walk(self, xp, N=20000, q=4e-4, seed=3):
        rng = xp.random.RandomState(seed)
        return xp.cumsum(rng.normal(0.0, float(np.sqrt(q)), N)), q

    def test_reduces_random_walk_noise(self, xp):
        """Smoothing a noisy random-walk phase lowers the RMS error vs truth."""
        truth, q = self._random_walk(xp)
        rng = xp.random.RandomState(11)
        r = 0.05
        noisy = truth + rng.normal(0.0, float(np.sqrt(r)), truth.shape[-1])
        smoothed = recovery.smooth_phase_wiener(
            noisy, process_variance=q, measurement_variance=r
        )
        g = slice(200, -200)  # trim FFT edge transients
        rms_noisy = calc_rms_phase_error(noisy[g], truth[g], xp=xp)
        rms_smooth = calc_rms_phase_error(smoothed[g], truth[g], xp=xp)
        assert rms_smooth < rms_noisy

    def test_preserves_linear_trend(self, xp):
        """A pure FOE ramp (+ tiny noise) survives the detrend/add-back path."""
        N = 8000
        n = xp.arange(N, dtype=xp.float64)
        slope = 1e-3
        ramp = slope * n
        rng = xp.random.RandomState(5)
        noisy = ramp + rng.normal(0.0, 0.01, N)
        smoothed = recovery.smooth_phase_wiener(
            noisy, process_variance=1e-6, measurement_variance=1e-2
        )
        g = slice(200, -200)
        # The recovered slope (via endpoints) matches the true ramp slope.
        est_slope = float((smoothed[g][-1] - smoothed[g][0]) / (n[g][-1] - n[g][0]))
        assert abs(est_slope - slope) < 0.05 * slope + 1e-5

    def test_shape_siso_and_mimo(self, xp):
        truth, q = self._random_walk(xp, N=4000)
        phi1d = recovery.smooth_phase_wiener(
            truth, process_variance=q, measurement_variance=0.05
        )
        assert phi1d.shape == truth.shape
        phi2d_in = xp.stack([truth, truth + 0.3])
        phi2d = recovery.smooth_phase_wiener(
            phi2d_in, process_variance=q, measurement_variance=0.05
        )
        assert phi2d.shape == phi2d_in.shape

    def test_derive_variances_from_physical_params(self, xp):
        """linewidth + sampling_rate derive q internally; r is given directly."""
        truth, _ = self._random_walk(xp, N=4000)
        phi = recovery.smooth_phase_wiener(
            truth, linewidth=100e3, sampling_rate=10e9, measurement_variance=0.05
        )
        assert phi.shape == truth.shape

    def test_invalid_params_raise(self, xp):
        truth, _ = self._random_walk(xp, N=100)
        with pytest.raises(ValueError, match="process_variance"):
            recovery.smooth_phase_wiener(truth, measurement_variance=0.05)


class TestCprTikhonov:
    @pytest.mark.parametrize(
        "order,modulation,block_size",
        [
            (4, "psk", 16),
            (4, "psk", 32),
            (4, "psk", 64),
            (16, "qam", 16),
            (16, "qam", 32),
            (64, "qam", 32),
        ],
    )
    def test_phase_residual(self, xp, order, modulation, block_size):
        """Tikhonov CPR: mean estimate within 0.1 rad of true carrier phase (mod M-fold)."""
        if modulation == "qam":
            sig = make_test_qam_signal(
                order=order, num_symbols=2048, sps=1, symbol_rate=FS, xp=xp
            )
        else:
            sig = make_test_psk_signal(
                order=order, num_symbols=2048, sps=1, symbol_rate=FS, xp=xp
            )
        phi_true = 0.3
        sig = sig.replace(samples=sig.samples * xp.exp(1j * phi_true))

        phase_est = recovery.estimate_carrier_phase(
            sig.samples,
            recovery.Tikhonov(
                linewidth_symbol_periods=1e-4, snr_db=SNR_DB, block_size=block_size
            ),
            constellation=getattr(Constellation, modulation)(order),
        ).value

        M = 4 if modulation == "qam" else order
        step = 2 * np.pi / M
        err = float(xp.mean(phase_est)) - phi_true
        err = err - step * round(err / step)
        assert abs(err) < 0.1

    def test_output_shape_siso(self, xp):
        """Tikhonov CPR: 1D input -> 1D output of same length."""
        sig = make_test_qam_signal(
            order=16, num_symbols=512, sps=1, symbol_rate=FS, xp=xp
        )
        phase = recovery.estimate_carrier_phase(
            sig.samples,
            recovery.Tikhonov(linewidth_symbol_periods=1e-4, snr_db=SNR_DB),
            constellation=Constellation.qam(16),
        ).value
        assert phase.shape == sig.samples.shape

    def test_output_shape_mimo(self, xp):
        """Tikhonov CPR: 2D input (C, N) -> 2D output (C, N)."""
        mimo, _ = make_test_mimo_samples(
            num_channels=2, order=16, num_symbols=512, sps=1, xp=xp
        )
        phase = recovery.estimate_carrier_phase(
            mimo,
            recovery.Tikhonov(linewidth_symbol_periods=1e-4, snr_db=SNR_DB),
            constellation=Constellation.qam(16),
        ).value
        assert phase.shape == mimo.shape

    def test_too_short_raises(self, xp):
        """Tikhonov CPR: signal shorter than block_size raises ValueError."""
        sig = make_test_qam_signal(
            order=4, num_symbols=20, sps=1, symbol_rate=FS, xp=xp
        )
        with pytest.raises(ValueError, match="shorter than block_size"):
            recovery.estimate_carrier_phase(
                sig.samples[:10],
                recovery.Tikhonov(
                    linewidth_symbol_periods=1e-4, snr_db=20, block_size=32
                ),
                constellation=Constellation.qam(4),
            )

    def test_invalid_smoother_raises(self):
        """Tikhonov CPR: an unknown smoother raises on construction."""
        with pytest.raises(ValueError, match="smoother"):
            recovery.Tikhonov(linewidth_symbol_periods=1e-4, snr_db=20, smoother="bad")

    @pytest.mark.parametrize("order,modulation", [(4, "psk"), (16, "qam")])
    def test_sskf_phase_residual(self, xp, order, modulation):
        """Tikhonov SSKF: mean estimate within 0.1 rad of true offset (mod M-fold)."""
        if modulation == "qam":
            sig = make_test_qam_signal(
                order=order, num_symbols=2048, sps=1, symbol_rate=FS, xp=xp
            )
        else:
            sig = make_test_psk_signal(
                order=order, num_symbols=2048, sps=1, symbol_rate=FS, xp=xp
            )
        phi_true = 0.3
        sig = sig.replace(samples=sig.samples * xp.exp(1j * phi_true))

        phase_est = recovery.estimate_carrier_phase(
            sig.samples,
            recovery.Tikhonov(
                linewidth_symbol_periods=1e-4, snr_db=SNR_DB, smoother="steady_state"
            ),
            constellation=getattr(Constellation, modulation)(order),
        ).value

        M = 4 if modulation == "qam" else order
        step = 2 * np.pi / M
        err = float(xp.mean(phase_est)) - phi_true
        err = err - step * round(err / step)
        assert abs(err) < 0.1

    def test_sskf_exact_close(self, xp):
        """SSKF and exact RTS produce similar phase estimates (within 0.05 rad RMS)."""
        sig = make_test_qam_signal(
            order=16, num_symbols=2048, sps=1, symbol_rate=FS, xp=xp
        )
        sig = sig.replace(samples=sig.samples * xp.exp(1j * 0.2))

        phi_exact = recovery.estimate_carrier_phase(
            sig.samples,
            recovery.Tikhonov(
                linewidth_symbol_periods=1e-4, snr_db=SNR_DB, smoother="rts"
            ),
            constellation=Constellation.qam(16),
        ).value
        phi_sskf = recovery.estimate_carrier_phase(
            sig.samples,
            recovery.Tikhonov(
                linewidth_symbol_periods=1e-4, snr_db=SNR_DB, smoother="steady_state"
            ),
            constellation=Constellation.qam(16),
        ).value
        rms_diff = float(xp.sqrt(xp.mean((phi_exact - phi_sskf) ** 2)))
        assert rms_diff < 0.05

    def test_smoother_reduces_noise_vs_vv(self, xp):
        """Tikhonov produces smoother phase trajectory than VV when σ_p² < σ_v²."""
        linewidth_symbol_periods = 1e-7
        snr_test = 15
        sig = make_test_psk_signal(
            order=4,
            num_symbols=2048,
            sps=1,
            symbol_rate=FS,
            snr_db=snr_test,
            seed=123,
            xp=xp,
        )
        sig = sig.replace(samples=sig.samples * xp.exp(1j * 0.3))

        phi_vv = recovery.estimate_carrier_phase(
            sig.samples,
            recovery.ViterbiViterbi(block_size=32),
            constellation=Constellation.psk(4),
        ).value
        phi_tik = recovery.estimate_carrier_phase(
            sig.samples,
            recovery.Tikhonov(
                linewidth_symbol_periods=linewidth_symbol_periods,
                snr_db=snr_test,
                block_size=32,
            ),
            constellation=Constellation.psk(4),
        ).value

        assert float(xp.std(phi_tik)) < float(xp.std(phi_vv))


class TestSignalInputTikhonov:
    """Signal-awareness for recover_carrier_phase_tikhonov."""

    def test_signal_input_uses_metadata(self, xp, xpt):
        """Signal input: modulation/order come from the signal's metadata."""
        sig = make_test_qam_signal(
            order=16, num_symbols=512, sps=1, symbol_rate=FS, xp=xp
        )

        phi_sig = recovery.estimate_carrier_phase(
            sig, recovery.Tikhonov(linewidth_symbol_periods=1e-5, snr_db=20)
        ).value
        phi_arr = recovery.estimate_carrier_phase(
            sig.samples,
            recovery.Tikhonov(linewidth_symbol_periods=1e-5, snr_db=20),
            constellation=Constellation.qam(16),
        ).value

        assert not isinstance(phi_sig, Signal)  # phase estimate stays a raw array
        xpt.assert_allclose(phi_sig, phi_arr)
