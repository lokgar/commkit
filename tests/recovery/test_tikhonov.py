"""Tikhonov / Wiener Kalman phase smoothers (RTS, SSKF)."""

import numpy as np
import pytest

from commkit import generate_psk, generate_qam, recovery, spectral
from commkit.core import Signal
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


def _psk_signal(xp, order, n_symbols, fo_hz=0.0, snr_db=SNR_DB, fs=FS, seed=42):
    """Generate a 1-SPS PSK signal with optional frequency offset and AWGN."""
    sig = generate_psk(
        order=order, num_symbols=n_symbols, sps=1, symbol_rate=fs, seed=seed
    )
    sig.samples = apply_awgn(sig.samples, esn0_db=snr_db, sps=1, seed=seed)
    if fo_hz != 0.0:
        sig.samples, _ = spectral.shift_frequency(sig.samples, fo_hz, fs)
    return sig


def _rms_phase_error(xp, phase_est, phase_true):
    """RMS of the phase error, after removing any constant offset (M-fold ambiguity)."""
    err = phase_est - phase_true
    # Remove the mean bias (accounts for the irreducible global ambiguity)
    err = err - float(xp.mean(err))
    return float(xp.sqrt(xp.mean(err**2)))


class TestWienerPhaseSmoother:
    """Zero-phase Wiener smoother for a random-walk carrier phase."""

    def _random_walk(self, xp, N=20000, q=4e-4, seed=3):
        rng = xp.random.RandomState(seed)
        return xp.cumsum(rng.normal(0.0, float(np.sqrt(q)), N)), q

    def test_reduces_random_walk_noise(self, backend_device, xp):
        """Smoothing a noisy random-walk phase lowers the RMS error vs truth."""
        truth, q = self._random_walk(xp)
        rng = xp.random.RandomState(11)
        r = 0.05
        noisy = truth + rng.normal(0.0, float(np.sqrt(r)), truth.shape[-1])
        smoothed = recovery.smooth_phase_wiener(
            noisy, process_variance=q, measurement_variance=r
        )
        g = slice(200, -200)  # trim FFT edge transients
        rms_noisy = _rms_phase_error(xp, noisy[g], truth[g])
        rms_smooth = _rms_phase_error(xp, smoothed[g], truth[g])
        assert rms_smooth < rms_noisy

    def test_preserves_linear_trend(self, backend_device, xp):
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

    def test_shape_siso_and_mimo(self, backend_device, xp):
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

    def test_derive_variances_from_physical_params(self, backend_device, xp):
        """linewidth + sampling_rate derive q internally; r is given directly."""
        truth, _ = self._random_walk(xp, N=4000)
        smoothed = recovery.smooth_phase_wiener(
            truth, linewidth=1e5, sampling_rate=4e9, measurement_variance=5e-3
        )
        assert smoothed.shape == truth.shape
        assert bool(xp.all(xp.isfinite(smoothed)))

    def test_missing_process_params_raise(self, backend_device, xp):
        truth, _ = self._random_walk(xp, N=512)
        with pytest.raises(ValueError, match="process_variance"):
            recovery.smooth_phase_wiener(truth, measurement_variance=0.05)

    def test_missing_measurement_params_raise(self, backend_device, xp):
        truth, q = self._random_walk(xp, N=512)
        with pytest.raises(ValueError, match="measurement_variance"):
            recovery.smooth_phase_wiener(truth, process_variance=q)

    def test_invalid_variance_raises(self, backend_device, xp):
        truth, _ = self._random_walk(xp, N=512)
        with pytest.raises(ValueError, match="must be > 0"):
            recovery.smooth_phase_wiener(
                truth, process_variance=0.0, measurement_variance=0.05
            )


class TestCprTikhonov:
    @pytest.mark.parametrize(
        "order,modulation,block_size",
        [
            (4, "psk", 16),
            (4, "psk", 32),
            (16, "qam", 16),
            (16, "qam", 32),
            (64, "qam", 32),
        ],
    )
    def test_phase_residual(self, backend_device, xp, order, modulation, block_size):
        """Tikhonov CPR: mean estimate within 0.1 rad of true carrier phase (mod M-fold)."""
        sig = (
            _qam_signal(xp, order, 2048)
            if modulation == "qam"
            else _psk_signal(xp, order, 2048)
        )
        phi_true = 0.3
        sig.samples = sig.samples * xp.exp(1j * phi_true)

        phase_est = recovery.recover_carrier_phase_tikhonov(
            sig.samples,
            modulation=modulation,
            order=order,
            linewidth_symbol_periods=1e-4,
            block_size=block_size,
            snr_db=SNR_DB,
        )

        M = 4 if modulation == "qam" else order
        step = 2 * np.pi / M
        err = float(xp.mean(phase_est)) - phi_true
        err = err - step * round(err / step)
        assert abs(err) < 0.1

    def test_output_shape_siso(self, backend_device, xp):
        """Tikhonov CPR: 1D input -> 1D output of same length."""
        sig = _qam_signal(xp, 16, 512)
        phase = recovery.recover_carrier_phase_tikhonov(
            sig.samples,
            modulation="qam",
            order=16,
            linewidth_symbol_periods=1e-4,
            snr_db=SNR_DB,
        )
        assert phase.shape == sig.samples.shape

    def test_output_shape_mimo(self, backend_device, xp):
        """Tikhonov CPR: 2D input (C, N) -> 2D output (C, N)."""
        sig_a = _qam_signal(xp, 16, 512)
        sig_b = _qam_signal(xp, 16, 512)
        mimo = xp.stack([sig_a.samples, sig_b.samples])
        phase = recovery.recover_carrier_phase_tikhonov(
            mimo,
            modulation="qam",
            order=16,
            linewidth_symbol_periods=1e-4,
            snr_db=SNR_DB,
        )
        assert phase.shape == mimo.shape

    def test_too_short_raises(self, backend_device, xp):
        """Tikhonov CPR: signal shorter than block_size raises ValueError."""
        sig = _qam_signal(xp, 4, 20)
        with pytest.raises(ValueError, match="shorter than block_size"):
            recovery.recover_carrier_phase_tikhonov(
                sig.samples[:10],
                modulation="qam",
                order=4,
                linewidth_symbol_periods=1e-4,
                block_size=32,
            )

    def test_invalid_method_raises(self, backend_device, xp):
        """Tikhonov CPR: unknown method raises ValueError."""
        sig = _qam_signal(xp, 16, 512)
        with pytest.raises(ValueError, match="Unknown method"):
            recovery.recover_carrier_phase_tikhonov(
                sig.samples,
                modulation="qam",
                order=16,
                linewidth_symbol_periods=1e-4,
                method="bad",
            )

    @pytest.mark.parametrize("order,modulation", [(4, "psk"), (16, "qam")])
    def test_sskf_phase_residual(self, backend_device, xp, order, modulation):
        """Tikhonov SSKF: mean estimate within 0.1 rad of true offset (mod M-fold)."""
        sig = (
            _qam_signal(xp, order, 2048)
            if modulation == "qam"
            else _psk_signal(xp, order, 2048)
        )
        phi_true = 0.3
        sig.samples = sig.samples * xp.exp(1j * phi_true)

        phase_est = recovery.recover_carrier_phase_tikhonov(
            sig.samples,
            modulation=modulation,
            order=order,
            linewidth_symbol_periods=1e-4,
            snr_db=SNR_DB,
            method="sskf",
        )

        M = 4 if modulation == "qam" else order
        step = 2 * np.pi / M
        err = float(xp.mean(phase_est)) - phi_true
        err = err - step * round(err / step)
        assert abs(err) < 0.1

    def test_sskf_exact_close(self, backend_device, xp):
        """SSKF and exact RTS produce similar phase estimates (within 0.05 rad RMS)."""
        sig = _qam_signal(xp, 16, 2048)
        sig.samples = sig.samples * xp.exp(1j * 0.2)

        phi_exact = recovery.recover_carrier_phase_tikhonov(
            sig.samples,
            modulation="qam",
            order=16,
            linewidth_symbol_periods=1e-4,
            snr_db=SNR_DB,
            method="exact",
        )
        phi_sskf = recovery.recover_carrier_phase_tikhonov(
            sig.samples,
            modulation="qam",
            order=16,
            linewidth_symbol_periods=1e-4,
            snr_db=SNR_DB,
            method="sskf",
        )
        rms_diff = float(xp.sqrt(xp.mean((phi_exact - phi_sskf) ** 2)))
        assert rms_diff < 0.05

    def test_smoother_reduces_noise_vs_vv(self, backend_device, xp):
        """Tikhonov produces smoother phase trajectory than VV when σ_p² < σ_v²."""
        linewidth_symbol_periods = 1e-7
        snr_test = 15
        sig = _psk_signal(xp, 4, 2048, snr_db=snr_test, seed=123)
        sig.samples = sig.samples * xp.exp(1j * 0.3)

        phi_vv = recovery.recover_carrier_phase_viterbi_viterbi(
            sig.samples, modulation="psk", order=4, block_size=32
        )
        phi_tik = recovery.recover_carrier_phase_tikhonov(
            sig.samples,
            modulation="psk",
            order=4,
            linewidth_symbol_periods=linewidth_symbol_periods,
            block_size=32,
            snr_db=snr_test,
        )

        assert float(xp.std(phi_tik)) < float(xp.std(phi_vv))


class TestSignalInputTikhonov:
    """Signal-awareness for recover_carrier_phase_tikhonov."""

    def test_signal_input_uses_metadata(self, backend_device, xp, xpt):
        """Signal input: modulation/order come from the signal's metadata."""
        sig = _qam_signal(xp, 16, 512)

        phi_sig = recovery.recover_carrier_phase_tikhonov(
            sig, linewidth_symbol_periods=1e-5
        )
        phi_arr = recovery.recover_carrier_phase_tikhonov(
            sig.samples, modulation="qam", order=16, linewidth_symbol_periods=1e-5
        )

        assert not isinstance(phi_sig, Signal)  # phase estimate stays a raw array
        xpt.assert_allclose(phi_sig, phi_arr)
