"""Tikhonov / Wiener Kalman phase smoothers (RTS, SSKF)."""

import numpy as np
import pytest

from commkit import recovery
from commkit.core import Signal
from commkit.mapping import Constellation
from tests.common import scoring, theory
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
        """A random walk (q = 4e-4) in white noise (r = 0.05) is smoothed to
        the two-sided Wiener error sqrt(q r / sqrt(q² + 4 q r)) = 0.047 rad
        within 10 % (the unsmoothed error is sqrt(r) = 0.22 rad)."""
        truth, q = self._random_walk(xp)
        rng = xp.random.RandomState(11)
        r = 0.05
        noisy = truth + rng.normal(0.0, float(np.sqrt(r)), truth.shape[-1])
        smoothed = recovery.smooth_phase_wiener(
            noisy, process_variance=q, measurement_variance=r
        )
        g = slice(200, -200)  # trim FFT edge transients
        rms = scoring.phase_rms(smoothed[g], truth[g])
        assert rms == pytest.approx(np.sqrt(theory.wiener_smoother_var(q, r)), rel=0.1)

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
    @pytest.mark.parametrize("smoother", ["rts", "steady_state"])
    @pytest.mark.parametrize("block_size", [16, 32, 64])
    def test_phase_residual(self, xp, block_size, smoother):
        """QPSK, static 0.3 rad at 30 dB: per-symbol RMS against the truth
        within 2x the M-th power block bound sqrt(1/(2 N SNR)) (smoothing
        across blocks can only lower it)."""
        sig = make_test_psk_signal(
            order=4, num_symbols=2048, sps=1, symbol_rate=FS, xp=xp
        )
        x = sig.samples * xp.exp(1j * 0.3).astype(sig.samples.dtype)
        est = recovery.estimate_carrier_phase(
            x,
            recovery.Tikhonov(
                linewidth=1e-4 * FS,
                snr_db=SNR_DB,
                block_size=block_size,
                smoother=smoother,
            ),
            constellation=Constellation.psk(4),
            sampling_rate=FS,
        ).value
        bound = np.sqrt(1.0 / (2 * block_size * 10 ** (SNR_DB / 10)))
        assert scoring.phase_rms(est, np.full(2048, 0.3), symmetry=4) < 2 * bound

    @pytest.mark.xfail(
        strict=True,
        reason="Tikhonov's observation variance 1/(M² SNR N) (tikhonov.py:243) "
        "omits the 4th-power self-noise of QAM: 16-QAM at 30 dB, N = 16, model "
        "4e-6 rad², actual Viterbi-Viterbi block error ~1e-2 rad².  The "
        "smoother trusts every block: per-symbol RMS 0.1012 rad vs 0.1013 for "
        "Viterbi-Viterbi; the Wiener smoother with the actual variance gives "
        "~0.067.",
    )
    def test_smooths_qam_self_noise(self, xp):
        """16-QAM, static phase, 16-symbol blocks: smoothing across blocks
        lowers the Viterbi-Viterbi error by at least 20 %."""
        sig = make_test_qam_signal(
            order=16, num_symbols=2048, sps=1, symbol_rate=FS, xp=xp
        )
        x = sig.samples * xp.exp(1j * 0.3).astype(sig.samples.dtype)
        c = Constellation.qam(16)
        truth = np.full(2048, 0.3)
        vv = recovery.estimate_carrier_phase(
            x, recovery.ViterbiViterbi(block_size=16), constellation=c
        ).value
        tik = recovery.estimate_carrier_phase(
            x,
            recovery.Tikhonov(linewidth=1e-4 * FS, snr_db=SNR_DB, block_size=16),
            constellation=c,
            sampling_rate=FS,
        ).value
        assert scoring.phase_rms(tik, truth, symmetry=4) < 0.8 * scoring.phase_rms(
            vv, truth, symmetry=4
        )

    def test_output_shape_siso(self, xp):
        """Tikhonov CPR: 1D input -> 1D output of same length."""
        sig = make_test_qam_signal(
            order=16, num_symbols=512, sps=1, symbol_rate=FS, xp=xp
        )
        phase = recovery.estimate_carrier_phase(
            sig.samples,
            recovery.Tikhonov(linewidth=1e-4 * FS, snr_db=SNR_DB),
            constellation=Constellation.qam(16),
            sampling_rate=FS,
        ).value
        assert phase.shape == sig.samples.shape

    def test_output_shape_mimo(self, xp):
        """Tikhonov CPR: 2D input (C, N) -> 2D output (C, N)."""
        mimo, _ = make_test_mimo_samples(
            num_channels=2, order=16, num_symbols=512, sps=1, xp=xp
        )
        phase = recovery.estimate_carrier_phase(
            mimo,
            recovery.Tikhonov(linewidth=1e-4 * FS, snr_db=SNR_DB),
            constellation=Constellation.qam(16),
            sampling_rate=FS,
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
                recovery.Tikhonov(linewidth=1e-4 * FS, snr_db=20, block_size=32),
                constellation=Constellation.qam(4),
                sampling_rate=FS,
            )

    def test_invalid_smoother_raises(self):
        """Tikhonov CPR: an unknown smoother raises on construction."""
        with pytest.raises(ValueError, match="smoother"):
            recovery.Tikhonov(linewidth=1e-4 * FS, snr_db=20, smoother="bad")

    def test_sskf_exact_close(self, xp):
        """SSKF and exact RTS produce similar phase estimates (within 0.05 rad RMS)."""
        sig = make_test_qam_signal(
            order=16, num_symbols=2048, sps=1, symbol_rate=FS, xp=xp
        )
        sig = sig.replace(samples=sig.samples * xp.exp(1j * 0.2))

        phi_exact = recovery.estimate_carrier_phase(
            sig.samples,
            recovery.Tikhonov(linewidth=1e-4 * FS, snr_db=SNR_DB, smoother="rts"),
            constellation=Constellation.qam(16),
            sampling_rate=FS,
        ).value
        phi_sskf = recovery.estimate_carrier_phase(
            sig.samples,
            recovery.Tikhonov(
                linewidth=1e-4 * FS, snr_db=SNR_DB, smoother="steady_state"
            ),
            constellation=Constellation.qam(16),
            sampling_rate=FS,
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
                linewidth=linewidth_symbol_periods * FS,
                snr_db=snr_test,
                block_size=32,
            ),
            constellation=Constellation.psk(4),
            sampling_rate=FS,
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
            sig, recovery.Tikhonov(linewidth=1e-5 * FS, snr_db=20)
        ).value
        phi_arr = recovery.estimate_carrier_phase(
            sig.samples,
            recovery.Tikhonov(linewidth=1e-5 * FS, snr_db=20),
            constellation=Constellation.qam(16),
            sampling_rate=FS,
        ).value

        assert not isinstance(phi_sig, Signal)  # phase estimate stays a raw array
        xpt.assert_allclose(phi_sig, phi_arr)


def test_linewidth_is_in_hz(xp, xpt):
    """The same Δν·T at two symbol rates gives the same estimate; the symbol
    period comes from the Signal, and array input needs sampling_rate."""
    sig = make_test_qam_signal(order=16, num_symbols=512, sps=1, symbol_rate=FS, xp=xp)
    fast = sig.replace(sampling_rate=1e3 * FS, symbol_rate=1e3 * FS)
    a = recovery.estimate_carrier_phase(sig, recovery.Tikhonov(1e2, 20)).value
    b = recovery.estimate_carrier_phase(fast, recovery.Tikhonov(1e5, 20)).value
    xpt.assert_allclose(a, b)
    with pytest.raises(ValueError, match="sampling_rate"):
        recovery.estimate_carrier_phase(
            sig.samples,
            recovery.Tikhonov(1e2, 20),
            constellation=Constellation.qam(16),
        )


def _wiener_record(constellation, linewidth_ts, esn0_db, n=20000, seed=1):
    """``constellation`` at 1 SPS under a Wiener phase (increment variance
    2πΔνT) with AWGN; returns ``(x, true_phase)``."""
    rng = np.random.default_rng(seed)
    s = constellation.points.astype(np.complex64)[
        rng.integers(0, constellation.order, n)
    ]
    phase = np.cumsum(rng.standard_normal(n) * np.sqrt(2 * np.pi * linewidth_ts))
    noise = np.sqrt(10 ** (-esn0_db / 10) / 2) * (
        rng.standard_normal(n) + 1j * rng.standard_normal(n)
    )
    return (s * np.exp(1j * phase) + noise).astype(np.complex64), phase


class TestWienerTracking:
    """Estimators following a Wiener phase, scored against the truth."""

    def test_tikhonov_beats_viterbi_viterbi_on_qpsk(self, xp):
        """QPSK, 2πΔνT = 2π·1e-5, Es/N0 12 dB, 8-symbol blocks: smoothing
        the block estimates across blocks lowers the error by more than 20 %
        (measured 0.037 vs 0.059 rad); neither slips (one quantum removed)."""
        c = Constellation.psk(4)
        x, phase = _wiener_record(c, 1e-5, 12.0)
        x = xp.asarray(x)
        vv = recovery.estimate_carrier_phase(
            x, recovery.ViterbiViterbi(block_size=8), constellation=c
        ).value
        tik = recovery.estimate_carrier_phase(
            x,
            recovery.Tikhonov(linewidth=1e-5 * FS, snr_db=12.0, block_size=8),
            constellation=c,
            sampling_rate=FS,
        ).value
        rms_vv = scoring.phase_rms(vv, phase, symmetry=4)
        rms_tik = scoring.phase_rms(tik, phase, symmetry=4)
        assert rms_tik < 0.8 * rms_vv, (rms_tik, rms_vv)
        assert rms_vv < 0.1
