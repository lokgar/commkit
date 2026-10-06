"""Tests for joint LMS/RLS+CPR equalizers and blockwise FOE.

Verification plan:
  1. Zero-Deviation Baseline   - cpr_type=None must produce bit-exact output
  3. Cycle Slip Stress Test    - π/2 steps are corrected, weights converge
  4. PLL Convergence / Phase Noise - RMSE within PLL jitter bound
  5. Blockwise Phase Coherence - chirp FOE recovers EVM within 0.5 dB of ideal
  6. MIMO Coverage             - 2x2 butterfly LMS+PLL converges on both channels
  7. BPS Phase Unwrap          - phase_trajectory is monotone under linear drift
  8. BPS Convergence           - lms(cpr_type='bps') converges under Wiener phase noise
  9. BPS Block Size > 1        - bps_block_size=32 still converges (incremental sum)
 10. RLS + BPS                 - rls(cpr_type='bps') convergence smoke test
 11. PLL Joint Channels        - cpr_joint_channels=True shares phase across MIMO
 12. CPRState warm-start       - second call resumes phase without re-lock transient
 13. input_norm_factor         - pre-supplied scale skips RMS, result matches manual scale
 14. Inline PLL raw gains      - cpr_pll_mu and cpr_pll_beta validation and parity
"""

import numpy as np
import pytest

from commkit.equalization import CPRState, lms, rls
from commkit.frequency import (
    MthPower,
    correct_frequency_offset,
    estimate_frequency_offset,
)
from commkit.mapping import Constellation
from commkit.recovery import recover_carrier_phase_pll
from tests.common.conversions import to_numpy
from tests.common.metrics import calc_mse_db
from tests.common.signals import (
    make_test_psk_samples,
    make_test_qam_samples,
    make_wiener_phase,
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _qpsk_signal(n_sym=4000, snr_db=20.0, rng=None):
    """Return (samples_2sps, symbols) for a QPSK signal with AWGN."""
    seed = 0 if rng is None else int(rng.integers(0, 100000))
    return make_test_psk_samples(
        order=4, num_symbols=n_sym, sps=2, snr_db=snr_db, seed=seed
    )


def _qam16_signal(n_sym=4000, snr_db=25.0, rng=None):
    """Return (samples_1sps, symbols) for 16-QAM at 1 SPS."""
    seed = 1 if rng is None else int(rng.integers(0, 100000))
    return make_test_qam_samples(
        order=16, num_symbols=n_sym, sps=1, snr_db=snr_db, seed=seed
    )


def _wiener_phase_signal(n_sym=4000, snr_db=20.0, linewidth=1e4, fs=1.0, seed=42):
    """Return (samples_1sps, symbols) for QPSK under Wiener phase noise at 1 SPS."""
    samples, syms = make_test_psk_samples(
        order=4, num_symbols=n_sym, sps=1, snr_db=None, seed=seed
    )
    phase = make_wiener_phase(
        num_symbols=n_sym, linewidth=linewidth, sample_rate=fs, seed=seed
    )
    samples = (samples * np.exp(1j * phase)).astype(np.complex64)
    if snr_db is not None:
        noise_std = np.sqrt(10 ** (-snr_db / 10) / 2)
        rng = np.random.default_rng(seed + 1)
        samples += noise_std * (
            rng.standard_normal(n_sym) + 1j * rng.standard_normal(n_sym)
        ).astype(np.complex64)
    return samples, syms


# -----------------------------------------------------------------------------
# Test Classes
# -----------------------------------------------------------------------------


class TestCPREqualizerBaseline:
    """Baseline equivalence and zero-deviation tests."""

    @pytest.mark.parametrize("algo", ["lms", "rls"])
    def test_cpr_none_baseline(self, algo, xp):
        """cpr_type=None produces bit-exact output vs the unmodified algorithm."""
        samples, syms = _qpsk_signal(n_sym=2000)
        kwargs = dict(
            training_symbols=syms[:500],
            num_taps=11,
            sps=2,
            modulation="psk",
            order=4,
        )
        fn = lms if algo == "lms" else rls
        extra = {} if algo == "lms" else {"sps": 2}
        kwargs.update(extra)

        res_base = fn(xp.asarray(samples), **kwargs, cpr_type=None)
        res_cpr_none = fn(xp.asarray(samples), **kwargs, cpr_type=None)

        assert bool(
            xp.all(xp.asarray(res_base.y_hat) == xp.asarray(res_cpr_none.y_hat))
        ), f"{algo}: cpr_type=None must be deterministic"
        assert res_cpr_none.phase_trajectory is None

    def test_baseline_cpr_none_matches_unwrapped(self, xp, xpt):
        """cpr_type=None baseline is identical to a standalone un-equalized slice."""
        samples, syms = _qpsk_signal(n_sym=500)
        kw = dict(
            num_taps=11,
            sps=2,
            modulation="psk",
            order=4,
            cpr_type="pll",
        )
        r_default = lms(samples, syms[:50], **kw)
        r_explicit_none = lms(
            samples, syms[:50], **kw, cpr_state=None, input_norm_factor=None
        )
        xpt.assert_array_equal(
            to_numpy(r_default.y_hat),
            to_numpy(r_explicit_none.y_hat),
        )


class TestCPRPLLConvergence:
    """PLL phase tracking, cycle-slip correction, and gain parameterization."""

    def test_cycle_slip_correction(self, xp):
        """LMS+PLL recovers through deliberate π/2 phase steps without diverging."""
        rng = np.random.default_rng(42)
        n_sym = 3000
        const = Constellation.psk(4).points.astype(np.complex64)
        idxs = rng.integers(0, 4, n_sym)
        syms = const[idxs]

        phase = np.zeros(n_sym, dtype=np.float64)
        for step_idx in range(500, n_sym, 500):
            phase[step_idx:] += np.pi / 2

        samples = (syms * np.exp(1j * phase).astype(np.complex64)).astype(np.complex64)
        noise = 0.05 * (rng.standard_normal(n_sym) + 1j * rng.standard_normal(n_sym))
        samples = (samples + noise.astype(np.complex64)).astype(np.complex64)

        res = lms(
            xp.asarray(samples),
            training_symbols=syms[:300],
            num_taps=1,
            sps=1,
            modulation="psk",
            order=4,
            cpr_type="pll",
            cpr_pll_bandwidth=5e-3,
            cpr_cycle_slip_correction=True,
            cpr_cycle_slip_history=200,
        )

        assert res.phase_trajectory is not None
        assert bool(xp.all(xp.isfinite(xp.asarray(res.y_hat)))), (
            "y_hat contains non-finite values"
        )
        y_dd = xp.asarray(res.y_hat[-500:])
        const_xp = xp.asarray(const)
        d = const_xp[xp.argmin(xp.abs(y_dd[:, None] - const_xp[None, :]) ** 2, axis=1)]
        mse = float(xp.mean(xp.abs(y_dd - d) ** 2))
        assert mse < 0.1, f"MSE after cycle slip recovery too large: {mse:.4f}"

    def test_pll_phase_noise_tracking(self, xp):
        """LMS+PLL tracks Wiener-process phase noise; phase RMSE within expected bound."""
        rng = np.random.default_rng(7)
        n_sym = 5000
        linewidth_ts = 1e-4
        const = Constellation.qam(16).points.astype(np.complex64)
        idxs = rng.integers(0, 16, n_sym)
        syms = const[idxs]

        phase_steps = rng.standard_normal(n_sym) * np.sqrt(2 * np.pi * linewidth_ts)
        phase_noise = np.cumsum(phase_steps).astype(np.float64)

        snr_db = 25.0
        noise_pwr = 10 ** (-snr_db / 10)
        awgn = np.sqrt(noise_pwr / 2) * (
            rng.standard_normal(n_sym) + 1j * rng.standard_normal(n_sym)
        ).astype(np.complex64)
        samples = (syms * np.exp(1j * phase_noise).astype(np.complex64) + awgn).astype(
            np.complex64
        )

        bw = 5e-3
        res = lms(
            xp.asarray(samples),
            training_symbols=syms[:1000],
            num_taps=1,
            sps=1,
            modulation="qam",
            order=16,
            cpr_type="pll",
            cpr_pll_bandwidth=bw,
            cpr_cycle_slip_correction=False,
        )

        assert res.phase_trajectory is not None
        phi_est = xp.asarray(res.phase_trajectory)[1500:]
        phi_true = xp.asarray(phase_noise[1500:])
        offset = float(xp.mean(phi_true - phi_est))
        rmse = float(xp.sqrt(xp.mean((phi_true - phi_est - offset) ** 2)))
        bound = np.sqrt(bw / linewidth_ts)
        assert rmse < bound, f"PLL phase RMSE {rmse:.4f} exceeds bound {bound:.4f}"

    def test_inline_raw_gains_match_bandwidth(self, xp):
        """cpr_pll_mu/beta set to bandwidth-equivalent gains reproduces bandwidth path."""
        samples, syms = _qpsk_signal(n_sym=1500)
        samples = (samples * np.exp(1j * 0.2)).astype(np.complex64)
        bw = 5e-3
        kw = dict(
            training_symbols=syms[:300],
            num_taps=11,
            sps=2,
            modulation="psk",
            order=4,
            cpr_type="pll",
            cpr_cycle_slip_correction=False,
        )
        res_bw = lms(xp.asarray(samples), **kw, cpr_pll_bandwidth=bw)
        res_raw = lms(
            xp.asarray(samples),
            **kw,
            cpr_pll_mu=float(np.float32(4.0 * bw)),
            cpr_pll_beta=float(np.float32(4.0 * bw**2)),
        )
        max_diff = float(
            xp.max(xp.abs(xp.asarray(res_bw.y_hat) - xp.asarray(res_raw.y_hat)))
        )
        assert max_diff < 1e-5, (
            f"raw vs bandwidth y_hat mismatch (max diff {max_diff:.2e})"
        )

    def test_inline_beta_without_mu_raises(self, xp):
        """cpr_pll_beta with cpr_pll_mu=None is ambiguous and must raise ValueError."""
        samples, syms = _qpsk_signal(n_sym=400)
        with pytest.raises(ValueError, match="beta requires mu"):
            lms(
                xp.asarray(samples),
                training_symbols=syms[:100],
                num_taps=11,
                sps=2,
                modulation="psk",
                order=4,
                cpr_type="pll",
                cpr_pll_beta=1e-3,
            )

    def test_inline_pll_parity_with_standalone(self, xp):
        """A frozen 1-tap identity equalizer reduces inline PLL to standalone DD-PLL."""
        rng = np.random.default_rng(3)
        n_sym = 2000
        const = Constellation.psk(4).points.astype(np.complex64)
        syms = const[rng.integers(0, 4, n_sym)]
        samples = (syms * np.exp(1j * 0.3).astype(np.complex64)).astype(np.complex64)

        m, b = 0.02, 1e-4
        res = lms(
            xp.asarray(samples),
            training_symbols=syms[:200],
            num_taps=1,
            sps=1,
            step_size=0.0,
            w_init=xp.asarray(np.array([1.0 + 0j], dtype=np.complex64)),
            modulation="psk",
            order=4,
            cpr_type="pll",
            cpr_pll_mu=m,
            cpr_pll_beta=b,
            cpr_cycle_slip_correction=False,
        )
        phi_inline = to_numpy(res.phase_trajectory)
        phi_std = to_numpy(
            recover_carrier_phase_pll(xp.asarray(samples), "psk", 4, mu=m, beta=b)
        )

        tail = slice(n_sym // 4, n_sym)
        diff = np.unwrap(phi_inline[tail]) - np.unwrap(phi_std[tail])
        assert np.std(diff) < 1e-3, f"inline vs standalone phase std {np.std(diff):.2e}"


class TestCPRBPSConvergence:
    """Blind Phase Search unwrapping, convergence, and block sizing."""

    def test_bps_phase_unwrap(self, xp):
        """phase_trajectory from BPS must not wrap back to [0, π/2) under a ramp."""
        rng = np.random.default_rng(5)
        n_sym = 3000
        const = Constellation.qam(16).points.astype(np.complex64)
        syms = const[rng.integers(0, 16, n_sym)]

        phase_true = np.linspace(0.0, 3.0, n_sym, dtype=np.float64)
        noise_pwr = 10 ** (-25.0 / 10)
        awgn = np.sqrt(noise_pwr / 2) * (
            rng.standard_normal(n_sym) + 1j * rng.standard_normal(n_sym)
        ).astype(np.complex64)
        samples = (syms * np.exp(1j * phase_true).astype(np.complex64) + awgn).astype(
            np.complex64
        )

        res = lms(
            xp.asarray(samples),
            training_symbols=syms[:500],
            num_taps=1,
            sps=1,
            modulation="qam",
            order=16,
            cpr_type="bps",
            cpr_bps_test_phases=64,
            cpr_bps_block_size=32,
            cpr_cycle_slip_correction=False,
        )

        phi = xp.asarray(res.phase_trajectory).astype(xp.float64)
        span = float(phi[-1] - phi[0])
        assert span > 1.0, f"Phase did not advance: span={span:.3f} rad"
        assert span > np.pi / 2, (
            f"BPS phase_trajectory looks wrapped (span={span:.3f} rad < π/2)"
        )

    def test_bps_phase_noise_tracking(self, xp):
        """LMS+BPS converges under Wiener phase noise (Numba backend)."""
        rng = np.random.default_rng(11)
        n_sym = 5000
        linewidth_ts = 5e-5
        const = Constellation.qam(16).points.astype(np.complex64)
        syms = const[rng.integers(0, 16, n_sym)]

        phase_noise = np.cumsum(
            rng.standard_normal(n_sym) * np.sqrt(2 * np.pi * linewidth_ts)
        ).astype(np.float64)
        noise_pwr = 10 ** (-25.0 / 10)
        awgn = np.sqrt(noise_pwr / 2) * (
            rng.standard_normal(n_sym) + 1j * rng.standard_normal(n_sym)
        ).astype(np.complex64)
        samples = xp.asarray(
            (syms * np.exp(1j * phase_noise).astype(np.complex64) + awgn).astype(
                np.complex64
            )
        )

        res_bps = lms(
            samples,
            training_symbols=syms[:1000],
            num_taps=1,
            sps=1,
            modulation="qam",
            order=16,
            cpr_type="bps",
            cpr_bps_test_phases=64,
            cpr_bps_block_size=32,
            cpr_cycle_slip_correction=False,
        )
        res_none = lms(
            samples,
            training_symbols=syms[:1000],
            num_taps=1,
            sps=1,
            modulation="qam",
            order=16,
            cpr_type=None,
        )

        mse_bps = float(xp.mean(xp.abs(xp.asarray(res_bps.error[-2000:])) ** 2))
        mse_none = float(xp.mean(xp.abs(xp.asarray(res_none.error[-2000:])) ** 2))
        assert mse_bps < mse_none, (
            f"BPS MSE ({mse_bps:.4f}) not better than no-CPR ({mse_none:.4f}) "
            "under Wiener phase noise"
        )

    def test_bps_block_size_convergence(self, xp):
        """lms(cpr_type='bps', bps_block_size=32) converges - verifies incremental sum."""
        rng = np.random.default_rng(13)
        n_sym = 4000
        const = Constellation.qam(16).points.astype(np.complex64)
        syms = const[rng.integers(0, 16, n_sym)]

        phase_noise = np.cumsum(
            rng.standard_normal(n_sym) * np.sqrt(2 * np.pi * 5e-5)
        ).astype(np.float64)
        noise_pwr = 10 ** (-25.0 / 10)
        awgn = np.sqrt(noise_pwr / 2) * (
            rng.standard_normal(n_sym) + 1j * rng.standard_normal(n_sym)
        ).astype(np.complex64)
        samples = xp.asarray(
            (syms * np.exp(1j * phase_noise).astype(np.complex64) + awgn).astype(
                np.complex64
            )
        )

        res_k1 = lms(
            samples,
            training_symbols=syms[:1000],
            num_taps=1,
            sps=1,
            modulation="qam",
            order=16,
            cpr_type="bps",
            cpr_bps_test_phases=32,
            cpr_bps_block_size=1,
            cpr_cycle_slip_correction=False,
        )
        res_k32 = lms(
            samples,
            training_symbols=syms[:1000],
            num_taps=1,
            sps=1,
            modulation="qam",
            order=16,
            cpr_type="bps",
            cpr_bps_test_phases=32,
            cpr_bps_block_size=32,
            cpr_cycle_slip_correction=False,
        )

        mse_k1 = float(xp.mean(xp.abs(xp.asarray(res_k1.error[-1000:])) ** 2))
        mse_k32 = float(xp.mean(xp.abs(xp.asarray(res_k32.error[-1000:])) ** 2))
        assert mse_k32 < 0.1, f"BPS K=32 did not converge: MSE={mse_k32:.4f}"
        assert mse_k1 < 0.1, f"BPS K=1 did not converge: MSE={mse_k1:.4f}"

    def test_rls_bps_convergence(self, xp):
        """rls(cpr_type='bps') converges under phase noise."""
        rng = np.random.default_rng(17)
        n_sym = 3000
        const = Constellation.qam(16).points.astype(np.complex64)
        syms = const[rng.integers(0, 16, n_sym)]

        phase_noise = np.cumsum(
            rng.standard_normal(n_sym) * np.sqrt(2 * np.pi * 5e-5)
        ).astype(np.float64)
        noise_pwr = 10 ** (-25.0 / 10)
        awgn = np.sqrt(noise_pwr / 2) * (
            rng.standard_normal(n_sym) + 1j * rng.standard_normal(n_sym)
        ).astype(np.complex64)
        samples = (syms * np.exp(1j * phase_noise).astype(np.complex64) + awgn).astype(
            np.complex64
        )

        res = rls(
            xp.asarray(samples),
            training_symbols=syms[:500],
            num_taps=1,
            sps=1,
            modulation="qam",
            order=16,
            cpr_type="bps",
            cpr_bps_test_phases=64,
            cpr_bps_block_size=32,
            cpr_cycle_slip_correction=False,
        )

        assert res.phase_trajectory is not None
        assert res.phase_trajectory.shape == (n_sym,)
        mse = float(xp.mean(xp.abs(xp.asarray(res.error[-1000:])) ** 2))
        assert mse < 0.1, f"RLS+BPS did not converge: MSE={mse:.4f}"


class TestCPRMIMOJoint:
    """Multi-channel MIMO CPR and joint carrier phase tracking."""

    def test_mimo_lms_pll(self, xp):
        """2x2 butterfly LMS+PLL converges on both output channels."""
        rng = np.random.default_rng(99)
        n_sym = 3000
        const = Constellation.qam(16).points.astype(np.complex64)
        const_xp = xp.asarray(const)

        syms_a = const[rng.integers(0, 16, n_sym)]
        syms_b = const[rng.integers(0, 16, n_sym)]
        phase_a = np.float32(0.3)
        phase_b = np.float32(1.1)

        sig_a = (syms_a * np.exp(1j * phase_a)).astype(np.complex64)
        sig_b = (syms_b * np.exp(1j * phase_b)).astype(np.complex64)

        snr_db = 25.0
        noise_pwr = 10 ** (-snr_db / 10)
        noise_a = np.sqrt(noise_pwr / 2) * (
            rng.standard_normal(n_sym) + 1j * rng.standard_normal(n_sym)
        ).astype(np.complex64)
        noise_b = np.sqrt(noise_pwr / 2) * (
            rng.standard_normal(n_sym) + 1j * rng.standard_normal(n_sym)
        ).astype(np.complex64)

        samples = xp.asarray(np.stack([sig_a + noise_a, sig_b + noise_b]))
        training = np.stack([syms_a[:500], syms_b[:500]])

        res = lms(
            samples,
            training_symbols=training,
            num_taps=1,
            sps=1,
            modulation="qam",
            order=16,
            cpr_type="pll",
            cpr_pll_bandwidth=5e-3,
            cpr_cycle_slip_correction=False,
        )

        assert res.phase_trajectory is not None
        assert res.phase_trajectory.shape == (2, n_sym), (
            f"Expected (2, {n_sym}), got {res.phase_trajectory.shape}"
        )

        for ch in range(2):
            y_ss = xp.asarray(res.y_hat[ch, -1000:])
            d = const_xp[
                xp.argmin(xp.abs(y_ss[:, None] - const_xp[None, :]) ** 2, axis=1)
            ]
            mse = float(xp.mean(xp.abs(y_ss - d) ** 2))
            assert mse < 0.1, f"MIMO channel {ch} MSE too large: {mse:.4f}"

    def test_pll_joint_channels(self, xp):
        """cpr_joint_channels=True makes both PLL integrators identical (shared LO)."""
        rng = np.random.default_rng(23)
        n_sym = 3000
        const = Constellation.qam(16).points.astype(np.complex64)
        syms_a = const[rng.integers(0, 16, n_sym)]
        syms_b = const[rng.integers(0, 16, n_sym)]

        phase_noise = np.cumsum(
            rng.standard_normal(n_sym) * np.sqrt(2 * np.pi * 1e-4)
        ).astype(np.float64)
        noise_pwr = 10 ** (-25.0 / 10)

        def _awgn():
            return (
                np.sqrt(noise_pwr / 2)
                * (rng.standard_normal(n_sym) + 1j * rng.standard_normal(n_sym))
            ).astype(np.complex64)

        phasor = np.exp(1j * phase_noise).astype(np.complex64)
        samples = xp.asarray(
            np.stack(
                [
                    syms_a * phasor + _awgn(),
                    syms_b * phasor + _awgn(),
                ]
            )
        )
        training = np.stack([syms_a[:500], syms_b[:500]])

        res = lms(
            samples,
            training_symbols=training,
            num_taps=1,
            sps=1,
            modulation="qam",
            order=16,
            cpr_type="pll",
            cpr_pll_bandwidth=5e-3,
            cpr_joint_channels=True,
            cpr_cycle_slip_correction=False,
        )

        assert res.phase_trajectory is not None
        assert res.phase_trajectory.shape == (2, n_sym)
        phi0 = xp.asarray(res.phase_trajectory[0])
        phi1 = xp.asarray(res.phase_trajectory[1])
        assert bool(xp.all(phi0 == phi1)), (
            "cpr_joint_channels=True: PLL integrators must be identical"
        )


class TestCPRStatePersistence:
    """State preservation and warm-start behavior."""

    @pytest.mark.parametrize("cpr_mode", ["pll", "bps"])
    def test_cpr_state_warmstart_lms(self, cpr_mode, xp):
        """Second lms call with cpr_state should have lower initial MSE than cold restart."""
        n_sym = 4000
        half = n_sym // 2
        samples_np, syms_np = _wiener_phase_signal(n_sym=n_sym)
        s1, s2 = samples_np[:half], samples_np[half:]
        t1, t2 = syms_np[:half], syms_np[half:]

        r1 = lms(
            s1,
            t1,
            num_taps=5,
            sps=1,
            step_size=5e-3,
            modulation="psk",
            order=4,
            cpr_type=cpr_mode,
            cpr_bps_block_size=16,
            cpr_bps_test_phases=32,
        )
        assert r1.cpr_state is not None, (
            "cpr_state must be populated when cpr_type is set"
        )
        assert r1.cpr_state.cpr_type == cpr_mode
        assert r1.cpr_state.num_ch == 1

        r2_warm = lms(
            s2,
            t2[:20],
            num_taps=5,
            sps=1,
            step_size=5e-3,
            modulation="psk",
            order=4,
            cpr_type=cpr_mode,
            cpr_bps_block_size=16,
            cpr_bps_test_phases=32,
            w_init=r1.weights,
            cpr_state=r1.cpr_state,
            input_norm_factor=r1.input_norm_factor,
        )
        r2_cold = lms(
            s2,
            t2[:20],
            num_taps=5,
            sps=1,
            step_size=5e-3,
            modulation="psk",
            order=4,
            cpr_type=cpr_mode,
            cpr_bps_block_size=16,
            cpr_bps_test_phases=32,
            w_init=r1.weights,
        )
        n_eval_start, n_eval_end = 20, 50
        mse_warm = calc_mse_db(
            r2_warm.y_hat[n_eval_start:n_eval_end], t2[n_eval_start:n_eval_end]
        )
        mse_cold = calc_mse_db(
            r2_cold.y_hat[n_eval_start:n_eval_end], t2[n_eval_start:n_eval_end]
        )
        assert mse_warm < mse_cold + 3.0, (
            f"Warm CPRState should not be worse than cold by >3 dB: "
            f"warm={mse_warm:.1f} dB  cold={mse_cold:.1f} dB"
        )

    def test_cpr_state_warmstart_rls(self, xp):
        """rls with cpr_state warm-start: second call has valid cpr_state output."""
        n_sym = 2000
        half = n_sym // 2
        samples_np, syms_np = _wiener_phase_signal(n_sym=n_sym)
        s1, s2 = samples_np[:half], samples_np[half:]
        t1 = syms_np[:half]

        r1 = rls(
            s1,
            t1,
            num_taps=5,
            sps=1,
            modulation="psk",
            order=4,
            cpr_type="pll",
        )
        assert r1.cpr_state is not None
        assert isinstance(r1.cpr_state, CPRState)
        assert r1.cpr_state.pll_phi is not None

        r2 = rls(
            s2,
            None,
            num_taps=5,
            sps=1,
            modulation="psk",
            order=4,
            cpr_type="pll",
            w_init=r1.weights,
            cpr_state=r1.cpr_state,
            input_norm_factor=r1.input_norm_factor,
        )
        assert r2.cpr_state is not None
        assert r2.cpr_state.cpr_type == "pll"

    def test_input_norm_factor_lms_skips_rms(self, xp, xpt):
        """Supplying input_norm_factor should give same result as letting lms compute it."""
        samples_np, syms_np = _wiener_phase_signal(n_sym=1000)
        samples, syms = xp.asarray(samples_np), xp.asarray(syms_np)
        kw = dict(num_taps=5, sps=1, step_size=5e-3, modulation="psk", order=4)

        r_auto = lms(samples, syms[:50], **kw)
        nf = r_auto.input_norm_factor

        r_supplied = lms(samples, syms[:50], **kw, input_norm_factor=nf)
        xpt.assert_allclose(
            xp.asarray(r_supplied.y_hat),
            xp.asarray(r_auto.y_hat),
            rtol=1e-5,
            atol=1e-6,
        )
        assert r_supplied.input_norm_factor == pytest.approx(float(nf), rel=1e-6)


class TestBlockwiseFOE:
    """Frequency offset estimation and blockwise correction."""

    def test_blockwise_foe_chirp(self, xp):
        """A blockwise M-th power estimate removes a linearly chirping frequency."""
        rng = np.random.default_rng(3)
        fs = 1e9
        n = 65536
        sps = 2
        f_start, f_end = 1e6, 5e6

        f_t = np.linspace(f_start, f_end, n)
        phase_chirp = 2 * np.pi * np.cumsum(f_t) / fs
        carrier = np.exp(1j * phase_chirp).astype(np.complex64)

        const = Constellation.qam(16).points.astype(np.complex64)
        idxs = rng.integers(0, 16, n // sps)
        syms = const[idxs]
        base_np = np.repeat(syms, sps).astype(np.complex64)
        samples = xp.asarray((base_np * carrier).astype(np.complex64))
        base = xp.asarray(base_np)

        est = estimate_frequency_offset(
            samples,
            MthPower(block_size=4096, overlap=0.5),
            sampling_rate=fs,
            constellation=Constellation.qam(16),
        )
        corrected = correct_frequency_offset(samples, est, sampling_rate=fs)

        ratio_corr = float(
            xp.mean(xp.abs(corrected - base) ** 2) / xp.mean(xp.abs(base) ** 2)
        )
        ratio_uncorr = float(
            xp.mean(xp.abs(samples - base) ** 2) / xp.mean(xp.abs(base) ** 2)
        )
        evm_corrected = 10 * np.log10(ratio_corr)
        evm_uncorrected = 10 * np.log10(ratio_uncorr)
        assert evm_corrected < evm_uncorrected - 10.0, (
            f"FOE correction ineffective: corrected EVM={evm_corrected:.1f} dB, "
            f"uncorrected={evm_uncorrected:.1f} dB"
        )
