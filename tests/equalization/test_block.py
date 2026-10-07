"""Tests for block_lms - frequency-domain block LMS equalizer.

Coverage:
  1. Output shapes - SISO and MIMO, with/without CPR
  2. Gradient descent - MSE decreases over blocks (identity channel, noise)
  3. Identity channel parity - with sufficient training, output ≈ input symbols
  4. initial_taps passthrough - warm-start weights are used
  5. store_weights shape - weights_history has expected layout
  6. num_train_symbols boundary - DA/DD switch is respected
  7. Last-block edge - n_sym not a multiple of block_size
  8. cpr validation - PLL raises TypeError
  9. BPS + CPR - phase_trajectory shape and MSE better than no CPR under phase noise
 10. BPS block_size vs BPS block_size independence - different values accepted
 11. MIMO butterfly convergence - 2x2, training on both channels
 12. state= warm-start - second block_lms call resumes BPS state seamlessly
 14. CUDA graph and transfer hygiene on GPU
"""

import numpy as np
import pytest

from commkit.core import Signal
from commkit.equalization import block_lms
from commkit.mapping import Constellation
from commkit.math import normalize
from commkit.recovery import BPS, PLL, CycleSlip
from tests.common.conversions import to_numpy
from tests.common.signals import (
    make_test_mimo_samples,
    make_test_qam_samples,
    make_wiener_phase,
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _qam16(n_sym=4096, snr_db=25.0, sps=2, rng=None):
    """Return (samples_sps, symbols) for 16-QAM with AWGN, repeat upsampling."""
    seed = 42 if rng is None else int(rng.integers(0, 100000))
    return make_test_qam_samples(
        order=16, num_symbols=n_sym, sps=sps, snr_db=snr_db, seed=seed
    )


def _wiener_qam16_block(n_sym=4096, snr_db=25.0, linewidth=5e3, seed=11):
    """Return (samples, symbols) for 16-QAM under Wiener phase noise."""
    samples, syms = make_test_qam_samples(
        order=16, num_symbols=n_sym, sps=1, snr_db=snr_db, seed=seed
    )
    phase = make_wiener_phase(
        num_symbols=n_sym,
        linewidth=linewidth,
        sample_rate=1.0,
        seed=seed,
        dtype=np.float64,
    )
    return (syms * np.exp(1j * phase)).astype(np.complex64), syms


def _wiener_qam16_trackable(n_sym, snr_db=25.0, sigma_phi=0.005, seed=33):
    """16-QAM under slow Wiener phase noise that BPS can track."""
    rng = np.random.default_rng(seed)
    const = Constellation.qam(16).points.astype(np.complex64)
    const = normalize(const, mode="average_power").astype(np.complex64)
    half = n_sym // 2
    syms_h = const[rng.integers(0, 16, half)]
    syms = np.concatenate([syms_h, rng.permutation(syms_h)])
    phase = np.cumsum(rng.normal(0.0, sigma_phi, n_sym))
    samples = (syms * np.exp(1j * phase)).astype(np.complex64)
    noise_std = float(np.sqrt(10 ** (-snr_db / 10) / 2))
    samples += noise_std * (
        rng.standard_normal(n_sym) + 1j * rng.standard_normal(n_sym)
    ).astype(np.complex64)
    return samples, syms


# -----------------------------------------------------------------------------
# Test Classes
# -----------------------------------------------------------------------------


class TestBlockLMSShapes:
    """Output array shape validation for SISO, MIMO, CPR, and history tracking."""

    def test_output_shape_siso(self, xp):
        samples, syms = _qam16(n_sym=1024, sps=2)
        r = block_lms(
            xp.asarray(samples),
            xp.asarray(syms),
            num_taps=11,
            sps=2,
            constellation=Constellation.qam(16),
            block_size=128,
        )
        n_sym = len(syms)
        assert r.y_hat.shape == (n_sym,), f"y_hat shape {r.y_hat.shape}"
        assert r.weights.shape == (11,), f"weights shape {r.weights.shape}"
        assert r.error.shape == (n_sym,)
        assert r.phase_trajectory is None

    def test_output_shape_mimo(self, xp):
        samples, training = make_test_mimo_samples(
            num_channels=2,
            order=16,
            num_symbols=1024,
            sps=2,
            snr_db=25.0,
            seed=1,
            xp=xp,
        )
        r = block_lms(
            samples,
            training,
            num_taps=11,
            sps=2,
            constellation=Constellation.qam(16),
            block_size=128,
        )
        assert r.y_hat.shape == (2, 1024)
        assert r.weights.shape == (2, 2, 11)
        assert r.error.shape == (2, 1024)

    def test_output_shape_bps(self, xp):
        samples, syms = _qam16(n_sym=1024, sps=2)
        r = block_lms(
            xp.asarray(samples),
            xp.asarray(syms),
            num_taps=11,
            sps=2,
            constellation=Constellation.qam(16),
            block_size=128,
            cpr=BPS(test_phases=16, block_size=32),
        )
        assert r.y_hat.shape == (1024,)
        assert r.phase_trajectory is not None
        assert r.phase_trajectory.shape == (1024,), (
            f"phase_trajectory shape {r.phase_trajectory.shape}"
        )

    def test_output_shape_bps_mimo(self, xp):
        samples, training = make_test_mimo_samples(
            num_channels=2,
            order=16,
            num_symbols=1024,
            sps=2,
            snr_db=25.0,
            seed=2,
            xp=xp,
        )
        r = block_lms(
            samples,
            training,
            num_taps=11,
            sps=2,
            constellation=Constellation.qam(16),
            block_size=128,
            cpr=BPS(test_phases=16),
        )
        assert r.phase_trajectory.shape == (2, 1024)

    def test_store_weights_shape_siso(self, xp):
        samples, syms = _qam16(n_sym=512, sps=2)
        r = block_lms(
            xp.asarray(samples),
            xp.asarray(syms),
            num_taps=11,
            sps=2,
            constellation=Constellation.qam(16),
            block_size=64,
            store_weights=True,
        )
        assert r.weights_history is not None
        assert r.weights_history.shape == (512, 11), (
            f"weights_history shape {r.weights_history.shape}"
        )

    def test_store_weights_shape_mimo(self, xp):
        samples, training = make_test_mimo_samples(
            num_channels=2, order=16, num_symbols=512, sps=2, snr_db=25.0, seed=3, xp=xp
        )
        r = block_lms(
            samples,
            training,
            num_taps=11,
            sps=2,
            constellation=Constellation.qam(16),
            block_size=64,
            store_weights=True,
        )
        assert r.weights_history.shape == (512, 2, 2, 11)


class TestBlockLMSConvergence:
    """Convergence characteristics under gradient descent, AWGN, and ISI."""

    def test_mse_decreases(self, xp):
        """MSE in the last quarter of the signal must be less than in the first quarter."""
        samples, syms = _qam16(n_sym=8192, snr_db=25.0, sps=2)
        r = block_lms(
            xp.asarray(samples),
            xp.asarray(syms[:512]),
            num_taps=11,
            sps=2,
            step_size=5e-4,
            constellation=Constellation.qam(16),
            block_size=128,
        )
        err = xp.abs(r.error) ** 2
        n = len(err)
        assert float(err[: n // 4].mean()) > float(err[3 * n // 4 :].mean()), (
            "MSE did not decrease from first to last quarter"
        )

    def test_identity_channel_convergence(self, xp):
        """On a near-identity channel, symbols after training should match reference."""
        samples, syms = _qam16(n_sym=4096, snr_db=30.0, sps=2)
        n_train = 1024
        syms_xp = xp.asarray(syms)
        r = block_lms(
            xp.asarray(samples),
            syms_xp,
            num_taps=11,
            sps=2,
            step_size=5e-4,
            constellation=Constellation.qam(16),
            block_size=128,
        )
        y_eval = r.y_hat[n_train:]
        s_eval = syms_xp[n_train:]
        evm = float(
            xp.sqrt(
                xp.mean(xp.abs(y_eval - s_eval) ** 2) / xp.mean(xp.abs(s_eval) ** 2)
            )
        )
        assert evm < 0.15, f"EVM {evm:.3f} too high - equalizer did not converge"

    def test_mimo_convergence(self, xp):
        """2x2 MIMO: both channels should converge to low EVM."""
        n_sym = 4096
        sps = 1
        samples, training = make_test_mimo_samples(
            num_channels=2,
            order=16,
            num_symbols=n_sym,
            sps=sps,
            snr_db=26.0,
            seed=5,
            xp=xp,
        )

        r = block_lms(
            samples,
            training,
            num_taps=5,
            sps=sps,
            step_size=2e-3,
            constellation=Constellation.qam(16),
            block_size=64,
        )
        n_eval = n_sym // 2
        for ch in range(2):
            evm = float(
                xp.sqrt(
                    xp.mean(xp.abs(r.y_hat[ch, n_eval:] - training[ch, n_eval:]) ** 2)
                    / xp.mean(xp.abs(training[ch, n_eval:]) ** 2)
                )
            )
            assert evm < 0.15, f"MIMO ch{ch} EVM {evm:.3f} too high"

    def test_isi_channel_convergence(self, xp):
        """Known 3-tap ISI channel: equalizer must converge across block boundaries."""
        rng = np.random.default_rng(99)
        n_sym = 2048
        sps = 1
        block_size = 32

        channel = np.array([0.1, 1.0, 0.1], dtype=np.complex64)
        const = Constellation.qam(4).points.astype(np.complex64)
        const = normalize(const, mode="average_power").astype(np.complex64)
        syms_np = const[rng.integers(0, 4, n_sym)].astype(np.complex64)
        received_np = np.convolve(syms_np, channel, mode="full")[:n_sym].astype(
            np.complex64
        )
        received = xp.asarray(received_np)
        syms = xp.asarray(syms_np)

        r = block_lms(
            received,
            syms,
            num_taps=5,
            sps=sps,
            step_size=5e-3,
            constellation=Constellation.qam(4),
            block_size=block_size,
        )
        n_eval = n_sym // 2
        evm = float(
            xp.sqrt(
                xp.mean(xp.abs(r.y_hat[n_eval:] - syms[n_eval:]) ** 2)
                / xp.mean(xp.abs(syms[n_eval:]) ** 2)
            )
        )
        assert evm < 0.10, (
            f"EVM {evm:.3f} - ISI equalization failed across block boundaries"
        )


class TestBlockLMSWeightHandling:
    """Initialization, warm-starting, and normalization parameter handling."""

    def test_initial_taps_used(self, xp):
        """Warm-starting from converged weights should give lower initial MSE."""
        samples, syms = _qam16(n_sym=4096, snr_db=25.0, sps=2)
        samples_xp = xp.asarray(samples)
        syms_xp = xp.asarray(syms)
        # First pass: converge weights
        r1 = block_lms(
            samples_xp,
            syms_xp,
            num_taps=11,
            sps=2,
            step_size=5e-4,
            constellation=Constellation.qam(16),
            block_size=128,
        )
        # Second pass: warm-start; MSE at start should be low
        r2 = block_lms(
            samples_xp,
            syms_xp,
            num_taps=11,
            sps=2,
            step_size=5e-4,
            constellation=Constellation.qam(16),
            block_size=128,
            initial_taps=r1.weights,
        )
        mse_cold_start = float(xp.mean(xp.abs(r1.error[:128]) ** 2))
        mse_warm_start = float(xp.mean(xp.abs(r2.error[:128]) ** 2))
        assert mse_warm_start < mse_cold_start, (
            f"warm-start MSE ({mse_warm_start:.4f}) not better than cold ({mse_cold_start:.4f})"
        )

    def test_num_train_symbols_respected(self, xp):
        """Training length is determined by the length of training_symbols passed in."""
        samples, syms = _qam16(n_sym=2048, snr_db=30.0, sps=2)
        samples_xp = xp.asarray(samples)
        syms_xp = xp.asarray(syms)
        # Pure DA (all training)
        r_da = block_lms(
            samples_xp,
            syms_xp,
            num_taps=11,
            sps=2,
            step_size=5e-4,
            constellation=Constellation.qam(16),
            block_size=128,
        )
        assert r_da.num_train_symbols == 2048

        # Pre-sliced training to 256
        r_clip = block_lms(
            samples_xp,
            syms_xp[..., :256],
            num_taps=11,
            sps=2,
            step_size=5e-4,
            constellation=Constellation.qam(16),
            block_size=128,
        )
        assert r_clip.num_train_symbols == 256


class TestBlockLMSCPRIntegration:
    """BPS carrier phase recovery, cycle slip correction, and state persistence."""

    def test_bps_block_size_independent(self, xp):
        """block_size=256 with BPS(block_size=16) must produce per-symbol phi."""
        samples, syms = _qam16(n_sym=1024, sps=2)
        r = block_lms(
            xp.asarray(samples),
            xp.asarray(syms),
            num_taps=11,
            sps=2,
            constellation=Constellation.qam(16),
            block_size=256,
            cpr=BPS(test_phases=16, block_size=16),
        )
        assert r.phase_trajectory.shape == (1024,)
        phi = r.phase_trajectory
        assert not bool(xp.all(phi == phi[0])), (
            "All phi identical - expected per-symbol variation with BPS(block_size=16)"
        )

    def test_bps_reduces_mse_under_phase_noise(self, xp):
        """With strong phase noise, BPS should produce lower steady-state MSE."""
        rng = np.random.default_rng(99)
        n_sym = 4096
        sps = 2
        const = Constellation.psk(4).points.astype(np.complex64)
        syms = const[rng.integers(0, 4, n_sym)]

        # Add random-walk phase noise
        phase_noise = np.cumsum(0.03 * rng.standard_normal(n_sym)).astype(np.float32)
        samples_pn = (
            np.repeat(syms * np.exp(1j * phase_noise), sps)
            + 0.1
            * (
                rng.standard_normal(2 * n_sym) + 1j * rng.standard_normal(2 * n_sym)
            ).astype(np.complex64)
        ).astype(np.complex64)

        n_eval = n_sym // 2
        r_no_cpr = block_lms(
            xp.asarray(samples_pn),
            xp.asarray(syms[:512]),
            num_taps=7,
            sps=sps,
            step_size=5e-4,
            block_size=128,
            constellation=Constellation.psk(4),
        )
        r_bps = block_lms(
            xp.asarray(samples_pn),
            xp.asarray(syms[:512]),
            num_taps=7,
            sps=sps,
            step_size=5e-4,
            block_size=128,
            constellation=Constellation.psk(4),
            cpr=BPS(test_phases=32, block_size=32),
        )

        mse_no_cpr = float(xp.mean(xp.abs(r_no_cpr.error[n_eval:]) ** 2))
        mse_bps = float(xp.mean(xp.abs(r_bps.error[n_eval:]) ** 2))
        assert mse_bps < mse_no_cpr, (
            f"BPS MSE ({mse_bps:.4f}) not better than no-CPR ({mse_no_cpr:.4f}) "
            "under phase noise"
        )

    def test_state_warmstart_block_lms_bps(self, xp):
        """block_lms with BPS carries the cross-block BPS state in result.state."""
        n_sym = 4096
        half = n_sym // 2
        samples_np, syms_np = _wiener_qam16_block(n_sym=n_sym)
        kw = dict(
            num_taps=11,
            sps=1,
            step_size=5e-4,
            constellation=Constellation.qam(16),
            cpr=BPS(test_phases=32, block_size=16),
        )

        r1 = block_lms(xp.asarray(samples_np[:half]), xp.asarray(syms_np[:half]), **kw)
        carrier = r1.state.carrier
        for name in ("prev4", "offset4", "d2_hist"):
            assert isinstance(getattr(carrier, name), np.ndarray), name
        assert r1.state.overlap % 1 == 0 and r1.state.block_size == 256

        ov = r1.state.overlap
        r2 = block_lms(
            xp.asarray(samples_np[half:]),
            xp.asarray(syms_np[half - ov : half - ov + 50]),
            **kw,
            state=r1.state,
        )
        assert r2.state.equalizer == "block_lms"
        assert r2.phase_trajectory is not None

    def test_block_lms_cycle_slip_correction(self, xp):
        """block_lms with cpr_cycle_slip_correction=True recovers through deliberate π/2 phase steps."""
        rng = np.random.default_rng(77)
        n_sym = 4096
        const = Constellation.qam(16).points.astype(np.complex64)
        const = normalize(const, mode="average_power").astype(np.complex64)
        syms = const[rng.integers(0, 16, n_sym)]

        # Inject a π/2 phase step every 512 symbols (well within block_size=256 boundaries)
        phase = np.zeros(n_sym, dtype=np.float64)
        for step_idx in range(512, n_sym, 512):
            phase[step_idx:] += np.pi / 2

        samples = (syms * np.exp(1j * phase).astype(np.complex64)).astype(np.complex64)
        noise_std = float(np.sqrt(10 ** (-25.0 / 10) / 2))
        samples += noise_std * (
            rng.standard_normal(n_sym) + 1j * rng.standard_normal(n_sym)
        ).astype(np.complex64)

        res = block_lms(
            xp.asarray(samples),
            xp.asarray(syms[:256]),
            num_taps=1,
            sps=1,
            step_size=1e-3,
            constellation=Constellation.qam(16),
            block_size=128,
            cpr=BPS(
                test_phases=64, block_size=32, cycle_slip=CycleSlip(threshold=np.pi / 4)
            ),
        )

        assert res.phase_trajectory is not None
        assert res.state.carrier.cs_buf_y is not None

        y_tail = res.y_hat[-n_sym // 4 :]
        const_xp = xp.asarray(const)
        d2 = xp.abs(y_tail[:, None] - const_xp[None, :]) ** 2
        decisions = const_xp[xp.argmin(d2, axis=1)]
        mse = float(xp.mean(xp.abs(y_tail - decisions) ** 2))
        assert mse < 0.05, (
            f"Steady-state MSE too large after cycle-slip correction: {mse:.4f}"
        )

    def test_block_lms_cycle_slip_regression_warmstart(self, xp):
        """The slip regression buffer is carried in the state and restored."""
        samples_np, syms_np = _wiener_qam16_block(n_sym=2048)
        half = 1024
        kw = dict(
            num_taps=11,
            sps=1,
            step_size=5e-4,
            constellation=Constellation.qam(16),
            cpr=BPS(test_phases=32, block_size=16, cycle_slip=CycleSlip()),
        )

        r1 = block_lms(xp.asarray(samples_np[:half]), xp.asarray(syms_np[:50]), **kw)
        assert r1.state.carrier.cs_buf_y is not None
        assert r1.state.carrier.cs_buf_n is not None

        r2 = block_lms(xp.asarray(samples_np[half:]), None, **kw, state=r1.state)
        assert r2.state.carrier.cs_buf_y is not None
        assert bool(xp.all(xp.isfinite(xp.asarray(r2.y_hat))))


class TestBlockLMSEdgeCases:
    """Degenerate inputs, non-multiple block sizes, error handling, and Signal containers."""

    def test_non_multiple_block_size(self, xp):
        samples, syms = _qam16(n_sym=1000, sps=2)
        r = block_lms(
            xp.asarray(samples),
            xp.asarray(syms),
            num_taps=11,
            sps=2,
            constellation=Constellation.qam(16),
            block_size=128,
        )
        assert r.y_hat.shape == (1000,)
        assert r.error.shape == (1000,)

    def test_pll_raises(self, xp):
        samples, syms = _qam16(n_sym=512, sps=2)
        with pytest.raises(TypeError, match="BPS"):
            block_lms(
                xp.asarray(samples),
                xp.asarray(syms),
                num_taps=11,
                sps=2,
                constellation=Constellation.qam(16),
                cpr=PLL(),
            )

    def test_single_tap(self, xp):
        """num_taps=1: degenerate equalizer - must not crash, output shape correct."""
        rng = np.random.default_rng(42)
        n_sym = 64
        const = Constellation.qam(4).points.astype(np.complex64)
        const = normalize(const, mode="average_power").astype(np.complex64)
        syms = xp.asarray(const[rng.integers(0, 4, n_sym)].astype(np.complex64))
        noise = 0.05 * xp.asarray(
            (rng.standard_normal(n_sym) + 1j * rng.standard_normal(n_sym)).astype(
                np.complex64
            )
        )
        samples = syms + noise

        r = block_lms(
            samples,
            syms,
            num_taps=1,
            sps=1,
            step_size=1e-2,
            constellation=Constellation.qam(4),
            block_size=16,
        )
        assert r.y_hat.shape == (n_sym,)

    def test_block_lms_signal_input(self, xp, xpt):
        """Signal input: sps is taken from the signal, y_hat becomes a Signal."""
        samples_np, syms_np = _qam16(n_sym=2048, sps=2)
        samples, syms = xp.asarray(samples_np), xp.asarray(syms_np)
        sig = Signal(samples=samples, sampling_rate=2e6, symbol_rate=1e6)
        kw = dict(num_taps=11, constellation=Constellation.qam(16), block_size=128)

        result_sig = block_lms(sig, syms[:200], **kw)
        result_arr = block_lms(samples, syms[:200], sps=2, **kw)

        assert isinstance(result_sig.signal, Signal)
        assert result_sig.signal.sampling_rate == 1e6
        xpt.assert_allclose(result_sig.signal.samples, result_arr.y_hat)
        xpt.assert_allclose(result_sig.y_hat, result_arr.y_hat)


class TestBlockLMSCUDAGraphAndPerformance:
    """GPU-specific tests: CUDA graph capture, transfer counts, and kernel parity."""

    @pytest.mark.gpu_only
    @pytest.mark.requires_kernel
    @pytest.mark.parametrize("cs_corr", [False, True])
    @pytest.mark.parametrize("n_sym", [100_000, 100_137])
    def test_block_lms_cuda_graph_matches_eager(self, cs_corr, n_sym, xp, xpt):
        rng = np.random.default_rng(5)
        const = normalize(Constellation.qam(16).points, mode="average_power").astype(
            np.complex64
        )
        syms = const[rng.integers(0, 16, n_sym)]
        phase = np.cumsum(rng.normal(0.0, 0.01, n_sym))
        samples = (syms * np.exp(1j * phase)).astype(np.complex64)
        noise = np.sqrt(10 ** (-25.0 / 10) / 2)
        samples += noise * (
            rng.standard_normal(n_sym) + 1j * rng.standard_normal(n_sym)
        ).astype(np.complex64)

        kw = dict(
            num_taps=21,
            sps=1,
            step_size=5e-4,
            block_size=256,
            constellation=Constellation.qam(16),
            cpr=BPS(cycle_slip=CycleSlip() if cs_corr else None),
        )
        x = xp.asarray(samples)
        t = xp.asarray(syms[:512])

        r_graph = block_lms(x, t, **kw, cuda_graph=True)
        r_eager = block_lms(x, t, **kw, cuda_graph=False)

        xpt.assert_array_equal(
            to_numpy(r_graph.y_hat),
            to_numpy(r_eager.y_hat),
        )
        xpt.assert_array_equal(
            to_numpy(r_graph.phase_trajectory),
            to_numpy(r_eager.phase_trajectory),
        )
        xpt.assert_array_equal(
            to_numpy(r_graph.weights),
            to_numpy(r_eager.weights),
        )

    @pytest.mark.gpu_only
    @pytest.mark.parametrize("cs_corr", [False, True])
    def test_block_lms_bps_loop_transfer_count_constant(self, cs_corr, xp, monkeypatch):
        if cs_corr:
            from commkit import _cuda

            if _cuda.get_kernel("cs_block") is None:
                pytest.skip("cs_block CUDA kernel unavailable - fallback transfers")

        import commkit.equalization._block._dd as eqmod

        real_to_device = eqmod.to_device
        counts = {"n": 0}

        def spy(data, device):
            counts["n"] += 1
            return real_to_device(data, device)

        monkeypatch.setattr(eqmod, "to_device", spy)

        def run(n_sym):
            samples_np, syms_np = _wiener_qam16_trackable(n_sym)
            counts["n"] = 0
            block_lms(
                xp.asarray(samples_np),
                xp.asarray(syms_np[:128]),
                num_taps=11,
                sps=1,
                step_size=5e-4,
                block_size=128,
                constellation=Constellation.qam(16),
                cpr=BPS(
                    test_phases=32,
                    block_size=16,
                    cycle_slip=CycleSlip() if cs_corr else None,
                ),
            )
            return counts["n"]

        n_small = run(512)
        n_large = run(2048)
        assert n_small == n_large, (
            f"to_device call count scales with block count: "
            f"{n_small} (4 blocks) vs {n_large} (16 blocks)"
        )

    @pytest.mark.gpu_only
    @pytest.mark.requires_kernel("cs_block")
    def test_block_lms_cycle_slip_kernel_matches_cpu_fallback(
        self, xp, xpt, monkeypatch
    ):
        from commkit import _cuda

        samples_np, syms_np = _wiener_qam16_trackable(2048, sigma_phi=0.02)
        kw = dict(
            num_taps=11,
            sps=1,
            step_size=5e-4,
            block_size=128,
            constellation=Constellation.qam(16),
            cpr=BPS(test_phases=32, block_size=16, cycle_slip=CycleSlip()),
        )

        r_kernel = block_lms(xp.asarray(samples_np), xp.asarray(syms_np[:128]), **kw)

        real_get_kernel = _cuda.get_kernel

        def no_cs_kernel(name, **spec):
            if name == "cs_block":
                return None
            return real_get_kernel(name, **spec)

        monkeypatch.setattr(_cuda, "get_kernel", no_cs_kernel)
        r_fallback = block_lms(xp.asarray(samples_np), xp.asarray(syms_np[:128]), **kw)

        xpt.assert_allclose(
            to_numpy(r_kernel.phase_trajectory),
            to_numpy(r_fallback.phase_trajectory),
            rtol=1e-6,
            atol=1e-6,
        )
        xpt.assert_allclose(
            to_numpy(r_kernel.y_hat),
            to_numpy(r_fallback.y_hat),
            rtol=1e-5,
            atol=1e-6,
        )
        for attr in ("cs_buf_y", "cs_buf_ptr", "cs_buf_n", "cs_stats"):
            xpt.assert_allclose(
                getattr(r_kernel.state.carrier, attr),
                getattr(r_fallback.state.carrier, attr),
                rtol=1e-8,
                atol=1e-8,
                err_msg=attr,
            )
