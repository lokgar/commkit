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
from commkit.equalization._block._engine import _Block
from commkit.mapping import Constellation
from commkit.math import normalize
from commkit.recovery import BPS, PLL, CycleSlip
from tests.common import scoring
from tests.common.conversions import to_numpy
from tests.common.signals import (
    make_channel_signal,
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


def _isi_record(
    constellation, num_symbols, *, seed=0, theta=None, num_channels=1, xp=None
):
    """RRC record at 2 sps through a 4-tap complex ISI channel and a
    0.3-sample delay (and a rotation for two channels), Es/N0 22 dB."""
    return make_channel_signal(
        constellation,
        num_symbols,
        isi_taps=[0.25, 1.0, 0.3j, -0.1],
        frac_delay=0.3,
        theta=theta,
        esn0_db=22.0,
        num_channels=num_channels,
        seed=seed,
        xp=xp,
    )


# -----------------------------------------------------------------------------
# Test Classes
# -----------------------------------------------------------------------------


class TestBlockLMSLayout:
    """EqualizerResult's documented layout for block LMS, also when the
    record is not a whole number of blocks: ``y_hat`` and ``error`` one
    value per symbol, ``weights`` ``(num_taps,)`` or ``(C, C, num_taps)``,
    ``phase_trajectory`` only with CPR, ``weights_history`` only with
    ``store_weights=True``."""

    @pytest.mark.parametrize("num_channels", [1, 2])
    @pytest.mark.parametrize("cpr", [None, BPS(test_phases=16, block_size=32)],
                             ids=["no-cpr", "bps"])  # fmt: skip
    def test_layout(self, xp, num_channels, cpr):
        n, taps = 1000, 11  # not a multiple of block_size
        if num_channels == 1:
            samples, syms = make_test_qam_samples(
                order=16, num_symbols=n, sps=2, seed=1
            )
        else:
            samples, syms = make_test_mimo_samples(num_channels=2, order=16,
                                                   num_symbols=n, sps=2, seed=1)  # fmt: skip
        kw = dict(num_taps=taps, sps=2, constellation=Constellation.qam(16),
                  block_size=128, cpr=cpr)  # fmt: skip
        r = block_lms(xp.asarray(samples), xp.asarray(syms), **kw)
        lead = () if num_channels == 1 else (num_channels,)
        w = (taps,) if num_channels == 1 else (num_channels, num_channels, taps)
        assert r.y_hat.shape == r.error.shape == (*lead, n)
        assert r.weights.shape == w
        if cpr is None:
            assert r.phase_trajectory is None
        else:
            assert r.phase_trajectory.shape == (*lead, n)
        assert r.weights_history is None
        stored = block_lms(xp.asarray(samples), xp.asarray(syms),
                           store_weights=True, **kw)  # fmt: skip
        assert stored.weights_history.shape == (n, *w)


class TestBlockLMSConvergence:
    """Convergence characteristics under gradient descent, AWGN, and ISI."""

    def test_identity_channel_convergence(self, xp):
        """1024 training symbols, then decisions, on a 4-tap complex ISI
        channel with a 0.3-sample delay (16-QAM, 2 sps, Es/N0 22 dB): the
        held-out second half has no symbol errors and an EVM within 1.4x the
        AWGN floor (0.079) at the default step; frozen taps keep the ISI."""
        c = Constellation.qam(16)
        rx = _isi_record(c, 8192, xp=xp)
        ref = to_numpy(rx.reference.symbols)
        kw = dict(num_taps=21, block_size=128)
        y = to_numpy(block_lms(rx, ref[:1024], **kw).y_hat)[4096:]
        assert scoring.ser(y, ref[4096:], c) == 0.0
        assert scoring.evm_rms(y, ref[4096:]) < 1.4 * 10 ** (-22 / 20)
        frozen = to_numpy(block_lms(rx, ref[:1024], step_size=0.0, **kw).y_hat)
        assert scoring.ser(frozen[4096:], ref[4096:], c) > 0.3

    def test_mimo_convergence(self, xp):
        """2x2 butterfly: two independent 16-QAM streams through the ISI
        channel and a 30-degree polarization rotation.  After 1024 training
        symbols each output carries its own stream without errors over the
        held-out half; frozen taps leave the streams mixed (SER ~0.7)."""
        c = Constellation.qam(16)
        rx = _isi_record(c, 8192, theta=np.deg2rad(30.0), num_channels=2, seed=1, xp=xp)
        ref = to_numpy(rx.reference.symbols)
        kw = dict(num_taps=21, block_size=128)
        y = to_numpy(block_lms(rx, ref[:, :1024], step_size=2e-3, **kw).y_hat)
        np.testing.assert_array_equal(scoring.ser(y[:, 4096:], ref[:, 4096:], c), 0.0)
        frozen = to_numpy(block_lms(rx, ref[:, :1024], step_size=0.0, **kw).y_hat)
        assert np.all(scoring.ser(frozen[:, 4096:], ref[:, 4096:], c) > 0.3)

    @pytest.mark.parametrize("block_size", [256, 2048])
    def test_default_step_converges_under_polarization_mixing(
        self, xp, xpt, block_size
    ):
        """The default step, min(0.5/block_size, 8e-3), untangles a 30-degree
        polarization rotation within four blocks at small and large blocks."""
        n_sym = 8 * block_size
        samples, training = make_test_mimo_samples(
            num_channels=2, order=16, num_symbols=n_sym, sps=2, snr_db=25.0, seed=7
        )
        th = np.deg2rad(30.0)
        rot = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
        mixed = xp.asarray((rot @ samples).astype(np.complex64))
        kw = dict(
            num_taps=21,
            sps=2,
            constellation=Constellation.qam(16),
            block_size=block_size,
        )
        r = block_lms(mixed, xp.asarray(training), **kw)
        r_explicit = block_lms(
            mixed,
            xp.asarray(training),
            step_size=min(0.5 / block_size, 8e-3),
            **kw,
        )
        xpt.assert_array_equal(r.y_hat, r_explicit.y_hat)

        t = xp.asarray(training[:, 4 * block_size :])
        y = r.y_hat[:, 4 * block_size :]
        evm = float(xp.sqrt(xp.mean(xp.abs(y - t) ** 2) / xp.mean(xp.abs(t) ** 2)))
        assert evm < 0.1, f"EVM {evm:.3f}: default step did not converge"

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
        """Taps converged on one record of a channel start a new record of
        the same channel near its optimum: the first two blocks' MSE against
        the reference is under a quarter of a cold start's (measured 0.010
        vs 0.116)."""
        c = Constellation.qam(16)
        kw = dict(num_taps=21, step_size=1e-3, block_size=128)
        first = _isi_record(c, 8192, seed=0, xp=xp)
        taps = block_lms(first, first.reference.symbols, **kw).weights
        second = _isi_record(c, 8192, seed=5, xp=xp)
        ref = to_numpy(second.reference.symbols)

        def head_mse(res):
            return float(np.mean(np.abs(to_numpy(res.y_hat)[:256] - ref[:256]) ** 2))

        cold = head_mse(block_lms(second, ref, **kw))
        warm = head_mse(block_lms(second, ref, **kw, initial_taps=taps))
        assert warm < cold / 4, (warm, cold)

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
        """QPSK under a random-walk phase (0.03 rad per symbol): block_lms
        with BPS keeps the output on the transmitted symbols after 512
        training symbols (no errors in the second half), without CPR the
        phase walks away (SER above 0.3)."""
        rng = np.random.default_rng(99)
        n_sym, sps = 4096, 2
        c = Constellation.psk(4)
        syms = c.points.astype(np.complex64)[rng.integers(0, 4, n_sym)]
        phase = np.cumsum(0.03 * rng.standard_normal(n_sym))
        noise = 0.1 * (
            rng.standard_normal(2 * n_sym) + 1j * rng.standard_normal(2 * n_sym)
        )
        x = xp.asarray(
            (np.repeat(syms * np.exp(1j * phase), sps) + noise).astype(np.complex64)
        )
        kw = dict(num_taps=7, sps=sps, step_size=5e-4, block_size=128, constellation=c)
        bps = block_lms(x, syms[:512], **kw, cpr=BPS(test_phases=32, block_size=32))
        none = block_lms(x, syms[:512], **kw)
        h = n_sym // 2
        assert scoring.ser(to_numpy(bps.y_hat)[h:], syms[h:], c) == 0.0
        assert scoring.ser(to_numpy(none.y_hat)[h:], syms[h:], c) > 0.3

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
        """Inline BPS slips on a smooth Wiener phase at low SNR (16-QAM,
        2πΔνT = 2π·1e-4, Es/N0 12 dB); the repair removes most of them.

        Slips are counted against the transmitted symbols: changes of the
        output's π/2 rotation between 64-symbol blocks.  Without the repair
        the same records slip (over 40 changes on 2 x 2e4 symbols), so the
        test fails if the repair does nothing.
        """
        c = Constellation.qam(16)
        counts = {True: 0, False: 0}
        for seed in range(2):
            rng = np.random.default_rng(seed)
            n = 20000
            syms = c.points.astype(np.complex64)[rng.integers(0, 16, n)]
            phase = np.cumsum(rng.standard_normal(n) * np.sqrt(2 * np.pi * 1e-4))
            noise = np.sqrt(10**-1.2 / 2) * (
                rng.standard_normal(n) + 1j * rng.standard_normal(n)
            )
            x = xp.asarray((syms * np.exp(1j * phase) + noise).astype(np.complex64))
            for repair in counts:
                res = block_lms(
                    x,
                    syms[:256],
                    num_taps=1,
                    sps=1,
                    step_size=0.0,
                    block_size=128,
                    constellation=c,
                    cpr=BPS(
                        test_phases=32,
                        block_size=16,
                        cycle_slip=CycleSlip() if repair else None,
                    ),
                )
                k = scoring.block_rotations(
                    to_numpy(res.y_hat)[256:], syms[256:], symmetry=4, block=64
                )
                counts[repair] += int(np.count_nonzero(np.diff(k)))
        assert counts[False] > 40, counts
        assert counts[True] < counts[False] / 4, counts

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

    def test_host_training_follows_the_samples(self, xp, xpt):
        """Host training symbols with device samples, like lms and rls."""
        samples, syms = _qam16(n_sym=1000, sps=2)
        kw = dict(num_taps=11, sps=2, constellation=Constellation.qam(16))
        r_host = block_lms(xp.asarray(samples), syms[:300], **kw)
        r_dev = block_lms(xp.asarray(samples), xp.asarray(syms[:300]), **kw)
        xpt.assert_array_equal(r_host.y_hat, r_dev.y_hat)

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


class TestBlockCopies:
    """The per-block copies at the block index (on the device for CuPy, so a
    captured graph replays them at the right offset) equal plain slicing."""

    def _run(self, xp, C=2, n_sym=10, block_size=4, sps=2, num_taps=3):
        rng = np.random.default_rng(0)
        n_pad = n_sym * sps + 1  # the last window runs past the data
        x = rng.standard_normal((C, n_pad)) + 1j * rng.standard_normal((C, n_pad))
        return _Block(
            xp=xp, was_1d=False, C=C, n_sym=n_sym, sps=sps, num_taps=num_taps,
            block_size=block_size, fftsize=16,
            x_padded=xp.asarray(x.astype(np.complex64)), lead=0, real_end=n_pad,
            resume=0, training=None, eq_norm=None, h=None,
            x_win=xp.full((C, 16), 7.0, dtype=xp.complex64),
            block_dev=None if xp is np else xp.zeros(1, dtype=xp.int64),
        )  # fmt: skip

    def test_window_take_put_follow_the_block_index(self, xp, xpt):
        run = self._run(xp)
        x = to_numpy(run.x_padded)
        src = xp.asarray(np.arange(2 * 10, dtype=np.float32).reshape(2, 10))
        dst = xp.zeros((2, 10), dtype=xp.float32)
        for b, B in ((0, 4), (1, 4), (2, 2)):
            run.block = b
            run.fill_window(B)
            start, need = b * 4 * 2, B * 2 + 2
            expected = np.zeros((2, 16), np.complex64)
            avail = min(need, x.shape[1] - start)
            expected[:, :avail] = x[:, start : start + avail]
            xpt.assert_array_equal(run.x_win, xp.asarray(expected))

            staged = xp.full((2, 4), -1.0, dtype=xp.float32)
            run.take(staged, src, B)
            xpt.assert_array_equal(staged[:, :B], src[:, 4 * b : 4 * b + B])
            run.put(dst, staged, B)
            run.advance()
        xpt.assert_array_equal(dst, src)
        if run.block_dev is not None:
            assert int(run.block_dev[0]) == 3


class TestBlockLMSCUDAGraphAndPerformance:
    """GPU-specific tests: CUDA graph capture, transfer counts, and kernel parity."""

    @pytest.mark.gpu_only
    @pytest.mark.parametrize("cpr", ["bps", "bps+cs", None])
    @pytest.mark.parametrize("n_sym", [100_000, 100_137])
    @pytest.mark.parametrize("n_train", [512, 20_000, "all"])
    def test_block_lms_cuda_graph_matches_eager(self, cpr, n_sym, n_train, xp, xpt):
        """Replayed decision-directed and training blocks equal the eager
        loop bit for bit (training blocks read staged symbols)."""
        rng = np.random.default_rng(5)
        const = normalize(Constellation.qam(16).points, mode="average_power").astype(
            np.complex64
        )
        syms = const[rng.integers(0, 16, n_sym)]
        phase = np.cumsum(rng.normal(0.0, 0.01, n_sym))
        samples = (syms * np.exp(1j * phase)).astype(np.complex64)
        noise = float(np.sqrt(10 ** (-25.0 / 10) / 2))
        samples += noise * (
            rng.standard_normal(n_sym) + 1j * rng.standard_normal(n_sym)
        ).astype(np.complex64)

        kw = dict(
            num_taps=21,
            sps=1,
            step_size=5e-4,
            block_size=256,
            constellation=Constellation.qam(16),
            cpr=None
            if cpr is None
            else BPS(cycle_slip=CycleSlip() if cpr == "bps+cs" else None),
        )
        x = xp.asarray(samples)
        t = xp.asarray(syms if n_train == "all" else syms[:n_train])

        r_graph = block_lms(x, t, **kw, cuda_graph=True)
        r_eager = block_lms(x, t, **kw, cuda_graph=False)

        xpt.assert_array_equal(
            to_numpy(r_graph.y_hat),
            to_numpy(r_eager.y_hat),
        )
        if cpr is not None:
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
    def test_block_lms_cycle_slip_kernel_matches_cpu(self, xp, xpt):
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
        r_cpu = block_lms(samples_np, syms_np[:128], **kw)

        xpt.assert_allclose(
            to_numpy(r_kernel.phase_trajectory),
            to_numpy(r_cpu.phase_trajectory),
            rtol=1e-6,
            atol=1e-6,
        )
        xpt.assert_allclose(
            to_numpy(r_kernel.y_hat),
            to_numpy(r_cpu.y_hat),
            rtol=1e-5,
            atol=1e-6,
        )
        for attr in ("cs_buf_y", "cs_buf_ptr", "cs_buf_n", "cs_stats"):
            xpt.assert_allclose(
                getattr(r_kernel.state.carrier, attr),
                getattr(r_cpu.state.carrier, attr),
                rtol=1e-8,
                atol=1e-8,
                err_msg=attr,
            )
