"""Equalizers with inline carrier recovery (lms/rls/block_lms + PLL/BPS), and
blockwise frequency-offset correction.

Recovery is scored against the transmitted symbols and the true phase
(``tests.common.scoring``), never against the receiver's own decisions: a
π/2-rotated 16-QAM output lies on its own grid.
"""

import numpy as np
import pytest

from commkit.equalization import lms, rls
from commkit.frequency import (
    MthPower,
    correct_frequency_offset,
    estimate_frequency_offset,
)
from commkit.mapping import Constellation
from commkit.recovery import (
    BPS,
    PLL,
    CycleSlip,
    correct_carrier_phase,
    estimate_carrier_phase,
)
from tests.common import scoring
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
        noise_std = float(np.sqrt(10 ** (-snr_db / 10) / 2))
        rng = np.random.default_rng(seed + 1)
        samples += noise_std * (
            rng.standard_normal(n_sym) + 1j * rng.standard_normal(n_sym)
        ).astype(np.complex64)
    return samples, syms


# -----------------------------------------------------------------------------
# Test Classes
# -----------------------------------------------------------------------------


def _wiener_record(order, linewidth_ts, esn0_db, seed, n):
    """16-QAM (or square ``order``) at 1 SPS under a Wiener phase with
    increment variance ``2π Δν T`` and AWGN; returns ``(x, symbols, phase,
    constellation)``."""
    rng = np.random.default_rng(seed)
    c = Constellation.qam(order)
    syms = c.points.astype(np.complex64)[rng.integers(0, order, n)]
    phase = np.cumsum(rng.standard_normal(n) * np.sqrt(2 * np.pi * linewidth_ts))
    noise = np.sqrt(10 ** (-esn0_db / 10) / 2) * (
        rng.standard_normal(n) + 1j * rng.standard_normal(n)
    )
    return (syms * np.exp(1j * phase) + noise).astype(np.complex64), syms, phase, c


def _slips(y, syms, start, block=64):
    """Changes of the output's π/2 rotation against the reference, counted
    over blocks of ``block`` symbols after ``start``."""
    k = scoring.block_rotations(y[start:], syms[start:], symmetry=4, block=block)
    return int(np.count_nonzero(np.diff(k)))


class TestCPREqualizerBaseline:
    """An inline CPR that never moves is the equalizer without CPR."""

    @pytest.mark.parametrize("algo", ["lms", "rls"])
    def test_zero_gain_pll_is_no_cpr(self, algo, xp, xpt):
        """A PLL with zero gains and zero initial phase applies no rotation,
        so its output equals ``cpr=None``, which reports no trajectory."""
        samples, syms = _qpsk_signal(n_sym=2000)
        fn = lms if algo == "lms" else rls
        kw = dict(
            training_symbols=syms[:500],
            num_taps=11,
            sps=2,
            constellation=Constellation.psk(4),
        )
        none = fn(xp.asarray(samples), **kw, cpr=None)
        frozen = fn(xp.asarray(samples), **kw, cpr=PLL(mu=0.0, beta=0.0))
        assert none.phase_trajectory is None
        xpt.assert_array_equal(
            frozen.phase_trajectory, xp.zeros_like(frozen.phase_trajectory)
        )
        xpt.assert_allclose(frozen.y_hat, none.y_hat, rtol=1e-5, atol=1e-6)


class TestCPRPLLConvergence:
    """PLL phase tracking, cycle-slip correction, and gain parameterization."""

    @pytest.mark.xfail(
        strict=True,
        reason="The PLL's CycleSlip repair adds slips instead of removing them: "
        "16-QAM, 2πΔνT = 2π·1e-4, Es/N0 16 dB, B·T = 2e-2, 4 records of 2e4 "
        "symbols: 10 rotation changes against the reference with "
        "CycleSlip(), 3 without (inline and standalone alike).  A PLL slip "
        "is a drift over ~1/B symbols, which the linear predictor follows.",
    )
    def test_cycle_slip_repair_reduces_pll_slips(self, xp):
        """With the repair the output changes its π/2 rotation against the
        transmitted symbols less often than without it."""
        counts = {True: 0, False: 0}
        for seed in range(4):
            x, syms, _, c = _wiener_record(16, 1e-4, 16.0, seed, 20000)
            for repair in counts:
                res = lms(
                    xp.asarray(x),
                    syms[:256],
                    num_taps=1,
                    sps=1,
                    step_size=0.0,
                    constellation=c,
                    cpr=PLL(
                        normalized_bandwidth=2e-2,
                        cycle_slip=CycleSlip() if repair else None,
                    ),
                )
                counts[repair] += _slips(to_numpy(res.y_hat), syms, 256)
        assert counts[True] < counts[False], counts

    def test_pll_phase_noise_tracking(self, xp):
        """LMS + PLL follows a Wiener phase (16-QAM, 2πΔνT = 2π·1e-4,
        Es/N0 25 dB) with the jitter of the loop.

        With ``mu = 4B`` the loop error is about first order, ``σ² ≈ σ_Δ²/(2μ)
        + μ N0/4`` (σ_Δ² = 2πΔνT; the detector gain |d|² averages 1): at
        B = 2e-2 that is 0.064 rad.  No symbol errors after training (the
        AWGN-only SER at 25 dB is ~1e-6).  The same record with B = 5e-3,
        whose phase-noise term alone is 0.13 rad, slips (baseline).
        """
        x, syms, phase, c = _wiener_record(16, 1e-4, 25.0, 7, 5000)
        sigma_d2, n0 = 2 * np.pi * 1e-4, 10 ** (-2.5)

        def run(bw):
            res = lms(
                xp.asarray(x),
                syms[:1000],
                num_taps=1,
                sps=1,
                constellation=c,
                cpr=PLL(normalized_bandwidth=bw),
            )
            return to_numpy(res.y_hat), to_numpy(res.phase_trajectory)

        y, phi = run(2e-2)
        mu = 4 * 2e-2
        predicted = np.sqrt(sigma_d2 / (2 * mu) + mu * n0 / 4)
        assert scoring.ser(y[1500:], syms[1500:], c) == 0.0
        assert scoring.phase_rms(phi[1500:], phase[1500:]) < 1.5 * predicted

        y_narrow, _ = run(5e-3)
        assert scoring.ser(y_narrow[1500:], syms[1500:], c) > 0.1

    def test_inline_raw_gains_match_bandwidth(self, xp):
        """cpr_pll_mu/beta set to bandwidth-equivalent gains reproduces bandwidth path."""
        samples, syms = _qpsk_signal(n_sym=1500)
        samples = (samples * np.exp(1j * 0.2)).astype(np.complex64)
        bw = 5e-3
        kw = dict(
            training_symbols=syms[:300],
            num_taps=11,
            sps=2,
            constellation=Constellation.psk(4),
        )
        res_bw = lms(xp.asarray(samples), **kw, cpr=PLL(normalized_bandwidth=bw))
        res_raw = lms(
            xp.asarray(samples),
            **kw,
            cpr=PLL(
                mu=float(np.float32(4.0 * bw)), beta=float(np.float32(4.0 * bw**2))
            ),
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
                syms[:100],
                num_taps=11,
                sps=2,
                constellation=Constellation.psk(4),
                cpr=PLL(beta=1e-3),
            )

    def test_inline_pll_phase_init_seeds_cold_start(self, xp):
        """PLL.phase_init is the first applied phase of a cold start."""
        samples, syms = _qpsk_signal(n_sym=400)
        res = lms(
            xp.asarray(samples),
            syms[:100],
            num_taps=11,
            sps=2,
            constellation=Constellation.psk(4),
            cpr=PLL(phase_init=0.3),
        )
        assert float(res.phase_trajectory[0]) == pytest.approx(0.3)

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
            syms[:200],
            num_taps=1,
            sps=1,
            step_size=0.0,
            initial_taps=xp.asarray(np.array([1.0 + 0j], dtype=np.complex64)),
            constellation=Constellation.psk(4),
            cpr=PLL(mu=m, beta=b),
        )
        phi_inline = to_numpy(res.phase_trajectory)
        phi_std = to_numpy(
            estimate_carrier_phase(
                xp.asarray(samples),
                PLL(mu=m, beta=b),
                constellation=Constellation.psk(4),
            ).value
        )

        tail = slice(n_sym // 4, n_sym)
        diff = np.unwrap(phi_inline[tail]) - np.unwrap(phi_std[tail])
        assert np.std(diff) < 1e-3, f"inline vs standalone phase std {np.std(diff):.2e}"


class TestCPRBPSConvergence:
    """Blind Phase Search unwrapping, convergence, and block sizing."""

    def test_bps_phase_unwrap(self, xp):
        """A 0 -> 3 rad ramp (beyond the π/2 search interval): the inline BPS
        trajectory, anchored by training, follows the ramp itself within
        0.05 rad RMS (measured 0.018) with no symbol errors; the tap is
        frozen so the trajectory carries the whole rotation."""
        rng = np.random.default_rng(5)
        n_sym = 3000
        c = Constellation.qam(16)
        syms = c.points.astype(np.complex64)[rng.integers(0, 16, n_sym)]
        phase_true = np.linspace(0.0, 3.0, n_sym)
        noise = np.sqrt(10**-2.5 / 2) * (
            rng.standard_normal(n_sym) + 1j * rng.standard_normal(n_sym)
        )
        x = xp.asarray((syms * np.exp(1j * phase_true) + noise).astype(np.complex64))
        res = lms(
            x,
            syms[:500],
            num_taps=1,
            sps=1,
            step_size=0.0,
            constellation=c,
            cpr=BPS(test_phases=64, block_size=32),
        )
        phi = to_numpy(res.phase_trajectory)
        assert scoring.phase_rms(phi[500:], phase_true[500:]) < 0.05
        assert scoring.ser(to_numpy(res.y_hat)[500:], syms[500:], c) == 0.0

    @pytest.mark.parametrize("algo", ["lms", "rls", "block_lms"])
    @pytest.mark.parametrize(
        "constellation",
        [Constellation.psk(8), Constellation.psk(2)],
        ids=["8psk", "bpsk"],
    )
    def test_bps_searches_the_constellation_symmetry(self, algo, constellation, xp):
        """Inline BPS on 8-PSK (π/4 symmetry) and BPSK (π) tracks a ramp
        itself, without symbol errors: the candidates span ``2π/S``, not
        ``π/2``."""
        from commkit.equalization import block_lms

        rng = np.random.default_rng(11)
        n_sym, n_train = 4000, 500
        points = constellation.points.astype(np.complex64)
        syms = points[rng.integers(0, points.size, n_sym)]
        phase = 0.2 + 1.5e-3 * np.arange(n_sym)  # inside the first branch, ±π/S
        awgn = 0.05 * (rng.standard_normal(n_sym) + 1j * rng.standard_normal(n_sym))
        samples = (syms * np.exp(1j * phase) + awgn).astype(np.complex64)
        cpr = BPS(test_phases=32, block_size=16, cycle_slip=CycleSlip())
        kw = dict(num_taps=1, sps=1, constellation=constellation, cpr=cpr)
        if algo == "lms":
            res = lms(xp.asarray(samples), syms[:n_train], step_size=1e-4, **kw)
        elif algo == "rls":
            res = rls(
                xp.asarray(samples), syms[:n_train], forgetting_factor=0.9999, **kw
            )
        else:
            res = block_lms(
                xp.asarray(samples),
                syms[:n_train],
                step_size=1e-4,
                block_size=32,
                **kw,
            )
        y = to_numpy(res.y_hat)[n_train:]
        ref = syms[n_train : n_train + y.size]
        decided = points[np.argmin(np.abs(y[:, None] - points[None, :]), axis=1)]
        assert np.count_nonzero(decided != ref) == 0
        # Slow taps: the CPR, not the taps, follows the ramp (modulo 2π/S).
        S = constellation.rotational_symmetry
        phi = to_numpy(res.phase_trajectory)[n_train:]
        residual = np.angle(
            np.exp(1j * S * (phase[n_train : n_train + phi.size] - phi))
        )
        assert np.std(residual / S) < 0.05

    def test_bps_phase_noise_tracking(self, xp):
        """LMS + BPS follows a Wiener phase (16-QAM, 2πΔνT = 2π·5e-5,
        Es/N0 25 dB): no symbol errors against the transmitted symbols after
        training, and the trajectory within 0.1 rad RMS of the true phase
        (BPS's 64 candidates over π/2 quantize to 0.007 rad; a 32-symbol
        window at 25 dB averages the noise well below 0.05 rad).  The tap is
        frozen so that the trajectory carries the whole rotation (an adapting
        tap takes a share of it).  Without CPR the same equalizer loses the
        phase (baseline)."""
        x, syms, phase, c = _wiener_record(16, 5e-5, 25.0, 11, 5000)
        kw = dict(num_taps=1, sps=1, step_size=0.0, constellation=c)
        res = lms(
            xp.asarray(x), syms[:1000], **kw, cpr=BPS(test_phases=64, block_size=32)
        )
        none = lms(xp.asarray(x), syms[:1000], **kw)
        assert scoring.ser(to_numpy(res.y_hat)[1000:], syms[1000:], c) == 0.0
        assert (
            scoring.phase_rms(to_numpy(res.phase_trajectory)[1000:], phase[1000:]) < 0.1
        )
        assert scoring.ser(to_numpy(none.y_hat)[1000:], syms[1000:], c) > 0.1

    @pytest.mark.parametrize("block_size", [8, 32])
    def test_bps_block_size_convergence(self, xp, block_size):
        """LMS + BPS with an 8- and a 32-symbol window tracks a Wiener phase
        without symbol errors against the reference.  (A one-symbol window
        fits many rotations of 16-QAM and slips: SER 0.83 on this record.)"""
        x, syms, _, c = _wiener_record(16, 5e-5, 25.0, 13, 4000)
        res = lms(
            xp.asarray(x),
            syms[:1000],
            num_taps=1,
            sps=1,
            constellation=c,
            cpr=BPS(test_phases=32, block_size=block_size),
        )
        assert scoring.ser(to_numpy(res.y_hat)[1000:], syms[1000:], c) == 0.0

    def test_rls_bps_convergence(self, xp):
        """RLS + BPS tracks a Wiener phase: no symbol errors against the
        reference after training (trajectory length is the record's)."""
        x, syms, _, c = _wiener_record(16, 5e-5, 25.0, 17, 3000)
        res = rls(
            xp.asarray(x),
            syms[:500],
            num_taps=1,
            sps=1,
            constellation=c,
            cpr=BPS(test_phases=64, block_size=32),
        )
        assert res.phase_trajectory.shape == (3000,)
        y = to_numpy(res.y_hat)
        assert scoring.ser(y[500:], syms[500 : y.size], c) == 0.0


class TestBlockSlipCarry:
    """A slip correction carries into the block unwrap state."""

    def test_trajectory_does_not_depend_on_the_block_size(self, xp):
        """With frozen taps the CPR must not depend on where the blocks are
        cut.  A fast 8-PSK phase step makes the slip corrector fire; its
        correction carried times 4 instead of times S = 8 left every later
        block start half a slip quantum (pi/8) away."""
        from commkit.equalization import block_lms

        c = Constellation.psk(8)
        rng = np.random.default_rng(4)
        n = 1536
        syms = c.points[rng.integers(0, 8, n)].astype(np.complex64)
        phase = np.zeros(n)
        phase[1000:1004] = np.linspace(0.0, np.pi / 4, 4)
        phase[1004:] = np.pi / 4
        noise = 0.01 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
        x = xp.asarray((syms * np.exp(1j * phase) + noise).astype(np.complex64))
        cpr = BPS(test_phases=32, block_size=2, cycle_slip=CycleSlip(threshold=0.3))

        def trajectory(block_size):
            res = block_lms(
                x,
                syms[:200],
                num_taps=1,
                sps=1,
                step_size=0.0,
                block_size=block_size,
                constellation=c,
                cpr=cpr,
            )
            return to_numpy(res.phase_trajectory).ravel()

        np.testing.assert_allclose(trajectory(64), trajectory(n), atol=1e-6)


class TestBPSTrainingAnchor:
    """Training symbols anchor the inline BPS phase."""

    @pytest.mark.parametrize("algo", ["lms", "rls", "block_lms"])
    @pytest.mark.parametrize("seed", [1, 5])
    def test_multi_tap_receiver_locks_without_rotation(self, algo, seed, xp):
        """A matched-filtered 16-QAM record at 2 sps through a 21-tap
        equalizer: the output matches the reference without a rotation and
        with the error rate of the noise alone.

        Blind BPS during training started from a few-symbol window that fits
        any rotation; the training error copied that phase into the taps and
        nothing fixed the absolute phase, so the output wandered and slipped
        (tail EVM above 100 %).  The data-aided estimate on the training
        symbols pins it.
        """
        import commkit as ck
        from commkit.equalization import block_lms

        c = Constellation.qam(16)
        tx = ck.generate(
            c, 8192, symbol_rate=32e9, sps=2, pulse=ck.RRC(rolloff=0.1), rng=seed
        )
        rx = ck.impairments.apply_awgn(tx, esn0_db=18, rng=3)
        rx = ck.filtering.matched_filter(rx).to("gpu" if xp is not np else "cpu")
        ref = to_numpy(tx.reference.symbols)
        kw = dict(num_taps=21, cpr=BPS(test_phases=64))
        train = ref[:2000]
        if algo == "lms":
            res = lms(
                rx, num_taps=21, step_size=1e-3, cpr=kw["cpr"], training_symbols=train
            )
        elif algo == "rls":
            res = rls(rx, forgetting_factor=0.999, training_symbols=train, **kw)
        else:
            res = block_lms(
                rx, step_size=1e-3, block_size=32, training_symbols=train, **kw
            )
        y = to_numpy(res.y_hat)[-3000:]
        r = ref[: to_numpy(res.y_hat).size][-3000:]
        assert abs(np.angle(np.vdot(r, y))) < 0.05

        def nearest(v):
            return np.argmin(np.abs(v[:, None] - c.points), axis=1)

        assert np.count_nonzero(nearest(y) != nearest(r)) / r.size < 5e-3


class TestCPRMIMOJoint:
    """Multi-channel MIMO CPR and joint carrier phase tracking."""

    def test_mimo_lms_pll(self, xp):
        """2x2 LMS + PLL with a different static phase per channel (0.3 and
        1.1 rad, beyond π/4 for the second): each output matches its own
        transmitted stream, with no rotation, after training."""
        rng = np.random.default_rng(99)
        n_sym = 3000
        c = Constellation.qam(16)
        pts = c.points.astype(np.complex64)
        syms = pts[rng.integers(0, 16, (2, n_sym))]
        rot = np.exp(1j * np.array([0.3, 1.1]))[:, None]
        noise = np.sqrt(10**-2.5 / 2) * (
            rng.standard_normal((2, n_sym)) + 1j * rng.standard_normal((2, n_sym))
        )
        samples = xp.asarray((syms * rot + noise).astype(np.complex64))
        res = lms(
            samples,
            syms[:, :500],
            num_taps=1,
            sps=1,
            constellation=c,
            cpr=PLL(normalized_bandwidth=5e-3),
        )
        assert res.phase_trajectory.shape == (2, n_sym)
        ser = scoring.ser(to_numpy(res.y_hat)[:, 500:], syms[:, 500:], c)
        np.testing.assert_array_equal(ser, [0.0, 0.0])

    def test_pll_joint_channels(self, xp):
        """joint_channels=True makes both PLL integrators identical (shared LO)."""
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
            training,
            num_taps=1,
            sps=1,
            constellation=Constellation.qam(16),
            cpr=PLL(normalized_bandwidth=5e-3, joint_channels=True),
        )

        assert res.phase_trajectory is not None
        assert res.phase_trajectory.shape == (2, n_sym)
        phi0 = xp.asarray(res.phase_trajectory[0])
        phi1 = xp.asarray(res.phase_trajectory[1])
        assert bool(xp.all(phi0 == phi1)), (
            "joint_channels=True: PLL integrators must be identical"
        )


class TestStatePersistence:
    """Continuation through state=: CPR resumes without a re-lock transient."""

    @pytest.mark.parametrize("cpr_mode", ["pll", "bps"])
    def test_state_warmstart_lms(self, cpr_mode, xp):
        """A continued call is not worse than a cold restart with the same taps."""
        n_sym = 4000
        half = n_sym // 2
        samples_np, syms_np = _wiener_phase_signal(n_sym=n_sym)
        cpr = PLL() if cpr_mode == "pll" else BPS(block_size=16, test_phases=32)
        kw = dict(
            num_taps=5,
            sps=1,
            step_size=5e-3,
            constellation=Constellation.psk(4),
            cpr=cpr,
        )

        r1 = lms(samples_np[:half], syms_np[:half], **kw)
        st = r1.state
        assert st is not None and st.carrier is not None
        assert st.cpr == cpr and st.num_channels == 1
        ov = st.overlap
        # The continued call starts at symbol half - ov; train 20 symbols.
        r2_warm = lms(
            samples_np[half:], syms_np[half - ov : half - ov + 20], **kw, state=st
        )
        r2_cold = lms(
            samples_np[half:], syms_np[half : half + 20], **kw, initial_taps=r1.weights
        )
        ref = syms_np[half + 20 : half + 50]
        mse_warm = calc_mse_db(r2_warm.y_hat[ov + 20 : ov + 50], ref)
        mse_cold = calc_mse_db(r2_cold.y_hat[20:50], ref)
        assert mse_warm < mse_cold + 3.0, (
            f"Warm state should not be worse than cold by >3 dB: "
            f"warm={mse_warm:.1f} dB  cold={mse_cold:.1f} dB"
        )

    def test_state_warmstart_rls(self, xp):
        """The RLS state carries P and the PLL state, and continues."""
        n_sym = 2000
        half = n_sym // 2
        samples_np, syms_np = _wiener_phase_signal(n_sym=n_sym)
        kw = dict(num_taps=5, sps=1, constellation=Constellation.psk(4), cpr=PLL())

        r1 = rls(samples_np[:half], syms_np[:half], **kw)
        st = r1.state
        assert st.inverse_correlation is not None
        assert st.inverse_correlation.dtype == np.complex128
        assert st.inverse_correlation.shape == (5, 5)
        assert st.carrier is not None

        r2 = rls(samples_np[half:], None, **kw, state=st)
        assert r2.state.equalizer == "rls"
        assert r2.y_hat.shape[-1] > 0


class TestBlockwiseFOE:
    """Frequency offset estimation and blockwise correction."""

    def test_blockwise_foe_chirp(self, xp):
        """A carrier chirping from 1 to 5 MHz (noiseless 16-QAM, 2 sps at
        1 GS/s): the blockwise M-th power correction leaves a residual whose
        frequency stays under 1 % of the 4 MHz sweep in the interior
        (measured 21 kHz), slow enough that BPS on the corrected symbols
        makes no errors (one quantum removed).  Without the correction BPS
        cannot follow the chirp (SER 0.7)."""
        rng = np.random.default_rng(3)
        fs, n, sps = 1e9, 65536, 2
        f_t = np.linspace(1e6, 5e6, n)
        carrier = np.exp(2j * np.pi * np.cumsum(f_t) / fs)
        c = Constellation.qam(16)
        syms = c.points.astype(np.complex64)[rng.integers(0, 16, n // sps)]
        base = np.repeat(syms, sps)
        samples = xp.asarray((base * carrier).astype(np.complex64))

        est = estimate_frequency_offset(
            samples,
            MthPower(block_size=4096, overlap=0.5),
            sampling_rate=fs,
            constellation=c,
        )
        corrected = to_numpy(correct_frequency_offset(samples, est, sampling_rate=fs))
        residual = np.unwrap(np.angle(corrected * np.conj(base)))
        w = 1024
        f_res = (residual[w:] - residual[:-w]) / w * fs / (2 * np.pi)
        assert np.max(np.abs(f_res[4096:-4096])) < 0.01 * 4e6

        def ser_after_bps(z):
            z = z[::sps]
            phi = estimate_carrier_phase(z, BPS(), constellation=c)
            return scoring.blind_ser(correct_carrier_phase(z, phi), syms, c)

        assert ser_after_bps(corrected) == 0.0
        assert ser_after_bps(to_numpy(samples)) > 0.3
