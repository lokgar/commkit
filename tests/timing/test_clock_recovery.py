"""Gardner clock recovery: estimate_sampling_clock_offset and
correct_sampling_clock_offset, scored against the clock offset and the
timing that apply_sampling_clock_offset and a fractional delay impose."""

import logging
import math

import numpy as np
import pytest

import commkit as ck
from commkit import RRC, Constellation
from commkit.timing import (
    Gardner,
    SamplingClockOffsetEstimate,
    correct_sampling_clock_offset,
    estimate_sampling_clock_offset,
)
from commkit.timing import clock_recovery as cr
from tests.common import scoring
from tests.common.conversions import to_numpy
from tests.common.reference_impl import gardner_reference
from tests.common.theory import ls_slope_std, raised_cosine_slope_energy

C16 = Constellation.qam(16)
DELAY = 0.37  # samples


def _received(
    xp,
    *,
    ppm,
    rolloff=0.1,
    esn0_db=20.0,
    num_symbols=2**17,
    sps=2,
    constellation=C16,
    num_channels=1,
    rng=1,
):
    tx = ck.generate(
        constellation,
        num_symbols=num_symbols,
        symbol_rate=32e9,
        sps=sps,
        pulse=RRC(rolloff=rolloff),
        num_channels=num_channels,
        rng=rng,
        device="gpu" if xp is not np else "cpu",
    )
    rx = ck.impairments.apply_awgn(tx, esn0_db=esn0_db, rng=rng + 1)
    rx = ck.timing.apply_delay(rx, delay=DELAY)
    rx = ck.impairments.apply_sampling_clock_offset(rx, offset_ppm=ppm)
    return ck.filtering.matched_filter(rx)


def _true_trajectory(num, ppm, sps=2):
    """Strobe of symbol k: input sample k·sps + DELAY, stretched by 1 + ε."""
    k = np.arange(num)
    return (k * sps + DELAY) * (1 + ppm * 1e-6) / sps - k


class TestEstimate:
    @pytest.mark.parametrize(
        ("rolloff", "method"),
        [(0.5, "classical"), (0.1, "classical"), (0.1, "power")],
    )
    @pytest.mark.parametrize("ppm", [20.0, -50.0])
    def test_offset_against_truth(self, xp, rolloff, method, ppm):
        """The offset is the least-squares slope of the trajectory over its
        last three quarters (K = 98 304 symbols); the trajectory's error
        against the true timing has rms σ (0.0026 to 0.018 symbol measured)
        and stays correlated over about the loop's memory, 1/(2·B·T) = 500
        symbols, so the slope's standard deviation is sqrt(12 σ² L / K³)
        (0.006 to 0.05 ppm).  Bound: 4 standard deviations.  Measured
        errors 0.006 to 0.06 ppm."""
        mf = _received(xp, ppm=ppm, rolloff=rolloff)
        est = estimate_sampling_clock_offset(mf, Gardner(method=method))
        traj = to_numpy(est.trajectory)
        err = traj - _true_trajectory(traj.size, ppm)
        k = traj.size - traj.size // 4
        sigma = float(np.std(err[traj.size // 4 :]))
        bound = 4 * ls_slope_std(sigma, 1 / (2 * 1e-3), k) * 1e6
        assert abs(float(est.value) - ppm) < bound, (float(est.value), bound)
        assert abs(float(np.mean(err[traj.size // 4 :]))) < 0.005  # lock point

    def test_joint_loop_shares_one_trajectory(self, xp):
        """Two polarizations sampled by one clock: one loop, the same
        trajectory and offset on both channels."""
        mf = _received(xp, ppm=30.0, num_channels=2, num_symbols=2**16)
        est = estimate_sampling_clock_offset(
            mf, Gardner(method="power", joint_channels=True)
        )
        traj = to_numpy(est.trajectory)
        assert traj.shape[0] == 2
        np.testing.assert_array_equal(traj[0], traj[1])
        value = to_numpy(est.value)
        assert value.shape == (2,) and value[0] == value[1]
        assert abs(value[0] - 30.0) < 0.5

    def test_matches_oracle(self, xp):
        """The kernel and its preparation against the plain-Python loop:
        strobe positions agree to float32 rounding."""
        mf = _received(xp, ppm=200.0, num_symbols=400, num_channels=2, esn0_db=25)
        x = to_numpy(mf.samples)
        for method in ("classical", "power"):
            ref = gardner_reference(
                x,
                sps=2,
                power=method == "power",
                normalized_bandwidth=3e-3,
                num_cal=397,
            )
            pos, _, _ = cr._track(
                x, 2, Gardner(normalized_bandwidth=3e-3, method=method)
            )
            k = min(ref.shape[1], pos.shape[1])
            np.testing.assert_allclose(pos[:, :k], ref[:, :k], atol=1e-4)

    @pytest.mark.parametrize("method", ["classical", "power"])
    @pytest.mark.parametrize(
        "single_block",
        [False, pytest.param(True, marks=pytest.mark.allow_warnings)],
    )
    def test_detector_gain_under_a_clock_offset(
        self, xp, method, single_block, monkeypatch
    ):
        """K_d at 300 and 900 ppm against its value at 0 ppm (rolloff 0.1).
        Calibrating at fixed strobes over M symbols scales the fitted
        amplitude by sinc(M·ε), so the gains grow by its inverse; blocks of
        100 symbols (0.1 symbol of drift at the default 1000 ppm bound)
        summed along the measured drift keep it.  Over 5 seeds the ratio
        measured 0.988 to 1.047 (classical) and 0.986 to 1.002 (power):
        bound 10 %.  Baseline, one block of 2048 symbols: 0.41 to 0.61 at
        300 ppm and 0.08 to 0.17 at 900 ppm, as sinc(2048·ε) = 0.49 and
        0.07 predict (the inflated gains slip, hence the warnings)."""
        if single_block:
            monkeypatch.setattr(cr, "_CAL_MIN_BLOCK", 2048)
        kd = {}
        for ppm in (0.0, 300.0, 900.0):
            mf = _received(xp, ppm=ppm, num_symbols=2**15)
            est = estimate_sampling_clock_offset(mf, Gardner(method=method))
            kd[ppm] = float(est.detector_gain)
            if not single_block:
                assert abs(float(est.value) - ppm) < 1.0
        ratios = np.array([kd[300.0], kd[900.0]]) / kd[0.0]
        if single_block:
            assert np.all(ratios < 0.7), ratios
        else:
            assert np.all(np.abs(ratios - 1.0) < 0.1), ratios

    def test_gain_sign_matters(self, xp, monkeypatch):
        """Baseline: with the loop's error sign flipped the loop locks to the
        other zero of the S-curve, half a symbol off the symbol instants (it
        still measures the slope, so the timing is what fails)."""
        mf = _received(xp, ppm=20.0, num_symbols=2**15)
        monkeypatch.setattr(Gardner, "gains", lambda self, kd: (-4e-3 / kd, -4e-6 / kd))
        traj = to_numpy(estimate_sampling_clock_offset(mf, Gardner()).trajectory)
        err = traj - _true_trajectory(traj.size, 20.0)
        wrapped = (err[traj.size // 2 :] + 0.5) % 1.0 - 0.5
        assert float(np.mean(np.abs(wrapped))) > 0.4


class TestCorrect:
    @pytest.mark.parametrize(
        ("rolloff", "method"), [(0.5, "classical"), (0.1, "power")]
    )
    def test_restores_the_symbol_grid(self, xp, rolloff, method):
        """After the loop, the symbol-rate samples carry the AWGN plus the
        ISI of the timing jitter, σ² · Σ_{n≠0} g'(n)² for the raised-cosine
        g (theory.raised_cosine_slope_energy), σ measured against the true
        timing.  Predicted 19.97 and 19.70 dB, measured 20.01 and 19.59 dB;
        bound 0.25 dB."""
        esn0_db = 20.0
        mf = _received(xp, ppm=20.0, rolloff=rolloff, esn0_db=esn0_db)
        est = estimate_sampling_clock_offset(mf, Gardner(method=method))
        traj = to_numpy(est.trajectory)
        sigma2 = float(np.var(traj[1000:] - _true_trajectory(traj.size, 20.0)[1000:]))
        isi = sigma2 * raised_cosine_slope_energy(rolloff)
        predicted = -10 * np.log10(10 ** (-esn0_db / 10) + isi)
        y = correct_sampling_clock_offset(mf, Gardner(method=method))
        assert y.sps == 2
        sym = ck.multirate.decimate_to_symbol_rate(y)
        assert sym.reference.symbols.shape[-1] == sym.samples.shape[-1]
        measured = ck.metrics.snr(sym, num_skip_symbols=1000)
        assert abs(measured - predicted) < 0.25, (measured, predicted)

    def test_static_correction_from_an_estimate(self, xp):
        """An estimate of an earlier capture resamples by 1 + ε: the next
        capture's residual offset is within 0.2 ppm of zero (the estimate's
        own error, about 0.05 ppm)."""
        first = _received(xp, ppm=40.0, rolloff=0.5, num_symbols=2**16, rng=1)
        second = _received(xp, ppm=40.0, rolloff=0.5, num_symbols=2**16, rng=5)
        est = estimate_sampling_clock_offset(first, Gardner())
        fixed = correct_sampling_clock_offset(second, est)
        residual = estimate_sampling_clock_offset(fixed, Gardner())
        assert abs(float(residual.value)) < 0.2
        assert fixed.reference.symbols.shape[-1] == fixed.samples.shape[-1] // 2

    @pytest.mark.allow_warnings
    @pytest.mark.parametrize("dtype", ["complex64", "complex128"])
    def test_static_correction_of_a_long_record(self, xp, dtype):
        """A tone at 0.2 cycles per sample through a 2000 ppm clock and back
        through the static correction, 2^20 samples: over the last sixteenth
        the error stays below -120 dB (measured -129.6 dB, two
        interpolations).  Instants formed in float32 gave -76.2 dB in
        complex64."""
        n, ppm, f = 2**20, 2000.0, 0.2
        x = xp.exp(2j * xp.pi * f * xp.arange(n)).astype(dtype)
        y = ck.impairments.apply_sampling_clock_offset(x, offset_ppm=ppm)
        est = SamplingClockOffsetEstimate(
            value=xp.asarray(ppm), trajectory=None, detector_gain=None
        )
        z = correct_sampling_clock_offset(y, est, sps=2)
        assert z.dtype == x.dtype
        err = (z - x[: z.shape[0]])[-n // 16 : -60]
        assert 10 * math.log10(float(xp.mean(xp.abs(err) ** 2))) < -120.0

    def test_equalizer_needs_clock_recovery(self, xp, caplog):
        """Baseline: at 50 ppm the symbols drift 13 symbol periods over the
        record, beyond an 11-tap equalizer's 5.5-symbol window, and trained
        LMS fails (SER near 0.7 over the last quarter; its decisions may
        collapse onto a few points, which LMS reports as a WARNING); after
        Gardner recovery the same equalizer decides every symbol of the last
        quarter right at 20 dB (16-QAM SER theory 1e-5), with no warning."""
        n = 2**18
        mf = _received(xp, ppm=50.0, rolloff=0.5, num_symbols=n)
        kw = {"num_taps": 11, "step_size": 1e-3}
        train = mf.reference.symbols[..., :4000]
        drifting = ck.equalization.lms(mf, train, **kw)
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="commkit"):
            recovered = ck.equalization.lms(
                correct_sampling_clock_offset(mf, Gardner()), train, **kw
            )
        assert caplog.text == ""
        ref = to_numpy(mf.reference.symbols)
        last = slice(n - n // 4, n)
        assert scoring.ser(to_numpy(drifting.y_hat)[last], ref[last], C16) > 0.3
        assert scoring.ser(to_numpy(recovered.y_hat)[last], ref[last], C16) < 1e-3


class TestLockChecks:
    """A loop too wide for its detector's noise slips; once its timing runs
    through the symbols the mean detector output is zero and only the
    integrator's bound keeps its frequency near the clock's."""

    def test_bound_keeps_a_slipping_loop_near_the_clock(self, xp, caplog):
        """Classical detector, rolloff 0.1, B·T = 1e-2: jitter large enough
        to slip throughout the record.  Within the default 1000 ppm bound the
        loop keeps returning to lock and the estimate stays inside the bound
        (measured 168 and 209 ppm against the true 50, CPU and GPU); the
        slip is reported."""
        mf = _received(xp, ppm=50.0)
        with caplog.at_level(logging.WARNING, logger="commkit"):
            est = estimate_sampling_clock_offset(mf, Gardner(normalized_bandwidth=1e-2))
        assert "slipped" in caplog.text
        assert abs(float(est.value) - 50.0) < 1000.0

    @pytest.mark.allow_warnings
    def test_without_the_bound_the_loop_runs_away(self, xp):
        """Baseline: the same loop with the bound at 89 000 ppm wanders off
        in frequency (-37 007 and -4287 ppm measured, CPU and GPU)."""
        mf = _received(xp, ppm=50.0)
        est = estimate_sampling_clock_offset(
            mf, Gardner(normalized_bandwidth=1e-2, max_offset_ppm=8.9e4)
        )
        assert abs(float(est.value)) > 2000.0

    def test_slip_reported(self, xp, caplog):
        """Power detector, rolloff 0.1, B·T = 2e-2: a few slips (3 to 5
        measured), each moving the timing by a whole symbol, against at
        most 0.46 symbol of jitter-driven movement over the window in the
        loops that do not slip."""
        mf = _received(xp, ppm=50.0)
        with caplog.at_level(logging.WARNING, logger="commkit"):
            correct_sampling_clock_offset(
                mf, Gardner(method="power", normalized_bandwidth=2e-2)
            )
        assert "slipped" in caplog.text

    def test_first_slip_on_a_known_trajectory(self):
        """A drifting, jittery trajectory with one slip at symbol 50 000:
        found where the window first spans the step (8 / B·T = 800 symbols
        before it), and nothing without the step."""
        rng = np.random.default_rng(0)
        k = np.arange(100_000)
        traj = k * 50e-6 + 0.05 * rng.standard_normal(k.size)
        assert cr._first_slip(traj[None], np.array([1e-2]))[0] == -1
        slipped = traj + (k >= 50_000)
        first = cr._first_slip(slipped[None], np.array([1e-2]))[0]
        assert 50_000 - 800 <= first < 50_000

    def test_offset_beyond_the_bound(self, xp, caplog):
        """1500 ppm with the default bound: the integrator stops at 1000 ppm
        and the proportional path carries the rest with a static timing lag
        (the slope still reads 1499.87 and 1499.83 ppm, CPU and GPU), and the warning names the
        bound; with max_offset_ppm=3000 there is no warning and the offset
        is measured within 0.5 ppm (1500.01 and 1499.97)."""
        mf = _received(xp, ppm=1500.0, rolloff=0.5, num_symbols=2**16)
        with caplog.at_level(logging.WARNING, logger="commkit"):
            estimate_sampling_clock_offset(mf, Gardner())
        assert "max_offset_ppm" in caplog.text
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="commkit"):
            est = estimate_sampling_clock_offset(mf, Gardner(max_offset_ppm=3000.0))
        assert caplog.text == ""
        assert abs(float(est.value) - 1500.0) < 0.5


class TestValidation:
    def test_needs_even_sps(self, xp):
        mf = _received(xp, ppm=0.0, num_symbols=256, sps=3)
        with pytest.raises(ValueError, match="even sps"):
            estimate_sampling_clock_offset(mf, Gardner())

    @pytest.mark.parametrize("method", ["classical", "power"])
    def test_no_clock_tone_raises(self, xp, method):
        """An unmodulated carrier has no transitions and no clock tone."""
        carrier = xp.ones(4096, dtype=xp.complex64)
        with pytest.raises(ValueError, match="no clock tone"):
            estimate_sampling_clock_offset(carrier, Gardner(method=method), sps=2)

    def test_method_type(self, xp):
        with pytest.raises(TypeError, match="Gardner"):
            estimate_sampling_clock_offset(xp.ones(64, xp.complex64), "gardner", sps=2)
        with pytest.raises(TypeError, match="Gardner"):
            correct_sampling_clock_offset(xp.ones(64, xp.complex64), 20.0, sps=2)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"normalized_bandwidth": 0.0},
            {"integral_gain": 1e-6},
            {"proportional_gain": -1.0},
            {"max_offset_ppm": 0.0},
            {"max_offset_ppm": 9e4},
            {"method": "squared"},
        ],
    )
    def test_gardner_validates(self, kwargs):
        with pytest.raises(ValueError):
            Gardner(**kwargs)

    def test_logs_the_offset(self, xp, caplog):
        mf = _received(xp, ppm=10.0, num_symbols=2**14)
        with caplog.at_level(logging.INFO, logger="commkit"):
            est = estimate_sampling_clock_offset(mf, Gardner())
        assert isinstance(est, SamplingClockOffsetEstimate)
        assert "ppm" in caplog.text
