"""Sequential adaptive equalizers (lms/rls/cma/rde) on the Numba backend."""

import numpy as np
import pytest

from commkit import equalization, generate
from commkit.core import Signal, SingleCarrierFrame
from commkit.equalization import EqualizerResult
from commkit.equalization.sequential._dd import _check_rls_divergence
from commkit.filtering import RRC
from commkit.mapping import Constellation
from tests.common import scoring
from tests.common.conversions import to_numpy
from tests.common.signals import make_channel_signal

ISI_TAPS = [0.25, 1.0, 0.3j, -0.1]


def _isi_record(constellation, num_symbols, *, esn0_db, seed=0, xp=None):
    """RRC record at 2 sps through a 4-tap complex ISI channel and a
    0.3-sample delay, with AWGN; an equalizer must undo it."""
    return make_channel_signal(
        constellation,
        num_symbols,
        sps=2,
        isi_taps=ISI_TAPS,
        frac_delay=0.3,
        esn0_db=esn0_db,
        seed=seed,
        xp=xp,
    )


class TestLMS:
    """Tests for the LMS adaptive equalizer."""

    def test_convergence_known_channel(self, xp):
        """Trained LMS on a 4-tap complex ISI channel with a 0.3-sample delay
        (16-QAM, 2 sps, Es/N0 22 dB) approaches the least-squares Wiener
        solution of the same 21-tap regressor: within 1.5 dB of its MSE over
        the second half of 2e4 symbols (measured 1.2 dB; the rest is gradient
        noise and lag).  The record is periodic, so the Wiener taps are exact
        on circular windows.  Frozen taps (μ = 0) are far from it."""
        rx = _isi_record(Constellation.qam(16), 20000, esn0_db=22.0, xp=xp)
        ref = to_numpy(rx.reference.symbols)
        x = to_numpy(rx.samples).astype(np.complex128)
        T, sps, half = 21, 2, 10000
        windows = (np.arange(ref.size)[:, None] * sps - T // 2 + np.arange(T)) % x.size
        w, *_ = np.linalg.lstsq(x[windows], ref, rcond=None)
        wiener_mse = float(np.mean(np.abs(x[windows] @ w - ref) ** 2))

        def tail_mse(step):
            y = to_numpy(equalization.lms(rx, ref, num_taps=T, step_size=step).y_hat)
            return float(np.mean(np.abs(y[half:] - ref[half:]) ** 2))

        assert 10 * np.log10(tail_mse(1e-3) / wiener_mse) < 1.5
        assert tail_mse(0.0) > 10 * wiener_mse

    def test_decision_directed_after_training(self, xp):
        """1000 training symbols, then decisions: the held-out second half
        has no symbol errors against the transmitted symbols and an EVM
        within 1.4x the AWGN floor (0.079 at 22 dB); frozen taps leave the
        ISI (SER above 0.3)."""
        c = Constellation.qam(16)
        rx = _isi_record(c, 6000, esn0_db=22.0, xp=xp)
        ref = to_numpy(rx.reference.symbols)

        def run(step):
            return to_numpy(
                equalization.lms(rx, ref[:1000], num_taps=21, step_size=step).y_hat
            )[3000:]

        y = run(1e-3)
        assert scoring.ser(y, ref[3000:], c) == 0.0
        assert scoring.evm_rms(y, ref[3000:]) < 1.4 * 10 ** (-22 / 20)
        assert scoring.ser(run(0.0), ref[3000:], c) > 0.3

    def test_requires_constellation_or_training(self, xp):
        """LMS should raise if neither training nor constellation is given."""

        sig = generate(
            Constellation.psk(4), 100, symbol_rate=1e6, sps=2, pulse=RRC(0.35), rng=0
        )
        rx = xp.asarray(sig.samples)
        with pytest.raises(ValueError):
            equalization.lms(rx, sps=2)


class TestRLS:
    """Tests for the RLS adaptive equalizer."""

    def test_convergence(self, xp):
        """RLS with 500 training symbols on the ISI channel: no symbol errors
        after training, EVM within 1.2x the AWGN floor (RLS has almost no
        misadjustment at λ = 0.999)."""
        c = Constellation.qam(16)
        rx = _isi_record(c, 3000, esn0_db=22.0, xp=xp)
        ref = to_numpy(rx.reference.symbols)
        res = equalization.rls(rx, ref[:500], num_taps=21, forgetting_factor=0.999)
        y = to_numpy(res.y_hat)[1000:]
        assert scoring.ser(y, ref[1000 : 1000 + y.size], c) == 0.0
        assert scoring.evm_rms(y, ref[1000 : 1000 + y.size]) < 1.2 * 10 ** (-22 / 20)

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_faster_convergence_than_lms(self, xp, seed):
        """RLS acquires a 3-tap ISI channel in about twice its 15 taps; LMS
        at step 0.01 needs at least 1.5x as long on the same record.

        Acquisition is the end of the last 20-symbol window whose MSE against
        the transmitted symbols is above 0.05 (the AWGN floor at 25 dB is
        0.003), so both run on the same symbols and indices.
        """
        c = Constellation.psk(4)
        rx = make_channel_signal(
            c, 1500, sps=1, isi_taps=[0.3, 1.0, 0.2], esn0_db=25.0, seed=seed, xp=xp
        )
        ref = to_numpy(rx.reference.symbols)

        def acquisition(y, w=20):
            y = to_numpy(y)
            e = np.convolve(np.abs(y - ref[: y.size]) ** 2, np.ones(w) / w, "valid")
            bad = np.nonzero(e >= 0.05)[0]
            return 0 if bad.size == 0 else int(bad[-1]) + w

        n_lms = acquisition(
            equalization.lms(rx, ref, num_taps=15, step_size=0.01).y_hat
        )
        n_rls = acquisition(
            equalization.rls(rx, ref, num_taps=15, forgetting_factor=0.99).y_hat
        )
        assert n_rls < 2 * 15 + 20, n_rls
        assert n_lms > 1.5 * n_rls, (n_lms, n_rls)


class TestCMA:
    """Tests for the CMA blind equalizer."""

    def test_convergence_qpsk(self, xp):
        """Blind CMA on QPSK through the ISI channel (Es/N0 18 dB): after one
        constant rotation (CMA is blind to the carrier phase) the second half
        has no symbol errors and an EVM within 1.4x the AWGN floor (0.126);
        frozen taps stay above twice the floor."""
        c = Constellation.psk(4)
        rx = _isi_record(c, 10000, esn0_db=18.0, xp=xp)
        ref = to_numpy(rx.reference.symbols)
        floor = 10 ** (-18 / 20)

        def evm(step):
            y = equalization.cma(rx, num_taps=21, step_size=step, constellation=c).y_hat
            y = scoring.resolve_blind(y, ref, c, skip=5000, rotation="any")[5000:]
            return scoring.ser(y, ref[5000:], c), scoring.evm_rms(y, ref[5000:])

        ser, adapted = evm(1e-3)
        assert ser == 0.0 and adapted < 1.4 * floor
        assert evm(0.0)[1] > 2 * floor


class TestRDE:
    """Tests for the Radius Directed Equalizer."""

    def test_convergence_16qam(self, xp):
        """Blind RDE on 16-QAM through the ISI channel (Es/N0 22 dB, 4e4
        symbols): after one constant rotation the second half has no symbol
        errors; frozen taps leave SER above 0.3.  (A step of 1e-4 is too slow
        for this record: SER 0.21.)"""
        c = Constellation.qam(16)
        rx = _isi_record(c, 40000, esn0_db=22.0, xp=xp)
        ref = to_numpy(rx.reference.symbols)

        def ser(step):
            y = equalization.rde(rx, num_taps=21, step_size=step, constellation=c)
            return scoring.blind_ser(y.y_hat, ref, c, skip=20000, rotation="any")

        assert ser(1e-3) == 0.0
        assert ser(0.0) > 0.3

    def test_refines_cma_on_16qam(self, xp):
        """RDE started from CMA's taps lowers the EVM on 16-QAM: CMA's error
        y(|y|² - R²) is not zero on the points of a multi-ring constellation,
        RDE's nearest-ring error is."""
        c = Constellation.qam(16)
        rx = _isi_record(c, 20000, esn0_db=22.0, xp=xp)
        ref = to_numpy(rx.reference.symbols)
        cma = equalization.cma(rx, num_taps=21, step_size=5e-4, constellation=c)
        rde = equalization.rde(
            rx, num_taps=21, step_size=5e-4, constellation=c, initial_taps=cma.weights
        )
        evm = {}
        for name, res in (("cma", cma), ("rde", rde)):
            y = scoring.resolve_blind(res.y_hat, ref, c, skip=10000, rotation="any")
            assert scoring.ser(y[10000:], ref[10000:], c) == 0.0, name
            evm[name] = scoring.evm_rms(y[10000:], ref[10000:])
        assert evm["rde"] < evm["cma"], evm


class TestResultLayout:
    """EqualizerResult's documented layout, in one place per algorithm:
    ``y_hat`` and ``error`` ``(N_sym,)`` or ``(C, N_sym)`` (RLS trims
    ``tail_trim = num_taps // 2`` symbols), ``weights`` ``(num_taps,)`` or
    ``(C, C, num_taps)``, and ``weights_history`` only with
    ``store_weights=True``, one row per processed symbol."""

    @pytest.mark.parametrize("name", ["lms", "rls", "cma", "rde"])
    @pytest.mark.parametrize("num_channels", [1, 2])
    def test_layout(self, xp, name, num_channels):
        c, n, taps = Constellation.psk(4), 400, 7
        sig = generate(c, n, symbol_rate=1e6, sps=2, pulse=RRC(0.35), rng=0,
                       num_channels=num_channels)  # fmt: skip
        args = [xp.asarray(sig.samples)]
        if name in ("lms", "rls"):
            args.append(xp.asarray(sig.reference.symbols))
        fn = getattr(equalization, name)
        kw = dict(num_taps=taps, constellation=c, sps=2)
        res = fn(*args, **kw)
        trim = taps // 2 if name == "rls" else 0
        lead = () if num_channels == 1 else (num_channels,)
        w = (taps,) if num_channels == 1 else (num_channels, num_channels, taps)
        assert res.tail_trim == trim
        assert res.y_hat.shape == res.error.shape == (*lead, n - trim)
        assert res.weights.shape == w
        assert res.weights_history is None
        stored = fn(*args, store_weights=True, **kw)
        assert stored.weights_history.shape == (n, *w)
        xpt_w = to_numpy(stored.weights_history[-1])
        np.testing.assert_allclose(xpt_w, to_numpy(stored.weights), rtol=1e-6)


class TestNonSquareLatticeDecisions:
    """The O(1) slicer is used only for square lattices with equal axes."""

    @pytest.mark.parametrize("name", ["lms", "rls", "block_lms"])
    def test_rectangular_constellation_decisions(self, xp, name):
        """A 4x4 rectangular lattice (imaginary spacing twice the real one)
        has sqrt(M) levels per axis but is not square; it was sliced on the
        real grid, and the decision-directed equalizers lost 25 % of the
        symbols on a clean channel."""
        re, im = np.array([-3.0, -1.0, 1.0, 3.0]), np.array([-6.0, -2.0, 2.0, 6.0])
        points = (re[:, None] + 1j * im[None, :]).ravel()
        c = Constellation(points / np.sqrt(np.mean(np.abs(points) ** 2)))
        rng = np.random.default_rng(0)
        s = c.points[rng.integers(0, 16, 8000)].astype(np.complex64)
        noise = 0.01 * (rng.standard_normal(s.size) + 1j * rng.standard_normal(s.size))
        x = xp.asarray((s + noise).astype(np.complex64))
        kw = {"block_size": 64} if name == "block_lms" else {}
        r = getattr(equalization, name)(
            x, xp.asarray(s[:500]), sps=1, num_taps=7, constellation=c, **kw
        )
        y = to_numpy(r.y_hat)[2000:7000]
        assert np.mean(c.demap(y) != c.demap(s[2000:7000])) == 0.0


class TestEdgeCases:
    """Tests for error paths, warnings, and edge-case inputs."""

    def _qpsk_rx(self, xp, n_symbols=500, seed=0):

        sig = generate(
            Constellation.psk(4),
            n_symbols,
            symbol_rate=1e6,
            sps=2,
            pulse=RRC(0.35),
            rng=seed,
        )
        return xp.ascontiguousarray(xp.asarray(sig.samples)), sig

    def test_lms_raises_no_constellation_numba(self, xp):
        """LMS Numba: ValueError when no modulation and no training symbols (DD impossible)."""
        rx, _ = self._qpsk_rx(xp)
        with pytest.raises(ValueError, match="needs a constellation"):
            equalization.lms(rx, num_taps=7, sps=2)

    def test_rls_warns_fractional_spacing(self, xp, caplog):
        """RLS at sps > 1 warns that its correlation matrix is ill-conditioned;
        at 1 sps it does not."""
        import logging

        rx, sig = self._qpsk_rx(xp, n_symbols=400)
        train = xp.asarray(sig.reference.symbols)
        kw = dict(num_taps=7, constellation=Constellation.psk(4))
        with caplog.at_level(logging.WARNING, logger="commkit"):
            equalization.rls(rx, train, sps=2, **kw)
        assert any("ill-conditioned" in r.getMessage() for r in caplog.records)
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="commkit"):
            equalization.rls(rx[::2], train, sps=1, **kw)
        assert not any("ill-conditioned" in r.getMessage() for r in caplog.records)

    def test_validate_sps_small_num_taps(self, xp, caplog):
        """_validate_sps should log a warning when num_taps < 4*sps."""
        import logging

        rx, sig = self._qpsk_rx(xp)
        with caplog.at_level(logging.WARNING, logger="commkit"):
            equalization.lms(
                rx,
                xp.asarray(sig.reference.symbols),
                num_taps=3,  # < 4*sps=8 -> should warn
                constellation=Constellation.psk(4),
                sps=2,
            )
        assert any("small" in r.message.lower() for r in caplog.records)

    def test_rde_no_modulation_falls_back_to_cma(self, xp, xpt):
        """Without a constellation RDE has the one radius 1, which is CMA
        with R² = 1: the two give the same output."""
        rx, _ = self._qpsk_rx(xp)
        kw = dict(num_taps=7, step_size=0.005, sps=2)
        xpt.assert_allclose(
            equalization.rde(rx, **kw).y_hat,
            equalization.cma(rx, **kw).y_hat,
            rtol=1e-5,
            atol=1e-6,
        )

    def test_lms_num_train_symbols_clamps_numba(self, xp):
        """LMS Numba: pre-sliced training_symbols limits DA phase length."""
        rx, sig = self._qpsk_rx(xp, n_symbols=1000)
        train = xp.asarray(sig.reference.symbols[:30])

        result = equalization.lms(
            rx,
            train,
            num_taps=7,
            step_size=0.05,
            constellation=Constellation.psk(4),
            sps=2,
        )

        assert result.num_train_symbols <= 30

    def test_lms_partial_training_needs_constellation(self, xp):
        """Decisions after the training need a constellation; none is guessed."""
        rx, sig = self._qpsk_rx(xp)
        train = xp.asarray(sig.reference.symbols)[..., :50]
        with pytest.raises(ValueError, match="needs a constellation"):
            equalization.lms(rx, train, num_taps=7, step_size=0.05, sps=2)

    def test_lms_full_training_needs_no_constellation(self, xp):
        """With every symbol trained there are no decisions to make."""
        rx, sig = self._qpsk_rx(xp)
        train = xp.asarray(sig.reference.symbols)
        result = equalization.lms(rx, train, num_taps=7, step_size=0.05, sps=2)
        assert result.num_train_symbols == result.y_hat.shape[-1]

    @pytest.mark.parametrize(
        ("taps", "best", "worst"),
        [([1.0, 0.6, 0.5, 0.4, 0.3], 8, 2), ([0.3, 0.4, 0.5, 0.6, 1.0], 2, 8)],
        ids=["post-cursor", "pre-cursor"],
    )
    def test_center_tap_override(self, xp, xpt, taps, best, worst):
        """``center_tap`` splits the taps around the main path: on a channel
        whose ISI lies on one side, the matching split reaches a clearly lower
        MSE against the reference (measured 0.054 vs 0.097), while the
        regressor window moves with it so the output stays aligned (frozen
        taps give the same output for any centre tap)."""
        c = Constellation.qam(16)
        rx = make_channel_signal(c, 6000, isi_taps=taps, esn0_db=30.0, xp=xp)
        ref = to_numpy(rx.reference.symbols)

        def mse(center):
            y = equalization.lms(
                rx, ref, num_taps=11, step_size=1e-3, center_tap=center
            ).y_hat
            return float(np.mean(np.abs(to_numpy(y)[3000:] - ref[3000:]) ** 2))

        assert mse(best) < 0.7 * mse(worst)
        frozen = dict(num_taps=11, step_size=0.0, constellation=c)
        xpt.assert_allclose(
            equalization.cma(rx, **frozen, center_tap=best).y_hat,
            equalization.cma(rx, **frozen).y_hat,
            rtol=1e-5,
            atol=1e-6,
        )


class TestNumbaBackendCoverage:
    """Tests targeting uncovered numba-backend code paths in LMS/RLS/CMA/RDE."""

    def _make_qpsk_rx(self, xp, n_symbols=1000, seed=0):

        channel = xp.array([0.1, 1.0, 0.15], dtype=xp.complex64)
        sig = generate(
            Constellation.psk(4),
            n_symbols,
            symbol_rate=1e6,
            sps=2,
            pulse=RRC(0.35),
            rng=seed,
        )
        rx = xp.convolve(xp.asarray(sig.samples), channel, mode="same")
        return xp.ascontiguousarray(rx), sig

    def test_lms_numba_pure_dd_no_training(self, xp):
        """Decision-directed LMS from a cold start, without training, opens
        a mild ISI channel (QPSK, Es/N0 18 dB): after one rotation of π/2
        multiples (decisions alone cannot fix the quadrant) the second half
        has no symbol errors and an EVM within 1.4x the AWGN floor; frozen
        taps stay above twice it."""
        c = Constellation.psk(4)
        rx = make_channel_signal(
            c, 6000, isi_taps=[0.15, 1.0, 0.2j], esn0_db=18.0, seed=1, xp=xp
        )
        ref = to_numpy(rx.reference.symbols)
        floor = 10 ** (-18 / 20)

        def score(step):
            y = equalization.lms(rx, num_taps=21, step_size=step, constellation=c)
            y = scoring.resolve_blind(y.y_hat, ref, c, skip=3000)[3000:]
            return scoring.ser(y, ref[3000:], c), scoring.evm_rms(y, ref[3000:])

        ser, evm = score(1e-3)
        assert ser == 0.0 and evm < 1.4 * floor
        assert score(0.0)[1] > 2 * floor

    def test_rls_divergence_guard(self, xp):
        """The guard behind rls() raises an actionable error on non-finite
        taps and passes finite ones (a unit test of the check: the RLS
        recursion's regularization keeps it finite on the inputs tried)."""
        good = xp.ones((2, 2, 5), dtype=xp.complex64)
        _check_rls_divergence(good, xp, 0.99, 0.01)
        bad = good.copy()
        bad[0, 0, 0] = float("nan")
        with pytest.raises(RuntimeError, match="RLS equalizer diverged"):
            _check_rls_divergence(bad, xp, 0.99, 0.01)

    def test_rls_numba_num_train_symbols(self, xp):
        """RLS numba: pre-sliced training_symbols limits DA phase length."""
        rx, sig = self._make_qpsk_rx(xp, n_symbols=800)
        train = xp.asarray(sig.reference.symbols[:50])

        result = equalization.rls(
            rx, train, sps=2, num_taps=7, constellation=Constellation.psk(4)
        )

        assert isinstance(result, EqualizerResult)
        assert result.num_train_symbols <= 50


class TestCmaPilotAided:
    """Tests for cma()/rde() with EqualizerPilots (hybrid PA mode)."""

    def _make_comb_frame_and_samples(self, xp):
        from commkit.backend import to_device
        from commkit.core import Preamble, SingleCarrierFrame

        preamble = Preamble(sequence_type="barker", num_symbols=13)
        frame = SingleCarrierFrame(
            payload_num_symbols=1008,
            pilot_pattern="comb",
            pilot_period=10,
            preamble=preamble,
        )
        sig = frame.to_signal(sps=2, symbol_rate=1e6, pulse=RRC(0.35))
        struct = frame.get_structure_map(unit="symbols", sps=1, include_preamble=False)
        samples_cpu = to_device(sig.samples, "cpu")
        pilot_syms_cpu = to_device(frame.pilot_symbols, "cpu")
        pilot_mask_bool = to_device(struct["pilots"], "cpu")
        n_body = int(pilot_mask_bool.size)
        return frame, samples_cpu, pilot_syms_cpu, pilot_mask_bool, n_body

    @pytest.mark.parametrize("name", ["cma", "rde"])
    def test_pilot_deboost_leaves_input_alone(self, name):
        """De-boosting the pilots works on a copy of a NumPy complex64 input."""
        rng = np.random.default_rng(0)
        x = (rng.standard_normal(400) + 1j * rng.standard_normal(400)).astype(
            np.complex64
        )
        before = x.copy()
        mask = np.zeros(200, dtype=bool)
        mask[::10] = True
        pilots = equalization.EqualizerPilots(
            np.full(20, 1 + 1j, np.complex64), mask, gain_db=3.0
        )
        getattr(equalization, name)(x, sps=2, num_taps=5, pilots=pilots)
        np.testing.assert_array_equal(x, before)

    @pytest.mark.parametrize("name", ["cma", "rde"])
    def test_pilots_resolve_the_phase(self, xp, name):
        """Pilots every 10th symbol fix the absolute phase a blind equalizer
        cannot see: on the ISI channel with a 1 rad carrier rotation the
        output matches the transmitted symbols with no ambiguity resolution
        (SER < 1e-3 over the last quarter of 4e4 symbols; the pilot gradient turns
        the taps at a rate set by the step and the pilot density, about 3e4
        symbols for 1 rad at 1e-3); without pilots it does not."""
        c = Constellation.qam(16)
        rx = _isi_record(c, 40000, esn0_db=22.0, seed=2)
        rx = rx.replace(samples=(rx.samples * np.exp(1j)).astype(np.complex64))
        rx = rx.to("gpu" if xp is not np else "cpu")
        ref = to_numpy(rx.reference.symbols)
        mask = np.zeros(ref.size, dtype=bool)
        mask[::10] = True
        pilots = equalization.EqualizerPilots(ref[mask], mask)
        fn = getattr(equalization, name)
        kw = dict(num_taps=21, step_size=1e-3, constellation=c)
        aided = to_numpy(fn(rx, **kw, pilots=pilots).y_hat)
        blind = to_numpy(fn(rx, **kw).y_hat)
        assert scoring.ser(aided[30000:], ref[30000:], c) < 1e-3
        assert scoring.ser(blind[30000:], ref[30000:], c) > 0.5

    def test_equalizer_pilots(self, xp):
        """EqualizerPilots checks its shapes; from_frame takes the frame's
        positions, symbols and gain, and a wrong length raises at the call."""
        n_sym, n_pilots = 100, 10
        mask = np.zeros(n_sym, dtype=bool)
        mask[np.linspace(0, n_sym - 1, n_pilots, dtype=int)] = True
        pilots = equalization.EqualizerPilots(np.ones((2, n_pilots)), mask)
        assert pilots.values.dtype == np.complex64 and pilots.gain_db == 0.0
        with pytest.raises(ValueError, match="K = 10 pilots"):
            equalization.EqualizerPilots(np.ones(9), mask)
        frame, samples_cpu, pilot_syms_cpu, pilot_mask_bool, n_body = (
            self._make_comb_frame_and_samples(xp)
        )
        from_frame = equalization.EqualizerPilots.from_frame(frame)
        assert from_frame.gain_db == frame.pilot_gain_db
        assert from_frame.mask.size == frame.num_symbols
        np.testing.assert_array_equal(
            from_frame.mask[frame.preamble.num_symbols :], pilot_mask_bool
        )
        np.testing.assert_array_equal(from_frame.values, pilot_syms_cpu)
        with pytest.raises(ValueError, match="output symbols"):
            equalization.cma(samples_cpu[:-20], sps=2, pilots=from_frame)
        with pytest.raises(TypeError, match="EqualizerPilots"):
            equalization.cma(samples_cpu, sps=2, pilots=(1, 2))
        with pytest.raises(ValueError, match="no pilots"):
            equalization.EqualizerPilots.from_frame(
                SingleCarrierFrame(payload_num_symbols=10)
            )


class TestSignalInputSequentialEqualizers:
    """Signal-awareness for lms, rls, cma, rde: sps from the signal, y_hat as
    a new Signal at the symbol rate."""

    def _rx_signal(self, xp, order=16, n_symbols=2000, sps=2, seed=0):
        sig = generate(
            Constellation.qam(order),
            n_symbols,
            symbol_rate=1e6,
            sps=sps,
            pulse=RRC(0.35),
            rng=seed,
        )
        return sig

    def test_lms_signal_input(self, xp, xpt):
        sig = self._rx_signal(xp)
        data = xp.asarray(sig.samples)

        result_sig = equalization.lms(
            sig, num_taps=11, constellation=Constellation.qam(16)
        )
        result_arr = equalization.lms(
            data, num_taps=11, sps=2, constellation=Constellation.qam(16)
        )

        assert isinstance(result_sig.signal, Signal)
        assert result_sig.signal.sampling_rate == 1e6
        xpt.assert_allclose(result_sig.signal.samples, result_arr.y_hat)
        xpt.assert_allclose(result_sig.y_hat, result_arr.y_hat)

    def test_rls_signal_input(self, xp, xpt):
        sig = self._rx_signal(xp, sps=1, n_symbols=500)
        data = xp.asarray(sig.samples)

        result_sig = equalization.rls(
            sig, num_taps=11, constellation=Constellation.qam(16)
        )
        result_arr = equalization.rls(
            data, num_taps=11, sps=1, constellation=Constellation.qam(16)
        )

        assert isinstance(result_sig.signal, Signal)
        assert result_sig.signal.sampling_rate == 1e6
        xpt.assert_allclose(result_sig.signal.samples, result_arr.y_hat)
        xpt.assert_allclose(result_sig.y_hat, result_arr.y_hat)

    def test_cma_signal_input(self, xp, xpt):
        sig = self._rx_signal(xp)
        data = xp.asarray(sig.samples)

        result_sig = equalization.cma(
            sig, num_taps=11, constellation=Constellation.qam(16)
        )
        result_arr = equalization.cma(
            data, num_taps=11, sps=2, constellation=Constellation.qam(16)
        )

        assert isinstance(result_sig.signal, Signal)
        assert result_sig.signal.sampling_rate == 1e6
        xpt.assert_allclose(result_sig.signal.samples, result_arr.y_hat)
        xpt.assert_allclose(result_sig.y_hat, result_arr.y_hat)

    def test_rde_signal_input(self, xp, xpt):
        sig = self._rx_signal(xp)
        data = xp.asarray(sig.samples)

        result_sig = equalization.rde(
            sig, num_taps=11, constellation=Constellation.qam(16)
        )
        result_arr = equalization.rde(
            data, num_taps=11, sps=2, constellation=Constellation.qam(16)
        )

        assert isinstance(result_sig.signal, Signal)
        assert result_sig.signal.sampling_rate == 1e6
        xpt.assert_allclose(result_sig.signal.samples, result_arr.y_hat)
        xpt.assert_allclose(result_sig.y_hat, result_arr.y_hat)


class TestKnownSymbolScale:
    """Training symbols and pilots are taken as given, on the scale of the
    unit-power constellation."""

    @staticmethod
    def _corner_symbols(n: int = 3000) -> np.ndarray:
        # Only the four corners of 16-QAM: sample power 1.8, far from 1, so a
        # renormalized training sequence would land 25% inside the grid.
        pts = Constellation.qam(16).points
        corners = pts[np.abs(pts) ** 2 > 1.5]
        rng = np.random.default_rng(0)
        return corners[rng.integers(0, 4, n)].astype(np.complex64)

    @pytest.mark.parametrize("name", ["lms", "rls", "block_lms"])
    def test_training_reproduces_the_points(self, name, xp):
        """Noiseless identity channel, trained throughout: y_hat -> s."""
        s = self._corner_symbols()
        kw = dict(sps=1, num_taps=3, constellation=Constellation.qam(16))
        if name == "lms":
            kw["step_size"] = 0.05
        if name == "block_lms":
            kw.update(step_size=5e-3, block_size=64)
        res = getattr(equalization, name)(xp.asarray(s), xp.asarray(s), **kw)
        y = to_numpy(res.y_hat)[-500:]
        assert np.max(np.abs(y - s[: res.y_hat.shape[-1]][-500:])) < 1e-3

    @pytest.mark.parametrize("name", ["cma", "block_cma"])
    def test_pilots_on_a_shaped_constellation(self, name, xp):
        """All-pilot blind run on a shaped constellation reproduces the points;
        1.x rescaled the pilots by 1/sqrt(E_PS) off the constellation."""
        c = Constellation.qam(16).shaped(nu=0.2)
        rng = np.random.default_rng(1)
        s = c.points[rng.choice(16, 3000, p=c.pmf)].astype(np.complex64)
        pilots = equalization.EqualizerPilots(s, np.ones(s.size, dtype=bool))
        kw = dict(sps=1, num_taps=3, constellation=c, pilots=pilots)
        if name == "cma":
            kw["step_size"] = 0.05
        else:
            kw.update(step_size=5e-3, block_size=64)
        res = getattr(equalization, name)(xp.asarray(s), **kw)
        y = to_numpy(res.y_hat)[-500:]
        assert np.max(np.abs(y - s[-500:])) < 1e-3
