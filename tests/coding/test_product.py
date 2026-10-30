"""Tests for staircase and concatenated codes, Chase decoding and the 400ZR
inner code (coding/product.py, coding/linear.py)."""

import itertools
from typing import Any

import numpy as np
import pytest

from commkit import generate, mapping, metrics
from commkit.backend import to_device
from commkit.coding import (
    BCHCode,
    Chase,
    Concatenated,
    HardIterations,
    Interleaver,
    LinearBlockCode,
    MaximumLikelihood,
    ReedSolomonCode,
    Stages,
    StaircaseCode,
    Syndrome,
    decode,
    encode,
)
from commkit.coding._gf2 import rank, row_reduce
from commkit.coding.product import _COMPONENT, _left, _pi_d
from commkit.impairments import apply_awgn
from tests.common.coding import random_bits
from tests.common.scoring import binomial_margin
from tests.common.theory import qam16_gray_ber


class TestOIF400ZRHamming:
    def test_parity_check_of_the_standard(self) -> None:
        """OIF-400ZR-01.0 clause 10.5: the code's checks span the clause's H."""
        i = np.arange(128)
        s = (i[None, :] >> np.arange(7)[:, None]) & 1
        s0, s1, s2 = s[0], s[1], s[2]
        s7 = (s0 & s2) | ((1 - s0) & (1 - s1) & (1 - s2)) | (s0 & s1 & (1 - s2))
        g = np.concatenate([s, s7[None], np.ones((1, 128), int)])
        order = [*range(63), *range(64, 95), *range(96, 111), *range(112, 119), 120, 122, 124,
                 63, 95, 111, 119, 121, 123, 125, 126, 127]  # fmt: skip
        h = g[:, order]
        code = LinearBlockCode.oif400zr()
        assert (code.n, code.k) == (128, 119)
        assert not np.any((code.generator.astype(int) @ h.T) % 2)
        assert rank(np.concatenate([h, code.parity_check])) == 9
        np.testing.assert_array_equal(code.generator[:, :119], np.eye(119))

    def test_distance_4(self) -> None:
        h = LinearBlockCode.oif400zr().parity_check
        cols = [int("".join(map(str, c)), 2) for c in h.T]
        assert len(set(cols)) == 128 and 0 not in cols
        assert not any((a ^ b) in set(cols) for a in cols for b in cols if a < b)


class TestChase:
    def test_large_list_equals_ml(self, xp: Any, xpt: Any) -> None:
        code = LinearBlockCode.hamming(4, extended=True)
        llrs = xp.asarray(
            (0.8 + np.random.default_rng(0).standard_normal(16 * 300)).astype(
                np.float32
            )
        )
        chase = decode(llrs, code=code, method=Chase(12, hard=Syndrome()))
        ml = decode(llrs, code=code, method=MaximumLikelihood())
        xpt.assert_array_equal(chase.bits, ml.bits)

    @pytest.mark.parametrize(
        ("test_positions", "beta"), [(3, None), (3, 5.0), (7, None), (1, None)]
    )
    def test_soft_output_matches_an_independent_chase(
        self, xp: Any, test_positions: int, beta: float | None
    ) -> None:
        """Decision and every soft output equal a Chase-II written here:
        candidates are the nearest codewords (by Hamming distance, unique
        for the perfect Hamming(7, 4)) of the hard decision with every
        combination of the least reliable bits flipped; the decision is the
        candidate of largest correlation ``Σ (1 - 2c) LLR``; bit ``i`` gets
        half the gap between the best candidates with each value there
        (max-log), else ``beta`` with the decision's sign.  ``beta=None``
        takes the word's largest competitor magnitude, or its largest input
        ``|LLR|`` when no candidate disagrees anywhere.  With 7 positions
        every codeword is a candidate, so the soft output is exact max-log.
        """
        code = LinearBlockCode.hamming(3)
        u = np.array(list(itertools.product([0, 1], repeat=code.k)))
        words = u @ code.generator.astype(int) % 2  # all 16 codewords
        num = 300
        llrs = (1.0 + np.random.default_rng(1).standard_normal((num, 7))).astype(
            np.float32
        )
        res = decode(
            xp.asarray(llrs.reshape(-1)),
            code=code,
            method=Chase(test_positions, beta=beta),
        )
        decided = np.empty((num, 7), int)
        soft = np.empty((num, 7))
        for r, x in enumerate(llrs.astype(np.float64)):
            hard = (x < 0).astype(int)
            least = np.argsort(np.abs(x))[:test_positions]
            cands = set()
            for flips in itertools.product([0, 1], repeat=test_positions):
                t = hard.copy()
                t[least] ^= np.array(flips)
                cands.add(tuple(words[np.argmin(np.sum(words != t, axis=1))]))
            cand = np.array(sorted(cands))
            metric = ((1 - 2 * cand) * x).sum(axis=1)
            best = np.argmax(metric)
            decided[r] = cand[best]
            gap = np.full(7, np.nan)
            for i in range(7):
                rivals = metric[cand[:, i] != cand[best, i]]
                if rivals.size:
                    gap[i] = (metric[best] - rivals.max()) / 2
            if beta is not None:
                fill = beta
            elif np.isfinite(gap).any():
                fill = np.nanmax(gap)
            else:
                fill = np.abs(x).max()
            soft[r] = np.where(np.isnan(gap), fill, gap) * (1 - 2 * decided[r])
        words_out = to_device(encode(res.bits, code=code), "cpu").reshape(num, 7)
        np.testing.assert_array_equal(words_out, decided)
        np.testing.assert_allclose(
            to_device(res.llrs, "cpu").reshape(num, 7), soft, rtol=1e-5, atol=1e-5
        )

    def test_bch_beyond_t(self, xp: Any, xpt: Any) -> None:
        """Three errors in BCH(31, 21) (t = 2) on the least reliable bits:
        hard decoding fails, Chase corrects."""
        code = BCHCode(5, 2)
        u = random_bits((40, code.k), xp, seed=2)
        c = encode(u, code=code)
        llrs = 4.0 * (1.0 - 2.0 * c.astype(xp.float32))
        llrs[:, [3, 11, 20]] *= -0.1
        hard = decode(llrs.reshape(-1), code=code)
        assert not bool((hard.bits == u.reshape(-1)).all())
        soft = decode(llrs.reshape(-1), code=code, method=Chase(4))
        xpt.assert_array_equal(soft.bits, u.reshape(-1))

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="test_positions"):
            Chase(0)
        with pytest.raises(ValueError, match="beta"):
            Chase(beta=-1.0)
        with pytest.raises(TypeError, match="does not decode"):
            decode(
                np.zeros(544 * 10, np.float32),
                code=ReedSolomonCode.kp4(),
                method=Chase(),
            )


class TestStaircaseComponent:
    def test_pi_d_is_a_permutation(self) -> None:
        pi = _pi_d()
        np.testing.assert_array_equal(np.sort(pi), np.arange(510))
        assert pi[0] == 478 and pi[8] == 0 and pi[509] == 477  # Table A.2

    def test_parity_matrix_is_the_reduced_form(self) -> None:
        """Clause A.7.3: H row-reduced with the identity on the right gives [P^T; I]."""
        h = _COMPONENT.parity_check
        assert h.shape == (32, 1022) and rank(h) == 32
        rref, pivots = row_reduce(h, reverse=True)
        np.testing.assert_array_equal(np.sort(pivots), np.arange(990, 1022))
        p = _COMPONENT._parity_matrix
        order = np.argsort(pivots)
        np.testing.assert_array_equal(rref[order][:, :990].T, p)

    def test_corrects_three_and_checks_the_extension(self, xp: Any, xpt: Any) -> None:
        rng = np.random.default_rng(3)
        info = rng.integers(0, 2, (300, 990)).astype(np.int8)
        words = np.concatenate(
            [info, (info @ _COMPONENT._parity_matrix % 2).astype(np.int8)], 1
        )
        assert not ((words.astype(int) @ _COMPONENT.parity_check.T) % 2).any()
        for count in (1, 2, 3):
            noisy = words.copy()
            for r in range(300):
                noisy[r, rng.choice(1022, count, replace=False)] ^= 1
            fixed, ok = _COMPONENT.correct(xp.asarray(noisy))
            xpt.assert_array_equal(fixed, xp.asarray(words))
            assert bool(ok.all())
        noisy = words.copy()
        for r in range(300):
            noisy[r, rng.choice(1022, 4, replace=False)] ^= 1
        fixed, ok = _COMPONENT.correct(xp.asarray(noisy))
        ok = to_device(ok, "cpu")
        fixed = to_device(fixed, "cpu")
        # Every word reported valid satisfies all 32 checks; the others are
        # returned as received.
        assert not ((fixed[ok].astype(int) @ _COMPONENT.parity_check.T) % 2).any()
        np.testing.assert_array_equal(fixed[~ok], noisy[~ok])
        assert ok.mean() < 0.2


class TestStaircase:
    def test_rows_are_component_codewords(self, xp: Any) -> None:
        code = StaircaseCode(2)
        assert (code.n, code.k) == (2 * 512 * 510 + 512 * 32, 2 * 512 * 478)
        c = to_device(encode(random_bits(code.k, xp, seed=4), code=code), "cpu")
        blocks = np.zeros((4, 512, 510), dtype=np.int8)  # B_-1, B_0, B_1, B_2
        blocks[1:3] = c[: 2 * 512 * 510].reshape(2, 512, 510)
        blocks[3, :, 478:] = c[2 * 512 * 510 :].reshape(512, 32)
        for i in range(1, 4):
            left = _left(blocks[i - 1][None], _pi_d())[0]
            word = np.concatenate([left, blocks[i]], axis=1)
            assert not ((word.astype(int) @ _COMPONENT.parity_check.T) % 2).any()

    @pytest.mark.parametrize("num_blocks", [
        pytest.param(2, id="smoke"),
        pytest.param(20, id="validation", marks=pytest.mark.slow),
    ])  # fmt: skip
    def test_threshold_of_g709_2(self, xp: Any, num_blocks: int) -> None:
        """Table A.1: input BER 4.5e-3 decodes (to 1e-15 in the standard);
        well above it, at 6.5e-3, the frame fails.  Smoke run: a frame of 2
        blocks (about 1e6 bits), so it checks the threshold's location, not
        the 1e-15 floor; the slow run has 20 blocks per frame."""
        code = StaircaseCode.g709_2(num_blocks)
        rng = np.random.default_rng(5)
        u = random_bits((2, code.k), xp, seed=6)
        c = encode(u, code=code)
        for ber, decodes in ((4.5e-3, True), (6.5e-3, False)):
            flips = xp.asarray((rng.random(c.shape) < ber).astype(np.int8))
            res = decode(c ^ flips, code=code)
            assert bool((res.bits == u).all()) == decodes
            assert bool(res.valid.all()) == decodes
        assert int(decode(c, code=code).iterations[0, 0]) == 1

    def test_valid_describes_the_final_frame(self, xp: Any, xpt: Any) -> None:
        """Four errors in one row of B_0 exceed its component (t = 3); B_1
        then corrects them through its left halves, within the one pass."""
        code = StaircaseCode(1)
        u = xp.zeros(code.k, dtype=np.int8)
        received = encode(u, code=code)
        received[10 * 510 + xp.arange(20, 24)] ^= 1
        res = decode(received, code=code, method=HardIterations(1))
        xpt.assert_array_equal(res.bits, u)
        assert res.valid.tolist() == [True]

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="blocks_per_frame"):
            StaircaseCode(0)
        with pytest.raises(ValueError, match="max_iterations"):
            HardIterations(0)
        assert repr(StaircaseCode(3)) == "StaircaseCode(blocks_per_frame=3)"


class TestConcatenated:
    def test_layout_and_round_trip(self, xp: Any, xpt: Any) -> None:
        outer = LinearBlockCode.hamming(4)  # (15, 11)
        inner = LinearBlockCode.hamming(3)  # (7, 4): 4 inner words, 1 pad bit
        code = Concatenated(outer, inner, Interleaver.random(15, rng=0))
        assert (code.n, code.k, code.num_inner) == (28, 11, 4)
        u = random_bits(11 * 20, xp, seed=7)
        c = encode(u, code=code)
        xpt.assert_array_equal(decode(c, code=code).bits, u)
        # One error per inner word: the inner code removes all of them.
        noisy = c.reshape(-1, 7).copy()
        noisy[:, 2] ^= 1
        xpt.assert_array_equal(decode(noisy.reshape(-1), code=code).bits, u)

    @pytest.mark.parametrize("num_frames", [
        pytest.param(2, id="smoke"),
        pytest.param(20, id="validation", marks=pytest.mark.slow),
    ])  # fmt: skip
    def test_cfec_threshold(self, backend_device: str, num_frames: int) -> None:
        """OIF 400ZR C-FEC: an input BER of 1.25e-2 gives 1e-15 at the output
        (clause 10).  16-QAM: error-free at a pre-FEC BER of 1.2e-2 (13.6 dB),
        failing at 1.5e-2 (13.2 dB); the pre-FEC BER is checked against Gray
        16-QAM theory first (4 sigma).  The inner decoder is Chase-II over the
        6 least reliable bits: with 4, 20 frames at 13.6 dB kept a post-FEC
        BER of 3e-4.  Smoke run: 2 frames (about 6e5 bits), a broken decoder
        or a misplaced threshold, not the 1e-15 floor; the slow run has 20."""
        code = Concatenated.cfec(num_frames)
        c = mapping.Constellation.qam(16)
        sig = generate(
            c, code.n // 4, symbol_rate=1e9, code=code, rng=8, device=backend_device
        )
        method = Stages(inner=Chase(6, hard=Syndrome(max_weight=1)))
        for esn0_db, decodes in ((13.6, True), (13.2, False)):
            rx = apply_awgn(sig, esn0_db=esn0_db, rng=9)
            p = qam16_gray_ber(10 ** (esn0_db / 10))
            pre = metrics.ber(rx)
            assert abs(pre - p) < binomial_margin(p, code.n), (pre, p)
            res = decode(mapping.compute_llr(rx, noise_var=10 ** (-esn0_db / 10)), code=code,
                         method=method)  # fmt: skip
            post = metrics.ber(res.bits, sig.reference.info_bits)
            assert (post == 0.0) == decodes, (pre, post)

    def test_ieee_802_3dj(self, xp: Any, xpt: Any) -> None:
        code = Concatenated.ieee_802_3dj()
        assert (code.n, code.k) == (136 * 128, 3 * 5140)
        rng = np.random.default_rng(10)
        u = random_bits(code.k, xp, seed=11)
        c = encode(u, code=code)
        llrs = (1.0 - 2.0 * c).astype(xp.float32) * 3.0
        flips = xp.asarray(rng.random(c.shape) < 3e-3)
        llrs = xp.where(flips, -0.5 * llrs, llrs)
        res = decode(
            llrs, code=code, method=Stages(inner=Chase(3, hard=Syndrome(max_weight=1)))
        )
        xpt.assert_array_equal(res.bits, u)

    def test_validation(self) -> None:
        h = LinearBlockCode.hamming(3)
        with pytest.raises(ValueError, match="interleaver"):
            Concatenated(h, h, Interleaver.random(5, rng=0))
        mixed = LinearBlockCode(generator=[[1, 1, 0, 1], [0, 1, 1, 1]])
        with pytest.raises(TypeError, match="not systematic"):
            Concatenated(h, mixed)
        assert "Concatenated(" in repr(Concatenated(h, h))
