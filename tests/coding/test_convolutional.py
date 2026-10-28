"""Tests for convolutional codes, Viterbi and BCJR (coding/convolutional.py)."""

import itertools
from typing import Any

import numpy as np
import pytest

from commkit.backend import to_device
from commkit.coding import BCJR, ConvolutionalCode, Viterbi, decode, encode
from tests.common.coding import random_bits


def _all_words(code: ConvolutionalCode) -> tuple[np.ndarray, np.ndarray]:
    info = np.array(list(itertools.product([0, 1], repeat=code.k)), dtype=np.int8)
    return info, to_device(encode(info, code=code), "cpu").astype(float)


class TestEncoding:
    def test_k7_is_polynomial_multiplication(self, xp: Any) -> None:
        """Each output stream is u(D) g(D) over GF(2); outputs interleaved."""
        code = ConvolutionalCode.k7(30)
        u = random_bits(30, xp, seed=1)
        c = to_device(encode(u, code=code), "cpu").reshape(-1, 2)
        assert c.shape == (36, 2)
        for j, g in enumerate((0o133, 0o171)):
            taps = [(g >> (6 - d)) & 1 for d in range(7)]
            stream = np.convolve(to_device(u, "cpu").astype(int), taps) % 2
            np.testing.assert_array_equal(c[:, j], stream)

    def test_tail_biting_wraps_around(self, xp: Any) -> None:
        code = ConvolutionalCode((0o7, 0o5), 3, 12, "tail-biting")
        u = to_device(random_bits(12, xp, seed=2), "cpu").astype(int)
        c = to_device(encode(xp.asarray(u.astype(np.int8)), code=code), "cpu").reshape(
            -1, 2
        )
        assert c.shape == (12, 2)
        for j, g in enumerate((0o7, 0o5)):
            taps = [(g >> (2 - d)) & 1 for d in range(3)]
            expect = [
                sum(taps[d] * u[(t - d) % 12] for d in range(3)) % 2 for t in range(12)
            ]
            np.testing.assert_array_equal(c[:, j], expect)


class TestDecoding:
    @pytest.mark.parametrize(
        "code",
        [ConvolutionalCode.k7(9), ConvolutionalCode((0o7, 0o5), 3, 10, "tail-biting")],
        ids=["k7-zero", "k3-tail-biting"],
    )
    def test_viterbi_is_maximum_likelihood(self, xp: Any, code) -> None:
        info, words = _all_words(code)
        llrs = (
            np.random.default_rng(3).standard_normal((150, code.n)).astype(np.float32)
        )
        best = info[np.argmax(llrs @ (1 - 2 * words).T, axis=1)]
        got = decode(xp.asarray(llrs), code=code, method=Viterbi()).bits
        np.testing.assert_array_equal(to_device(got, "cpu").reshape(150, -1), best)

    def test_bcjr_is_bitwise_map(self, xp: Any) -> None:
        """Exact BCJR: a-posteriori LLRs of the coded bits equal brute-force
        MAP; max-log decisions equal Viterbi's."""
        code = ConvolutionalCode((0o7, 0o5), 3, 8)
        _, words = _all_words(code)
        llrs = (
            np.random.default_rng(4).standard_normal((100, code.n)).astype(np.float32)
        )
        metric = -(llrs.astype(float) @ words.T)
        expect = np.stack(
            [
                np.logaddexp.reduce(metric[:, words[:, j] == 0], axis=1)
                - np.logaddexp.reduce(metric[:, words[:, j] == 1], axis=1)
                for j in range(code.n)
            ],
            axis=1,
        )
        x = xp.asarray(llrs)
        exact = decode(x, code=code, method=BCJR(exact=True))
        np.testing.assert_allclose(
            to_device(exact.llrs, "cpu").reshape(100, -1), expect, atol=1e-4
        )
        maxlog = decode(x, code=code, method=BCJR())
        viterbi = decode(x, code=code, method=Viterbi())
        np.testing.assert_array_equal(
            to_device(maxlog.bits, "cpu"), to_device(viterbi.bits, "cpu")
        )

    def test_noiseless(self, xp: Any, xpt: Any) -> None:
        code = ConvolutionalCode.k7(200)
        u = random_bits((3, 200), xp, seed=5)
        xpt.assert_array_equal(decode(encode(u, code=code), code=code).bits, u)

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="constraint_length"):
            ConvolutionalCode((0o7, 0o5), 1, 10)
        with pytest.raises(ValueError, match="generators"):
            ConvolutionalCode((0o7,), 3, 10)
        with pytest.raises(ValueError, match="generators"):
            ConvolutionalCode((0o17, 0o5), 3, 10)
        with pytest.raises(ValueError, match="block_length"):
            ConvolutionalCode((0o7, 0o5), 3, 0)
        with pytest.raises(ValueError, match="termination"):
            ConvolutionalCode((0o7, 0o5), 3, 10, "flush")
        with pytest.raises(ValueError, match="at least K - 1"):
            ConvolutionalCode((0o7, 0o5), 3, 1, "tail-biting")
        tb = ConvolutionalCode((0o7, 0o5), 3, 4, "tail-biting")
        with pytest.raises(ValueError, match="Viterbi"):
            decode(np.zeros(8, np.float32), code=tb, method=BCJR())
        assert (
            repr(ConvolutionalCode.k7(5))
            == "ConvolutionalCode((0o133, 0o171), K=7, k=5)"
        )
