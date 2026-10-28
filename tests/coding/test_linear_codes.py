"""Tests for linear block codes and their decoders (coding/linear.py)."""

from itertools import product
from typing import Any

import numpy as np
import pytest

from commkit.backend import to_device
from commkit.coding import (
    LinearBlockCode,
    MaximumLikelihood,
    Syndrome,
    decode,
    encode,
)
from commkit.coding import linear as linear_module
from commkit.coding._gf2 import rank
from tests.common.coding import random_bits


def _codewords(code: LinearBlockCode) -> np.ndarray:
    """All 2^k codewords, rows in the order of the information words."""
    u = np.array(list(product([0, 1], repeat=code.k)), dtype=int)
    return (u @ code.generator) % 2


def _min_distance(code: LinearBlockCode) -> int:
    return int(_codewords(code)[1:].sum(axis=1).min())


def _no_three_columns_sum_to_zero(h: np.ndarray) -> bool:
    """Distance >= 4: columns distinct and nonzero, and no column is the sum
    of two others (invariant under row operations on H)."""
    cols = [int("".join(map(str, c)), 2) for c in h.T.astype(int)]
    if 0 in cols or len(set(cols)) != len(cols):
        return False
    s = set(cols)
    return not any((cols[i] ^ cols[j]) in s for i in range(len(cols)) for j in range(i))


class TestConstruction:
    @pytest.mark.parametrize("r", [2, 3, 4, 5, 6])
    def test_hamming(self, r: int) -> None:
        code = LinearBlockCode.hamming(r)
        assert (code.n, code.k) == (2**r - 1, 2**r - 1 - r)
        assert not np.any((code.generator.astype(int) @ code.parity_check.T) % 2)
        np.testing.assert_array_equal(code.generator[:, : code.k], np.eye(code.k))
        if code.k <= 12:
            assert _min_distance(code) == 3

    @pytest.mark.parametrize("r", [3, 4])
    def test_extended_hamming_distance_4(self, r: int) -> None:
        code = LinearBlockCode.hamming(r, extended=True)
        assert (code.n, code.k) == (2**r, 2**r - 1 - r)
        assert _min_distance(code) == 4

    @pytest.mark.parametrize(
        ("r", "length", "nk"), [(8, 128, (128, 119)), (7, None, (128, 120))]
    )
    def test_optical_inner_codes(self, r, length, nk) -> None:
        """The SECDED inner codes of 400ZR and IEEE 802.3dj: distance 4."""
        code = LinearBlockCode.hamming(r, extended=True, length=length)
        assert (code.n, code.k) == nk
        assert _no_three_columns_sum_to_zero(code.parity_check)

    def test_shortened_keeps_distance(self) -> None:
        code = LinearBlockCode.hamming(4, length=11)
        assert (code.n, code.k) == (11, 7)
        assert _min_distance(code) == 3

    def test_from_parity_check(self) -> None:
        rng = np.random.default_rng(0)
        h = rng.integers(0, 2, (5, 14))
        h[4] = h[0] ^ h[1]  # dependent row: k = n - rank
        code = LinearBlockCode.from_parity_check(h)
        assert code.k == 14 - rank(h)
        assert not np.any((code.generator.astype(int) @ h.T) % 2)

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            LinearBlockCode(generator=np.ones(4))
        with pytest.raises(ValueError, match="only 0 and 1"):
            LinearBlockCode(generator=[[1, 2, 0]])
        with pytest.raises(ValueError, match="full row rank"):
            LinearBlockCode(generator=[[1, 1, 0], [1, 1, 0]])
        with pytest.raises(ValueError, match="r must be"):
            LinearBlockCode.hamming(1)
        with pytest.raises(ValueError, match="length"):
            LinearBlockCode.hamming(3, length=3)
        with pytest.raises(ValueError, match="empty"):
            LinearBlockCode.from_parity_check(np.eye(3))
        with pytest.raises(ValueError, match="binary"):
            LinearBlockCode.from_parity_check([[0, 2]])
        with pytest.raises(ValueError, match="max_weight"):
            Syndrome(max_weight=-1)

    def test_repr(self) -> None:
        assert repr(LinearBlockCode.hamming(3)) == "LinearBlockCode(n=7, k=4)"


class TestEncoding:
    def test_codewords_satisfy_parity(self, xp: Any) -> None:
        code = LinearBlockCode.hamming(8, extended=True, length=128)
        c = to_device(encode(random_bits(119 * 20, xp), code=code), "cpu")
        assert not np.any((c.reshape(20, 128).astype(int) @ code.parity_check.T) % 2)

    def test_non_systematic_generator_round_trip(self, xp: Any, xpt: Any) -> None:
        """Decoding returns u with u G = c also when G is not systematic."""
        base = LinearBlockCode.hamming(3)
        mix = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1]])
        code = LinearBlockCode(generator=(mix @ base.generator) % 2)
        u = random_bits(4 * 50, xp, seed=2)
        c = encode(u, code=code)
        for method in (Syndrome(), MaximumLikelihood()):
            xpt.assert_array_equal(decode(c, code=code, method=method).bits, u)


class TestSyndrome:
    @pytest.mark.parametrize(
        "code",
        [
            LinearBlockCode.hamming(3),
            LinearBlockCode.hamming(4, extended=True),
            LinearBlockCode.hamming(7, extended=True),
        ],
        ids=str,
    )
    def test_corrects_every_single_error(self, xp: Any, xpt: Any, code) -> None:
        n = code.n
        u = random_bits((n, code.k), xp, seed=3)
        c = encode(u, code=code)
        errors = xp.eye(n, dtype=np.int8)  # block i has its error at position i
        res = decode((c ^ errors).reshape(-1), code=code)
        xpt.assert_array_equal(res.bits, u.reshape(-1))
        assert bool(res.valid.all())
        assert bool((res.iterations == 1).all())

    def test_extended_detects_double_errors(self, xp: Any, xpt: Any) -> None:
        code = LinearBlockCode.hamming(4, extended=True)
        pairs = [(i, j) for i in range(16) for j in range(i)]
        u = random_bits((len(pairs), code.k), xp, seed=4)
        c = to_device(encode(u, code=code), "cpu")
        for row, (i, j) in enumerate(pairs):
            c[row, [i, j]] ^= 1
        res = decode(
            xp.asarray(c.reshape(-1)), code=code, method=Syndrome(max_weight=1)
        )
        assert not bool(res.valid.any())
        # Uncorrected: the information bits of the hard decisions.
        xpt.assert_array_equal(res.bits, xp.asarray(c[:, : code.k].reshape(-1)))

    def test_complete_decoding_is_minimum_distance(self, xp: Any, xpt: Any) -> None:
        """Without max_weight every syndrome decodes to a nearest codeword."""
        code = LinearBlockCode.hamming(4, extended=True)
        words = _codewords(code)
        received = random_bits((200, code.n), xp, seed=5)
        res = decode(received.reshape(-1), code=code)
        assert bool(res.valid.all())
        decoded = (
            to_device(res.bits, "cpu").reshape(200, -1).astype(int) @ code.generator
        ) % 2
        rx = to_device(received, "cpu").astype(int)
        nearest = np.abs(rx[:, None, :] - words[None]).sum(-1).min(axis=1)
        np.testing.assert_array_equal(np.abs(rx - decoded).sum(-1), nearest)

    def test_table_limits(self, monkeypatch) -> None:
        rng = np.random.default_rng(6)
        wide = LinearBlockCode(
            generator=np.concatenate(
                [np.eye(10, dtype=int), rng.integers(0, 2, (10, 30))], axis=1
            )
        )
        with pytest.raises(ValueError, match="table"):
            decode(np.zeros(40, np.int8), code=wide)
        monkeypatch.setattr(linear_module, "_MAX_PATTERNS", 10)
        with pytest.raises(ValueError, match="max_weight"):
            # A code no other test decodes: its table is not cached yet.
            decode(
                np.zeros(32, np.int8), code=LinearBlockCode.hamming(5, extended=True)
            )


class TestMaximumLikelihood:
    @pytest.mark.parametrize(
        "code",
        [
            LinearBlockCode.hamming(3),
            LinearBlockCode(
                generator=np.concatenate(
                    [
                        np.eye(6, dtype=int),
                        np.random.default_rng(7).integers(0, 2, (6, 6)),
                    ],
                    axis=1,
                )
            ),
        ],
        ids=str,
    )
    def test_equals_brute_force_argmax(
        self, xp: Any, xpt: Any, code, monkeypatch
    ) -> None:
        rng = np.random.default_rng(8)
        llrs = rng.standard_normal((300, code.n)).astype(np.float32)
        words = _codewords(code)
        best = np.argmax(llrs @ (1 - 2 * words).T, axis=1)
        u = np.array(list(product([0, 1], repeat=code.k)))[best]
        res = decode(
            xp.asarray(llrs.reshape(-1)), code=code, method=MaximumLikelihood()
        )
        xpt.assert_array_equal(res.bits, xp.asarray(u.reshape(-1).astype(np.int8)))
        # Chunked over rows and codewords: same decisions.
        monkeypatch.setattr(linear_module, "_ML_CHUNK", 64)
        res = decode(
            xp.asarray(llrs.reshape(-1)), code=code, method=MaximumLikelihood()
        )
        xpt.assert_array_equal(res.bits, xp.asarray(u.reshape(-1).astype(np.int8)))

    def test_k_limit(self) -> None:
        with pytest.raises(ValueError, match="exceeds"):
            decode(
                np.zeros(31, np.float32),
                code=LinearBlockCode.hamming(5),
                method=MaximumLikelihood(),
            )
