"""Tests for punctured and shortened codes (coding/ratematch.py)."""

import itertools
from typing import Any

import numpy as np
import pytest

from commkit.coding import (
    CRC,
    LDPCCode,
    LinearBlockCode,
    MaximumLikelihood,
    MinSum,
    Punctured,
    Shortened,
    decode,
    encode,
)
from tests.common.coding import random_bits

HAMMING = LinearBlockCode.hamming(4, extended=True)  # (16, 11), distance 4


class TestPunctured:
    def test_drops_positions_and_decodes_erasures(self, xp: Any, xpt: Any) -> None:
        code = Punctured(HAMMING, [15])  # the overall parity bit: Hamming(15, 11)
        assert (code.n, code.k) == (15, 11)
        u = random_bits(11 * 30, xp, seed=1)
        c = encode(u, code=code)
        full = encode(u, code=HAMMING).reshape(30, 16)
        xpt.assert_array_equal(c.reshape(30, 15), full[:, :15])
        # One error per block is still corrected through the erasure.
        llrs = (1.0 - 2.0 * c.reshape(30, 15)).astype(xp.float32)
        llrs[xp.arange(30), xp.arange(30) % 15] *= -1
        res = decode(llrs.reshape(-1), code=code, method=MaximumLikelihood())
        xpt.assert_array_equal(res.bits, u)

    def test_soft_output_keeps_the_layout(self, xp: Any) -> None:
        ldpc = LDPCCode.dvbs2("1/2", "short")
        code = Punctured(ldpc, np.arange(ldpc.n - 100, ldpc.n))
        u = random_bits(code.k, xp, seed=2)
        res = decode(encode(u, code=code), code=code)
        assert res.llrs.shape == (code.n,)
        assert bool(res.valid.all())

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="distinct"):
            Punctured(HAMMING, [1, 1])
        with pytest.raises(ValueError, match="integers in"):
            Punctured(HAMMING, [16])
        with pytest.raises(ValueError, match="whole codeword"):
            Punctured(HAMMING, np.arange(16))
        with pytest.raises(TypeError, match="code object"):
            Punctured("code", [0])
        with pytest.raises(TypeError, match="does not decode"):
            decode(
                np.zeros(15, np.float32), code=Punctured(HAMMING, [15]), method=MinSum()
            )
        assert (
            repr(Punctured(HAMMING, [15]))
            == "Punctured(LinearBlockCode(n=16, k=11), 1 bits)"
        )


class TestShortened:
    @pytest.mark.parametrize(
        "inner",
        [HAMMING, LDPCCode.dvbs2("1/2", "short"), CRC.nr("16", 40)],
        ids=["hamming", "ldpc", "crc"],
    )
    def test_round_trip(self, xp: Any, xpt: Any, inner) -> None:
        code = Shortened(inner, 5)
        assert (code.n, code.k) == (inner.n - 5, inner.k - 5)
        u = random_bits(code.k * 4, xp, seed=3)
        c = encode(u, code=code)
        assert c.shape == (code.n * 4,)
        res = decode(c, code=code)
        xpt.assert_array_equal(res.bits, u)
        assert bool(res.valid.all())

    def test_known_zeros_help_decoding(self, xp: Any, xpt: Any) -> None:
        """Shortened Hamming(16, 11) to (10, 5): ML decoding is over the 32
        words of the subcode, with the 6 shortened information bits known to
        be 0, not over the mother code with them erased.  Both decisions are
        computed here by brute force from the generator (systematic, so the
        shortened bits are positions 5-10 and the sent ones 0-4 and 11-15);
        at Es/N0 near 0 dB they differ on about half the blocks, so a decoder
        that treated the known zeros as erasures fails."""
        code = Shortened(HAMMING, 6)
        kept = np.r_[0:5, 11:16]
        u_all = np.array(list(itertools.product([0, 1], repeat=HAMMING.k)))
        mother = u_all @ HAMMING.generator.astype(int) % 2
        sub = mother[(u_all[:, 5:] == 0).all(axis=1)][:, kept]  # info = first 5
        rng = np.random.default_rng(4)
        num = 200
        words = sub[rng.integers(0, sub.shape[0], num)]
        llrs = (1.5 * (1 - 2 * words) + 1.5 * rng.standard_normal(words.shape)).astype(
            np.float32
        )
        ml = sub[np.argmax((1 - 2 * sub) @ llrs.T, axis=0)][:, :5]
        full = np.zeros((num, HAMMING.n), np.float32)
        full[:, kept] = llrs
        erased = mother[np.argmax((1 - 2 * mother) @ full.T, axis=0)][:, :5]
        assert (ml != erased).any(axis=1).sum() > num // 4  # the test can tell
        res = decode(
            xp.asarray(llrs.reshape(-1)), code=code, method=MaximumLikelihood()
        )
        xpt.assert_array_equal(
            res.bits, xp.asarray(ml.reshape(-1), dtype=res.bits.dtype)
        )

    def test_info_positions_of_nested_wrappers(self, xp: Any, xpt: Any) -> None:
        ldpc = LDPCCode.dvbs2("1/2", "short")
        code = Shortened(Punctured(ldpc, np.arange(ldpc.n - 50, ldpc.n)), 100)
        u = random_bits(code.k, xp, seed=5)
        c = encode(u, code=code)
        xpt.assert_array_equal(c[: code.k], u)  # still systematic
        xpt.assert_array_equal(decode(c, code=code).bits, u)

    def test_validation(self) -> None:
        mixed = LinearBlockCode(generator=[[1, 1, 0, 1], [0, 1, 1, 1]])
        with pytest.raises(TypeError, match="not systematic"):
            Shortened(mixed, 1)
        with pytest.raises(TypeError, match="not systematic"):
            Shortened(Punctured(mixed, [3]), 1)
        with pytest.raises(ValueError, match="num_bits"):
            Shortened(HAMMING, 11)
        punctured_info = Punctured(HAMMING, [0])
        with pytest.raises(TypeError, match="not systematic"):
            Shortened(punctured_info, 1)
        assert "Shortened(" in repr(Shortened(HAMMING, 1))
