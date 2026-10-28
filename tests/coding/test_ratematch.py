"""Tests for punctured and shortened codes (coding/ratematch.py)."""

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
        """Two errors in a shortened Hamming(16, 11) block: ML with the known
        zeros corrects what the unshortened code could not see."""
        code = Shortened(HAMMING, 6)  # (10, 5)
        u = random_bits(5 * 20, xp, seed=4)
        c = encode(u, code=code).reshape(20, 10)
        llrs = (2.0 * (1.0 - 2.0 * c)).astype(xp.float32)
        llrs[:, 0] *= -0.2  # one weak wrong bit per block
        res = decode(llrs.reshape(-1), code=code, method=MaximumLikelihood())
        xpt.assert_array_equal(res.bits, u)

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
