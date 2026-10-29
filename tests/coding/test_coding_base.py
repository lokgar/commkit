"""Tests for the encode/decode verbs and DecodeResult (coding/base.py)."""

import warnings
from typing import Any

import numpy as np
import pytest

from commkit import PrecisionWarning, backend
from commkit.coding import (
    LinearBlockCode,
    MaximumLikelihood,
    Syndrome,
    decode,
    encode,
)
from tests.common.coding import random_bits

H74 = LinearBlockCode.hamming(3)


class TestEncode:
    def test_layout_per_channel(self, xp: Any, xpt: Any) -> None:
        """(C, B*k) -> (C, B*n); each channel's blocks encode on their own."""
        u = random_bits((3, 4 * 5), xp)
        c = encode(u, code=H74)
        assert c.shape == (3, 7 * 5)
        assert c.dtype == np.int8
        for ch in range(3):
            xpt.assert_array_equal(c[ch], encode(u[ch], code=H74))

    def test_bool_input(self, xp: Any, xpt: Any) -> None:
        u = random_bits(8, xp)
        xpt.assert_array_equal(encode(u.astype(bool), code=H74), encode(u, code=H74))

    def test_empty(self, xp: Any) -> None:
        out = encode(xp.zeros((2, 0), dtype=np.int8), code=H74)
        assert out.shape == (2, 0)

    def test_errors(self, xp: Any) -> None:
        with pytest.raises(TypeError, match="code must be a code object"):
            encode(random_bits(8, xp), code="hamming")
        with pytest.raises(ValueError, match="multiple of the block length 4"):
            encode(random_bits(6, xp), code=H74)
        with pytest.raises(TypeError, match="integers or bool"):
            encode(xp.zeros(8, dtype=xp.float32), code=H74)
        with pytest.raises(ValueError):
            encode(xp.zeros((2, 2, 4), dtype=np.int8), code=H74)

    @pytest.mark.parametrize(
        "bits",
        [[2, 0, 0, 0], [0, -1, 0, 0], [0, 0, 256, 0]],
        ids=["two", "minus-one", "wraps-to-zero-in-int8"],
    )
    def test_nonbinary_bits(self, xp: Any, bits: list[int]) -> None:
        """Values other than 0 and 1 raise instead of encoding another payload."""
        with pytest.raises(ValueError, match="0 or 1"):
            encode(xp.asarray(bits, dtype=xp.int16), code=H74)


class TestDecode:
    def test_result_layout(self, xp: Any) -> None:
        llrs = xp.ones((2, 7 * 3), dtype=xp.float32)
        res = decode(llrs, code=H74)
        assert res.bits.shape == (2, 12) and res.bits.dtype == np.int8
        assert res.valid.shape == (2, 3) and res.valid.dtype == bool
        assert res.iterations.shape == (2, 3) and res.iterations.dtype == np.int32
        assert res.llrs is None
        res1 = decode(llrs[0], code=H74)
        assert res1.bits.shape == (12,) and res1.valid.shape == (3,)

    def test_hard_input_is_the_sign(self, xp: Any, xpt: Any) -> None:
        """Integer bits decode as LLRs of sign (1 - 2b): ML on hard input is
        minimum Hamming distance."""
        bits = random_bits(7 * 40, xp, seed=1)
        llrs = (1.0 - 2.0 * bits).astype(xp.float32)
        for method in (Syndrome(), MaximumLikelihood()):
            a = decode(bits, code=H74, method=method)
            b = decode(llrs, code=H74, method=method)
            xpt.assert_array_equal(a.bits, b.bits)

    def test_double_warns(self, xp: Any) -> None:
        with pytest.warns(PrecisionWarning):
            decode(xp.ones(7, dtype=xp.float64), code=H74)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            decode(xp.ones(7, dtype=xp.float32), code=H74)

    def test_empty(self, xp: Any) -> None:
        res = decode(xp.zeros((2, 0), dtype=xp.float32), code=H74)
        assert res.bits.shape == (2, 0) and res.valid.shape == (2, 0)

    def test_errors(self, xp: Any) -> None:
        llrs = xp.ones(7, dtype=xp.float32)
        with pytest.raises(TypeError, match="code must be a code object"):
            decode(llrs, code=None)
        with pytest.raises(TypeError, match="does not decode"):
            decode(llrs, code=H74, method="syndrome")
        with pytest.raises(ValueError, match="multiple of the block length 7"):
            decode(xp.ones(8, dtype=xp.float32), code=H74)
        with pytest.raises(TypeError, match="real"):
            decode(xp.ones(7, dtype=xp.complex64), code=H74)


class TestCodeValue:
    def test_equality_and_hash(self) -> None:
        a = LinearBlockCode.hamming(4)
        b = LinearBlockCode(generator=a.generator.copy())
        assert a == b and hash(a) == hash(b)
        assert a != LinearBlockCode.hamming(4, extended=True)
        assert a != "code"
        assert a.rate == pytest.approx(11 / 15)

    def test_arrays_are_read_only(self) -> None:
        with pytest.raises(ValueError):
            H74.generator[0, 0] = 1

    def test_equal_codes_share_device_tables(self, xp: Any) -> None:
        a = LinearBlockCode.hamming(5)
        b = LinearBlockCode(generator=a.generator.copy())
        encode(random_bits(26, xp), code=a)
        tables = dict(backend._DEVICE_TABLES[a])
        encode(random_bits(26, xp), code=b)
        assert backend._DEVICE_TABLES[b].keys() == tables.keys()
        for key, value in tables.items():
            assert backend._DEVICE_TABLES[b][key] is value
