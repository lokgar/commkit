"""Tests for the private bit packing helpers (commkit/_bits.py)."""

from typing import Any

import numpy as np

from commkit._bits import pack_bits, unpack_bits


class TestBits:
    def test_unpack_bits_matches_manual_shift(self, xp: Any, xpt: Any) -> None:
        """unpack_bits(indices, k) must match the direct bit-shift idiom it replaced."""
        k = 4
        indices = xp.arange(2**k, dtype=xp.int32)
        indices_np = np.arange(2**k, dtype=np.int32)
        expected = (
            (indices_np[:, None] >> np.arange(k - 1, -1, -1, dtype=np.int32)) & 1
        ).astype(np.int8)

        result = unpack_bits(indices, k)
        assert result.dtype == xp.int8 or result.dtype == np.int8
        assert result.shape == (2**k, k)
        xpt.assert_array_equal(result, xp.asarray(expected))

    def test_pack_inverts_unpack(self, xp: Any, xpt: Any) -> None:
        """pack_bits(unpack_bits(i, k)) gives back i, MSB first, int64."""
        indices = xp.arange(32, dtype=xp.int64)
        packed = pack_bits(unpack_bits(indices, 5))
        assert packed.dtype == xp.int64
        xpt.assert_array_equal(packed, indices)
        xpt.assert_array_equal(pack_bits(xp.asarray([[1, 0, 0]])), xp.asarray([4]))
