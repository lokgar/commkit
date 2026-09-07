"""Tests for constellation geometry and Gray labelling."""

from typing import Any

import numpy as np
import pytest

from commkit import mapping
from commkit.mapping.gray import (
    _gray_ask,
    _gray_psk,
    _gray_qam_cross,
    nearest_constellation_index,
    square_qam_slicer_params,
    unpack_bits,
)


class TestGrayCode:
    """Tests for Gray code generation and binary conversion properties."""

    def test_gray_code_edge_cases(self, xp: Any, xpt: Any) -> None:
        """Verify Gray code generation for boundary bit depths."""
        # n = 0
        xpt.assert_array_equal(mapping.gray_code(0), xp.array([0]))
        xpt.assert_array_equal(mapping.gray_to_binary(0), xp.array([0]))

        # n = 2
        # binary: 0, 1, 2, 3 -> gray: 0, 1, 3, 2
        # gray_to_binary inverse: mapping[0]=0, mapping[1]=1, mapping[3]=2, mapping[2]=3
        xpt.assert_array_equal(mapping.gray_to_binary(2), xp.array([0, 1, 3, 2]))

        # n < 0
        with pytest.raises(ValueError, match="n must be non-negative"):
            mapping.gray_code(-1)
        with pytest.raises(ValueError, match="n must be non-negative"):
            mapping.gray_to_binary(-1)

    def test_gray_code_zero(self, xp: Any, xpt: Any) -> None:
        """Verify Gray code for 0 bits returns single-element zero array."""
        xpt.assert_array_equal(mapping.gray_code(0), xp.array([0]))
        xpt.assert_array_equal(mapping.gray_to_binary(0), xp.array([0]))

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


class TestGrayConstellation:
    """Tests for Gray-coded constellation generation across modulation schemes."""

    def test_gray_constellation_advanced(self, xp: Any) -> None:
        """Verify constellation generation edge cases."""
        # 1. Unipolar via argument
        const_unipol = mapping.gray_constellation("pam", 4, unipolar=True)
        assert xp.min(const_unipol) >= 0

        # 2. Custom scheme check
        const_custom = mapping.gray_constellation("pam", 4)
        assert len(const_custom) == 4

        # 3. Order error
        with pytest.raises(ValueError, match="at least 2"):
            mapping.gray_constellation("psk", 1)

        # 4. QAM non-power-of-2
        with pytest.raises(ValueError, match="power of 2"):
            mapping.gray_constellation("qam", 7)

        # 5. Unknown modulation
        with pytest.raises(ValueError, match="Unsupported modulation type"):
            mapping.gray_constellation("unknown", 4)

    def test_qam_cross_fallback(self) -> None:
        """Trigger the fallback in cross-QAM for small N (e.g. 8-QAM)."""
        res = _gray_qam_cross(8)
        assert res.shape == (8,)

    def test_constellation_unsupported(self) -> None:
        """Verify error for unknown modulation type."""
        with pytest.raises(ValueError, match="Unsupported modulation type"):
            mapping.gray_constellation("chaos", 4)

    def test_constellation_order_error(self) -> None:
        """Verify errors for non-matching orders."""
        with pytest.raises(ValueError, match="Order must be at least 2"):
            mapping.gray_constellation("qam", 1)

        with pytest.raises(ValueError, match="Order must be power of 2"):
            mapping.gray_constellation("qam", 10)

        with pytest.raises(ValueError, match="Order must be power of 2"):
            _gray_psk(3)

        with pytest.raises(ValueError, match="Order must be power of 2"):
            _gray_ask(6)

    def test_constellation_unsupported_string(self) -> None:
        """Test that unsupported strings raise ValueError."""
        with pytest.raises(
            ValueError, match="Unsupported modulation type: custom-unknown"
        ):
            mapping.gray_constellation("custom-unknown", 4)


class TestConstellationSlicing:
    """Tests for square-QAM slicing parameters and nearest-point indexing."""

    def test_square_qam_slicer_params_valid(self) -> None:
        """16-QAM is square-sliceable: side=4, uniform lev_min/d_grid."""
        const = mapping.gray_constellation("qam", 16)
        side, lev_min, d_grid = square_qam_slicer_params(const)
        assert side == 4
        levels = np.unique(np.round(const.real, 6))
        assert np.isclose(lev_min, levels[0])
        assert np.isclose(d_grid, levels[1] - levels[0])

    def test_square_qam_slicer_params_non_square(self) -> None:
        """PSK / cross-QAM constellations fall back to side=0 (O(M) search)."""
        const_psk = mapping.gray_constellation("psk", 8)
        side, _, _ = square_qam_slicer_params(const_psk)
        assert side == 0

        const_cross = mapping.gray_constellation("qam", 32)
        side, _, _ = square_qam_slicer_params(const_cross)
        assert side == 0

    def test_square_qam_slicer_params_non_uniform_grid(self) -> None:
        """Degenerate grid collapsing to fewer than sqrt(M) levels is rejected (side=0)."""
        i_levels = np.array([-3.0, -1.0, 1.0, 1.0])
        q_levels = np.array([-3.0, -1.0, 1.0, 3.0])
        const = (i_levels[:, None] + 1j * q_levels[None, :]).ravel()
        assert len(const) == 16

        side, lev_min, d_grid = square_qam_slicer_params(const)
        assert side == 0
        assert lev_min == np.float32(0.0)
        assert d_grid == np.float32(1.0)

    def test_nearest_constellation_index_matches_unchunked_argmin(
        self, xp: Any, xpt: Any
    ) -> None:
        """Chunked search agrees with plain unchunked argmin on active backend."""
        const_np = mapping.gray_constellation("qam", 16)
        rng = np.random.default_rng(0)
        x_np = const_np[rng.integers(0, 16, size=1000)] + (
            0.01 * rng.standard_normal(1000) + 0.01j * rng.standard_normal(1000)
        )
        x_np = x_np.astype(np.complex64)

        const = xp.asarray(const_np)
        x = xp.asarray(x_np)

        result = nearest_constellation_index(x, const, chunk=7)
        expected = np.argmin(np.abs(x_np[:, None] - const_np[None, :]), axis=1)
        xpt.assert_array_equal(result, xp.asarray(expected))
