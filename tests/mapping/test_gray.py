"""Tests for constellation geometry and Gray labelling."""

from typing import Any

import numpy as np
import pytest

from commkit import mapping
from commkit.mapping.constellation import _slicer_params
from commkit.mapping.gray import (
    _gray_ask,
    _gray_psk,
    _gray_qam_cross,
    _nearest_index,
    _unpack_bits,
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
        """_unpack_bits(indices, k) must match the direct bit-shift idiom it replaced."""
        k = 4
        indices = xp.arange(2**k, dtype=xp.int32)
        indices_np = np.arange(2**k, dtype=np.int32)
        expected = (
            (indices_np[:, None] >> np.arange(k - 1, -1, -1, dtype=np.int32)) & 1
        ).astype(np.int8)

        result = _unpack_bits(indices, k)
        assert result.dtype == xp.int8 or result.dtype == np.int8
        assert result.shape == (2**k, k)
        xpt.assert_array_equal(result, xp.asarray(expected))


class TestGrayConstellation:
    """Tests for Gray-coded constellation generation across modulation schemes."""

    def test_gray_constellation_advanced(self, xp: Any) -> None:
        """Verify constellation generation edge cases."""
        # 1. Unipolar via argument
        const_unipol = mapping.gray._gray_points("pam", 4, unipolar=True)
        assert xp.min(const_unipol) >= 0

        # 2. Custom scheme check
        const_custom = mapping.gray._gray_points("pam", 4)
        assert len(const_custom) == 4

        # 3. Order error
        with pytest.raises(ValueError, match="at least 2"):
            mapping.gray._gray_points("psk", 1)

        # 4. QAM non-power-of-2
        with pytest.raises(ValueError, match="power of 2"):
            mapping.gray._gray_points("qam", 7)

        # 5. Unknown modulation
        with pytest.raises(ValueError, match="Unsupported modulation type"):
            mapping.gray._gray_points("unknown", 4)

    def test_qam_cross_fallback(self) -> None:
        """Trigger the fallback in cross-QAM for small N (e.g. 8-QAM)."""
        res = _gray_qam_cross(8)
        assert res.shape == (8,)

    def test_constellation_unsupported(self) -> None:
        """Verify error for unknown modulation type."""
        with pytest.raises(ValueError, match="Unsupported modulation type"):
            mapping.gray._gray_points("chaos", 4)

    def test_constellation_order_error(self) -> None:
        """Verify errors for non-matching orders."""
        with pytest.raises(ValueError, match="Order must be at least 2"):
            mapping.gray._gray_points("qam", 1)

        with pytest.raises(ValueError, match="Order must be power of 2"):
            mapping.gray._gray_points("qam", 10)

        with pytest.raises(ValueError, match="Order must be power of 2"):
            _gray_psk(3)

        with pytest.raises(ValueError, match="Order must be power of 2"):
            _gray_ask(6)

    def test_constellation_unsupported_string(self) -> None:
        """Test that unsupported strings raise ValueError."""
        with pytest.raises(
            ValueError, match="Unsupported modulation type: custom-unknown"
        ):
            mapping.gray._gray_points("custom-unknown", 4)


class TestConstellationSlicing:
    """Tests for square-QAM slicing parameters and nearest-point indexing."""

    def test_slicer_params_square_qam(self) -> None:
        """16-QAM is square-sliceable: side=4, uniform lev_min/step."""
        c = mapping.Constellation.qam(16)
        side, lev_min, step = _slicer_params(c)
        assert side == 4
        levels = np.unique(np.round(c.points.real, 6))
        assert np.isclose(lev_min, levels[0])
        assert np.isclose(step, levels[1] - levels[0])

    @pytest.mark.parametrize(
        ("re", "im"),
        [
            ([-3.0, -1.0, 1.0, 3.0], [-6.0, -2.0, 2.0, 6.0]),  # other imag grid
            ([-3.0, -1.0, 1.0, 5.0], [-3.0, -1.0, 1.0, 5.0]),  # uneven levels
        ],
        ids=["rectangular", "uneven"],
    )
    def test_slicer_params_reject_non_square_lattices(self, re, im) -> None:
        """sqrt(M) levels per axis are not enough: a slicer that snaps both
        axes to one grid needs the same uniform levels on both."""
        points = (np.array(re)[:, None] + 1j * np.array(im)[None, :]).ravel()
        assert _slicer_params(mapping.Constellation(points))[0] == 0

    def test_slicer_params_non_square(self) -> None:
        """PSK, cross QAM and no constellation fall back to side=0."""
        assert _slicer_params(mapping.Constellation.psk(8))[0] == 0
        assert _slicer_params(mapping.Constellation.qam(32))[0] == 0
        assert _slicer_params(None) == (0, np.float32(0.0), np.float32(1.0))

    def test_nearest_constellation_index_matches_unchunked_argmin(
        self, xp: Any, xpt: Any
    ) -> None:
        """Chunked search agrees with plain unchunked argmin on active backend."""
        const_np = mapping.gray._gray_points("qam", 16)
        rng = np.random.default_rng(0)
        x_np = const_np[rng.integers(0, 16, size=1000)] + (
            0.01 * rng.standard_normal(1000) + 0.01j * rng.standard_normal(1000)
        )
        x_np = x_np.astype(np.complex64)

        const = xp.asarray(const_np)
        x = xp.asarray(x_np)

        result = _nearest_index(x, const, chunk=7)
        expected = np.argmin(np.abs(x_np[:, None] - const_np[None, :]), axis=1)
        xpt.assert_array_equal(result, xp.asarray(expected))


_DECISION_CASES = [
    mapping.Constellation.psk(16),
    mapping.Constellation.qam(128),
    mapping.Constellation.qam(8),
    mapping.Constellation.pam(4),
    # Two rings of 8 points (16-APSK-like), natural-binary labels.
    mapping.Constellation(
        np.concatenate(
            [
                np.exp(2j * np.pi * np.arange(8) / 8),
                2 * np.exp(2j * np.pi * (np.arange(8) + 0.5) / 8),
            ]
        )
    ),
]
_DECISION_IDS = ["16psk", "cross128qam", "8qam", "pam4", "two-rings"]


def _noisy_points(constellation: Any, n: int, seed: int) -> np.ndarray:
    """Points plus noise at the scale of their spacing, single precision."""
    rng = np.random.default_rng(seed)
    pts = constellation.points
    x = pts[rng.integers(0, pts.size, n)] + 0.15 * rng.standard_normal(n)
    if np.iscomplexobj(pts):
        x = x + 0.15j * rng.standard_normal(n)
    return x.astype(np.complex64 if np.iscomplexobj(x) else np.float32)


@pytest.mark.gpu_only
class TestNearestIndexGPU:
    """GPU hard decisions of constellations without a square lattice."""

    @pytest.mark.parametrize("constellation", _DECISION_CASES, ids=_DECISION_IDS)
    def test_kernel_matches_cpu(self, xp: Any, xpt: Any, constellation: Any) -> None:
        x = _noisy_points(constellation, 20000, seed=1)
        pts = constellation.points.astype(constellation._storage_dtype())
        xpt.assert_array_equal(
            _nearest_index(xp.asarray(x), pts), xp.asarray(_nearest_index(x, pts))
        )

    @pytest.mark.parametrize("constellation", _DECISION_CASES, ids=_DECISION_IDS)
    def test_double_input_is_chunked_like_cpu(
        self, xp: Any, xpt: Any, constellation: Any, monkeypatch: Any
    ) -> None:
        """Double input, outside the kernel's float32, is searched in chunks."""
        from commkit.mapping import gray

        monkeypatch.setattr(gray, "_GPU_CHUNK_ELEMENTS", 999 * constellation.order)
        x = _noisy_points(constellation, 5000, seed=2).astype(
            np.complex128 if constellation.is_complex else np.float64
        )
        pts = constellation.points
        xpt.assert_array_equal(
            _nearest_index(xp.asarray(x), pts), xp.asarray(_nearest_index(x, pts))
        )

    def test_demap_matches_cpu(self, xp: Any, xpt: Any) -> None:
        c = mapping.Constellation.psk(8)
        x = _noisy_points(c, 3000, seed=3).reshape(2, -1)
        xpt.assert_array_equal(c.demap(xp.asarray(x)), xp.asarray(c.demap(x)))
