"""Tests for the square-lattice structure of constellations (mapping/_lattice.py)."""

import numpy as np
import pytest

from commkit.mapping import Constellation
from commkit.mapping._lattice import separable_axes, slicer_params, square_grid


class TestSquareGrid:
    def test_square_qam_lut_indexes_the_points(self) -> None:
        """lut[i * side + q] is the point at level i on the real axis and q
        on the imaginary one."""
        c = Constellation.qam(64)
        lev_min, step, side, lut = square_grid(c)
        i, q = np.divmod(np.arange(c.order), side)
        expected = (lev_min + i * step) + 1j * (lev_min + q * step)
        np.testing.assert_allclose(c.points[lut], expected, atol=1e-12)

    def test_gray_qam_is_separable_and_psk_is_not(self) -> None:
        """Gray QAM factors into two axes of sqrt(M) levels; PSK has no grid."""
        axes = separable_axes(Constellation.qam(16))
        assert axes is not None
        assert [a.levels.size for a in axes] == [4, 4]
        assert square_grid(Constellation.psk(8)) is None
        assert separable_axes(Constellation.psk(8)) is None


class TestSlicerParams:
    def test_slicer_params_square_qam(self) -> None:
        """16-QAM is square-sliceable: side=4, uniform lev_min/step."""
        c = Constellation.qam(16)
        side, lev_min, step = slicer_params(c)
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
        assert slicer_params(Constellation(points))[0] == 0

    def test_slicer_params_non_square(self) -> None:
        """PSK, cross QAM and no constellation fall back to side=0."""
        assert slicer_params(Constellation.psk(8))[0] == 0
        assert slicer_params(Constellation.qam(32))[0] == 0
        assert slicer_params(None) == (0, np.float32(0.0), np.float32(1.0))
