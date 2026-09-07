"""Tests for diagnostic/analysis smoothers (non-causal, plotting/estimation-only)."""

from typing import Any

import pytest

from commkit import smoothing


class TestMovingAverage:
    """Tests for 1D moving average boxcar filter modes."""

    def test_moving_average_same_mode(self, xp: Any, xpt: Any) -> None:
        """mode='same' returns an edge-aware boxcar average, same length as input."""
        x = xp.ones(64)
        out = smoothing.moving_average(x, window=5, mode="same")
        assert out.shape == x.shape
        xpt.assert_allclose(out, x, rtol=1e-6)

    def test_moving_average_valid_mode_shrinks(self, xp: Any, xpt: Any) -> None:
        """mode='valid' shrinks the output by window - 1."""
        n, window = 64, 5
        x = xp.arange(n, dtype=xp.float64)
        out = smoothing.moving_average(x, window=window, mode="valid")
        assert out.shape[-1] == n - (window - 1)
        expected = xp.arange((window - 1) / 2, n - (window - 1) / 2)
        xpt.assert_allclose(out, expected, rtol=1e-6)

    def test_moving_average_invalid_mode_raises(self, xp: Any) -> None:
        """Unknown mode raises ValueError."""
        x = xp.ones(16)
        with pytest.raises(ValueError):
            smoothing.moving_average(x, window=3, mode="bogus")


class TestPolynomialAndDensitySmoothing:
    """Tests for Savitzky-Golay filtering and 2D Gaussian density smoothing."""

    def test_savgol_smooth_matches_input_shape(self, xp: Any, xpt: Any) -> None:
        """Savitzky-Golay smoothing preserves shape and fits a low-order polynomial exactly."""
        n = 101
        t = xp.linspace(-1, 1, n)
        x = 2.0 + 3.0 * t - t**2
        out = smoothing.savgol_smooth(x, window=11, polyorder=2)
        assert out.shape == x.shape
        xpt.assert_allclose(out, x, atol=1e-8)

    def test_smooth_density_2d_preserves_shape_and_mass(self, xp: Any) -> None:
        """Gaussian blur preserves array shape and roughly preserves total mass."""
        hist = xp.zeros((32, 32))
        hist[16, 16] = 100.0
        out = smoothing.smooth_density_2d(hist, sigma=1.0)
        assert out.shape == hist.shape
        total_before = float(xp.sum(hist))
        total_after = float(xp.sum(out))
        assert total_after == pytest.approx(total_before, rel=0.05)
