"""Tests for the remaining helpers (peak interpolation, ZC roots, linear trend)."""

import numpy as np
import pytest

from commkit import helpers


class TestParabolicAndSequences:
    """Parabolic peak interpolation and MIMO ZC roots."""

    def test_parabolic_peak_offset_recovers_known_offset(self, xp):
        """A synthetic parabola with a known sub-bin peak must be recovered exactly."""
        k_true = 2.3
        nearest = round(k_true)
        offset_true = k_true - nearest

        def y(k):
            return -((k - k_true) ** 2) + 10.0

        y_prev, y_curr, y_next = y(nearest - 1), y(nearest), y(nearest + 1)
        delta = helpers._parabolic_peak_offset(
            xp.asarray(y_prev), xp.asarray(y_curr), xp.asarray(y_next), xp, log=False
        )
        assert float(delta) == pytest.approx(offset_true, abs=1e-9)

    def test_parabolic_peak_offset_degenerate_denom_returns_zero(self, xp):
        """A flat triplet must return delta=0, not NaN/Inf."""
        y_prev = xp.asarray(1.0)
        y_curr = xp.asarray(1.0)
        y_next = xp.asarray(1.0)
        delta = helpers._parabolic_peak_offset(y_prev, y_curr, y_next, xp, log=False)
        assert float(delta) == 0.0

    def test_parabolic_peak_offset_log_mode_host_scalars(self):
        """log=True must work on plain host scalars with xp=numpy."""
        delta = helpers._parabolic_peak_offset(0.5, 1.0, 0.6, np, log=True)
        assert isinstance(float(delta), float)
        assert -0.5 <= float(delta) <= 0.5

    def test_zc_mimo_root(self, xp):
        """zc_mimo_root assigns distinct roots cycling from base_root in [1, length-1]."""
        from commkit.helpers import zc_mimo_root

        assert zc_mimo_root(0, 1, 13) == 1
        assert zc_mimo_root(1, 1, 13) == 2
        assert zc_mimo_root(2, 1, 13) == 3

        assert zc_mimo_root(0, 10, 13) == 10
        assert zc_mimo_root(1, 10, 13) == 11
        assert zc_mimo_root(2, 10, 13) == 12
        assert zc_mimo_root(3, 10, 13) == 1

        for k in range(12):
            r = zc_mimo_root(k, 1, 13)
            assert 1 <= r <= 12


class TestLinearTrend:
    """Estimation and removal of linear phase/carrier ramps."""

    def test_linear_trend_slope_per_sample(self, xp, xpt):
        """linear_trend_slope: recovers a known per-channel slope in units/sample."""
        n = 512
        idx = np.arange(n, dtype=np.float64)
        y = np.stack([0.25 * idx + 3.0, -0.75 * idx - 11.0])
        slope = helpers.linear_trend_slope(xp.asarray(y))
        xpt.assert_allclose(slope, xp.asarray(np.array([0.25, -0.75])), rtol=1e-9)

    def test_linear_trend_slope_with_explicit_axis(self, xp, xpt):
        """linear_trend_slope: a non-uniform x axis gives a slope per unit x."""
        x = np.array([0.0, 1.0, 4.0, 9.0, 16.0])
        y = (2.0 * x + 5.0)[None, :]
        slope = helpers.linear_trend_slope(xp.asarray(y), x=xp.asarray(x))
        xpt.assert_allclose(slope, xp.asarray(np.array([2.0])), rtol=1e-9)

    def test_linear_trend_slope_stays_on_device(self, xp):
        """linear_trend_slope: the result is a device array (no implicit transfer)."""
        y = xp.asarray(np.random.default_rng(1).normal(size=(2, 64)))
        assert isinstance(helpers.linear_trend_slope(y), xp.ndarray)

    def test_remove_linear_trend_strips_ramp_and_keeps_mean(self, xp, xpt):
        """remove_linear_trend: ramp removed, mean preserved, slope reported."""
        n = 1024
        idx = np.arange(n, dtype=np.float64)
        rng = np.random.default_rng(7)
        fluct = rng.normal(scale=0.01, size=n)
        y = 0.05 * idx + 2.0 + fluct
        y2 = xp.asarray(y[None, :])

        detrended, slope = helpers.remove_linear_trend(y2)
        xpt.assert_allclose(slope, xp.asarray(np.array([0.05])), atol=1e-4)
        assert float(xp.mean(detrended)) == pytest.approx(float(np.mean(y)), abs=1e-9)
        xpt.assert_allclose(
            detrended[0] - float(np.mean(y)),
            xp.asarray(fluct - fluct.mean()),
            atol=5e-3,
        )

    def test_remove_linear_trend_degenerate_length(self, xp):
        """remove_linear_trend: a single-sample record does not divide by zero."""
        y = xp.asarray(np.array([[4.0]]))
        detrended, slope = helpers.remove_linear_trend(y)
        assert np.isfinite(float(slope[0]))
        assert float(detrended[0, 0]) == 4.0
