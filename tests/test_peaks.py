"""Tests for sub-bin peak location (commkit/_peaks.py)."""

import numpy as np
import pytest

from commkit._peaks import parabolic_peak_offset


class TestParabolicPeakOffset:
    def test_parabolic_peak_offset_recovers_known_offset(self, xp):
        """A synthetic parabola with a known sub-bin peak must be recovered exactly."""
        k_true = 2.3
        nearest = round(k_true)
        offset_true = k_true - nearest

        def y(k):
            return -((k - k_true) ** 2) + 10.0

        y_prev, y_curr, y_next = y(nearest - 1), y(nearest), y(nearest + 1)
        delta = parabolic_peak_offset(
            xp.asarray(y_prev), xp.asarray(y_curr), xp.asarray(y_next), xp, log=False
        )
        assert float(delta) == pytest.approx(offset_true, abs=1e-9)

    def test_parabolic_peak_offset_degenerate_denom_returns_zero(self, xp):
        """A flat triplet must return delta=0, not NaN/Inf."""
        y_prev = xp.asarray(1.0)
        y_curr = xp.asarray(1.0)
        y_next = xp.asarray(1.0)
        delta = parabolic_peak_offset(y_prev, y_curr, y_next, xp, log=False)
        assert float(delta) == 0.0

    def test_parabolic_peak_offset_log_mode_host_scalars(self):
        """log=True must work on plain host scalars with xp=numpy."""
        delta = parabolic_peak_offset(0.5, 1.0, 0.6, np, log=True)
        assert isinstance(float(delta), float)
        assert -0.5 <= float(delta) <= 0.5
