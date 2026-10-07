"""Tests for the overlapping Allan deviation estimator."""

from typing import Any

import numpy as np
import pytest

from commkit import analysis

R = 32e9


class TestAllanDeviation:
    """Tests for overlapping Allan deviation estimation across noise regimes."""

    def test_allan_deviation_white_fm_slope(self, xp: Any) -> None:
        """White-FM frequency noise yields an Allan deviation slope of approximately -1/2 in log-log."""
        n = 1 << 16
        rng = np.random.default_rng(31)
        df = rng.normal(0, 1e5, n)
        out = analysis.allan_deviation(xp.asarray(df), sampling_rate=R, num_taus=20)
        tau, adev = out.tau_s, out.adev
        good = np.isfinite(adev) & (adev > 0)
        slope = np.polyfit(np.log(tau[good]), np.log(adev[good]), 1)[0]
        assert slope == pytest.approx(-0.5, abs=0.15)

    def test_allan_deviation_output_fields(self, xp: Any) -> None:
        """A frozen AllanDeviation with matching tau and adev lengths."""
        df = xp.zeros(1024)
        out = analysis.allan_deviation(df, sampling_rate=R, num_taus=10)
        assert isinstance(out, analysis.AllanDeviation)
        assert len(out.tau_s) == len(out.adev)
        with pytest.raises(AttributeError):
            out.adev = None  # type: ignore[misc]
