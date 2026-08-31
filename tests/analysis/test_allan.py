"""Tests for the overlapping Allan deviation estimator.

The white-FM slope is checked against its analytic log-log value.  Inputs are
built on the active backend via the ``xp`` fixture.
"""

import numpy as np
import pytest

from commkit import analysis

R = 32e9  # symbol rate (Baud)
T = 1.0 / R


def test_allan_deviation_white_fm_slope(xp):
    """White-FM frequency noise => Allan deviation slope ≈ -1/2 in log-log."""
    n = 1 << 16
    rng = np.random.default_rng(31)
    df = rng.normal(0, 1e5, n)  # white frequency noise
    out = analysis.allan_deviation(xp.asarray(df), R, n_taus=20)
    tau, adev = out["tau_s"], out["adev"]
    good = np.isfinite(adev) & (adev > 0)
    slope = np.polyfit(np.log(tau[good]), np.log(adev[good]), 1)[0]
    assert slope == pytest.approx(-0.5, abs=0.15)
