"""Overlapping Allan deviation, checked against its definition and closed forms.

For frequency samples ``y`` at ``τ0 = 1/f_s`` and averaging ``m`` samples
(``τ = m τ0``), the overlapping Allan variance is half the mean square of
the differences of adjacent ``m``-sample averages, ``ȳ_{k+m} - ȳ_k``, over
every ``k``.  White frequency noise of standard deviation ``σ`` gives
``σ/√m``; a linear drift ``y = a t`` gives ``|a| τ / √2``.
"""

from typing import Any

import numpy as np
import pytest

from commkit import analysis

R = 32e9


def _adev_by_definition(y: np.ndarray, m: int) -> float:
    """Overlapping ADEV from moving averages, written from the definition."""
    avg = np.convolve(y, np.ones(m) / m, mode="valid")  # ȳ_k, k = 0 .. N - m
    d = avg[m:] - avg[:-m]
    return float(np.sqrt(np.mean(d**2) / 2.0))


class TestAllanDeviation:
    def test_matches_the_definition(self, xp: Any) -> None:
        """Every point equals the definition computed independently."""
        y = np.random.default_rng(5).normal(0.0, 1e5, 3000)
        out = analysis.allan_deviation(xp.asarray(y), sampling_rate=R, num_taus=12)
        for tau, adev in zip(out.taus, out.adev, strict=True):
            m = int(round(tau * R))
            assert adev == pytest.approx(_adev_by_definition(y, m), rel=1e-9)

    def test_white_fm_level(self, xp: Any) -> None:
        """White FM of σ = 100 kHz: ADEV = σ/√m at τ = m/f_s, within the
        estimator's spread (overlapping averages over 2^16 samples: under
        10 % up to m = 256), so both the slope -1/2 and the Hz scale hold."""
        sigma, n = 1e5, 1 << 16
        y = np.random.default_rng(31).normal(0.0, sigma, n)
        ms = np.array([1, 4, 16, 64, 256])
        out = analysis.allan_deviation(xp.asarray(y), sampling_rate=R, taus=ms / R)
        np.testing.assert_allclose(out.taus, ms / R)
        np.testing.assert_allclose(out.adev, sigma / np.sqrt(ms), rtol=0.1)

    def test_linear_drift(self, xp: Any) -> None:
        """A frequency ramp y = a t + b: ADEV = |a| τ / √2, exactly (the
        offset b cancels in the differences)."""
        a, b = -3e15, 2e6  # Hz/s, Hz
        t = np.arange(10000) / R
        out = analysis.allan_deviation(
            xp.asarray(a * t + b), sampling_rate=R, num_taus=10
        )
        np.testing.assert_allclose(out.adev, abs(a) * out.taus / np.sqrt(2), rtol=1e-6)

    def test_constant_frequency_has_no_deviation(self, xp: Any) -> None:
        """A constant frequency (or none) has zero Allan deviation."""
        for y in (np.zeros(1024), np.full(1024, 5e6)):
            out = analysis.allan_deviation(xp.asarray(y), sampling_rate=R, num_taus=10)
            np.testing.assert_allclose(out.adev, 0.0, atol=1e-6)

    def test_channels_are_independent(self, xp: Any) -> None:
        """Two channels with different white-FM levels give their own ADEV."""
        rng = np.random.default_rng(2)
        y = np.stack([rng.normal(0, 1e5, 1 << 14), rng.normal(0, 3e5, 1 << 14)])
        out = analysis.allan_deviation(xp.asarray(y), sampling_rate=R, taus=[1 / R])
        np.testing.assert_allclose(out.adev[:, 0], [1e5, 3e5], rtol=0.05)

    def test_result_is_frozen(self, xp: Any) -> None:
        out = analysis.allan_deviation(xp.zeros(1024), sampling_rate=R, num_taus=10)
        assert isinstance(out, analysis.AllanDeviation)
        assert len(out.taus) == len(out.adev)
        with pytest.raises(AttributeError):
            out.adev = None  # type: ignore[misc]
