"""Tests for drift / phase-noise separation and frequency-wander metrics."""

from typing import Any

import numpy as np
import pytest

from commkit import analysis
from tests.common.conversions import to_numpy

R = 32e9
T = 1.0 / R


class TestFrequencyDriftAnalysis:
    """Tests for phase drift vs phase noise separation and wander statistics."""

    def test_frequency_drift_metrics_sinusoid(
        self, backend_device: str, xp: Any
    ) -> None:
        """Verify frequency drift metrics on a known sinusoidal wander."""
        n = 1 << 16
        amp, periods = 4e6, 10.0
        t = np.arange(n) * T
        fm = periods / (n * T)
        drift_phase = 2.0 * np.pi * np.cumsum(amp * np.sin(2 * np.pi * fm * t)) * T
        m = analysis.frequency_drift_metrics(xp.asarray(drift_phase), R)
        assert m["std"] == pytest.approx(amp / np.sqrt(2.0), rel=0.05)
        assert m["pp"] == pytest.approx(2.0 * amp, rel=0.10)

    def test_separate_drift_phase_noise_splits(
        self, backend_device: str, xp: Any
    ) -> None:
        """Verify frequency split separates slow wander from fast phase jitter."""
        n = 1 << 16
        t = np.arange(n) * T
        slow = 0.5 * np.sin(2 * np.pi * 1e5 * t)
        rng = np.random.default_rng(21)
        fast = rng.normal(0, 0.05, n)
        phi = slow + fast
        drift, pn = analysis.separate_drift_phase_noise(xp.asarray(phi), R, cutoff=1e6)
        assert drift.shape == pn.shape == (n,)

        edge = 200
        dr = to_numpy(drift)[edge:-edge]
        sl = slow[edge:-edge]
        assert float(np.std(dr - sl)) < 0.05
