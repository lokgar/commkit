"""Tests for drift / phase-noise separation and frequency-wander metrics.

Each estimator is checked against an *analytic* ground truth: a deterministic
sinusoidal frequency wander and a slow-tone-plus-jitter phase.  Inputs are
built on the active backend via the ``xp`` fixture.
"""

import numpy as np
import pytest

from commkit import analysis
from commkit.backend import to_device

R = 32e9  # symbol rate (Baud)
T = 1.0 / R


def test_frequency_drift_metrics_sinusoid(xp):
    n = 1 << 16
    amp, periods = 4e6, 10.0
    t = np.arange(n) * T
    fm = periods / (n * T)
    # drift phase whose derivative is amp·sin(2π fm t)
    drift_phase = 2.0 * np.pi * np.cumsum(amp * np.sin(2 * np.pi * fm * t)) * T
    m = analysis.frequency_drift_metrics(xp.asarray(drift_phase), R)
    assert m["std"] == pytest.approx(amp / np.sqrt(2.0), rel=0.05)
    assert m["pp"] == pytest.approx(2.0 * amp, rel=0.10)


def test_separate_drift_phase_noise_splits(xp):
    n = 1 << 16
    t = np.arange(n) * T
    # slow drift (100 kHz tone in phase) + fast jitter
    slow = 0.5 * np.sin(2 * np.pi * 1e5 * t)
    rng = np.random.default_rng(21)
    fast = rng.normal(0, 0.05, n)
    phi = slow + fast
    drift, pn = analysis.separate_drift_phase_noise(xp.asarray(phi), R, cutoff=1e6)
    assert drift.shape == pn.shape == (n,)
    # Drift tracks the slow tone; pn carries the fast jitter.
    edge = 200
    dr = np.asarray(to_device(drift, "cpu"))[edge:-edge]
    sl = slow[edge:-edge]
    assert float(np.std(dr - sl)) < 0.05
