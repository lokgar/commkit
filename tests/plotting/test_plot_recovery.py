"""Tests for the carrier-phase plots: each draws a real CarrierPhaseEstimate."""

from typing import Any

import numpy as np
import pytest

from commkit import recovery
from commkit.mapping import Constellation
from commkit.plotting.recovery import (
    plot_carrier_phase_trajectory,
    plot_pilot_phase_estimate,
    plot_pilot_tone_phase_estimate,
    plot_pilot_tones_phase_estimate,
)
from commkit.spectral import add_pilot_tone
from tests.common.signals import make_offset_qpsk

FS = 1e6


def _qpsk(xp: Any, n: int = 4096, channels: int = 1, df: float = 0.0, seed: int = 0):
    """QPSK at 1 sample per symbol with a frequency offset ``df`` (Hz at FS)."""
    return make_offset_qpsk(
        xp, n=n, channels=channels, df=df, seed=seed, sampling_rate=FS
    )


class TestPlotCarrierPhase:
    """Carrier-phase diagnostics from CarrierPhaseEstimates."""

    def test_carrier_phase_trajectory_estimate(self, xp: Any) -> None:
        """A block estimate marks its block centres."""
        x, _ = _qpsk(xp, n=1024)
        est = recovery.estimate_carrier_phase(
            x,
            recovery.ViterbiViterbi(block_size=64),
            constellation=Constellation.psk(4),
        )
        fig, ax = plot_carrier_phase_trajectory(est, num_train_symbols=100, show=False)
        assert len(ax.lines) == 1 + len(est.block_centers) + 1

    def test_carrier_phase_trajectory_array(self, xp: Any) -> None:
        phi = xp.cumsum(xp.ones((2, 100)) * 0.01, axis=1)
        fig, ax = plot_carrier_phase_trajectory(phi, show=False)
        assert len(ax.lines) == 2

    def test_pilot_phase_carrier_estimate(self, xp: Any) -> None:
        """PilotAided: pilot panel plus the interpolated trajectory."""
        x, s = _qpsk(xp, df=1e3)
        idx = np.arange(0, 4096, 32)
        est = recovery.estimate_carrier_phase(x, recovery.PilotAided(idx, s[0, idx]))
        fig, axes = plot_pilot_phase_estimate(est, sampling_rate=FS, show=False)
        assert np.asarray(axes, dtype=object).shape == (1, 2)
        _, ax = plot_pilot_phase_estimate(
            est, sampling_rate=FS, quantity="trajectory", label="first"
        )
        _, ax = plot_pilot_phase_estimate(
            est,
            sampling_rate=FS,
            quantity="trajectory",
            label="second",
            ax=ax,
            legend=True,
        )
        assert [line.get_label() for line in ax.lines] == ["first", "second"]
        _, ax = plot_pilot_phase_estimate(
            est, sampling_rate=FS, quantity="pilots", xlim=(0, 100)
        )
        assert ax.get_xlim() == (0, 100)

    @pytest.mark.allow_warnings
    def test_pilot_tone_phase_estimate(self, xp: Any) -> None:
        x, _ = _qpsk(xp, n=8192)
        method = recovery.PilotTone(frequency=3e5, bandwidth=2e4)
        x = add_pilot_tone(x, frequency=3e5, sampling_rate=FS)
        est = recovery.estimate_carrier_phase(x, method, sampling_rate=FS)
        fig, axes = plot_pilot_tone_phase_estimate(
            est, x, method=method, sampling_rate=FS, show=False
        )
        assert len(axes) == 2
        fig, ax = plot_pilot_tone_phase_estimate(
            est, x, method=method, sampling_rate=FS, quantity="phase", label="first"
        )
        _, ax = plot_pilot_tone_phase_estimate(
            est,
            x,
            method=method,
            sampling_rate=FS,
            quantity="phase",
            ax=ax,
            label="second",
            legend=True,
        )
        assert [line.get_label() for line in ax.lines] == ["first", "second"]
        _, ax = plot_pilot_tone_phase_estimate(
            est,
            x,
            method=method,
            sampling_rate=FS,
            quantity="spectrum",
            xlim=(0, FS / 2),
        )
        assert ax.get_xlim() == (0, FS / 2)
        fig, ax = plot_pilot_tone_phase_estimate(
            est,
            x,
            method=method,
            sampling_rate=FS,
            quantity="spectrum",
            label="first",
            annotations=False,
        )
        _, ax = plot_pilot_tone_phase_estimate(
            est,
            x,
            method=method,
            sampling_rate=FS,
            quantity="spectrum",
            ax=ax,
            label="second",
            annotations=False,
            legend=True,
        )
        assert len(fig.axes) == 1
        assert [line.get_label() for line in ax.lines] == ["first", "second"]

    @pytest.mark.allow_warnings
    def test_pilot_tones_phase_estimate(self, xp: Any) -> None:
        x, _ = _qpsk(xp, n=8192)
        tones = (2.5e5, 3.5e5)
        for f in tones:
            x = add_pilot_tone(x, frequency=f, sampling_rate=FS)
        est = recovery.estimate_carrier_phase(
            x, recovery.PilotTones(tones, bandwidth=2e4), sampling_rate=FS
        )
        fig, axes = plot_pilot_tones_phase_estimate(est, show=False)
        assert len(axes) == 2
        _, ax = plot_pilot_tones_phase_estimate(est, quantity="phase", label="first")
        _, ax = plot_pilot_tones_phase_estimate(
            est, quantity="phase", label="second", ax=ax, legend=True
        )
        assert [line.get_label() for line in ax.lines] == ["first", "second"]
        _, ax = plot_pilot_tones_phase_estimate(
            est, quantity="differential", label="test"
        )
        assert any("test" in line.get_label() for line in ax.lines)

    def test_wrong_estimate_raises(self, xp: Any) -> None:
        x, _ = _qpsk(xp, n=1024)
        est = recovery.estimate_carrier_phase(
            x,
            recovery.ViterbiViterbi(block_size=64),
            constellation=Constellation.psk(4),
        )
        with pytest.raises(ValueError, match="PilotTones"):
            plot_pilot_tones_phase_estimate(est)
