"""plot_carrier_phase_estimate (the panels of each method's real estimate) and
plot_carrier_phase_trajectory (a phase array)."""

from typing import Any

import numpy as np
import pytest

from commkit import recovery
from commkit.core import Signal
from commkit.mapping import Constellation
from commkit.plotting.recovery import (
    plot_carrier_phase_estimate,
    plot_carrier_phase_trajectory,
)
from commkit.spectral import add_pilot_tone
from tests.common.signals import make_offset_qpsk

FS = 1e6


class TestPlotCarrierPhase:
    """Carrier-phase diagnostics from CarrierPhaseEstimates."""

    def test_carrier_phase_trajectory_estimate(self, xp: Any) -> None:
        """A block estimate marks its block centres."""
        x, _ = make_offset_qpsk(xp, sampling_rate=FS, n=1024)
        est = recovery.estimate_carrier_phase(
            x,
            recovery.ViterbiViterbi(block_size=64),
            constellation=Constellation.psk(4),
        )
        fig, ax = plot_carrier_phase_estimate(est, show=False)
        assert len(ax.lines) == 1 + len(est.block_centers)
        with pytest.raises(TypeError, match="plot_carrier_phase_estimate"):
            plot_carrier_phase_trajectory(est)  # type: ignore[arg-type]

    def test_carrier_phase_trajectory_array(self, xp: Any) -> None:
        phi = xp.cumsum(xp.ones((2, 100)) * 0.01, axis=1)
        fig, ax = plot_carrier_phase_trajectory(phi, num_train_symbols=50, show=False)
        assert len(ax.lines) == 2 + 1

    def test_pilot_phase_carrier_estimate(self, xp: Any) -> None:
        """PilotInterpolation: pilot panel plus the interpolated trajectory."""
        x, s = make_offset_qpsk(xp, sampling_rate=FS, df=1e3)
        idx = np.arange(0, 4096, 32)
        est = recovery.estimate_carrier_phase(
            x, recovery.PilotInterpolation(idx, s[0, idx])
        )
        fig, axes = plot_carrier_phase_estimate(est, show=False)
        assert np.asarray(axes, dtype=object).shape == (1, 2)
        _, ax = plot_carrier_phase_estimate(est, quantity="trajectory", label="first")
        _, ax = plot_carrier_phase_estimate(
            est,
            quantity="trajectory",
            label="second",
            ax=ax,
            legend=True,
        )
        assert [line.get_label() for line in ax.lines] == ["first", "second"]
        _, ax = plot_carrier_phase_estimate(est, quantity="pilots", xlim=(0, 100))
        assert ax.get_xlim() == (0, 100)

    def test_pilot_tone_phase_estimate(self, xp: Any) -> None:
        x, _ = make_offset_qpsk(xp, sampling_rate=FS, n=8192)
        method = recovery.PilotTone(frequency=3e5, bandwidth=2e4)
        x = add_pilot_tone(x, frequency=3e5, sampling_rate=FS)
        est = recovery.estimate_carrier_phase(x, method, sampling_rate=FS)
        fig, axes = plot_carrier_phase_estimate(est, x, show=False)
        assert len(axes) == 2
        fig, ax = plot_carrier_phase_estimate(est, x, quantity="phase", label="first")
        _, ax = plot_carrier_phase_estimate(
            est,
            x,
            quantity="phase",
            ax=ax,
            label="second",
            legend=True,
        )
        assert [line.get_label() for line in ax.lines] == ["first", "second"]
        _, ax = plot_carrier_phase_estimate(
            est,
            x,
            quantity="spectrum",
            xlim=(0, FS / 2),
        )
        assert ax.get_xlim() == (0, FS / 2)
        fig, ax = plot_carrier_phase_estimate(
            est,
            x,
            quantity="spectrum",
            label="first",
            annotations=False,
        )
        _, ax = plot_carrier_phase_estimate(
            est,
            x,
            quantity="spectrum",
            ax=ax,
            label="second",
            annotations=False,
            legend=True,
        )
        assert len(fig.axes) == 1
        assert [line.get_label() for line in ax.lines] == ["first", "second"]

    def test_pilot_tones_phase_estimate(self, xp: Any) -> None:
        x, _ = make_offset_qpsk(xp, sampling_rate=FS, n=8192)
        tones = (2.5e5, 3.5e5)
        for f in tones:
            x = add_pilot_tone(x, frequency=f, sampling_rate=FS)
        est = recovery.estimate_carrier_phase(
            x, recovery.PilotTones(tones, bandwidth=2e4), sampling_rate=FS
        )
        fig, axes = plot_carrier_phase_estimate(est, show=False)
        assert len(axes) == 2
        _, ax = plot_carrier_phase_estimate(est, quantity="phase", label="first")
        _, ax = plot_carrier_phase_estimate(
            est, quantity="phase", label="second", ax=ax, legend=True
        )
        assert [line.get_label() for line in ax.lines] == ["first", "second"]
        _, ax = plot_carrier_phase_estimate(est, quantity="differential", label="test")
        assert any("test" in line.get_label() for line in ax.lines)

    def test_quantity_follows_the_estimate(self, xp: Any) -> None:
        """Each estimate draws its own panels; another raises naming them."""
        x, _ = make_offset_qpsk(xp, sampling_rate=FS, n=1024)
        est = recovery.estimate_carrier_phase(
            x,
            recovery.ViterbiViterbi(block_size=64),
            constellation=Constellation.psk(4),
        )
        with pytest.raises(ValueError, match="draws trajectory"):
            plot_carrier_phase_estimate(est, quantity="differential")
        with pytest.raises(TypeError, match="CarrierPhaseEstimate"):
            plot_carrier_phase_estimate(est.value)  # type: ignore[arg-type]

    def test_pilot_tone_without_samples(self, xp: Any) -> None:
        """The estimate carries its method and rate: without the samples
        only the phase is drawn, and the spectrum asks for them; a Signal at
        another rate is a property conflict."""
        x, _ = make_offset_qpsk(xp, sampling_rate=FS, n=8192)
        x = add_pilot_tone(x, frequency=3e5, sampling_rate=FS)
        method = recovery.PilotTone(frequency=3e5, bandwidth=2e4)
        est = recovery.estimate_carrier_phase(x, method, sampling_rate=FS)
        assert est.method == method and est.sampling_rate == FS
        _, ax = plot_carrier_phase_estimate(est)
        assert ax.get_title().startswith("CPR - Pilot Tone - Recovered")
        with pytest.raises(ValueError, match="samples="):
            plot_carrier_phase_estimate(est, quantity="spectrum")
        other = Signal(samples=x, sampling_rate=2 * FS, symbol_rate=FS / 2)
        with pytest.raises(ValueError, match="sampling_rate"):
            plot_carrier_phase_estimate(est, other)
