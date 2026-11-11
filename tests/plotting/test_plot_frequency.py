"""Tests for the frequency-offset plots: each draws a real FrequencyOffsetEstimate."""

from typing import Any

import numpy as np
import pytest

from commkit import frequency
from commkit.plotting.frequency import (
    plot_frequency_offset_blockwise_result,
    plot_frequency_offset_spectrum,
    plot_mm_autocorrelation,
)
from commkit.plotting.recovery import plot_pilot_phase_estimate
from tests.common.signals import make_offset_qpsk

FS = 1e6


def _qpsk(xp: Any, n: int = 4096, channels: int = 1, df: float = 0.0, seed: int = 0):
    """QPSK at 1 sample per symbol with a frequency offset ``df`` (Hz at FS)."""
    return make_offset_qpsk(
        xp, n=n, channels=channels, df=df, seed=seed, sampling_rate=FS
    )


class TestPlotFrequencySync:
    """Frequency-offset diagnostics from FrequencyOffsetEstimates."""

    @pytest.mark.parametrize("channels", [1, 2])
    def test_mm_autocorrelation(self, xp: Any, channels: int) -> None:
        x, _ = _qpsk(xp, channels=channels, df=2e3)
        est = frequency.estimate_frequency_offset(
            x, frequency.MengaliMorelli(), sampling_rate=FS
        )
        fig, axes = plot_mm_autocorrelation(est, sampling_rate=FS, show=False)
        assert fig is not None
        _, selected = plot_mm_autocorrelation(
            est, sampling_rate=FS, quantity="magnitude", label="measured"
        )
        first = selected if channels == 1 else selected[0]
        assert first.lines[0].get_label() == "measured"
        _, selected = plot_mm_autocorrelation(
            est, sampling_rate=FS, quantity="phase", annotations=False
        )
        first = selected if channels == 1 else selected[0]
        assert not first.lines

    def test_mm_autocorrelation_needs_mm_estimate(self, xp: Any) -> None:
        x, _ = _qpsk(xp, df=2e3)
        est = frequency.estimate_frequency_offset(
            x, frequency.MthPower(power=4), sampling_rate=FS
        )
        with pytest.raises(ValueError, match="autocorrelation"):
            plot_mm_autocorrelation(est, sampling_rate=FS)

    def test_frequency_offset_spectrum(self, xp: Any) -> None:
        """The estimate line is drawn at the estimated offset."""
        x, _ = _qpsk(xp, df=3e3)
        est = frequency.estimate_frequency_offset(
            x, frequency.MthPower(power=4), sampling_rate=FS
        )
        fig, ax = plot_frequency_offset_spectrum(
            est, search_range=(-1e4, 1e4), show=False
        )
        f_hat = float(est.value)
        assert f_hat == pytest.approx(3e3, abs=50)
        assert any(ln.get_xdata()[0] == pytest.approx(f_hat) for ln in ax.lines[1:])
        _, ax = plot_frequency_offset_spectrum(est, label="first", annotations=False)
        _, ax = plot_frequency_offset_spectrum(
            est, ax=ax, label="second", annotations=False, legend=True
        )
        assert [line.get_label() for line in ax.lines] == ["first", "second"]

    def test_frequency_offset_blockwise_result(self, xp: Any) -> None:
        """The recomputed trajectory follows the block estimates."""
        n = 8192
        x, _ = _qpsk(xp, n=n, df=2e3)
        est = frequency.estimate_frequency_offset(
            x, frequency.MthPower(power=4, block_size=1024), sampling_rate=FS
        )
        fig, axes = plot_frequency_offset_blockwise_result(
            est, num_samples=n, sampling_rate=FS, max_points=0, show=False
        )
        df_line = axes[0].lines[0].get_ydata()  # Hz, prefixed by the axis
        assert np.median(df_line) == pytest.approx(2e3, abs=100.0)
        fig.canvas.draw()
        assert axes[0].get_ylabel().endswith("[kHz]")

    def test_pilot_phase_frequency_estimate(self, xp: Any) -> None:
        """A PilotSlope estimate labels the fit with its offset."""
        x, s = _qpsk(xp, df=1e3)
        idx = np.arange(0, 4096, 64)
        est = frequency.estimate_frequency_offset(
            x, frequency.PilotSlope(idx, s[0, idx]), sampling_rate=FS
        )
        fig, axes = plot_pilot_phase_estimate(est, sampling_rate=FS, show=False)
        assert np.asarray(axes, dtype=object).shape == (1, 1)
        with pytest.raises(ValueError, match="carrier-phase estimate"):
            plot_pilot_phase_estimate(est, sampling_rate=FS, quantity="trajectory")


def test_selected_blockwise_phase_overlay(xp: Any) -> None:
    """The integrated phase panel can be compared on one axis."""
    x, _ = _qpsk(xp, n=8192, df=2e3)
    est = frequency.estimate_frequency_offset(
        x, frequency.MthPower(power=4, block_size=1024), sampling_rate=FS
    )
    fig, ax = plot_frequency_offset_blockwise_result(
        est, num_samples=8192, sampling_rate=FS, quantity="phase", label="first"
    )
    fig2, ax2 = plot_frequency_offset_blockwise_result(
        est,
        num_samples=8192,
        sampling_rate=FS,
        quantity="phase",
        label="second",
        ax=ax,
        legend=True,
    )
    assert fig2 is fig and ax2 is ax
    assert [line.get_label() for line in ax.lines] == ["first", "second"]
