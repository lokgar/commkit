"""Tests for the timing plots: each draws a real TimingEstimate."""

from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from commkit import timing
from commkit.core import Preamble, SingleCarrierFrame
from commkit.plotting.timing import plot_timing_correlation
from tests.common.signals import make_offset_qpsk

FS = 1e6


def _qpsk(xp: Any, n: int = 4096, channels: int = 1, df: float = 0.0, seed: int = 0):
    """QPSK at 1 sample per symbol with a frequency offset ``df`` (Hz at FS)."""
    return make_offset_qpsk(
        xp, n=n, channels=channels, df=df, seed=seed, sampling_rate=FS
    )


class TestPlotTimingSync:
    """Timing correlation from a TimingEstimate."""

    def _estimate(self, xp: Any, channels: int = 1):
        frame = SingleCarrierFrame(
            payload_num_symbols=200,
            preamble=Preamble(
                sequence_type="barker", num_symbols=13, num_channels=channels
            ),
            num_channels=channels,
        )
        sig = frame.to_signal(sps=1, symbol_rate=FS)
        pad = np.zeros((channels, 37), dtype=np.complex64)
        x = np.concatenate([pad, np.atleast_2d(sig.samples)], axis=-1)
        sig = sig.replace(samples=xp.asarray(x[0] if channels == 1 else x))
        return timing.estimate_timing(sig)

    @pytest.mark.parametrize("channels", [1, 2])
    def test_timing_correlation(self, xp: Any, channels: int) -> None:
        """One overall/zoom row per channel; the marked peak is the start."""
        est = self._estimate(xp, channels)
        fig, axes = plot_timing_correlation(est, show=False)
        assert np.asarray(axes, dtype=object).shape == (channels, 2)
        starts = np.atleast_1d(
            np.asarray(est.integer.get() if xp is not np else est.integer)
        )
        assert int(starts[0]) == 37
        # The peak line sits at the estimated start.
        vlines = [
            ln.get_xdata()[0]
            for ln in axes[0][0].lines[1:]
            if len(set(ln.get_xdata())) == 1
        ]
        assert 37 in vlines

    def test_timing_correlation_show(self, xp: Any) -> None:
        with patch("matplotlib.pyplot.show"):
            assert plot_timing_correlation(self._estimate(xp), show=True) is None


def test_selected_timing_detail_overlay(xp: Any) -> None:
    """Timing details can be overlaid on one axis for comparison."""
    est = TestPlotTimingSync()._estimate(xp)
    fig, ax = plot_timing_correlation(est, quantity="detail", label="first")
    fig2, ax2 = plot_timing_correlation(
        est, quantity="detail", ax=ax, label="second", xlim=(20, 60), legend=False
    )
    assert fig2 is fig and ax2 is ax
    assert ax.get_xlim() == (20, 60)
    labels = [line.get_label() for line in ax.lines]
    assert "first" in labels and "second" in labels
    assert ax.get_legend() is None
