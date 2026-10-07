"""Tests for time-domain waveform rendering."""

from typing import Any
from unittest.mock import patch

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pytest

from commkit.plotting import plot_psd, plot_time_domain


class TestPlotTimeDomain:
    """Tests for time-domain waveform rendering."""

    def test_time_domain(self, xp: Any) -> None:
        """Verify time-domain waveform plotting."""
        samples = xp.arange(100)
        fig, ax = plot_time_domain(samples, sampling_rate=10.0, show=False)
        assert fig is not None

    def test_time_domain_mimo_grid(self, xp: Any) -> None:
        """Verify MIMO time-domain plotting uses an optimized grid layout."""
        samples = xp.random.randn(4, 100)
        fig, axes = plot_time_domain(samples, sampling_rate=10.0, show=False)
        assert fig is not None
        assert axes.shape == (2, 2)

    @pytest.mark.parametrize("plot_func", [plot_psd, plot_time_domain])
    def test_multichannel_plots(self, xp: Any, plot_func: Any) -> None:
        """Verify multichannel plotting for PSD and Time-Domain."""
        samples = xp.random.randn(2, 256)
        fig, axes = plot_func(samples, sampling_rate=1.0, show=False)
        assert fig is not None
        assert axes.size == 2

    def test_time_domain_limits(self, caplog: Any, xp: Any) -> None:
        """Verify symbol limit warnings in time_domain."""
        sig = xp.ones(100)
        with patch("matplotlib.pyplot.show"):
            plot_time_domain(sig, sampling_rate=1.0, num_symbols=200, sps=1.0)
        assert "Limit exceeds number of symbols" in caplog.text

    def test_time_domain_auto_scale(self, xp: Any) -> None:
        """Verify time axis uses an SI-prefixed engineering tick formatter."""
        sig = xp.ones(100)
        _, ax = plot_time_domain(sig, sampling_rate=1e10)
        assert ax.get_xlabel() == "Time [s]"
        fmt = ax.xaxis.get_major_formatter()
        assert isinstance(fmt, mticker.EngFormatter)
        assert "n" in fmt(3e-9)
        assert "p" in fmt(3e-12)
        assert "µ" in fmt(3e-6)
        assert "m" in fmt(3e-3)

    def test_time_domain_multichannel_axes_array(self, xp: Any) -> None:
        """plot_time_domain() multichannel with axes array normalizes axes to 2D layout."""
        samples = xp.random.randn(2, 1000).astype(xp.float32)
        fig0, axes0 = plt.subplots(1, 2)
        result = plot_time_domain(samples, sampling_rate=1e6, ax=axes0, show=False)
        assert result is not None

    def test_time_domain_multichannel_show(self, xp: Any) -> None:
        """plot_time_domain() multichannel with show=True calls plt.show() and returns None."""
        samples = xp.random.randn(2, 1000).astype(xp.float32)
        with patch("matplotlib.pyplot.show"):
            result = plot_time_domain(samples, sampling_rate=1e6, show=True)
        assert result is None

    def test_time_domain_siso_show(self, xp: Any) -> None:
        """plot_time_domain() 1D with show=True calls plt.show() and returns None."""
        samples = xp.random.randn(500).astype(xp.float32)
        with patch("matplotlib.pyplot.show"):
            result = plot_time_domain(samples, sampling_rate=1e6, show=True)
        assert result is None
