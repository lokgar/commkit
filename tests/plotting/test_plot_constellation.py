"""Tests for constellation diagram plotting and decision overlay."""

import logging
from typing import Any
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest

from commkit.core import Signal
from commkit.plotting import plot_constellation, plot_ideal_constellation


class TestPlotConstellation:
    """Tests for constellation diagram plotting and decision overlay."""

    def test_constellation_1d(self, xp: Any) -> None:
        """Verify basic constellation density plot generation."""
        samples = xp.random.randn(1000) + 1j * xp.random.randn(1000)
        fig, ax = plot_constellation(samples, bins=50, show=False)
        assert fig is not None
        assert ax is not None

    def test_constellation_overlay_ideal(self, xp: Any) -> None:
        """Verify constellation plot with theoretical overlay enabled."""
        samples = xp.random.randn(1000) + 1j * xp.random.randn(1000)
        fig, ax = plot_constellation(
            samples, bins=50, overlay_ideal=True, modulation="qam", order=16, show=False
        )
        assert fig is not None

    def test_constellation_mimo(self, xp: Any) -> None:
        """Verify MIMO constellation plotting uses an optimized grid layout."""
        samples = xp.random.randn(4, 1000) + 1j * xp.random.randn(4, 1000)
        fig, axes = plot_constellation(samples, bins=50, show=False)
        assert fig is not None
        assert axes.shape == (2, 2)

    def test_ideal_constellation_basic(self, xp: Any) -> None:
        """Verify ideal constellation plotting."""
        fig, ax = plot_ideal_constellation("qam", 16, show=False)
        assert fig is not None

        ret = plot_ideal_constellation("invalid", 4, show=False)
        assert ret is None

    def test_constellation_histogram_overlay_error(self, xp: Any) -> None:
        """Verify warning when overlaying ideal on histogram constellation with bad mod."""
        samples = xp.random.randn(100) + 1j * xp.random.randn(100)
        plot_constellation(
            samples,
            bins=10,
            overlay_ideal=True,
            modulation="invalid",
            order=4,
            show=False,
        )

    def test_constellation_histogram_overlay_warning(
        self, caplog: Any, xp: Any
    ) -> None:
        """Verify warning when overlaying ideal on histogram."""
        caplog.set_level(logging.WARNING)
        plot_constellation(
            xp.ones(10) + 1j, bins=10, overlay_ideal=True, modulation=None
        )
        assert "Modulation and order must be provided" in caplog.text

    def test_constellation_real_samples(self, xp: Any) -> None:
        """Constellation with real (non-complex) samples warns and converts to complex."""
        samples = xp.ones(100, dtype=xp.float32)
        with patch("commkit.plotting.logger.warning"):
            result = plot_constellation(samples, show=False)
        assert result is not None

    def test_constellation_multichannel_single_axis_warning(
        self, xp: Any, caplog: Any
    ) -> None:
        """Constellation multichannel with single axis warns and overlays all channels."""
        samples = xp.ones((2, 100), dtype=xp.complex64)
        fig0, ax0 = plt.subplots()
        caplog.set_level(logging.WARNING)
        result = plot_constellation(samples, ax=ax0, show=False)
        assert result is not None
        assert "Overlaying plots" in caplog.text

    def test_constellation_multichannel_axes_array(self, xp: Any) -> None:
        """Constellation multichannel with axes array normalizes to 2D layout."""
        samples = xp.ones((2, 100), dtype=xp.complex64)
        fig0, axes0 = plt.subplots(1, 2)
        result = plot_constellation(samples, ax=axes0, show=False)
        assert result is not None

    def test_constellation_multichannel_show(self, xp: Any) -> None:
        """Constellation multichannel with show=True calls plt.show() and returns None."""
        samples = xp.ones((2, 100), dtype=xp.complex64)
        with patch("matplotlib.pyplot.show"):
            result = plot_constellation(samples, show=True)
        assert result is None

    def test_constellation_siso_show(self, xp: Any) -> None:
        """Constellation SISO with show=True calls plt.show() and returns None."""
        samples = xp.ones(100, dtype=xp.complex64)
        with patch("matplotlib.pyplot.show"):
            result = plot_constellation(samples, show=True)
        assert result is None

    def test_ideal_constellation_custom_ax(self, xp: Any) -> None:
        """plot_ideal_constellation() with provided ax uses that axis's figure."""
        fig0, ax0 = plt.subplots()
        result = plot_ideal_constellation(modulation="psk", order=4, ax=ax0, show=False)
        assert result is not None

    def test_ideal_constellation_show(self, xp: Any) -> None:
        """plot_ideal_constellation() with show=True calls plt.show() and returns None."""
        with patch("matplotlib.pyplot.show"):
            result = plot_ideal_constellation(modulation="qam", order=16, show=True)
        assert result is None

    def test_constellation_vmin_vmax(self, xp: Any) -> None:
        """plot_constellation() with vmin/vmax sets color scale bounds on histogram."""
        samples = (xp.random.randn(500) + 1j * xp.random.randn(500)).astype(
            xp.complex64
        )
        result = plot_constellation(samples, vmin=0.0, vmax=1.0, show=False)
        assert result is not None

    @pytest.mark.parametrize("channels", [1, 2])
    def test_constellation_signal_optional_metadata_fallback(
        self, xp: Any, channels: int
    ) -> None:
        """Signal container constellation plotting falls back cleanly on optional metadata."""
        samples = xp.asarray([0.0, 1.0] * 50, dtype=xp.complex64)
        if channels == 2:
            samples = xp.stack([samples, samples])
        sig = Signal(samples=samples, sampling_rate=1e6, symbol_rate=1e6)
        fig, axes = plot_constellation(
            sig, modulation="PAM", order=2, unipolar=True, overlay_ideal=True
        )
        for ax in np.asarray(axes).flat:
            assert len(ax.collections) == 1
            points = ax.collections[0].get_offsets()
            assert points.shape == (2, 2)
            assert np.min(points[:, 0]) == 0.0
            assert np.max(points[:, 0]) > 0.0
