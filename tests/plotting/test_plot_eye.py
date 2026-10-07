"""Tests for eye diagram plotting (line, hist, multichannel, and validation)."""

from typing import Any
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pytest

from commkit.core import Signal
from commkit.plotting import _plot_eye_traces, plot_eye_diagram


class TestPlotEyeDiagram:
    """Tests for eye diagram plotting (line, hist, multichannel, and validation)."""

    @pytest.mark.parametrize("type", ["line", "hist"])
    def test_eye_diagram_real(self, xp: Any, type: str) -> None:
        """Verify eye diagram generation for real-valued signals."""
        samples = xp.random.randn(1000)
        sps = 4
        fig, ax = plot_eye_diagram(samples, sps=sps, kind=type, show=False)
        assert fig is not None
        assert ax is not None

    def test_eye_diagram_complex(self, xp: Any) -> None:
        """Verify eye diagram generation for complex-valued signals (I/Q)."""
        samples = xp.random.randn(1000) + 1j * xp.random.randn(1000)
        sps = 4

        fig, ax = plot_eye_diagram(samples, sps=sps, show=False)
        assert fig is not None
        assert isinstance(ax, (list, tuple, np.ndarray))
        assert len(ax) == 2

        fig, axes = plt.subplots(2, 1)
        fig, ax_ret = plot_eye_diagram(samples, sps=sps, ax=axes, show=False)
        assert ax_ret is axes

    def test_eye_diagram_multichannel(self, xp: Any) -> None:
        """Verify eye diagram for multichannel complex signals."""
        samples = xp.random.randn(2, 100) + 1j * xp.random.randn(2, 100)
        fig, axes = plot_eye_diagram(samples, sps=4, show=False)
        assert fig is not None
        assert axes.shape == (2, 2)

    def test_eye_diagram_axis_error(self, xp: Any) -> None:
        """Verify error when too few axes are provided for multichannel eye diagram."""
        samples = xp.zeros((2, 100))
        fig, ax = plt.subplots(1)
        with pytest.raises(ValueError, match="Not enough axes"):
            plot_eye_diagram(samples, sps=4, ax=[ax], show=False)

        with pytest.raises(ValueError, match="must provide a list of axes"):
            plot_eye_diagram(samples, sps=4, ax=ax, show=False)

    def test_eye_diagram_sps_error(self, xp: Any) -> None:
        """Verify error for non-integer sps."""
        with pytest.raises(ValueError, match="sps to be a positive integer"):
            plot_eye_diagram(xp.ones(10), sps=2.5)

    def test_eye_diagram_trace_len_error(self, xp: Any) -> None:
        """Verify error when signal too short for eye trace."""
        with pytest.raises(
            ValueError, match="Signal is shorter than the required trace length"
        ):
            _plot_eye_traces(
                xp.ones(5),
                sps=10,
                num_symbols=2,
                ax=MagicMock(),
                kind="line",
                title=None,
            )

    def test_eye_diagram_invalid_type(self, xp: Any) -> None:
        """Verify error for unknown eye type."""
        with pytest.raises(ValueError, match="Unknown kind"):
            _plot_eye_traces(
                xp.ones(100),
                sps=4,
                num_symbols=2,
                ax=MagicMock(),
                kind="magic",
                title=None,
            )

    def test_eye_diagram_multichannel_axes_array(self, xp: Any) -> None:
        """Eye diagram multichannel with axes array reshapes them for per-channel plotting."""
        samples = xp.random.randn(2, 1000).astype(xp.float32)
        fig0, axes0 = plt.subplots(2, 1)
        fig, axes = plot_eye_diagram(samples, sps=4, ax=list(axes0), show=False)
        assert fig is not None

    def test_eye_diagram_multichannel_show(self, xp: Any) -> None:
        """Eye diagram multichannel with show=True calls plt.show() and returns None."""
        samples = xp.random.randn(2, 1000).astype(xp.float32)
        with patch("matplotlib.pyplot.show"):
            result = plot_eye_diagram(samples, sps=4, show=True)
        assert result is None

    def test_eye_diagram_real_with_ax(self, xp: Any) -> None:
        """Eye diagram for real signal with a single pre-existing axis uses it directly."""
        samples = xp.random.randn(500).astype(xp.float32)
        fig0, ax0 = plt.subplots()
        fig, ax = plot_eye_diagram(samples, sps=4, ax=ax0, show=False)
        assert fig is not None

    def test_eye_diagram_siso_show(self, xp: Any) -> None:
        """plot_eye_diagram() 1D with show=True calls plt.show() and returns None."""
        samples = xp.random.randn(1000).astype(xp.float32)
        with patch("matplotlib.pyplot.show"):
            result = plot_eye_diagram(samples, sps=4, show=True)
        assert result is None

    def test_eye_diagram_complex_single_ax_error(self, xp: Any) -> None:
        """plot_eye_diagram() with complex signal and single ax raises ValueError requiring two axes."""
        samples = (xp.random.randn(500) + 1j * xp.random.randn(500)).astype(
            xp.complex64
        )
        fig0, ax0 = plt.subplots()
        with pytest.raises(ValueError, match="complex"):
            plot_eye_diagram(samples, sps=4, ax=ax0, show=False)

    def test_eye_diagram_dense_line(self, xp: Any) -> None:
        """plot_eye_diagram() 'line' type with num_traces>5000 triggers downsampling skip path."""
        samples = xp.random.randn(10200).astype(xp.float32)
        fig, ax = plot_eye_diagram(
            samples, sps=2, kind="line", num_symbols=2, show=False
        )
        assert fig is not None

    def test_eye_diagram_dense_hist(self, xp: Any) -> None:
        """plot_eye_diagram() 'hist' type with num_traces>20000 triggers downsampling skip path."""
        samples = xp.random.randn(40100).astype(xp.float32)
        fig, ax = plot_eye_diagram(
            samples, sps=2, kind="hist", num_symbols=2, show=False
        )
        assert fig is not None

    @pytest.mark.parametrize("channels", [1, 2])
    def test_eye_signal_preserves_window_length(self, xp: Any, channels: int) -> None:
        """Eye diagram from Signal object preserves configured trace window length."""
        samples = xp.sin(xp.arange(100) * 0.4)
        if channels == 2:
            samples = xp.stack([samples, samples])
        sig = Signal(samples=samples, sampling_rate=4e6, symbol_rate=1e6)
        fig, axes = plot_eye_diagram(sig, num_symbols=3, kind="line", show=False)
        for ax in np.asarray(axes).flat:
            assert ax.lines
            assert len(ax.lines[0].get_xdata()) == 3 * 4 + 1

    @pytest.mark.parametrize("sps", [0, -1, float("nan"), float("inf")])
    def test_eye_rejects_invalid_sps(self, xp: Any, sps: Any) -> None:
        """plot_eye_diagram rejects non-positive or non-integral SPS."""
        with pytest.raises(ValueError, match="sps to be a positive integer"):
            plot_eye_diagram(xp.ones(100), sps=sps)
