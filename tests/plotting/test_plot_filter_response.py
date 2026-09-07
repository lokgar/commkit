"""Tests for digital filter response plotting."""

from typing import Any
from unittest.mock import patch

import matplotlib.pyplot as plt

from commkit import filtering
from commkit.plotting import plot_filter_response


class TestPlotFilterResponse:
    """Tests for filter response plotting."""

    def test_filter_response(self, xp: Any) -> None:
        """Verify filter response plotting (Impulse, Mag, Phase, Group Delay)."""
        taps = xp.array([1, 0.5, 0.25])
        fig, axes = plot_filter_response(taps, sps=1.0, show=False)
        assert fig is not None
        assert len(axes) == 4

    def test_filter_response_ba(self, xp: Any) -> None:
        """Verify filter response plotting accepts a general (b, a) IIR system."""
        b = xp.array([0.1, 0.2, 0.1])
        a = xp.array([1.0, -0.5, 0.1])
        fig, axes = plot_filter_response((b, a), sps=1.0, show=False)
        assert fig is not None
        assert len(axes) == 4

    def test_filter_response_sos(self, xp: Any) -> None:
        """Verify filter response plotting accepts SOS input, with sampling_rate."""
        import scipy.signal

        sos = xp.asarray(scipy.signal.butter(4, 0.1, btype="low", output="sos"))
        fig, axes = plot_filter_response(
            sos, sampling_rate=1e9, n_impulse=200, show=False
        )
        assert fig is not None
        assert len(axes) == 4

    def test_filter_response_axis_error(self, xp: Any) -> None:
        """Verify behavior when wrong number of axes are provided for filter response."""
        taps = xp.array([1, 0, 0, 1])
        fig, ax = plt.subplots(1)
        plot_filter_response(taps, ax=ax, show=False)

        fig, axes3 = plt.subplots(3, 1)
        plot_filter_response(taps, ax=axes3, show=False)

        fig, axes4 = plt.subplots(2, 2)
        fig2, axes = plot_filter_response(taps, ax=axes4, show=False)
        assert fig2 is fig
        assert len(axes) == 4

    def test_filter_response_no_sps(self, xp: Any) -> None:
        """Test filter_response without sps."""
        plot_filter_response(xp.ones(10))

    def test_filter_response_complex_taps(self, xp: Any) -> None:
        """plot_filter_response() with complex taps plots I/Q components separately."""
        taps = filtering.rrc_taps(sps=4, span=4, rolloff=0.35)
        complex_taps = taps.astype(complex)
        result = plot_filter_response(complex_taps, sps=4, show=False)
        assert result is not None

    def test_filter_response_show(self, xp: Any) -> None:
        """plot_filter_response() with show=True calls plt.show() and returns None."""
        taps = filtering.rrc_taps(sps=4, span=4, rolloff=0.35)
        with patch("matplotlib.pyplot.show"):
            result = plot_filter_response(taps, sps=4, show=True)
        assert result is None
