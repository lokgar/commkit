"""Tests for digital filter response plotting."""

from typing import Any
from unittest.mock import patch

import matplotlib.pyplot as plt
import pytest

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
            sos, sampling_rate=1e9, num_impulse_samples=200, show=False
        )
        assert fig is not None
        assert len(axes) == 4

    def test_filter_response_axis_error(self, xp: Any) -> None:
        """The dashboard takes exactly four axes; any other count raises."""
        taps = xp.array([1, 0, 0, 1])
        fig, ax = plt.subplots(1)
        with pytest.raises(ValueError, match="4 axes, got 1"):
            plot_filter_response(taps, ax=ax, show=False)

        fig, axes3 = plt.subplots(3, 1)
        with pytest.raises(ValueError, match="4 axes, got 3"):
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
        taps = filtering.RRC(0.35, span=4).taps(4)
        complex_taps = taps.astype(complex)
        result = plot_filter_response(complex_taps, sps=4, show=False)
        assert result is not None

    def test_filter_response_show(self, xp: Any) -> None:
        """plot_filter_response() with show=True calls plt.show() and returns None."""
        taps = filtering.RRC(0.35, span=4).taps(4)
        with patch("matplotlib.pyplot.show"):
            result = plot_filter_response(taps, sps=4, show=True)
        assert result is None


def test_magnitude_overlay_respects_theme_and_labels(xp: Any) -> None:
    """Two filters share an axis, its theme color cycle and a usable legend."""
    taps = xp.asarray([0.25, 0.5, 0.25], dtype=xp.float32)
    fig, ax = plot_filter_response(taps, quantity="magnitude", label="first")
    fig2, ax2 = plot_filter_response(
        taps,
        quantity="magnitude",
        ax=ax,
        label="second",
        legend=True,
        xlim=(0, 0.25),
        ylim=(-40, 2),
    )
    assert fig2 is fig and ax2 is ax
    assert [line.get_label() for line in ax.lines] == ["first", "second"]
    assert ax.lines[0].get_color() != ax.lines[1].get_color()
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["first", "second"]
    assert ax.get_xlim() == (0, 0.25)


def test_unknown_quantity_raises() -> None:
    with pytest.raises(ValueError, match="quantity"):
        plot_filter_response([1.0], quantity="unknown")


def test_custom_frequency_grid(xp: Any) -> None:
    """A response plot draws exactly the user-supplied frequency grid."""
    import numpy as np

    taps = xp.asarray([1.0, 0.5])
    frequencies = xp.asarray([-200.0, 0.0, 100.0])
    _, ax = plot_filter_response(
        taps, quantity="phase", sampling_rate=1000, frequencies=frequencies
    )
    np.testing.assert_allclose(ax.lines[0].get_xdata(), [-200.0, 0.0, 100.0])


def test_complex_denominator_impulse(xp: Any) -> None:
    """The impulse plot shows both components even when only a is complex."""
    b = xp.asarray([1.0], dtype=xp.float32)
    a = xp.asarray([1.0, -0.5j], dtype=xp.complex64)
    _, ax = plot_filter_response((b, a), quantity="impulse", num_impulse_samples=10)
    assert len(ax.lines) == 2
    assert abs(ax.lines[1].get_ydata()[1]) > 0
