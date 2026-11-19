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


def test_long_fir_time_panels_reach_end(xp: Any) -> None:
    """A 2001-tap FIR shows its full centred impulse and settled step response."""
    import numpy as np

    taps = xp.asarray(filtering.fir_taps(sampling_rate=1.0, num_taps=2001, cutoff=0.1))
    taps_host = np.asarray(taps.get() if hasattr(taps, "get") else taps)

    fig_dash, axes_dash = plot_filter_response(taps, sps=1.0, show=False)
    fig_dash.canvas.draw()
    imp_dash = axes_dash[0].lines[0]
    x_dash = imp_dash.get_xdata()
    y_dash = imp_dash.get_ydata()
    assert len(x_dash) == 2001
    assert x_dash[0] == -1000.0
    assert x_dash[-1] == 1000.0
    assert x_dash[int(np.argmax(np.abs(y_dash)))] == 0.0
    np.testing.assert_allclose(y_dash, taps_host, rtol=1e-6, atol=1e-6)
    plt.close(fig_dash)

    fig_imp, ax_imp = plot_filter_response(
        taps, quantity="impulse", sps=1.0, show=False
    )
    fig_imp.canvas.draw()
    imp_line = ax_imp.lines[0]
    x_imp = imp_line.get_xdata()
    y_imp = imp_line.get_ydata()
    assert len(x_imp) == 2001
    assert x_imp[0] == -1000.0
    assert x_imp[-1] == 1000.0
    assert x_imp[int(np.argmax(np.abs(y_imp)))] == 0.0
    np.testing.assert_allclose(y_imp, taps_host, rtol=1e-6, atol=1e-6)
    plt.close(fig_imp)

    fig_step, ax_step = plot_filter_response(taps, quantity="step", sps=1.0, show=False)
    fig_step.canvas.draw()
    step_line = ax_step.lines[0]
    x_step = step_line.get_xdata()
    y_step = step_line.get_ydata()
    assert len(x_step) > 2001
    assert x_step[0] == 0.0
    assert x_step[-1] > 2000.0
    np.testing.assert_allclose(
        y_step[2000:],
        np.full(len(y_step) - 2000, np.sum(taps_host)),
        rtol=1e-6,
        atol=1e-6,
    )
    plt.close(fig_step)

    for q in (None, "impulse", "step"):
        fig_exp, ax_exp = plot_filter_response(
            taps, quantity=q, num_impulse_samples=120, sps=1.0, show=False
        )
        target_ax = ax_exp[0] if q is None else ax_exp
        x_exp = target_ax.lines[0].get_xdata()
        assert len(x_exp) == 120
        assert x_exp[0] == 0.0
        assert x_exp[-1] == 119.0
        plt.close(fig_exp)


def test_slow_iir_time_panels_reach_end(xp: Any) -> None:
    """A low-cutoff IIR extends past 500 samples until its response settles."""
    import numpy as np
    import scipy.signal

    sos = xp.asarray(scipy.signal.butter(4, 1e-3, output="sos"))

    fig_imp, ax_imp = plot_filter_response(sos, quantity="impulse", show=False)
    y_imp = ax_imp.lines[0].get_ydata()
    assert len(y_imp) > 5000
    assert abs(y_imp[-1]) / np.max(np.abs(y_imp)) < 1e-4
    plt.close(fig_imp)

    fig_step, ax_step = plot_filter_response(sos, quantity="step", show=False)
    y_step = ax_step.lines[0].get_ydata()
    assert len(y_step) > 5000
    assert abs(y_step[-1] - 1.0) < 1e-4
    plt.close(fig_step)


def test_invalid_num_impulse_samples_raises() -> None:
    for bad in (0, -5, True):
        with pytest.raises(ValueError, match="num_impulse_samples"):
            plot_filter_response([1.0], num_impulse_samples=bad)  # type: ignore[arg-type]
