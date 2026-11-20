"""Tests for digital filter response plotting."""

from typing import Any
from unittest.mock import patch

import matplotlib.pyplot as plt
import pytest

import commkit.plotting.theme as theme
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

    fig_step, ax_step = plot_filter_response(
        taps, quantity="step", sps=1.0, causal=True, show=False
    )
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
            taps, quantity=q, num_impulse_samples=120, sps=1.0, causal=True, show=False
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


# -----------------------------------------------------------------------------
# relative and annotations
# -----------------------------------------------------------------------------


def _curves(ax: Any) -> list[Any]:
    """y data of the responses on ``ax``, guides excluded."""
    import numpy as np

    return [
        np.asarray(line.get_ydata())
        for line in ax.lines
        if line.get_gid() != "commkit-3db-guide"
    ]


@pytest.mark.parametrize("dtype", ["float64", "complex128"])
def test_relative_removes_the_tap_scale(xp: Any, dtype: str) -> None:
    """Taps scaled by 7 plot the same relative impulse, step and magnitude.

    Without ``relative`` the curves differ by 7 (and 20 log10 7 = 16.9 dB);
    with it the impulse peaks at magnitude 1 and the magnitude at 0 dB.
    """
    import numpy as np

    taps = xp.asarray(filtering.RRC(0.2, span=8).taps(8), dtype=dtype)
    if dtype == "complex128":  # a frequency shift: response peak off DC
        taps = taps * xp.exp(2j * np.pi * 0.05 * xp.arange(taps.size)).astype(dtype)
    for q in ("impulse", "step", "magnitude"):
        _, ax = plot_filter_response(taps, quantity=q, sps=8, relative=True)
        plot_filter_response(7 * taps, quantity=q, sps=8, relative=True, ax=ax)
        curves = _curves(ax)
        half = len(curves) // 2
        for a, b in zip(curves[:half], curves[half:], strict=True):
            np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
        _, raw = plot_filter_response(taps, quantity=q, sps=8)
        plot_filter_response(7 * taps, quantity=q, sps=8, ax=raw)
        a, b = _curves(raw)[0], _curves(raw)[half]
        assert not np.allclose(a, b, atol=1e-3)
        plt.close("all")

    _, ax = plot_filter_response(7 * taps, quantity="impulse", sps=8, relative=True)
    y = _curves(ax)
    magnitude = np.abs(y[0] + 1j * y[1]) if len(y) == 2 else np.abs(y[0])
    assert np.max(magnitude) == pytest.approx(1.0, abs=1e-6)
    _, ax = plot_filter_response(7 * taps, quantity="magnitude", relative=True)
    assert np.max(_curves(ax)[0]) == pytest.approx(0.0, abs=1e-4)
    assert ax.get_ylabel() == "Relative Magnitude [dB]"
    plt.close("all")


def test_relative_uses_the_peak_gain_not_dc(xp: Any) -> None:
    """A high-pass filter peaks at 0 dB though its DC gain is zero, and a
    low-pass step response scaled to unit peak gain settles at 1."""
    import numpy as np

    hp = xp.asarray(
        filtering.butterworth_sos(sampling_rate=1.0, cutoff=0.1, order=4, btype="high")
    )
    sos = hp.copy()
    sos[0, :3] *= 3  # gain 3, +9.5 dB
    _, ax = plot_filter_response(sos, quantity="magnitude", relative=True)
    y = _curves(ax)[0]
    assert np.max(y) == pytest.approx(0.0, abs=1e-3)
    assert y[0] < -100  # DC

    lp = xp.asarray(
        filtering.butterworth_sos(sampling_rate=1.0, cutoff=0.1, order=4)
    ).copy()
    lp[0, :3] *= 5
    _, ax = plot_filter_response(lp, quantity="step", relative=True)
    assert _curves(ax)[0][-1] == pytest.approx(1.0, abs=1e-4)
    plt.close("all")


def test_relative_leaves_phase_and_group_delay(xp: Any) -> None:
    """A positive gain changes neither phase nor group delay."""
    import numpy as np

    taps = xp.asarray([0.1, 0.7, 0.3, -0.2])
    for q in ("phase", "group_delay"):
        _, ax = plot_filter_response(taps, quantity=q)
        plot_filter_response(5 * taps, quantity=q, relative=True, ax=ax)
        a, b = _curves(ax)
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
        plt.close("all")


def test_relative_all_zero_filter_stays_zero(xp: Any) -> None:
    import numpy as np

    _, axes = plot_filter_response(xp.zeros(5), relative=True, annotations=True)
    assert not np.any(_curves(axes[0])[0])
    plt.close("all")


def test_annotations_draw_the_3db_line(xp: Any) -> None:
    """The guide sits 3 dB below each curve's peak, in the accent color also
    on a single curve (the dashboard included); relative curves share one
    guide at -3 dB; legends leave it out."""
    import numpy as np

    taps = xp.asarray([0.25, 0.5, 0.25])
    _, ax = plot_filter_response(taps, quantity="magnitude")
    assert len(ax.lines) == 1  # off by default

    _, ax = plot_filter_response(taps, quantity="magnitude", annotations=True)
    plot_filter_response(
        4 * taps, quantity="magnitude", annotations=True, ax=ax, label="x4"
    )
    guides = [line for line in ax.lines if line.get_gid() == "commkit-3db-guide"]
    curves = [line for line in ax.lines if line.get_gid() != "commkit-3db-guide"]
    assert len(guides) == 2
    for guide, curve in zip(guides, curves, strict=True):
        assert guide.get_ydata()[0] == pytest.approx(np.max(curve.get_ydata()) - 3)
        assert guide.get_color() == theme._accent()
    assert guides[1].get_ydata()[0] == pytest.approx(20 * np.log10(4) - 3, abs=1e-4)

    _, ax = plot_filter_response(
        taps, quantity="magnitude", relative=True, annotations=True, label="a"
    )
    plot_filter_response(
        4 * taps,
        quantity="magnitude",
        relative=True,
        annotations=True,
        ax=ax,
        label="b",
        legend=True,
    )
    guides = [line for line in ax.lines if line.get_gid() == "commkit-3db-guide"]
    assert len(guides) == 1
    assert guides[0].get_ydata()[0] == pytest.approx(-3.0, abs=1e-4)
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["a", "b"]

    _, axes = plot_filter_response(taps, annotations=True)
    assert sum(len(a.lines) for a in axes) == 5  # four responses, one guide
    (guide,) = [line for line in axes[1].lines if line.get_gid() == "commkit-3db-guide"]
    assert guide.get_color() == theme._accent()  # not the curve's color
    plt.close("all")


# -----------------------------------------------------------------------------
# time reference, units and guides
# -----------------------------------------------------------------------------


def _guides(ax: Any, gid: str) -> list[Any]:
    return [line for line in ax.lines if line.get_gid() == gid]


@pytest.mark.parametrize("num_taps", [9, 10])
def test_time_reference_follows_fir_filter(xp: Any, num_taps: int) -> None:
    """Aligned (default) and causal panels place the taps where
    ``fir_filter`` puts them, and measure the delay from the same tap.

    The expectation is the output of ``fir_filter`` for an impulse, not the
    plot's own arithmetic: tap ``j`` lands ``j - (L - 1) // 2`` samples from
    the impulse (``j`` with ``causal=True``).  A symmetric filter delays by
    ``(L - 1) / 2`` causally, which leaves 0 (odd ``L``) or half a sample
    (even ``L``) after ``fir_filter``'s alignment on tap ``(L - 1) // 2``.
    """
    import numpy as np

    taps = np.hanning(num_taps + 2)[1:-1].astype(np.float64)
    for causal in (False, True):
        impulse = np.zeros(64)
        impulse[30] = 1.0
        out = np.asarray(
            ck_host(
                filtering.fir_filter(
                    xp.asarray(impulse), xp.asarray(taps), causal=causal
                )
            )
        )
        _, ax = plot_filter_response(
            xp.asarray(taps), quantity="impulse", causal=causal
        )
        t = ax.lines[0].get_xdata()
        np.testing.assert_allclose(out[30 + t.astype(int)], taps, atol=1e-12)

        _, ax = plot_filter_response(
            xp.asarray(taps), quantity="group_delay", causal=causal
        )
        expected = (num_taps - 1) / 2 - (0 if causal else (num_taps - 1) // 2)
        delay = ax.lines[0].get_ydata()
        np.testing.assert_allclose(delay[np.isfinite(delay)][:100], expected, atol=1e-6)

        _, ax = plot_filter_response(xp.asarray(taps), quantity="phase", causal=causal)
        f = ax.lines[0].get_xdata()
        phase = ax.lines[0].get_ydata()
        np.testing.assert_allclose(
            phase[:50], -2 * np.pi * f[:50] * expected, atol=1e-6
        )
        plt.close("all")


def ck_host(a: Any) -> Any:
    from commkit.backend import to_device

    return to_device(a, "cpu")


def test_iir_ignores_causal(xp: Any) -> None:
    """IIR filters are causal whatever ``causal`` says: time starts at 0."""
    import numpy as np

    sos = xp.asarray(filtering.butterworth_sos(sampling_rate=1.0, cutoff=0.1, order=4))
    for q in ("impulse", "group_delay"):
        _, ax = plot_filter_response(sos, quantity=q)
        plot_filter_response(sos, quantity=q, causal=True, ax=ax)
        a, b = ax.lines[0].get_xydata(), ax.lines[1].get_xydata()
        np.testing.assert_array_equal(a, b)
        if q == "impulse":
            assert a[0, 0] == 0.0
    plt.close("all")


def test_units_follow_sps_and_sampling_rate(xp: Any) -> None:
    """Samples without ``sps``, symbol periods and rates with it, seconds and
    hertz with a sampling rate."""
    import numpy as np

    taps = xp.asarray(filtering.RRC(0.2, span=6).taps(4))
    _, raw = plot_filter_response(taps, quantity="group_delay", causal=True)
    assert raw.get_xlabel() == "Frequency [Cycles/Sample]"
    assert raw.get_ylabel() == "Group Delay [Samples]"
    _, sym = plot_filter_response(taps, quantity="group_delay", sps=4, causal=True)
    assert sym.get_xlabel() == "Frequency [Symbol Rates]"
    assert sym.get_ylabel() == "Group Delay [Symbol Periods]"
    np.testing.assert_allclose(sym.lines[0].get_xdata(), 4 * raw.lines[0].get_xdata())
    np.testing.assert_allclose(
        sym.lines[0].get_ydata(), raw.lines[0].get_ydata() / 4, rtol=1e-6
    )
    _, imp = plot_filter_response(taps, quantity="impulse")
    assert imp.get_xlabel() == "Time [Samples]"
    assert imp.lines[0].get_xdata()[0] == -12.0
    _, hz = plot_filter_response(taps, quantity="magnitude", sps=4, sampling_rate=8e9)
    assert hz.get_xlabel().startswith("Frequency [")
    assert hz.lines[0].get_xdata()[-1] < 4e9
    with pytest.raises(ValueError, match="sps"):
        plot_filter_response(taps, sps=0)
    plt.close("all")


def test_symbol_instant_markers_show_nyquist_zero_crossings(xp: Any) -> None:
    """On an RC pulse the markers sit on zeros except at t = 0; on an RRC
    pulse (not a Nyquist pulse by itself) they do not.

    The RC taps vanish at every nonzero symbol instant by construction, up to
    rounding (|p(kT)| / p(0) below 1e-16 here); the RRC's neighbors are
    several percent of its peak.
    """
    import numpy as np

    for pulse, nyquist in (
        (filtering.RC(0.2), True),
        (filtering.RRC(0.2), False),
    ):
        _, ax = plot_filter_response(
            xp.asarray(pulse.taps(8)), quantity="impulse", sps=8, relative=True,
            annotations=True,
        )  # fmt: skip
        (marks,) = _guides(ax, "commkit-symbol-instants")
        t, y = marks.get_xdata(), marks.get_ydata()
        np.testing.assert_allclose(t, np.round(t))  # whole symbol periods
        assert len(t) == 33  # the default span of 32 symbols
        assert y[t == 0] == pytest.approx(1.0)
        off = np.abs(y[t != 0])
        assert bool(off.max() < 1e-6) is nyquist
        plt.close("all")
    for sps in (None, 1, 2.5):
        _, ax = plot_filter_response(
            xp.asarray([0.5, 1.0, 0.5]), quantity="impulse", sps=sps, annotations=True
        )
        assert not _guides(ax, "commkit-symbol-instants")
        plt.close("all")


def test_step_final_value_guide(xp: Any) -> None:
    import numpy as np

    sos = xp.asarray(
        filtering.chebyshev1_sos(sampling_rate=1.0, cutoff=0.1, order=4, ripple=1.0)
    )
    _, ax = plot_filter_response(sos, quantity="step", annotations=True)
    (guide,) = _guides(ax, "commkit-final-value-guide")
    assert guide.get_ydata()[0] == pytest.approx(ax.lines[0].get_ydata()[-1])
    assert guide.get_color() == theme._accent()
    # Even order: settles at the bottom of the 1 dB ripple, 10^(-1/20).
    assert guide.get_ydata()[0] == pytest.approx(10 ** (-1 / 20), abs=1e-3)
    assert np.all(np.isfinite(ax.lines[0].get_ydata()))
    plt.close("all")


def test_half_symbol_rate_guide(xp: Any) -> None:
    """The guide sits at half the symbol rate, once per axis, on both sides
    for a two-sided (complex) response; RC is at -6 dB there, RRC at -3 dB."""
    import numpy as np

    rc = xp.asarray(filtering.RC(0.2).taps(8))
    _, ax = plot_filter_response(
        rc, quantity="magnitude", sps=8, relative=True, annotations=True
    )
    plot_filter_response(
        xp.asarray(filtering.RRC(0.2).taps(8)), quantity="magnitude", sps=8,
        relative=True, annotations=True, ax=ax,
    )  # fmt: skip
    (edge,) = _guides(ax, "commkit-nyquist-guide")
    assert edge.get_xdata()[0] == 0.5
    curves = [line for line in ax.lines if line.get_gid() is None]
    # Truncated to 32 symbols, the response peaks 0.015 dB above DC.
    for curve, level in zip(curves, (-6.03, -3.02), strict=True):
        assert np.interp(0.5, curve.get_xdata(), curve.get_ydata()) == pytest.approx(
            level, abs=0.05
        )

    _, ax = plot_filter_response(
        rc.astype(xp.complex64), quantity="magnitude", sps=8, sampling_rate=8e9,
        annotations=True,
    )  # fmt: skip
    positions = sorted(
        line.get_xdata()[0] for line in _guides(ax, "commkit-nyquist-guide")
    )
    assert positions == [-0.5e9, 0.5e9]
    _, ax = plot_filter_response(rc, quantity="magnitude", annotations=True)
    assert not _guides(ax, "commkit-nyquist-guide")
    plt.close("all")


def test_flat_panels_do_not_zoom_into_rounding(xp: Any) -> None:
    """Autoscaled frequency panels span at least 0.01 samples, 0.1 dB and
    0.01 rad, so a flat curve draws flat; a curve that varies more, an
    overlay and an explicit ``ylim`` are unaffected."""
    import numpy as np

    taps = xp.asarray(filtering.fir_taps(sampling_rate=1e6, num_taps=101, cutoff=1e5))
    for causal, centre in ((False, 0.0), (True, 50e-6)):
        _, ax = plot_filter_response(
            taps, sampling_rate=1e6, quantity="group_delay", causal=causal
        )
        lo, hi = ax.get_ylim()
        assert hi - lo >= 0.01e-6 * (1 - 1e-9)
        assert lo < centre < hi and hi - lo < 1e-6  # around it, not ±0.05 s
        plt.close("all")
    _, ax = plot_filter_response(xp.asarray([0.0, 1.0]), quantity="magnitude")
    lo, hi = ax.get_ylim()
    assert hi - lo >= 0.1 * (1 - 1e-9) and lo < 0 < hi
    _, ax = plot_filter_response(
        xp.asarray([0.25, 0.5, 0.25]), quantity="phase", nfft=64
    )
    lo, hi = ax.get_ylim()
    assert hi - lo >= 0.01 * (1 - 1e-9)

    sos = xp.asarray(filtering.butterworth_sos(sampling_rate=1.0, cutoff=0.1, order=4))
    _, ax = plot_filter_response(taps, quantity="group_delay", causal=True)
    plot_filter_response(sos, quantity="group_delay", ax=ax)
    y = ax.lines[1].get_ydata()
    lo, hi = ax.get_ylim()
    assert lo <= np.nanmin(y) and hi >= np.nanmax(y)  # the overlay still fits
    _, ax = plot_filter_response(
        taps, quantity="group_delay", ylim=(49.9, 50.1), causal=True
    )
    assert ax.get_ylim() == (49.9, 50.1)
    plt.close("all")


@pytest.mark.parametrize("causal", [False, True])
def test_phase_has_no_jumps_at_stopband_zeros(xp: Any, causal: bool) -> None:
    """The phase of a linear-phase low-pass is the straight line of its
    group delay over the whole band, stopband included.

    Its response changes sign at each of some 40 stopband zeros, a jump of π
    that ``np.unwrap`` turns up or down by rounding: drawn that way, the
    aligned phase climbs a staircase to about 25 rad instead of staying at
    0.  The line is exact up to the response's rounding (1e-10 rad here).
    """
    import numpy as np

    taps = xp.asarray(filtering.fir_taps(sampling_rate=1.0, num_taps=101, cutoff=0.1))
    _, ax = plot_filter_response(taps, quantity="phase", causal=causal)
    f, phase = ax.lines[0].get_xdata(), ax.lines[0].get_ydata()
    delay = 50 if causal else 0
    np.testing.assert_allclose(phase, -2 * np.pi * f * delay, atol=1e-6)
    plt.close("all")
