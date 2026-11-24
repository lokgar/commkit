"""plot_frequency_offset_estimate: the panels of each method's real estimate."""

from typing import Any

import numpy as np
import pytest

from commkit import frequency
from commkit.plotting.frequency import plot_frequency_offset_estimate as plot_foe
from tests.common.signals import make_offset_qpsk

FS = 1e6


class TestPlotFrequencySync:
    """Frequency-offset diagnostics from FrequencyOffsetEstimates."""

    @pytest.mark.parametrize("channels", [1, 2])
    def test_mm_autocorrelation(self, xp: Any, channels: int) -> None:
        x, _ = make_offset_qpsk(xp, sampling_rate=FS, channels=channels, df=2e3)
        est = frequency.estimate_frequency_offset(
            x, frequency.MengaliMorelli(), sampling_rate=FS
        )
        fig, axes = plot_foe(est, show=False)
        assert fig is not None
        _, selected = plot_foe(est, quantity="magnitude", label="measured")
        first = selected if channels == 1 else selected[0]
        assert first.lines[0].get_label() == "measured"
        _, selected = plot_foe(est, quantity="phase", annotations=False)
        first = selected if channels == 1 else selected[0]
        assert not first.lines

    def test_mm_autocorrelation_needs_mm_estimate(self, xp: Any) -> None:
        x, _ = make_offset_qpsk(xp, sampling_rate=FS, df=2e3)
        est = frequency.estimate_frequency_offset(
            x, frequency.MthPower(power=4), sampling_rate=FS
        )
        # An MthPower estimate has no autocorrelation panel.
        with pytest.raises(ValueError, match="draws spectrum"):
            plot_foe(est, quantity="magnitude")

    def test_frequency_offset_spectrum(self, xp: Any) -> None:
        """The estimate line is drawn at the estimated offset."""
        x, _ = make_offset_qpsk(xp, sampling_rate=FS, df=3e3)
        est = frequency.estimate_frequency_offset(
            x, frequency.MthPower(power=4), sampling_rate=FS
        )
        fig, ax = plot_foe(est, show=False)
        f_hat = float(est.value)
        assert f_hat == pytest.approx(3e3, abs=50)
        assert any(ln.get_xdata()[0] == pytest.approx(f_hat) for ln in ax.lines[1:])
        _, ax = plot_foe(est, label="first", annotations=False)
        _, ax = plot_foe(est, ax=ax, label="second", annotations=False, legend=True)
        assert [line.get_label() for line in ax.lines] == ["first", "second"]

    def test_frequency_offset_blockwise_result(self, xp: Any) -> None:
        """The recomputed trajectory follows the block estimates."""
        n = 8192
        x, _ = make_offset_qpsk(xp, sampling_rate=FS, n=n, df=2e3)
        est = frequency.estimate_frequency_offset(
            x, frequency.MthPower(power=4, block_size=1024), sampling_rate=FS
        )
        fig, axes = plot_foe(est, max_points=0, show=False)
        df_line = axes[0].lines[0].get_ydata()  # Hz, prefixed by the axis
        assert np.median(df_line) == pytest.approx(2e3, abs=100.0)
        fig.canvas.draw()
        assert axes[0].get_ylabel().endswith("[kHz]")

    def test_pilot_phase_frequency_estimate(self, xp: Any) -> None:
        """A PilotSlope estimate labels the fit with its offset."""
        x, s = make_offset_qpsk(xp, sampling_rate=FS, df=1e3)
        idx = np.arange(0, 4096, 64)
        est = frequency.estimate_frequency_offset(
            x, frequency.PilotSlope(idx, s[0, idx]), sampling_rate=FS
        )
        fig, axes = plot_foe(est, show=False)
        assert np.asarray(axes, dtype=object).shape == (1, 1)
        fit = [ln.get_label() for ln in np.asarray(axes).ravel()[0].lines]
        f_hat = float(est.value)
        assert f_hat == pytest.approx(1e3, abs=20)
        assert any(f"{f_hat:.3f} Hz" in label for label in fit)
        with pytest.raises(ValueError, match="draws pilots"):
            plot_foe(est, quantity="trajectory")


def test_blockwise_draws_every_channel(xp: Any) -> None:
    """Each channel's trajectory follows its own offset, on a time axis.

    Offsets 1 kHz and -2 kHz; the 1024-symbol 4th-power blocks resolve
    them to a few Hz, so the medians sit within 100 Hz.  Drawing channel 0
    twice (the earlier behavior) puts both curves at 1 kHz and fails.
    """
    from commkit.impairments import apply_frequency_offset

    x, _ = make_offset_qpsk(xp, sampling_rate=FS, n=8192, channels=2)
    x = apply_frequency_offset(x, offset=[1e3, -2e3], sampling_rate=FS)
    est = frequency.estimate_frequency_offset(
        x, frequency.MthPower(power=4, block_size=1024), sampling_rate=FS
    )
    _, axes = plot_foe(est, max_points=0)
    curves = [ln for ln in axes[0].lines if ln.get_label().startswith("Channel")]
    assert [ln.get_label() for ln in curves] == ["Channel 0", "Channel 1"]
    medians = [float(np.median(ln.get_ydata())) for ln in curves]
    assert medians == pytest.approx([1e3, -2e3], abs=100.0)
    # The abscissa is in seconds: the record lasts 8192 / FS.
    assert float(curves[0].get_xdata()[-1]) == pytest.approx(8191 / FS)


def test_selected_blockwise_phase_overlay(xp: Any) -> None:
    """The integrated phase panel can be compared on one axis."""
    x, _ = make_offset_qpsk(xp, sampling_rate=FS, n=8192, df=2e3)
    est = frequency.estimate_frequency_offset(
        x, frequency.MthPower(power=4, block_size=1024), sampling_rate=FS
    )
    fig, ax = plot_foe(est, quantity="phase", label="first")
    fig2, ax2 = plot_foe(
        est,
        quantity="phase",
        label="second",
        ax=ax,
        legend=True,
    )
    assert fig2 is fig and ax2 is ax
    assert [line.get_label() for line in ax.lines] == ["first", "second"]


def test_estimates_without_diagnostics_raise(xp: Any) -> None:
    """SpectralPeak's estimate and a combined() one hold only the offset;
    anything but an estimate is a TypeError."""
    x, _ = make_offset_qpsk(xp, sampling_rate=FS, channels=2, df=2e3)
    peak = frequency.estimate_frequency_offset(
        x, frequency.SpectralPeak(), sampling_rate=FS
    )
    with pytest.raises(ValueError, match="no diagnostics"):
        plot_foe(peak)
    mth = frequency.estimate_frequency_offset(
        x, frequency.MthPower(power=4), sampling_rate=FS
    )
    with pytest.raises(ValueError, match="no diagnostics"):
        plot_foe(mth.combined())
    with pytest.raises(TypeError, match="FrequencyOffsetEstimate"):
        plot_foe(xp.ones(4))  # type: ignore[arg-type]


def test_estimate_carries_its_axes(xp: Any) -> None:
    """The estimate holds the sampling rate and record length its plots
    need, so no axis is given twice."""
    x, _ = make_offset_qpsk(xp, sampling_rate=FS, n=8192, df=2e3)
    est = frequency.estimate_frequency_offset(
        x, frequency.MengaliMorelli(block_size=1024), sampling_rate=FS
    )
    assert est.sampling_rate == FS and est.num_samples == 8192
    assert est.combined().num_samples == 8192
