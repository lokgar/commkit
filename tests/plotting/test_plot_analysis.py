"""Tests for the analysis plots (frequency drift, Allan deviation, linewidth).

The plots take the results of ``commkit.analysis``, so every test builds a
real result and checks that the plot draws its fields.
"""

from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from commkit import analysis
from commkit.analysis import (
    BetaSeparation,
    DshFmFit,
    DshIncrement,
    DshLorentzian,
    IncrementSlope,
    IncrementSubtract,
)
from commkit.plotting.analysis import (
    plot_allan_deviation,
    plot_carrier_phase_characterization,
    plot_carrier_phase_decomposition,
    plot_dsh_beat_psd,
    plot_frequency_drift,
    plot_frequency_noise_psd,
    plot_increment_variance,
)
from tests.common.signals import make_dsh_beat, make_wiener_phase

R = 1e9
FS = 500e6


def _phase(xp: Any, channels: int = 1, n: int = 1 << 14) -> Any:
    rows = [
        make_wiener_phase(
            num_symbols=n, linewidth=1e6, sample_rate=R, seed=s, dtype=np.float64
        )
        for s in range(channels)
    ]
    return xp.asarray(rows[0] if channels == 1 else np.stack(rows))


def _beat(xp: Any) -> tuple[Any, int]:
    m = 200
    z, _ = make_dsh_beat(2e6, 1 << 16, m, 80e6, snr_db=25, seed=12)
    return xp.asarray(z), m


class TestPlotAnalysis:
    """Drift, Allan and linewidth plots from analysis results."""

    @pytest.mark.parametrize("channels", [1, 2])
    def test_frequency_drift(self, xp: Any, channels: int) -> None:
        """One trace per channel, on the time axis of the sampling rate."""
        drift = analysis.frequency_drift(_phase(xp, channels), sampling_rate=R)
        fig, ax = plot_frequency_drift(drift, sampling_rate=R, reference_amplitude=6e3)
        traces = [ln for ln in ax.lines if len(ln.get_xdata()) > 2]
        assert len(traces) == channels
        assert traces[0].get_xdata()[1] == pytest.approx(1.0 / R)

    def test_allan_deviation(self, xp: Any) -> None:
        """The deviation is drawn at its averaging times, plus the guide."""
        drift = analysis.frequency_drift(_phase(xp), sampling_rate=R)
        allan = analysis.allan_deviation(drift.df, sampling_rate=R, num_taus=12)
        fig, ax = plot_allan_deviation(allan, reference_slopes=True)
        np.testing.assert_allclose(ax.lines[0].get_xdata(), allan.taus)
        assert len(ax.lines) == 2
        _, ax = plot_allan_deviation(
            allan, xlim=(1e-9, 1e-6), ylim=(1e-5, 1), legend=False
        )
        assert ax.get_xlim() == (1e-9, 1e-6)
        assert ax.get_ylim() == (1e-5, 1)
        assert ax.get_legend() is None
        _, ax = plot_allan_deviation(allan, reference_slopes=False, label="before")
        _, ax = plot_allan_deviation(
            allan, reference_slopes=False, ax=ax, label="after", legend=True
        )
        assert [t.get_text() for t in ax.get_legend().get_texts()] == [
            "before",
            "after",
        ]
        assert ax.lines[0].get_color() != ax.lines[1].get_color()

    @pytest.mark.allow_warnings  # a short phase: the β-area is unresolved
    def test_frequency_noise_psd_beta(self, xp: Any) -> None:
        """BetaSeparation: PSD, β-line, integration fill and floor guide."""
        est = analysis.estimate_linewidth(
            _phase(xp), BetaSeparation(f_min=1e6, f_max=1e8), sampling_rate=R
        )
        fig, ax = plot_frequency_noise_psd(est)
        labels = [ln.get_label() for ln in ax.lines]
        assert r"$\beta$-separation line" in labels
        assert len(ax.collections) >= 1  # the β-area fill
        floor = [ln for ln in ax.lines if "White-FM" in ln.get_label()][0]
        assert floor.get_ydata()[0] == pytest.approx(est.linewidth_floor / np.pi)
        _, ax = plot_frequency_noise_psd(est, label="first", annotations=False)
        _, ax = plot_frequency_noise_psd(
            est, ax=ax, label="second", annotations=False, legend=True
        )
        labels = [line.get_label() for line in ax.lines]
        assert any("first" in label for label in labels)
        assert any("second" in label for label in labels)
        assert not ax.collections

    def test_frequency_noise_psd_dsh(self, xp: Any) -> None:
        """DshFmFit: the floor guide sits at the estimate itself."""
        z, m = _beat(xp)
        est = analysis.estimate_linewidth(z, DshFmFit(delay=m / FS), sampling_rate=FS)
        fig, ax = plot_frequency_noise_psd(est, title="DSH")
        floor = [ln for ln in ax.lines if "White-FM" in ln.get_label()][0]
        assert floor.get_ydata()[0] == pytest.approx(est.value / np.pi)

    @pytest.mark.parametrize("dsh", [False, True])
    def test_increment_variance(self, xp: Any, dsh: bool) -> None:
        """Measured points, the fit and the intercept guide."""
        if dsh:
            z, m = _beat(xp)
            est = analysis.estimate_linewidth(
                z, DshIncrement(delay=m / FS), sampling_rate=FS
            )
        else:
            est = analysis.estimate_linewidth(
                _phase(xp), IncrementSlope(), sampling_rate=R
            )
        fig, ax = plot_increment_variance(est)
        np.testing.assert_allclose(ax.lines[0].get_ydata(), est.increment_var[0])
        assert len(ax.lines) == 3  # points, fit, intercept
        _, ax = plot_increment_variance(est, label="first", annotations=False)
        _, ax = plot_increment_variance(
            est, ax=ax, label="second", annotations=False, legend=True
        )
        assert [line.get_label() for line in ax.lines] == ["first", "second"]

    def test_increment_variance_subtract(self, xp: Any) -> None:
        """IncrementSubtract has no fit: only its lag-1 point is drawn."""
        est = analysis.estimate_linewidth(
            _phase(xp), IncrementSubtract(), sampling_rate=R
        )
        fig, ax = plot_increment_variance(est)
        assert len(ax.lines) == 1

    @pytest.mark.allow_warnings
    def test_dsh_beat_psd(self, xp: Any) -> None:
        """Centred on the peak, with both width contours at the method's depth."""
        z, m = _beat(xp)
        method = DshLorentzian(delay=m / FS, level_db=15.0)
        est = analysis.estimate_linewidth(z, method, sampling_rate=FS)
        fig, ax = plot_dsh_beat_psd(est)
        depths = [ln.get_ydata()[0] for ln in ax.lines[1:]]
        assert -15.0 in depths and -3.01 in depths
        # Zoomed on the line: twice the -15 dB width on each side.
        w = 2 * np.sqrt(10**1.5 - 1) * est.value
        assert ax.get_xlim() == pytest.approx((-2 * w, 2 * w))
        with patch("matplotlib.pyplot.show"):
            assert plot_dsh_beat_psd(est, show=True) is None

    @pytest.mark.parametrize(
        ("plot", "match"),
        [
            (plot_frequency_noise_psd, "BetaSeparation or DshFmFit"),
            (plot_dsh_beat_psd, "DshLorentzian"),
        ],
    )
    def test_wrong_estimate_raises(self, xp: Any, plot: Any, match: str) -> None:
        est = analysis.estimate_linewidth(_phase(xp), IncrementSlope(), sampling_rate=R)
        with pytest.raises(ValueError, match=match):
            plot(est)

    @pytest.mark.allow_warnings  # a short phase: the β-area is unresolved
    def test_carrier_phase_characterization(self, xp: Any) -> None:
        """The dashboard draws the four results in a 2x2 grid."""
        phi = _phase(xp)
        drift_phase, _ = analysis.separate_drift_phase_noise(
            phi, sampling_rate=R, cutoff=1e7
        )
        drift = analysis.frequency_drift(drift_phase, sampling_rate=R)
        fig, axes = plot_carrier_phase_characterization(
            phi,
            drift=drift,
            linewidth=analysis.estimate_linewidth(
                phi, BetaSeparation(), sampling_rate=R
            ),
            allan=analysis.allan_deviation(drift.df, sampling_rate=R),
            sampling_rate=R,
            drift_phase=drift_phase,
            drift_cutoff=1e7,
            reference_amplitude=1e5,
        )
        assert axes.shape == (2, 2)
        assert all(ax.lines for ax in axes.flat)
        fig2, reused = plot_carrier_phase_characterization(
            phi,
            drift=drift,
            linewidth=analysis.estimate_linewidth(
                phi, BetaSeparation(), sampling_rate=R
            ),
            allan=analysis.allan_deviation(drift.df, sampling_rate=R),
            sampling_rate=R,
            ax=axes,
        )
        assert fig2 is fig
        assert all(a is b for a, b in zip(reused.flat, axes.flat, strict=True))


def test_carrier_phase_decomposition(xp: Any) -> None:
    """Verify plot_carrier_phase_decomposition with and without drift, and with n_train."""
    phi_raw = xp.linspace(0, 4 * np.pi, 300)
    phi_smooth = xp.linspace(0, 4 * np.pi, 300) + 0.05
    fig, axes = plot_carrier_phase_decomposition(
        phi=phi_raw,
        drift=phi_smooth,
        symbol_rate=1e6,
        num_train_symbols=25,
        show=False,
    )
    assert fig is not None

    fig2, axes2 = plot_carrier_phase_decomposition(
        phi=phi_raw,
        symbol_rate=1e6,
        show=False,
    )
    assert fig2 is not None
