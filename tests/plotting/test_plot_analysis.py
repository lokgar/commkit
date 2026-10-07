"""Tests for analysis plotting functions (frequency drift, Allan deviation, DSH, phase noise)."""

from typing import Any

import numpy as np

from commkit.analysis import AllanDeviation, BetaSeparationLinewidth, FrequencyDrift
from commkit.plotting.analysis import (
    plot_allan_deviation,
    plot_carrier_phase_characterization,
    plot_dsh_beat_psd,
    plot_frequency_drift,
    plot_frequency_noise_psd,
    plot_increment_variance,
)


class TestPlotAnalysis:
    """Tests for physical signal analysis and laser diagnostics plots."""

    def test_frequency_drift_siso(self, xp: Any) -> None:
        """Verify plot_frequency_drift for SISO frequency trajectory."""
        df = xp.linspace(-5e3, 5e3, 100)
        fig, ax = plot_frequency_drift(df, symbol_rate=1e6, amp_ref=6e3, show=False)
        assert fig is not None

    def test_frequency_drift_mimo(self, xp: Any) -> None:
        """Verify plot_frequency_drift for dual-pol / multichannel."""
        df = xp.stack([xp.linspace(-2e3, 2e3, 50), xp.linspace(-1e3, 3e3, 50)])
        fig, ax = plot_frequency_drift(df, symbol_rate=1e6, show=False)
        assert fig is not None

    def test_allan_deviation(self, xp: Any) -> None:
        """Verify plot_allan_deviation with white-FM guide slope."""
        tau = xp.logspace(-6, -2, 20)
        adev = 1e4 / xp.sqrt(tau)
        fig, ax = plot_allan_deviation(tau, adev, reference_slopes=True, show=False)
        assert fig is not None

    def test_frequency_noise_psd(self, xp: Any) -> None:
        """Verify plot_frequency_noise_psd with plateau bins, beta-line, and used mask."""
        f = xp.logspace(3, 7, 200)
        S_f = xp.full_like(f, 1e4)
        fig, ax = plot_frequency_noise_psd(
            f, S_f, floor=1e4 * np.pi, band=(1e4, 1e6), show=False
        )
        assert fig is not None

        beta = (8 * np.log(2) / np.pi**2) * f
        above = S_f > beta
        used = (f >= 1e4) & (f <= 1e6)
        fig2, ax2 = plot_frequency_noise_psd(
            f,
            S_f,
            beta_line=beta,
            floor=1e4 * np.pi,
            band=(1e4, 1e6),
            above=above,
            used=used,
            show=False,
        )
        assert fig2 is not None

    def test_increment_variance(self, xp: Any) -> None:
        """Verify plot_increment_variance."""
        lags = xp.logspace(-6, -2, 10)
        var = 2 * np.pi * 1e5 * lags
        fig, ax = plot_increment_variance(lags, var, slope=2 * np.pi * 1e5, show=False)
        assert fig is not None

    def test_dsh_beat_psd(self, xp: Any) -> None:
        """Verify plot_dsh_beat_psd with linewidths and show=True."""
        from unittest.mock import patch

        f = xp.linspace(-1e6, 1e6, 256)
        psd = 1.0 / (1.0 + (f / 1e5) ** 2)
        fig, ax = plot_dsh_beat_psd(
            f, psd, f_peak=0.0, linewidth=1e5, linewidth_3db=2e5, show=False
        )
        assert fig is not None

        with patch("matplotlib.pyplot.show"):
            ret = plot_dsh_beat_psd(f, psd, show=True)
        assert ret is None

    def test_carrier_phase_characterization(self, xp: Any) -> None:
        """Verify plot_carrier_phase_characterization full dashboard."""
        f = xp.logspace(3, 6, 100)
        S_f = xp.full(100, 1e4)
        beta = (8 * np.log(2) / np.pi**2) * f
        report = {
            "phi": xp.cumsum(xp.random.randn(500) * 0.05),
            "drift": xp.linspace(0, 1, 500),
            "drift_metrics": FrequencyDrift(
                df=xp.linspace(-100, 100, 500), std=0.0, pp=0.0, max_abs=0.0
            ),
            "linewidth_beta": BetaSeparationLinewidth(
                linewidth=1e4,
                linewidth_floor=1e4,
                n_segments=8,
                area_hz2=0.0,
                f=f,
                S_f=S_f,
                beta_line=beta,
                above=S_f > beta,
                used=(f >= 1e4) & (f <= 1e5),
                band=(1e4, 1e5),
            ),
            "allan": AllanDeviation(
                tau_s=xp.logspace(-5, -2, 10), adev=xp.full(10, 1e3)
            ),
        }
        fig, axes = plot_carrier_phase_characterization(
            report,
            symbol_rate=1e6,
            drift_cutoff=1e6,
            band=(1e4, 1e5),
            amp_ref=120.0,
            show=False,
        )
        assert fig is not None
        assert axes.shape == (2, 2)
