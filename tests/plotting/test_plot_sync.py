"""Tests for synchronization plotting functions (timing, carrier frequency, carrier phase).

The diagnostics take the estimates of the public estimators, so every test
builds a real estimate and checks that the plot draws its fields.
"""

from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from commkit import frequency, recovery, timing
from commkit.core import Preamble, SingleCarrierFrame
from commkit.mapping import Constellation
from commkit.plotting.sync import (
    plot_carrier_phase_decomposition,
    plot_carrier_phase_trajectory,
    plot_frequency_offset_blockwise_result,
    plot_frequency_offset_spectrum,
    plot_mm_autocorrelation,
    plot_pilot_phase_estimate,
    plot_pilot_tone_phase_estimate,
    plot_pilot_tones_phase_estimate,
    plot_timing_correlation,
)
from commkit.spectral import add_pilot_tone

FS = 1e6


def _qpsk(xp: Any, n: int = 4096, channels: int = 1, df: float = 0.0, seed: int = 0):
    """QPSK at 1 sample per symbol with a frequency offset ``df`` (Hz at FS)."""
    rng = np.random.default_rng(seed)
    pts = Constellation.psk(4).points
    s = pts[rng.integers(0, 4, (channels, n))]
    x = s * np.exp(2j * np.pi * df * np.arange(n) / FS)
    x = x + 0.05 * (rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape))
    x = x.astype(np.complex64)
    return xp.asarray(x[0] if channels == 1 else x), s


class TestPlotTimingSync:
    """Timing correlation from a TimingEstimate."""

    def _estimate(self, xp: Any, channels: int = 1):
        frame = SingleCarrierFrame(
            payload_len=200,
            preamble=Preamble(sequence_type="barker", length=13, num_streams=channels),
            num_streams=channels,
        )
        sig = frame.to_signal(sps=1, symbol_rate=FS)
        pad = np.zeros((channels, 37), dtype=np.complex64)
        x = np.concatenate([pad, np.atleast_2d(sig.samples)], axis=-1)
        sig = sig.replace(samples=xp.asarray(x[0] if channels == 1 else x))
        return timing.estimate_timing(sig)

    @pytest.mark.parametrize("channels", [1, 2])
    def test_timing_correlation(self, xp: Any, channels: int) -> None:
        """One overall/zoom row per channel; the marked peak is the start."""
        est = self._estimate(xp, channels)
        fig, axes = plot_timing_correlation(est, show=False)
        assert np.asarray(axes, dtype=object).shape == (channels, 2)
        starts = np.atleast_1d(
            np.asarray(est.integer.get() if xp is not np else est.integer)
        )
        assert int(starts[0]) == 37
        # The peak line sits at the estimated start.
        vlines = [
            ln.get_xdata()[0]
            for ln in axes[0][0].lines[1:]
            if len(set(ln.get_xdata())) == 1
        ]
        assert 37 in vlines

    def test_timing_correlation_show(self, xp: Any) -> None:
        with patch("matplotlib.pyplot.show"):
            assert plot_timing_correlation(self._estimate(xp), show=True) is None


class TestPlotFrequencySync:
    """Frequency-offset diagnostics from FrequencyOffsetEstimates."""

    @pytest.mark.parametrize("channels", [1, 2])
    def test_mm_autocorrelation(self, xp: Any, channels: int) -> None:
        x, _ = _qpsk(xp, channels=channels, df=2e3)
        est = frequency.estimate_frequency_offset(
            x, frequency.MengaliMorelli(), sampling_rate=FS
        )
        fig, axes = plot_mm_autocorrelation(est, sampling_rate=FS, show=False)
        assert fig is not None

    def test_mm_autocorrelation_needs_mm_estimate(self, xp: Any) -> None:
        x, _ = _qpsk(xp, df=2e3)
        est = frequency.estimate_frequency_offset(
            x, frequency.MthPower(power=4), sampling_rate=FS
        )
        with pytest.raises(ValueError, match="autocorrelation"):
            plot_mm_autocorrelation(est, sampling_rate=FS)

    def test_frequency_offset_spectrum(self, xp: Any) -> None:
        """The estimate line is drawn at the estimated offset."""
        x, _ = _qpsk(xp, df=3e3)
        est = frequency.estimate_frequency_offset(
            x, frequency.MthPower(power=4), sampling_rate=FS
        )
        fig, ax = plot_frequency_offset_spectrum(
            est, search_range=(-1e4, 1e4), show=False
        )
        f_hat = float(est.value)
        assert f_hat == pytest.approx(3e3, abs=50)
        assert any(ln.get_xdata()[0] == pytest.approx(f_hat) for ln in ax.lines[1:])

    def test_frequency_offset_blockwise_result(self, xp: Any) -> None:
        """The recomputed trajectory follows the block estimates."""
        n = 8192
        x, _ = _qpsk(xp, n=n, df=2e3)
        est = frequency.estimate_frequency_offset(
            x, frequency.MthPower(power=4, block_size=1024), sampling_rate=FS
        )
        fig, axes = plot_frequency_offset_blockwise_result(
            est, num_samples=n, sampling_rate=FS, max_points=0, show=False
        )
        df_line = axes[0].lines[0].get_ydata()  # kHz
        assert np.median(df_line) == pytest.approx(2.0, abs=0.1)

    def test_pilot_phase_frequency_estimate(self, xp: Any) -> None:
        """A PilotSymbols estimate labels the fit with its offset."""
        x, s = _qpsk(xp, df=1e3)
        idx = np.arange(0, 4096, 64)
        est = frequency.estimate_frequency_offset(
            x, frequency.PilotSymbols(idx, s[0, idx]), sampling_rate=FS
        )
        fig, axes = plot_pilot_phase_estimate(est, sampling_rate=FS, show=False)
        assert np.asarray(axes, dtype=object).shape == (1, 1)


class TestPlotCarrierPhase:
    """Carrier-phase diagnostics from CarrierPhaseEstimates."""

    def test_carrier_phase_trajectory_estimate(self, xp: Any) -> None:
        """A block estimate marks its block centres."""
        x, _ = _qpsk(xp, n=1024)
        est = recovery.estimate_carrier_phase(
            x,
            recovery.ViterbiViterbi(block_size=64),
            constellation=Constellation.psk(4),
        )
        fig, ax = plot_carrier_phase_trajectory(est, n_train=100, show=False)
        assert len(ax.lines) == 1 + len(est.block_centers) + 1

    def test_carrier_phase_trajectory_array(self, xp: Any) -> None:
        phi = xp.cumsum(xp.ones((2, 100)) * 0.01, axis=1)
        fig, ax = plot_carrier_phase_trajectory(phi, show=False)
        assert len(ax.lines) == 2

    def test_pilot_phase_carrier_estimate(self, xp: Any) -> None:
        """PilotAided: pilot panel plus the interpolated trajectory."""
        x, s = _qpsk(xp, df=1e3)
        idx = np.arange(0, 4096, 32)
        est = recovery.estimate_carrier_phase(x, recovery.PilotAided(idx, s[0, idx]))
        fig, axes = plot_pilot_phase_estimate(est, sampling_rate=FS, show=False)
        assert np.asarray(axes, dtype=object).shape == (1, 2)

    def test_pilot_tone_phase_estimate(self, xp: Any) -> None:
        x, _ = _qpsk(xp, n=8192)
        method = recovery.PilotTone(frequency=3e5, bandwidth=2e4)
        x = add_pilot_tone(x, frequency=3e5, sampling_rate=FS)
        est = recovery.estimate_carrier_phase(x, method, sampling_rate=FS)
        fig, axes = plot_pilot_tone_phase_estimate(
            est, x, method=method, sampling_rate=FS, show=False
        )
        assert len(axes) == 2

    def test_pilot_tones_phase_estimate(self, xp: Any) -> None:
        x, _ = _qpsk(xp, n=8192)
        tones = (2.5e5, 3.5e5)
        for f in tones:
            x = add_pilot_tone(x, frequency=f, sampling_rate=FS)
        est = recovery.estimate_carrier_phase(
            x, recovery.PilotTones(tones, bandwidth=2e4), sampling_rate=FS
        )
        fig, axes = plot_pilot_tones_phase_estimate(est, show=False)
        assert len(axes) == 2

    def test_wrong_estimate_raises(self, xp: Any) -> None:
        x, _ = _qpsk(xp, n=1024)
        est = recovery.estimate_carrier_phase(
            x,
            recovery.ViterbiViterbi(block_size=64),
            constellation=Constellation.psk(4),
        )
        with pytest.raises(ValueError, match="PilotTones"):
            plot_pilot_tones_phase_estimate(est)

    def test_carrier_phase_decomposition(self, xp: Any) -> None:
        """Verify plot_carrier_phase_decomposition with and without drift, and with n_train."""
        phi_raw = xp.linspace(0, 4 * np.pi, 300)
        phi_smooth = xp.linspace(0, 4 * np.pi, 300) + 0.05
        fig, axes = plot_carrier_phase_decomposition(
            phi=phi_raw,
            drift=phi_smooth,
            symbol_rate=1e6,
            n_train=25,
            show=False,
        )
        assert fig is not None

        fig2, axes2 = plot_carrier_phase_decomposition(
            phi=phi_raw,
            symbol_rate=1e6,
            show=False,
        )
        assert fig2 is not None
