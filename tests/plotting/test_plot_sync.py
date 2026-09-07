"""Tests for synchronization plotting functions (timing, carrier frequency, carrier phase)."""

from typing import Any
from unittest.mock import patch

import numpy as np

from commkit.plotting.sync import (
    plot_carrier_phase_decomposition,
    plot_carrier_phase_trajectory,
    plot_frequency_offset_blockwise_result,
    plot_frequency_offset_spectrum,
    plot_mm_autocorrelation,
    plot_pilot_phase_estimate,
    plot_pilot_tone_phase_estimate,
    plot_timing_correlation,
)


class TestPlotTimingSync:
    """Tests for timing correlation and synchronization diagnostic plots."""

    def test_timing_correlation_siso(self, xp: Any) -> None:
        """Verify plot_timing_correlation with SISO input."""
        corr = xp.zeros(200, dtype=xp.float32)
        corr[100] = 5.0
        corr[98:103] = xp.array([1.0, 3.0, 5.0, 3.0, 1.0])
        fig, axes = plot_timing_correlation(
            corr_mag=corr,
            peak_indices=100,
            norm_factors=1.0,
            threshold=0.5,
            offset=10,
            show=False,
        )
        assert fig is not None
        assert axes.shape == (1, 2)

    def test_timing_correlation_mimo(self, xp: Any) -> None:
        """Verify plot_timing_correlation with multi-channel MIMO input."""
        corr = xp.zeros((2, 200), dtype=xp.float32)
        corr[0, 80] = 4.0
        corr[1, 120] = 4.5
        fig, axes = plot_timing_correlation(
            corr_mag=corr,
            peak_indices=xp.array([80, 120]),
            norm_factors=xp.array([1.0, 1.0]),
            threshold=0.4,
            show=False,
        )
        assert fig is not None
        assert axes.shape == (2, 2)

    def test_timing_correlation_show(self, xp: Any) -> None:
        """Verify plot_timing_correlation with show=True returns None."""
        corr = xp.ones(50)
        with patch("matplotlib.pyplot.show"):
            res = plot_timing_correlation(corr, 25, 1.0, 0.5, show=True)
        assert res is None


class TestPlotFrequencySync:
    """Tests for frequency offset estimation and spectrum diagnostic plots."""

    def test_mm_autocorrelation_siso(self, xp: Any) -> None:
        """Verify plot_mm_autocorrelation for 1D lag autocorrelation."""
        lags = xp.array([0.9 + 0.1j, 0.8 + 0.2j, 0.7 + 0.3j], dtype=xp.complex64)
        fig, axes = plot_mm_autocorrelation(
            lags, f_est=10e3, sampling_rate=1e6, M=4, show=False
        )
        assert fig is not None

    def test_mm_autocorrelation_mimo(self, xp: Any) -> None:
        """Verify plot_mm_autocorrelation for multi-channel input."""
        lags = xp.ones((2, 4), dtype=xp.complex64)
        fig, axes = plot_mm_autocorrelation(
            lags, f_est=[10e3, -5e3], sampling_rate=1e6, M=4, show=False
        )
        assert fig is not None

    def test_frequency_offset_spectrum(self, xp: Any) -> None:
        """Verify plot_frequency_offset_spectrum with search range overlay."""
        nfft = 128
        mag = xp.ones(nfft, dtype=xp.float32)
        mag[64] = 10.0
        freqs = xp.fft.fftfreq(nfft, d=1.0 / 1e6)
        fig, ax = plot_frequency_offset_spectrum(
            mag_spectrum=mag,
            freqs=freqs,
            M=4,
            k_peaks=64,
            f_estimates=[15e3],
            search_range=(-50e3, 50e3),
            show=False,
        )
        assert fig is not None

    def test_frequency_offset_blockwise_result(self, xp: Any) -> None:
        """Verify plot_frequency_offset_blockwise_result."""
        t_centers = xp.array([50.0, 150.0, 250.0])
        df_estimates = xp.array([24e3, 25e3, 26e3])
        n_grid = xp.arange(300.0)
        df_dense = xp.linspace(24e3, 26e3, 300)
        phase_trajectory = xp.cumsum(df_dense) * (2 * np.pi / 1e6)

        fig, axes = plot_frequency_offset_blockwise_result(
            t_centers=t_centers,
            df_estimates=df_estimates,
            n_grid=n_grid,
            df_dense=df_dense,
            phase_trajectory=phase_trajectory,
            show=False,
        )
        assert fig is not None


class TestPlotPhaseSync:
    """Tests for carrier phase trajectory and pilot-aided sync plotting."""

    def test_carrier_phase_trajectory_siso(self, xp: Any) -> None:
        """Verify plot_carrier_phase_trajectory with SISO phase array."""
        phi = xp.linspace(0, np.pi, 200)
        fig, ax = plot_carrier_phase_trajectory(
            phi, block_centers=[50, 100, 150], n_train=30, show=False
        )
        assert fig is not None

    def test_carrier_phase_trajectory_mimo(self, xp: Any) -> None:
        """Verify plot_carrier_phase_trajectory with MIMO phase array."""
        phi = xp.stack([xp.linspace(0, 1, 100), xp.linspace(0.5, 1.5, 100)])
        fig, ax = plot_carrier_phase_trajectory(phi, show=False)
        assert fig is not None

    def test_pilot_phase_estimate(self, xp: Any) -> None:
        """Verify plot_pilot_phase_estimate unwrapped/detrended view and full trajectory."""
        pilot_indices = xp.arange(10, 500, 10)
        phases = xp.linspace(-1.0, 1.0, len(pilot_indices))
        fig, ax = plot_pilot_phase_estimate(
            pilot_indices=pilot_indices,
            phi_pilots_u=phases,
            f_est=5e3,
            sampling_rate=1e6,
            show=False,
        )
        assert fig is not None

        # Exercise with full interpolated trajectory
        phi_full = xp.zeros((1, 500))
        fig2, axes2 = plot_pilot_phase_estimate(
            pilot_indices=pilot_indices,
            phi_pilots_u=phases,
            phi_full=phi_full,
            f_est=5e3,
            sampling_rate=1e6,
            show=False,
        )
        assert fig2 is not None

    def test_pilot_tone_phase_estimate(self, xp: Any) -> None:
        """Verify plot_pilot_tone_phase_estimate."""
        freqs = xp.linspace(-500e3, 500e3, 256)
        mag = xp.ones(256)
        window = xp.ones(256)
        theta = xp.linspace(0.1, 0.9, 128)
        fig, ax = plot_pilot_tone_phase_estimate(
            freqs=freqs,
            mag_spectrum=mag,
            window=window,
            f_tones=100e3,
            theta=theta,
            tone_frequency=100e3,
            bandwidth=20e3,
            show=False,
        )
        assert fig is not None

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

    def test_pilot_tones_phase_estimate(self, xp: Any) -> None:
        """Verify plot_pilot_tones_phase_estimate for MRC common-phase tracking."""
        from commkit.plotting.sync import plot_pilot_tones_phase_estimate

        delta = [xp.zeros(100), xp.linspace(0, 0.1, 100)]
        phi = xp.linspace(0, 0.5, 100)
        fig, axes = plot_pilot_tones_phase_estimate(
            delta=delta,
            phi=phi,
            ref=0,
            used=[0, 1],
            show=False,
        )
        assert fig is not None
        assert len(axes) == 2
