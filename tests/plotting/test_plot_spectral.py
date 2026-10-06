"""Tests for Power Spectral Density (PSD) and spectral/spectrogram plotting."""

import logging
from typing import Any
from unittest.mock import patch

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from commkit import generate, plotting, spectral
from commkit.filtering import RRC
from commkit.mapping import Constellation
from commkit.plotting import plot_psd, plot_spectrogram


class TestPlotSpectralAndPSD:
    """Tests for Power Spectral Density (PSD) and spectral plotting."""

    def test_psd(self, xp: Any) -> None:
        """Verify Power Spectral Density (PSD) plotting."""
        samples = xp.random.randn(256)
        fig, ax = plot_psd(samples, sampling_rate=10.0, show=False)
        assert fig is not None

        fig2, ax2 = plot_psd(
            samples,
            sampling_rate=10.0,
            nperseg=64,
            window=("kaiser", 8.0),
            noverlap=32,
            nfft=128,
            scaling="spectrum",
            show=False,
        )
        assert fig2 is not None

    def test_psd_mimo_grid(self, xp: Any) -> None:
        """Verify MIMO PSD plotting uses an optimized grid layout."""
        samples = xp.random.randn(4, 256)
        fig, axes = plot_psd(samples, sampling_rate=10.0, show=False)
        assert fig is not None
        assert axes.shape == (2, 2)

    def test_psd_wavelength(self, xp: Any) -> None:
        """Verify PSD plotting with wavelength axis."""
        samples = xp.random.randn(256)
        fig, ax = plot_psd(
            samples, sampling_rate=1e15, x_axis="wavelength", domain="OPT", show=False
        )
        assert fig is not None
        assert "Wavelength" in ax.get_xlabel()

    def test_psd_auto_scale(self, xp: Any) -> None:
        """Verify PSD frequency axis uses an SI-prefixed engineering tick formatter."""
        samples = xp.random.randn(256)
        fig, ax = plot_psd(samples, sampling_rate=2e12, show=False)
        assert ax.get_xlabel() == "Frequency [Hz]"
        fmt = ax.xaxis.get_major_formatter()
        assert isinstance(fmt, mticker.EngFormatter)
        assert "T" in fmt(2e12)

        fig, ax = plot_psd(samples, sampling_rate=2e6, show=False)
        assert "M" in ax.xaxis.get_major_formatter()(2e6)

    def test_psd_axis_overlay_warning(self, caplog: Any, xp: Any) -> None:
        """Verify warning when single axis is provided for multichannel PSD."""
        sig = xp.ones((2, 256))
        fig, ax = plt.subplots()
        with patch("matplotlib.pyplot.show"):
            plot_psd(sig, ax=ax)
        assert "Multiple channels detected but single axis provided" in caplog.text

    def test_psd_wavelength_warning(self, caplog: Any, xp: Any) -> None:
        """Verify wavelength warning."""
        sig = xp.ones(256)
        with patch("matplotlib.pyplot.show"):
            plot_psd(sig, x_axis="wavelength", domain="RF")
        assert (
            "Wavelength plotting is typically used for optical signals" in caplog.text
        )

    def test_psd_auto_scale_small(self, xp: Any) -> None:
        """Test auto-scaling for small frequencies (Hz)."""
        sig = xp.ones(256)
        fig, ax = plot_psd(sig, sampling_rate=1.0)
        assert "Hz" in ax.get_xlabel()

    def test_psd_ghz_scaling(self, xp: Any) -> None:
        """PSD with GHz-range sampling rate triggers GHz scale factor."""
        samples = xp.random.randn(256).astype(xp.float32)
        fig, ax = plot_psd(samples, sampling_rate=5e9, show=False)
        assert fig is not None

    def test_psd_khz_scaling(self, xp: Any) -> None:
        """PSD with kHz-range sampling rate triggers kHz scale factor."""
        samples = xp.random.randn(256).astype(xp.float32)
        fig, ax = plot_psd(samples, sampling_rate=5e3, show=False)
        assert fig is not None

    def test_psd_hz_scaling(self, xp: Any) -> None:
        """PSD with Hz-range sampling rate uses default Hz unit."""
        samples = xp.random.randn(256).astype(xp.float32)
        fig, ax = plot_psd(samples, sampling_rate=100.0, show=False)
        assert fig is not None

    def test_psd_xlim_ylim(self, xp: Any) -> None:
        """PSD with xlim and ylim parameters applies axis limits."""
        samples = xp.random.randn(256).astype(xp.float32)
        fig, ax = plot_psd(
            samples, sampling_rate=1e6, xlim=(-0.4, 0.4), ylim=(-80, 0), show=False
        )
        assert fig is not None

    def test_psd_show(self, xp: Any) -> None:
        """PSD with show=True calls plt.show() and returns None."""
        samples = xp.random.randn(256).astype(xp.float32)
        with patch("matplotlib.pyplot.show"):
            result = plot_psd(samples, sampling_rate=1e6, show=True)
        assert result is None

    def test_psd_multichannel_show(self, xp: Any) -> None:
        """PSD multichannel with show=True calls plt.show() and returns None."""
        samples = xp.random.randn(2, 256).astype(xp.float32)
        with patch("matplotlib.pyplot.show"):
            result = plot_psd(samples, sampling_rate=1e6, show=True)
        assert result is None

    def test_psd_multichannel_single_axis_warning(self, xp: Any, caplog: Any) -> None:
        """PSD multichannel with single axis warns and overlays all channels on it."""
        samples = xp.random.randn(2, 256).astype(xp.float32)
        fig0, ax0 = plt.subplots()
        caplog.set_level(logging.WARNING)
        fig, axes = plot_psd(samples, sampling_rate=1e6, ax=ax0, show=False)
        assert "Overlaying plots" in caplog.text

    def test_psd_multichannel_axes_array(self, xp: Any) -> None:
        """PSD multichannel with axes ndarray normalizes axes to 2D layout."""
        samples = xp.random.randn(2, 256).astype(xp.float32)
        fig0, axes0 = plt.subplots(1, 2)
        fig, axes = plot_psd(samples, sampling_rate=1e6, ax=axes0, show=False)
        assert fig is not None


class TestPlotSpectrogram:
    """Tests for spectrogram plotting."""

    def test_spectrogram_plot_siso(self, xp: Any) -> None:
        """Verify spectrogram plotting for SISO (1D) signals."""
        fs = 100.0
        t = xp.arange(1000) / fs
        samples = xp.sin(2 * xp.pi * 20.0 * t)

        fig, ax = plot_spectrogram(samples, sampling_rate=fs, show=False)
        assert fig is not None
        assert ax is not None

        fig, ax = plot_spectrogram(
            samples,
            sampling_rate=fs,
            xlim=(-10.0, 30.0),
            ylim=(1.0, 5.0),
            show=False,
        )
        assert fig is not None
        assert ax is not None

        with patch("matplotlib.pyplot.show"):
            res = plot_spectrogram(samples, sampling_rate=fs, show=True)
        assert res is None

    def test_spectrogram_plot_mimo(self, xp: Any) -> None:
        """Verify spectrogram plotting for MIMO (2D) signals generates subplots."""
        fs = 100.0
        t = xp.arange(1000) / fs
        samples = xp.stack([xp.sin(2 * xp.pi * 20.0 * t), xp.cos(2 * xp.pi * 10.0 * t)])

        fig, axes = plot_spectrogram(samples, sampling_rate=fs, show=False)
        assert fig is not None
        assert isinstance(axes, np.ndarray)
        assert axes.size == 2

    def test_signal_spectrogram_convenience(self, xp: Any) -> None:
        """Verify core.Signal.spectrogram and plot_spectrogram convenience methods."""
        fs = 100.0
        sig = generate(
            Constellation.psk(4),
            100,
            symbol_rate=10.0,
            sps=int(fs / 10.0),
            pulse=RRC(0.35),
            rng=42,
        )
        sig = sig.replace(samples=xp.asarray(sig.samples))

        spec = spectral.spectrogram(sig, nperseg=64, noverlap=32)
        assert isinstance(spec.frequencies, xp.ndarray)
        assert isinstance(spec.times, xp.ndarray)
        assert isinstance(spec.values, xp.ndarray)
        assert len(spec.frequencies) == 64
        assert spec.values.shape[-1] == len(spec.times)

        fig, ax = plotting.plot_spectrogram(sig, nperseg=64, show=False)
        assert fig is not None
        assert ax is not None
