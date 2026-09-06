"""Tests for signal visualization and plotting tools."""

import logging
from typing import Any
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pytest

from commkit import filtering, generate_psk, plotting, spectral
from commkit.core import Signal
from commkit.plotting import (
    _create_subplot_grid,
    _plot_eye_traces,
    apply_default_theme,
    plot_constellation,
    plot_eye_diagram,
    plot_filter_response,
    plot_ideal_constellation,
    plot_psd,
    plot_spectrogram,
    plot_time_domain,
)


class TestPlotEyeDiagram:
    """Tests for eye diagram plotting (line, hist, multichannel, and validation)."""

    @pytest.mark.parametrize("type", ["line", "hist"])
    def test_eye_diagram_real(
        self, backend_device: str, xp: Any, type: str
    ) -> None:
        """Verify eye diagram generation for real-valued signals."""
        samples = xp.random.randn(1000)
        sps = 4
        fig, ax = plot_eye_diagram(samples, sps=sps, type=type, show=False)
        assert fig is not None
        assert ax is not None
        plt.close("all")

    def test_eye_diagram_complex(self, backend_device: str, xp: Any) -> None:
        """Verify eye diagram generation for complex-valued signals (I/Q)."""
        samples = xp.random.randn(1000) + 1j * xp.random.randn(1000)
        sps = 4

        fig, ax = plot_eye_diagram(samples, sps=sps, show=False)
        assert fig is not None
        assert isinstance(ax, (list, tuple, np.ndarray))
        assert len(ax) == 2
        plt.close("all")

        fig, axes = plt.subplots(2, 1)
        fig, ax_ret = plot_eye_diagram(samples, sps=sps, ax=axes, show=False)
        assert ax_ret is axes
        plt.close("all")

    def test_eye_diagram_multichannel(self, backend_device: str, xp: Any) -> None:
        """Verify eye diagram for multichannel complex signals."""
        samples = xp.random.randn(2, 100) + 1j * xp.random.randn(2, 100)
        fig, axes = plot_eye_diagram(samples, sps=4, show=False)
        assert fig is not None
        assert axes.shape == (2, 2)
        plt.close("all")

    def test_eye_diagram_axis_error(self, backend_device: str, xp: Any) -> None:
        """Verify error when too few axes are provided for multichannel eye diagram."""
        samples = xp.zeros((2, 100))
        fig, ax = plt.subplots(1)
        with pytest.raises(ValueError, match="Not enough axes"):
            plot_eye_diagram(samples, sps=4, ax=[ax], show=False)
        plt.close("all")

        with pytest.raises(ValueError, match="must provide a list of axes"):
            plot_eye_diagram(samples, sps=4, ax=ax, show=False)
        plt.close("all")

    def test_eye_diagram_sps_error(self, backend_device: str, xp: Any) -> None:
        """Verify error for non-integer sps."""
        with pytest.raises(ValueError, match="sps to be a positive integer"):
            plot_eye_diagram(xp.ones(10), sps=2.5)
            plt.close("all")

    def test_eye_diagram_trace_len_error(self, backend_device: str, xp: Any) -> None:
        """Verify error when signal too short for eye trace."""
        with pytest.raises(
            ValueError, match="Signal is shorter than the required trace length"
        ):
            _plot_eye_traces(
                xp.ones(5), sps=10, num_symbols=2, ax=MagicMock(), type="line", title=None
            )
            plt.close("all")

    def test_eye_diagram_invalid_type(self, backend_device: str, xp: Any) -> None:
        """Verify error for unknown eye type."""
        with pytest.raises(ValueError, match="Unknown type"):
            _plot_eye_traces(
                xp.ones(100), sps=4, num_symbols=2, ax=MagicMock(), type="magic", title=None
            )
            plt.close("all")

    def test_eye_diagram_multichannel_axes_array(
        self, backend_device: str, xp: Any
    ) -> None:
        """Eye diagram multichannel with axes array reshapes them for per-channel plotting."""
        samples = xp.random.randn(2, 1000).astype(xp.float32)
        fig0, axes0 = plt.subplots(2, 1)
        fig, axes = plot_eye_diagram(samples, sps=4, ax=list(axes0), show=False)
        assert fig is not None
        plt.close("all")

    def test_eye_diagram_multichannel_show(
        self, backend_device: str, xp: Any
    ) -> None:
        """Eye diagram multichannel with show=True calls plt.show() and returns None."""
        samples = xp.random.randn(2, 1000).astype(xp.float32)
        with patch("matplotlib.pyplot.show"):
            result = plot_eye_diagram(samples, sps=4, show=True)
        assert result is None
        plt.close("all")

    def test_eye_diagram_real_with_ax(self, backend_device: str, xp: Any) -> None:
        """Eye diagram for real signal with a single pre-existing axis uses it directly."""
        samples = xp.random.randn(500).astype(xp.float32)
        fig0, ax0 = plt.subplots()
        fig, ax = plot_eye_diagram(samples, sps=4, ax=ax0, show=False)
        assert fig is not None
        plt.close("all")

    def test_eye_diagram_siso_show(self, backend_device: str, xp: Any) -> None:
        """plot_eye_diagram() 1D with show=True calls plt.show() and returns None."""
        samples = xp.random.randn(1000).astype(xp.float32)
        with patch("matplotlib.pyplot.show"):
            result = plot_eye_diagram(samples, sps=4, show=True)
        assert result is None
        plt.close("all")

    def test_eye_diagram_complex_single_ax_error(
        self, backend_device: str, xp: Any
    ) -> None:
        """plot_eye_diagram() with complex signal and single ax raises ValueError requiring two axes."""
        samples = (xp.random.randn(500) + 1j * xp.random.randn(500)).astype(xp.complex64)
        fig0, ax0 = plt.subplots()
        with pytest.raises(ValueError, match="complex"):
            plot_eye_diagram(samples, sps=4, ax=ax0, show=False)
        plt.close("all")

    def test_eye_diagram_dense_line(self, backend_device: str, xp: Any) -> None:
        """plot_eye_diagram() 'line' type with num_traces>5000 triggers downsampling skip path."""
        samples = xp.random.randn(10200).astype(xp.float32)
        fig, ax = plot_eye_diagram(samples, sps=2, type="line", num_symbols=2, show=False)
        assert fig is not None
        plt.close("all")

    def test_eye_diagram_dense_hist(self, backend_device: str, xp: Any) -> None:
        """plot_eye_diagram() 'hist' type with num_traces>20000 triggers downsampling skip path."""
        samples = xp.random.randn(40100).astype(xp.float32)
        fig, ax = plot_eye_diagram(samples, sps=2, type="hist", num_symbols=2, show=False)
        assert fig is not None
        plt.close("all")

    @pytest.mark.parametrize("channels", [1, 2])
    def test_eye_signal_preserves_window_length(
        self, backend_device: str, xp: Any, channels: int
    ) -> None:
        """Eye diagram from Signal object preserves configured trace window length."""
        samples = xp.sin(xp.arange(100) * 0.4)
        if channels == 2:
            samples = xp.stack([samples, samples])
        sig = Signal(samples=samples, sampling_rate=4e6, symbol_rate=1e6)
        fig, axes = plot_eye_diagram(sig, num_symbols=3, type="line", show=False)
        try:
            for ax in np.asarray(axes).flat:
                assert ax.lines
                assert len(ax.lines[0].get_xdata()) == 3 * 4 + 1
        finally:
            plt.close(fig)

    @pytest.mark.parametrize("sps", [0, -1, float("nan"), float("inf")])
    def test_eye_rejects_invalid_sps(
        self, backend_device: str, xp: Any, sps: Any
    ) -> None:
        """plot_eye_diagram rejects non-positive or non-integral SPS."""
        with pytest.raises(ValueError, match="sps to be a positive integer"):
            plot_eye_diagram(xp.ones(100), sps=sps)


class TestPlotConstellation:
    """Tests for constellation diagram plotting and decision overlay."""

    def test_constellation_1d(self, backend_device: str, xp: Any) -> None:
        """Verify basic constellation density plot generation."""
        samples = xp.random.randn(1000) + 1j * xp.random.randn(1000)
        fig, ax = plot_constellation(samples, bins=50, show=False)
        assert fig is not None
        assert ax is not None
        plt.close("all")

    def test_constellation_overlay_ideal(self, backend_device: str, xp: Any) -> None:
        """Verify constellation plot with theoretical overlay enabled."""
        samples = xp.random.randn(1000) + 1j * xp.random.randn(1000)
        fig, ax = plot_constellation(
            samples, bins=50, overlay_ideal=True, modulation="qam", order=16, show=False
        )
        assert fig is not None
        plt.close("all")

    def test_constellation_mimo(self, backend_device: str, xp: Any) -> None:
        """Verify MIMO constellation plotting uses an optimized grid layout."""
        samples = xp.random.randn(4, 1000) + 1j * xp.random.randn(4, 1000)
        fig, axes = plot_constellation(samples, bins=50, show=False)
        assert fig is not None
        assert axes.shape == (2, 2)
        plt.close("all")

    def test_ideal_constellation_basic(self, backend_device: str, xp: Any) -> None:
        """Verify ideal constellation plotting."""
        fig, ax = plot_ideal_constellation("qam", 16, show=False)
        assert fig is not None
        plt.close("all")

        ret = plot_ideal_constellation("invalid", 4, show=False)
        assert ret is None
        plt.close("all")

    def test_constellation_histogram_overlay_error(
        self, backend_device: str, xp: Any
    ) -> None:
        """Verify warning when overlaying ideal on histogram constellation with bad mod."""
        samples = xp.random.randn(100) + 1j * xp.random.randn(100)
        plot_constellation(
            samples, bins=10, overlay_ideal=True, modulation="invalid", order=4, show=False
        )
        plt.close("all")

    def test_constellation_histogram_overlay_warning(
        self, caplog: Any, backend_device: str, xp: Any
    ) -> None:
        """Verify warning when overlaying ideal on histogram."""
        caplog.set_level(logging.WARNING)
        plot_constellation(xp.ones(10) + 1j, bins=10, overlay_ideal=True, modulation=None)
        assert "Modulation and order must be provided" in caplog.text
        plt.close("all")

    def test_constellation_real_samples(
        self, backend_device: str, xp: Any
    ) -> None:
        """Constellation with real (non-complex) samples warns and converts to complex."""
        samples = xp.ones(100, dtype=xp.float32)
        with patch("commkit.plotting.logger.warning"):
            result = plot_constellation(samples, show=False)
        assert result is not None
        plt.close("all")

    def test_constellation_multichannel_single_axis_warning(
        self, backend_device: str, xp: Any, caplog: Any
    ) -> None:
        """Constellation multichannel with single axis warns and overlays all channels."""
        samples = xp.ones((2, 100), dtype=xp.complex64)
        fig0, ax0 = plt.subplots()
        caplog.set_level(logging.WARNING)
        result = plot_constellation(samples, ax=ax0, show=False)
        assert result is not None
        assert "Overlaying plots" in caplog.text
        plt.close("all")

    def test_constellation_multichannel_axes_array(
        self, backend_device: str, xp: Any
    ) -> None:
        """Constellation multichannel with axes array normalizes to 2D layout."""
        samples = xp.ones((2, 100), dtype=xp.complex64)
        fig0, axes0 = plt.subplots(1, 2)
        result = plot_constellation(samples, ax=axes0, show=False)
        assert result is not None
        plt.close("all")

    def test_constellation_multichannel_show(
        self, backend_device: str, xp: Any
    ) -> None:
        """Constellation multichannel with show=True calls plt.show() and returns None."""
        samples = xp.ones((2, 100), dtype=xp.complex64)
        with patch("matplotlib.pyplot.show"):
            result = plot_constellation(samples, show=True)
        assert result is None
        plt.close("all")

    def test_constellation_siso_show(self, backend_device: str, xp: Any) -> None:
        """Constellation SISO with show=True calls plt.show() and returns None."""
        samples = xp.ones(100, dtype=xp.complex64)
        with patch("matplotlib.pyplot.show"):
            result = plot_constellation(samples, show=True)
        assert result is None
        plt.close("all")

    def test_ideal_constellation_custom_ax(
        self, backend_device: str, xp: Any
    ) -> None:
        """plot_ideal_constellation() with provided ax uses that axis's figure."""
        fig0, ax0 = plt.subplots()
        result = plot_ideal_constellation(modulation="psk", order=4, ax=ax0, show=False)
        assert result is not None
        plt.close("all")

    def test_ideal_constellation_show(self, backend_device: str, xp: Any) -> None:
        """plot_ideal_constellation() with show=True calls plt.show() and returns None."""
        with patch("matplotlib.pyplot.show"):
            result = plot_ideal_constellation(modulation="qam", order=16, show=True)
        assert result is None
        plt.close("all")

    def test_constellation_vmin_vmax(self, backend_device: str, xp: Any) -> None:
        """plot_constellation() with vmin/vmax sets color scale bounds on histogram."""
        samples = (xp.random.randn(500) + 1j * xp.random.randn(500)).astype(xp.complex64)
        result = plot_constellation(samples, vmin=0.0, vmax=1.0, show=False)
        assert result is not None
        plt.close("all")

    @pytest.mark.parametrize("channels", [1, 2])
    def test_constellation_signal_optional_metadata_fallback(
        self, backend_device: str, xp: Any, channels: int
    ) -> None:
        """Signal container constellation plotting falls back cleanly on optional metadata."""
        samples = xp.asarray([0.0, 1.0] * 50, dtype=xp.complex64)
        if channels == 2:
            samples = xp.stack([samples, samples])
        sig = Signal(samples=samples, sampling_rate=1e6, symbol_rate=1e6)
        fig, axes = plot_constellation(
            sig, modulation="PAM", order=2, unipolar=True, overlay_ideal=True
        )
        try:
            for ax in np.asarray(axes).flat:
                assert len(ax.collections) == 1
                points = ax.collections[0].get_offsets()
                assert points.shape == (2, 2)
                assert np.min(points[:, 0]) == 0.0
                assert np.max(points[:, 0]) > 0.0
        finally:
            plt.close(fig)


class TestPlotSpectralAndPSD:
    """Tests for Power Spectral Density (PSD) and spectral plotting."""

    def test_psd(self, backend_device: str, xp: Any) -> None:
        """Verify Power Spectral Density (PSD) plotting."""
        samples = xp.random.randn(256)
        fig, ax = plot_psd(samples, sampling_rate=10.0, show=False)
        assert fig is not None
        plt.close("all")

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
        plt.close("all")

    def test_psd_mimo_grid(self, backend_device: str, xp: Any) -> None:
        """Verify MIMO PSD plotting uses an optimized grid layout."""
        samples = xp.random.randn(4, 256)
        fig, axes = plot_psd(samples, sampling_rate=10.0, show=False)
        assert fig is not None
        assert axes.shape == (2, 2)
        plt.close("all")

    def test_psd_wavelength(self, backend_device: str, xp: Any) -> None:
        """Verify PSD plotting with wavelength axis."""
        samples = xp.random.randn(256)
        fig, ax = plot_psd(
            samples, sampling_rate=1e15, x_axis="wavelength", domain="OPT", show=False
        )
        assert fig is not None
        assert "Wavelength" in ax.get_xlabel()
        plt.close("all")

    def test_psd_auto_scale(self, backend_device: str, xp: Any) -> None:
        """Verify PSD frequency axis uses an SI-prefixed engineering tick formatter."""
        samples = xp.random.randn(256)
        fig, ax = plot_psd(samples, sampling_rate=2e12, show=False)
        assert ax.get_xlabel() == "Frequency [Hz]"
        fmt = ax.xaxis.get_major_formatter()
        assert isinstance(fmt, mticker.EngFormatter)
        assert "T" in fmt(2e12)
        plt.close("all")

        fig, ax = plot_psd(samples, sampling_rate=2e6, show=False)
        assert "M" in ax.xaxis.get_major_formatter()(2e6)
        plt.close("all")

    def test_multichannel_overlay(self, backend_device: str, xp: Any) -> None:
        """Verify overlaying multichannel signals on a single axis."""
        samples = xp.random.randn(2, 256)
        fig, ax = plt.subplots()
        plot_psd(samples, sampling_rate=1.0, ax=ax, show=False)
        plot_time_domain(samples, sampling_rate=1.0, ax=ax, show=False)
        plt.close("all")

    def test_psd_axis_overlay_warning(
        self, caplog: Any, backend_device: str, xp: Any
    ) -> None:
        """Verify warning when single axis is provided for multichannel PSD."""
        sig = xp.ones((2, 256))
        fig, ax = plt.subplots()
        with patch("matplotlib.pyplot.show"):
            plot_psd(sig, ax=ax)
        assert "Multiple channels detected but single axis provided" in caplog.text
        plt.close("all")

    def test_psd_wavelength_warning(
        self, caplog: Any, backend_device: str, xp: Any
    ) -> None:
        """Verify wavelength warning."""
        sig = xp.ones(256)
        with patch("matplotlib.pyplot.show"):
            plot_psd(sig, x_axis="wavelength", domain="RF")
        assert "Wavelength plotting is typically used for optical signals" in caplog.text
        plt.close("all")

    def test_psd_auto_scale_small(self, backend_device: str, xp: Any) -> None:
        """Test auto-scaling for small frequencies (Hz)."""
        sig = xp.ones(256)
        fig, ax = plot_psd(sig, sampling_rate=1.0)
        assert "Hz" in ax.get_xlabel()
        plt.close("all")

    def test_psd_ghz_scaling(self, backend_device: str, xp: Any) -> None:
        """PSD with GHz-range sampling rate triggers GHz scale factor."""
        samples = xp.random.randn(256).astype(xp.float32)
        fig, ax = plot_psd(samples, sampling_rate=5e9, show=False)
        assert fig is not None
        plt.close("all")

    def test_psd_khz_scaling(self, backend_device: str, xp: Any) -> None:
        """PSD with kHz-range sampling rate triggers kHz scale factor."""
        samples = xp.random.randn(256).astype(xp.float32)
        fig, ax = plot_psd(samples, sampling_rate=5e3, show=False)
        assert fig is not None
        plt.close("all")

    def test_psd_hz_scaling(self, backend_device: str, xp: Any) -> None:
        """PSD with Hz-range sampling rate uses default Hz unit."""
        samples = xp.random.randn(256).astype(xp.float32)
        fig, ax = plot_psd(samples, sampling_rate=100.0, show=False)
        assert fig is not None
        plt.close("all")

    def test_psd_xlim_ylim(self, backend_device: str, xp: Any) -> None:
        """PSD with xlim and ylim parameters applies axis limits."""
        samples = xp.random.randn(256).astype(xp.float32)
        fig, ax = plot_psd(
            samples, sampling_rate=1e6, xlim=(-0.4, 0.4), ylim=(-80, 0), show=False
        )
        assert fig is not None
        plt.close("all")

    def test_psd_show(self, backend_device: str, xp: Any) -> None:
        """PSD with show=True calls plt.show() and returns None."""
        samples = xp.random.randn(256).astype(xp.float32)
        with patch("matplotlib.pyplot.show"):
            result = plot_psd(samples, sampling_rate=1e6, show=True)
        assert result is None
        plt.close("all")

    def test_psd_multichannel_show(self, backend_device: str, xp: Any) -> None:
        """PSD multichannel with show=True calls plt.show() and returns None."""
        samples = xp.random.randn(2, 256).astype(xp.float32)
        with patch("matplotlib.pyplot.show"):
            result = plot_psd(samples, sampling_rate=1e6, show=True)
        assert result is None
        plt.close("all")

    def test_psd_multichannel_single_axis_warning(
        self, backend_device: str, xp: Any, caplog: Any
    ) -> None:
        """PSD multichannel with single axis warns and overlays all channels on it."""
        samples = xp.random.randn(2, 256).astype(xp.float32)
        fig0, ax0 = plt.subplots()
        caplog.set_level(logging.WARNING)
        fig, axes = plot_psd(samples, sampling_rate=1e6, ax=ax0, show=False)
        assert "Overlaying plots" in caplog.text
        plt.close("all")

    def test_psd_multichannel_axes_array(
        self, backend_device: str, xp: Any
    ) -> None:
        """PSD multichannel with axes ndarray normalizes axes to 2D layout."""
        samples = xp.random.randn(2, 256).astype(xp.float32)
        fig0, axes0 = plt.subplots(1, 2)
        fig, axes = plot_psd(samples, sampling_rate=1e6, ax=axes0, show=False)
        assert fig is not None
        plt.close("all")


class TestPlotTimeDomainAndSpectrogram:
    """Tests for time-domain waveform and spectrogram rendering."""

    def test_time_domain(self, backend_device: str, xp: Any) -> None:
        """Verify time-domain waveform plotting."""
        samples = xp.arange(100)
        fig, ax = plot_time_domain(samples, sampling_rate=10.0, show=False)
        assert fig is not None
        plt.close("all")

    def test_time_domain_mimo_grid(self, backend_device: str, xp: Any) -> None:
        """Verify MIMO time-domain plotting uses an optimized grid layout."""
        samples = xp.random.randn(4, 100)
        fig, axes = plot_time_domain(samples, sampling_rate=10.0, show=False)
        assert fig is not None
        assert axes.shape == (2, 2)
        plt.close("all")

    @pytest.mark.parametrize("plot_func", [plot_psd, plot_time_domain])
    def test_multichannel_plots(
        self, backend_device: str, xp: Any, plot_func: Any
    ) -> None:
        """Verify multichannel plotting for PSD and Time-Domain."""
        samples = xp.random.randn(2, 256)
        fig, axes = plot_func(samples, sampling_rate=1.0, show=False)
        assert fig is not None
        assert axes.size == 2
        plt.close("all")

    def test_time_domain_limits(
        self, caplog: Any, backend_device: str, xp: Any
    ) -> None:
        """Verify symbol limit warnings in time_domain."""
        sig = xp.ones(100)
        with patch("matplotlib.pyplot.show"):
            plot_time_domain(sig, num_symbols=200, sps=1.0)
        assert "Limit exceeds number of symbols" in caplog.text
        plt.close("all")

    def test_time_domain_auto_scale(self, backend_device: str, xp: Any) -> None:
        """Verify time axis uses an SI-prefixed engineering tick formatter."""
        sig = xp.ones(100)
        _, ax = plot_time_domain(sig, sampling_rate=1e10)
        assert ax.get_xlabel() == "Time [s]"
        fmt = ax.xaxis.get_major_formatter()
        assert isinstance(fmt, mticker.EngFormatter)
        assert "n" in fmt(3e-9)
        assert "p" in fmt(3e-12)
        assert "µ" in fmt(3e-6)
        assert "m" in fmt(3e-3)
        plt.close("all")

    def test_time_domain_multichannel_axes_array(
        self, backend_device: str, xp: Any
    ) -> None:
        """plot_time_domain() multichannel with axes array normalizes axes to 2D layout."""
        samples = xp.random.randn(2, 1000).astype(xp.float32)
        fig0, axes0 = plt.subplots(1, 2)
        result = plot_time_domain(samples, sampling_rate=1e6, ax=axes0, show=False)
        assert result is not None
        plt.close("all")

    def test_time_domain_multichannel_show(
        self, backend_device: str, xp: Any
    ) -> None:
        """plot_time_domain() multichannel with show=True calls plt.show() and returns None."""
        samples = xp.random.randn(2, 1000).astype(xp.float32)
        with patch("matplotlib.pyplot.show"):
            result = plot_time_domain(samples, sampling_rate=1e6, show=True)
        assert result is None
        plt.close("all")

    def test_time_domain_siso_show(self, backend_device: str, xp: Any) -> None:
        """plot_time_domain() 1D with show=True calls plt.show() and returns None."""
        samples = xp.random.randn(500).astype(xp.float32)
        with patch("matplotlib.pyplot.show"):
            result = plot_time_domain(samples, sampling_rate=1e6, show=True)
        assert result is None
        plt.close("all")

    def test_spectrogram_plot_siso(self, backend_device: str, xp: Any) -> None:
        """Verify spectrogram plotting for SISO (1D) signals."""
        fs = 100.0
        t = xp.arange(1000) / fs
        samples = xp.sin(2 * xp.pi * 20.0 * t)

        fig, ax = plot_spectrogram(samples, sampling_rate=fs, show=False)
        assert fig is not None
        assert ax is not None
        plt.close("all")

        fig, ax = plot_spectrogram(
            samples,
            sampling_rate=fs,
            xlim=(-10.0, 30.0),
            ylim=(1.0, 5.0),
            show=False,
        )
        assert fig is not None
        assert ax is not None
        plt.close("all")

        with patch("matplotlib.pyplot.show"):
            res = plot_spectrogram(samples, sampling_rate=fs, show=True)
        assert res is None
        plt.close("all")

    def test_spectrogram_plot_mimo(self, backend_device: str, xp: Any) -> None:
        """Verify spectrogram plotting for MIMO (2D) signals generates subplots."""
        fs = 100.0
        t = xp.arange(1000) / fs
        samples = xp.stack([xp.sin(2 * xp.pi * 20.0 * t), xp.cos(2 * xp.pi * 10.0 * t)])

        fig, axes = plot_spectrogram(samples, sampling_rate=fs, show=False)
        assert fig is not None
        assert isinstance(axes, np.ndarray)
        assert axes.size == 2
        plt.close("all")

    def test_signal_spectrogram_convenience(
        self, backend_device: str, xp: Any
    ) -> None:
        """Verify core.Signal.spectrogram and plot_spectrogram convenience methods."""
        fs = 100.0
        sig = generate_psk(
            symbol_rate=10.0,
            num_symbols=100,
            order=4,
            sps=int(fs / 10.0),
            seed=42,
        )
        sig.samples = xp.asarray(sig.samples)

        f, t, Sxx = spectral.spectrogram(sig, nperseg=64, noverlap=32)
        assert isinstance(f, xp.ndarray)
        assert isinstance(t, xp.ndarray)
        assert isinstance(Sxx, xp.ndarray)
        assert len(f) == 64
        assert Sxx.shape[-1] == len(t)

        fig, ax = plotting.plot_spectrogram(sig, nperseg=64, show=False)
        assert fig is not None
        assert ax is not None
        plt.close("all")


class TestPlotFilterAndEqualizer:
    """Tests for filter response and equalizer convergence plotting."""

    def test_filter_response(self, backend_device: str, xp: Any) -> None:
        """Verify filter response plotting (Impulse, Mag, Phase, Group Delay)."""
        taps = xp.array([1, 0.5, 0.25])
        fig, axes = plot_filter_response(taps, sps=1.0, show=False)
        assert fig is not None
        assert len(axes) == 4
        plt.close("all")

    def test_filter_response_ba(self, backend_device: str, xp: Any) -> None:
        """Verify filter response plotting accepts a general (b, a) IIR system."""
        b = xp.array([0.1, 0.2, 0.1])
        a = xp.array([1.0, -0.5, 0.1])
        fig, axes = plot_filter_response((b, a), sps=1.0, show=False)
        assert fig is not None
        assert len(axes) == 4
        plt.close("all")

    def test_filter_response_sos(self, backend_device: str, xp: Any) -> None:
        """Verify filter response plotting accepts SOS input, with sampling_rate."""
        import scipy.signal

        sos = xp.asarray(scipy.signal.butter(4, 0.1, btype="low", output="sos"))
        fig, axes = plot_filter_response(sos, sampling_rate=1e9, n_impulse=200, show=False)
        assert fig is not None
        assert len(axes) == 4
        plt.close("all")

    def test_filter_response_axis_error(
        self, backend_device: str, xp: Any
    ) -> None:
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
        plt.close("all")

    def test_filter_response_no_sps(self, backend_device: str, xp: Any) -> None:
        """Test filter_response without sps."""
        plot_filter_response(xp.ones(10))
        plt.close("all")

    def test_filter_response_complex_taps(
        self, backend_device: str, xp: Any
    ) -> None:
        """plot_filter_response() with complex taps plots I/Q components separately."""
        taps = filtering.rrc_taps(sps=4, span=4, rolloff=0.35)
        complex_taps = taps.astype(complex)
        result = plot_filter_response(complex_taps, sps=4, show=False)
        assert result is not None
        plt.close("all")

    def test_filter_response_show(self, backend_device: str, xp: Any) -> None:
        """plot_filter_response() with show=True calls plt.show() and returns None."""
        taps = filtering.rrc_taps(sps=4, span=4, rolloff=0.35)
        with patch("matplotlib.pyplot.show"):
            result = plot_filter_response(taps, sps=4, show=True)
        assert result is None
        plt.close("all")

    def test_equalizer_result_mimo_weights(
        self, backend_device: str, xp: Any
    ) -> None:
        """equalizer_result with MIMO error/weights plots per-channel error and weights."""
        from commkit import equalization
        from commkit.plotting import plot_equalizer_result

        n_symbols = 400
        sig = generate_psk(
            symbol_rate=1e6,
            num_symbols=n_symbols,
            order=4,
            pulse_shape="rrc",
            sps=2,
            num_streams=2,
            seed=0,
        )
        rx_mimo = xp.asarray(sig.samples)
        train_mimo = xp.asarray(sig.source_symbols)

        result = equalization.lms(
            rx_mimo,
            training_symbols=train_mimo,
            num_taps=7,
            step_size=0.05,
            modulation="psk",
            order=4,
            backend="numba",
        )

        fig, axes = plot_equalizer_result(result, smoothing=10)
        assert fig is not None
        assert len(axes) == 2
        plt.close("all")

    def test_equalizer_result_custom_axes(
        self, backend_device: str, xp: Any
    ) -> None:
        """equalizer_result with pre-existing axes uses them rather than creating new figures."""
        from commkit import equalization
        from commkit.plotting import plot_equalizer_result

        sig = generate_psk(
            symbol_rate=1e6, num_symbols=200, order=4, pulse_shape="rrc", sps=2, seed=0
        )
        result = equalization.lms(
            xp.asarray(sig.samples),
            training_symbols=xp.asarray(sig.source_symbols),
            num_taps=7,
            step_size=0.05,
            modulation="psk",
            order=4,
            backend="numba",
        )

        fig0, axes0 = plt.subplots(1, 2)
        fig_ret, axes_ret = plot_equalizer_result(result, ax=axes0)
        assert fig_ret is not None
        plt.close("all")

    def test_equalizer_result_show(self, backend_device: str, xp: Any) -> None:
        """equalizer_result with show=True calls plt.show() and returns None."""
        from commkit import equalization
        from commkit.plotting import plot_equalizer_result

        sig = generate_psk(
            symbol_rate=1e6, num_symbols=200, order=4, pulse_shape="rrc", sps=2, seed=0
        )
        result = equalization.lms(
            xp.asarray(sig.samples),
            training_symbols=xp.asarray(sig.source_symbols),
            num_taps=7,
            step_size=0.05,
            modulation="psk",
            order=4,
            backend="numba",
        )

        with patch("matplotlib.pyplot.show"):
            ret = plot_equalizer_result(result, show=True)
        assert ret is None
        plt.close("all")

    def test_equalizer_result_short_smoothing_siso(
        self, backend_device: str, xp: Any
    ) -> None:
        """plot_equalizer_result() SISO where len(mse) <= smoothing uses raw mse."""
        from commkit import equalization
        from commkit.plotting import plot_equalizer_result

        sig = generate_psk(
            symbol_rate=1e6, num_symbols=50, order=4, pulse_shape="rrc", sps=2, seed=0
        )
        result = equalization.lms(
            xp.asarray(sig.samples),
            training_symbols=xp.asarray(sig.source_symbols),
            num_taps=5,
            step_size=0.05,
            modulation="psk",
            order=4,
            backend="numba",
        )
        fig, axes = plot_equalizer_result(result, smoothing=1000)
        assert fig is not None
        plt.close("all")

    def test_equalizer_result_short_smoothing_mimo(
        self, backend_device: str, xp: Any
    ) -> None:
        """plot_equalizer_result() MIMO where len(mse) <= smoothing uses raw mse."""
        from commkit import equalization
        from commkit.plotting import plot_equalizer_result

        sig = generate_psk(
            symbol_rate=1e6,
            num_symbols=50,
            order=4,
            pulse_shape="rrc",
            sps=2,
            num_streams=2,
            seed=0,
        )
        result = equalization.lms(
            xp.asarray(sig.samples),
            training_symbols=xp.asarray(sig.source_symbols),
            num_taps=5,
            step_size=0.05,
            modulation="psk",
            order=4,
            backend="numba",
        )
        fig, axes = plot_equalizer_result(result, smoothing=1000)
        assert fig is not None
        plt.close("all")


class TestPlotThemeAndGridUtilities:
    """Tests for default theme configuration and subplot grid computations."""

    def test_apply_theme(self) -> None:
        """Verify applying the visual theme."""
        apply_default_theme()

    @patch("matplotlib.font_manager.findfont")
    def test_apply_theme_fallback(self, mock_find: Any) -> None:
        """Trigger the font fallback in apply_default_theme."""
        mock_find.side_effect = ValueError("Font not found")
        apply_default_theme()

    def test_subplot_grid(self, backend_device: str, xp: Any) -> None:
        """Verify subplot grid calculation."""
        assert _create_subplot_grid(1) == (1, 1)
        assert _create_subplot_grid(2) == (1, 2)
        assert _create_subplot_grid(3) == (2, 2)
        assert _create_subplot_grid(5) == (3, 2)
        plt.close("all")
