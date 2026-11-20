"""Tests for Power Spectral Density (PSD) and spectral/spectrogram plotting."""

import logging
from typing import Any
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest

from commkit import generate, plotting, spectral
from commkit.filtering import RRC
from commkit.mapping import Constellation
from commkit.plotting import plot_psd, plot_spectrogram
from tests.common.conversions import to_numpy


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

    def test_psd_domain_from_the_signal(self, caplog: Any, xp: Any) -> None:
        """A Signal's domain and carrier place the axis; a conflict raises."""
        from commkit import Signal

        sig = Signal(
            samples=xp.random.randn(256), sampling_rate=1e12, symbol_rate=1e11,
            domain="OPT", center_frequency=193.1e12,
        )  # fmt: skip
        fig, ax = plot_psd(sig, x_axis="wavelength", show=False)
        assert "Wavelength" in ax.get_xlabel()
        assert "optical" not in caplog.text
        lo, hi = ax.get_lines()[0].get_xdata()[[0, -1]]
        assert 1548 < min(lo, hi) < max(lo, hi) < 1557
        with pytest.raises(ValueError, match="domain"):
            plot_psd(sig, domain="RF", show=False)
        with pytest.raises(ValueError, match="domain='RF' or 'OPT'"):
            plot_psd(xp.ones(256), sampling_rate=1.0, center_frequency=1e9,
                     domain="DIG", show=False)  # fmt: skip
        with pytest.raises(ValueError, match="domain must be"):
            plot_psd(xp.ones(256), sampling_rate=1.0, domain="IF", show=False)

    def test_psd_auto_scale(self, xp: Any) -> None:
        """The frequency axis carries one SI prefix, in the label's unit."""
        samples = xp.random.randn(256)
        fig, ax = plot_psd(samples, sampling_rate=2e12, show=False)
        fig.canvas.draw()
        assert ax.get_xlabel() == "Frequency [THz]"
        assert "0.4" in [
            t.get_text() for t in ax.get_xticklabels()
        ]  # one-sided: 0-1 THz

        fig, ax = plot_psd(samples, sampling_rate=2e6, show=False)
        fig.canvas.draw()
        assert ax.get_xlabel() == "Frequency [MHz]"

    def test_psd_axis_overlay_warning(self, caplog: Any, xp: Any) -> None:
        """Verify warning when single axis is provided for multichannel PSD."""
        sig = xp.ones((2, 256))
        fig, ax = plt.subplots()
        with patch("matplotlib.pyplot.show"):
            plot_psd(sig, sampling_rate=1.0, ax=ax)
        assert "Multiple channels detected but single axis provided" in caplog.text

    def test_psd_wavelength_warning(self, caplog: Any, xp: Any) -> None:
        """Verify wavelength warning."""
        sig = xp.ones(256)
        with patch("matplotlib.pyplot.show"):
            plot_psd(sig, sampling_rate=1.0, x_axis="wavelength", domain="RF")
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
            num_symbols=100,
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


class TestPlotFacts:
    """sampling_rate and sps are properties: from a Signal, required for arrays."""

    def test_conflicting_sampling_rate_raises(self, xp: Any) -> None:
        from commkit.core import Signal

        sig = Signal(samples=xp.ones(256), sampling_rate=2.0, symbol_rate=1.0)
        with pytest.raises(ValueError, match="conflicts"):
            plot_psd(sig, sampling_rate=4.0)

    def test_array_without_sampling_rate_raises(self, xp: Any) -> None:
        with pytest.raises(ValueError, match="requires sampling_rate"):
            plot_psd(xp.ones(256))

    def test_eye_array_without_sps_raises(self, xp: Any) -> None:
        from commkit.plotting import plot_eye_diagram

        with pytest.raises(ValueError, match="requires sps"):
            plot_eye_diagram(xp.ones(256))


class TestSpectrogramImage:
    """The spectrogram is an image whose cells sit on the STFT grid."""

    def test_image_values_and_extent(self, xp: Any) -> None:
        from matplotlib.image import AxesImage

        rng = np.random.default_rng(0)
        x = xp.asarray(
            (rng.standard_normal(8192) + 1j * rng.standard_normal(8192)).astype(
                np.complex64
            )
        )
        fs = 1e6
        _, ax = plot_spectrogram(x, sampling_rate=fs, nperseg=128)
        (im,) = [a for a in ax.get_images() if isinstance(a, AxesImage)]
        spec = spectral.spectrogram(x, sampling_rate=fs, nperseg=128)
        f = to_numpy(spec.frequencies)
        t = to_numpy(spec.times)
        expected = 10 * np.log10(to_numpy(spec.values) + 1e-20).T
        np.testing.assert_allclose(im.get_array(), expected, rtol=1e-6)
        df, dt = f[1] - f[0], t[1] - t[0]
        np.testing.assert_allclose(
            im.get_extent(),
            [f[0] - df / 2, f[-1] + df / 2, t[0] - dt / 2, t[-1] + dt / 2],
        )

    def test_crop_happens_before_drawing(self, xp: Any) -> None:
        x = xp.asarray(np.random.default_rng(1).standard_normal(8192))
        _, ax = plot_spectrogram(
            x, sampling_rate=1e6, nperseg=128, xlim=(1e5, 2e5), ylim=(1e-3, 4e-3)
        )
        lo, hi, t0, t1 = ax.get_images()[0].get_extent()
        assert 0.99e5 - 1e6 / 128 <= lo and hi <= 2e5 + 1e6 / 128
        assert t0 >= 1e-3 - 1e-3 and t1 <= 4e-3 + 1e-3


def test_psd_comparison_legend(xp: Any) -> None:
    """A PSD overlay can show a legend without Matplotlib calls."""
    x = xp.ones(256, dtype=xp.float32)
    fig, ax = plot_psd(x, sampling_rate=1, label="before")
    _, ax = plot_psd(x, sampling_rate=1, ax=ax, label="after", legend=True)
    assert [text.get_text() for text in ax.get_legend().get_texts()] == [
        "before",
        "after",
    ]


# -----------------------------------------------------------------------------
# relative, annotations, dynamic range
# -----------------------------------------------------------------------------


def _rrc_signal(backend_device: str, num_symbols: int = 2**15) -> Any:
    return generate(
        Constellation.qam(16),
        num_symbols=num_symbols,
        symbol_rate=1e9,
        sps=8,
        pulse=RRC(0.2),
        rng=1,
        device=backend_device,
    )


def test_psd_relative_compares_shapes(backend_device: str) -> None:
    """A signal 20 dB stronger draws the same relative PSD, peaking at 0 dB."""
    sig = _rrc_signal(backend_device, 2**12)
    loud = sig.replace(samples=10 * sig.samples)
    _, ax = plot_psd(sig, relative=True)
    plot_psd(loud, relative=True, ax=ax)
    a, b = (line.get_ydata() for line in ax.lines)
    band = a > -50  # deeper, single-precision rounding of the samples shows
    np.testing.assert_allclose(a[band], b[band], atol=1e-3)
    assert np.max(a) == pytest.approx(0.0)
    assert ax.get_ylabel() == "Relative PSD [dB]"
    _, raw = plot_psd(sig)
    plot_psd(loud, ax=raw)
    a, b = (line.get_ydata() for line in raw.lines)
    np.testing.assert_allclose((b - a)[band], 20.0, atol=1e-3)
    plt.close("all")


def test_psd_3db_guide_reads_the_rrc_bandwidth(backend_device: str) -> None:
    """The -3 dB line crosses an RRC spectrum at half the symbol rate.

    ``|H(Rs/2)|² = 1/2`` for any roll-off.  The guide hangs from the noisy
    Welch peak: about 250 averaged segments leave each bin a relative
    standard deviation near 6 %, so the largest of some 400 in-band bins sits
    about 0.8 dB above the mean level, and the line with it.  At the
    roll-off's slope there (68 dB per symbol rate for β = 0.2), that moves
    the crossing inward by about 0.01 Rs; the tolerance is 0.03 Rs.
    """
    _, ax = plot_psd(_rrc_signal(backend_device), nperseg=1024, relative=True,
                     annotations=True)  # fmt: skip
    (guide,) = [line for line in ax.lines if line.get_gid() == "commkit-3db-guide"]
    level = guide.get_ydata()[0]
    assert level == pytest.approx(-3.0)
    f, y = ax.lines[0].get_xdata(), ax.lines[0].get_ydata()
    upper = f > 0
    above = np.nonzero(upper & (y >= level))[0]
    edge = f[above[-1]]
    assert edge / 1e9 == pytest.approx(0.5, abs=0.03)
    plt.close("all")


def test_psd_guides_shared_when_relative(backend_device: str) -> None:
    sig = _rrc_signal(backend_device, 2**12)
    _, ax = plot_psd(sig, relative=True, annotations=True, label="a")
    plot_psd(sig, relative=True, annotations=True, ax=ax, label="b", legend=True)
    guides = [line for line in ax.lines if line.get_gid() == "commkit-3db-guide"]
    assert len(guides) == 1
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["a", "b"]
    _, ax = plot_psd(sig, annotations=True)
    plot_psd(sig.replace(samples=2 * sig.samples), annotations=True, ax=ax)
    guides = [line for line in ax.lines if line.get_gid() == "commkit-3db-guide"]
    assert len(guides) == 2
    assert guides[1].get_ydata()[0] - guides[0].get_ydata()[0] == pytest.approx(
        20 * np.log10(2), abs=1e-3
    )
    plt.close("all")


def test_spectrogram_relative_and_dynamic_range(backend_device: str) -> None:
    """Relative levels peak at 0 dB; the dynamic range sets the color
    limits to (peak - range, peak); explicit vmin wins."""
    sig = _rrc_signal(backend_device, 2**12)
    _, ax = plot_spectrogram(sig)
    raw = ax.images[0].get_array()
    _, ax = plot_spectrogram(sig, relative=True, dynamic_range_db=40)
    image = ax.images[0]
    np.testing.assert_allclose(image.get_array(), raw - raw.max(), atol=1e-4)
    assert image.get_clim() == (-40.0, 0.0)
    assert ax.figure.axes[-1].get_ylabel() == "Relative Level [dB]"
    _, ax = plot_spectrogram(sig, dynamic_range_db=30)
    assert ax.images[0].get_clim() == pytest.approx((raw.max() - 30, raw.max()))
    _, ax = plot_spectrogram(sig, dynamic_range_db=30, vmin=-500)
    assert ax.images[0].get_clim()[0] == -500
    for bad in ({"mode": "angle", "relative": True}, {"dynamic_range_db": 0}):
        with pytest.raises(ValueError):
            plot_spectrogram(sig, **bad)
    plt.close("all")
