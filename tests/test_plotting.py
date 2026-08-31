"""Tests for signal visualization and plotting tools."""

import logging
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pytest

from commkit import filtering, generate_psk, plotting, spectral
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


@pytest.mark.parametrize("type", ["line", "hist"])
def test_eye_diagram_real(backend_device, xp, type):
    """Verify eye diagram generation for real-valued signals."""
    # Random binary data
    samples = xp.random.randn(1000)
    sps = 4

    # Run plot
    try:
        fig, ax = plot_eye_diagram(samples, sps=sps, type=type, show=False)
        assert fig is not None
        assert ax is not None
        plt.close("all")
    except ImportError:
        if backend_device == "gpu":
            pytest.skip("Skipping GPU plot test if import failure")
        raise


def test_eye_diagram_complex(backend_device, xp):
    """Verify eye diagram generation for complex-valued signals (I/Q)."""
    samples = xp.random.randn(1000) + 1j * xp.random.randn(1000)
    sps = 4

    # Default (no ax provided) -> should create 2 subplots
    fig, ax = plot_eye_diagram(samples, sps=sps, show=False)
    assert fig is not None
    assert isinstance(ax, (list, tuple, np.ndarray))
    assert len(ax) == 2
    plt.close("all")

    # With provided ax
    fig, axes = plt.subplots(2, 1)
    fig, ax_ret = plot_eye_diagram(samples, sps=sps, ax=axes, show=False)
    assert ax_ret is axes
    plt.close("all")


def test_time_domain(backend_device, xp):
    """Verify time-domain waveform plotting."""
    samples = xp.arange(100)
    fig, ax = plot_time_domain(samples, sampling_rate=10.0, show=False)
    assert fig is not None
    plt.close("all")


def test_psd(backend_device, xp):
    """Verify Power Spectral Density (PSD) plotting."""
    samples = xp.random.randn(256)
    fig, ax = plot_psd(samples, sampling_rate=10.0, show=False)
    assert fig is not None
    plt.close("all")

    # Test custom parameters (window, noverlap, nfft, scaling)
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


def test_filter_response(backend_device, xp):
    """Verify filter response plotting (Impulse, Mag, Phase)."""
    taps = xp.array([1, 0.5, 0.25])
    fig, axes = plot_filter_response(taps, sps=1.0, show=False)
    assert fig is not None
    assert len(axes) == 3
    plt.close("all")


def test_constellation_1d(backend_device, xp):
    """Verify basic constellation density plot generation."""
    samples = xp.random.randn(1000) + 1j * xp.random.randn(1000)
    fig, ax = plot_constellation(samples, bins=50, show=False)
    assert fig is not None
    assert ax is not None
    plt.close("all")


def test_constellation_overlay_ideal(backend_device, xp):
    """Verify constellation plot with theoretical overlay enabled."""
    samples = xp.random.randn(1000) + 1j * xp.random.randn(1000)
    fig, ax = plot_constellation(
        samples, bins=50, overlay_ideal=True, modulation="qam", order=16, show=False
    )
    assert fig is not None
    plt.close("all")


def test_constellation_mimo(backend_device, xp):
    """Verify MIMO constellation plotting uses an optimized grid layout."""
    # 4 channels -> should be 2x2 grid
    samples = xp.random.randn(4, 1000) + 1j * xp.random.randn(4, 1000)
    fig, axes = plot_constellation(samples, bins=50, show=False)
    assert fig is not None
    assert axes.shape == (2, 2)  # 2x2 grid
    plt.close("all")


def test_psd_mimo_grid(backend_device, xp):
    """Verify MIMO PSD plotting uses an optimized grid layout."""
    # 4 channels -> should be 2x2 grid
    samples = xp.random.randn(4, 256)
    fig, axes = plot_psd(samples, sampling_rate=10.0, show=False)
    assert fig is not None
    assert axes.shape == (2, 2)  # 2x2 grid, not (1, 4)
    plt.close("all")


def test_time_domain_mimo_grid(backend_device, xp):
    """Verify MIMO time-domain plotting uses an optimized grid layout."""
    # 4 channels -> should be 2x2 grid
    samples = xp.random.randn(4, 100)
    fig, axes = plot_time_domain(samples, sampling_rate=10.0, show=False)
    assert fig is not None
    assert axes.shape == (2, 2)  # 2x2 grid, not (1, 4)
    plt.close("all")


@pytest.mark.parametrize("plot_func", [plot_psd, plot_time_domain])
def test_multichannel_plots(backend_device, xp, plot_func):
    """Verify multichannel plotting for PSD and Time-Domain."""
    samples = xp.random.randn(2, 256)
    fig, axes = plot_func(samples, sampling_rate=1.0, show=False)
    assert fig is not None
    # For 2 channels, it should return 2 axes
    assert axes.size == 2
    plt.close("all")


def test_eye_diagram_multichannel(backend_device, xp):
    """Verify eye diagram for multichannel complex signals."""
    samples = xp.random.randn(2, 100) + 1j * xp.random.randn(2, 100)
    # 2 channels, complex -> 2x2 grid = 4 subplots
    fig, axes = plot_eye_diagram(samples, sps=4, show=False)
    assert fig is not None
    assert axes.shape == (2, 2)
    plt.close("all")


def test_eye_diagram_axis_error(backend_device, xp):
    """Verify error when too few axes are provided for multichannel eye diagram."""
    samples = xp.zeros((2, 100))
    fig, ax = plt.subplots(1)
    with pytest.raises(ValueError, match="Not enough axes"):
        plot_eye_diagram(samples, sps=4, ax=[ax], show=False)
    plt.close("all")

    with pytest.raises(ValueError, match="must provide a list of axes"):
        plot_eye_diagram(samples, sps=4, ax=ax, show=False)
    plt.close("all")


def test_apply_theme():
    """Verify applying the visual theme."""
    apply_default_theme()


def test_psd_wavelength(backend_device, xp):
    """Verify PSD plotting with wavelength axis."""
    samples = xp.random.randn(256)
    # Wavelength requires positive frequencies, using real signal gets +/- f
    fig, ax = plot_psd(
        samples, sampling_rate=1e15, x_axis="wavelength", domain="OPT", show=False
    )
    assert fig is not None
    assert "Wavelength" in ax.get_xlabel()
    plt.close("all")


def test_psd_auto_scale(backend_device, xp):
    """Verify PSD auto-scaling for different frequency ranges."""
    samples = xp.random.randn(256)

    # THz
    fig, ax = plot_psd(samples, sampling_rate=2e12, show=False)
    assert "THz" in ax.get_xlabel()
    plt.close("all")

    # MHz
    fig, ax = plot_psd(samples, sampling_rate=2e6, show=False)
    assert "MHz" in ax.get_xlabel()
    plt.close("all")


def test_multichannel_overlay(backend_device, xp):
    """Verify overlaying multichannel signals on a single axis."""
    samples = xp.random.randn(2, 256)
    fig, ax = plt.subplots()

    # PSD overlay
    plot_psd(samples, sampling_rate=1.0, ax=ax, show=False)

    # Time domain overlay
    plot_time_domain(samples, sampling_rate=1.0, ax=ax, show=False)
    plt.close("all")


def test_filter_response_axis_error(backend_device, xp):
    """Verify behavior when wrong number of axes are provided for filter response."""
    taps = xp.array([1, 0, 0, 1])
    fig, ax = plt.subplots(1)
    # Should warn and create new figure
    plot_filter_response(taps, ax=ax, show=False)

    # Should accept list of 3 axes
    fig, axes = plt.subplots(3, 1)
    plot_filter_response(taps, ax=axes, show=False)
    plt.close("all")


def test_ideal_constellation_basic(backend_device, xp):
    """Verify ideal constellation plotting."""
    fig, ax = plot_ideal_constellation("qam", 16, show=False)
    assert fig is not None
    plt.close("all")

    # Error case
    ret = plot_ideal_constellation("invalid", 4, show=False)
    assert ret is None
    plt.close("all")


def test_constellation_histogram_overlay_error(backend_device, xp):
    """Verify warning when overlaying ideal on histogram constellation with bad mod."""
    samples = xp.random.randn(100) + 1j * xp.random.randn(100)
    # bins > 0 triggers histogram mode
    plot_constellation(
        samples, bins=10, overlay_ideal=True, modulation="invalid", order=4, show=False
    )
    plt.close("all")


@patch("matplotlib.font_manager.findfont")
def test_apply_theme_fallback(mock_find):
    """Trigger the font fallback in apply_default_theme."""
    mock_find.side_effect = ValueError("Font not found")
    apply_default_theme()


def test_subplot_grid(backend_device, xp):
    """Verify subplot grid calculation."""
    assert _create_subplot_grid(1) == (1, 1)
    assert _create_subplot_grid(2) == (1, 2)
    assert _create_subplot_grid(3) == (2, 2)
    assert _create_subplot_grid(5) == (3, 2)
    plt.close("all")


def test_psd_axis_overlay_warning(caplog, backend_device, xp):
    """Verify warning when single axis is provided for multichannel PSD."""
    sig = xp.ones((2, 256))
    fig, ax = plt.subplots()
    with patch("matplotlib.pyplot.show"):
        plot_psd(sig, ax=ax)
    assert "Multiple channels detected but single axis provided" in caplog.text
    plt.close("all")


def test_psd_wavelength_warning(caplog, backend_device, xp):
    """Verify wavelength warning."""
    sig = xp.ones(256)
    with patch("matplotlib.pyplot.show"):
        plot_psd(sig, x_axis="wavelength", domain="RF")
    assert "Wavelength plotting is typically used for optical signals" in caplog.text
    plt.close("all")


def test_psd_auto_scale_small(backend_device, xp):
    """Test auto-scaling for small frequencies (Hz)."""
    sig = xp.ones(256)
    fig, ax = plot_psd(sig, sampling_rate=1.0)
    assert "Hz" in ax.get_xlabel()
    plt.close("all")


def test_time_domain_limits(caplog, backend_device, xp):
    """Verify symbol limit warnings in time_domain."""
    sig = xp.ones(100)
    # request 200 symbols when only 100 samples available (sps=1)
    with patch("matplotlib.pyplot.show"):
        plot_time_domain(sig, num_symbols=200, sps=1.0)
    assert "Limit exceeds number of symbols" in caplog.text
    plt.close("all")


def test_time_domain_auto_scale(backend_device, xp):
    """Verify time axis auto-scaling."""
    sig = xp.ones(100)

    # ns
    _, ax = plot_time_domain(sig, sampling_rate=1e10)
    assert "ns" in ax.get_xlabel()
    plt.close("all")

    # ps
    _, ax = plot_time_domain(sig, sampling_rate=1e13)
    assert "ps" in ax.get_xlabel()
    plt.close("all")

    # us
    _, ax = plot_time_domain(sig, sampling_rate=1e7)
    assert "µs" in ax.get_xlabel()
    plt.close("all")

    # ms
    _, ax = plot_time_domain(sig, sampling_rate=1e4)
    assert "ms" in ax.get_xlabel()
    plt.close("all")


def test_eye_diagram_sps_error(backend_device, xp):
    """Verify error for non-integer sps."""
    with pytest.raises(ValueError, match="sps must be an integer"):
        plot_eye_diagram(xp.ones(10), sps=2.5)
        plt.close("all")


def test_eye_diagram_trace_len_error(backend_device, xp):
    """Verify error when signal too short for eye trace."""
    with pytest.raises(
        ValueError, match="Signal is shorter than the required trace length"
    ):
        _plot_eye_traces(
            xp.ones(5), sps=10, num_symbols=2, ax=MagicMock(), type="line", title=None
        )
        plt.close("all")


def test_eye_diagram_invalid_type(backend_device, xp):
    """Verify error for unknown eye type."""
    with pytest.raises(ValueError, match="Unknown type"):
        _plot_eye_traces(
            xp.ones(100), sps=4, num_symbols=2, ax=MagicMock(), type="magic", title=None
        )
        plt.close("all")


def test_filter_response_no_sps(backend_device, xp):
    """Test filter_response without sps."""
    plot_filter_response(xp.ones(10))
    plt.close("all")


def test_constellation_histogram_overlay_warning(caplog, backend_device, xp):
    """Verify warning when overlaying ideal on histogram."""
    caplog.set_level(logging.WARNING)
    # Constellation with bins > 0 triggers histogram
    # If modulation is None, it should warn
    plot_constellation(xp.ones(10) + 1j, bins=10, overlay_ideal=True, modulation=None)
    assert "Modulation and order must be provided" in caplog.text
    plt.close("all")


# -----------------------------------------------------------------------------
# PSD - FREQUENCY SCALING AND SHOW
# -----------------------------------------------------------------------------


def test_psd_ghz_scaling(backend_device, xp):
    """PSD with GHz-range sampling rate triggers GHz scale factor."""
    samples = xp.random.randn(256).astype(xp.float32)
    fig, ax = plot_psd(samples, sampling_rate=5e9, show=False)
    assert fig is not None
    plt.close("all")


def test_psd_khz_scaling(backend_device, xp):
    """PSD with kHz-range sampling rate triggers kHz scale factor."""
    samples = xp.random.randn(256).astype(xp.float32)
    fig, ax = plot_psd(samples, sampling_rate=5e3, show=False)
    assert fig is not None
    plt.close("all")


def test_psd_hz_scaling(backend_device, xp):
    """PSD with Hz-range sampling rate uses default Hz unit."""
    samples = xp.random.randn(256).astype(xp.float32)
    fig, ax = plot_psd(samples, sampling_rate=100.0, show=False)
    assert fig is not None
    plt.close("all")


def test_psd_xlim_ylim(backend_device, xp):
    """PSD with xlim and ylim parameters applies axis limits."""
    samples = xp.random.randn(256).astype(xp.float32)
    fig, ax = plot_psd(
        samples, sampling_rate=1e6, xlim=(-0.4, 0.4), ylim=(-80, 0), show=False
    )
    assert fig is not None
    plt.close("all")


def test_psd_show(backend_device, xp):
    """PSD with show=True calls plt.show() and returns None."""
    samples = xp.random.randn(256).astype(xp.float32)
    with patch("matplotlib.pyplot.show"):
        result = plot_psd(samples, sampling_rate=1e6, show=True)
    assert result is None
    plt.close("all")


def test_psd_multichannel_show(backend_device, xp):
    """PSD multichannel with show=True calls plt.show() and returns None."""
    samples = xp.random.randn(2, 256).astype(xp.float32)
    with patch("matplotlib.pyplot.show"):
        result = plot_psd(samples, sampling_rate=1e6, show=True)
    assert result is None
    plt.close("all")


def test_psd_multichannel_single_axis_warning(backend_device, xp, caplog):
    """PSD multichannel with single axis warns and overlays all channels on it."""
    samples = xp.random.randn(2, 256).astype(xp.float32)
    fig0, ax0 = plt.subplots()
    caplog.set_level(logging.WARNING)
    fig, axes = plot_psd(samples, sampling_rate=1e6, ax=ax0, show=False)
    assert "Overlaying plots" in caplog.text
    plt.close("all")


# -----------------------------------------------------------------------------
# CONSTELLATION - REAL SAMPLES, MULTICHANNEL, SHOW
# -----------------------------------------------------------------------------


def test_constellation_real_samples(backend_device, xp):
    """Constellation with real (non-complex) samples warns and converts to complex."""
    samples = xp.ones(100, dtype=xp.float32)
    with patch("commkit.plotting.logger.warning"):
        result = plot_constellation(samples, show=False)
    assert result is not None
    plt.close("all")


def test_constellation_multichannel_single_axis_warning(backend_device, xp, caplog):
    """Constellation multichannel with single axis warns and overlays all channels."""
    samples = xp.ones((2, 100), dtype=xp.complex64)
    fig0, ax0 = plt.subplots()
    caplog.set_level(logging.WARNING)
    result = plot_constellation(samples, ax=ax0, show=False)
    assert result is not None
    assert "Overlaying plots" in caplog.text
    plt.close("all")


def test_constellation_multichannel_axes_array(backend_device, xp):
    """Constellation multichannel with axes array normalizes to 2D layout."""
    samples = xp.ones((2, 100), dtype=xp.complex64)
    fig0, axes0 = plt.subplots(1, 2)
    result = plot_constellation(samples, ax=axes0, show=False)
    assert result is not None
    plt.close("all")


def test_constellation_multichannel_show(backend_device, xp):
    """Constellation multichannel with show=True calls plt.show() and returns None."""
    samples = xp.ones((2, 100), dtype=xp.complex64)
    with patch("matplotlib.pyplot.show"):
        result = plot_constellation(samples, show=True)
    assert result is None
    plt.close("all")


def test_constellation_siso_show(backend_device, xp):
    """Constellation SISO with show=True calls plt.show() and returns None."""
    samples = xp.ones(100, dtype=xp.complex64)
    with patch("matplotlib.pyplot.show"):
        result = plot_constellation(samples, show=True)
    assert result is None
    plt.close("all")


# -----------------------------------------------------------------------------
# EQUALIZER DIAGNOSTICS - MIMO, CUSTOM AXES, SHOW
# -----------------------------------------------------------------------------


def test_equalizer_result_mimo_weights(backend_device, xp):
    """equalizer_result with MIMO error/weights plots per-channel error curves and weight matrices."""
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


def test_equalizer_result_custom_axes(backend_device, xp):
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


def test_equalizer_result_show(backend_device, xp):
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


# -----------------------------------------------------------------------------
# EYE DIAGRAM - AXES ARRAY, SHOW
# -----------------------------------------------------------------------------


def test_eye_diagram_multichannel_axes_array(backend_device, xp):
    """Eye diagram multichannel with axes array reshapes them for per-channel plotting."""
    samples = xp.random.randn(2, 1000).astype(xp.float32)
    fig0, axes0 = plt.subplots(2, 1)
    fig, axes = plot_eye_diagram(samples, sps=4, ax=list(axes0), show=False)
    assert fig is not None
    plt.close("all")


def test_eye_diagram_multichannel_show(backend_device, xp):
    """Eye diagram multichannel with show=True calls plt.show() and returns None."""
    samples = xp.random.randn(2, 1000).astype(xp.float32)
    with patch("matplotlib.pyplot.show"):
        result = plot_eye_diagram(samples, sps=4, show=True)
    assert result is None
    plt.close("all")


def test_eye_diagram_real_with_ax(backend_device, xp):
    """Eye diagram for real signal with a single pre-existing axis uses it directly."""
    samples = xp.random.randn(500).astype(xp.float32)
    fig0, ax0 = plt.subplots()
    fig, ax = plot_eye_diagram(samples, sps=4, ax=ax0, show=False)
    assert fig is not None
    plt.close("all")


# -----------------------------------------------------------------------------
# PSD - MULTICHANNEL AXES ARRAY
# -----------------------------------------------------------------------------


def test_psd_multichannel_axes_array(backend_device, xp):
    """PSD multichannel with axes ndarray normalizes axes to 2D layout."""
    samples = xp.random.randn(2, 256).astype(xp.float32)
    fig0, axes0 = plt.subplots(1, 2)
    fig, axes = plot_psd(samples, sampling_rate=1e6, ax=axes0, show=False)
    assert fig is not None
    plt.close("all")


# -----------------------------------------------------------------------------
# TIME DOMAIN - MULTICHANNEL AXES, SHOW, SISO SHOW
# -----------------------------------------------------------------------------


def test_time_domain_multichannel_axes_array(backend_device, xp):
    """plot_time_domain() multichannel with axes array normalizes axes to 2D layout."""
    samples = xp.random.randn(2, 1000).astype(xp.float32)
    fig0, axes0 = plt.subplots(1, 2)
    result = plot_time_domain(samples, sampling_rate=1e6, ax=axes0, show=False)
    assert result is not None
    plt.close("all")


def test_time_domain_multichannel_show(backend_device, xp):
    """plot_time_domain() multichannel with show=True calls plt.show() and returns None."""
    samples = xp.random.randn(2, 1000).astype(xp.float32)
    with patch("matplotlib.pyplot.show"):
        result = plot_time_domain(samples, sampling_rate=1e6, show=True)
    assert result is None
    plt.close("all")


def test_time_domain_siso_show(backend_device, xp):
    """plot_time_domain() 1D with show=True calls plt.show() and returns None."""
    samples = xp.random.randn(500).astype(xp.float32)
    with patch("matplotlib.pyplot.show"):
        result = plot_time_domain(samples, sampling_rate=1e6, show=True)
    assert result is None
    plt.close("all")


# -----------------------------------------------------------------------------
# EYE DIAGRAM - SISO SHOW, COMPLEX SINGLE AX ERROR, DENSE SIGNAL
# -----------------------------------------------------------------------------


def test_eye_diagram_siso_show(backend_device, xp):
    """plot_eye_diagram() 1D with show=True calls plt.show() and returns None."""
    samples = xp.random.randn(1000).astype(xp.float32)
    with patch("matplotlib.pyplot.show"):
        result = plot_eye_diagram(samples, sps=4, show=True)
    assert result is None
    plt.close("all")


def test_eye_diagram_complex_single_ax_error(backend_device, xp):
    """plot_eye_diagram() with complex signal and single ax raises ValueError requiring two axes."""
    samples = (xp.random.randn(500) + 1j * xp.random.randn(500)).astype(xp.complex64)
    fig0, ax0 = plt.subplots()
    with pytest.raises(ValueError, match="complex"):
        plot_eye_diagram(samples, sps=4, ax=ax0, show=False)
    plt.close("all")


def test_eye_diagram_dense_line(backend_device, xp):
    """plot_eye_diagram() 'line' type with num_traces>5000 triggers downsampling skip path."""
    # sps=2, N=10200, trace_len=4: num_traces = (10200-4)//2+1 = 5099 > 5000
    samples = xp.random.randn(10200).astype(xp.float32)
    fig, ax = plot_eye_diagram(samples, sps=2, type="line", num_symbols=2, show=False)
    assert fig is not None
    plt.close("all")


def test_eye_diagram_dense_hist(backend_device, xp):
    """plot_eye_diagram() 'hist' type with num_traces>20000 triggers downsampling skip path."""
    # sps=2, N=40100, trace_len=4: num_traces = (40100-4)//2+1 = 20049 > 20000
    samples = xp.random.randn(40100).astype(xp.float32)
    fig, ax = plot_eye_diagram(samples, sps=2, type="hist", num_symbols=2, show=False)
    assert fig is not None
    plt.close("all")


# -----------------------------------------------------------------------------
# FILTER RESPONSE - COMPLEX TAPS, SHOW
# -----------------------------------------------------------------------------


def test_filter_response_complex_taps(backend_device, xp):
    """plot_filter_response() with complex taps plots I/Q components separately."""
    taps = filtering.rrc_taps(sps=4, span=4, rolloff=0.35)
    complex_taps = taps.astype(complex)
    result = plot_filter_response(complex_taps, sps=4, show=False)
    assert result is not None
    plt.close("all")


def test_filter_response_show(backend_device, xp):
    """plot_filter_response() with show=True calls plt.show() and returns None."""
    taps = filtering.rrc_taps(sps=4, span=4, rolloff=0.35)
    with patch("matplotlib.pyplot.show"):
        result = plot_filter_response(taps, sps=4, show=True)
    assert result is None
    plt.close("all")


# -----------------------------------------------------------------------------
# IDEAL CONSTELLATION - CUSTOM AX, SHOW
# -----------------------------------------------------------------------------


def test_ideal_constellation_custom_ax(backend_device, xp):
    """plot_ideal_constellation() with provided ax uses that axis's figure."""
    fig0, ax0 = plt.subplots()
    result = plot_ideal_constellation(modulation="psk", order=4, ax=ax0, show=False)
    assert result is not None
    plt.close("all")


def test_ideal_constellation_show(backend_device, xp):
    """plot_ideal_constellation() with show=True calls plt.show() and returns None."""
    with patch("matplotlib.pyplot.show"):
        result = plot_ideal_constellation(modulation="qam", order=16, show=True)
    assert result is None
    plt.close("all")


# -----------------------------------------------------------------------------
# CONSTELLATION - VMIN/VMAX
# -----------------------------------------------------------------------------


def test_constellation_vmin_vmax(backend_device, xp):
    """plot_constellation() with vmin/vmax sets color scale bounds on the histogram plot."""
    samples = (xp.random.randn(500) + 1j * xp.random.randn(500)).astype(xp.complex64)
    result = plot_constellation(samples, vmin=0.0, vmax=1.0, show=False)
    assert result is not None
    plt.close("all")


# -----------------------------------------------------------------------------
# EQUALIZER RESULT - SHORT SMOOTHING WINDOW (SISO AND MIMO)
# -----------------------------------------------------------------------------


def test_equalizer_result_short_smoothing_siso(backend_device, xp):
    """plot_equalizer_result() SISO where len(mse) <= smoothing uses raw mse without smoothing."""
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
    # smoothing=1000 >> len(error)=25 -> uses raw mse without smoothing
    fig, axes = plot_equalizer_result(result, smoothing=1000)
    assert fig is not None
    plt.close("all")


def test_equalizer_result_short_smoothing_mimo(backend_device, xp):
    """plot_equalizer_result() MIMO where len(mse) <= smoothing uses raw mse without smoothing."""
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
    # smoothing=1000 >> len(error)=25 -> uses raw mse without smoothing
    fig, axes = plot_equalizer_result(result, smoothing=1000)
    assert fig is not None
    plt.close("all")


def test_spectrogram_plot_siso(backend_device, xp):
    """Verify spectrogram plotting for SISO (1D) signals."""
    from unittest.mock import patch

    fs = 100.0
    t = xp.arange(1000) / fs
    samples = xp.sin(2 * xp.pi * 20.0 * t)

    # 1. Basic plot
    fig, ax = plot_spectrogram(samples, sampling_rate=fs, show=False)
    assert fig is not None
    assert ax is not None
    plt.close("all")

    # 2. Plot with xlim (frequency limit) and ylim (time limit)
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

    # 3. Test show=True path (returns None)
    with patch("matplotlib.pyplot.show"):
        res = plot_spectrogram(samples, sampling_rate=fs, show=True)
    assert res is None
    plt.close("all")


def test_spectrogram_plot_mimo(backend_device, xp):
    """Verify spectrogram plotting for MIMO (2D) signals generates subplots."""
    fs = 100.0
    t = xp.arange(1000) / fs
    samples = xp.stack([xp.sin(2 * xp.pi * 20.0 * t), xp.cos(2 * xp.pi * 10.0 * t)])

    fig, axes = plot_spectrogram(samples, sampling_rate=fs, show=False)
    assert fig is not None
    # For 2-channel MIMO signal, it should create a grid of 1x2 or 2x1 subplots
    assert isinstance(axes, np.ndarray)
    assert axes.size == 2
    plt.close("all")


def test_signal_spectrogram_convenience(backend_device, xp):
    """Verify core.Signal.spectrogram and plot_spectrogram convenience methods."""

    fs = 100.0
    sig = generate_psk(
        symbol_rate=10.0,
        num_symbols=100,
        order=4,
        sps=int(fs / 10.0),
        seed=42,
    )
    # Ensure samples match the backend device
    sig.samples = xp.asarray(sig.samples)

    # 1. Calculation convenience method
    f, t, Sxx = spectral.spectrogram(sig, nperseg=64, noverlap=32)
    assert isinstance(f, xp.ndarray)
    assert isinstance(t, xp.ndarray)
    assert isinstance(Sxx, xp.ndarray)
    assert len(f) == 64
    assert Sxx.shape[-1] == len(t)

    # 2. Plotting convenience method
    fig, ax = plotting.plot_spectrogram(sig, nperseg=64, show=False)
    assert fig is not None
    assert ax is not None
    plt.close("all")
