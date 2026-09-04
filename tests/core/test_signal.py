"""Tests for the base Signal class and its core signal processing methods."""

from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest

from commkit import (
    filtering,
    generate,
    generate_pam,
    generate_psk,
    generate_psqam,
    generate_qam,
    mapping,
    metrics,
    multirate,
    plotting,
    spectral,
)
from commkit.core import Signal


def test_signal_initialization(backend_device, xp):
    """Verify Signal initialization from basic Python lists and device-aware backend tracking."""
    # Test with list
    data = [1, 2, 3, 4]
    # Signal automatically moves to GPU if available (controlled by backend_device fixture)
    s = Signal(samples=data, sampling_rate=1.0, symbol_rate=1.0)
    assert isinstance(s.samples, xp.ndarray)

    assert s.sampling_rate == 1.0
    assert s.symbol_rate == 1.0


def test_signal_validation_heuristics(backend_device, xp):
    """Verify Signal validation for higher dimensions and Time-Last heuristic."""
    # 1. Dimension > 2
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="Only 1D"):
        Signal(samples=xp.zeros((2, 2, 10)), sampling_rate=1.0, symbol_rate=1.0)

    # 2. Time-Last heuristic (Time, Channels) -> (Channels, Time)
    # Heuristic: s0 > s1 and s0 > 32
    data_wrong = xp.zeros((100, 2))
    s = Signal(samples=data_wrong, sampling_rate=1.0, symbol_rate=1.0)
    assert s.samples.shape == (2, 100)  # Should be transposed


def test_signal_auto_symbols(backend_device, xp):
    """Verify source_symbols derivation from source_bits in post-init."""

    bits = xp.array([0, 1, 0, 0], dtype="int8")
    # BPSK mapping: 0 -> -1, 1 -> 1
    s = Signal(
        samples=xp.ones(10),
        sampling_rate=1.0,
        symbol_rate=1.0,
        source_bits=bits,
        mod_scheme="PSK",
        mod_order=2,
    )
    assert s.source_symbols is not None
    assert len(s.source_symbols) == 4

    # 262: hyphenated mod
    s2 = Signal(
        samples=xp.ones(10),
        sampling_rate=1.0,
        symbol_rate=1.0,
        source_bits=bits,
        mod_scheme="PSK-MY",
        mod_order=2,
    )
    assert s2.source_symbols is not None
    # Check if derivation handled the "PSK" vs "mapping.map_bits" etc.


def test_signal_properties(backend_device, xp):
    """Verify core time-domain and rate properties of the Signal object."""
    # Create data directly on device using xp
    data = xp.zeros(100)
    fs = 100.0
    sym_rate = 10.0

    s = Signal(samples=data, sampling_rate=fs, symbol_rate=sym_rate)

    assert s.duration == 1.0
    assert s.sps == 10.0
    assert len(s.time_axis()) == 100


def test_signal_add_pilot_tone(backend_device, xp):
    """add_pilot_tone returns a new Signal with the tone and snapped freq recorded."""
    sig = generate_qam(num_symbols=256, sps=8, symbol_rate=1e6, order=16, seed=3)
    fs = sig.sampling_rate
    df = fs / sig.samples.shape[-1]
    before = sig.samples.copy()

    ret = spectral.add_pilot_tone(sig, 2.0e6, power_ratio_db=-12.0)

    assert ret is not sig  # pure: a new Signal is returned
    assert ret.pilot_tone_frequency is not None
    # Frequency and power are recorded as 1-D per-channel arrays (SISO -> len 1).
    assert isinstance(ret.pilot_tone_frequency, np.ndarray)
    assert ret.pilot_tone_frequency.shape == (1,)
    np.testing.assert_array_equal(ret.pilot_tone_power_ratio_db, [-12.0])
    f_p = float(ret.pilot_tone_frequency[0])
    # Recorded frequency is on the FFT grid and near the request.
    assert abs(round(f_p / df) - f_p / df) < 1e-9
    assert abs(f_p - 2.0e6) <= df / 2 + 1.0
    # Samples actually changed.
    assert float(xp.max(xp.abs(ret.samples - before))) > 0.0


def test_signal_methods(backend_device, xp):
    """Verify common Signal methods like upsampling and FIR filtering."""
    # Test upsample
    data = xp.array([1.0 + 0j, -1.0 + 0j])
    s = Signal(samples=data, sampling_rate=1.0, symbol_rate=1.0)

    s = multirate.upsample(s, 2)
    assert s.sampling_rate == 2.0
    assert s.samples.shape[0] == 4

    # Test fir_filter
    taps = xp.array([1.0])
    s = filtering.fir_filter(s, taps)
    # should be unchanged roughly


def test_signal_resample_sps(backend_device, xp):
    """Verify Signal resampling using target samples per symbol (SPS)."""
    data = xp.ones(100)
    # create signal with sps=4 (fs=4, sym_rate=1)
    s = Signal(samples=data, sampling_rate=4.0, symbol_rate=1.0)
    assert s.sps == 4.0

    # resample to sps=8
    s = multirate.resample(s, sps_out=8.0)
    assert s.sps == 8.0
    assert s.sampling_rate == 8.0
    assert s.samples.size == 200


def test_welch_psd(backend_device, xp):
    """Verify Welch PSD estimation within the Signal object."""
    data = xp.random.randn(1000) + 1j * xp.random.randn(1000)
    s = Signal(samples=data, sampling_rate=100.0, symbol_rate=10.0)

    f, p = spectral.welch_psd(s, nperseg=64)
    assert f.shape == p.shape
    assert isinstance(f, xp.ndarray)

    # Test custom parameters
    f2, p2 = spectral.welch_psd(
        s,
        nperseg=64,
        window=("kaiser", 8.0),
        noverlap=32,
        nfft=128,
        scaling="spectrum",
    )
    assert len(f2) == 128
    assert f2.shape == p2.shape


def test_signal_print_info(backend_device, xp, capsys):
    """Verify print_info() execution and output detection."""
    data = xp.zeros(10)
    s = Signal(samples=data, sampling_rate=100.0, symbol_rate=10.0)
    s.print_info()
    captured = capsys.readouterr()
    assert "Spectral Domain" in captured.out or captured.out == ""


def test_shaping_filter_taps_error(backend_device, xp):
    """Verify that shaping_filter_taps raises errors for unconfigured or unknown shapes."""
    data = xp.zeros(10)
    s = Signal(samples=data, sampling_rate=100.0, symbol_rate=10.0)

    with pytest.raises(ValueError, match="No pulse shape defined"):
        filtering.shaping_filter_taps(s)

    s.pulse_shape = "invalid_shape"
    with pytest.raises(ValueError, match="Unknown pulse shape"):
        filtering.shaping_filter_taps(s)


def test_signal_clone(backend_device, xp, xpt):
    """Verify Signal.clone() deep-copies arrays and preserves device context."""
    data = xp.array([1, 2, 3])
    source_bits = xp.array([0, 1, 0])
    s = Signal(
        samples=data,
        sampling_rate=1.0,
        symbol_rate=1.0,
        source_bits=source_bits,
    )
    s_copy = s.clone()

    assert s_copy is not s
    xpt.assert_allclose(s.samples, s_copy.samples)
    assert s_copy.samples is not s.samples
    assert s_copy.source_bits is not s.source_bits
    assert s_copy.backend == s.backend


def test_signal_clone_can_be_explicitly_shallow(backend_device, xp):
    """Signal.clone(deep=False) shares array metadata intentionally."""
    s = Signal(
        samples=xp.arange(8),
        sampling_rate=2.0,
        symbol_rate=1.0,
        source_bits=xp.arange(4),
    )

    shallow = s.clone(deep=False)

    assert shallow is not s
    assert shallow.samples is s.samples
    assert shallow.source_bits is s.source_bits


def test_signal_with_samples_shares_provenance_and_invalidates_caches(
    backend_device, xp, xpt
):
    """Functional sample replacement avoids copying old samples or provenance."""
    frame = {"cached": xp.arange(4)}
    s = Signal(
        samples=xp.arange(8, dtype=xp.float32),
        sampling_rate=2.0,
        symbol_rate=1.0,
        source_bits=xp.arange(4),
        source_symbols=xp.asarray([1.0, -1.0]),
        frame=frame,
    )
    s.resolved_symbols = xp.asarray([1.0, -1.0])
    s.resolved_bits = xp.asarray([0, 1])
    old_samples = s.samples
    replacement = xp.arange(4, dtype=xp.float32) + 10

    result = s.with_samples(replacement, sampling_rate=1.0)

    assert result is not s
    assert result.samples is replacement
    assert result.samples is not old_samples
    assert result.source_bits is s.source_bits
    assert result.source_symbols is s.source_symbols
    assert result.frame is frame
    assert result.resolved_symbols is None
    assert result.resolved_bits is None
    assert s.samples is old_samples
    assert s.resolved_symbols is not None
    assert s.resolved_bits is not None
    assert result.sampling_rate == 1.0
    xpt.assert_array_equal(result.samples, replacement)


def test_signal_with_samples_can_preserve_resolved_caches(backend_device, xp):
    """Proven-safe internal transforms can explicitly retain resolved caches."""
    s = Signal(samples=xp.arange(8), sampling_rate=2.0, symbol_rate=1.0)
    s.resolved_symbols = xp.asarray([1.0, -1.0])
    s.resolved_bits = xp.asarray([0, 1])

    result = s.with_samples(s.samples.copy(), _preserve_resolved=True)

    assert result.resolved_symbols is s.resolved_symbols
    assert result.resolved_bits is s.resolved_bits


def test_signal_with_samples_validates_replacement_and_metadata(backend_device, xp):
    """Replacement samples and metadata pass through assignment validation."""
    from pydantic import ValidationError

    s = Signal(samples=xp.arange(8), sampling_rate=2.0, symbol_rate=1.0)

    with pytest.raises(ValidationError, match="greater than 0"):
        s.with_samples(s.samples.copy(), sampling_rate=0.0)
    with pytest.raises((ValueError, ValidationError), match="Only 1D"):
        s.with_samples(xp.zeros((2, 2, 2)))


def test_signal_shift_frequency(backend_device, xp, xpt):
    """Verify frequency shifting logic and resulting spectral peak positioning."""
    fs = 100.0
    # Simple DC signal (freq 0)
    data = xp.ones(100, dtype="complex128")
    s = Signal(samples=data, sampling_rate=fs, symbol_rate=10.0)

    # Offset by 20 Hz
    s = spectral.shift_frequency(s, 20.0)

    # 1. Check metadata
    assert s.digital_frequency_offset == 20.0

    # 2. Check signal content
    t = xp.arange(100) / fs
    expected = xp.exp(1j * 2 * xp.pi * 20.0 * t)
    xpt.assert_allclose(s.samples, expected)

    # 3. Check accumulation
    s = spectral.shift_frequency(s, 5.0)
    assert s.digital_frequency_offset == 25.0

    # Check approximate freq
    f, p = spectral.welch_psd(s, nperseg=64)
    peak = f[xp.argmax(p)]
    # 25 Hz expected
    assert abs(peak - 25.0) < (fs / 64)


def test_signal_resolution_and_demap(backend_device, xp, xpt):
    """Verify manual symbol resolution and bit demapping with caching."""
    # Generate a simple BPSK signal at 4 SPS
    symbol_rate = 1e6
    sps = 4
    num_symbols = 100
    sig = generate_psk(
        order=2,
        num_symbols=num_symbols,
        sps=sps,
        symbol_rate=symbol_rate,
        seed=42,
    )

    # Initially resolved attributes should be None
    assert sig.resolved_symbols is None
    assert sig.resolved_bits is None

    # Calling demap_symbols_hard before resolve_symbols should raise ValueError
    with pytest.raises(ValueError, match="No resolved symbols available"):
        sig = mapping.demap_symbols_hard(sig)

    # Resolve symbols with offset
    sig = multirate.resolve_symbols(sig, offset=0)
    assert sig.resolved_symbols is not None
    assert len(sig.resolved_symbols) == num_symbols
    assert isinstance(sig.resolved_symbols, xp.ndarray)

    # Demap
    sig = mapping.demap_symbols_hard(sig)
    assert sig.resolved_bits is not None
    assert len(sig.resolved_bits) == num_symbols

    # Metrics should now work
    evm_pct, evm_db = metrics.evm(sig)
    assert evm_pct >= 0

    snr_db = metrics.snr(sig)
    assert snr_db > 0

    # Test BER with manual reference bits
    ref_bits = sig.source_bits
    if ref_bits is not None:
        # ber() requires resolved_bits (populated by demap_symbols_hard above)
        ber = metrics.ber(sig, bits_tx=ref_bits)
        assert 0 <= ber <= 1

    # Test that BER raises if resolved_bits is missing
    sig.resolved_bits = None
    with pytest.raises(ValueError, match="No resolved bits available"):
        metrics.ber(sig, bits_tx=ref_bits)


def test_signal_decimate_to_symbol_rate(backend_device, xp):
    """Verify downsampling Signal to symbols."""
    data = xp.ones(40, dtype="complex128")
    s = Signal(samples=data, sampling_rate=4.0, symbol_rate=1.0)
    s = multirate.decimate_to_symbol_rate(s, offset=0)
    assert len(s.samples) == 10
    assert s.sampling_rate == 1.0


def test_signal_downsample_warning(backend_device, xp):
    """Verify warning when downsampling already 1 SPS signal."""
    data = xp.ones(10, dtype="complex128")
    s = Signal(samples=data, sampling_rate=1.0, symbol_rate=1.0)
    multirate.decimate_to_symbol_rate(s)  # Should just warn


def test_signal_mimo_fir_coverage(backend_device, xp):
    """Verify FIR filtering on multichannel signals."""
    data = xp.ones((2, 100), dtype="complex128")
    s = Signal(samples=data, sampling_rate=1.0, symbol_rate=1.0)
    # Filter with delay
    taps = xp.array([1.0, 0.5])
    s = filtering.fir_filter(s, taps)
    assert s.samples.shape == (2, 100)
    # y[1] should be 1.5
    assert abs(float(s.samples[0, 1].real) - 1.5) < 1e-10


def test_signal_gaussian_coverage(backend_device, xp):
    """Verify Gaussian Signal generation."""
    # Use PSK with Gaussian pulse shaping
    s = generate_psk(
        order=2,
        pulse_shape="gaussian",
        symbol_rate=1e6,
        num_symbols=100,
        sps=8,
        duty_cycle=0.5,
    )
    assert s.pulse_shape == "gaussian"
    assert s.duty_cycle == 0.5


def test_signal_jax_interop(backend_device, xp, xpt):
    """Verify JAX interoperability."""
    try:
        import jax
    except ImportError:
        pytest.skip("JAX not installed")

    s = Signal(samples=xp.ones(10), sampling_rate=1.0, symbol_rate=1.0)

    # Export to JAX
    jax_arr = s.export_samples_to_jax()
    assert isinstance(jax_arr, jax.Array)

    # Update from JAX
    jax_arr = jax_arr * 2.0
    s.update_samples_from_jax(jax_arr)
    xpt.assert_allclose(s.samples, 2.0)


def test_signal_properties_coverage(backend_device, xp):
    """Access Signal properties to ensure coverage."""
    s = Signal(samples=xp.zeros(100), sampling_rate=10.0, symbol_rate=2.0)

    # sp property
    assert s.sp is not None

    # duration property
    assert s.duration == 10.0

    # backend
    assert s.backend in ("CPU", "GPU")


def test_signal_validate_samples_transposition(backend_device, xp):
    """Cover the transposition warning heuristic in Signal."""
    # Create (Time, Channels) where Time >> Channels and Time > 32
    # e.g. (100, 2)
    data = xp.zeros((100, 2))

    # Signal expects (Channels, Time) usually, but logic detects (Time, Channels)
    # and transposes it, logging a warning.
    # We verify the shape is flipped to (2, 100).
    sig = Signal(samples=data, sampling_rate=1.0, symbol_rate=1.0)

    assert sig.samples.shape == (2, 100)

    assert sig.bits_per_symbol is None


def test_signal_wrappers(backend_device, xp):
    """
    Test the wrapper methods on Signal to ensure they call the underlying modules.
    We just check they run without error.
    """

    sig = Signal(
        samples=xp.zeros(100, dtype="complex64"), sampling_rate=100.0, symbol_rate=10.0
    )

    # Print info
    sig.print_info()

    # Properties
    assert sig.duration == 1.0
    assert sig.num_streams == 1

    # wrappers
    # Use small nperseg to match signal length
    f, p = spectral.welch_psd(sig, nperseg=32)
    assert len(f) > 0

    # Clean up any existing figures from previous tests
    plt.close("all")

    # Plotting wrappers (just call them, assume plotting logic tested elsewhere)
    # We pass show=False to avoid blocking
    plotting.plot_psd(sig, show=False, nperseg=32)
    plotting.plot_psd(
        sig,
        show=False,
        nperseg=32,
        window=("kaiser", 8.0),
        noverlap=16,
        nfft=64,
        scaling="spectrum",
    )
    plotting.plot_time_domain(sig, num_symbols=10, show=False)
    plotting.plot_eye_diagram(sig, show=False)
    plotting.plot_constellation(sig, show=False)

    # Clean up figures to avoid RuntimeWarning
    plt.close("all")


def test_signal_duration_mimo(backend_device, xp):
    """Verify duration property for MIMO (2D) signal."""
    data = xp.zeros((2, 200))
    s = Signal(samples=data, sampling_rate=100.0, symbol_rate=10.0)
    assert s.duration == 2.0  # 200 / 100


def test_signal_bits_per_symbol_set(backend_device, xp):
    """Verify bits_per_symbol property when mod_order is set."""
    s = Signal(samples=xp.zeros(10), sampling_rate=1.0, symbol_rate=1.0, mod_order=16)
    assert s.bits_per_symbol == 4

    s2 = Signal(samples=xp.zeros(10), sampling_rate=1.0, symbol_rate=1.0, mod_order=64)
    assert s2.bits_per_symbol == 6


def test_rzpam_odd_sps(backend_device, xp):
    """Verify RZ-PAM raises error for odd SPS."""
    with pytest.raises(ValueError, match="sps.*must be even"):
        generate_pam(
            order=2,
            num_symbols=10,
            sps=3,
            symbol_rate=1e3,
            rz=True,
        )


def test_rzpam_multi_stream(backend_device, xp):
    """Verify RZ-PAM multi-stream reshape produces correctly shaped multichannel output."""
    sig = generate_pam(
        order=2,
        num_symbols=10,
        sps=4,
        symbol_rate=1e3,
        rz=True,
        num_streams=2,
        pulse_shape="rect",
    )
    # Should have 2 channels
    assert sig.samples.ndim == 2
    assert sig.samples.shape[0] == 2
    assert sig.source_bits is not None
    assert sig.source_symbols is not None


@pytest.mark.parametrize(
    "factory,kwargs",
    [
        (
            generate,
            dict(
                num_symbols=10,
                sps=3.5,
                symbol_rate=1e6,
                modulation="qam",
                order=16,
            ),
        ),
        (
            generate_psk,
            dict(num_symbols=10, sps=3.5, symbol_rate=1e6, order=4),
        ),
        (
            generate_qam,
            dict(num_symbols=10, sps=3.5, symbol_rate=1e6, order=16),
        ),
        (
            generate_pam,
            dict(num_symbols=10, sps=3.5, symbol_rate=1e6, order=4),
        ),
        (
            generate_psqam,
            dict(num_symbols=10, sps=3.5, symbol_rate=1e6, order=64, nu=0.3),
        ),
    ],
)
def test_noninteger_sps_raises(backend_device, factory, kwargs):
    """Non-integer sps must raise at generation time, before any samples are produced."""
    with pytest.raises(ValueError, match="sps must be a positive integer"):
        factory(**kwargs)


def test_integer_valued_float_sps_accepted(backend_device):
    """sps=4.0 (integer-valued float) must succeed and produce correct sample count."""
    sig = generate_qam(num_symbols=100, sps=4.0, symbol_rate=1e6, order=16)
    assert sig.samples.shape[-1] == 100 * 4
    assert sig.sps == 4.0


def test_resolve_symbols_sps_errors(backend_device, xp):
    """Verify resolve_symbols error paths for invalid SPS values."""
    # SPS < 1 (symbol_rate > sampling_rate)
    s = Signal(
        samples=xp.ones(10, dtype="complex64"), sampling_rate=1.0, symbol_rate=2.0
    )
    with pytest.raises(ValueError, match="Symbol rate must be >= 1"):
        s = multirate.resolve_symbols(s)

    # Non-integer SPS
    s2 = Signal(
        samples=xp.ones(10, dtype="complex64"), sampling_rate=3.0, symbol_rate=2.0
    )
    with pytest.raises(ValueError, match="Symbol rate must be an integer"):
        s2 = multirate.resolve_symbols(s2)


def test_demap_without_modulation(backend_device, xp):
    """Verify demap_symbols_hard raises error without modulation metadata."""
    s = Signal(
        samples=xp.ones(10, dtype="complex64"), sampling_rate=1.0, symbol_rate=1.0
    )
    s.resolved_symbols = xp.ones(10, dtype="complex64")
    with pytest.raises(ValueError, match="Modulation scheme and order required"):
        s = mapping.demap_symbols_hard(s)


def test_evm_no_reference(backend_device, xp):
    """Verify evm raises when no reference is available."""
    s = Signal(
        samples=xp.ones(10, dtype="complex64"), sampling_rate=1.0, symbol_rate=1.0
    )
    s.resolved_symbols = xp.ones(10, dtype="complex64")
    with pytest.raises(ValueError, match="No reference available"):
        metrics.evm(s)


def test_evm_no_resolved(backend_device, xp):
    """Verify evm raises when no resolved_symbols are present."""
    s = Signal(
        samples=xp.ones(10, dtype="complex64"),
        sampling_rate=1.0,
        symbol_rate=1.0,
        source_symbols=xp.ones(10, dtype="complex64"),
    )
    with pytest.raises(ValueError, match="No resolved symbols available"):
        metrics.evm(s)


def test_snr_no_reference(backend_device, xp):
    """Verify snr raises when no reference is available."""
    s = Signal(
        samples=xp.ones(10, dtype="complex64"), sampling_rate=1.0, symbol_rate=1.0
    )
    s.resolved_symbols = xp.ones(10, dtype="complex64")
    with pytest.raises(ValueError, match="No reference available"):
        metrics.snr(s)


def test_snr_no_resolved(backend_device, xp):
    """Verify snr raises when no resolved_symbols are present."""
    s = Signal(
        samples=xp.ones(10, dtype="complex64"),
        sampling_rate=1.0,
        symbol_rate=1.0,
        source_symbols=xp.ones(10, dtype="complex64"),
    )
    with pytest.raises(ValueError, match="No resolved symbols available"):
        metrics.snr(s)


def test_ber_no_reference(backend_device, xp):
    """Verify ber raises when no reference bits are available."""
    s = Signal(
        samples=xp.ones(10, dtype="complex64"), sampling_rate=1.0, symbol_rate=1.0
    )
    s.resolved_bits = xp.array([0, 1, 0, 1])
    with pytest.raises(ValueError, match="No reference bits available"):
        metrics.ber(s)


def test_signal_rz_modscheme_flags(backend_device, xp):
    """Verify RZ modulation flags in __init__."""
    bits = xp.array([0, 1], dtype="int8")
    s = Signal(
        samples=xp.ones(10),
        sampling_rate=1.0,
        symbol_rate=1.0,
        source_bits=bits,
        mod_scheme="PAM",
        mod_order=2,
        mod_rz=True,
    )
    # source_symbols should be derived from bits using "PAM" mapping
    assert s.source_symbols is not None
    assert len(s.source_symbols) == 2
    assert s.mod_rz is True


# -----------------------------------------------------------------------------
# SIGNAL EVM / SNR / BER - num_train_symbols explicit discard
# -----------------------------------------------------------------------------


def test_evm_with_explicit_num_train_symbols(backend_device, xp):
    """evm(num_train_symbols=N) discards leading symbols from resolved_symbols."""
    from commkit import equalization

    n_symbols = 600
    n_train = 100
    orig = generate_psk(
        symbol_rate=1e6,
        num_symbols=n_symbols,
        order=4,
        pulse_shape="rrc",
        sps=2,
        seed=7,
    )
    result = equalization.lms(
        xp.asarray(orig.samples),
        training_symbols=orig.source_symbols[:n_train],
        sps=2,
        num_taps=7,
        backend="numba",
    )
    n_train = result.num_train_symbols
    rx = Signal(
        samples=result.y_hat,
        sampling_rate=orig.symbol_rate,
        symbol_rate=orig.symbol_rate,
        mod_scheme="psk",
        mod_order=4,
        source_symbols=orig.source_symbols[..., : result.y_hat.shape[-1]],
    )
    rx = multirate.resolve_symbols(rx)

    evm_pct, evm_db = metrics.evm(rx, num_train_symbols=n_train)
    assert np.isfinite(float(evm_db))
    assert float(evm_pct) > 0

    evm_pct2, evm_db2 = metrics.evm(rx)
    assert np.isfinite(float(evm_db2))


def test_snr_with_explicit_num_train_symbols(backend_device, xp):
    """snr(num_train_symbols=N) discards leading symbols before computing SNR."""
    from commkit import equalization

    n_symbols = 600
    orig = generate_psk(
        symbol_rate=1e6,
        num_symbols=n_symbols,
        order=4,
        pulse_shape="rrc",
        sps=2,
        seed=8,
    )
    result = equalization.lms(
        xp.asarray(orig.samples),
        training_symbols=orig.source_symbols[:100],
        sps=2,
        num_taps=7,
        backend="numba",
    )
    rx = Signal(
        samples=result.y_hat,
        sampling_rate=orig.symbol_rate,
        symbol_rate=orig.symbol_rate,
        mod_scheme="psk",
        mod_order=4,
        source_symbols=orig.source_symbols[..., : result.y_hat.shape[-1]],
    )
    rx = multirate.resolve_symbols(rx)

    snr_val = metrics.snr(rx, num_train_symbols=result.num_train_symbols)
    assert np.isfinite(snr_val)

    snr_notrim = metrics.snr(rx)
    assert np.isfinite(snr_notrim)


def test_ber_with_explicit_num_train_symbols(backend_device, xp):
    """ber(num_train_symbols=N) discards leading bits before computing BER."""
    from commkit import equalization

    n_symbols = 600
    orig = generate_psk(
        symbol_rate=1e6,
        num_symbols=n_symbols,
        order=4,
        pulse_shape="rrc",
        sps=2,
        seed=11,
    )
    result = equalization.lms(
        xp.asarray(orig.samples),
        training_symbols=orig.source_symbols[:100],
        sps=2,
        num_taps=7,
        backend="numba",
    )
    rx = Signal(
        samples=result.y_hat,
        sampling_rate=orig.symbol_rate,
        symbol_rate=orig.symbol_rate,
        mod_scheme="psk",
        mod_order=4,
        source_symbols=orig.source_symbols[..., : result.y_hat.shape[-1]],
        source_bits=orig.source_bits[..., : result.y_hat.shape[-1] * 2],
    )
    rx = multirate.resolve_symbols(rx)
    rx = mapping.demap_symbols_hard(rx)

    ber_val = metrics.ber(rx, num_train_symbols=result.num_train_symbols)
    assert np.isfinite(float(ber_val))
    assert 0.0 <= float(ber_val) <= 1.0

    ber_notrim = metrics.ber(rx)
    assert np.isfinite(float(ber_notrim))
    assert 0.0 <= float(ber_notrim) <= 1.0


def test_rls_tail_trim_field(backend_device, xp):
    """rls() tail_trim field equals num_taps // 2 and y_hat is shortened accordingly."""
    from commkit import equalization

    n_symbols = 400
    num_taps = 7
    orig = generate_psk(
        symbol_rate=1e6,
        num_symbols=n_symbols,
        order=4,
        pulse_shape="rrc",
        sps=2,
        seed=3,
    )
    result = equalization.rls(
        xp.asarray(orig.samples),
        training_symbols=orig.source_symbols,
        sps=2,
        num_taps=num_taps,
        modulation="psk",
        order=4,
        backend="numba",
    )
    assert result.tail_trim == num_taps // 2
    assert result.y_hat.shape[-1] == n_symbols - result.tail_trim


def test_plot_constellation_at_symbol_rate(backend_device, xp):
    """plot_constellation on a 1-SPS Signal (built from lms y_hat) should succeed."""
    from commkit import equalization

    sig = generate_psk(
        symbol_rate=1e6, num_symbols=200, order=4, pulse_shape="rrc", sps=2, seed=0
    )
    result = equalization.lms(
        xp.asarray(sig.samples),
        training_symbols=sig.source_symbols,
        sps=2,
        num_taps=7,
        backend="numba",
    )
    rx_1sps = Signal(
        samples=result.y_hat,
        sampling_rate=sig.symbol_rate,
        symbol_rate=sig.symbol_rate,
        mod_scheme="psk",
        mod_order=4,
    )
    plot_result = plotting.plot_constellation(rx_1sps, show=False)
    assert plot_result is not None
    plt.close("all")


def test_plot_constellation_overlay_source_mimo(backend_device, xp):
    """Signal.plot_constellation with MIMO signal and overlay_source=True."""
    sig = generate_psk(
        symbol_rate=1e6,
        num_symbols=200,
        order=4,
        pulse_shape="rrc",
        sps=1,
        num_streams=2,
        seed=0,
    )
    result = plotting.plot_constellation(sig, overlay_source=True, show=False)
    assert result is not None
    plt.close("all")


def test_plot_constellation_show(backend_device, xp):
    """Signal.plot_constellation(show=True) should call plt.show() and return None."""
    sig = generate_psk(
        symbol_rate=1e6, num_symbols=100, order=4, pulse_shape="rrc", sps=1, seed=0
    )
    with patch("matplotlib.pyplot.show"):
        result = plotting.plot_constellation(sig, show=True)
    assert result is None
    plt.close("all")


def test_plot_constellation_overlay_source_siso(backend_device, xp):
    """SISO signal with overlay_source=True uses the single-axes scatter path."""
    sig = generate_psk(
        symbol_rate=1e6, num_symbols=200, order=4, pulse_shape="rrc", sps=1, seed=0
    )
    assert sig.num_streams == 1
    assert sig.source_symbols is not None

    result = plotting.plot_constellation(sig, overlay_source=True, show=False)
    assert result is not None
    plt.close("all")


# -----------------------------------------------------------------------------
# FACTORY METHOD TESTS
# -----------------------------------------------------------------------------


def test_pam_waveform(backend_device, xp):
    """Verify basic PAM signal generation produces samples on the active device."""
    sig = generate_pam(order=2, unipolar=False, num_symbols=10, sps=4, symbol_rate=1e3)
    assert sig.samples.size > 0
    assert isinstance(sig.samples, xp.ndarray)
    assert sig.mod_scheme is not None


def test_rzpam_waveform(backend_device, xp):
    """Verify Return-to-Zero PAM signal generation and pulse-shape validation."""
    sig = generate_pam(
        order=2,
        unipolar=False,
        num_symbols=10,
        sps=4,
        symbol_rate=1e3,
        rz=True,
        pulse_shape="rect",
    )
    assert sig.samples.size > 0
    assert isinstance(sig.samples, xp.ndarray)

    with pytest.raises(ValueError, match="not allowed for RZ PAM"):
        generate_pam(
            order=2,
            unipolar=False,
            num_symbols=10,
            sps=4,
            symbol_rate=1e3,
            rz=True,
            pulse_shape="rrc",
        )


def test_qam_waveform(backend_device, xp):
    """Verify QAM signal generation populates samples and modulation metadata."""
    sig = generate_qam(order=16, num_symbols=10, sps=4, symbol_rate=1e3)
    assert sig.samples.size > 0
    assert isinstance(sig.samples, xp.ndarray)
    assert sig.mod_order == 16


def test_psk_waveform(backend_device, xp, xpt):
    """Verify PSK signal generation, metadata, and unit-magnitude constellation."""
    sig = generate_psk(order=8, num_symbols=50, sps=2, symbol_rate=1e6, seed=0)
    assert sig.samples.size > 0
    assert isinstance(sig.samples, xp.ndarray)
    assert sig.mod_order == 8
    assert sig.mod_scheme is not None
    # All PSK symbols should lie on the unit circle
    syms = sig.source_symbols
    if syms is not None:
        magnitudes = xp.abs(syms)
        xpt.assert_allclose(magnitudes, xp.ones_like(magnitudes), atol=1e-5)


def test_signal_generate(backend_device, xp):
    """Verify generate() produces correct metadata for any modulation."""
    sig = generate(
        num_symbols=100,
        sps=4,
        symbol_rate=1e6,
        modulation="qam",
        order=16,
        pulse_shape="rrc",
        seed=1,
    )
    assert sig.samples.size > 0
    assert isinstance(sig.samples, xp.ndarray)
    assert sig.symbol_rate == 1e6
    assert sig.mod_order == 16
    assert sig.source_bits is not None
    assert sig.source_symbols is not None
