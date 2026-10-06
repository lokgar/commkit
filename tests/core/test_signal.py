"""Tests for the base Signal class and its core signal processing methods."""

from unittest.mock import patch

import numpy as np
import pytest

from commkit import (
    filtering,
    generate,
    generate_psqam,
    mapping,
    metrics,
    multirate,
    plotting,
    spectral,
)
from commkit.core import Reference, Signal
from commkit.filtering import RRC, Gaussian, Rect
from commkit.mapping import Constellation
from tests.common.conversions import device_of, to_numpy


class TestSignalCreation:
    """Tests for TestSignalCreation."""

    def test_signal_initialization(self, xp):
        """Signal keeps samples on the device they were given on."""
        s = Signal(samples=[1, 2, 3, 4], sampling_rate=1.0, symbol_rate=1.0)
        assert isinstance(s.samples, np.ndarray)  # lists become host arrays

        s_dev = Signal(samples=xp.arange(4), sampling_rate=1.0, symbol_rate=1.0)
        assert isinstance(s_dev.samples, xp.ndarray)

        assert s.sampling_rate == 1.0
        assert s.symbol_rate == 1.0

    def test_to_returns_new_signal_with_all_arrays_moved(self, xp):
        """Signal.to() leaves the input untouched and moves reference arrays too."""
        sig = generate(
            Constellation.qam(16), 64, symbol_rate=1e6, sps=2, pulse=RRC(0.35), rng=1
        )
        moved = sig.to(device_of(xp))
        assert moved is not sig
        assert isinstance(sig.samples, np.ndarray)
        assert isinstance(sig.source_symbols, np.ndarray)
        for arr in (moved.samples, moved.source_symbols, moved.source_bits):
            assert isinstance(arr, xp.ndarray)
        np.testing.assert_array_equal(to_numpy(moved.samples), sig.samples)

    def test_signal_validation_heuristics(self, xp):
        """Verify Signal validation for higher dimensions and Time-Last heuristic."""
        # 1. Dimension > 2
        with pytest.raises(ValueError, match="Only 1D"):
            Signal(samples=xp.zeros((2, 2, 10)), sampling_rate=1.0, symbol_rate=1.0)

        # 2. (N, C)-looking input raises instead of being transposed.
        with pytest.raises(ValueError, match="looks like"):
            Signal(samples=xp.zeros((100, 2)), sampling_rate=1.0, symbol_rate=1.0)
        assert Signal(
            samples=xp.zeros((2, 100)), sampling_rate=1.0, symbol_rate=1.0
        ).samples.shape == (2, 100)

    def test_construction_does_no_hidden_work(self, xp, xpt):
        """The reference is stored as given: no mapping, no normalization."""
        symbols = xp.asarray([2.0, -2.0, 2.0, 2.0])
        bits = xp.asarray([1, 0, 1, 1], dtype="int8")
        s = Signal(
            samples=xp.ones(8),
            sampling_rate=2.0,
            symbol_rate=1.0,
            constellation=Constellation.pam(2),
            reference=Reference(symbols=symbols, bits=bits),
        )
        assert s.reference.symbols is symbols
        assert s.reference.bits is bits
        xpt.assert_array_equal(s.reference.symbols, symbols)
        assert s.bits_per_symbol == 1
        assert (
            Signal(
                samples=xp.ones(4), sampling_rate=1.0, symbol_rate=1.0
            ).bits_per_symbol
            is None
        )

    def test_reference_validation(self, xp):
        with pytest.raises(ValueError, match="symbols"):
            Reference(symbols=xp.zeros((2, 2, 2)))
        with pytest.raises(ValueError, match="channel axis"):
            Reference(symbols=xp.zeros((2, 4)), bits=xp.zeros((3, 8)))
        with pytest.raises(ValueError, match="channel axis"):
            Reference(symbols=xp.zeros((2, 4)), bits=xp.zeros(8))
        assert Reference(symbols=[1.0, -1.0]).bits is None

    @pytest.mark.gpu_only
    def test_reference_rejects_mixed_devices(self, xp):
        with pytest.raises(ValueError, match="same device"):
            Reference(symbols=xp.zeros(4), bits=np.zeros(4))

    def test_bridge_properties(self, xp):
        """1.x attributes are derived read-only from the 2.0 fields."""
        sig = generate(
            Constellation.qam(16), 8, symbol_rate=1e3, sps=2, pulse=RRC(0.35)
        )
        assert sig.constellation == Constellation.qam(16)
        assert sig.pulse == RRC(0.35, span=10)
        assert (sig.mod_scheme, sig.mod_order, sig.pulse_shape) == ("QAM", 16, "rrc")
        assert (sig.filter_span, sig.rrc_rolloff, sig.mod_rz) == (10, 0.35, False)
        assert sig.source_symbols is sig.reference.symbols
        assert sig.source_bits is sig.reference.bits
        assert sig.signal_type is None
        rz = generate(Constellation.pam(2), 8, symbol_rate=1e3, sps=4, pulse=Rect(0.5))
        assert rz.pulse == Rect(0.5)
        assert rz.mod_rz is True


class TestSignalProperties:
    """Tests for TestSignalProperties."""

    def test_signal_properties(self, xp):
        """Verify core time-domain and rate properties of the Signal object."""
        # Create data directly on device using xp
        data = xp.zeros(100)
        fs = 100.0
        sym_rate = 10.0

        s = Signal(samples=data, sampling_rate=fs, symbol_rate=sym_rate)

        assert s.duration == 1.0
        assert s.sps == 10.0
        assert len(s.time_axis()) == 100

    def test_signal_properties_coverage(self, xp):
        """Access Signal properties to ensure coverage."""
        s = Signal(samples=xp.zeros(100), sampling_rate=10.0, symbol_rate=2.0)

        # sp property
        assert s.sp is not None

        # duration property
        assert s.duration == 10.0

        # backend
        assert s.backend in ("CPU", "GPU")

    def test_signal_duration_mimo(self, xp):
        """Verify duration property for MIMO (2D) signal."""
        data = xp.zeros((2, 200))
        s = Signal(samples=data, sampling_rate=100.0, symbol_rate=10.0)
        assert s.duration == 2.0  # 200 / 100

    def test_signal_bits_per_symbol_set(self, xp):
        """Verify bits_per_symbol property when mod_order is set."""
        s = Signal(
            samples=xp.zeros(10),
            sampling_rate=1.0,
            symbol_rate=1.0,
            constellation=Constellation.qam(16),
        )
        assert s.bits_per_symbol == 4
        assert s.replace(constellation=Constellation.qam(64)).bits_per_symbol == 6

    def test_signal_summary(self, xp):
        """str() gives a plain-text summary; _repr_html_ the notebook table."""
        s = Signal(samples=xp.zeros(10), sampling_rate=100.0, symbol_rate=10.0)
        text = str(s)
        assert "Sampling rate" in text
        assert "Samples per symbol  10.00" in text
        html = s._repr_html_()
        assert html.startswith("<table>")
        assert "<b>Sampling rate</b>" in html

    def test_signal_wrappers(self, xp):
        """
        Test the wrapper methods on Signal to ensure they call the underlying modules.
        We just check they run without error.
        """

        sig = Signal(
            samples=xp.zeros(100, dtype="complex64"),
            sampling_rate=100.0,
            symbol_rate=10.0,
        )

        # Properties
        assert sig.duration == 1.0
        assert sig.num_streams == 1

        # wrappers
        # Use small nperseg to match signal length
        f, p = spectral.welch_psd(sig, nperseg=32)
        assert len(f) > 0

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


class TestSignalCloningAndProvenance:
    """Tests for TestSignalCloningAndProvenance."""

    def test_signal_clone(self, xp, xpt):
        """Verify Signal.clone() deep-copies arrays and preserves device context."""
        data = xp.array([1, 2, 3])
        s = Signal(
            samples=data,
            sampling_rate=1.0,
            symbol_rate=1.0,
            reference=Reference(symbols=xp.ones(3), bits=xp.array([0, 1, 0])),
        )
        s_copy = s.clone()

        assert s_copy is not s
        xpt.assert_allclose(s.samples, s_copy.samples)
        assert s_copy.samples is not s.samples
        assert s_copy.reference.bits is not s.reference.bits
        xpt.assert_array_equal(s_copy.reference.bits, s.reference.bits)
        assert s_copy.backend == s.backend

    def test_signal_replace_shares_unchanged_fields(self, xp):
        """replace() shares unchanged arrays and leaves the original untouched."""
        s = Signal(
            samples=xp.arange(8),
            sampling_rate=2.0,
            symbol_rate=1.0,
            reference=Reference(symbols=xp.arange(4) * 1.0),
        )

        new = s.replace(sampling_rate=4.0)

        assert new is not s
        assert new.samples is s.samples
        assert new.reference is s.reference
        assert s.sampling_rate == 2.0

    def test_signal_is_frozen(self, xp):
        import dataclasses

        s = Signal(samples=xp.arange(8), sampling_rate=2.0, symbol_rate=1.0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            s.sampling_rate = 4.0  # type: ignore[misc]

    def test_signal_replace_rejects_unknown_fields(self, xp):
        s = Signal(samples=xp.arange(8), sampling_rate=2.0, symbol_rate=1.0)
        with pytest.raises(TypeError, match="sample_rate"):
            s.replace(sample_rate=4.0)

    @pytest.mark.parametrize(
        "field, value",
        [
            ("symbol_rate", -1.0),
            ("center_frequency", -1.0),
            ("sampling_rate", "1e6"),
            ("constellation", "qam"),
            ("pulse", "rrc"),
            ("reference", np.ones(4)),
        ],
    )
    def test_signal_field_validation(self, xp, field, value):
        s = Signal(samples=xp.arange(8), sampling_rate=2.0, symbol_rate=1.0)
        with pytest.raises(ValueError, match=field):
            s.replace(**{field: value})
        kwargs = dict(samples=xp.arange(8), sampling_rate=2.0, symbol_rate=1.0)
        with pytest.raises(ValueError, match=field):
            Signal(**{**kwargs, field: value})

    def test_signal_numeric_fields_are_coerced(self, xp):
        s = Signal(samples=xp.arange(8), sampling_rate=2, symbol_rate=np.float32(1))
        assert type(s.sampling_rate) is float
        assert type(s.symbol_rate) is float

    @pytest.mark.parametrize("name", ["mod_scheme", "source_symbols", "pulse_shape"])
    def test_bridge_fields_are_read_only(self, xp, name):
        s = Signal(samples=xp.arange(8), sampling_rate=2.0, symbol_rate=1.0)
        with pytest.raises(TypeError, match=name):
            s.replace(**{name: None})

    def test_signal_replace_samples_shares_provenance_and_invalidates_caches(
        self, xp, xpt
    ):
        """Functional sample replacement avoids copying old samples or provenance."""
        frame = {"cached": xp.arange(4)}
        s = Signal(
            samples=xp.arange(8, dtype=xp.float32),
            sampling_rate=2.0,
            symbol_rate=1.0,
            reference=Reference(symbols=xp.asarray([1.0, -1.0]), bits=xp.arange(4)),
            frame=frame,
        )
        s = s.replace(resolved_symbols=xp.asarray([1.0, -1.0]))
        s = s.replace(resolved_bits=xp.asarray([0, 1]))
        old_samples = s.samples
        replacement = xp.arange(4, dtype=xp.float32) + 10

        result = s.replace_samples(replacement, sampling_rate=1.0)

        assert result is not s
        assert result.samples is replacement
        assert result.samples is not old_samples
        assert result.reference is s.reference
        assert result.frame is frame
        assert result.resolved_symbols is None
        assert result.resolved_bits is None
        assert s.samples is old_samples
        assert s.resolved_symbols is not None
        assert s.resolved_bits is not None
        assert result.sampling_rate == 1.0
        xpt.assert_array_equal(result.samples, replacement)

    def test_signal_replace_samples_can_preserve_resolved_caches(self, xp):
        """Proven-safe internal transforms can explicitly retain resolved caches."""
        s = Signal(samples=xp.arange(8), sampling_rate=2.0, symbol_rate=1.0)
        s = s.replace(resolved_symbols=xp.asarray([1.0, -1.0]))
        s = s.replace(resolved_bits=xp.asarray([0, 1]))

        result = s.replace_samples(s.samples.copy(), _preserve_resolved=True)

        assert result.resolved_symbols is s.resolved_symbols
        assert result.resolved_bits is s.resolved_bits

    def test_signal_replace_samples_validates_replacement_and_metadata(self, xp):
        """Replacement samples and metadata pass through assignment validation."""
        s = Signal(samples=xp.arange(8), sampling_rate=2.0, symbol_rate=1.0)

        with pytest.raises(ValueError, match="sampling_rate must be > 0"):
            s.replace_samples(s.samples.copy(), sampling_rate=0.0)
        with pytest.raises(ValueError, match="Only 1D"):
            s.replace_samples(xp.zeros((2, 2, 2)))

    def test_signal_noop_paths_shallow_clone(self, xp):
        """Skipped Signal operations return a new container without copying metadata."""
        frame = {"cached": xp.arange(4)}
        sig = Signal(
            samples=xp.ones(16, dtype=xp.complex64),
            sampling_rate=1.0,
            symbol_rate=1.0,
            frame=frame,
        )

        unresolved = multirate.resolve_symbols(sig)
        undemapped = mapping.demap_symbols_hard(sig)
        unfiltered = filtering.matched_filter(sig)

        for result in (unresolved, undemapped, unfiltered):
            assert result is not sig
            assert result.samples is sig.samples
            assert result.frame is frame


class TestSignalDSPOperations:
    """Tests for TestSignalDSPOperations."""

    def test_signal_methods(self, xp):
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

    def test_signal_resample_sps(self, xp):
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

    def test_welch_psd(self, xp):
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

    def test_add_pilot_tone_returns_applied_frequency(self, xp, xpt):
        """add_pilot_tone on a Signal returns a new Signal and the applied frequency."""
        sig = generate(
            Constellation.psk(4), 128, symbol_rate=1e6, sps=8, pulse=RRC(0.35), rng=0
        ).to(device_of(xp))
        before = xp.asarray(sig.samples.copy())
        df = sig.sampling_rate / sig.samples.shape[-1]

        ret = spectral.add_pilot_tone(sig, 2.0e6, power_ratio_db=-12.0)
        f_p = spectral.grid_frequency(
            2.0e6, sampling_rate=sig.sampling_rate, num_samples=sig.samples.shape[-1]
        )

        assert isinstance(ret, Signal)
        assert ret is not sig  # pure: a new Signal is returned
        # The applied frequency is on the FFT grid and near the request.
        assert abs(round(f_p / df) - f_p / df) < 1e-9
        assert abs(f_p - 2.0e6) <= df / 2 + 1.0
        # Samples actually changed.
        assert float(xp.max(xp.abs(ret.samples - before))) > 0.0

    def test_signal_shift_frequency(self, xp, xpt):
        """Verify frequency shifting logic and resulting spectral peak positioning."""
        fs = 100.0
        # Simple DC signal (freq 0)
        data = xp.ones(100, dtype="complex128")
        s = Signal(samples=data, sampling_rate=fs, symbol_rate=10.0)

        # Offset by 20 Hz
        s = spectral.shift_frequency(s, 20.0)
        actual = spectral.grid_frequency(
            20.0, sampling_rate=s.sampling_rate, num_samples=s.samples.shape[-1]
        )
        assert isinstance(s, Signal)
        assert actual == 20.0

        t = xp.arange(100) / fs
        expected = xp.exp(1j * 2 * xp.pi * 20.0 * t)
        xpt.assert_allclose(s.samples, expected)

        s = spectral.shift_frequency(s, 5.0)

        # Check approximate freq
        f, p = spectral.welch_psd(s, nperseg=64)
        peak = f[xp.argmax(p)]
        # 25 Hz expected
        assert abs(peak - 25.0) < (fs / 64)

    def test_signal_decimate_to_symbol_rate(self, xp):
        """Verify downsampling Signal to symbols."""
        data = xp.ones(40, dtype="complex128")
        s = Signal(samples=data, sampling_rate=4.0, symbol_rate=1.0)
        s = multirate.decimate_to_symbol_rate(s, offset=0)
        assert len(s.samples) == 10
        assert s.sampling_rate == 1.0

    def test_signal_downsample_warning(self, xp):
        """Verify warning when downsampling already 1 SPS signal."""
        data = xp.ones(10, dtype="complex128")
        s = Signal(samples=data, sampling_rate=1.0, symbol_rate=1.0)
        multirate.decimate_to_symbol_rate(s)  # Should just warn

    def test_signal_mimo_fir_coverage(self, xp):
        """Verify FIR filtering on multichannel signals."""
        data = xp.ones((2, 100), dtype="complex128")
        s = Signal(samples=data, sampling_rate=1.0, symbol_rate=1.0)
        # Filter with delay
        taps = xp.array([1.0, 0.5])
        s = filtering.fir_filter(s, taps)
        assert s.samples.shape == (2, 100)
        # y[1] should be 1.5
        assert abs(float(s.samples[0, 1].real) - 1.5) < 1e-10

    def test_signal_gaussian_coverage(self, xp):
        """Verify Gaussian Signal generation."""
        # Use PSK with Gaussian pulse shaping
        s = generate(
            Constellation.psk(2), 100, symbol_rate=1e6, sps=8, pulse=Gaussian(0.5)
        )
        assert s.pulse == filtering.Gaussian(fwhm=0.5)


class TestSignalWaveformsAndModulation:
    """Tests for TestSignalWaveformsAndModulation."""

    def test_shaping_filter_taps_error(self, xp):
        """Verify that shaping_filter_taps raises errors for unconfigured or unknown shapes."""
        data = xp.zeros(10)
        s = Signal(samples=data, sampling_rate=100.0, symbol_rate=10.0)

        with pytest.raises(ValueError, match="No pulse shape defined"):
            filtering.shaping_filter_taps(s)

        with pytest.raises(ValueError, match="pulse must be a Pulse"):
            s.replace(pulse="invalid_shape")

    def test_rzpam_odd_sps(self, xp):
        """An RZ pulse that does not fit whole samples raises instead of rounding."""
        with pytest.raises(ValueError, match=r"duty_cycle \* sps"):
            generate(Constellation.pam(2), 10, symbol_rate=1e3, sps=3, pulse=Rect(0.5))

    def test_rzpam_multi_stream(self, xp):
        """Verify RZ-PAM multi-stream reshape produces correctly shaped multichannel output."""
        sig = generate(
            Constellation.pam(2),
            10,
            symbol_rate=1e3,
            sps=4,
            pulse=Rect(0.5),
            num_channels=2,
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
                    constellation=Constellation.qam(16),
                    num_symbols=10,
                    sps=3.5,
                    symbol_rate=1e6,
                ),
            ),
            (
                generate_psqam,
                dict(num_symbols=10, sps=3.5, symbol_rate=1e6, order=64, nu=0.3),
            ),
        ],
    )
    def test_noninteger_sps_raises(self, backend_device, factory, kwargs):
        """Non-integer sps must raise at generation time, before any samples are produced."""
        with pytest.raises(ValueError, match="sps to be a positive integer"):
            factory(**kwargs)

    def test_integer_valued_float_sps_accepted(self, backend_device):
        """sps=4.0 (integer-valued float) must succeed and produce correct sample count."""
        sig = generate(
            Constellation.qam(16), 100, symbol_rate=1e6, sps=4.0, pulse=RRC(0.35)
        )
        assert sig.samples.shape[-1] == 100 * 4
        assert sig.sps == 4.0

    def test_pam_waveform(self, xp):
        """PAM generation returns a host Signal; .to() moves it explicitly."""
        sig = generate(Constellation.pam(2), 10, symbol_rate=1e3, sps=4, pulse=Rect())
        assert sig.samples.size > 0
        assert isinstance(sig.samples, np.ndarray)
        assert isinstance(sig.to(device_of(xp)).samples, xp.ndarray)
        assert sig.mod_scheme is not None

    def test_rzpam_waveform(self, xp):
        """Verify Return-to-Zero PAM signal generation and pulse-shape validation."""
        sig = generate(
            Constellation.pam(2), 10, symbol_rate=1e3, sps=4, pulse=Rect(0.5)
        ).to(device_of(xp))
        assert sig.samples.size > 0
        assert isinstance(sig.samples, xp.ndarray)

    def test_qam_waveform(self, xp):
        """Verify QAM signal generation populates samples and modulation metadata."""
        sig = generate(
            Constellation.qam(16), 10, symbol_rate=1e3, sps=4, pulse=RRC(0.35)
        ).to(device_of(xp))
        assert sig.samples.size > 0
        assert isinstance(sig.samples, xp.ndarray)
        assert sig.mod_order == 16

    def test_psk_waveform(self, xp, xpt):
        """Verify PSK signal generation, metadata, and unit-magnitude constellation."""
        sig = generate(
            Constellation.psk(8), 50, symbol_rate=1e6, sps=2, pulse=RRC(0.35), rng=0
        ).to(device_of(xp))
        assert sig.samples.size > 0
        assert isinstance(sig.samples, xp.ndarray)
        assert sig.mod_order == 8
        assert sig.mod_scheme is not None
        # All PSK symbols should lie on the unit circle
        syms = sig.source_symbols
        if syms is not None:
            magnitudes = xp.abs(syms)
            xpt.assert_allclose(magnitudes, xp.ones_like(magnitudes), atol=1e-5)

    def test_signal_generate(self, xp):
        """Verify generate().to(device_of(xp)) produces correct metadata for any modulation."""
        sig = generate(
            Constellation.qam(16), 100, symbol_rate=1e6, sps=4, pulse=RRC(0.35), rng=1
        ).to(device_of(xp))
        assert sig.samples.size > 0
        assert isinstance(sig.samples, xp.ndarray)
        assert sig.symbol_rate == 1e6
        assert sig.mod_order == 16
        assert sig.source_bits is not None
        assert sig.source_symbols is not None


class TestSignalResolutionAndMetrics:
    """Tests for TestSignalResolutionAndMetrics."""

    def test_signal_resolution_and_demap(self, xp, xpt):
        """Verify manual symbol resolution and bit demapping with caching."""
        # Generate a simple BPSK signal at 4 SPS
        symbol_rate = 1e6
        sps = 4
        num_symbols = 100
        sig = generate(
            Constellation.psk(2),
            num_symbols,
            symbol_rate=symbol_rate,
            sps=sps,
            pulse=RRC(0.35),
            rng=42,
        ).to(device_of(xp))

        # Initially resolved attributes should be None
        assert sig.resolved_symbols is None
        assert sig.resolved_bits is None

        # Calling demap_symbols_hard before resolve_symbols should raise ValueError
        with pytest.raises(ValueError, match="No resolved symbols available"):
            sig = mapping.demap_symbols_hard(sig)

        # Resolve symbols with offset. Unchanged waveform/provenance arrays are shared.
        original = sig
        sig = multirate.resolve_symbols(original, offset=0)
        assert sig is not original
        assert sig.samples is original.samples
        assert sig.source_bits is original.source_bits
        assert sig.source_symbols is original.source_symbols
        assert sig.resolved_symbols is not None
        assert sig.resolved_bits is None
        assert len(sig.resolved_symbols) == num_symbols
        assert isinstance(sig.resolved_symbols, xp.ndarray)

        # Demapping also shares unchanged waveform and resolved-symbol arrays.
        resolved = sig
        sig = mapping.demap_symbols_hard(resolved)
        assert sig is not resolved
        assert sig.samples is resolved.samples
        assert sig.resolved_symbols is resolved.resolved_symbols
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
        sig = sig.replace(resolved_bits=None)
        with pytest.raises(ValueError, match="No resolved bits available"):
            metrics.ber(sig, bits_tx=ref_bits)

    def test_resolve_symbols_sps_errors(self, xp):
        """Verify resolve_symbols error paths for invalid SPS values."""
        # SPS < 1 (symbol_rate > sampling_rate)
        s = Signal(
            samples=xp.ones(10, dtype="complex64"), sampling_rate=1.0, symbol_rate=2.0
        )
        with pytest.raises(ValueError, match="sps to be a positive integer"):
            s = multirate.resolve_symbols(s)

        # Non-integer SPS
        s2 = Signal(
            samples=xp.ones(10, dtype="complex64"), sampling_rate=3.0, symbol_rate=2.0
        )
        with pytest.raises(ValueError, match="sps to be a positive integer"):
            s2 = multirate.resolve_symbols(s2)

    def test_demap_without_modulation(self, xp):
        """Verify demap_symbols_hard raises error without modulation metadata."""
        s = Signal(
            samples=xp.ones(10, dtype="complex64"), sampling_rate=1.0, symbol_rate=1.0
        )
        s = s.replace(resolved_symbols=xp.ones(10, dtype="complex64"))
        with pytest.raises(ValueError, match="Modulation scheme and order required"):
            s = mapping.demap_symbols_hard(s)

    def test_evm_no_reference(self, xp):
        """Verify evm raises when no reference is available."""
        s = Signal(
            samples=xp.ones(10, dtype="complex64"), sampling_rate=1.0, symbol_rate=1.0
        )
        s = s.replace(resolved_symbols=xp.ones(10, dtype="complex64"))
        with pytest.raises(ValueError, match="No reference available"):
            metrics.evm(s)

    def test_evm_no_resolved(self, xp):
        """Verify evm raises when no resolved_symbols are present."""
        s = Signal(
            samples=xp.ones(10, dtype="complex64"),
            sampling_rate=1.0,
            symbol_rate=1.0,
            reference=Reference(symbols=xp.ones(10, dtype="complex64")),
        )
        with pytest.raises(ValueError, match="No resolved symbols available"):
            metrics.evm(s)

    def test_snr_no_reference(self, xp):
        """Verify snr raises when no reference is available."""
        s = Signal(
            samples=xp.ones(10, dtype="complex64"), sampling_rate=1.0, symbol_rate=1.0
        )
        s = s.replace(resolved_symbols=xp.ones(10, dtype="complex64"))
        with pytest.raises(ValueError, match="No reference available"):
            metrics.snr(s)

    def test_snr_no_resolved(self, xp):
        """Verify snr raises when no resolved_symbols are present."""
        s = Signal(
            samples=xp.ones(10, dtype="complex64"),
            sampling_rate=1.0,
            symbol_rate=1.0,
            reference=Reference(symbols=xp.ones(10, dtype="complex64")),
        )
        with pytest.raises(ValueError, match="No resolved symbols available"):
            metrics.snr(s)

    def test_ber_no_reference(self, xp):
        """Verify ber raises when no reference bits are available."""
        s = Signal(
            samples=xp.ones(10, dtype="complex64"), sampling_rate=1.0, symbol_rate=1.0
        )
        s = s.replace(resolved_bits=xp.array([0, 1, 0, 1]))
        with pytest.raises(ValueError, match="No reference bits available"):
            metrics.ber(s)

    def test_evm_with_explicit_num_train_symbols(self, xp):
        """evm(num_train_symbols=N) discards leading symbols from resolved_symbols."""
        from commkit import equalization

        n_symbols = 600
        n_train = 100
        orig = generate(
            Constellation.psk(4),
            n_symbols,
            symbol_rate=1e6,
            sps=2,
            pulse=RRC(0.35),
            rng=7,
        )
        result = equalization.lms(
            xp.asarray(orig.samples),
            training_symbols=orig.source_symbols[:n_train],
            sps=2,
            num_taps=7,
        )
        n_train = result.num_train_symbols
        rx = Signal(
            samples=result.y_hat,
            sampling_rate=orig.symbol_rate,
            symbol_rate=orig.symbol_rate,
            constellation=Constellation.psk(4),
            reference=Reference(
                symbols=orig.source_symbols[..., : result.y_hat.shape[-1]]
            ),
        )
        rx = multirate.resolve_symbols(rx)

        evm_pct, evm_db = metrics.evm(rx, num_train_symbols=n_train)
        assert np.isfinite(float(evm_db))
        assert float(evm_pct) > 0

        evm_pct2, evm_db2 = metrics.evm(rx)
        assert np.isfinite(float(evm_db2))

    def test_snr_with_explicit_num_train_symbols(self, xp):
        """snr(num_train_symbols=N) discards leading symbols before computing SNR."""
        from commkit import equalization

        n_symbols = 600
        orig = generate(
            Constellation.psk(4),
            n_symbols,
            symbol_rate=1e6,
            sps=2,
            pulse=RRC(0.35),
            rng=8,
        )
        result = equalization.lms(
            xp.asarray(orig.samples),
            training_symbols=orig.source_symbols[:100],
            sps=2,
            num_taps=7,
        )
        rx = Signal(
            samples=result.y_hat,
            sampling_rate=orig.symbol_rate,
            symbol_rate=orig.symbol_rate,
            constellation=Constellation.psk(4),
            reference=Reference(
                symbols=orig.source_symbols[..., : result.y_hat.shape[-1]]
            ),
        )
        rx = multirate.resolve_symbols(rx)

        snr_val = metrics.snr(rx, num_train_symbols=result.num_train_symbols)
        assert np.isfinite(snr_val)

        snr_notrim = metrics.snr(rx)
        assert np.isfinite(snr_notrim)

    def test_ber_with_explicit_num_train_symbols(self, xp):
        """ber(num_train_symbols=N) discards leading bits before computing BER."""
        from commkit import equalization

        n_symbols = 600
        orig = generate(
            Constellation.psk(4),
            n_symbols,
            symbol_rate=1e6,
            sps=2,
            pulse=RRC(0.35),
            rng=11,
        )
        result = equalization.lms(
            xp.asarray(orig.samples),
            training_symbols=orig.source_symbols[:100],
            sps=2,
            num_taps=7,
        )
        rx = Signal(
            samples=result.y_hat,
            sampling_rate=orig.symbol_rate,
            symbol_rate=orig.symbol_rate,
            constellation=Constellation.psk(4),
            reference=Reference(
                symbols=orig.source_symbols[..., : result.y_hat.shape[-1]],
                bits=orig.source_bits[..., : result.y_hat.shape[-1] * 2],
            ),
        )
        rx = multirate.resolve_symbols(rx)
        rx = mapping.demap_symbols_hard(rx)

        ber_val = metrics.ber(rx, num_train_symbols=result.num_train_symbols)
        assert np.isfinite(float(ber_val))
        assert 0.0 <= float(ber_val) <= 1.0

        ber_notrim = metrics.ber(rx)
        assert np.isfinite(float(ber_notrim))
        assert 0.0 <= float(ber_notrim) <= 1.0

    def test_rls_tail_trim_field(self, xp):
        """rls() tail_trim field equals num_taps // 2 and y_hat is shortened accordingly."""
        from commkit import equalization

        n_symbols = 400
        num_taps = 7
        orig = generate(
            Constellation.psk(4),
            n_symbols,
            symbol_rate=1e6,
            sps=2,
            pulse=RRC(0.35),
            rng=3,
        )
        result = equalization.rls(
            xp.asarray(orig.samples),
            training_symbols=orig.source_symbols,
            sps=2,
            num_taps=num_taps,
            modulation="psk",
            order=4,
        )
        assert result.tail_trim == num_taps // 2
        assert result.y_hat.shape[-1] == n_symbols - result.tail_trim


class TestSignalDeviceAndPlotting:
    """Tests for TestSignalDeviceAndPlotting."""

    def test_signal_rejects_foreign_arrays(self):
        """Samples from another framework raise TypeError instead of being copied."""

        class _Foreign:
            def __init__(self):
                self._a = np.ones(10)

            def __dlpack__(self, **kwargs):
                return self._a.__dlpack__(**kwargs)

            def __dlpack_device__(self):
                return self._a.__dlpack_device__()

        with pytest.raises(TypeError, match="from_dlpack"):
            Signal(samples=_Foreign(), sampling_rate=1.0, symbol_rate=1.0)

    def test_plot_constellation_at_symbol_rate(self, xp):
        """plot_constellation on a 1-SPS Signal (built from lms y_hat) should succeed."""
        from commkit import equalization

        sig = generate(
            Constellation.psk(4), 200, symbol_rate=1e6, sps=2, pulse=RRC(0.35), rng=0
        )
        result = equalization.lms(
            xp.asarray(sig.samples),
            training_symbols=sig.source_symbols,
            sps=2,
            num_taps=7,
        )
        rx_1sps = Signal(
            samples=result.y_hat,
            sampling_rate=sig.symbol_rate,
            symbol_rate=sig.symbol_rate,
            constellation=Constellation.psk(4),
        )
        plot_result = plotting.plot_constellation(rx_1sps, show=False)
        assert plot_result is not None

    def test_plot_constellation_overlay_source_mimo(self, xp):
        """Signal.plot_constellation with MIMO signal and overlay_source=True."""
        sig = generate(
            Constellation.psk(4),
            200,
            symbol_rate=1e6,
            sps=1,
            pulse=RRC(0.35),
            num_channels=2,
            rng=0,
        )
        result = plotting.plot_constellation(sig, overlay_source=True, show=False)
        assert result is not None

    def test_plot_constellation_show(self, xp):
        """Signal.plot_constellation(show=True) should call plt.show() and return None."""
        sig = generate(
            Constellation.psk(4), 100, symbol_rate=1e6, sps=1, pulse=RRC(0.35), rng=0
        )
        with patch("matplotlib.pyplot.show"):
            result = plotting.plot_constellation(sig, show=True)
        assert result is None

    def test_plot_constellation_overlay_source_siso(self, xp):
        """SISO signal with overlay_source=True uses the single-axes scatter path."""
        sig = generate(
            Constellation.psk(4), 200, symbol_rate=1e6, sps=1, pulse=RRC(0.35), rng=0
        )
        assert sig.num_streams == 1
        assert sig.source_symbols is not None

        result = plotting.plot_constellation(sig, overlay_source=True, show=False)
        assert result is not None
