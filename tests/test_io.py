"""Tests for commkit.io - save_npz / load_npz round-trips."""

from typing import Any

import numpy as np
import pytest

import commkit
from commkit import (
    Preamble,
    Signal,
    SingleCarrierFrame,
    filtering,
    generate_psqam,
    generate_qam,
    mapping,
    metrics,
    multirate,
)
from commkit.io import load_npz, save_npz
from tests.common.conversions import to_numpy
from tests.common.signals import (
    make_test_frame_signal,
    make_test_mimo_signal,
    make_test_qam_signal,
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _siso_signal() -> Signal:
    return make_test_qam_signal(
        num_symbols=256, sps=4, symbol_rate=1e9, order=16, snr_db=None, seed=0
    )


def _mimo_signal() -> Signal:
    return make_test_mimo_signal(
        num_channels=2, num_symbols=128, sps=4, symbol_rate=1e9, order=4, seed=1
    )


def _frame_signal() -> Signal:
    return make_test_frame_signal(
        payload_len=200, preamble_len=13, sps=4, symbol_rate=1e9, payload_mod_order=16
    )


# -----------------------------------------------------------------------------
# Test Classes
# -----------------------------------------------------------------------------


class TestNPZSaveLoadSISO:
    """Tests for saving and loading SISO signals and their metadata."""

    def test_roundtrip_samples(self, tmp_path: Any, xpt: Any) -> None:
        """SISO samples round-trip with bit-exact preservation."""
        sig = _siso_signal()
        p = tmp_path / "sig.npz"
        save_npz(sig, p)
        sig2 = load_npz(p)
        xpt.assert_array_equal(to_numpy(sig.samples), to_numpy(sig2.samples))

    def test_roundtrip_scalar_metadata(self, tmp_path: Any) -> None:
        """Physical and modulation scalar metadata round-trip accurately."""
        sig = _siso_signal()
        p = tmp_path / "sig.npz"
        save_npz(sig, p)
        sig2 = load_npz(p)

        assert sig2.sampling_rate == sig.sampling_rate
        assert sig2.symbol_rate == sig.symbol_rate
        assert sig2.mod_scheme == sig.mod_scheme
        assert sig2.mod_order == sig.mod_order
        assert sig2.mod_unipolar == sig.mod_unipolar
        assert sig2.pulse_shape == sig.pulse_shape
        assert sig2.rrc_rolloff == sig.rrc_rolloff
        assert sig2.filter_span == sig.filter_span
        assert sig2.spectral_domain == sig.spectral_domain
        assert sig2.physical_domain == sig.physical_domain
        assert sig2.center_frequency == sig.center_frequency
        assert sig2.digital_frequency_offset == sig.digital_frequency_offset
        assert sig2.pilot_tone_frequency == sig.pilot_tone_frequency

    def test_roundtrip_pilot_tone_frequency(self, tmp_path: Any, xpt: Any) -> None:
        """pilot_tone_frequency round-trips: None when absent, 1-D array when set."""
        sig = _siso_signal()
        assert sig.pilot_tone_frequency is None

        sig.pilot_tone_frequency = 2.5e9
        assert isinstance(sig.pilot_tone_frequency, np.ndarray)
        p = tmp_path / "tone.npz"
        save_npz(sig, p)
        sig2 = load_npz(p)
        assert isinstance(sig2.pilot_tone_frequency, np.ndarray)
        xpt.assert_array_equal(sig2.pilot_tone_frequency, [2.5e9])

    def test_roundtrip_pilot_tone_power_ratio_db(self, tmp_path: Any, xpt: Any) -> None:
        """pilot_tone_power_ratio_db round-trips correctly."""
        sig = _siso_signal()
        assert sig.pilot_tone_power_ratio_db is None

        sig.pilot_tone_power_ratio_db = -12.0
        p = tmp_path / "psr.npz"
        save_npz(sig, p)
        sig2 = load_npz(p)
        xpt.assert_array_equal(sig2.pilot_tone_power_ratio_db, [-12.0])

    def test_roundtrip_source_bits(self, tmp_path: Any, xpt: Any) -> None:
        """Source bits round-trip identically."""
        sig = _siso_signal()
        assert sig.source_bits is not None
        p = tmp_path / "sig.npz"
        save_npz(sig, p)
        sig2 = load_npz(p)
        xpt.assert_array_equal(to_numpy(sig.source_bits), to_numpy(sig2.source_bits))

    def test_roundtrip_source_symbols(self, tmp_path: Any, xpt: Any) -> None:
        """Source symbols round-trip identically."""
        sig = _siso_signal()
        assert sig.source_symbols is not None
        p = tmp_path / "sig.npz"
        save_npz(sig, p)
        sig2 = load_npz(p)
        xpt.assert_allclose(
            to_numpy(sig.source_symbols), to_numpy(sig2.source_symbols), atol=1e-7
        )

    def test_extension_appended_automatically(self, tmp_path: Any, xpt: Any) -> None:
        """Saving/loading without extension appends .npz automatically."""
        sig = _siso_signal()
        p_no_ext = tmp_path / "capture"
        save_npz(sig, p_no_ext)
        assert (tmp_path / "capture.npz").exists()

        sig2 = load_npz(p_no_ext)
        xpt.assert_array_equal(to_numpy(sig.samples), to_numpy(sig2.samples))

    def test_roundtrip_signal_type_none(self, tmp_path: Any) -> None:
        """Signal without signal_type should load with signal_type=None."""
        sig = _siso_signal()
        assert sig.signal_type is None
        p = tmp_path / "plain.npz"
        save_npz(sig, p)
        sig2 = load_npz(p)
        assert sig2.signal_type is None

    def test_no_source_arrays_when_none(self, tmp_path: Any) -> None:
        """Signal with no source_bits/source_symbols should load without them."""
        sig = Signal(
            samples=np.random.randn(512) + 1j * np.random.randn(512),
            sampling_rate=1e9,
            symbol_rate=250e6,
        )
        assert sig.source_bits is None
        assert sig.source_symbols is None

        p = tmp_path / "raw.npz"
        save_npz(sig, p)
        sig2 = load_npz(p)
        assert sig2.source_bits is None
        assert sig2.source_symbols is None


class TestNPZSaveLoadMIMO:
    """Tests for saving and loading MIMO signals."""

    def test_roundtrip_mimo(self, tmp_path: Any, xpt: Any) -> None:
        """MIMO 2D sample array shape and values round-trip."""
        sig = _mimo_signal()
        assert sig.samples.ndim == 2
        p = tmp_path / "mimo.npz"
        save_npz(sig, p)
        sig2 = load_npz(p)
        assert sig2.samples.shape == sig.samples.shape
        xpt.assert_array_equal(to_numpy(sig.samples), to_numpy(sig2.samples))

    def test_roundtrip_pilot_tone_frequency_per_channel(
        self, tmp_path: Any, xpt: Any
    ) -> None:
        """Per-channel pilot frequencies round-trip as an array."""
        sig = _mimo_signal()
        sig.pilot_tone_frequency = [2.5e9, -3.0e9]
        assert isinstance(sig.pilot_tone_frequency, np.ndarray)
        p = tmp_path / "tones.npz"
        save_npz(sig, p)
        sig2 = load_npz(p)
        assert isinstance(sig2.pilot_tone_frequency, np.ndarray)
        xpt.assert_array_equal(sig2.pilot_tone_frequency, [2.5e9, -3.0e9])

    def test_roundtrip_pilot_tone_power_ratio_db_mimo(
        self, tmp_path: Any, xpt: Any
    ) -> None:
        """Per-channel pilot power ratios round-trip."""
        mimo = _mimo_signal()
        mimo.pilot_tone_power_ratio_db = [-10.0, -8.0]
        assert isinstance(mimo.pilot_tone_power_ratio_db, np.ndarray)
        save_npz(mimo, tmp_path / "psr_mimo.npz")
        mimo2 = load_npz(tmp_path / "psr_mimo.npz")
        xpt.assert_array_equal(mimo2.pilot_tone_power_ratio_db, [-10.0, -8.0])


class TestNPZSaveLoadFrame:
    """Tests for saving and loading frame signals and structural metadata."""

    def test_roundtrip_frame_metadata(self, tmp_path: Any) -> None:
        """Single-Carrier Frame geometry and slots survive round-trip."""
        sig = _frame_signal()
        assert sig.signal_type == "Single-Carrier Frame"
        assert sig.frame is not None

        p = tmp_path / "frame.npz"
        save_npz(sig, p)
        sig2 = load_npz(p)

        assert sig2.signal_type == "Single-Carrier Frame"
        assert sig2.frame is not None
        assert sig2.frame.payload_len == sig.frame.payload_len
        assert sig2.frame.payload_mod_scheme == sig.frame.payload_mod_scheme
        assert sig2.frame.payload_mod_order == sig.frame.payload_mod_order

    def test_roundtrip_zc_preamble_kwargs(self, tmp_path: Any) -> None:
        """Zadoff-Chu preamble root and length kwargs round-trip."""
        frame = SingleCarrierFrame(
            payload_len=100,
            preamble=Preamble(sequence_type="zc", length=31, root=7),
        )
        sig = frame.to_signal(sps=4, symbol_rate=1e9)
        assert sig.frame.preamble.root == 7

        p = tmp_path / "zc.npz"
        save_npz(sig, p)
        sig2 = load_npz(p)
        assert sig2.frame.preamble.root == 7


class TestNPZNoPickle:
    """Archives hold only numeric and unicode arrays and load without pickle."""

    def test_archive_has_no_object_arrays(self, tmp_path: Any) -> None:
        save_npz(_siso_signal(), tmp_path / "sig.npz")
        with np.load(tmp_path / "sig.npz", allow_pickle=False) as data:
            for key in data.files:
                assert data[key].dtype != object, key

    def test_frame_archive_has_no_object_arrays(self, tmp_path: Any) -> None:
        save_npz(make_test_frame_signal(), tmp_path / "frame.npz")
        with np.load(tmp_path / "frame.npz", allow_pickle=False) as data:
            assert "__frame_metadata__" in data.files
            for key in data.files:
                assert data[key].dtype != object, key

    def test_pickled_metadata_is_rejected_without_unpickling(
        self, tmp_path: Any
    ) -> None:
        """An object array (old YAML format, or a crafted file) is never unpickled."""

        class _Payload:
            def __reduce__(self):
                return (_mark_unpickled, ())

        np.savez(
            tmp_path / "evil.npz",
            samples=np.zeros(8, np.complex64),
            __metadata__=np.array(_Payload(), dtype=object),
        )
        _UNPICKLED.clear()
        with pytest.raises(ValueError, match="not JSON"):
            load_npz(tmp_path / "evil.npz")
        assert not _UNPICKLED


_UNPICKLED: list[bool] = []


def _mark_unpickled() -> bool:
    _UNPICKLED.append(True)
    return True


class TestNPZCompressionAndCaches:
    """Tests for compression options and symbol/bit cache preservation."""

    def test_include_cache_false_by_default(self, tmp_path: Any) -> None:
        """Resolved caches are omitted from archive by default."""
        sig = _siso_signal()
        sig = multirate.resolve_symbols(sig)
        assert sig.resolved_symbols is not None

        p = tmp_path / "sig.npz"
        save_npz(sig, p)

        data = np.load(p, allow_pickle=False)
        assert "resolved_symbols" not in data.files
        assert "resolved_bits" not in data.files

    def test_include_cache_roundtrip(self, tmp_path: Any, xpt: Any) -> None:
        """Resolved caches round-trip when include_cache=True."""
        sig = _siso_signal()
        sig = multirate.resolve_symbols(sig)
        assert sig.resolved_symbols is not None

        p = tmp_path / "sig_cache.npz"
        save_npz(sig, p, include_cache=True)
        sig2 = load_npz(p)

        xpt.assert_allclose(
            to_numpy(sig.resolved_symbols), to_numpy(sig2.resolved_symbols), atol=1e-7
        )

    def test_uncompressed_roundtrip(self, tmp_path: Any, xpt: Any) -> None:
        """Uncompressed archive round-trips identically."""
        sig = _siso_signal()
        p = tmp_path / "uncompressed.npz"
        save_npz(sig, p, compressed=False)
        sig2 = load_npz(p)
        xpt.assert_array_equal(to_numpy(sig.samples), to_numpy(sig2.samples))

    def test_compressed_smaller_than_uncompressed(self, tmp_path: Any) -> None:
        """Compressed archive is no larger than uncompressed archive."""
        sig = _siso_signal()
        p_c = tmp_path / "c.npz"
        p_u = tmp_path / "u.npz"
        save_npz(sig, p_c, compressed=True)
        save_npz(sig, p_u, compressed=False)
        assert p_c.stat().st_size <= p_u.stat().st_size


class TestNPZDeviceHandling:
    """Tests for device-aware signal loading and backend residency."""

    def test_exported_from_package(self) -> None:
        """Package root exports save_npz and load_npz."""
        assert hasattr(commkit, "save_npz")
        assert hasattr(commkit, "load_npz")

    @pytest.mark.gpu_only
    def test_roundtrip_device_gpu(self, tmp_path: Any, xpt: Any) -> None:
        """GPU signal round-trips to GPU device placement."""
        sig = _siso_signal()
        p = tmp_path / "sig_gpu.npz"
        save_npz(sig, p)
        sig_gpu = load_npz(p, device="gpu")
        assert sig_gpu.backend == "GPU"
        xpt.assert_array_equal(to_numpy(sig.samples), to_numpy(sig_gpu.samples))

    @pytest.mark.gpu_only
    def test_auto_device_uses_gpu_when_available(
        self, backend_device: str, tmp_path: Any
    ) -> None:
        """device='auto' chooses GPU if CuPy is available."""
        sig = _siso_signal()
        p = tmp_path / "auto.npz"
        save_npz(sig, p)
        sig2 = load_npz(p)
        assert sig2.backend == "GPU"

    def test_psqam_pmf_roundtrip(self, tmp_path: Any, xpt: Any) -> None:
        """PS-QAM signal PMF, mod_scheme, and order round-trip."""
        sig = generate_psqam(1000, sps=4, symbol_rate=10e9, order=64, entropy=5.0)
        save_npz(sig, tmp_path / "psqam")
        loaded = load_npz(tmp_path / "psqam.npz", device="cpu")
        assert loaded.ps_pmf is not None
        xpt.assert_allclose(to_numpy(loaded.ps_pmf), to_numpy(sig.ps_pmf), rtol=1e-6)
        assert loaded.mod_scheme == "PS-QAM"
        assert loaded.mod_order == 64

    def test_free_function_pipeline_metrics_survive_roundtrip(
        self, tmp_path: Any, xpt: Any
    ) -> None:
        """Signal processed via free functions preserves reproducible metrics."""
        sig = generate_qam(num_symbols=2000, sps=4, symbol_rate=10e9, order=16, seed=7)
        sig = filtering.matched_filter(sig)
        sig = multirate.resolve_symbols(sig)
        sig = mapping.demap_symbols_hard(sig)

        evm_before = metrics.evm(sig)
        ber_before = metrics.ber(sig)

        p = tmp_path / "pipeline.npz"
        save_npz(sig, p, include_cache=True)
        loaded = load_npz(p, device="cpu")

        xpt.assert_allclose(
            to_numpy(sig.resolved_symbols), to_numpy(loaded.resolved_symbols), atol=1e-7
        )
        xpt.assert_array_equal(
            to_numpy(sig.resolved_bits), to_numpy(loaded.resolved_bits)
        )
        xpt.assert_array_equal(to_numpy(sig.source_bits), to_numpy(loaded.source_bits))

        assert metrics.evm(loaded) is not None
        xpt.assert_allclose(metrics.evm(loaded)[0], evm_before[0], rtol=1e-5)
        xpt.assert_allclose(float(metrics.ber(loaded)), float(ber_before), rtol=1e-9)
