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
    generate,
    mapping,
    metrics,
    multirate,
)
from commkit.filtering import RRC
from commkit.io import load_npz, save_npz
from commkit.mapping import Constellation
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
        payload_len=203, preamble_len=13, sps=4, symbol_rate=1e9, payload_mod_order=16
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
        assert sig2.center_frequency == sig.center_frequency
        assert sig2.constellation == sig.constellation
        assert sig2.constellation.family == sig.constellation.family

    @pytest.mark.parametrize(
        "pulse",
        [
            filtering.RRC(0.2, span=6),
            filtering.RC(0.5),
            filtering.Gaussian(0.7, span=4),
            filtering.Rect(0.5, 0.1),
            filtering.SmoothRect(0.3, 0.5, 8),
            None,
        ],
    )
    def test_roundtrip_pulse(self, tmp_path: Any, pulse: Any) -> None:
        sig = _siso_signal().replace(pulse=pulse)
        save_npz(sig, tmp_path / "pulse.npz")
        assert load_npz(tmp_path / "pulse.npz").pulse == pulse

    def test_roundtrip_custom_constellation(self, tmp_path: Any) -> None:
        c = mapping.Constellation(
            [-3.0, -1.0, 1.0, 3.0], bit_labels=[[0, 0], [0, 1], [1, 1], [1, 0]]
        ).shaped(nu=0.1)
        sig = _siso_signal().replace(constellation=c)
        save_npz(sig, tmp_path / "custom.npz")
        loaded = load_npz(tmp_path / "custom.npz").constellation
        assert loaded == c
        assert loaded.family is None

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


class TestNPZSaveLoadFrame:
    """Tests for saving and loading frame signals and structural metadata."""

    def test_roundtrip_frame_metadata(self, tmp_path: Any) -> None:
        """Single-Carrier Frame geometry and slots survive round-trip."""
        sig = _frame_signal()
        assert sig.frame is not None

        p = tmp_path / "frame.npz"
        save_npz(sig, p)
        sig2 = load_npz(p)

        assert sig2.frame is not None
        assert sig2.frame.payload_len == sig.frame.payload_len
        assert sig2.frame == sig.frame

    def test_roundtrip_zc_preamble_kwargs(self, tmp_path: Any) -> None:
        """Zadoff-Chu preamble root and length kwargs round-trip."""
        frame = SingleCarrierFrame(
            payload_len=100,
            preamble=Preamble(sequence_type="zc", length=31, root=7),
        )
        sig = frame.to_signal(sps=4, symbol_rate=1e9, pulse=RRC(0.35))
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

    def test_default_device_is_cpu(self, tmp_path: Any) -> None:
        """load_npz loads to the CPU unless a device is requested, GPU or not."""
        p = tmp_path / "default.npz"
        save_npz(_siso_signal(), p)
        assert load_npz(p).backend == "CPU"

    def test_psqam_pmf_roundtrip(self, tmp_path: Any, xpt: Any) -> None:
        """PS-QAM signal PMF, mod_scheme, and order round-trip."""
        sig = generate(
            Constellation.qam(64).shaped(entropy=5.0),
            1000,
            symbol_rate=10e9,
            sps=4,
            pulse=RRC(0.35),
        )
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
        sig = generate(
            Constellation.qam(16), 2000, symbol_rate=10e9, sps=4, pulse=RRC(0.35), rng=7
        )
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


def test_frame_constellations_roundtrip(tmp_path: Any) -> None:
    frame = SingleCarrierFrame(
        payload_len=21,
        payload_constellation=mapping.Constellation.qam(64).shaped(nu=0.05),
        pilot_pattern="comb",
        pilot_period=4,
        pilot_constellation=mapping.Constellation.psk(8),
    )
    sig = frame.to_signal(sps=2, symbol_rate=1e6, pulse=RRC(0.35))
    save_npz(sig, tmp_path / "f.npz")
    loaded = load_npz(tmp_path / "f.npz")
    assert loaded.frame == frame
    np.testing.assert_array_equal(loaded.frame.payload_symbols, frame.payload_symbols)
