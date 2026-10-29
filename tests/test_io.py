"""Tests for commkit.io - save_npz / load_npz round-trips."""

import json
from typing import Any

import numpy as np
import pytest

import commkit
from commkit import (
    Preamble,
    Signal,
    SingleCarrierFrame,
    coding,
    filtering,
    generate,
    mapping,
    metrics,
    multirate,
)
from commkit.coding import LinearBlockCode
from commkit.core import extract_payload
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
        payload_num_symbols=203,
        preamble_num_symbols=13,
        sps=4,
        symbol_rate=1e9,
        payload_mod_order=16,
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
        assert sig2.domain == sig.domain == "DIG"
        assert sig2.constellation == sig.constellation
        assert sig2.constellation.family == sig.constellation.family

    @pytest.mark.parametrize("domain", ["RF", "OPT"])
    def test_roundtrip_domain(self, tmp_path: Any, domain: str) -> None:
        """The domain and the carrier round-trip."""
        sig = _siso_signal().replace(domain=domain, center_frequency=193.1e12)
        p = tmp_path / "sig.npz"
        save_npz(sig, p)
        sig2 = load_npz(p)
        assert (sig2.domain, sig2.center_frequency) == (domain, 193.1e12)

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
        assert sig.reference.bits is not None
        p = tmp_path / "sig.npz"
        save_npz(sig, p)
        sig2 = load_npz(p)
        xpt.assert_array_equal(
            to_numpy(sig.reference.bits), to_numpy(sig2.reference.bits)
        )

    def test_roundtrip_source_symbols(self, tmp_path: Any, xpt: Any) -> None:
        """Source symbols round-trip identically."""
        sig = _siso_signal()
        assert sig.reference.symbols is not None
        p = tmp_path / "sig.npz"
        save_npz(sig, p)
        sig2 = load_npz(p)
        xpt.assert_allclose(
            to_numpy(sig.reference.symbols), to_numpy(sig2.reference.symbols), atol=1e-7
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
        """A Signal without a reference loads without one."""
        sig = Signal(
            samples=np.random.randn(512) + 1j * np.random.randn(512),
            sampling_rate=1e9,
            symbol_rate=250e6,
        )
        assert sig.reference is None

        p = tmp_path / "raw.npz"
        save_npz(sig, p)
        sig2 = load_npz(p)
        assert sig2.reference is None


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
        assert sig2.frame.payload_num_symbols == sig.frame.payload_num_symbols
        assert sig2.frame == sig.frame

    def test_roundtrip_zc_preamble_kwargs(self, tmp_path: Any) -> None:
        """Zadoff-Chu preamble root and length kwargs round-trip."""
        frame = SingleCarrierFrame(
            payload_num_symbols=100,
            preamble=Preamble(sequence_type="zc", num_symbols=31, root=7),
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
    """Tests for compression options and 1.x cache entries."""

    def test_1x_cache_entries_are_ignored(self, tmp_path: Any, xpt: Any) -> None:
        """1.x archives with resolved_symbols/resolved_bits still load; the
        derived caches are dropped."""
        sig = _siso_signal()
        p = tmp_path / "sig.npz"
        save_npz(sig, p)
        with np.load(p, allow_pickle=False) as data:
            entries = dict(data)
        entries["resolved_symbols"] = entries["samples"][::2]
        entries["resolved_bits"] = np.zeros(4, dtype=np.int8)
        np.savez(p, **entries)

        loaded = load_npz(p)
        assert not hasattr(loaded, "resolved_symbols")
        xpt.assert_array_equal(to_numpy(loaded.samples), to_numpy(sig.samples))

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

    def test_invalid_device_raises_before_loading(self, tmp_path: Any) -> None:
        with pytest.raises(ValueError, match="device"):
            load_npz(tmp_path / "missing.npz", device="tpu")

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
        assert loaded.constellation.pmf is not None
        xpt.assert_allclose(loaded.constellation.pmf, sig.constellation.pmf, rtol=1e-6)
        assert loaded.constellation.family == "qam"
        assert loaded.constellation.order == 64

    def test_free_function_pipeline_metrics_survive_roundtrip(
        self, tmp_path: Any, xpt: Any
    ) -> None:
        """Signal processed via free functions preserves reproducible metrics."""
        sig = generate(
            Constellation.qam(16), 2000, symbol_rate=10e9, sps=4, pulse=RRC(0.35), rng=7
        )
        sig = filtering.matched_filter(sig)
        sig = multirate.decimate_to_symbol_rate(sig)

        evm_before = metrics.evm(sig)
        ber_before = metrics.ber(sig)

        p = tmp_path / "pipeline.npz"
        save_npz(sig, p)
        loaded = load_npz(p, device="cpu")

        xpt.assert_array_equal(
            to_numpy(sig.reference.bits), to_numpy(loaded.reference.bits)
        )
        xpt.assert_allclose(metrics.evm(loaded), evm_before, rtol=1e-5)
        assert metrics.ber(loaded) == ber_before


def _assert_same_signal(a: Signal, b: Signal) -> None:
    """Every field of ``b`` equals ``a``'s, the frame's generated data included."""
    np.testing.assert_array_equal(to_numpy(a.samples), to_numpy(b.samples))
    assert a.samples.dtype == b.samples.dtype
    for name in (
        "sampling_rate", "symbol_rate", "center_frequency", "domain",
        "constellation", "pulse",
    ):  # fmt: skip
        assert getattr(a, name) == getattr(b, name), name
    for ra, rb in [(a.reference, b.reference)] + (
        [(a.frame._payload_reference(), b.frame._payload_reference())]
        if a.frame is not None
        else []
    ):
        assert (ra is None) == (rb is None)
        if ra is None:
            continue
        for name in ("symbols", "bits", "info_bits"):
            va, vb = getattr(ra, name), getattr(rb, name)
            assert (va is None) == (vb is None), name
            if va is not None:
                np.testing.assert_array_equal(to_numpy(va), to_numpy(vb))
                assert va.dtype == vb.dtype, name
        assert (ra.info_bits is ra.bits) == (rb.info_bits is rb.bits)
        assert ra.code_block == rb.code_block
        assert ra.code == rb.code
    assert a.frame == b.frame
    if a.frame is not None:
        for name in ("pilot_symbols", "pilot_bits", "body_symbols"):
            va, vb = getattr(a.frame, name), getattr(b.frame, name)
            assert (va is None) == (vb is None), name
            if va is not None:
                np.testing.assert_array_equal(va, vb)


_SHAPED = Constellation.qam(16).shaped(entropy=3.5)


@pytest.mark.parametrize(
    "make",
    [
        lambda: generate(Constellation.qam(16), 70, symbol_rate=1e9, rng=0),
        lambda: generate(
            Constellation.qam(16),
            70,
            symbol_rate=1e9,
            num_channels=2,
            code=LinearBlockCode.hamming(3),
            rng=0,
        ),
        lambda: generate(
            Constellation.qam(16),
            symbol_rate=1e9,
            num_blocks=2,
            code=coding.LDPCCode.dvbs2("1/2", "short"),
            rng=0,
        ),
        lambda: generate(_SHAPED, 400, symbol_rate=1e9, rng=0, sps=2, pulse=RRC(0.1)),
        lambda: generate(
            Constellation.qam(16),
            10,
            symbol_rate=1e9,
            code=LinearBlockCode.hamming(3),
            pad=True,
            rng=0,
        ).replace(domain="OPT", center_frequency=193.1e12),
        lambda: SingleCarrierFrame(
            payload_num_symbols=402,
            payload_constellation=_SHAPED,
            preamble=Preamble(num_symbols=13),
            pilot_pattern="block",
            pilot_period=8,
            pilot_block_num_symbols=2,
            guard_type="cp",
            guard_num_symbols=4,
            num_streams=2,
        ).to_signal(sps=2, symbol_rate=1e9, pulse=RRC(0.2)),
        lambda: SingleCarrierFrame(
            payload_blocks=12,
            payload_code=LinearBlockCode.hamming(3),
            pilot_pattern="comb",
            pilot_period=4,
        ).to_signal(sps=1, symbol_rate=1e9),
        lambda: SingleCarrierFrame(
            payload_num_symbols=21,
            payload_code=LinearBlockCode.hamming(3),
            payload_pad=True,
        ).to_signal(sps=1, symbol_rate=1e9),
    ],
    ids=[
        "plain",
        "coded-mimo",
        "ldpc",
        "shaped",
        "padded-optical",
        "shaped-frame",
        "coded-frame",
        "padded-frame",
    ],
)
def test_everything_is_recovered(tmp_path: Any, make) -> None:
    """A loaded Signal equals the saved one in every field and generated array."""
    sig = make()
    save_npz(sig, tmp_path / "s")
    _assert_same_signal(sig, load_npz(tmp_path / "s"))


def test_extracted_payload_decodes_after_load(tmp_path: Any) -> None:
    """The stored reference code decodes the loaded payload."""
    frame = SingleCarrierFrame(
        payload_blocks=4,
        payload_code=coding.LDPCCode.dvbs2("1/2", "short"),
        payload_constellation=Constellation.qam(16),
    )
    save_npz(extract_payload(frame.to_signal(sps=1, symbol_rate=1e9)), tmp_path / "p")
    ref = load_npz(tmp_path / "p").reference
    assert ref.code == frame.payload_code
    res = coding.decode(ref.bits, code=ref.code)
    np.testing.assert_array_equal(res.bits, ref.info_bits)


def test_shared_payload_code_is_stored_once(tmp_path: Any) -> None:
    """The payload reference names the frame's code instead of a second copy."""
    frame = SingleCarrierFrame(
        payload_blocks=4,
        payload_code=coding.LDPCCode.dvbs2("1/2", "short"),
        payload_constellation=Constellation.qam(16),
    )
    save_npz(frame.to_signal(sps=1, symbol_rate=1e9), tmp_path / "f")
    with np.load(tmp_path / "f.npz") as data:
        assert not any(k.startswith("frame_payload_reference_code") for k in data)


class TestNPZCodedReference:
    """The reference's source bits are stored only when they differ."""

    def test_coded_roundtrip(self, tmp_path: Any, xp: Any, xpt: Any) -> None:
        device = "gpu" if xp is not np else "cpu"
        sig = generate(
            Constellation.qam(16),
            70,
            symbol_rate=1e9,
            code=LinearBlockCode.hamming(3),
            rng=0,
            device=device,
        )
        save_npz(sig, tmp_path / "coded")
        assert "reference_info_bits" in np.load(tmp_path / "coded.npz")
        back = load_npz(tmp_path / "coded.npz", device=device)
        assert back.reference.code_block == (7, 4)
        assert back.reference.code == LinearBlockCode.hamming(3)
        xpt.assert_array_equal(back.reference.info_bits, sig.reference.info_bits)
        xpt.assert_array_equal(back.reference.bits, sig.reference.bits)

    def test_padded_roundtrip(self, tmp_path: Any) -> None:
        """Filler after the last codeword survives: num_blocks is derived."""
        sig = generate(
            Constellation.qam(16), 10, symbol_rate=1e9,
            code=LinearBlockCode.hamming(3), pad=True, rng=0,
        )  # fmt: skip
        save_npz(sig, tmp_path / "padded")
        back = load_npz(tmp_path / "padded.npz").reference
        assert back.num_blocks == sig.reference.num_blocks == 5
        np.testing.assert_array_equal(back.bits, sig.reference.bits)
        np.testing.assert_array_equal(back.info_bits, sig.reference.info_bits)

    def test_uncoded_stores_bits_once(self, tmp_path: Any) -> None:
        save_npz(_siso_signal(), tmp_path / "plain")
        assert "reference_info_bits" not in np.load(tmp_path / "plain.npz")
        ref = load_npz(tmp_path / "plain.npz").reference
        assert ref.info_bits is ref.bits

    def test_shaped_frame_keeps_the_matcher_input(self, tmp_path: Any) -> None:
        frame = SingleCarrierFrame(
            payload_num_symbols=400,
            payload_constellation=Constellation.qam(16).shaped(nu=0.1),
        )
        save_npz(frame.to_signal(sps=1, symbol_rate=1e9), tmp_path / "shaped")
        loaded = load_npz(tmp_path / "shaped.npz").frame
        np.testing.assert_array_equal(loaded.payload_info_bits, frame.payload_info_bits)
        assert (
            loaded._payload_reference().code_block
            == frame._payload_reference().code_block
        )
        plain = SingleCarrierFrame(payload_num_symbols=40)
        save_npz(plain.to_signal(sps=1, symbol_rate=1e9), tmp_path / "plain_frame")
        back = load_npz(tmp_path / "plain_frame.npz").frame
        assert back.payload_info_bits is back.payload_bits

    @pytest.mark.parametrize(
        "code, constellation",
        [
            (LinearBlockCode.hamming(3), Constellation.qam(16)),
            (coding.LDPCCode.dvbs2("1/2", "short"), Constellation.qam(16)),
            (
                coding.PAS(
                    coding.LDPCCode.dvbs2("3/4", "short"),
                    Constellation.qam(64).shaped(entropy=5.0),
                ),
                Constellation.qam(64).shaped(entropy=5.0),
            ),
        ],
        ids=["hamming", "ldpc", "pas"],
    )
    def test_frame_with_payload_code(self, tmp_path: Any, code, constellation) -> None:
        """The payload code is stored; the loaded frame encodes and decodes alike."""
        frame = SingleCarrierFrame(
            payload_blocks=12,
            payload_code=code,
            payload_constellation=constellation,
            pilot_pattern="comb",
            pilot_period=4,
        )
        sig = frame.to_signal(sps=1, symbol_rate=1e9)
        save_npz(sig, tmp_path / "f")
        back = load_npz(tmp_path / "f").frame
        assert back.payload_code == code
        assert back == frame
        np.testing.assert_array_equal(back.payload_info_bits, frame.payload_info_bits)
        np.testing.assert_array_equal(back.payload_bits, frame.payload_bits)
        np.testing.assert_array_equal(
            coding.encode(back.payload_info_bits, code=back.payload_code),
            coding.encode(frame.payload_info_bits, code=code),
        )


_CODES = [
    LinearBlockCode.hamming(4, extended=True),
    coding.LDPCCode.dvbs2("2/3", "short"),
    coding.ReedSolomonCode.kp4(),
    coding.BCHCode(6, 3),
    coding.StaircaseCode.g709_2(),
    coding.Concatenated.cfec(),
    coding.Concatenated.ieee_802_3dj(),
    coding.PolarCode.nr(k=64, n=128, crc="11"),
    coding.ConvolutionalCode.k7(40, termination="tail-biting"),
    coding.TurboCode.lte(40),
    coding.CRC.ethernet(64),
    coding.Punctured(LinearBlockCode.hamming(4, extended=True), [15]),
    coding.Shortened(coding.ReedSolomonCode.kp4(), 10),
    coding.CCDM(Constellation.pam(4).shaped(entropy=1.5), num_symbols=16),
    coding.PAS(
        coding.LDPCCode.dvbs2("3/4", "short"),
        Constellation.qam(64).shaped(entropy=5.0),
    ),
]


@pytest.mark.parametrize("code", _CODES, ids=lambda c: type(c).__name__)
def test_code_roundtrip(code) -> None:
    """Every code family is stored as its class and fields, arrays aside."""
    from commkit.io import _get_code, _put_code

    arrays: dict = {}
    stored = json.loads(json.dumps(_put_code(code, "c", arrays)))
    back = _get_code(stored, arrays, "test")
    assert type(back) is type(code)
    assert back == code
    assert (back.n, back.k) == (code.n, code.k)


def test_unknown_code_type_is_rejected() -> None:
    """Only commkit.coding classes are rebuilt, never an arbitrary name."""
    from commkit.io import _get_code

    with pytest.raises(ValueError, match="Unknown code type 'Popen'"):
        _get_code({"type": "Popen", "fields": {}}, {}, "test")


def test_frame_constellations_roundtrip(tmp_path: Any) -> None:
    frame = SingleCarrierFrame(
        payload_num_symbols=21,
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
