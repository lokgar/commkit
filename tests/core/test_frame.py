"""Tests for SingleCarrierFrame structure and waveform generation."""

import dataclasses
from typing import Any

import numpy as np
import pytest

from commkit.coding import CCDM, LinearBlockCode, decode, encode
from commkit.core import Preamble, SingleCarrierFrame, extract_payload
from commkit.core.generation import shape_pulse
from commkit.filtering import RRC
from commkit.mapping import Constellation
from tests.common.conversions import device_of, to_numpy
from tests.common.periodic import circular_matched_samples


class TestSingleCarrierFrameBasics:
    """Tests for basic frame generation, preambles, guards, and normalization."""

    def test_sc_frame_none(self, xp: Any) -> None:
        """Verify basic frame generation with no pilots or guard intervals."""
        frame = SingleCarrierFrame(payload_num_symbols=100, pilot_pattern="none")
        sig = frame.to_signal(sps=1)
        assert len(sig.samples) == 100
        assert sig.symbol_rate == 1e6
        assert sig.frame is frame
        assert sig.frame.payload_num_symbols == 100

    def test_sc_frame_comb(self, xp: Any) -> None:
        """Verify 'comb' pilot insertion logic and resulting sequence length."""
        frame = SingleCarrierFrame(
            payload_num_symbols=9, pilot_pattern="comb", pilot_period=4
        )
        mask, length = frame._generate_pilot_mask()
        assert length == 12
        assert xp.sum(mask) == 3

        sig = frame.to_signal(sps=1)
        assert len(sig.samples) == 12
        assert len(sig.frame.pilot_symbols) >= 3

    def test_sc_frame_block(self, xp: Any) -> None:
        """Verify 'block' pilot insertion logic and resulting sequence length."""
        frame = SingleCarrierFrame(
            payload_num_symbols=10,
            pilot_pattern="block",
            pilot_period=4,
            pilot_block_num_symbols=2,
        )
        mask, length = frame._generate_pilot_mask()
        assert length == 20
        assert xp.sum(mask) == 10

        sig = frame.to_signal(sps=1)
        assert len(sig.samples) == 20

    def test_sc_frame_guard_zero(self, xp: Any, xpt: Any) -> None:
        """Verify zero-insertion guard interval (GI) padding."""
        frame = SingleCarrierFrame(
            payload_num_symbols=100, guard_type="zero", guard_num_symbols=20
        )
        sig = frame.to_signal(sps=1)
        assert len(sig.samples) == 120
        xpt.assert_array_equal(sig.samples[-20:], 0)
        assert sig.frame.guard_num_symbols == 20
        assert sig.frame.guard_type == "zero"

    def test_sc_frame_guard_cp(self, xp: Any, xpt: Any) -> None:
        """Verify cyclic prefix (CP) guard interval generation."""
        frame = SingleCarrierFrame(
            payload_num_symbols=100, guard_type="cp", guard_num_symbols=20
        )
        sig = frame.to_signal(sps=1)
        assert len(sig.samples) == 120
        xpt.assert_allclose(sig.samples[:20], sig.samples[-20:])
        assert sig.frame.guard_type == "cp"

    def test_sc_frame_preamble(self, xp: Any, xpt: Any) -> None:
        """Verify that auto-generated preambles are prepended to the frame."""
        preamble = Preamble(sequence_type="barker", num_symbols=13)
        frame = SingleCarrierFrame(payload_num_symbols=100, preamble=preamble)
        sig = frame.to_signal(sps=1)
        assert len(sig.samples) == 113
        xpt.assert_allclose(sig.samples[:13], preamble.symbols)
        assert sig.frame.preamble.num_symbols == 13

    def test_sc_frame_bit_first(self, xp: Any) -> None:
        """Verify Frame preserves source bits (bit-first architecture)."""
        frame = SingleCarrierFrame(payload_num_symbols=100, payload_seed=42)
        bits = frame.payload_bits
        assert bits is not None
        assert bits.size == 200

        sig = frame.to_signal(sps=1)
        assert sig.reference is None

    def test_preamble_to_signal(self, xp: Any) -> None:
        """Verify Preamble.to_signal() standalone signal generation."""
        preamble = Preamble(sequence_type="barker", num_symbols=13)
        sig = preamble.to_signal(sps=4, symbol_rate=1e6, pulse=RRC(0.35))

        assert len(sig.samples) == 13 * 4
        assert sig.constellation is None
        assert sig.reference is None

    def test_single_stream_preamble_is_broadcast(self) -> None:
        """A one-stream Barker preamble starts every stream of a MIMO frame."""
        frame = SingleCarrierFrame(
            payload_num_symbols=40,
            preamble=Preamble(sequence_type="barker", num_symbols=13),
            num_streams=2,
        )
        x = frame.to_signal(sps=1, symbol_rate=1e6).samples
        assert x.shape == (2, 13 + 40)
        np.testing.assert_array_equal(x[0, :13], x[1, :13])

    def test_independent_preamble_normalization(self, xp: Any) -> None:
        """Verify frame normalization contract for independent section scaling."""
        preamble = Preamble(sequence_type="barker", num_symbols=13)
        frame = SingleCarrierFrame(
            payload_num_symbols=100, preamble=preamble, pilot_pattern="none"
        )

        sps = 4
        sig = frame.to_signal(sps=sps, pulse=RRC(0.5)).to(device_of(xp))
        preamble_len_samples = 13 * sps
        preamble_section = sig.samples[:preamble_len_samples]
        body_section = sig.samples[preamble_len_samples:]

        def iq_peak(s: Any) -> float:
            return float(xp.maximum(xp.max(xp.abs(s.real)), xp.max(xp.abs(s.imag))))

        avg_sample_power = float(xp.mean(xp.abs(sig.samples) ** 2))
        expected = 1.0 / sps
        assert abs(avg_sample_power - expected) < 1e-2

        pp = iq_peak(preamble_section)
        bp = iq_peak(body_section)
        assert abs(pp - bp) < 0.05


class TestSingleCarrierFrameStructureMap:
    """Tests for get_structure_map segment identification and boundary indexing."""

    def test_sc_frame_structure_map(self, xp: Any) -> None:
        """Verify get_structure_map identifies preamble, body, pilot, and guard boundaries."""
        preamble = Preamble(sequence_type="barker", num_symbols=2)
        frame = SingleCarrierFrame(
            payload_num_symbols=6,
            preamble=preamble,
            pilot_pattern="comb",
            pilot_period=2,
            guard_type="zero",
            guard_num_symbols=5,
        )

        struct = frame.get_structure_map(unit="symbols", include_preamble=True)
        _, body_num_symbols = frame._generate_pilot_mask()
        total_num_symbols = 2 + body_num_symbols + 5

        assert len(struct["preamble"]) == total_num_symbols
        assert xp.sum(struct["preamble"]) == 2
        assert xp.sum(struct["pilots"]) > 0
        assert xp.sum(struct["payload"]) == 6
        assert xp.sum(struct["guard"]) == 5

        sps = 4
        struct_s = frame.get_structure_map(
            unit="samples", sps=sps, include_preamble=True
        )
        assert len(struct_s["preamble"]) == total_num_symbols * sps
        assert xp.sum(struct_s["preamble"]) == 2 * sps

    def test_sc_frame_structure_map_cp(self, xp: Any) -> None:
        """Verify get_structure_map with Cyclic Prefix (CP) guard interval."""
        frame = SingleCarrierFrame(
            payload_num_symbols=10, guard_type="cp", guard_num_symbols=5
        )
        struct = frame.get_structure_map(unit="symbols", include_preamble=True)
        assert struct["guard"][0]
        assert struct["guard"][4]
        assert not struct["guard"][5]

    def test_structure_map_default_no_preamble_no_guard(self, xp: Any) -> None:
        """Verify default behavior: no preamble, zero guard."""
        frame = SingleCarrierFrame(payload_num_symbols=100)
        struct = frame.get_structure_map(include_preamble=False)

        assert "preamble" not in struct
        assert "guard" in struct
        assert len(struct["payload"]) == 100
        assert xp.sum(struct["payload"]) == 100
        assert xp.sum(struct["guard"]) == 0

    def test_structure_map_include_preamble(self, xp: Any) -> None:
        """Verify explicit include_preamble=True in structure map."""
        preamble = Preamble(sequence_type="barker", num_symbols=13)
        frame = SingleCarrierFrame(payload_num_symbols=100, preamble=preamble)
        struct = frame.get_structure_map(include_preamble=True)

        assert "preamble" in struct
        assert len(struct["preamble"]) == 113
        assert xp.sum(struct["preamble"]) == 13
        assert xp.sum(struct["payload"]) == 100

    def test_structure_map_no_preamble_with_zero_guard(self, xp: Any) -> None:
        """Verify behavior with zero guard when preamble is excluded."""
        frame = SingleCarrierFrame(
            payload_num_symbols=100, guard_type="zero", guard_num_symbols=20
        )
        struct = frame.get_structure_map(include_preamble=False)

        assert "preamble" not in struct
        assert "guard" in struct
        assert len(struct["payload"]) == 120
        assert xp.sum(struct["payload"]) == 100
        assert xp.sum(struct["guard"]) == 20

    def test_structure_map_no_preamble_with_cp_guard(self, xp: Any) -> None:
        """Verify behavior with CP guard when preamble is excluded."""
        frame = SingleCarrierFrame(
            payload_num_symbols=100, guard_type="cp", guard_num_symbols=20
        )
        struct = frame.get_structure_map(include_preamble=False)

        assert "preamble" not in struct
        assert "guard" not in struct
        assert len(struct["payload"]) == 100
        assert xp.sum(struct["payload"]) == 100

    def test_structure_map_with_pilots_no_preamble(self, xp: Any) -> None:
        """Verify pilot mask is correct when preamble is excluded."""
        frame = SingleCarrierFrame(
            payload_num_symbols=10, pilot_pattern="comb", pilot_period=2
        )
        struct = frame.get_structure_map()

        assert len(struct["pilots"]) == 20
        assert xp.sum(struct["pilots"]) == 10
        assert xp.sum(struct["payload"]) == 10

    def test_structure_map_samples_unit(self, xp: Any) -> None:
        """Verify unit='samples' with include_preamble=False."""
        frame = SingleCarrierFrame(payload_num_symbols=10)
        sps = 4
        struct = frame.get_structure_map(unit="samples", sps=sps)

        assert len(struct["payload"]) == 10 * sps
        assert xp.sum(struct["payload"]) == 10 * sps


class TestSingleCarrierFramePilots:
    """Tests for pilot generation, validation, caching, and power boost."""

    def test_comb_pilot_period_le1_error(self, xp: Any) -> None:
        """comb pilot_period <= 1 raises ValueError on construction."""
        with pytest.raises(ValueError, match="pilot_period must be > 1"):
            SingleCarrierFrame(
                payload_num_symbols=10, pilot_pattern="comb", pilot_period=1
            )

    def test_block_pilot_period_le_block_len_error(self, xp: Any) -> None:
        """block pilot_period <= pilot_block_num_symbols raises ValueError on construction."""
        with pytest.raises(
            ValueError, match="pilot_period must be > pilot_block_num_symbols"
        ):
            SingleCarrierFrame(
                payload_num_symbols=10,
                pilot_pattern="block",
                pilot_period=3,
                pilot_block_num_symbols=3,
            )

    def test_block_without_pilot_block_len_error(self, xp: Any) -> None:
        """The block pattern needs at least one pilot per period."""
        with pytest.raises(ValueError, match="pilot_block_num_symbols must be >= 1"):
            SingleCarrierFrame(
                payload_num_symbols=12, pilot_pattern="block", pilot_period=4
            )

    def test_comb_is_a_block_of_one_pilot(self, xp: Any) -> None:
        """comb places one pilot at the start of each period, like block with 1."""
        comb = SingleCarrierFrame(
            payload_num_symbols=12, pilot_pattern="comb", pilot_period=4
        )
        block = SingleCarrierFrame(
            payload_num_symbols=12,
            pilot_pattern="block",
            pilot_period=4,
            pilot_block_num_symbols=1,
        )
        mask, length = comb._generate_pilot_mask()
        assert length == 16
        np.testing.assert_array_equal(np.flatnonzero(mask), [0, 4, 8, 12])
        np.testing.assert_array_equal(mask, block._generate_pilot_mask()[0])

    def test_pilot_bits_none_when_no_pilots(self, xp: Any) -> None:
        """pilot_bits returns None when pilot_pattern='none'."""
        frame = SingleCarrierFrame(payload_num_symbols=20, pilot_pattern="none")
        assert frame.pilot_bits is None

    def test_pilot_symbols_none_when_no_pilots(self, xp: Any) -> None:
        """pilot_symbols returns None when pilot_pattern='none'."""
        frame = SingleCarrierFrame(payload_num_symbols=20, pilot_pattern="none")
        assert frame.pilot_symbols is None

    def test_pilot_gain_db_siso(self, xp: Any) -> None:
        """Non-zero pilot_gain_db boosts pilot symbols for SISO."""
        frame_nogain = SingleCarrierFrame(
            payload_num_symbols=20,
            pilot_pattern="comb",
            pilot_period=5,
            pilot_gain_db=0.0,
        )
        frame_gain = SingleCarrierFrame(
            payload_num_symbols=20,
            pilot_pattern="comb",
            pilot_period=5,
            pilot_gain_db=6.0,
        )
        body_nogain = frame_nogain.body_symbols
        body_gain = frame_gain.body_symbols
        mask, _ = frame_gain._generate_pilot_mask()
        pilot_power_nogain = float(xp.mean(xp.abs(xp.asarray(body_nogain)[mask]) ** 2))
        pilot_power_gain = float(xp.mean(xp.abs(xp.asarray(body_gain)[mask]) ** 2))
        assert pilot_power_gain > pilot_power_nogain * 3.5

    def test_pilot_gain_db_mimo(self, xp: Any) -> None:
        """Non-zero pilot_gain_db boosts pilot symbols for MIMO."""
        frame = SingleCarrierFrame(
            payload_num_symbols=20,
            pilot_pattern="comb",
            pilot_period=5,
            pilot_gain_db=6.0,
            num_streams=2,
        )
        body = frame.body_symbols
        assert body.shape[0] == 2
        assert body.ndim == 2

    def test_pilot_bits_with_pilots(self, xp: Any) -> None:
        """pilot_bits on a frame with comb pilots generates pilot bits."""
        frame = SingleCarrierFrame(
            payload_num_symbols=21, pilot_pattern="comb", pilot_period=4
        )
        bits = frame.pilot_bits
        assert bits is not None
        assert len(bits) > 0

    def test_pilot_bits_double_access(self, xp: Any) -> None:
        """Accessing pilot_bits twice returns consistent cached results."""
        frame = SingleCarrierFrame(
            payload_num_symbols=21, pilot_pattern="comb", pilot_period=4
        )
        bits1 = frame.pilot_bits
        bits2 = frame.pilot_bits
        assert bits1 is not None
        assert bits2 is not None
        assert len(bits1) == len(bits2)

    def test_pilot_symbols_with_pilots(self, xp: Any) -> None:
        """pilot_symbols on a frame with block pilots generates pilot symbols."""
        frame = SingleCarrierFrame(
            payload_num_symbols=20,
            pilot_pattern="block",
            pilot_period=4,
            pilot_block_num_symbols=2,
        )
        syms = frame.pilot_symbols
        assert syms is not None
        assert len(syms) > 0


class TestSingleCarrierFrameDivisibility:
    """payload_num_symbols must fill whole pilot periods; it is never changed silently."""

    def test_comb_rejects_partial_period(self, xp: Any) -> None:
        with pytest.raises(ValueError, match="use 9 or 12"):
            SingleCarrierFrame(
                payload_num_symbols=10, pilot_pattern="comb", pilot_period=4
            )
        frame = SingleCarrierFrame(
            payload_num_symbols=12, pilot_pattern="comb", pilot_period=4
        )
        mask, length = frame._generate_pilot_mask()
        assert length == 16
        assert int(xp.sum(mask)) == 4

    def test_block_rejects_partial_period(self, xp: Any) -> None:
        with pytest.raises(ValueError, match="use 8 or 10"):
            SingleCarrierFrame(
                payload_num_symbols=9,
                pilot_pattern="block",
                pilot_period=4,
                pilot_block_num_symbols=2,
            )
        frame = SingleCarrierFrame(
            payload_num_symbols=10,
            pilot_pattern="block",
            pilot_period=4,
            pilot_block_num_symbols=2,
        )
        mask, length = frame._generate_pilot_mask()
        assert length == 20
        assert int(xp.sum(mask)) == 10

    def test_short_payload_suggests_one_period(self, xp: Any) -> None:
        with pytest.raises(ValueError, match=r"use 3\."):
            SingleCarrierFrame(
                payload_num_symbols=2, pilot_pattern="comb", pilot_period=4
            )

    def test_comb_no_snap_when_divisible(self, xp: Any) -> None:
        """payload_num_symbols already divisible by data_per_period is not modified."""
        frame = SingleCarrierFrame(
            payload_num_symbols=9, pilot_pattern="comb", pilot_period=4
        )
        assert frame.payload_num_symbols == 9

    def test_block_no_snap_when_divisible(self, xp: Any) -> None:
        """payload_num_symbols already divisible by data_per_block is not modified."""
        frame = SingleCarrierFrame(
            payload_num_symbols=10,
            pilot_pattern="block",
            pilot_period=4,
            pilot_block_num_symbols=2,
        )
        assert frame.payload_num_symbols == 10


class TestFrameConstellations:
    """payload_constellation / pilot_constellation."""

    def test_payload_and_pilots_use_their_constellations(self) -> None:
        frame = SingleCarrierFrame(
            payload_num_symbols=21,
            payload_constellation=Constellation.qam(16),
            pilot_pattern="comb",
            pilot_period=4,
            pilot_constellation=Constellation.psk(2),
        )
        assert np.isin(
            frame.payload_symbols, Constellation.qam(16).points.astype("complex64")
        ).all()
        assert np.isin(
            frame.pilot_symbols, Constellation.psk(2).points.astype("complex64")
        ).all()

    def test_shaped_payload(self) -> None:
        c = Constellation.qam(64).shaped(entropy=5.0)
        frame = SingleCarrierFrame(payload_num_symbols=2000, payload_constellation=c)
        np.testing.assert_array_equal(c.map(frame.payload_bits), frame.payload_symbols)
        sig = frame.to_signal(sps=2, symbol_rate=1e6, pulse=RRC(0.35))
        np.testing.assert_array_equal(sig.frame.payload_constellation.pmf, c.pmf)

    def test_invalid_constellations(self) -> None:
        with pytest.raises(ValueError, match="payload_constellation"):
            SingleCarrierFrame(payload_constellation="qam")
        with pytest.raises(ValueError, match="must not be shaped"):
            SingleCarrierFrame(pilot_constellation=Constellation.qam(16).shaped(nu=0.1))

    def test_layout_needs_no_data(self) -> None:
        frame = SingleCarrierFrame(
            payload_num_symbols=21, pilot_pattern="comb", pilot_period=4
        )
        frame.get_structure_map()
        assert frame._cache == {}


class TestExtractPayload:
    """extract_payload: the payload of a full 1-SPS frame, reference attached."""

    @staticmethod
    def _frame(num_streams: int = 1, guard_type: str = "cp") -> SingleCarrierFrame:
        return SingleCarrierFrame(
            payload_num_symbols=90,
            payload_constellation=Constellation.qam(16),
            preamble=Preamble(
                sequence_type="barker", num_symbols=13, num_streams=num_streams
            ),
            pilot_pattern="comb",
            pilot_period=10,
            pilot_gain_db=3.0,
            guard_type=guard_type,
            guard_num_symbols=7,
            num_streams=num_streams,
        )

    @pytest.mark.parametrize("guard_type", ["cp", "zero"])
    @pytest.mark.parametrize("num_streams", [1, 2])
    def test_payload_and_reference(self, num_streams, guard_type, xp, xpt):
        frame = self._frame(num_streams, guard_type)
        sig = frame.to_signal(sps=1, symbol_rate=1e6).to(
            "gpu" if xp is not np else "cpu"
        )
        out = extract_payload(sig)
        ref = out.reference.symbols
        assert out.frame is None
        rf = sig.replace(domain="RF", center_frequency=2.4e9)
        assert extract_payload(rf).domain == "RF"
        assert extract_payload(rf).center_frequency == 2.4e9
        assert out.constellation == Constellation.qam(16)
        assert out.samples.shape == np.shape(frame.payload_symbols)
        xpt.assert_array_equal(ref, xp.asarray(frame.payload_symbols))
        xpt.assert_array_equal(out.reference.bits, xp.asarray(frame.payload_bits))
        # The samples are the payload symbols up to the frame's per-stream scale.
        num = xp.sum(out.samples * xp.conj(ref), axis=-1, keepdims=True)
        scale = num / xp.sum(xp.abs(ref) ** 2, axis=-1, keepdims=True)
        xpt.assert_allclose(out.samples, scale * ref, rtol=1e-5, atol=1e-6)

    def test_coded_payload(self, xp, xpt):
        """payload_code forwards to generate; the reference keeps both bit
        streams on the samples' device."""
        code = LinearBlockCode.hamming(3)
        frame = SingleCarrierFrame(
            payload_num_symbols=70,
            payload_constellation=Constellation.qam(16),
            payload_code=code,
            num_streams=2,
        )
        np.testing.assert_array_equal(
            encode(frame.payload_info_bits, code=code), frame.payload_bits
        )
        assert frame.payload_info_bits.shape == (2, 160)
        sig = frame.to_signal(sps=1, symbol_rate=1e6).to(
            "gpu" if xp is not np else "cpu"
        )
        ref = extract_payload(sig).reference
        assert ref.code_block == (7, 4) and ref.code is code
        xpt.assert_array_equal(ref.info_bits, xp.asarray(frame.payload_info_bits))
        xpt.assert_array_equal(ref.bits, xp.asarray(frame.payload_bits))
        uncoded = SingleCarrierFrame(payload_num_symbols=10)
        assert uncoded.payload_info_bits is uncoded.payload_bits

    def test_shaped_payload_keeps_the_matcher_input(self, xp, xpt):
        c = Constellation.qam(16).shaped(entropy=3.5)
        frame = SingleCarrierFrame(payload_num_symbols=500, payload_constellation=c)
        sig = frame.to_signal(sps=1, symbol_rate=1e6).to(
            "gpu" if xp is not np else "cpu"
        )
        ref = extract_payload(sig).reference
        matcher = CCDM(c, block_length=500)
        assert ref.code_block == (2000, matcher.k) and ref.code == matcher
        xpt.assert_array_equal(ref.info_bits, xp.asarray(frame.payload_info_bits))
        xpt.assert_array_equal(decode(ref.bits, code=ref.code).bits, ref.info_bits)

    def test_coded_payload_validation(self):
        with pytest.raises(ValueError, match="payload_code.n=7"):
            SingleCarrierFrame(
                payload_num_symbols=10, payload_code=LinearBlockCode.hamming(3)
            )
        with pytest.raises(TypeError, match="code object"):
            SingleCarrierFrame(payload_num_symbols=7, payload_code="hamming")
        with pytest.raises(ValueError, match="fewer than one codeword"):
            SingleCarrierFrame(
                payload_num_symbols=3, payload_code=LinearBlockCode.hamming(3)
            )
        with pytest.raises(ValueError, match="need a payload_code"):
            SingleCarrierFrame(payload_num_symbols=10, payload_pad=True)

    def test_coded_payload_suggests_lengths_with_pilots(self):
        """Whole codewords (every 7 QPSK symbols: 2 codewords of 7 bits) and
        whole pilot periods (3 payload symbols each) meet every 21 symbols."""
        with pytest.raises(ValueError, match="payload_num_symbols=21 or 42"):
            SingleCarrierFrame(
                payload_num_symbols=30,
                payload_code=LinearBlockCode.hamming(3),
                pilot_pattern="comb",
                pilot_period=4,
            )

    def test_padded_payload(self, xp, xpt):
        """payload_pad: whole codewords, then filler; any payload_num_symbols fits."""
        code = LinearBlockCode.hamming(3)
        frame = SingleCarrierFrame(
            payload_num_symbols=30,
            payload_code=code,
            payload_pad=True,
            pilot_pattern="comb",
            pilot_period=4,
            num_streams=2,
        )
        sig = frame.to_signal(sps=1, symbol_rate=1e6).to(
            "gpu" if xp is not np else "cpu"
        )
        ref = extract_payload(sig).reference
        assert ref.num_blocks == 8  # 60 bits: 8 codewords and 4 filler bits
        assert ref.bits.shape == (2, 60) and ref.info_bits.shape == (2, 32)
        res = decode(ref.bits[:, : ref.num_blocks * code.n], code=code)
        xpt.assert_array_equal(res.bits, ref.info_bits)

    def test_sized_in_codewords(self):
        """payload_blocks: the payload is the symbols its codewords fill."""
        code = LinearBlockCode.hamming(3)
        frame = SingleCarrierFrame(payload_blocks=6, payload_code=code)
        assert frame.payload_num_symbols == 21  # 42 bits of QPSK
        assert frame.payload_info_bits.shape == (24,)
        # With pilots: 6 codewords fill 7 periods of 3 payload symbols.
        piloted = SingleCarrierFrame(
            payload_blocks=6, payload_code=code, pilot_pattern="comb", pilot_period=4
        )
        assert piloted.payload_num_symbols == 21
        # replace() passes both sizes, which agree.
        again = dataclasses.replace(frame, payload_seed=3)
        assert again.payload_num_symbols == 21 and again.payload_blocks == 6

    def test_codeword_sizing_errors(self):
        code = LinearBlockCode.hamming(3)
        with pytest.raises(ValueError, match="payload_blocks=2 or 4"):
            SingleCarrierFrame(payload_blocks=3, payload_code=code)
        with pytest.raises(ValueError, match="disagree"):
            SingleCarrierFrame(
                payload_num_symbols=14, payload_blocks=6, payload_code=code
            )
        with pytest.raises(ValueError, match="Give one of them"):
            SingleCarrierFrame(payload_blocks=6, payload_code=code, payload_pad=True)
        with pytest.raises(ValueError, match="need a payload_code"):
            SingleCarrierFrame(payload_blocks=6)
        c = Constellation.qam(16).shaped(entropy=3.5)
        with pytest.raises(ValueError, match="composition"):
            SingleCarrierFrame(
                payload_num_symbols=25, payload_constellation=c,
                payload_code=CCDM(c, block_length=10), payload_pad=True,
            )  # fmt: skip

    def test_matched_payload_in_blocks(self, xp, xpt):
        c = Constellation.qam(16).shaped(entropy=3.5)
        matcher = CCDM(c, block_length=100)
        frame = SingleCarrierFrame(
            payload_blocks=3, payload_constellation=c, payload_code=matcher
        )
        assert frame.payload_num_symbols == 300
        ref = extract_payload(frame.to_signal(sps=1, symbol_rate=1e6)).reference
        assert ref.num_blocks == 3
        xpt.assert_array_equal(
            to_numpy(decode(ref.bits, code=matcher).bits), to_numpy(ref.info_bits)
        )

    def test_num_symbols_is_the_frame_length(self):
        for kwargs in (
            {},
            {"guard_type": "cp", "guard_num_symbols": 16},
            {"pilot_pattern": "comb", "pilot_period": 4, "payload_num_symbols": 300},
        ):
            frame = SingleCarrierFrame(
                preamble=Preamble(sequence_type="zc", num_symbols=31),
                **{"guard_num_symbols": 8, **kwargs},
            )
            sig = frame.to_signal(sps=2, symbol_rate=1e6)
            assert sig.samples.shape[-1] == 2 * frame.num_symbols

    def test_without_frame_raises(self):
        from commkit import generate

        sig = generate(Constellation.qam(16), 100, symbol_rate=1e6)
        with pytest.raises(ValueError, match="no frame"):
            extract_payload(sig)

    def test_oversampled_raises(self):
        sig = self._frame().to_signal(sps=2, symbol_rate=1e6)
        with pytest.raises(ValueError, match="one sample per symbol"):
            extract_payload(sig)

    def test_partial_frame_raises(self):
        sig = self._frame().to_signal(sps=1, symbol_rate=1e6)
        sig = sig.replace(samples=sig.samples[5:])
        with pytest.raises(ValueError, match="full frame"):
            extract_payload(sig)


_PERIODIC_FRAMES = [
    dict(),
    dict(preamble=Preamble(num_symbols=13), guard_type="zero", guard_num_symbols=8),
    dict(preamble=Preamble(num_symbols=13), guard_type="cp", guard_num_symbols=8),
    dict(
        preamble=Preamble(num_symbols=13),
        num_streams=2,
        guard_type="zero",
        guard_num_symbols=8,
    ),
    dict(
        preamble=Preamble(sequence_type="zc", num_symbols=31, num_streams=2),
        num_streams=2,
        pilot_pattern="comb",
        pilot_period=8,
    ),
]


class TestPeriodicFrame:
    """to_signal(periodic=True): the frame shaped once, as one loop period."""

    @pytest.mark.parametrize("kw", _PERIODIC_FRAMES)
    def test_unshaped_matches_linear(self, backend_device: str, kw: dict) -> None:
        """Without a pulse there is nothing to wrap: section gains, guard and
        layout must equal the linear path."""
        frame = SingleCarrierFrame(payload_num_symbols=196, **kw)
        a = frame.to_signal(sps=2, device=backend_device)
        b = frame.to_signal(sps=2, periodic=True, device=backend_device)
        assert b.samples.dtype == a.samples.dtype
        np.testing.assert_allclose(to_numpy(b.samples), to_numpy(a.samples), atol=1e-6)
        assert b.frame is frame

    @pytest.mark.parametrize("kw", _PERIODIC_FRAMES[:2])
    def test_section_edges_recovered(self, kw: dict) -> None:
        """A circular matched filter recovers every section, edges included,
        up to one gain per section; the linear frame breaks at the wrap."""
        pulse = RRC(0.1, span=32)
        frame = SingleCarrierFrame(payload_num_symbols=196, **kw)
        pre = 0 if frame.preamble is None else frame.preamble.num_symbols
        sections = [frame.body_symbols]
        if pre:
            sections.insert(0, frame.preamble.symbols)
        errors = {}
        for periodic in (False, True):
            sig = frame.to_signal(sps=4, pulse=pulse, periodic=periodic)
            y = circular_matched_samples(sig.samples, pulse.taps(4), 4)
            start, err = 0, []
            for ref in sections:
                seg = y[start : start + ref.size]
                g = np.vdot(ref, seg) / np.vdot(ref, ref)
                err.append(np.abs(seg / g - ref))
                start += ref.size
            errors[periodic] = np.concatenate(err)
        assert errors[True].max() < 0.03
        assert errors[False].max() > 0.1

    def test_preamble_forwards_periodic(self, xpt: Any) -> None:
        preamble = Preamble(sequence_type="zc", num_symbols=31)
        sig = preamble.to_signal(sps=4, symbol_rate=1e6, pulse=RRC(0.1), periodic=True)
        xpt.assert_allclose(
            sig.samples,
            shape_pulse(preamble.symbols, sps=4, pulse=RRC(0.1), periodic=True),
        )


class TestFrameDevice:
    """to_signal(device=) shapes on the device and matches the CPU waveform."""

    def test_frame_matches_cpu(self, backend_device: str, xp: Any) -> None:
        frame = SingleCarrierFrame(
            payload_num_symbols=196,
            num_streams=2,
            preamble=Preamble(sequence_type="zc", num_symbols=31),
            pilot_pattern="comb",
            pilot_period=8,
            guard_type="cp",
            guard_num_symbols=16,
        )
        ref = frame.to_signal(sps=2, pulse=RRC(0.2))
        sig = frame.to_signal(sps=2, pulse=RRC(0.2), device=backend_device)
        assert isinstance(sig.samples, xp.ndarray)
        assert sig.samples.dtype == ref.samples.dtype
        np.testing.assert_allclose(to_numpy(sig.samples), ref.samples, atol=1e-5)

    def test_preamble_device(self, backend_device: str, xp: Any) -> None:
        sig = Preamble(sequence_type="barker", num_symbols=13).to_signal(
            sps=4, symbol_rate=1e6, device=backend_device
        )
        assert isinstance(sig.samples, xp.ndarray)

    def test_invalid_device(self) -> None:
        with pytest.raises(ValueError, match="device"):
            SingleCarrierFrame(payload_num_symbols=10).to_signal(sps=1, device="tpu")
