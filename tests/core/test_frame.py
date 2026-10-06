"""Tests for SingleCarrierFrame structure and waveform generation."""

from typing import Any

import pytest

from commkit.core import Preamble, SingleCarrierFrame
from tests.common.conversions import device_of


class TestSingleCarrierFrameBasics:
    """Tests for basic frame generation, preambles, guards, and normalization."""

    def test_sc_frame_none(self, xp: Any) -> None:
        """Verify basic frame generation with no pilots or guard intervals."""
        frame = SingleCarrierFrame(
            payload_len=100, symbol_rate=1e6, pilot_pattern="none"
        )
        sig = frame.to_signal(sps=1, pulse_shape="none")
        assert len(sig.samples) == 100
        assert sig.symbol_rate == 1e6
        assert sig.signal_type == "Single-Carrier Frame"
        assert sig.frame.payload_len == 100

    def test_sc_frame_comb(self, xp: Any) -> None:
        """Verify 'comb' pilot insertion logic and resulting sequence length."""
        frame = SingleCarrierFrame(
            payload_len=9, symbol_rate=1e6, pilot_pattern="comb", pilot_period=4
        )
        mask, length = frame._generate_pilot_mask()
        assert length == 12
        assert xp.sum(mask) == 3

        sig = frame.to_signal(sps=1, pulse_shape="none")
        assert len(sig.samples) == 12
        assert len(sig.frame.pilot_symbols) >= 3

    def test_sc_frame_block(self, xp: Any) -> None:
        """Verify 'block' pilot insertion logic and resulting sequence length."""
        frame = SingleCarrierFrame(
            payload_len=10,
            symbol_rate=1e6,
            pilot_pattern="block",
            pilot_period=4,
            pilot_block_len=2,
        )
        mask, length = frame._generate_pilot_mask()
        assert length == 20
        assert xp.sum(mask) == 10

        sig = frame.to_signal(sps=1, pulse_shape="none")
        assert len(sig.samples) == 20

    def test_sc_frame_guard_zero(self, xp: Any, xpt: Any) -> None:
        """Verify zero-insertion guard interval (GI) padding."""
        frame = SingleCarrierFrame(
            payload_len=100, symbol_rate=1e6, guard_type="zero", guard_len=20
        )
        sig = frame.to_signal(sps=1, pulse_shape="none")
        assert len(sig.samples) == 120
        xpt.assert_array_equal(sig.samples[-20:], 0)
        assert sig.frame.guard_len == 20
        assert sig.frame.guard_type == "zero"

    def test_sc_frame_guard_cp(self, xp: Any, xpt: Any) -> None:
        """Verify cyclic prefix (CP) guard interval generation."""
        frame = SingleCarrierFrame(
            payload_len=100, symbol_rate=1e6, guard_type="cp", guard_len=20
        )
        sig = frame.to_signal(sps=1, pulse_shape="none")
        assert len(sig.samples) == 120
        xpt.assert_allclose(sig.samples[:20], sig.samples[-20:])
        assert sig.frame.guard_type == "cp"

    def test_sc_frame_preamble(self, xp: Any, xpt: Any) -> None:
        """Verify that auto-generated preambles are prepended to the frame."""
        preamble = Preamble(sequence_type="barker", length=13)
        frame = SingleCarrierFrame(payload_len=100, symbol_rate=1e6, preamble=preamble)
        sig = frame.to_signal(sps=1, pulse_shape="none")
        assert len(sig.samples) == 113
        xpt.assert_allclose(sig.samples[:13], preamble.symbols)
        assert sig.frame.preamble.length == 13

    def test_sc_frame_bit_first(self, xp: Any) -> None:
        """Verify Frame preserves source bits (bit-first architecture)."""
        frame = SingleCarrierFrame(
            payload_len=100, symbol_rate=1e6, payload_mod_order=4, payload_seed=42
        )
        bits = frame.payload_bits
        assert bits is not None
        assert bits.size == 200

        sig = frame.to_signal(sps=1, pulse_shape="none")
        assert sig.source_bits is None

    def test_preamble_to_signal(self, xp: Any) -> None:
        """Verify Preamble.to_signal() standalone signal generation."""
        preamble = Preamble(sequence_type="barker", length=13)
        sig = preamble.to_signal(sps=4, symbol_rate=1e6, pulse_shape="rrc")

        assert len(sig.samples) == 13 * 4
        assert sig.mod_scheme is None
        assert sig.source_symbols is None

    def test_independent_preamble_normalization(self, xp: Any) -> None:
        """Verify frame normalization contract for independent section scaling."""
        preamble = Preamble(sequence_type="barker", length=13)
        frame = SingleCarrierFrame(
            payload_len=100, symbol_rate=1e6, preamble=preamble, pilot_pattern="none"
        )

        sps = 4
        sig = frame.to_signal(sps=sps, pulse_shape="rrc", rrc_rolloff=0.5).to(
            device_of(xp)
        )
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
        preamble = Preamble(sequence_type="barker", length=2)
        frame = SingleCarrierFrame(
            payload_len=6,
            symbol_rate=1e6,
            preamble=preamble,
            pilot_pattern="comb",
            pilot_period=2,
            guard_type="zero",
            guard_len=5,
        )

        struct = frame.get_structure_map(unit="symbols", include_preamble=True)
        _, body_len = frame._generate_pilot_mask()
        total_len = 2 + body_len + 5

        assert len(struct["preamble"]) == total_len
        assert xp.sum(struct["preamble"]) == 2
        assert xp.sum(struct["pilots"]) > 0
        assert xp.sum(struct["payload"]) == 6
        assert xp.sum(struct["guard"]) == 5

        sps = 4
        struct_s = frame.get_structure_map(
            unit="samples", sps=sps, include_preamble=True
        )
        assert len(struct_s["preamble"]) == total_len * sps
        assert xp.sum(struct_s["preamble"]) == 2 * sps

    def test_sc_frame_structure_map_cp(self, xp: Any) -> None:
        """Verify get_structure_map with Cyclic Prefix (CP) guard interval."""
        frame = SingleCarrierFrame(
            payload_len=10, symbol_rate=1e6, guard_type="cp", guard_len=5
        )
        struct = frame.get_structure_map(unit="symbols", include_preamble=True)
        assert struct["guard"][0]
        assert struct["guard"][4]
        assert not struct["guard"][5]

    def test_structure_map_default_no_preamble_no_guard(self, xp: Any) -> None:
        """Verify default behavior: no preamble, zero guard."""
        frame = SingleCarrierFrame(payload_len=100)
        struct = frame.get_structure_map(include_preamble=False)

        assert "preamble" not in struct
        assert "guard" in struct
        assert len(struct["payload"]) == 100
        assert xp.sum(struct["payload"]) == 100
        assert xp.sum(struct["guard"]) == 0

    def test_structure_map_include_preamble(self, xp: Any) -> None:
        """Verify explicit include_preamble=True in structure map."""
        preamble = Preamble(sequence_type="barker", length=13)
        frame = SingleCarrierFrame(payload_len=100, preamble=preamble)
        struct = frame.get_structure_map(include_preamble=True)

        assert "preamble" in struct
        assert len(struct["preamble"]) == 113
        assert xp.sum(struct["preamble"]) == 13
        assert xp.sum(struct["payload"]) == 100

    def test_structure_map_no_preamble_with_zero_guard(self, xp: Any) -> None:
        """Verify behavior with zero guard when preamble is excluded."""
        frame = SingleCarrierFrame(payload_len=100, guard_type="zero", guard_len=20)
        struct = frame.get_structure_map(include_preamble=False)

        assert "preamble" not in struct
        assert "guard" in struct
        assert len(struct["payload"]) == 120
        assert xp.sum(struct["payload"]) == 100
        assert xp.sum(struct["guard"]) == 20

    def test_structure_map_no_preamble_with_cp_guard(self, xp: Any) -> None:
        """Verify behavior with CP guard when preamble is excluded."""
        frame = SingleCarrierFrame(payload_len=100, guard_type="cp", guard_len=20)
        struct = frame.get_structure_map(include_preamble=False)

        assert "preamble" not in struct
        assert "guard" not in struct
        assert len(struct["payload"]) == 100
        assert xp.sum(struct["payload"]) == 100

    def test_structure_map_with_pilots_no_preamble(self, xp: Any) -> None:
        """Verify pilot mask is correct when preamble is excluded."""
        frame = SingleCarrierFrame(payload_len=10, pilot_pattern="comb", pilot_period=2)
        struct = frame.get_structure_map()

        assert len(struct["pilots"]) == 20
        assert xp.sum(struct["pilots"]) == 10
        assert xp.sum(struct["payload"]) == 10

    def test_structure_map_samples_unit(self, xp: Any) -> None:
        """Verify unit='samples' with include_preamble=False."""
        frame = SingleCarrierFrame(payload_len=10)
        sps = 4
        struct = frame.get_structure_map(unit="samples", sps=sps)

        assert len(struct["payload"]) == 10 * sps
        assert xp.sum(struct["payload"]) == 10 * sps


class TestSingleCarrierFramePilots:
    """Tests for pilot generation, validation, caching, and power boost."""

    def test_comb_pilot_period_le1_error(self, xp: Any) -> None:
        """comb pilot_period <= 1 raises ValueError."""
        frame = SingleCarrierFrame(
            payload_len=10, symbol_rate=1e6, pilot_pattern="comb", pilot_period=1
        )
        with pytest.raises(ValueError, match="pilot_period must be > 1"):
            frame._generate_pilot_mask()

    def test_block_pilot_period_le_block_len_error(self, xp: Any) -> None:
        """block pilot_period <= pilot_block_len raises ValueError."""
        frame = SingleCarrierFrame(
            payload_len=10,
            symbol_rate=1e6,
            pilot_pattern="block",
            pilot_period=3,
            pilot_block_len=3,
        )
        with pytest.raises(ValueError, match="pilot_period must be > pilot_block_len"):
            frame._generate_pilot_mask()

    def test_pilot_bits_none_when_no_pilots(self, xp: Any) -> None:
        """pilot_bits returns None when pilot_pattern='none'."""
        frame = SingleCarrierFrame(
            payload_len=20, symbol_rate=1e6, pilot_pattern="none"
        )
        assert frame.pilot_bits is None

    def test_pilot_symbols_none_when_no_pilots(self, xp: Any) -> None:
        """pilot_symbols returns None when pilot_pattern='none'."""
        frame = SingleCarrierFrame(
            payload_len=20, symbol_rate=1e6, pilot_pattern="none"
        )
        assert frame.pilot_symbols is None

    def test_pilot_gain_db_siso(self, xp: Any) -> None:
        """Non-zero pilot_gain_db boosts pilot symbols for SISO."""
        frame_nogain = SingleCarrierFrame(
            payload_len=20,
            symbol_rate=1e6,
            pilot_pattern="comb",
            pilot_period=5,
            pilot_gain_db=0.0,
        )
        frame_gain = SingleCarrierFrame(
            payload_len=20,
            symbol_rate=1e6,
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
            payload_len=20,
            symbol_rate=1e6,
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
            payload_len=20, symbol_rate=1e6, pilot_pattern="comb", pilot_period=4
        )
        bits = frame.pilot_bits
        assert bits is not None
        assert len(bits) > 0

    def test_pilot_bits_double_access(self, xp: Any) -> None:
        """Accessing pilot_bits twice returns consistent cached results."""
        frame = SingleCarrierFrame(
            payload_len=20, symbol_rate=1e6, pilot_pattern="comb", pilot_period=4
        )
        bits1 = frame.pilot_bits
        bits2 = frame.pilot_bits
        assert bits1 is not None
        assert bits2 is not None
        assert len(bits1) == len(bits2)

    def test_pilot_symbols_with_pilots(self, xp: Any) -> None:
        """pilot_symbols on a frame with block pilots generates pilot symbols."""
        frame = SingleCarrierFrame(
            payload_len=20,
            symbol_rate=1e6,
            pilot_pattern="block",
            pilot_period=4,
            pilot_block_len=2,
        )
        syms = frame.pilot_symbols
        assert syms is not None
        assert len(syms) > 0


class TestSingleCarrierFrameDivisibility:
    """Tests for automatic payload length snapping to pilot block boundaries."""

    def test_comb_snaps_payload_len(self, xp: Any) -> None:
        """payload_len non-divisible by data_per_period is snapped up."""
        frame = SingleCarrierFrame(
            payload_len=10, symbol_rate=1e6, pilot_pattern="comb", pilot_period=4
        )
        assert frame.payload_len == 12
        mask, length = frame._generate_pilot_mask()
        assert length == 16
        assert int(xp.sum(mask)) == 4

    def test_block_snaps_payload_len(self, xp: Any) -> None:
        """payload_len non-divisible by data_per_block is snapped up."""
        frame = SingleCarrierFrame(
            payload_len=9,
            symbol_rate=1e6,
            pilot_pattern="block",
            pilot_period=4,
            pilot_block_len=2,
        )
        assert frame.payload_len == 10
        mask, length = frame._generate_pilot_mask()
        assert length == 20
        assert int(xp.sum(mask)) == 10

    def test_comb_no_snap_when_divisible(self, xp: Any) -> None:
        """payload_len already divisible by data_per_period is not modified."""
        frame = SingleCarrierFrame(
            payload_len=9, symbol_rate=1e6, pilot_pattern="comb", pilot_period=4
        )
        assert frame.payload_len == 9

    def test_block_no_snap_when_divisible(self, xp: Any) -> None:
        """payload_len already divisible by data_per_block is not modified."""
        frame = SingleCarrierFrame(
            payload_len=10,
            symbol_rate=1e6,
            pilot_pattern="block",
            pilot_period=4,
            pilot_block_len=2,
        )
        assert frame.payload_len == 10
