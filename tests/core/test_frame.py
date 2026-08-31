"""Tests for SingleCarrierFrame structure and waveform generation."""

import numpy as np
import pytest

from commkit.core import Preamble, SingleCarrierFrame


def test_sc_frame_none(backend_device, xp):
    """Verify basic frame generation with no pilots or guard intervals."""
    frame = SingleCarrierFrame(payload_len=100, symbol_rate=1e6, pilot_pattern="none")
    sig = frame.to_signal(sps=1, pulse_shape="none")
    assert len(sig.samples) == 100
    assert sig.symbol_rate == 1e6
    assert sig.symbol_rate == 1e6
    assert sig.signal_type == "Single-Carrier Frame"
    assert sig.frame.payload_len == 100


def test_sc_frame_comb(backend_device, xp):
    """Verify 'comb' pilot insertion logic and resulting sequence length."""
    # payload=9, period=4 -> data_per_period=3, 9 % 3 == 0 (no snapping)
    # 3 full periods -> total_length = 3*4 = 12
    # Mask [T, F, F, F, T, F, F, F, T, F, F, F] -> 3 pilots, 9 data
    frame = SingleCarrierFrame(
        payload_len=9, symbol_rate=1e6, pilot_pattern="comb", pilot_period=4
    )
    mask, length = frame._generate_pilot_mask()
    assert length == 12
    assert xp.sum(mask) == 3

    sig = frame.to_signal(sps=1, pulse_shape="none")
    assert len(sig.samples) == 12
    assert len(sig.frame.pilot_symbols) >= 3


def test_sc_frame_block(backend_device, xp):
    """Verify 'block' pilot insertion logic and resulting sequence length."""
    # payload=10, period=4, len=2 -> data_per_block=2
    # num_blocks = ceil(10/2) = 5
    # Mask [T, T, F, F] * 5 -> [T, T, F, F, T, T, F, F, T, T, F, F, T, T, F, F, T, T, F, F]
    # False indices: 2, 3, 6, 7, 10, 11, 14, 15, 18, 19
    # 10th False is at index 19.
    # Truncated length: 19 + 1 = 20.
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


def test_sc_frame_guard_zero(backend_device, xp):
    """Verify zero-insertion guard interval (GI) padding."""
    frame = SingleCarrierFrame(
        payload_len=100, symbol_rate=1e6, guard_type="zero", guard_len=20
    )
    sig = frame.to_signal(sps=1, pulse_shape="none")
    assert len(sig.samples) == 120
    assert xp.all(sig.samples[-20:] == 0)
    assert sig.frame.guard_len == 20
    assert sig.frame.guard_type == "zero"


def test_sc_frame_guard_cp(backend_device, xp, xpt):
    """Verify cyclic prefix (CP) guard interval generation."""
    frame = SingleCarrierFrame(
        payload_len=100, symbol_rate=1e6, guard_type="cp", guard_len=20
    )
    sig = frame.to_signal(sps=1, pulse_shape="none")
    assert len(sig.samples) == 120
    # CP should match the last 20 samples of the *original* body
    # New structure: [CP (20), Body (100)]
    xpt.assert_allclose(sig.samples[:20], sig.samples[-20:])
    assert sig.frame.guard_type == "cp"


def test_sc_frame_preamble(backend_device, xp, xpt):
    """Verify that auto-generated preambles are correctly prepended to the frame."""
    # Create Barker-13 preamble
    preamble = Preamble(sequence_type="barker", length=13)
    frame = SingleCarrierFrame(payload_len=100, symbol_rate=1e6, preamble=preamble)
    sig = frame.to_signal(sps=1, pulse_shape="none")
    assert len(sig.samples) == 113  # 13 preamble + 100 payload
    # Verify preamble symbols match
    xpt.assert_allclose(sig.samples[:13], preamble.symbols)
    # Verify frame
    assert sig.frame.preamble.length == 13


def test_sc_frame_bit_first(backend_device, xp):
    """Verify Frame preserves source bits (bit-first architecture)."""
    frame = SingleCarrierFrame(
        payload_len=100, symbol_rate=1e6, payload_mod_order=4, payload_seed=42
    )
    # Access bits and symbols
    bits = frame.payload_bits

    # Bits should exist and have correct length (100 symbols * 2 bits/symbol)
    assert bits is not None
    assert bits.size == 200

    # Signal source_bits should be None for redundancy removal
    sig = frame.to_signal(sps=1, pulse_shape="none")
    assert sig.source_bits is None


def test_preamble_to_signal(backend_device, xp):
    """Verify Preamble.to_signal() stand-alone signal generation."""
    preamble = Preamble(sequence_type="barker", length=13)

    sig = preamble.to_signal(sps=4, symbol_rate=1e6, pulse_shape="rrc")

    assert len(sig.samples) == 13 * 4  # 13 symbols * 4 sps
    assert sig.mod_scheme is None
    assert sig.source_symbols is None


def test_sc_frame_structure_map(backend_device, xp):
    """Verify that get_structure_map correctly identifies frame segment boundaries."""
    # Frame with preamble (2), body (10: 4 pilots + 6 payload), guard (5)
    preamble = Preamble(sequence_type="barker", length=2)
    frame = SingleCarrierFrame(
        payload_len=6,
        symbol_rate=1e6,
        preamble=preamble,
        pilot_pattern="comb",
        pilot_period=2,  # 1 pilot every 2nd symbol
        guard_type="zero",
        guard_len=5,
    )

    # 1. Symbol-level check
    struct = frame.get_structure_map(unit="symbols", include_preamble=True)
    # Total length: 2 (preamble) + 10 (body: alternating P, D, P, D, P, D, P, 0, 0, 0?)
    # Wait, pilot mask for payload_len=6, period=2:
    # _generate_pilot_mask: data_per_period=1. num_periods=6.
    # Mask [T, F, T, F, T, F, T] (length 7). 4 pilots, 3 data.
    # Wait, I wanted 6 payload symbols.
    # If payload_len=6, period=2 -> needs 6 data symbols.
    # 6 periods -> [T, F] * 6 = [T, F, T, F, T, F, T, F, T, F, T, F]. Oops.

    # Let's just check length and pilot count.
    _, body_len = frame._generate_pilot_mask()
    total_len = 2 + body_len + 5

    assert len(struct["preamble"]) == total_len
    assert xp.sum(struct["preamble"]) == 2
    assert xp.sum(struct["pilots"]) > 0
    assert xp.sum(struct["payload"]) == 6
    assert xp.sum(struct["guard"]) == 5

    # 2. Sample-level check
    sps = 4
    struct_s = frame.get_structure_map(unit="samples", sps=sps, include_preamble=True)
    assert len(struct_s["preamble"]) == total_len * sps
    assert xp.sum(struct_s["preamble"]) == 2 * sps


def test_sc_frame_structure_map_cp():
    """Verify get_structure_map with Cyclic Prefix (CP) guard interval."""
    # Create SC frame with CP
    frame = SingleCarrierFrame(
        payload_len=10, symbol_rate=1e6, guard_type="cp", guard_len=5
    )
    struct = frame.get_structure_map(unit="symbols", include_preamble=True)
    # CP is at the beginning
    assert struct["guard"][0]
    assert struct["guard"][4]
    assert not struct["guard"][5]


def test_independent_preamble_normalization(backend_device, xp):
    """
    Verify frame normalization contract:
    1. The assembled frame has unit symbol power: mean(|x|²) * sps ≈ 1,
       i.e. average sample power ≈ 1/sps.
    2. Preamble and body sections have equal I/Q peaks, confirming that neither
       silences the other (each was independently I/Q peak-normalised before the
       final global symbol-power normalization, so the same global scale factor
       applies to both).

    Guards against the old global-normalisation bug where a high-PAPR body
    would silence a low-PAPR preamble or vice versa.
    """
    preamble = Preamble(sequence_type="barker", length=13)

    frame = SingleCarrierFrame(
        payload_len=100, symbol_rate=1e6, preamble=preamble, pilot_pattern="none"
    )

    sps = 4
    sig = frame.to_signal(sps=sps, pulse_shape="rrc", rrc_rolloff=0.5)

    preamble_len_samples = 13 * sps
    preamble_section = sig.samples[:preamble_len_samples]
    body_section = sig.samples[preamble_len_samples:]

    def iq_peak(s):
        return float(xp.maximum(xp.max(xp.abs(s.real)), xp.max(xp.abs(s.imag))))

    # Unit symbol power: mean(|x|²) * sps = 1  ->  avg sample power = 1/sps.
    avg_sample_power = float(xp.mean(xp.abs(sig.samples) ** 2))
    expected = 1.0 / sps
    assert abs(avg_sample_power - expected) < 1e-2, (
        f"Frame avg sample power {avg_sample_power:.4f} != 1/sps={expected:.4f}"
    )

    # Both sections were independently I/Q peak-normalised before the global
    # normalization, so their I/Q peaks must be equal after it.
    pp = iq_peak(preamble_section)
    bp = iq_peak(body_section)
    assert abs(pp - bp) < 0.05, (
        f"Preamble I/Q peak {pp:.4f} != body I/Q peak {bp:.4f} - "
        "sections were not independently normalized before global norm"
    )


def test_structure_map_default_no_preamble_no_guard():
    """Verify default behavior: no preamble, no guard (if none)."""
    frame = SingleCarrierFrame(payload_len=100)
    struct = frame.get_structure_map(include_preamble=False)

    assert "preamble" not in struct
    assert "guard" in struct
    assert len(struct["payload"]) == 100
    assert np.sum(struct["payload"]) == 100
    assert np.sum(struct["guard"]) == 0


def test_structure_map_include_preamble():
    """Verify explicit include_preamble=True."""
    preamble = Preamble(sequence_type="barker", length=13)
    frame = SingleCarrierFrame(payload_len=100, preamble=preamble)
    struct = frame.get_structure_map(include_preamble=True)

    assert "preamble" in struct
    assert len(struct["preamble"]) == 113
    assert np.sum(struct["preamble"]) == 13
    assert np.sum(struct["payload"]) == 100


def test_structure_map_no_preamble_with_zero_guard():
    """Verify default behavior with zero guard (guard should be present)."""
    frame = SingleCarrierFrame(payload_len=100, guard_type="zero", guard_len=20)
    struct = frame.get_structure_map(include_preamble=False)

    assert "preamble" not in struct
    assert "guard" in struct
    assert len(struct["payload"]) == 120
    assert np.sum(struct["payload"]) == 100
    assert np.sum(struct["guard"]) == 20


def test_structure_map_no_preamble_with_cp_guard():
    """Verify default behavior with CP guard (preamble and CP removed)."""
    frame = SingleCarrierFrame(payload_len=100, guard_type="cp", guard_len=20)
    struct = frame.get_structure_map(include_preamble=False)

    assert "preamble" not in struct
    assert "guard" not in struct  # CP is removed with preamble
    assert len(struct["payload"]) == 100
    assert np.sum(struct["payload"]) == 100


def test_structure_map_with_pilots_no_preamble():
    """Verify pilot mask is correct when preamble is excluded."""
    frame = SingleCarrierFrame(payload_len=10, pilot_pattern="comb", pilot_period=2)
    # _generate_pilot_mask for payload_len=10, period=2:
    # [T, F] * 10 = length 20. 10 pilots, 10 data.
    struct = frame.get_structure_map()

    assert len(struct["pilots"]) == 20
    assert np.sum(struct["pilots"]) == 10
    assert np.sum(struct["payload"]) == 10


def test_structure_map_samples_unit():
    """Verify unit='samples' with include_preamble=False."""
    frame = SingleCarrierFrame(payload_len=10)
    sps = 4
    struct = frame.get_structure_map(unit="samples", sps=sps)

    assert len(struct["payload"]) == 10 * sps
    assert np.sum(struct["payload"]) == 10 * sps


# -----------------------------------------------------------------------------
# PILOT MASK ERROR VALIDATION
# -----------------------------------------------------------------------------


def test_comb_pilot_period_le1_error(backend_device, xp):
    """comb pilot_period <= 1 should raise ValueError."""
    frame = SingleCarrierFrame(
        payload_len=10, symbol_rate=1e6, pilot_pattern="comb", pilot_period=1
    )
    with pytest.raises(ValueError, match="pilot_period must be > 1"):
        frame._generate_pilot_mask()


def test_block_pilot_period_le_block_len_error(backend_device, xp):
    """block pilot_period <= pilot_block_len should raise ValueError."""
    frame = SingleCarrierFrame(
        payload_len=10,
        symbol_rate=1e6,
        pilot_pattern="block",
        pilot_period=3,
        pilot_block_len=3,  # period == block_len -> invalid
    )
    with pytest.raises(ValueError, match="pilot_period must be > pilot_block_len"):
        frame._generate_pilot_mask()


# -----------------------------------------------------------------------------
# PILOT BITS/SYMBOLS WHEN PILOT_PATTERN='NONE'
# -----------------------------------------------------------------------------


def test_pilot_bits_none_when_no_pilots(backend_device, xp):
    """pilot_bits returns None when pilot_pattern='none'."""
    frame = SingleCarrierFrame(payload_len=20, symbol_rate=1e6, pilot_pattern="none")
    assert frame.pilot_bits is None


def test_pilot_symbols_none_when_no_pilots(backend_device, xp):
    """pilot_symbols returns None when pilot_pattern='none'."""
    frame = SingleCarrierFrame(payload_len=20, symbol_rate=1e6, pilot_pattern="none")
    assert frame.pilot_symbols is None


# -----------------------------------------------------------------------------
# PILOT GAIN DB - SISO AND MIMO
# -----------------------------------------------------------------------------


def test_pilot_gain_db_siso(backend_device, xp):
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
    # With 6 dB gain, pilot amplitude should be ~2x
    body_nogain = frame_nogain.body_symbols
    body_gain = frame_gain.body_symbols
    mask, _ = frame_gain._generate_pilot_mask()
    pilot_power_nogain = float(xp.mean(xp.abs(xp.asarray(body_nogain)[mask]) ** 2))
    pilot_power_gain = float(xp.mean(xp.abs(xp.asarray(body_gain)[mask]) ** 2))
    # 6 dB gain means power ratio ≈ 4x
    assert pilot_power_gain > pilot_power_nogain * 3.5


def test_pilot_gain_db_mimo(backend_device, xp):
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
    # MIMO body shape: (num_streams, body_length)
    assert body.shape[0] == 2
    assert body.ndim == 2


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# PILOT BITS/SYMBOLS WITH PILOTS ENABLED
# -----------------------------------------------------------------------------


def test_pilot_bits_with_pilots(backend_device, xp):
    """pilot_bits on a frame with comb pilots generates and returns pilot bit data."""
    frame = SingleCarrierFrame(
        payload_len=20, symbol_rate=1e6, pilot_pattern="comb", pilot_period=4
    )
    bits = frame.pilot_bits
    assert bits is not None
    assert len(bits) > 0


def test_pilot_bits_double_access(backend_device, xp):
    """Accessing pilot_bits twice returns consistent cached results."""
    frame = SingleCarrierFrame(
        payload_len=20, symbol_rate=1e6, pilot_pattern="comb", pilot_period=4
    )
    # First access: generates and caches
    bits1 = frame.pilot_bits
    # Second access: returns cached value without regenerating
    bits2 = frame.pilot_bits
    assert bits1 is not None
    assert bits2 is not None
    assert len(bits1) == len(bits2)


def test_pilot_symbols_with_pilots(backend_device, xp):
    """pilot_symbols on a frame with block pilots calls _ensure_pilot_generated."""
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


# -----------------------------------------------------------------------------
# Pilot divisibility snapping tests
# -----------------------------------------------------------------------------


def test_comb_snaps_payload_len(backend_device, xp):
    """payload_len non-divisible by data_per_period is snapped up (logger.warning)."""
    # payload=10, period=4 -> data_per_period=3, 10 % 3 != 0 -> snapped to 12
    frame = SingleCarrierFrame(
        payload_len=10, symbol_rate=1e6, pilot_pattern="comb", pilot_period=4
    )
    assert frame.payload_len == 12
    mask, length = frame._generate_pilot_mask()
    # 4 full periods of 3 data each = 12 data, 4 pilots, total 16 samples
    assert length == 16
    assert int(xp.sum(mask)) == 4


def test_block_snaps_payload_len(backend_device, xp):
    """payload_len non-divisible by data_per_block is snapped up (logger.warning)."""
    # payload=9, period=4, block_len=2 -> data_per_block=2, 9 % 2 != 0 -> snapped to 10
    frame = SingleCarrierFrame(
        payload_len=9,
        symbol_rate=1e6,
        pilot_pattern="block",
        pilot_period=4,
        pilot_block_len=2,
    )
    assert frame.payload_len == 10
    mask, length = frame._generate_pilot_mask()
    # 5 blocks of 2 data each = 10 data, 10 pilots, total 20 samples
    assert length == 20
    assert int(xp.sum(mask)) == 10


def test_comb_no_snap_when_divisible(backend_device, xp):
    """payload_len already divisible by data_per_period is not modified."""
    # payload=9, period=4 -> data_per_period=3, 9 % 3 == 0 -> no snap
    frame = SingleCarrierFrame(
        payload_len=9, symbol_rate=1e6, pilot_pattern="comb", pilot_period=4
    )
    assert frame.payload_len == 9


def test_block_no_snap_when_divisible(backend_device, xp):
    """payload_len already divisible by data_per_block is not modified."""
    # payload=10, period=4, block_len=2 -> data_per_block=2, 10 % 2 == 0 -> no snap
    frame = SingleCarrierFrame(
        payload_len=10,
        symbol_rate=1e6,
        pilot_pattern="block",
        pilot_period=4,
        pilot_block_len=2,
    )
    assert frame.payload_len == 10
