"""Tests for hard bit mapping and demapping."""

import numpy as np
import pytest

from commkit import mapping


def test_qam_mapping(xp):
    """Verify QAM mapping produces the correct number of symbols."""
    # Use xp to create array on device
    bits = xp.array([0, 0, 0, 0, 1, 1, 1, 1])

    syms = mapping.map_bits(bits, modulation="qam", order=16)

    assert isinstance(syms, xp.ndarray)
    assert len(syms) == 2


def test_psk_mapping(xp):
    """Verify PSK mapping produces the correct number of symbols."""
    bits = xp.array([0, 1])

    syms = mapping.map_bits(bits, modulation="psk", order=2)

    assert isinstance(syms, xp.ndarray)
    assert len(syms) == 2


def test_demap_dimensions_mimo(xp, xpt):
    """Test that demap_symbols_hard preserves multidimensional structure."""
    modulation = "qam"
    order = 4

    # Creates input shape (2 streams, 4 symbols)
    # Total symbols = 8
    # Total bits = 16
    bits_in = xp.zeros(16, dtype="int32")
    bits_in[::2] = 1

    # Map to symbols (flat) first
    symbols_flat = mapping.map_bits(bits_in, modulation, order)

    # Reshape to (2, 4) to simulate MIMO
    symbols_mimo = symbols_flat.reshape(2, 4)

    # Demap
    bits_out = mapping.demap_symbols_hard(symbols_mimo, modulation, order)

    # Verify strict shape compliance
    # We expect (2, 4 * 2) = (2, 8)
    expected_shape = (2, 8)
    assert bits_out.shape == expected_shape
    # Flatten both to compare content (bits_out is flat if reshape logic failed, but shape check guards it)
    xpt.assert_array_equal(bits_out.flatten(), bits_in)


def test_8qam_mapping(xp, xpt):
    """Verify 8-QAM (rectangular) mapping and round-trip."""
    bits = xp.array([0, 0, 0, 1, 1, 1], dtype="int32")
    syms = mapping.map_bits(bits, modulation="qam", order=8)
    assert len(syms) == 2

    bits_out = mapping.demap_symbols_hard(syms, modulation="qam", order=8)
    xpt.assert_array_equal(bits, bits_out)


def test_cross_qam_32_mapping(xp, xpt):
    """Verify 32-QAM (Cross) mapping and round-trip."""
    # 5 bits per symbol. 2 symbols = 10 bits.
    bits = xp.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype="int32")
    syms = mapping.map_bits(bits, modulation="qam", order=32)
    assert len(syms) == 2

    bits_out = mapping.demap_symbols_hard(syms, modulation="qam", order=32)
    xpt.assert_array_equal(bits, bits_out)


def test_map_demap_unipolar(xp, xpt):
    """Verify bit mapping and demapping with unipolar ASK/PAM."""
    bits = xp.array([0, 1, 1, 0])
    # Map to unipolar
    syms = mapping.map_bits(bits, "ask", 4, unipolar=True)
    assert xp.all(syms >= 0)

    # Demap from unipolar
    bits_rx = mapping.demap_symbols_hard(syms, "ask", 4, unipolar=True)
    xpt.assert_array_equal(bits, bits_rx)

    # LLR signs should recover the same bits
    llrs = mapping.compute_llr(syms, "ask", 4, noise_var=0.1, unipolar=True)
    bits_soft = (np.asarray(llrs) < 0).astype("int32")
    xpt.assert_array_equal(bits_soft, bits)


def test_mapping_more(xp):
    """Verify more mapping edge cases."""
    # Order error in map_bits
    with pytest.raises(ValueError, match="at least 2"):
        mapping.map_bits(xp.array([0, 0]), "psk", 1)

    # Order error in demap_symbols_hard
    with pytest.raises(ValueError, match="at least 2"):
        mapping.demap_symbols_hard(xp.array([0]), "psk", 1)


def test_mapping_order_errors(xp):
    """Verify order validation for power-of-2 requirements."""
    # map_bits
    with pytest.raises(ValueError, match="power of 2"):
        mapping.map_bits(xp.array([0, 0]), "psk", 3)

    # compute_llr
    with pytest.raises(ValueError, match="power of 2"):
        mapping.compute_llr(xp.array([0.1]), "psk", 3, noise_var=0.1)


def test_demap_symbols_empty_shape(xp):
    """Verify demapping behavior for effectively scalar inputs."""
    # 0-dim array (scalar)
    s = xp.array(1.0)
    # constellation will be [-1, 1] for psk-2
    # index 0 is -1, index 1 is 1.
    # so 1.0 -> index 1 -> bit 1.
    bits = mapping.demap_symbols_hard(s, "psk", 2)
    assert bits.size == 1
    assert int(bits.item()) == 1


def test_map_bits_divisibility(xp):
    """Verify error when bit count is not divisible by bits per symbol."""
    bits = xp.array([1, 0, 1])  # 3 bits
    with pytest.raises(ValueError, match="must be divisible by bits per symbol"):
        mapping.map_bits(bits, "qam", 16)  # bits_per_symbol = 4


def test_map_bits_fixed_dtypes(xp):
    """Verify map_bits returns complex64 for PSK/QAM and float32 for ASK/PAM."""
    bits = xp.array([0, 1, 0, 1])

    out_ask = mapping.map_bits(bits, "ask", 4)
    assert out_ask.dtype == "float32"

    out_pam = mapping.map_bits(bits, "pam", 4)
    assert out_pam.dtype == "float32"

    out_qam = mapping.map_bits(bits, "qam", 4)
    assert out_qam.dtype == "complex64"

    out_psk = mapping.map_bits(bits, "psk", 4)
    assert out_psk.dtype == "complex64"


def test_demap_symbols_returns_int8(xp):
    """Verify demap_symbols_hard returns int8 bits matching source_bits dtype."""
    bits = xp.array([0, 1, 0, 1], dtype="int8")
    symbols = mapping.map_bits(bits, "qam", 4)
    demapped = mapping.demap_symbols_hard(symbols, "qam", 4)
    assert demapped.dtype == "int8"
