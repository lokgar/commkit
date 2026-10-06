"""Tests for hard bit mapping and demapping."""

from typing import Any

import pytest

from commkit import mapping
from commkit.mapping import Constellation


class TestBitMapping:
    """Tests for bit-to-symbol mapping across modulations and orders."""

    def test_qam_mapping(self, xp: Any) -> None:
        """Verify QAM mapping produces the correct number of symbols on device."""
        bits = xp.array([0, 0, 0, 0, 1, 1, 1, 1])
        syms = mapping.map_bits(bits, constellation=Constellation.qam(16))

        assert isinstance(syms, xp.ndarray)
        assert len(syms) == 2

    def test_psk_mapping(self, xp: Any) -> None:
        """Verify PSK mapping produces the correct number of symbols on device."""
        bits = xp.array([0, 1])
        syms = mapping.map_bits(bits, constellation=Constellation.psk(2))

        assert isinstance(syms, xp.ndarray)
        assert len(syms) == 2

    def test_8qam_mapping(self, xp: Any, xpt: Any) -> None:
        """Verify 8-QAM (rectangular) mapping and round-trip."""
        bits = xp.array([0, 0, 0, 1, 1, 1], dtype="int32")
        syms = mapping.map_bits(bits, constellation=Constellation.qam(8))
        assert len(syms) == 2

        bits_out = mapping.demap_symbols_hard(syms, constellation=Constellation.qam(8))
        xpt.assert_array_equal(bits, bits_out)

    def test_cross_qam_32_mapping(self, xp: Any, xpt: Any) -> None:
        """Verify 32-QAM (Cross) mapping and round-trip."""
        bits = xp.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype="int32")
        syms = mapping.map_bits(bits, constellation=Constellation.qam(32))
        assert len(syms) == 2

        bits_out = mapping.demap_symbols_hard(syms, constellation=Constellation.qam(32))
        xpt.assert_array_equal(bits, bits_out)

    def test_map_demap_unipolar(self, xp: Any, xpt: Any) -> None:
        """Verify bit mapping and demapping with unipolar ASK/PAM."""
        bits = xp.array([0, 1, 1, 0])
        syms = mapping.map_bits(bits, constellation=Constellation.pam(4, unipolar=True))
        xpt.assert_array_equal(syms >= 0, True)

        bits_rx = mapping.demap_symbols_hard(
            syms, constellation=Constellation.pam(4, unipolar=True)
        )
        xpt.assert_array_equal(bits, bits_rx)

        llrs = mapping.compute_llr(
            syms, noise_var=0.1, constellation=Constellation.pam(4, unipolar=True)
        )
        bits_soft = (llrs < 0).astype("int32")
        xpt.assert_array_equal(bits_soft, bits)

    def test_map_bits_fixed_dtypes(self, xp: Any) -> None:
        """Verify map_bits returns complex64 for PSK/QAM and float32 for ASK/PAM."""
        bits = xp.array([0, 1, 0, 1])

        out_ask = mapping.map_bits(bits, constellation=Constellation.pam(4))
        assert out_ask.dtype == "float32"

        out_pam = mapping.map_bits(bits, constellation=Constellation.pam(4))
        assert out_pam.dtype == "float32"

        out_qam = mapping.map_bits(bits, constellation=Constellation.qam(4))
        assert out_qam.dtype == "complex64"

        out_psk = mapping.map_bits(bits, constellation=Constellation.psk(4))
        assert out_psk.dtype == "complex64"


class TestBitDemapping:
    """Tests for symbol-to-bit hard demapping."""

    def test_demap_dimensions_mimo(self, xp: Any, xpt: Any) -> None:
        """Test that demap_symbols_hard preserves multidimensional structure."""
        order = 4
        constellation = Constellation.qam(order)

        bits_in = xp.zeros(16, dtype="int32")
        bits_in[::2] = 1

        symbols_flat = mapping.map_bits(bits_in, constellation=constellation)
        symbols_mimo = symbols_flat.reshape(2, 4)

        bits_out = mapping.demap_symbols_hard(symbols_mimo, constellation=constellation)
        assert bits_out.shape == (2, 8)
        xpt.assert_array_equal(bits_out.flatten(), bits_in)

    def test_demap_symbols_empty_shape(self, xp: Any) -> None:
        """Verify demapping behavior for effectively scalar inputs."""
        s = xp.array(1.0)
        bits = mapping.demap_symbols_hard(s, constellation=Constellation.psk(2))
        assert bits.size == 1
        assert int(bits.item()) == 1

    def test_demap_symbols_returns_int8(self, xp: Any) -> None:
        """Verify demap_symbols_hard returns int8 bits matching source_bits dtype."""
        bits = xp.array([0, 1, 0, 1], dtype="int8")
        symbols = mapping.map_bits(bits, constellation=Constellation.qam(4))
        demapped = mapping.demap_symbols_hard(
            symbols, constellation=Constellation.qam(4)
        )
        assert demapped.dtype == "int8"


class TestBitMappingValidation:
    """Tests for argument validation and error branches in bit mapping."""

    def test_constellation_must_be_an_object(self, xp: Any) -> None:
        """Strings are rejected by name; constellation= is required for arrays."""
        with pytest.raises(TypeError, match="Constellation"):
            mapping.map_bits(xp.array([0, 0]), constellation="qpsk")
        with pytest.raises(TypeError, match="Constellation"):
            mapping.demap_symbols_hard(xp.array([0j]), constellation="qpsk")
        with pytest.raises(TypeError, match="Constellation"):
            mapping.compute_llr(xp.array([0j]), noise_var=0.1, constellation="qpsk")
        with pytest.raises(ValueError, match="constellation="):
            mapping.demap_symbols_hard(xp.array([0j]))
        with pytest.raises(ValueError, match="constellation="):
            mapping.compute_llr(xp.array([0j]), noise_var=0.1)

    def test_map_bits_divisibility(self, xp: Any) -> None:
        """Bit sequence length not divisible by bits per symbol raises ValueError."""
        bits = xp.array([1, 0, 1])
        with pytest.raises(ValueError, match="multiple of bits_per_symbol"):
            mapping.map_bits(bits, constellation=Constellation.qam(16))
