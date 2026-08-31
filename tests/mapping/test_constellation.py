"""Tests for the Constellation value object (mapping.constellation)."""

import numpy as np
import pytest

from commkit import mapping
from commkit.mapping import Constellation


def test_gray_constructor_matches_free_functions():
    """Constellation.gray points/labels match gray_constellation + bit table."""
    c = Constellation.gray("qam", 16)
    np.testing.assert_array_equal(c.points, mapping.gray_constellation("qam", 16))
    assert c.bit_labels.shape == (16, 4)
    assert c.bits_per_symbol == 4
    # Natural-binary labels: row i is the MSB-first bits of i.
    expected = ((np.arange(16)[:, None] >> np.arange(3, -1, -1)) & 1).astype(np.int8)
    np.testing.assert_array_equal(c.bit_labels, expected)


def test_gray_base_is_cached():
    """The pmf-less base is cached and returns the identical object."""
    assert Constellation.gray("qam", 16) is Constellation.gray("qam", 16)


def test_power_uniform_and_shaped():
    """power() equals constellation_power for uniform and shaped grids."""
    c_uniform = Constellation.gray("qam", 16)
    assert c_uniform.power() == pytest.approx(1.0, abs=1e-6)

    pmf = mapping.maxwell_boltzmann(16, 0.05)
    c_shaped = Constellation.gray("qam", 16, pmf=pmf)
    assert c_shaped.pmf is not None
    expected = mapping.constellation_power(c_shaped.points, pmf)
    assert c_shaped.power() == pytest.approx(expected, rel=1e-9)
    # Shaping a normalised grid lowers the average power below 1.
    assert c_shaped.power() < 1.0


def test_map_demap_roundtrip(xp, xpt):
    """map() then demap() recovers the bits, matching the free functions."""
    c = Constellation.gray("qam", 16)
    bits = xp.array([0, 0, 0, 0, 1, 1, 1, 1], dtype="int8")
    syms = c.map(bits)
    xpt.assert_array_equal(syms, mapping.map_bits(bits, "qam", 16))
    bits_out = c.demap(syms)
    xpt.assert_array_equal(bits_out, bits)


def test_unipolar_carried_through(xp):
    """A unipolar Constellation maps to a strictly non-negative grid."""
    c = Constellation.gray("ask", 4, unipolar=True)
    assert c.unipolar is True
    bits = xp.array([0, 1, 1, 0])
    syms = c.map(bits)
    assert bool(xp.all(syms >= 0))


def test_llr_matches_free_function(xp, xpt):
    """llr() delegates to compute_llr with this constellation's settings."""
    c = Constellation.gray("qam", 16)
    bits = xp.array([0, 0, 0, 0, 1, 1, 1, 1], dtype="int32")
    syms = mapping.map_bits(bits, "qam", 16)
    llr_obj = np.asarray(c.llr(syms, noise_var=0.1, output="numpy"))
    llr_ref = np.asarray(
        mapping.compute_llr(syms, "qam", 16, noise_var=0.1, output="numpy")
    )
    np.testing.assert_allclose(llr_obj, llr_ref, atol=1e-6)
