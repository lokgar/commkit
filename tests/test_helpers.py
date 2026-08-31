"""Tests for helpers routines (normalization, interpolation, random generation)."""

import numpy as np
import pytest

from commkit import helpers


class Unconvertible:
    """Object that raises an exception during np.asarray conversion."""

    def __array__(self):
        raise TypeError("Cannot convert to array")


# -----------------------------------------------------------------------------
# RANDOM GENERATION
# -----------------------------------------------------------------------------


def test_random_bits(backend_device, xp):
    """Verify random bit generation produces correct length, binary values, and device."""
    bits = helpers.generate_bits(100, seed=42)
    assert len(bits) == 100
    assert xp.all((bits == 0) | (bits == 1))
    assert isinstance(bits, xp.ndarray)


def test_random_bits_no_seed(backend_device, xp):
    """Verify generate_bits without a seed produces a result on the active device."""
    bits = helpers.generate_bits(100)
    assert isinstance(bits, xp.ndarray)
    assert len(bits) == 100


def test_random_symbols_unipolar(backend_device, xp):
    """Verify unipolar flag in generate_symbols produces non-negative values."""
    syms = helpers.generate_symbols(10, "ask", 4, unipolar=True)
    assert xp.all(syms >= 0)


# -----------------------------------------------------------------------------
# NORMALIZATION
# -----------------------------------------------------------------------------


def test_normalize(backend_device, xp):
    """Verify peak and average-power normalization modes."""
    data = xp.array([1.0, 2.0, 0.5])

    norm = helpers.normalize(data, mode="peak")
    assert xp.isclose(xp.max(xp.abs(norm)), 1.0)

    norm_power = helpers.normalize(data, mode="average_power")
    assert xp.isclose(float(xp.mean(xp.abs(norm_power) ** 2)), 1.0)


def test_normalize_peak_complex_envelope(backend_device, xp):
    """peak mode normalizes by complex envelope, not per-component I/Q.

    After normalization max(|x[n]|) == 1.0. A subsequent frequency rotation
    must not push real or imaginary parts outside [-1, 1].
    """
    # Sample with large imaginary relative to real: envelope = sqrt(0.6^2 + 0.8^2) = 1.0
    # Before fix: per-component max was 0.8, so norm_factor=0.8 -> envelope after = 1.25
    data = xp.array([0.6 + 0.8j, -0.3 + 0.4j, 0.1 - 0.2j])
    norm = helpers.normalize(data, mode="peak")

    # Complex envelope peak must be exactly 1.0
    assert xp.isclose(xp.max(xp.abs(norm)), 1.0)

    # Neither component may exceed 1.0 (would violate the envelope bound)
    assert float(xp.max(xp.abs(norm.real))) <= 1.0 + 1e-6
    assert float(xp.max(xp.abs(norm.imag))) <= 1.0 + 1e-6

    # After a 45-degree rotation (worst case for per-component spread),
    # both components must still be within [-1, 1].
    import numpy as _np

    rotated = norm * _np.exp(1j * _np.pi / 4)
    assert float(xp.max(xp.abs(rotated.real))) <= 1.0 + 1e-6
    assert float(xp.max(xp.abs(rotated.imag))) <= 1.0 + 1e-6


def test_normalize_unity_gain(backend_device, xp):
    """Verify unity-gain normalization (sum of elements = 1)."""
    data = xp.array([1.0, 2.0, 3.0, 4.0])
    norm = helpers.normalize(data, mode="unity_gain")
    assert xp.isclose(xp.sum(norm), 1.0)


def test_normalize_zeros(backend_device, xp):
    """Verify all-zero array normalization returns zeros without NaN."""
    x = xp.zeros(10)
    out = helpers.normalize(x, mode="unit_energy")
    assert xp.all(out == 0)
    assert out.shape == (10,)


def test_normalize_invalid_mode(backend_device, xp):
    """Verify unknown normalization mode raises ValueError."""
    with pytest.raises(ValueError, match="Unknown normalization mode"):
        helpers.normalize(xp.ones(5), mode="invalid")


# -----------------------------------------------------------------------------
# SI FORMATTING
# -----------------------------------------------------------------------------


def test_format_si(backend_device, xp):
    """Verify SI-prefix formatting for common magnitudes."""
    assert helpers.format_si(None) == "None"
    assert helpers.format_si(0) == "0.00 Hz"
    assert "1.00 MHz" in helpers.format_si(1e6, "Hz")
    assert "500.00 mV" in helpers.format_si(0.5, "V")
    assert "Hz" in helpers.format_si(100)


# -----------------------------------------------------------------------------
# ARRAY VALIDATION
# -----------------------------------------------------------------------------


def test_validate_array(backend_device, xp):
    """Verify array validation: None passthrough, list conversion, complex_only, error paths."""
    assert helpers.validate_array(None) is None

    arr = helpers.validate_array([1, 2, 3])
    assert isinstance(arr, (xp.ndarray, np.ndarray))

    arr_c = helpers.validate_array(xp.array([1, 2], dtype=float), complex_only=True)
    assert xp.iscomplexobj(arr_c)

    with pytest.raises(ValueError, match="Expected numeric array"):
        helpers.validate_array("not an array")

    with pytest.raises(ValueError, match="Expected numeric array"):
        helpers.validate_array(np.array(["a", "b"]))


def test_validate_array_complex_only(backend_device, xp):
    """Verify complex_only flag zero-extends the imaginary part."""
    arr = xp.array([1, 2, 3], dtype=float)
    out = helpers.validate_array(arr, complex_only=True)
    assert xp.iscomplexobj(out)
    assert xp.all(out.real == arr)
    assert xp.all(out.imag == 0)


def test_validate_array_exception(backend_device, xp):
    """Verify the except-block in validate_array raises ValueError for unconvertible input."""
    obj = Unconvertible()
    with pytest.raises(ValueError, match="Could not convert"):
        helpers.validate_array(obj)


# -----------------------------------------------------------------------------
# RMS
# -----------------------------------------------------------------------------


def test_rms_axis(backend_device, xp, xpt):
    """Verify RMS over all elements and per-row."""
    x = xp.array([[1.0, 1.0], [2.0, 2.0]])
    xpt.assert_allclose(helpers.rms(x), xp.sqrt(2.5))
    xpt.assert_allclose(helpers.rms(x, axis=1), [1.0, 2.0])


# -----------------------------------------------------------------------------
# MIMO PREAMBLE EXPANSION
# -----------------------------------------------------------------------------


def test_zc_mimo_root(backend_device, xp):
    """zc_mimo_root assigns distinct roots cycling from base_root in [1, length-1]."""
    from commkit.helpers import zc_mimo_root

    # base_root=1, N=13 -> roots 1,2,3,4
    assert zc_mimo_root(0, 1, 13) == 1
    assert zc_mimo_root(1, 1, 13) == 2
    assert zc_mimo_root(2, 1, 13) == 3

    # wraps at length-1=12 -> root 12 then back to 1
    assert zc_mimo_root(0, 10, 13) == 10
    assert zc_mimo_root(1, 10, 13) == 11
    assert zc_mimo_root(2, 10, 13) == 12
    assert zc_mimo_root(3, 10, 13) == 1  # wraps

    # All roots must be in [1, length-1]
    for k in range(12):
        r = zc_mimo_root(k, 1, 13)
        assert 1 <= r <= 12


# -----------------------------------------------------------------------------
# DTYPE PRESERVATION TESTS
# -----------------------------------------------------------------------------


def test_normalize_preserves_float32_dtype(backend_device, xp):
    """normalize: float32 input -> float32 output across all modes."""
    x = xp.asarray(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    for mode in ("unity_gain", "unit_energy", "peak", "average_power"):
        out = helpers.normalize(x, mode=mode)
        assert out.dtype == xp.float32, (
            f"mode={mode!r}: expected float32, got {out.dtype}"
        )


def test_rms_preserves_float32_dtype(backend_device, xp):
    """rms: float32 input -> float32 output."""
    x = xp.asarray(np.ones(64, dtype=np.float32))
    out = helpers.rms(x)
    assert out.dtype == xp.float32, f"Expected float32, got {out.dtype}"


def test_normalize_preserves_complex64_dtype(backend_device, xp):
    """normalize: complex64 input -> complex64 output."""
    x = xp.asarray(np.array([1 + 1j, 2 + 2j], dtype=np.complex64))
    for mode in ("unit_energy", "peak", "average_power"):
        out = helpers.normalize(x, mode=mode)
        assert out.dtype == xp.complex64, (
            f"mode={mode!r}: expected complex64, got {out.dtype}"
        )
