"""Tests for waveform synthesis primitives (expand, shape_pulse)."""

import numpy as np
import pytest

from commkit.core import generation


def test_expand(backend_device, xp, xpt):
    """Verify up-sampling by zero-stuffing correctly inserts zeros."""
    data = xp.array([1, 2, 3], dtype="float32")
    factor = 3
    expanded = generation.expand(data, factor)

    # Expected: 1, 0, 0, 2, 0, 0, 3, 0, 0
    expected = xp.array([1, 0, 0, 2, 0, 0, 3, 0, 0])
    xpt.assert_array_equal(expanded, expected)


def test_shape_pulse_variants(backend_device, xp):
    """Verify shape_pulse produces correct lengths for RC and sinc shapes."""
    symbols = xp.array([1, -1, 1, -1])

    res_rc = generation.shape_pulse(symbols, sps=4, pulse_shape="rc")
    assert len(res_rc) == 16

    res_sinc = generation.shape_pulse(symbols, sps=4, pulse_shape="sinc")
    assert len(res_sinc) == 16

    with pytest.raises(ValueError, match="Not implemented pulse shape"):
        generation.shape_pulse(symbols, sps=4, pulse_shape="magic")


def test_smoothrect_pulse(backend_device, xp):
    """Verify smoothrect pulse shaping output length."""
    symbols = xp.array([1, 1])
    res = generation.shape_pulse(
        symbols, sps=8, pulse_shape="smoothrect", filter_span=4, rise_time=0.05
    )
    assert len(res) == 16


def test_shape_pulse_none_with_rz(backend_device, xp):
    """Verify shape_pulse with pulse_shape='none' and rz=True expands using rect."""
    symbols = xp.array([1, -1, 1], dtype=xp.complex64)
    result = generation.shape_pulse(symbols, sps=4, pulse_shape="none", rz=True)
    assert result is not None
    assert len(result) > 0


def test_shape_pulse_preserves_complex64_dtype(backend_device, xp):
    """shape_pulse: complex64 symbols -> complex64 waveform."""
    rng = np.random.default_rng(12)
    syms = xp.asarray(
        (rng.standard_normal(100) + 1j * rng.standard_normal(100)).astype(np.complex64)
    )
    out = generation.shape_pulse(syms, sps=4, pulse_shape="rrc")
    assert out.dtype == xp.complex64, f"Expected complex64, got {out.dtype}"
