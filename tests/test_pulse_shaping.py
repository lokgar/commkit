"""Tests for pulse shaping, matched filtering, and filter tap generation in Signal objects."""

import pytest

from commkit import filtering, generate_pam, generate_qam
from commkit.core import Signal


def test_signal_pulse_params(backend_device, xp):
    """Verify that pulse shaping parameters (e.g., rolloff) are correctly stored and utilized."""
    sig = generate_qam(
        order=4,
        num_symbols=10,
        sps=4,
        symbol_rate=1e3,
        pulse_shape="rrc",
        rrc_rolloff=0.5,
    )
    assert sig.pulse_shape == "rrc"
    assert getattr(sig, "pulse_params", None) is None
    assert sig.rrc_rolloff == 0.5

    # Check if taps generation works
    taps = filtering.shaping_filter_taps(sig)
    assert taps is not None
    assert len(taps) > 0


def test_matched_filter_auto_taps(backend_device, xp):
    """Verify that matched_filter correctly auto-generates and applies the correct taps."""
    sig = generate_pam(
        order=2,
        unipolar=False,
        num_symbols=100,
        sps=4,
        symbol_rate=1e3,
        pulse_shape="rrc",
    )

    # Copy signal to compare
    sig_before = sig.clone()

    # Apply matched filter (should auto-generate taps)
    sig = filtering.matched_filter(sig)

    # Check that samples changed (filtering happened)
    assert not xp.allclose(sig.samples, sig_before.samples)


def test_rzpam_pulse_params(backend_device, xp):
    """Verify pulse parameters for Return-to-Zero (RZ) PAM signals."""
    sig = generate_pam(
        order=2,
        unipolar=False,
        num_symbols=10,
        sps=4,
        symbol_rate=1e3,
        rz=True,
        pulse_shape="smoothrect",
        rise_time=0.1,
    )
    assert sig.pulse_shape == "smoothrect"
    assert sig.rise_time == 0.1

    taps = filtering.shaping_filter_taps(sig)
    assert len(taps) > 0


def test_rz_rect_taps_length(backend_device, xp, xpt):
    """Verify that RZ rectangular pulse taps have the correct half-symbol length."""
    # RZ rect with sps=4 should have length 2 (sps * 0.5)
    sig = generate_pam(
        order=2,
        unipolar=False,
        num_symbols=10,
        sps=4,
        symbol_rate=1e3,
        rz=True,
        pulse_shape="rect",
    )
    taps = filtering.shaping_filter_taps(sig)
    assert len(taps) == 2
    xpt.assert_allclose(taps, xp.ones(2))


def test_unknown_pulse_shape(backend_device, xp):
    """Verify that attempting to generate taps for unsupported shapes raises errors."""
    sig = Signal(samples=[1, 2], sampling_rate=10, symbol_rate=5)
    # No pulse shape
    with pytest.raises(ValueError, match="No pulse shape defined"):
        filtering.shaping_filter_taps(sig)

    sig.pulse_shape = "unknown_shape"
    with pytest.raises(ValueError, match="Unknown pulse shape"):
        filtering.shaping_filter_taps(sig)


def test_rect_pulse_taps(backend_device, xp, xpt):
    """Verify that standard rectangular pulse shaping produces all-ones taps."""
    # Rect pulse should return ones
    sig = generate_pam(
        order=2,
        unipolar=False,
        num_symbols=10,
        sps=4,
        symbol_rate=1e3,
        pulse_shape="rect",
    )
    taps = filtering.shaping_filter_taps(sig)
    xpt.assert_allclose(taps, xp.ones(4))


def test_gaussian_shaping_filter_taps(backend_device, xp):
    """shaping_filter_taps for gaussian pulse shape returns valid taps."""
    sig = Signal(
        samples=xp.ones(40, dtype="complex64"),
        sampling_rate=4e3,
        symbol_rate=1e3,
        pulse_shape="gaussian",
        filter_span=4,
        duty_cycle=0.5,
    )
    taps = filtering.shaping_filter_taps(sig)
    assert taps is not None
    assert len(taps) > 0


def test_rc_shaping_filter_taps(backend_device, xp):
    """shaping_filter_taps for rc pulse shape returns valid taps."""
    sig = Signal(
        samples=xp.ones(40, dtype="complex64"),
        sampling_rate=4e3,
        symbol_rate=1e3,
        pulse_shape="rc",
        filter_span=4,
        rc_rolloff=0.5,
    )
    taps = filtering.shaping_filter_taps(sig)
    assert taps is not None
    assert len(taps) > 0


def test_matched_filter_logs_error_for_no_pulse_shape(backend_device, xp):
    """matched_filter() with no pulse_shape logs an error and returns a copy unchanged."""
    sig = Signal(
        samples=xp.ones(10, dtype="complex64"),
        sampling_rate=4e3,
        symbol_rate=1e3,
        # No pulse_shape set
    )
    # Should NOT raise, but log an error and return an unchanged copy
    result = filtering.matched_filter(sig)
    assert result is not sig
    assert bool(xp.all(result.samples == sig.samples))
