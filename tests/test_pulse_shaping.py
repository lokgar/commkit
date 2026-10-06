"""Tests for pulse shaping, matched filtering, and filter tap generation in Signal objects."""

import dataclasses
from typing import Any

import numpy as np
import pytest

from commkit import filtering, generate_pam, generate_qam
from commkit.core import Signal


class TestShapingFilterTaps:
    """Tests for automatic shaping filter tap generation from Signal metadata."""

    def test_signal_pulse_params(self, xp: Any) -> None:
        """Verify pulse shaping parameters (e.g. rolloff) are correctly stored and utilized."""
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

        taps = filtering.shaping_filter_taps(sig)
        assert taps is not None
        assert len(taps) > 0

    def test_rzpam_pulse_params(self, xp: Any) -> None:
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

    def test_rz_rect_taps_length(self, xp: Any, xpt: Any) -> None:
        """Verify that RZ rectangular pulse taps have the correct half-symbol length."""
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

    def test_rect_pulse_taps(self, xp: Any, xpt: Any) -> None:
        """Verify that standard rectangular pulse shaping produces all-ones taps."""
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

    def test_gaussian_shaping_filter_taps(self, xp: Any) -> None:
        """shaping_filter_taps for gaussian pulse shape returns valid taps."""
        sig = Signal(
            samples=xp.ones(40, dtype="complex64"),
            sampling_rate=4e3,
            symbol_rate=1e3,
            pulse=filtering.Gaussian(fwhm=0.5, span=4),
        )
        taps = filtering.shaping_filter_taps(sig)
        assert taps is not None
        assert len(taps) > 0

    def test_rc_shaping_filter_taps(self, xp: Any) -> None:
        """shaping_filter_taps for rc pulse shape returns valid taps."""
        sig = Signal(
            samples=xp.ones(40, dtype="complex64"),
            sampling_rate=4e3,
            symbol_rate=1e3,
            pulse=filtering.RC(0.5, span=4),
        )
        taps = filtering.shaping_filter_taps(sig)
        assert taps is not None
        assert len(taps) > 0

    def test_unknown_pulse_shape(self, xp: Any) -> None:
        """Verify unsupported or missing pulse shapes raise ValueError."""
        sig = Signal(samples=[1, 2], sampling_rate=10, symbol_rate=5)
        with pytest.raises(ValueError, match="No pulse shape defined"):
            filtering.shaping_filter_taps(sig)


class TestMatchedFilterAuto:
    """Tests for matched filtering with auto-derived taps."""

    def test_matched_filter_auto_taps(self, xp: Any) -> None:
        """Verify that matched_filter correctly auto-generates and applies taps."""
        sig = generate_pam(
            order=2,
            unipolar=False,
            num_symbols=100,
            sps=4,
            symbol_rate=1e3,
            pulse_shape="rrc",
        )
        sig_before = sig.clone()
        sig = filtering.matched_filter(sig)
        assert not xp.allclose(sig.samples, sig_before.samples)

    def test_matched_filter_logs_error_for_no_pulse_shape(self, xp: Any) -> None:
        """matched_filter() with no pulse_shape returns an unchanged copy."""
        sig = Signal(
            samples=xp.ones(10, dtype="complex64"), sampling_rate=4e3, symbol_rate=1e3
        )
        result = filtering.matched_filter(sig)
        assert result is not sig
        assert bool(xp.all(result.samples == sig.samples))


class TestPulseObjects:
    """Pulse value objects: delegation, definitions, validation."""

    @pytest.mark.parametrize(
        "pulse, expected",
        [
            (filtering.RRC(0.2), lambda: filtering.rrc_taps(4, rolloff=0.2, span=10)),
            (filtering.RC(0.3, span=6), lambda: filtering.rc_taps(4, 0.3, 6)),
            (
                filtering.Gaussian(0.8),
                lambda: filtering.gaussian_taps(4, span=10, duty_cycle=0.8),
            ),
            (filtering.Rect(0.5, 0.1), lambda: filtering.rect_taps(4, 0.5, 0.1)),
            (
                filtering.SmoothRect(0.3, 0.5),
                lambda: filtering.smoothrect_taps(4, 10, 0.3, 0.5),
            ),
        ],
    )
    def test_taps_match_design_functions(self, pulse, expected) -> None:
        np.testing.assert_array_equal(pulse.taps(4), np.asarray(expected()))

    def test_rc_has_zero_isi(self) -> None:
        sps = 8
        h = filtering.RC(0.25).taps(sps)
        c = len(h) // 2
        samples = h[c % sps :: sps]
        others = np.delete(samples, c // sps)
        assert np.max(np.abs(others)) < 1e-12 * h[c]

    def test_rrc_matched_pair_has_near_zero_isi(self) -> None:
        sps = 8
        h = filtering.RRC(0.25, span=20).taps(sps)
        g = np.convolve(h, h)
        c = len(g) // 2
        others = np.delete(g[c % sps :: sps], c // sps)
        assert np.max(np.abs(others)) < 5e-3 * g[c]

    @pytest.mark.parametrize("fwhm", [0.5, 1.0, 1.5])
    def test_gaussian_width_is_fwhm(self, fwhm) -> None:
        sps = 256
        h = filtering.Gaussian(fwhm).taps(sps)
        width = np.count_nonzero(h >= h.max() / 2) / sps
        assert width == pytest.approx(fwhm, abs=2 / sps)

    @pytest.mark.parametrize("rise_time", [0.1, 0.22, 0.4])
    def test_smoothrect_rise_time_is_10_to_90(self, rise_time) -> None:
        sps = 256
        h = filtering.SmoothRect(rise_time).taps(sps)
        edge = (h / h.max())[: len(h) // 2]
        rise = (np.argmax(edge >= 0.9) - np.argmax(edge >= 0.1)) / sps
        assert rise == pytest.approx(rise_time, abs=3 / sps)

    def test_rect_width_is_duty_cycle(self) -> None:
        assert len(filtering.Rect(0.5).taps(8)) == 4
        np.testing.assert_array_equal(filtering.Rect().taps(4), np.ones(4))

    def test_rect_rejects_fractional_sps(self) -> None:
        with pytest.raises(ValueError):
            filtering.Rect().taps(2.5)

    @pytest.mark.parametrize(
        "factory",
        [
            lambda: filtering.RRC(1.5),
            lambda: filtering.RC(-0.1),
            lambda: filtering.RRC(0.1, span=0),
            lambda: filtering.RRC(0.1, span=2.5),
            lambda: filtering.Gaussian(0.0),
            lambda: filtering.Rect(0.0),
            lambda: filtering.Rect(1.2),
            lambda: filtering.Rect(0.5, rise_time=0.3),
            lambda: filtering.SmoothRect(0.0),
            lambda: filtering.SmoothRect(0.2, duty_cycle=2.0),
        ],
    )
    def test_validation(self, factory) -> None:
        with pytest.raises(ValueError):
            factory()

    def test_value_semantics(self) -> None:
        assert filtering.RRC(0.1) == filtering.RRC(0.1, span=10)
        assert filtering.RRC(0.1) != filtering.RC(0.1)
        assert len({filtering.RRC(0.1), filtering.RRC(0.1)}) == 1
        with pytest.raises(dataclasses.FrozenInstanceError):
            filtering.RRC(0.1).rolloff = 0.2  # type: ignore[misc]
        assert isinstance(filtering.SmoothRect(), filtering.Pulse)
