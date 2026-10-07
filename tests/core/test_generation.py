"""Tests for generate() and the waveform synthesis primitives."""

from typing import Any

import numpy as np
import pytest

from commkit.core import generate, generation
from commkit.filtering import RC, RRC, Rect, SmoothRect
from commkit.mapping import Constellation
from tests.common.conversions import to_numpy


class TestWaveformSynthesis:
    """Tests for zero-stuffing upsampler and pulse-shaping filters."""

    def test_expand_zero_stuffing(self, xp: Any, xpt: Any) -> None:
        """Verify up-sampling by zero-stuffing correctly inserts zeros."""
        data = xp.array([1, 2, 3], dtype="float32")
        factor = 3
        expanded = generation.expand(data, factor=factor)

        expected = xp.array([1, 0, 0, 2, 0, 0, 3, 0, 0], dtype="float32")
        xpt.assert_array_equal(expanded, expected)

    def test_shape_pulse_variants(self, xp: Any) -> None:
        """Verify shape_pulse produces correct lengths for RC and sinc shapes."""
        symbols = xp.array([1, -1, 1, -1], dtype=xp.float32)

        res_rc = generation.shape_pulse(symbols, sps=4, pulse=RC(0.35))
        assert len(res_rc) == 16

        res_sinc = generation.shape_pulse(symbols, sps=4, pulse=RRC(0.0))
        assert len(res_sinc) == 16

        taps = RRC(0.35).taps(4)
        np.testing.assert_allclose(
            to_numpy(generation.shape_pulse(symbols, sps=4, pulse=xp.asarray(taps))),
            to_numpy(generation.shape_pulse(symbols, sps=4, pulse=RRC(0.35))),
        )

    def test_smoothrect_pulse(self, xp: Any) -> None:
        """Verify smoothrect pulse shaping output length."""
        symbols = xp.array([1, 1], dtype=xp.float32)
        res = generation.shape_pulse(
            symbols, sps=8, pulse=SmoothRect(rise_time=0.05, span=4)
        )
        assert len(res) == 16

    def test_shape_pulse_rz_rect(self, xp: Any, xpt: Any) -> None:
        """An RZ rect pulse is on for half of each symbol period."""
        symbols = xp.array([1, -1, 1], dtype=xp.complex64)
        result = generation.shape_pulse(symbols, sps=4, pulse=Rect(0.5))
        assert len(result) == 12
        assert int(xp.sum(xp.abs(result) > 1e-6)) == 6

    def test_shape_pulse_preserves_complex64_dtype(self, xp: Any) -> None:
        """shape_pulse: complex64 symbols -> complex64 waveform."""
        rng = np.random.default_rng(12)
        syms = xp.asarray(
            (rng.standard_normal(100) + 1j * rng.standard_normal(100)).astype(
                np.complex64
            )
        )
        out = generation.shape_pulse(syms, sps=4, pulse=RRC(0.35))
        assert out.dtype == xp.complex64, f"Expected complex64, got {out.dtype}"


class TestGenerate:
    """generate(constellation, ...): reference, waveform and reproducibility."""

    @pytest.mark.parametrize(
        "constellation",
        [
            Constellation.qam(16),
            Constellation.psk(8),
            Constellation.pam(4, unipolar=True),
        ],
    )
    def test_reference_is_exact_constellation_points(self, constellation) -> None:
        sig = generate(constellation, 500, symbol_rate=1e6, num_channels=2, rng=0)
        ref = sig.reference
        assert ref.symbols.shape == (2, 500)
        assert ref.bits.shape == (2, 500 * constellation.bits_per_symbol)
        points = constellation.points.astype(ref.symbols.dtype)
        assert np.isin(ref.symbols, points).all()
        np.testing.assert_array_equal(constellation.map(ref.bits), ref.symbols)

    def test_unshaped_waveform_at_one_sps_is_scaled_reference(self) -> None:
        sig = generate(Constellation.qam(16), 1000, symbol_rate=1e6, rng=1)
        scale = sig.samples / sig.reference.symbols
        np.testing.assert_allclose(scale, scale[0], rtol=1e-6)
        assert np.mean(np.abs(sig.samples) ** 2) == pytest.approx(1.0, rel=1e-6)

    def test_shaped_draws_follow_pmf(self) -> None:
        c = Constellation.qam(16).shaped(nu=0.1)
        sig = generate(c, 200_000, symbol_rate=1e6, rng=2)
        idx = np.argmin(np.abs(sig.reference.symbols[:, None] - c.points), axis=1)
        freq = np.bincount(idx, minlength=16) / idx.size
        np.testing.assert_allclose(freq, c.pmf, atol=4e-3)
        np.testing.assert_array_equal(c.map(sig.reference.bits), sig.reference.symbols)

    def test_int_seed_matches_default_rng_bits(self) -> None:
        """An int rng draws the bits default_rng(seed).integers would."""
        sig = generate(Constellation.qam(16), 100, symbol_rate=1e6, rng=7)
        expected = np.random.default_rng(7).integers(0, 2, 400, dtype="int8")
        np.testing.assert_array_equal(sig.reference.bits, expected)

    def test_generator_rng_and_cpu_output(self) -> None:
        gen = np.random.default_rng(3)
        a = generate(Constellation.psk(4), 64, symbol_rate=1e6, rng=gen)
        b = generate(Constellation.psk(4), 64, symbol_rate=1e6, rng=gen)
        assert not np.array_equal(a.reference.bits, b.reference.bits)
        assert isinstance(a.samples, np.ndarray)

    def test_pulse_object_and_raw_taps_agree(self) -> None:
        kw = dict(symbol_rate=1e6, sps=4, rng=4)
        a = generate(Constellation.qam(4), 200, pulse=RRC(0.2), **kw)
        b = generate(Constellation.qam(4), 200, pulse=RRC(0.2).taps(4), **kw)
        np.testing.assert_array_equal(a.samples, b.samples)
        assert a.pulse == RRC(0.2)
        assert b.pulse is None
        assert a.sampling_rate == 4e6

    @pytest.mark.parametrize(
        "kwargs, error",
        [
            (dict(constellation="qam"), TypeError),
            (dict(num_symbols=0), ValueError),
            (dict(num_channels=1.5), ValueError),
            (dict(sps=2.5), ValueError),
        ],
    )
    def test_invalid_arguments(self, kwargs, error) -> None:
        args = dict(constellation=Constellation.qam(4), num_symbols=10, symbol_rate=1.0)
        with pytest.raises(error):
            generate(**{**args, **kwargs})
