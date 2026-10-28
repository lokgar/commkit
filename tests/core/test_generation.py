"""Tests for generate() and the waveform synthesis primitives."""

from typing import Any

import numpy as np
import pytest

from commkit.coding import LinearBlockCode, encode
from commkit.core import generate, generation
from commkit.filtering import RC, RRC, Rect, SmoothRect
from commkit.mapping import Constellation
from tests.common.conversions import to_numpy
from tests.common.periodic import circular_matched_samples, middle_period, symbol_power


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


class TestPeriodicShaping:
    """shape_pulse/generate(periodic=True): one period of a looped waveform."""

    @pytest.mark.parametrize(
        "pulse, sps, num_symbols, reps",
        [
            (RRC(0.1), 4, 64, 3),
            (Rect(0.5), 4, 32, 3),  # even number of taps
            (RRC(0.1).taps(2), 2, 40, 3),  # raw taps
            (RRC(0.1, span=32), 2, 5, 9),  # filter longer than the record
        ],
    )
    def test_equals_middle_of_tiled_linear_shaping(
        self, xp: Any, pulse: Any, sps: int, num_symbols: int, reps: int
    ) -> None:
        rng = np.random.default_rng(0)
        syms = rng.standard_normal((2, num_symbols)) + 1j * rng.standard_normal(
            (2, num_symbols)
        )
        syms = xp.asarray(syms.astype(np.complex64))
        out = generation.shape_pulse(syms, sps=sps, pulse=pulse, periodic=True)
        expected = middle_period(
            lambda s: generation.shape_pulse(s, sps=sps, pulse=pulse), syms, reps=reps
        )
        assert out.dtype == xp.complex64
        assert out.shape == (2, num_symbols * sps)
        np.testing.assert_allclose(
            to_numpy(out), symbol_power(expected, sps), atol=2e-6
        )

    def test_unshaped_is_unchanged(self, xp: Any, xpt: Any) -> None:
        syms = xp.asarray(np.array([1, -1, 1j, -1j], dtype=np.complex64))
        xpt.assert_allclose(
            generation.shape_pulse(syms, sps=2, periodic=True),
            generation.shape_pulse(syms, sps=2),
        )

    def test_edge_symbols_recovered_like_interior(self) -> None:
        """A circular matched filter recovers the first and last symbols of a
        periodic record as cleanly as the interior; not so for a linear one."""
        pulse = RRC(0.1, span=32)
        errors = {}
        for periodic in (False, True):
            sig = generate(
                Constellation.qam(16),
                512,
                symbol_rate=1e6,
                sps=4,
                pulse=pulse,
                periodic=periodic,
                rng=0,
            )
            y = circular_matched_samples(sig.samples, pulse.taps(4), 4)
            r = sig.reference.symbols
            errors[periodic] = np.abs(y * np.vdot(r, r) / np.vdot(r, y) - r)
        assert errors[True].max() < 0.02
        assert errors[False][0] > 0.3

    def test_generate_reference_unchanged(self) -> None:
        kw = dict(symbol_rate=1e6, sps=2, pulse=RRC(0.1), num_channels=2, rng=3)
        a = generate(Constellation.qam(16), 256, **kw)
        b = generate(Constellation.qam(16), 256, periodic=True, **kw)
        np.testing.assert_array_equal(a.reference.bits, b.reference.bits)
        assert b.samples.shape == a.samples.shape
        assert np.mean(np.abs(b.samples) ** 2) == pytest.approx(0.5, rel=1e-5)


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
            (dict(device="tpu"), ValueError),
        ],
    )
    def test_invalid_arguments(self, kwargs, error) -> None:
        args = dict(constellation=Constellation.qam(4), num_symbols=10, symbol_rate=1.0)
        with pytest.raises(error):
            generate(**{**args, **kwargs})


class TestGenerateDevice:
    """generate(..., device=): built on the device, same data as on the CPU."""

    @pytest.mark.parametrize(
        "constellation", [Constellation.qam(16), Constellation.qam(16).shaped(nu=0.1)]
    )
    def test_matches_cpu(self, backend_device: str, xp: Any, constellation) -> None:
        kw = dict(symbol_rate=1e6, sps=2, pulse=RRC(0.1), num_channels=2, rng=5)
        ref = generate(constellation, 1000, **kw)
        sig = generate(constellation, 1000, device=backend_device.upper(), **kw)
        for arr in (sig.samples, sig.reference.symbols, sig.reference.bits):
            assert isinstance(arr, xp.ndarray)
        np.testing.assert_array_equal(to_numpy(sig.reference.bits), ref.reference.bits)
        np.testing.assert_array_equal(
            to_numpy(sig.reference.symbols), ref.reference.symbols
        )
        assert sig.samples.dtype == ref.samples.dtype
        np.testing.assert_allclose(to_numpy(sig.samples), ref.samples, atol=1e-5)


class TestGenerateCoded:
    """generate(..., code=): information bits encoded, the coded bits mapped."""

    def test_reference_holds_both_bit_streams(
        self, backend_device: str, xp: Any
    ) -> None:
        code = LinearBlockCode.hamming(3)
        c = Constellation.qam(16)
        sig = generate(
            c,
            700,
            symbol_rate=1e6,
            num_channels=2,
            code=code,
            rng=0,
            device=backend_device,
        )
        ref = sig.reference
        assert ref.code_block == (7, 4)
        assert ref.bits.shape == (2, 2800) and ref.info_bits.shape == (2, 1600)
        assert isinstance(ref.info_bits, xp.ndarray)
        np.testing.assert_array_equal(
            to_numpy(encode(ref.info_bits, code=code)), to_numpy(ref.bits)
        )
        np.testing.assert_array_equal(to_numpy(c.map(ref.bits)), to_numpy(ref.symbols))
        cpu = generate(c, 700, symbol_rate=1e6, num_channels=2, code=code, rng=0)
        np.testing.assert_array_equal(to_numpy(ref.info_bits), cpu.reference.info_bits)

    def test_single_channel_layout(self) -> None:
        sig = generate(
            Constellation.psk(4), 14, symbol_rate=1.0, code=LinearBlockCode.hamming(3)
        )
        assert sig.reference.bits.shape == (28,)
        assert sig.reference.info_bits.shape == (16,)

    def test_uncoded_is_unchanged(self) -> None:
        """Without code the bits are the plain draw, shared as info_bits."""
        sig = generate(Constellation.qam(16), 100, symbol_rate=1e6, rng=7)
        expected = np.random.default_rng(7).integers(0, 2, 400, dtype="int8")
        np.testing.assert_array_equal(sig.reference.bits, expected)
        assert sig.reference.info_bits is sig.reference.bits

    def test_invalid(self) -> None:
        code = LinearBlockCode.hamming(3)
        with pytest.raises(ValueError, match="num_symbols=7 or 14"):
            generate(Constellation.qam(16), 10, symbol_rate=1.0, code=code)
        with pytest.raises(ValueError, match=r"use num_symbols=7\."):
            generate(Constellation.qam(16), 3, symbol_rate=1.0, code=code)
        with pytest.raises(ValueError, match="shaped"):
            generate(
                Constellation.qam(16).shaped(nu=0.1), 7, symbol_rate=1.0, code=code
            )
        with pytest.raises(TypeError, match="code object"):
            generate(Constellation.qam(16), 7, symbol_rate=1.0, code="hamming")
