"""Tests for commkit.math (normalization, RMS, dB conversions)."""

import numpy as np
import pytest

from commkit import math as ckmath
from commkit.math import _linear_trend_slope, _remove_linear_trend


class TestNormalization:
    """Normalization modes and RMS computation."""

    def test_normalize(self, xp):
        """Verify peak and average-power normalization modes."""
        data = xp.array([1.0, 2.0, 0.5])
        norm = ckmath.normalize(data, mode="peak")
        assert xp.isclose(xp.max(xp.abs(norm)), 1.0)

        norm_power = ckmath.normalize(data, mode="average_power")
        assert xp.isclose(float(xp.mean(xp.abs(norm_power) ** 2)), 1.0)

    def test_normalize_peak_complex_envelope(self, xp):
        """peak mode normalizes by complex envelope, not per-component I/Q."""
        data = xp.array([0.6 + 0.8j, -0.3 + 0.4j, 0.1 - 0.2j])
        norm = ckmath.normalize(data, mode="peak")

        assert xp.isclose(xp.max(xp.abs(norm)), 1.0)
        assert float(xp.max(xp.abs(norm.real))) <= 1.0 + 1e-6
        assert float(xp.max(xp.abs(norm.imag))) <= 1.0 + 1e-6

        rotated = norm * np.exp(1j * np.pi / 4)
        assert float(xp.max(xp.abs(rotated.real))) <= 1.0 + 1e-6
        assert float(xp.max(xp.abs(rotated.imag))) <= 1.0 + 1e-6

    def test_normalize_dac_peak(self, xp, xpt):
        """dac_peak mode normalizes by max(peak_|Re|, peak_|Im|), not complex envelope."""
        data = xp.array([0.6 + 0.8j, -0.3 + 0.4j, 0.1 - 0.2j])
        norm = ckmath.normalize(data, mode="dac_peak")

        assert xp.isclose(xp.max(xp.abs(norm.real)), 0.6 / 0.8)
        assert xp.isclose(xp.max(xp.abs(norm.imag)), 1.0)

        data_2d = xp.array([[1.0 + 2.0j, 0.5 + 0.5j], [4.0 + 1.0j, 1.0 + 1.0j]])
        norm_2d = ckmath.normalize(data_2d, mode="dac_peak", axis=-1)
        row_max = xp.maximum(
            xp.max(xp.abs(norm_2d.real), axis=-1), xp.max(xp.abs(norm_2d.imag), axis=-1)
        )
        xpt.assert_allclose(row_max, 1.0)

    def test_normalize_unity_gain(self, xp):
        """unity_gain normalizes by sum of elements."""
        data = xp.array([1.0, 2.0, 3.0])
        norm = ckmath.normalize(data, mode="unity_gain")
        assert xp.isclose(xp.sum(norm), 1.0)

    def test_normalize_zeros(self, xp, xpt):
        """Normalizing an all-zero array returns all zeros without NaN."""
        zeros = xp.zeros(5)
        norm = ckmath.normalize(zeros, mode="peak")
        xpt.assert_array_equal(norm, 0)

    def test_normalize_invalid_mode(self, xp):
        """Invalid mode raises ValueError."""
        data = xp.array([1.0, 2.0])
        with pytest.raises(ValueError, match="Unknown normalization mode"):
            ckmath.normalize(data, mode="invalid_mode")

    def test_normalize_preserves_float32_dtype(self, xp):
        """normalize: float32 input -> float32 output across all modes."""
        x = xp.asarray(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        for mode in ("unity_gain", "unit_energy", "peak", "average_power"):
            out = ckmath.normalize(x, mode=mode)
            assert out.dtype == xp.float32, (
                f"mode={mode!r}: expected float32, got {out.dtype}"
            )

    def test_rms_preserves_float32_dtype(self, xp):
        """rms: float32 input -> float32 output."""
        x = xp.asarray(np.ones(64, dtype=np.float32))
        out = ckmath.rms(x)
        assert out.dtype == xp.float32, f"Expected float32, got {out.dtype}"

    def test_normalize_preserves_complex64_dtype(self, xp):
        """normalize: complex64 input -> complex64 output."""
        x = xp.asarray(np.array([1 + 1j, 2 + 2j], dtype=np.complex64))
        for mode in ("unit_energy", "peak", "average_power"):
            out = ckmath.normalize(x, mode=mode)
            assert out.dtype == xp.complex64, (
                f"mode={mode!r}: expected complex64, got {out.dtype}"
            )

    def test_rms_axis(self, xp, xpt):
        """Verify RMS over all elements and per-row."""
        x = xp.array([[1.0, 1.0], [2.0, 2.0]])
        xpt.assert_allclose(ckmath.rms(x), xp.sqrt(2.5))
        xpt.assert_allclose(ckmath.rms(x, axis=1), [1.0, 2.0])


class TestDecibels:
    """dB / linear conversions."""

    def test_db_to_linear_power_vs_amplitude(self, xp):
        """power=True uses 10x convention; power=False uses 20x."""
        assert ckmath.db_to_linear(10.0, power=True) == pytest.approx(10.0)
        assert ckmath.db_to_linear(20.0, power=False) == pytest.approx(10.0)

    def test_linear_to_db_is_inverse_of_db_to_linear(self, xp):
        """linear_to_db and db_to_linear are inverses of each other."""
        val = 15.5
        assert ckmath.linear_to_db(
            ckmath.db_to_linear(val, power=True), power=True
        ) == pytest.approx(val)
        assert ckmath.linear_to_db(
            ckmath.db_to_linear(val, power=False), power=False
        ) == pytest.approx(val)

    def test_linear_to_db_zero_is_negative_inf_no_warning(self, xp):
        """linear_to_db(0) returns -inf cleanly without warning."""
        res = ckmath.linear_to_db(0.0)
        assert np.isneginf(res)


class TestLinearTrend:
    """Estimation and removal of linear phase/carrier ramps."""

    def test_linear_trend_slope_per_sample(self, xp, xpt):
        """_linear_trend_slope: recovers a known per-channel slope in units/sample."""
        n = 512
        idx = np.arange(n, dtype=np.float64)
        y = np.stack([0.25 * idx + 3.0, -0.75 * idx - 11.0])
        slope = _linear_trend_slope(xp.asarray(y))
        xpt.assert_allclose(slope, xp.asarray(np.array([0.25, -0.75])), rtol=1e-9)

    def test_linear_trend_slope_with_explicit_axis(self, xp, xpt):
        """_linear_trend_slope: a non-uniform x axis gives a slope per unit x."""
        x = np.array([0.0, 1.0, 4.0, 9.0, 16.0])
        y = (2.0 * x + 5.0)[None, :]
        slope = _linear_trend_slope(xp.asarray(y), x=xp.asarray(x))
        xpt.assert_allclose(slope, xp.asarray(np.array([2.0])), rtol=1e-9)

    def test_linear_trend_slope_stays_on_device(self, xp):
        """_linear_trend_slope: the result is a device array (no implicit transfer)."""
        y = xp.asarray(np.random.default_rng(1).normal(size=(2, 64)))
        assert isinstance(_linear_trend_slope(y), xp.ndarray)

    def test_remove_linear_trend_strips_ramp_and_keeps_mean(self, xp, xpt):
        """_remove_linear_trend: ramp removed, mean preserved, slope reported."""
        n = 1024
        idx = np.arange(n, dtype=np.float64)
        rng = np.random.default_rng(7)
        fluct = rng.normal(scale=0.01, size=n)
        y = 0.05 * idx + 2.0 + fluct
        y2 = xp.asarray(y[None, :])

        detrended, slope = _remove_linear_trend(y2)
        xpt.assert_allclose(slope, xp.asarray(np.array([0.05])), atol=1e-4)
        assert float(xp.mean(detrended)) == pytest.approx(float(np.mean(y)), abs=1e-9)
        xpt.assert_allclose(
            detrended[0] - float(np.mean(y)),
            xp.asarray(fluct - fluct.mean()),
            atol=5e-3,
        )

    def test_remove_linear_trend_degenerate_length(self, xp):
        """_remove_linear_trend: a single-sample record does not divide by zero."""
        y = xp.asarray(np.array([[4.0]]))
        detrended, slope = _remove_linear_trend(y)
        assert np.isfinite(float(slope[0]))
        assert float(detrended[0, 0]) == 4.0
