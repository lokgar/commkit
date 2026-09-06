"""Tests for helpers routines (normalization, interpolation, random generation)."""

import numpy as np
import pytest

from commkit import helpers


class Unconvertible:
    """Object that raises an exception during np.asarray conversion."""

    def __array__(self):
        raise TypeError("Cannot convert to array")


class TestRandomGeneration:
    """Random bits and symbol generation helpers."""

    def test_random_bits(self, backend_device, xp):
        """Verify random bit generation produces correct length, binary values, and device."""
        bits = helpers.generate_bits(100, seed=42)
        assert len(bits) == 100
        assert xp.all((bits == 0) | (bits == 1))
        assert isinstance(bits, xp.ndarray)

    def test_random_bits_no_seed(self, backend_device, xp):
        """Verify generate_bits without a seed produces a result on the active device."""
        bits = helpers.generate_bits(100)
        assert isinstance(bits, xp.ndarray)
        assert len(bits) == 100

    def test_random_symbols_unipolar(self, backend_device, xp):
        """Verify unipolar flag in generate_symbols produces non-negative values."""
        syms = helpers.generate_symbols(10, "ask", 4, unipolar=True)
        assert xp.all(syms >= 0)


class TestNormalization:
    """Normalization modes and RMS computation."""

    def test_normalize(self, backend_device, xp):
        """Verify peak and average-power normalization modes."""
        data = xp.array([1.0, 2.0, 0.5])
        norm = helpers.normalize(data, mode="peak")
        assert xp.isclose(xp.max(xp.abs(norm)), 1.0)

        norm_power = helpers.normalize(data, mode="average_power")
        assert xp.isclose(float(xp.mean(xp.abs(norm_power) ** 2)), 1.0)

    def test_normalize_peak_complex_envelope(self, backend_device, xp):
        """peak mode normalizes by complex envelope, not per-component I/Q."""
        data = xp.array([0.6 + 0.8j, -0.3 + 0.4j, 0.1 - 0.2j])
        norm = helpers.normalize(data, mode="peak")

        assert xp.isclose(xp.max(xp.abs(norm)), 1.0)
        assert float(xp.max(xp.abs(norm.real))) <= 1.0 + 1e-6
        assert float(xp.max(xp.abs(norm.imag))) <= 1.0 + 1e-6

        rotated = norm * np.exp(1j * np.pi / 4)
        assert float(xp.max(xp.abs(rotated.real))) <= 1.0 + 1e-6
        assert float(xp.max(xp.abs(rotated.imag))) <= 1.0 + 1e-6

    def test_normalize_dac_peak(self, backend_device, xp, xpt):
        """dac_peak mode normalizes by max(peak_|Re|, peak_|Im|), not complex envelope."""
        data = xp.array([0.6 + 0.8j, -0.3 + 0.4j, 0.1 - 0.2j])
        norm = helpers.normalize(data, mode="dac_peak")

        assert xp.isclose(xp.max(xp.abs(norm.real)), 0.6 / 0.8)
        assert xp.isclose(xp.max(xp.abs(norm.imag)), 1.0)

        data_2d = xp.array([[1.0 + 2.0j, 0.5 + 0.5j], [4.0 + 1.0j, 1.0 + 1.0j]])
        norm_2d = helpers.normalize(data_2d, mode="dac_peak", axis=-1)
        row_max = xp.maximum(
            xp.max(xp.abs(norm_2d.real), axis=-1), xp.max(xp.abs(norm_2d.imag), axis=-1)
        )
        xpt.assert_allclose(row_max, 1.0)

    def test_normalize_unity_gain(self, backend_device, xp):
        """unity_gain normalizes by sum of elements."""
        data = xp.array([1.0, 2.0, 3.0])
        norm = helpers.normalize(data, mode="unity_gain")
        assert xp.isclose(xp.sum(norm), 1.0)

    def test_normalize_zeros(self, backend_device, xp):
        """Normalizing an all-zero array returns all zeros without NaN."""
        zeros = xp.zeros(5)
        norm = helpers.normalize(zeros, mode="peak")
        assert xp.all(norm == 0)

    def test_normalize_invalid_mode(self, backend_device, xp):
        """Invalid mode raises ValueError."""
        data = xp.array([1.0, 2.0])
        with pytest.raises(ValueError, match="Unknown normalization mode"):
            helpers.normalize(data, mode="invalid_mode")

    def test_normalize_preserves_float32_dtype(self, backend_device, xp):
        """normalize: float32 input -> float32 output across all modes."""
        x = xp.asarray(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        for mode in ("unity_gain", "unit_energy", "peak", "average_power"):
            out = helpers.normalize(x, mode=mode)
            assert out.dtype == xp.float32, (
                f"mode={mode!r}: expected float32, got {out.dtype}"
            )

    def test_rms_preserves_float32_dtype(self, backend_device, xp):
        """rms: float32 input -> float32 output."""
        x = xp.asarray(np.ones(64, dtype=np.float32))
        out = helpers.rms(x)
        assert out.dtype == xp.float32, f"Expected float32, got {out.dtype}"

    def test_normalize_preserves_complex64_dtype(self, backend_device, xp):
        """normalize: complex64 input -> complex64 output."""
        x = xp.asarray(np.array([1 + 1j, 2 + 2j], dtype=np.complex64))
        for mode in ("unit_energy", "peak", "average_power"):
            out = helpers.normalize(x, mode=mode)
            assert out.dtype == xp.complex64, (
                f"mode={mode!r}: expected complex64, got {out.dtype}"
            )

    def test_rms_axis(self, backend_device, xp, xpt):
        """Verify RMS over all elements and per-row."""
        x = xp.array([[1.0, 1.0], [2.0, 2.0]])
        xpt.assert_allclose(helpers.rms(x), xp.sqrt(2.5))
        xpt.assert_allclose(helpers.rms(x, axis=1), [1.0, 2.0])


class TestParabolicAndMath:
    """Parabolic peak interpolation and dB / linear conversions."""

    def test_parabolic_peak_offset_recovers_known_offset(self, backend_device, xp):
        """A synthetic parabola with a known sub-bin peak must be recovered exactly."""
        k_true = 2.3
        nearest = round(k_true)
        offset_true = k_true - nearest

        def y(k):
            return -((k - k_true) ** 2) + 10.0

        y_prev, y_curr, y_next = y(nearest - 1), y(nearest), y(nearest + 1)
        delta = helpers._parabolic_peak_offset(
            xp.asarray(y_prev), xp.asarray(y_curr), xp.asarray(y_next), xp, log=False
        )
        assert float(delta) == pytest.approx(offset_true, abs=1e-9)

    def test_parabolic_peak_offset_degenerate_denom_returns_zero(
        self, backend_device, xp
    ):
        """A flat triplet must return delta=0, not NaN/Inf."""
        y_prev = xp.asarray(1.0)
        y_curr = xp.asarray(1.0)
        y_next = xp.asarray(1.0)
        delta = helpers._parabolic_peak_offset(y_prev, y_curr, y_next, xp, log=False)
        assert float(delta) == 0.0

    def test_parabolic_peak_offset_log_mode_host_scalars(self):
        """log=True must work on plain host scalars with xp=numpy."""
        delta = helpers._parabolic_peak_offset(0.5, 1.0, 0.6, np, log=True)
        assert isinstance(float(delta), float)
        assert -0.5 <= float(delta) <= 0.5

    def test_db_to_linear_power_vs_amplitude(self, backend_device, xp):
        """power=True uses 10x convention; power=False uses 20x."""
        assert helpers.db_to_linear(10.0, power=True) == pytest.approx(10.0)
        assert helpers.db_to_linear(20.0, power=False) == pytest.approx(10.0)

    def test_linear_to_db_is_inverse_of_db_to_linear(self, backend_device, xp):
        """linear_to_db and db_to_linear are inverses of each other."""
        val = 15.5
        assert helpers.linear_to_db(
            helpers.db_to_linear(val, power=True), power=True
        ) == pytest.approx(val)
        assert helpers.linear_to_db(
            helpers.db_to_linear(val, power=False), power=False
        ) == pytest.approx(val)

    def test_linear_to_db_zero_is_negative_inf_no_warning(self, backend_device, xp):
        """linear_to_db(0) returns -inf cleanly without warning."""
        res = helpers.linear_to_db(0.0)
        assert np.isneginf(res)

    def test_format_si(self, backend_device, xp):
        """Verify SI-prefix formatting for common magnitudes."""
        assert helpers.format_si(None) == "None"
        assert helpers.format_si(0) == "0.00 Hz"
        assert "1.00 MHz" in helpers.format_si(1e6, "Hz")
        assert "500.00 mV" in helpers.format_si(0.5, "V")
        assert "Hz" in helpers.format_si(100)

    def test_zc_mimo_root(self, backend_device, xp):
        """zc_mimo_root assigns distinct roots cycling from base_root in [1, length-1]."""
        from commkit.helpers import zc_mimo_root

        assert zc_mimo_root(0, 1, 13) == 1
        assert zc_mimo_root(1, 1, 13) == 2
        assert zc_mimo_root(2, 1, 13) == 3

        assert zc_mimo_root(0, 10, 13) == 10
        assert zc_mimo_root(1, 10, 13) == 11
        assert zc_mimo_root(2, 10, 13) == 12
        assert zc_mimo_root(3, 10, 13) == 1

        for k in range(12):
            r = zc_mimo_root(k, 1, 13)
            assert 1 <= r <= 12


class TestValidationHelpers:
    """Array validation and input coercion helpers."""

    def test_validate_array(self, backend_device, xp):
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

    def test_validate_array_complex_only(self, backend_device, xp):
        """Verify complex_only flag zero-extends the imaginary part."""
        arr = xp.array([1, 2, 3], dtype=float)
        out = helpers.validate_array(arr, complex_only=True)
        assert xp.iscomplexobj(out)
        assert xp.all(out.real == arr)
        assert xp.all(out.imag == 0)

    def test_validate_array_exception(self, backend_device, xp):
        """Verify the except-block in validate_array raises ValueError for unconvertible input."""
        obj = Unconvertible()
        with pytest.raises(ValueError, match="Could not convert"):
            helpers.validate_array(obj)


class TestShapeHelpers:
    """Dimensional promotion, squeezing, channel broadcasting, and layout round-trips."""

    def test_as_2d_promotes_siso(self, backend_device, xp, xpt):
        """as_2d: (N,) -> (1, N) with was_1d=True."""
        x = xp.asarray(np.arange(8.0))
        x2, was_1d = helpers.as_2d(x)
        assert was_1d is True
        assert x2.shape == (1, 8)
        xpt.assert_allclose(x2[0], x)

    def test_as_2d_passes_mimo_through_without_copy(self, backend_device, xp):
        """as_2d: (C, N) is returned as the same object (no copy, no promotion)."""
        x = xp.asarray(np.arange(12.0).reshape(3, 4))
        x2, was_1d = helpers.as_2d(x)
        assert was_1d is False
        assert x2 is x

    @pytest.mark.parametrize("shape", [(), (2, 3, 4)])
    def test_as_2d_rejects_unsupported_ndim(self, backend_device, xp, shape):
        """as_2d: 0-d and 3-D inputs raise instead of silently passing through."""
        x = xp.asarray(np.zeros(shape))
        with pytest.raises(ValueError, match="SISO|MIMO"):
            helpers.as_2d(x, name="samples")

    def test_as_2d_error_message_names_the_variable(self, backend_device, xp):
        """as_2d: the error quotes the caller's variable name."""
        x = xp.asarray(np.zeros((2, 2, 2)))
        with pytest.raises(ValueError, match="ref_symbols"):
            helpers.as_2d(x, name="ref_symbols")

    def test_restore_1d_single_and_multiple(self, backend_device, xp, xpt):
        """restore_1d: squeezes one or many outputs, bare return for a single one."""
        a = xp.asarray(np.arange(4.0))[None, :]
        b = xp.asarray(np.arange(4.0, 8.0))[None, :]

        out = helpers.restore_1d(True, a)
        assert out.shape == (4,)

        oa, ob = helpers.restore_1d(True, a, b)
        assert oa.shape == (4,) and ob.shape == (4,)
        xpt.assert_allclose(ob, xp.asarray(np.arange(4.0, 8.0)))

        ka, kb = helpers.restore_1d(False, a, b)
        assert ka is a and kb is b

    def test_restore_1d_requires_an_array(self, backend_device):
        """restore_1d: calling with no arrays is a programming error."""
        with pytest.raises(ValueError, match="at least one"):
            helpers.restore_1d(True)

    @pytest.mark.parametrize("shape", [(8,), (3, 8)])
    def test_as_2d_restore_1d_round_trip(self, backend_device, xp, xpt, shape):
        """as_2d + restore_1d is the identity for both layouts."""
        x = xp.asarray(np.random.default_rng(0).normal(size=shape))
        x2, was_1d = helpers.as_2d(x)
        xpt.assert_allclose(helpers.restore_1d(was_1d, x2), x)

    def test_broadcast_channels_shared_and_per_channel(self, backend_device, xp, xpt):
        """broadcast_channels: (L,) and (1, L) expand; (C, L) passes through."""
        ref = xp.asarray(np.arange(5.0))
        out = helpers.broadcast_channels(ref, 3)
        assert out.shape == (3, 5)
        xpt.assert_allclose(out[2], ref)

        out1 = helpers.broadcast_channels(ref[None, :], 3)
        assert out1.shape == (3, 5)

        per_ch = xp.asarray(np.arange(15.0).reshape(3, 5))
        assert helpers.broadcast_channels(per_ch, 3) is per_ch

    def test_broadcast_channels_rejects_mismatch(self, backend_device, xp):
        """broadcast_channels: a channel count that is neither C nor 1 raises."""
        ref = xp.asarray(np.zeros((2, 5)))
        with pytest.raises(ValueError, match="channels"):
            helpers.broadcast_channels(ref, 3)
        with pytest.raises(ValueError, match="1-D|2-D"):
            helpers.broadcast_channels(xp.asarray(np.zeros((2, 2, 5))), 2)

    def test_require_channels(self, backend_device, xp):
        """require_channels: exact (C, N) passes; SISO and wrong counts raise."""
        x = xp.asarray(np.zeros((2, 16)))
        assert helpers.require_channels(x, 2) is x
        with pytest.raises(ValueError, match="2-D"):
            helpers.require_channels(xp.asarray(np.zeros(16)), 2)
        with pytest.raises(ValueError, match="2-D"):
            helpers.require_channels(xp.asarray(np.zeros((3, 16))), 2, name="samples")

    def test_to_report_scalar(self, backend_device, xp):
        """to_report_scalar: length-1 -> float, (C,) -> host array, device input OK."""
        single = helpers.to_report_scalar(xp.asarray(np.array([3.5])))
        assert isinstance(single, float) and single == 3.5

        multi = helpers.to_report_scalar(xp.asarray(np.array([1.0, 2.0])))
        assert isinstance(multi, np.ndarray)
        assert multi.dtype == np.float64
        np.testing.assert_allclose(multi, [1.0, 2.0])

        assert helpers.to_report_scalar(xp.asarray(np.float64(2.0))) == 2.0
        assert helpers.to_report_scalar(7) == 7.0

    def test_shape_helpers_work_on_jax_arrays(self, jax):
        """as_2d/restore_1d are pure indexing: valid on JAX arrays as well."""
        import jax.numpy as jnp

        x = jnp.arange(6.0)
        x2, was_1d = helpers.as_2d(x)
        assert was_1d is True and x2.shape == (1, 6)
        assert helpers.restore_1d(was_1d, x2).shape == (6,)
        assert helpers.broadcast_channels(x, 2, jnp).shape == (2, 6)


class TestLinearTrend:
    """Estimation and removal of linear phase/carrier ramps."""

    def test_linear_trend_slope_per_sample(self, backend_device, xp, xpt):
        """linear_trend_slope: recovers a known per-channel slope in units/sample."""
        n = 512
        idx = np.arange(n, dtype=np.float64)
        y = np.stack([0.25 * idx + 3.0, -0.75 * idx - 11.0])
        slope = helpers.linear_trend_slope(xp.asarray(y))
        xpt.assert_allclose(slope, xp.asarray(np.array([0.25, -0.75])), rtol=1e-9)

    def test_linear_trend_slope_with_explicit_axis(self, backend_device, xp, xpt):
        """linear_trend_slope: a non-uniform x axis gives a slope per unit x."""
        x = np.array([0.0, 1.0, 4.0, 9.0, 16.0])
        y = (2.0 * x + 5.0)[None, :]
        slope = helpers.linear_trend_slope(xp.asarray(y), x=xp.asarray(x))
        xpt.assert_allclose(slope, xp.asarray(np.array([2.0])), rtol=1e-9)

    def test_linear_trend_slope_stays_on_device(self, backend_device, xp):
        """linear_trend_slope: the result is a device array (no implicit transfer)."""
        y = xp.asarray(np.random.default_rng(1).normal(size=(2, 64)))
        assert isinstance(helpers.linear_trend_slope(y), xp.ndarray)

    def test_remove_linear_trend_strips_ramp_and_keeps_mean(
        self, backend_device, xp, xpt
    ):
        """remove_linear_trend: ramp removed, mean preserved, slope reported."""
        n = 1024
        idx = np.arange(n, dtype=np.float64)
        rng = np.random.default_rng(7)
        fluct = rng.normal(scale=0.01, size=n)
        y = 0.05 * idx + 2.0 + fluct
        y2 = xp.asarray(y[None, :])

        detrended, slope = helpers.remove_linear_trend(y2)
        xpt.assert_allclose(slope, xp.asarray(np.array([0.05])), atol=1e-4)
        assert float(xp.mean(detrended)) == pytest.approx(float(np.mean(y)), abs=1e-9)
        xpt.assert_allclose(
            detrended[0] - float(np.mean(y)),
            xp.asarray(fluct - fluct.mean()),
            atol=5e-3,
        )

    def test_remove_linear_trend_degenerate_length(self, backend_device, xp):
        """remove_linear_trend: a single-sample record does not divide by zero."""
        y = xp.asarray(np.array([[4.0]]))
        detrended, slope = helpers.remove_linear_trend(y)
        assert np.isfinite(float(slope[0]))
        assert float(detrended[0, 0]) == 4.0
