"""Tests for helpers routines (normalization, interpolation, random generation)."""

import numpy as np
import pytest

from commkit import helpers
from commkit.core import Signal


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


def test_normalize_dac_peak(backend_device, xp):
    """dac_peak mode normalizes by max(peak_|Re|, peak_|Im|), not the complex envelope.

    Unlike "peak", this brings the dominant I/Q *component* to 1.0 rather
    than the complex envelope, so it maximises DAC range utilisation - the
    envelope may exceed 1.0 for points off the I/Q axes (e.g. sqrt(2) for a
    45-degree QAM point).
    """
    # Dominant component is imaginary: peak_|Re|=0.6, peak_|Im|=0.8
    data = xp.array([0.6 + 0.8j, -0.3 + 0.4j, 0.1 - 0.2j])
    norm = helpers.normalize(data, mode="dac_peak")

    assert xp.isclose(xp.max(xp.abs(norm.real)), 0.6 / 0.8)
    assert xp.isclose(xp.max(xp.abs(norm.imag)), 1.0)

    # Per-channel (axis=-1): each row normalized independently.
    data_2d = xp.array([[1.0 + 2.0j, 0.5 + 0.5j], [4.0 + 1.0j, 1.0 + 1.0j]])
    norm_2d = helpers.normalize(data_2d, mode="dac_peak", axis=-1)
    row_max = xp.maximum(
        xp.max(xp.abs(norm_2d.real), axis=-1), xp.max(xp.abs(norm_2d.imag), axis=-1)
    )
    assert xp.allclose(row_max, 1.0)


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


# -----------------------------------------------------------------------------
# ARRAY SHAPE HELPERS
# -----------------------------------------------------------------------------


def test_as_2d_promotes_siso(backend_device, xp, xpt):
    """as_2d: (N,) -> (1, N) with was_1d=True."""
    x = xp.asarray(np.arange(8.0))
    x2, was_1d = helpers.as_2d(x)
    assert was_1d is True
    assert x2.shape == (1, 8)
    xpt.assert_allclose(x2[0], x)


def test_as_2d_passes_mimo_through_without_copy(backend_device, xp):
    """as_2d: (C, N) is returned as the same object (no copy, no promotion)."""
    x = xp.asarray(np.arange(12.0).reshape(3, 4))
    x2, was_1d = helpers.as_2d(x)
    assert was_1d is False
    assert x2 is x


@pytest.mark.parametrize("shape", [(), (2, 3, 4)])
def test_as_2d_rejects_unsupported_ndim(backend_device, xp, shape):
    """as_2d: 0-d and 3-D inputs raise instead of silently passing through."""
    x = xp.asarray(np.zeros(shape))
    with pytest.raises(ValueError, match="SISO|MIMO"):
        helpers.as_2d(x, name="samples")


def test_as_2d_error_message_names_the_variable(backend_device, xp):
    """as_2d: the error quotes the caller's variable name."""
    x = xp.asarray(np.zeros((2, 2, 2)))
    with pytest.raises(ValueError, match="ref_symbols"):
        helpers.as_2d(x, name="ref_symbols")


def test_restore_1d_single_and_multiple(backend_device, xp, xpt):
    """restore_1d: squeezes one or many outputs, bare return for a single one."""
    a = xp.asarray(np.arange(4.0))[None, :]
    b = xp.asarray(np.arange(4.0, 8.0))[None, :]

    out = helpers.restore_1d(True, a)
    assert out.shape == (4,)

    oa, ob = helpers.restore_1d(True, a, b)
    assert oa.shape == (4,) and ob.shape == (4,)
    xpt.assert_allclose(ob, xp.asarray(np.arange(4.0, 8.0)))

    # was_1d False -> untouched
    ka, kb = helpers.restore_1d(False, a, b)
    assert ka is a and kb is b


def test_restore_1d_requires_an_array(backend_device):
    """restore_1d: calling with no arrays is a programming error."""
    with pytest.raises(ValueError, match="at least one"):
        helpers.restore_1d(True)


@pytest.mark.parametrize("shape", [(8,), (3, 8)])
def test_as_2d_restore_1d_round_trip(backend_device, xp, xpt, shape):
    """as_2d + restore_1d is the identity for both layouts."""
    x = xp.asarray(np.random.default_rng(0).normal(size=shape))
    x2, was_1d = helpers.as_2d(x)
    xpt.assert_allclose(helpers.restore_1d(was_1d, x2), x)


def test_broadcast_channels_shared_and_per_channel(backend_device, xp, xpt):
    """broadcast_channels: (L,) and (1, L) expand; (C, L) passes through."""
    ref = xp.asarray(np.arange(5.0))
    out = helpers.broadcast_channels(ref, 3)
    assert out.shape == (3, 5)
    xpt.assert_allclose(out[2], ref)

    out1 = helpers.broadcast_channels(ref[None, :], 3)
    assert out1.shape == (3, 5)

    per_ch = xp.asarray(np.arange(15.0).reshape(3, 5))
    assert helpers.broadcast_channels(per_ch, 3) is per_ch


def test_broadcast_channels_rejects_mismatch(backend_device, xp):
    """broadcast_channels: a channel count that is neither C nor 1 raises."""
    ref = xp.asarray(np.zeros((2, 5)))
    with pytest.raises(ValueError, match="channels"):
        helpers.broadcast_channels(ref, 3)
    with pytest.raises(ValueError, match="1-D|2-D"):
        helpers.broadcast_channels(xp.asarray(np.zeros((2, 2, 5))), 2)


def test_require_channels(backend_device, xp):
    """require_channels: exact (C, N) passes; SISO and wrong counts raise."""
    x = xp.asarray(np.zeros((2, 16)))
    assert helpers.require_channels(x, 2) is x
    with pytest.raises(ValueError, match="2-D"):
        helpers.require_channels(xp.asarray(np.zeros(16)), 2)
    with pytest.raises(ValueError, match="2-D"):
        helpers.require_channels(xp.asarray(np.zeros((3, 16))), 2, name="samples")


def test_to_report_scalar(backend_device, xp):
    """to_report_scalar: length-1 -> float, (C,) -> host array, device input OK."""
    single = helpers.to_report_scalar(xp.asarray(np.array([3.5])))
    assert isinstance(single, float) and single == 3.5

    multi = helpers.to_report_scalar(xp.asarray(np.array([1.0, 2.0])))
    assert isinstance(multi, np.ndarray)
    assert multi.dtype == np.float64
    np.testing.assert_allclose(multi, [1.0, 2.0])

    # 0-d and plain Python scalars collapse to float too.
    assert helpers.to_report_scalar(xp.asarray(np.float64(2.0))) == 2.0
    assert helpers.to_report_scalar(7) == 7.0


def test_shape_helpers_work_on_jax_arrays():
    """as_2d/restore_1d are pure indexing: valid on JAX arrays as well."""
    try:
        import jax.numpy as jnp
    except ImportError:
        pytest.skip("JAX not installed")

    try:
        x = jnp.arange(6.0)
    except RuntimeError as err:  # pragma: no cover - environment dependent
        # JAX and CuPy fight over the same device pool; when CuPy has already
        # claimed it, JAX cannot allocate. Nothing to do with the helpers.
        pytest.skip(f"JAX could not allocate on this device: {err}")

    x2, was_1d = helpers.as_2d(x)
    assert was_1d is True and x2.shape == (1, 6)
    assert helpers.restore_1d(was_1d, x2).shape == (6,)
    assert helpers.broadcast_channels(x, 2, jnp).shape == (2, 6)


# -----------------------------------------------------------------------------
# LINEAR TREND HELPERS
# -----------------------------------------------------------------------------


def test_linear_trend_slope_per_sample(backend_device, xp, xpt):
    """linear_trend_slope: recovers a known per-channel slope in units/sample."""
    n = 512
    idx = np.arange(n, dtype=np.float64)
    y = np.stack([0.25 * idx + 3.0, -0.75 * idx - 11.0])
    slope = helpers.linear_trend_slope(xp.asarray(y))
    xpt.assert_allclose(slope, xp.asarray(np.array([0.25, -0.75])), rtol=1e-9)


def test_linear_trend_slope_with_explicit_axis(backend_device, xp, xpt):
    """linear_trend_slope: a non-uniform x axis gives a slope per unit x."""
    x = np.array([0.0, 1.0, 4.0, 9.0, 16.0])
    y = (2.0 * x + 5.0)[None, :]
    slope = helpers.linear_trend_slope(xp.asarray(y), x=xp.asarray(x))
    xpt.assert_allclose(slope, xp.asarray(np.array([2.0])), rtol=1e-9)


def test_linear_trend_slope_stays_on_device(backend_device, xp):
    """linear_trend_slope: the result is a device array (no implicit transfer)."""
    y = xp.asarray(np.random.default_rng(1).normal(size=(2, 64)))
    assert isinstance(helpers.linear_trend_slope(y), xp.ndarray)


def test_remove_linear_trend_strips_ramp_and_keeps_mean(backend_device, xp, xpt):
    """remove_linear_trend: ramp removed, mean preserved, slope reported."""
    n = 1024
    idx = np.arange(n, dtype=np.float64)
    rng = np.random.default_rng(7)
    fluct = rng.normal(scale=0.01, size=n)
    y = 0.05 * idx + 2.0 + fluct
    y2 = xp.asarray(y[None, :])

    detrended, slope = helpers.remove_linear_trend(y2)
    xpt.assert_allclose(slope, xp.asarray(np.array([0.05])), atol=1e-4)
    # Mean is untouched; the residual is the injected fluctuation.
    assert float(xp.mean(detrended)) == pytest.approx(float(np.mean(y)), abs=1e-9)
    xpt.assert_allclose(
        detrended[0] - float(np.mean(y)), xp.asarray(fluct - fluct.mean()), atol=5e-3
    )


def test_remove_linear_trend_degenerate_length(backend_device, xp):
    """remove_linear_trend: a single-sample record does not divide by zero."""
    y = xp.asarray(np.array([[4.0]]))
    detrended, slope = helpers.remove_linear_trend(y)
    assert np.isfinite(float(slope[0]))
    assert float(detrended[0, 0]) == 4.0


# -----------------------------------------------------------------------------
# SIGNAL UNWRAP/REWRAP
# -----------------------------------------------------------------------------


def test_unwrap_signal_passes_raw_array_through(backend_device, xp):
    """unwrap_signal: array input is returned unchanged with signal=None."""
    x = xp.asarray(np.arange(8.0))
    arr, sig = helpers.unwrap_signal(x)
    assert arr is x
    assert sig is None


def test_unwrap_signal_extracts_samples(backend_device, xp):
    """unwrap_signal: Signal input yields .samples and the originating Signal."""
    data = xp.asarray(np.arange(8, dtype=np.complex64))
    s = Signal(samples=data, sampling_rate=8e9, symbol_rate=1e9)
    arr, sig = helpers.unwrap_signal(s)
    assert arr is s.samples
    assert sig is s


def test_unwrap_signal_alternate_field(backend_device, xp):
    """unwrap_signal: field= reads an attribute other than .samples."""
    data = xp.asarray(np.arange(4, dtype=np.complex64))
    s = Signal(samples=data, sampling_rate=8e9, symbol_rate=1e9)
    s.resolved_symbols = xp.asarray(np.arange(4, dtype=np.complex64) + 1)
    arr, sig = helpers.unwrap_signal(s, field="resolved_symbols")
    assert arr is s.resolved_symbols
    assert sig is s


def test_rewrap_signal_passes_raw_array_through(backend_device, xp):
    """rewrap_signal: sig=None returns the array unchanged."""
    x = xp.asarray(np.arange(8.0))
    out = helpers.rewrap_signal(None, x)
    assert out is x


def test_rewrap_signal_builds_a_copy_with_new_samples(backend_device, xp, xpt):
    """rewrap_signal: returns a Signal copy with .samples replaced, original untouched."""
    data = xp.asarray(np.arange(8, dtype=np.complex64))
    s = Signal(samples=data, sampling_rate=8e9, symbol_rate=1e9)
    result = data * 2

    out = helpers.rewrap_signal(s, result)

    assert isinstance(out, Signal)
    assert out is not s
    xpt.assert_allclose(out.samples, result)
    xpt.assert_allclose(s.samples, data)  # original Signal is untouched


def test_rewrap_signal_applies_metadata_kwargs(backend_device, xp):
    """rewrap_signal: keyword args are set on the returned copy via setattr."""
    data = xp.asarray(np.arange(8, dtype=np.complex64))
    s = Signal(samples=data, sampling_rate=8e9, symbol_rate=1e9)

    out = helpers.rewrap_signal(s, data[::2], sampling_rate=s.symbol_rate)

    assert out.sampling_rate == 1e9
    assert s.sampling_rate == 8e9  # original unaffected


def test_unwrap_rewrap_signal_round_trip(backend_device, xp, xpt):
    """unwrap_signal + rewrap_signal reproduces the array-in/array-out,
    Signal-in/Signal-out contract for both input kinds."""
    data = xp.asarray(np.arange(8, dtype=np.complex64))

    arr, sig = helpers.unwrap_signal(data)
    out = helpers.rewrap_signal(sig, arr * 2)
    assert not isinstance(out, Signal)
    xpt.assert_allclose(out, data * 2)

    s = Signal(samples=data, sampling_rate=8e9, symbol_rate=1e9)
    arr2, sig2 = helpers.unwrap_signal(s)
    out2 = helpers.rewrap_signal(sig2, arr2 * 2)
    assert isinstance(out2, Signal)
    xpt.assert_allclose(out2.samples, data * 2)
