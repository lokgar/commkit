"""Tests for commkit._array (shape and validation helpers)."""

import numpy as np
import pytest

from commkit import _array


class Unconvertible:
    """A foreign array-like: exposes ``__array__`` but is not NumPy/CuPy."""

    def __array__(self):
        raise TypeError("Cannot convert to array")


class TestValidationHelpers:
    """Array validation and input coercion _array."""

    def test_validate_array(self, xp):
        """Verify array validation: None passthrough, list conversion, complex_only, error paths."""
        assert _array.validate_array(None) is None

        arr = _array.validate_array([1, 2, 3])
        assert isinstance(arr, (xp.ndarray, np.ndarray))

        arr_c = _array.validate_array(xp.array([1, 2], dtype=float), complex_only=True)
        assert xp.iscomplexobj(arr_c)

        with pytest.raises(ValueError, match="Expected numeric array"):
            _array.validate_array("not an array")

        with pytest.raises(ValueError, match="Expected numeric array"):
            _array.validate_array(np.array(["a", "b"]))

    def test_validate_array_complex_only(self, xp, xpt):
        """Verify complex_only flag zero-extends the imaginary part."""
        arr = xp.array([1, 2, 3], dtype=float)
        out = _array.validate_array(arr, complex_only=True)
        assert xp.iscomplexobj(out)
        xpt.assert_array_equal(out.real, arr)
        xpt.assert_array_equal(out.imag, 0)

    def test_validate_array_exception(self, xp):
        """Input NumPy cannot convert (a ragged list) raises ValueError."""
        with pytest.raises(ValueError, match="Could not convert"):
            _array.validate_array([[1.0], [1.0, 2.0]])

    def test_validate_array_rejects_foreign_arrays(self, xp):
        """Objects exposing an array protocol but not NumPy/CuPy raise TypeError."""
        with pytest.raises(TypeError, match="from_dlpack"):
            _array.validate_array(Unconvertible())


class TestShapeHelpers:
    """Dimensional promotion, squeezing, channel broadcasting, and layout round-trips."""

    def test_as_2d_promotes_siso(self, xp, xpt):
        """as_2d: (N,) -> (1, N) with was_1d=True."""
        x = xp.asarray(np.arange(8.0))
        x2, was_1d = _array.as_2d(x)
        assert was_1d is True
        assert x2.shape == (1, 8)
        xpt.assert_allclose(x2[0], x)

    def test_as_2d_passes_mimo_through_without_copy(self, xp):
        """as_2d: (C, N) is returned as the same object (no copy, no promotion)."""
        x = xp.asarray(np.arange(12.0).reshape(3, 4))
        x2, was_1d = _array.as_2d(x)
        assert was_1d is False
        assert x2 is x

    @pytest.mark.parametrize("shape", [(), (2, 3, 4)])
    def test_as_2d_rejects_unsupported_ndim(self, xp, shape):
        """as_2d: 0-d and 3-D inputs raise instead of silently passing through."""
        x = xp.asarray(np.zeros(shape))
        with pytest.raises(ValueError, match="SISO|MIMO"):
            _array.as_2d(x, name="samples")

    def test_as_2d_error_message_names_the_variable(self, xp):
        """as_2d: the error quotes the caller's variable name."""
        x = xp.asarray(np.zeros((2, 2, 2)))
        with pytest.raises(ValueError, match="ref_symbols"):
            _array.as_2d(x, name="ref_symbols")

    def test_restore_1d_single_and_multiple(self, xp, xpt):
        """restore_1d: squeezes one or many outputs, bare return for a single one."""
        a = xp.asarray(np.arange(4.0))[None, :]
        b = xp.asarray(np.arange(4.0, 8.0))[None, :]

        out = _array.restore_1d(True, a)
        assert out.shape == (4,)

        oa, ob = _array.restore_1d(True, a, b)
        assert oa.shape == (4,) and ob.shape == (4,)
        xpt.assert_allclose(ob, xp.asarray(np.arange(4.0, 8.0)))

        ka, kb = _array.restore_1d(False, a, b)
        assert ka is a and kb is b

    def test_restore_1d_requires_an_array(self, backend_device):
        """restore_1d: calling with no arrays is a programming error."""
        with pytest.raises(ValueError, match="at least one"):
            _array.restore_1d(True)

    @pytest.mark.parametrize("shape", [(8,), (3, 8)])
    def test_as_2d_restore_1d_round_trip(self, xp, xpt, shape):
        """as_2d + restore_1d is the identity for both layouts."""
        x = xp.asarray(np.random.default_rng(0).normal(size=shape))
        x2, was_1d = _array.as_2d(x)
        xpt.assert_allclose(_array.restore_1d(was_1d, x2), x)

    def test_broadcast_channels_shared_and_per_channel(self, xp, xpt):
        """broadcast_channels: (L,) and (1, L) expand; (C, L) passes through."""
        ref = xp.asarray(np.arange(5.0))
        out = _array.broadcast_channels(ref, 3)
        assert out.shape == (3, 5)
        xpt.assert_allclose(out[2], ref)

        out1 = _array.broadcast_channels(ref[None, :], 3)
        assert out1.shape == (3, 5)

        per_ch = xp.asarray(np.arange(15.0).reshape(3, 5))
        assert _array.broadcast_channels(per_ch, 3) is per_ch

    def test_broadcast_channels_rejects_mismatch(self, xp):
        """broadcast_channels: a channel count that is neither C nor 1 raises."""
        ref = xp.asarray(np.zeros((2, 5)))
        with pytest.raises(ValueError, match="channels"):
            _array.broadcast_channels(ref, 3)
        with pytest.raises(ValueError, match="1-D|2-D"):
            _array.broadcast_channels(xp.asarray(np.zeros((2, 2, 5))), 2)

    def test_require_channels(self, xp):
        """require_channels: exact (C, N) passes; SISO and wrong counts raise."""
        x = xp.asarray(np.zeros((2, 16)))
        assert _array.require_channels(x, 2) is x
        with pytest.raises(ValueError, match="2-D"):
            _array.require_channels(xp.asarray(np.zeros(16)), 2)
        with pytest.raises(ValueError, match="2-D"):
            _array.require_channels(xp.asarray(np.zeros((3, 16))), 2, name="samples")

    def test_to_report_scalar(self, xp):
        """to_report_scalar: length-1 -> float, (C,) -> host array, device input OK."""
        single = _array.to_report_scalar(xp.asarray(np.array([3.5])))
        assert isinstance(single, float) and single == 3.5

        multi = _array.to_report_scalar(xp.asarray(np.array([1.0, 2.0])))
        assert isinstance(multi, np.ndarray)
        assert multi.dtype == np.float64
        np.testing.assert_allclose(multi, [1.0, 2.0])

        assert _array.to_report_scalar(xp.asarray(np.float64(2.0))) == 2.0
        assert _array.to_report_scalar(7) == 7.0
