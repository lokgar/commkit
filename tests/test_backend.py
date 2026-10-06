"""Tests for the backend management and device-agnostic dispatch system."""

import warnings
from typing import Any

import numpy as np
import pytest

from commkit import backend

warnings.filterwarnings("ignore", message=".*cupyx.jit.rawkernel is experimental.*")


class TestGetArrayModule:
    """Tests for array module resolution across CPU and GPU backends."""

    def test_numpy_array(self) -> None:
        """Verify that get_array_module correctly identifies NumPy for host data."""
        arr_cpu = np.array([1, 2, 3])
        assert backend.get_array_module(arr_cpu) == np

    def test_list_default(self) -> None:
        """Verify that a Python list defaults to NumPy."""
        assert backend.get_array_module([1, 2, 3]) == np


class TestToDevice:
    """Tests for explicit device transfer via to_device."""

    def test_to_device_transfer(self, backend_device: str, xp: Any, xpt: Any) -> None:
        """Verify data transfer between CPU and the target device."""
        data = np.array([1, 2, 3])
        device_data = backend.to_device(data, backend_device)

        assert isinstance(device_data, xp.ndarray)
        xpt.assert_allclose(backend.to_device(device_data, "cpu"), data)

        if backend_device == "cpu":
            assert backend.get_array_module(device_data) == np
        elif backend_device == "gpu":
            import cupy as cp

            assert backend.get_array_module(device_data) == cp

    @pytest.mark.gpu_only
    def test_to_device_cpu_fetches_gpu_array(self) -> None:
        """to_device(x, "cpu") brings a CuPy array to the host."""
        import cupy as cp

        host = backend.to_device(cp.arange(5), "cpu")
        assert isinstance(host, np.ndarray)
        assert np.array_equal(host, [0, 1, 2, 3, 4])

    def test_to_device_list_input(self) -> None:
        """Verify to_device handles plain list input by converting to ndarray."""
        result = backend.to_device([1, 2, 3], "cpu")
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, [1, 2, 3])

    def test_to_device_errors(self) -> None:
        """Test error paths in to_device for unknown target devices."""
        with pytest.raises(ValueError, match="Unknown device"):
            backend.to_device(np.array([1]), "tpu")


class TestBackendDispatch:
    """Tests for backend.dispatch returning array and scipy modules."""

    def test_dispatch_array(self, backend_device: str, xp: Any) -> None:
        """Verify the dispatch system returns correct array and signal modules."""
        data = np.array([1, 2, 3])
        data_in = backend.to_device(data, backend_device)

        out_data, out_xp, out_sp = backend.dispatch(data_in)

        assert out_xp == xp
        assert isinstance(out_data, xp.ndarray)
        assert hasattr(out_sp, "signal")

    def test_dispatch_list(self) -> None:
        """Test dispatch with list input converting to ndarray."""
        from commkit import multirate

        data, x, s = backend.dispatch([1, 2, 3])
        assert isinstance(data, x.ndarray)
        assert x in (np, getattr(multirate, "cp", None))


class _ForeignArray:
    """Stand-in for a JAX/PyTorch array: DLPack-capable, but not NumPy/CuPy."""

    def __init__(self, data):
        self._a = np.asarray(data)

    def __dlpack__(self, **kwargs):
        return self._a.__dlpack__(**kwargs)

    def __dlpack_device__(self):
        return self._a.__dlpack_device__()


class TestForeignArrays:
    """Arrays from other frameworks are rejected, never silently copied."""

    def test_dispatch_rejects_foreign_array(self) -> None:
        with pytest.raises(TypeError, match="from_dlpack"):
            backend.dispatch(_ForeignArray([1.0, 2.0]))

    def test_dispatch_accepts_python_and_numpy_scalars(self) -> None:
        for value in ([1, 2, 3], (1.0, 2.0), 3.0, np.float32(2.0)):
            data, xp, _ = backend.dispatch(value)
            assert xp is np and isinstance(data, np.ndarray)

    def test_explicit_dlpack_conversion_works(self) -> None:
        data, xp, _ = backend.dispatch(np.from_dlpack(_ForeignArray([1.0, 2.0])))
        assert xp is np
        np.testing.assert_array_equal(data, [1.0, 2.0])
