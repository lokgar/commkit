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


class TestJaxInterop:
    """Tests for interoperability between CommKit backends and JAX."""

    def test_jax_interop_roundtrip(
        self, backend_device: str, xp: Any, xpt: Any, jax: Any
    ) -> None:
        """Verify interoperability between core backends and JAX using DLPack."""
        import jax.numpy as jnp

        data = xp.array([1.0, 2.0, 3.0])

        jax_arr = backend.to_jax(data)
        assert isinstance(jax_arr, jnp.ndarray)

        back_arr = backend.from_jax(jax_arr)
        if backend_device == "cpu":
            assert isinstance(back_arr, np.ndarray)
        elif backend_device == "gpu":
            assert isinstance(back_arr, (np.ndarray, xp.ndarray))

        xpt.assert_allclose(backend.to_device(back_arr, "cpu"), [1.0, 2.0, 3.0])

    def test_jax_conversions(self, xp: Any, xpt: Any, jax: Any) -> None:
        """Test JAX conversion utilities with real JAX if available."""
        import jax.numpy as jnp

        from commkit import Signal

        arr_np = np.array([1, 2, 3])
        arr_jax = backend.to_jax(arr_np)
        assert isinstance(arr_jax, jnp.ndarray)

        arr_back = backend.from_jax(arr_jax)
        assert isinstance(arr_back, np.ndarray)
        xpt.assert_array_equal(arr_back, arr_np)

        sig = Signal(samples=xp.asarray(arr_np), sampling_rate=1.0, symbol_rate=1.0)
        jax_sig = sig.export_samples_to_jax()
        assert isinstance(jax_sig, jnp.ndarray)

        sig.update_samples_from_jax(jax_sig)
        assert isinstance(sig.samples, xp.ndarray)
        xpt.assert_allclose(sig.samples, xp.asarray(arr_np))

    def test_to_jax_list_and_scalar(self, jax: Any) -> None:
        """Verify to_jax handles list and scalar inputs by converting via jnp.asarray."""
        import jax.numpy as jnp

        result = backend.to_jax([1.0, 2.0, 3.0])
        assert isinstance(result, jnp.ndarray)
        np.testing.assert_allclose(np.asarray(result), [1.0, 2.0, 3.0])

        result_scalar = backend.to_jax(42.0)
        assert isinstance(result_scalar, jnp.ndarray)
        assert float(result_scalar) == 42.0

    def test_to_jax_explicit_device(self, jax: Any) -> None:
        """Verify to_jax with explicit device placement places the array on requested device."""
        import jax.numpy as jnp

        result = backend.to_jax(np.array([1.0, 2.0]), device="cpu")
        assert isinstance(result, jnp.ndarray)
        assert result.device.platform == "cpu"

        with pytest.raises(ValueError, match="not available"):
            backend.to_jax(np.array([1.0]), device="tpu")
