"""Tests for the backend management and device-agnostic dispatch system."""

import warnings

import numpy as np
import pytest

from commkit import backend

warnings.filterwarnings("ignore", message=".*cupyx.jit.rawkernel is experimental.*")


def test_get_array_module():
    """Verify that get_array_module correctly identifies NumPy for host data."""
    # Test CPU
    arr_cpu = np.array([1, 2, 3])
    assert backend.get_array_module(arr_cpu) == np

    # Test List (should default to NumPy)
    assert backend.get_array_module([1, 2, 3]) == np


def test_to_device(backend_device, xp):
    """Verify data transfer between CPU and the target device."""
    data = np.array([1, 2, 3])

    # Move to target device
    device_data = backend.to_device(data, backend_device)

    assert isinstance(device_data, xp.ndarray)
    assert np.allclose(backend.to_device(device_data, "cpu"), data)

    if backend_device == "cpu":
        assert backend.get_array_module(device_data) == np
    elif backend_device == "gpu":
        # If gpu test runs, we assume cupy is present (checked by fixture)
        import cupy as cp

        assert backend.get_array_module(device_data) == cp


def test_dispatch(backend_device, xp):
    """Verify the dispatch system returns correct array and signal modules."""
    data = np.array([1, 2, 3])

    # Pre-move data to device manually to simulate input from that device
    data_in = backend.to_device(data, backend_device)

    out_data, out_xp, out_sp = backend.dispatch(data_in)

    assert out_xp == xp
    assert isinstance(out_data, xp.ndarray)
    assert hasattr(out_sp, "signal")  # Check if it looks like scipy/cupyx.scipy


def test_cpu_only_toggle():
    """Verify that forcing CPU mode correctly disables GPU detection."""
    # Ensure clean state
    # Save original state to restore
    original_force = backend._FORCE_CPU

    backend.use_cpu_only(False)
    # Check if cupy is physically available on system
    # We can't easily check this without potentially importing cupy if we haven't already,
    # but backend.is_cupy_available() relies on module level _CUPY_AVAILABLE.

    # Force CPU
    backend.use_cpu_only(True)
    assert backend.is_cupy_available() is False

    # Restore
    backend.use_cpu_only(False)
    # If it was available before force, it should be available now.

    # Restore original state
    backend.use_cpu_only(original_force)


def test_get_array_module_robust_under_force_cpu():
    """A CuPy array must resolve to CuPy even under use_cpu_only().

    use_cpu_only() governs default *placement* of new arrays (is_cupy_available()
    -> False), not the module of data that already lives on the GPU. Reporting
    NumPy for a CuPy array would mis-dispatch GPU data into NumPy code paths.
    """
    original_force = backend._FORCE_CPU
    backend.use_cpu_only(False)
    if not backend.is_cupy_available():
        backend.use_cpu_only(original_force)
        pytest.skip("CuPy not available")

    import cupy as cp

    arr = cp.arange(4)
    try:
        backend.use_cpu_only(True)
        # Placement flag flips off...
        assert backend.is_cupy_available() is False
        # ...but type inspection still resolves the actual backend, and
        # dispatch() keeps the array module and scipy module paired.
        assert backend.get_array_module(arr) is cp
        assert backend.get_scipy_module(cp).__name__.startswith("cupyx")
        _, out_xp, out_sp = backend.dispatch(arr)
        assert out_xp is cp
        assert hasattr(out_sp, "signal")
    finally:
        backend.use_cpu_only(original_force)


def test_to_device_cpu_fetches_gpu_array_under_force():
    """to_device(x, "cpu") must bring a CuPy array to host even under force.

    Regression: the CPU path gated the ``.get()`` on ``is_cupy_available()``,
    which use_cpu_only() forces to False, so an existing GPU array fell through
    to ``np.asarray(cupy_array)`` and raised "Implicit conversion ... use .get()".
    The transfer dispatches on the actual array type, like get_array_module, so a
    GPU array stays fetchable regardless of the placement flag.  This off-diagonal
    state (force CPU + a live CuPy array) is the one the device-parametrized
    fixtures cannot reach, since they flip the flag and the array module together.
    """
    original_force = backend._FORCE_CPU
    backend.use_cpu_only(False)
    if not backend.is_cupy_available():
        backend.use_cpu_only(original_force)
        pytest.skip("CuPy not available")

    import cupy as cp

    arr = cp.arange(5)
    try:
        backend.use_cpu_only(True)
        assert backend.is_cupy_available() is False
        host = backend.to_device(arr, "cpu")
        assert isinstance(host, np.ndarray)
        assert np.array_equal(host, [0, 1, 2, 3, 4])
    finally:
        backend.use_cpu_only(original_force)


def test_jax_interop(backend_device, xp):
    """Verify interoperability between core backends and JAX using DLPack."""
    try:
        import jax.numpy as jnp
    except ImportError:
        pytest.skip("JAX not installed")

    # Create data on backend device
    data = xp.array([1.0, 2.0, 3.0])

    # To JAX
    if backend_device == "cpu":
        # Ensure CPU only is enforced within this test context
        backend.use_cpu_only(True)

    jax_arr = backend.to_jax(data)
    assert isinstance(jax_arr, jnp.ndarray)

    # From JAX
    # Note: from_jax might return numpy or cupy depending on context/availability
    # If on GPU, we expect JAX to likely stay on GPU if jax[cuda] is working,
    # but strictly from_jax should return backend-compatible array

    back_arr = backend.from_jax(jax_arr)

    if backend_device == "cpu":
        assert isinstance(back_arr, np.ndarray)
    elif backend_device == "gpu":
        # Ideally it should be cupy, but fallback to numpy is valid if DLPack fails or JAX is CPU-only
        assert isinstance(back_arr, (np.ndarray, xp.ndarray))

    # Test round trip values
    assert np.allclose(backend.to_device(back_arr, "cpu"), [1.0, 2.0, 3.0])

    # Reset configuration to avoid polluting other tests if fixture doesn't handle it
    backend.use_cpu_only(False)


def test_use_cpu_only(xp):
    """Test use_cpu_only forces CPU backend."""
    from commkit import backend

    # Save original state
    original_force = backend._FORCE_CPU

    try:
        backend.use_cpu_only(True)
        assert backend.is_cupy_available() is False
        assert backend.get_array_module(np.array([1])) == np

        # Should raise ImportError if we try to force GPU
        with pytest.raises(ImportError):
            backend.to_device(np.array([1]), "gpu")

    finally:
        # Restore state
        backend.use_cpu_only(original_force)


def test_to_device_errors(xp):
    """Test error paths in to_device."""
    from commkit import backend

    with pytest.raises(ValueError, match="Unknown device"):
        backend.to_device(np.array([1]), "tpu")


def test_dispatch_list(xp):
    """Test dispatch with list input."""
    # Should convert list to array
    from commkit import backend, multirate

    data, x, s = backend.dispatch([1, 2, 3])
    assert isinstance(data, x.ndarray)
    assert x in (np, getattr(multirate, "cp", None))  # generic check


def test_jax_conversions(backend_device, xp, xpt):
    """Test JAX conversion utilities with real JAX if available."""
    try:
        import jax.numpy as jnp
    except ImportError:
        pytest.skip("JAX not installed")

    from commkit import Signal, backend

    # 1. Numpy -> JAX
    arr_np = np.array([1, 2, 3])
    arr_jax = backend.to_jax(arr_np)
    assert isinstance(arr_jax, jnp.ndarray)

    # 2. JAX -> Numpy
    arr_back = backend.from_jax(arr_jax)
    assert isinstance(arr_back, np.ndarray)
    assert np.allclose(arr_back, arr_np)

    # 3. Signal methods
    sig = Signal(samples=arr_np, sampling_rate=1.0, symbol_rate=1.0)
    jax_sig = sig.export_samples_to_jax()
    assert isinstance(jax_sig, jnp.ndarray)

    sig.update_samples_from_jax(jax_sig)
    assert isinstance(sig.samples, xp.ndarray)
    xpt.assert_allclose(sig.samples, xp.asarray(arr_np))


def test_to_device_list_input(xp):
    """Verify to_device handles plain list input by converting to ndarray."""
    result = backend.to_device([1, 2, 3], "cpu")
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, [1, 2, 3])


def test_to_jax_list_and_scalar(xp):
    """Verify to_jax handles list and scalar inputs by converting via jnp.asarray."""
    try:
        import jax.numpy as jnp
    except ImportError:
        pytest.skip("JAX not installed")

    # List input -> general case (jnp.asarray)
    result = backend.to_jax([1.0, 2.0, 3.0])
    assert isinstance(result, jnp.ndarray)
    assert np.allclose(np.asarray(result), [1.0, 2.0, 3.0])

    # Scalar input
    result_scalar = backend.to_jax(42.0)
    assert isinstance(result_scalar, jnp.ndarray)
    assert float(result_scalar) == 42.0


def test_to_jax_explicit_device(xp):
    """Verify to_jax with explicit device placement places the array on the requested device."""
    try:
        import jax.numpy as jnp
    except ImportError:
        pytest.skip("JAX not installed")

    # CPU device should always be available
    result = backend.to_jax(np.array([1.0, 2.0]), device="cpu")
    assert isinstance(result, jnp.ndarray)
    assert result.device.platform == "cpu"

    # Invalid device should raise
    with pytest.raises(ValueError, match="not available"):
        backend.to_jax(np.array([1.0]), device="tpu")
