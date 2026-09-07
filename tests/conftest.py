"""Configuration and shared fixtures for the commkit test suite.

This module provides the `backend_device` and `xp` fixtures, allowing tests to run
transparently on both CPU (NumPy) and GPU (CuPy) backends.

The `--device` CLI option (cpu | gpu | all) controls which backends are exercised.
The default is set to "all" in pyproject.toml [tool.pytest.ini_options] addopts.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from commkit import backend

try:
    import cupy as cp

    _CUPY_AVAILABLE = True
except ImportError:
    cp = None
    _CUPY_AVAILABLE = False


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "gpu_only: mark test as requiring a GPU backend")
    config.addinivalue_line("markers", "cpu_only: mark test as CPU-only")
    config.addinivalue_line(
        "markers",
        "requires_kernel(name): mark test as requiring a specific CUDA kernel",
    )


def pytest_addoption(parser):
    """Add custom CLI options for device selection."""
    parser.addoption(
        "--device",
        action="store",
        default="cpu",
        help="Device to run tests on: cpu, gpu, or all",
    )


def pytest_generate_tests(metafunc):
    """Parametrize the backend_device fixture based on the --device option and markers."""
    if "backend_device" in metafunc.fixturenames:
        device_opt = metafunc.config.getoption("--device")
        is_gpu_only = bool(metafunc.definition.get_closest_marker("gpu_only"))
        is_cpu_only = bool(metafunc.definition.get_closest_marker("cpu_only"))

        if is_gpu_only:
            params = ["gpu"]
        elif is_cpu_only:
            params = ["cpu"]
        elif device_opt == "all":
            params = ["cpu", "gpu"]
        elif device_opt == "gpu":
            params = ["gpu"]
        else:
            params = ["cpu"]

        metafunc.parametrize("backend_device", params, indirect=True)


def pytest_collection_modifyitems(config, items):
    """Filter items based on --device selection so incompatible tests are deselected rather than skipped."""
    device_opt = config.getoption("--device")
    if device_opt == "cpu":
        items[:] = [
            item
            for item in items
            if not (
                item.get_closest_marker("gpu_only")
                or item.get_closest_marker("requires_kernel")
            )
        ]
    elif device_opt == "gpu":
        items[:] = [item for item in items if not item.get_closest_marker("cpu_only")]


@pytest.fixture(scope="session", autouse=True)
def _ensure_jax_precision():
    """Ensure JAX uses float64/complex128 precision across the entire test suite."""
    from tests.common.conversions import ensure_jax_x64

    ensure_jax_x64()


@pytest.fixture(autouse=True)
def _autoclose_figures():
    """Ensure all matplotlib figures are closed after each test to prevent resource leaks."""
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


@pytest.fixture
def jax():
    """Fixture providing the JAX module, cleanly skipping if not installed."""
    return pytest.importorskip("jax", reason="JAX not installed")


@pytest.fixture
def backend_device(request):
    """
    Fixture that returns the current backend device name.

    Skips GPU tests if CuPy is not available or functional.
    Forces CPU mode when device is 'cpu' to ensure isolation.

    Parameters
    ----------
    request : _pytest.fixtures.FixtureRequest
        The request object for the fixture.

    Returns
    -------
    str
        One of {"cpu", "gpu"}.
    """
    device = request.param
    device_opt = request.config.getoption("--device")
    if device == "gpu":
        if device_opt == "cpu":
            pytest.skip("Test requires GPU, but --device=cpu was selected")
        backend.use_cpu_only(False)
        if not _CUPY_AVAILABLE:
            pytest.skip("CuPy not installed, skipping GPU tests")
        try:
            # Aggressive check for a functional GPU context
            cp.zeros(1)
            try:
                cp.random.randn(1)
            except ImportError:
                raise
        except Exception as e:
            pytest.skip(f"CuPy installed but not functional (missing libs?): {e}")

    elif device == "cpu":
        # Force CPU to prevent accidental GPU usage in "cpu" tests
        backend.use_cpu_only(True)

    marker = request.node.get_closest_marker("requires_kernel")
    if marker:
        kernel_name = marker.args[0] if marker.args else None
        from tests.common.kernel_utils import skip_unless_kernel_available

        skip_unless_kernel_available(kernel_name, backend_device=device)

    try:
        yield device
    finally:
        # Always restore default state so later tests are not affected
        backend.use_cpu_only(False)


@pytest.fixture
def xp(backend_device):
    """
    Fixture that returns the array module for the current backend.

    Returns
    -------
    module
        Either `numpy` or `cupy`.
    """
    if backend_device == "cpu":
        return np
    elif backend_device == "gpu":
        return cp
    return np


@pytest.fixture
def xpt(backend_device):
    """
    Fixture that returns the testing module for the current backend.

    Provides backend-aware assertion functions like ``assert_allclose`` and
    ``assert_array_equal``.  Using this fixture instead of bare
    ``assert xp.allclose(...)`` gives rich failure messages (mismatched
    elements, max diff, shapes) and avoids implicit device-to-host transfers
    when running on GPU.

    Returns
    -------
    module
        Either ``numpy.testing`` (CPU) or ``cupy.testing`` (GPU).
    """
    if backend_device == "gpu":
        import cupy.testing as cpt

        return cpt
    import numpy.testing as npt

    return npt
