"""Configuration and shared fixtures for the commkit test suite.

This module provides the `backend_device` and `xp` fixtures, allowing tests to run
transparently on both CPU (NumPy) and GPU (CuPy) backends.

The `--device` CLI option (cpu | gpu | all) controls which backends are exercised.
The default is set to "all" in pyproject.toml [tool.pytest.ini_options] addopts.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from commkit import backend

try:
    import cupy as cp

    _CUPY_AVAILABLE = True
except ImportError:
    cp = None
    _CUPY_AVAILABLE = False


def pytest_addoption(parser):
    """Add custom CLI options for device selection."""
    parser.addoption(
        "--device",
        action="store",
        default="cpu",
        help="Device to run tests on: cpu, gpu, or all",
    )


def pytest_generate_tests(metafunc):
    """Parametrize the backend_device fixture based on the --device option."""
    if "backend_device" in metafunc.fixturenames:
        device_opt = metafunc.config.getoption("--device")
        if device_opt == "all":
            params = ["cpu", "gpu"]
        elif device_opt == "gpu":
            params = ["gpu"]
        else:
            params = ["cpu"]

        metafunc.parametrize("backend_device", params, indirect=True)


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
    if device == "gpu":
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
