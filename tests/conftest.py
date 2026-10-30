"""Configuration and shared fixtures for the commkit test suite.

This module provides the `backend_device` and `xp` fixtures, allowing tests to run
transparently on both CPU (NumPy) and GPU (CuPy) backends.

The `--device` CLI option (cpu | gpu | all) controls which backends are exercised.
The default is set to "all" in pyproject.toml [tool.pytest.ini_options] addopts.
"""

import logging
import re

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")


try:
    import cupy as cp

    _CUPY_AVAILABLE = True
except ImportError:
    cp = None
    _CUPY_AVAILABLE = False


def pytest_configure(config):
    """Register custom markers and the PrecisionWarning error filter."""
    # The library's own paths must never reduce double input silently.  Added
    # here rather than in pyproject's filterwarnings: pytest resolves those
    # categories while reading its config, which would import commkit before
    # pytest-cov starts and hide every module-level line from coverage.
    config.addinivalue_line("filterwarnings", "error::commkit.backend.PrecisionWarning")
    config.addinivalue_line("markers", "gpu_only: mark test as requiring a GPU backend")
    config.addinivalue_line("markers", "cpu_only: mark test as CPU-only")
    config.addinivalue_line(
        "markers", "slow: long statistical validation, run with --run-slow"
    )
    config.addinivalue_line(
        "markers",
        "allow_warnings: the test provokes a commkit WARNING log on purpose",
    )


def pytest_addoption(parser):
    """Add custom CLI options for device selection."""
    parser.addoption(
        "--device",
        action="store",
        default="cpu",
        help="Device to run tests on: cpu, gpu, or all",
    )
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Also run the long statistical validation tests (marker: slow)",
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
        items[:] = [item for item in items if not item.get_closest_marker("gpu_only")]
    elif device_opt == "gpu":
        items[:] = [item for item in items if not item.get_closest_marker("cpu_only")]
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="slow validation; run with --run-slow")
        for item in items:
            if item.get_closest_marker("slow"):
                item.add_marker(skip_slow)


@pytest.fixture(autouse=True)
def _autoclose_figures():
    """Ensure all matplotlib figures are closed after each test to prevent resource leaks."""
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


# WARNING records that report what the caller asked for, not a suspect
# result, and that a test cannot avoid; each is a log-level issue of the
# library, kept here until it logs them below WARNING.
_NOT_SUSPECT = re.compile(
    "|".join([
        r"is running on CPU \(NumPy\)",  # block equalizers' device advice
        r"^RLS tail trim:",  # the documented trim of every RLS call
        r"correct_timing\(mode='slice'\): output is shorter",  # what slice does
        r"num_taps=\d+ is small for sps=1\.",  # fractional-spacing advice at 1 sps
        r"font not found",  # the plotting theme on a machine without Roboto
    ])
)  # fmt: skip


class _WarningRecords(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        if not _NOT_SUSPECT.search(record.getMessage()):
            self.records.append(record)


@pytest.fixture(autouse=True)
def _fail_on_commkit_warnings(request):
    """Fail a test whose library calls log a WARNING (a returned but suspect
    result: no convergence, a fit out of range, a misconfigured call), unless
    it is marked ``allow_warnings`` or takes ``caplog`` to assert the log."""
    if request.node.get_closest_marker("allow_warnings") or (
        "caplog" in request.fixturenames
    ):
        yield
        return
    handler = _WarningRecords()
    log = logging.getLogger("commkit")
    log.addHandler(handler)
    try:
        yield
    finally:
        log.removeHandler(handler)
    if handler.records:
        lines = "; ".join(
            f"{r.name}:{r.lineno}: {r.getMessage()}" for r in handler.records[:3]
        )
        pytest.fail(f"commkit logged {len(handler.records)} warning(s): {lines}")


@pytest.fixture
def backend_device(request):
    """
    Fixture that returns the current backend device name.

    Skips GPU tests if CuPy is not installed (fails under an explicit
    ``--device=gpu``) and fails if it is installed but not functional.  The device only
    selects where tests build their inputs (via ``xp``); the library never
    moves data on its own.

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
        # An explicit --device=gpu must test the GPU: a missing or broken
        # CuPy fails instead of passing as skips.  --device=all skips only
        # when CuPy is not installed; installed but broken still fails.
        if not _CUPY_AVAILABLE:
            if device_opt == "gpu":
                pytest.fail("--device=gpu was requested but CuPy is not installed")
            pytest.skip("CuPy not installed, skipping GPU tests")
        try:
            cp.zeros(1)
            cp.random.randn(1)
        except Exception as e:
            pytest.fail(f"CuPy is installed but not functional: {e}")

    yield device


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
