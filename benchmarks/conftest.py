"""Fixtures for the CommKit benchmark suite.

Mirrors the ``backend_device``/``xp`` fixtures from ``tests/conftest.py`` so
benchmarks parametrize over CPU/GPU exactly like the test suite, and adds a
``sync`` fixture so timed callables include GPU kernel completion.

Run explicitly (the default ``uv run pytest`` collects ``tests/`` only):

    uv run pytest benchmarks/ --benchmark-only --device=gpu
    uv run pytest benchmarks/ --benchmark-only --device=all \
        --benchmark-save=<label> --benchmark-storage=file://benchmarks/baselines
"""

import logging

import numpy as np
import pytest

from commkit import backend

try:
    import cupy as cp

    _CUPY_AVAILABLE = True
except ImportError:
    cp = None
    _CUPY_AVAILABLE = False

# Benchmarks measure compute, not logging: silence per-channel INFO chatter.
logging.getLogger("commkit").setLevel(logging.WARNING)


def pytest_addoption(parser):
    # Guarded: tests/conftest.py registers the same option when both
    # directories are collected in one invocation.
    try:
        parser.addoption(
            "--device",
            action="store",
            default="cpu",
            help="Device to run benchmarks on: cpu, gpu, or all",
        )
    except ValueError:
        pass


def pytest_generate_tests(metafunc):
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
    """Current backend device name; skips GPU benches without functional CuPy."""
    device = request.param
    if device == "gpu":
        backend.use_cpu_only(False)
        if not _CUPY_AVAILABLE:
            pytest.skip("CuPy not installed, skipping GPU benchmarks")
        try:
            cp.zeros(1)
        except Exception as e:  # pragma: no cover - environment-dependent
            pytest.skip(f"CuPy installed but not functional: {e}")
    elif device == "cpu":
        backend.use_cpu_only(True)
    try:
        yield device
    finally:
        backend.use_cpu_only(False)


@pytest.fixture
def xp(backend_device):
    """Array module for the current backend (numpy or cupy)."""
    return cp if backend_device == "gpu" else np


@pytest.fixture
def sync(backend_device):
    """Device synchronization callable - call at the end of every timed body.

    On GPU, wall time without a sync measures only kernel *launches*; the
    returned callable blocks until the current stream has drained.
    """
    if backend_device == "gpu":

        def _sync():
            cp.cuda.get_current_stream().synchronize()

    else:

        def _sync():
            pass

    return _sync
