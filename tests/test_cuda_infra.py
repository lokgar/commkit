"""Tests for the commkit._cuda kernel infrastructure.

Covers the availability probe, the get_kernel fallback contract (None on
CPU-only machines, warn-once on compile failure), the in-process kernel
cache, and an end-to-end compile + launch of the self-test kernel on GPU.
"""

import logging

import numpy as np
import pytest

from commkit import _cuda
from commkit._cuda import compiler


@pytest.fixture(autouse=True)
def _reset_warned_kernels(monkeypatch):
    """Isolate the warn-once bookkeeping between tests."""
    monkeypatch.setattr(_cuda, "_warned_kernels", set())


def test_get_kernel_returns_none_on_cpu(backend_device):
    if backend_device != "cpu":
        pytest.skip("CPU-leg fallback test")
    # The cpu leg of backend_device forces use_cpu_only(True), which must
    # make the probe report unavailable and get_kernel fall back cleanly.
    assert _cuda.is_available() is False
    assert _cuda.get_kernel("selftest_scale") is None


def test_get_kernel_unknown_name_raises(backend_device):
    with pytest.raises(KeyError, match="selftest_scale"):
        _cuda.get_kernel("no_such_kernel")


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_selftest_kernel_compiles_and_runs(backend_device, xp, xpt, dtype):
    if backend_device != "gpu":
        pytest.skip("requires a CUDA device")
    if not _cuda.is_available():
        pytest.skip("CUDA device below compute capability 7.0")

    launch = _cuda.get_kernel("selftest_scale", dtype=dtype)
    assert launch is not None

    rng = np.random.RandomState(42)
    x = xp.asarray(rng.randn(1 << 16).astype(dtype))
    y = launch(x, 2.5)

    assert y.dtype == x.dtype
    assert y.shape == x.shape
    xpt.assert_allclose(y, 2.5 * x, rtol=1e-6)


def test_selftest_kernel_rejects_wrong_dtype(backend_device, xp):
    if backend_device != "gpu":
        pytest.skip("requires a CUDA device")
    if not _cuda.is_available():
        pytest.skip("CUDA device below compute capability 7.0")

    launch = _cuda.get_kernel("selftest_scale", dtype="float32")
    with pytest.raises(TypeError, match="float32"):
        launch(xp.zeros(8, dtype=xp.float64), 1.0)


def test_kernel_cache_returns_same_object(backend_device):
    if backend_device != "gpu":
        pytest.skip("requires a CUDA device")
    if not _cuda.is_available():
        pytest.skip("CUDA device below compute capability 7.0")

    k1 = compiler.get_raw_kernel("selftest", "selftest_scale<float>")
    k2 = compiler.get_raw_kernel("selftest", "selftest_scale<float>")
    assert k1 is k2


def test_compile_failure_warns_once_and_returns_none(
    backend_device, monkeypatch, caplog
):
    if backend_device != "gpu":
        pytest.skip("requires a CUDA device")
    if not _cuda.is_available():
        pytest.skip("CUDA device below compute capability 7.0")

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated NVRTC failure")

    monkeypatch.setattr(compiler, "get_raw_kernel", _boom)

    with caplog.at_level(logging.WARNING, logger="commkit"):
        assert _cuda.get_kernel("selftest_scale") is None
        assert _cuda.get_kernel("selftest_scale") is None

    warnings = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "selftest_scale" in r.getMessage()
    ]
    assert len(warnings) == 1
    assert "falling back" in warnings[0].getMessage()


def test_read_source_ships_cu_files():
    src = compiler.read_source("selftest")
    assert "__global__ void selftest_scale" in src
