"""Tests for the commkit._cuda kernel infrastructure.

Covers kernel lookup (unknown names and compile failures raise), the
in-process kernel cache, an end-to-end compile + launch of the self-test
kernel on GPU, and CUDA-graph replay.
"""

from typing import Any

import numpy as np
import pytest

from commkit import _cuda
from commkit._cuda import compiler
from commkit._cuda.graphs import GraphRunner
from tests.common.kernel_utils import skip_unless_gpu


class TestKernelLookup:
    """Kernel names and packaged sources."""

    def test_get_kernel_unknown_name_raises(self) -> None:
        """Attempting to load an unregistered kernel name raises KeyError."""
        with pytest.raises(KeyError, match="selftest_scale"):
            _cuda.get_kernel("no_such_kernel")

    def test_read_source_ships_cu_files(self) -> None:
        """Verify that packaged CUDA .cu kernel sources are discoverable via importlib."""
        src = compiler.read_source("selftest")
        assert "__global__ void selftest_scale" in src


@pytest.mark.gpu_only
class TestCudaKernelExecution:
    """Tests for compiling and launching CUDA kernels on GPU."""

    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    def test_selftest_kernel_compiles_and_runs(
        self, backend_device: str, xp: Any, xpt: Any, dtype: str
    ) -> None:
        """Verify selftest_scale compiles and computes accurately on GPU."""
        skip_unless_gpu(backend_device)

        launch = _cuda.get_kernel("selftest_scale", dtype=dtype)

        rng = np.random.RandomState(42)
        x = xp.asarray(rng.randn(1 << 16).astype(dtype))
        y = launch(x, 2.5)

        assert y.dtype == x.dtype
        assert y.shape == x.shape
        xpt.assert_allclose(y, 2.5 * x, rtol=1e-6)

    def test_selftest_kernel_rejects_wrong_dtype(
        self, backend_device: str, xp: Any
    ) -> None:
        """Kernel launcher raises TypeError if array dtype does not match template."""
        skip_unless_gpu(backend_device)

        launch = _cuda.get_kernel("selftest_scale", dtype="float32")
        with pytest.raises(TypeError, match="float32"):
            launch(xp.zeros(8, dtype=xp.float64), 1.0)


@pytest.mark.gpu_only
class TestCudaCompiler:
    """Kernel compilation: the in-process cache, and failures that raise."""

    def test_kernel_cache_returns_same_object(self, backend_device: str) -> None:
        """Subsequent compilations of the same kernel return the cached raw kernel."""
        skip_unless_gpu(backend_device)

        k1 = compiler.get_raw_kernel("selftest", "selftest_scale<float>")
        k2 = compiler.get_raw_kernel("selftest", "selftest_scale<float>")
        assert k1 is k2

    def test_compile_failure_raises(
        self, backend_device: str, monkeypatch: Any
    ) -> None:
        """An NVRTC failure reaches the caller; there is no slower path."""
        skip_unless_gpu(backend_device)

        def _boom(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("simulated NVRTC failure")

        monkeypatch.setattr(compiler, "get_raw_kernel", _boom)
        with pytest.raises(RuntimeError, match="simulated NVRTC failure"):
            _cuda.get_kernel("selftest_scale")


class TestGraphRunner:
    """Warm, capture and replay of keyed bodies (commkit._cuda.graphs)."""

    def test_disabled_runs_every_body_eagerly(self) -> None:
        calls = []
        with GraphRunner("test", enabled=False) as graphs:
            for _ in range(4):
                assert graphs.run("k", lambda: calls.append(1) or len(calls)) == len(
                    calls
                )
        assert len(calls) == 4

    @pytest.mark.gpu_only
    def test_warms_captures_then_replays(self, backend_device: str, xp: Any) -> None:
        a = xp.zeros(8, dtype=xp.float32)
        calls = []

        def body() -> Any:
            calls.append(1)
            a[...] += 1  # in place: a replay repeats the captured address
            return a

        with GraphRunner("test", enabled=True) as graphs:
            outs = [graphs.run("k", body) for _ in range(5)]
            graphs.run(None, body)  # eager whatever the key's state
        assert len(calls) == 3  # warmup, capture, eager
        assert outs[0] is a and outs[1] is a and outs[2:] == [None] * 3
        np.testing.assert_array_equal(a.get(), np.full(8, 6.0, np.float32))

    @pytest.mark.gpu_only
    def test_failed_capture_raises(self, backend_device: str, xp: Any) -> None:
        a = xp.zeros(8, dtype=xp.float32)

        def body() -> None:
            a[...] += 1
            float(a.sum())  # a host sync is illegal during capture

        with pytest.raises(RuntimeError, match="test: CUDA-graph capture failed"):
            with GraphRunner("test", enabled=True) as graphs:
                for _ in range(2):
                    graphs.run("k", body)
