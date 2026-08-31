"""Device-timing and profiling helpers for the benchmark suite.

``CudaEventTimer`` measures pure device time (diagnostic - acceptance criteria
quote pytest-benchmark wall time); ``nvtx_range`` annotates DSP stages so
``nsys profile uv run pytest benchmarks/...`` produces a readable timeline and
D2H transfers can be counted per stage.
"""

from contextlib import contextmanager

try:
    import cupy as cp
except ImportError:  # pragma: no cover - CPU-only environments
    cp = None


class CudaEventTimer:
    """Pure GPU-side timing via a pair of CUDA events.

    Usage::

        t = CudaEventTimer()
        t.start()
        ...  # launch kernels
        ms = t.stop()  # synchronizes, returns elapsed device ms
    """

    def __init__(self):
        if cp is None:
            raise RuntimeError("CudaEventTimer requires CuPy")
        self._start = cp.cuda.Event()
        self._stop = cp.cuda.Event()

    def start(self):
        self._start.record()

    def stop(self) -> float:
        self._stop.record()
        self._stop.synchronize()
        return cp.cuda.get_elapsed_time(self._start, self._stop)


@contextmanager
def nvtx_range(name: str):
    """NVTX range marker; silent no-op when CuPy/NVTX is unavailable."""
    pushed = False
    if cp is not None:
        try:
            cp.cuda.nvtx.RangePush(name)
            pushed = True
        except Exception:
            pass
    try:
        yield
    finally:
        if pushed:
            cp.cuda.nvtx.RangePop()
