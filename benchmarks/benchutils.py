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


def assert_converged(result, syms, constellation, *, skip, max_ser, blind=False):
    """Fail when an equalizer benchmark did not converge.

    A workload that the algorithm cannot handle times a failure mode (slip
    storms, divergence) instead of the operating point, so every equalizer
    benchmark checks its symbol error rate after ``skip`` symbols.  Blind
    equalizers leave a phase ambiguity, which is resolved against the
    reference first.
    """
    from commkit.backend import to_device
    from commkit.recovery import resolve_phase_ambiguity

    y = to_device(result.y_hat, "cpu")[..., skip:]
    ref = to_device(syms, "cpu")[..., skip : skip + y.shape[-1]]
    if blind:
        y = resolve_phase_ambiguity(y, ref, constellation=constellation)
    ser = float((constellation.demap(y) != constellation.demap(ref)).mean())
    assert ser <= max_ser, f"benchmark workload did not converge: SER {ser:.3g}"
