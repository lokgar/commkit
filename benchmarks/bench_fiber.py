"""Benchmarks: split-step fiber propagation and digital backpropagation.

A dual-polarization 32 GBd field at 64 GS/s over 4 x 80 km.  The forward
model runs 25 steps per span (100 FFT pairs and Kerr stages); on the GPU
the PMD-free spans from the third on replay a captured CUDA graph.  The
``plain`` GPU variant forces the unfused CuPy stages: the fused kernels must
beat it (AGENTS kernel policy).
"""

import pytest
from workloads import fiber_workload

from commkit import _fiber
from commkit.filtering import correct_fiber_propagation
from commkit.impairments import Fiber, apply_fiber_propagation

ROUNDS = dict(rounds=3, warmup_rounds=1, iterations=1)
FS = 64e9
LINK = dict(
    fiber=Fiber(),
    span_length_km=80.0,
    num_spans=4,
    launch_power_dbm=4.0,
    sampling_rate=FS,
)


def _stages(backend_device, monkeypatch, path):
    if path == "plain":
        if backend_device != "gpu":
            pytest.skip("the plain/fused split exists on the GPU only")
        monkeypatch.setattr(
            _fiber, "_make_ops", lambda x, xp, *args: _fiber._Ops(xp, x.dtype, *args)
        )


@pytest.mark.parametrize("path", ["auto", "plain"])
@pytest.mark.parametrize("n", [2**16, 2**20], ids=["N64k", "N1M"])
def bench_fiber_propagation(benchmark, backend_device, xp, sync, monkeypatch, n, path):
    _stages(backend_device, monkeypatch, path)
    x = xp.asarray(fiber_workload(n))

    def run():
        out = apply_fiber_propagation(x, steps_per_span=25, **LINK)
        sync()
        return out

    benchmark.pedantic(run, **ROUNDS)


@pytest.mark.parametrize("path", ["auto", "plain"])
@pytest.mark.parametrize("steps", [1, 10])
def bench_dbp(benchmark, backend_device, xp, sync, monkeypatch, steps, path):
    _stages(backend_device, monkeypatch, path)
    x = xp.asarray(fiber_workload(2**20))

    def run():
        out = correct_fiber_propagation(x, steps_per_span=steps, **LINK)
        sync()
        return out

    benchmark.pedantic(run, **ROUNDS)
