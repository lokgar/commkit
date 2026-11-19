"""End-to-end Signal pipeline benchmarks for copy-cost regression tracking."""

from __future__ import annotations

import gc
import tracemalloc

import pytest
from workloads import pipeline_signal

from commkit import equalization, filtering, multirate
from commkit.impairments import apply_awgn

ROUNDS = dict(rounds=3, warmup_rounds=1, iterations=1)


def _pipeline(sig, xp, sync):
    out = apply_awgn(sig, esn0_db=20, rng=7)
    out = filtering.matched_filter(out)
    out = multirate.resample(out, sps_out=2)
    out = equalization.apply_taps(
        out, xp.asarray([1.0 + 0.0j], dtype=xp.complex64), normalize=False, sps=2
    )
    sync()
    return out


def _gpu_peak(call, xp, sync):
    """Bytes a fresh, private CuPy pool holds after ``call()``: its high-water mark.

    The default pool cannot measure this: an allocation that fits in a
    partially used chunk left by earlier work adds nothing to its
    ``total_bytes()``, so the result depended on what ran before.
    """
    pool = xp.cuda.MemoryPool()
    with xp.cuda.using_allocator(pool.malloc):
        result = call()
        sync()
        peak = pool.total_bytes()
        del result
    pool.free_all_blocks()
    return peak


def _allocator_peak(call, backend_device, xp, sync):
    """Incremental allocator high-water memory of one call of ``call``."""
    gc.collect()
    if backend_device == "gpu":
        return _gpu_peak(call, xp, sync)

    tracemalloc.start()
    try:
        result = call()
        _, peak = tracemalloc.get_traced_memory()
        del result
    finally:
        tracemalloc.stop()
    return peak


def _profile_rewrap_copy_cost(sig, backend_device, xp, sync):
    """Contrast Phase 1 replacement with the legacy deep-copy implementation."""
    replacement = sig.samples.copy()

    def legacy_rewrap():
        result = sig.clone()
        result = result.replace(samples=replacement)
        return result

    optimized_peak = _allocator_peak(
        lambda: sig.replace(samples=replacement), backend_device, xp, sync
    )
    legacy_peak = _allocator_peak(legacy_rewrap, backend_device, xp, sync)
    assert optimized_peak < legacy_peak, (
        "Signal.replace() no longer improves on legacy deep-copy rewrapping"
    )
    return {
        "rewrap_peak_bytes": optimized_peak,
        "legacy_rewrap_peak_bytes": legacy_peak,
        "rewrap_saved_bytes": legacy_peak - optimized_peak,
    }


@pytest.mark.parametrize("case", ["plain", "frame"], ids=["plain", "frame-backed"])
def bench_signal_pipeline(benchmark, backend_device, xp, sync, case):
    """Track wall time and peak CPU/GPU allocation for representative composition."""
    sig = pipeline_signal(frame=case == "frame").to(backend_device)

    def run():
        return _pipeline(sig, xp, sync)

    key = (
        "peak_gpu_pool_bytes"
        if backend_device == "gpu"
        else "peak_cpu_tracemalloc_bytes"
    )
    benchmark.extra_info[key] = _allocator_peak(run, backend_device, xp, sync)
    benchmark.extra_info.update(
        _profile_rewrap_copy_cost(sig, backend_device, xp, sync)
    )
    benchmark.extra_info["input_sample_bytes"] = sig.samples.nbytes
    benchmark.extra_info["frame_backed"] = case == "frame"
    benchmark.pedantic(run, **ROUNDS)
