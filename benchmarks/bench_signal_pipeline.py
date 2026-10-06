"""End-to-end Signal pipeline benchmarks for copy-cost regression tracking."""

from __future__ import annotations

import gc
import tracemalloc

import pytest

from commkit import equalization, filtering, generate, multirate
from commkit.core import Preamble, Reference, SingleCarrierFrame
from commkit.filtering import RRC
from commkit.impairments import apply_awgn
from commkit.mapping import Constellation

ROUNDS = dict(rounds=3, warmup_rounds=1, iterations=1)


def _plain_signal():
    return generate(
        Constellation.qam(16),
        32_768,
        symbol_rate=1e6,
        sps=4,
        pulse=RRC(0.35, span=8),
        rng=42,
    )


def _frame_signal():
    frame = SingleCarrierFrame(
        # 1057 pilot periods of 31 payload symbols: the length the 1.x frame
        # snapped 32_768 up to, so the workload matches the 0002 baseline.
        payload_len=32_798,
        payload_constellation=Constellation.qam(16),
        payload_seed=42,
        preamble=Preamble(sequence_type="barker", length=13),
        pilot_pattern="comb",
        pilot_period=32,
    )
    # Materialize all lazy provenance arrays so deep-copy overhead is visible.
    payload_bits = frame.payload_bits
    payload_symbols = frame.payload_symbols
    _ = frame.pilot_bits, frame.pilot_symbols
    sig = frame.to_signal(sps=4, symbol_rate=1e6, pulse=RRC(0.35, span=8))
    sig = sig.replace(reference=Reference(symbols=payload_symbols, bits=payload_bits))
    sig = sig.replace(resolved_symbols=payload_symbols)
    sig = sig.replace(resolved_bits=payload_bits)
    return sig


def _pipeline(sig, xp, sync):
    out = apply_awgn(sig, esn0_db=20, rng=7)
    out = filtering.matched_filter(out)
    out = multirate.resample(out, sps_out=2)
    out = equalization.apply_taps(
        out, xp.asarray([1.0 + 0.0j], dtype=xp.complex64), normalize=False
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


def _profile_peak_memory(run, backend_device, xp, sync):
    """Return incremental allocator high-water memory for one pipeline run."""
    gc.collect()
    if backend_device == "gpu":
        return {"peak_gpu_pool_bytes": _gpu_peak(run, xp, sync)}

    tracemalloc.start()
    try:
        result = run()
        _, peak = tracemalloc.get_traced_memory()
        del result
    finally:
        tracemalloc.stop()
    return {"peak_cpu_tracemalloc_bytes": peak}


def _allocator_peak(call, backend_device, xp, sync):
    """Measure incremental allocator peak for one container replacement."""
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
        lambda: sig.replace_samples(replacement), backend_device, xp, sync
    )
    legacy_peak = _allocator_peak(legacy_rewrap, backend_device, xp, sync)
    assert optimized_peak < legacy_peak, (
        "Signal.replace_samples() no longer improves on legacy deep-copy rewrapping"
    )
    return {
        "rewrap_peak_bytes": optimized_peak,
        "legacy_rewrap_peak_bytes": legacy_peak,
        "rewrap_saved_bytes": legacy_peak - optimized_peak,
    }


@pytest.mark.parametrize("case", ["plain", "frame"], ids=["plain", "frame-backed"])
def bench_signal_pipeline(benchmark, backend_device, xp, sync, case):
    """Track wall time and peak CPU/GPU allocation for representative composition."""
    sig = (_plain_signal() if case == "plain" else _frame_signal()).to(backend_device)

    def run():
        return _pipeline(sig, xp, sync)

    benchmark.extra_info.update(_profile_peak_memory(run, backend_device, xp, sync))
    benchmark.extra_info.update(
        _profile_rewrap_copy_cost(sig, backend_device, xp, sync)
    )
    benchmark.extra_info["input_sample_bytes"] = sig.samples.nbytes
    benchmark.extra_info["frame_backed"] = case == "frame"
    benchmark.pedantic(run, **ROUNDS)
