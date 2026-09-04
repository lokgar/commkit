"""End-to-end Signal pipeline benchmarks for copy-cost regression tracking."""

from __future__ import annotations

import gc
import tracemalloc

import pytest

from commkit import equalization, filtering, multirate
from commkit.core import Preamble, SingleCarrierFrame, generate_qam
from commkit.impairments import apply_awgn

ROUNDS = dict(rounds=3, warmup_rounds=1, iterations=1)


def _plain_signal():
    return generate_qam(
        order=16,
        num_symbols=32_768,
        sps=4,
        symbol_rate=1e6,
        pulse_shape="rrc",
        filter_span=8,
        seed=42,
    )


def _frame_signal():
    frame = SingleCarrierFrame(
        payload_len=32_768,
        payload_mod_scheme="QAM",
        payload_mod_order=16,
        payload_seed=42,
        preamble=Preamble(sequence_type="barker", length=13),
        pilot_pattern="comb",
        pilot_period=32,
    )
    # Materialize all lazy provenance arrays so deep-copy overhead is visible.
    payload_bits = frame.payload_bits
    payload_symbols = frame.payload_symbols
    _ = frame.pilot_bits, frame.pilot_symbols
    sig = frame.to_signal(sps=4, symbol_rate=1e6, filter_span=8)
    sig.source_bits = payload_bits
    sig.source_symbols = payload_symbols
    sig.resolved_symbols = payload_symbols
    sig.resolved_bits = payload_bits
    return sig


def _pipeline(sig, xp, sync):
    out = apply_awgn(sig, esn0_db=20, seed=7)
    out = filtering.matched_filter(out)
    out = multirate.resample(out, sps_out=2)
    out = equalization.apply_taps(
        out, xp.asarray([1.0 + 0.0j], dtype=xp.complex64), normalize=False
    )
    sync()
    return out


def _profile_peak_memory(run, backend_device, xp, sync):
    """Return incremental allocator high-water memory for one pipeline run."""
    gc.collect()
    if backend_device == "gpu":
        pool = xp.get_default_memory_pool()
        pool.free_all_blocks()
        sync()
        baseline = pool.total_bytes()
        result = run()
        sync()
        peak = pool.total_bytes() - baseline
        del result
        return {"peak_gpu_pool_bytes": peak}

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
        pool = xp.get_default_memory_pool()
        pool.free_all_blocks()
        sync()
        baseline = pool.total_bytes()
        result = call()
        sync()
        peak = pool.total_bytes() - baseline
        del result
        pool.free_all_blocks()
        return peak

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
        result.samples = replacement
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
    sig = _plain_signal() if case == "plain" else _frame_signal()

    def run():
        return _pipeline(sig, xp, sync)

    benchmark.extra_info.update(_profile_peak_memory(run, backend_device, xp, sync))
    benchmark.extra_info.update(
        _profile_rewrap_copy_cost(sig, backend_device, xp, sync)
    )
    benchmark.extra_info["input_sample_bytes"] = sig.samples.nbytes
    benchmark.extra_info["frame_backed"] = case == "frame"
    benchmark.pedantic(run, **ROUNDS)
