"""Benchmarks: sequential adaptive equalizers.

``backend='numba'`` with GPU input measures the documented D2H round-trip +
CPU-loop case; ``backend='jax'`` measures the per-symbol ``lax.scan`` that
block-update mode is gated against (≥ 10x at D=16).
"""

import pytest
from workloads import mimo_equalizer_workload

from commkit.equalization import cma, lms, rls

ROUNDS = dict(rounds=3, warmup_rounds=1, iterations=1)


@pytest.mark.parametrize("eq_backend", ["numba", "jax"])
def bench_lms(benchmark, backend_device, xp, sync, eq_backend):
    samples, syms = mimo_equalizer_workload(n_sym=50_000, order=16, sps=2)
    x = xp.asarray(samples)
    t = xp.asarray(syms)
    device = backend_device if eq_backend == "jax" else "cpu"

    def run():
        r = lms(
            x,
            t,
            num_taps=21,
            sps=2,
            step_size=1e-3,
            modulation="qam",
            order=16,
            backend=eq_backend,
            device=device,
        )
        sync()
        return r

    benchmark.pedantic(run, **ROUNDS)


@pytest.mark.parametrize("eq_backend", ["jax", "xp"])
@pytest.mark.parametrize("block_len", [16])
def bench_lms_block(benchmark, backend_device, xp, sync, eq_backend, block_len):
    """Block-update LMS.  The gate: ``[gpu-jax]`` here vs. the
    per-symbol ``bench_lms[gpu-jax]`` scan should be >= 10x faster at D=16.
    ``[*-xp]`` is the array-native NumPy/CuPy path (no JAX)."""
    samples, syms = mimo_equalizer_workload(n_sym=50_000, order=16, sps=2)
    x = xp.asarray(samples)
    t = xp.asarray(syms)
    device = backend_device if eq_backend == "jax" else "cpu"

    def run():
        r = lms(
            x,
            t,
            num_taps=21,
            sps=2,
            step_size=1e-3,
            modulation="qam",
            order=16,
            backend=eq_backend,
            device=device,
            update_mode="block",
            block_len=block_len,
        )
        sync()
        return r

    benchmark.pedantic(run, **ROUNDS)


@pytest.mark.parametrize("eq_backend", ["numba", "jax"])
def bench_lms_bps(benchmark, backend_device, xp, sync, eq_backend):
    # LMS + inline BPS carrier-phase recovery (cpr_type='bps').  Exercises the
    # _get_jax_lms_cpr scan whose per-symbol BPS metric is an incremental
    # running sum: O(B*M)/symbol instead of re-scoring all KB
    # buffer slots.  A large block size (KB=64) is where that saving is largest.
    samples, syms = mimo_equalizer_workload(n_sym=20_000, order=16, sps=2)
    x = xp.asarray(samples)
    t = xp.asarray(syms)
    device = backend_device if eq_backend == "jax" else "cpu"

    if eq_backend == "jax":
        # BPS metric/unwrap run in float64 - JAX needs x64 enabled or it
        # silently downcasts.  Restore afterwards (other JAX legs run f32).
        import jax

        prev_x64 = bool(jax.config.jax_enable_x64)
        jax.config.update("jax_enable_x64", True)

    def run():
        r = lms(
            x,
            t,
            num_taps=21,
            sps=2,
            step_size=1e-3,
            modulation="qam",
            order=16,
            cpr_type="bps",
            cpr_bps_test_phases=64,
            cpr_bps_block_size=64,
            backend=eq_backend,
            device=device,
        )
        sync()
        return r

    try:
        benchmark.pedantic(run, **ROUNDS)
    finally:
        if eq_backend == "jax":
            jax.config.update("jax_enable_x64", prev_x64)


@pytest.mark.parametrize("eq_backend", ["numba", "jax"])
def bench_cma(benchmark, backend_device, xp, sync, eq_backend):
    samples, _ = mimo_equalizer_workload(n_sym=50_000, order=4, sps=2)
    x = xp.asarray(samples)
    device = backend_device if eq_backend == "jax" else "cpu"

    def run():
        r = cma(
            x,
            num_taps=21,
            sps=2,
            step_size=1e-3,
            modulation="qam",
            order=4,
            backend=eq_backend,
            device=device,
        )
        sync()
        return r

    benchmark.pedantic(run, **ROUNDS)


@pytest.mark.parametrize("eq_backend", ["numba", "jax"])
def bench_rls(benchmark, backend_device, xp, sync, eq_backend):
    # sps=1: the library itself warns that fractionally-spaced RLS is
    # ill-conditioned - benchmark the supported symbol-spaced regime.
    samples, syms = mimo_equalizer_workload(n_sym=20_000, order=16, sps=1)
    x = xp.asarray(samples)
    t = xp.asarray(syms)
    device = backend_device if eq_backend == "jax" else "cpu"

    if eq_backend == "jax":
        # RLS-JAX mandates x64 (complex128 P matrix); restore afterwards so
        # the LMS/CMA JAX benches keep their representative f32 config.
        import jax

        prev_x64 = bool(jax.config.jax_enable_x64)
        jax.config.update("jax_enable_x64", True)

    def run():
        r = rls(
            x,
            t,
            num_taps=21,
            sps=1,
            modulation="qam",
            order=16,
            backend=eq_backend,
            device=device,
        )
        sync()
        return r

    try:
        benchmark.pedantic(run, **ROUNDS)
    finally:
        if eq_backend == "jax":
            jax.config.update("jax_enable_x64", prev_x64)
