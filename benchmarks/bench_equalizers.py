"""Benchmarks: sequential adaptive equalizers (Numba).

The adaptation always runs as a Numba loop on the CPU.  ``[gpu]`` legs feed
CuPy input, so the delta against ``[cpu]`` is the documented host round trip
(one device-to-host copy of the input, one host-to-device copy of the outputs).
"""

from workloads import mimo_equalizer_workload

from commkit.equalization import cma, lms, rls

ROUNDS = dict(rounds=3, warmup_rounds=1, iterations=1)


def bench_lms(benchmark, backend_device, xp, sync):
    samples, syms = mimo_equalizer_workload(n_sym=50_000, order=16, sps=2)
    x = xp.asarray(samples)
    t = xp.asarray(syms)

    def run():
        r = lms(x, t, num_taps=21, sps=2, step_size=1e-3, modulation="qam", order=16)
        sync()
        return r

    benchmark.pedantic(run, **ROUNDS)


def bench_lms_bps(benchmark, backend_device, xp, sync):
    # LMS + inline BPS carrier-phase recovery; a large BPS block (64) stresses
    # the per-symbol metric update.
    samples, syms = mimo_equalizer_workload(n_sym=20_000, order=16, sps=2)
    x = xp.asarray(samples)
    t = xp.asarray(syms)

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
        )
        sync()
        return r

    benchmark.pedantic(run, **ROUNDS)


def bench_cma(benchmark, backend_device, xp, sync):
    samples, _ = mimo_equalizer_workload(n_sym=50_000, order=4, sps=2)
    x = xp.asarray(samples)

    def run():
        r = cma(x, num_taps=21, sps=2, step_size=1e-3, modulation="qam", order=4)
        sync()
        return r

    benchmark.pedantic(run, **ROUNDS)


def bench_rls(benchmark, backend_device, xp, sync):
    # sps=1: the library itself warns that fractionally-spaced RLS is
    # ill-conditioned - benchmark the supported symbol-spaced regime.
    samples, syms = mimo_equalizer_workload(n_sym=20_000, order=16, sps=1)
    x = xp.asarray(samples)
    t = xp.asarray(syms)

    def run():
        r = rls(x, t, num_taps=21, sps=1, modulation="qam", order=16)
        sync()
        return r

    benchmark.pedantic(run, **ROUNDS)
