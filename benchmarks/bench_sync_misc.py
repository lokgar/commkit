"""Benchmarks: recovery/metrics functions with per-channel host syncs."""

from workloads import bps_workload, rotated_symbols_workload

from commkit import recovery
from commkit.metrics import evm

ROUNDS = dict(rounds=3, warmup_rounds=1, iterations=1)
N_SYM = 100_000


def bench_viterbi_viterbi_cs(benchmark, backend_device, xp, sync):
    x = xp.asarray(bps_workload(order=4, n_sym=N_SYM, num_ch=2, linewidth_hz=1e4))

    def run():
        out = recovery.recover_carrier_phase_viterbi_viterbi(
            x, "qam", 4, block_size=64, cycle_slip_correction=True
        )
        sync()
        return out

    benchmark.pedantic(run, **ROUNDS)


def bench_resolve_phase_ambiguity(benchmark, backend_device, xp, sync):
    y_np, ref_np = rotated_symbols_workload(order=16, n_sym=N_SYM, num_ch=2)
    y = xp.asarray(y_np)
    ref = xp.asarray(ref_np)

    def run():
        out = recovery.resolve_phase_ambiguity(y, ref, "qam", 16)
        sync()
        return out

    benchmark.pedantic(run, **ROUNDS)


def bench_evm(benchmark, backend_device, xp, sync):
    y_np, ref_np = rotated_symbols_workload(
        order=16, n_sym=N_SYM, num_ch=2, rot_quadrants=0
    )
    y = xp.asarray(y_np)
    ref = xp.asarray(ref_np)

    def run():
        out = evm(y, ref)
        sync()
        return out

    benchmark.pedantic(run, **ROUNDS)
