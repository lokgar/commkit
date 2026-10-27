"""Benchmarks: recovery/metrics functions with per-channel host syncs."""

from workloads import FS, bps_workload, iq_imbalance_workload, rotated_symbols_workload

from commkit import frequency, recovery
from commkit.impairments import Lowdin, correct_iq_imbalance
from commkit.mapping import Constellation
from commkit.metrics import evm

ROUNDS = dict(rounds=3, warmup_rounds=1, iterations=1)
N_SYM = 100_000


def bench_viterbi_viterbi_cs(benchmark, backend_device, xp, sync):
    x = xp.asarray(bps_workload(order=4, n_sym=N_SYM, num_ch=2, linewidth_hz=1e3))

    def run():
        out = recovery.estimate_carrier_phase(
            x,
            recovery.ViterbiViterbi(block_size=64, cycle_slip=recovery.CycleSlip()),
            constellation=Constellation.qam(4),
        ).value
        sync()
        return out

    benchmark.pedantic(run, **ROUNDS)


def bench_resolve_phase_ambiguity(benchmark, backend_device, xp, sync):
    y_np, ref_np = rotated_symbols_workload(order=16, n_sym=N_SYM, num_ch=2)
    y = xp.asarray(y_np)
    ref = xp.asarray(ref_np)

    def run():
        out = recovery.resolve_phase_ambiguity(
            y, ref, constellation=Constellation.qam(16)
        )
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


def bench_pll(benchmark, backend_device, xp, sync):
    x = xp.asarray(bps_workload(order=16, n_sym=N_SYM, num_ch=2))

    def run():
        out = recovery.estimate_carrier_phase(
            x, recovery.PLL(), constellation=Constellation.qam(16)
        ).value
        sync()
        return out

    benchmark.pedantic(run, **ROUNDS)


def bench_mengali_morelli_blockwise(benchmark, backend_device, xp, sync):
    x = xp.asarray(bps_workload(order=4, n_sym=10 * N_SYM, num_ch=2))

    def run():
        out = frequency.estimate_frequency_offset(
            x,
            frequency.MengaliMorelli(block_size=1024),
            sampling_rate=FS,
            constellation=Constellation.qam(4),
        ).value
        sync()
        return out

    benchmark.pedantic(run, **ROUNDS)


def bench_correct_iq_imbalance(benchmark, backend_device, xp, sync):
    x = xp.asarray(iq_imbalance_workload(10 * N_SYM))

    def run():
        out = correct_iq_imbalance(x, Lowdin())
        sync()
        return out

    benchmark.pedantic(run, **ROUNDS)
