"""Benchmarks: standalone Blind Phase Search CPR.

``bps/128cross/N1e6/C2`` is the large non-square case (CPU: Numba kernel,
GPU: fused CUDA kernel).
"""

import pytest
from workloads import bps_workload

from commkit import recovery
from commkit.mapping import Constellation

ROUNDS = dict(rounds=3, warmup_rounds=1, iterations=1)


@pytest.mark.parametrize(
    "label,order",
    [
        ("16qam-square", 16),  # exercises the O(1) GRID fast path
        ("128cross-table", 128),  # non-square: Numba (CPU) / TABLE kernel (GPU)
    ],
)
def bench_bps(benchmark, backend_device, xp, sync, label, order):
    # Smaller on the CPU, as recorded in the baselines.
    n_sym = 20_000 if backend_device == "cpu" else 200_000
    x = xp.asarray(bps_workload(order=order, n_sym=n_sym, num_ch=2))

    def run():
        out = recovery.estimate_carrier_phase(
            x,
            recovery.BPS(test_phases=64, block_size=32),
            constellation=Constellation.qam(order),
        ).value
        sync()
        return out

    benchmark.pedantic(run, **ROUNDS)


def bench_bps_128cross_N1e6_C2(benchmark, backend_device, xp, sync):
    x = xp.asarray(bps_workload(order=128, n_sym=1_000_000, num_ch=2))

    def run():
        out = recovery.estimate_carrier_phase(
            x,
            recovery.BPS(test_phases=64, block_size=32),
            constellation=Constellation.qam(128),
        ).value
        sync()
        return out

    benchmark.pedantic(run, **ROUNDS)
