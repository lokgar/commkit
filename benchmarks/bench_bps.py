"""Benchmarks: standalone Blind Phase Search CPR.

``bps/128cross/N1e6/C2`` (GPU-only - the non-square CPU path
at that size is prohibitively slow, which is rather the point).
"""

import pytest
from workloads import bps_workload

from commkit import recovery

ROUNDS = dict(rounds=3, warmup_rounds=1, iterations=1)


@pytest.mark.parametrize(
    "label,order",
    [
        ("16qam-square", 16),  # exercises the O(1) GRID fast path
        ("128cross-table", 128),  # exercises the (CHUNK, B, M) TABLE path
    ],
)
def bench_bps(benchmark, backend_device, xp, sync, label, order):
    # The non-square NumPy path is slow; keep the CPU baseline tractable.
    n_sym = 20_000 if backend_device == "cpu" else 200_000
    x = xp.asarray(bps_workload(order=order, n_sym=n_sym, num_ch=2))

    def run():
        out = recovery.recover_carrier_phase_bps(
            x, "qam", order, num_test_phases=64, block_size=32
        )
        sync()
        return out

    benchmark.pedantic(run, **ROUNDS)


def bench_bps_128cross_N1e6_C2(benchmark, backend_device, xp, sync):
    if backend_device != "gpu":
        pytest.skip("Workload is GPU-only")
    x = xp.asarray(bps_workload(order=128, n_sym=1_000_000, num_ch=2))

    def run():
        out = recovery.recover_carrier_phase_bps(
            x, "qam", 128, num_test_phases=64, block_size=32
        )
        sync()
        return out

    benchmark.pedantic(run, **ROUNDS)
