"""Benchmarks: bit mapping (``Constellation.map``) and hard decisions.

``demap`` covers both decision paths: the per-axis snap of square QAM and
the nearest-point search of PSK and cross QAM (``gray._nearest_index``).
"""

import pytest
from workloads import llr_workload

from commkit.mapping import Constellation

ROUNDS = dict(rounds=3, warmup_rounds=1, iterations=1)
N_SYM = 2**20

CONSTELLATIONS = {
    "qam256": Constellation.qam(256),
    "qam128cross": Constellation.qam(128),
    "psk16": Constellation.psk(16),
}


@pytest.mark.parametrize("order", [16, 256])
def bench_map(benchmark, backend_device, xp, sync, order):
    _, bits_np, _ = llr_workload(order=order, n_sym=N_SYM)
    bits = xp.asarray(bits_np)
    c = Constellation.qam(order)

    def run():
        out = c.map(bits)
        sync()
        return out

    benchmark.pedantic(run, **ROUNDS)


@pytest.mark.parametrize("name", list(CONSTELLATIONS))
def bench_demap(benchmark, backend_device, xp, sync, name):
    c = CONSTELLATIONS[name]
    rx_np, _, _ = llr_workload(order=c.order, n_sym=N_SYM, constellation=c)
    rx = xp.asarray(rx_np)

    def run():
        out = c.demap(rx)
        sync()
        return out

    benchmark.pedantic(run, **ROUNDS)
