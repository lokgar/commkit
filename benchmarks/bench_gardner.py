"""Benchmarks: Gardner clock recovery (CPU Numba loop; GPU input makes one
round trip and resamples on the device)."""

from types import SimpleNamespace

import pytest
from benchutils import assert_converged
from workloads import clock_offset_workload

from commkit.mapping import Constellation
from commkit.multirate import decimate_to_symbol_rate
from commkit.timing import Gardner, correct_sampling_clock_offset

ROUNDS = dict(rounds=3, warmup_rounds=1, iterations=1)
N_SYM = 2**18


@pytest.mark.parametrize("method", ["classical", "power"])
def bench_correct_sampling_clock_offset(benchmark, backend_device, xp, sync, method):
    samples, symbols = clock_offset_workload(N_SYM)
    x = xp.asarray(samples)

    def run():
        out = correct_sampling_clock_offset(x, Gardner(method=method), sps=2)
        sync()
        return out

    y = benchmark.pedantic(run, **ROUNDS)
    sym = decimate_to_symbol_rate(y, sps=2)
    assert_converged(
        SimpleNamespace(y_hat=sym),
        symbols,
        Constellation.qam(16),
        skip=1000,
        max_ser=1e-3,
    )
