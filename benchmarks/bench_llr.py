"""Benchmarks: soft demapping (``compute_llr``) and GMI from LLRs.

LLRs are returned on the input's device.  The 0002 baseline was recorded with
the former JAX implementation (``output="input"``, which on the GPU included a
device-to-host-to-device copy); later baselines measure the NumPy/CuPy one.
"""

import numpy as np
import pytest
from workloads import llr_workload

from commkit.mapping import Constellation, compute_llr
from commkit.metrics import gmi

ROUNDS = dict(rounds=3, warmup_rounds=1, iterations=1)
N_SYM = 2**18


@pytest.mark.parametrize("method", ["maxlog", "exact"])
@pytest.mark.parametrize("order", [16, 64, 256])
def bench_compute_llr(benchmark, backend_device, xp, sync, order, method):
    rx_np, _, noise_var = llr_workload(order=order, n_sym=N_SYM)
    rx = xp.asarray(rx_np)

    def run():
        out = compute_llr(
            rx,
            noise_var=noise_var,
            constellation=Constellation.qam(order),
            method=method,
        )
        sync()
        return out

    benchmark.pedantic(run, **ROUNDS)


@pytest.mark.parametrize("order", [16, 256])
def bench_gmi(benchmark, backend_device, xp, sync, order):
    rx_np, bits_np, noise_var = llr_workload(order=order, n_sym=N_SYM)
    c = Constellation.qam(order)
    llrs = xp.asarray(
        np.asarray(compute_llr(rx_np, noise_var=noise_var, constellation=c))
    )
    bits = xp.asarray(bits_np.reshape(-1))

    def run():
        out = gmi(llrs, bits, constellation=c)
        sync()
        return out

    benchmark.pedantic(run, **ROUNDS)
