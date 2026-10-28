"""Benchmarks: LDPC encoding and decoding, Reed-Solomon decoding (``commkit.coding``).

DVB-S2 rate 1/2 normal frames over QPSK at Es/N0 = 1.5 dB (0.5 dB above
the ETSI operating point), where every codeword decodes; each decode
asserts that, so a regression that stops convergence fails instead of
timing 50 iterations of a failure.
"""

import numpy as np
import pytest
from workloads import ldpc_workload

from commkit.backend import to_device
from commkit.coding import LDPCCode, MinSum, ReedSolomonCode, SumProduct, decode, encode

ROUNDS = dict(rounds=3, warmup_rounds=1, iterations=1)
CODE = LDPCCode.dvbs2("1/2")
NUM_CODEWORDS = 64


def bench_ldpc_encode(benchmark, backend_device, xp, sync):
    rng = np.random.default_rng(0)
    bits = xp.asarray(rng.integers(0, 2, NUM_CODEWORDS * CODE.k, dtype=np.int8))

    def run():
        out = encode(bits, code=CODE)
        sync()
        return out

    benchmark.pedantic(run, **ROUNDS)


@pytest.mark.parametrize(
    "method", [MinSum(), SumProduct()], ids=["min_sum", "sum_product"]
)
def bench_ldpc_decode(benchmark, backend_device, xp, sync, method):
    llrs_np, info = ldpc_workload(CODE, NUM_CODEWORDS, esn0_db=1.5)
    llrs = xp.asarray(llrs_np)

    def run():
        out = decode(llrs, code=CODE, method=method)
        sync()
        return out

    res = benchmark.pedantic(run, **ROUNDS)
    assert np.array_equal(to_device(res.bits, "cpu"), info), "decoder did not converge"


def bench_kp4_decode(benchmark, backend_device, xp, sync):
    """RS(544, 514) over GF(2^10), 2048 codewords at a pre-FEC BER of 1e-3
    (about 5 symbol errors per codeword, all correctable)."""
    code = ReedSolomonCode.kp4()
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, (2048, code.k), dtype=np.int8)
    words = np.asarray(encode(bits, code=code))
    received = xp.asarray(words ^ (rng.random(words.shape) < 1e-3).astype(np.int8))

    def run():
        out = decode(received, code=code)
        sync()
        return out

    res = benchmark.pedantic(run, **ROUNDS)
    assert np.array_equal(to_device(res.bits, "cpu"), bits), "decoder failed"
