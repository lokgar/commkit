"""Benchmarks: encoding and decoding of the code families (``commkit.coding``).

LDPC: DVB-S2 rate 1/2 normal frames over QPSK at Es/N0 = 1.5 dB (0.5 dB
above the ETSI operating point). The other decoders run over QPSK at an
Es/N0 where every codeword of their workload decodes. Each decode asserts
that, so a regression that stops convergence fails instead of timing the
iterations of a failure.
"""

import numpy as np
import pytest
from workloads import coded_workload

from commkit.backend import to_device
from commkit.coding import (
    CCDM,
    SC,
    SCL,
    Chase,
    ConvolutionalCode,
    LDPCCode,
    LinearBlockCode,
    MinSum,
    PolarCode,
    ReedSolomonCode,
    SumProduct,
    Syndrome,
    TurboCode,
    TurboIterations,
    Viterbi,
    decode,
    encode,
)
from commkit.mapping import Constellation

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
    llrs_np, info = coded_workload(CODE, NUM_CODEWORDS, esn0_db=1.5)
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


def _bench_decode(benchmark, xp, sync, code, method, num_codewords, esn0_db):
    """Time ``decode`` on ``coded_workload``; every codeword must decode."""
    llrs_np, info = coded_workload(code, num_codewords, esn0_db=esn0_db)
    llrs = xp.asarray(llrs_np)

    def run():
        out = decode(llrs, code=code, method=method)
        sync()
        return out

    res = benchmark.pedantic(run, **ROUNDS)
    assert np.array_equal(to_device(res.bits, "cpu"), info), "decoder failed"


@pytest.mark.parametrize(
    ("method", "esn0_db"), [(SC(), 3.0), (SCL(8), 2.0)], ids=["sc", "scl8"]
)
def bench_polar_decode(benchmark, backend_device, xp, sync, method, esn0_db):
    """PolarCode.nr(512, 1024) with the NR CRC-16, 64 codewords."""
    code = PolarCode.nr(512, 1024, crc="16")
    _bench_decode(benchmark, xp, sync, code, method, 64, esn0_db)


@pytest.mark.parametrize("termination", ["zero", "tail-biting"])
def bench_viterbi_decode(benchmark, backend_device, xp, sync, termination):
    """The K = 7 code, 16 blocks of 1000 bits, at Es/N0 = 5 dB."""
    code = ConvolutionalCode.k7(1000, termination=termination)
    _bench_decode(benchmark, xp, sync, code, Viterbi(), 16, 5.0)


@pytest.mark.parametrize("exact", [False, True], ids=["maxlog", "exact"])
def bench_turbo_decode(benchmark, backend_device, xp, sync, exact):
    """TurboCode.lte(1024), 32 blocks, 6 iterations, at Es/N0 = 0 dB."""
    code = TurboCode.lte(1024)
    _bench_decode(benchmark, xp, sync, code, TurboIterations(6, exact=exact), 32, 0.0)


def bench_chase_decode(benchmark, backend_device, xp, sync):
    """OIF 400ZR Hamming(128, 119), Chase(4) over Syndrome(max_weight=1),
    4096 codewords at Es/N0 = 10 dB."""
    code = LinearBlockCode.oif400zr()
    method = Chase(4, hard=Syndrome(max_weight=1))
    _bench_decode(benchmark, xp, sync, code, method, 4096, 10.0)


def bench_ccdm_decode(benchmark, backend_device, xp, sync):
    """Dematching of 8 blocks of 20000 PS-64-QAM symbols (labels, no noise)."""
    code = CCDM(Constellation.qam(64).shaped(entropy=5.0), block_length=20_000)
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, 8 * code.k, dtype=np.int8)
    labels = xp.asarray(np.asarray(encode(bits, code=code)))

    def run():
        out = decode(labels, code=code)
        sync()
        return out

    res = benchmark.pedantic(run, **ROUNDS)
    assert np.array_equal(to_device(res.bits, "cpu"), bits), "dematcher failed"
