"""Benchmarks: block_lms frequency-domain equalizer.

``block_lms/bps+cs/16qam/N1e5/C2`` - BPS CPR
with per-symbol cycle-slip correction enabled.

Two block sizes are tracked:

* ``bench_block_lms`` (block_size=256) - the deliberate stress case: on GPU
  the per-block work is too small to amortize the ~30-50 kernel launches per
  iteration, so wall time is dominated by fixed launch overhead.
* ``bench_block_lms_large`` (block_size=2048) - the recommended GPU operating
  point with launch overhead amortized.  Guards against regressions that
  scale with block size (per-element work, intermediate-tensor growth) which
  the overhead-bound 256 case would mask, and provides the throughput ceiling
  that graph capture at small block sizes is judged against.
* ``bench_block_lms_dd`` (block_size=256, short training prefix) - the
  decision-directed steady state, the realistic operating mode and the only
  one that exercises the CUDA-graph path (graph capture covers
  full DD blocks only; the fully-trained ``bench_block_lms`` above runs the
  eager loop because every block is a training block).  This is the headline
  Point-7 number; a graph regression / silent fallback shows up here as a
  jump back to the eager ~800 ms.

``bench_block_lms``/``_large`` pass the full symbol sequence as training, so
they measure the eager loop on both backends regardless of ``cuda_graph``.
"""

import pytest
from workloads import mimo_equalizer_workload

from commkit.equalization import block_lms

ROUNDS = dict(rounds=3, warmup_rounds=1, iterations=1)
N_SYM = 100_000
# Short data-aided preamble for the DD benchmark: enough to seed the taps,
# small enough that the bulk of the run is decision-directed (graph-eligible).
N_TRAIN_DD = 512

CPR_CONFIGS = [
    ("no-cpr", dict()),
    ("bps", dict(cpr_type="bps")),
    ("bps+cs", dict(cpr_type="bps", cpr_cycle_slip_correction=True)),
]


def _bench_block_lms(benchmark, xp, sync, cpr_kwargs, block_size, n_train=None):
    linewidth = 1e4 if cpr_kwargs else 0.0
    samples, syms = mimo_equalizer_workload(
        n_sym=N_SYM, order=16, sps=2, linewidth_hz=linewidth
    )
    x = xp.asarray(samples)
    t = xp.asarray(syms if n_train is None else syms[:, :n_train])

    def run():
        r = block_lms(
            x,
            t,
            num_taps=21,
            sps=2,
            modulation="qam",
            order=16,
            block_size=block_size,
            **cpr_kwargs,
        )
        sync()
        return r

    benchmark.pedantic(run, **ROUNDS)


@pytest.mark.parametrize("label,cpr_kwargs", CPR_CONFIGS)
def bench_block_lms(benchmark, backend_device, xp, sync, label, cpr_kwargs):
    _bench_block_lms(benchmark, xp, sync, cpr_kwargs, block_size=256)


@pytest.mark.parametrize("label,cpr_kwargs", CPR_CONFIGS)
def bench_block_lms_large(benchmark, backend_device, xp, sync, label, cpr_kwargs):
    _bench_block_lms(benchmark, xp, sync, cpr_kwargs, block_size=2048)


@pytest.mark.parametrize("label,cpr_kwargs", CPR_CONFIGS)
def bench_block_lms_dd(benchmark, backend_device, xp, sync, label, cpr_kwargs):
    _bench_block_lms(
        benchmark, xp, sync, cpr_kwargs, block_size=256, n_train=N_TRAIN_DD
    )
