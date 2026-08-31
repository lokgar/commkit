"""Benchmarks: blind frequency-domain equalizers (block_cma, block_rde).

The blind, phase-directed siblings of ``block_lms`` share its overlap-save FDAF
core (forward butterfly + frequency-domain gradient) and its CUDA-graph capture
path: the host-sync-free per-block body is captured once and replayed, so wall
time at small ``block_size`` is launch-overhead-bound only on the eager
(``cuda_graph=False``) path.

Two block sizes are tracked, matching ``bench_block_lms``:

* ``bench_block_blind`` (block_size=256) - the launch-overhead-bound stress
  case.  On GPU with graph capture this collapses the per-block kernel launches
  into one replay; a graph regression / silent fallback shows up here as a jump
  back toward the eager loop (a ~7-8x slowdown vs the captured path).
* ``bench_block_blind_large`` (block_size=2048) - the launch-overhead-amortized
  operating point; guards block-size-scaling regressions (per-element work,
  intermediate-tensor growth) that the 256 case would mask.

The blind engine has no CPR and no training prefix, so every full block is
graph-eligible (unlike ``block_lms``, where only decision-directed blocks are).
"""

import pytest
from workloads import mimo_equalizer_workload

from commkit.equalization import block_cma, block_rde

ROUNDS = dict(rounds=3, warmup_rounds=1, iterations=1)
N_SYM = 100_000

EQUALIZERS = [
    ("cma", block_cma),
    ("rde", block_rde),
]


def _bench_block_blind(benchmark, xp, sync, eq_fn, block_size):
    samples, _ = mimo_equalizer_workload(n_sym=N_SYM, order=16, sps=2)
    x = xp.asarray(samples)

    def run():
        r = eq_fn(
            x,
            num_taps=21,
            sps=2,
            modulation="qam",
            order=16,
            block_size=block_size,
        )
        sync()
        return r

    benchmark.pedantic(run, **ROUNDS)


@pytest.mark.parametrize("label,eq_fn", EQUALIZERS)
def bench_block_blind(benchmark, backend_device, xp, sync, label, eq_fn):
    _bench_block_blind(benchmark, xp, sync, eq_fn, block_size=256)


@pytest.mark.parametrize("label,eq_fn", EQUALIZERS)
def bench_block_blind_large(benchmark, backend_device, xp, sync, label, eq_fn):
    _bench_block_blind(benchmark, xp, sync, eq_fn, block_size=2048)
