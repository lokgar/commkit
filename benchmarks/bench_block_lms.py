"""Benchmarks: block_lms frequency-domain equalizer.

``block_lms/bps+cs/16qam/N1e5/C2`` - BPS CPR
with per-symbol cycle-slip correction enabled.

Two block sizes are tracked:

* ``bench_block_lms`` (block_size=256) - the deliberate stress case: on GPU
  the per-block work is too small to amortize the per-block graph replay
  and the work outside the graph, so wall time is dominated by fixed
  overhead.
* ``bench_block_lms_large`` (block_size=2048) - the recommended GPU operating
  point with that overhead amortized.  Guards against regressions that
  scale with block size (per-element work, intermediate-tensor growth) which
  the overhead-bound 256 case would mask.
* ``bench_block_lms_dd`` (block_size=256, short training prefix) - the
  decision-directed steady state, the realistic operating mode.

``bench_block_lms``/``_large`` pass the full symbol sequence as training.
On the GPU, training blocks and decision-directed blocks are each replayed
from their own CUDA graph, so all three exercise the graph path; a silent
fallback to the eager loop shows up as a several-fold jump.
"""

import pytest
from benchutils import assert_converged
from workloads import mimo_equalizer_workload

from commkit.equalization import block_lms
from commkit.mapping import Constellation
from commkit.recovery import BPS, CycleSlip

ROUNDS = dict(rounds=3, warmup_rounds=1, iterations=1)
N_SYM = 100_000
# Short data-aided preamble for the DD benchmark: enough to seed the taps,
# small enough that the bulk of the run is decision-directed (graph-eligible).
N_TRAIN_DD = 512

CPR_CONFIGS = [
    ("no-cpr", dict()),
    (
        "bps",
        dict(
            cpr=BPS(),
        ),
    ),
    (
        "bps+cs",
        dict(
            cpr=BPS(cycle_slip=CycleSlip()),
        ),
    ),
]


# Converging step sizes for the 30-degree mixing of the workload: the block
# update's stability ceiling falls with block_size.
STEP_SIZE = {256: 3e-3, 2048: 5e-4}


def _bench_block_lms(benchmark, xp, sync, cpr_kwargs, block_size, n_train=None):
    linewidth = 100.0 if cpr_kwargs else 0.0  # linewidth x symbol time = 1e-4
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
            constellation=Constellation.qam(16),
            block_size=block_size,
            step_size=STEP_SIZE[block_size],
            **cpr_kwargs,
        )
        sync()
        return r

    r = benchmark.pedantic(run, **ROUNDS)
    assert_converged(r, syms, Constellation.qam(16), skip=4096, max_ser=1e-3)


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
