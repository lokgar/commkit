"""Benchmark: ``import commkit`` cost in a fresh interpreter.

Each round spawns a new Python process, so the timing includes interpreter
start-up (a constant offset) plus everything ``import commkit`` triggers:
submodule imports, Matplotlib theme setup, and GPU probing.  The number of
modules loaded is recorded in ``extra_info`` - it tracks the lazy-import work
independently of timing noise.
"""

import subprocess
import sys

ROUNDS = dict(rounds=5, warmup_rounds=1, iterations=1)

_CHILD = "import sys, commkit; print(len(sys.modules))"


def _import_once() -> int:
    out = subprocess.run(
        [sys.executable, "-c", _CHILD],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(out.stdout.strip().splitlines()[-1])


def bench_import_commkit(benchmark):
    benchmark.extra_info["num_modules"] = _import_once()
    benchmark.pedantic(_import_once, **ROUNDS)
