"""Record a benchmark baseline that does not depend on run order.

Usage::

    uv run python benchmarks/record_baseline.py NAME [--passes 3]

Runs every ``benchmarks/bench_*.py`` file in its own pytest process, ``passes``
times, and keeps for each benchmark the pass with the lowest median.  The
result is written to ``benchmarks/baselines/<machine>/NNNN_NAME.json`` with
the next free number, in pytest-benchmark's format, so ``--benchmark-compare``
reads it.

Why: in one full-suite process, small GPU benchmarks that follow the large
equalizer workloads ran 2-12x slower than alone (the state carried over was
not the CuPy memory pool), and single GPU runs swing by +-20-40 %.  A fresh
process per file and the best of several passes remove both.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STORAGE = ROOT / "baselines"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("name", help="baseline name, e.g. v2_0")
    parser.add_argument("--passes", type=int, default=3)
    args = parser.parse_args()

    files = sorted(ROOT.glob("bench_*.py"))
    best: dict[str, dict] = {}
    template: dict | None = None
    with tempfile.TemporaryDirectory() as tmp:
        for p in range(args.passes):
            for f in files:
                out = Path(tmp) / f"{p}_{f.stem}.json"
                cmd = [
                    sys.executable,
                    "-m",
                    "pytest",
                    str(f),
                    "--benchmark-only",
                    "--device=all",
                    "-q",
                    "-p",
                    "no:randomly",
                    f"--benchmark-json={out}",
                ]
                subprocess.run(cmd, cwd=ROOT.parent, check=True, capture_output=True)
                data = json.loads(out.read_text())
                template = template or data
                for b in data["benchmarks"]:
                    old = best.get(b["fullname"])
                    if old is None or b["stats"]["median"] < old["stats"]["median"]:
                        best[b["fullname"]] = b
            print(f"pass {p + 1}/{args.passes}: {len(best)} benchmarks")

    assert template is not None
    template["benchmarks"] = sorted(best.values(), key=lambda b: b["fullname"])
    machine = STORAGE / next(iter(sorted(d.name for d in STORAGE.iterdir())))
    numbers = [int(p.name[:4]) for p in machine.glob("[0-9][0-9][0-9][0-9]_*.json")]
    target = machine / f"{max(numbers, default=0) + 1:04d}_{args.name}.json"
    target.write_text(json.dumps(template, indent=4))
    print(f"wrote {target.relative_to(ROOT.parent)}")


if __name__ == "__main__":
    main()
