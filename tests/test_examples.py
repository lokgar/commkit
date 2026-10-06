"""Smoke tests: every script in examples/ runs end to end.

Each example runs in a fresh interpreter with a non-interactive Matplotlib
backend. The examples select CuPy themselves when it works, so on a GPU
machine this also exercises the GPU path. A failure here means a library
change broke a documented workflow.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = sorted((Path(__file__).parents[1] / "examples").glob("*.py"))


@pytest.mark.cpu_only
@pytest.mark.parametrize("script", EXAMPLES, ids=lambda p: p.stem)
def test_example_runs(script: Path, backend_device: str) -> None:
    env = {**os.environ, "MPLBACKEND": "Agg"}
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=script.parent,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
