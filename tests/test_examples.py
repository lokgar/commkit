"""Smoke tests: every notebook in examples/ runs end to end.

Each notebook executes top to bottom in a fresh kernel with a non-interactive
Matplotlib backend. The notebooks select CuPy themselves when it works, so on a
GPU machine this also exercises the GPU path. A failure here means a library
change broke a documented workflow. The committed notebooks carry no outputs
(the executed copy is discarded).
"""

import subprocess
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

EXAMPLES = sorted((Path(__file__).parents[1] / "examples").glob("*.ipynb"))


def test_examples_exist() -> None:
    assert EXAMPLES, "no notebooks found in examples/"


@pytest.mark.cpu_only
@pytest.mark.parametrize("notebook", EXAMPLES, ids=lambda p: p.stem)
def test_example_runs(
    notebook: Path, backend_device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MPLBACKEND", "Agg")
    nb = nbformat.read(notebook, as_version=4)
    client = NotebookClient(
        nb,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(notebook.parent)}},
    )
    client.execute()


def _committed_text(notebook: Path) -> str:
    """The notebook as git would commit it: the staged copy, which the
    nbstripout clean filter has stripped, else the file (no git checkout)."""
    root = notebook.parents[1]
    rel = notebook.relative_to(root).as_posix()
    proc = subprocess.run(
        ["git", "show", f":{rel}"], cwd=root, capture_output=True, text=True
    )
    return proc.stdout if proc.returncode == 0 else notebook.read_text()


@pytest.mark.parametrize("notebook", EXAMPLES, ids=lambda p: p.stem)
def test_example_has_no_outputs(notebook: Path) -> None:
    """Committed notebooks carry no outputs (.gitattributes + nbstripout).

    Checks the staged copy, so a notebook run locally (outputs in the
    working tree, stripped on staging) does not fail the suite.
    """
    nb = nbformat.reads(_committed_text(notebook), as_version=4)
    dirty = [
        i
        for i, c in enumerate(nb.cells)
        if c.cell_type == "code" and (c.get("outputs") or c.get("execution_count"))
    ]
    assert not dirty, f"cells with outputs: {dirty}"
