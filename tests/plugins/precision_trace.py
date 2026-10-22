"""pytest plugin: find 64-bit sample-rate work inside commkit (audit tool).

    uv run pytest --device=cpu -p tests.plugins.precision_trace \
        --ignore=tests/test_examples.py

Two kinds of site are reported in the terminal summary, with their commkit
source line:

- ``PROMOTE``: an operation on single-precision data that came through
  ``dispatch`` produced a float64/complex128 array of at least 512 elements
  (the NEP 50 trap, a float64 operand, a double reference);
- ``FRESH``: a NumPy math function (exp, cos, sin, sqrt, log, cumsum, angle,
  unwrap, ...) or a SciPy filter called from commkit returned such an array
  (a double phasor or response built and cast afterwards).

Every site is either a bug or one of the deliberate float64 uses listed in
AGENTS.md (phases before unwrap, phase walks, recursion state, IIR sections,
small host arrays, and complex128 input passed on purpose).  The tagged
inputs are an ndarray subclass, which a few code paths reject: failures
under this plugin are expected and say nothing about the code.  NumPy only;
CuPy and the compiled kernels are not traced.
"""

import collections
import functools
import importlib
import pkgutil
import sys
import traceback
from typing import Any

import numpy as np

_MIN = 512
_DOUBLE = (np.dtype(np.float64), np.dtype(np.complex128))
_SINGLE = (np.dtype(np.float32), np.dtype(np.complex64))
_HITS: collections.Counter[str] = collections.Counter()


def _site() -> str | None:
    for f in reversed(traceback.extract_stack()[:-3]):
        path = f.filename.replace("\\", "/")
        if "/commkit/commkit/" in path and "/plotting/" not in path:
            line = (f.line or "").strip()[:90]
            return f"{path.split('/commkit/commkit/')[1]}:{f.lineno}: {line}"
    return None


class _Tagged(np.ndarray):
    def __array_ufunc__(self, ufunc: Any, method: str, *inputs: Any, **kw: Any) -> Any:
        args = [np.asarray(a) if isinstance(a, _Tagged) else a for a in inputs]
        if "out" in kw:
            kw["out"] = tuple(
                np.asarray(o) if isinstance(o, _Tagged) else o for o in kw["out"]
            )
        out = getattr(ufunc, method)(*args, **kw)

        def tag(o: Any) -> Any:
            if isinstance(o, np.ndarray) and o.dtype in _DOUBLE and o.size >= _MIN:
                site = _site()
                if site:
                    _HITS[f"PROMOTE {site} [{ufunc.__name__} -> {o.dtype}]"] += 1
                return o
            if isinstance(o, np.ndarray) and o.dtype in _SINGLE:
                return o.view(_Tagged)
            return o

        return tuple(tag(o) for o in out) if isinstance(out, tuple) else tag(out)


def _install() -> None:
    import scipy.signal

    import commkit
    import commkit.backend as backend

    for info in pkgutil.walk_packages(commkit.__path__, "commkit."):
        if "plotting" not in info.name:
            importlib.import_module(info.name)
    original = backend.dispatch

    @functools.wraps(original)
    def dispatch(data: Any, *args: Any, **kwargs: Any) -> Any:
        x, xp, sp = original(data, *args, **kwargs)
        if xp is np and type(x) is np.ndarray and x.dtype in _SINGLE:
            x = x.view(_Tagged)
        return x, xp, sp

    for name, module in list(sys.modules.items()):
        if name.startswith("commkit") and getattr(module, "dispatch", None) is original:
            module.dispatch = dispatch  # type: ignore[attr-defined]

    def watch(owner: Any, name: str) -> None:
        f = getattr(owner, name)

        @functools.wraps(f)
        def g(*args: Any, **kwargs: Any) -> Any:
            r = f(*args, **kwargs)
            if isinstance(r, np.ndarray) and r.dtype in _DOUBLE and r.size >= _MIN:
                site = _site()
                if site:
                    _HITS[f"FRESH {site} [{name} -> {r.dtype}]"] += 1
            return r

        setattr(owner, name, g)

    for name in ("exp", "cos", "sin", "sqrt", "log", "log1p", "cumsum", "angle",
                 "unwrap", "arctan2", "convolve"):  # fmt: skip
        watch(np, name)
    for name in ("sosfilt", "sosfiltfilt", "fftconvolve", "upfirdn",
                 "resample_poly", "lfilter", "filtfilt"):  # fmt: skip
        watch(scipy.signal, name)


def pytest_configure(config: Any) -> None:
    _install()


def pytest_terminal_summary(terminalreporter: Any) -> None:
    terminalreporter.section("64-bit sample-rate work in commkit")
    if not _HITS:
        terminalreporter.write_line("none")
    for key, count in sorted(_HITS.items()):
        terminalreporter.write_line(f"{count:6d}  {key}")
