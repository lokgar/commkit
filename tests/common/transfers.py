"""Host-device copies made by commkit, traced with ``sys.monitoring``.

Shared by the audit plugin ``tests/plugins/transfer_trace.py`` (one session-
wide recorder) and by tests that bound the copies of one call
(:func:`recording`).  A copy is:

- ``D2H``: ``cupy.ndarray.get``, directly or through ``cupy.asnumpy`` or
  :func:`commkit.backend.to_device`;
- ``H2D``: ``cupy.asarray`` / ``cupy.array`` of a NumPy array, or
  ``cupy.ndarray.set``.

It is charged to the innermost commkit frame outside ``backend`` (so a
:func:`~commkit.backend.to_device` call is charged to its caller); a copy
with no commkit frame on the stack (a test's own) is ignored.  Calls are
traced rather than patched, since ``cupy.ndarray.get`` cannot be patched,
and nothing in commkit is replaced, so every path to a copy is seen.
"""

import contextlib
import importlib.util
import os
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from types import CodeType, FrameType
from typing import Any

import numpy as np

#: Copies smaller than this are not recorded (scalars, small tables).
MIN_SIZE = 512


def _package_dir() -> str:
    """The directory of the commkit package, found without importing it.

    Sites are matched against it, not against a ``commkit/commkit/`` path
    fragment, so the trace works in a checkout of any name (a worktree).
    """
    spec = importlib.util.find_spec("commkit")
    assert spec is not None and spec.origin is not None, "commkit is not importable"
    return os.path.dirname(os.path.realpath(spec.origin)).replace("\\", "/") + "/"


_PKG = _package_dir()


def relative(path: str) -> str | None:
    """``path`` relative to the commkit package, or None outside it."""
    path = path.replace("\\", "/")
    return path[len(_PKG) :] if path.startswith(_PKG) else None


def _site() -> tuple[str, int] | None:
    # 0: _site, 1: the callback, 2: the frame making the call.
    frame: FrameType | None = sys._getframe(2)
    while frame is not None:
        path = frame.f_code.co_filename.replace("\\", "/")
        if path.endswith("/commkit/backend.py") and frame.f_code.co_qualname.startswith(
            "_device_tables"
        ):
            return "backend.py::_device_tables", frame.f_lineno
        if "/site-packages/cupy" in path or path.endswith("/commkit/backend.py"):
            frame = frame.f_back
            continue
        rel = relative(path)
        if rel is None:
            return None
        return f"{rel}::{frame.f_code.co_qualname}", frame.f_lineno
    return None


Recorder = Callable[[str, int, str, int], None]  # (kind, size, site, line)


def install(tool: int, name: str, record: Recorder) -> Callable[[], None]:
    """Trace copies under ``sys.monitoring`` tool ``tool``; return the
    function that stops tracing and frees the tool id."""
    import cupy as cp

    get, set_ = cp.ndarray.get, cp.ndarray.set
    uploads = (cp.asarray, cp.array)
    disable = sys.monitoring.DISABLE

    def on_call(code: CodeType, offset: int, f: Any, arg0: Any) -> Any:
        # ``x.get()`` passes the method descriptor and x; ``h = x.get; h()``
        # passes the bound method and its first argument.
        bound = getattr(f, "__self__", None)
        method = getattr(f, "__name__", None) if isinstance(bound, cp.ndarray) else None
        kind, size = None, 0
        if f is get or f is set_:
            kind, size = ("D2H" if f is get else "H2D"), getattr(arg0, "size", 0)
        elif method in ("get", "set"):
            kind, size = ("D2H" if method == "get" else "H2D"), bound.size
        elif any(f is u for u in uploads):
            if isinstance(arg0, np.ndarray):
                kind, size = "H2D", arg0.size
        elif relative(code.co_filename) is None:
            # The callee at a call site outside commkit does not change, so
            # stop watching it.  Inside commkit it may (``xp.asarray`` with
            # xp = numpy first, cupy later), so keep watching.
            return disable
        if kind and size >= MIN_SIZE:
            found = _site()
            if found:
                record(kind, int(size), found[0], found[1])
        return None

    monitoring = sys.monitoring
    monitoring.use_tool_id(tool, name)
    monitoring.register_callback(tool, monitoring.events.CALL, on_call)
    monitoring.set_events(tool, monitoring.events.CALL)

    def uninstall() -> None:
        monitoring.set_events(tool, 0)
        monitoring.register_callback(tool, monitoring.events.CALL, None)
        monitoring.free_tool_id(tool)

    return uninstall


@dataclass(frozen=True)
class Transfer:
    kind: str
    size: int
    site: str
    line: int


@contextlib.contextmanager
def recording() -> Iterator[list[Transfer]]:
    """Record the copies commkit makes inside the block.

    Uses ``sys.monitoring`` tool 4, so it runs alongside the audit plugin
    (tool 3) and coverage (tool 1)::

        with recording() as copies:
            plot_psd(x, sampling_rate=fs)
        assert max(c.size for c in copies) <= limit
    """
    copies: list[Transfer] = []
    uninstall = install(
        sys.monitoring.PROFILER_ID + 2,
        "commkit-transfer-recording",
        lambda kind, size, site, line: copies.append(Transfer(kind, size, site, line)),
    )
    try:
        yield copies
    finally:
        uninstall()
