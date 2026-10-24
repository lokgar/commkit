"""CUDA-graph replay of repeated GPU work (private).

Warm up once, capture, then replay.

Used by the block equalizers (one graph per kind of block) and the fiber
solver (one graph per span).  :class:`GraphRunner` owns a non-blocking
stream that every step runs on, eager or replayed, so a captured graph and
the eager work around it stay in order.  Per key, the first
:meth:`GraphRunner.run` runs the body eagerly (priming cuFFT plans and the
memory pool, so the capture allocates nothing), the second captures it and
every later call replays it: one launch instead of one per kernel.  A failed
capture logs a warning and turns replay off for the rest of the run.

The stream is one per device and thread, reused across runs: CuPy's memory
pool caches freed blocks per stream, so a fresh stream per call would grow
the pool by a run's working set on every call.

A body is capturable only if it reads and writes fixed buffers: a replay
repeats the captured addresses, so per-call inputs must be staged into those
buffers outside the graph.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable, Hashable
from typing import Any

from ..logger import logger

_local = threading.local()


def _stream() -> Any:
    """This thread's non-blocking stream on the current device."""
    import cupy as cp

    streams = _local.__dict__.setdefault("streams", {})
    device = cp.cuda.Device().id
    if device not in streams:
        streams[device] = cp.cuda.Stream(non_blocking=True)
    return streams[device]


class GraphRunner:
    """Run keyed bodies eagerly or from captured CUDA graphs.

    Parameters
    ----------
    name : str
        Prefix of the capture-failure warning.
    enabled : bool
        Capture and replay; ``False`` (or CuPy unavailable) runs every body
        eagerly on the current stream.

    Use as a context manager around all the work of one run: entering
    orders the work queued so far before the runner's stream and makes it
    current; leaving synchronizes it.
    """

    def __init__(self, name: str, enabled: bool) -> None:
        self.name = name
        self.enabled = enabled
        self._stream: Any = None
        self._graphs: dict[Hashable, Any] = {}
        self._warmed: set[Hashable] = set()
        if enabled:
            try:
                self._stream = _stream()
            except Exception:
                self.enabled = False

    def __enter__(self) -> GraphRunner:
        if self._stream is not None:
            import cupy as cp

            ready = cp.cuda.Event()
            ready.record()  # on the caller's stream
            self._stream.wait_event(ready)
            self._stream.__enter__()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._stream is not None:
            self._stream.__exit__(*exc)
            self._stream.synchronize()

    def run(self, key: Hashable, body: Callable[[], Any]) -> Any:
        """Run ``body()`` for ``key``; ``None`` or ``False`` runs it eagerly.

        Returns what ``body`` returns, or ``None`` when a graph was replayed.
        """
        if not self.enabled or key is None or key is False:
            return body()
        if key in self._graphs:
            self._graphs[key].launch()
            return None
        if key not in self._warmed:
            self._warmed.add(key)
            return body()
        try:
            self._stream.begin_capture()
            out = body()
            graph = self._stream.end_capture()
        except Exception as exc:  # pragma: no cover - hw/version dependent
            with contextlib.suppress(Exception):
                self._stream.end_capture()
            self.disable()
            logger.warning(
                "%s: CUDA-graph capture failed (%s); continuing eagerly.",
                self.name,
                exc,
            )
            return body()  # the captured work never ran
        self._graphs[key] = graph
        graph.launch()
        return out

    def disable(self) -> None:
        """Drop the captured graphs and run every later body eagerly."""
        self._graphs.clear()
        self.enabled = False
