"""Peak host memory of a call, for tests of bounded working memory."""

import tracemalloc
from collections.abc import Callable
from typing import Any


def peak_bytes(fn: Callable[[], Any]) -> tuple[int, Any]:
    """Peak bytes allocated while ``fn()`` runs (NumPy reports to tracemalloc).

    Returns ``(peak, result)``; the result's own bytes are part of the peak.
    """
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        result = fn()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak, result
