"""Oracles for periodic (circular) processing, independent of commkit._circular."""

from collections.abc import Callable
from typing import Any

import numpy as np

from tests.common.conversions import to_numpy


def middle_period(fn: Callable[[Any], Any], x: Any, *, reps: int = 3) -> np.ndarray:
    """One period of ``fn`` applied to ``x`` repeated ``reps`` times (odd).

    For a linear filter whose transients fit within ``reps // 2`` periods,
    the middle period of the output equals circular processing of ``x``.
    """
    tiled = np.tile(to_numpy(x), (1,) * (np.ndim(x) - 1) + (reps,))
    y = to_numpy(fn(tiled))
    n = y.shape[-1] // reps
    k = reps // 2
    return y[..., k * n : (k + 1) * n]


def symbol_power(x: Any, sps: int) -> np.ndarray:
    """Scale ``x`` per row to E[|x|²] = 1/sps, as ``shape_pulse`` does."""
    x = to_numpy(x)
    return x / np.sqrt(np.mean(np.abs(x) ** 2, axis=-1, keepdims=True) * sps)


def circular_matched_samples(x: Any, taps: np.ndarray, sps: int) -> np.ndarray:
    """Circular matched filter with ``taps`` (by an explicit DFT), sampled
    at the symbol instants ``k * sps``."""
    x = to_numpy(x)
    n = x.shape[-1]
    h = np.conj(np.asarray(taps)[::-1])
    kernel = np.zeros(n, dtype=complex)
    for i, tap in enumerate(h):
        kernel[(i - (h.size - 1) // 2) % n] += tap
    y = np.fft.ifft(np.fft.fft(x, axis=-1) * np.fft.fft(kernel), axis=-1)
    return y[..., ::sps]
