"""Band-limited interpolation at arbitrary instants (private).

Behind :func:`~commkit.impairments.apply_sampling_clock_offset`,
:func:`~commkit.impairments.apply_sampling_jitter` and the output of
:func:`~commkit.timing.correct_sampling_clock_offset`; the Gardner loop's
detector inlines the same Numba helper with a shorter half-width.

The interpolator is a windowed sinc of ``2 · HALF_WIDTH`` taps under a
4-term Blackman-Harris window, zero outside the record.  Instants come in
two forms:

- :func:`interpolate_at`: an integer index ``base`` and a fraction ``frac``
  per instant, for instants that are themselves sample-rate data (jitter);
- :func:`interpolate_piecewise`: instants linear between float64 knots,
  ``knots[k] + j · (knots[k+1] - knots[k]) / sub`` for ``j = 0 .. sub - 1``,
  which covers a constant rate (two knots) and the strobes of clock
  recovery.  Each instant is formed in float64 (a scalar in the CPU kernel,
  a chunk of ``_CHUNK`` instants on the GPU), so the accuracy does not fall
  with the record length.

CPU: a Numba kernel in ``_numba_interpolation.py``, parallel over the
instants, taps in float64 registers.  GPU: CuPy gathers of strided windows,
taps in the samples' precision, chunked over the instants.  Both return the
input's dtype on its device.
"""

import math
from typing import Any

import numpy as np

from .backend import ArrayType, dispatch

# Half-width of the interpolator (2·HALF_WIDTH taps).  Measured error of a
# complex tone: below -119 dB up to 0.4 cycles per sample (80 % of the
# Nyquist band), -36 dB at 0.45.
HALF_WIDTH = 24
# 4-term Blackman-Harris window w(u) = a0 - a1 cos φ + a2 cos 2φ - a3 cos 3φ,
# φ = π(u / half + 1) for a tap at distance u from the instant.
BLACKMAN_HARRIS = (0.35875, 0.48829, 0.14128, 0.01168)
_CHUNK = 1 << 14


def _get_numba_interpolation() -> Any:
    """The CPU kernels."""
    from . import _numba_interpolation

    return _numba_interpolation


def interpolate_at(
    x: ArrayType, base: ArrayType, frac: ArrayType, *, margin: int = 0
) -> ArrayType:
    """``x`` ``(C, N)`` at the instants ``base + frac``: ``base`` ``(M,)`` int64,
    ``frac`` ``(M,)`` in ``[0, 1)``.  ``base`` may reach ``margin`` samples
    beyond either end.  Returns ``(C, M)``."""
    x, xp, _ = dispatch(x)
    out = xp.empty((x.shape[0], base.shape[0]), dtype=x.dtype)
    if xp is np:
        kernels = _get_numba_interpolation()
        re, im, write_imag = _parts(out)
        kernels.interp_at(x, base, frac, HALF_WIDTH, re, im, write_imag)
        return out
    m = base.shape[0]
    _gather_gpu(
        x,
        xp,
        lambda start, stop: (base[start:stop], frac[start:stop]),
        m,
        margin,
        out,
        rows=None,
    )
    return out


def interpolate_piecewise(x: ArrayType, knots: np.ndarray, sub: int) -> ArrayType:
    """``x`` ``(C, N)`` at instants linear between ``knots``: host float64
    ``(C, K)``, or ``(1, K)`` shared by all channels; ``sub`` instants per
    interval, ``sub · (K - 1)`` per channel.  Returns ``(C, sub · (K - 1))``."""
    x, xp, _ = dispatch(x)
    knots = np.ascontiguousarray(knots, dtype=np.float64)
    C, n = x.shape
    m = sub * (knots.shape[1] - 1)
    out = xp.empty((C, m), dtype=x.dtype)
    if m == 0:
        return out
    if xp is np:
        kernels = _get_numba_interpolation()
        re, im, write_imag = _parts(out)
        kernels.interp_piecewise(x, knots, sub, HALF_WIDTH, re, im, write_imag)
        return out
    lo = math.floor(float(knots.min()))
    hi = math.ceil(float(knots.max()))
    margin = max(0, -lo, hi - (n - 1)) + 1
    dev_knots = xp.asarray(knots)
    real = x.real.dtype

    def instants(start: int, stop: int, r: int) -> tuple[ArrayType, ArrayType]:
        idx = xp.arange(start, stop)
        k = idx // sub
        a = dev_knots[r, k]
        pos = a + (idx - k * sub) * ((dev_knots[r, k + 1] - a) / sub)
        whole = xp.floor(pos)
        return whole.astype(xp.int64), (pos - whole).astype(real)

    _gather_gpu(x, xp, instants, m, margin, out, rows=knots.shape[0])
    return out


def _parts(out: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    """Writable real and imaginary views of ``out`` (a dummy for real
    ``out``) and whether to write the imaginary part."""
    if np.iscomplexobj(out):
        return out.real, out.imag, True
    return out, np.empty((1, 1), dtype=out.dtype), False


def _gather_gpu(
    x: ArrayType,
    xp: Any,
    instants: Any,
    m: int,
    margin: int,
    out: ArrayType,
    *,
    rows: int | None,
) -> None:
    """Fill ``out`` ``(C, m)`` on the GPU, ``_CHUNK`` instants at a time.
    ``instants(start, stop)`` (or ``instants(start, stop, r)`` per knot row
    ``r`` when ``rows`` is given) returns the index and fraction."""
    K = HALF_WIDTH
    real = x.real.dtype
    C = x.shape[0]
    pad = K + margin
    padded = xp.pad(x, ((0, 0), (pad, pad)))
    # Row r of windows[c] is padded[c, r : r + 2K], a strided view: gathering
    # whole rows is about twice as fast as indexing every tap.
    from cupy.lib.stride_tricks import sliding_window_view

    windows = [sliding_window_view(padded[c], 2 * K) for c in range(C)]
    offsets = xp.arange(-K + 1, K + 1)
    k = offsets.astype(real)
    # sin(π(k - μ)) = -(-1)^k sin(πμ): one sine per instant, and the window's
    # cos 2φ, cos 3φ from cos φ (Chebyshev), one cosine per tap.
    neg_alt = -((-1.0) ** offsets).astype(real)
    a0, a1, a2, a3 = BLACKMAN_HARRIS
    for start in range(0, m, _CHUNK):
        stop = min(start + _CHUNK, m)
        for c in range(C):
            if rows is None:
                if c > 0:
                    break  # instants and taps shared by all channels
                base, frac = instants(start, stop)
            else:
                base, frac = instants(start, stop, c if rows > 1 else 0)
            mu = frac[:, None].astype(real, copy=False)
            u = k[None, :] - mu
            # sin(πμ) = sin(π(1 - μ)), on the nearer side: 1 - μ is exact,
            # while the rounded π times a μ near 1 loses the small result
            # (2 % at μ = 1 - 1e-6 in float32).
            sin_pi_mu = xp.sin(math.pi * xp.minimum(mu, 1.0 - mu))
            c1 = xp.cos((u / K + 1.0) * math.pi)
            window = (
                (a0 - a2) - (a1 - 3.0 * a3) * c1 + c1 * c1 * (2.0 * a2 - 4.0 * a3 * c1)
            )
            # u = 0 only at μ = 0, where the instant is a sample: tap 1 there
            on_sample = u == 0
            sinc = xp.where(
                on_sample,
                1.0,
                neg_alt[None, :] * sin_pi_mu / (math.pi * xp.where(on_sample, 1.0, u)),
            )
            taps = (sinc * window).astype(real, copy=False)
            first = base + (pad - K + 1)  # window start per instant
            channels = range(C) if rows is None else (c,)
            for ch in channels:
                out[ch, start:stop] = xp.sum(windows[ch][first] * taps, axis=-1)
