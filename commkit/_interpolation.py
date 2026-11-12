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
  recovery.  Each instant is formed in float64 (a scalar per instant in
  both kernels), so the accuracy does not fall with the record length.

CPU: a Numba kernel in ``_numba_interpolation.py``, parallel over the
instants, taps in float64 registers.  GPU: the CUDA kernel ``interpolate``
(``_cuda/src/interpolate.cu``), one thread per instant, taps in the
samples' real precision, shared by the channels that share the instant.
Both return the input's dtype on its device.
"""

from typing import Any

import numpy as np

from ._cuda import get_kernel
from .backend import ArrayType, dispatch

# Half-width of the interpolator (2·HALF_WIDTH taps).  Measured error of a
# complex tone: below -119 dB up to 0.4 cycles per sample (80 % of the
# Nyquist band), -36 dB at 0.45.
HALF_WIDTH = 24
# 4-term Blackman-Harris window w(u) = a0 - a1 cos φ + a2 cos 2φ - a3 cos 3φ,
# φ = π(u / half + 1) for a tap at distance u from the instant.  The CUDA
# kernel (_cuda/src/interpolate.cu) repeats the coefficients.
BLACKMAN_HARRIS = (0.35875, 0.48829, 0.14128, 0.01168)


def _get_numba_interpolation() -> Any:
    """The CPU kernels."""
    from . import _numba_interpolation

    return _numba_interpolation


def interpolate_at(x: ArrayType, base: ArrayType, frac: ArrayType) -> ArrayType:
    """``x`` ``(C, N)`` at the instants ``base + frac``: ``base`` ``(M,)`` int64,
    ``frac`` ``(M,)`` in ``[0, 1)`` in ``x``'s real dtype; ``base`` may lie
    beyond either end.  Returns ``(C, M)``."""
    x, xp, _ = dispatch(x)
    if xp is np:
        out = np.empty((x.shape[0], base.shape[0]), dtype=x.dtype)
        kernels = _get_numba_interpolation()
        re, im, write_imag = _parts(out)
        kernels.interp_at(x, base, frac, HALF_WIDTH, re, im, write_imag)
        return out
    launch = get_kernel("interpolate", half=HALF_WIDTH)
    return launch(xp.ascontiguousarray(x), base=base, frac=frac)


def interpolate_piecewise(x: ArrayType, knots: np.ndarray, sub: int) -> ArrayType:
    """``x`` ``(C, N)`` at instants linear between ``knots``: host float64
    ``(C, K)``, or ``(1, K)`` shared by all channels; ``sub`` instants per
    interval, ``sub · (K - 1)`` per channel.  Returns ``(C, sub · (K - 1))``."""
    x, xp, _ = dispatch(x)
    knots = np.ascontiguousarray(knots, dtype=np.float64)
    if xp is np:
        out = np.empty((x.shape[0], sub * (knots.shape[1] - 1)), dtype=x.dtype)
        if out.size:
            kernels = _get_numba_interpolation()
            re, im, write_imag = _parts(out)
            kernels.interp_piecewise(x, knots, sub, HALF_WIDTH, re, im, write_imag)
        return out
    launch = get_kernel("interpolate", half=HALF_WIDTH)
    return launch(xp.ascontiguousarray(x), knots=xp.asarray(knots), sub=sub)


def _parts(out: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    """Writable real and imaginary views of ``out`` (a dummy for real
    ``out``) and whether to write the imaginary part."""
    if np.iscomplexobj(out):
        return out.real, out.imag, True
    return out, np.empty((1, 1), dtype=out.dtype), False
