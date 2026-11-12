"""Numba windowed-sinc interpolation (private).

The CPU kernels of :mod:`commkit._interpolation`, imported on first use
through its ``_get_numba_interpolation``, and the inline helper
:func:`interp` that the Gardner loop (``timing/_numba_loops.py``) calls with
its own half-width.  Numba caches a kernel against its own source file
only: after editing :func:`interp`, clear the ``__pycache__`` of
``commkit/timing`` too.

Taps and sums are float64 scalars; samples outside ``0 .. N - 1`` count as
zero.  The window's cosine advances by a rotation per tap, so an instant
costs one sine and one sine-cosine pair besides the multiply-adds.
"""

import numba
import numpy as np

from ._interpolation import BLACKMAN_HARRIS

_A0, _A1, _A2, _A3 = BLACKMAN_HARRIS


@numba.njit(inline="always", nogil=True)
def interp(x, c, p, half):
    """Row ``c`` of ``x`` at the position ``p`` (float64): ``(re, im)``."""
    n = x.shape[1]
    i = int(np.floor(p))
    mu = p - i
    if mu == 0.0:
        if 0 <= i < n:
            return float(x[c, i].real), float(x[c, i].imag)
        return 0.0, 0.0
    # sin(π(t - μ)) = -(-1)^t sin(πμ), and sin(πμ) on the nearer side of 1
    s = np.sin(np.pi * min(mu, 1.0 - mu))
    # window phase φ_t = π((t - μ) / half + 1), advanced by π / half per tap
    phi = np.pi * ((1.0 - half - mu) / half + 1.0)
    cw = np.cos(phi)
    sw = np.sin(phi)
    cd = np.cos(np.pi / half)
    sd = np.sin(np.pi / half)
    sign = 1.0 if (half - 1) % 2 == 0 else -1.0  # (-1)^t at t = 1 - half
    inside = i - half + 1 >= 0 and i + half < n
    re = 0.0
    im = 0.0
    for t in range(1 - half, half + 1):
        window = (
            (_A0 - _A2)
            - (_A1 - 3.0 * _A3) * cw
            + cw * cw * (2.0 * _A2 - 4.0 * _A3 * cw)
        )
        h = -sign * s / (np.pi * (t - mu)) * window
        k = i + t
        if inside or 0 <= k < n:
            v = x[c, k]
            re += h * v.real
            im += h * v.imag
        sign = -sign
        cw, sw = cw * cd - sw * sd, sw * cd + cw * sd
    return re, im


@numba.njit(cache=True, fastmath=True, nogil=True, parallel=True)
def interp_at(x, base, frac, half, out_re, out_im, write_imag):
    """Every row of ``x`` at ``base[m] + frac[m]``, into ``out_re`` (and
    ``out_im`` when ``write_imag``), ``(C, M)``."""
    C = x.shape[0]
    for m in numba.prange(base.shape[0]):
        p = float(base[m]) + float(frac[m])
        for c in range(C):
            re, im = interp(x, c, p, half)
            out_re[c, m] = re
            if write_imag:
                out_im[c, m] = im


@numba.njit(cache=True, fastmath=True, nogil=True, parallel=True)
def interp_piecewise(x, knots, sub, half, out_re, out_im, write_imag):
    """Row ``c`` of ``x`` at ``knots[r, k] + j · (knots[r, k+1] - knots[r, k]) / sub``
    for output ``m = k · sub + j``, with ``r = c`` (or 0 for one knot row)."""
    C = x.shape[0]
    shared = knots.shape[0] == 1
    for m in numba.prange(out_re.shape[1]):
        k = m // sub
        j = m - k * sub
        for c in range(C):
            r = 0 if shared else c
            a = knots[r, k]
            p = a + j * ((knots[r, k + 1] - a) / sub)
            re, im = interp(x, c, p, half)
            out_re[c, m] = re
            if write_imag:
                out_im[c, m] = im
