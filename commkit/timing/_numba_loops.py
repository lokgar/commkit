"""Numba loop of Gardner clock recovery (private).

The compiled CPU loop behind :func:`commkit.timing.estimate_sampling_clock_offset`
and :func:`commkit.timing.correct_sampling_clock_offset` with a
:class:`~commkit.timing.Gardner`.  Imported on first use through
``_get_numba_gardner`` and ``_get_numba_ted_curve`` in
``timing/clock_recovery.py``, so ``import commkit`` does not load Numba.

Conventions: ``x`` ``(C, N)`` complex64, normalized to unit mean power and
zero-padded by ``pad`` samples at both ends (at least ``2 · sps + 10``:
the loop reads a symbol before its first strobe, the interpolator eight
samples on); positions are float64 sample indices into the padded record.
The detector interpolates with a 16-tap windowed sinc (the output is
resampled afterwards by the 48-tap interpolator of :mod:`commkit.multirate`).  The
detector output is negative when the strobes are late, for both methods:

- classical: ``e = Re{m_k · conj(y_{k-1} - y_k)}``;
- power: ``e = -|m_k|² · (|y_{k-1}|² - |y_k|²)``,

with ``y_k`` at the strobe ``p_k`` and ``y_{k-1}``, ``m_k`` one and half a
symbol period ``sps · (1 + ν)`` before it.  Placed by the integrator's rate
alone, they move together with the timing error and not with the
proportional correction of the previous symbol, and the correction from
``e_k`` acts on the spacing after the next strobe: otherwise the detector
answers the data-dependent spacing it set itself, which biases the lock
point in proportion to the gain (0.016 symbol at ``B·T = 1e-3`` for a
0.1-rolloff pulse, against 0.002 now).
"""

import numba
import numpy as np

_fast = numba.njit(cache=True, fastmath=True, nogil=True)
_step = numba.njit(inline="always", nogil=True)

# Half-width of the detector's windowed-sinc interpolator (2·_HALF taps,
# Blackman-Harris window): error of a tone -84 dB at 0.275 cycles per sample
# (the band edge of a 0.1-rolloff pulse at 2 sps), -27 dB at 0.375.
_HALF = 8


@_step
def _interp(x, c, p):
    """Windowed-sinc interpolation of channel ``c`` of ``x`` at ``p``."""
    i = int(np.floor(p))
    mu = p - i
    if mu == 0.0:
        return x[c, i].real, x[c, i].imag
    s = np.sin(np.pi * mu)
    re = 0.0
    im = 0.0
    sign = 1.0 if (_HALF - 1) % 2 == 0 else -1.0  # (-1)^t at t = -(_HALF - 1)
    for t in range(-_HALF + 1, _HALF + 1):
        u = t - mu
        # sin(π(t - μ)) = -(-1)^t · sin(πμ)
        sinc = -sign * s / (np.pi * u)
        cw = np.cos(np.pi * (u / _HALF + 1.0))
        window = (
            0.35875 - 0.48829 * cw + 0.14128 * (2.0 * cw * cw - 1.0)
        ) - 0.01168 * (4.0 * cw * cw * cw - 3.0 * cw)
        h = sinc * window
        re += h * x[c, i + t].real
        im += h * x[c, i + t].imag
        sign = -sign
    return re, im


@_step
def _ted(x, c, p_prev, p_mid, p, power):
    """Timing error at the strobe ``p`` (negative when late)."""
    a_re, a_im = _interp(x, c, p_prev)
    m_re, m_im = _interp(x, c, p_mid)
    b_re, b_im = _interp(x, c, p)
    if power:
        pa = a_re * a_re + a_im * a_im
        pb = b_re * b_re + b_im * b_im
        return -(m_re * m_re + m_im * m_im) * (pa - pb)
    return m_re * (a_re - b_re) + m_im * (a_im - b_im)


@_fast
def ted_curve(x, pad, sps, offsets, num_symbols, power):
    """Mean detector output at fixed strobe offsets (open loop).

    Strobes at ``pad + (k + offset) · sps`` for ``k = 1 .. num_symbols``,
    one row of means per channel: ``(C, len(offsets))`` float64.
    """
    C = x.shape[0]
    J = offsets.shape[0]
    out = np.zeros((C, J))
    for c in range(C):
        for j in range(J):
            acc = 0.0
            for k in range(1, num_symbols + 1):
                p = pad + (k + offsets[j]) * sps
                acc += _ted(x, c, p - sps, p - 0.5 * sps, p, power)
            out[c, j] = acc / num_symbols
    return out


@_step
def _clamp_step(step, sps):
    """Keep the strobe spacing within ±10 % of ``sps``."""
    lo = sps / 1.1
    hi = sps * 1.1
    if step < lo:
        return lo
    if step > hi:
        return hi
    return step


@_step
def _track(x, c0, c1, pad, sps, p0, mu, beta, nu_max, power, n_valid, pos, err):
    """One loop over channels ``c0 .. c1 - 1`` (their mean error drives it);
    fills ``pos`` and ``err`` and returns the number of strobes.  The
    integrator stays within ``±nu_max``."""
    nu = 0.0
    e_last = 0.0
    p = p0
    step = float(sps)
    k = 0
    K = pos.shape[0]
    last = pad + n_valid - 1.0
    while k < K and p <= last:
        period = sps * (1.0 + nu)
        e = 0.0
        for c in range(c0, c1):
            e += _ted(x, c, p - period, p - 0.5 * period, p, power)
        e /= c1 - c0
        pos[k] = p
        err[k] = e
        k += 1
        step = _clamp_step(sps * (1.0 + nu + mu * e_last), sps)
        nu = min(max(nu + beta * e_last, -nu_max), nu_max)
        e_last = e
        p += step
    return k


@numba.njit(cache=True, fastmath=True, nogil=True, parallel=True)
def gardner_loop(x, pad, sps, p0, mu, beta, nu_max, power, joint, n_valid, num_out):
    """Track the symbol strobes of every channel (or one joint loop).

    Parameters
    ----------
    x : numpy.ndarray
        ``(C, N + 2 · pad)`` complex64, unit power, zero-padded.
    pad : int
        Zero padding at each end, in samples.
    sps : float
        Nominal samples per symbol.
    p0 : numpy.ndarray
        First strobe per row in padded indices, float64 (one row per channel,
        or one for the joint loop).
    mu, beta : numpy.ndarray
        Proportional and integral gains per row, float64: the strobe spacing
        is ``sps · (1 + ν + μ·e)`` and ``ν += β·e``.
    nu_max : float
        Bound of the integrator ``|ν|``, the largest clock offset tracked
        (fraction, not ppm).
    power, joint : bool
        Power detector; one loop driven by the mean error of all channels.
    n_valid : int
        Unpadded record length ``N``; strobes stop at its last sample.
    num_out : int
        Strobe buffer length.

    Returns
    -------
    pos : numpy.ndarray
        Strobe positions, float64 ``(C, num_out)`` (``(1, num_out)`` joint).
    err : numpy.ndarray
        Detector output at each strobe, the same shape.
    count : numpy.ndarray
        Strobes filled per row, int64.
    """
    C = x.shape[0]
    rows = 1 if joint else C
    pos = np.zeros((rows, num_out))
    err = np.zeros((rows, num_out))
    count = np.zeros(rows, dtype=np.int64)
    for r in numba.prange(rows):
        c0 = 0 if joint else r
        c1 = C if joint else r + 1
        count[r] = _track(
            x,
            c0,
            c1,
            pad,
            sps,
            p0[r],
            mu[r],
            beta[r],
            nu_max,
            power,
            n_valid,
            pos[r],
            err[r],
        )
    return pos, err, count
