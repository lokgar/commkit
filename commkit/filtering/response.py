"""Frequency, impulse and step responses of digital filters.

:func:`filter_response` (returning a :class:`FilterResponse`),
:func:`impulse_response` and :func:`step_response` accept FIR taps,
second-order sections, or a ``(b, a)`` transfer function, and compute on the
coefficients' device.  Frequencies are in cycles per sample unless a sampling
rate in Hz is given.
"""

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy

from .. import _fft
from ..backend import ArrayType, _to_host, dispatch
from ..math import _phasor
from .linear import iir_filter

__all__ = ["FilterResponse", "filter_response", "impulse_response", "step_response"]


@dataclass(frozen=True)
class FilterResponse:
    """Frequency response and group delay of a digital filter.

    Attributes
    ----------
    frequencies : ndarray
        Frequency grid in Hz, or cycles per sample without a sampling rate.
    response : ndarray
        Complex frequency response on the input device.
    group_delay : ndarray
        Group delay in seconds, or samples without a sampling rate. Undefined
        values at response zeros are ``nan``.
    """

    frequencies: ArrayType
    response: ArrayType
    group_delay: ArrayType


def _system(system: Any) -> tuple[str, ArrayType, ArrayType | None, Any, Any]:
    """Validate a filter representation and select its device."""
    if isinstance(system, tuple):
        if len(system) != 2:
            raise ValueError("system must be FIR taps, SOS, or a (b, a) pair")
        b, xp, sp = dispatch(system[0])
        a, axp, _ = dispatch(system[1])
        if xp is not axp:
            raise ValueError("system coefficients must be on the same device")
        if b.ndim != 1 or a.ndim != 1 or not b.size or not a.size:
            raise ValueError("system coefficients b and a must be nonempty 1-D arrays")
        return "ba", b, a, xp, sp
    x, xp, sp = dispatch(system)
    if x.ndim == 1 and x.size:
        return "fir", x, None, xp, sp
    if x.ndim == 2 and x.shape[0] and x.shape[1] == 6:
        return "sos", x, None, xp, sp
    raise ValueError("system must be nonempty FIR taps, SOS, or a (b, a) pair")


def filter_response(
    system: Any,
    *,
    nfft: int = 2048,
    sampling_rate: float | None = None,
    frequencies: ArrayType | None = None,
) -> FilterResponse:
    """Calculate the frequency response and group delay of a digital filter.

    Parameters
    ----------
    system : array_like or (b, a) tuple
        FIR taps, second-order sections ``(S, 6)``, or transfer coefficients.
    nfft : int, default 2048
        Number of frequency points when ``frequencies`` is not given: from
        DC to just below Nyquist for real coefficients, and over the whole
        band ``[-fs/2, fs/2)`` for complex ones, whose response differs at
        negative frequencies.
    sampling_rate : float, optional
        Sampling rate in Hz. Without it, frequencies are cycles per sample
        and group delays are samples; otherwise they are Hz and seconds.
    frequencies : array_like, optional
        Frequencies to evaluate, in Hz or cycles per sample when no sampling
        rate is given. Must be one-dimensional and on the coefficients' device.
        When omitted, a uniform grid of ``nfft`` points is used.

    Returns
    -------
    FilterResponse
        Frequency grid, complex response and group delay on the input device.

    Notes
    -----
    The group delay is the sum of the section delays for SOS filters. It is
    undefined at a zero or pole and is returned as ``nan`` there. For FIR
    taps that are conjugate symmetric or antisymmetric (linear phase, to
    within a few rounding steps) it is exactly ``(L - 1) / 2`` samples
    wherever it is defined. Values
    indistinguishable from zero in the coefficient precision are also marked
    undefined. Frequency grids and phase wrapping use float64; coefficient
    arithmetic follows the coefficient dtype.
    """
    if not isinstance(nfft, (int, np.integer)) or nfft <= 0:
        raise ValueError("nfft must be a positive integer")
    if sampling_rate is not None and (
        not np.isfinite(sampling_rate) or sampling_rate <= 0
    ):
        raise ValueError("sampling_rate must be positive and finite")
    kind, b, a, xp, _ = _system(system)
    scale = float(sampling_rate) if sampling_rate is not None else 1.0
    two_sided = frequencies is None and bool(
        xp.iscomplexobj(b) or (a is not None and xp.iscomplexobj(a))
    )
    if two_sided:
        f = (xp.arange(nfft, dtype=xp.float64) - nfft // 2) * (scale / nfft)
    elif frequencies is None:
        f = xp.arange(nfft, dtype=xp.float64) * (scale / (2 * nfft))
    else:
        f, fxp, _ = dispatch(frequencies)
        if fxp is not xp:
            raise ValueError("frequencies must be on the system's device")
        if f.ndim != 1 or f.size == 0 or xp.iscomplexobj(f):
            raise ValueError("frequencies must be a nonempty real 1-D array")
        if not bool(xp.all(xp.isfinite(f))):
            raise ValueError("frequencies must be finite")
        f = f.astype(xp.float64, copy=False)
    # Wrap before forming the phase: multiplying a large frequency by 2π
    # first discards its fractional cycle, especially in single precision.
    cycles = xp.remainder(f, scale) / scale
    dtype = xp.result_type(b.dtype, a.dtype if a is not None else b.dtype, xp.complex64)
    z = _phasor(-2 * np.pi * cycles, dtype, xp)

    def evaluate(coeff: ArrayType) -> tuple[ArrayType, ArrayType]:
        coeff = coeff.astype(dtype, copy=False)
        weighted_coeff = coeff * xp.arange(coeff.size, dtype=coeff.real.dtype)
        if coeff.size > 16 and frequencies is None:
            # Frequencies lie on a DFT grid (2*nfft points from DC, or nfft
            # points shifted to start at -fs/2). Folding coefficients
            # preserves their response there even when the filter is longer.
            fft_size = nfft if two_sided else 2 * nfft
            if coeff.size > fft_size:
                pad = (-coeff.size) % fft_size
                coeff = xp.pad(coeff, (0, pad)).reshape(-1, fft_size).sum(axis=0)
                weighted_coeff = (
                    xp.pad(weighted_coeff, (0, pad)).reshape(-1, fft_size).sum(axis=0)
                )
            if two_sided:
                value = xp.fft.fftshift(_fft.fft(coeff, n=fft_size))
                weighted = xp.fft.fftshift(_fft.fft(weighted_coeff, n=fft_size))
            else:
                value = _fft.fft(coeff, n=fft_size)[:nfft]
                weighted = _fft.fft(weighted_coeff, n=fft_size)[:nfft]
        else:
            value = xp.polyval(coeff[::-1], z)
            weighted = xp.polyval(weighted_coeff[::-1], z)
        return value, weighted

    response = xp.ones(f.size, dtype=dtype)
    delay = xp.zeros(f.size, dtype=xp.empty((), dtype=dtype).real.dtype)
    sections = (
        ((b[i, :3], b[i, 3:]) for i in range(b.shape[0]))
        if kind == "sos"
        else ((b, a),)
    )
    for num, den in sections:
        h, hn = evaluate(num)
        if den is None:
            d = xp.ones_like(h)
            dn = xp.zeros_like(h)
        else:
            d, dn = evaluate(den)
        # Evaluating a mathematical zero on the unit circle can leave a
        # rounding-sized residual (e.g. sin(π)); its delay is not meaningful.
        eps = xp.finfo(delay.dtype).eps
        h_floor = eps * xp.sum(xp.abs(num))
        d_floor = eps * xp.sum(xp.abs(den)) if den is not None else 0.0
        valid = (xp.abs(h) > h_floor) & (xp.abs(d) > d_floor)
        with np.errstate(divide="ignore", invalid="ignore"):
            response *= h / d
            delay += xp.where(valid, xp.real(hn / h - dn / d), xp.nan)
    if (kind == "fir" or (kind == "ba" and a is not None and a.size == 1)) and (
        _linear_phase(b, xp)
    ):
        # Exactly (L - 1) / 2 wherever the response is not zero.  The ratio
        # above carries rounding that grows as 1 / |H| next to the stopband
        # zeros (about 1e-7 relative at -130 dB), which an autoscaled plot of
        # the constant delay magnifies into spikes.
        delay = xp.where(xp.isnan(delay), delay, xp.full_like(delay, (b.size - 1) / 2))
    return FilterResponse(f, response, delay / scale)


def _linear_phase(taps: ArrayType, xp: Any) -> bool:
    """Whether ``taps`` are conjugate symmetric or antisymmetric, up to a few
    rounding steps: a linear-phase FIR filter, whose group delay is
    ``(L - 1) / 2`` at every frequency where its response is not zero."""
    if taps.size < 2:
        return True
    mirrored = xp.conj(taps[::-1])
    eps = xp.finfo(xp.result_type(taps.dtype, xp.float32)).eps
    tol = 4 * eps * xp.max(xp.abs(taps))
    return bool(
        xp.max(xp.abs(taps - mirrored)) <= tol or xp.max(xp.abs(taps + mirrored)) <= tol
    )


_IIR_DECAY = 1e-5
_IIR_FALLBACK_SAMPLES = 500
_IIR_MAX_SAMPLES = 100_000
_FIR_STEP_MIN_MARGIN = 16


def _fir_default_samples(num_taps: int, *, step: bool) -> int:
    """Default sample count for an FIR impulse or unit-step response."""
    if not step:
        return num_taps
    return num_taps + max(_FIR_STEP_MIN_MARGIN, num_taps // 10)


def _max_pole_radius(kind: str, b: ArrayType, a: ArrayType | None) -> float:
    """Largest pole magnitude from the denominator coefficients."""
    if kind == "sos":
        dens = _to_host(b[:, 3:])
        dens = dens.astype(
            np.complex128 if np.iscomplexobj(dens) else np.float64, copy=False
        )
        if not np.all(np.isfinite(dens)):
            return float("nan")
        max_r = 0.0
        for row in dens:
            roots = np.roots(row)
            if roots.size:
                max_r = max(max_r, float(np.max(np.abs(roots))))
        return max_r
    assert a is not None
    den = _to_host(a)
    den = den.astype(np.complex128 if np.iscomplexobj(den) else np.float64, copy=False)
    if not np.all(np.isfinite(den)):
        return float("nan")
    roots = np.roots(den)
    return float(np.max(np.abs(roots))) if roots.size else 0.0


def _default_num_samples(
    kind: str, b: ArrayType, a: ArrayType | None, *, step: bool
) -> int:
    """Derive a default time-response length from the filter structure."""
    if kind == "fir":
        return _fir_default_samples(int(b.size), step=step)
    num_len = 2 * int(b.shape[0]) + 1 if kind == "sos" else int(b.size)
    min_len = _fir_default_samples(num_len, step=step)
    r = _max_pole_radius(kind, b, a)
    if r == 0.0:
        return min_len
    if not math.isfinite(r) or r >= 1.0:
        return max(_IIR_FALLBACK_SAMPLES, min_len)
    log_r = math.log(r)
    if log_r >= 0.0:
        return max(_IIR_FALLBACK_SAMPLES, min_len)
    ratio = math.log(_IIR_DECAY) / log_r
    decay = _IIR_MAX_SAMPLES if ratio >= _IIR_MAX_SAMPLES else int(math.ceil(ratio))
    return min(max(min_len, (num_len - 1) + decay), max(_IIR_MAX_SAMPLES, min_len))


def _time_response(system: Any, num_samples: int | None, step: bool) -> ArrayType:
    """Impulse (``step=False``) or unit-step response of ``num_samples``."""
    if num_samples is not None and (
        isinstance(num_samples, bool)
        or not isinstance(num_samples, (int, np.integer))
        or num_samples <= 0
    ):
        raise ValueError("num_samples must be a positive integer")
    kind, b, a, xp, sp = _system(system)
    if kind == "ba" and a is not None and a.size == 1:
        kind, b = "fir", b / a[0]
    n = (
        _default_num_samples(kind, b, a, step=step)
        if num_samples is None
        else int(num_samples)
    )
    dtype = xp.result_type(b.dtype, a.dtype if a is not None else b.dtype, xp.float32)
    if kind == "fir":
        # The impulse response of an FIR filter is its taps, the step response
        # their running sum: no recursion, so nothing to filter.
        h = xp.zeros(n, dtype=dtype)
        count = min(n, b.size)
        h[:count] = b[:count]
        return xp.cumsum(h) if step else h
    if step:
        x = xp.ones(n, dtype=dtype)
    else:
        x = xp.zeros(n, dtype=dtype)
        x[0] = 1
    if x.dtype.kind == "c":
        # iir_filter takes real sections; complex coefficients run through
        # SciPy's direct form in their own precision.
        if kind == "sos":
            return sp.signal.sosfilt(b, x)
        return sp.signal.lfilter(b, a, x)
    # The sections go to the host, as iir_filter takes them (its kernels and
    # their state transitions read float64 host sections; tf2sos is SciPy's
    # root finding): a few coefficients, while x stays on its device.
    if kind == "sos":
        sos = _to_host(b, np.float64)
        sos = sos / sos[:, 3:4]
    else:
        sos = scipy.signal.tf2sos(
            _to_host(b, np.float64),
            _to_host(a, np.float64),
        )
    # iir_filter keeps the sections and the state in double precision, which
    # poles near z = 1 need, and the samples in their own precision.
    return iir_filter(x, sos)


def impulse_response(system: Any, *, num_samples: int | None = None) -> ArrayType:
    """Calculate a filter's causal impulse response.

    Parameters
    ----------
    system : array_like or (b, a) tuple
        FIR taps, second-order sections, or transfer coefficients.
    num_samples : int, optional
        Number of output samples, including zeros after an FIR response.
        ``None`` uses all taps for an FIR filter, or for an IIR filter the
        decay of its slowest pole to ``1e-5`` (-100 dB), capped at 100 000
        samples (falling back to 500 when a pole has magnitude ``>= 1``).

    Returns
    -------
    ndarray
        Impulse response, ``(N,)``, on the coefficients' device.

    Notes
    -----
    Real IIR filters run through :func:`iir_filter`, with the sections and
    the state in double precision whatever the coefficients' precision (a
    ``(b, a)`` pair is first converted to sections); the response keeps the
    coefficients' precision, at least single.  Complex IIR coefficients run
    through SciPy's direct form in their own precision.
    """
    return _time_response(system, num_samples, False)


def step_response(system: Any, *, num_samples: int | None = None) -> ArrayType:
    """Calculate a filter's causal unit-step response.

    Parameters
    ----------
    system : array_like or (b, a) tuple
        FIR taps, second-order sections, or transfer coefficients.
    num_samples : int, optional
        Number of output samples. ``None`` uses the FIR tap count plus a
        settling margin (at least 16 samples or 10 % of the taps), or for an
        IIR filter the decay of its slowest pole to ``1e-5`` (-100 dB),
        capped at 100 000 samples (falling back to 500 when a pole has
        magnitude ``>= 1``).

    Returns
    -------
    ndarray
        Step response, ``(N,)``, on the coefficients' device.

    Notes
    -----
    Computed as :func:`impulse_response` is; for FIR taps it is their running
    sum.
    """
    return _time_response(system, num_samples, True)
