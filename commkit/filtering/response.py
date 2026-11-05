"""Frequency, impulse and step responses of digital filters.

:func:`filter_response` (returning a :class:`FilterResponse`),
:func:`impulse_response` and :func:`step_response` accept FIR taps,
second-order sections, or a ``(b, a)`` transfer function, and compute on the
coefficients' device.  Frequencies are in cycles per sample unless a sampling
rate in Hz is given.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy

from .. import _fft
from ..backend import ArrayType, dispatch, to_device
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
    undefined at a zero or pole and is returned as ``nan`` there. Values
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
    return FilterResponse(f, response, delay / scale)


def _time_response(system: Any, num_samples: int, step: bool) -> ArrayType:
    """Impulse (``step=False``) or unit-step response of ``num_samples``."""
    if not isinstance(num_samples, (int, np.integer)) or num_samples <= 0:
        raise ValueError("num_samples must be a positive integer")
    kind, b, a, xp, sp = _system(system)
    if kind == "ba" and a is not None and a.size == 1:
        kind, b = "fir", b / a[0]
    dtype = xp.result_type(b.dtype, a.dtype if a is not None else b.dtype, xp.float32)
    if kind == "fir":
        # The impulse response of an FIR filter is its taps, the step response
        # their running sum: no recursion, so nothing to filter.
        h = xp.zeros(num_samples, dtype=dtype)
        count = min(num_samples, b.size)
        h[:count] = b[:count]
        return xp.cumsum(h) if step else h
    if step:
        x = xp.ones(num_samples, dtype=dtype)
    else:
        x = xp.zeros(num_samples, dtype=dtype)
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
        sos = np.asarray(to_device(b, "cpu"), dtype=np.float64)
        sos = sos / sos[:, 3:4]
    else:
        sos = scipy.signal.tf2sos(
            np.asarray(to_device(b, "cpu"), dtype=np.float64),
            np.asarray(to_device(a, "cpu"), dtype=np.float64),
        )
    # iir_filter keeps the sections and the state in double precision, which
    # poles near z = 1 need, and the samples in their own precision.
    return iir_filter(x, sos)


def impulse_response(system: Any, *, num_samples: int = 500) -> ArrayType:
    """Calculate a filter's causal impulse response.

    Parameters
    ----------
    system : array_like or (b, a) tuple
        FIR taps, second-order sections, or transfer coefficients.
    num_samples : int, default 500
        Number of output samples, including zeros after an FIR response.

    Returns
    -------
    ndarray
        Impulse response, ``(num_samples,)``, on the coefficients' device.

    Notes
    -----
    Real IIR filters run through :func:`iir_filter`, with the sections and
    the state in double precision whatever the coefficients' precision (a
    ``(b, a)`` pair is first converted to sections); the response keeps the
    coefficients' precision, at least single.  Complex IIR coefficients run
    through SciPy's direct form in their own precision.
    """
    return _time_response(system, num_samples, False)


def step_response(system: Any, *, num_samples: int = 500) -> ArrayType:
    """Calculate a filter's causal unit-step response.

    Parameters
    ----------
    system : array_like or (b, a) tuple
        FIR taps, second-order sections, or transfer coefficients.
    num_samples : int, default 500
        Number of output samples.

    Returns
    -------
    ndarray
        Step response, ``(num_samples,)``, on the coefficients' device.

    Notes
    -----
    Computed as :func:`impulse_response` is; for FIR taps it is their running
    sum.
    """
    return _time_response(system, num_samples, True)
