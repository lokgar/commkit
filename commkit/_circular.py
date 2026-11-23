"""Circular (periodic) convolution (private).

Shared by ``shape_pulse`` and ``fir_filter`` when their input is one period of
a periodic sequence: the record is filtered as if it repeated forever, so the
output loops seamlessly (e.g. on a streaming AWG).  The taps are aligned like
``scipy.signal.resample_poly`` and ``convolve(mode="same")``: tap
``(L - 1) // 2`` lands on the output sample, so a symmetric filter adds no
delay.
"""

import numpy as np

from . import _fft
from .backend import ArrayType, _to_host, dispatch


def circular_kernel(
    taps: ArrayType, length: int, center: int | None = None
) -> np.ndarray:
    """Fold ``taps`` into a length-``length`` circular kernel on the host.

    Tap ``center`` goes to index 0; taps longer than ``length`` wrap around
    and add up, which is exactly circular convolution with them.

    Parameters
    ----------
    taps : array_like
        Filter taps, ``(L,)``, on any device.
    length : int
        Period of the sequence the kernel is applied to.
    center : int, optional
        Tap that lands on the output sample; ``None`` takes ``(L - 1) // 2``.

    Returns
    -------
    numpy.ndarray
        Kernel ``(length,)`` with the dtype of ``taps``.
    """
    h = _to_host(taps)
    kernel = np.zeros(length, dtype=h.dtype)
    c = (h.size - 1) // 2 if center is None else center
    idx = (np.arange(h.size) - c) % length
    np.add.at(kernel, idx, h)
    return kernel


def convolve_periodic(
    samples: ArrayType, taps: ArrayType, center: int | None = None
) -> ArrayType:
    """Circular convolution of each row of ``samples`` with ``taps``.

    Parameters
    ----------
    samples : array_like
        One period of the input, time on the last axis.
    taps : array_like
        Filter taps, ``(L,)``; any ``L``.
    center : int, optional
        Tap that lands on the output sample; ``None`` takes ``(L - 1) // 2``.

    Returns
    -------
    ndarray
        Filtered period, same shape, dtype and device as ``samples``.  Real
        samples and real taps give a real result.
    """
    x, xp, _ = dispatch(samples)
    n = x.shape[-1]
    h_raw = xp.asarray(taps)
    real = not (xp.iscomplexobj(x) or xp.iscomplexobj(h_raw))
    real_dtype = x.real.dtype if x.real.dtype.kind == "f" else np.dtype(np.float32)
    out_dtype = real_dtype if real else np.result_type(real_dtype, np.complex64)
    if n == 0:
        return xp.empty(x.shape, dtype=out_dtype)
    # Fold taps on the data's device (avoids a record-length H2D copy on GPU).
    precision = (
        np.result_type(real_dtype, np.complex64)
        if xp.iscomplexobj(h_raw)
        else real_dtype
    )
    h = h_raw.astype(precision, copy=False).ravel()
    c = (h.size - 1) // 2 if center is None else center
    kernel = xp.zeros(n, dtype=precision)
    if h.size <= n:
        kernel[: h.size] = h
        if c:
            kernel = xp.roll(kernel, -c)
    else:
        idx = (xp.arange(h.size) - c) % n
        xp.add.at(kernel, idx, h)
    if real:
        h_f = _fft.rfft(kernel)
        out = _fft.irfft(_fft.rfft(x, axis=-1) * h_f, n=n, axis=-1)
    else:
        h_f = _fft.fft(kernel)
        out = _fft.ifft(_fft.fft(x, axis=-1) * h_f, axis=-1)
    return out.astype(out_dtype, copy=False)
