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
from .backend import ArrayType, dispatch, to_device


def circular_kernel(taps: ArrayType, length: int) -> np.ndarray:
    """Fold ``taps`` into a length-``length`` circular kernel on the host.

    Tap ``(L - 1) // 2`` goes to index 0; taps longer than ``length`` wrap
    around and add up, which is exactly circular convolution with them.

    Parameters
    ----------
    taps : array_like
        Filter taps, ``(L,)``, on any device.
    length : int
        Period of the sequence the kernel is applied to.

    Returns
    -------
    numpy.ndarray
        Kernel ``(length,)`` with the dtype of ``taps``.
    """
    h = np.asarray(to_device(taps, "cpu"))
    kernel = np.zeros(length, dtype=h.dtype)
    idx = (np.arange(h.size) - (h.size - 1) // 2) % length
    np.add.at(kernel, idx, h)
    return kernel


def convolve_periodic(samples: ArrayType, taps: ArrayType) -> ArrayType:
    """Circular convolution of each row of ``samples`` with ``taps``.

    Parameters
    ----------
    samples : array_like
        One period of the input, time on the last axis.
    taps : array_like
        Filter taps, ``(L,)``; any ``L``.

    Returns
    -------
    ndarray
        Filtered period, same shape, dtype and device as ``samples``.  Real
        samples and real taps give a real result.
    """
    x, xp, _ = dispatch(samples)
    n = x.shape[-1]
    kernel = circular_kernel(taps, n)
    real = not (xp.iscomplexobj(x) or np.iscomplexobj(kernel))
    # Kernel at the sample precision, so complex64 data is not promoted.
    precision = np.dtype(x.real.dtype)
    if np.iscomplexobj(kernel):
        precision = np.result_type(precision, np.complex64)
    kernel = kernel.astype(precision, copy=False)
    if real:
        h_f = _fft.rfft(xp.asarray(kernel))
        out = _fft.irfft(_fft.rfft(x, axis=-1) * h_f, n=n, axis=-1)
    else:
        h_f = _fft.fft(xp.asarray(kernel))
        out = _fft.ifft(_fft.fft(x, axis=-1) * h_f, axis=-1)
    return out.astype(x.dtype, copy=False)
