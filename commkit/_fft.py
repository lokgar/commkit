"""FFTs on the data's device (private).

Every FFT in the library goes through these four functions:

- CPU: ``scipy.fft`` on every core.  It keeps complex64 in single precision,
  as cuFFT does, and is about 3x faster than ``numpy.fft`` on one core
  (NumPy 2 computes complex64 transforms in double precision).
- GPU: ``cupy.fft``, which caches its cuFFT plans per shape.
"""

from typing import Any

from .backend import ArrayType, _is_cupy_array

# scipy.fft worker threads: -1 uses every core, as the Numba kernels do.
_WORKERS = -1


def _call(name: str, x: ArrayType, n: int | None, axis: int) -> Any:
    if _is_cupy_array(x):
        import cupy

        return getattr(cupy.fft, name)(x, n=n, axis=axis)
    import scipy.fft

    return getattr(scipy.fft, name)(x, n=n, axis=axis, workers=_WORKERS)


def fft(x: ArrayType, n: int | None = None, axis: int = -1) -> ArrayType:
    """Forward complex FFT along ``axis``."""
    return _call("fft", x, n, axis)


def ifft(x: ArrayType, n: int | None = None, axis: int = -1) -> ArrayType:
    """Inverse complex FFT along ``axis``."""
    return _call("ifft", x, n, axis)


def rfft(x: ArrayType, n: int | None = None, axis: int = -1) -> ArrayType:
    """Forward FFT of real input along ``axis`` (non-negative frequencies)."""
    return _call("rfft", x, n, axis)


def irfft(x: ArrayType, n: int | None = None, axis: int = -1) -> ArrayType:
    """Inverse of :func:`rfft`; ``n`` is the output length."""
    return _call("irfft", x, n, axis)
