"""FFTs on the data's device (private).

Every FFT in the library goes through :func:`fft`, :func:`ifft`, :func:`rfft`
and :func:`irfft`:

- CPU: ``scipy.fft`` on every core.  It keeps complex64 in single precision,
  as cuFFT does, and is about 3x faster than ``numpy.fft`` on one core
  (NumPy 2 computes complex64 transforms in double precision).
- GPU: ``cupy.fft``, which caches its cuFFT plans per shape.
"""

from typing import Any

from .backend import ArrayType, _is_cupy_array

# scipy.fft worker threads: -1 uses every core, as the Numba kernels do.
_WORKERS = -1


def _call(
    name: str,
    x: ArrayType,
    n: int | None,
    axis: int,
    overwrite_x: bool = False,
    norm: str = "backward",
) -> Any:
    if _is_cupy_array(x):
        if overwrite_x or norm != "backward":
            # cupyx.scipy.fft transforms a C2C input in place when allowed
            # (same cuFFT plans and cache as cupy.fft).
            import cupyx.scipy.fft

            return getattr(cupyx.scipy.fft, name)(
                x, n=n, axis=axis, norm=norm, overwrite_x=overwrite_x
            )
        import cupy

        return getattr(cupy.fft, name)(x, n=n, axis=axis)
    import scipy.fft

    return getattr(scipy.fft, name)(
        x, n=n, axis=axis, norm=norm, workers=_WORKERS, overwrite_x=overwrite_x
    )


def fft(
    x: ArrayType,
    n: int | None = None,
    axis: int = -1,
    *,
    overwrite_x: bool = False,
    norm: str = "backward",
) -> ArrayType:
    """Forward complex FFT along ``axis``.

    ``overwrite_x`` lets the transform reuse ``x``'s buffer (in place on the
    GPU for complex input); always use the returned array.  ``norm`` as in
    ``scipy.fft``.
    """
    return _call("fft", x, n, axis, overwrite_x, norm)


def ifft(
    x: ArrayType,
    n: int | None = None,
    axis: int = -1,
    *,
    overwrite_x: bool = False,
    norm: str = "backward",
) -> ArrayType:
    """Inverse complex FFT along ``axis``.

    Keywords as in :func:`fft`; ``norm="forward"`` leaves the inverse
    unscaled.
    """
    return _call("ifft", x, n, axis, overwrite_x, norm)


def rfft(x: ArrayType, n: int | None = None, axis: int = -1) -> ArrayType:
    """Forward FFT of real input along ``axis`` (non-negative frequencies)."""
    return _call("rfft", x, n, axis)


def irfft(x: ArrayType, n: int | None = None, axis: int = -1) -> ArrayType:
    """Inverse of :func:`rfft`; ``n`` is the output length."""
    return _call("irfft", x, n, axis)
