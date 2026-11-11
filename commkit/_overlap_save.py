"""Overlap-save block FFT filtering (private).

Shared by ``filtering.fir_filter`` (per-channel convolution of long records),
``equalization.zf_equalizer`` and ``equalization.apply_taps`` (per-bin MIMO
matrix multiply) through :func:`ols_apply`: the record is cut into overlapping FFT blocks, the
caller processes each block's bins, and each inverse block is kept without
a guard at either end, which holds the filter's transients (a quarter block
by default, so the blocks overlap by half).  The blocks are transformed in
batches, so the spectra take bounded memory whatever the record length.
"""

from collections.abc import Callable

from . import _fft
from .backend import ArrayType, dispatch

# Spectrum elements per batch of blocks (128 MB of complex64).
_OLS_ELEMENTS = 1 << 24


def ols_apply(
    samples: ArrayType,
    N_fft: int,
    process: Callable[[ArrayType], ArrayType],
    *,
    guard: int | None = None,
    real: bool = False,
) -> ArrayType:
    """Overlap-save filtering of ``(num_ch, N)`` samples, ``(num_ch, N)`` out.

    Parameters
    ----------
    samples : array_like
        Input samples, ``(num_ch, N)``, already on their device.
    N_fft : int
        FFT block size, a power of 2.
    process : callable
        Maps the spectra of a batch of blocks, ``(num_ch, blocks, N_fft)``,
        to the filtered spectra, ``(num_ch, blocks, N_fft)``.
    guard : int, optional
        Samples discarded at each end of an inverse block, at least the
        filter's extent on either side of lag 0 (its length for a causal
        filter), below ``N_fft // 2``.  ``None`` takes ``N_fft // 4``.
    real : bool, default False
        Real samples and a real filter: the blocks go through ``rfft`` and
        ``irfft``, and ``process`` sees the ``N_fft // 2 + 1`` bins of each.

    Returns
    -------
    ndarray
        The centers of the inverse blocks, ``(num_ch, N)``: complex, or real
        with ``real``.
    """
    _, xp, _ = dispatch(samples)
    num_ch, N = samples.shape
    discard = N_fft // 4 if guard is None else guard  # symmetric guard
    B = N_fft - 2 * discard  # hop
    num_blocks = (N + B - 1) // B

    # Pre-pad by discard so the first kept sample is sample 0; post-pad to
    # fill the last window.
    padded = xp.pad(samples, ((0, 0), (discard, num_blocks * B - N + discard)))
    stride = padded.strides
    windows = xp.lib.stride_tricks.as_strided(  # a view, no copy
        padded,
        shape=(num_ch, num_blocks, N_fft),
        strides=(stride[0], B * stride[1], stride[1]),
    )
    batch = max(1, _OLS_ELEMENTS // (num_ch * N_fft))
    out = None
    for b0 in range(0, num_blocks, batch):
        if real:
            spectra = _fft.rfft(windows[:, b0 : b0 + batch], n=N_fft, axis=-1)
            blocks = _fft.irfft(process(spectra), n=N_fft, axis=-1)
        else:
            spectra = _fft.fft(windows[:, b0 : b0 + batch], n=N_fft, axis=-1)
            blocks = _fft.ifft(process(spectra), n=N_fft, axis=-1)
        if out is None:
            out = xp.empty((num_ch, num_blocks, B), dtype=blocks.dtype)
        out[:, b0 : b0 + batch] = blocks[:, :, discard : discard + B]
    assert out is not None
    return out.reshape(num_ch, -1)[:, :N]
