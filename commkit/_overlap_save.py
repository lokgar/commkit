"""Overlap-save block FFT filtering (private).

Shared by ``filtering.ols_fir_filter`` (per-channel convolution) and
``equalization.zf_equalizer`` (per-bin MIMO matrix multiply) through
:func:`ols_apply`: the record is cut into 50 %-overlapping FFT blocks, the
caller processes each block's bins, and the center of each inverse block is
kept.  The blocks are transformed in batches, so the spectra take bounded
memory whatever the record length.
"""

from collections.abc import Callable

from . import _fft
from .backend import ArrayType, dispatch

# Spectrum elements per batch of blocks (128 MB of complex64).
_OLS_ELEMENTS = 1 << 24


def ols_apply(
    samples: ArrayType, N_fft: int, process: Callable[[ArrayType], ArrayType]
) -> ArrayType:
    """Overlap-save filtering of ``(num_ch, N)`` samples, ``(num_ch, N)`` out.

    Parameters
    ----------
    samples : array_like
        Input samples, ``(num_ch, N)``, already on their device.
    N_fft : int
        FFT block size, a power of 2 with ``N_fft // 4`` at least the filter
        length, so the guard of a quarter block on each side holds the
        causal and anti-causal transients.
    process : callable
        Maps the spectra of a batch of blocks, ``(num_ch, blocks, N_fft)``,
        to the filtered spectra, ``(num_ch, blocks, N_fft)``.

    Returns
    -------
    ndarray
        The centers of the inverse blocks, complex, ``(num_ch, N)``.
    """
    _, xp, _ = dispatch(samples)
    num_ch, N = samples.shape
    B = N_fft // 2  # 50 % hop
    discard = N_fft // 4  # symmetric guard
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
        spectra = _fft.fft(windows[:, b0 : b0 + batch], n=N_fft, axis=-1)
        blocks = _fft.ifft(process(spectra), n=N_fft, axis=-1)
        if out is None:
            out = xp.empty((num_ch, num_blocks, B), dtype=blocks.dtype)
        out[:, b0 : b0 + batch] = blocks[:, :, discard : discard + B]
    assert out is not None
    return out.reshape(num_ch, -1)[:, :N]
