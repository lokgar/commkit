"""Overlap-save block FFT scaffolding (private).

Shared by ``filtering.ols_fir_filter`` (per-channel convolution) and
``equalization.zf_equalizer`` (per-bin MIMO matrix multiply): the forward pass
cuts the record into 50 %-overlapping FFT blocks, the caller processes each
bin, and the backward pass inverts and keeps the centre of each block.
"""

from typing import Any

from .backend import ArrayType, dispatch


def ols_forward(samples: ArrayType, N_fft: int) -> tuple[ArrayType, dict[str, Any]]:
    """
    Overlap-and-save forward pass: block windowing and batch FFT.

    This is the shared OLS scaffolding used by both ``ols_fir_filter`` (SISO
    scalar convolution) and ``zf_equalizer`` (MIMO per-bin matrix multiply).
    It should be called on samples that have already been dispatched to the
    correct backend and shaped as ``(num_ch, N)``.

    Parameters
    ----------
    samples : array_like
        Input samples. Shape: ``(num_ch, N)``. Must be 2-D.
    N_fft : int
        FFT block size. Must be a power of 2 and satisfy
        ``N_fft // 4 >= filter_length`` so the causal/anti-causal guard
        regions fully contain the filter transients.

    Returns
    -------
    Y : array_like
        Batch FFT of all OLS windows. Shape: ``(num_ch, num_blocks, N_fft)``.
    meta : dict
        Scaffold parameters required by ``ols_backward``:
        ``{'N': int, 'B': int, 'discard': int, 'num_blocks': int}``.
    """
    _, xp, _ = dispatch(samples)
    num_ch, N = samples.shape
    B = N_fft // 2  # 50 % hop - maximises block reuse
    discard = N_fft // 4  # symmetric guard: absorbs causal & anti-causal transients
    num_blocks = (N + B - 1) // B

    # Pre-pad by discard so the first valid output aligns with sample 0.
    # Post-pad to fill the last block window completely.
    pad_left = discard
    pad_right = num_blocks * B - N + discard
    samples_padded = xp.pad(samples, ((0, 0), (pad_left, pad_right)))

    # Zero-copy window extraction via as_strided (view, not copy).
    stride = samples_padded.strides
    windows = xp.lib.stride_tricks.as_strided(
        samples_padded,
        shape=(num_ch, num_blocks, N_fft),
        strides=(stride[0], B * stride[1], stride[1]),
    )

    Y = xp.fft.fft(windows, n=N_fft, axis=-1)  # (num_ch, num_blocks, N_fft)
    meta = {"N": N, "B": B, "discard": discard, "num_blocks": num_blocks}
    return Y, meta


def ols_backward(X_hat_f: ArrayType, meta: dict[str, Any]) -> ArrayType:
    """
    Overlap-and-save backward pass: batch IFFT, symmetric discard, reshape.

    Parameters
    ----------
    X_hat_f : array_like
        Frequency-domain blocks after per-bin processing.
        Shape: ``(num_ch, num_blocks, N_fft)``.
    meta : dict
        Scaffold parameters returned by ``ols_forward``.

    Returns
    -------
    array_like
        Time-domain output trimmed to the original signal length ``N``.
        Shape: ``(num_ch, N)``.
    """
    _, xp, _ = dispatch(X_hat_f)
    N = meta["N"]
    B = meta["B"]
    discard = meta["discard"]
    N_fft = X_hat_f.shape[-1]
    num_ch = X_hat_f.shape[0]

    x_hat = xp.fft.ifft(X_hat_f, n=N_fft, axis=-1)
    # Keep the center B samples of each block (symmetric discard of guard regions).
    valid = x_hat[:, :, discard : discard + B]
    out = valid.reshape(num_ch, -1)[:, :N]
    return out
