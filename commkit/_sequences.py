"""Synchronization sequences (private).

Barker and Zadoff-Chu generators and the per-stream ZC root assignment.
They live below ``core`` so that ``core.frame`` can build preambles without
importing a DSP module; ``commkit.timing`` exposes the generators publicly.
"""

import numpy as np

from .logger import logger

# Standard Barker codes
_BARKER_SEQUENCES = {
    2: [1, -1],
    3: [1, 1, -1],
    4: [1, 1, -1, 1],
    5: [1, 1, 1, -1, 1],
    7: [1, 1, 1, -1, -1, 1, -1],
    11: [1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1],
    13: [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1],
}


def barker_sequence(length: int) -> np.ndarray:
    """
    Generates a Barker sequence of the specified length.

    Barker sequences are binary sequences (+1, -1) with optimal cyclic
    auto-correlation properties, where the sidelobes are at most 1. They are
    widely used for frame synchronization and pulse compression.

    Parameters
    ----------
    length : {2, 3, 4, 5, 7, 11, 13}
        Total length of the Barker sequence.

    Returns
    -------
    array_like
        BPSK symbols (+1.0, -1.0). Shape: (length,).
        NumPy array; move it with ``to_device`` if needed.

    Raises
    ------
    ValueError
        If the requested length is not a valid Barker length.

    Examples
    --------
    >>> barker_sequence(7)
    array([ 1.,  1.,  1., -1., -1.,  1., -1.], dtype=float32)
    """
    if length not in _BARKER_SEQUENCES:
        valid = sorted(_BARKER_SEQUENCES.keys())
        raise ValueError(f"No Barker sequence of length {length}. Valid: {valid}")

    seq = np.array(_BARKER_SEQUENCES[length], dtype="float32")

    logger.debug("Generated Barker-%s sequence.", length)
    return seq


def zadoff_chu_sequence(length: int, *, root: int = 1) -> np.ndarray:
    r"""
    Generates a Zadoff-Chu (ZC) synchronization sequence.

    ZC sequences are Complex-valued, Constant Amplitude Zero
    Auto-Correlation (CAZAC) sequences. They possess the unique property
    that their periodic auto-correlation is zero at all non-zero lags,
    and their DFT is also a ZC sequence. This makes them ideal for
    timing and frequency synchronization in systems like LTE and 5G NR.

    Parameters
    ----------
    length : int
        The sequence length (N_ZC). For optimal cross-correlation
        properties, this should be a prime number.
    root : int, default 1
        The root index (u). Must be relatively prime to `length`.

    Returns
    -------
    array_like
        Complex Zadoff-Chu symbols of unit magnitude.
        Shape: (length,). Data type: `complex64`.

    Notes
    -----
    - For odd lengths: x[n] = exp(-j * pi * u * n * (n + 1) / N_ZC)
    - For even lengths: x[n] = exp(-j * pi * u * n^2 / N_ZC)
    - ZC sequences have exceptionally low Peak-to-Average Power Ratio (PAPR).
    """
    if length < 1:
        raise ValueError("Length must be positive.")
    if root < 1 or root >= length:
        raise ValueError(f"Root must be in [1, {length - 1}].")

    xp = np

    # ZC formula: x[n] = exp(-j * pi * u * n * (n+1) / N)
    n = xp.arange(length)
    if length % 2 == 0:
        # Even length: x[n] = exp(-j * pi * u * n^2 / N)
        seq = xp.exp(-1j * xp.pi * root * n * n / length)
    else:
        # Odd length: x[n] = exp(-j * pi * u * n * (n+1) / N)
        seq = xp.exp(-1j * xp.pi * root * n * (n + 1) / length)

    out: np.ndarray = seq.astype(xp.complex64)

    logger.debug("Generated ZC sequence: length=%s, root=%s.", length, root)
    return out


def zc_mimo_root(stream_idx: int, base_root: int, length: int) -> int:
    """
    Returns the Zadoff-Chu root for TX stream ``stream_idx`` in a MIMO preamble.

    Assigns a deterministic unique root to each TX stream by cycling through
    distinct roots starting from ``base_root``, wrapping in the range
    ``[1, length-1]``.  For prime ``length`` all roots are valid CAZAC
    sequences; any two distinct roots are near-orthogonal with cross-correlation
    magnitude ``1/sqrt(length)`` at every lag.

    Parameters
    ----------
    stream_idx : int
        TX stream index (0-based).
    base_root : int
        ZC root assigned to stream 0.  Must be in ``[1, length-1]``.
    length : int
        Sequence length (should be prime for the CAZAC property).

    Returns
    -------
    int
        ZC root for stream ``stream_idx``, guaranteed in ``[1, length-1]``.

    Examples
    --------
    >>> [zc_mimo_root(k, 1, 13) for k in range(4)]
    [1, 2, 3, 4]
    >>> [zc_mimo_root(k, 10, 13) for k in range(4)]
    [10, 11, 12, 1]
    """
    return ((base_root - 1 + stream_idx) % (length - 1)) + 1
