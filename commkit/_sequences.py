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


def barker_sequence(num_symbols: int) -> np.ndarray:
    """Generate a Barker sequence.

    Barker sequences are binary (+1, -1) sequences whose aperiodic
    autocorrelation sidelobes are at most 1 in magnitude, used for frame
    synchronization and pulse compression.

    Parameters
    ----------
    num_symbols : {2, 3, 4, 5, 7, 11, 13}
        Sequence length in symbols.

    Returns
    -------
    ndarray
        Host float32 BPSK symbols (+1, -1), ``(num_symbols,)``; move them with
        ``to_device`` if needed.

    Raises
    ------
    ValueError
        If ``num_symbols`` is not a Barker length.

    Examples
    --------
    >>> barker_sequence(7)
    array([ 1.,  1.,  1., -1., -1.,  1., -1.], dtype=float32)
    """
    if num_symbols not in _BARKER_SEQUENCES:
        valid = sorted(_BARKER_SEQUENCES.keys())
        raise ValueError(f"No Barker sequence of length {num_symbols}. Valid: {valid}")

    seq = np.array(_BARKER_SEQUENCES[num_symbols], dtype="float32")

    logger.debug("Generated Barker-%s sequence.", num_symbols)
    return seq


def zadoff_chu_sequence(num_symbols: int, *, root: int = 1) -> np.ndarray:
    r"""Generate a Zadoff-Chu (ZC) synchronization sequence.

    ZC sequences are constant-amplitude zero-autocorrelation (CAZAC)
    sequences: the periodic autocorrelation vanishes at every non-zero lag,
    and the DFT of a ZC sequence is again one.  They serve timing and
    frequency synchronization (LTE, 5G NR).

    Parameters
    ----------
    num_symbols : int
        Sequence length ``N_ZC``; a prime gives the lowest cross-correlation
        between roots.
    root : int, default 1
        Root index ``u`` in ``[1, num_symbols)``, coprime to ``num_symbols`` (not
        checked).

    Returns
    -------
    ndarray
        Host complex64 symbols of unit magnitude, ``(num_symbols,)``.

    Raises
    ------
    ValueError
        If ``num_symbols < 1`` or ``root`` is outside ``[1, num_symbols)``.

    Notes
    -----
    - For odd lengths: x[n] = exp(-j * pi * u * n * (n + 1) / N_ZC)
    - For even lengths: x[n] = exp(-j * pi * u * n^2 / N_ZC)
    """
    if num_symbols < 1:
        raise ValueError(f"num_symbols must be positive, got {num_symbols}.")
    if root < 1 or root >= num_symbols:
        raise ValueError(f"Root must be in [1, {num_symbols - 1}].")

    xp = np

    # ZC formula: x[n] = exp(-j * pi * u * n * (n+1) / N)
    n = xp.arange(num_symbols)
    if num_symbols % 2 == 0:
        # Even length: x[n] = exp(-j * pi * u * n^2 / N)
        seq = xp.exp(-1j * xp.pi * root * n * n / num_symbols)
    else:
        # Odd length: x[n] = exp(-j * pi * u * n * (n+1) / N)
        seq = xp.exp(-1j * xp.pi * root * n * (n + 1) / num_symbols)

    out: np.ndarray = seq.astype(xp.complex64)

    logger.debug("Generated ZC sequence: num_symbols=%s, root=%s.", num_symbols, root)
    return out


def zc_mimo_root(channel_index: int, base_root: int, num_symbols: int) -> int:
    """Return the Zadoff-Chu root for channel ``channel_index`` in a MIMO preamble.

    Assigns a deterministic unique root to each channel by cycling through
    distinct roots starting from ``base_root``, wrapping in the range
    ``[1, num_symbols-1]``.  For a prime ``num_symbols`` all roots are valid CAZAC
    sequences; any two distinct roots are near-orthogonal with cross-correlation
    magnitude ``1/sqrt(num_symbols)`` at every lag.

    Parameters
    ----------
    channel_index : int
        Channel index (0-based).
    base_root : int
        ZC root assigned to channel 0.  Must be in ``[1, num_symbols-1]``.
    num_symbols : int
        Sequence length (should be prime for the CAZAC property).

    Returns
    -------
    int
        ZC root for channel ``channel_index``, guaranteed in ``[1, num_symbols-1]``.

    Examples
    --------
    >>> [zc_mimo_root(k, 1, 13) for k in range(4)]
    [1, 2, 3, 4]
    >>> [zc_mimo_root(k, 10, 13) for k in range(4)]
    [10, 11, 12, 1]
    """
    return ((base_root - 1 + channel_index) % (num_symbols - 1)) + 1
