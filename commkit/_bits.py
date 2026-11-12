"""MSB-first bit rows and the integers they label (private).

:func:`pack_bits` and :func:`unpack_bits`, on the device of their input,
shared by :mod:`commkit.mapping` (bit labels of constellation points) and
:mod:`commkit.coding` (codewords and syndromes as integers).  Below both in
the layering, so the value objects of ``coding`` need not import a DSP
module.
"""

from .backend import ArrayType, dispatch


def pack_bits(labels: ArrayType) -> ArrayType:
    """MSB-first bit rows ``(..., k)`` to integers, int64, on their device.

    The inverse of :func:`unpack_bits`.
    """
    labels, xp, _ = dispatch(labels)
    k = labels.shape[-1]
    weights = xp.left_shift(1, xp.arange(k - 1, -1, -1, dtype=xp.int64))
    return (labels.astype(xp.int64) * weights).sum(axis=-1)


def unpack_bits(indices: ArrayType, k: int) -> ArrayType:
    """Unpack integer indices into their k-bit binary representation (MSB-first).

    Runs with the array module of ``indices``, so a GPU-resident
    index array (e.g. from hard-decision demapping on CuPy) unpacks without
    a host round trip.

    Parameters
    ----------
    indices : array_like, shape (N,), integer dtype
    k : int
        Number of bits per index.

    Returns
    -------
    ndarray, shape (N, k), int8
        MSB-first bit pattern per index, on the device of ``indices``.
    """
    indices, xp, _ = dispatch(indices)
    shifts = xp.arange(k - 1, -1, -1, dtype="int32")
    return ((indices[:, xp.newaxis] >> shifts) & 1).astype(xp.int8)
