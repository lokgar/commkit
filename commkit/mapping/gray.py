"""Gray codes and Gray-labeled constellation geometry.

:func:`gray_code` and its inverse :func:`gray_to_binary`, and the private
generators of the Gray-mapped PSK, ASK (PAM), square-, 8-rect- and cross-QAM
constellations behind the :class:`Constellation` factories, together with
the nearest-point search of hard decisions.  Codes and constellations are
built with NumPy on the host.
"""

from functools import lru_cache
from types import ModuleType

import numpy as np

from .. import _cuda
from ..backend import ArrayType, dispatch
from ..logger import logger

__all__ = ["gray_code", "gray_to_binary"]


@lru_cache(maxsize=128)
def gray_code(n: int) -> np.ndarray:
    """Generate the binary-reflected Gray code of ``n`` bits.

    ``gray_code(n)[i] = i ^ (i >> 1)``: consecutive entries differ in one
    bit, so neighboring constellation points labeled in this order differ
    in one bit, which minimizes the bit errors per symbol error.

    Parameters
    ----------
    n : int
        Number of bits, ``>= 0``.

    Returns
    -------
    numpy.ndarray
        Host integer code words, ``(2**n,)``.

    Examples
    --------
    >>> gray_code(2)
    array([0, 1, 3, 2])
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return np.array([0], dtype=int)

    # Vectorized implementation: s ^ (s >> 1)
    i = np.arange(1 << n, dtype=int)

    return i ^ (i >> 1)


@lru_cache(maxsize=128)
def gray_to_binary(n: int) -> np.ndarray:
    """Invert the Gray code: the binary index of each Gray code word.

    ``gray_to_binary(n)[gray_code(n)[i]] = i``, so ``gray_to_binary(n)[g]``
    is the position of code word ``g`` in the Gray sequence.

    Parameters
    ----------
    n : int
        Number of bits, ``>= 0``.

    Returns
    -------
    numpy.ndarray
        Host integers, ``(2**n,)``.

    Examples
    --------
    >>> gray_to_binary(3)
    array([0, 1, 3, 2, 7, 6, 4, 5])
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return np.array([0], dtype=int)

    # gray_code gives: natural_binary -> gray_symbol
    # We want: gray_symbol -> natural_binary (inverse)
    gray = gray_code(n)
    inverse = np.zeros(1 << n, dtype=int)
    inverse[gray] = np.arange(1 << n)

    return inverse


def _is_square_qam(modulation: str, order: int) -> bool:
    """True if ``(modulation, order)`` builds a square (not cross/8-rect) QAM grid.

    Square QAM (order a perfect square, e.g. 4/16/64/256/1024) is a uniform
    per-axis grid; 8-QAM and cross-QAM (odd bits-per-symbol orders, e.g.
    32/128/512) are not.
    """
    if "qam" not in modulation.lower():
        return False
    if order == 8:
        return False
    k = int(np.log2(order))
    return k % 2 == 0


@lru_cache(maxsize=128)
def _gray_points(
    modulation: str,
    order: int,
    normalize: bool = True,
    unipolar: bool = False,
) -> np.ndarray:
    """Gray-labeled constellation points, indexed by their label.

    ``points[i]`` is the point whose bit label is the binary value of ``i``.

    Parameters
    ----------
    modulation : {"psk", "qam", "ask", "pam"}
        Modulation family; a name containing ``"unipol"`` also sets
        ``unipolar``.
    order : int
        Number of points, a power of 2.
    normalize : bool, default True
        Scale to unit average power ``Es = 1``, so that formats combined in
        one frame (PSK pilots with a QAM payload) start at the same power.
    unipolar : bool, default False
        Shift ASK/PAM points to non-negative values.

    Returns
    -------
    numpy.ndarray
        Points, ``(order,)``: complex for PSK and QAM, real for ASK.
    """
    logger.debug(
        "Generating Gray-coded constellation: modulation=%s, order=%s, normalize=%s",
        modulation,
        order,
        normalize,
    )
    modulation = modulation.lower()

    # Extract core modulation scheme
    if "psk" in modulation:
        modulation = "psk"
    elif "qam" in modulation:
        modulation = "qam"
    elif "ask" in modulation or "pam" in modulation:
        modulation = "ask"

    if order < 2:
        raise ValueError("Order must be at least 2 for modulation")

    if modulation == "psk":
        result = _gray_psk(order)
    elif modulation == "ask":
        result = _gray_ask(order, unipolar=unipolar)
    elif modulation == "qam":
        k = int(np.log2(order))
        if 2**k != order:
            raise ValueError(f"Order must be power of 2 for {modulation}")

        # Check for 8-QAM
        if order == 8:
            result = _gray_qam_8_rect()
        elif _is_square_qam(modulation, order):
            result = _gray_qam_square(order)
        else:
            result = _gray_qam_cross(order)
    else:
        raise ValueError(f"Unsupported modulation type: {modulation}")

    if normalize:
        if modulation == "psk":
            # PSK is already on unit circle (E_s = 1)
            pass
        else:
            from ..math import normalize as _normalize

            result = _normalize(result, mode="average_power")

    return result


def _pack(labels: np.ndarray) -> np.ndarray:
    """MSB-first bit rows to integers, int64: the inverse of :func:`_unpack_bits`."""
    k = labels.shape[1]
    return labels.astype(np.int64) @ (1 << np.arange(k - 1, -1, -1, dtype=np.int64))


def _unpack_bits(indices: ArrayType, k: int) -> ArrayType:
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


# GPU input outside the kernel's limits (double precision, more than 1024
# points): budget of the (chunk, M) distance matrix per chunk, in elements.
_GPU_CHUNK_ELEMENTS = 1 << 22


def _nearest_index(
    x: ArrayType, constellation: ArrayType, chunk: int = 4096
) -> ArrayType:
    """Nearest-constellation-point index for each element of ``x``.

    On the CPU, chunking over the ``N`` axis bounds the peak memory of the
    ``(N, M)`` distance matrix.  On the GPU, single-precision input and up
    to 1024 points take one launch of the ``bps_min_d2`` kernel's
    nearest-point mode (no distance matrix; it computes in float32, the
    limit of its table); other input runs the chunked loop with chunks of
    ``_GPU_CHUNK_ELEMENTS`` distances, so a long record is not hundreds of
    small launches.

    Parameters
    ----------
    x : array_like
        Flat query points, ``(N,)``, NumPy or CuPy.
    constellation : array_like
        Reference constellation points, ``(M,)``.  Moved to ``x``'s device.
    chunk : int, default 4096
        ``N``-axis chunk size on the CPU (and the minimum on the GPU).

    Returns
    -------
    ndarray
        int64 index into ``constellation`` of the nearest point, ``(N,)``, on
        the device of ``x``.  Ties go to the lower index.
    """
    x, xp, _ = dispatch(x)
    constellation = xp.asarray(constellation)
    if xp is not np:
        if x.dtype in (np.complex64, np.float32) and constellation.size <= 1024:
            return _nearest_index_kernel(x, constellation, xp)
        chunk = max(chunk, _GPU_CHUNK_ELEMENTS // constellation.size)
    indices = xp.empty(len(x), dtype=xp.int64)
    for n0 in range(0, len(x), chunk):
        n1 = min(n0 + chunk, len(x))
        d = xp.abs(x[n0:n1, xp.newaxis] - constellation[xp.newaxis, :])
        indices[n0:n1] = xp.argmin(d, axis=1)
    return indices


def _nearest_index_kernel(
    x: ArrayType, constellation: ArrayType, xp: ModuleType
) -> ArrayType:
    """GPU nearest-point indices from ``bps_min_d2``, ``(N,)`` int64.

    For single-precision ``x`` and at most 1024 points.  The kernel compares
    squared distances in float32, which can break a tie of the distances
    differently from ``abs`` only where two points are equally near up to
    float32 rounding.
    """
    if x.size == 0:
        return xp.empty(0, dtype=xp.int64)
    kernel = _cuda.get_kernel("bps_min_d2", mode="table", return_argmin=True)
    _, idx = kernel(
        x.astype(np.complex64, copy=False).reshape(1, -1),
        xp.ones(1, dtype=np.complex64),
        constellation.astype(np.complex64, copy=False),
    )
    return idx.reshape(-1).astype(np.int64)


def _gray_psk(order: int) -> np.ndarray:
    """Generate an M-PSK constellation with Gray mapping.

    Parameters
    ----------
    order : int
        Modulation order (must be a power of 2).

    Returns
    -------
    numpy.ndarray
        Complex array of PSK constellation points.
    """
    # Bits per symbol
    k = int(np.log2(order))
    if 2**k != order:
        raise ValueError("Order must be power of 2 for psk")

    # Gray codes for k bits
    gray = gray_code(k)

    # BPSK Special Case: strictly real values
    if order == 2:
        points = np.array([-1.0, 1.0], dtype=float)
        constellation = np.zeros(order, dtype=float)
        constellation[gray] = points
        return constellation

    # Geometric points (phases)
    # Phases are 0, 2pi/M, ..., 2pi(M-1)/M
    phases = np.arange(order) * 2 * np.pi / order
    points = np.exp(1j * phases)

    constellation = np.zeros(order, dtype=complex)
    constellation[gray] = points

    return constellation


def _gray_ask(order: int, unipolar: bool = False) -> np.ndarray:
    """Generate an M-ASK constellation with Gray mapping.

    By default, returns bipolar values centered at zero. If unipolar=True,
    shifts values to start from zero.

    Parameters
    ----------
    order : int
        Modulation order (must be a power of 2).
    unipolar : bool, default False
        Shift the levels to start at zero.

    Returns
    -------
    numpy.ndarray
        Real-valued array of ASK constellation points.
    """
    k = int(np.log2(order))
    if 2**k != order:
        raise ValueError("Order must be power of 2 for ask")

    gray = gray_code(k)
    # Points are symmetric: -M+1, -M+3, ..., M-3, M-1
    points = np.linspace(-order + 1, order - 1, order)

    if unipolar:
        points = points - np.min(points)

    constellation = np.zeros(order, dtype=float)
    constellation[gray] = points

    return constellation


def _gray_qam_square(order: int) -> np.ndarray:
    """Generate a Square M-QAM constellation with Gray mapping.

    Constructed as the Cartesian product of two Gray-coded ASK
    constellations. Valid for orders where log2(M) is even.

    Parameters
    ----------
    order : int
        Modulation order (must be an even power of 2, e.g., 16, 64).

    Returns
    -------
    numpy.ndarray
        Complex array of square QAM constellation points.
    """
    # M = 2^(2k). I bits = k, Q bits = k.
    k_total = int(np.log2(order))
    n = k_total // 2  # bits per axis

    # Generate ASK for each axis
    m_axis = 2**n
    pam = _gray_ask(m_axis)

    # Cartesian product
    # Symbol s (2n bits). High n bits -> I, Low n bits -> Q.
    s = np.arange(order)
    mask_q = (1 << n) - 1

    idx_i = s >> n
    idx_q = s & mask_q

    i_vals = pam[idx_i]
    q_vals = pam[idx_q]

    constellation = i_vals + 1j * q_vals

    return np.asarray(constellation)


def _gray_qam_8_rect() -> np.ndarray:
    """Generate an 8-QAM constellation with Rectangular Gray mapping.

    A 4-level Gray I axis (the two MSBs) times a 2-level Q axis (the LSB):
    label ``b2 b1 b0`` sits at ``I = gray4[b2 b1]``, ``Q = gray2[b0]``.

    Returns
    -------
    numpy.ndarray
        Complex array of 8-QAM constellation points.
    """
    return (_gray_ask(4)[:, None] + 1j * _gray_ask(2)[None, :]).ravel()


def _gray_qam_cross(order: int) -> np.ndarray:
    """Generate a Cross-QAM constellation using a Quasi-Gray mapping.

    Cross constellations are used for QAM orders where log2(M) is odd (e.g., 32, 128)
    to maintain a more circular/compact shape than a purely rectangular grid.

    Parameters
    ----------
    order : int
        Modulation order (must be an odd power of 2).

    Returns
    -------
    numpy.ndarray
        Complex array of cross QAM constellation points.
    """
    # M = 2^(2k+1).
    k_total = int(np.log2(order))

    # I bits n = ceil(k_total/2) = (k+1)//2
    # Q bits m = floor(k_total/2) = k // 2
    n = (k_total + 1) // 2
    m = k_total // 2

    width = 2**n
    height = 2**m

    # 1. Inverse Gray tables of the underlying rectangular grid
    #    (Gray symbol -> geometric index)
    inv_gray_i = gray_to_binary(n)
    inv_gray_q = gray_to_binary(m)

    # 2. Decompose input Symbols into I and Q components
    s = np.arange(order)
    mask_q = (1 << m) - 1
    idx_i_sym = s >> m
    idx_q_sym = s & mask_q  # This is the Gray-coded Symbol Value for Q

    # 3. Map Symbols to Geometric Indices (0..Width-1, 0..Height-1)
    geo_i = inv_gray_i[idx_i_sym]
    geo_q = inv_gray_q[idx_q_sym]

    # 4. Determine Folding Parameters
    n_shift = 0
    if n >= 3:
        n_shift = 2 ** (n - 3)

    if n_shift == 0:
        # No wings to fold: the rectangular grid is the constellation.
        val_i = (-width + 1 + 2 * geo_i).astype(float)
        val_q = (-height + 1 + 2 * geo_q).astype(float)
        return np.asarray(val_i + 1j * val_q)

    # 5. Identify Wings
    mask_left = geo_i < n_shift
    mask_right = geo_i >= (width - n_shift)

    # 6. Calculate Rotation/Translation
    center_offset = (width - height) // 2

    # Left Wing -> Top Cap
    rotated_i_left = geo_q + center_offset
    rotated_q_left = height + (n_shift - 1 - geo_i)

    # Right Wing -> Bottom Cap
    rotated_i_right = geo_q + center_offset
    rotated_q_right = -1 - (geo_i - (width - n_shift))

    # Apply Updates
    geo_i_final = geo_i.copy()
    geo_q_final = geo_q.copy()

    geo_i_final = np.where(mask_left, rotated_i_left, geo_i_final)
    geo_i_final = np.where(mask_right, rotated_i_right, geo_i_final)

    geo_q_final = np.where(mask_left, rotated_q_left, geo_q_final)
    geo_q_final = np.where(mask_right, rotated_q_right, geo_q_final)

    # Convert Final Geometric Indices to Values
    final_i_vals = (-width + 1 + 2 * geo_i_final).astype(float)
    final_q_vals = (-height + 1 + 2 * geo_q_final).astype(float)

    constellation = final_i_vals + 1j * final_q_vals

    return np.asarray(constellation)
