"""
Constellation geometry and Gray labelling.

Gray code sequences, their inverse, and the Gray-mapped constellation
generators for PSK / ASK / square-, 8-rect-, and cross-QAM.  These are the
geometric foundation consumed by the bit mapping, soft demapping, and
probabilistic-shaping layers.

Note: codes and constellations are generated using NumPy (host-side).
"""

from functools import lru_cache

import numpy as np

from ..logger import logger

__all__ = [
    "gray_code",
    "gray_constellation",
    "gray_to_binary",
]


@lru_cache(maxsize=128)
def gray_code(n: int) -> np.ndarray:
    """
    Generates a Gray code sequence for `n` bits.

    Gray coding ensures that adjacent symbols in the constellation differ
    by only one bit, minimizing bit error rate (BER) for a given symbol
    error rate (SER).

    Parameters
    ----------
    n : int
        Number of bits.

    Returns
    -------
    np.ndarray
        Array of integers representing the Gray code sequence.
        Shape: (2^n,).

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
    """
    Computes the inverse Gray code mapping: symbol index to bit pattern.

    In soft demapping, it is necessary to identify which bits correspond to
    each constellation point. Since `gray_code(n)[i]` provides the Gray
    code for natural binary `i`, this inverse mapping finds the natural
    binary representation corresponding to a specific Gray-coded index.

    Parameters
    ----------
    n : int
        Number of bits per symbol.

    Returns
    -------
    np.ndarray
        Array where `result[s]` provides the natural binary integer
        representing the bit pattern for symbol `s`. Shape: (2^n,).
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


@lru_cache(maxsize=128)
def gray_constellation(
    modulation: str,
    order: int,
    normalize: bool = True,
    unipolar: bool = False,
) -> np.ndarray:
    """
    Generate constellation points with Gray mapping.

    The returned array is indexed by the symbol's natural binary value.
    `constellation[i]` is the complex/float value for the symbol representing
    the bit pattern for integer `i`.

    Parameters
    ----------
    modulation : {"psk", "qam", "ask", "pam"}
        Modulation type. If string contains 'unipolar', unipolar=True is triggered.
    order : int
        Modulation order (number of symbols). Must be a power of 2.
    normalize : bool, default True
        If True, scales the constellation to unit average power (E_s = 1).
    unipolar : bool, default False
        If True, shifts the constellation to be strictly non-negative (for 'ask'/'pam').
        If the modulation string contains 'unipol', this is automatically forced to True.

    Returns
    -------
    np.ndarray
        Array of constellation points (NumPy). Shape: (order,).
        Complex for 'psk' and 'qam', Float for 'ask'.

    Notes
    -----
    By default, constellations are normalized to **unit average power (E_s = 1)**.
    This ensures that when different modulation schemes are mixed (e.g., PSK
    pilots with QAM payload), they start from a consistent power baseline.
    """
    logger.debug(
        f"Generating Gray-coded constellation: modulation={modulation}, order={order}, normalize={normalize}"
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
        elif k % 2 == 0:
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
            from .. import helpers

            result = helpers.normalize(result, mode="average_power")

    return result


def _gray_psk(order: int) -> np.ndarray:
    """
    Generates an M-PSK constellation with Gray mapping.

    Parameters
    ----------
    order : int
        Modulation order (must be a power of 2).

    Returns
    -------
    np.ndarray
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
    """
    Generates an M-ASK constellation with Gray mapping.

    By default, returns bipolar values centered at zero. If unipolar=True,
    shifts values to start from zero.

    Parameters
    ----------
    order : int
        Modulation order (must be a power of 2).

    Returns
    -------
    np.ndarray
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
    """
    Generates a Square M-QAM constellation with Gray mapping.

    Constructed as the Cartesian product of two Gray-coded ASK
    constellations. Valid for orders where log2(M) is even.

    Parameters
    ----------
    order : int
        Modulation order (must be an even power of 2, e.g., 16, 64).

    Returns
    -------
    np.ndarray
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

    return constellation


def _gray_qam_8_rect() -> np.ndarray:
    """
    Generates an 8-QAM constellation with Rectangular Gray mapping.

    Returns
    -------
    np.ndarray
        Complex array of 8-QAM constellation points.
    """

    points = np.zeros(8, dtype=complex)

    # 000 (I=-3, Q=-1)
    points[0] = -3.0 - 1.0j

    # 001 (I=-3, Q=+1)
    points[1] = -3.0 + 1.0j

    # 010 (I=-1, Q=-1)
    points[2] = -1.0 - 1.0j

    # 011 (I=-1, Q=+1)
    points[3] = -1.0 + 1.0j

    # 100 (I=+3, Q=-1)
    points[4] = 3.0 - 1.0j

    # 101 (I=+3, Q=+1)
    points[5] = 3.0 + 1.0j

    # 110 (I=+1, Q=-1)
    points[6] = 1.0 - 1.0j

    # 111 (I=+1, Q=+1)
    points[7] = 1.0 + 1.0j

    return points


def _gray_qam_cross(order: int) -> np.ndarray:
    """
    Generates a Cross-QAM constellation using a Quasi-Gray mapping.

    Cross constellations are used for QAM orders where log2(M) is odd (e.g., 32, 128)
    to maintain a more circular/compact shape than a purely rectangular grid.

    Parameters
    ----------
    order : int
        Modulation order (must be an odd power of 2).

    Returns
    -------
    np.ndarray
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

    # 1. Generate Gray sequences for underlying Rectangular Grid
    gray_i_seq = gray_code(n)
    gray_q_seq = gray_code(m)

    # 2. Build Inverse Lookup Tables (Symbol -> Geometric Index)
    inv_gray_i = np.zeros(width, dtype=int)
    inv_gray_i[gray_i_seq] = np.arange(width)

    inv_gray_q = np.zeros(height, dtype=int)
    inv_gray_q[gray_q_seq] = np.arange(height)

    # 3. Decompose input Symbols into I and Q components
    s = np.arange(order)
    mask_q = (1 << m) - 1
    idx_i_sym = s >> m
    idx_q_sym = s & mask_q  # This is the Gray-coded Symbol Value for Q

    # 4. Map Symbols to Geometric Indices (0..Width-1, 0..Height-1)
    geo_i = inv_gray_i[idx_i_sym]
    geo_q = inv_gray_q[idx_q_sym]

    # 5. Determine Folding Parameters
    n_shift = 0
    if n >= 3:
        n_shift = 2 ** (n - 3)

    if n_shift == 0:
        # Fallback
        val_i = (-width + 1 + 2 * geo_i).astype(float)
        val_q = (-height + 1 + 2 * geo_q).astype(float)
        return val_i + 1j * val_q

    # 6. Identify Wings
    mask_left = geo_i < n_shift
    mask_right = geo_i >= (width - n_shift)

    # 7. Calculate Rotation/Translation
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

    return constellation
