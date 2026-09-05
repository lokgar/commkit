"""
Hard bit mapping and demapping.

Maps a flat bit sequence to Gray-coded symbols (``map_bits``) and recovers the
most-likely bits from received symbols via minimum-distance decoding
(``demap_symbols_hard``).
"""

import numpy as np

from ..backend import ArrayType, dispatch
from ..core._signal_adapter import prepare_signal_input
from ..core.signal import Signal
from ..logger import logger
from .gray import (
    _is_square_qam,
    gray_code,
    gray_constellation,
    nearest_constellation_index,
    unpack_bits,
)
from .shaping import rescale_ps_symbols

__all__ = ["demap_symbols_hard", "map_bits"]

# map_bits takes a bit sequence, not a signal - Signal-awareness does not
# apply (bits aren't a field a Signal carries).  demap_symbols_hard below
# reads .resolved_symbols and writes .resolved_bits.


def map_bits(
    bits: ArrayType,
    modulation: str,
    order: int,
    unipolar: bool = False,
) -> ArrayType:
    """
    Maps a bit sequence to complex or real symbols.

    This function follows a bit-first architecture: it takes a flat sequence
    of bits and packs them into symbols according to the modulation scheme
    and Gray mapping.

    Output dtype is ``complex64`` for PSK/QAM and ``float32`` for ASK/PAM.

    Parameters
    ----------
    bits : array_like
        Input binary sequence (0s and 1s).
    modulation : {"psk", "qam", "ask", "pam"}
        Modulation scheme.
    order : int
        Modulation order (number of symbols).
    unipolar : bool, default False
        Trigger unipolar mapping for ASK/PAM.

    Returns
    -------
    array_like
        Array of symbols. Shape: (N_bits / log2(order),).
        Backend (NumPy/CuPy) matches the input `bits`.
    """
    logger.debug("Mapping bits to %s %s-level symbols.", modulation.upper(), order)
    bits, xp, _ = dispatch(bits)

    if order < 2:
        raise ValueError("Order must be at least 2 for modulation")

    k = int(np.log2(order))
    if 2**k != order:
        raise ValueError(f"Order must be a power of 2, got {order}")

    if len(bits) % k != 0:
        raise ValueError(
            f"Number of bits ({len(bits)}) must be divisible by bits per symbol ({k})"
        )

    # Reshape bits into symbols (N/k, k)
    num_symbols = len(bits) // k

    # Pack bits into integer indices
    bits_reshaped = bits.reshape((num_symbols, k))
    powers = 2 ** xp.arange(k - 1, -1, -1, dtype=int)
    indices = xp.sum(bits_reshaped * powers, axis=1)

    # Get constellation (returns NumPy)
    constellation = gray_constellation(modulation, order, unipolar=unipolar)

    # Ensure constellation is on the same backend and dtype
    constellation = xp.asarray(constellation)

    # ASK/PAM constellations are real-valued; PSK/QAM are complex.
    mod_lower = modulation.lower()

    if "ask" in mod_lower or "pam" in mod_lower:
        constellation = constellation.astype(xp.float32)
    else:
        constellation = constellation.astype(xp.complex64)

    # Map indices to points
    return constellation[indices]


def demap_symbols_hard(
    symbols: ArrayType | Signal,
    modulation: str | None = None,
    order: int | None = None,
    unipolar: bool = False,
    pmf: np.ndarray | None = None,
) -> ArrayType | Signal:
    """
    Maps complex symbols back to a sequence of bits (hard decisions).

    This function performs minimum Euclidean distance decoding to find the
    most likely bit pattern for each received symbol.

    Parameters
    ----------
    symbols : array_like
        Input array of received symbols. Shape: (..., N_symbols).
    modulation : {"psk", "qam", "ask"}
        Modulation type.  Required for array input; for :class:`Signal`
        input, used only as a fallback when the signal's ``mod_scheme`` is
        unset.
    order : int
        Modulation order.  Required for array input; for :class:`Signal`
        input, used only as a fallback when the signal's ``mod_order`` is
        unset.
    unipolar : bool, default False
        Trigger unipolar demapping for ASK/PAM.
    pmf : np.ndarray, optional
        Symbol PMF of shape ``(order,)`` for PS-QAM.  When provided, the
        input symbols are scaled by ``sqrt(E_PS)`` (where
        ``E_PS = Σ P(s_m) |s_m|²`` on the normalised grid) before the
        nearest-neighbour search, mapping unit-avg-power resolved symbols
        back to the ``{s_m}`` grid used by ``gray_constellation``.
        Use this when ``symbols`` comes from
        ``resolved_symbols`` of a PS-QAM signal.
        Has no effect for uniform modulations.  For :class:`Signal` input,
        used only as a fallback when the signal's ``ps_pmf`` is unset.

    Returns
    -------
    array_like
        Sequence of bits (0s and 1s). Shape: (..., N_symbols * log2(order)).
        The backend (NumPy/CuPy) matches the input `symbols`.
    """
    context = prepare_signal_input(
        symbols, function_name="demap_symbols_hard()", field="resolved_symbols"
    )
    if context.signal is not None:
        sig = context.signal
        if sig.signal_type is not None:
            logger.warning(
                "demap_symbols_hard() called on a frame-generated signal - skipping. "
                "Extract the payload segment via frame.get_structure_map() and build "
                "a plain Signal before demapping."
            )
            return sig._shallow_clone()
        mod = context.optional("mod_scheme", modulation)
        ord_ = context.optional("mod_order", order)
        if mod is None or ord_ is None:
            raise ValueError("Modulation scheme and order required for demapping.")
        if sig.resolved_symbols is None:
            raise ValueError(
                "No resolved symbols available. Call resolve_symbols(sig) first."
            )
        eff_pmf = context.optional("ps_pmf", pmf)
        # Output and input use different derived fields, so replace the target
        # field explicitly.
        resolved_bits = _demap_symbols_hard_array(
            context.array,
            mod,
            ord_,
            unipolar=context.optional("mod_unipolar", unipolar),
            pmf=eff_pmf,
        )
        return context.replace_field("resolved_bits", resolved_bits)

    if modulation is None or order is None:
        raise ValueError("demap_symbols_hard() requires modulation and order.")
    return _demap_symbols_hard_array(symbols, modulation, order, unipolar, pmf)


def _demap_symbols_hard_array(
    symbols: ArrayType,
    modulation: str,
    order: int,
    unipolar: bool = False,
    pmf: np.ndarray | None = None,
) -> ArrayType:
    """Array-only hard-decision implementation."""

    logger.debug("Demapping %s %s-level symbols to bits.", modulation.upper(), order)
    symbols, xp, _ = dispatch(symbols)

    # Capture original shape to restore structure later
    # If input is (N,), output (N*k,)
    # If input is (C, N), output (C, N*k)
    original_shape = symbols.shape

    # Ensure 1D for vectorized distance broadcasting
    symbols_flat = symbols.flatten()

    # Get constellation
    constellation = gray_constellation(modulation, order, unipolar=unipolar)

    # PS-QAM: receive-path symbols at unit average power live on the
    # ``{s_m/sqrt(E_PS)}`` grid.  Rescale by ``sqrt(E_PS)`` to bring them back
    # to the ``{s_m}`` grid that ``gray_constellation`` returns, so the
    # nearest-neighbour search is exact.  No-op for uniform modulations.
    symbols_flat = rescale_ps_symbols(symbols_flat, xp, modulation, order, pmf)

    constellation = xp.asarray(constellation)

    # 1. Find nearest constellation point (Hard Decision)
    k = int(np.log2(order))
    is_sq_qam = _is_square_qam(modulation, order)

    if is_sq_qam:
        # O(1) component-wise rounding - no (N, M) distance matrix.
        # Levels are evenly spaced; round each axis independently.
        n_ax = k // 2  # bits per axis
        side = 2**n_ax  # points per axis
        levels = xp.sort(xp.unique(constellation.real))  # (side,)
        lev_min = float(levels[0])
        d_grid = float(levels[1] - levels[0])
        # Gray LUT: sorted-level index (geometric) -> natural-binary symbol index
        gray_lut = xp.asarray(gray_code(n_ax))  # (side,)
        g_i = xp.clip(
            xp.round((symbols_flat.real - lev_min) / d_grid).astype(xp.int64),
            0,
            side - 1,
        )
        g_q = xp.clip(
            xp.round((symbols_flat.imag - lev_min) / d_grid).astype(xp.int64),
            0,
            side - 1,
        )
        indices = (gray_lut[g_i] << n_ax) | gray_lut[g_q]  # (N_flat,)
    else:
        # General path: chunk N to bound peak memory at (CHUNK_N, M_const).
        indices = nearest_constellation_index(symbols_flat, constellation)

    # 2. Convert indices to bits: (N, k)
    bits = unpack_bits(indices, k)

    # 3. Reshape to restore original structure
    # bits is currently (Total_Symbols, k)
    # We flatten it to (Total_Symbols * k)
    flat_bits = bits.flatten()

    # Calculate new shape: replace last dimension D with D*k
    # If original shape was (D1, D2, ..., Dn), new shape is (D1, D2, ..., Dn * k)
    if len(original_shape) > 0:
        new_shape = list(original_shape)
        new_shape[-1] = new_shape[-1] * k
        return flat_bits.reshape(new_shape)
    else:
        # Scalar input case (if supported), returns 1D array of bits
        return flat_bits
