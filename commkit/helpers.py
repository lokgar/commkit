"""General library utility functions."""

from typing import Any

import numpy as np

from .backend import ArrayType, dispatch, get_array_module, is_cupy_available, to_device
from .logger import logger

try:
    import cupy as cp
except ImportError:
    cp = None


def generate_bits(length: int, seed: int | None = None) -> ArrayType:
    """
    Generates a sequence of random binary bits (0s and 1s).

    Uses `numpy.random.default_rng()` for consistent seed behavior across
    different platforms and backends.

    Parameters
    ----------
    length : int
        Total number of bits to generate.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    array_like
        Array of bits (0 or 1). Shape: (length,).
        Data type is `int8`.
    """
    logger.debug(f"Generating {length} random bits (seed={seed}).")
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=length, dtype="int8")

    if is_cupy_available():
        bits = to_device(bits, "gpu")

    return bits


def generate_symbols(
    num_symbols: int,
    modulation: str,
    order: int,
    seed: int | None = None,
    unipolar: bool = False,
) -> ArrayType:
    """
    Generates a sequence of random modulation symbols.

    This is a high-level utility that combines bit generation and mapping
    to produce synthetic symbol sequences.

    Parameters
    ----------
    num_symbols : int
        Number of symbols to generate.
    modulation : {"psk", "qam", "ask"}
        The modulation scheme identifier.
    order : int
        Modulation order (e.g., 4, 16, 64).
    seed : int, optional
        Random seed for reproducible results.
    unipolar : bool, default False
        If True, use unipolar constellation (ASK/PAM).

    Returns
    -------
    array_like
        Array of symbols on the active device (CPU or GPU).
        Dtype is ``complex64`` for PSK/QAM, ``float32`` for ASK/PAM.
    """
    from . import mapping

    k = int(np.log2(order))
    bits = generate_bits(num_symbols * k, seed=seed)
    return mapping.map_bits(bits, modulation, order, unipolar=unipolar)


def rms(x: ArrayType, axis: int | None = None, keepdims: bool = False) -> ArrayType:
    """
    Computes the Root-Mean-Square (RMS) value of an array.

    RMS is defined as: sqrt(E[|x|^2]).

    Parameters
    ----------
    x : array_like
        Input array.
    axis : int, optional
        Axis along which to compute the RMS. If None, computes global RMS.
    keepdims : bool, default False
        If True, the reduced axes are left in the result as dimensions with size one.

    Returns
    -------
    array_like or float
        The RMS value of the input.
    """
    x, xp, _ = dispatch(x)
    # RMS = ||x||₂ / √N  ->  linalg.norm routes through BLAS (DZNRM2/SNRM2),
    # eliminating the abs(x)**2 and mean() intermediate allocations.
    n = x.size if axis is None else x.shape[axis]
    # xp.sqrt(Python int) returns float64; cast n to x's real dtype so that
    # float32 norms are not silently promoted to float64.
    return xp.linalg.norm(x, axis=axis, keepdims=keepdims) / xp.sqrt(
        xp.asarray(n, dtype=x.real.dtype)
    )


def normalize(
    x: ArrayType, mode: str = "unity_gain", axis: int | None = None, sps: int = 1
) -> ArrayType:
    """
    Normalizes an array according to the specified strategy.

    Parameters
    ----------
    x : array_like
        Input signal or filter taps.
    mode : {"unity_gain", "unit_energy", "peak", "average_power", "symbol_power"}, default "unity_gain"
        Normalization strategy:
        - "unity_gain": Sum of elements is 1.0 (DC gain normalization).
          Preserves signal levels (e.g., 5V -> 5V). Used for general filters.
        - "unit_energy": L2-norm is 1.0 (sum(|x|^2) = 1).
          Preserves total energy/noise power. Used for pulse shaping and matched filters.
        - "peak": Peak complex envelope is 1.0 (max_n |x[n]| = 1).
          For complex signals this normalizes by the maximum instantaneous magnitude,
          so |x[n]| <= 1 for all n. This bound is invariant under any
          unit-magnitude operation (frequency shifts, phase rotations, equalization),
          making it the correct choice for DSP chains. For real signals the behavior
          is identical: max_n |x[n]| = 1.
        - "average_power": Mean sample power is 1.0 (E[|x|^2] = 1 per sample).
          Normalizes the composite complex signal power at the sample level.
          Used for symbol constellations at 1 sps and for display/plotting.
          **Not suitable for oversampled waveforms**: for a Nyquist pulse with
          unit-energy taps at ``sps`` samples/symbol the natural average sample
          power is ``Es/sps``, so ``"average_power"`` would inflate all samples
          by ``√sps`` and break Es/N0 calibration.
        - "symbol_power": Unit symbol energy regardless of oversampling factor.
          Norm factor is ``rms(x) * √sps``, so the output satisfies
          ``E[|x|²] * sps = 1`` (i.e. average sample power = 1/sps).
          This is the correct mode for pulse-shaped waveforms: all pulse types
          (zero-stuffed, rect, RRC, Gaussian, ...) end up at the same power level
          and ``apply_awgn`` can use ``Es = signal_power * sps = 1`` directly.
          Requires ``sps`` parameter. At ``sps=1`` it is identical to
          ``"average_power"``.
    axis : int, optional
        The axis along which to compute the normalization factor.
        If `None`, normalizes the entire array globally.
    sps : int, default 1
        Samples per symbol. Only used by the ``"symbol_power"`` mode.

    Returns
    -------
    array_like
        The normalized array.
    """
    logger.debug(f"Normalizing array (mode: {mode}, axis={axis}, sps={sps}).")
    x, xp, _ = dispatch(x)

    # keepdims for proper broadcasting when axis is specified
    keepdims = axis is not None

    if mode == "unity_gain":
        # DC gain = 1: H(0) = sum(h) = 1
        # Use case: filter taps where you want unity passband gain
        norm_factor = xp.sum(x, axis=axis, keepdims=keepdims)

    elif mode == "unit_energy":
        # L2 norm = 1: ||x||₂ = 1
        # Use case: matched filter taps (preserves SNR after correlation)
        # linalg.norm routes through BLAS (DNRM2/DZNRM2 on CPU, cuBLAS on GPU):
        # numerically superior (compensated summation) and avoids intermediate allocations.
        norm_factor = xp.linalg.norm(x, axis=axis, keepdims=keepdims)

    elif mode == "peak":
        # Complex envelope peak: max(|x[n]|) = 1.
        # For complex signals this is the instantaneous magnitude, not the
        # per-component max. The bound is invariant under frequency shifts and
        # phase rotations, unlike per-component (I/Q) normalization which can
        # allow |x[n]| up to sqrt(2) and therefore violate bounds after rotation.
        norm_factor = xp.max(xp.abs(x), axis=axis, keepdims=keepdims)

    elif mode == "average_power":
        # RMS = 1: sqrt(mean(|x|²)) = 1, so mean(|x|²) = 1
        # Use case: 1-sps symbol sequences and constellation normalization.
        norm_factor = rms(x, axis=axis, keepdims=keepdims)

    elif mode == "symbol_power":
        # Symbol-power norm: rms(x) * √sps = 1  ->  mean(|x|²) * sps = 1
        # Equivalent to average_power at 1 sps; at higher sps it accounts for
        # the 1/sps dilution produced by Nyquist pulse shaping with unit-energy
        # taps, leaving Es = 1 per symbol for all pulse shapes.
        # This is the same correction used in the equalizer's _normalize_inputs:
        #   sym_rms = global_rms * √sps
        norm_factor = rms(x, axis=axis, keepdims=keepdims) * xp.asarray(
            sps**0.5, dtype=x.real.dtype
        )

    else:
        raise ValueError(f"Unknown normalization mode: {mode}")

    # Handle division by zero safely for both NumPy and CuPy.
    # Avoid control flow based on data values to prevent host-device synchronization.
    # Use ones_like instead of the literal 1.0 (float64) to preserve float32 dtype.
    safe_norm = xp.where(norm_factor == 0, xp.ones_like(norm_factor), norm_factor)
    result = x / safe_norm

    # If norm_factor is 0, the input was all zeros -> output should also be zeros
    return xp.where(norm_factor == 0, xp.zeros(x.shape, dtype=x.dtype), result)


def format_si(value: float | None, unit: str = "Hz") -> str:
    """
    Formats a numeric value into a human-readable string with SI prefixes.

    Automatically selects the appropriate SI prefix (e.g., k, M, G, m, u, n)
    based on the magnitude of the value. Supports a wide range from
    femto (10^-15) to Peta (10^15).

    Parameters
    ----------
    value : float or None
        The numeric value to format. If `None`, returns "None".
    unit : str, default "Hz"
        The unit suffix to append (e.g., 'Hz', 'Baud', 's', 'W').

    Returns
    -------
    str
        The formatted string (e.g., '10.00 MHz', '50.00 ns').
    """
    if value is None:
        return "None"

    if abs(value) == 0:
        return f"0.00 {unit}"

    # Standard SI prefixes
    si_units = {
        -5: "f",
        -4: "p",
        -3: "n",
        -2: "µ",
        -1: "m",
        0: "",
        1: "k",
        2: "M",
        3: "G",
        4: "T",
        5: "P",
    }

    rank = int(np.floor(np.log10(abs(value)) / 3))
    # clamp to supported range
    rank = max(min(si_units.keys()), min(rank, max(si_units.keys())))

    scaled = value / (1000.0**rank)
    return f"{scaled:.2f} {si_units[rank]}{unit}"


def validate_array(
    v: Any, name: str = "array", complex_only: bool = False
) -> ArrayType:
    """
    Validates and coerces input data into a numeric array.

    Existing NumPy or CuPy arrays are passed through unchanged (preserving
    device placement). All other inputs (Python scalars, lists, tuples) are
    coerced to NumPy via ``np.asarray``; there is no automatic promotion to
    CuPy for non-array inputs. Optionally enforces complex-valued dtype.

    Parameters
    ----------
    v : array_like or any
        Input data to validate.
    name : str, default "array"
        Variable name used in error messages.
    complex_only : bool, default False
        If True, ensures the resulting array is complex-valued.

    Returns
    -------
    array_like
        NumPy or CuPy array (CuPy only when ``v`` was already a CuPy array).

    Raises
    ------
    ValueError
        If the input cannot be converted to a supported array type.
    """
    if v is None:
        return None

    # Coerce lists/tuples or other array-likes to numpy arrays initially
    if not isinstance(v, (np.ndarray, getattr(cp, "ndarray", type(None)))):
        try:
            v = np.asarray(v)
        except Exception as err:
            raise ValueError(
                f"Could not convert {name} of type {type(v)} to array."
            ) from err

    # Ensure it's a numeric array (not object, string, etc.)
    if v.dtype.kind not in "biufc":
        raise ValueError(
            f"Expected numeric array for {name}, got dtype {v.dtype} (kind {v.dtype.kind})"
        )

    if complex_only and not np.iscomplexobj(v):
        xp = get_array_module(v)
        # Preserve single-precision: float32 -> complex64, everything else -> complex128
        complex_dtype = xp.complex64 if v.dtype == xp.float32 else xp.complex128
        v = v.astype(complex_dtype)

    return v


def cross_correlate_fft(
    samples: ArrayType,
    template: ArrayType,
    mode: str = "full",
) -> ArrayType:
    """
    Vectorized FFT-based cross-correlation.

    Computes the cross-correlation of ``samples`` with ``template`` using
    the frequency-domain multiplication approach. Handles 1D and 2D
    (multichannel) inputs natively via ``axis=-1`` broadcasting - no
    Python loops over channels.

    Parameters
    ----------
    samples : array_like
        Input samples. Shape: ``(N,)`` or ``(C, N)``.
    template : array_like
        Reference sequence. Shape: ``(L,)`` or ``(C, L)``.
        If ``(1, L)`` and samples is ``(C, N)``, the template is
        broadcast across all channels.
    mode : {"full", "same", "valid", "positive_lags"}, default "full"
        Output size:
        - ``"full"``: length ``N + L - 1``.
        - ``"same"``: length ``N`` (centered).
        - ``"valid"``: length ``max(N, L) - min(N, L) + 1``.
        - ``"positive_lags"``: length ``N`` (lags 0 ... N-1 only). Returns a
          zero-copy view of the raw circular-correlation output - no
          ``concatenate`` and no reordering. Use this when negative lags are
          not needed (e.g. frame timing search within a bounded window).

    Returns
    -------
    array_like
        Complex cross-correlation with shape matching the input
        dimensionality and the selected ``mode``.
    """
    samples, xp, _ = dispatch(samples)
    template = xp.asarray(template)

    was_1d = samples.ndim == 1
    if was_1d:
        samples = samples[None, :]
    if template.ndim == 1:
        template = template[None, :]

    N = samples.shape[-1]
    L = template.shape[-1]
    full_len = N + L - 1

    # Smallest power-of-2 >= full_len for FFT efficiency.
    # `(full_len - 1).bit_length()` is the canonical integer-only formula;
    # `full_len.bit_length()` would round up even when full_len is already a power of 2.
    n_fft = 1 << (full_len - 1).bit_length()

    # FFT-based correlation: R[k] = IFFT(FFT(samples) * conj(FFT(template)))
    # Circular correlation places positive lags at 0..N-1 and negative lags
    # wrap to n_fft-(L-1)..n_fft-1.  Rearrange to match scipy layout:
    # lags [-(L-1), ..., -1, 0, 1, ..., N-1]  (total = N + L - 1).
    SIG = xp.fft.fft(samples, n_fft, axis=-1)
    TPL = xp.fft.fft(template, n_fft, axis=-1)
    corr_circ = xp.fft.ifft(SIG * xp.conj(TPL), axis=-1)

    # Gather negative lags (indices n_fft-(L-1) .. n_fft-1) then positive (0 .. N-1)
    neg_lags = corr_circ[..., n_fft - L + 1 :]  # length L-1
    pos_lags = corr_circ[..., :N]  # length N
    corr = xp.concatenate([neg_lags, pos_lags], axis=-1)  # length N+L-1

    # Apply mode trimming
    if mode == "positive_lags":
        corr = corr_circ[..., :N]  # zero-copy view; lags 0 ... N-1
    elif mode == "same":
        start = (L - 1) // 2
        corr = corr[..., start : start + N]
    elif mode == "valid":
        valid_len = max(N, L) - min(N, L) + 1
        start = min(N, L) - 1
        corr = corr[..., start : start + valid_len]
    # mode == "full": no trimming needed

    if was_1d:
        return corr[0]
    return corr


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


def cpr_pll_gains(bandwidth: float):
    """Convert normalised loop bandwidth to PI gains (mu, beta).

    Uses the standard 2nd-order loop approximation for a critically-damped
    (ζ = 1) PI loop:  μ ≈ 4·B_L,  β ≈ 4·B_L².  (With ``ωₙT = √β = 2B`` and
    ``ζ = μ/(2√β) = 1``.)

    Parameters
    ----------
    bandwidth : float
        Normalised one-sided loop bandwidth as a fraction of the symbol rate,
        e.g. ``1e-3`` for a narrow loop.

    Returns
    -------
    mu, beta : float32
    """
    mu = np.float32(4.0 * bandwidth)
    beta = np.float32(4.0 * bandwidth**2)
    return mu, beta


def resolve_pll_gains(bandwidth: float, mu: float | None, beta: float | None):
    """Resolve decision-directed PLL PI gains from a raw/bandwidth parameterization.

    Shared by the inline equalizer PLL (``lms``/``rls`` with ``cpr_type='pll'``)
    and the standalone ``recover_carrier_phase_pll``, so
    the bandwidth->gain mapping is defined in exactly one place.

    Precedence
    ----------
    * ``mu`` given -> raw PI gains; ``beta`` defaults to ``0.0`` (1st-order loop).
    * ``mu`` is ``None`` -> derive critically-damped (ζ=1) gains ``μ=4B, β=4B²``
      from ``bandwidth`` via ``cpr_pll_gains``.

    ``beta`` without ``mu`` is ambiguous and raises ``ValueError``.

    Returns
    -------
    mu, beta : float
    """
    if mu is not None:
        return float(mu), float(beta if beta is not None else 0.0)
    if beta is not None:  # beta without mu is ambiguous
        raise ValueError("beta requires mu to be set (or use the bandwidth shortcut).")
    return cpr_pll_gains(bandwidth)
