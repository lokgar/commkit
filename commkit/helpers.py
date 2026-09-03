"""General library utility functions."""

from typing import TYPE_CHECKING, Any, overload

import numpy as np

from .backend import ArrayType, dispatch, get_array_module, is_cupy_available, to_device
from .logger import logger

if TYPE_CHECKING:
    from .core import Signal

try:
    import cupy as cp
except ImportError:
    cp = None


# ---------------------------------------------------------------------------
# Random generation
# ---------------------------------------------------------------------------


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
    logger.debug("Generating %s random bits (seed=%s).", length, seed)
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


# ---------------------------------------------------------------------------
# Power / normalization
# ---------------------------------------------------------------------------


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
    mode : {"unity_gain", "unit_energy", "peak", "average_power", "symbol_power", "dac_peak"}, default "unity_gain"
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
        - "dac_peak": Per-channel ``max(peak_|Re|, peak_|Im|)`` is 1.0.
          Brings the dominant I/Q component to 1.0 while preserving the I/Q
          ratio - unlike ``"peak"`` (complex-envelope ``max(|x[n]|)``), which
          leaves components at ``<= 1/sqrt(2)`` for square QAM/PSK whose
          envelope peak sits at 45°.  Maximises DAC range utilisation for a
          signal section independent of modulation format or constellation
          phase geometry (e.g. a frame's preamble and body, normalised
          separately so each uses the full DAC range).
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
    logger.debug("Normalizing array (mode: %s, axis=%s, sps=%s).", mode, axis, sps)
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

    elif mode == "dac_peak":
        # Per-channel max(peak_|Re|, peak_|Im|) = 1: brings the dominant I/Q
        # component to 1.0, preserving the I/Q ratio.
        norm_factor = xp.maximum(
            xp.max(xp.abs(x.real), axis=axis, keepdims=keepdims),
            xp.max(xp.abs(x.imag), axis=axis, keepdims=keepdims),
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


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Correlation & sequences
# ---------------------------------------------------------------------------


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

    samples, was_1d = as_2d(samples, name="samples")
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


# ---------------------------------------------------------------------------
# CPR / PLL loop gains
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Array shape helpers
# ---------------------------------------------------------------------------
#
# CommKit's SISO/MIMO convention is ``(N,)`` / ``(C, N)`` with time on the last
# axis (CLAUDE.md, "Array Shapes").  Nearly every DSP entry point therefore
# promotes a 1-D input to ``(1, N)``, runs one vectorized channel-batched
# implementation, and squeezes the leading axis back off on the way out.  These
# helpers are that idiom, defined once, so the promotion is validated the same
# way everywhere instead of silently passing 3-D (or 0-d) input through to a
# confusing downstream broadcast error.
#
# They are pure indexing/broadcast operations and therefore correct on every
# array type the library sees - NumPy, CuPy, and JAX - without dispatching.


def as_2d(arr: ArrayType, *, name: str = "array") -> tuple[ArrayType, bool]:
    """
    Promotes a SISO ``(N,)`` array to the MIMO layout ``(1, N)``.

    The canonical entry half of the library's SISO/MIMO shape idiom; pair it
    with :func:`restore_1d` to squeeze the promoted axis back off the outputs.

    Parameters
    ----------
    arr : array_like
        Input array, ``(N,)`` (SISO) or ``(C, N)`` (MIMO, time last).
        Must already be an array (call ``dispatch`` first); no conversion or
        host transfer is performed.
    name : str, default "array"
        Variable name used in the error message.

    Returns
    -------
    arr_2d : array_like
        ``arr[None, :]`` for 1-D input, ``arr`` itself (no copy) for 2-D.
    was_1d : bool
        Whether the promotion happened - pass this to :func:`restore_1d`.

    Raises
    ------
    ValueError
        If ``arr`` is 0-d or has more than two dimensions.  CommKit signals
        carry at most a channel axis and a time axis; a 3-D input is a caller
        error, not a batch dimension.
    """
    ndim = np.ndim(arr)  # reads ``arr.ndim``; never converts CuPy/JAX to host
    if ndim == 1:
        return arr[None, :], True
    if ndim == 2:
        return arr, False
    raise ValueError(
        f"{name} must be 1-D (N,) for SISO or 2-D (C, N) for MIMO with time on "
        f"the last axis; got ndim={ndim}."
    )


@overload
def restore_1d(was_1d: bool, arr: ArrayType, /) -> ArrayType: ...


@overload
def restore_1d(
    was_1d: bool, arr: ArrayType, arr2: ArrayType, /, *rest: ArrayType
) -> tuple[ArrayType, ...]: ...


def restore_1d(was_1d: bool, *arrays: ArrayType) -> ArrayType | tuple[ArrayType, ...]:
    """
    Undoes :func:`as_2d` on one or more outputs.

    Parameters
    ----------
    was_1d : bool
        The flag returned by :func:`as_2d`.
    *arrays : array_like
        Channel-batched results, each ``(1, ...)`` when ``was_1d`` is True.

    Returns
    -------
    array_like or tuple of array_like
        ``arr[0]`` per input when ``was_1d``, otherwise the inputs unchanged.
        A single input returns bare (not a 1-tuple), so both
        ``out = restore_1d(was_1d, out)`` and
        ``drift, pn = restore_1d(was_1d, drift, pn)`` read naturally.
    """
    if not arrays:
        raise ValueError("restore_1d() requires at least one array.")
    out = tuple(a[0] for a in arrays) if was_1d else arrays
    return out[0] if len(out) == 1 else out


def broadcast_channels(
    ref: ArrayType, num_channels: int, xp: Any = None, *, name: str = "reference"
) -> ArrayType:
    """
    Broadcasts a shared reference sequence across ``num_channels`` channels.

    Replaces the ad-hoc ``if ref.ndim == 1: ref = ref[None, :]`` promotion used
    at every reference/pilot input, adding the channel-count check that the
    bare promotion leaves to a later, far more opaque broadcast failure.

    Parameters
    ----------
    ref : array_like
        Reference sequence: ``(L,)`` shared by all channels, ``(1, L)``
        (broadcast), or ``(C, L)`` (per-channel).
    num_channels : int
        Number of channels ``C`` the reference must cover.
    xp : module, optional
        Array module to broadcast with.  Inferred from ``ref`` when omitted.
    name : str, default "reference"
        Variable name used in the error message.

    Returns
    -------
    array_like
        A ``(C, L)`` view.  Shared references are returned as a **read-only
        broadcast view** (no data is copied); call ``.copy()`` before writing.

    Raises
    ------
    ValueError
        If ``ref`` is not 1-D/2-D, or its channel count is neither
        ``num_channels`` nor 1.
    """
    if xp is None:
        xp = get_array_module(ref)
    ndim = np.ndim(ref)
    if ndim == 1:
        return xp.broadcast_to(ref[None, :], (num_channels, ref.shape[-1]))
    if ndim == 2:
        c = ref.shape[0]
        if c == num_channels:
            return ref
        if c == 1:
            return xp.broadcast_to(ref, (num_channels, ref.shape[-1]))
        raise ValueError(
            f"{name} has {c} channels, which matches neither the signal's "
            f"{num_channels} channels nor 1 (broadcast)."
        )
    raise ValueError(f"{name} must be 1-D (L,) or 2-D (C, L); got ndim={ndim}.")


def require_channels(
    arr: ArrayType,
    num_channels: int,
    *,
    name: str = "samples",
    description: str | None = None,
) -> ArrayType:
    """
    Validates a strict MIMO layout with an exact channel count.

    For entry points that are *only* defined for a fixed number of channels -
    dual-polarization channel models and polarization equalizers - where a
    SISO input is a caller error rather than something to promote.

    Parameters
    ----------
    arr : array_like
        Input array; must be ``(num_channels, N)``.
    num_channels : int
        Required channel count (e.g. ``2`` for dual-pol).
    name : str, default "samples"
        Variable name used in the error message.
    description : str, optional
        Domain wording for the requirement, e.g. ``"dual-pol input with shape
        (2, N)"``.  Defaults to a generic ``"a 2-D (C, N) array"``.

    Returns
    -------
    array_like
        ``arr`` unchanged.

    Raises
    ------
    ValueError
        If ``arr`` is not 2-D or does not have exactly ``num_channels`` rows.
    """
    ndim = np.ndim(arr)
    if ndim != 2 or arr.shape[0] != num_channels:
        shape = tuple(arr.shape) if hasattr(arr, "shape") else np.shape(arr)
        what = description or f"a 2-D ({num_channels}, N) array"
        raise ValueError(
            f"{name} must be {what} with time on the last axis; got shape {shape}."
        )
    return arr


def to_report_scalar(values: Any) -> float | np.ndarray:
    """
    Collapses a per-channel result to a Python float, or a host NumPy array.

    The reporting-layer counterpart of :func:`as_2d`: channel-batched metrics
    are computed as ``(C,)`` vectors, but a SISO caller wants a plain float
    back.  Device arrays are transferred to the host internally, so this is
    safe to call on a CuPy result directly.

    Parameters
    ----------
    values : array_like or scalar
        Per-channel metric, ``(C,)`` (or 0-d / scalar).

    Returns
    -------
    float or numpy.ndarray
        A Python float when a single value is present, otherwise a
        ``float64`` NumPy array.
    """
    arr = np.asarray(to_device(values, "cpu"), dtype=np.float64)
    return float(arr.reshape(-1)[0]) if arr.size == 1 else arr


# ---------------------------------------------------------------------------
# Three-point (log-)parabolic sub-bin/sub-sample peak fit
# ---------------------------------------------------------------------------


def _parabolic_peak_offset(
    y_prev: ArrayType,
    y_curr: ArrayType,
    y_next: ArrayType,
    xp: Any,
    *,
    log: bool = False,
    log_eps: float = 1e-300,
    denom_eps: float | None = None,
) -> ArrayType:
    r"""Three-point (log-)parabolic sub-bin/sub-sample peak-offset fit.

    Fits a parabola through three samples straddling a peak (bins/samples
    k-1, k, k+1) - or through their logs, for the standard log-parabolic
    (Gaussian-equivalent) fit - and returns the offset of the true peak
    relative to the center sample, clipped to ``[-0.5, 0.5]``:

        delta = 0.5 * (y_prev - y_next) / (y_prev - 2*y_curr + y_next)

    Shared by every three-point peak-interpolation site in the library:
    the FOE M-th-power estimator's magnitude-domain fit
    (``frequency.estimate_frequency_offset_mth_power``, ``log=False``), the
    two log-parabolic tone-refinement estimators
    (``frequency.find_bias_tone``, ``frequency._refine_tones_from_spectrum``,
    ``log=True``), and the fractional-delay estimator
    (``timing.estimate_fractional_delay``, either fit) - which first
    phase-rotates a complex peak onto the real axis (a preprocessing step
    outside this function's scope) before calling this with its own
    ``log_eps``/``denom_eps`` tuning.  Inputs may be plain Python floats
    with ``xp=numpy`` (host scalars) or device arrays (NumPy/CuPy) - the
    arithmetic is expressed purely through ``xp``, so both execution models
    are supported by the same implementation.

    Parameters
    ----------
    y_prev, y_curr, y_next : array_like or float
        Samples at bins/positions k-1, k, k+1.
    xp : module
        Array module (``numpy``/``cupy``) providing ``log``, ``maximum``,
        ``abs``, ``where``, ``ones_like``, ``zeros_like``, ``clip``.
    log : bool, default False
        If True, fits to ``log(max(y, log_eps))`` (log-parabolic / Gaussian
        fit - standard for spectral-magnitude tone estimation). If False,
        fits directly to ``y`` (magnitude-domain fit).
    log_eps : float, default 1e-300
        Clamp floor before taking the log. Only used when ``log=True``.
    denom_eps : float, optional
        Threshold below which the fit denominator is treated as degenerate
        (returns ``delta=0`` instead of dividing). Defaults to ``1e-30`` when
        ``log=True``, ``1e-15`` when ``log=False`` - the values already in
        use at every call site except ``timing.estimate_fractional_delay``,
        which passes its own tuning explicitly.

    Returns
    -------
    delta : same type as inputs
        Sub-bin/sub-sample offset in ``[-0.5, 0.5]``.
    """
    if denom_eps is None:
        denom_eps = 1e-30 if log else 1e-15
    if log:
        y_prev = xp.log(xp.maximum(y_prev, log_eps))
        y_curr = xp.log(xp.maximum(y_curr, log_eps))
        y_next = xp.log(xp.maximum(y_next, log_eps))

    denom = y_prev - 2.0 * y_curr + y_next
    valid = xp.abs(denom) > denom_eps
    safe_denom = xp.where(valid, denom, xp.ones_like(denom))
    raw = 0.5 * (y_prev - y_next) / safe_denom
    delta = xp.where(valid, raw, xp.zeros_like(raw))
    return xp.clip(delta, -0.5, 0.5)


# ---------------------------------------------------------------------------
# Linear-trend (least-squares slope) helpers
# ---------------------------------------------------------------------------


def _centered_axis(n: int, x: Any, xp: Any) -> tuple[ArrayType, ArrayType]:
    """Mean-removed abscissa and its sum of squares, as device arrays."""
    if x is None:
        # Analytic form for x = arange(n): Σ(i - (n-1)/2)² = n(n²-1)/12, which
        # avoids a reduction and is exact in float64 for realistic record
        # lengths (n < 2⁵², well inside the mantissa).
        xc = xp.arange(n, dtype=xp.float64) - 0.5 * (n - 1)
        denom = xp.asarray(n * (n * n - 1.0) / 12.0, dtype=xp.float64)
    else:
        x_arr = xp.asarray(x, dtype=xp.float64)
        xc = x_arr - xp.mean(x_arr)
        denom = xp.sum(xc * xc)
    # Guard n == 1 (or a degenerate axis) without a host sync on the value.
    return xc, xp.where(denom > 0.0, denom, xp.ones_like(denom))


def linear_trend_slope(y: ArrayType, *, x: Any = None, xp: Any = None) -> ArrayType:
    r"""
    Per-channel least-squares slope of a phase (or any) record.

    Ordinary least squares on the centred normal equations,

    .. math:: \hat{a} = \frac{\sum_k (x_k - \bar{x})(y_k - \bar{y})}
                             {\sum_k (x_k - \bar{x})^2},

    evaluated for every channel in one vectorized pass and returned **on the
    input backend** - no host synchronization, so the caller decides when (and
    whether) to transfer.

    Parameters
    ----------
    y : array_like
        Record to fit, ``(C, N)`` with the fit axis last (promote SISO with
        :func:`as_2d` first).
    x : array_like, optional
        Abscissa, ``(N,)``.  Defaults to the sample index ``arange(N)``, so the
        slope is then in *units of y per sample*.  Pass a time axis in seconds
        to get a slope per second (e.g. for non-uniform pilot positions).
    xp : module, optional
        Array module; inferred from ``y`` when omitted.

    Returns
    -------
    array_like
        Slope per channel, ``(C,)``, ``float64``, on the input backend.
    """
    if xp is None:
        y, xp, _ = dispatch(y)
    n = y.shape[-1]
    xc, denom = _centered_axis(n, x, xp)
    # Subtracting the per-channel mean is mathematically redundant (xc is
    # already centred) but keeps the products small when the record carries a
    # large constant offset - phase trajectories routinely do.
    yc = y - xp.mean(y, axis=-1, keepdims=True)
    return xp.sum(yc * xc, axis=-1) / denom


def remove_linear_trend(y: ArrayType, *, x: Any = None) -> tuple[ArrayType, ArrayType]:
    r"""
    Removes the per-channel least-squares linear trend, preserving the mean.

    On an unwrapped phase record the linear term *is* the mean frequency
    offset, so this is the canonical "strip the residual FOE, keep the phase
    fluctuation" step shared by the pilot-tone recovery and the DSH
    fine-frequency stage.  Only the slope term is subtracted, so the mean phase
    (and hence any constant offset) survives.

    Parameters
    ----------
    y : array_like
        Record to detrend, ``(C, N)`` with time on the last axis.
    x : array_like, optional
        Abscissa, ``(N,)``; defaults to the sample index (slope per sample).

    Returns
    -------
    detrended : array_like
        ``y`` minus the fitted slope term, ``float64``, on the input backend.
    slope : array_like
        Fitted slope per channel, ``(C,)`` - in units of ``y`` per unit ``x``
        (per sample by default).
    """
    y, xp, _ = dispatch(y)
    n = y.shape[-1]
    xc, denom = _centered_axis(n, x, xp)
    yc = y - xp.mean(y, axis=-1, keepdims=True)
    slope = xp.sum(yc * xc, axis=-1) / denom
    return y - slope[..., None] * xc[None, :], slope


# ---------------------------------------------------------------------------
# Signal unwrap/rewrap helpers
# ---------------------------------------------------------------------------


def unwrap_signal(
    x: "ArrayType | Signal", *, field: str = "samples"
) -> tuple[ArrayType, "Signal | None"]:
    """
    Extracts the working array from a :class:`~commkit.core.Signal`, or
    passes a raw array through unchanged.

    The canonical entry half of the library's Signal-awareness idiom; pair it
    with :func:`rewrap_signal` so a function transparently returns an array
    for array input and a :class:`Signal` for :class:`Signal` input.

    Parameters
    ----------
    x : array_like or Signal
        Input samples, or a :class:`Signal` wrapping them.
    field : str, default "samples"
        Attribute to read off ``x`` when it is a :class:`Signal`. Most DSP
        functions operate on ``.samples``; a few (e.g. hard-decision mapping)
        instead read ``.resolved_symbols``.

    Returns
    -------
    array : array_like
        ``getattr(x, field)`` for :class:`Signal` input, or ``x`` itself for
        array input. Not yet passed through :func:`~commkit.backend.dispatch`.
    signal : Signal or None
        The originating :class:`Signal`, or ``None`` for array input - pass
        this straight to :func:`rewrap_signal`.
    """
    # Local import: commkit.core.signal imports this module, so importing
    # Signal at module scope here would be circular (see io.py for the same
    # pattern).
    from .core import Signal

    if isinstance(x, Signal):
        return getattr(x, field), x
    return x, None


@overload
def rewrap_signal(sig: None, array: ArrayType, /, **metadata: Any) -> ArrayType: ...


@overload
def rewrap_signal(sig: "Signal", array: ArrayType, /, **metadata: Any) -> "Signal": ...


def rewrap_signal(
    sig: "Signal | None", array: ArrayType, /, **metadata: Any
) -> "ArrayType | Signal":
    """
    Rebuilds a :class:`~commkit.core.Signal` around a result array, or
    passes the array through unchanged.

    The inverse of :func:`unwrap_signal`. ``sig`` is normally the value
    :func:`unwrap_signal` returned alongside the array being processed.

    Parameters
    ----------
    sig : Signal or None
        The originating :class:`Signal` from :func:`unwrap_signal`, or
        ``None`` to pass ``array`` through unchanged (the array-input case).
    array : array_like
        The result to store on the copy's ``.samples``.
    **metadata
        Additional fields to set on the copy via ``setattr`` (e.g.
        ``sampling_rate=sig.symbol_rate`` after decimating to symbol rate).

    Returns
    -------
    array_like or Signal
        ``array`` unchanged when ``sig`` is ``None``; otherwise a
        :meth:`Signal.copy` with ``.samples = array`` and ``**metadata``
        applied.
    """
    if sig is None:
        return array
    new = sig.copy()
    new.samples = array
    for key, value in metadata.items():
        setattr(new, key, value)
    return new


def _cd_beta2_length(
    dispersion_ps_nm_km: float, fiber_length_km: float, center_wavelength_nm: float
) -> float:
    """Chromatic-dispersion SI-unit conversion: the beta_2 * L product.

    Shared physics conversion between ``filtering.compensate_chromatic_dispersion``
    (the inverse EDC transfer function) and
    ``impairments.channel.linear.apply_chromatic_dispersion`` (the forward
    impairment) - both build their own signed transfer function ``H`` from
    this value; only the sign of the frequency-domain exponent differs
    between them.

        beta_2 = -D * lambda^2 / (2*pi*c)

    Parameters
    ----------
    dispersion_ps_nm_km : float
        Fiber dispersion parameter D in ps / (nm * km).
    fiber_length_km : float
        Fiber span length in km.
    center_wavelength_nm : float
        Center wavelength in nm (e.g. 1550 for C-band).

    Returns
    -------
    float
        The ``beta_2 * L`` product in s² (SI units).
    """
    D = dispersion_ps_nm_km * 1e-12 / (1e-9 * 1e3)  # s / m²
    lam = center_wavelength_nm * 1e-9  # m
    c = 2.998e8  # m/s
    L = fiber_length_km * 1e3  # m
    return -(D * lam**2) / (2.0 * np.pi * c) * L  # s²  (β₂·L product)
