"""
Digital filtering and pulse shaping.

This module provides routines for design and application of digital filters
commonly used in communication systems. It supports both standard FIR filters
and specialized pulse-shaping filters, with high-performance execution on
both CPU and GPU backends.
"""

import numpy as np
import scipy

from .backend import ArrayType, dispatch, to_device
from .core.signal import Signal
from .helpers import as_2d, normalize, restore_1d, rewrap_signal, unwrap_signal
from .logger import logger

# -----------------------------------------------------------------------------
# FILTER DESIGN - TAP GENERATORS (array-only)
# -----------------------------------------------------------------------------
# rect_taps:     Hard or trapezoidal rectangular pulse taps
# gaussian_taps: Gaussian filter taps
# smoothrect_taps: Gaussian-smoothed rectangular pulse taps
# rrc_taps:      Root Raised Cosine filter taps
# rc_taps:       Raised Cosine filter taps
# lowpass_taps, highpass_taps, bandpass_taps, bandstop_taps: FIR filters
#
# All build taps from parameters alone - no signal input, so none are
# Signal-aware (same category as gray_code/barker_sequence).


def rect_taps(sps: int, duty_cycle: float = 1.0, rise_time: float = 0.0) -> np.ndarray:
    """
    Generates rectangular or trapezoidal pulse-shaping filter taps.

    With ``rise_time=0`` (default) the output is a hard rectangular pulse of
    width ``duty_cycle`` symbol periods.  With ``rise_time > 0`` the leading
    and trailing edges are replaced by linear ramps, producing an isosceles
    trapezoidal pulse that models a slew-rate-limited driver or modulator.

    Parameters
    ----------
    sps : int
        Samples per symbol.
    duty_cycle : float, default 1.0
        Total pulse width in symbol periods, including both ramps.
        Must be in the range ``(0, 1]``.
    rise_time : float, default 0.0
        Duration of each linear ramp (10%->90% of amplitude is the full ramp
        here; the ramp spans the full ``rise_time``) in symbol periods.
        Must satisfy ``rise_time <= duty_cycle / 2``; otherwise the ramps
        overlap and no flat top exists.

    Returns
    -------
    ndarray
        Pulse taps (unnormalized). Shape: ``(N_taps,)``.

    Raises
    ------
    ValueError
        If ``rise_time > duty_cycle / 2``.
    """
    if rise_time > duty_cycle / 2:
        raise ValueError(
            f"rise_time ({rise_time}) must be <= duty_cycle / 2 ({duty_cycle / 2:.3f}); "
            "ramps would overlap with no flat top."
        )

    n_total = int(round(sps * duty_cycle))
    if n_total < 1:
        n_total = 1

    if rise_time == 0.0:
        h = np.ones(n_total)
    else:
        n_ramp = int(round(sps * rise_time))
        n_flat = n_total - 2 * n_ramp
        if n_flat < 0:
            n_flat = 0
        ramp_up = np.linspace(0.0, 1.0, n_ramp, endpoint=False)
        ramp_dn = np.linspace(1.0, 0.0, n_ramp, endpoint=False)
        flat = np.ones(n_flat)
        h = np.concatenate([ramp_up, flat, ramp_dn])

    logger.debug(
        "Generating Rect taps: sps=%s, duty_cycle=%s, rise_time=%s, n_taps=%s",
        sps,
        duty_cycle,
        rise_time,
        len(h),
    )
    return h


def gaussian_taps(sps: float, span: int = 4, duty_cycle: float = 1.0) -> np.ndarray:
    """
    Generates Gaussian pulse-shaping filter taps.

    The Gaussian filter is typically used in GMSK/GFSK modulation to minimize
    occupied bandwidth while introducing controlled Inter-Symbol Interference (ISI).

    Parameters
    ----------
    sps : float
        Samples per symbol.
    span : int, default 4
        Total filter span in symbols. The number of taps will be ``span * sps + 1``
        to ensure symmetry.
    duty_cycle : float, default 1.0
        Full-Width at Half-Maximum (FWHM) of the Gaussian pulse in symbol periods.
        Smaller values produce a narrower pulse (lower ISI but wider bandwidth).
        The Bandwidth-Time product is derived internally as
        ``bt = √2·ln(2) / (π·duty_cycle)``.

    Returns
    -------
    ndarray
        Gaussian filter taps normalized to unit energy.
        Shape: (N_taps,).
    """
    # Convert duty_cycle (FWHM in symbol periods) to BT product.
    # FWHM of h(t) = exp(-(π·t/α)²) is α·√(ln2)/π = √(ln2/2)/B·√(ln2)/π = ln2/(π·B).
    # Wait - using the standard relation:
    #   FWHM = √(2·ln2) · σ_freq,  where B = 1/(2π·σ_freq)  ->  BT = √(ln2/2)/π
    # More directly: FWHM_time = √(ln2/2) / (π·B) which gives BT = √(ln2/2)/π·(1/FWHM)
    # Rearranged: bt = √2·ln(2) / (π·duty_cycle)
    bt = np.sqrt(2) * np.log(2) / (np.pi * duty_cycle)
    logger.debug(
        "Generating Gaussian taps: sps=%s, span=%s, duty_cycle=%s (bt=%.4f)",
        sps,
        span,
        duty_cycle,
        bt,
    )
    # Ensure odd number of taps to have a center peak
    num_taps = int(span * sps)
    if num_taps % 2 == 0:
        num_taps += 1

    t = np.linspace(-span / 2, span / 2, num_taps)

    # Gaussian function
    # h(t) = (sqrt(pi)/alpha) * exp(-(pi*t/alpha)^2)
    # where alpha = sqrt(ln(2)/2)/B
    alpha = np.sqrt(np.log(2) / 2) / bt
    h = (np.sqrt(np.pi) / alpha) * np.exp(-((np.pi * t / alpha) ** 2))

    return normalize(h, "unit_energy")


def smoothrect_taps(
    sps: int, span: int, rise_time: float = 0.22, duty_cycle: float = 1.0
) -> ArrayType:
    """
    Generates a perfectly centered Gaussian-smoothed rectangular pulse.

    This method uses the analytical closed-form solution (Error Function)
    to avoid the 0.5 sample shift artifact typically caused by convolving
    odd/even discrete arrays.

    Parameters
    ----------
    sps : int
        Samples per symbol.
    span : int
        Filter span in symbols. The number of taps will be approximately ``span * sps``.
    rise_time : float, default 0.22
        10%-90% edge transition duration in symbol periods. Smaller values produce
        sharper edges (closer to a hard rect); larger values yield softer transitions
        (approaching a Gaussian pulse). Converted internally to the Gaussian sigma via
        ``σ = rise_time / (2·√2·erfinv(0.8))``.
    duty_cycle : float, default 1.0
        Width of the underlying rectangular pulse in symbol periods. Use 1.0 for NRZ
        and 0.5 for RZ signaling.

    Returns
    -------
    ndarray
        Gaussian-smoothed rectangular pulse taps normalized to unit energy.
        Shape: (N_taps,).
    """
    logger.debug(
        "Generating SmoothRect taps: sps=%s, span=%s, rise_time=%s, duty_cycle=%s",
        sps,
        span,
        rise_time,
        duty_cycle,
    )
    # Ensure odd number of taps to have a center peak
    num_taps = int(span * sps)
    if num_taps % 2 == 0:
        num_taps += 1

    t = np.linspace(-span / 2, span / 2, num_taps)

    # Convert rise_time to Gaussian sigma.
    # rise_time (10%-90%) = 2·√2·erfinv(0.8)·σ  ->  σ = rise_time / (2·√2·erfinv(0.8))
    sigma = rise_time / (2 * np.sqrt(2) * float(scipy.special.erfinv(0.8)))

    # Analytical Formula (Convolved Rect and Gaussian)
    # The underlying rect spans [-duty_cycle/2, +duty_cycle/2].
    # Convolution of rect with Gaussian = difference of error functions.
    w_half = duty_cycle / 2.0
    h = 0.5 * (
        scipy.special.erf((t + w_half) / (sigma * np.sqrt(2)))
        - scipy.special.erf((t - w_half) / (sigma * np.sqrt(2)))
    )

    return normalize(h, "unit_energy")


def rrc_taps(sps: float, rolloff: float = 0.35, span: int = 8) -> np.ndarray:
    """
    Generates Root Raised Cosine (RRC) filter taps.

    RRC filters are used at both the transmitter (pulse shaping) and
    receiver (matched filtering) to satisfy the Nyquist ISI criterion.

    Parameters
    ----------
    sps : float
        Samples per symbol.
    rolloff : float, default 0.35
        Roll-off factor (alpha), range [0, 1].
    span : int, default 8
        Filter span in symbols.

    Returns
    -------
    ndarray
        RRC filter taps normalized to unit energy.
        Shape: (N_taps,).
    """
    logger.debug("Generating RRC taps: sps=%s, rolloff=%s, span=%s", sps, rolloff, span)
    # Ensure odd number of taps
    num_taps = int(span * sps)
    if num_taps % 2 == 0:
        num_taps += 1

    t = np.linspace(-span / 2, span / 2, num_taps)

    # Avoid division by zero
    # 1. t = 0
    # 2. t = +/- 1/(4*rolloff)

    # Initialize array
    h = np.zeros_like(t)

    # Case 1: t = 0
    idx_0 = np.isclose(t, 0)
    h = np.where(idx_0, 1.0 - rolloff + (4 * rolloff / np.pi), h)

    # Case 2: t = +/- 1/(4*rolloff)
    if rolloff > 0:
        idx_singularity = np.isclose(np.abs(t), 1 / (4 * rolloff))
        h = np.where(
            idx_singularity,
            (rolloff / np.sqrt(2))
            * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * rolloff))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * rolloff))
            ),
            h,
        )
    else:
        idx_singularity = np.zeros_like(t, dtype=bool)

    # Case 3: General case
    idx_general = ~(idx_0 | idx_singularity)

    numer = np.sin(np.pi * t * (1 - rolloff)) + 4 * rolloff * t * np.cos(
        np.pi * t * (1 + rolloff)
    )
    denom = np.pi * t * (1 - (4 * rolloff * t) ** 2)

    # Avoid invalid value warning by making den safe
    denom_safe = np.where(idx_general, denom, 1.0)
    h = np.where(idx_general, numer / denom_safe, h)

    return normalize(h, "unit_energy")


def rc_taps(sps: float, rolloff: float = 0.35, span: int = 8) -> ArrayType:
    """
    Generates Raised Cosine (RC) filter taps.

    Parameters
    ----------
    sps : float
        Samples per symbol.
    rolloff : float, default 0.35
        Roll-off factor (alpha), range [0, 1].
    span : int, default 8
        Filter span in symbols.

    Returns
    -------
    ndarray
        RC filter taps normalized to unit energy.
        Shape: (N_taps,).
    """
    logger.debug("Generating RC taps: sps=%s, rolloff=%s, span=%s", sps, rolloff, span)
    # Ensure odd number of taps
    num_taps = int(span * sps)
    if num_taps % 2 == 0:
        num_taps += 1

    t = np.linspace(-span / 2, span / 2, num_taps)

    # Avoid division by zero
    # Singularities at t = +/- 1 / (2 * rolloff)

    # Initialize array
    h = np.zeros_like(t)

    # General case mask
    # Denominator: 1 - (2 * rolloff * t)**2
    # Singularity when 2 * rolloff * |t| = 1 => |t| = 1 / (2 * rolloff)

    if rolloff > 0:
        idx_singularity = np.isclose(np.abs(t), 1 / (2 * rolloff))
        # Value at singularity: (pi / 4) * sinc(1 / (2 * rolloff))
        # sinc(x) = sin(pi * x) / (pi * x)
        # arg = 1 / (2 * rolloff)
        # val = (pi / 4) * sin(pi * arg) / (pi * arg)
        #     = (pi / 4) * sin(pi / (2 * rolloff)) * (2 * rolloff / pi)
        #     = (rolloff / 2) * sin(pi / (2 * rolloff))
        val_singularity = (rolloff / 2) * np.sin(np.pi / (2 * rolloff))
        h = np.where(idx_singularity, val_singularity, h)
    else:
        idx_singularity = np.zeros_like(t, dtype=bool)

    idx_general = ~idx_singularity

    # h(t) = sinc(t) * cos(pi * alpha * t) / (1 - (2 * alpha * t)^2)
    # sinc(t) = sin(pi * t) / (pi * t) (normalized sinc)

    # To avoid t=0 in sinc division, use np.sinc which handles 0 safely
    sinc_t = np.sinc(t)
    cos_t = np.cos(np.pi * rolloff * t)
    denom = 1 - (2 * rolloff * t) ** 2

    # We masked out where denom is 0, so safe to divide where idx_general is true
    # However we compute everywhere then mask, so denom should not be 0 to avoid warning/NaN if backend evals strict
    # backend.where usually evals both branches
    # So we set denom to 1 where it is 0
    denom_safe = np.where(idx_singularity, 1.0, denom)

    res = sinc_t * cos_t / denom_safe
    h = np.where(idx_general, res, h)

    return normalize(h, "unit_energy")


def lowpass_taps(
    sampling_rate: float,
    num_taps: int,
    cutoff: float,
    window: str = "hamming",
) -> ArrayType:
    """
    Design Lowpass FIR filter using the window method.

    Parameters
    ----------
    sampling_rate : float
        The sampling rate of the signal in Hz.
    num_taps : int
        Number of filter coefficients.
    cutoff : float
        Cutoff frequency in Hz.
    window : str, default "hamming"
        Type of window function to apply (e.g., 'hamming', 'blackman').

    Returns
    -------
    ndarray
        Filter taps with 0 dB passband gain.
        Shape: (num_taps,).
    """
    logger.debug("Designing Lowpass FIR: cutoff=%s Hz, taps=%s", cutoff, num_taps)
    h = scipy.signal.firwin(
        num_taps, cutoff, window=window, fs=sampling_rate, pass_zero=True
    )
    return h


def highpass_taps(
    sampling_rate: float,
    num_taps: int,
    cutoff: float,
    window: str = "hamming",
) -> ArrayType:
    """
    Design Highpass FIR filter using the window method.

    Parameters
    ----------
    sampling_rate : float
        The sampling rate of the signal in Hz.
    num_taps : int
        Number of filter coefficients. For highpass filters, this should
        typically be an odd integer to avoid a zero at the Nyquist frequency.
    cutoff : float
        Cutoff frequency in Hz.
    window : str, default "hamming"
        Type of window function to apply.

    Returns
    -------
    ndarray
        Filter taps with 0 dB passband gain.
        Shape: (num_taps,).
    """
    logger.debug("Designing Highpass FIR: cutoff=%s Hz, taps=%s", cutoff, num_taps)
    # pass_zero=False for highpass
    h = scipy.signal.firwin(
        num_taps, cutoff, window=window, fs=sampling_rate, pass_zero=False
    )
    return h


def bandpass_taps(
    sampling_rate: float,
    num_taps: int,
    low_cutoff: float,
    high_cutoff: float,
    window: str = "hamming",
) -> ArrayType:
    """
    Design Bandpass FIR filter using the window method.

    Parameters
    ----------
    sampling_rate : float
        The sampling rate of the signal in Hz.
    num_taps : int
        Number of filter coefficients.
    low_cutoff : float
        Lower cutoff frequency in Hz.
    high_cutoff : float
        Upper cutoff frequency in Hz.
    window : str, default "hamming"
        Type of window function to apply.

    Returns
    -------
    ndarray
        Filter taps with 0 dB passband gain.
        Shape: (num_taps,).
    """
    logger.debug(
        "Designing Bandpass FIR: range=[%s, %s] Hz, taps=%s",
        low_cutoff,
        high_cutoff,
        num_taps,
    )
    # pass_zero=False for bandpass
    h = scipy.signal.firwin(
        num_taps,
        [low_cutoff, high_cutoff],
        window=window,
        fs=sampling_rate,
        pass_zero=False,
    )
    return h


def bandstop_taps(
    sampling_rate: float,
    num_taps: int,
    low_cutoff: float,
    high_cutoff: float,
    window: str = "hamming",
) -> ArrayType:
    """
    Design Bandstop FIR filter using the window method.

    Parameters
    ----------
    sampling_rate : float
        The sampling rate of the signal in Hz.
    num_taps : int
        Number of filter coefficients. Should typically be odd.
    low_cutoff : float
        Lower cutoff frequency in Hz.
    high_cutoff : float
        Upper cutoff frequency in Hz.
    window : str, default "hamming"
        Type of window function to apply.

    Returns
    -------
    ndarray
        Filter taps with 0 dB passband gain.
        Shape: (num_taps,).
    """
    logger.debug(
        "Designing Bandstop FIR: range=[%s, %s] Hz, taps=%s",
        low_cutoff,
        high_cutoff,
        num_taps,
    )
    # pass_zero=True for bandstop
    h = scipy.signal.firwin(
        num_taps,
        [low_cutoff, high_cutoff],
        window=window,
        fs=sampling_rate,
        pass_zero=True,
    )
    return h


# -----------------------------------------------------------------------------
# FILTERING OPERATIONS (Signal-aware)
# -----------------------------------------------------------------------------
# _ols_forward:  OLS block windowing + batch FFT (shared scaffold)
# _ols_backward: OLS batch IFFT + symmetric discard + reshape (shared scaffold)
# ols_fir_filter: Public OLS FIR convolution (long-tap / memory-bounded)
# shaping_filter_taps: Reconstruct pulse-shaping taps from a Signal's own
#   metadata (pulse_shape/sps/rolloff) - takes a Signal, returns taps, used
#   by matched_filter below to derive default taps for Signal input.
# fir_filter: Generic FIR filtering operation (short-to-medium taps)
# matched_filter: Apply matched filter (time-reversed conjugate of pulse shape)
#
# shape_pulse (TX symbol -> waveform synthesis) lives in core/generation.py,
# not here: it is a signal-construction primitive, not a transform on an
# existing Signal's samples (see CLAUDE.md, "Signal-Awareness").


def _ols_forward(samples: ArrayType, N_fft: int):
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
        Scaffold parameters required by ``_ols_backward``:
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


def _ols_backward(X_hat_f: ArrayType, meta: dict) -> ArrayType:
    """
    Overlap-and-save backward pass: batch IFFT, symmetric discard, reshape.

    Parameters
    ----------
    X_hat_f : array_like
        Frequency-domain blocks after per-bin processing.
        Shape: ``(num_ch, num_blocks, N_fft)``.
    meta : dict
        Scaffold parameters returned by ``_ols_forward``.

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


def ols_fir_filter(
    samples: ArrayType | Signal,
    taps: ArrayType,
    N_fft: int | None = None,
    center: bool = True,
) -> ArrayType | Signal:
    """
    Overlap-and-save FIR filter for long-tap or large-signal convolution.

    Implements the overlap-and-save (OLS) block-processing algorithm, which
    processes the signal in fixed-size FFT blocks. This makes it suitable
    for filters with long impulse responses (e.g., chromatic dispersion
    compensation, group-delay equalizers) where a single full-signal FFT
    would be memory-prohibitive on GPU.

    For short filters on moderate-length signals, ``fir_filter`` (which
    uses scipy's FFT convolution) is equally efficient and simpler.

    Parameters
    ----------
    samples : array_like or Signal
        Input signal. Shape: ``(N,)`` for SISO or ``(C, N)`` for
        multi-channel.  A :class:`Signal` returns a new filtered
        :class:`Signal`.
    taps : array_like
        FIR filter coefficients. Shape: ``(L,)``.
    N_fft : int, optional
        FFT block size. Must be a power of 2. Defaults to
        ``max(1024, next_power_of_2(4 * L))`` so that the 25 % guard
        region is at least ``L`` samples long.
    center : bool, default True
        When ``True`` (default), the output alignment matches
        ``fir_filter`` (scipy ``mode='same'``, center-aligned at tap
        ``L // 2``).  The output at position ``n`` equals
        ``sum_k x[n + L//2 - k] * taps[k]``, which is correct for
        pulse-shaped signals where the filter group delay must be
        compensated before symbol sampling.

        When ``False``, the output is the causal linear convolution
        ``y[n] = sum_k x[n-k] * taps[k]`` (equivalent to
        ``numpy.convolve(x, taps, mode='full')[:N]``).  Use this when
        you need the raw causal impulse response (e.g. measuring filter
        step response) or when writing CD/dispersion compensation where
        the two-sided inverse filter alignment is handled externally.

    Returns
    -------
    array_like
        Filtered signal, same shape as ``samples``.

    Notes
    -----
    A symmetric guard of ``N_fft // 4`` samples is discarded from each
    block edge, so ``N_fft // 4 >= len(taps)`` must hold.

    The ``center=True`` path post-pads the input by ``L // 2`` zeros
    before OLS processing and trims the same number of leading output
    samples - a zero-copy shift that costs one extra OLS block at most.
    """
    x, sig = unwrap_signal(samples)
    if sig is not None:
        return rewrap_signal(sig, ols_fir_filter(x, taps, N_fft=N_fft, center=center))

    samples, xp, _ = dispatch(samples)
    taps = xp.asarray(taps)
    is_real = not xp.iscomplexobj(samples) and not xp.iscomplexobj(taps)
    out_dtype = samples.dtype  # capture before any reshape

    # Signal drives precision: cast taps to match signal so float64 tap
    # generators do not silently upcast complex64 signals via FFT multiply.
    target_tap_dtype = (
        samples.real.dtype if not xp.iscomplexobj(taps) else samples.dtype
    )
    if taps.dtype != target_tap_dtype:
        taps = taps.astype(target_tap_dtype)

    L = len(taps)
    half = L // 2

    samples, was_1d = as_2d(samples, name="samples")

    N = samples.shape[-1]

    if N_fft is None:
        N_fft = max(1024, 1 << (max(1, 4 * L) - 1).bit_length())

    logger.debug(
        "ols_fir_filter: L=%s, N=%s, N_fft=%s, num_ch=%s, center=%s",
        L,
        N,
        N_fft,
        samples.shape[0],
        center,
    )

    H = xp.fft.fft(taps, n=N_fft)  # frequency response of the filter

    if center:
        # Post-pad by half so the OLS can compute full_conv[half : half+N].
        # This matches scipy's mode='same' (center-aligned, group-delay compensated),
        # which is required for correct eye-opening after pulse-shaped filtering.
        samples_ext = xp.pad(samples, ((0, 0), (0, half)))
        Y, meta = _ols_forward(samples_ext, N_fft)
        X_hat_f = Y * H
        out_ext = _ols_backward(X_hat_f, meta)  # shape: (num_ch, N + half)
        out = out_ext[:, half:]  # trim leading half -> shape: (num_ch, N)
    else:
        Y, meta = _ols_forward(samples, N_fft)
        X_hat_f = Y * H
        out = _ols_backward(X_hat_f, meta)

    if is_real:
        out = out.real  # strip IFFT imaginary noise for real inputs
    elif out.dtype != out_dtype:
        out = out.astype(
            out_dtype
        )  # guard complex inputs (e.g. complex64 -> complex128)
    return restore_1d(was_1d, out)


def shaping_filter_taps(sig: Signal) -> ArrayType:
    """
    Compute pulse-shaping filter taps from a :class:`Signal`'s metadata.

    Reconstructs the transmit pulse-shaping taps from ``pulse_shape`` and the
    associated parameters (``sps``, ``filter_span``, roll-offs, ``duty_cycle``,
    ``rise_time``) stored on the signal.  The taps are returned on the signal's
    current backend.

    Parameters
    ----------
    sig : Signal
        Signal carrying valid ``pulse_shape`` metadata.

    Returns
    -------
    array_like
        Generated filter taps on the signal's device.

    Raises
    ------
    ValueError
        If ``pulse_shape`` is missing or unsupported.
    """
    if not sig.pulse_shape or sig.pulse_shape == "none":
        raise ValueError("No pulse shape defined for this signal.")
    logger.info("Generating shaping filter taps (shape: %s).", sig.pulse_shape)

    # Use stored duty_cycle for RZ; NRZ always uses the full symbol period.
    duty_cycle = sig.duty_cycle if sig.mod_rz else 1.0

    if sig.pulse_shape == "rect":
        taps = rect_taps(int(sig.sps), duty_cycle=duty_cycle, rise_time=sig.rise_time)
    elif sig.pulse_shape == "smoothrect":
        taps = smoothrect_taps(
            sps=int(sig.sps),
            span=sig.filter_span,
            rise_time=sig.rise_time,
            duty_cycle=duty_cycle,
        )
    elif sig.pulse_shape == "gaussian":
        taps = gaussian_taps(
            sps=sig.sps, span=sig.filter_span, duty_cycle=sig.duty_cycle
        )
    elif sig.pulse_shape == "rrc":
        taps = rrc_taps(sps=sig.sps, span=sig.filter_span, rolloff=sig.rrc_rolloff)
    elif sig.pulse_shape == "rc":
        taps = rc_taps(sps=sig.sps, span=sig.filter_span, rolloff=sig.rc_rolloff)
    else:
        raise ValueError(f"Unknown pulse shape: {sig.pulse_shape}")

    return to_device(taps, sig.backend)


def fir_filter(
    samples: ArrayType | Signal, taps: ArrayType, axis: int = -1
) -> ArrayType | Signal:
    """
    Apply a Finite Impulse Response (FIR) filter to signal samples.

    The filter is applied via FFT-based convolution for high throughput,
    efficiently handling both CPU and GPU backends.

    Parameters
    ----------
    samples : array_like or Signal
        Input signal samples. Shape: (..., N_samples).  A :class:`Signal`
        returns a new filtered :class:`Signal`.
    taps : array_like
        FIR filter coefficients (impulse response). Shape: (N_taps,).
    axis : int, default -1
        The axis along which the filter is applied (typically the Time axis).

    Returns
    -------
    array_like or Signal
        Filtered samples with the same shape as `samples` (mode='same').
    """
    x, sig = unwrap_signal(samples)
    if sig is not None:
        return rewrap_signal(sig, fir_filter(x, taps, axis=-1))

    logger.debug(
        "Applying FIR filter via convolution (%s taps, axis=%s).", len(taps), axis
    )
    samples, xp, sp = dispatch(samples)

    # Ensure taps are on the correct backend
    taps = xp.asarray(taps)

    # Signal drives precision: cast taps to match signal dtype so numpy/scipy
    # type-promotion rules do not silently upcast float32/complex64 signals.
    target_tap_dtype = (
        samples.real.dtype if not xp.iscomplexobj(taps) else samples.dtype
    )
    if taps.dtype != target_tap_dtype:
        taps = taps.astype(target_tap_dtype)

    if samples.ndim > 1:
        # Ensure axis is positive
        axis = axis % samples.ndim

        new_shape = [1] * samples.ndim
        new_shape[axis] = len(taps)
        taps_nd = taps.reshape(new_shape)

        result = sp.signal.convolve(samples, taps_nd, mode="same", method="fft")
    else:
        # 1D case
        result = sp.signal.convolve(samples, taps, mode="same", method="fft")

    # Belt-and-suspenders: scipy may still promote internally (version-dependent)
    if result.dtype != samples.dtype:
        result = result.astype(samples.dtype)
    return result


def matched_filter(
    samples: ArrayType | Signal,
    pulse_taps: ArrayType | None = None,
    taps_normalization: str = "unit_energy",
    axis: int = -1,
) -> ArrayType | Signal:
    """
    Applies a matched filter to the received signal.

    The matched filter is the time-reversed complex conjugate of the pulse
    shaping filter. It maximizes the Signal-to-Noise Ratio (SNR) in the
    presence of AWGN.

    Parameters
    ----------
    samples : array_like or Signal
        Input received samples. Shape: (..., N_samples).  A :class:`Signal`
        returns a new matched-filtered :class:`Signal`; when ``pulse_taps`` is
        omitted, the taps are derived from the signal's ``pulse_shape`` metadata
        via :func:`shaping_filter_taps`.
    pulse_taps : array_like, optional
        Taps of the pulse-shaping filter used at the transmitter.
        Shape: (N_taps,).  Required for raw-array input.
    taps_normalization : {"unit_energy", "unity_gain"}, default "unit_energy"
        Designates how the matched filter taps are normalized.
    axis : int, default -1
        The axis along which to apply the filter.

    Returns
    -------
    array_like or Signal
        Matched filtered samples. Shape: (..., N_samples).
    """
    x, sig = unwrap_signal(samples)
    if sig is not None:
        taps = pulse_taps
        if taps is None:
            try:
                taps = shaping_filter_taps(sig)
            except ValueError as e:
                logger.error("Cannot apply matched filter: %s", e)
                return sig.copy()
        return rewrap_signal(
            sig, matched_filter(x, taps, taps_normalization=taps_normalization, axis=-1)
        )

    if pulse_taps is None:
        raise ValueError("matched_filter() requires pulse_taps for array input.")

    logger.debug("Applying Matched Filter (taps length=%s).", len(pulse_taps))
    samples, xp, _ = dispatch(samples)

    # Matched filter is conjugate and time-reversed version of pulse
    # Ensure pulse_taps is on correct backend
    pulse_taps = xp.asarray(pulse_taps)
    matched_taps = xp.conj(pulse_taps[::-1])

    if taps_normalization == "unity_gain":
        matched_taps = normalize(matched_taps, mode="unity_gain")
    elif taps_normalization == "unit_energy":
        matched_taps = normalize(matched_taps, mode="unit_energy")
    else:
        raise ValueError(
            f"Unknown taps_normalization: {taps_normalization!r}. "
            "Use 'unity_gain' or 'unit_energy'."
        )

    return fir_filter(samples, matched_taps, axis=axis)


# -----------------------------------------------------------------------------
# CHROMATIC DISPERSION (Signal-aware)
# -----------------------------------------------------------------------------


def compensate_chromatic_dispersion(
    samples: ArrayType | Signal,
    sampling_rate: float | None = None,
    dispersion_ps_nm_km: float | None = None,
    fiber_length_km: float | None = None,
    center_wavelength_nm: float | None = None,
) -> ArrayType | Signal:
    """
    Electronic dispersion compensation (EDC) for chromatic dispersion.

    Applies the inverse of the CD frequency-domain transfer function to remove
    chromatic dispersion accumulated over a fiber link:

        H_EDC(f) = exp(j/2 * beta_2 * (2*pi*f)^2 * L)

    where

        beta_2 = -D * lambda^2 / (2*pi*c)

    and D is the dispersion parameter, lambda is the center wavelength,
    c is the speed of light, and L is the fiber length.

    Parameters
    ----------
    samples : array_like or Signal
        Complex baseband signal. Shape: ``(N,)`` (SISO) or ``(C, N)`` (MIMO).
    sampling_rate : float, optional
        Sampling rate in Hz.  Required for array input; defaults to the
        signal's ``sampling_rate`` for :class:`Signal` input.
    dispersion_ps_nm_km : float
        Fiber dispersion parameter D in ps / (nm * km).
        Standard SMF-28: 17 ps/(nm*km) at 1550 nm.
    fiber_length_km : float
        Fiber span length in km.
    center_wavelength_nm : float
        Center wavelength in nm (e.g. 1550 for C-band).

    Returns
    -------
    array_like or Signal
        CD-compensated signal, same shape, dtype, and backend as input.  A
        :class:`Signal` returns a new compensated :class:`Signal`.

    See Also
    --------
    commkit.impairments.apply_chromatic_dispersion :
        Apply the forward CD impairment (use before this function in simulation).

    Examples
    --------
    >>> cd_free = compensate_chromatic_dispersion(
    ...     received, dispersion_ps_nm_km=17.0, fiber_length_km=80.0,
    ...     center_wavelength_nm=1550.0, sampling_rate=fs)
    """
    x, sig = unwrap_signal(samples)
    if sig is not None:
        result = compensate_chromatic_dispersion(
            x,
            sampling_rate if sampling_rate is not None else sig.sampling_rate,
            dispersion_ps_nm_km,
            fiber_length_km,
            center_wavelength_nm,
        )
        return rewrap_signal(sig, result)

    if sampling_rate is None:
        raise ValueError(
            "compensate_chromatic_dispersion() requires sampling_rate for array input."
        )
    if (
        dispersion_ps_nm_km is None
        or fiber_length_km is None
        or center_wavelength_nm is None
    ):
        raise ValueError(
            "compensate_chromatic_dispersion() requires dispersion_ps_nm_km, "
            "fiber_length_km, and center_wavelength_nm."
        )

    logger.info(
        "Compensating CD (D=%s ps/nm/km, L=%s km, λ=%s nm).",
        dispersion_ps_nm_km,
        fiber_length_km,
        center_wavelength_nm,
    )

    samples, xp, _ = dispatch(samples)
    samples, was_1d = as_2d(samples, name="samples")
    C, N = samples.shape

    # Convert to SI
    D = dispersion_ps_nm_km * 1e-12 / (1e-9 * 1e3)  # s / m²
    lam = center_wavelength_nm * 1e-9  # m
    c = 2.998e8  # m/s
    L = fiber_length_km * 1e3  # m
    beta2 = -(D * lam**2) / (2.0 * np.pi * c) * L  # s²  (β₂·L product)

    omega = 2.0 * np.pi * xp.fft.fftfreq(N, d=1.0 / sampling_rate)
    H = xp.exp(1j * (beta2 / 2.0) * omega**2)

    S_F = xp.fft.fft(samples, axis=-1)
    out_F = S_F * H[None, :]
    result = xp.fft.ifft(out_F, axis=-1)

    if result.dtype != samples.dtype:
        result = result.astype(samples.dtype)

    return restore_1d(was_1d, result)
