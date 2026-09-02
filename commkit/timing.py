"""
Timing synchronization utilities.

This module provides routines for time synchronization, including the
generation of optimal synchronization sequences (Barker, Zadoff-Chu),
robust integer timing offset estimation via cross-correlation, and
fractional timing offset estimation and correction.
"""

import logging
from typing import Union, overload

import numpy as np

from .backend import ArrayType, dispatch, is_cupy_available, to_device
from .core import Preamble, Signal
from .helpers import as_2d, restore_1d, rewrap_signal, unwrap_signal
from .logger import logger

# Window length for DFT-upsampling in estimate_fractional_delay()
_DFT_WINDOW = 33

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


def barker_sequence(length: int) -> ArrayType:
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
        Backend depends on system availability (CuPy if available, else NumPy).

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

    if is_cupy_available():
        seq = to_device(seq, "gpu")

    logger.debug("Generated Barker-%s sequence.", length)
    return seq


def zadoff_chu_sequence(length: int, root: int = 1) -> ArrayType:
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

    # Determine backend
    if is_cupy_available():
        import cupy as cp

        xp = cp
    else:
        xp = np

    # ZC formula: x[n] = exp(-j * pi * u * n * (n+1) / N)
    n = xp.arange(length)
    if length % 2 == 0:
        # Even length: x[n] = exp(-j * pi * u * n^2 / N)
        seq = xp.exp(-1j * xp.pi * root * n * n / length)
    else:
        # Odd length: x[n] = exp(-j * pi * u * n * (n+1) / N)
        seq = xp.exp(-1j * xp.pi * root * n * (n + 1) / length)

    seq = seq.astype(xp.complex64)

    logger.debug("Generated ZC sequence: length=%s, root=%s.", length, root)
    return seq


def estimate_fractional_delay(
    correlation: ArrayType,
    peak_indices: ArrayType,
    dft_upsample: int = 1,
    method: str = "log-parabolic",
) -> ArrayType:
    """
    Estimates sub-sample timing offset via parabolic interpolation.

    Given a correlation array and the integer peak positions, fits a
    parabola (or Gaussian) through the three points around each peak to
    estimate the fractional offset with sub-sample precision.

    When the input is **complex**, the peak is phase-rotated to the real
    axis before fitting. This preserves the true peak shape (unlike
    fitting to ``|R|``) and improves accuracy by 2-5x for typical
    matched-filter outputs.

    If ``dft_upsample > 1``, performs a "Zoom FFT" (DFT zero-padding)
    around the peak to interpolate the correlation function onto a finer
    grid before fitting. This significantly reduces bias error.

    Parameters
    ----------
    correlation : array_like
        Correlation values - complex or real magnitude.
        Shape: ``(N,)`` or ``(C, N)``.
    peak_indices : array_like
        Integer peak positions. Shape: scalar or ``(C,)``.
    dft_upsample : int, default 1
        Upsampling factor for DFT-based interpolation.
        Values > 1 perform zero-padded FFT interpolation on a window
        around the peak (typically 33 samples).
    method : {'parabolic', 'log-parabolic'}, default 'log-parabolic'
        Fitting method:
        - 'parabolic': Standard parabolic fit. Good for general peaks.
        - 'log-parabolic': Fits a parabola to log(y), equivalent to a
          Gaussian fit. Often more accurate for bandlimited pulses.

    Returns
    -------
    array_like
        Fractional offset per channel, in [-0.5, 0.5). Shape: ``(C,)`` or scalar.
    """
    correlation, xp, _ = dispatch(correlation)
    peak_indices = xp.asarray(peak_indices)
    scalar_input = peak_indices.ndim == 0

    if correlation.ndim == 1:
        correlation = correlation[None, :]  # (1, N)
    if peak_indices.ndim == 0:
        peak_indices = peak_indices[None]  # (1,)

    N = correlation.shape[-1]
    C = correlation.shape[0]
    ch_idx = xp.arange(C)

    k = peak_indices.astype(int)
    mu = xp.zeros(C, dtype=correlation.real.dtype)

    def _calculate_mu(r_prev, r_curr, r_next, xp, method):
        if xp.iscomplexobj(r_curr):
            phase = xp.exp(-1j * xp.angle(r_curr))
            alpha = (r_prev * phase).real
            beta = (r_curr * phase).real
            gamma = (r_next * phase).real
        else:
            alpha = r_prev
            beta = r_curr
            gamma = r_next

        if method == "log-parabolic":
            eps = 1e-12
            alpha = xp.maximum(alpha, eps)
            beta = xp.maximum(beta, eps)
            gamma = xp.maximum(gamma, eps)
            alpha, beta, gamma = xp.log(alpha), xp.log(beta), xp.log(gamma)

        denom = 2.0 * (alpha - 2.0 * beta + gamma)
        safe_denom = xp.where(xp.abs(denom) > 1e-12, denom, xp.ones_like(denom))
        mu_val = (alpha - gamma) / safe_denom

        # Mask invalid fits
        valid = xp.abs(denom) > 1e-12
        mu_val = xp.where(valid, mu_val, xp.zeros_like(mu_val))
        return xp.clip(mu_val, -0.5, 0.5)

    half_W = _DFT_WINDOW // 2
    interior_mask = (k >= half_W) & (k < N - half_W)

    # -------------------------------------------------------------------------
    # Path A: DFT Upsampling
    # -------------------------------------------------------------------------
    if dft_upsample > 1:
        if xp.any(interior_mask):
            valid_indices = xp.where(interior_mask)[0]
            k_valid = k[interior_mask]

            offsets = xp.arange(-half_W, half_W + 1)
            gather_idx = k_valid[:, None] + offsets[None, :]
            windows = correlation[valid_indices[:, None], gather_idx]

            specs = xp.fft.fft(windows, axis=-1)
            pos_limit = (_DFT_WINDOW + 1) // 2
            neg_len = _DFT_WINDOW - pos_limit
            target_len = _DFT_WINDOW * dft_upsample
            padded_specs = xp.zeros((len(valid_indices), target_len), dtype=specs.dtype)
            padded_specs[:, :pos_limit] = specs[:, :pos_limit]
            padded_specs[:, -neg_len:] = specs[:, pos_limit:]

            upsampled = xp.fft.ifft(padded_specs, axis=-1) * dft_upsample
            up_mag = xp.abs(upsampled)
            k_up = xp.argmax(up_mag, axis=-1)
            k_up_safe = xp.clip(k_up, 1, target_len - 2)

            row_idx = xp.arange(len(valid_indices))
            r_prev = upsampled[row_idx, k_up_safe - 1]
            r_curr = upsampled[row_idx, k_up_safe]
            r_next = upsampled[row_idx, k_up_safe + 1]

            mu_up = _calculate_mu(r_prev, r_curr, r_next, xp, method)

            # Position relative to window start: k_up + mu_up
            # Center of window is at index (half_W * dft_upsample)?
            # No, standard ZoomFFT mapping.
            # Time axis of upsampled is 0 to W*M-1.
            # Original sample k corresponds to center of window.
            # Window covers [k - half_W, k + half_W].
            # Index 0 of upsampled corresponds to k - half_W.
            # So offset from k is: -half_W + (k_up + mu_up) / M.
            offset_samples = -half_W + (k_up_safe + mu_up) / dft_upsample
            mu[interior_mask] = offset_samples

    # -------------------------------------------------------------------------
    # Path B: Standard (Primary when M=1, fallback for edge channels when M>1)
    # -------------------------------------------------------------------------
    # The trigger is determined purely by which channels still need a result:
    #   - M=1: all channels (Path A was skipped entirely)
    #   - M>1: only ~interior_mask channels (edges Path A cannot process)
    # Do NOT use `mu == 0` as a proxy - zero is a valid fractional offset and
    # would wrongly overwrite DFT results for on-centre correlation peaks.
    if dft_upsample == 1:
        calc_mask = xp.ones(C, dtype=bool)
    else:
        calc_mask = ~interior_mask  # only edge channels need the fallback

    if xp.any(calc_mask):
        interior_all = (k >= 1) & (k < N - 1)
        k_all_safe = xp.clip(k, 1, N - 2)

        r_prev = correlation[ch_idx, k_all_safe - 1]
        r_curr = correlation[ch_idx, k_all_safe]
        r_next = correlation[ch_idx, k_all_safe + 1]

        mu_std = _calculate_mu(r_prev, r_curr, r_next, xp, method)
        mu_std = xp.where(interior_all, mu_std, xp.zeros_like(mu_std))

        if dft_upsample == 1:
            mu = mu_std
        else:
            # Merge: DFT result for interior channels, standard for edge channels
            mu = xp.where(interior_mask, mu, mu_std)

    if scalar_input:
        return mu[0]
    return mu


@overload
def fft_fractional_delay(samples: ArrayType, delay: float | ArrayType) -> ArrayType: ...


@overload
def fft_fractional_delay(samples: Signal, delay: float | ArrayType) -> Signal: ...


def fft_fractional_delay(
    samples: ArrayType | Signal,
    delay: float | ArrayType,
) -> ArrayType | Signal:
    """
    Applies fractional sample delay using FFT-based frequency-domain method.

    This is the mathematically ideal method for bandlimited signals. It
    applies the delay as a phase shift in the frequency domain, which is
    equivalent to ideal sinc interpolation in the time domain. Unlike
    polynomial interpolators (e.g., Farrow), this method perfectly preserves
    signal power and has no bandwidth limitations.

    Parameters
    ----------
    samples : array_like or Signal
        Input signal. Shape: (N,) or (C, N).  A :class:`Signal` returns a
        new delayed :class:`Signal`.
    delay : float or array_like
        Fractional delay in samples. Positive = delay (shift right).
        Scalar applies the same delay to all channels.
        Array of shape (C,) applies per-channel delays.

    Returns
    -------
    array_like or Signal
        Delayed signal with the same shape as input.

    Notes
    -----
    Applies Y(f) = X(f) * exp(-j * 2*pi * f * delay / N) - equivalent to
    ideal sinc interpolation with perfect power preservation.
    """
    x, sig = unwrap_signal(samples)
    if sig is not None:
        return rewrap_signal(sig, fft_fractional_delay(x, delay))

    samples, xp, _ = dispatch(samples)
    samples, was_1d = as_2d(samples, name="samples")

    C, N = samples.shape

    # Convert delay to array
    if isinstance(delay, (int, float)):
        delay_arr = xp.full(C, delay, dtype=samples.real.dtype)
    else:
        delay_arr = xp.asarray(delay, dtype=samples.real.dtype)
        if delay_arr.ndim == 0:
            delay_arr = delay_arr[None]

    # FFT
    spec = xp.fft.fft(samples, axis=-1)

    # Frequency axis: normalized frequencies in cycles/sample
    freqs = xp.fft.fftfreq(N, d=1.0)

    # Phase shift: exp(-j * 2 * pi * f * delay)
    # Positive delay -> phase ramp that shifts signal to the right.
    # Computed at float64 accuracy, then cast to match spec's dtype to prevent
    # -2j * xp.pi (complex128) from promoting complex64 spectra.
    phase_shift = xp.exp(-2j * xp.pi * freqs[None, :] * delay_arr[:, None])
    phase_shift = phase_shift.astype(spec.dtype)

    # Apply phase shift
    spec_delayed = spec * phase_shift

    # IFFT
    result = xp.fft.ifft(spec_delayed, axis=-1)

    # Dtype restoration: mirrors the impairments.py pattern.
    if not xp.iscomplexobj(samples):
        # Real input: fractional delay is a real-valued operation
        result = result.real
    elif result.dtype != samples.dtype:
        # Complex input: ifft may return complex128 from complex64 input
        result = result.astype(samples.dtype)

    return restore_1d(was_1d, result)


def estimate_timing(
    samples: ArrayType | Signal,
    reference: Union[ArrayType, "Preamble"] | None = None,
    threshold: float = 3.0,
    sps: int | None = None,
    pulse_shape: str | None = None,
    filter_params: dict | None = None,
    search_range: tuple[int, int] | None = None,
    dft_upsample: int = 1,
    fractional_method: str = "log-parabolic",
    debug_plot: bool = False,
) -> tuple[ArrayType, ArrayType]:
    """
    Estimates integer and fractional timing offsets via cross-correlation.

    Slides a cross-correlation between the received signal and a known reference
    to find the integer-sample timing offset per channel, then estimates the
    fractional offset via parabolic interpolation on the correlation peak.

    For MIMO references with shape ``(C_tx, L)``, every template is correlated
    against every RX channel; each channel takes the strongest peak across all
    (template, lag) pairs.  This preserves per-channel hardware skew and is
    robust to polarization swap or mixing.

    Parameters
    ----------
    samples : array_like or Signal
        Received signal samples. Shape: ``(N,)`` or ``(C, N)``.
    reference : array_like or Preamble
        Reference for correlation.  A Preamble object is reconstructed via
        ``Preamble.to_signal()`` using ``sps`` and ``pulse_shape``; a raw
        array (shape ``(L,)`` or ``(C_tx, L)``) is used directly.
    threshold : float, default 3.0
        Peak-to-average power ratio threshold for peak detection.
    sps : int, optional
        Samples per symbol.  Required when ``reference`` is a Preamble.
    pulse_shape : str, optional
        Pulse shaping filter type used when ``reference`` is a Preamble.
        Defaults to ``'rrc'``.
    filter_params : dict, optional
        Extra pulse shaper parameters when ``reference`` is a Preamble.
    search_range : tuple of int, optional
        ``(start, end)`` sample range to restrict the search.
    dft_upsample : int, default 1
        DFT-based upsampling factor for high-precision fractional estimation.
    fractional_method : {'parabolic', 'log-parabolic'}, default 'log-parabolic'
        Fitting method for fractional delay estimation.
    debug_plot : bool, default False
        If ``True``, plots the correlation magnitude for debugging.

    Returns
    -------
    integer_offsets : ArrayType
        Integer sample offsets where the reference sequence begins,
        per RX channel.  Shape: ``(N_channels,)``.  Each channel's
        offset is estimated independently so hardware skew between
        channels is preserved.
    fractional_offsets : ArrayType
        Sub-sample timing offsets in [-0.5, 0.5) per channel.
        Shape: ``(N_channels,)``.

    Raises
    ------
    ValueError
        If no correlation peak is found that satisfies the threshold
        criteria, or if ``reference`` is not provided.

    Notes
    -----
    The returned integer offset is the sample index of the first reference
    sample.  For oversampled signals, pass a shaped reference at the same sps
    for best timing SNR.
    """
    from .helpers import cross_correlate_fft

    samples, sig = unwrap_signal(samples)
    if sig is not None:
        return estimate_timing(
            samples,
            reference=reference,
            threshold=threshold,
            sps=sps,
            pulse_shape=pulse_shape,
            filter_params=filter_params,
            search_range=search_range,
            dft_upsample=dft_upsample,
            fractional_method=fractional_method,
            debug_plot=debug_plot,
        )

    # 1. Resolve Inputs & Metadata
    if filter_params is None:
        filter_params = {}

    if not hasattr(samples, "ndim"):
        raise TypeError(
            f"estimate_timing() expects an array of samples, got {type(samples).__name__}."
        )

    sig_array, xp, _ = dispatch(samples)

    # 2. Build reference waveform
    ref_waveform = None

    if isinstance(reference, Preamble):
        # Preamble object: reconstruct the shaped waveform at the target SPS.
        if sps is None:
            raise ValueError("SPS must be provided when using a Preamble object.")

        ref_waveform = xp.asarray(
            reference.to_signal(
                sps=sps,
                symbol_rate=1.0,
                pulse_shape=pulse_shape or "rrc",
                **filter_params,
            ).samples
        )
        # ensure 2-D (C_tx, L*sps) for the correlation engine
        if ref_waveform.ndim == 1:
            ref_waveform = ref_waveform[None, :]

    elif reference is not None:
        # Raw array: use directly as the correlation template (e.g. the known
        # TX waveform or a slice of the payload itself).
        ref_waveform = xp.asarray(reference)
    else:
        raise ValueError(
            "A 'reference' sequence must be provided (Preamble object or raw array)."
        )

    # 3. Correlation Strategy
    # Signal: (C, N) or (N,)
    # Reference: (C, L) or (L,) or (1, L)

    # Ensure dimensions match for broadcast/multichannel correlation
    if sig_array.ndim == 1:
        sig_array = sig_array[None, :]  # Treat as 1 channel

    if ref_waveform.ndim == 1:
        ref_waveform = ref_waveform[None, :]  # Treat as 1 template

    # Ensure reference is on the same device as the signal.
    ref_waveform = to_device(ref_waveform, "cpu" if xp is np else "gpu")

    num_sig_ch = sig_array.shape[0]

    # Apply search range
    offset = 0
    if search_range is not None:
        start, end = search_range
        sig_processing = sig_array[:, start:end]
        offset = start
    else:
        sig_processing = sig_array

    # === Vectorized Correlation (FFT) via shared helper ===
    L = ref_waveform.shape[-1]
    C_tx = ref_waveform.shape[0]

    if C_tx > 1:
        # Correlate every TX template against every RX channel.
        # corr_all[rx, tx, lag] - shape (C_rx, C_tx, N_lag)
        corr_all = xp.stack(
            [
                cross_correlate_fft(
                    sig_processing, ref_waveform[t : t + 1], mode="positive_lags"
                )
                for t in range(C_tx)
            ],
            axis=1,
        )

        # === Greedy per-channel peak finding ===
        # Each RX channel takes its strongest peak across all (template, lag).
        # The lag is an independent per-channel argmax, so hardware skew is
        # preserved; the template choice only selects which peak height is read,
        # not its position (all streams share the symbol clock).  Robust to a
        # polarization swap, mixing, or a missing/weak stream.  Polarization
        # identity is not resolved here (see recovery.resolve_channel_permutation).
        corr_all_mag = xp.abs(corr_all)  # (C_rx, C_tx, N_lag)
        best_tx = xp.argmax(xp.max(corr_all_mag, axis=-1), axis=-1)  # (C_rx,)
        corr = xp.take_along_axis(corr_all, best_tx[:, None, None], axis=1)[
            :, 0
        ]  # (C_rx, N_lag) complex - on-device gather, no per-channel host sync
        corr_incoherent = xp.abs(corr)  # (C_rx, N_lag)
        peak_indices = xp.argmax(corr_incoherent, axis=-1)  # (C_rx,)
    else:
        corr = cross_correlate_fft(sig_processing, ref_waveform, mode="positive_lags")

    # Magnitude
    corr_mag = xp.abs(corr)

    # === Per-Channel Analysis ===
    # For SISO / broadcast single template: find peak from magnitude
    if C_tx == 1:
        peak_indices = xp.argmax(corr_mag, axis=-1)  # Shape (C,)

    # === Per-Channel Normalization ===
    # Normalized correlation metric: use local signal energy around each
    # detected peak (L-sample window) instead of global mean.  This gives
    # metric ≈ 1.0 for a perfectly matched reference regardless of the
    # noise/data content elsewhere in the signal.
    e_ref = xp.sum(xp.abs(ref_waveform) ** 2, axis=-1)  # (C_tx,) or scalar
    if e_ref.ndim == 0:
        e_ref = xp.full(num_sig_ch, e_ref)
    else:
        e_ref = xp.full(num_sig_ch, xp.mean(e_ref))

    # Local signal energy: sum |sig|² in the L-sample window at each peak.
    # Batch the peak indices to host once instead of one sync per channel.
    N_sig = sig_processing.shape[-1]
    peaks_np = to_device(peak_indices, "cpu")
    e_s = xp.empty(num_sig_ch, dtype=sig_processing.real.dtype)
    for ch in range(num_sig_ch):
        pk = int(peaks_np[ch])
        end_idx = min(pk + L, N_sig)
        e_s[ch] = xp.sum(xp.abs(sig_processing[ch, pk:end_idx]) ** 2)

    norm_factors = xp.sqrt(e_ref * e_s)
    norm_factors = xp.maximum(norm_factors, 1e-12)

    # Calculate per-channel coherence (diagnostic mathematically absolute bound, [0, 1])
    if C_tx > 1:
        peak_vals = xp.max(
            corr_incoherent, axis=-1
        )  # (C,) incoherent - matches peak detection
        mean_vals = xp.mean(corr_incoherent, axis=-1)
    else:
        peak_vals = xp.max(corr_mag, axis=-1)  # (C,)
        mean_vals = xp.mean(corr_mag, axis=-1)

    coherence = peak_vals / norm_factors
    coherence = xp.clip(coherence, 0.0, 1.0)

    # Primary timing metric: PAPR (Peak-to-Average magnitude)
    # This evaluates visual prominence against the noise floor, ensuring
    # extreme robustness to CFO phase-rotation and pulse-shape mismatch.
    mean_vals = xp.maximum(mean_vals, 1e-12)
    metrics = peak_vals / mean_vals

    # Log coherence diagnostics - one batched D2H per array, not per channel
    coherence_np = to_device(coherence, "cpu")
    metrics_np = to_device(metrics, "cpu")
    for ch in range(num_sig_ch):
        c_val = float(coherence_np[ch])
        p_val = float(metrics_np[ch])
        if c_val < 0.5 and p_val >= threshold:
            logger.warning(
                "Channel %s: Peak phase coherence is very low (%.2f), but "
                "peak is visually prominent (PAPR=%.1f >= %s). This "
                "suggests strong Carrier Frequency Offset (CFO) or "
                "uncompensated dispersion destroying phase alignment "
                "over the sequence length.",
                ch,
                c_val,
                p_val,
                threshold,
            )
        else:
            logger.debug(
                "Channel %s: Peak prominence = %.1f, coherence = %.2f", ch, p_val, c_val
            )

    # === Threshold Check === (reuses the host copy - no further syncs)
    max_metric = float(metrics_np.max())
    if max_metric < threshold:
        raise ValueError(
            f"No correlation peak above threshold {threshold} (max: {max_metric:.3f})"
        )
    for _ch in range(num_sig_ch):
        _m = float(metrics_np[_ch])
        if _m < threshold:
            logger.warning(
                "Channel %s: correlation metric %.3f is below threshold "
                "%s. Integer offset for this channel may be unreliable.",
                _ch,
                _m,
                threshold,
            )

    # === Skew Check (Robust) ===
    if num_sig_ch > 1:
        valid_mask = metrics > threshold

        if xp.sum(valid_mask) > 1:
            valid_peaks = peak_indices[valid_mask]
            spread = int(xp.max(valid_peaks) - xp.min(valid_peaks))

            if spread > 0:
                logger.warning(
                    "Skew detected among valid channels! Valid Peaks: %s. "
                    "Spread: %s samples.",
                    valid_peaks.tolist(),
                    spread,
                )
            else:
                logger.info("Channels aligned (spread %s).", spread)

    # Each channel's peak is found independently so hardware skew is preserved.
    integer_offsets = xp.maximum(0, peak_indices + offset)

    if debug_plot:
        from . import plotting as _plotting

        _plotting.plot_timing_correlation(
            corr_mag=to_device(corr_mag, "cpu"),
            peak_indices=to_device(peak_indices, "cpu"),
            norm_factors=to_device(mean_vals, "cpu"),
            threshold=threshold,
            offset=offset,
            show=True,
        )

    # === Fractional Timing (Parabolic Interpolation) ===
    fractional_offsets = estimate_fractional_delay(
        corr, peak_indices, dft_upsample=dft_upsample, method=fractional_method
    )

    if logger.isEnabledFor(logging.INFO):
        # Three .tolist() host syncs, only for this summary line.
        logger.info(
            "Timing estimated. Integer: %s, Fractional: %s, Metrics: %s",
            integer_offsets.tolist(),
            fractional_offsets.tolist(),
            metrics.tolist(),
        )

    return integer_offsets, fractional_offsets


def correct_timing(
    samples: ArrayType | Signal,
    integer_offset: int | ArrayType,
    fractional_offset: float | ArrayType = 0.0,
    mode: str = "circular",
) -> ArrayType | Signal:
    """
    Combined integer and fractional timing correction.

    Applies an integer sample shift followed by fractional sample
    interpolation using FFT-based frequency-domain delay.

    Parameters
    ----------
    samples : array_like or Signal
        Input signal. Shape: (N,) or (C, N).  A :class:`Signal` returns a
        new timing-corrected :class:`Signal`.
    integer_offset : int or array_like
        Integer sample offset(s) to correct. Positive values shift
        the signal left (i.e., remove leading samples).
        Scalar or shape (C,) for per-channel offsets.
    fractional_offset : float or array_like, default 0.0
        Fractional sample delay(s) in [-0.5, 0.5) to correct via
        FFT-based interpolation. Scalar or shape (C,).
    mode : {'circular', 'zero', 'slice'}, default 'circular'
        How to handle boundary samples after the integer shift:

        - ``'circular'``: Wrap-around (``xp.roll``). Output has the
          same length as input.  Correct for periodic signals or
          unit tests; **not** suitable for burst frames.
        - ``'zero'``: Shift left, fill trailing samples with zeros.
          Same output shape as input.  Correct for burst reception
          where tail wrapping would corrupt the payload.
        - ``'slice'``: Discard leading samples; no tail artifact.
          For a scalar offset the output length is ``N - offset``.
          For per-channel offsets the output length is
          ``N - max(offset)`` so all channels are aligned to the
          same common overlap region - the correct approach for
          offline MIMO timing-skew correction.

    Returns
    -------
    array_like or Signal
        Timing-corrected signal.  Same shape as input for
        ``'circular'`` and ``'zero'``; shorter for ``'slice'``.  A
        :class:`Signal` returns a new corrected :class:`Signal`.

    Notes
    -----
    The reference sequence is not stripped from the output; sample 0 is the
    first reference sample.  Fractional delay uses FFT-based phase shift
    (ideal for bandlimited signals; no power loss).  For ``mode='slice'``,
    fractional delay is applied to the full buffer before slicing so the
    wrap-around stays at the buffer ends.
    """
    x, sig = unwrap_signal(samples)
    if sig is not None:
        result = correct_timing(
            x, integer_offset, fractional_offset=fractional_offset, mode=mode
        )
        return rewrap_signal(sig, result)

    samples, xp, _ = dispatch(samples)
    samples, was_1d = as_2d(samples, name="samples")

    num_ch = samples.shape[0]
    N = samples.shape[-1]

    # Pre-evaluate the fractional-correction flag so mode='slice' can apply the
    # delay before the slice. Same logic that used to live just before the
    # post-integer fractional call below - relocated, not changed.
    if isinstance(fractional_offset, (int, float)):
        apply_frac = abs(fractional_offset) > 1e-9
    else:
        fractional_offset = xp.asarray(fractional_offset)
        apply_frac = bool(xp.any(xp.abs(fractional_offset) > 1e-9))

    # mode='slice': apply fractional delay on the *full* pre-slice buffer.
    # fft_fractional_delay treats its input as circular; applying it after the
    # slice would wrap the slice's trailing edge back into its new sample 0 -
    # exactly the frame boundary we just aligned to. Applying first puts the
    # wrap at the physical buffer ends, the leading one of which is then
    # discarded by the slice; only the tail of the slice carries any residual
    # ~sinc-tail artefact, well away from the equalizer training start.
    if mode == "slice" and apply_frac:
        samples = fft_fractional_delay(samples, -fractional_offset)

    # === Integer correction (integer shift) ===
    integer_offset = xp.asarray(integer_offset)

    if integer_offset.ndim == 0:
        # --- Scalar: same shift for all channels ---
        shift = int(integer_offset)
        if mode == "circular":
            samples = xp.roll(samples, -shift, axis=-1)
        elif mode == "zero":
            result = xp.zeros_like(samples)
            if shift > 0:
                result[..., : N - shift] = samples[..., shift:]
            elif shift < 0:
                result[..., -shift:] = samples[..., : N + shift]
            samples = result
        elif mode == "slice":
            samples = samples[..., shift:]
        else:
            raise ValueError(
                f"Unknown mode {mode!r}. Choose 'circular', 'zero', or 'slice'."
            )

    else:
        # --- Per-channel: vectorized gather (avoids one GPU->CPU sync per channel) ---
        integer_shift = integer_offset.astype(xp.int64)  # (C,) on device
        col_base = xp.arange(N, dtype=xp.int64)[None, :]  # (1, N)
        row_idx = xp.arange(num_ch)[:, None]  # (C, 1)

        if mode == "circular":
            col_idx = (col_base + integer_shift[:, None]) % N  # (C, N)
            samples = samples[row_idx, col_idx]

        elif mode == "zero":
            col_raw = col_base + integer_shift[:, None]  # (C, N)
            gathered = samples[row_idx, xp.clip(col_raw, 0, N - 1)]
            samples = xp.where(col_raw < N, gathered, xp.zeros_like(gathered))

        elif mode == "slice":
            # Align all channels to common overlap: N - max(offset) samples
            max_shift = int(xp.max(integer_shift))  # one GPU sync
            common_len = N - max_shift
            col_idx_s = (
                xp.arange(common_len, dtype=xp.int64)[None, :] + integer_shift[:, None]
            )  # (C, common_len)
            samples = samples[row_idx, col_idx_s]

        else:
            raise ValueError(
                f"Unknown mode {mode!r}. Choose 'circular', 'zero', or 'slice'."
            )

    # === Fractional correction (via FFT) ===
    # For mode='slice' this was already applied above on the full pre-slice
    # buffer; here we only handle 'circular' / 'zero', whose output length
    # equals the input length so the wrap location is unchanged either way.
    if apply_frac and mode != "slice":
        samples = fft_fractional_delay(samples, -fractional_offset)

    if mode == "slice":
        logger.warning(
            "correct_timing(mode='slice'): output is shorter than input "
            "(trimmed by %s samples). Signal length metadata (e.g. "
            "duration) will no longer match the original.",
            int(xp.max(xp.asarray(integer_offset))),
        )
    logger.info(
        "Timing corrected: integer=applied, fractional=%s, mode=%r.",
        "applied" if apply_frac else "skipped",
        mode,
    )

    return restore_1d(was_1d, samples)
