"""Timing synchronization.

:func:`estimate_timing` finds the integer and fractional timing offset by
cross-correlation (:func:`cross_correlate_fft`) with a template or a
:class:`~commkit.core.Preamble` (a :class:`TimingEstimate`), and
:func:`correct_timing` removes it.  :func:`estimate_fractional_delay` and
:func:`fft_fractional_delay` measure and apply sub-sample delays.  The
sequences :func:`barker_sequence` and :func:`zadoff_chu_sequence` build
preambles.
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import _fft
from ._array import as_2d, restore_1d
from ._sequences import barker_sequence, zadoff_chu_sequence
from .backend import ArrayType, dispatch, to_device
from .core import Preamble, Signal
from .core._signal_adapter import S, adapt_signal, require_integer_sps
from .filtering import Pulse
from .logger import logger
from .math import _derotate

__all__ = [
    "TimingEstimate",
    "barker_sequence",
    "correct_timing",
    "cross_correlate_fft",
    "estimate_fractional_delay",
    "estimate_timing",
    "fft_fractional_delay",
    "zadoff_chu_sequence",
]

# Window length for DFT-upsampling in estimate_fractional_delay()
_DFT_WINDOW = 33

# -----------------------------------------------------------------------------
# CORRELATION AND PEAK INTERPOLATION (array-only)
# -----------------------------------------------------------------------------


def cross_correlate_fft(
    samples: ArrayType,
    template: ArrayType,
    *,
    mode: str = "full",
) -> ArrayType:
    """Cross-correlate samples with a template through the FFT.

    ``R[k] = Σ_n samples[n + k] · conj(template[n])``, computed as
    ``IFFT(FFT(samples) · conj(FFT(template)))`` on a power-of-2 grid of at
    least ``N + L - 1`` points, for all channels at once (``(C, N)`` input,
    time on the last axis).

    Parameters
    ----------
    samples : array_like
        Samples, ``(N,)`` or ``(C, N)``.
    template : array_like
        Reference sequence, ``(L,)`` or ``(C, L)``.  A ``(1, L)`` template is
        shared by all channels.
    mode : {"full", "same", "valid", "positive_lags"}, default "full"
        Output size:

        - ``"full"``: lags ``-(L - 1) .. N - 1``, length ``N + L - 1``
          (the layout of ``scipy.signal.correlate``).
        - ``"same"``: length ``N``, centered.
        - ``"valid"``: length ``max(N, L) - min(N, L) + 1``.
        - ``"positive_lags"``: lags ``0 .. N - 1`` only, a view of the
          circular correlation, for a timing search that needs no negative
          lags.

    Returns
    -------
    ndarray
        Complex correlation, ``(M,)`` or ``(C, M)`` as ``samples``, with
        ``M`` set by ``mode``.
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
    SIG = _fft.fft(samples, n_fft, axis=-1)
    TPL = _fft.fft(template, n_fft, axis=-1)
    corr_circ = _fft.ifft(SIG * xp.conj(TPL), axis=-1)

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
    r"""Offset of a peak from three samples by a (log-)parabolic fit.

    Fits a parabola through the samples at ``k-1``, ``k`` and ``k+1`` around
    a peak, or through their logarithms (the log-parabolic fit, equivalent to
    a Gaussian), and returns the offset of the vertex from ``k``, clipped to
    ``[-0.5, 0.5]``::

        delta = 0.5 * (y_prev - y_next) / (y_prev - 2*y_curr + y_next)

    Every three-point peak fit of the library uses it: ``frequency.MthPower``
    on magnitudes (``log=False``), ``frequency.BiasTone`` and
    ``frequency._refine_tones_from_spectrum`` (``log=True``), and
    :func:`estimate_fractional_delay` with either fit, which first rotates a
    complex peak onto the real axis and passes its own ``log_eps`` and
    ``denom_eps``.  The arithmetic goes through ``xp`` only, so the inputs
    may be host floats (``xp=numpy``) or device arrays.

    Parameters
    ----------
    y_prev, y_curr, y_next : array_like or float
        Samples at ``k-1``, ``k`` and ``k+1``.
    xp : module
        Array module (``numpy``/``cupy``) providing ``log``, ``maximum``,
        ``abs``, ``where``, ``ones_like``, ``zeros_like``, ``clip``.
    log : bool, default False
        Fit ``log(max(y, log_eps))``, the usual fit for spectral magnitudes,
        instead of ``y``.
    log_eps : float, default 1e-300
        Floor applied before the logarithm, with ``log=True`` only.
    denom_eps : float, optional
        Denominator magnitude below which the fit is degenerate and returns
        0.  Defaults to ``1e-30`` with ``log=True`` and ``1e-15`` without.

    Returns
    -------
    delta : ndarray or float
        Offset from the center sample in ``[-0.5, 0.5]``, the type of the
        inputs.
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


# -----------------------------------------------------------------------------
# CORRELATION-DOMAIN ESTIMATION (array-only)
# -----------------------------------------------------------------------------
# estimate_fractional_delay operates on a correlation array (e.g. from
# cross_correlate_fft), not on raw IQ samples or any Signal field, so it is
# not Signal-aware.


def estimate_fractional_delay(
    correlation: ArrayType,
    peak_indices: ArrayType,
    *,
    dft_upsample: int = 1,
    fit: str = "log-parabolic",
) -> ArrayType:
    """Estimate the sub-sample offset of correlation peaks.

    Fits a parabola (or, on the logarithm, a Gaussian) through the three
    samples around each integer peak.  Complex input is rotated so that the
    peak lies on the real axis before the fit, which keeps the peak's shape
    (a fit to ``|R|`` distorts it).

    With ``dft_upsample > 1`` the correlation in a 33-sample window around
    the peak is first interpolated onto a finer grid by DFT zero-padding,
    which reduces the bias of the three-point fit.  Peaks closer than 16
    samples to an end of the record fall back to the plain fit.

    Parameters
    ----------
    correlation : array_like
        Correlation, complex or real magnitude, ``(N,)`` or ``(C, N)``.
    peak_indices : array_like
        Integer peak positions, scalar or ``(C,)``.
    dft_upsample : int, default 1
        Interpolation factor of the DFT zero-padding; 1 fits the samples
        directly.
    fit : {'parabolic', 'log-parabolic'}, default 'log-parabolic'
        Peak fit:

        - ``'parabolic'``: a parabola through the samples.
        - ``'log-parabolic'``: a parabola through their logarithms, a
          Gaussian fit, usually more accurate for band-limited pulses.

    Returns
    -------
    ndarray
        Offset of each peak from ``peak_indices`` in samples, 0-d for a scalar
        ``peak_indices``, else ``(C,)``.  In ``[-0.5, 0.5]`` with
        ``dft_upsample=1``; with upsampling, the offset of the interpolated
        peak.  0 for a peak on the first or last sample.
    """
    if fit not in ("parabolic", "log-parabolic"):
        raise ValueError(f"fit must be 'parabolic' or 'log-parabolic', got {fit!r}.")
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

    def _calculate_mu(
        r_prev: ArrayType, r_curr: ArrayType, r_next: ArrayType, xp: Any, fit: str
    ) -> ArrayType:
        if xp.iscomplexobj(r_curr):
            phase = xp.exp(-1j * xp.angle(r_curr))
            alpha = (r_prev * phase).real
            beta = (r_curr * phase).real
            gamma = (r_next * phase).real
        else:
            alpha = r_prev
            beta = r_curr
            gamma = r_next

        # Same three-point (log-)parabolic fit as frequency.py's peak
        # estimators (_parabolic_peak_offset), just in the
        # phase-rotated-to-real-axis coordinate used here.  denom_eps=5e-13
        # exactly reproduces this function's original degeneracy threshold,
        # which was checked against 2*(alpha - 2*beta + gamma) rather than
        # (alpha - 2*beta + gamma) directly.
        return _parabolic_peak_offset(
            alpha,
            beta,
            gamma,
            xp,
            log=(fit == "log-parabolic"),
            log_eps=1e-12,
            denom_eps=5e-13,
        )

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

            specs = _fft.fft(windows, axis=-1)
            pos_limit = (_DFT_WINDOW + 1) // 2
            neg_len = _DFT_WINDOW - pos_limit
            target_len = _DFT_WINDOW * dft_upsample
            padded_specs = xp.zeros((len(valid_indices), target_len), dtype=specs.dtype)
            padded_specs[:, :pos_limit] = specs[:, :pos_limit]
            padded_specs[:, -neg_len:] = specs[:, pos_limit:]

            upsampled = _fft.ifft(padded_specs, axis=-1) * dft_upsample
            up_mag = xp.abs(upsampled)
            k_up = xp.argmax(up_mag, axis=-1)
            k_up_safe = xp.clip(k_up, 1, target_len - 2)

            row_idx = xp.arange(len(valid_indices))
            r_prev = upsampled[row_idx, k_up_safe - 1]
            r_curr = upsampled[row_idx, k_up_safe]
            r_next = upsampled[row_idx, k_up_safe + 1]

            mu_up = _calculate_mu(r_prev, r_curr, r_next, xp, fit)

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
    # would wrongly overwrite DFT results for on-center correlation peaks.
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

        mu_std = _calculate_mu(r_prev, r_curr, r_next, xp, fit)
        mu_std = xp.where(interior_all, mu_std, xp.zeros_like(mu_std))

        if dft_upsample == 1:
            mu = mu_std
        else:
            # Merge: DFT result for interior channels, standard for edge channels
            mu = xp.where(interior_mask, mu, mu_std)

    if scalar_input:
        return mu[0]
    return mu


# -----------------------------------------------------------------------------
# TIMING SYNCHRONIZATION (Signal-aware)
# -----------------------------------------------------------------------------
# fft_fractional_delay / correct_timing rewrap to a Signal (pass-through);
# estimate_timing unwraps the input but returns raw (integer, fractional)
# offset arrays.


def fft_fractional_delay(samples: S, *, delay: float | ArrayType) -> S:
    """Delay samples by a fractional number of samples in the frequency domain.

    The delay is a linear phase on the spectrum, ideal (periodic sinc)
    interpolation for band-limited signals: unlike polynomial interpolators
    (Farrow), it keeps the power and the whole band.  The record is treated
    as periodic, so samples shifted past one end reappear at the other.

    Parameters
    ----------
    samples : array_like or Signal
        Input samples, ``(N,)`` or ``(C, N)``.  A :class:`Signal` returns a
        new delayed :class:`Signal`.
    delay : float or array_like
        Delay in samples (any real value); positive shifts the signal later.
        A scalar applies to all channels, a ``(C,)`` array per channel.

    Returns
    -------
    ndarray or Signal
        Delayed samples, same shape and dtype as the input (real for real
        input).

    Notes
    -----
    ``Y[k] = X[k] · exp(-j 2π k · delay / N)`` with ``k`` the signed FFT bin
    index.
    """
    signal_adapter = adapt_signal(samples, function_name="fft_fractional_delay()")
    x, xp, _ = dispatch(signal_adapter.array)
    x, was_1d = as_2d(x, name="samples")

    C, N = x.shape

    # Convert delay to array
    if isinstance(delay, (int, float)):
        delay_arr = xp.full(C, delay, dtype=xp.float64)
    else:
        delay_arr = xp.asarray(delay, dtype=xp.float64)
        if delay_arr.ndim == 0:
            delay_arr = delay_arr[None]

    # FFT
    spec = _fft.fft(x, axis=-1)

    # Frequency axis: normalized frequencies in cycles/sample
    freqs = xp.fft.fftfreq(N, d=1.0)

    # spec · exp(-j 2π f delay): positive delay shifts the signal right.  The
    # phase in float64, the rotation in the spectrum's precision.
    spec_delayed = _derotate(spec, 2.0 * np.pi * freqs[None, :] * delay_arr[:, None])

    # IFFT
    result = _fft.ifft(spec_delayed, axis=-1)

    # Dtype restoration: mirrors the impairments.py pattern.
    if not xp.iscomplexobj(x):
        # Real input: fractional delay is a real-valued operation
        result = result.real
    elif result.dtype != x.dtype:
        # Complex input: ifft may return complex128 from complex64 input
        result = result.astype(x.dtype)

    return signal_adapter.wrap_samples(restore_1d(was_1d, result))


@dataclass(frozen=True)
class TimingEstimate:
    """Result of :func:`estimate_timing`.

    Per-channel fields follow the rank rule: 0-d for ``(N,)`` input and
    ``(C,)`` for ``(C, N)`` input, on the input's device.

    Attributes
    ----------
    integer : ndarray
        Sample index where the template starts (``int64``).
    fractional : ndarray
        Sub-sample offset of the correlation peak (see
        :func:`estimate_fractional_delay`; in ``[-0.5, 0.5]`` without DFT
        upsampling).
    metric : ndarray
        Peak-to-mean ratio of ``|correlation|`` (the detection metric).
    coherence : ndarray
        Peak ``|correlation|`` over the template and local signal energy, in
        ``[0, 1]``; low values with a high ``metric`` point to a frequency
        offset or dispersion.
    correlation : ndarray
        Complex correlation at non-negative lags ``0 .. N_search - 1`` of the
        searched window, ``(N_search,)`` or ``(C, N_search)``; the data
        ``plot_timing_correlation`` draws.
    search_start : int
        First sample of the searched window: lag ``k`` is sample
        ``search_start + k``.
    """

    integer: ArrayType
    fractional: ArrayType
    metric: ArrayType
    coherence: ArrayType
    correlation: ArrayType
    search_start: int = 0

    @property
    def value(self) -> ArrayType:
        """Total offset in samples, ``integer + fractional``."""
        return self.integer + self.fractional


def estimate_timing(
    samples: ArrayType | Signal,
    *,
    template: ArrayType | Preamble | None = None,
    threshold: float = 3.0,
    sps: float | None = None,
    pulse: Pulse | ArrayType | None = None,
    search_range: tuple[int, int] | None = None,
    dft_upsample: int = 1,
    fit: str = "log-parabolic",
) -> TimingEstimate:
    """Estimate the timing offset by cross-correlation with a template.

    Finds the integer start of the template and the sub-sample offset of the
    correlation peak.

    For a multi-stream template ``(C_tx, L)`` every stream is correlated with
    every receive channel and each channel takes its strongest peak, which
    keeps per-channel skew and tolerates swapped or mixed polarizations.

    Parameters
    ----------
    samples : array_like or Signal
        Received samples, ``(N,)`` or ``(C, N)``.
    template : array_like or Preamble, optional
        Known sequence to find.  An array (``(L,)`` or ``(C_tx, L)``) is used
        as is; a :class:`Preamble` is shaped at ``sps`` with ``pulse``.
        Defaults to the preamble of the Signal's frame.
    threshold : float, default 3.0
        Minimum peak-to-mean ratio of ``|correlation|``.  A channel below it
        logs a warning; the call raises only when every channel is below.
    sps : float, optional
        Samples per symbol, needed to shape a Preamble.  Taken from the
        Signal; a value that disagrees with it raises.
    pulse : Pulse or array_like, optional
        Pulse that shapes a Preamble template.  Defaults to the Signal's
        ``pulse``; ``None`` leaves it unshaped (zero-stuffed).
    search_range : (int, int), optional
        ``(start, stop)`` samples to search; defaults to the whole record.
    dft_upsample : int, default 1
        DFT interpolation factor for the fractional fit.
    fit : {'parabolic', 'log-parabolic'}, default 'log-parabolic'
        Peak fit for the fractional offset.

    Returns
    -------
    TimingEstimate
        ``integer`` is the sample where the template starts.

    Raises
    ------
    ValueError
        If no template is available or no channel's peak reaches
        ``threshold``.
    """
    signal_adapter = adapt_signal(samples, function_name="estimate_timing()")
    sig = signal_adapter.signal
    if template is None and sig is not None and sig.frame is not None:
        template = getattr(sig.frame, "preamble", None)
    if template is None:
        raise ValueError(
            "estimate_timing() needs a template: pass template= (array or "
            "Preamble) or a Signal whose frame has a preamble."
        )
    sig_array, xp, _ = dispatch(signal_adapter.array)
    sig_array, was_1d = as_2d(sig_array, name="samples")

    if isinstance(template, Preamble):
        if sig is None and sps is None:
            raise ValueError("SPS must be provided when using a Preamble template.")
        sps_int = require_integer_sps(
            signal_adapter.resolve_fact("sps", sps), "estimate_timing()"
        )
        pulse = signal_adapter.resolve_choice("pulse", pulse)
        ref_waveform = xp.asarray(
            template.to_signal(sps=sps_int, symbol_rate=1.0, pulse=pulse).samples
        )
    else:
        ref_waveform = xp.asarray(template)
    if ref_waveform.ndim == 1:
        ref_waveform = ref_waveform[None, :]
    # Ensure the template is on the same device as the signal.
    ref_waveform = to_device(ref_waveform, "cpu" if xp is np else "gpu")

    num_sig_ch = sig_array.shape[0]

    # Apply search range
    offset = 0
    if search_range is not None:
        start, end = search_range
        sig_processing = sig_array[:, start:end]
        offset = int(start)
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

    # === Fractional Timing (Parabolic Interpolation) ===
    fractional_offsets = estimate_fractional_delay(
        corr, peak_indices, dft_upsample=dft_upsample, fit=fit
    )

    if logger.isEnabledFor(logging.INFO):
        # Three .tolist() host syncs, only for this summary line.
        logger.info(
            "Timing estimated. Integer: %s, Fractional: %s, Metrics: %s",
            integer_offsets.tolist(),
            fractional_offsets.tolist(),
            metrics.tolist(),
        )

    integer_offsets, fractional_offsets, metrics, coherence, corr = restore_1d(
        was_1d, integer_offsets, fractional_offsets, metrics, coherence, corr
    )
    return TimingEstimate(
        integer=integer_offsets.astype(xp.int64),
        fractional=fractional_offsets,
        metric=metrics,
        coherence=coherence,
        correlation=corr,
        search_start=offset,
    )


def correct_timing(
    samples: S,
    how: TimingEstimate | float | ArrayType,
    *,
    mode: str = "circular",
) -> S:
    """Remove a timing offset: integer shift, then FFT fractional delay.

    Parameters
    ----------
    samples : array_like or Signal
        Input samples, ``(N,)`` or ``(C, N)``.
    how : TimingEstimate, float or array_like
        A :class:`TimingEstimate`, or the offset in samples (scalar or
        ``(C,)``), which is split into its nearest integer and a fraction
        in ``[-0.5, 0.5)``.  The input's sample at the offset becomes
        sample 0.
    mode : {'circular', 'zero', 'slice'}, default 'circular'
        How to handle boundary samples after the integer shift:

        - ``'circular'``: wrap around (``roll``); same length.  For periodic
          signals; not for bursts.
        - ``'zero'``: shift left and zero-fill the tail; same length.
        - ``'slice'``: drop the leading samples.  With per-channel offsets
          the output has ``N - max(offset)`` samples so all channels share the
          overlap.  A Signal's reference keeps the symbol periods left.

    Returns
    -------
    ndarray or Signal
        Corrected samples; shorter for ``'slice'``, which also logs a
        warning.

    Raises
    ------
    ValueError
        On an unknown ``mode``.

    Notes
    -----
    The template is not stripped: sample 0 is its first sample.  For
    ``mode='slice'`` the fractional delay is applied to the full buffer
    before slicing, so the FFT wrap-around falls in the discarded part.
    """

    if mode not in ("circular", "zero", "slice"):
        raise ValueError(
            f"Unknown mode {mode!r}. Choose 'circular', 'zero', or 'slice'."
        )
    signal_adapter = adapt_signal(samples, function_name="correct_timing()")
    x, xp, _ = dispatch(signal_adapter.array)
    x, was_1d = as_2d(x, name="samples")
    if isinstance(how, TimingEstimate):
        integer_offset = xp.asarray(how.integer)
        fractional_offset = xp.asarray(how.fractional)
    else:
        total = xp.asarray(how, dtype=xp.float64)
        integer_offset = xp.floor(total + 0.5).astype(xp.int64)
        fractional_offset = total - integer_offset

    num_ch = x.shape[0]
    N = x.shape[-1]

    # Pre-evaluate the fractional-correction flag so mode='slice' can apply the
    # delay before the slice. Same logic that used to live just before the
    # post-integer fractional call below - relocated, not changed.
    apply_frac = bool(xp.any(xp.abs(fractional_offset) > 1e-9))

    # mode='slice': apply fractional delay on the *full* pre-slice buffer.
    # fft_fractional_delay treats its input as circular; applying it after the
    # slice would wrap the slice's trailing edge back into its new sample 0 -
    # exactly the frame boundary we just aligned to. Applying first puts the
    # wrap at the physical buffer ends, the leading one of which is then
    # discarded by the slice; only the tail of the slice carries any residual
    # ~sinc-tail artifact, well away from the equalizer training start.
    if mode == "slice" and apply_frac:
        x = fft_fractional_delay(x, delay=-fractional_offset)

    # === Integer correction (integer shift) ===

    if integer_offset.ndim == 0:
        # --- Scalar: same shift for all channels ---
        shift = int(integer_offset)
        if mode == "circular":
            x = xp.roll(x, -shift, axis=-1)
        elif mode == "zero":
            result = xp.zeros_like(x)
            if shift > 0:
                result[..., : N - shift] = x[..., shift:]
            elif shift < 0:
                result[..., -shift:] = x[..., : N + shift]
            x = result
        elif mode == "slice":
            x = x[..., shift:]

    else:
        # --- Per-channel: vectorized gather (avoids one GPU->CPU sync per channel) ---
        integer_shift = integer_offset.astype(xp.int64)  # (C,) on device
        col_base = xp.arange(N, dtype=xp.int64)[None, :]  # (1, N)
        row_idx = xp.arange(num_ch)[:, None]  # (C, 1)

        if mode == "circular":
            col_idx = (col_base + integer_shift[:, None]) % N  # (C, N)
            x = x[row_idx, col_idx]

        elif mode == "zero":
            col_raw = col_base + integer_shift[:, None]  # (C, N)
            gathered = x[row_idx, xp.clip(col_raw, 0, N - 1)]
            x = xp.where(col_raw < N, gathered, xp.zeros_like(gathered))

        elif mode == "slice":
            # Align all channels to common overlap: N - max(offset) samples
            max_shift = int(xp.max(integer_shift))  # one GPU sync
            common_len = N - max_shift
            col_idx_s = (
                xp.arange(common_len, dtype=xp.int64)[None, :] + integer_shift[:, None]
            )  # (C, common_len)
            x = x[row_idx, col_idx_s]

    # === Fractional correction (via FFT) ===
    # For mode='slice' this was already applied above on the full pre-slice
    # buffer; here we only handle 'circular' / 'zero', whose output length
    # equals the input length so the wrap location is unchanged either way.
    if apply_frac and mode != "slice":
        x = fft_fractional_delay(x, delay=-fractional_offset)

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

    out = restore_1d(was_1d, x)
    sig = signal_adapter.signal
    if mode == "slice" and sig is not None and sig.reference is not None:
        # Alignment invariant: keep only the symbol periods still present.
        num_symbols = int(out.shape[-1] // sig.sps)
        return signal_adapter.wrap_samples(
            out, reference=sig.reference.head(num_symbols)
        )
    return signal_adapter.wrap_samples(out)
