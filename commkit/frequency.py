"""
Frequency offset estimation and correction utilities.

This module provides routines for carrier frequency offset (FOE) estimation
and correction, including blind spectral methods, multi-lag autocorrelation,
pilot-aided estimation, blockwise time-varying FOE, and constant-offset
complex mixing correction.
"""

from collections.abc import Callable

import numpy as np

from .backend import ArrayType, dispatch, to_device
from .core._signal_adapter import adapt_signal
from .core.signal import Signal
from .helpers import (
    _parabolic_peak_offset,
    as_2d,
    broadcast_channels,
    linear_trend_slope,
    restore_1d,
)
from .logger import logger


def _modulation_power_m(modulation: str, order: int) -> int:
    """
    Return the exponent M for M-th power spectral methods.

    Parameters
    ----------
    modulation : str
        Modulation type string (case-insensitive).
    order : int
        Modulation order.

    Returns
    -------
    int
        M = ``order`` for PSK; M = 4 for QAM and other schemes.

    Notes
    -----
    M=4 is exact only for **square** QAM constellations (4, 16, 64, 256, ...)
    which have perfect 4-fold rotational symmetry.  For cross-QAM (32, 128,
    512-QAM) the 4th power leaves residual modulation spurs; a warning is
    emitted.  For PAM/ASK the M-th power law does not apply; prefer
    pilot-aided or data-aided estimators.
    """
    mod = modulation.lower()
    if "psk" in mod:
        if order > 4:
            logger.warning(
                "%s-PSK: M=%sth-power raises noise variance by M² - "
                "VV/FOE reliability degrades severely for order > 4. "
                "Prefer BPS or pilot-aided CPR for 8-PSK and higher.",
                order,
                order,
            )
        return order  # M-th power exactly removes M-PSK modulation

    if "qam" in mod:
        side = int(order**0.5)
        if side * side == order:
            return 4  # Square QAM: 4-fold rotational symmetry, 4th power is exact
        # Cross-QAM (32, 128, 512-QAM): 4-fold symmetry is only approximate
        logger.warning(
            "%s-QAM is not square - 4th-power FOE/CPR will have residual "
            "modulation spurs. Prefer pilot-aided or data-aided estimation.",
            order,
        )
        return 4

    # PAM, ASK, or unrecognised scheme
    logger.warning(
        "Modulation '%s' (order %s): M=4 is a heuristic. 4th-power methods "
        "are unreliable for non-QAM/PSK formats. Prefer pilot-aided or "
        "data-aided estimation.",
        modulation,
        order,
    )
    return 4


# Lazy-compiled Numba kernel for the M&M iterative bootstrap.
_NUMBA_MM: dict = {}


def _get_numba_mm_bootstrap():
    """JIT-compile and cache the Numba M&M iterative bootstrap kernel.

    Returns
    -------
    callable
        Numba-compiled ``_mm_bootstrap_loop``.
    """
    if "mm" not in _NUMBA_MM:
        import numba

        @numba.njit(cache=True, fastmath=True, nogil=True)
        def _mm_bootstrap_loop(theta, amp, M_val, fs):
            """Iterative Mengali-Morelli bootstrap compiled to machine code.

            Predicts each lag's phase from the running weighted frequency
            estimate accumulated from all previous lags, then folds it
            into the weighted sum.  Sequential data dependency prevents
            vectorisation; Numba removes Python-interpreter overhead.

            Parameters
            ----------
            theta : (L,) float64
                Wrapped phase of R[m] at each lag (output of np.angle).
            amp : (L,) float64
                Magnitude |R[m]| at each lag (output of np.abs).
            M_val : float64
                Modulation power (1 for data-aided/generic, order for PSK,
                4 for QAM).
            fs : float64
                Sampling rate in Hz.

            Returns
            -------
            float64
                Estimated frequency offset in Hz.
            """
            two_pi = 2.0 * np.pi
            L = len(theta)

            # Lag 1 initialisation (m=1, m²=1)
            Theta_0 = theta[0]
            w0 = amp[0] * amp[0]  # m² · |R|² at m=1
            f_hat = Theta_0 * fs / (two_pi * M_val)
            w_sum = w0 if w0 > 1e-30 else 1e-30
            wf_sum = w_sum * f_hat

            for m_idx in range(1, L):
                m_val = float(m_idx + 1)
                predicted = two_pi * f_hat * M_val * m_val / fs
                diff = predicted - theta[m_idx]
                # round() is the C math round - unboxed, no Python overhead
                correction = round(diff / two_pi)
                Theta_m = theta[m_idx] + two_pi * correction
                f_m = Theta_m * fs / (two_pi * m_val * M_val)
                w_m = m_val * m_val * amp[m_idx] * amp[m_idx]
                w_sum += w_m
                wf_sum += w_m * f_m
                f_hat = wf_sum / w_sum

            return f_hat

        _NUMBA_MM["mm"] = _mm_bootstrap_loop

    return _NUMBA_MM["mm"]


# -----------------------------------------------------------------------------
# FREQUENCY OFFSET ESTIMATION (Signal-aware)
# -----------------------------------------------------------------------------
# estimate_frequency_offset_mth_power / mengali_morelli / pilot_symbols and
# find_bias_tone unwrap the input but return a float/ndarray estimate, never
# a Signal (see CLAUDE.md, "Signal-Awareness", rule R5).


def estimate_frequency_offset_mth_power(
    samples: ArrayType | Signal,
    sampling_rate: float | None = None,
    modulation: str | None = None,
    order: int | None = None,
    search_range: tuple[float, float] | None = None,
    nfft: int | None = None,
    interpolation: str = "jacobsen",
    combine_channels: bool = False,
    debug_plot: bool = False,
) -> float | np.ndarray:
    """
    Estimates frequency offset using the M-th power law (nonlinear spectral method).

    Raises the signal to the M-th power to eliminate PSK/QAM modulation,
    producing a tone at M·Δf.  A spectral peak search with sub-bin
    interpolation gives frequency resolution well below the FFT bin width.

    Parameters
    ----------
    samples : array_like or Signal
        Complex IQ samples. Shape: (N,) or (C, N). For MIMO, channel
        magnitude spectra are summed for robust peak detection, then
        per-channel sub-bin interpolation is applied at the shared peak bin.
        For :class:`Signal` input, the signal's own ``sampling_rate`` /
        ``mod_scheme`` / ``mod_order`` take priority over the corresponding
        arguments below; a set ``modulation``/``order`` argument is used only
        as a fallback when the signal's metadata is unset, with a warning.
    sampling_rate : float, optional
        Sampling rate in Hz.  Required for array input; ignored for
        :class:`Signal` input, which always uses the signal's own
        ``sampling_rate``.
    modulation : str, optional
        Modulation scheme (case-insensitive): 'psk', 'qam', 'bpsk', etc.
        Required for array input; for :class:`Signal` input, used only as a
        fallback when the signal's ``mod_scheme`` is unset.
    order : int, optional
        Modulation order (2, 4, 16, 64, ...).  Required for array input;
        for :class:`Signal` input, used only as a fallback when the signal's
        ``mod_order`` is unset.
    search_range : tuple of float, optional
        ``(f_min, f_max)`` in Hz to limit the frequency offset search.
        The spectral search is mapped to ``[M·f_min, M·f_max]``.
        Default: full spectrum.
    nfft : int, optional
        FFT size. Default: next power of 2 ≥ len(samples).
    interpolation : {'jacobsen', 'parabolic'}, default 'jacobsen'
        Sub-bin interpolation method.

        * ``'jacobsen'``: uses complex FFT values around the peak bin -
          corrects rectangular-window sinc bias analytically.  More
          accurate than parabolic for short observation windows.
        * ``'parabolic'``: classic parabolic fit on FFT magnitudes.
    combine_channels : bool, default False
        For MIMO inputs (C > 1): if ``True``, return a single
        magnitude-weighted mean estimate as ``float``; if ``False``
        (default), return per-channel estimates as ``np.ndarray`` of
        shape ``(C,)``.  SISO inputs always return ``float``.
    debug_plot : bool, default False
        If ``True``, opens a diagnostic figure showing the M-th power
        spectrum per channel with the detected peak and sub-bin result.

    Returns
    -------
    float or np.ndarray
        Estimated frequency offset in Hz. For SISO or when
        ``combine_channels=True``: scalar ``float``. For MIMO with
        ``combine_channels=False``: ``np.ndarray`` of shape ``(C,)``
        with one estimate per channel.

    Notes
    -----
    M is determined by the modulation type:

    - PSK / BPSK: M = ``order`` (e.g. 4 for QPSK, 8 for 8-PSK).
    - PSK / BPSK: M = order (e.g. 4 for QPSK, 8 for 8-PSK).
    - QAM: M = 4 (the 4th power removes quadrature phase; residual
      amplitude modulation is suppressed by subtracting the per-channel
      mean of signal^M before the FFT).

    Lock range: [-fs/(2M), fs/(2M)]. For QPSK at 1 GHz -> +/- 125 MHz.
    Use search_range to reduce false-peak probability.

    Jacobsen interpolation:

    delta = Re[ (X[k-1] - X[k+1]) / (2*X[k] - X[k-1] - X[k+1]) ]

    where X are complex FFT values at the peak bin k and its
    neighbours. Unlike the parabolic fit (which operates on magnitudes
    and has a sinc-function bias for small NFFT), the Jacobsen estimator
    is unbiased for a rectangular window.
    """
    signal_adapter = adapt_signal(
        samples, function_name="estimate_frequency_offset_mth_power()"
    )
    samples = signal_adapter.array
    sampling_rate = signal_adapter.resolve_required("sampling_rate", sampling_rate)
    modulation = signal_adapter.resolve_optional("mod_scheme", modulation)
    order = signal_adapter.resolve_optional("mod_order", order)
    if modulation is None or order is None:
        raise ValueError(
            "estimate_frequency_offset_mth_power() requires modulation and order "
            "for array input."
        )

    samples, xp, _ = dispatch(samples)
    samples, was_1d = as_2d(samples, name="samples")
    C, N = samples.shape

    if N < 8:
        raise ValueError(
            f"Signal too short for spectral FOE (N={N}). Minimum 8 samples required."
        )

    M = _modulation_power_m(modulation, order)
    mod_lower = modulation.lower()

    if "qam" in mod_lower and order >= 64:
        logger.warning(
            "QAM order %s: M-th power lock range is ±fs/8 = ±%.0f Hz and AM "
            "spectral spreading reduces accuracy for high-order "
            "constellations. Consider estimate_frequency_offset_mengali_"
            "morelli (blind) or find_bias_tone (pilot tone).",
            order,
            sampling_rate / 8,
        )

    if nfft is None:
        nfft = 1 << int(np.ceil(np.log2(N)))  # guaranteed Python int

    # Promote for numerical accuracy during power computation: (C, N)
    s_c = samples.astype(
        xp.complex128 if samples.dtype == xp.complex64 else samples.dtype
    )

    # M-th power removes modulation -> tone at M·Δf; shape stays (C, N)
    x_M = s_c**M

    # For QAM: subtract per-channel mean to suppress DC spike from amplitude residuals
    if M == 4 and "qam" in mod_lower:
        x_M = x_M - xp.mean(x_M, axis=-1, keepdims=True)

    # Batched FFT across all channels: single kernel call on GPU -> (C, nfft)
    X_M = xp.fft.fft(x_M, n=nfft, axis=-1)
    # CPU freq array; device version allocated only when search_range masking is needed.
    freqs_np = np.fft.fftfreq(nfft, d=1.0 / sampling_rate)  # (nfft,) - always on CPU

    mag = xp.abs(X_M)  # (C, nfft)

    # Restrict search to [M·f_min, M·f_max] when search_range is given;
    # do this BEFORE zeroing DC so the mask operates on the raw spectrum.
    if search_range is not None:
        tone_lo = M * min(search_range)
        tone_hi = M * max(search_range)
        freqs = xp.asarray(freqs_np)  # device version only when mask is needed
        mask = (freqs >= tone_lo) & (freqs <= tone_hi)  # (nfft,)
        if not bool(xp.any(mask)):
            raise ValueError(
                f"search_range {search_range} Hz produces an empty search "
                f"window in the M={M} scaled spectrum."
            )
        mag = xp.where(mask[None, :], mag, xp.zeros_like(mag))

    # Zero DC after masking (Δf = 0 is degenerate; also removes any residual
    # QAM DC not caught by mean subtraction above).
    mag[:, 0] = 0.0

    # For MIMO, sum spectra across channels for coherent accumulation.
    # For SISO this is a no-op (C=1).
    mag_combined = xp.sum(mag, axis=0)  # (nfft,)

    k_peak = int(xp.argmax(mag_combined))
    k_prev = (k_peak - 1) % nfft  # circular - correct at Nyquist edge (k=nfft-1)
    k_next = (k_peak + 1) % nfft

    # Vectorised sub-bin interpolation across all C channels at the shared peak bin.
    # Extract 3 neighbouring bins for all channels at once -> 1 device->host transfer
    # instead of 3C individual scalar round-trips.
    X_km1 = X_M[:, k_prev]  # (C,) complex, on device
    X_k0 = X_M[:, k_peak]
    X_kp1 = X_M[:, k_next]

    if interpolation == "jacobsen":
        d_vec = 2.0 * X_k0 - X_km1 - X_kp1  # (C,) complex
        safe_d = xp.where(xp.abs(d_vec) > 1e-30, d_vec, xp.ones_like(d_vec))
        mu_raw = ((X_km1 - X_kp1) / safe_d).real  # (C,) float
        mu_vec = xp.clip(
            xp.where(xp.abs(d_vec) > 1e-30, mu_raw, xp.zeros_like(mu_raw)),
            -0.5,
            0.5,
        )
    else:  # parabolic - use the already-computed magnitude array
        a_vec = mag[:, k_prev]  # (C,) float
        b_vec = mag[:, k_peak]
        cc_vec = mag[:, k_next]
        mu_vec = _parabolic_peak_offset(a_vec, b_vec, cc_vec, xp, log=False)

    mu_np = to_device(mu_vec, "cpu")  # one transfer: (C,) floats
    f_per_ch = [
        float((freqs_np[k_peak] + float(mu_np[c]) * (sampling_rate / nfft)) / M)
        for c in range(C)
    ]

    logger.info(
        "FOE (M-th power, M=%s): %s Hz [nfft=%s, interp=%s, search_range=%s]",
        M,
        [f"{f:.2f}" for f in f_per_ch],
        nfft,
        interpolation,
        search_range,
    )

    if debug_plot:
        from . import plotting as _plotting

        _plotting.plot_frequency_offset_spectrum(
            mag_spectrum=to_device(mag, "cpu"),
            freqs=freqs_np,
            M=M,
            k_peaks=np.array([k_peak] * C),
            f_estimates=f_per_ch,
            search_range=search_range,
            show=True,
        )

    if was_1d:
        return float(f_per_ch[0])
    if combine_channels:
        weights = to_device(mag[:, k_peak], "cpu").tolist()  # one batch transfer
        combined = float(np.average(f_per_ch, weights=weights))
        logger.info(
            "FOE (M-th power, M=%s): combined=%.2f Hz (magnitude-weighted "
            "mean of %s channels)",
            M,
            combined,
            C,
        )
        return combined
    return np.array(f_per_ch)


def estimate_frequency_offset_mengali_morelli(
    samples: ArrayType | Signal,
    sampling_rate: float | None = None,
    modulation: str | None = None,
    order: int | None = None,
    ref_signal: ArrayType | None = None,
    max_lag: int | None = None,
    combine_channels: bool = False,
    debug_plot: bool = False,
) -> float | np.ndarray:
    """
    Estimates frequency offset via the Mengali-Morelli multi-lag autocorrelation.

    Uses multiple autocorrelation lags m = 1 ... L combined with MVUE weights
    to extend the lock range to the full Nyquist interval [-fs/2, fs/2]
    while remaining Cramér-Rao-efficient.  This is the recommended estimator
    when the frequency offset may be large (exceeding the Kay / differential
    lock range of fs/(2M)) and a pilot or data-aided reference is
    available for pre-processing.

    Three input modes:

    1. **Data-aided** (``ref_signal`` provided): derotates samples by the
       known reference before estimating.
    2. **Blind M-PSK/QAM** (``modulation`` + ``order``): applies M-th power
       pre-processing.  Lock range remains [-fs/2, fs/2] for all lags
       after bootstrap unwrapping from lag 1.
    3. **Generic blind** (no arguments): assumes a constant-envelope signal.

    Parameters
    ----------
    samples : array_like or Signal
        Complex IQ samples. Shape: (N,) or (C, N).  For :class:`Signal`
        input, the signal's own ``sampling_rate``/``mod_scheme``/``mod_order``
        take priority over the corresponding arguments below.
    sampling_rate : float, optional
        Sampling rate in Hz.  Required for array input; ignored for
        :class:`Signal` input, which always uses the signal's own
        ``sampling_rate``.
    modulation : str, optional
        Modulation type (case-insensitive). Required for blind M-th power mode.
        Ignored when ``ref_signal`` is provided.  For :class:`Signal` input,
        used only as a fallback when the signal's ``mod_scheme`` is unset.
    order : int, optional
        Modulation order. Required with ``modulation`` for blind mode.  For
        :class:`Signal` input, used only as a fallback when the signal's
        ``mod_order`` is unset.
    ref_signal : array_like, optional
        Ideal transmitted waveform used to derotate ``samples`` before
        autocorrelation.  Computes ``y[n] = samples[n] * conj(ref[n])``,
        cancelling the data modulation and leaving a tone at delta_f.
        Must be sample-aligned with ``samples`` (call ``estimate_timing``
        first).  Shape: ``(N,)``, ``(1, N)``, or ``(C, N)``.
    max_lag : int, optional
        Maximum autocorrelation lag L.  Default: ``N // 4``, clamped to
        ``[1, N // 2]``.  Increasing L improves noise averaging at the cost
        of using shorter sub-sequences for each lag.
    combine_channels : bool, default False
        For MIMO inputs (C > 1): if ``True``, return a single
        autocorrelation-energy-weighted mean estimate as ``float``; if
        ``False`` (default), return per-channel estimates as
        ``np.ndarray`` of shape ``(C,)``.
        SISO inputs always return ``float``.
    debug_plot : bool, default False
        If ``True``, opens a diagnostic figure showing per-channel
        autocorrelation magnitude ``|R[m]|`` and wrapped phase ``∠R[m]``
        vs lag, with the expected phase ramp overlaid.

    Returns
    -------
    float or np.ndarray
        Estimated frequency offset in Hz. For SISO or when
        ``combine_channels=True``: scalar ``float``. For MIMO with
        ``combine_channels=False``: ``np.ndarray`` of shape ``(C,)``
        with one estimate per channel.

    Notes
    -----
    Computes the unbiased autocorrelation R[m] at lags 1 ... L via the
    Wiener-Khinchin theorem (2 FFTs), bootstraps the phase from lag 1 to
    resolve 2*pi ambiguities at higher lags, then combines per-lag estimates
    with m^2 * |R[m]|^2 MVUE weights.

    Lock range: [-fs/(2M), fs/(2M)] for blind M-th power mode;
    [-fs/2, fs/2] for data-aided or generic blind mode.
    """
    signal_adapter = adapt_signal(
        samples, function_name="estimate_frequency_offset_mengali_morelli()"
    )
    samples = signal_adapter.array
    sampling_rate = signal_adapter.resolve_required("sampling_rate", sampling_rate)
    modulation = signal_adapter.resolve_optional("mod_scheme", modulation)
    order = signal_adapter.resolve_optional("mod_order", order)

    samples, xp, _ = dispatch(samples)
    samples, was_1d = as_2d(samples, name="samples")
    C, N = samples.shape

    # Pre-process: data-aided, blind M-th power, or generic blind
    if ref_signal is not None:
        ref, _, _ = dispatch(ref_signal)
        ref = broadcast_channels(xp.asarray(ref), C, xp, name="ref_signal")
        y = samples * xp.conj(ref)  # derotate -> complex tone at Δf
        M = 1
    elif modulation is not None and order is not None:
        M = _modulation_power_m(modulation, order)
        y = samples**M  # removes PSK/QAM modulation
    else:
        y = samples
        M = 1

    # Choose max_lag L
    L = max_lag if max_lag is not None else N // 4
    L = max(1, min(L, N // 2))

    # Autocorrelation at all lags 1..L via the Wiener-Khinchin theorem.
    # IFFT(|FFT(y)|²)[l] = Σ_n conj(y[n]) · y[n+l]  (linear, not circular,
    # when zero-padded to nfft_r ≥ N+L).
    # This replaces a Python loop of L GPU kernel launches with 2 FFT calls,
    # keeping all heavy computation on the device (GPU or CPU backend).
    nfft_r = 1 << int(np.ceil(np.log2(N + L)))  # smallest power-of-2 ≥ N+L
    Y_r = xp.fft.fft(y, n=nfft_r, axis=-1)  # (C, nfft_r)
    R_all = xp.fft.ifft(Y_r * xp.conj(Y_r), axis=-1)  # (C, nfft_r) per-ch autocorr

    # Unbiased per-channel autocorrelation at lags 1..L: R_per_ch[c, m-1] = R[c, m] / (N-m)
    lags_xp = xp.arange(1, L + 1, dtype=xp.float64)  # (L,) on device
    R_per_ch = R_all[:, 1 : L + 1] / (N - lags_xp[None, :])  # (C, L)

    # Transfer R_per_ch (C, L) to CPU once; run the iterative bootstrap per channel.
    # The bootstrap is a sequential scan (each step depends on the previous),
    # so Numba on CPU beats GPU here. Single transfer avoids repeated device round-trips.
    R_per_ch_np = to_device(R_per_ch, "cpu")  # (C, L) complex128

    _mm_kernel = _get_numba_mm_bootstrap()
    f_per_ch = []
    for c_idx in range(C):
        R_c_np = R_per_ch_np[c_idx]  # (L,)
        theta_c = np.angle(R_c_np)
        amp_c = np.abs(R_c_np)
        f_per_ch.append(
            float(_mm_kernel(theta_c, amp_c, float(M), float(sampling_rate)))
        )

    mode_str = "data-aided" if ref_signal is not None else f"blind M={M}"
    logger.info(
        "FOE (Mengali-Morelli, %s): %s Hz [L=%s lags, N=%s]",
        mode_str,
        [f"{f:.2f}" for f in f_per_ch],
        L,
        N,
    )

    if debug_plot:
        from . import plotting as _plotting

        _plotting.plot_mm_autocorrelation(
            R_np=R_per_ch_np,
            f_est=f_per_ch,
            sampling_rate=sampling_rate,
            M=M,
            show=True,
        )

    if was_1d:
        return float(f_per_ch[0])
    if combine_channels:
        # Weight by total autocorrelation energy per channel
        weights = [float(np.sum(np.abs(R_per_ch_np[c]) ** 2)) for c in range(C)]
        combined = float(np.average(f_per_ch, weights=weights))
        logger.info(
            "FOE (Mengali-Morelli, %s): combined=%.2f Hz "
            "(autocorrelation-weighted mean of %s channels)",
            mode_str,
            combined,
            C,
        )
        return combined
    return np.array(f_per_ch)


def estimate_frequency_offset_pilot_symbols(
    samples: ArrayType | Signal,
    sampling_rate: float | None = None,
    pilot_indices: ArrayType | None = None,
    pilot_values: ArrayType | None = None,
    snr_weighted: bool = True,
    combine_channels: bool = False,
    debug_plot: bool = False,
) -> float | np.ndarray:
    """
    Estimates frequency offset from pilot symbols via phase slope fitting.

    Extracts the received phase at each pilot position, demodulates against
    the known pilot values to obtain the residual phase, unwraps the pilot
    phase sequence, then fits a (optionally SNR-weighted) least-squares line
    to the unwrapped phase as a function of pilot sample time.  The slope
    gives the frequency offset: Δf = slope / (2π).

    Parameters
    ----------
    samples : array_like or Signal
        Received complex samples. Shape: (N,) or (C, N).  For :class:`Signal`
        input, the signal's own ``sampling_rate`` always wins over a
        supplied ``sampling_rate``.
    sampling_rate : float, optional
        Sampling rate in Hz.  Required for array input; ignored for
        :class:`Signal` input, which always uses the signal's own
        ``sampling_rate``.
    pilot_indices : array_like of int
        Sample indices of pilot positions in increasing order. Shape: (P,).
        Must be unique and sorted.  Supports any pilot arrangement:

        * **Comb (scattered):** uniform grid, e.g. every 16th sample.
          Lock range determined by comb spacing.
        * **Block (contiguous cluster):** e.g. ``[0, 1, ..., L-1]``.
        * **Multi-block:** e.g. a front preamble and a mid-burst pilot
          cluster.  Lock range is set by the **largest gap** between any
          two consecutive pilot indices (see Notes).
    pilot_values : array_like
        Known transmitted pilot symbols at the corresponding indices.
        Shape: (P,) for shared pilots (broadcast to all MIMO channels),
        or (C, P) for per-channel pilots.
    snr_weighted : bool, default True
        If ``True``, weights each pilot by its received power ``|r|²``
        (SNR proxy) in the least-squares phase-slope fit.  This is the
        **WLSQ** estimator and is significantly more robust when pilot SNR
        varies across the block (e.g. due to PMD nulls or spectral ripple).
        Set to ``False`` for the standard unweighted OLS slope.
    combine_channels : bool, default False
        For MIMO inputs (C > 1): if ``True``, return a single
        pilot-power-weighted mean estimate as ``float``; if ``False``
        (default), return per-channel estimates as ``np.ndarray`` of
        shape ``(C,)``.
        SISO inputs always return ``float``.
    debug_plot : bool, default False
        If ``True``, opens a diagnostic figure showing the unwrapped pilot
        phase sequence and the fitted frequency-slope line.

    Returns
    -------
    float or np.ndarray
        Estimated frequency offset in Hz. For SISO or when
        ``combine_channels=True``: scalar ``float``. For MIMO with
        ``combine_channels=False``: ``np.ndarray`` of shape ``(C,)``
        with one estimate per channel.

    Notes
    -----
    The demodulated pilot phase follows:

    phi_hat[k] = 2*pi * delta_f * t_k + phi_0 + noise

    where t_k = pilot_indices[k] / f_s.

    **Unweighted (OLS):** the minimum-variance unbiased estimator for
    equal-noise pilots:

    delta_f_hat = (1 / (2*pi)) * sum_k ((t_k - t_bar) * (phi_hat[k] - phi_bar)) / sum_k ((t_k - t_bar)^2)

    **SNR-weighted (WLSQ):** pilots are weighted by received power
    v_k = |r_k|^2 (normalised to unit mean), giving:

    delta_f_hat = (1 / (2*pi)) * sum_k (v_k * (t_k - t_bar_v) * (phi_hat[k] - phi_bar_v)) / sum_k (v_k * (t_k - t_bar_v)^2)

    where t_bar_v and phi_bar_v are the weighted means.  This reduces variance by 30-50 % when pilot SNR
    varies significantly across the burst.

    **Lock range:** xp.unwrap bridges each gap between consecutive
    pilot indices.  The gap that limits the lock range is the largest one:

    |delta_f| < f_s / (2 * max_gap)

    where ``max_gap`` is the maximum spacing (in samples) between any two
    consecutive entries of ``pilot_indices``.
    """
    signal_adapter = adapt_signal(
        samples, function_name="estimate_frequency_offset_pilot_symbols()"
    )
    samples = signal_adapter.array
    sampling_rate = signal_adapter.resolve_required("sampling_rate", sampling_rate)
    if pilot_indices is None or pilot_values is None:
        raise ValueError(
            "estimate_frequency_offset_pilot_symbols() requires pilot_indices and "
            "pilot_values."
        )

    samples, xp, _ = dispatch(samples)
    samples, was_1d = as_2d(samples, name="samples")
    C, N = samples.shape

    pilot_indices_np = to_device(pilot_indices, "cpu").astype(np.intp)
    pilot_indices_xp = xp.asarray(pilot_indices_np)  # device copy for GPU fancy-index
    pilot_values_xp = xp.asarray(pilot_values)
    P = len(pilot_indices_np)

    pilot_values_xp = broadcast_channels(pilot_values_xp, C, xp, name="pilot_values")

    # Extract and demodulate: phase = angle(r · conj(s)) = 2π·Δf·t + φ₀ + noise
    r_pilots = samples[:, pilot_indices_xp]  # (C, P)
    phi_pilots = xp.angle(r_pilots * xp.conj(pilot_values_xp))  # (C, P)

    # Unwrap in float64; cp.unwrap preserves input dtype so cast before calling
    phi_pilots_u = xp.unwrap(phi_pilots.astype(xp.float64), axis=-1)  # (C, P)

    t_xp = xp.asarray(pilot_indices_np.astype(np.float64)) / sampling_rate  # (P,)

    if snr_weighted:
        # WLSQ: weight each pilot by received power |r|² independently per channel.
        # Per-channel weights (C, P) handle independent spatial fading correctly;
        # when all channels have equal SNR the weights reduce to the shared case.
        pwr = xp.abs(r_pilots) ** 2  # (C, P)
        pwr_mean = xp.mean(pwr, axis=-1, keepdims=True)  # (C, 1)
        v = pwr / (pwr_mean + 1e-30)  # (C, P) normalised per-channel weights

        # Per-channel weighted means
        v_sum = xp.sum(v, axis=-1, keepdims=True)  # (C, 1)
        t_mean_v = xp.sum(v * t_xp[None, :], axis=-1, keepdims=True) / v_sum  # (C, 1)
        t_c = t_xp[None, :] - t_mean_v  # (C, P) centred times
        phi_mean_v = xp.sum(v * phi_pilots_u, axis=-1, keepdims=True) / v_sum  # (C, 1)
        phi_c = phi_pilots_u - phi_mean_v  # (C, P)

        # Per-channel weighted normal equations: slope[c] = Σ_p v[c,p]·t_c[c,p]·φ_c[c,p]
        #                                                    / Σ_p v[c,p]·t_c[c,p]²
        t_var_w = xp.sum(v * t_c**2, axis=-1)  # (C,)
        safe_denom = xp.where(xp.abs(t_var_w) > 1e-30, t_var_w, xp.ones_like(t_var_w))
        slopes = xp.sum(v * phi_c * t_c, axis=-1) / safe_denom  # (C,)
    else:
        # Unweighted OLS: centered normal equations (Tretter 1985 MVUE), on the
        # shared least-squares slope helper - which keeps Σ(t-t̄)² on device
        # instead of syncing it back as a Python float.
        slopes = linear_trend_slope(phi_pilots_u, x=t_xp, xp=xp)  # (C,) rad/s

    max_gap = int(np.max(np.diff(pilot_indices_np))) if P > 1 else 0
    lock_range = sampling_rate / (2 * max_gap) if max_gap > 0 else float("inf")
    wt_str = "WLSQ" if snr_weighted else "OLS"

    slopes_np = to_device(slopes, "cpu")  # one batched D2H instead of C syncs
    f_per_ch = [float(slopes_np[c]) / (2.0 * np.pi) for c in range(C)]

    logger.info(
        "FOE (pilots, %s): %s Hz [P=%s, max_gap=%s samples, lock_range=±%.1f Hz]",
        wt_str,
        [f"{f:.2f}" for f in f_per_ch],
        P,
        max_gap,
        lock_range,
    )

    if debug_plot:
        from . import plotting as _plotting

        _plotting.plot_pilot_phase_estimate(
            pilot_indices=pilot_indices_np,
            phi_pilots_u=to_device(phi_pilots_u, "cpu"),
            f_est=f_per_ch,
            sampling_rate=sampling_rate,
            show=True,
        )

    if was_1d:
        return float(f_per_ch[0])
    if combine_channels:
        # Weight by mean received pilot power per channel
        pwr_per_ch = to_device(xp.mean(xp.abs(r_pilots) ** 2, axis=-1), "cpu")  # (C,)
        weights = [float(pwr_per_ch[c]) for c in range(C)]
        combined = float(np.average(f_per_ch, weights=weights))
        logger.info(
            "FOE (pilots, %s): combined=%.2f Hz (pilot-power-weighted "
            "mean of %s channels)",
            wt_str,
            combined,
            C,
        )
        return combined
    return np.array(f_per_ch)


def find_bias_tone(
    seg: ArrayType | Signal,
    sampling_rate: float | None = None,
    target_frequency: float | None = None,
    search_band: float | None = None,
) -> float:
    """
    Locate a CW pilot / bias tone in the spectrum of a 1-D complex segment.

    Finds the spectral peak in ``seg`` (optionally restricted to a search
    window) and refines its frequency via **log-parabolic sub-bin
    interpolation** on the three bins around the argmax.  The result is
    modulation-agnostic: no nonlinear transform of the data is applied,
    so there is no AM-induced spectral spreading regardless of constellation
    order.

    Parameters
    ----------
    seg : array_like or Signal
        1-D complex IQ samples.  Must reside on a single backend (CPU or GPU).
        For :class:`Signal` input, the signal's own ``sampling_rate`` always
        wins over a supplied ``sampling_rate``.
    sampling_rate : float, optional
        Sampling rate in Hz.  Required for array input; ignored for
        :class:`Signal` input, which always uses the signal's own
        ``sampling_rate``.
    target_frequency : float, optional
        Centre of the frequency search window in Hz.  Must be paired with
        ``search_band``.  If both are given the argmax is restricted to
        ``[target_frequency - search_band, target_frequency + search_band]``.
        Use this after a coarse correction has placed the tone near a known
        position; without it the wideband data signal typically wins the
        argmax.
    search_band : float, optional
        Half-width of the search window in Hz.  Must be paired with
        ``target_frequency``.

    Returns
    -------
    float
        Refined bias-tone frequency in Hz (always a Python ``float``).

    Raises
    ------
    ValueError
        If ``seg`` is not 1-D, is shorter than 4 samples, only one of
        ``target_frequency`` / ``search_band`` is given, or the search window
        maps to an empty set of FFT bins.

    Notes
    -----
    Log-parabolic sub-bin interpolation fits the log-magnitude of the
    three bins around the argmax peak k:

    y_-  = log(max(|X[k-1]|, epsilon))
    y_0  = log(max(|X[k]|,   epsilon))
    y_+  = log(max(|X[k+1]|, epsilon))
    d    = y_- - 2y_0 + y_+
    delta = 0.5 * (y_- - y_+) / d    if |d| > epsilon
    f_refined = f[k] + delta * f_s / N

    Log-parabolic interpolation better matches the Gaussian shape of a
    windowed spectral peak than standard parabolic (magnitude-domain) fits,
    reducing estimation bias for non-integer tone frequencies.
    """
    signal_adapter = adapt_signal(seg, function_name="find_bias_tone()")
    seg = signal_adapter.array
    sampling_rate = signal_adapter.resolve_required("sampling_rate", sampling_rate)
    if (target_frequency is None) != (search_band is None):
        raise ValueError(
            "target_frequency and search_band must both be provided or both omitted."
        )

    seg, xp, _ = dispatch(seg)
    if seg.ndim != 1:
        raise ValueError(
            f"find_bias_tone expects a 1-D segment, got shape {seg.shape}."
        )
    N = len(seg)
    if N < 4:
        raise ValueError(
            f"Segment too short for bias tone search (N={N}). Minimum 4 samples required."
        )

    nfft = 1 << int(np.ceil(np.log2(N)))  # next power of 2 ≥ N
    X = xp.fft.fft(seg, n=nfft)
    mag = xp.abs(X)  # (nfft,)

    freqs_np = np.fft.fftfreq(nfft, d=1.0 / sampling_rate)  # (nfft,) always on CPU

    if target_frequency is not None:
        assert search_band is not None
        lo = target_frequency - abs(search_band)
        hi = target_frequency + abs(search_band)
        freqs_xp = xp.asarray(freqs_np)
        mask = (freqs_xp >= lo) & (freqs_xp <= hi)
        if not bool(xp.any(mask)):
            raise ValueError(
                f"target_frequency={target_frequency} ± search_band={search_band} produces an "
                f"empty search window for fs={sampling_rate} Hz, nfft={nfft}."
            )
        mag_search = xp.where(mask, mag, xp.zeros_like(mag))
    else:
        mag_search = mag

    k = int(xp.argmax(mag_search))
    k_prev = (k - 1) % nfft  # circular - correct at spectral edges
    k_next = (k + 1) % nfft

    # Transfer 3 neighbourhood magnitudes to CPU - scalars, negligible transfer cost
    delta = float(
        _parabolic_peak_offset(
            float(mag[k_prev]), float(mag[k]), float(mag[k_next]), np, log=True
        )
    )

    f_refined = float(freqs_np[k]) + delta * (sampling_rate / nfft)

    logger.debug(
        "find_bias_tone: peak bin %s (%.2f Hz), delta=%.4f -> %.2f Hz "
        "[nfft=%s, window=%s]",
        k,
        freqs_np[k],
        delta,
        f_refined,
        nfft,
        "full" if target_frequency is None else f"{target_frequency}±{search_band} Hz",
    )

    return float(f_refined)


def _refine_tones_from_spectrum(
    X: ArrayType,
    sampling_rate: float,
    targets,
    search_band: float,
    rows=None,
) -> np.ndarray:
    """Batched log-parabolic tone refinement on a precomputed full-record FFT.

    Device-side counterpart of ``find_bias_tone`` for callers that already hold
    the record's spectrum: refines each ``targets[t]`` by an argmax search over
    ``targets[t] ± search_band`` in the spectrum row ``rows[t]``, followed by
    the same three-bin log-parabolic sub-bin fit.  No zero-padding is applied
    (the sub-bin fit compensates for the coarser bin grid), no extra FFTs are
    run, and all per-tone work stays on device - the refined frequencies come
    back in a **single** host transfer.

    Parameters
    ----------
    X : (R, N) array
        Full FFT (``xp.fft.fft`` bin order) of the rows to search.
    sampling_rate : float
        Sampling rate in Hz.
    targets : sequence of float
        Nominal tone frequencies to refine (length T).
    search_band : float
        Half-width of the per-tone search window in Hz.
    rows : sequence of int, optional
        Spectrum row for each target.  Defaults to ``range(T)`` (requires
        ``R == T``).

    Returns
    -------
    (T,) np.ndarray
        Refined frequencies in Hz (float64, on host).
    """
    X, xp, _ = dispatch(X)
    X, _ = as_2d(X, name="X")
    N = X.shape[-1]
    df = sampling_rate / N
    targets = [float(f) for f in targets]
    if rows is None:
        rows = list(range(len(targets)))
    neigh = xp.asarray([-1, 0, 1], dtype=xp.int64)

    refined = []
    for f_t, row in zip(targets, rows):
        # Signed candidate bins covering [f_t - band, f_t + band] on the N-grid.
        b_lo = max(int(np.ceil((f_t - abs(search_band)) / df)), -(N // 2))
        b_hi = min(int(np.floor((f_t + abs(search_band)) / df)), (N - 1) // 2)
        if b_hi < b_lo:
            raise ValueError(
                f"target_frequency={f_t} ± search_band={search_band} produces an "
                f"empty search window for fs={sampling_rate} Hz, N={N}."
            )
        cand = xp.arange(b_lo, b_hi + 1, dtype=xp.int64)  # signed bins
        spec_row = X[int(row)]
        mag2 = xp.abs(spec_row[cand % N]) ** 2
        kb = cand[xp.argmax(mag2)]  # 0-d signed peak bin, stays on device
        # Three-bin log-parabolic fit.
        mags = xp.abs(spec_row[(kb + neigh) % N]).astype(xp.float64)
        delta = _parabolic_peak_offset(mags[0], mags[1], mags[2], xp, log=True)
        refined.append((kb.astype(xp.float64) + delta) * df)

    return to_device(xp.stack(refined), "cpu")


# -----------------------------------------------------------------------------
# FREQUENCY OFFSET CORRECTION (Signal-aware)
# -----------------------------------------------------------------------------
# Both apply a correction and return corrected samples in the same domain, so
# unlike the estimators above they rewrap to a Signal when given one.


def correct_frequency_offset_blockwise(
    samples: ArrayType | Signal,
    sampling_rate: float | None = None,
    block_size: int | None = None,
    overlap: float | None = None,
    estimator: Callable[[np.ndarray, float], float] | None = None,
    combine_channels: bool = False,
    debug_plot: bool = False,
) -> ArrayType | Signal:
    """
    Estimate and correct a time-varying frequency offset in one call.

    Divides the signal into overlapping blocks, calls an arbitrary
    ``estimator(block_1d_cpu, fs) -> float`` independently on each channel,
    interpolates the per-block estimates with **PCHIP** (monotone, no
    overshoot), integrates to a phase trajectory, and applies the correction
    by complex rotation - returning corrected samples on the same device and
    dtype as the input.

    Parameters
    ----------
    samples : array_like or Signal
        Complex IQ samples.  Shape: ``(N,)`` or ``(C, N)``.  A :class:`Signal`
        returns a new corrected :class:`Signal`; its own ``sampling_rate``
        always wins over a supplied ``sampling_rate``.
    sampling_rate : float, optional
        Sampling rate in Hz.  Required for array input; ignored for
        :class:`Signal` input, which always uses the signal's own
        ``sampling_rate``.
    block_size : int
        Number of samples per analysis block.
    overlap : float
        Fractional overlap between consecutive blocks, in ``[0, 1)``.  Block
        centres are spaced ``step = round(block_size * (1 - overlap))``
        samples apart.
    estimator : callable
        Signature ``(block: np.ndarray, fs: float) -> float``.  Receives a
        1-D NumPy complex array on CPU and returns the estimated instantaneous
        frequency offset in Hz.  Additional arguments can be bound with
        ``partial``.

    combine_channels : bool, default False
        For MIMO inputs (C > 1):

        * ``False`` - estimate and correct each channel independently.
          Each channel's own per-block estimates drive a separate PCHIP
          interpolation and phase trajectory.
        * ``True`` - average the per-channel block estimates, run one PCHIP
          on the average, and apply the same phase trajectory to every channel.
          Use this for coherent MIMO (e.g. polarisation diversity) where all
          channels share the same laser / frequency offset.

    debug_plot : bool, default False
        If ``True``, opens a diagnostic figure showing the per-block frequency
        estimates, PCHIP-interpolated trajectory, and integrated phase
        trajectory.  For MIMO with ``combine_channels=False``, shows channel 0.

    Returns
    -------
    array_like or Signal
        Frequency-offset-corrected samples, **same shape and dtype as the
        input**, on the same backend device.  A :class:`Signal` returns a
        new corrected :class:`Signal`.

    Notes
    -----
    Pipeline (per channel):

    1. Slice into overlapping blocks centred at t_k = s_k + block_size / 2.
    2. Call estimator(block_np, fs) on each CPU block.
    3. Interpolate estimates with scipy.interpolate.PchipInterpolator;
       clamp to [t_0, t_{B-1}] (no extrapolation).
    4. Integrate: theta(n) = (2 * pi / fs) * cumsum(delta_f).
    5. Apply: y[n] = x[n] * exp(-j * theta[n]).
    """
    signal_adapter = adapt_signal(
        samples, function_name="correct_frequency_offset_blockwise()"
    )
    samples = signal_adapter.array
    sampling_rate = signal_adapter.resolve_required("sampling_rate", sampling_rate)
    if block_size is None or overlap is None or estimator is None:
        raise ValueError(
            "correct_frequency_offset_blockwise() requires block_size, overlap, "
            "and estimator."
        )
    if not (0.0 <= overlap < 1.0):
        raise ValueError(f"overlap must be in [0, 1), got {overlap}.")

    samples, xp, _ = dispatch(samples)
    samples_2d, was_1d = as_2d(samples, name="samples")
    C, N = samples_2d.shape

    step = max(1, round(block_size * (1.0 - overlap)))
    starts = list(range(0, N - block_size + 1, step))
    if not starts:
        starts = [0]
        _bsize = N
    else:
        _bsize = block_size

    t_centers = np.array([s + _bsize / 2.0 for s in starts], dtype=np.float64)
    B = len(starts)
    n_grid = np.arange(N, dtype=np.float64)

    # Per-channel per-block estimation: df_all[c, k]
    df_all = np.empty((C, B), dtype=np.float64)
    for c in range(C):
        sig1d = np.asarray(to_device(samples_2d[c], "cpu"))
        for k, s in enumerate(starts):
            df_all[c, k] = float(estimator(sig1d[s : s + _bsize], sampling_rate))

    # Averaged or per-channel frequency estimates for interpolation
    if combine_channels and C > 1:
        df_for_interp = df_all.mean(axis=0, keepdims=True)  # (1, B) - shared
    else:
        df_for_interp = df_all  # (C, B) - per-channel

    C_interp = df_for_interp.shape[0]

    # Pre-import PCHIP once if multiple blocks
    if B > 1:
        from scipy.interpolate import PchipInterpolator

        n_clamped = np.clip(n_grid, t_centers[0], t_centers[-1])

    # PCHIP interpolation + cumulative integration -> (C_interp, N) phase array
    theta_np = np.empty((C_interp, N), dtype=np.float64)
    df_dense_plot: np.ndarray | None = None
    for c in range(C_interp):
        df_est = df_for_interp[c]
        if B == 1:
            df_dense = np.full(N, df_est[0], dtype=np.float64)
        else:
            df_dense = PchipInterpolator(t_centers, df_est)(n_clamped)
        theta_np[c] = (2.0 * np.pi / sampling_rate) * np.cumsum(df_dense)
        if debug_plot and c == 0:
            df_dense_plot = df_dense

    # Broadcast averaged trajectory to all channels if needed
    if combine_channels and C > 1:
        theta_np_full = np.broadcast_to(theta_np, (C, N)).copy()
    else:
        theta_np_full = theta_np  # (C, N)

    # Apply correction on the original device
    theta_xp = xp.asarray(theta_np_full)  # (C, N) on device
    two_pi = 2.0 * np.pi
    phase_f64 = theta_xp.astype(xp.float64)
    phase_wrapped = (phase_f64 - xp.round(phase_f64 / two_pi) * two_pi).astype(
        xp.float32
    )
    phasor = xp.exp(-1j * phase_wrapped).astype(samples_2d.dtype)
    corrected_2d = samples_2d * phasor

    df_log = df_all.mean(axis=0) if (combine_channels and C > 1) else df_all[0]
    logger.debug(
        "correct_frequency_offset_blockwise: C=%s, B=%s blocks, "
        "freq range=[%.2f, %.2f] Hz, total phase drift=%.3f rad",
        C,
        B,
        df_log.min(),
        df_log.max(),
        float(theta_np_full[0, -1]),
    )
    logger.info(
        "correct_frequency_offset_blockwise:  mean = %+.1f Hz,  "
        "std = %.1f Hz,  range = [%+.1f, %+.1f] Hz (%s segments)",
        df_all.mean(),
        df_all.std(),
        df_all.min(),
        df_all.max(),
        len(starts),
    )

    if debug_plot and df_dense_plot is not None:
        from . import plotting as _plotting

        title = (
            f"correct_frequency_offset_blockwise - channel 0 of {C}"
            if C > 1 and not combine_channels
            else "correct_frequency_offset_blockwise"
        )
        _plotting.plot_frequency_offset_blockwise_result(
            t_centers=t_centers,
            df_estimates=df_for_interp[0],
            n_grid=n_grid,
            df_dense=df_dense_plot,
            phase_trajectory=theta_np_full[0],
            show=True,
            title=title,
        )

    return signal_adapter.wrap_samples(restore_1d(was_1d, corrected_2d))


def correct_static_frequency_offset(
    samples: ArrayType | Signal,
    sampling_rate: float | None = None,
    offset: float | np.ndarray | None = None,
) -> ArrayType | Signal:
    """
    Applies a **constant** frequency offset correction via exact complex mixing.

    Multiplies ``samples`` by exp(-j * 2 * pi * delta_f * n / fs) where
    *n* starts at 0 at the first sample of the passed array.

    Warning: designed for full-signal correction of a single static offset.
    Calling it on sub-blocks breaks phase continuity (phasor restarts at n = 0).
    For time-varying or block-streamed correction use ``correct_frequency_offset_blockwise``.

    Unlike ``shift_frequency``, this function applies the
    correction **without bin quantization**, preserving the full precision of
    a sub-bin estimate (e.g. from Jacobsen interpolation in
    ``estimate_frequency_offset_mth_power``).

    Parameters
    ----------
    samples : array_like or Signal
        Input signal samples. Shape: (..., N).  A :class:`Signal` returns a
        new corrected :class:`Signal`; its own ``sampling_rate`` always
        wins over a supplied ``sampling_rate``.
    sampling_rate : float, optional
        Sampling rate in Hz.  Required for array input; ignored for
        :class:`Signal` input, which always uses the signal's own
        ``sampling_rate``.
    offset : float or np.ndarray
        Estimated frequency offset in Hz. Either a scalar (same correction
        applied to all channels) or a 1-D array of shape ``(C,)`` as
        returned by the per-channel ``estimate_frequency_offset_*``
        functions when ``combine_channels=False``.  A per-channel array
        requires ``samples`` to have shape ``(C, N)``.

    Returns
    -------
    array_like or Signal
        Frequency-corrected samples, same shape and dtype as input.  A
        :class:`Signal` returns a new corrected :class:`Signal`.
    """
    signal_adapter = adapt_signal(
        samples, function_name="correct_static_frequency_offset()"
    )
    samples = signal_adapter.array
    sampling_rate = signal_adapter.resolve_required("sampling_rate", sampling_rate)
    if offset is None:
        raise ValueError("correct_static_frequency_offset() requires offset.")

    samples, xp, _ = dispatch(samples)
    offset_arr = xp.asarray(offset)
    per_channel = offset_arr.ndim >= 1 and offset_arr.size > 1

    if per_channel:
        logger.debug(
            "Applying per-channel frequency offset correction: %s Hz "
            "(sampling_rate=%.0f Hz)",
            [f"{f:.4f}" for f in offset_arr.flat],
            sampling_rate,
        )
    else:
        logger.debug(
            "Applying frequency offset correction: %.4f Hz (sampling_rate=%.0f Hz)",
            float(offset_arr.reshape(-1)[0]),
            sampling_rate,
        )

    n = samples.shape[-1]
    t = xp.arange(n, dtype=xp.float64) / sampling_rate  # float64 - correct for all N

    if xp.iscomplexobj(samples):
        target_dtype = samples.dtype
    else:
        target_dtype = xp.complex64 if samples.dtype == xp.float32 else xp.complex128

    # For complex64 targets: wrap to [-π, π] in float64, then cast to float32 before exp -
    # matching the correct_carrier_phase / JAX BPS pattern in recovery.py / equalization.py.
    # Wrapping is essential: casting a large unbounded ramp (e.g. 6000 rad) directly to
    # float32 causes trig argument-reduction error (~|phase|·2⁻²³); wrapping first bounds
    # the mantissa range to [-π, π] so float32 error is only ~3.7x10⁻⁷ rad.
    # Avoids the original (C, N) complex128 intermediate for complex64 targets.
    dtype_real = xp.float32 if target_dtype == xp.complex64 else xp.float64
    two_pi = 2.0 * np.pi

    if per_channel:
        # Build a (C, N) mixer - one distinct tone per channel
        C = samples.shape[0]
        offsets_xp = xp.asarray(offset_arr.reshape(-1)[:C], dtype=xp.float64)  # (C,)
        phase = -2.0 * xp.pi * offsets_xp[:, None] * t[None, :]  # (C, N) float64
        phase_exp = (phase - xp.round(phase / two_pi) * two_pi).astype(dtype_real)
        mixer = xp.exp(1j * phase_exp).astype(target_dtype)
    else:
        # Scalar path - single mixer broadcast over all channels.  offset_arr
        # may be a genuine 0-d array or a size-1 array of any shape (e.g. a
        # single-channel (1,) per-channel estimate) - reshape before the
        # scalar cast so both are accepted on every backend (recent NumPy
        # and CuPy both reject float() on non-0-d arrays outright).
        phase = -2.0 * xp.pi * float(offset_arr.reshape(-1)[0]) * t  # (N,) float64
        phase_exp = (phase - xp.round(phase / two_pi) * two_pi).astype(dtype_real)
        mixer = xp.exp(1j * phase_exp).astype(target_dtype)
        if samples.ndim > 1:
            mixer = mixer.reshape((1,) * (samples.ndim - 1) + (-1,))

    return signal_adapter.wrap_samples(samples * mixer)
