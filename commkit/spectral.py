"""
Spectral analysis and frequency-domain processing.

This module provides high-performance routines for spectral estimation and
manipulation, optimized for both CPU and GPU backends. It supports Welch's
Power Spectral Density (PSD) method and phase-continuous frequency shifting.
"""

from collections.abc import Sequence
from typing import Any, cast

import numpy as np

from .backend import ArrayType, dispatch
from .core.signal import Signal
from .logger import logger


def shift_frequency(
    samples: ArrayType | Signal,
    offset: float,
    sampling_rate: float | None = None,
) -> tuple[ArrayType, float] | Signal:
    """
    Applies a frequency offset (complex mixing) to a signal.

    This function shifts the signal spectrum by a specified offset in Hz
    by multiplying the samples with a complex phasor:
    s_shifted(t) = s(t) * e^(j * 2 * pi * f_offset * t)

    To maintain phase continuity and prevent spectral leakage when the
    signal is treated as periodic (e.g., in circular convolution or
    FFT-based operations), the applied offset is quantized to the
    fundamental frequency resolution of the signal (df = f_s / N).

    Parameters
    ----------
    samples : array_like or Signal
        Input signal samples. Shape: (..., N_samples).
    offset : float
        Target frequency shift in Hz. Positive values shift the spectrum
        towards higher frequencies.
    sampling_rate : float
        Sampling rate in Hz.

    Returns
    -------
    shifted_samples : array_like
        The frequency-shifted signal on the same backend as the input.
    actual_offset : float
        The actual quantized frequency shift applied to the signal.

    Notes
    -----
    The quantization ensures that the applied shift corresponds to an
    integer number of cycles over the signal duration, which is critical
    for preserving the circularity of the signal's phase.

    When ``samples`` is a :class:`Signal`, a new :class:`Signal` is returned
    with the shift applied and ``digital_frequency_offset`` accumulated;
    ``sampling_rate`` is taken from the signal.
    """
    if isinstance(samples, Signal):
        sig = samples
        new = sig.copy()
        out = shift_frequency(sig.samples, offset, sig.sampling_rate)
        assert isinstance(out, tuple)  # array input -> (samples, actual_offset)
        shifted, actual = out
        new.samples = shifted
        if new.digital_frequency_offset is None:
            new.digital_frequency_offset = 0.0
        new.digital_frequency_offset += actual
        return new

    if sampling_rate is None:
        raise ValueError("shift_frequency() requires sampling_rate for array input.")

    samples, xp, _ = dispatch(samples)

    # Axis -1 is time
    n = samples.shape[-1]
    df = sampling_rate / n

    # Quantize offset to nearest bin to ensure phase continuity
    k = xp.round(offset / df)
    actual_offset = k * df

    if not xp.isclose(offset, actual_offset):
        logger.warning(
            f"Requested offset {offset:.3f} Hz quantized to {actual_offset:.3f} Hz "
            f"(step {df:.3f} Hz) to maintain phase continuity."
        )
    else:
        logger.debug(f"Applying frequency offset: {actual_offset:.3f} Hz.")

    # Time vector
    t = xp.arange(n) / sampling_rate

    # Apply mixing
    # exp(j * 2 * pi * f * t)
    # Phase is computed at float64 accuracy (xp.pi is float64), then the mixer
    # is cast to the signal's complex precision to prevent silent promotion of
    # complex64/float32 signals to complex128/float64.
    phase = 2 * xp.pi * actual_offset * t
    mixer = xp.exp(1j * phase)  # complex128
    if xp.iscomplexobj(samples):
        target_cdtype = samples.dtype
    else:
        target_cdtype = xp.complex64 if samples.dtype == xp.float32 else xp.complex128
    mixer = mixer.astype(target_cdtype)

    # Broadcast mixer to match samples shape: (1, ..., 1, N)
    if samples.ndim > 1:
        mixer = mixer.reshape((1,) * (samples.ndim - 1) + (-1,))

    return samples * mixer, float(actual_offset)


def add_pilot_tone(
    samples: ArrayType | Signal,
    sampling_rate: float | None = None,
    frequency: float | Sequence[float] | None = None,
    power_ratio_db: float | Sequence[float] = -15.0,
    phase_init: float = 0.0,
    renormalize: bool = False,
) -> tuple[ArrayType, float | list[float]] | Signal:
    r"""
    Add a continuous-wave (CW) pilot tone to a baseband waveform.

    Superimposes a * exp(j*(2*pi*f_p*n/f_s + phi_0)) on the oversampled samples.
    The tone acquires the same carrier frequency offset and phase noise as the
    data; at the receiver its phase directly recovers both - see
    ``recover_carrier_phase_pilot_tone``.

    Apply to a pulse-shaped oversampled waveform before channel impairments.
    Place the tone in a guard band: (1+beta)/2 * R_s < |f_p| < f_s/2.

    Parameters
    ----------
    samples : array_like
        Complex baseband samples. Shape: ``(N,)`` (SISO) or ``(C, N)`` (MIMO).
        The tone is added to every channel, each scaled to its own power.
    frequency : float or sequence of float
        Requested tone frequency fp in Hz, in ``(-f_s/2, f_s/2)``.
        A **scalar** places the same tone on every channel.  A **sequence**
        of length ``C`` places one tone per channel (channel ``c`` gets
        ``frequency[c]``) - distinct per-channel tones enable, e.g.,
        tone-based polarization demultiplexing
        (``demultiplex_polarization_tones``).  Each value is quantized to the
        nearest FFT bin ``f_s/N`` (see Notes); the **actual** applied
        frequency(ies) are returned.
    sampling_rate : float
        Sampling rate fs in Hz.
    power_ratio_db : float or sequence of float, default -15.0
        Pilot-to-signal power ratio (PSR) in dB: 10*log10(P_tone / P_signal).
        Typical range -20 to -10 dB.  A **scalar** applies the same PSR to every
        channel; a **sequence** of length ``C`` sets one PSR per channel
        (mirroring per-channel ``frequency``).
    phase_init : float, default 0.0
        Initial tone phase phi_0 in radians, common to all channels.  Acts as
        a known phase reference; it appears as a constant offset in the
        recovered phase and is absorbed by the usual post-CPR ambiguity
        resolution.
    renormalize : bool, default False
        If ``True``, rescale each channel after adding the tone so its mean
        power matches the input (preserves the library power invariant E[|x|²] = 1/sps).
        If ``False``, total power rises by 1 + 10^(PSR/10).

    Returns
    -------
    samples : array_like
        Samples with the pilot tone added, same shape, dtype, and backend as
        the input.
    actual_frequency : float or list of float
        The grid-quantized tone frequency(ies) in Hz actually applied (see
        Notes).  A **scalar** ``frequency`` returns a single ``float``; a
        per-channel **sequence** returns a ``list`` of ``C`` floats.  Store
        this (e.g. in ``pilot_tone_frequency``) and pass it to the receiver,
        since it - not the requested value - is where the tone(s) sit.

    Raises
    ------
    ValueError
        If any requested frequency lies outside ``(-fs/2, fs/2)``, or if a
        per-channel sequence is given whose length does not equal ``C``.

    Notes
    -----
    Each requested frequency is snapped to the nearest FFT bin (fs/N) so the
    tone completes an integer number of cycles per buffer - ensuring seamless
    playback on an AWG/DAC.  The quantization error is at most fs/(2N).
    The phase ramp is accumulated in float64 to avoid trig argument-reduction
    error for large N.

    When ``samples`` is a :class:`Signal`, the sampling rate is taken from the
    signal, so the **second positional argument is the frequency** (i.e. call
    ``add_pilot_tone(sig, freq, ...)``).  A new :class:`Signal` is returned with
    ``pilot_tone_frequency`` / ``pilot_tone_power_ratio_db`` recorded.
    """
    if isinstance(samples, Signal):
        sig = samples
        # Signal rate is implicit; the second positional carries the frequency.
        freq = frequency if frequency is not None else sampling_rate
        if freq is None:
            raise ValueError("add_pilot_tone() requires a frequency.")
        new = sig.copy()
        out = add_pilot_tone(
            sig.samples,
            sig.sampling_rate,
            freq,
            power_ratio_db=power_ratio_db,
            phase_init=phase_init,
            renormalize=renormalize,
        )
        assert isinstance(out, tuple)  # array input -> (samples, actual_frequency)
        new.samples, new.pilot_tone_frequency = out
        new.pilot_tone_power_ratio_db = power_ratio_db
        return new

    if sampling_rate is None or frequency is None:
        raise ValueError(
            "add_pilot_tone() requires sampling_rate and frequency for array input."
        )

    samples, xp, _ = dispatch(samples)
    was_1d = samples.ndim == 1
    if was_1d:
        samples = samples[None, :]  # (1, N)
    C, N = samples.shape

    # Normalise ``frequency`` to a per-channel (C,) host array.  A scalar is
    # broadcast to every channel (and returns a scalar for back-compat); a
    # sequence must supply exactly one frequency per channel.
    scalar_input = np.ndim(frequency) == 0
    if scalar_input:
        f_req = [float(cast(float, frequency))] * C
    else:
        f_req = [float(f) for f in cast(Sequence[float], frequency)]
        if len(f_req) != C:
            raise ValueError(
                f"frequency sequence has length {len(f_req)} but the signal has "
                f"C={C} channel(s); supply one frequency per channel."
            )

    nyq = sampling_rate / 2.0
    for f in f_req:
        if not (-nyq < f < nyq):
            raise ValueError(
                f"frequency={f} must lie in (-fs/2, fs/2) = (±{nyq:.3g}) Hz."
            )

    # Snap each tone to the FFT bin grid so it is buffer-periodic (loop-seamless
    # on an AWG/DAC), mirroring shift_frequency's quantization.
    df = sampling_rate / N
    actual = [float(round(f / df) * df) for f in f_req]
    for f_in, f_out in zip(f_req, actual):
        if abs(f_out - f_in) > 1e-12 * max(1.0, abs(f_in)):
            logger.warning(
                f"add_pilot_tone: requested {f_in:.3f} Hz quantized to "
                f"{f_out:.3f} Hz (grid step fs/N={df:.3f} Hz) for "
                f"buffer-periodic (loop-seamless) playback."
            )

    # Normalise ``power_ratio_db`` to a per-channel (C,) list, mirroring how
    # ``frequency`` is handled: a scalar broadcasts to every channel; a sequence
    # must supply exactly one PSR per channel.
    scalar_power = np.ndim(power_ratio_db) == 0
    if scalar_power:
        psr_req = [float(cast(float, power_ratio_db))] * C
    else:
        psr_req = [float(p) for p in cast(Sequence[float], power_ratio_db)]
        if len(psr_req) != C:
            raise ValueError(
                f"power_ratio_db sequence has length {len(psr_req)} but the signal "
                f"has C={C} channel(s); supply one PSR per channel."
            )

    # Per-channel signal power and the tone amplitude that realises the PSR.
    p_signal = xp.mean(xp.abs(samples) ** 2, axis=-1, keepdims=True)  # (C, 1) float
    psr_lin = (10.0 ** (xp.asarray(psr_req, dtype=xp.float64) / 10.0)).reshape(
        C, 1
    )  # (C, 1)
    amp = xp.sqrt(p_signal * psr_lin)  # (C, 1)

    # Per-channel phase ramp (C, N) in float64; wrap to [-π, π) before exp so
    # complex64 targets avoid argument-reduction error on long ramps
    # (cf. correct_static_frequency_offset).
    two_pi = 2.0 * xp.pi
    n = xp.arange(N, dtype=xp.float64)  # (N,)
    f_ch = xp.asarray(actual, dtype=xp.float64).reshape(C, 1)  # (C, 1)
    phase = two_pi * f_ch * n[None, :] / sampling_rate + phase_init  # (C, N) float64
    phase = phase - xp.round(phase / two_pi) * two_pi

    dtype_real = xp.float32 if samples.dtype == xp.complex64 else xp.float64
    tone = xp.exp(1j * phase.astype(dtype_real)).astype(samples.dtype)  # (C, N)
    out = samples + amp.astype(samples.dtype) * tone  # (C, N)

    if renormalize:
        # Restore each channel to its original mean power.
        p_out = xp.mean(xp.abs(out) ** 2, axis=-1, keepdims=True)  # (C, 1)
        out = out * xp.sqrt(p_signal / p_out).astype(samples.dtype)

    f_log = f"{actual[0]:.3g} Hz" if scalar_input else f"{actual} Hz"
    psr_log = f"{psr_req[0]:.1f} dB" if scalar_power else f"{psr_req} dB"
    logger.info(
        f"add_pilot_tone: f_p={f_log}, PSR={psr_log}, "
        f"phase_init={phase_init:.3g} rad, "
        f"renormalize={renormalize} [C={C}, N={N}]"
    )

    out = out[0] if was_1d else out
    actual_frequency: float | list[float] = actual[0] if scalar_input else actual
    return out, actual_frequency


def welch_psd(
    samples: ArrayType | Signal,
    sampling_rate: float | None = None,
    nperseg: int = 256,
    detrend: str | bool | None = False,
    average: str | None = "mean",
    window: str | tuple[Any, ...] | Any = "hann",
    noverlap: int | None = None,
    nfft: int | None = None,
    scaling: str = "density",
    return_onesided: bool | None = None,
    axis: int = -1,
) -> tuple[ArrayType, ArrayType]:
    """
    Estimates the Power Spectral Density (PSD) using Welch's method.

    Welch's method provides a lower-variance estimate of the PSD
    compared to a raw periodogram by averaging spectra computed over
    overlapping segments of the signal.

    Parameters
    ----------
    samples : array_like or Signal
        Input signal samples. Shape: (..., N_samples).
    sampling_rate : float
        Sampling rate in Hz.
    nperseg : int, default 256
        Length of each segment. A longer segment increases frequency
        resolution but also increases the variance of the estimate.
    detrend : str or bool, default False
        Specifies how to detrend each segment (e.g., 'constant', 'linear').
    average : {"mean", "median"}, default "mean"
        Method to use for averaging segments. Median is more robust to
        transient outliers.
    window : str or tuple or array_like, default "hann"
        Desired window to use. If `window` is a string or tuple, it is
        passed to `scipy.signal.get_window` to generate the window values.
    noverlap : int, optional
        Number of points to overlap between segments. If None,
        `noverlap = nperseg // 2`.
    nfft : int, optional
        Length of the FFT used, if a zero padded FFT is desired. If None,
        the FFT length is `nperseg`.
    scaling : {"density", "spectrum"}, default "density"
        Selects between computing the power spectral density ('density')
        where Pxx has units of V**2/Hz and computing the power spectrum
        ('spectrum') where Pxx has units of V**2.
    return_onesided : bool, optional
        If True, returns a one-sided spectrum (frequencies 0 to f_s/2)
        for real-valued data. For complex data, only two-sided spectra
        (frequencies -f_s/2 to f_s/2) are supported.
        Axis along which to compute the PSD.

    Returns
    -------
    f : array_like
        Array of sample frequencies.
    Pxx : array_like
        Power spectral density (linear scale, units: V^2/Hz).

    Raises
    ------
    ValueError
        If `return_onesided` set to True for complex-valued inputs.
    """
    if isinstance(samples, Signal):
        sig = samples
        return welch_psd(
            sig.samples,
            sig.sampling_rate,
            nperseg=nperseg,
            detrend=detrend,
            average=average,
            window=window,
            noverlap=noverlap,
            nfft=nfft,
            scaling=scaling,
            return_onesided=return_onesided,
            axis=-1,
        )

    if sampling_rate is None:
        raise ValueError("welch_psd() requires sampling_rate for array input.")

    samples, xp, sp = dispatch(samples)
    is_complex = xp.iscomplexobj(samples)

    if return_onesided is None:
        return_onesided = not is_complex

    # scipy.signal.welch returns onesided by default for real, two-sided for complex
    # unless return_onesided is explicitly set.
    # Note: scipy's return_onesided argument serves to force one-sided for real data.
    # It cannot force one-sided for complex data (always raises error).
    # For complex data, it always returns two-sided (0 to fs).

    if is_complex and return_onesided:
        raise ValueError("Cannot compute one-sided PSD for complex data.")

    f, Pxx = sp.signal.welch(
        samples,
        fs=sampling_rate,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend=detrend,
        return_onesided=return_onesided,
        scaling=scaling,
        axis=axis,
        average=average,
    )

    if not return_onesided:
        # Shift zero frequency to center
        # f is typically 1D array of frequencies
        f = xp.fft.fftshift(f)
        # Pxx needs shift along the frequency axis
        Pxx = xp.fft.fftshift(Pxx, axes=axis)

    return f, Pxx


def spectrogram(
    samples: ArrayType | Signal,
    sampling_rate: float | None = None,
    window: str | tuple[Any, ...] | Any = "hann",
    nperseg: int = 256,
    noverlap: int | None = None,
    nfft: int | None = None,
    detrend: str | bool | None = False,
    return_onesided: bool | None = None,
    scaling: str = "density",
    axis: int = -1,
    mode: str = "psd",
) -> tuple[ArrayType, ArrayType, ArrayType]:
    """
    Computes a spectrogram with consecutive Fourier transforms.

    Parameters
    ----------
    samples : array_like or Signal
        Input signal samples. Shape: (..., N_samples).
    sampling_rate : float
        Sampling rate in Hz.
    window : str or tuple or array_like, default "hann"
        Desired window to use. If `window` is a string or tuple, it is
        passed to `scipy.signal.get_window` to generate the window values.
    nperseg : int, default 256
        Length of each segment. A longer segment increases frequency
        resolution but also increases the variance of the estimate.
    noverlap : int, optional
        Number of points to overlap between segments. If None,
        `noverlap = nperseg // 2`.
    nfft : int, optional
        Length of the FFT used, if a zero padded FFT is desired. If None,
        the FFT length is `nperseg`.
    detrend : str or bool, default False
        Specifies how to detrend each segment (e.g., 'constant', 'linear').
    return_onesided : bool, optional
        If True, returns a one-sided spectrum (frequencies 0 to f_s/2)
        for real-valued data. For complex data, only two-sided spectra
        are supported.
    scaling : {"density", "spectrum"}, default "density"
        Selects between computing the power spectral density ('density')
        where Sxx has units of V**2/Hz and computing the power spectrum
        ('spectrum') where Sxx has units of V**2.
    axis : int, default -1
        The axis along which to compute the spectrogram.
    mode : {"psd", "complex", "magnitude", "angle", "phase"}, default "psd"
        Type of spectrogram to return. Options are 'psd', 'complex',
        'magnitude', 'angle', 'phase'.

    Returns
    -------
    f : array_like
        Array of sample frequencies.
    t : array_like
        Array of segment times.
    Sxx : array_like
        Spectrogram of the signal.

    Raises
    ------
    ValueError
        If `return_onesided` set to True for complex-valued inputs.
    """
    if isinstance(samples, Signal):
        sig = samples
        return spectrogram(
            sig.samples,
            sig.sampling_rate,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            detrend=detrend,
            return_onesided=return_onesided,
            scaling=scaling,
            axis=-1,
            mode=mode,
        )

    if sampling_rate is None:
        raise ValueError("spectrogram() requires sampling_rate for array input.")

    samples, xp, sp = dispatch(samples)
    is_complex = xp.iscomplexobj(samples)

    if return_onesided is None:
        return_onesided = not is_complex

    if is_complex and return_onesided:
        raise ValueError("Cannot compute one-sided spectrogram for complex data.")

    f, t, Sxx = sp.signal.spectrogram(
        samples,
        fs=sampling_rate,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend=detrend,
        return_onesided=return_onesided,
        scaling=scaling,
        axis=axis,
        mode=mode,
    )

    if not return_onesided:
        # Shift zero frequency to center
        f = xp.fft.fftshift(f)
        # Sxx frequency axis is at position axis_pos in output
        ndim = samples.ndim
        axis_pos = axis % ndim
        Sxx = xp.fft.fftshift(Sxx, axes=axis_pos)

    return f, t, Sxx
