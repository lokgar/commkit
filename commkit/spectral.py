"""Spectral estimation and frequency shifting.

:func:`welch_psd` (Welch's averaged periodogram) and :func:`spectrogram`
(a :class:`Spectrogram`); :func:`shift_frequency` (complex mixing) and
:func:`add_pilot_tone` (a CW tone), both at a frequency snapped to the FFT
grid so that a periodic record stays periodic, and :func:`grid_frequency`,
which returns that grid frequency.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from ._array import as_2d, restore_1d
from .backend import ArrayType, dispatch
from .core._signal_adapter import S, adapt_signal
from .core.signal import Signal
from .logger import logger
from .math import _derotate, _phasor

__all__ = [
    "Spectrogram",
    "add_pilot_tone",
    "grid_frequency",
    "shift_frequency",
    "spectrogram",
    "welch_psd",
]


def _validate_and_shift(
    xp: Any, is_complex: bool, return_onesided: bool | None, label: str
) -> tuple[bool, Callable[..., tuple[ArrayType, ...]]]:
    """Resolve ``return_onesided`` and build the matching shift of the result.

    Shared by :func:`welch_psd` and :func:`spectrogram`: ``return_onesided``
    defaults to ``not is_complex``, a one-sided request for complex data is
    rejected, and a two-sided result has its frequency axis (and the data
    indexed by it) fftshifted to centered order, ``-fs/2`` to ``fs/2``.

    Parameters
    ----------
    xp : module
        Array module providing ``fft.fftshift``.
    is_complex : bool
        Whether the input samples are complex-valued.
    return_onesided : bool or None
        The caller-supplied value (``None`` triggers the default).
    label : str
        Name for the error message (``"PSD"``, ``"spectrogram"``).

    Returns
    -------
    return_onesided : bool
        The resolved value to pass through to the ``scipy.signal`` call.
    shift : callable
        ``shift(f, *arrays_with_axis) -> (f, *arrays)`` where each element of
        ``arrays_with_axis`` is an ``(array, axis)`` pair.  No-op when
        ``return_onesided`` is ``True``; otherwise fftshifts ``f`` and every
        array at its given axis.
    """
    if return_onesided is None:
        return_onesided = not is_complex
    if is_complex and return_onesided:
        raise ValueError(f"Cannot compute one-sided {label} for complex data.")

    def shift(
        f: ArrayType, *arrays_with_axis: tuple[ArrayType, int]
    ) -> tuple[ArrayType, ...]:
        if return_onesided:
            return (f, *(a for a, _ in arrays_with_axis))
        f_shifted = xp.fft.fftshift(f)
        shifted = (xp.fft.fftshift(a, axes=ax) for a, ax in arrays_with_axis)
        return (f_shifted, *shifted)

    return return_onesided, shift


def grid_frequency(
    frequency: float | Sequence[float],
    *,
    sampling_rate: float,
    num_samples: int,
) -> float | np.ndarray:
    """The FFT-bin frequency that ``shift_frequency`` / ``add_pilot_tone`` apply.

    Both functions snap a requested frequency to the nearest multiple of the
    bin spacing ``sampling_rate / num_samples`` (ties to even), so the tone or
    shift completes a whole number of cycles over the record.  This returns
    that applied value, for example to tell a receiver where a pilot tone
    sits.

    Parameters
    ----------
    frequency : float or sequence of float
        Requested frequency (or one per channel) in Hz.
    sampling_rate : float
        Sampling rate in Hz.
    num_samples : int
        Record length ``N`` along the time axis (``samples.shape[-1]``).

    Returns
    -------
    float or numpy.ndarray
        The applied frequency in Hz; an array for sequence input.

    Examples
    --------
    >>> grid_frequency(1.03e6, sampling_rate=8e6, num_samples=1000)
    1032000.0
    >>> tone = grid_frequency(f, sampling_rate=sig.sampling_rate,
    ...                       num_samples=sig.samples.shape[-1])
    """
    if num_samples < 1:
        raise ValueError(f"num_samples must be >= 1, got {num_samples}.")
    df = sampling_rate / num_samples
    snapped = np.round(np.asarray(frequency, dtype=np.float64) / df) * df
    return float(snapped) if snapped.ndim == 0 else snapped


def shift_frequency(
    samples: S,
    *,
    frequency: float,
    sampling_rate: float | None = None,
) -> S:
    """Shift the spectrum by a frequency (complex mixing).

    ``y[n] = x[n] · exp(j 2π f n / f_s)``, with ``f`` snapped to the FFT grid
    ``f_s / N`` (see :func:`grid_frequency`; a warning is logged when it
    moves), so the shift completes a whole number of cycles over the record
    and a periodic record stays periodic.

    Parameters
    ----------
    samples : array_like or Signal
        Input samples, time on the last axis.
    frequency : float
        Requested frequency shift in Hz; positive values move the spectrum
        up.
    sampling_rate : float, optional
        Sampling rate in Hz.  Taken from the Signal; required for array
        input.  A value that disagrees with the Signal raises.

    Returns
    -------
    ndarray or Signal
        The shifted samples, complex, on the input's device.  The applied
        shift is ``grid_frequency(frequency, sampling_rate=fs, num_samples=N)``.
    """
    signal_adapter = adapt_signal(samples, function_name="shift_frequency()")
    x = signal_adapter.array
    sampling_rate = signal_adapter.resolve_fact("sampling_rate", sampling_rate)

    x, xp, _ = dispatch(x)

    # Axis -1 is time
    n = x.shape[-1]
    df = sampling_rate / n

    # Quantize offset to nearest bin to ensure phase continuity
    actual_offset = grid_frequency(
        frequency, sampling_rate=sampling_rate, num_samples=n
    )

    if not np.isclose(frequency, actual_offset):
        logger.warning(
            "Requested offset %.3f Hz quantized to %.3f Hz (step %.3f Hz) "
            "to maintain phase continuity.",
            frequency,
            actual_offset,
            df,
        )
    else:
        logger.debug("Applying frequency offset: %.3f Hz.", actual_offset)

    # x · exp(j 2π f t): the phase in float64, the rotation in the samples'
    # precision (one fused kernel for complex64 on the GPU).
    phase = (2.0 * np.pi * actual_offset / sampling_rate) * xp.arange(
        n, dtype=xp.float64
    )
    return signal_adapter.wrap_samples(_derotate(x, -phase))


def add_pilot_tone(
    samples: S,
    *,
    frequency: float | Sequence[float],
    sampling_rate: float | None = None,
    power_ratio_db: float | Sequence[float] = -15.0,
    phase_init: float = 0.0,
    renormalize: bool = False,
) -> S:
    r"""Add a continuous-wave (CW) pilot tone to a baseband waveform.

    Adds ``a · exp(j (2π f_p n / f_s + φ_0))`` to the samples.  The tone
    acquires the same carrier frequency offset and phase noise as the data,
    so its phase at the receiver recovers both (see
    :class:`commkit.recovery.PilotTone`).

    Apply it to the pulse-shaped, oversampled waveform before the channel
    impairments, with the tone in a guard band:
    ``(1 + rolloff) / 2 · R_s < |f_p| < f_s / 2``.

    Parameters
    ----------
    samples : array_like or Signal
        Complex baseband samples, ``(N,)`` or ``(C, N)``.  The tone is added
        to every channel, scaled to that channel's power.
    frequency : float or sequence of float
        Requested tone frequency ``f_p`` in Hz, in ``(-f_s/2, f_s/2)``.  A
        scalar places the same tone on every channel; a sequence of length
        ``C`` places one tone per channel (channel ``c`` gets
        ``frequency[c]``), as for tone-based polarization demultiplexing
        (:func:`~commkit.equalization.demultiplex_polarization_tones_static`).
        Each value is snapped to the FFT grid ``f_s/N`` (see Notes);
        :func:`grid_frequency` gives the applied value.
    sampling_rate : float, optional
        Sampling rate in Hz.  Taken from the Signal; required for array
        input.  A value that disagrees with the Signal raises.
    power_ratio_db : float or sequence of float, default -15.0
        Pilot-to-signal power ratio (PSR) in dB, ``10·log10(P_tone /
        P_signal)``; typically -20 to -10 dB.  A scalar applies to every
        channel, a sequence of length ``C`` sets one PSR per channel.
    phase_init : float, default 0.0
        Initial tone phase ``φ_0`` in radians, common to all channels.  Acts as
        a known phase reference; it appears as a constant offset in the
        recovered phase and is absorbed by the usual post-CPR ambiguity
        resolution.
    renormalize : bool, default False
        Rescale each channel after adding the tone so that its mean power
        matches the input (keeping ``E[|x|²] = 1/sps``).  Otherwise the power
        rises by the factor ``1 + 10^(PSR/10)``.

    Returns
    -------
    ndarray or Signal
        Samples with the pilot tone added, same shape, dtype and device as
        the input.  The tones sit at
        ``grid_frequency(frequency, sampling_rate=fs, num_samples=N)``, not at
        the requested values; pass that to the receiver.

    Raises
    ------
    ValueError
        If any requested frequency lies outside ``(-fs/2, fs/2)``, or if a
        per-channel sequence is given whose length does not equal ``C``.

    Notes
    -----
    Each requested frequency is snapped to the nearest FFT bin ``f_s/N`` (a
    warning is logged when it moves), so the tone completes a whole number of
    cycles per record and loops seamlessly on an AWG/DAC.  The snapping error
    is at most ``f_s/(2N)``.  The phase ramp is computed and wrapped in
    float64, so long records keep their phase accuracy in complex64.
    """
    signal_adapter = adapt_signal(samples, function_name="add_pilot_tone()")
    x = signal_adapter.array
    sampling_rate = signal_adapter.resolve_fact("sampling_rate", sampling_rate)

    x, xp, _ = dispatch(x)
    x, was_1d = as_2d(x, name="samples")
    C, N = x.shape

    # Normalize ``frequency`` to a per-channel (C,) host array.  A scalar is
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
    actual = [
        float(grid_frequency(f, sampling_rate=sampling_rate, num_samples=N))
        for f in f_req
    ]
    for f_in, f_out in zip(f_req, actual):
        if abs(f_out - f_in) > 1e-12 * max(1.0, abs(f_in)):
            logger.warning(
                "add_pilot_tone: requested %.3f Hz quantized to %.3f Hz "
                "(grid step fs/N=%.3f Hz) for buffer-periodic "
                "(loop-seamless) playback.",
                f_in,
                f_out,
                df,
            )

    # Normalize ``power_ratio_db`` to a per-channel (C,) list, mirroring how
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

    # Per-channel signal power and the tone amplitude that realizes the PSR.
    p_signal = xp.mean(xp.abs(x) ** 2, axis=-1, keepdims=True)  # (C, 1) float
    psr_lin = (10.0 ** (xp.asarray(psr_req, dtype=xp.float64) / 10.0)).reshape(
        C, 1
    )  # (C, 1)
    amp = xp.sqrt(p_signal * psr_lin)  # (C, 1)

    # Per-channel phase ramp (C, N) in float64, wrapped by _phasor before the
    # trigonometry in the samples' precision (long ramps keep their accuracy).
    n = xp.arange(N, dtype=xp.float64)  # (N,)
    f_ch = xp.asarray(actual, dtype=xp.float64).reshape(C, 1)  # (C, 1)
    phase = 2.0 * np.pi * f_ch * n[None, :] / sampling_rate + phase_init  # (C, N)
    tone = _phasor(phase, np.result_type(x.dtype, np.complex64), xp)
    tone = tone.astype(x.dtype, copy=False)  # (C, N); real input keeps the cosine
    out = x + amp.astype(x.dtype) * tone  # (C, N)

    if renormalize:
        # Restore each channel to its original mean power.
        p_out = xp.mean(xp.abs(out) ** 2, axis=-1, keepdims=True)  # (C, 1)
        out = out * xp.sqrt(p_signal / p_out).astype(x.dtype)

    f_log = f"{actual[0]:.3g} Hz" if scalar_input else f"{actual} Hz"
    psr_log = f"{psr_req[0]:.1f} dB" if scalar_power else f"{psr_req} dB"
    logger.info(
        "add_pilot_tone: f_p=%s, PSR=%s, phase_init=%.3g rad, "
        "renormalize=%s [C=%s, N=%s]",
        f_log,
        psr_log,
        phase_init,
        renormalize,
        C,
        N,
    )

    samples_out = restore_1d(was_1d, out)
    return signal_adapter.wrap_samples(samples_out)


def welch_psd(
    samples: ArrayType | Signal,
    *,
    sampling_rate: float | None = None,
    nperseg: int = 256,
    detrend: str | bool | None = False,
    average: str | None = "mean",
    window: str | tuple[Any, ...] | Any = "hann",
    noverlap: int | None = None,
    nfft: int | None = None,
    scaling: str = "density",
    return_onesided: bool | None = None,
) -> tuple[ArrayType, ArrayType]:
    """Estimate the power spectral density (PSD) with Welch's method.

    Averaging the periodograms of overlapping, windowed segments gives a
    lower-variance estimate than one periodogram of the record
    (``scipy.signal.welch`` on the input's device).

    Parameters
    ----------
    samples : array_like or Signal
        Input samples, ``(N,)`` or ``(C, N)``.
    sampling_rate : float, optional
        Sampling rate in Hz.  Taken from the Signal; required for array
        input.  A value that disagrees with the Signal raises.
    nperseg : int, default 256
        Segment length in samples: longer segments resolve frequency more
        finely, at a higher variance (fewer segments to average).
    detrend : str or bool, default False
        How to detrend each segment (``"constant"``, ``"linear"``, or
        ``False`` for none).
    average : {"mean", "median"}, default "mean"
        How the segment periodograms are averaged; the median is more robust
        to transient outliers.
    window : str or tuple or array_like, default "hann"
        Window; a string or tuple is passed to ``scipy.signal.get_window``.
    noverlap : int, optional
        Samples of overlap between segments; ``None`` gives ``nperseg // 2``.
    nfft : int, optional
        FFT length, for zero-padded segments; ``None`` gives ``nperseg``.
    scaling : {"density", "spectrum"}, default "density"
        Power spectral density (units² per Hz) or power spectrum (units²).
    return_onesided : bool, optional
        One-sided spectrum (``0`` to ``f_s/2``), only for real input.
        ``None`` gives one-sided for real and two-sided for complex input.

    Returns
    -------
    f : ndarray
        Frequencies in Hz, ``(F,)``, ascending (a two-sided spectrum runs
        from ``-f_s/2``).
    Pxx : ndarray
        Linear PSD (or power spectrum), ``(F,)`` or ``(C, F)``.

    Raises
    ------
    ValueError
        If ``return_onesided=True`` for complex input.
    """
    signal_adapter = adapt_signal(samples, function_name="welch_psd()")
    x = signal_adapter.array
    sampling_rate = signal_adapter.resolve_fact("sampling_rate", sampling_rate)
    axis = -1

    x, xp, sp = dispatch(x)
    is_complex = xp.iscomplexobj(x)

    # scipy.signal.welch returns onesided by default for real, two-sided for complex
    # unless return_onesided is explicitly set.
    # Note: scipy's return_onesided argument serves to force one-sided for real data.
    # It cannot force one-sided for complex data (always raises error).
    # For complex data, it always returns two-sided (0 to fs).
    return_onesided, shift = _validate_and_shift(xp, is_complex, return_onesided, "PSD")

    f, Pxx = sp.signal.welch(
        x,
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

    f, Pxx = shift(f, (Pxx, axis))
    return f, Pxx


@dataclass(frozen=True)
class Spectrogram:
    """Result of :func:`spectrogram`.

    Attributes
    ----------
    frequencies : ndarray
        Frequencies in Hz, ``(F,)``, ascending (two-sided spectra are
        fftshifted).
    times : ndarray
        Segment center times in seconds, ``(T,)``.
    values : ndarray
        ``(F, T)`` or ``(C, F, T)``; power density, power, complex, magnitude
        or phase depending on ``mode``.
    """

    frequencies: ArrayType
    times: ArrayType
    values: ArrayType


def spectrogram(
    samples: ArrayType | Signal,
    *,
    sampling_rate: float | None = None,
    window: str | tuple[Any, ...] | Any = "hann",
    nperseg: int = 256,
    noverlap: int | None = None,
    nfft: int | None = None,
    detrend: str | bool | None = False,
    return_onesided: bool | None = None,
    scaling: str = "density",
    mode: str = "psd",
) -> Spectrogram:
    """Compute a spectrogram: the spectra of consecutive segments.

    ``scipy.signal.spectrogram`` on the input's device, with the frequency
    axis of a two-sided result in ascending order.

    Parameters
    ----------
    samples : array_like or Signal
        Input samples, ``(N,)`` or ``(C, N)``.
    sampling_rate : float, optional
        Sampling rate in Hz.  Taken from the Signal; required for array
        input.  A value that disagrees with the Signal raises.
    window : str or tuple or array_like, default "hann"
        Window; a string or tuple is passed to ``scipy.signal.get_window``.
    nperseg : int, default 256
        Segment length in samples: longer segments resolve frequency more
        finely and time more coarsely.
    noverlap : int, optional
        Samples of overlap between segments; ``None`` gives ``nperseg // 8``
        (SciPy's default for spectrograms).
    nfft : int, optional
        FFT length, for zero-padded segments; ``None`` gives ``nperseg``.
    detrend : str or bool, default False
        How to detrend each segment (``"constant"``, ``"linear"``, or
        ``False`` for none).
    return_onesided : bool, optional
        One-sided spectrum (``0`` to ``f_s/2``), only for real input.
        ``None`` gives one-sided for real and two-sided for complex input.
    scaling : {"density", "spectrum"}, default "density"
        Power spectral density (units² per Hz) or power spectrum (units²).
    mode : {"psd", "complex", "magnitude", "angle", "phase"}, default "psd"
        What ``values`` holds: power, the complex STFT, its magnitude, its
        wrapped angle or its unwrapped phase.

    Returns
    -------
    Spectrogram
        ``frequencies``, ``times`` and ``values``.

    Raises
    ------
    ValueError
        If ``return_onesided=True`` for complex input.
    """
    signal_adapter = adapt_signal(samples, function_name="spectrogram()")
    x = signal_adapter.array
    sampling_rate = signal_adapter.resolve_fact("sampling_rate", sampling_rate)
    axis = -1

    x, xp, sp = dispatch(x)
    is_complex = xp.iscomplexobj(x)

    return_onesided, shift = _validate_and_shift(
        xp, is_complex, return_onesided, "spectrogram"
    )

    f, t, Sxx = sp.signal.spectrogram(
        x,
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

    # Sxx frequency axis is at position axis_pos in output
    axis_pos = axis % x.ndim
    f, Sxx = shift(f, (Sxx, axis_pos))
    return Spectrogram(frequencies=f, times=t, values=Sxx)
