"""Optical/electronic source impairments (laser/oscillator phase noise)."""

import math

import numpy as np

from .._array import as_2d, restore_1d
from ..backend import dispatch
from ..core._signal_adapter import S, adapt_signal
from ..logger import logger

__all__ = ["apply_phase_noise", "generate_phase_noise"]

# -----------------------------------------------------------------------------
# PHASE-NOISE SYNTHESIS (array-only)
# -----------------------------------------------------------------------------
# generate_phase_noise: builds a phase trajectory from parameters - there is
# no Signal yet to unwrap (same category as the other generate_* factories).


def _phase_trajectory(
    shape: tuple[int, int],
    sampling_rate: float,
    linewidth: float,
    flicker: float,
    flicker_f_min: float | None,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    NumPy float64 phase trajectories with one-sided FM-noise PSD

        S_f(f) = linewidth / pi  +  flicker / f      [Hz^2/Hz].

    The white-FM part is generated exactly as a discrete Wiener walk
    (per-sample increments N(0, 2*pi*linewidth/f_s)); the flicker part by
    spectral shaping of white frequency noise.  Generated on the CPU so a
    given seed yields the identical trajectory on every backend.
    """
    num_samples = shape[-1]
    phi = np.zeros(shape, dtype=np.float64)

    if linewidth > 0.0:
        std = math.sqrt(2.0 * math.pi * linewidth / sampling_rate)
        phi += np.cumsum(rng.normal(0.0, std, shape), axis=-1)

    if flicker > 0.0:
        f = np.fft.rfftfreq(num_samples, 1.0 / sampling_rate)
        f_min = (
            flicker_f_min if flicker_f_min is not None else sampling_rate / num_samples
        )
        # A unit-variance white input has one-sided PSD 2/f_s, so shaping to
        # S_f = flicker/f requires the amplitude gain sqrt(flicker/f * f_s/2).
        gain = np.sqrt(flicker / np.maximum(f, f_min)) * math.sqrt(sampling_rate / 2.0)
        spec = np.fft.rfft(rng.normal(0.0, 1.0, shape), axis=-1)
        df = np.fft.irfft(spec * gain, num_samples, axis=-1)
        phi += 2.0 * math.pi * np.cumsum(df, axis=-1) / sampling_rate

    return phi


def generate_phase_noise(
    *,
    num_samples: int,
    sampling_rate: float,
    linewidth: float = 0.0,
    flicker: float = 0.0,
    flicker_f_min: float | None = None,
    num_channels: int = 1,
    rng: int | np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generates laser/oscillator phase-noise trajectories phi[n] in radians.

    The instantaneous-frequency (FM) noise follows the standard power-law
    model with a white and a flicker component:

        S_f(f) = linewidth / pi  +  flicker / f      [Hz^2/Hz, one-sided]

    * ``linewidth`` is the Lorentzian (white-FM / Wiener) linewidth
      delta_nu: the phase performs a random walk with per-sample increments
      N(0, 2*pi*delta_nu / f_s) and the field spectrum is a Lorentzian of
      FWHM delta_nu.
    * ``flicker`` is the 1/f FM coefficient h_-1: technical noise (current
      source, temperature, acoustics) that dominates below the corner
      frequency f_c = pi * h_-1 / delta_nu where the two terms cross.

    Returning the trajectory itself (rather than a rotated signal) makes the
    ground truth available for estimator validation; apply it with
    ``samples * xp.exp(1j * phi)`` or via :func:`apply_phase_noise`.

    Parameters
    ----------
    num_samples : int
        Trajectory length per stream.
    sampling_rate : float
        Sampling rate in Hz.
    linewidth : float, default 0.0
        White-FM (Lorentzian) linewidth delta_nu in Hz.
    flicker : float, default 0.0
        Flicker-FM coefficient h_-1 in Hz^2 (one-sided ``S_f = h_-1 / f``).
    flicker_f_min : float, optional
        Frequency below which the flicker shaping is held flat (the 1/f
        divergence must be capped).  Defaults to the record resolution
        ``sampling_rate / num_samples``.
    num_channels : int, default 1
        Number of independent trajectories.
    rng : int, numpy.random.Generator or None
        Random source (an int seeds ``numpy.random.default_rng``).

    Returns
    -------
    numpy.ndarray
        Phase in radians, ``float64``, on the host (move it with
        ``to_device``).  Shape ``(num_samples,)`` for ``num_channels=1``,
        else ``(num_channels, num_samples)``.

    Notes
    -----
    The trajectory is generated on the host with a NumPy Generator, so a
    given seed gives the same trajectory whichever device it is used on.
    """
    logger.info(
        "Generating phase noise (linewidth=%.3g Hz, flicker=%.3g Hz², %s stream(s)).",
        linewidth,
        flicker,
        num_channels,
    )

    phi = _phase_trajectory(
        (num_channels, num_samples),
        sampling_rate,
        linewidth,
        flicker,
        flicker_f_min,
        np.random.default_rng(rng),
    )
    if num_channels == 1:
        phi = phi[0]
    return phi


# -----------------------------------------------------------------------------
# PHASE-NOISE APPLICATION (Signal-aware)
# -----------------------------------------------------------------------------


def apply_phase_noise(
    samples: S,
    *,
    linewidth: float,
    sampling_rate: float | None = None,
    flicker: float = 0.0,
    flicker_f_min: float | None = None,
    shared_lo: bool = False,
    rng: int | np.random.Generator | None = None,
) -> S:
    """
    Adds laser / oscillator phase noise to a signal.

    Each sample is rotated by an accumulated phase drawn from the power-law
    FM-noise model of :func:`generate_phase_noise` (white-FM Wiener walk
    plus optional 1/f flicker):

        r[n] = s[n] * exp(j * phi[n])

    Parameters
    ----------
    samples : array_like or Signal
        Complex baseband signal. Shape: ``(N,)`` (SISO) or ``(C, N)`` (MIMO).
    linewidth : float
        Combined transmitter + receiver laser linewidth delta_nu in Hz.
        Typical values: 100 kHz (narrow-linewidth laser) to 10 MHz (DFB).
    sampling_rate : float, optional
        Sampling rate in Hz.  Taken from the Signal; required for array
        input.  A value that disagrees with the Signal raises.
    flicker : float, default 0.0
        Flicker-FM coefficient h_-1 in Hz^2 (one-sided ``S_f = h_-1 / f``).
    flicker_f_min : float, optional
        Low-frequency cap for the flicker shaping; see
        :func:`generate_phase_noise`.
    shared_lo : bool, default False
        When ``False`` (default), each channel receives independent phase noise
        (separate oscillators / lasers per TX-RX path).
        When ``True``, a single phase noise trajectory is shared across all
        channels (common local oscillator in a coherent system).
    rng : int, numpy.random.Generator or None
        Random source for the trajectories (drawn on the host).

    Returns
    -------
    array_like or Signal
        Phase-noise-impaired signal, same shape, dtype, and backend as input.
        A :class:`Signal` returns a new impaired :class:`Signal`.

    Examples
    --------
    >>> noisy = apply_phase_noise(x, linewidth=100e3, sampling_rate=fs, rng=0)
    >>> noisy = apply_phase_noise(sig, linewidth=100e3)  # Signal input
    """
    signal_adapter = adapt_signal(samples, function_name="apply_phase_noise()")
    sampling_rate = signal_adapter.resolve_fact("sampling_rate", sampling_rate)

    logger.info(
        "Applying phase noise (linewidth=%.3g Hz, flicker=%.3g Hz², shared_lo=%s).",
        linewidth,
        flicker,
        shared_lo,
    )

    x, xp, _ = dispatch(signal_adapter.array)
    x, was_1d = as_2d(x, name="samples")
    C, N = x.shape

    num_trajectories = 1 if shared_lo else C
    phase = xp.asarray(
        _phase_trajectory(
            (num_trajectories, N),
            sampling_rate,
            linewidth,
            flicker,
            flicker_f_min,
            np.random.default_rng(rng),
        )
    )
    result = x * xp.exp(1j * phase)  # (1, N) broadcasts across channels

    if result.dtype != x.dtype:
        result = result.astype(x.dtype)

    return signal_adapter.wrap_samples(restore_1d(was_1d, result))
