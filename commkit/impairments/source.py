"""Laser and oscillator phase noise.

:func:`generate_phase_noise` draws a phase trajectory from the power-law
FM-noise model (white FM, the Lorentzian linewidth, plus optional 1/f
flicker FM) and :func:`apply_phase_noise` rotates the samples by one.
"""

import math
from types import ModuleType

import numpy as np

from .. import _fft
from .._array import as_2d, restore_1d
from .._random import standard_normal
from ..backend import ArrayType, _module_for, dispatch
from ..core._signal_adapter import S, adapt_signal, require_device
from ..logger import logger
from ..math import _derotate

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
    xp: ModuleType,
    periodic: bool = False,
) -> ArrayType:
    """float64 phase trajectories on ``xp``'s device with one-sided FM-noise PSD

        S_f(f) = linewidth / pi  +  flicker / f      [Hz^2/Hz].

    The white-FM part is generated exactly as a discrete Wiener walk
    (per-sample increments N(0, 2*pi*linewidth/f_s)); the flicker part by
    spectral shaping of white frequency noise.  Both draw float32 noise and
    accumulate in float64.  Like AWGN, the noise is drawn
    on the device: a seed gives the same statistics on every device, not the
    same realization.

    ``periodic`` takes the same realization and subtracts one constant from
    every increment so that they sum to a multiple of 2π: the trajectory then
    closes on itself, and the step from the last sample back to the first is
    an ordinary increment.
    """
    num_samples = shape[-1]
    phi = xp.zeros(shape, dtype=xp.float64)

    # The random steps are drawn in float32 (half the generator's work and
    # memory) and summed in float64: the walk reaches thousands of radians,
    # where float32 would resolve only about 1e-4 rad.
    if linewidth > 0.0:
        std = math.sqrt(2.0 * math.pi * linewidth / sampling_rate)
        steps = standard_normal(rng, shape, dtype=np.float32, xp=xp)
        phi += std * xp.cumsum(steps, axis=-1, dtype=xp.float64)

    if flicker > 0.0:
        f = xp.fft.rfftfreq(num_samples, 1.0 / sampling_rate)
        f_min = (
            flicker_f_min if flicker_f_min is not None else sampling_rate / num_samples
        )
        # A unit-variance white input has one-sided PSD 2/f_s, so shaping to
        # S_f = flicker/f requires the amplitude gain sqrt(flicker/f * f_s/2).
        f32 = xp.maximum(f, f_min).astype(xp.float32)
        gain = xp.sqrt(xp.float32(flicker) / f32) * xp.float32(
            math.sqrt(sampling_rate / 2.0)
        )
        white = standard_normal(rng, shape, dtype=np.float32, xp=xp)
        spectrum = _fft.rfft(white, axis=-1) * gain
        df = _fft.irfft(spectrum, num_samples, axis=-1)
        phi += (2.0 * math.pi / sampling_rate) * xp.cumsum(
            df, axis=-1, dtype=xp.float64
        )

    if periodic:
        # phi[-1] is the sum of all N increments, the wrap step included.
        two_pi = 2.0 * math.pi
        excess = phi[..., -1:] - two_pi * xp.round(phi[..., -1:] / two_pi)
        phi -= excess * (xp.arange(1, num_samples + 1, dtype=xp.float64) / num_samples)

    return phi


def generate_phase_noise(
    *,
    num_samples: int,
    sampling_rate: float,
    linewidth: float = 0.0,
    flicker: float = 0.0,
    flicker_f_min: float | None = None,
    num_channels: int = 1,
    periodic: bool = False,
    rng: int | np.random.Generator | None = None,
    device: str = "cpu",
) -> ArrayType:
    """Generate laser or oscillator phase-noise trajectories ``φ[n]`` in radians.

    The instantaneous-frequency (FM) noise follows the power-law model with a
    white and a flicker component (one-sided, in Hz²/Hz):

        S_f(f) = linewidth / π  +  flicker / f

    * ``linewidth`` is the Lorentzian (white-FM, Wiener) linewidth ``Δν``:
      the phase is a random walk with per-sample increments
      ``N(0, 2π Δν / f_s)``, and the field spectrum is a Lorentzian of FWHM
      ``Δν``.
    * ``flicker`` is the 1/f FM coefficient ``h_-1``: technical noise
      (current source, temperature, acoustics) that dominates below the
      corner frequency ``f_c = π h_-1 / Δν`` where the two terms cross.

    The trajectory itself (rather than a rotated signal) is the ground truth
    for validating estimators.  :func:`apply_phase_noise` draws and applies
    one in a single step; to apply this one, rotate in the samples'
    precision (``samples * xp.exp(1j * phi)`` promotes complex64 to
    complex128).

    Parameters
    ----------
    num_samples : int
        Trajectory length per stream.
    sampling_rate : float
        Sampling rate in Hz.
    linewidth : float, default 0.0
        White-FM (Lorentzian) linewidth ``Δν`` in Hz.
    flicker : float, default 0.0
        Flicker-FM coefficient ``h_-1`` in Hz² (one-sided ``S_f = h_-1 / f``).
    flicker_f_min : float, optional
        Frequency below which the flicker shaping is held flat (the 1/f
        divergence must be capped).  Defaults to the record resolution
        ``sampling_rate / num_samples``.
    num_channels : int, default 1
        Number of independent trajectories.
    periodic : bool, default False
        Make each trajectory one period of a periodic phase (modulo 2π), so
        a record carrying it loops seamlessly, e.g. on a streaming AWG.  See
        Notes.
    rng : int, numpy.random.Generator or None
        Random source (an int seeds ``numpy.random.default_rng``); the noise
        itself is drawn on ``device``.
    device : {"cpu", "gpu"}, default "cpu"
        Where the trajectory is drawn and returned.

    Returns
    -------
    ndarray
        Phase in radians, ``float64``, on ``device``.  Shape
        ``(num_samples,)`` for ``num_channels=1``, else
        ``(num_channels, num_samples)``.

    Notes
    -----
    The noise is drawn on ``device``, like AWGN: a seed gives the same
    trajectory on every run on one device, but CPU and GPU realizations
    differ (their statistics agree).  For the identical trajectory on both,
    generate it on the CPU and move it with ``to_device``.

    A random walk does not end where it started, so a looped record would
    jump in phase at every repetition.  ``periodic=True`` subtracts the same
    constant from every increment so that the N increments, the one from the
    last sample back to the first included, sum to a multiple of 2π.  This
    is a frequency offset of at most ``sampling_rate / (2 * num_samples)``;
    the statistics at lags much shorter than the record are unchanged.
    """
    device = require_device(device, "generate_phase_noise()")
    logger.info(
        "Generating phase noise (linewidth=%.3g Hz, flicker=%.3g Hz², "
        "%s stream(s), periodic=%s).",
        linewidth,
        flicker,
        num_channels,
        periodic,
    )

    phi = _phase_trajectory(
        (num_channels, num_samples),
        sampling_rate,
        linewidth,
        flicker,
        flicker_f_min,
        np.random.default_rng(rng),
        _module_for(device),
        periodic,
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
    periodic: bool = False,
    rng: int | np.random.Generator | None = None,
) -> S:
    """Add laser or oscillator phase noise.

    Each sample is rotated by a phase drawn from the power-law FM-noise
    model of :func:`generate_phase_noise` (a white-FM Wiener walk plus
    optional 1/f flicker): ``r[n] = s[n] · exp(j φ[n])``.

    Parameters
    ----------
    samples : array_like or Signal
        Complex baseband samples, ``(N,)`` or ``(C, N)``.
    linewidth : float
        Combined transmitter and receiver laser linewidth ``Δν`` in Hz;
        typically 100 kHz (narrow-linewidth laser) to 10 MHz (DFB).
    sampling_rate : float, optional
        Sampling rate in Hz.  Taken from the Signal; required for array
        input.  A value that disagrees with the Signal raises.
    flicker : float, default 0.0
        Flicker-FM coefficient ``h_-1`` in Hz² (one-sided ``S_f = h_-1 / f``).
    flicker_f_min : float, optional
        Low-frequency cap for the flicker shaping; see
        :func:`generate_phase_noise`.
    shared_lo : bool, default False
        ``False``: each channel gets independent phase noise (separate
        lasers per path).  ``True``: one trajectory is shared by all
        channels (a common local oscillator).
    periodic : bool, default False
        Treat ``samples`` as one period of a looped record (e.g. for a
        streaming AWG) and make the phase noise periodic too; see
        :func:`generate_phase_noise`.
    rng : int, numpy.random.Generator or None
        Random source for the trajectories, drawn on the input's device
        (see :func:`generate_phase_noise`).

    Returns
    -------
    ndarray or Signal
        The rotated samples, same shape, dtype and device as the input.

    Examples
    --------
    >>> noisy = apply_phase_noise(x, linewidth=100e3, sampling_rate=fs, rng=0)
    >>> noisy = apply_phase_noise(sig, linewidth=100e3)  # Signal input
    """
    signal_adapter = adapt_signal(samples, function_name="apply_phase_noise()")
    sampling_rate = signal_adapter.resolve_fact("sampling_rate", sampling_rate)

    logger.info(
        "Applying phase noise (linewidth=%.3g Hz, flicker=%.3g Hz², shared_lo=%s, "
        "periodic=%s).",
        linewidth,
        flicker,
        shared_lo,
        periodic,
    )

    x, xp, _ = dispatch(signal_adapter.array)
    x, was_1d = as_2d(x, name="samples")
    C, N = x.shape

    num_trajectories = 1 if shared_lo else C
    phase = _phase_trajectory(
        (num_trajectories, N),
        sampling_rate,
        linewidth,
        flicker,
        flicker_f_min,
        np.random.default_rng(rng),
        xp,
        periodic,
    )
    # x · e^{jφ} in the samples' precision ((1, N) broadcasts across channels).
    result = _derotate(x, -phase)

    return signal_adapter.wrap_samples(restore_1d(was_1d, result))
