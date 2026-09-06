"""Shared metrics and evaluation functions for tests."""

from typing import Any

import numpy as np

from commkit import backend
from commkit.mapping import gray_constellation
from tests.common.conversions import to_numpy


def calc_mse_db(y: Any, ref: Any, xp: Any = None) -> float:
    """Calculate Mean Squared Error in dB relative to reference signal power.

    Parameters
    ----------
    y : ArrayLike
        Estimated or recovered symbol array.
    ref : ArrayLike
        Reference symbol array.
    xp : module, optional
        Array module. If None, dispatched from `y`.

    Returns
    -------
    float
        Normalized MSE in dB.
    """
    if xp is None:
        xp = backend.get_array_module(y)
    err = float(xp.mean(xp.abs(y - ref) ** 2))
    sig = float(xp.mean(xp.abs(ref) ** 2))
    return float(10.0 * np.log10(err / (sig + 1e-30) + 1e-30))


def calc_tail_mse_db(error: Any, n: int = 400, xp: Any = None) -> float:
    """Calculate MSE in dB over the last `n` samples of an error sequence.

    Useful for validating equalizer convergence without initial transient penalty.

    Parameters
    ----------
    error : ArrayLike
        Error sequence from equalizer or adaptive filter.
    n : int
        Number of tail samples to evaluate.
    xp : module, optional
        Array module.

    Returns
    -------
    float
        Tail MSE in dB.
    """
    e = to_numpy(error)[-n:]
    return float(10.0 * np.log10(float(np.mean(np.abs(e) ** 2)) + 1e-30))


def calc_rms_phase_error(phase_est: Any, phase_true: Any, xp: Any = None) -> float:
    """Calculate Root Mean Square (RMS) phase error, removing any constant bias.

    Accounts for irreducible global phase ambiguity by subtracting mean bias.

    Parameters
    ----------
    phase_est : ArrayLike
        Estimated phase trajectory in radians.
    phase_true : ArrayLike
        True phase trajectory in radians.
    xp : module, optional
        Array module.

    Returns
    -------
    float
        Zero-mean RMS phase error in radians.
    """
    if xp is None:
        xp = backend.get_array_module(phase_est)
    err = phase_est - phase_true
    err = err - float(xp.mean(err))
    return float(xp.sqrt(xp.mean(err**2)))


def calc_dispersion(y: Any, order: int = 16, mod: str = "qam") -> float:
    """Calculate phase-blind radial dispersion to nearest constellation ring power.

    A convergence metric for blind equalizers (CMA/RDE) that carry residual phase ambiguity.

    Parameters
    ----------
    y : ArrayLike
        Equalizer output symbols.
    order : int
        Constellation modulation order (e.g. 4 for QPSK, 16 for 16-QAM).
    mod : str
        Modulation scheme name ('qam', 'psk', 'pam').

    Returns
    -------
    float
        Radial dispersion metric value.
    """
    const = gray_constellation(mod, order)
    const = const / np.sqrt(np.mean(np.abs(const) ** 2))
    r2 = np.abs(const) ** 2
    a2 = np.abs(to_numpy(y)) ** 2
    return float(np.mean(np.min((a2[:, None] - r2[None, :]) ** 2, axis=1)))


def calc_freq_response(taps: Any, nfft: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized frequencies and magnitude frequency response of a tap array.

    Parameters
    ----------
    taps : ArrayLike
        Filter impulse response coefficients.
    nfft : int
        FFT size.

    Returns
    -------
    freqs : np.ndarray
        Normalized frequencies in [-0.5, 0.5) cycles/sample.
    H : np.ndarray
        Magnitude response |H(f)|.
    """
    t_np = to_numpy(taps)
    H = np.abs(np.fft.fft(t_np, nfft))
    freqs = np.fft.fftfreq(nfft)
    return freqs, H
