"""Chromatic dispersion physics (private).

Single source of the fiber dispersion model used by
``impairments.apply_chromatic_dispersion`` (the channel) and
``filtering.correct_chromatic_dispersion`` (its inverse):

- ``beta2_length``: the accumulated dispersion D·L [ps/nm] and λ [nm] to the
  β₂·L product [s²], ``β₂ = -D λ² / (2π c)``; ``beta3_length`` adds the
  accumulated slope S·L [ps/nm²] for the β₃·L product [s³]
  (``commkit._fiber``).  The channel passes D·L of a fiber, the receiver the
  accumulated dispersion it knows.
- ``dispersion_phase``: the phase ``-(β₂Lω²/2 + β₃Lω³/6)`` on the FFT grid
  (also used by ``commkit._fiber``).
- ``transfer_function``: ``H(ω) = exp(-j β₂ L ω² / 2)`` on the FFT grid of
  ``num_samples`` samples, or its inverse ``exp(+j β₂ L ω² / 2)``.
- ``spread_samples`` and ``compensation_taps``: the group-delay spread over
  the sampled band and the FIR form of the inverse
  (``filtering.dispersion_compensation_taps``).

Sign convention: NumPy's FFT uses the engineering kernel ``exp(-jωt)``, so a
field propagating through the fiber is multiplied by ``exp(-j β₂ L ω² / 2)``.
Its group delay ``τ(ω) = -dφ/dω = β₂ L ω`` makes the higher frequency
(shorter wavelength) arrive first in anomalous fiber (D > 0, β₂ < 0), with
``Δτ = D L Δλ``.
"""

from types import ModuleType
from typing import Any

import numpy as np
import scipy.linalg
import scipy.special

from . import _fft
from ._array import as_2d, restore_1d
from ._constants import SPEED_OF_LIGHT as _SPEED_OF_LIGHT
from .backend import ArrayType, dispatch
from .math import _phasor


def beta2_length(*, dispersion_ps_nm: float, center_wavelength_nm: float) -> float:
    """The ``β₂ · L`` product in s² for the accumulated dispersion D·L in ps/nm."""
    dl = dispersion_ps_nm * 1e-12 / 1e-9  # s/m
    lam = center_wavelength_nm * 1e-9  # m
    return float(-(dl * lam**2) / (2.0 * np.pi * _SPEED_OF_LIGHT))


def beta3_length(
    *,
    dispersion_ps_nm: float,
    dispersion_slope_ps_nm2: float,
    center_wavelength_nm: float,
) -> float:
    """The ``β₃ · L`` product in s³ for D·L, the accumulated slope S·L and λ.

    From ``D = -(2πc/λ²) β₂`` and ``dω/dλ = -2πc/λ²``:
    ``β₃ = (S + 2D/λ) · (λ² / (2πc))²``, times the length.
    """
    dl = dispersion_ps_nm * 1e-12 / 1e-9  # s/m
    sl = dispersion_slope_ps_nm2 * 1e-12 / (1e-9 * 1e-9)  # s/m²
    lam = center_wavelength_nm * 1e-9  # m
    k = lam**2 / (2.0 * np.pi * _SPEED_OF_LIGHT)
    return float((sl + 2.0 * dl / lam) * k**2)


def dispersion_phase(
    num_samples: int,
    *,
    sampling_rate: float,
    beta2_length: float,
    beta3_length: float = 0.0,
    xp: ModuleType,
) -> ArrayType:
    """Propagation phase ``-(β₂L ω²/2 + β₃L ω³/6)`` on the ``fftfreq`` grid.

    float64 on ``xp``'s device, because it reaches thousands of radians.
    """
    omega = 2.0 * np.pi * xp.fft.fftfreq(num_samples, d=1.0 / sampling_rate)
    phase = (beta2_length / 2.0) * omega**2
    if beta3_length:
        phase += (beta3_length / 6.0) * omega**3
    return -phase


def transfer_function(
    num_samples: int,
    *,
    sampling_rate: float,
    beta2_length: float,
    inverse: bool,
    xp: ModuleType,
    dtype: Any = np.complex128,
) -> ArrayType:
    """Fiber transfer function on the ``fftfreq`` grid (or its inverse).

    The phase is computed in float64 and the response is returned as
    ``dtype``, so complex64 data is filtered in single precision instead of
    being promoted.
    """
    phase = dispersion_phase(
        num_samples, sampling_rate=sampling_rate, beta2_length=beta2_length, xp=xp
    )
    return _phasor(-phase if inverse else phase, dtype, xp)


def spread_samples(
    beta2_length: float, *, sampling_rate: float, bandwidth: float
) -> float:
    """Group-delay spread over a band, in samples.

    ``τ(ω) = β₂ L ω`` over ``|f| <= bandwidth / 2`` spans
    ``2π |β₂ L| · bandwidth`` seconds; over the whole sampled band
    (``bandwidth = fs``) this is the FIR length that covers the whole spread.
    """
    return 2.0 * np.pi * abs(beta2_length) * bandwidth * sampling_rate


def _band_integral(a: float, n: np.ndarray, half_band: float) -> np.ndarray:
    """``(1/2π) ∫_{-W}^{W} exp(j(a θ² + n θ)) dθ`` by Fresnel integrals.

    Completing the square, ``a θ² + n θ = a (θ + n/2a)² - n²/4a``.
    """
    scale = np.sqrt(2.0 * abs(a) / np.pi)
    shift = n / (2.0 * a)
    s_hi, c_hi = scipy.special.fresnel(scale * (half_band + shift))
    s_lo, c_lo = scipy.special.fresnel(scale * (-half_band + shift))
    integral = np.sqrt(np.pi / (2.0 * abs(a))) * (
        (c_hi - c_lo) + 1j * np.sign(a) * (s_hi - s_lo)
    )
    phase = np.exp(-1j * n**2 / (4.0 * a))
    return np.asarray(phase * integral / (2.0 * np.pi), dtype=np.complex128)


#: Tikhonov term added to the diagonal of the band's Gram matrix (whose
#: diagonal is ``bandwidth / fs``): bounds the gain outside the band.
_LS_REGULARIZATION = 1e-6


def compensation_taps(
    beta2_length: float, *, sampling_rate: float, bandwidth: float, num_taps: int
) -> np.ndarray:
    """Least-squares FIR fit of the inverse response over a band.

    Minimizes ``∫_{|θ|<W} |Σ h[n] e^{-jθn} - exp(j a θ²)|² dθ`` with
    ``θ = ω / fs``, ``W = π · bandwidth / fs`` and ``a = β₂ L fs² / 2``, for
    ``n`` centred on tap ``(num_taps - 1) // 2``: ``Q h = p`` with the Toeplitz Gram matrix
    ``Q[m, k] = (W/π) sinc(W (m-k) / π)`` and ``p[m]`` the band integral of
    the response, exact through Fresnel integrals.  Over the whole band
    (``W = π``) ``Q`` is the identity and ``h = p``, the truncated
    band-limited impulse response.  Host complex128.
    """
    n = np.arange(num_taps) - (num_taps - 1) // 2
    a = beta2_length * sampling_rate**2 / 2.0
    if a == 0.0:
        return np.asarray(n == 0, dtype=np.complex128)
    half_band = np.pi * bandwidth / sampling_rate
    p = _band_integral(a, n, half_band)
    if bandwidth >= sampling_rate:
        return p
    col = (half_band / np.pi) * np.sinc(half_band * np.arange(num_taps) / np.pi)
    col[0] += _LS_REGULARIZATION
    return np.asarray(scipy.linalg.solve_toeplitz(col, p), dtype=np.complex128)


def apply_dispersion(
    samples: ArrayType,
    *,
    sampling_rate: float,
    dispersion_ps_nm: float,
    center_wavelength_nm: float,
    inverse: bool,
) -> ArrayType:
    """Filter samples with the fiber's dispersion response or its inverse.

    ``(N,)`` or ``(C, N)`` samples keep their shape, dtype and device.
    """
    samples, xp, _ = dispatch(samples)
    samples, was_1d = as_2d(samples, name="samples")
    h = transfer_function(
        samples.shape[-1],
        sampling_rate=sampling_rate,
        beta2_length=beta2_length(
            dispersion_ps_nm=dispersion_ps_nm,
            center_wavelength_nm=center_wavelength_nm,
        ),
        inverse=inverse,
        xp=xp,
        dtype=np.result_type(samples.dtype, np.complex64),
    )
    result = _fft.ifft(_fft.fft(samples, axis=-1) * h[None, :], axis=-1)
    if result.dtype != samples.dtype:
        result = result.astype(samples.dtype)
    return restore_1d(was_1d, result)
