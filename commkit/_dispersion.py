"""Chromatic dispersion physics (private).

Single source of the fiber dispersion model used by
``impairments.apply_chromatic_dispersion`` (the channel) and the receiver's
dispersion compensation (its inverse):

- ``beta2_length``: D [ps/(nm km)], L [km], λ [nm] to the β₂·L product [s²],
  ``β₂ = -D λ² / (2π c)``.
- ``transfer_function``: ``H(ω) = exp(-j β₂ L ω² / 2)`` on the FFT grid of
  ``num_samples`` samples, or its inverse ``exp(+j β₂ L ω² / 2)``.

Sign convention: NumPy's FFT uses the engineering kernel ``exp(-jωt)``, so a
field propagating through the fiber is multiplied by ``exp(-j β₂ L ω² / 2)``.
Its group delay ``τ(ω) = -dφ/dω = β₂ L ω`` makes the higher frequency
(shorter wavelength) arrive first in anomalous fiber (D > 0, β₂ < 0), with
``Δτ = D L Δλ``.
"""

from types import ModuleType

import numpy as np

from ._array import as_2d, restore_1d
from .backend import ArrayType, dispatch

# Speed of light used by the 1.x conversion, kept so results do not change.
_SPEED_OF_LIGHT = 2.998e8  # m/s


def beta2_length(
    *,
    dispersion_ps_nm_km: float,
    fiber_length_km: float,
    center_wavelength_nm: float,
) -> float:
    """The ``β₂ · L`` product in s² for D, L and λ in engineering units."""
    d = dispersion_ps_nm_km * 1e-12 / (1e-9 * 1e3)  # s/m²
    lam = center_wavelength_nm * 1e-9  # m
    length = fiber_length_km * 1e3  # m
    return float(-(d * lam**2) / (2.0 * np.pi * _SPEED_OF_LIGHT) * length)


def transfer_function(
    num_samples: int,
    *,
    sampling_rate: float,
    beta2_length: float,
    inverse: bool,
    xp: ModuleType,
) -> ArrayType:
    """Fiber transfer function on the ``fftfreq`` grid (or its inverse)."""
    omega = 2.0 * np.pi * xp.fft.fftfreq(num_samples, d=1.0 / sampling_rate)
    sign = 1.0 if inverse else -1.0
    return xp.exp(sign * 1j * (beta2_length / 2.0) * omega**2)


def apply_dispersion(
    samples: ArrayType,
    *,
    sampling_rate: float,
    dispersion_ps_nm_km: float,
    fiber_length_km: float,
    center_wavelength_nm: float,
    inverse: bool,
) -> ArrayType:
    """Filter ``(N,)`` or ``(C, N)`` samples with the fiber response or its
    inverse, keeping shape, dtype and device."""
    samples, xp, _ = dispatch(samples)
    samples, was_1d = as_2d(samples, name="samples")
    h = transfer_function(
        samples.shape[-1],
        sampling_rate=sampling_rate,
        beta2_length=beta2_length(
            dispersion_ps_nm_km=dispersion_ps_nm_km,
            fiber_length_km=fiber_length_km,
            center_wavelength_nm=center_wavelength_nm,
        ),
        inverse=inverse,
        xp=xp,
    )
    result = xp.fft.ifft(xp.fft.fft(samples, axis=-1) * h[None, :], axis=-1)
    if result.dtype != samples.dtype:
        result = result.astype(samples.dtype)
    return restore_1d(was_1d, result)
