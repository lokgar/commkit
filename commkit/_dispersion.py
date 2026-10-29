"""Chromatic dispersion physics (private).

Single source of the fiber dispersion model used by
``impairments.apply_chromatic_dispersion`` (the channel) and
``filtering.correct_chromatic_dispersion`` (its inverse):

- ``beta2_length``: D [ps/(nm km)], L [km], λ [nm] to the β₂·L product [s²],
  ``β₂ = -D λ² / (2π c)``; ``beta3_length`` adds the slope S [ps/(nm² km)]
  for the β₃·L product [s³] (``commkit._fiber``).
- ``dispersion_phase``: the phase ``-(β₂Lω²/2 + β₃Lω³/6)`` on the FFT grid
  (also used by ``commkit._fiber``).
- ``transfer_function``: ``H(ω) = exp(-j β₂ L ω² / 2)`` on the FFT grid of
  ``num_samples`` samples, or its inverse ``exp(+j β₂ L ω² / 2)``.

Sign convention: NumPy's FFT uses the engineering kernel ``exp(-jωt)``, so a
field propagating through the fiber is multiplied by ``exp(-j β₂ L ω² / 2)``.
Its group delay ``τ(ω) = -dφ/dω = β₂ L ω`` makes the higher frequency
(shorter wavelength) arrive first in anomalous fiber (D > 0, β₂ < 0), with
``Δτ = D L Δλ``.
"""

from types import ModuleType
from typing import Any

import numpy as np

from . import _fft
from ._array import as_2d, restore_1d
from ._constants import SPEED_OF_LIGHT as _SPEED_OF_LIGHT
from .backend import ArrayType, dispatch
from .math import _phasor


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


def beta3_length(
    *,
    dispersion_ps_nm_km: float,
    dispersion_slope_ps_nm2_km: float,
    fiber_length_km: float,
    center_wavelength_nm: float,
) -> float:
    """The ``β₃ · L`` product in s³ for D, the slope S = dD/dλ, L and λ.

    From ``D = -(2πc/λ²) β₂`` and ``dω/dλ = -2πc/λ²``:
    ``β₃ = (S + 2D/λ) · (λ² / (2πc))²``.
    """
    d = dispersion_ps_nm_km * 1e-12 / (1e-9 * 1e3)  # s/m²
    slope = dispersion_slope_ps_nm2_km * 1e-12 / (1e-9 * 1e-9 * 1e3)  # s/m³
    lam = center_wavelength_nm * 1e-9  # m
    length = fiber_length_km * 1e3  # m
    k = lam**2 / (2.0 * np.pi * _SPEED_OF_LIGHT)
    return float((slope + 2.0 * d / lam) * k**2 * length)


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


def apply_dispersion(
    samples: ArrayType,
    *,
    sampling_rate: float,
    dispersion_ps_nm_km: float,
    fiber_length_km: float,
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
            dispersion_ps_nm_km=dispersion_ps_nm_km,
            fiber_length_km=fiber_length_km,
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
