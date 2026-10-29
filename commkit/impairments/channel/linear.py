"""Linear fiber-channel impairments.

:func:`apply_chromatic_dispersion`, :func:`apply_pmd` (a single
differential-group-delay section) and :func:`apply_polarization_mixing` (a
rotation of the state of polarization).
"""

import math
from types import ModuleType
from typing import Any

import numpy as np

from ... import _fft
from ..._array import require_channels
from ..._dispersion import apply_dispersion
from ..._fiber import jones_section
from ...backend import ArrayType, dispatch
from ...core._signal_adapter import S, adapt_signal
from ...logger import logger
from ...math import _phasor

__all__ = [
    "apply_chromatic_dispersion",
    "apply_pmd",
    "apply_polarization_mixing",
]


def _jones_rotation(theta: float, xp: ModuleType, dtype: Any) -> ArrayType:
    """Static 2x2 Jones rotation matrix ``R(theta) = [[cos, -sin], [sin, cos]]``.

    Shared builder for ``apply_pmd``'s rotation into the PSP frame,
    ``R(-theta)`` (the section of ``_fiber.jones_section``), and
    ``apply_polarization_mixing``'s static-``theta`` path.  The time-varying
    path of ``apply_polarization_mixing`` builds no matrix (one 2x2 matrix
    per sample is not worth building); it applies the same rotation by
    broadcasting per sample.
    """
    c = math.cos(theta)
    s = math.sin(theta)
    return xp.array([[c, -s], [s, c]], dtype=dtype)


def apply_pmd(
    samples: S,
    *,
    dgd: float,
    theta: float = 0.0,
    sampling_rate: float | None = None,
) -> S:
    """Apply first-order polarization-mode dispersion (PMD) to a dual-pol field.

    One section with the frequency-dependent Jones matrix

        H(f) = R(θ) · diag(exp(-jπfτ), exp(+jπfτ)) · R(-θ)

    with the differential group delay (DGD) ``τ``, the orientation ``θ`` of
    the principal states of polarization (PSPs) and the rotation
    ``R(θ) = [[cos θ, -sin θ], [sin θ, cos θ]]``.  ``R(-θ)`` projects the
    field onto the PSPs, which are delayed by ``±τ/2`` (a differential delay
    of ``τ``), and ``R(θ)`` rotates back.  ``θ`` is part of the PMD model,
    not a separate rotation: with ``τ = 0`` the section is the identity.
    For a frequency-independent rotation use
    :func:`apply_polarization_mixing`; for PMD distributed along a fiber,
    :func:`apply_fiber_propagation` with ``Fiber(pmd_ps_sqrt_km=...)``.

    Parameters
    ----------
    samples : array_like or Signal
        Dual-polarization field, ``(2, N)``.
    dgd : float
        Differential group delay ``τ`` in seconds.
    theta : float, default 0.0
        PSP orientation ``θ`` in radians: 0 aligns the PSPs with the X and Y
        axes (no coupling between them), ``π/4`` couples them most.
    sampling_rate : float, optional
        Sampling rate in Hz.  Taken from the Signal; required for array
        input.  A value that disagrees with the Signal raises.

    Returns
    -------
    ndarray or Signal
        The distorted field, same shape, dtype and device as the input.

    Raises
    ------
    ValueError
        If the input is not ``(2, N)``.

    Examples
    --------
    >>> x = sig.samples  # shape (2, N), dual-pol
    >>> distorted = apply_pmd(x, dgd=5e-12, theta=np.pi / 5, sampling_rate=fs)
    >>> distorted = apply_pmd(sig, dgd=5e-12, theta=np.pi / 5)  # Signal input
    """
    signal_adapter = adapt_signal(samples, function_name="apply_pmd()")
    sampling_rate = signal_adapter.resolve_fact("sampling_rate", sampling_rate)

    logger.info("Applying PMD (DGD=%.2e s, theta=%.3f rad).", dgd, theta)

    x, xp, _ = dispatch(signal_adapter.array)

    require_channels(
        x, 2, name="samples", description="dual-pol input with shape (2, N)"
    )

    N = x.shape[1]
    freqs = xp.fft.fftfreq(N, d=1.0 / sampling_rate)

    # H(f) = R(+θ) · diag(e^{-jπfτ}, e^{+jπfτ}) · R(-θ): one PMD section with
    # the real rotation U = R(-θ) into the PSP frame (U^H = R(+θ)).
    S_F = _fft.fft(x, axis=-1)  # (2, N)
    phase = xp.pi * freqs * dgd  # float64; responses in the spectrum's precision
    h_neg = _phasor(-phase, S_F.dtype, xp)
    out_F = jones_section(
        S_F, _jones_rotation(-theta, np, np.float64), h_neg, xp.conj(h_neg)
    )
    result = _fft.ifft(out_F, axis=-1)

    # Preserve input dtype (ifft may produce complex128 from complex64 input)
    if result.dtype != x.dtype:
        result = result.astype(x.dtype)

    return signal_adapter.wrap_samples(result)


def apply_polarization_mixing(
    samples: S,
    *,
    theta: float | ArrayType,
    drift_rad_per_sample: float = 0.0,
) -> S:
    """Rotate the state of polarization, statically or over time.

    A frequency-independent real rotation,
    ``[x'[n], y'[n]] = R(θ[n]) · [x[n], y[n]]`` with
    ``R(θ) = [[cos θ, -sin θ], [sin θ, cos θ]]``: unlike :func:`apply_pmd`,
    there is no differential group delay.  For testing polarization-diverse
    receivers and modeling slow SOP drift.

    Parameters
    ----------
    samples : array_like or Signal
        Dual-polarization field, ``(2, N)``.
    theta : float or array_like
        Rotation angle in radians: a scalar for one rotation of every sample
        (a ramp with ``drift_rad_per_sample``), or ``(N,)`` angles, one per
        sample.
    drift_rad_per_sample : float, default 0.0
        Linear drift for a scalar ``theta``:
        ``θ[n] = theta + drift_rad_per_sample · n``.  Ignored for an array
        ``theta``.

    Returns
    -------
    ndarray or Signal
        The rotated field, same shape, dtype and device as the input.

    Raises
    ------
    ValueError
        If the input is not ``(2, N)`` or an array ``theta`` is not ``(N,)``.

    Examples
    --------
    >>> # Static 45° rotation
    >>> rotated = apply_polarization_mixing(samples, theta=np.pi / 4)

    >>> # Slow linear SOP drift: 1 mrad per sample
    >>> drifted = apply_polarization_mixing(
    ...     samples, theta=0.0, drift_rad_per_sample=1e-3)
    """
    signal_adapter = adapt_signal(samples, function_name="apply_polarization_mixing()")

    logger.info(
        "Applying polarization mixing (theta=%s, drift=%.3g rad/sym).",
        theta if np.ndim(theta) == 0 else "array",
        drift_rad_per_sample,
    )

    x, xp, _ = dispatch(signal_adapter.array)

    require_channels(
        x, 2, name="samples", description="dual-pol input with shape (2, N)"
    )

    N = x.shape[1]

    # Build angle trajectory
    if np.ndim(theta) == 0:
        scalar_theta = float(theta)
        if drift_rad_per_sample != 0.0:
            angles = (
                xp.arange(N, dtype=xp.float64) * drift_rad_per_sample + scalar_theta
            )
        else:
            # Static: scalar path - avoid building (N,) array
            R = _jones_rotation(scalar_theta, xp, x.dtype)
            result = R @ x
            if result.dtype != x.dtype:
                result = result.astype(x.dtype)
            return signal_adapter.wrap_samples(result)
    else:
        angles = xp.asarray(theta, dtype=xp.float64)
        if angles.shape != (N,):
            raise ValueError(
                f"theta array must have shape (N,)={(N,)}, got {angles.shape}."
            )

    # Time-varying: vectorized per-sample rotation via broadcasting
    # R(θ[n]) applied to each column of x
    # Angles in float64, the rotation in the samples' precision.
    cos_t = xp.cos(angles).astype(x.real.dtype)  # (N,)
    sin_t = xp.sin(angles).astype(x.real.dtype)  # (N,)

    Ex, Ey = x[0], x[1]
    Ex_out = cos_t * Ex - sin_t * Ey
    Ey_out = sin_t * Ex + cos_t * Ey

    result = xp.stack([Ex_out, Ey_out])  # (2, N)

    if result.dtype != x.dtype:
        result = result.astype(x.dtype)

    return signal_adapter.wrap_samples(result)


def apply_chromatic_dispersion(
    samples: S,
    *,
    dispersion_ps_nm_km: float,
    fiber_length_km: float,
    center_wavelength_nm: float,
    sampling_rate: float | None = None,
) -> S:
    """Apply chromatic dispersion (CD) in the frequency domain.

    Multiplies the spectrum by ``H(ω) = exp(-j β₂ L ω² / 2)`` with
    ``β₂ = -D λ² / (2π c)``: the dispersion parameter ``D``, the center
    wavelength ``λ``, the speed of light ``c`` and the fiber length ``L``
    (second-order dispersion only, without loss or nonlinearity; see
    :func:`apply_fiber_propagation` for those).

    Parameters
    ----------
    samples : array_like or Signal
        Complex baseband samples, ``(N,)`` or ``(C, N)``.
    dispersion_ps_nm_km : float
        Dispersion parameter ``D`` in ps/(nm km); standard single-mode fiber
        (G.652) has 17 at 1550 nm.  Positive is anomalous (``β₂ < 0``).
    fiber_length_km : float
        Fiber length in km.
    center_wavelength_nm : float
        Center wavelength in nm (e.g. 1550 for the C band).
    sampling_rate : float, optional
        Sampling rate in Hz.  Taken from the Signal; required for array
        input.  A value that disagrees with the Signal raises.

    Returns
    -------
    ndarray or Signal
        The dispersed samples, same shape, dtype and device as the input.

    See Also
    --------
    commkit.filtering.correct_chromatic_dispersion :
        Remove CD in the receiver (electronic dispersion compensation).

    Examples
    --------
    >>> distorted = apply_chromatic_dispersion(
    ...     x, dispersion_ps_nm_km=17.0, fiber_length_km=80.0,
    ...     center_wavelength_nm=1550.0, sampling_rate=fs)
    >>> distorted = apply_chromatic_dispersion(  # Signal input
    ...     sig, dispersion_ps_nm_km=17.0, fiber_length_km=80.0,
    ...     center_wavelength_nm=1550.0)
    """
    signal_adapter = adapt_signal(samples, function_name="apply_chromatic_dispersion()")
    sampling_rate = signal_adapter.resolve_fact("sampling_rate", sampling_rate)

    logger.info(
        "Applying CD (D=%s ps/nm/km, L=%s km, λ=%s nm).",
        dispersion_ps_nm_km,
        fiber_length_km,
        center_wavelength_nm,
    )

    result = apply_dispersion(
        signal_adapter.array,
        sampling_rate=sampling_rate,
        dispersion_ps_nm_km=dispersion_ps_nm_km,
        fiber_length_km=fiber_length_km,
        center_wavelength_nm=center_wavelength_nm,
        inverse=False,
    )
    return signal_adapter.wrap_samples(result)
