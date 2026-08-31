"""Linear fiber-channel impairments: chromatic dispersion, PMD, SOP mixing."""

import math

import numpy as np

from ...backend import ArrayType, dispatch
from ...logger import logger

__all__ = [
    "apply_chromatic_dispersion",
    "apply_pmd",
    "apply_polarization_mixing",
]


def apply_pmd(
    samples: ArrayType,
    sampling_rate: float,
    dgd: float,
    theta: float = 0.0,
) -> ArrayType:
    """
    Applies first-order Polarization Mode Dispersion (PMD) to a dual-pol signal.

    Models an uncompensated channel segment as a frequency-dependent Jones matrix:

        H(f) = R(+theta) * diag(e^(-j*pi*f*tau), e^(+j*pi*f*tau)) * R(-theta)

    where tau is the differential group delay (DGD), theta is the PSP orientation
    angle, and R(theta) is the 2x2 Jones rotation matrix.

    The DGD is applied in the *principal states of polarisation* (PSP) frame:
    R(-theta) projects the signal onto the PSPs, the differential delay +/- pi*f*tau
    is applied to each PSP, then R(+theta) rotates back to the lab frame. The two PSPs
    experience equal and opposite group delays, giving a total differential delay of
    tau seconds.

    theta is **not** a separate bulk rotation - it is the PSP orientation angle that
    is intrinsic to the PMD model. For a frequency-independent polarization rotation
    with no DGD use ``apply_polarization_mixing`` instead.

    The operation is fully vectorised in the frequency domain and
    backend-agnostic (NumPy / CuPy).

    Parameters
    ----------
    samples : array_like
        Dual-polarization signal. Shape: ``(2, N_samples)``.
    sampling_rate : float
        Sampling rate in Hz.
    dgd : float
        Differential group delay tau in seconds.
        Set to ``0`` to apply pure SOP rotation with no delay (equivalent
        to ``apply_polarization_mixing``).
    theta : float, default 0.0
        PSP orientation angle theta in radians.  Determines how
        much energy couples between X and Y polarisations.
        theta = 0 -> PSPs aligned with lab axes (no cross-coupling);
        theta = pi/4 -> maximum coupling.

    Returns
    -------
    array_like
        PMD-distorted signal, same backend/shape as input.

    Raises
    ------
    ValueError
        If input is not 2-dimensional with first axis == 2.

    Examples
    --------
    >>> samples = sig.samples  # shape (2, N), dual-pol
    >>> distorted = apply_pmd(samples, sig.sampling_rate, dgd=5e-12, theta=np.pi/5)
    """
    logger.info("Applying PMD (DGD=%.2e s, theta=%.3f rad).", dgd, theta)

    samples, xp, _ = dispatch(samples)

    if samples.ndim != 2 or samples.shape[0] != 2:
        raise ValueError(
            f"apply_pmd requires dual-pol input with shape (2, N). "
            f"Got shape {samples.shape}."
        )

    N = samples.shape[1]
    freqs = xp.fft.fftfreq(N, d=1.0 / sampling_rate)

    c = math.cos(theta)
    s = math.sin(theta)
    # H(f) = R(+θ) · diag(D) · R(-θ)
    # R(-θ): rotate INTO the principal-state-of-polarisation (PSP) frame
    Rfwd = xp.array([[c, s], [-s, c]], dtype=samples.dtype)
    # R(+θ): rotate back to the lab frame
    Rinv = xp.array([[c, -s], [s, c]], dtype=samples.dtype)

    phase = xp.pi * freqs * dgd
    D = xp.stack([xp.exp(-1j * phase), xp.exp(1j * phase)])  # (2, N)

    S_F = xp.fft.fft(samples, axis=-1)  # (2, N)

    # Apply: rotate to PSP frame -> DGD delay -> rotate back to lab frame
    out_F = Rinv @ (D * (Rfwd @ S_F))

    result = xp.fft.ifft(out_F, axis=-1)

    # Preserve input dtype (ifft may produce complex128 from complex64 input)
    if result.dtype != samples.dtype:
        result = result.astype(samples.dtype)

    return result


def apply_polarization_mixing(
    samples: ArrayType,
    theta: float | ArrayType,
    drift_rate_rad_per_sym: float = 0.0,
) -> ArrayType:
    """
    Applies a static or time-varying polarization rotation (pure SOP mixing).

    Models a frequency-independent 2x2 Jones rotation matrix:

        [Ex'[n]; Ey'[n]] = R(theta[n]) * [Ex[n]; Ey[n]]

        where R(theta) = [[cos(theta), -sin(theta)]; [sin(theta), cos(theta)]]

    Unlike ``apply_pmd``, there is no differential group delay - this is a
    pure bulk polarization rotation.  Useful for testing polarization-diverse
    receivers and modelling slow SOP drift.

    Parameters
    ----------
    samples : array_like
        Dual-polarization signal. Shape: ``(2, N_samples)``.
    theta : float or array_like of shape ``(N_samples,)``
        Rotation angle(s) in radians.

        * **Scalar** - static rotation: the same R(theta) is applied to every sample.
        * **Array of shape** ``(N,)`` - time-varying SOP: one angle per sample,
          applied sample-by-sample via vectorised broadcasting.

        When ``theta`` is a scalar and ``drift_rate_rad_per_sym != 0``, the
        trajectory is extended as a linear ramp:
        ``theta[n] = theta + drift_rate_rad_per_sym * n``.
    drift_rate_rad_per_sym : float, default 0.0
        Linear SOP drift rate in radians per sample.  Only used when ``theta``
        is a scalar.  Ignored when ``theta`` is an array.

    Returns
    -------
    array_like
        Rotated dual-polarization signal, same shape, dtype, and backend as
        input.

    Raises
    ------
    ValueError
        If input is not 2-dimensional with first axis == 2.

    Examples
    --------
    >>> # Static 45° rotation
    >>> rotated = apply_polarization_mixing(samples, theta=np.pi / 4)

    >>> # Slow linear SOP drift: 1 mrad per symbol
    >>> drifted = apply_polarization_mixing(samples, theta=0.0,
    ...                                     drift_rate_rad_per_sym=1e-3)
    """
    logger.info(
        "Applying polarization mixing (theta=%s, drift=%.3g rad/sym).",
        theta if np.ndim(theta) == 0 else "array",
        drift_rate_rad_per_sym,
    )

    samples, xp, _ = dispatch(samples)

    if samples.ndim != 2 or samples.shape[0] != 2:
        raise ValueError(
            f"apply_polarization_mixing requires dual-pol input with shape (2, N). "
            f"Got shape {samples.shape}."
        )

    N = samples.shape[1]

    # Build angle trajectory
    if np.ndim(theta) == 0:
        scalar_theta = float(theta)
        if drift_rate_rad_per_sym != 0.0:
            angles = (
                xp.arange(N, dtype=xp.float64) * drift_rate_rad_per_sym + scalar_theta
            )
        else:
            # Static: scalar path - avoid building (N,) array
            c = math.cos(scalar_theta)
            s = math.sin(scalar_theta)
            R = xp.array([[c, -s], [s, c]], dtype=samples.dtype)
            result = R @ samples
            if result.dtype != samples.dtype:
                result = result.astype(samples.dtype)
            return result
    else:
        angles = xp.asarray(theta, dtype=xp.float64)
        if angles.shape != (N,):
            raise ValueError(
                f"theta array must have shape (N,)={(N,)}, got {angles.shape}."
            )

    # Time-varying: vectorised per-sample rotation via broadcasting
    # R(θ[n]) applied to each column of samples
    cos_t = xp.cos(angles)  # (N,)
    sin_t = xp.sin(angles)  # (N,)

    Ex, Ey = samples[0], samples[1]
    Ex_out = cos_t * Ex - sin_t * Ey
    Ey_out = sin_t * Ex + cos_t * Ey

    result = xp.stack([Ex_out, Ey_out])  # (2, N)

    if result.dtype != samples.dtype:
        result = result.astype(samples.dtype)

    return result


def apply_chromatic_dispersion(
    samples: ArrayType,
    sampling_rate: float,
    dispersion_ps_nm_km: float,
    fiber_length_km: float,
    center_wavelength_nm: float,
) -> ArrayType:
    """
    Applies chromatic dispersion (CD) to a signal in the frequency domain.

    Multiplies the signal spectrum by the CD transfer function:

        H_CD(f) = exp(-j/2 * beta_2 * (2*pi*f)^2 * L)

    where

        beta_2 = -D * lambda^2 / (2*pi*c)

    and D is the dispersion parameter, lambda is the center wavelength,
    c is the speed of light, and L is the fiber length.

    Parameters
    ----------
    samples : array_like
        Complex baseband signal. Shape: ``(N,)`` (SISO) or ``(C, N)`` (MIMO).
    sampling_rate : float
        Sampling rate in Hz.
    dispersion_ps_nm_km : float
        Fiber dispersion parameter D in ps / (nm * km).
        Standard SMF-28: 17 ps/(nm*km) at 1550 nm.
    fiber_length_km : float
        Fiber span length in km.
    center_wavelength_nm : float
        Center wavelength in nm (e.g. 1550 for C-band).

    Returns
    -------
    array_like
        CD-impaired signal, same shape, dtype, and backend as input.

    See Also
    --------
    commkit.filtering.compensate_chromatic_dispersion :
        Remove CD in the receiver (electronic dispersion compensation).

    Examples
    --------
    >>> distorted = apply_chromatic_dispersion(
    ...     sig.samples, dispersion_ps_nm_km=17.0, fiber_length_km=80.0,
    ...     center_wavelength_nm=1550.0, sampling_rate=sig.sampling_rate)
    """
    logger.info(
        "Applying CD (D=%s ps/nm/km, L=%s km, λ=%s nm).",
        dispersion_ps_nm_km,
        fiber_length_km,
        center_wavelength_nm,
    )

    samples, xp, _ = dispatch(samples)
    was_1d = samples.ndim == 1
    if was_1d:
        samples = samples[None, :]
    _, N = samples.shape

    # Convert to SI
    D = dispersion_ps_nm_km * 1e-12 / (1e-9 * 1e3)  # s / m²
    lam = center_wavelength_nm * 1e-9  # m
    c = 2.998e8  # m/s
    L = fiber_length_km * 1e3  # m
    beta2 = -(D * lam**2) / (2.0 * np.pi * c) * L  # s²  (β₂·L product)

    omega = 2.0 * np.pi * xp.fft.fftfreq(N, d=1.0 / sampling_rate)
    H = xp.exp(-1j * (beta2 / 2.0) * omega**2)

    S_F = xp.fft.fft(samples, axis=-1)
    out_F = S_F * H[None, :]
    result = xp.fft.ifft(out_F, axis=-1)

    if result.dtype != samples.dtype:
        result = result.astype(samples.dtype)

    if was_1d:
        return result[0]
    return result
