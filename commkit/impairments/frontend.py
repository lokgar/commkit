"""Transceiver front-end IQ-imbalance application and compensation.

Application and blind compensation are kept together because they are one
device model (the widely-linear I/Q mixing) and are read as a pair.
"""

import math

from ..backend import ArrayType, dispatch
from ..helpers import as_2d, restore_1d
from ..logger import logger

__all__ = [
    "apply_iq_imbalance",
    "compensate_iq_imbalance_gram_schmidt",
    "compensate_iq_imbalance_lowdin",
]


def apply_iq_imbalance(
    samples: ArrayType,
    amplitude_imbalance_db: float,
    phase_imbalance_deg: float,
) -> ArrayType:
    """
    Applies IQ imbalance to a complex baseband signal.

    Models the widely linear mixing that occurs when the I and Q branches of a
    receiver have mismatched gain and/or non-orthogonal phase:

        r[n] = K1 * s[n] + K2 * s*[n]

    where

        K1 = (1 + g * e^(j*phi)) / 2,
        K2 = (1 - g * e^(-j*phi)) / 2

    and g = 10^(A / 20) is the I/Q amplitude ratio and phi is the phase error in radians.

    Parameters
    ----------
    samples : array_like
        Complex baseband signal. Shape: ``(N,)`` (SISO) or ``(C, N)`` (MIMO).
    amplitude_imbalance_db : float
        Amplitude imbalance between I and Q branches in dB.  Positive values
        mean Q has higher gain than I.  Use ``0.0`` for no amplitude mismatch.
    phase_imbalance_deg : float
        Phase error between I and Q branches in degrees.  Use ``0.0`` for no
        phase mismatch.

    Returns
    -------
    array_like
        Imbalanced signal, same shape and dtype as input.

    Examples
    --------
    >>> r = apply_iq_imbalance(s, amplitude_imbalance_db=1.0, phase_imbalance_deg=3.0)
    """
    logger.info(
        "Applying IQ imbalance (amplitude=%.2f dB, phase=%.2f deg).",
        amplitude_imbalance_db,
        phase_imbalance_deg,
    )

    samples, xp, _ = dispatch(samples)

    g = 10.0 ** (amplitude_imbalance_db / 20.0)
    phi = math.radians(phase_imbalance_deg)

    # Mixing coefficients: r = K1*s + K2*conj(s)
    K1 = complex(0.5 * (1.0 + g * math.cos(phi)), 0.5 * g * math.sin(phi))
    K2 = complex(0.5 * (1.0 - g * math.cos(phi)), -0.5 * g * math.sin(phi))

    result = K1 * samples + K2 * xp.conj(samples)

    if result.dtype != samples.dtype:
        result = result.astype(samples.dtype)

    return result


def compensate_iq_imbalance_lowdin(samples: ArrayType) -> ArrayType:
    """
    Blind IQ imbalance compensation via Löwdin symmetric orthogonalisation.

    Treats the I and Q components as a 2-D real vector and applies the
    symmetric whitening transform W = M^(-1/2) (where M is the 2x2 second-moment matrix)
    so that the corrected I and Q channels have equal power and zero cross-correlation.
    Unlike Gram-Schmidt, the transform is symmetric: both branches are adjusted
    equally, minimising the total distortion introduced.

    The output power equals the input power.

    Parameters
    ----------
    samples : array_like
        Complex baseband signal. Shape: ``(N,)`` (SISO) or ``(C, N)`` (MIMO).

    Returns
    -------
    array_like
        IQ-corrected signal, same shape and dtype as input.

    Notes
    -----
    Per channel: forms the 2x2 second-moment matrix M = X*X.T/N from X = [I; Q],
    then applies whitening W = M^(-1/2) via symmetric eigendecomposition.

    Examples
    --------
    >>> r = apply_iq_imbalance(s, amplitude_imbalance_db=1.5, phase_imbalance_deg=4.0)
    >>> s_hat = compensate_iq_imbalance_lowdin(r)
    """
    logger.info("Applying Löwdin IQ imbalance compensation.")

    samples, xp, _ = dispatch(samples)

    samples, was_1d = as_2d(samples, name="samples")

    C, N = samples.shape
    result = xp.empty_like(samples)

    for ch in range(C):
        r = samples[ch]  # (N,)
        P_in = xp.mean(xp.abs(r) ** 2)

        # 2xN real data matrix: rows = [I, Q]
        X = xp.stack([r.real, r.imag])  # (2, N)

        # 2x2 second-moment matrix
        M = (X @ X.T) / N  # (2, 2)

        # Symmetric whitening: W = M^{-1/2} = V @ diag(1/sqrt(lam)) @ V.T
        lam, V = xp.linalg.eigh(M)  # lam: (2,), V: (2, 2)
        W = (V * (1.0 / xp.sqrt(lam))) @ V.T  # (2, 2)

        # Apply whitening - X_corr has identity second-moment matrix
        X_corr = W @ X  # (2, N)

        # Restore input power: E[|s_hat|^2] = P_in
        s_corr = (X_corr[0] + 1j * X_corr[1]) * xp.sqrt(P_in / 2.0)

        if s_corr.dtype != samples.dtype:
            s_corr = s_corr.astype(samples.dtype)

        result[ch] = s_corr

    return restore_1d(was_1d, result)


def compensate_iq_imbalance_gram_schmidt(samples: ArrayType) -> ArrayType:
    """
    Blind IQ imbalance compensation via Gram-Schmidt sequential orthogonalisation.

    Uses the I branch as the reference axis.  The Q branch is orthogonalised
    against I and both are normalised to unit RMS before being recombined.
    This is the classical GSOP approach used in analogue front-end calibration.

    The output power equals the input power.

    Parameters
    ----------
    samples : array_like
        Complex baseband signal. Shape: ``(N,)`` (SISO) or ``(C, N)`` (MIMO).

    Returns
    -------
    array_like
        IQ-corrected signal, same shape and dtype as input.

    Notes
    -----
    *Algorithm* (per channel):

    1. Normalize I to unit RMS: I_hat = I / sigma_I.
    2. Remove I-projection from Q: Q_perp = Q - <I_hat, Q> * I_hat.
    3. Normalize Q_perp to unit RMS: Q_hat = Q_perp / sigma_Q_perp.
    4. Recombine and rescale to preserve input power.

    Examples
    --------
    >>> r = apply_iq_imbalance(s, amplitude_imbalance_db=1.5, phase_imbalance_deg=4.0)
    >>> s_hat = compensate_iq_imbalance_gram_schmidt(r)
    """
    logger.info("Applying Gram-Schmidt IQ imbalance compensation.")

    samples, xp, _ = dispatch(samples)

    samples, was_1d = as_2d(samples, name="samples")

    C, N = samples.shape
    result = xp.empty_like(samples)

    for ch in range(C):
        r = samples[ch]  # (N,)
        P_in = xp.mean(xp.abs(r) ** 2)

        I = r.real  # noqa: E741
        Q = r.imag

        # Step 1: Normalise I (reference branch)
        sigma_I = xp.sqrt(xp.mean(I**2))
        I_norm = I / sigma_I

        # Step 2: Orthogonalise Q against I
        rho = xp.mean(I_norm * Q)  # scalar projection coefficient
        Q_orth = Q - rho * I_norm

        # Step 3: Normalise orthogonalised Q
        sigma_Q = xp.sqrt(xp.mean(Q_orth**2))
        Q_norm = Q_orth / sigma_Q

        # Step 4: Recombine and restore input power
        s_corr = (I_norm + 1j * Q_norm) * xp.sqrt(P_in / 2.0)

        if s_corr.dtype != samples.dtype:
            s_corr = s_corr.astype(samples.dtype)

        result[ch] = s_corr

    return restore_1d(was_1d, result)
