"""Transceiver front-end IQ-imbalance application and compensation.

Application and blind compensation are kept together because they are one
device model (the widely-linear I/Q mixing) and are read as a pair.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType

from .._array import as_2d, restore_1d
from ..backend import ArrayType, dispatch
from ..core._signal_adapter import S, adapt_signal
from ..logger import logger
from ..math import db_to_linear

__all__ = [
    "GramSchmidt",
    "Lowdin",
    "apply_iq_imbalance",
    "correct_iq_imbalance",
]


def apply_iq_imbalance(
    samples: S,
    *,
    amplitude_imbalance_db: float,
    phase_imbalance_deg: float,
) -> S:
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
    samples : array_like or Signal
        Complex baseband signal. Shape: ``(N,)`` (SISO) or ``(C, N)`` (MIMO).
    amplitude_imbalance_db : float
        Amplitude imbalance between I and Q branches in dB.  Positive values
        mean Q has higher gain than I.  Use ``0.0`` for no amplitude mismatch.
    phase_imbalance_deg : float
        Phase error between I and Q branches in degrees.  Use ``0.0`` for no
        phase mismatch.

    Returns
    -------
    array_like or Signal
        Imbalanced signal, same shape and dtype as input.  A :class:`Signal`
        returns a new imbalanced :class:`Signal`.

    Examples
    --------
    >>> r = apply_iq_imbalance(s, amplitude_imbalance_db=1.0, phase_imbalance_deg=3.0)
    """
    signal_adapter = adapt_signal(samples, function_name="apply_iq_imbalance()")

    logger.info(
        "Applying IQ imbalance (amplitude=%.2f dB, phase=%.2f deg).",
        amplitude_imbalance_db,
        phase_imbalance_deg,
    )

    x, xp, _ = dispatch(signal_adapter.array)

    g = db_to_linear(amplitude_imbalance_db, power=False)
    phi = math.radians(phase_imbalance_deg)

    # Mixing coefficients: r = K1*s + K2*conj(s)
    K1 = complex(0.5 * (1.0 + g * math.cos(phi)), 0.5 * g * math.sin(phi))
    K2 = complex(0.5 * (1.0 - g * math.cos(phi)), -0.5 * g * math.sin(phi))

    result = K1 * x + K2 * xp.conj(x)

    if result.dtype != x.dtype:
        result = result.astype(x.dtype)

    return signal_adapter.wrap_samples(result)


# -----------------------------------------------------------------------------
# BLIND IQ-IMBALANCE CORRECTION
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Lowdin:
    """Löwdin symmetric orthogonalization of the I and Q branches.

    Whitens ``X = [I; Q]`` with ``W = M^(-1/2)``, ``M = X Xᵀ / N`` (symmetric
    eigendecomposition), so the corrected branches have equal power and zero
    correlation.  The transform is symmetric: both branches are adjusted
    equally, which minimizes the total distortion.
    """


@dataclass(frozen=True)
class GramSchmidt:
    """Gram-Schmidt orthogonalization with I as the reference branch.

    Normalizes I to unit RMS, removes its projection from Q, then normalizes
    Q (the classical GSOP front-end calibration).
    """


def _lowdin(r: ArrayType, xp: ModuleType) -> tuple[ArrayType, ArrayType]:
    N = r.shape[0]
    X = xp.stack([r.real, r.imag])  # (2, N), rows = [I, Q]
    M = (X @ X.T) / N  # 2x2 second-moment matrix
    lam, V = xp.linalg.eigh(M)
    W = (V * (1.0 / xp.sqrt(lam))) @ V.T  # M^(-1/2)
    X_corr = W @ X  # identity second-moment matrix
    return X_corr[0], X_corr[1]


def _gram_schmidt(r: ArrayType, xp: ModuleType) -> tuple[ArrayType, ArrayType]:
    i_branch, q_branch = r.real, r.imag
    i_norm = i_branch / xp.sqrt(xp.mean(i_branch**2))
    q_orth = q_branch - xp.mean(i_norm * q_branch) * i_norm
    q_norm = q_orth / xp.sqrt(xp.mean(q_orth**2))
    return i_norm, q_norm


_IQ_CORRECTORS: dict[
    type, Callable[[ArrayType, ModuleType], tuple[ArrayType, ArrayType]]
] = {Lowdin: _lowdin, GramSchmidt: _gram_schmidt}


def correct_iq_imbalance(samples: S, how: Lowdin | GramSchmidt) -> S:
    """
    Blind IQ-imbalance correction.

    Orthogonalizes the I and Q branches of each channel with the algorithm
    ``how`` and restores the channel's input power.  Undoes
    :func:`apply_iq_imbalance` up to a common gain and rotation.

    Parameters
    ----------
    samples : array_like or Signal
        Complex baseband samples, ``(N,)`` or ``(C, N)``.
    how : Lowdin or GramSchmidt
        Orthogonalization algorithm.

    Returns
    -------
    array_like or Signal
        Corrected samples, same shape, dtype and device as the input.

    Examples
    --------
    >>> r = apply_iq_imbalance(s, amplitude_imbalance_db=1.5, phase_imbalance_deg=4.0)
    >>> s_hat = correct_iq_imbalance(r, Lowdin())
    """
    correct_fn = _IQ_CORRECTORS.get(type(how))
    if correct_fn is None:
        raise TypeError(
            "correct_iq_imbalance(): how must be Lowdin() or GramSchmidt(), got "
            f"{type(how).__name__}."
        )
    signal_adapter = adapt_signal(samples, function_name="correct_iq_imbalance()")
    logger.info("Correcting IQ imbalance (%s).", type(how).__name__)
    x, xp, _ = dispatch(signal_adapter.array)
    return signal_adapter.wrap_samples(_apply_iq_correction(x, xp, correct_fn))


def _apply_iq_correction(
    samples: ArrayType,
    xp: ModuleType,
    correct_fn: Callable[[ArrayType, ModuleType], tuple[ArrayType, ArrayType]],
) -> ArrayType:
    """Shared per-channel scaffold for the blind IQ-imbalance compensators.

    ``correct_fn(r, xp) -> (comp0, comp1)`` computes one channel's corrected
    real I/Q-like components (algorithm-specific: Löwdin symmetric whitening
    or Gram-Schmidt orthogonalisation) from the complex channel ``r`` (shape
    ``(N,)``).  Shares the per-channel input-power measurement, power
    restoration, dtype cast, and channel loop between the two compensators -
    the correction math itself stays in ``correct_fn``.
    """
    samples, was_1d = as_2d(samples, name="samples")

    C, N = samples.shape
    result = xp.empty_like(samples)

    for ch in range(C):
        r = samples[ch]  # (N,)
        P_in = xp.mean(xp.abs(r) ** 2)

        comp0, comp1 = correct_fn(r, xp)

        # Restore input power: E[|s_hat|^2] = P_in
        s_corr = (comp0 + 1j * comp1) * xp.sqrt(P_in / 2.0)

        if s_corr.dtype != samples.dtype:
            s_corr = s_corr.astype(samples.dtype)

        result[ch] = s_corr

    return restore_1d(was_1d, result)
