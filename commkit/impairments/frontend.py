"""Transceiver IQ imbalance.

:func:`apply_iq_imbalance` and its blind compensation
:func:`correct_iq_imbalance` (:class:`Lowdin` or
:class:`GramSchmidt` orthogonalization) are kept together because they are
one device model, the widely linear mixing of I and Q, and are read as a
pair.
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
    """Apply IQ imbalance to a complex baseband signal.

    The widely linear mixing of I and Q branches with mismatched gain and
    non-orthogonal phase, ``r = I + j·g·e^{jφ}·Q`` for ``s = I + jQ``:

        r[n] = K1 · s[n] + K2 · conj(s[n]),
        K1 = (1 + g e^{jφ}) / 2,   K2 = (1 - g e^{jφ}) / 2,

    with the Q-to-I amplitude ratio ``g = 10^(A/20)`` and the phase error
    ``φ`` in radians.

    Parameters
    ----------
    samples : array_like or Signal
        Complex baseband samples, ``(N,)`` or ``(C, N)``.
    amplitude_imbalance_db : float
        Amplitude imbalance ``A`` between the branches in dB; positive gives
        Q the higher gain, ``0.0`` none.
    phase_imbalance_deg : float
        Phase error ``φ`` of the Q branch in degrees; ``0.0`` for none.

    Returns
    -------
    ndarray or Signal
        The imbalanced samples, same shape, dtype and device as the input.

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


# The correctors map the per-channel branch statistics ``E[I²]``, ``E[Q²]``
# and ``E[IQ]`` (float64, ``(C,)``) to the 2x2 transform ``[I'; Q'] = A [I; Q]``
# of each channel, as the four entries ``(a_ii, a_iq, a_qi, a_qq)``.
_Coefficients = tuple[ArrayType, ArrayType, ArrayType, ArrayType]


def _lowdin(
    sii: ArrayType, sqq: ArrayType, siq: ArrayType, xp: ModuleType
) -> _Coefficients:
    # M^(-1/2) of M = [[sii, siq], [siq, sqq]] in closed form: the square root
    # of a 2x2 SPD matrix is (M + s·1) / t, with s = sqrt(det M) and
    # t = sqrt(tr M + 2s), so M^(-1/2) = t · (M + s·1)^(-1).
    det_s = xp.sqrt(sii * sqq - siq * siq)
    t = xp.sqrt(sii + sqq + 2.0 * det_s)
    a, d = sii + det_s, sqq + det_s
    scale = t / (a * d - siq * siq)
    return d * scale, -siq * scale, -siq * scale, a * scale


def _gram_schmidt(
    sii: ArrayType, sqq: ArrayType, siq: ArrayType, xp: ModuleType
) -> _Coefficients:
    # I / rms(I); Q minus its projection on I, siq/sii · I, over its rms,
    # sqrt(sqq - siq²/sii).
    inv_i = 1.0 / xp.sqrt(sii)
    inv_q = 1.0 / xp.sqrt(sqq - siq * siq / sii)
    return inv_i, xp.zeros_like(sii), -(siq / sii) * inv_q, inv_q


_IQ_CORRECTORS: dict[
    type, Callable[[ArrayType, ArrayType, ArrayType, ModuleType], _Coefficients]
] = {Lowdin: _lowdin, GramSchmidt: _gram_schmidt}


def correct_iq_imbalance(samples: S, how: Lowdin | GramSchmidt) -> S:
    """Correct IQ imbalance blindly.

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
    ndarray or Signal
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
    correct_fn: Callable[[ArrayType, ArrayType, ArrayType, ModuleType], _Coefficients],
) -> ArrayType:
    """Shared scaffold for the blind IQ-imbalance compensators, all channels
    at once.

    Measures the branch statistics, takes the 2x2 transform of
    ``correct_fn`` (Löwdin symmetric whitening or Gram-Schmidt
    orthogonalization; both give unit-power, uncorrelated branches), folds in
    the gain that restores each channel's input power, and writes the
    corrected real and imaginary parts in the samples' precision.
    """
    samples, was_1d = as_2d(samples, name="samples")
    i_branch, q_branch = samples.real, samples.imag
    # Means in the samples' precision (pairwise on NumPy, a tree on CuPy,
    # whose float64-accumulating reduction over few long rows is 10x
    # slower); the 2x2 algebra in float64.
    sii = xp.mean(i_branch * i_branch, axis=-1).astype(xp.float64)
    sqq = xp.mean(q_branch * q_branch, axis=-1).astype(xp.float64)
    siq = xp.mean(i_branch * q_branch, axis=-1).astype(xp.float64)

    a_ii, a_iq, a_qi, a_qq = correct_fn(sii, sqq, siq, xp)
    # Restore input power: E[|s_hat|^2] = E[|r|^2], each branch at half.
    gain = xp.sqrt((sii + sqq) / 2.0)
    real = i_branch.dtype

    def coef(a: ArrayType) -> ArrayType:
        return (a * gain).astype(real)[:, None]

    result = xp.empty_like(samples)
    result.real[...] = coef(a_ii) * i_branch + coef(a_iq) * q_branch
    result.imag[...] = coef(a_qi) * i_branch + coef(a_qq) * q_branch
    return restore_1d(was_1d, result)
