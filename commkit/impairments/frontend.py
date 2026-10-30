"""Transceiver IQ imbalance.

:func:`apply_iq_imbalance`, its blind measurement
:func:`estimate_iq_imbalance` and its compensation
:func:`correct_iq_imbalance` (:class:`Lowdin` or :class:`GramSchmidt`
orthogonalization, or the inverse of an :class:`IQImbalanceEstimate`) are
kept together because they are one device model, the widely linear mixing
of I and Q.
"""

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType

from .._array import as_2d, restore_1d
from ..backend import ArrayType, dispatch, to_device
from ..core._signal_adapter import S, adapt_signal
from ..logger import logger
from ..math import db_to_linear

__all__ = [
    "GramSchmidt",
    "IQImbalanceEstimate",
    "Lowdin",
    "apply_iq_imbalance",
    "correct_iq_imbalance",
    "estimate_iq_imbalance",
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
# IQ-IMBALANCE ESTIMATION
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class IQImbalanceEstimate:
    """Result of :func:`estimate_iq_imbalance`, in the units of
    :func:`apply_iq_imbalance`.

    Per-channel fields follow the rank rule (0-d for ``(N,)`` input, ``(C,)``
    for ``(C, N)``) and stay on the input's device.

    Attributes
    ----------
    amplitude_imbalance_db : ndarray
        Q-to-I amplitude ratio ``A`` in dB; positive when Q has the higher
        gain.
    phase_imbalance_deg : ndarray
        Phase error ``φ`` of the Q branch in degrees.
    """

    amplitude_imbalance_db: ArrayType
    phase_imbalance_deg: ArrayType


def estimate_iq_imbalance(samples: ArrayType) -> IQImbalanceEstimate:
    """Measure IQ imbalance blindly, as the parameters of
    :func:`apply_iq_imbalance`.

    Assumes the undistorted signal is circular, its I and Q of equal power
    and uncorrelated, as square QAM, PSK and Gaussian noise are.  The
    imbalanced branches ``r_I = I - g sin φ · Q`` and ``r_Q = g cos φ · Q``
    then have the second moments

        E[r_Q²] = σ² g² cos² φ,   E[r_I r_Q] = -σ² g² sin φ cos φ,
        E[r_I²] - E[r_I r_Q]² / E[r_Q²] = σ²,

    which give ``tan φ = -E[r_I r_Q] / E[r_Q²]`` and ``g`` in closed form.

    Parameters
    ----------
    samples : array_like or Signal
        Complex baseband samples, ``(N,)`` or ``(C, N)``.  A carrier
        frequency offset does no harm (it rotates the signal, which stays
        circular), but the imbalance must sit after it, at the receiver.

    Returns
    -------
    IQImbalanceEstimate
        Per channel, on the input's device.

    Notes
    -----
    The moments are estimated from ``N`` samples, so the spread of the
    estimate falls as ``1/sqrt(N)``: about ``57°/sqrt(N)`` in phase for
    QPSK, 0.06° at ``N = 10^6``.

    Examples
    --------
    >>> r = apply_iq_imbalance(s, amplitude_imbalance_db=0.5, phase_imbalance_deg=3.0)
    >>> est = estimate_iq_imbalance(r)  # ≈ 0.5 dB, 3.0°
    >>> s_hat = correct_iq_imbalance(r, est)
    """
    signal_adapter = adapt_signal(samples, function_name="estimate_iq_imbalance()")
    x, xp, _ = dispatch(signal_adapter.array)
    x2, was_1d = as_2d(x, name="samples")
    sii, sqq, siq = _branch_moments(x2, xp)
    g2 = (sqq * sqq + siq * siq) / (sii * sqq - siq * siq)
    amplitude_db = 10.0 * xp.log10(g2)
    phase_deg = xp.degrees(xp.arctan2(-siq, sqq))
    if was_1d:
        amplitude_db, phase_deg = amplitude_db[0], phase_deg[0]
    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "Estimated IQ imbalance: amplitude=%s dB, phase=%s deg.",
            to_device(amplitude_db, "cpu").round(2),
            to_device(phase_deg, "cpu").round(2),
        )
    return IQImbalanceEstimate(amplitude_db, phase_deg)


def _branch_moments(
    samples: ArrayType, xp: ModuleType
) -> tuple[ArrayType, ArrayType, ArrayType]:
    """``E[I²]``, ``E[Q²]``, ``E[IQ]`` per channel of ``(C, N)`` samples,
    float64 ``(C,)``.

    Means in the samples' precision (pairwise on NumPy, a tree on CuPy,
    whose float64-accumulating reduction over few long rows is 10x slower).
    """
    i_branch, q_branch = samples.real, samples.imag
    sii = xp.mean(i_branch * i_branch, axis=-1).astype(xp.float64)
    sqq = xp.mean(q_branch * q_branch, axis=-1).astype(xp.float64)
    siq = xp.mean(i_branch * q_branch, axis=-1).astype(xp.float64)
    return sii, sqq, siq


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


def correct_iq_imbalance(
    samples: S, how: Lowdin | GramSchmidt | IQImbalanceEstimate
) -> S:
    """Correct IQ imbalance, blindly or from an estimate.

    With :class:`Lowdin` or :class:`GramSchmidt`, orthogonalizes the I and Q
    branches of each channel and restores the channel's input power; this
    undoes :func:`apply_iq_imbalance` up to a common gain and rotation.  With
    an :class:`IQImbalanceEstimate` (from :func:`estimate_iq_imbalance`,
    possibly of an earlier capture), inverts the widely linear model exactly,
    ``s = (conj(K1) · r - K2 · conj(r)) / (|K1|² - |K2|²)``.

    Parameters
    ----------
    samples : array_like or Signal
        Complex baseband samples, ``(N,)`` or ``(C, N)``.
    how : Lowdin, GramSchmidt or IQImbalanceEstimate
        Orthogonalization algorithm, or the imbalance to invert (0-d, or
        ``(C,)`` per channel).

    Returns
    -------
    ndarray or Signal
        Corrected samples, same shape, dtype and device as the input.

    Examples
    --------
    >>> r = apply_iq_imbalance(s, amplitude_imbalance_db=1.5, phase_imbalance_deg=4.0)
    >>> s_hat = correct_iq_imbalance(r, Lowdin())
    >>> s_hat = correct_iq_imbalance(r, estimate_iq_imbalance(r))
    """
    correct_fn = _IQ_CORRECTORS.get(type(how))
    if correct_fn is None and not isinstance(how, IQImbalanceEstimate):
        raise TypeError(
            "correct_iq_imbalance(): how must be Lowdin(), GramSchmidt() or an "
            f"IQImbalanceEstimate, got {type(how).__name__}."
        )
    signal_adapter = adapt_signal(samples, function_name="correct_iq_imbalance()")
    logger.info("Correcting IQ imbalance (%s).", type(how).__name__)
    x, xp, _ = dispatch(signal_adapter.array)
    if correct_fn is None:
        assert isinstance(how, IQImbalanceEstimate)
        return signal_adapter.wrap_samples(_invert_iq_imbalance(x, xp, how))
    return signal_adapter.wrap_samples(_apply_iq_correction(x, xp, correct_fn))


def _invert_iq_imbalance(
    samples: ArrayType, xp: ModuleType, how: IQImbalanceEstimate
) -> ArrayType:
    """The exact inverse of :func:`apply_iq_imbalance`, with common or
    per-channel parameters."""
    samples, was_1d = as_2d(samples, name="samples")
    num_channels = samples.shape[0]

    def column(value: ArrayType, name: str) -> ArrayType:
        arr = xp.asarray(value, dtype=xp.float64).reshape(-1)
        if arr.size not in (1, num_channels):
            raise ValueError(
                f"correct_iq_imbalance(): how.{name} has {arr.size} channels "
                f"for {num_channels}."
            )
        return arr[:, None]

    g = 10.0 ** (column(how.amplitude_imbalance_db, "amplitude_imbalance_db") / 20.0)
    phi = xp.radians(column(how.phase_imbalance_deg, "phase_imbalance_deg"))
    ge = g * xp.exp(1j * phi)
    k1, k2 = (1.0 + ge) / 2.0, (1.0 - ge) / 2.0
    det = g * xp.cos(phi)  # |K1|² - |K2|²
    a = (xp.conj(k1) / det).astype(samples.dtype)
    b = (-k2 / det).astype(samples.dtype)
    return restore_1d(was_1d, a * samples + b * xp.conj(samples))


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
    # The 2x2 algebra in float64.
    sii, sqq, siq = _branch_moments(samples, xp)

    a_ii, a_iq, a_qi, a_qq = correct_fn(sii, sqq, siq, xp)
    # Restore input power: E[|s_hat|^2] = E[|r|^2], each branch at half.
    gain = xp.sqrt((sii + sqq) / 2.0)
    real = i_branch.dtype

    def coef(a: ArrayType) -> ArrayType:
        """A per-channel coefficient, scaled by ``gain``, as a ``(C, 1)`` column."""
        return (a * gain).astype(real)[:, None]

    result = xp.empty_like(samples)
    result.real[...] = coef(a_ii) * i_branch + coef(a_iq) * q_branch
    result.imag[...] = coef(a_qi) * i_branch + coef(a_qq) * q_branch
    return restore_1d(was_1d, result)
