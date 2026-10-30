"""Truth-based scoring of receiver outputs, through the public API.

Every score compares an output with the transmitted reference or the true
impairment, never with the receiver's own decisions: a constellation with a
rotational symmetry maps a rotated output onto its own grid, so a distance to
the nearest point cannot see a wrong quadrant, a slip or a duplicated stream.

Only the legitimate ambiguities are removed: one rotation per stream
(:func:`commkit.recovery.resolve_phase_ambiguity`) and one bijective channel
permutation (:func:`commkit.recovery.resolve_channel_permutation`) for blind
receivers, each fixed over the whole scored interval.
"""

from typing import Any

import numpy as np

from commkit.recovery import resolve_channel_permutation, resolve_phase_ambiguity

from .conversions import to_numpy


def ser(y: Any, ref: Any, constellation: Any) -> Any:
    """Symbol error rate of ``y`` against ``ref``: a float for ``(N,)``, one
    value per channel ``(C,)`` for ``(C, N)``.  Decisions are the nearest
    constellation points (not bits: one wrong symbol is one error)."""
    y, ref = to_numpy(y), to_numpy(ref)
    if y.shape != ref.shape:
        raise ValueError(f"shapes differ: {y.shape} and {ref.shape}")
    d_y = to_numpy(constellation._nearest(y.reshape(-1))).reshape(y.shape)
    d_r = to_numpy(constellation._nearest(ref.reshape(-1))).reshape(ref.shape)
    rate = np.mean(d_y != d_r, axis=-1)
    return float(rate) if y.ndim == 1 else rate


def resolve_blind(
    y: Any,
    ref: Any,
    constellation: Any,
    *,
    skip: int = 0,
    permute: bool = True,
    rotation: str = "symmetry",
) -> np.ndarray:
    """``y`` with the ambiguities of a blind receiver removed, estimated
    after ``skip``: one channel permutation (bijective, so two outputs locked
    onto the same source cannot both score), then one rotation per stream.

    ``rotation="symmetry"`` removes a multiple of ``2π/S``, what blind
    carrier recovery leaves; ``"any"`` removes one arbitrary constant angle,
    ``∠ Σ y conj(s)``, what a phase-blind equalizer (CMA, RDE) leaves
    without carrier recovery.  Either way a single value per stream: a
    rotation that drifts or jumps still counts as errors.
    """
    y, ref = to_numpy(y), to_numpy(ref)[..., : to_numpy(y).shape[-1]]
    if permute and y.ndim == 2 and y.shape[0] > 1:
        y = to_numpy(resolve_channel_permutation(y, ref, num_skip_symbols=skip))
    if rotation == "any":
        corr = np.sum(y[..., skip:] * np.conj(ref[..., skip:]), axis=-1)
        return y * np.exp(-1j * np.angle(corr))[..., None].astype(y.dtype)
    return to_numpy(
        resolve_phase_ambiguity(
            y, ref, constellation=constellation, num_skip_symbols=skip
        )
    )


def blind_ser(
    y: Any,
    ref: Any,
    constellation: Any,
    *,
    skip: int = 0,
    permute: bool = True,
    rotation: str = "symmetry",
) -> Any:
    """SER after :func:`resolve_blind`, over the symbols after ``skip``."""
    y = resolve_blind(
        y, ref, constellation, skip=skip, permute=permute, rotation=rotation
    )
    ref = to_numpy(ref)[..., : y.shape[-1]]
    return ser(y[..., skip:], ref[..., skip:], constellation)


def evm_rms(y: Any, ref: Any) -> Any:
    """RMS error vector magnitude against the reference (not in %)."""
    y, ref = to_numpy(y), to_numpy(ref)
    num = np.mean(np.abs(y - ref) ** 2, axis=-1)
    den = np.mean(np.abs(ref) ** 2, axis=-1)
    out = np.sqrt(num / den)
    return float(out) if y.ndim == 1 else out


def phase_rms(est: Any, truth: Any, *, symmetry: int | None = None) -> float:
    """RMS of ``est - truth`` in radians.  With ``symmetry``, one whole
    quantum ``2π/symmetry`` (the median one) is removed from the whole
    record, the only shift a blind estimator cannot see; a slip in the middle
    still counts."""
    err = np.asarray(to_numpy(est), dtype=np.float64) - np.asarray(
        to_numpy(truth), dtype=np.float64
    )
    if symmetry is not None:
        q = 2.0 * np.pi / symmetry
        err = err - np.round(np.median(err) / q) * q
    return float(np.sqrt(np.mean(err**2)))


def block_rotations(y: Any, ref: Any, *, symmetry: int, block: int) -> np.ndarray:
    """The rotation index ``k`` (output ≈ reference · e^{j 2πk/symmetry}) of
    each block of ``block`` symbols, from the angle of ``Σ y conj(s)``."""
    y, ref = to_numpy(y).ravel(), to_numpy(ref).ravel()[: to_numpy(y).size]
    n = (y.size // block) * block
    corr = np.sum((y[:n] * np.conj(ref[:n])).reshape(-1, block), axis=1)
    return np.round(np.angle(corr) / (2.0 * np.pi / symmetry)).astype(int) % symmetry


def zero_failure_upper_bound(num_trials: int, confidence: float = 0.95) -> float:
    """One-sided upper bound on a failure rate after 0 failures in
    ``num_trials`` independent trials: ``1 - (1 - confidence)^(1/n)``."""
    return 1.0 - (1.0 - confidence) ** (1.0 / num_trials)


def binomial_margin(p: float, n: int, z: float = 4.0) -> float:
    """``z`` standard deviations of an error-rate estimate from ``n`` trials."""
    return z * float(np.sqrt(p * (1.0 - p) / n))
