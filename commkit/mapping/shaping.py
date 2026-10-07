"""
Probabilistic shaping.

Maxwell-Boltzmann priors over any constellation, the shaping parameter that
reaches a target entropy, and the pmf-weighted constellation power.  Shaped
constellations are built with :meth:`Constellation.shaped`, which uses these
functions.

The shaping parameter ``nu`` is defined on the points scaled to a minimum
distance of 2, which is the odd-integer grid for QAM and PAM, so ``nu``
matches the literature (``P(s) ∝ exp(-nu |s|^2)`` on that grid) for every
scale of the same geometry.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

from ..backend import ArrayType, to_device

if TYPE_CHECKING:
    from .constellation import Constellation

__all__ = ["maxwell_boltzmann", "optimal_nu"]


def _constellation_power(
    constellation: ArrayType, pmf: ArrayType | None = None
) -> float:
    r"""Average symbol power ``E[|s|^2]`` of a constellation.

    For a uniform constellation (``pmf=None``) this is the unweighted mean
    power ``mean(|s|^2)``.  For probabilistic shaping it is the pmf-weighted
    power ``Σ_m P(s_m) |s_m|^2`` - the quantity written ``E_PS`` when the
    constellation is on the normalised grid (where it is ``< 1``).

    The value is *grid-agnostic*: it reports the average power of the
    constellation exactly as passed - ``≈1`` for a normalised uniform grid,
    ``E_PS < 1`` for a normalised shaped grid, or the raw integer-grid energy
    (e.g. ``10`` for an unnormalised 16-QAM).  Callers apply their own
    rescaling (``√E_PS`` on the received symbols, or ``1/√E_PS`` on the
    constellation) using the returned value.

    This is the single source of truth for PS-QAM power across the library;
    prefer it over the inline ``Σ pmf·|s|^2`` idiom.

    Parameters
    ----------
    constellation : array_like
        Constellation points, shape ``(M,)``.  NumPy, CuPy, or list.
    pmf : array_like, optional
        Per-point probabilities, shape ``(M,)``, aligned with
        ``constellation`` and summing to 1 (e.g. from
        :func:`maxwell_boltzmann`).  ``None`` (uniform) returns the
        unweighted mean power.

    Returns
    -------
    float
        Host scalar ``E[|s|^2]``.

    Raises
    ------
    ValueError
        If ``pmf`` is supplied and its length does not match the
        constellation.
    """
    const = np.asarray(to_device(constellation, "cpu"))
    energies = np.abs(const).astype(np.float64) ** 2
    if pmf is None:
        return float(np.mean(energies))
    pmf_arr = np.asarray(to_device(pmf, "cpu"), dtype=np.float64).ravel()
    if pmf_arr.shape[0] != energies.shape[0]:
        raise ValueError(
            f"pmf length {pmf_arr.shape[0]} does not match constellation "
            f"length {energies.shape[0]}."
        )
    return float(np.dot(pmf_arr, energies))


def maxwell_boltzmann(constellation: Constellation, *, nu: float) -> np.ndarray:
    r"""Maxwell-Boltzmann prior over the points of a constellation.

    ``P(s_m) = exp(-nu |s'_m|^2) / Z``, where ``s'`` are the points scaled to
    a minimum distance of 2 (the odd-integer grid for QAM and PAM), so ``nu``
    is the literature value and does not depend on the constellation's scale.

    Parameters
    ----------
    constellation : Constellation
        Points to shape; any existing pmf is ignored.
    nu : float
        Shaping parameter, ``nu >= 0``.  ``nu = 0`` gives the uniform prior;
        larger values favour low-energy points.

    Returns
    -------
    np.ndarray
        pmf of shape ``(M,)``, float64, aligned with ``constellation.points``
        and summing to 1.
    """
    if nu < 0:
        raise ValueError(f"nu must be >= 0, got {nu}.")
    return _mb_pmf(_grid_energies(constellation.points), float(nu)).copy()


def optimal_nu(constellation: Constellation, *, entropy: float) -> float:
    r"""Shaping parameter ``nu`` whose Maxwell-Boltzmann prior has ``entropy``.

    Solves ``H(maxwell_boltzmann(constellation, nu=nu)) = entropy`` by
    bracketing and Brent's method (tolerance 1e-12).

    Parameters
    ----------
    constellation : Constellation
        Points to shape; any existing pmf is ignored.
    entropy : float
        Target entropy in bits per symbol, in ``(0, log2(M)]``.

    Returns
    -------
    float
        ``nu >= 0``; ``0.0`` when ``entropy = log2(M)``.
    """
    energies = _grid_energies(constellation.points)
    if np.ptp(energies) < 1e-9:
        raise ValueError(
            "constellation: every point has the same energy, so shaping cannot "
            "change the entropy."
        )
    return _nu_for_entropy(energies, float(entropy))


# -----------------------------------------------------------------------------
# Internals (shared with Constellation.shaped)
# -----------------------------------------------------------------------------


def _grid_energies(points: np.ndarray) -> np.ndarray:
    """Point energies on the grid scaled to a minimum distance of 2."""
    d = np.abs(points[:, None] - points[None, :])
    d_min = d[~np.eye(points.size, dtype=bool)].min()
    energies: np.ndarray = np.abs(points * (2.0 / d_min)) ** 2
    return energies


@lru_cache(maxsize=256)
def _mb_pmf_cached(energies: bytes, nu: float) -> np.ndarray:
    e = np.frombuffer(energies, dtype=np.float64)
    log_p = -nu * e
    p: np.ndarray = np.exp(log_p - log_p.max())
    p = p / p.sum()
    p.setflags(write=False)
    return p


def _mb_pmf(energies: np.ndarray, nu: float) -> np.ndarray:
    """Read-only MB pmf for ``energies`` (cached by value)."""
    return _mb_pmf_cached(np.ascontiguousarray(energies, np.float64).tobytes(), nu)


def _entropy_bits(pmf: np.ndarray) -> float:
    p = pmf[pmf > 0]
    return float(-np.sum(p * np.log2(p)))


def _nu_for_entropy(energies: np.ndarray, entropy: float) -> float:
    from scipy.optimize import brentq

    max_h = np.log2(energies.size)
    if not 0 < entropy <= max_h:
        raise ValueError(f"entropy must be in (0, {max_h:g}] bits, got {entropy}.")
    if np.isclose(entropy, max_h, atol=1e-8):
        return 0.0

    def gap(nu: float) -> float:
        return _entropy_bits(_mb_pmf(energies, nu)) - entropy

    nu_hi = 0.01
    while gap(nu_hi) > 0:
        nu_hi *= 10.0
    return float(brentq(gap, 0.0, nu_hi, xtol=1e-12, rtol=1e-12))
