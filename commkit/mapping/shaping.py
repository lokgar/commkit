"""
Probabilistic shaping (PS-QAM).

Maxwell-Boltzmann symbol distributions over a QAM constellation, the entropy /
shaping-parameter inversion, MB sampling, and the pmf-weighted constellation
power that ties the PS scale conventions together.
"""

from functools import lru_cache

import numpy as np

from ..backend import ArrayType, to_device
from .gray import gray_constellation

__all__ = [
    "constellation_power",
    "maxwell_boltzmann",
    "optimal_nu",
    "ps_entropy",
    "rescale_ps_symbols",
    "sample_ps_symbols",
]


def constellation_power(
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


def rescale_ps_symbols(
    rx: ArrayType, xp, modulation: str, order: int, pmf: ArrayType | None
) -> ArrayType:
    r"""Rescale unit-avg-power PS-QAM symbols ``{c·s_m}`` back to ``{s_m}``.

    Receive-path symbols normalised to unit average power (e.g. via
    ``resolve_symbols``) place PS-QAM symbols on the ``{s_m / sqrt(E_PS)}``
    grid (``c = 1/sqrt(E_PS)``).  Nearest-neighbour searches against
    :func:`gray_constellation` (hard demapping, EVM, SER, MI/LLR) expect
    symbols on the ``{s_m}`` grid instead.  This applies the exact
    deterministic correction ``rx -> rx·sqrt(E_PS)``.  No-op for uniform
    modulations (``pmf is None``) or when ``E_PS ≈ 1``.

    Parameters
    ----------
    rx : array_like
        Received symbols, any backend.
    xp : module
        ``rx``'s array module (NumPy/CuPy), used to build the dtype-matched
        scale factor.
    modulation, order : constellation spec, as accepted by
        :func:`gray_constellation`.
    pmf : array_like or None
        Symbol PMF of shape ``(order,)``.  ``None`` is a no-op.

    Returns
    -------
    array_like
        ``rx``, rescaled (or unchanged), same shape/backend as ``rx``.
    """
    if pmf is None:
        return rx
    e_ps = constellation_power(gray_constellation(modulation, order), pmf)
    if e_ps < 1.0 - 1e-6:
        rx = rx * xp.asarray(np.sqrt(e_ps), dtype=rx.real.dtype)
    return rx


@lru_cache(maxsize=256)
def maxwell_boltzmann(order: int, nu: float) -> np.ndarray:
    """
    Computes the Maxwell-Boltzmann PMF over a QAM constellation.

    P(s_m) = exp(-nu * |s_m|^2) / Z(nu), where |s_m|^2 is on the unnormalized
    integer grid (literature-compatible nu scale).  Indexed consistently with
    ``gray_constellation``.

    Parameters
    ----------
    order : int
        QAM modulation order (must be a power of 2, e.g. 16, 64, 256).
    nu : float
        Shaping parameter nu >= 0.  nu = 0 returns the uniform distribution.
        Larger nu concentrates probability on lower-energy points.

    Returns
    -------
    np.ndarray
        PMF array of shape ``(order,)``, dtype float64, summing to 1.
        Cached by ``(order, nu)``.
    """
    # Energies computed on the unnormalized integer grid for literature-compatible ν.
    # The PMF index ordering matches gray_constellation (normalized), since both
    # use the same Gray-code index assignment.
    unnorm_constellation = gray_constellation("qam", order, normalize=False)
    if nu == 0.0:
        return np.full(order, 1.0 / order, dtype=np.float64)
    energies = np.abs(unnorm_constellation) ** 2  # (M,) unnormalized energies
    log_p = -nu * energies
    log_p -= log_p.max()  # shift for numerical stability before exp
    p = np.exp(log_p)
    return p / p.sum()


def ps_entropy(order: int, nu: float) -> float:
    r"""
    Computes the per-symbol Shannon entropy under a Maxwell-Boltzmann distribution.

    H(X) = -sum_m P(s_m) * log2(P(s_m)) [bits/symbol]

    Parameters
    ----------
    order : int
        QAM modulation order.
    nu : float
        MB shaping parameter nu >= 0. nu = 0 returns log2(order).

    Returns
    -------
    float
        Entropy in bits per symbol. In the range (0, log2(order)].
    """
    pmf = maxwell_boltzmann(order, nu)
    nonzero = pmf > 0
    return float(-np.sum(pmf[nonzero] * np.log2(pmf[nonzero])))


def optimal_nu(order: int, entropy_bits: float) -> tuple:
    r"""
    Finds the MB shaping parameter ``ν`` that achieves a target per-symbol entropy.

    Uses ``scipy.optimize.brentq`` to bisect on
    ``ps_entropy(order, ν) - entropy_bits = 0``.

    Parameters
    ----------
    order : int
        QAM modulation order.
    entropy_bits : float
        Target per-symbol entropy in bits. Must be in ``(0, log₂(order)]``.

    Returns
    -------
    nu : float
        Shaping parameter achieving the target entropy.
    achieved_entropy : float
        Actual entropy at the returned ``nu`` (may differ by ``< 1e-6`` bits).

    Raises
    ------
    ValueError
        If ``entropy_bits`` is outside ``(0, log₂(order)]``.
    """
    from scipy.optimize import brentq

    max_h = np.log2(order)
    if not (0 < entropy_bits <= max_h):
        raise ValueError(
            f"entropy_bits must be in (0, {max_h:.3f}] for {order}-QAM, "
            f"got {entropy_bits:.4f}"
        )
    if np.isclose(entropy_bits, max_h, atol=1e-8):
        return 0.0, float(max_h)

    def _obj(nu):
        return ps_entropy(order, nu) - entropy_bits

    # Grow upper bound until entropy drops below target
    nu_hi = 0.01
    while ps_entropy(order, nu_hi) > entropy_bits:
        nu_hi *= 10.0

    nu_opt = float(brentq(_obj, 0.0, nu_hi, xtol=1e-9, rtol=1e-9))
    return nu_opt, ps_entropy(order, nu_opt)


def sample_ps_symbols(
    num_symbols: int,
    order: int,
    pmf: np.ndarray,
    seed: int | None = None,
) -> np.ndarray:
    """
    Draws QAM symbols from a Maxwell-Boltzmann distribution.

    Parameters
    ----------
    num_symbols : int
        Number of symbols to generate.
    order : int
        QAM modulation order.
    pmf : np.ndarray
        Symbol PMF of shape ``(order,)``. Typically from
        ``maxwell_boltzmann``.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Complex-valued symbols drawn from the MB distribution.
        Shape: ``(num_symbols,)``, dtype ``complex64``.
        All values lie exactly on the normalized QAM constellation grid.
    """
    rng = np.random.default_rng(seed)
    constellation = gray_constellation("qam", order).astype(np.complex64)
    indices = rng.choice(order, size=num_symbols, p=np.asarray(pmf, dtype=np.float64))
    return constellation[indices]
