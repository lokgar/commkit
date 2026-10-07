"""
Soft-decision demapping (log-likelihood ratios).

Max-log and exact (log-sum-exp) LLRs with an optional probabilistic-shaping
prior, computed with NumPy or CuPy on the input's device.  Sign convention:
positive LLR -> bit 0 more likely.
"""

from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING

import numpy as np

from ..backend import ArrayType, dispatch
from ..core._signal_adapter import adapt_signal
from ..core.signal import Signal
from ..logger import logger

if TYPE_CHECKING:
    from .constellation import Constellation

__all__ = ["compute_llr"]

# Upper bound on the (chunk, bits, M/2) intermediate, in elements.  Chunking
# over symbols keeps peak memory independent of the record length.
_CHUNK_ELEMENTS = 1 << 23


def compute_llr(
    symbols: ArrayType | Signal,
    *,
    noise_var: float,
    constellation: Constellation | None = None,
    method: str = "maxlog",
) -> ArrayType:
    """
    Bit log-likelihood ratios for soft-decision decoding.

    Positive LLR means bit 0 is more likely; the magnitude is the confidence.
    The constellation's ``pmf`` (if any) enters as the symbol prior.
    Computed on the input's device in float32, chunked over symbols so memory
    stays bounded for any record length.  Differentiable LLRs belong to
    commax.

    Parameters
    ----------
    symbols : array_like or Signal
        Received symbols at one sample per symbol, shape ``(..., N)``, on the
        constellation's scale.  A :class:`Signal` must be at one sample per
        symbol.
    noise_var : float
        Complex noise variance ``sigma^2 = E[|n|^2]`` on the constellation's
        scale.  For unit-power constellations, ``sigma^2 = 10^(-EsN0_dB/10)``.
    constellation : Constellation, optional
        Points, bit labels and prior.  Defaults to the Signal's
        ``constellation``; required for array input.
    method : {"maxlog", "exact"}, default "maxlog"
        ``"maxlog"`` keeps the largest term; ``"exact"`` uses log-sum-exp.

    Returns
    -------
    array_like
        float32 LLRs of shape ``(..., N * k)`` on the input's device; the
        ``k`` bits of each symbol are adjacent (MSB first).

    Notes
    -----
    Max-log: ``LLR_b = max_{s: b=0} m(s) - max_{s: b=1} m(s)``; exact:
    ``LLR_b = log sum_{s: b=0} e^{m(s)} - log sum_{s: b=1} e^{m(s)}``, with
    ``m(s) = -|r - s|^2 / sigma^2 + log P(s)``.
    """
    from .constellation import Constellation

    signal_adapter = adapt_signal(symbols, function_name="compute_llr()")
    x = signal_adapter.symbol_array()
    constellation = signal_adapter.resolve_choice("constellation", constellation)
    if constellation is None:
        raise ValueError(
            "compute_llr() needs a constellation: pass constellation= or a "
            "Signal that has one."
        )
    if not isinstance(constellation, Constellation):
        raise TypeError(
            "compute_llr(): constellation must be a Constellation, got "
            f"{type(constellation).__name__}; use e.g. Constellation.qam(16)."
        )
    logger.debug("Computing LLRs for %r (method=%s).", constellation, method)
    return constellation.llr(x, noise_var=noise_var, method=method)


def _llr(
    symbols: ArrayType,
    points: np.ndarray,
    bit_labels: np.ndarray,
    pmf: np.ndarray | None,
    noise_var: float,
    method: str,
) -> ArrayType:
    """LLRs of ``symbols`` against host ``points`` labelled by ``bit_labels``.

    Shared by :func:`compute_llr` and :meth:`Constellation.llr`.  Returns
    float32 LLRs of shape ``(..., N * k)`` on the input's device.
    """
    if method not in ("maxlog", "exact"):
        raise ValueError(f"Unknown method: {method}. Use 'maxlog' or 'exact'.")
    order, k = bit_labels.shape
    symbols, xp, _ = dispatch(symbols)
    is_complex = symbols.dtype.kind == "c"
    sym_flat = symbols.reshape(-1).astype(xp.complex64 if is_complex else xp.float32)

    const = xp.asarray(points.astype(np.complex64 if is_complex else np.float32))
    # Column indices of the M/2 points whose bit b is 0 (resp. 1): (k, M/2).
    idx0 = xp.asarray(
        np.stack([np.flatnonzero(bit_labels[:, b] == 0) for b in range(k)])
    )
    idx1 = xp.asarray(
        np.stack([np.flatnonzero(bit_labels[:, b] == 1) for b in range(k)])
    )

    inv_sigma2 = np.float32(1.0 / max(noise_var, 1e-20))
    if pmf is not None:
        log_pmf = np.log(np.clip(np.asarray(pmf, dtype=np.float64), 1e-40, None))
        log_pmf_dev = xp.asarray(log_pmf.astype(np.float32))
    else:
        log_pmf_dev = None  # uniform prior: a constant that cancels

    n = sym_flat.shape[0]
    chunk = max(1, _CHUNK_ELEMENTS // (k * order))
    llrs = xp.empty((n, k), dtype=xp.float32)
    for n0 in range(0, n, chunk):
        x = sym_flat[n0 : n0 + chunk]
        diff = x[:, None] - const[None, :]  # (chunk, M)
        # log-likelihood (up to a shared constant): -|x - s|^2/sigma^2 + log P(s)
        metric = (
            -(diff.real**2 + diff.imag**2) * inv_sigma2
            if is_complex
            else -(diff**2) * inv_sigma2
        )
        if log_pmf_dev is not None:
            metric = metric + log_pmf_dev
        m0 = metric[:, idx0]  # (chunk, k, M/2)
        m1 = metric[:, idx1]
        if method == "maxlog":
            llrs[n0 : n0 + chunk] = m0.max(axis=-1) - m1.max(axis=-1)
        else:
            llrs[n0 : n0 + chunk] = _logsumexp(m0, xp) - _logsumexp(m1, xp)

    out_shape = (*symbols.shape[:-1], symbols.shape[-1] * k)
    return llrs.reshape(out_shape)


def _logsumexp(a: ArrayType, xp: ModuleType) -> ArrayType:
    """Stable log-sum-exp over the last axis."""
    peak = a.max(axis=-1, keepdims=True)
    return xp.log(xp.exp(a - peak).sum(axis=-1)) + peak[..., 0]
