"""Soft-decision demapping (log-likelihood ratios): :func:`compute_llr`.

Max-log and exact (log-sum-exp) LLRs with an optional probabilistic-shaping
prior, computed in float32 on the input's device.  Sign convention: a
positive LLR means bit 0 is more likely,
``LLR_k = log P(b_k = 0 | y) / P(b_k = 1 | y)``.

The arithmetic lives in the private ``_demap`` module, below
:class:`Constellation`, which computes the LLRs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..backend import ArrayType
from ..core._signal_adapter import adapt_signal
from ..core.signal import Signal
from ..logger import logger
from ._demap import _noise_variance

if TYPE_CHECKING:
    from .constellation import Constellation

__all__ = ["compute_llr"]


def compute_llr(
    symbols: ArrayType | Signal,
    *,
    noise_var: float | None = None,
    esn0_db: float | None = None,
    constellation: Constellation | None = None,
    method: str = "maxlog",
) -> ArrayType:
    """Compute bit log-likelihood ratios (LLRs) for soft-decision decoding.

    Positive LLR means bit 0 is more likely; the magnitude is the confidence.
    The constellation's ``pmf`` (if any) enters as the symbol prior.
    Computed on the input's device in float32 by one compiled pass per
    symbol (Numba on the CPU, a CUDA kernel on the GPU), so memory stays
    bounded for any record length.  Square QAM whose labels and prior factor
    per axis (Gray QAM, shaped or not) is demapped per axis, which gives the
    same LLRs from 2·sqrt(M) metrics instead of M.
    Differentiable LLRs belong to commax.

    Parameters
    ----------
    symbols : array_like or Signal
        Received symbols at one sample per symbol, ``(..., N)``, on the
        constellation's scale.  A :class:`Signal` must be at one sample per
        symbol.
    noise_var : float, optional
        Variance ``sigma^2 = E[|n|^2]`` of the noise in ``symbols``, on the
        constellation's scale: of circular complex noise for complex
        symbols, of real noise for real ones (what
        :func:`~commkit.metrics.estimate_noise_variance` measures).  Give
        this or ``esn0_db``.
    esn0_db : float, optional
        Es/N0 of the channel in dB, instead of ``noise_var``: Es is the
        constellation's average energy, and the noise variance follows as
        for :func:`~commkit.impairments.apply_awgn` (``Es / (Es/N0)`` for
        complex symbols, half of it for real ones).
    constellation : Constellation, optional
        Points, bit labels and prior.  Defaults to the Signal's
        ``constellation``; required for array input.
    method : {"maxlog", "exact"}, default "maxlog"
        ``"maxlog"`` keeps the largest term; ``"exact"`` uses log-sum-exp.

    Returns
    -------
    ndarray
        float32 LLRs, ``(..., N * k)``, on the input's device; the
        ``k`` bits of each symbol are adjacent (MSB first).

    Raises
    ------
    ValueError
        If neither or both of ``noise_var`` and ``esn0_db`` are given, and
        for GPU input and a constellation of more than 16 bits per symbol,
        the limit of the CUDA kernels.

    Notes
    -----
    Max-log: ``LLR_b = max_{s: b=0} m(s) - max_{s: b=1} m(s)``; exact:
    ``LLR_b = log sum_{s: b=0} e^{m(s)} - log sum_{s: b=1} e^{m(s)}``, with
    ``m(s) = -|r - s|^2 / sigma^2 + log P(s)`` for complex symbols and
    ``m(s) = -(r - s)^2 / (2 sigma^2) + log P(s)`` for real ones: the
    Gaussian log-likelihood of the noise the symbols carry.
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
    noise_var = _noise_variance(
        noise_var, esn0_db, constellation, x.dtype.kind != "c", "compute_llr()"
    )
    logger.debug("Computing LLRs for %r (method=%s).", constellation, method)
    return constellation.llr(x, noise_var=noise_var, method=method)
