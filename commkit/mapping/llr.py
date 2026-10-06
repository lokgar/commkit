"""
Soft-decision demapping (log-likelihood ratios).

Max-log and exact (log-sum-exp) LLRs with an optional probabilistic-shaping
prior, computed with NumPy or CuPy on the input's device.  Sign convention:
positive LLR -> bit 0 more likely.
"""

import numpy as np

from ..backend import ArrayType, dispatch
from ..core._signal_adapter import adapt_signal
from ..core.signal import Signal
from ..logger import logger
from .gray import gray_constellation, unpack_bits

__all__ = ["compute_llr"]

# Upper bound on the (chunk, bits, M/2) intermediate, in elements.  Chunking
# over symbols keeps peak memory independent of the record length.
_CHUNK_ELEMENTS = 1 << 23


def compute_llr(
    symbols: ArrayType | Signal,
    modulation: str | None = None,
    order: int | None = None,
    noise_var: float | None = None,
    method: str = "maxlog",
    unipolar: bool = False,
    pmf: np.ndarray | None = None,
) -> ArrayType:
    """
    Compute Log-Likelihood Ratios (LLRs) for soft-decision decoding.

    Positive LLR -> bit 0 more likely; negative -> bit 1; magnitude = confidence.
    Computed on the input's device (NumPy or CuPy) in float32, chunked over
    symbols so memory stays bounded for any record length.  Differentiable
    LLRs belong to commax.

    Parameters
    ----------
    symbols : array_like or Signal
        Received noisy symbols. Shape: (..., N_symbols). NumPy or CuPy.
        A :class:`Signal` supplies ``resolved_symbols`` and defaults
        ``modulation``/``order``/``pmf`` from its metadata when not given
        explicitly.
    modulation : {"psk", "qam", "ask"}, optional
        Modulation type.  Required for array input; for :class:`Signal`
        input, used only as a fallback when the signal's ``mod_scheme`` is
        unset.
    order : int, optional
        Modulation order.  Required for array input; for :class:`Signal`
        input, used only as a fallback when the signal's ``mod_order`` is
        unset.
    noise_var : float
        Complex noise variance sigma^2 referenced to the normalised
        constellation (unit avg power).  sigma^2 = 10^(-EsN0_dB / 10).
    method : {"maxlog", "exact"}, default "maxlog"
        LLR algorithm. ``"maxlog"`` is faster; ``"exact"`` uses log-sum-exp.
    unipolar : bool, default False
        Use unipolar constellation for ASK/PAM.
    pmf : np.ndarray, optional
        Symbol PMF of shape ``(order,)`` for PS-QAM.  Pass
        ``maxwell_boltzmann(order, nu)`` to incorporate the non-uniform prior.
        ``None`` assumes uniform prior.  For :class:`Signal` input, used
        only as a fallback when the signal's ``ps_pmf`` is unset.

    Returns
    -------
    array_like
        LLR values, float32, on the input's device.
        Shape: (..., N_symbols * log2(order)); the ``log2(order)`` bits of
        each symbol are adjacent (MSB first).

    Notes
    -----
    Max-Log: LLR_k ≈ (1/sigma^2) * (min_{S_1^k} |r-s|^2 - min_{S_0^k} |r-s|^2).
    Exact: LLR_k = log sum_{S_0^k} exp(-|r-s|^2/sigma^2) - log sum_{S_1^k} ...

    For PS-QAM, ``symbols`` must be on the same scale as
    ``gray_constellation`` (unit avg power).  After
    ``resolve_symbols`` the receiver renormalises;
    use ``gmi`` instead for correct scale.
    """
    signal_adapter = adapt_signal(
        symbols, function_name="compute_llr()", field="resolved_symbols"
    )
    symbols = signal_adapter.array
    if signal_adapter.signal is not None:
        if symbols is None:
            raise ValueError(
                "No resolved symbols available. Call resolve_symbols(sig) first."
            )
        modulation = signal_adapter.resolve_optional("mod_scheme", modulation)
        order = signal_adapter.resolve_optional("mod_order", order)
        pmf = signal_adapter.resolve_optional("ps_pmf", pmf)

    if modulation is None or order is None:
        raise ValueError("compute_llr() requires modulation and order for array input.")
    if noise_var is None:
        raise ValueError("compute_llr() requires noise_var.")
    if symbols is None:
        raise ValueError("compute_llr() requires resolved symbols.")
    logger.debug(
        "Computing LLRs for %s %s-level (method=%s).", modulation.upper(), order, method
    )

    k = int(np.log2(order))
    if 2**k != order:
        raise ValueError(f"Order must be a power of 2, got {order}")
    const = gray_constellation(modulation, order, unipolar=unipolar)
    labels = unpack_bits(np.arange(order, dtype="int32"), k)  # (M, k), MSB first
    return _llr(symbols, const, labels, pmf, noise_var, method)


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


def _logsumexp(a: ArrayType, xp) -> ArrayType:
    """Stable log-sum-exp over the last axis."""
    peak = a.max(axis=-1, keepdims=True)
    return xp.log(xp.exp(a - peak).sum(axis=-1)) + peak[..., 0]
