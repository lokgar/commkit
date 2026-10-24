"""Soft-decision demapping (log-likelihood ratios): :func:`compute_llr`.

Max-log and exact (log-sum-exp) LLRs with an optional probabilistic-shaping
prior, computed in float32 on the input's device.  Sign convention: a
positive LLR means bit 0 is more likely,
``LLR_k = log P(b_k = 0 | y) / P(b_k = 1 | y)``.
"""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from .. import _cuda
from ..backend import ArrayType, dispatch
from ..core._signal_adapter import adapt_signal
from ..core.signal import Signal
from ..logger import logger

if TYPE_CHECKING:
    from .constellation import Constellation

__all__ = ["compute_llr"]

# Upper bound on the (chunk, bits, M/2) intermediate of the CuPy fallback
# (no CUDA kernel, or more than 16 bits per symbol), in elements.  Chunking
# over symbols keeps peak memory independent of the record length.
_CHUNK_ELEMENTS = 1 << 23


def compute_llr(
    symbols: ArrayType | Signal,
    *,
    noise_var: float,
    constellation: Constellation | None = None,
    method: str = "maxlog",
) -> ArrayType:
    """Compute bit log-likelihood ratios (LLRs) for soft-decision decoding.

    Positive LLR means bit 0 is more likely; the magnitude is the confidence.
    The constellation's ``pmf`` (if any) enters as the symbol prior.
    Computed on the input's device in float32 by one compiled pass per
    symbol (Numba on the CPU, a CUDA kernel on the GPU), so memory stays
    bounded for any record length.  On the CPU, square QAM whose labels and
    prior factor per axis (Gray QAM, shaped or not) is demapped per axis,
    which gives the same LLRs from 2·sqrt(M) metrics instead of M.
    Differentiable LLRs belong to commax.

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
    ndarray
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


_NUMBA_LLR: dict[str, Callable[..., None]] = {}

# Exact LLRs sum exp(m - P) with one peak P per symbol; a half-sum below this
# (its largest term under about e^-60 relative to P) is redone relative to its
# own max, so float32 underflow never costs more than ~1e-9 relative error.
_EXACT_FLOOR = np.float32(1e-26)
# Symbols per prange task: amortizes the scratch buffers of the CPU kernels.
_LLR_TASK = 256


def _get_numba_llr(name: str = "llr") -> Callable[..., None]:
    """Numba LLR kernels for the CPU, parallel over blocks of symbols.

    ``"llr"`` computes per symbol the M metrics
    ``m = -|x - s|^2 / sigma^2 + log P(s)``; ``"llr_separable"`` the
    sqrt(M) metrics of each axis of a separable constellation
    (:func:`_llr_separable`).  Each bit's LLR then comes from the metrics of
    the points whose bit is 0 and of those whose bit is 1:

    - max-log: the max over the first minus the max over the second;
    - exact: sums of ``e = exp(m - peak)``, computed once per point with the
      symbol's peak metric (which cancels in the difference), so one
      exponential per point instead of one per point and bit.

    float32, as the GPU paths.
    """
    if name not in _NUMBA_LLR:
        import numba

        floor = _EXACT_FLOOR
        task = _LLR_TASK
        jit = numba.njit(cache=True, fastmath=True, nogil=True, inline="always")

        @jit
        def prepare_exp(metric, e):  # type: ignore[no-untyped-def]
            """``e = exp(metric - peak)`` with the largest metric ``peak``."""
            peak = metric[0]
            for m in range(1, metric.shape[0]):
                if metric[m] > peak:
                    peak = metric[m]
            for m in range(metric.shape[0]):
                e[m] = np.exp(metric[m] - peak)
            return peak

        @jit
        def log_half_sum(metric, e, idx, v, b, peak):  # type: ignore[no-untyped-def]
            """log sum_j exp(metric[idx[v, b, j]] - peak), via ``e`` unless
            the half is underflowed there, else relative to its own max."""
            half = idx.shape[2]
            s = np.float32(0.0)
            for j in range(half):
                s += e[idx[v, b, j]]
            if s >= floor:
                return np.log(s)
            p = metric[idx[v, b, 0]]
            for j in range(1, half):
                p = max(p, metric[idx[v, b, j]])
            s = np.float32(0.0)
            for j in range(half):
                s += np.exp(metric[idx[v, b, j]] - p)
            return np.log(s) + (p - peak)

        @jit
        def bit_llr(metric, e, idx, b, peak, exact):  # type: ignore[no-untyped-def]
            """LLR of bit ``b``; ``idx[v, b]`` are the points with bit ``v``."""
            if exact:
                return log_half_sum(metric, e, idx, 0, b, peak) - log_half_sum(
                    metric, e, idx, 1, b, peak
                )
            p0 = metric[idx[0, b, 0]]
            p1 = metric[idx[1, b, 0]]
            for j in range(1, idx.shape[2]):
                p0 = max(p0, metric[idx[0, b, j]])
                p1 = max(p1, metric[idx[1, b, j]])
            return p0 - p1

        @jit
        def axis_llrs(x, lev, logp, idx, bits, inv_s2, exact, metric, e, out, i):  # type: ignore[no-untyped-def]
            """LLRs of one axis's bits into ``out[i, bits]``."""
            for m in range(lev.shape[0]):
                d = x - lev[m]
                metric[m] = -(d * d) * inv_s2 + logp[m]
            peak = prepare_exp(metric, e) if exact else np.float32(0.0)
            for b in range(bits.shape[0]):
                out[i, bits[b]] = bit_llr(metric, e, idx, b, peak, exact)

        @numba.njit(cache=True, fastmath=True, nogil=True, parallel=True)
        def llr_kernel(
            x_re: np.ndarray,
            x_im: np.ndarray,
            c_re: np.ndarray,
            c_im: np.ndarray,
            log_pmf: np.ndarray,
            idx: np.ndarray,
            inv_s2: np.float32,
            exact: bool,
            out: np.ndarray,
        ) -> None:
            n = x_re.shape[0]
            M = c_re.shape[0]
            k = idx.shape[1]
            for t in numba.prange((n + task - 1) // task):  # type: ignore[attr-defined,no-untyped-call]
                metric = np.empty(M, dtype=np.float32)
                e = np.empty(M, dtype=np.float32)
                for i in range(t * task, min(n, (t + 1) * task)):
                    for m in range(M):
                        dr = x_re[i] - c_re[m]
                        di = x_im[i] - c_im[m]
                        metric[m] = -(dr * dr + di * di) * inv_s2 + log_pmf[m]
                    peak = prepare_exp(metric, e) if exact else np.float32(0.0)
                    for b in range(k):
                        out[i, b] = bit_llr(metric, e, idx, b, peak, exact)

        @numba.njit(cache=True, fastmath=True, nogil=True, parallel=True)
        def llr_separable_kernel(
            x_re: np.ndarray,
            x_im: np.ndarray,
            lev_re: np.ndarray,
            lev_im: np.ndarray,
            logp_re: np.ndarray,
            logp_im: np.ndarray,
            idx_re: np.ndarray,
            idx_im: np.ndarray,
            bits_re: np.ndarray,
            bits_im: np.ndarray,
            inv_s2: np.float32,
            exact: bool,
            out: np.ndarray,
        ) -> None:
            n = x_re.shape[0]
            for t in numba.prange((n + task - 1) // task):  # type: ignore[attr-defined,no-untyped-call]
                m_re = np.empty(lev_re.shape[0], dtype=np.float32)
                e_re = np.empty(lev_re.shape[0], dtype=np.float32)
                m_im = np.empty(lev_im.shape[0], dtype=np.float32)
                e_im = np.empty(lev_im.shape[0], dtype=np.float32)
                for i in range(t * task, min(n, (t + 1) * task)):
                    axis_llrs(
                        x_re[i],
                        lev_re,
                        logp_re,
                        idx_re,
                        bits_re,
                        inv_s2,
                        exact,
                        m_re,
                        e_re,
                        out,
                        i,
                    )
                    axis_llrs(
                        x_im[i],
                        lev_im,
                        logp_im,
                        idx_im,
                        bits_im,
                        inv_s2,
                        exact,
                        m_im,
                        e_im,
                        out,
                        i,
                    )

        _NUMBA_LLR["llr"] = llr_kernel
        _NUMBA_LLR["llr_separable"] = llr_separable_kernel
    kernel: Callable[..., None] = _NUMBA_LLR[name]
    return kernel


def _half_indices(bit_labels: np.ndarray) -> np.ndarray:
    """Indices of the points whose bit b is 0 and 1: ``(2, k, M/2)`` int64."""
    k = bit_labels.shape[1]
    return np.stack(
        [
            np.stack([np.flatnonzero(bit_labels[:, b] == v) for b in range(k)])
            for v in (0, 1)
        ]
    ).astype(np.int64)


def _check_method(method: str) -> None:
    if method not in ("maxlog", "exact"):
        raise ValueError(f"Unknown method: {method}. Use 'maxlog' or 'exact'.")


def _log_prior(pmf: np.ndarray | None, order: int) -> np.ndarray:
    """float32 log prior; zeros (a constant that cancels) for no prior."""
    if pmf is None:
        return np.zeros(order, dtype=np.float32)
    return np.log(np.clip(np.asarray(pmf, np.float64), 1e-40, None)).astype(np.float32)


def _llr(
    symbols: ArrayType,
    points: np.ndarray,
    bit_labels: np.ndarray,
    pmf: np.ndarray | None,
    noise_var: float,
    method: str,
) -> ArrayType:
    """LLRs of ``symbols`` against host ``points`` labeled by ``bit_labels``.

    Shared by :func:`compute_llr` and :meth:`Constellation.llr`.  Returns
    float32 LLRs of shape ``(..., N * k)`` on the input's device.
    """
    _check_method(method)
    order, k = bit_labels.shape
    symbols, xp, _ = dispatch(symbols)
    is_complex = symbols.dtype.kind == "c"
    sym_flat = symbols.reshape(-1).astype(xp.complex64 if is_complex else xp.float32)
    inv_sigma2 = np.float32(1.0 / max(noise_var, 1e-20))
    log_pmf = _log_prior(pmf, order)

    n = sym_flat.shape[0]
    llrs = xp.empty((n, k), dtype=xp.float32)
    if xp is np:
        # CPU: one parallel pass, no (chunk, k, M/2) intermediates.
        _get_numba_llr()(
            np.ascontiguousarray(sym_flat.real, dtype=np.float32),
            np.ascontiguousarray(
                sym_flat.imag if is_complex else np.zeros(n), dtype=np.float32
            ),
            np.ascontiguousarray(np.real(points), dtype=np.float32),
            np.ascontiguousarray(
                np.imag(points) if is_complex else np.zeros(order), dtype=np.float32
            ),
            log_pmf,
            _half_indices(bit_labels),
            inv_sigma2,
            method == "exact",
            llrs,
        )
        return llrs.reshape((*symbols.shape[:-1], symbols.shape[-1] * k))
    kernel = _cuda.get_kernel("llr") if k <= 16 else None
    if kernel is not None:
        # GPU: one thread per symbol, no (chunk, k, M/2) intermediates.
        labels = (bit_labels.astype(np.int64) << np.arange(k - 1, -1, -1)).sum(-1)
        llrs = kernel(
            sym_flat.astype(xp.complex64, copy=False),
            xp.asarray(points.astype(np.complex64)),
            xp.asarray(log_pmf),
            xp.asarray(labels.astype(np.int32)),
            k=k,
            inv_s2=float(inv_sigma2),
            exact=method == "exact",
        )
        return llrs.reshape((*symbols.shape[:-1], symbols.shape[-1] * k))
    # CuPy fallback (no CUDA kernel, or more than 16 bits per symbol).
    const = xp.asarray(points.astype(np.complex64 if is_complex else np.float32))
    idx0, idx1 = (xp.asarray(i) for i in _half_indices(bit_labels))  # (k, M/2)
    log_pmf_dev = None if pmf is None else xp.asarray(log_pmf)
    chunk = max(1, _CHUNK_ELEMENTS // (k * order))
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


class _Axis(NamedTuple):
    """One axis of a separable constellation (see ``Constellation._axes``)."""

    levels: np.ndarray  # (L,) real levels on this axis
    labels: np.ndarray  # (L, k_axis) labels of this axis's bits
    pmf: np.ndarray | None  # (L,) marginal prior
    bits: np.ndarray  # (k_axis,) positions of these bits in the symbol label


def _llr_separable(
    symbols: np.ndarray, axes: tuple[_Axis, _Axis], noise_var: float, method: str
) -> np.ndarray:
    """CPU LLRs of a separable constellation, one pass over both axes.

    The metric splits as ``m(s) = m_re(s_re) + m_im(s_im)`` and every bit
    depends on one axis, so the other axis's factor cancels from its LLR in
    both methods: 2·sqrt(M) metrics per symbol instead of M.
    """
    _check_method(method)
    re, im = axes
    flat = np.asarray(symbols).reshape(-1)
    k = re.bits.size + im.bits.size
    llrs = np.empty((flat.size, k), dtype=np.float32)
    _get_numba_llr("llr_separable")(
        np.ascontiguousarray(flat.real, dtype=np.float32),
        np.ascontiguousarray(flat.imag, dtype=np.float32),
        re.levels.astype(np.float32),
        im.levels.astype(np.float32),
        _log_prior(re.pmf, re.levels.size),
        _log_prior(im.pmf, im.levels.size),
        _half_indices(re.labels),
        _half_indices(im.labels),
        re.bits.astype(np.int64),
        im.bits.astype(np.int64),
        np.float32(1.0 / max(noise_var, 1e-20)),
        method == "exact",
        llrs,
    )
    return llrs.reshape((*symbols.shape[:-1], symbols.shape[-1] * k))


def _logsumexp(a: ArrayType, xp: ModuleType) -> ArrayType:
    """Stable log-sum-exp over the last axis."""
    peak = a.max(axis=-1, keepdims=True)
    return xp.log(xp.exp(a - peak).sum(axis=-1)) + peak[..., 0]
