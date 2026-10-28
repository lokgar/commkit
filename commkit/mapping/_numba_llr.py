"""Numba kernels of :func:`commkit.mapping.compute_llr` (private).

The CPU LLR kernels, parallel over blocks of symbols, documented by the
getter ``_get_numba_llr`` of :mod:`commkit.mapping.llr`: :func:`llr_kernel`
(the metrics of every point), :func:`llr_separable_kernel` (the metrics of
the levels of each axis) and :func:`llr_separable_maxlog_kernel` (the
closed form).  Imported on first use, so ``import commkit`` does not load
Numba; at module level, so Numba's disk cache holds the kernels, which
call each other's helpers.
"""

import numba
import numpy as np

# Exact LLRs sum exp(m - P) with one peak P per symbol; a half-sum below this
# (its largest term under about e^-60 relative to P) is redone relative to its
# own max, so float32 underflow never costs more than ~1e-9 relative error.
_EXACT_FLOOR = np.float32(1e-26)
# Symbols per prange task: amortizes the scratch buffers of the CPU kernels.
_LLR_TASK = 256
# A step shared by the kernels, inlined into each before it is compiled.
_inline = numba.njit(cache=True, fastmath=True, nogil=True, inline="always")


@_inline
def prepare_exp(metric, e):  # type: ignore[no-untyped-def]
    """``e = exp(metric - peak)`` with the largest metric ``peak``."""
    peak = metric[0]
    for m in range(1, metric.shape[0]):
        if metric[m] > peak:
            peak = metric[m]
    for m in range(metric.shape[0]):
        e[m] = np.exp(metric[m] - peak)
    return peak


@_inline
def log_half_sum(metric, e, idx, v, b, peak):  # type: ignore[no-untyped-def]
    """``log sum_j exp(metric[idx[v, b, j]] - peak)`` of one half.

    Summed from ``e``, or relative to the half's own maximum where
    it underflows in ``e``.
    """
    half = idx.shape[2]
    s = np.float32(0.0)
    for j in range(half):
        s += e[idx[v, b, j]]
    if s >= _EXACT_FLOOR:
        return np.log(s)
    p = metric[idx[v, b, 0]]
    for j in range(1, half):
        p = max(p, metric[idx[v, b, j]])
    s = np.float32(0.0)
    for j in range(half):
        s += np.exp(metric[idx[v, b, j]] - p)
    return np.log(s) + (p - peak)


@_inline
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


@_inline
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
    """LLRs from the metrics of every point (see ``_get_numba_llr``)."""
    n = x_re.shape[0]
    M = c_re.shape[0]
    k = idx.shape[1]
    for t in numba.prange((n + _LLR_TASK - 1) // _LLR_TASK):  # type: ignore[attr-defined,no-untyped-call]
        metric = np.empty(M, dtype=np.float32)
        e = np.empty(M, dtype=np.float32)
        for i in range(t * _LLR_TASK, min(n, (t + 1) * _LLR_TASK)):
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
    """LLRs from the metrics of the levels of each axis."""
    n = x_re.shape[0]
    for t in numba.prange((n + _LLR_TASK - 1) // _LLR_TASK):  # type: ignore[attr-defined,no-untyped-call]
        m_re = np.empty(lev_re.shape[0], dtype=np.float32)
        e_re = np.empty(lev_re.shape[0], dtype=np.float32)
        m_im = np.empty(lev_im.shape[0], dtype=np.float32)
        e_im = np.empty(lev_im.shape[0], dtype=np.float32)
        for i in range(t * _LLR_TASK, min(n, (t + 1) * _LLR_TASK)):
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


@_inline
def axis_maxlog(x, lev, lab, lo, hi, pos, inv_s2, out, i):  # type: ignore[no-untyped-def]
    """Closed-form max-log LLRs of one axis's bits into ``out[i, pos]``."""
    L = lev.shape[0]
    kb = pos.shape[0]
    s = int(np.rint((x - lev[0]) / (lev[1] - lev[0])))
    s = min(max(s, 0), L - 1)
    t = x - lev[s]
    # Near a midpoint the rounded quotient can land one level off:
    # keep the nearer of s and its neighbor towards x.
    s2 = s + 1 if t > 0 else s - 1
    if 0 <= s2 < L:
        t2 = x - lev[s2]
        if t2 * t2 < t * t:
            s = s2
            t = t2
    m_own = -(t * t) * inv_s2
    for b in range(kb):
        # Both halves are never empty: one of lo, hi is a level.
        m_other = np.float32(-3.0e38)
        j = lo[b, s]
        if j >= 0:
            t = x - lev[j]
            m_other = max(m_other, -(t * t) * inv_s2)
        j = hi[b, s]
        if j >= 0:
            t = x - lev[j]
            m_other = max(m_other, -(t * t) * inv_s2)
        if (lab[s] >> (kb - 1 - b)) & 1:
            out[i, pos[b]] = m_other - m_own
        else:
            out[i, pos[b]] = m_own - m_other


@numba.njit(cache=True, fastmath=True, nogil=True, parallel=True)
def llr_separable_maxlog_kernel(
    x_re: np.ndarray,
    x_im: np.ndarray,
    lev_re: np.ndarray,
    lab_re: np.ndarray,
    lo_re: np.ndarray,
    hi_re: np.ndarray,
    pos_re: np.ndarray,
    lev_im: np.ndarray,
    lab_im: np.ndarray,
    lo_im: np.ndarray,
    hi_im: np.ndarray,
    pos_im: np.ndarray,
    inv_s2: np.float32,
    out: np.ndarray,
) -> None:
    """Closed-form max-log LLRs of a uniform separable constellation."""
    n = x_re.shape[0]
    for t in numba.prange((n + _LLR_TASK - 1) // _LLR_TASK):  # type: ignore[attr-defined,no-untyped-call]
        for i in range(t * _LLR_TASK, min(n, (t + 1) * _LLR_TASK)):
            axis_maxlog(x_re[i], lev_re, lab_re, lo_re, hi_re, pos_re, inv_s2, out, i)
            axis_maxlog(x_im[i], lev_im, lab_im, lo_im, hi_im, pos_im, inv_s2, out, i)
