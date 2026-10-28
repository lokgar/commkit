"""Soft-decision demapping (log-likelihood ratios): :func:`compute_llr`.

Max-log and exact (log-sum-exp) LLRs with an optional probabilistic-shaping
prior, computed in float32 on the input's device.  Sign convention: a
positive LLR means bit 0 is more likely,
``LLR_k = log P(b_k = 0 | y) / P(b_k = 1 | y)``.

Three paths: the metrics of every point (any constellation); the metrics
of the levels of each axis of a separable square lattice (exact LLRs, or a
shaping prior); and, for max-log LLRs of a uniform separable lattice, a
closed form.  There the largest metric of each bit's half is that of the
nearest level with that bit value: the snapped level for its own value,
and for the other value the nearer of the two levels just outside the
snapped level's run of equal bits (1 + 2k metrics per axis instead of
sqrt(M)), on the GPU at every order and on the CPU from 256-QAM, below
which the level metrics are as fast.  A prior moves the maxima off the
nearest levels, so it keeps the level metrics.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from .. import _cuda
from ..backend import ArrayType, _device_tables, dispatch
from ..core._signal_adapter import adapt_signal
from ..core.signal import Signal
from ..logger import logger
from .gray import _pack

if TYPE_CHECKING:
    from .constellation import Constellation

__all__ = ["compute_llr"]


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
        float32 LLRs, ``(..., N * k)``, on the input's device; the
        ``k`` bits of each symbol are adjacent (MSB first).

    Raises
    ------
    ValueError
        For GPU input and a constellation of more than 16 bits per symbol,
        the limit of the CUDA kernels.

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


def _get_numba_llr(name: str = "llr") -> Callable[..., None]:
    """Numba LLR kernels for the CPU, parallel over blocks of symbols.

    ``"llr"`` computes per symbol the M metrics
    ``m = -|x - s|^2 / sigma^2 + log P(s)``; ``"llr_separable"`` the
    sqrt(M) metrics of each axis of a separable constellation
    (:func:`_llr_separable`); ``"llr_separable_maxlog"`` the closed-form
    max-log LLRs of a uniform separable constellation (module docstring),
    1 + 2k metrics per axis.  The first two take each bit's LLR from the
    metrics of the points whose bit is 0 and of those whose bit is 1:

    - max-log: the max over the first minus the max over the second;
    - exact: sums of ``e = exp(m - peak)``, computed once per point with the
      symbol's peak metric (which cancels in the difference), so one
      exponential per point instead of one per point and bit.

    float32, as the GPU paths.
    """
    from . import _numba_llr

    kernels: dict[str, Callable[..., None]] = {
        "llr": _numba_llr.llr_kernel,
        "llr_separable": _numba_llr.llr_separable_kernel,
        "llr_separable_maxlog": _numba_llr.llr_separable_maxlog_kernel,
    }
    return kernels[name]


def _half_indices(bit_labels: np.ndarray) -> np.ndarray:
    """Indices of the points whose bit b is 0 and 1: ``(2, k, M/2)`` int64."""
    k = bit_labels.shape[1]
    return np.stack(
        [
            np.stack([np.flatnonzero(bit_labels[:, b] == v) for b in range(k)])
            for v in (0, 1)
        ]
    ).astype(np.int64)


def _opposite_levels(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest levels of the other bit value, for the closed-form max-log.

    For bit ``b`` and level ``m`` of one axis (``labels`` ``(L, k)``, levels
    ascending): ``lo[b, m]``, the highest level below ``m`` whose bit ``b``
    differs from level ``m``'s, and ``hi[b, m]``, the lowest one above it;
    -1 where there is none.  These are the levels just outside the run of
    equal bits that holds ``m``.  int32 ``(k, L)`` each.
    """
    L, k = labels.shape
    lo = np.empty((k, L), dtype=np.int32)
    hi = np.empty((k, L), dtype=np.int32)
    for b in range(k):
        col = labels[:, b]
        change = np.flatnonzero(col[1:] != col[:-1])  # level j differs from j + 1
        run = np.concatenate([[0], np.cumsum(col[1:] != col[:-1])])
        starts = np.concatenate([[0], change + 1])
        ends = np.concatenate([change, [L - 1]])
        lo[b] = starts[run] - 1
        hi[b] = np.where(ends[run] + 1 < L, ends[run] + 1, -1)
    return lo, hi


# Fewest levels per axis from which the CPU takes the closed form.  Against
# the level metrics, 2^18 symbols: 1.21x at 16-QAM (4 levels), 1.05x at 64,
# 0.65x at 256, 0.40x at 1024, 0.23x at 4096; on the GPU it is faster at
# every order, so the GPU always takes it.
_CLOSED_FORM_MIN_LEVELS_CPU = 16


def _closed_form(axes: tuple[_Axis, _Axis], method: str, min_levels: int = 2) -> bool:
    """Max-log LLRs of a uniform separable constellation, with at least
    ``min_levels`` levels per axis: the closed form."""
    return method == "maxlog" and all(
        a.pmf is None and a.levels.size >= min_levels for a in axes
    )


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
    owner: Constellation | None = None,
) -> ArrayType:
    """LLRs of ``symbols`` against host ``points`` labeled by ``bit_labels``.

    Behind :meth:`Constellation.llr` (and so :func:`compute_llr`), which
    passes itself as ``owner`` to cache the GPU tables.  Returns float32
    LLRs, ``(..., N * k)``, on the input's device.
    """
    _check_method(method)
    order, k = bit_labels.shape
    symbols, xp, _ = dispatch(symbols)
    is_complex = symbols.dtype.kind == "c"
    sym_flat = symbols.reshape(-1).astype(xp.complex64 if is_complex else xp.float32)
    inv_sigma2 = np.float32(1.0 / max(noise_var, 1e-20))
    log_pmf = _log_prior(pmf, order)

    n = sym_flat.shape[0]
    if xp is np:
        # CPU: one parallel pass, no (chunk, k, M/2) intermediates.
        llrs = np.empty((n, k), dtype=np.float32)
        _get_numba_llr()(
            np.ascontiguousarray(sym_flat.real, dtype=np.float32),
            np.ascontiguousarray(
                sym_flat.imag if is_complex else np.zeros(n, np.float32),
                dtype=np.float32,
            ),
            np.ascontiguousarray(np.real(points), dtype=np.float32),
            np.ascontiguousarray(
                np.imag(points) if is_complex else np.zeros(order, np.float32),
                dtype=np.float32,
            ),
            log_pmf,
            _half_indices(bit_labels),
            inv_sigma2,
            method == "exact",
            llrs,
        )
        return llrs.reshape((*symbols.shape[:-1], symbols.shape[-1] * k))
    # GPU: one thread per symbol, likewise.
    _require_gpu_bits(k)

    def build() -> tuple[np.ndarray, ...]:
        return (
            points.astype(np.complex64),
            log_pmf,
            _packed_labels(bit_labels),
        )

    pts_dev, log_pmf_dev, labels_dev = _device_tables(owner, "llr", xp, build)
    llrs = _cuda.get_kernel("llr")(
        sym_flat.astype(xp.complex64, copy=False),
        pts_dev,
        log_pmf_dev,
        labels_dev,
        k=k,
        inv_s2=float(inv_sigma2),
        exact=method == "exact",
    )
    return llrs.reshape((*symbols.shape[:-1], symbols.shape[-1] * k))


def _require_gpu_bits(k: int) -> None:
    """Raise for more bits per symbol than the CUDA LLR kernels handle."""
    if k > _cuda._LLR_MAX_K:
        raise ValueError(
            f"compute_llr(): GPU input supports at most {_cuda._LLR_MAX_K} bits "
            f"per symbol, the constellation has {k}"
        )


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
    both methods: 2·sqrt(M) metrics per symbol instead of M.  Max-log LLRs
    without a prior take the closed form (module docstring) from 256-QAM
    (``_CLOSED_FORM_MIN_LEVELS_CPU``), where it is faster on the CPU.
    """
    _check_method(method)
    re, im = axes
    flat = np.asarray(symbols).reshape(-1)
    k = re.bits.size + im.bits.size
    llrs = np.empty((flat.size, k), dtype=np.float32)
    if _closed_form(axes, method, _CLOSED_FORM_MIN_LEVELS_CPU):
        _get_numba_llr("llr_separable_maxlog")(
            np.ascontiguousarray(flat.real, dtype=np.float32),
            np.ascontiguousarray(flat.imag, dtype=np.float32),
            *(
                t
                for a in axes
                for t in (
                    a.levels.astype(np.float32),
                    _packed_labels(a.labels),
                    *_opposite_levels(a.labels),
                    a.bits.astype(np.int64),
                )
            ),
            np.float32(1.0 / max(noise_var, 1e-20)),
            llrs,
        )
        return llrs.reshape((*symbols.shape[:-1], symbols.shape[-1] * k))
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


def _llr_separable_gpu(
    symbols: ArrayType,
    axes: tuple[_Axis, _Axis],
    noise_var: float,
    method: str,
    owner: Constellation | None = None,
) -> ArrayType:
    """GPU LLRs of a separable constellation (as :func:`_llr_separable`)."""
    _check_method(method)
    re, im = axes
    k = re.bits.size + im.bits.size
    _require_gpu_bits(k)
    symbols, xp, _ = dispatch(symbols)
    x = symbols.reshape(-1).astype(xp.complex64, copy=False)
    inv_s2 = float(np.float32(1.0 / max(noise_var, 1e-20)))

    if _closed_form(axes, method):

        def build_closed() -> tuple[np.ndarray, ...]:
            return tuple(
                t
                for a in axes
                for t in (
                    a.levels.astype(np.float32),
                    _packed_labels(a.labels),
                    *_opposite_levels(a.labels),
                    a.bits.astype(np.int32),
                )
            )

        closed = _device_tables(owner, "llr_separable_maxlog", xp, build_closed)
        llrs = _cuda.get_kernel("llr_separable_maxlog")(
            x, closed[:5], closed[5:], inv_s2=inv_s2
        )
        return llrs.reshape((*symbols.shape[:-1], symbols.shape[-1] * k))

    def build() -> tuple[np.ndarray, ...]:
        return tuple(
            t
            for a in axes
            for t in (
                a.levels.astype(np.float32),
                _log_prior(a.pmf, a.levels.size),
                _packed_labels(a.labels),
                a.bits.astype(np.int32),
            )
        )

    tables = _device_tables(owner, "llr_separable", xp, build)
    llrs = _cuda.get_kernel("llr_separable")(
        x, tables[:4], tables[4:], inv_s2=inv_s2, exact=method == "exact"
    )
    return llrs.reshape((*symbols.shape[:-1], symbols.shape[-1] * k))


def _packed_labels(bit_labels: np.ndarray) -> np.ndarray:
    """MSB-first bit rows to int32 labels, the CUDA kernels' layout."""
    return _pack(bit_labels).astype(np.int32)
