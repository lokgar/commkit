"""Blind phase search (BPS) carrier-phase recovery: :class:`BPS`."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..backend import ArrayType, dispatch
from ..logger import logger
from ..mapping._lattice import slicer_params
from ..math import normalize
from ._common import _Context, _interp_block_phases, _mfold_unwrap, _num_blocks, _numba
from .corrections import CycleSlip, _log_phase_summary, _repair_slips
from .estimates import BlockPhaseEstimate

__all__ = ["BPS"]


@dataclass(frozen=True)
class BPS:
    """Blind phase search (BPS).

    Tests ``test_phases`` candidate rotations over one ambiguity interval
    ``[0, 2π/M)``, ``M`` the constellation's rotational symmetry (an interval
    of ``π/2`` for QAM); in each block it selects the candidate with the
    smallest summed squared distance to the nearest constellation points,
    then unwraps the block phases M-fold and interpolates them linearly to
    one value per symbol.

    Parameters
    ----------
    test_phases : int, default 64
        Number of candidate phases B.  Resolution is ``2π/(M·B)`` rad.
    block_size : int, default 32
        Symbols per block for metric averaging.  Values below 4 make the
        M-fold unwrap unreliable (a noisy argmin jumps between non-adjacent
        candidates).
    joint_channels : bool, default False
        MIMO: sum the distance metrics across channels before the argmin and
        give every channel the one trajectory (a shared LO; about ``C``
        times lower variance).
    cycle_slip : CycleSlip, optional
        Repair cycle slips in the block phases before interpolation.

    Notes
    -----
    A global ``2π/M`` ambiguity remains: the trajectory starts on the branch
    nearest zero, ``[-π/M, π/M)``, as for :class:`ViterbiViterbi` and
    :class:`PLL`, so a carrier phase near ``±π/M`` can come out one ``2π/M``
    step off.  No blind estimator can tell those apart; training symbols,
    pilots or :func:`resolve_phase_ambiguity` against a reference resolve it.

    Memory: the CPU kernels and the GPU kernel stream the nearest-point
    search (square QAM by an O(1) per-axis slicer).  The GPU beyond its
    kernel's limits (complex128, more than 128 test phases, a table of more
    than 1024 points) builds chunks of about 2^22 distance elements.
    """

    test_phases: int = 64
    block_size: int = 32
    joint_channels: bool = False
    cycle_slip: CycleSlip | None = None

    def __post_init__(self) -> None:
        if self.test_phases < 1:
            raise ValueError(f"BPS: test_phases must be >= 1, got {self.test_phases}.")
        if self.block_size < 1:
            raise ValueError(f"BPS: block_size must be >= 1, got {self.block_size}.")


def _bps(symbols: ArrayType, method: BPS, ctx: _Context) -> BlockPhaseEstimate:
    """BPS phase of ``(C, N)`` symbols."""

    constellation = ctx.need_constellation(method)
    num_test_phases = method.test_phases
    block_size = method.block_size
    joint_channels = method.joint_channels

    symbols, xp, _ = dispatch(symbols)
    C, N = symbols.shape

    # Normalize each channel to unit average power so the metric is computed at
    # the same scale as the (unit-power) constellation.  BPS is a phase estimator; it must be
    # amplitude-agnostic.
    symbols = normalize(symbols, mode="average_power", axis=-1)

    # Candidate test phases over one ambiguity interval [0, 2π/M), in float64.
    M = int(constellation.rotational_symmetry)
    B = num_test_phases
    candidates = xp.arange(B, dtype=xp.float64) * (2.0 * np.pi / M / B)

    N_blocks = _num_blocks(N, block_size)

    # Very small block_size makes the M-fold phase unwrap unreliable: with only
    # one or two symbols per block the noise on the distance-metric argmin causes
    # large candidate-index jumps between consecutive blocks, triggering false
    # M-fold unwrap corrections.  Traced at DEBUG: the size is the caller's
    # choice, and slips it causes show in the result.
    if block_size < 4:
        logger.debug(
            "CPR (BPS): block_size=%s is very small. Averaging the distance "
            "metric over only %s symbol(s) per block makes the M-fold "
            "phase unwrap unreliable. Recommended minimum: block_size ≥ 4.",
            block_size,
            block_size,
        )

    # Phasors of the B candidates, computed once in float64 and cast to the
    # symbols' precision.
    dtype_c = xp.complex64 if symbols.dtype == xp.complex64 else xp.complex128
    phasors = xp.exp(-1j * candidates.astype(xp.float64)).astype(dtype_c)  # (B,)
    metrics_all = _bps_metrics(symbols, phasors, constellation, block_size, N_blocks)

    # Phase estimation: joint (sum metrics across channels) or independent per channel.
    if joint_channels and C > 1:
        metrics_all = xp.sum(metrics_all, axis=0, keepdims=True)  # (1, N_blocks, B)
    best_k = xp.argmin(metrics_all, axis=-1)  # (R, N_blocks)
    phi_b = candidates[best_k].astype(xp.float64)
    # Principal branch [-π/M, π/M), as Viterbi-Viterbi and the PLL: the M-fold
    # unwrap ignores 2π/M steps, so this only puts the first block on the
    # branch nearest zero instead of the edge of the candidate range.
    phi_b = xp.where(phi_b >= np.pi / M, phi_b - 2.0 * np.pi / M, phi_b)
    phi_u = _mfold_unwrap(phi_b, M, xp)
    phi_u = _repair_slips(phi_u, method.cycle_slip, M, block_size)
    phi_full = _interp_block_phases(phi_u, N, block_size, xp)
    if phi_u.shape[0] != C:  # joint: one trajectory for every channel
        phi_full = xp.broadcast_to(phi_full, (C, N)).copy()
        phi_u = xp.broadcast_to(phi_u, (C, N_blocks)).copy()

    mode_str = "joint" if (joint_channels and C > 1) else "independent"
    _log_phase_summary(
        phi_full,
        "CPR (BPS, B=%s, %s)",
        (B, mode_str),
        "[%s blocks x %s symbols, C=%s, cycle_slip=%s]",
        (N_blocks, block_size, C, method.cycle_slip is not None),
    )

    return BlockPhaseEstimate(
        value=phi_full,
        block_centers=np.arange(N_blocks, dtype=np.float64) * block_size
        + block_size / 2,
        block_phase=phi_u,
    )


def _bps_metrics(
    symbols: ArrayType,
    phasors: ArrayType,
    constellation: Any,
    block_size: int,
    n_blocks: int,
) -> ArrayType:
    """Block metric of each candidate phase, ``(C, n_blocks, B)``.

    The sum over a block of each symbol's squared distance to the nearest
    point of the unit-power constellation after rotation by a candidate, in
    the symbols' real precision.  The nearest point comes from the O(1)
    per-axis slicer for square QAM and from a scan of the points otherwise.
    The GPU runs the fused kernel within its limits (complex64, at most 128
    phases, a table of at most 1024 points) and the chunked array path
    beyond them; the CPU runs the Numba kernels.
    """
    xp = dispatch(symbols)[1]
    grid = slicer_params(constellation)
    B = phasors.size
    if xp is np:
        return _bps_metrics_numba(
            symbols, phasors, constellation, grid, block_size, n_blocks
        )
    if symbols.dtype == xp.complex64 and B <= 128:
        if grid[0] > 0 or constellation.order <= 1024:
            return _bps_metrics_kernel(
                symbols, phasors, constellation, grid, block_size, n_blocks
            )
    return _bps_metrics_array(
        symbols, phasors, constellation, grid, block_size, n_blocks
    )


def _bps_metrics_kernel(
    symbols: ArrayType,
    phasors: ArrayType,
    constellation: Any,
    grid: tuple[int, Any, Any],
    block_size: int,
    n_blocks: int,
) -> ArrayType:
    """:func:`_bps_metrics` by the fused CUDA kernel (CuPy, complex64).

    One call per chunk covers all channels and writes only the minima, so
    the chunk can be far larger than the array path's; its ``(B, C, n)``
    output is block-summed into the ``(C, n_blocks, B)`` layout.
    """
    from .. import _cuda

    xp = dispatch(symbols)[1]
    side, lev_min, d_grid = grid
    C, B = symbols.shape[0], phasors.size
    kern = _cuda.get_kernel("bps_min_d2", mode="grid" if side > 0 else "table")
    points = None if side > 0 else xp.asarray(constellation.points.astype(np.complex64))
    metrics = xp.empty((C, n_blocks, B), dtype=xp.float32)
    chunk = ((131072 + block_size - 1) // block_size) * block_size
    n_trunc = n_blocks * block_size
    for n0 in range(0, n_trunc, chunk):
        n1 = min(n0 + chunk, n_trunc)
        if side > 0:
            md = kern(
                symbols[:, n0:n1],
                phasors,
                lev_min=float(lev_min),
                d_grid=float(d_grid),
                side=side,
            )
        else:
            md = kern(symbols[:, n0:n1], phasors, constellation=points)
        b0, n_b = n0 // block_size, (n1 - n0) // block_size
        metrics[:, b0 : b0 + n_b] = (
            md.reshape(B, C, n_b, block_size).sum(axis=3).transpose(1, 2, 0)
        )
    return metrics


def _bps_metrics_numba(
    symbols: np.ndarray,
    phasors: np.ndarray,
    constellation: Any,
    grid: tuple[int, Any, Any],
    block_size: int,
    n_blocks: int,
) -> np.ndarray:
    """:func:`_bps_metrics` by the Numba kernels, over channels and blocks."""
    side, lev_min, d_grid = grid
    C, B = symbols.shape[0], phasors.size
    n_trunc = n_blocks * block_size
    real = np.float32 if symbols.dtype == np.complex64 else np.float64
    metrics = np.empty((C, n_blocks, B), dtype=np.float64)
    data = (
        np.ascontiguousarray(symbols[:, :n_trunc].real, dtype=real),
        np.ascontiguousarray(symbols[:, :n_trunc].imag, dtype=real),
        np.ascontiguousarray(phasors.real, dtype=real),
        np.ascontiguousarray(phasors.imag, dtype=real),
    )
    if side > 0:
        _numba().bps_grid(*data, real(lev_min), real(d_grid), side, block_size, metrics)
    else:
        points = np.asarray(constellation.points, dtype=np.complex128)
        _numba().bps_table(
            *data,
            np.ascontiguousarray(points.real, dtype=real),
            np.ascontiguousarray(points.imag, dtype=real),
            block_size,
            metrics,
        )
    return metrics.astype(real)


def _bps_metrics_array(
    symbols: ArrayType,
    phasors: ArrayType,
    constellation: Any,
    grid: tuple[int, Any, Any],
    block_size: int,
    n_blocks: int,
) -> ArrayType:
    """:func:`_bps_metrics` with array operations: the GPU beyond its kernel.

    Chunks of whole blocks keep the ``(chunk, B)`` arrays, or
    ``(chunk, B, M)`` for a table, within about 2^22 elements.
    """
    xp = dispatch(symbols)[1]
    side, lev_min_f32, d_grid_f32 = grid
    lev_min, d_grid = float(lev_min_f32), float(d_grid_f32)
    C, B = symbols.shape[0], phasors.size
    real = xp.float32 if symbols.dtype == xp.complex64 else xp.float64
    points = xp.asarray(constellation.points.astype(phasors.dtype))
    per_symbol = B * (1 if side > 0 else points.size)
    chunk = max(block_size, (2**22 // per_symbol) // block_size * block_size)
    n_trunc = n_blocks * block_size
    metrics = xp.zeros((C, n_blocks, B), dtype=real)
    for ch in range(C):
        sym = symbols[ch, :n_trunc]
        for n0 in range(0, n_trunc, chunk):
            n1 = min(n0 + chunk, n_trunc)
            x_rot = sym[n0:n1, None] * phasors[None, :]  # (chunk, B)
            if side > 0:
                # Nearest grid level per component.
                r_idx = xp.clip(
                    xp.round((x_rot.real - lev_min) / d_grid).astype(xp.int64),
                    0,
                    side - 1,
                )
                i_idx = xp.clip(
                    xp.round((x_rot.imag - lev_min) / d_grid).astype(xp.int64),
                    0,
                    side - 1,
                )
                r_near = lev_min + r_idx.astype(real) * d_grid
                i_near = lev_min + i_idx.astype(real) * d_grid
                min_d = (
                    (x_rot.real - r_near) ** 2 + (x_rot.imag - i_near) ** 2
                ).astype(real)
            else:
                d_sq = xp.abs(x_rot[:, :, None] - points[None, None, :]) ** 2
                min_d = xp.min(d_sq, axis=-1).astype(real)
            b0, n_b = n0 // block_size, (n1 - n0) // block_size
            metrics[ch, b0 : b0 + n_b] = min_d.reshape(n_b, block_size, B).sum(axis=1)
    return metrics
