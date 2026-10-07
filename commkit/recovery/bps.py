"""Blind Phase Search (BPS) carrier phase recovery."""

from dataclasses import dataclass

import numpy as np

from ..backend import ArrayType, dispatch
from ..logger import logger
from ._common import _check_blocks, _Context, _Phase
from .corrections import CycleSlip, _log_phase_summary, _repair_slips

__all__ = ["BPS"]


@dataclass(frozen=True)
class BPS:
    """
    Blind Phase Search.

    Tests ``test_phases`` candidate rotations over one ambiguity interval
    ``[0, 2π/M)``, ``M`` the constellation's rotational symmetry (``π/2``
    for QAM), selects per block the candidate that minimises the summed
    minimum squared distance to the constellation, M-fold-unwraps the block
    phases and interpolates linearly to per-symbol resolution.

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
        give every channel the one trajectory (shared LO, ~√C lower
        variance).
    cycle_slip : CycleSlip, optional
        Repair cycle slips in the block phases before interpolation.

    Notes
    -----
    A global ``2π/M`` ambiguity remains; resolve it against a reference.

    Memory: the general (non-square) path builds a ``(1024, B, M)``
    distance tensor per chunk; square QAM uses an O(1) per-axis slicer.
    """

    test_phases: int = 64
    block_size: int = 32
    joint_channels: bool = False
    cycle_slip: CycleSlip | None = None

    def __post_init__(self) -> None:
        if self.test_phases < 1:
            raise ValueError(f"test_phases must be >= 1, got {self.test_phases}.")
        if self.block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {self.block_size}.")


_NUMBA_BPS: dict = {}


def _get_numba_bps_table():
    """Numba kernel for the BPS block metric with an arbitrary constellation.

    ``metrics[c, b, k] = sum_{n in block b} min_m |x[c, n] e^{-j theta_k} - s_m|^2``
    for ``(C, N_blocks, B)``, parallel over channels and blocks.  It replaces
    the NumPy ``(CHUNK, B, M)`` distance tensor of the non-square CPU path;
    square QAM has its own O(1) slicer.  Distances are float32 for complex64
    input (as in the NumPy path), the block sums float64.
    """
    if "table" not in _NUMBA_BPS:
        import numba

        @numba.njit(cache=True, fastmath=True, nogil=True, parallel=True)
        def bps_table(x_re, x_im, ph_re, ph_im, c_re, c_im, block_size, out):
            C = x_re.shape[0]
            n_blocks = out.shape[1]
            B = ph_re.shape[0]
            M = c_re.shape[0]
            for job in numba.prange(C * n_blocks):
                c = job // n_blocks
                b = job - c * n_blocks
                for k in range(B):
                    acc = 0.0
                    for n in range(b * block_size, (b + 1) * block_size):
                        xr = x_re[c, n] * ph_re[k] - x_im[c, n] * ph_im[k]
                        xi = x_re[c, n] * ph_im[k] + x_im[c, n] * ph_re[k]
                        best = (xr - c_re[0]) ** 2 + (xi - c_im[0]) ** 2
                        for m in range(1, M):
                            d = (xr - c_re[m]) ** 2 + (xi - c_im[m]) ** 2
                            if d < best:
                                best = d
                        acc += best
                    out[c, b, k] = acc

        _NUMBA_BPS["table"] = bps_table
    return _NUMBA_BPS["table"]


def _bps(symbols: ArrayType, method: BPS, ctx: _Context) -> _Phase:
    """BPS phase of ``(C, N)`` symbols."""
    from ..mapping.gray import _square_qam_slicer_params
    from ..math import normalize

    constellation = ctx.need_constellation(method)
    num_test_phases = method.test_phases
    block_size = method.block_size
    joint_channels = method.joint_channels

    symbols, xp, _ = dispatch(symbols)
    C, N = symbols.shape

    # Normalise each channel to unit average power so the metric is computed at
    # the same scale as the (unit-power) constellation.  BPS is a phase estimator; it must be
    # amplitude-agnostic.
    symbols = normalize(symbols, mode="average_power", axis=-1)

    # Reference constellation (unit power; a shaped constellation is already
    # on the {s_m/sqrt(E_PS)} grid of the unit-power input).
    const_np = np.asarray(constellation.points, dtype=np.complex128)
    const_xp = xp.asarray(const_np)  # (M_const,)

    # Candidate test phases over one ambiguity interval [0, 2π/M).
    M = int(constellation.rotational_symmetry)
    B = num_test_phases
    candidates = xp.arange(B, dtype=symbols.real.dtype) * (2.0 * np.pi / M / B)

    N_blocks = _check_blocks(N, block_size)
    N_trunc = N_blocks * block_size

    # Very small block_size makes the M-fold phase unwrap unreliable: with only
    # one or two symbols per block the noise on the distance-metric argmin causes
    # large candidate-index jumps between consecutive blocks, triggering false
    # M-fold unwrap corrections.  Warn early so users diagnose this easily.
    if block_size < 4:
        logger.warning(
            "CPR (BPS): block_size=%s is very small. Averaging the distance "
            "metric over only %s symbol(s) per block makes the M-fold "
            "phase unwrap unreliable. Recommended minimum: block_size ≥ 4.",
            block_size,
            block_size,
        )

    # block_centers[b] = b * block_size + block_size/2  (consistent with VV)
    block_centers = xp.arange(N_blocks, dtype=xp.float64) * block_size + block_size / 2

    all_positions = xp.arange(N, dtype=xp.float64)

    # Pre-compute interpolation indices and weights (identical for every channel).
    # block b is "to the left" of position n when its centre b*bs + bs/2 <= n
    #   => b <= (n - bs/2) / bs  => idx_left = floor((n - bs/2) / bs)
    idx_left = xp.clip(
        xp.floor((all_positions - block_size / 2) / block_size).astype(xp.int64),
        0,
        N_blocks - 2,
    )  # (N,)
    idx_right = idx_left + 1  # (N,)
    t_interp = xp.clip(
        (all_positions - block_centers[idx_left]) / block_size, 0.0, 1.0
    )  # (N,)

    # Pre-compute phasors for all B candidates once (avoid redundant exp per channel)
    dtype_c = xp.complex64 if symbols.dtype == xp.complex64 else xp.complex128
    phasors = xp.exp(-1j * candidates.astype(xp.float64)).astype(dtype_c)  # (B,)

    # For square QAM (order a perfect square): the nearest constellation point
    # can be found in O(1) per symbol via per-component rounding, eliminating
    # the (CHUNK, B, M_const) distance tensor entirely.
    side, lev_min_f32, d_grid_f32 = _square_qam_slicer_params(const_np)
    is_sq_qam = side > 0
    lev_min = float(lev_min_f32)
    d_grid = float(d_grid_f32)

    float_dtype = xp.float32 if symbols.dtype == xp.complex64 else xp.float64

    # Fused CUDA kernel (CuPy + complex64 only): computes the per-symbol
    # min-distance metric for all B candidate phases and all C channels in a
    # single pass, avoiding the materialized (CHUNK, B[, M]) intermediates of
    # the xp path.  None => fall back to the xp implementation below.
    _kern = None
    if xp is not np and symbols.dtype == xp.complex64 and B <= 128:
        if is_sq_qam or const_xp.size <= 1024:
            from .. import _cuda

            _kern = _cuda.get_kernel(
                "bps_min_d2", mode="grid" if is_sq_qam else "table"
            )

    # Chunk size for N axis: bounds peak memory of the distance tensor.
    # Always a multiple of block_size so each chunk covers a whole number of
    # blocks exactly.  Rounded up to the nearest multiple ≥ 1024.
    CHUNK_N = max(block_size, ((1024 + block_size - 1) // block_size) * block_size)

    # Accumulate per-channel distance metrics (N_blocks, B) for all channels.
    metrics_all = xp.zeros((C, N_blocks, B), dtype=float_dtype)

    if _kern is not None:
        # One kernel call per chunk covering all C channels; output (B, C, n)
        # is block-summed and transposed into metrics_all's (C, n_b, B)
        # layout.  The kernel writes only the minima, so the chunk can be far
        # larger than the tensor-bounded CHUNK_N of the xp path.
        chunk_gpu = ((131072 + block_size - 1) // block_size) * block_size
        const_c64 = None if is_sq_qam else const_xp.astype(xp.complex64)
        for n0 in range(0, N_trunc, chunk_gpu):
            n1 = min(n0 + chunk_gpu, N_trunc)
            if is_sq_qam:
                md = _kern(
                    symbols[:, n0:n1],
                    phasors,
                    lev_min=lev_min,
                    d_grid=d_grid,
                    side=side,
                )
            else:
                md = _kern(symbols[:, n0:n1], phasors, constellation=const_c64)
            b0 = n0 // block_size
            n_b = (n1 - n0) // block_size
            metrics_all[:, b0 : b0 + n_b] = (
                md.reshape(B, C, n_b, block_size).sum(axis=3).transpose(1, 2, 0)
            )

    elif xp is np and not is_sq_qam:
        # Non-square constellation on the CPU: the Numba kernel streams the
        # nearest-point search instead of materializing (CHUNK, B, M).
        real = np.float32 if symbols.dtype == np.complex64 else np.float64
        metrics_f64 = np.empty((C, N_blocks, B), dtype=np.float64)
        _get_numba_bps_table()(
            np.ascontiguousarray(symbols[:, :N_trunc].real, dtype=real),
            np.ascontiguousarray(symbols[:, :N_trunc].imag, dtype=real),
            np.ascontiguousarray(phasors.real, dtype=real),
            np.ascontiguousarray(phasors.imag, dtype=real),
            np.ascontiguousarray(const_np.real, dtype=real),
            np.ascontiguousarray(const_np.imag, dtype=real),
            block_size,
            metrics_f64,
        )
        metrics_all = metrics_f64.astype(float_dtype)

    else:
        for ch in range(C):
            sym = symbols[ch, :N_trunc]  # (N_trunc,)

            for n0 in range(0, N_trunc, CHUNK_N):
                n1 = min(n0 + CHUNK_N, N_trunc)
                x_rot = sym[n0:n1, None] * phasors[None, :]  # (CHUNK, B)

                if is_sq_qam:
                    # O(1) nearest-point: round each component to the nearest grid level
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
                    r_near = lev_min + r_idx.astype(float_dtype) * d_grid  # (CHUNK, B)
                    i_near = lev_min + i_idx.astype(float_dtype) * d_grid  # (CHUNK, B)
                    chunk_min_d = (
                        (x_rot.real - r_near) ** 2 + (x_rot.imag - i_near) ** 2
                    ).astype(float_dtype)
                else:
                    # General: (CHUNK, B, M_const) - bounded by CHUNK_N
                    d_sq = xp.abs(x_rot[:, :, None] - const_xp[None, None, :]) ** 2
                    chunk_min_d = xp.min(d_sq, axis=-1).astype(float_dtype)

                b0 = n0 // block_size
                n_b = (n1 - n0) // block_size
                metrics_all[ch, b0 : b0 + n_b] = chunk_min_d.reshape(
                    n_b, block_size, B
                ).sum(axis=1)

    # Phase estimation: joint (sum metrics across channels) or independent per channel.
    if joint_channels and C > 1:
        metrics_all = xp.sum(metrics_all, axis=0, keepdims=True)  # (1, N_blocks, B)
    best_k = xp.argmin(metrics_all, axis=-1)  # (R, N_blocks)
    phi_b = candidates[best_k]
    phi_u = xp.unwrap(phi_b.astype(xp.float64) * M, axis=-1) / M
    phi_u = _repair_slips(phi_u, xp, method.cycle_slip, M)
    # Per row: a 1-D gather is about twice as fast as the 2-D fancy index.
    phi_full = xp.empty((phi_u.shape[0], N), dtype=xp.float64)
    for r, row in enumerate(phi_u):
        phi_full[r] = row[idx_left] * (1.0 - t_interp) + row[idx_right] * t_interp
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

    return _Phase(
        phase=phi_full,
        block_centers=np.arange(N_blocks, dtype=np.float64) * block_size
        + block_size / 2,
        block_phase=phi_u,
    )
