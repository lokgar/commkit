"""Blind Phase Search (BPS) carrier phase recovery."""

import logging

import numpy as np

from ..backend import ArrayType, dispatch, to_device
from ..helpers import as_2d, restore_1d
from ..logger import logger
from .corrections import correct_cycle_slips


def recover_carrier_phase_bps(
    symbols: ArrayType,
    modulation: str,
    order: int,
    num_test_phases: int = 64,
    block_size: int = 32,
    joint_channels: bool = False,
    cycle_slip_correction: bool = False,
    cycle_slip_history: int = 100,
    cycle_slip_threshold: float = np.pi / 4,
    pmf: np.ndarray | None = None,
    debug_plot: bool = False,
) -> ArrayType:
    """
    Carrier phase recovery via Blind Phase Search (BPS).

    Tests ``num_test_phases`` candidate rotation angles over ``[0, π/2)``
    (exploiting 4-fold QAM symmetry), selects the candidate that minimises
    the block-averaged sum of minimum squared distances to the reference
    constellation, and interpolates to per-symbol resolution.

    Parameters
    ----------
    symbols : array_like
        1-SPS complex symbols after matched filter. Shape: (N,) or (C, N).
    modulation : str
        Modulation scheme (case-insensitive). Used to fetch the reference
        constellation via
        ``gray_constellation``.
    order : int
        Modulation order.
    num_test_phases : int, default 64
        Number of candidate phase offsets B. Resolution is ``π/(2B)``
        rad per step. More candidates improve accuracy at higher compute cost.
    block_size : int, default 32
        Number of symbols per block for error-metric averaging.
        Very small values (< 4) make the 4-fold phase unwrap unreliable
        because noise on a single-symbol metric causes the best-candidate
        index to jump between non-adjacent phase bins between consecutive
        blocks, triggering false unwrap corrections.  Recommended
        minimum: ``block_size ≥ 4``.
    joint_channels : bool, default False
        For MIMO inputs (C > 1): if ``True``, sum the distance metrics
        across all channels before selecting the best phase candidate.
        The resulting single phase trajectory is broadcast to all C rows
        of the output (all channels identical before ambiguity resolution).
        Reduces phase estimation variance by ~√C for shared-LO systems.
        Has no effect for SISO (C = 1).
    cycle_slip_correction : bool, default False
        If ``True``, apply cycle-slip detection and correction
        (``correct_cycle_slips``) to the block-phase trajectory
        after 4-fold unwrap, before interpolation.
    cycle_slip_history : int, default 100
        ``history_length`` passed to ``correct_cycle_slips``.
        Number of past corrected blocks used for linear extrapolation.
    cycle_slip_threshold : float, default π/4
        ``threshold`` passed to ``correct_cycle_slips`` (radians).
    pmf : np.ndarray, optional
        Symbol PMF of shape ``(order,)`` for PS-QAM.  When provided, the
        reference constellation is scaled by ``1/sqrt(E_PS)`` (where
        ``E_PS = Σ P(s_m) |s_m|²`` on the normalised grid) so the
        nearest-neighbour distance metric matches the scale of the
        unit-avg-power input.  Without this, mid-shell PS points cross
        decision boundaries in the BPS metric and bias the phase estimate.
        No-op for uniform modulations.
    debug_plot : bool, default False
        If ``True``, opens a diagnostic figure showing the per-symbol phase
        trajectory alongside the block-phase estimates.

    Returns
    -------
    array_like
        Per-symbol phase estimate in radians. Shape matches ``symbols``.
        Same backend as input.

    Notes
    -----
    Tests B candidate rotations over [0, pi/2), selects the one minimising
    block-averaged minimum Euclidean distance, then 4-fold unwraps.  A global
    pi/2 ambiguity remains - resolve via a pilot or preamble reference.

    Memory: the distance tensor scales as N * B * M * 8 bytes; reduce
    ``num_test_phases`` or segment length for high-order constellations.
    """
    from ..helpers import normalize
    from ..mapping import constellation_power, gray_constellation

    symbols, xp, _ = dispatch(symbols)
    symbols, was_1d = as_2d(symbols, name="symbols")
    C, N = symbols.shape

    # Normalise each channel to unit average power so the metric is computed at
    # the same scale as the reference constellation (gray_constellation returns
    # unit-average-power points).  BPS is a phase estimator; it must be
    # amplitude-agnostic.
    symbols = normalize(symbols, mode="average_power", axis=-1)

    # Reference constellation on the same device
    const_np = gray_constellation(modulation, order)

    # PS-QAM: unit-avg-power input lives on the ``{s_m/sqrt(E_PS)}`` grid.
    # Rescale the comparison constellation to the same grid so the nearest-
    # neighbour distance metric is correct.  Skip on uniform PMF.
    if pmf is not None:
        e_ps = constellation_power(const_np, pmf)
        if e_ps < 1.0 - 1e-6:
            const_np = const_np / np.sqrt(e_ps)

    const_xp = xp.asarray(const_np)  # (M_const,)

    # Candidate test phases over [0, π/2)
    B = num_test_phases
    candidates = xp.arange(B, dtype=symbols.real.dtype) * (np.pi / 2.0 / B)  # (B,)

    N_trunc = (N // block_size) * block_size
    N_blocks = N_trunc // block_size

    if N_blocks == 0:
        raise ValueError(
            f"Signal length {N} is shorter than block_size={block_size}. "
            "Reduce block_size or use a longer symbol sequence."
        )

    # Very small block_size makes the 4-fold phase unwrap unreliable: with only
    # one or two symbols per block the noise on the distance-metric argmin causes
    # large candidate-index jumps between consecutive blocks, triggering false
    # 4-fold unwrap corrections.  Warn early so users diagnose this easily.
    if block_size < 4:
        logger.warning(
            "CPR (BPS): block_size=%s is very small. Averaging the distance "
            "metric over only %s symbol(s) per block makes the 4-fold "
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
    side = int(order**0.5)
    is_sq_qam = ("qam" in modulation.lower()) and (side * side == order)
    if is_sq_qam:
        # Sorted unique real levels of the constellation (shape: (side,))
        levels = xp.sort(xp.unique(const_xp.real))
        d_grid = float(levels[1] - levels[0])  # uniform grid spacing
        lev_min = float(levels[0])

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

    phi_full = xp.zeros((C, N), dtype=xp.float64)
    phi_blocks = xp.zeros((C, N_blocks), dtype=xp.float64)

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
                    r_near = levels[r_idx]  # (CHUNK, B)
                    i_near = levels[i_idx]  # (CHUNK, B)
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
        metric_joint = xp.sum(metrics_all, axis=0)  # (N_blocks, B)
        best_k_joint = xp.argmin(metric_joint, axis=-1)  # (N_blocks,)
        phi_b_joint = candidates[best_k_joint]  # (N_blocks,)
        phi_u_joint = xp.unwrap(phi_b_joint.astype(xp.float64) * 4, axis=-1) / 4
        if cycle_slip_correction:
            phi_u_joint_np = correct_cycle_slips(
                to_device(phi_u_joint, "cpu"),
                4,
                cycle_slip_history,
                cycle_slip_threshold,
            )
            phi_u_joint = xp.asarray(phi_u_joint_np)
        for ch in range(C):
            phi_full[ch] = (
                phi_u_joint[idx_left] * (1.0 - t_interp)
                + phi_u_joint[idx_right] * t_interp
            )
            phi_blocks[ch] = phi_u_joint
    else:
        for ch in range(C):
            metric = metrics_all[ch]  # (N_blocks, B)
            best_k = xp.argmin(metric, axis=-1)  # (N_blocks,)
            phi_b = candidates[best_k]  # (N_blocks,)
            phi_u = xp.unwrap(phi_b.astype(xp.float64) * 4, axis=-1) / 4
            if cycle_slip_correction:
                phi_u_np = correct_cycle_slips(
                    to_device(phi_u, "cpu"), 4, cycle_slip_history, cycle_slip_threshold
                )
                phi_u = xp.asarray(phi_u_np)
            phi_full[ch] = (
                phi_u[idx_left] * (1.0 - t_interp) + phi_u[idx_right] * t_interp
            )
            phi_blocks[ch] = phi_u

    # Host copy of the full phase trajectory is needed only for the INFO summary
    # below and the optional debug plot - skip the transfer + reductions when
    # neither is active (the device phi_full drives the actual correction).
    _want_log = logger.isEnabledFor(logging.INFO)
    if _want_log or debug_plot:
        phi_full_np = to_device(phi_full, "cpu")
    if _want_log:
        phi_mean_deg = float(np.mean(phi_full_np)) * 180.0 / np.pi
        phi_std_deg = float(np.std(phi_full_np)) * 180.0 / np.pi
        mode_str = "joint" if (joint_channels and C > 1) else "independent"
        logger.info(
            "CPR (BPS, B=%s, %s): phase mean=%.2f°, std=%.2f° [%s blocks "
            "x %s symbols, C=%s, cycle_slip_correction=%s]",
            B,
            mode_str,
            phi_mean_deg,
            phi_std_deg,
            N_blocks,
            block_size,
            C,
            cycle_slip_correction,
        )

    if debug_plot:
        from .. import plotting as _plotting

        _plotting.plot_carrier_phase_trajectory(
            phi_full=phi_full_np,
            show=True,
            title="CPR - Blind Phase Search",
        )

    return restore_1d(was_1d, phi_full)
