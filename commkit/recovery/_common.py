"""Shared helpers for the recovery package (block-phase estimation, logging)."""

from __future__ import annotations

import numpy as np

from ..backend import to_device


def _vv_block_phase(
    symbols2d, xp, M: int, modulation: str, block_size: int, joint_channels: bool
):
    """Viterbi-Viterbi (M-th power) block-phase estimator.

    Reshapes into blocks, unit-circle-normalises QAM (so the M-th-power bias
    correction is exact by the 4-fold rotational symmetry of the
    constellation), sums the M-th power per block, 4-fold-unwraps the
    resulting block-phase trajectory, applies the QAM ``pi/M`` bias
    correction, and - for MIMO in non-joint mode - aligns every channel's
    M-fold branch to channel 0's.

    Shared core of ``recover_carrier_phase_viterbi_viterbi`` and
    ``recover_carrier_phase_tikhonov`` (which extends this with a Kalman
    smoother before cycle-slip correction and interpolation): both consume
    this **raw** (unwrapped, bias-corrected, MIMO-aligned) block-phase
    trajectory before any smoothing / cycle-slip-correction / interpolation,
    which stays in the caller.

    Parameters
    ----------
    symbols2d : (C, N) complex array, any backend
        1-sps symbols, already 2-D (``as_2d``'d) and ``dispatch``'d.
    xp : module
        ``symbols2d``'s array module (NumPy/CuPy).
    M : int
        Modulation M-th power (see ``frequency._modulation_power_m``).
    modulation : str
        Modulation scheme string (matched case-insensitively for the "qam"
        unit-circle-normalisation / bias-correction branch).
    block_size : int
        Symbols per block.  The caller has already validated
        ``N // block_size > 0``.
    joint_channels : bool
        If ``True`` and ``C > 1``, sum the M-th-power block phasors across
        channels before angle/unwrap, producing a single joint trajectory
        broadcast (as independent copies) to all ``C`` rows of the return
        value - matching ``phi_blocks_out[ch] = phi_u_joint`` in the
        non-shared per-caller loop this replaces.

    Returns
    -------
    phi_u : (C, N_blocks) float64 array, same backend as ``symbols2d``
        Raw block-phase trajectory - identical across rows in joint mode.
    block_centers : (N_blocks,) float64 array
        Block centre positions in symbols, for interpolation.
    all_positions : (N,) float64 array
        Per-symbol positions, for interpolation.
    """
    C, N = symbols2d.shape
    N_trunc = (N // block_size) * block_size
    N_blocks = N_trunc // block_size

    # Reshape for block processing: (C, N_blocks, block_size).
    # Promote to complex128 for the M-th power - identical to estimate_frequency_offset_mth_power.
    # On GPU, complex64^4 loses precision near the ±π/M unwrap boundary, causing
    # spurious branch flips for high-order QAM with small block sizes.
    blocks = symbols2d[:, :N_trunc].reshape(C, N_blocks, block_size)
    blocks_c = blocks.astype(
        xp.complex128 if blocks.dtype == xp.complex64 else blocks.dtype
    )

    # For QAM, project to unit circle before the M-th power (normalized VV).
    # This removes outer-ring amplitude dominance and makes the π/M QAM bias
    # correction exact (by the 4-fold rotational symmetry of the constellation).
    # PSK is already constant-modulus; normalization is a no-op.
    is_qam = "qam" in modulation.lower()
    if is_qam:
        mag = xp.abs(blocks_c)
        blocks_c = blocks_c / xp.maximum(mag, 1e-15 * xp.max(mag))

    S_b = xp.sum(blocks_c**M, axis=-1)  # (C, N_blocks)

    # Block centre positions for interpolation (uniform spacing = block_size)
    block_centers = xp.arange(N_blocks, dtype=xp.float64) * block_size + block_size / 2
    all_positions = xp.arange(N, dtype=xp.float64)

    if joint_channels and C > 1:
        # Sum M-th-power phasors across channels -> single block-phase trajectory
        S_b_joint = xp.sum(S_b, axis=0)  # (N_blocks,)
        phi_raw_joint = xp.angle(S_b_joint) / M
        phi_u_joint = xp.unwrap((phi_raw_joint * M).astype(xp.float64)) / M
        if is_qam:
            phi_u_joint = phi_u_joint - (np.pi / M)
        phi_u = xp.broadcast_to(phi_u_joint, (C, N_blocks)).copy()
    else:
        # Raw block phase in [-π/M, π/M)
        phi_raw = xp.angle(S_b) / M  # (C, N_blocks)

        # M-fold unwrap: scale into 2π domain, unwrap, re-scale back.
        # Cast to float64 before unwrap - cp.unwrap preserves input dtype so float32
        # would lose precision during the discontinuity test (diff vs 2π threshold).
        phi_u = (
            xp.unwrap((phi_raw * M).astype(xp.float64), axis=-1) / M
        )  # (C, N_blocks)

        # QAM bias correction.
        if is_qam:
            phi_u = phi_u - (np.pi / M)

        # MIMO M-fold alignment: align every channel to channel 0's branch.
        # Skipped in joint mode (all channels share the same trajectory).
        if C > 1:
            # All per-channel means on device, one batched D2H, vectorized shift
            # (instead of one float() sync + one rounding per channel).
            diffs_np = to_device(xp.mean(phi_u[1:] - phi_u[0:1], axis=-1), "cpu")
            k_np = np.round(diffs_np * M / (2 * np.pi))
            phi_u[1:] = phi_u[1:] - xp.asarray(k_np)[:, None] * (2 * np.pi / M)

    return phi_u, block_centers, all_positions
