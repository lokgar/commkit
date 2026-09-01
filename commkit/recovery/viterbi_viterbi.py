"""Viterbi-Viterbi (V&V) carrier phase recovery."""

import logging

import numpy as np

from ..backend import ArrayType, dispatch, to_device
from ..frequency import _modulation_power_m
from ..helpers import as_2d, restore_1d
from ..logger import logger
from .corrections import correct_cycle_slips


def recover_carrier_phase_viterbi_viterbi(
    symbols: ArrayType,
    modulation: str,
    order: int,
    block_size: int = 32,
    joint_channels: bool = False,
    cycle_slip_correction: bool = False,
    cycle_slip_history: int = 100,
    cycle_slip_threshold: float = np.pi / 4,
    debug_plot: bool = False,
) -> ArrayType:
    """
    Carrier phase recovery via the Viterbi-Viterbi (M-th power) algorithm.

    Block-based blind phase estimation for PSK and QAM symbols. Raises each
    block of symbols to the M-th power to remove modulation, extracts the
    block phase, resolves the M-fold ambiguity by unwrapping, then
    interpolates to per-symbol resolution.

    Parameters
    ----------
    symbols : array_like
        1-SPS complex symbols after matched filter. Shape: (N,) or (C, N).
    modulation : str
        Modulation scheme (case-insensitive): 'psk', 'qam', etc.
    order : int
        Modulation order.
    block_size : int, default 32
        Number of symbols per estimation block. Larger blocks reduce
        variance but reduce tracking bandwidth for fast phase noise.
        Typical range: 16-128 for QAM; as low as 1 for PSK (data cancels
        exactly in the M-th power for M-PSK constellations).
    joint_channels : bool, default False
        For MIMO inputs (C > 1): if ``True``, sum the M-th-power block
        phasors ``S_b`` across all channels before phase extraction.
        The resulting single trajectory is broadcast to all C output rows.
        Reduces variance by ~√C for shared-LO systems.  SISO-safe.
    cycle_slip_correction : bool, default False
        If ``True``, apply cycle-slip detection and correction
        (``correct_cycle_slips``) after M-fold unwrap, before
        interpolation.
    cycle_slip_history : int, default 100
        ``history_length`` passed to ``correct_cycle_slips``.
    cycle_slip_threshold : float, default π/4
        ``threshold`` passed to ``correct_cycle_slips`` (radians).
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
    Each block: S_b = sum s[n]^M, phi_hat_b = angle(S_b) / M. Block phases
    are M-fold unwrapped; a global 2*pi/M ambiguity always remains.

    For QAM with order > 4, block averaging suppresses M-th-power data residuals;
    minimum reliable block_size scales as ~4*ceil(sqrt(order)).  For high phase
    noise prefer ``recover_carrier_phase_bps`` (no unwrap required).
    """
    symbols, xp, _ = dispatch(symbols)
    symbols, was_1d = as_2d(symbols, name="symbols")
    C, N = symbols.shape

    M = _modulation_power_m(modulation, order)

    N_trunc = (N // block_size) * block_size
    N_blocks = N_trunc // block_size

    if N_blocks == 0:
        raise ValueError(
            f"Signal length {N} is shorter than block_size={block_size}. "
            "Reduce block_size or use a longer symbol sequence."
        )

    # For QAM with order > 4 the M-th power of individual symbols does NOT cancel
    # the data modulation (unlike PSK, where every M-PSK point gives (c/|c|)^M = 1).
    # Sufficient block averaging is required so that the block-phase variance stays
    # below the π/M unwrap threshold.  The practical minimum scales as 4·ceil(√order).
    if "qam" in modulation.lower() and order > 4:
        _min_bs = max(8, 4 * int(np.ceil(order**0.5)))
        if block_size < _min_bs:
            logger.warning(
                "CPR (VV): block_size=%s is too small for %s-QAM. Individual QAM symbols' M-th powers do not cancel the data modulation; insufficient averaging causes block-phase variance that exceeds the π/M unwrap threshold, producing persistent 2π/M phase slips. Recommended minimum for %s-QAM: block_size ≥ %s.",
                block_size,
                order,
                order,
                _min_bs,
            )

    # Reshape for block processing: (C, N_blocks, block_size).
    # Promote to complex128 for the M-th power - identical to estimate_frequency_offset_mth_power.
    # On GPU, complex64^4 loses precision near the ±π/M unwrap boundary, causing
    # spurious branch flips for high-order QAM with small block sizes.
    blocks = symbols[:, :N_trunc].reshape(C, N_blocks, block_size)
    blocks_c = blocks.astype(
        xp.complex128 if blocks.dtype == xp.complex64 else blocks.dtype
    )

    # For QAM, project to unit circle before the M-th power (normalized VV).
    # This removes outer-ring amplitude dominance and makes the π/M QAM bias
    # correction exact (by the 4-fold rotational symmetry of the constellation).
    # PSK is already constant-modulus; normalization is a no-op.
    if "qam" in modulation.lower():
        mag = xp.abs(blocks_c)
        blocks_c = blocks_c / xp.maximum(mag, 1e-15 * xp.max(mag))

    S_b = xp.sum(blocks_c**M, axis=-1)  # (C, N_blocks)

    # Block centre positions for interpolation (uniform spacing = block_size)
    block_centers = xp.arange(N_blocks, dtype=xp.float64) * block_size + block_size / 2
    all_positions = xp.arange(N, dtype=xp.float64)

    phi_full = xp.zeros((C, N), dtype=xp.float64)
    phi_blocks_out = xp.zeros((C, N_blocks), dtype=xp.float64)

    if joint_channels and C > 1:
        # Sum M-th-power phasors across channels -> single block-phase trajectory
        S_b_joint = xp.sum(S_b, axis=0)  # (N_blocks,)
        phi_raw_joint = xp.angle(S_b_joint) / M
        phi_u_joint = xp.unwrap((phi_raw_joint * M).astype(xp.float64)) / M
        if "qam" in modulation.lower():
            phi_u_joint = phi_u_joint - (np.pi / M)
        if cycle_slip_correction:
            phi_u_joint_np = correct_cycle_slips(
                to_device(phi_u_joint, "cpu"),
                4,
                cycle_slip_history,
                cycle_slip_threshold,
            )
            phi_u_joint = xp.asarray(phi_u_joint_np)
        phi_interp = xp.interp(all_positions, block_centers, phi_u_joint)
        for ch in range(C):
            phi_full[ch] = phi_interp
            phi_blocks_out[ch] = phi_u_joint
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
        if "qam" in modulation.lower():
            phi_u = phi_u - (np.pi / M)

        # MIMO M-fold alignment: align every channel to channel 0's branch.
        # Skipped in joint mode (all channels share the same trajectory).
        if C > 1:
            # All per-channel means on device, one batched D2H, vectorized shift
            # (instead of one float() sync + one rounding per channel).
            diffs_np = to_device(xp.mean(phi_u[1:] - phi_u[0:1], axis=-1), "cpu")
            k_np = np.round(diffs_np * M / (2 * np.pi))
            phi_u[1:] = phi_u[1:] - xp.asarray(k_np)[:, None] * (2 * np.pi / M)

        # xp.interp is 1D-only; loop over C channels.
        for ch in range(C):
            phi_u_ch = phi_u[ch]
            if cycle_slip_correction:
                phi_u_ch_np = correct_cycle_slips(
                    to_device(phi_u_ch, "cpu"),
                    4,
                    cycle_slip_history,
                    cycle_slip_threshold,
                )
                phi_u_ch = xp.asarray(phi_u_ch_np)
            phi_full[ch] = xp.interp(all_positions, block_centers, phi_u_ch)
            phi_blocks_out[ch] = phi_u_ch

    # Host copy of the trajectory is needed only for the INFO summary and the
    # optional debug plot; skip the transfer + reductions otherwise (the device
    # phi_full drives the actual correction and is what gets returned).
    _want_log = logger.isEnabledFor(logging.INFO)
    if _want_log or debug_plot:
        phi_full_np = to_device(phi_full, "cpu")
    if _want_log:
        phi_mean_deg = float(np.mean(phi_full_np)) * 180.0 / np.pi
        phi_std_deg = float(np.std(phi_full_np)) * 180.0 / np.pi
        mode_str = "joint" if (joint_channels and C > 1) else "independent"
        logger.info(
            "CPR (Viterbi-Viterbi, M=%s, %s): phase mean=%.2f°, std=%.2f° [%s blocks x %s symbols, C=%s, cycle_slip_correction=%s]",
            M,
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
            block_centers=to_device(block_centers, "cpu"),
            phi_blocks=to_device(phi_blocks_out, "cpu"),
            show=True,
            title="CPR - Viterbi-Viterbi",
        )

    return restore_1d(was_1d, phi_full)
