"""Viterbi-Viterbi (V&V) carrier phase recovery."""

import numpy as np

from ..backend import ArrayType, dispatch, to_device
from ..core._signal_adapter import adapt_signal
from ..core.signal import Signal
from ..frequency import _modulation_power_m
from ..helpers import as_2d, restore_1d
from ..logger import logger
from ._common import _vv_block_phase
from .corrections import _log_phase_summary, correct_cycle_slips


def recover_carrier_phase_viterbi_viterbi(
    symbols: ArrayType | Signal,
    modulation: str | None = None,
    order: int | None = None,
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
    symbols : array_like or Signal
        1-SPS complex symbols after matched filter. Shape: (N,) or (C, N).
        A :class:`Signal` supplies ``modulation``/``order`` from its
        metadata when not given explicitly.
    modulation : str, optional
        Modulation scheme (case-insensitive): 'psk', 'qam', etc.  Required
        for array input; for :class:`Signal` input, used only as a fallback
        when the signal's ``mod_scheme`` is unset.
    order : int, optional
        Modulation order.  Required for array input; for :class:`Signal`
        input, used only as a fallback when the signal's ``mod_order`` is
        unset.
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
    signal_adapter = adapt_signal(
        symbols, function_name="recover_carrier_phase_viterbi_viterbi()"
    )
    symbols = signal_adapter.array
    modulation = signal_adapter.resolve_optional("mod_scheme", modulation)
    order = signal_adapter.resolve_optional("mod_order", order)

    if modulation is None or order is None:
        raise ValueError(
            "recover_carrier_phase_viterbi_viterbi() requires modulation and order "
            "for array input."
        )

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
                "CPR (VV): block_size=%s is too small for %s-QAM. "
                "Individual QAM symbols' M-th powers do not cancel the "
                "data modulation; insufficient averaging causes "
                "block-phase variance that exceeds the π/M unwrap "
                "threshold, producing persistent 2π/M phase slips. "
                "Recommended minimum for %s-QAM: block_size ≥ %s.",
                block_size,
                order,
                order,
                _min_bs,
            )

    phi_u, block_centers, all_positions = _vv_block_phase(
        symbols, xp, M, modulation, block_size, joint_channels
    )

    phi_full = xp.zeros((C, N), dtype=xp.float64)
    phi_blocks_out = xp.zeros((C, N_blocks), dtype=xp.float64)

    if joint_channels and C > 1:
        # phi_u's rows are broadcast-identical copies of the joint trajectory.
        phi_u_joint = phi_u[0]
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

    mode_str = "joint" if (joint_channels and C > 1) else "independent"
    phi_full_np = _log_phase_summary(
        phi_full,
        "CPR (Viterbi-Viterbi, M=%s, %s)",
        (M, mode_str),
        "[%s blocks x %s symbols, C=%s, cycle_slip_correction=%s]",
        (N_blocks, block_size, C, cycle_slip_correction),
        debug_plot=debug_plot,
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
