"""MAP Tikhonov carrier phase recovery with RTS/SSKF smoothers."""

import logging

import numpy as np

from ..backend import ArrayType, dispatch, to_device
from ..core.signal import Signal
from ..frequency import _modulation_power_m
from ..helpers import as_2d, restore_1d, unwrap_signal
from ..logger import logger
from .corrections import correct_cycle_slips

_NUMBA_RTS: dict = {}


def _get_numba_rts_smoother():
    """JIT-compile and cache the Numba RTS-smoother kernel.

    Returns
    -------
    callable
        Numba-compiled ``_rts_loop``.
    """
    if "rts" not in _NUMBA_RTS:
        import numba

        @numba.njit(cache=True, fastmath=True, nogil=True)
        def _rts_loop(phi_obs, sigma_p2, sigma_v2):
            """Rauch-Tung-Striebel smoother - Numba inner kernel.

            Parameters
            ----------
            phi_obs : (B,) float64
            sigma_p2 : float64
            sigma_v2 : float64

            Returns
            -------
            (B,) float64
            """
            B = len(phi_obs)
            x_filt = np.empty(B, dtype=np.float64)
            P_filt = np.empty(B, dtype=np.float64)
            x_pred = np.empty(B, dtype=np.float64)
            P_pred = np.empty(B, dtype=np.float64)

            x_filt[0] = phi_obs[0]
            P_filt[0] = sigma_v2

            for k in range(1, B):
                x_pred[k] = x_filt[k - 1]
                P_pred[k] = P_filt[k - 1] + sigma_p2
                K = P_pred[k] / (P_pred[k] + sigma_v2)
                x_filt[k] = x_pred[k] + K * (phi_obs[k] - x_pred[k])
                P_filt[k] = (1.0 - K) * P_pred[k]

            x_smooth = x_filt.copy()
            for k in range(B - 2, -1, -1):
                G = P_filt[k] / P_pred[k + 1]
                x_smooth[k] = x_filt[k] + G * (x_smooth[k + 1] - x_pred[k + 1])

            return x_smooth

        _NUMBA_RTS["rts"] = _rts_loop

    return _NUMBA_RTS["rts"]


def _rts_smoother_1d(
    phi_obs: np.ndarray,
    sigma_p2: float,
    sigma_v2: float,
) -> np.ndarray:
    """Rauch-Tung-Striebel (RTS) Kalman smoother for a 1-D random-walk state.

    Uses the Numba-compiled kernel (``_get_numba_rts_smoother``) when
    available; falls back to a pure-Python loop otherwise.  Always runs on
    CPU - call with a NumPy array; the caller is responsible for
    ``to_device`` conversion.

    State model  : x[k+1] = x[k] + w[k],   w ~ N(0, sigma_p2)
    Observation  : y[k]   = x[k] + v[k],   v ~ N(0, sigma_v2)

    Parameters
    ----------
    phi_obs : (B,) float64
        Noisy block-phase observations in radians (e.g. from VV).
    sigma_p2 : float
        Process noise variance per block (Wiener phase noise increment).
    sigma_v2 : float
        Observation noise variance (VV estimator variance per block).

    Returns
    -------
    (B,) float64
        MAP-smoothed phase trajectory.
    """
    return _get_numba_rts_smoother()(phi_obs, float(sigma_p2), float(sigma_v2))


def _sskf_smoother_1d(
    phi_obs: ArrayType,
    sigma_p2: float,
    sigma_v2: float,
    sp,
    xp,
) -> ArrayType:
    """Steady-state Kalman smoother via zero-phase IIR filter (filtfilt).

    Approximates the RTS smoother by replacing the sequential Kalman
    recurrence with a 1st-order IIR filter whose gain is solved analytically
    from the discrete algebraic Riccati equation.  The bidirectional
    ``filtfilt`` call makes it equivalent to the RTS smoother in steady state.

    Backend-aware: uses ``sp.signal.filtfilt`` where ``sp`` is
    ``scipy`` (CPU) or ``cupyx.scipy`` (GPU) as returned by
    ``dispatch``.

    The approximation is excellent when ``B >> 1/K_∞``
    (typically ``B > 20``).  For ``B < 7`` (``filtfilt`` minimum), falls
    back to the exact ``_rts_smoother_1d`` on CPU.

    Parameters
    ----------
    phi_obs : (B,) float64, on the target device
        Noisy block-phase observations in radians.
    sigma_p2, sigma_v2 : float
        Process and observation noise variances per block.
    sp : module
        ``scipy`` or ``cupyx.scipy``, from ``dispatch``.
    xp : module
        ``numpy`` or ``cupy``, from ``dispatch``.

    Returns
    -------
    (B,) float64, same device as ``phi_obs``.
    """
    # filtfilt requires at least padlen * 2 + 1 samples; padlen = 3 * max(len(b), len(a)) = 6
    if len(phi_obs) < 7:
        phi_np = to_device(phi_obs, "cpu")
        return xp.asarray(_rts_smoother_1d(phi_np, sigma_p2, sigma_v2))

    # Steady-state prediction error covariance from discrete Riccati equation:
    #   p² - σ_p²·p - σ_p²·σ_v² = 0  ->  p = (σ_p² + √(σ_p⁴ + 4σ_p²σ_v²)) / 2
    p_ss = (sigma_p2 + float(np.sqrt(sigma_p2**2 + 4.0 * sigma_p2 * sigma_v2))) / 2.0
    K_ss = p_ss / (p_ss + sigma_v2)

    # Forward IIR:  y[k] = (1-K)·y[k-1] + K·x[k]
    #   H(z) = K / (1 - (1-K)·z⁻¹)
    # filtfilt applies forward + backward  ->  zero-phase, ≡ RTS smoother at
    # steady state.
    b = [K_ss]
    a = [1.0, -(1.0 - K_ss)]
    return sp.signal.filtfilt(b, a, phi_obs)


def recover_carrier_phase_tikhonov(
    symbols: ArrayType | Signal,
    modulation: str | None = None,
    order: int | None = None,
    linewidth_symbol_periods: float | None = None,
    block_size: int = 32,
    snr_db: float | None = None,
    method: str = "exact",
    joint_channels: bool = False,
    cycle_slip_correction: bool = False,
    cycle_slip_history: int = 100,
    cycle_slip_threshold: float = np.pi / 4,
    debug_plot: bool = False,
) -> ArrayType:
    r"""
    Carrier phase recovery via MAP estimation with a Tikhonov/Wiener phase
    noise prior.

    Extends the Viterbi-Viterbi block estimator with a Kalman smoother
    matched to the laser phase noise statistics.  Two smoother backends are
    available via ``method``:

    * ``'exact'`` - full Rauch-Tung-Striebel (RTS) smoother; Numba-compiled.
      Exact for all sequence lengths; runs on CPU.
    * ``'sskf'`` - steady-state Kalman filter approximation via zero-phase
      IIR (``filtfilt``); backend-aware (stays on GPU when input is on GPU).
      Approximation holds for ``N_blocks >> 1/K_∞`` (~20+ blocks typical).

    Parameters
    ----------
    symbols : array_like or Signal
        1-SPS complex symbols after matched filter and FOE.
        Shape: ``(N,)`` or ``(C, N)``.  A :class:`Signal` supplies
        ``modulation``/``order`` from its metadata when not given explicitly.
    modulation : str, optional
        Modulation scheme (case-insensitive): ``'psk'``, ``'qam'``, etc.
        Required for array input; for :class:`Signal` input, used only as a
        fallback when the signal's ``mod_scheme`` is unset.
    order : int, optional
        Modulation order.  Required for array input; for :class:`Signal`
        input, used only as a fallback when the signal's ``mod_order`` is
        unset.
    linewidth_symbol_periods : float
        Combined linewidth-symbol-time product delta_nu * T_s.
        Typical values: ``1e-5`` (narrow laser, 32 GBd), ``5e-4`` (wide
        laser / high baud rate).  Sets the Kalman process noise variance:
        sigma_p^2 = 2*pi * delta_nu * T_s * N_b.
    block_size : int, default 32
        Symbols per VV estimation block.  Same trade-off as for
        ``recover_carrier_phase_viterbi_viterbi``.
    snr_db : float or None, default None
        Per-symbol SNR in dB.  Used to compute the VV observation noise
        variance sigma_v^2 ≈ 1 / (M^2 * SNR * N_b).
        If ``None``, defaults to 20 dB with a warning - provide the actual
        operating SNR for the optimal smoother bandwidth.
    method : {'exact', 'sskf'}, default 'exact'
        Smoother implementation:

        * ``'exact'``: full RTS smoother (``_rts_smoother_1d``); Numba
          kernel when available.  Sequential CPU recurrence; exact for any
          ``N_blocks``.  On GPU inputs this forces a full device-to-host
          transfer of the block-phase trajectory (and back), stalling the
          GPU pipeline - prefer ``'sskf'`` for GPU-resident signals.
        * ``'sskf'``: steady-state approximation via ``filtfilt``
          (``_sskf_smoother_1d``); runs on the input device (GPU-native
          when data is on GPU, no host transfer).  Excellent for
          ``N_blocks ≥ 20``; for ``N_blocks < 7`` silently falls back to
          ``'exact'``.
    joint_channels : bool, default False
        For MIMO inputs (C > 1): if ``True``, sum the M-th-power block
        phasors across all channels before the VV phase extraction and
        Kalman smoother.  The single smoothed trajectory is broadcast to
        all C output rows.  Reduces variance by ~√C for shared-LO systems.
    cycle_slip_correction : bool, default False
        If ``True``, apply cycle-slip detection and correction
        (``correct_cycle_slips``) after the Kalman smoother, before
        interpolation.
    cycle_slip_history : int, default 100
        ``history_length`` passed to ``correct_cycle_slips``.
    cycle_slip_threshold : float, default π/4
        ``threshold`` passed to ``correct_cycle_slips`` (radians).
    debug_plot : bool, default False
        If ``True``, opens a diagnostic figure showing the per-symbol phase
        trajectory with the Kalman-smoothed block phases.

    Returns
    -------
    array_like
        Per-symbol phase estimate in radians.  Shape matches ``symbols``.
        Same backend as input.

    Notes
    -----
    VV block phases are Kalman-smoothed with sigma_p^2 = 2*pi*linewidth*T_s*N_b
    and sigma_v^2 ≈ 1/(M^2 * SNR * N_b), then interpolated to per-symbol
    resolution.  A residual 2*pi/M ambiguity always remains.
    """
    x, sig = unwrap_signal(symbols)
    if sig is not None:
        mod = sig.mod_scheme
        if mod is None:
            mod = modulation
            if mod is not None:
                logger.warning(
                    "recover_carrier_phase_tikhonov(): Signal has no "
                    "mod_scheme set; falling back to supplied modulation=%r.",
                    mod,
                )
        ord_ = sig.mod_order
        if ord_ is None:
            ord_ = order
            if ord_ is not None:
                logger.warning(
                    "recover_carrier_phase_tikhonov(): Signal has no "
                    "mod_order set; falling back to supplied order=%r.",
                    ord_,
                )
        return recover_carrier_phase_tikhonov(
            x,
            mod,
            ord_,
            linewidth_symbol_periods,
            block_size=block_size,
            snr_db=snr_db,
            method=method,
            joint_channels=joint_channels,
            cycle_slip_correction=cycle_slip_correction,
            cycle_slip_history=cycle_slip_history,
            cycle_slip_threshold=cycle_slip_threshold,
            debug_plot=debug_plot,
        )

    if modulation is None or order is None:
        raise ValueError(
            "recover_carrier_phase_tikhonov() requires modulation and order for "
            "array input."
        )
    if linewidth_symbol_periods is None:
        raise ValueError(
            "recover_carrier_phase_tikhonov() requires linewidth_symbol_periods."
        )

    if method not in ("exact", "sskf"):
        raise ValueError(f"Unknown method {method!r}. Choose 'exact' or 'sskf'.")

    symbols, xp, sp = dispatch(symbols)
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

    # Same data-residual constraint as VV: for QAM with order > 4 the M-th power
    # does not cancel per symbol.  Block phase variance can exceed π/M before the
    # Kalman smoother is applied, causing unwrap slips that the smoother cannot fix.
    if "qam" in modulation.lower() and order > 4:
        _min_bs = max(8, 4 * int(np.ceil(order**0.5)))
        if block_size < _min_bs:
            logger.warning(
                "CPR (Tikhonov): block_size=%s is too small for %s-QAM. "
                "Block phases are estimated via Viterbi-Viterbi; the "
                "data-residual constraint is identical - see "
                "recover_carrier_phase_viterbi_viterbi. Recommended "
                "minimum for %s-QAM: block_size ≥ %s.",
                block_size,
                order,
                order,
                _min_bs,
            )

    # Smoother noise parameters
    if snr_db is None:
        logger.warning(
            "CPR (Tikhonov): snr_db not provided - defaulting to 20 dB. "
            "Pass the operating SNR for the optimal smoother bandwidth."
        )
        snr_lin = 100.0  # 20 dB default
    else:
        snr_lin = 10.0 ** (snr_db / 10.0)

    sigma_p2 = float(2.0 * np.pi * linewidth_symbol_periods * block_size)
    sigma_v2 = float(1.0 / (M**2 * snr_lin * block_size))

    # VV block phase estimation with unit-circle normalisation for QAM
    blocks = symbols[:, :N_trunc].reshape(C, N_blocks, block_size)
    blocks_c = blocks.astype(
        xp.complex128 if blocks.dtype == xp.complex64 else blocks.dtype
    )
    if "qam" in modulation.lower():
        mag = xp.abs(blocks_c)
        blocks_c = blocks_c / xp.maximum(mag, 1e-15 * xp.max(mag))

    S_b = xp.sum(blocks_c**M, axis=-1)  # (C, N_blocks)

    block_centers = xp.arange(N_blocks, dtype=xp.float64) * block_size + block_size / 2
    all_positions = xp.arange(N, dtype=xp.float64)
    phi_full = xp.zeros((C, N), dtype=xp.float64)

    if joint_channels and C > 1:
        # Sum M-th-power phasors -> single VV estimate -> single Kalman pass
        S_b_joint = xp.sum(S_b, axis=0)  # (N_blocks,)
        phi_raw_joint = xp.angle(S_b_joint) / M
        phi_u_joint = xp.unwrap((phi_raw_joint * M).astype(xp.float64)) / M
        if "qam" in modulation.lower():
            phi_u_joint = phi_u_joint - (np.pi / M)

        # Kalman smoother on the joint trajectory
        if method == "exact":
            phi_u_joint_np = to_device(phi_u_joint, "cpu")
            phi_smooth_joint_np = _rts_smoother_1d(phi_u_joint_np, sigma_p2, sigma_v2)
            phi_smooth_joint = xp.asarray(phi_smooth_joint_np)
        else:
            phi_smooth_joint = _sskf_smoother_1d(
                phi_u_joint, sigma_p2, sigma_v2, sp, xp
            )
            phi_smooth_joint_np = to_device(phi_smooth_joint, "cpu")

        if cycle_slip_correction:
            phi_smooth_joint_np = correct_cycle_slips(
                to_device(phi_smooth_joint, "cpu"),
                4,
                cycle_slip_history,
                cycle_slip_threshold,
            )
            phi_smooth_joint = xp.asarray(phi_smooth_joint_np)

        phi_interp = xp.interp(all_positions, block_centers, phi_smooth_joint)
        for ch in range(C):
            phi_full[ch] = phi_interp
        phi_smooth_np = np.tile(to_device(phi_smooth_joint, "cpu"), (C, 1))
    else:
        phi_raw = xp.angle(S_b) / M
        phi_u = (
            xp.unwrap((phi_raw * M).astype(xp.float64), axis=-1) / M
        )  # (C, N_blocks)

        if "qam" in modulation.lower():
            phi_u = phi_u - (np.pi / M)

        if C > 1:
            # All per-channel means on device, one batched D2H, vectorized shift
            # (instead of one float() sync + one rounding per channel).
            diffs_np = to_device(xp.mean(phi_u[1:] - phi_u[0:1], axis=-1), "cpu")
            k_np = np.round(diffs_np * M / (2 * np.pi))
            phi_u[1:] = phi_u[1:] - xp.asarray(k_np)[:, None] * (2 * np.pi / M)

        # Kalman smoother - dispatch on method
        if method == "exact":
            phi_u_np = to_device(phi_u, "cpu")  # (C, N_blocks) float64
            phi_smooth_np = np.empty_like(phi_u_np)
            for ch in range(C):
                phi_smooth_np[ch] = _rts_smoother_1d(phi_u_np[ch], sigma_p2, sigma_v2)
            phi_smooth = xp.asarray(phi_smooth_np)
        else:  # method == "sskf"
            phi_smooth = xp.empty_like(phi_u)
            for ch in range(C):
                phi_smooth[ch] = _sskf_smoother_1d(
                    phi_u[ch], sigma_p2, sigma_v2, sp, xp
                )
            phi_smooth_np = to_device(phi_smooth, "cpu")

        for ch in range(C):
            phi_s_ch = phi_smooth[ch]
            if cycle_slip_correction:
                phi_s_ch_np = correct_cycle_slips(
                    to_device(phi_s_ch, "cpu"),
                    4,
                    cycle_slip_history,
                    cycle_slip_threshold,
                )
                phi_s_ch = xp.asarray(phi_s_ch_np)
                phi_smooth_np[ch] = to_device(phi_s_ch, "cpu")
            phi_full[ch] = xp.interp(all_positions, block_centers, phi_s_ch)

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
            "CPR (Tikhonov-%s, M=%s, %s): phase mean=%.2f°, std=%.2f° "
            "[%s blocks x %s, σ_p²=%.2e, σ_v²=%.2e, C=%s, "
            "cycle_slip_correction=%s]",
            method.upper(),
            M,
            mode_str,
            phi_mean_deg,
            phi_std_deg,
            N_blocks,
            block_size,
            sigma_p2,
            sigma_v2,
            C,
            cycle_slip_correction,
        )

    if debug_plot:
        from .. import plotting as _plotting

        _plotting.plot_carrier_phase_trajectory(
            phi_full=phi_full_np,
            block_centers=to_device(block_centers, "cpu"),
            phi_blocks=phi_smooth_np,
            show=True,
            title=f"CPR - Tikhonov-{method.upper()}",
        )

    return restore_1d(was_1d, phi_full)
