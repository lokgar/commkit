"""Decision-directed PLL carrier phase recovery."""

import logging

import numpy as np

from ..backend import ArrayType, dispatch, to_device
from ..core.signal import Signal
from ..helpers import as_2d, restore_1d, unwrap_signal
from ..logger import logger
from .corrections import correct_cycle_slips

_NUMBA_PLL: dict = {}


def _get_numba_dd_pll():
    """JIT-compile and cache the Numba DD-PLL sample-wise loop kernel.

    Returns
    -------
    callable
        Numba-compiled ``_dd_pll_loop``.
    """
    if "dd_pll" not in _NUMBA_PLL:
        import numba

        @numba.njit(cache=True, fastmath=True, nogil=True)
        def _dd_pll_loop(
            sym_r,
            sym_i,
            const_r,
            const_i,
            mu,
            beta,
            phi0,
            freq0,
            is_sq_qam,
            levels,
            d_grid,
            lev_min,
            side,
        ):
            """Inner DD-PLL loop compiled to machine code by Numba.

            Parameters
            ----------
            sym_r, sym_i : (N,) float64
                Real and imaginary parts of received symbols.
            const_r, const_i : (M,) float64
                Real and imaginary parts of reference constellation.
            mu : float64
                Proportional (phase) gain - corrects the instantaneous phase error.
            beta : float64
                Integral (frequency) gain - tracks residual frequency drift.
                Set to 0.0 for a 1st-order loop.
            phi0 : float64
                Initial phase state in radians.
            freq0 : float64
                Initial frequency correction state in radians/symbol.
            is_sq_qam : bool
                True when the constellation is a square QAM grid.  Enables the
                O(1) rounding decision path instead of the O(M) linear search.
            levels : (side,) float64
                Sorted unique axis levels for square QAM (ignored when not sq_qam).
            d_grid : float64
                Grid spacing (levels[1] - levels[0]).
            lev_min : float64
                Minimum level value (levels[0]).
            side : int
                Number of points per axis (sqrt of constellation order).

            Returns
            -------
            phase_est : (N,) float64
                Per-symbol phase trajectory φ[n].
            """
            N = len(sym_r)
            M = len(const_r)
            phase_est = np.empty(N, dtype=np.float64)
            phi = phi0
            freq = freq0

            for n in range(N):
                # Rotate received symbol by current phase estimate:
                # y[n] = s[n] · exp(-jφ[n])
                cos_phi = np.cos(phi)
                sin_phi = np.sin(phi)
                yr = sym_r[n] * cos_phi + sym_i[n] * sin_phi
                yi = -sym_r[n] * sin_phi + sym_i[n] * cos_phi

                # Hard decision: argmin_{c ∈ C} |y - c|²
                if is_sq_qam:
                    # O(1) grid rounding for square QAM
                    r_idx = int(round((yr - lev_min) / d_grid))
                    if r_idx < 0:
                        r_idx = 0
                    elif r_idx >= side:
                        r_idx = side - 1
                    d_r = levels[r_idx]
                    i_idx = int(round((yi - lev_min) / d_grid))
                    if i_idx < 0:
                        i_idx = 0
                    elif i_idx >= side:
                        i_idx = side - 1
                    d_i = levels[i_idx]
                else:
                    min_d2 = (yr - const_r[0]) ** 2 + (yi - const_i[0]) ** 2
                    d_r = const_r[0]
                    d_i = const_i[0]
                    for k in range(1, M):
                        d2 = (yr - const_r[k]) ** 2 + (yi - const_i[k]) ** 2
                        if d2 < min_d2:
                            min_d2 = d2
                            d_r = const_r[k]
                            d_i = const_i[k]

                # Cross-product phase error:  e = Im(y · d*) = yi·d_r - yr·d_i
                e = yi * d_r - yr * d_i

                # Record the phase used to derotate symbol n - before the update.
                phase_est[n] = phi

                # 2nd-order loop filter (reduces to 1st order when beta=0):
                #   φ[n+1] = φ[n] + μ·e[n] + ν[n]
                #   ν[n]   = ν[n-1] + β·e[n]
                phi = phi + mu * e + freq
                freq = freq + beta * e

            return phase_est

        _NUMBA_PLL["dd_pll"] = _dd_pll_loop

    return _NUMBA_PLL["dd_pll"]


def _get_numba_dd_pll_joint():
    """JIT-compile and cache the joint-channel DD-PLL PI kernel.

    Averages the cross-product phase error across C channels at each symbol
    before updating the single shared phase/frequency state.  This is the
    MVUE joint estimator for shared-LO systems.

    Returns
    -------
    callable
        Numba-compiled ``_dd_pll_joint_loop``.
    """
    if "dd_pll_joint" not in _NUMBA_PLL:
        import numba

        @numba.njit(cache=True, fastmath=True, nogil=True)
        def _dd_pll_joint_loop(
            sym_r,
            sym_i,
            const_r,
            const_i,
            mu,
            beta,
            phi0,
            freq0,
            is_sq_qam,
            levels,
            d_grid,
            lev_min,
            side,
        ):
            """Joint-channel DD-PLL with PI loop filter.

            Parameters
            ----------
            sym_r, sym_i : (C, N) float64
                Real and imaginary parts of received symbols, all channels.
            const_r, const_i : (M,) float64
                Reference constellation.
            mu, beta, phi0, freq0 : float64
                Loop parameters - same semantics as ``_dd_pll_loop``.
            is_sq_qam : bool
                Enables O(1) rounding decision for square QAM grids.
            levels : (side,) float64
            d_grid, lev_min : float64
            side : int

            Returns
            -------
            phase_est : (N,) float64
                Single shared phase trajectory (broadcast to all channels by caller).
            """
            C = sym_r.shape[0]
            N = sym_r.shape[1]
            M = len(const_r)
            phase_est = np.empty(N, dtype=np.float64)
            phi = phi0
            freq = freq0

            for n in range(N):
                cos_phi = np.cos(phi)
                sin_phi = np.sin(phi)
                e_sum = 0.0
                for c in range(C):
                    yr = sym_r[c, n] * cos_phi + sym_i[c, n] * sin_phi
                    yi = -sym_r[c, n] * sin_phi + sym_i[c, n] * cos_phi
                    if is_sq_qam:
                        r_idx = int(round((yr - lev_min) / d_grid))
                        if r_idx < 0:
                            r_idx = 0
                        elif r_idx >= side:
                            r_idx = side - 1
                        d_r = levels[r_idx]
                        i_idx = int(round((yi - lev_min) / d_grid))
                        if i_idx < 0:
                            i_idx = 0
                        elif i_idx >= side:
                            i_idx = side - 1
                        d_i = levels[i_idx]
                    else:
                        min_d2 = (yr - const_r[0]) ** 2 + (yi - const_i[0]) ** 2
                        d_r = const_r[0]
                        d_i = const_i[0]
                        for k in range(1, M):
                            d2 = (yr - const_r[k]) ** 2 + (yi - const_i[k]) ** 2
                            if d2 < min_d2:
                                min_d2 = d2
                                d_r = const_r[k]
                                d_i = const_i[k]
                    e_sum += yi * d_r - yr * d_i
                # Average error across channels - MVUE for shared LO
                e = e_sum / float(C)
                phase_est[n] = phi
                phi = phi + mu * e + freq
                freq = freq + beta * e

            return phase_est

        _NUMBA_PLL["dd_pll_joint"] = _dd_pll_joint_loop

    return _NUMBA_PLL["dd_pll_joint"]


def recover_carrier_phase_pll(
    symbols: ArrayType | Signal,
    modulation: str | None = None,
    order: int | None = None,
    mu: float | None = 1e-2,
    beta: float | None = None,
    phase_init: float = 0.0,
    loop_bandwidth_normalized: float = 1e-3,
    joint_channels: bool = False,
    cycle_slip_correction: bool = False,
    cycle_slip_history: int = 100,
    cycle_slip_threshold: float = np.pi / 4,
    debug_plot: bool = False,
) -> ArrayType:
    r"""
    Carrier phase recovery via a Decision-Directed Phase-Locked Loop (DD-PLL).

    Tracks the carrier phase symbol-by-symbol using hard decisions as phase
    references.  A 1st-order loop (``beta=0``) corrects static or slowly
    varying phase noise; a 2nd-order loop (``beta > 0``) additionally tracks
    a residual frequency offset left over after coarse FOE.

    This is the standard streaming CPR for hardware implementations: it is
    modulation-format agnostic (works for any QAM/PSK order) and converges
    much faster than block-based methods (VV, BPS) after equalizer pull-in.

    Note: the DD-PLL requires reliable decisions at the input.  For a cold
    start the first ~1/mu symbols may show slow convergence; a common strategy
    is to pre-converge with BPS or a short preamble and pass the phase as
    ``phase_init``.

    Parameters
    ----------
    symbols : array_like or Signal
        1-SPS complex symbols after matched filtering and FOE.
        Shape: ``(N,)`` or ``(C, N)``.  A :class:`Signal` supplies
        ``modulation``/``order`` from its metadata when not given explicitly.
    modulation : str, optional
        Modulation scheme (case-insensitive): ``'qam'``, ``'psk'``, etc.
        Used to fetch the reference constellation via
        ``gray_constellation``.  Required for array input; defaults to the
        signal's ``mod_scheme`` for :class:`Signal` input.
    order : int, optional
        Modulation order (4, 16, 64, ...).  Required for array input;
        defaults to the signal's ``mod_order`` for :class:`Signal` input.
    mu : float or None, default 1e-2
        Proportional gain - controls convergence speed and steady-state
        jitter.  Larger ``mu`` converges faster but amplifies noise.
        Typical range: ``1e-3`` (high-SNR, high-order QAM) to ``5e-2``
        (QPSK, low latency).  Set ``mu=None`` to opt into the
        ``loop_bandwidth_normalized`` shortcut instead (see below).  These
        gains are interchangeable with the inline equalizer PLL's
        ``cpr_pll_mu``/``cpr_pll_beta`` (``lms``/``rls``, ``cpr_type='pll'``).
    beta : float or None, default None
        Integral gain - enables 2nd-order frequency tracking.  ``None`` (or
        ``0.0``) gives a 1st-order loop; set ``beta > 0`` when a residual
        frequency offset remains after FOE (e.g. ``beta ≈ mu² / 4``).
        Requires ``mu`` to be set (passing ``beta`` with ``mu=None`` raises
        ``ValueError``).
    phase_init : float, default 0.0
        Initial phase state in radians.  Use the last sample of a
        preceding BPS or pilot-aided estimate to warm-start the loop.
    loop_bandwidth_normalized : float, default 1e-3
        Critically-damped (ζ=1) loop bandwidth shortcut, used only when
        ``mu is None``.  Normalized one-sided bandwidth in ``(0, 0.5)``;
        gains are derived as mu = 4*B_L, beta = 4*B_L^2.
    joint_channels : bool, default False
        For MIMO inputs (C > 1): if ``True``, average the cross-product
        phase error across all channels at each symbol before updating the
        shared loop state.  Both polarisations drive a single phase/frequency
        trajectory, giving ~√C variance reduction for shared-LO systems.
        The output ``phi_full[ch]`` rows are all identical.
        Has no effect for SISO (C = 1).
    cycle_slip_correction : bool, default False
        If ``True``, apply ``correct_cycle_slips`` to the per-symbol
        phase trajectory after the loop, to detect and fix sudden ``π/2``
        jumps caused by incorrect hard decisions near the branch boundary.
    cycle_slip_history : int, default 100
        ``history_length`` passed to ``correct_cycle_slips``.
        Default is higher than for block-phase methods because the trajectory
        is per-symbol (not per-block).
    cycle_slip_threshold : float, default π/4
        ``threshold`` passed to ``correct_cycle_slips`` (radians).

    Returns
    -------
    array_like
        Per-symbol phase estimate φ[n] in radians.
        Shape matches ``symbols``.  Same backend as input.

    Notes
    -----
    Inner loop: derotate by phi_hat, hard-decide, compute cross-product error
    e[n] = Im(y[n] * d_hat*[n]), update phi_hat[n+1] = phi_hat[n] + mu*e + nu,
    nu += beta*e.  Numba-compiled on CPU; GPU inputs are offloaded transparently.

    A global M-fold phase ambiguity always remains - resolve via a pilot or
    preamble reference after CPR.
    """
    from ..helpers import normalize, resolve_pll_gains
    from ..mapping import gray_constellation

    x, sig = unwrap_signal(symbols)
    if sig is not None:
        return recover_carrier_phase_pll(
            x,
            modulation or sig.mod_scheme,
            order or sig.mod_order,
            mu=mu,
            beta=beta,
            phase_init=phase_init,
            loop_bandwidth_normalized=loop_bandwidth_normalized,
            joint_channels=joint_channels,
            cycle_slip_correction=cycle_slip_correction,
            cycle_slip_history=cycle_slip_history,
            cycle_slip_threshold=cycle_slip_threshold,
            debug_plot=debug_plot,
        )

    if modulation is None or order is None:
        raise ValueError(
            "recover_carrier_phase_pll() requires modulation and order for array input."
        )

    # Resolve PI gains: raw mu/beta if given, else the critically-damped
    # bandwidth shortcut (mu=None).  Validate the bandwidth only on that path.
    if mu is None and not (0.0 < loop_bandwidth_normalized < 0.5):
        raise ValueError(
            f"loop_bandwidth_normalized must be in (0, 0.5), got {loop_bandwidth_normalized}."
        )
    mu, beta = resolve_pll_gains(loop_bandwidth_normalized, mu, beta)

    symbols, xp, _ = dispatch(symbols)
    symbols, was_1d = as_2d(symbols, name="symbols")
    C, N = symbols.shape

    # Normalise to unit average power so the effective loop gain is mu regardless
    # of input amplitude.  The error signal is e[n] = Im(y[n]*d_hat*), which
    # scales with signal amplitude; without this, the effective gain is mu*A
    # (where A is the RMS amplitude), making loop bandwidth input-dependent.
    symbols = normalize(symbols, mode="average_power", axis=-1)

    # Constellation on CPU (decisions are scalar operations in the loop)
    const_np = gray_constellation(modulation, order).astype(np.complex128)
    const_r = const_np.real.copy()
    const_i = const_np.imag.copy()

    # Square-QAM O(1) decision parameters.  For square QAM (order a perfect
    # square, e.g. 4/16/64/256/1024) the constellation is a uniform grid and
    # the nearest point can be found by rounding to the closest axis level.
    import math as _math

    _sq_root = _math.isqrt(order)
    _is_sq_qam = ("qam" in modulation.lower()) and (_sq_root * _sq_root == order)
    if _is_sq_qam:
        _levels = np.unique(const_np.real).astype(np.float64)
        _d_grid = float(_levels[1] - _levels[0]) if len(_levels) > 1 else 1.0
        _lev_min = float(_levels[0])
        _side = _sq_root
    else:
        _levels = np.empty(0, dtype=np.float64)
        _d_grid = 1.0
        _lev_min = 0.0
        _side = 0

    # Move to CPU for sequential processing
    if xp is not np:
        symbols_cpu = to_device(symbols, "cpu")
    else:
        symbols_cpu = symbols

    phi_full = np.zeros((C, N), dtype=np.float64)
    use_joint = joint_channels and C > 1

    # Pre-build (C, N) float64 views used by joint kernels
    if use_joint:
        symbols_np = symbols_cpu.astype(np.complex128)
        sym_r_all = np.ascontiguousarray(symbols_np.real)  # (C, N) float64
        sym_i_all = np.ascontiguousarray(symbols_np.imag)

    if use_joint:
        j_kernel = _get_numba_dd_pll_joint()
        phi_joint = j_kernel(
            sym_r_all,
            sym_i_all,
            const_r,
            const_i,
            float(mu),
            float(beta),
            float(phase_init),
            0.0,
            _is_sq_qam,
            _levels,
            _d_grid,
            _lev_min,
            _side,
        )
        for ch in range(C):
            phi_full[ch] = phi_joint
    else:
        pi_kernel = _get_numba_dd_pll()
        for ch in range(C):
            sym = symbols_cpu[ch].astype(np.complex128)
            phi_full[ch] = pi_kernel(
                sym.real.copy(),
                sym.imag.copy(),
                const_r,
                const_i,
                float(mu),
                float(beta),
                float(phase_init),
                0.0,
                _is_sq_qam,
                _levels,
                _d_grid,
                _lev_min,
                _side,
            )
    loop_order = "2nd" if beta > 0.0 else "1st"
    loop_desc = f"PI {loop_order}-order, mu={mu}, beta={beta}"

    if cycle_slip_correction:
        if use_joint:
            # All rows are identical - correct once and broadcast
            phi_full[0] = correct_cycle_slips(
                phi_full[0],
                symmetry=4,
                history_length=cycle_slip_history,
                threshold=cycle_slip_threshold,
            )
            for ch in range(1, C):
                phi_full[ch] = phi_full[0]
        else:
            for ch in range(C):
                phi_full[ch] = correct_cycle_slips(
                    phi_full[ch],
                    symmetry=4,
                    history_length=cycle_slip_history,
                    threshold=cycle_slip_threshold,
                )

    # Move result back to original device
    if xp is not np:
        phi_full = xp.asarray(phi_full)

    if logger.isEnabledFor(logging.INFO):
        # Two reductions + host syncs, needed only for the summary below.
        phi_mean_deg = float(np.mean(phi_full)) * 180.0 / np.pi
        phi_std_deg = float(np.std(phi_full)) * 180.0 / np.pi
        logger.info(
            "CPR (DD-PLL, %s): phase mean=%.2f°, std=%.2f° [C=%s]",
            loop_desc,
            phi_mean_deg,
            phi_std_deg,
            C,
        )

    if debug_plot:
        from .. import plotting as _plotting

        _plotting.plot_carrier_phase_trajectory(
            phi_full=phi_full if xp is np else to_device(phi_full, "cpu"),
            show=True,
            title=f"CPR - DD-PLL ({loop_desc})",
        )

    return restore_1d(was_1d, phi_full)
