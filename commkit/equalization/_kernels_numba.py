"""Numba-compiled sequential equalizer kernels (CPU hot loops)."""

from __future__ import annotations

import numpy as np

# -----------------------------------------------------------------------------
# NUMBA LAZY LOADER
# -----------------------------------------------------------------------------

_NUMBA_CACHE: dict = {}


def _get_numba():
    """Lazy loader for Numba.

    Returns the ``numba`` module if installed, else ``None``.
    """
    if "numba" not in _NUMBA_CACHE:
        try:
            import numba

            _NUMBA_CACHE["numba"] = numba
        except ImportError:
            _NUMBA_CACHE["numba"] = None
    return _NUMBA_CACHE.get("numba")


# -----------------------------------------------------------------------------
# NUMBA KERNELS - ADAPTIVE EQUALIZER LOOPS
# -----------------------------------------------------------------------------
#
# Each factory lazily compiles a @njit kernel on first call and caches it in
# _NUMBA_KERNELS.  Numba specialises over argument *types* (all complex64/
# float32/int32/bool), so a single compiled version handles any (C, T, M).
#
# All kernels share the same calling conventions:
#   - x_padded   : (C, N_pad) complex64, C-contiguous
#   - W          : (C, C, T)  complex64, modified **in-place** - caller owns
#   - y_out      : (N_sym, C) complex64, pre-allocated output
#   - e_out      : (N_sym, C) complex64, pre-allocated output
#   - w_hist_out : (N_sym, C, C, T) if store_weights=True, else (1, C, C, T)
#
# Performance tricks:
#   - fastmath=True   - LLVM unsafe-math: enables SIMD vectorisation of
#                       inner tap loops (AVX-512 on modern CPUs)
#   - cache=True      - compiled LLVM IR cached to __pycache__ (*.nbi/*.nbc)
#   - Pre-allocated temporaries (X_wins, y, e, x_bar, Px ...) - no heap
#                       allocation inside the hot symbol loop
#   - Manual argmin slicer - explicit loop over M; LLVM can SIMD-vectorise
#                       the distance computation for small constellations
#   - In-place W update - no copy per step, minimal working-set pressure
#   - np.conj() on scalars - valid in Numba 0.64+

_NUMBA_KERNELS: dict = {}


def _get_numba_lms():
    """JIT-compile and cache the Numba LMS butterfly loop kernel.

    Returns
    -------
    lms_loop : numba-compiled callable
        See kernel source for argument shapes and semantics.
    """
    if "lms" not in _NUMBA_KERNELS:
        numba_mod = _get_numba()
        if numba_mod is None:
            raise ImportError("Numba is required for backend='numba'.")

        @numba_mod.njit(cache=True, fastmath=True, nogil=True)
        def lms_loop(
            x_padded,
            training,
            constellation,
            W,
            step_size,
            n_train,
            stride,
            store_weights,
            y_out,
            e_out,
            w_hist_out,
            sq_lev_min,
            sq_d_grid,
            sq_side,
        ):
            # x_padded      : (C, N_pad)          complex64
            # training      : (C, N_sym)           complex64
            # constellation : (M,)                 complex64
            # W             : (C, C, num_taps)      complex64 - modified in-place
            # step_size     : float32              - LMS step size μ; stable when 0 < μ < 2/(C·num_taps·P_x)
            # n_train       : int32                - training/DD boundary
            # stride        : int                  - sps (== 2 for T/2-spaced)
            # store_weights : bool
            # y_out         : (N_sym, C)            complex64 - pre-allocated
            # e_out         : (N_sym, C)            complex64 - pre-allocated
            # w_hist_out    : (N_sym or 1, C, C, T) complex64 - pre-allocated
            C = W.shape[0]
            num_taps = W.shape[2]
            n_sym = y_out.shape[0]
            M = len(constellation)

            X_wins = np.empty((C, num_taps), dtype=np.complex64)
            y = np.empty(C, dtype=np.complex64)
            e = np.empty(C, dtype=np.complex64)

            for idx in range(n_sym):
                sample_idx = idx * stride

                # Window extraction
                for c in range(C):
                    for t in range(num_taps):
                        X_wins[c, t] = x_padded[c, sample_idx + t]

                # Butterfly forward pass: y[i] = Σ_j conj(W[i,j]) · X_wins[j]
                for i in range(C):
                    acc_re = 0.0
                    acc_im = 0.0
                    for j in range(C):
                        for t in range(num_taps):
                            w = W[i, j, t]
                            x = X_wins[j, t]
                            acc_re += np.float64(w.real) * np.float64(
                                x.real
                            ) + np.float64(w.imag) * np.float64(x.imag)
                            acc_im += np.float64(w.real) * np.float64(
                                x.imag
                            ) - np.float64(w.imag) * np.float64(x.real)
                    y[i] = acc_re + 1j * acc_im
                    y_out[idx, i] = acc_re + 1j * acc_im

                # Desired symbol: training or decision-directed slicer
                for i in range(C):
                    if idx < n_train:
                        d_i = training[i, idx]
                    elif sq_side > np.int32(0):
                        ir = np.int32(np.round((y[i].real - sq_lev_min) / sq_d_grid))
                        if ir < np.int32(0):
                            ir = np.int32(0)
                        if ir >= sq_side:
                            ir = sq_side - np.int32(1)
                        ii = np.int32(np.round((y[i].imag - sq_lev_min) / sq_d_grid))
                        if ii < np.int32(0):
                            ii = np.int32(0)
                        if ii >= sq_side:
                            ii = sq_side - np.int32(1)
                        nr = sq_lev_min + np.float32(ir) * sq_d_grid
                        ni = sq_lev_min + np.float32(ii) * sq_d_grid
                        d_i = np.complex64(nr + ni * np.complex64(1j))
                    else:
                        min_dist = np.float32(1e38)
                        min_idx = 0
                        for k in range(M):
                            diff = y[i] - constellation[k]
                            dist = diff.real * diff.real + diff.imag * diff.imag
                            if dist < min_dist:
                                min_dist = dist
                                min_idx = k
                        d_i = constellation[min_idx]
                    e[i] = d_i - y[i]
                    e_out[idx, i] = e[i]

                # Weight update: W[i,j,t] += μ * conj(e[i]) * X_wins[j,t]
                for i in range(C):
                    ce_i = np.conj(e[i])
                    for j in range(C):
                        for t in range(num_taps):
                            W[i, j, t] = W[i, j, t] + step_size * ce_i * X_wins[j, t]

                if store_weights:
                    for i in range(C):
                        for j in range(C):
                            for t in range(num_taps):
                                w_hist_out[idx, i, j, t] = W[i, j, t]

        _NUMBA_KERNELS["lms"] = lms_loop
    return _NUMBA_KERNELS["lms"]


def _get_numba_rls():
    """JIT-compile and cache the Numba Leaky-RLS butterfly loop kernel.

    Returns
    -------
    rls_loop : numba-compiled callable
        See kernel source for argument shapes and semantics.
    """
    if "rls" not in _NUMBA_KERNELS:
        numba_mod = _get_numba()
        if numba_mod is None:
            raise ImportError("Numba is required for backend='numba'.")

        @numba_mod.njit(cache=True, fastmath=True, nogil=True)
        def rls_loop(
            x_padded,
            training,
            constellation,
            W,
            P,
            lam,
            leakage,
            n_train,
            n_update_halt,
            stride,
            store_weights,
            y_out,
            e_out,
            w_hist_out,
            sq_lev_min,
            sq_d_grid,
            sq_side,
        ):
            # x_padded      : (C, N_pad)                  complex64
            # training      : (C, N_sym)                   complex64
            # constellation : (M,)                         complex64
            # W             : (C, C, num_taps)              complex64 - in-place
            # P             : (C*num_taps, C*num_taps)      complex128 - in-place
            # lam           : float32  - forgetting factor λ ∈ (0, 1]
            # leakage       : float32  - weight-decay coefficient γ ∈ [0, 1)
            # n_train       : int32    - training/DD boundary
            # n_update_halt : int32    - freeze W,P beyond this index
            # stride        : int      - sps
            # store_weights : bool
            # y_out         : (N_sym, C)                   complex64
            # e_out         : (N_sym, C)                   complex64
            # w_hist_out    : (N_sym or 1, C, C, T)        complex64
            C = W.shape[0]
            num_taps = W.shape[2]
            n_sym = y_out.shape[0]
            N = C * num_taps  # regressor dimension
            M = len(constellation)

            x_bar = np.empty(N, dtype=np.complex128)
            Px = np.empty(N, dtype=np.complex128)
            xH_P = np.empty(N, dtype=np.complex128)
            k = np.empty(N, dtype=np.complex128)
            y = np.empty(C, dtype=np.complex64)
            e = np.empty(C, dtype=np.complex64)

            lam_f64 = np.float64(lam)
            leak_term = np.float32(1.0) - np.float32(leakage)

            for idx in range(n_sym):
                sample_idx = idx * stride

                # Flatten regressor directly from input
                for j in range(C):
                    for t in range(num_taps):
                        x_bar[j * num_taps + t] = x_padded[j, sample_idx + t]

                # Butterfly forward pass
                for i in range(C):
                    acc_re = 0.0
                    acc_im = 0.0
                    for j in range(C):
                        for t in range(num_taps):
                            w = W[i, j, t]
                            x = x_bar[j * num_taps + t]
                            acc_re += np.float64(w.real) * np.float64(
                                x.real
                            ) + np.float64(w.imag) * np.float64(x.imag)
                            acc_im += np.float64(w.real) * np.float64(
                                x.imag
                            ) - np.float64(w.imag) * np.float64(x.real)
                    y[i] = acc_re + 1j * acc_im
                    y_out[idx, i] = acc_re + 1j * acc_im

                # Desired and error
                for i in range(C):
                    if idx < n_train:
                        d_i = training[i, idx]
                    elif sq_side > np.int32(0):
                        ir = np.int32(np.round((y[i].real - sq_lev_min) / sq_d_grid))
                        if ir < np.int32(0):
                            ir = np.int32(0)
                        if ir >= sq_side:
                            ir = sq_side - np.int32(1)
                        ii = np.int32(np.round((y[i].imag - sq_lev_min) / sq_d_grid))
                        if ii < np.int32(0):
                            ii = np.int32(0)
                        if ii >= sq_side:
                            ii = sq_side - np.int32(1)
                        nr = sq_lev_min + np.float32(ir) * sq_d_grid
                        ni = sq_lev_min + np.float32(ii) * sq_d_grid
                        d_i = np.complex64(nr + ni * np.complex64(1j))
                    else:
                        min_dist = np.float32(1e38)
                        min_idx = 0
                        for kk in range(M):
                            diff = y[i] - constellation[kk]
                            dist = diff.real * diff.real + diff.imag * diff.imag
                            if dist < min_dist:
                                min_dist = dist
                                min_idx = kk
                        d_i = constellation[min_idx]
                    e[i] = d_i - y[i]
                    e_out[idx, i] = e[i]

                # Kalman gain: Px = P @ x_bar  (in-place, no heap allocation)
                for ii in range(N):
                    acc_re = 0.0
                    acc_im = 0.0
                    for jj in range(N):
                        p_val = P[ii, jj]
                        x_val = x_bar[jj]
                        acc_re += p_val.real * x_val.real - p_val.imag * x_val.imag
                        acc_im += p_val.real * x_val.imag + p_val.imag * x_val.real
                    Px[ii] = acc_re + 1j * acc_im

                # denom = λ + real(conj(x_bar) · Px)
                denom = lam_f64
                for jj in range(N):
                    denom = denom + (np.conj(x_bar[jj]) * Px[jj]).real

                # k = Px / denom
                inv_denom = np.float64(1.0) / denom
                for ii in range(N):
                    k[ii] = Px[ii] * inv_denom

                if idx < n_update_halt:
                    # Leaky W update: W[i,j,t] = (1-γ)W[i,j,t] + k[j*T+t]*conj(e[i])
                    for i in range(C):
                        ce_i = np.conj(e[i])
                        for j in range(C):
                            for t in range(num_taps):
                                W[i, j, t] = np.complex64(
                                    leak_term * W[i, j, t] + k[j * num_taps + t] * ce_i
                                )

                    # Exploit Hermitian symmetry P = P^H: for each j, (x^H P)[j] = conj((Px)[j]).
                    # Reuse the already-computed Px to avoid a second O(N²) mat-vec.
                    for jj in range(N):
                        xH_P[jj] = np.conj(Px[jj])

                    # Riccati + Hermitian re-symmetrization in one pass (upper
                    # triangle only). Halves memory traffic vs. two separate
                    # O(N²) loops; correctness is identical.
                    for ii in range(N):
                        for jj in range(ii, N):
                            val = (P[ii, jj] - k[ii] * xH_P[jj]) / lam_f64
                            if ii == jj:
                                P[ii, ii] = np.complex128(val.real)
                            else:
                                conj_val = (P[jj, ii] - k[jj] * xH_P[ii]) / lam_f64
                                avg = (val + np.conj(conj_val)) * np.float64(0.5)
                                P[ii, jj] = avg
                                P[jj, ii] = np.conj(avg)

                if store_weights:
                    for i in range(C):
                        for j in range(C):
                            for t in range(num_taps):
                                w_hist_out[idx, i, j, t] = W[i, j, t]

        _NUMBA_KERNELS["rls"] = rls_loop
    return _NUMBA_KERNELS["rls"]


def _get_numba_lms_cpr():
    """JIT-compile and cache the Numba LMS+CPR butterfly loop kernel.

    Extends the baseline LMS kernel with an inline carrier phase tracker
    (PLL or single-shot BPS) and a causal cycle-slip corrector.

    cpr_mode 1 = PLL (DD second-order PI loop).
    cpr_mode 2 = BPS (block-averaged Blind Phase Search, window=bps_block_size).

    Returns
    -------
    lms_cpr_loop : numba-compiled callable
    """
    if "lms_cpr" not in _NUMBA_KERNELS:
        numba_mod = _get_numba()
        if numba_mod is None:
            raise ImportError("Numba is required for backend='numba'.")

        @numba_mod.njit(cache=True, fastmath=False, nogil=True)
        def lms_cpr_loop(
            x_padded,
            training,
            constellation,
            bps_phases_neg,
            bps_angles,
            bps_block_size,
            bps_joint_channels,
            W,
            step_size,
            n_train,
            stride,
            store_weights,
            cpr_mode,
            pll_mu,
            pll_beta,
            symmetry,
            cs_enabled,
            cs_threshold,
            pll_phi,
            pll_freq,
            cs_buf_x,
            cs_buf_y,
            cs_buf_ptr,
            cs_buf_n,
            cs_stats,
            bps_prev4,
            y_out,
            e_out,
            phase_out,
            w_hist_out,
            sq_lev_min,
            sq_d_grid,
            sq_side,
        ):
            # x_padded          : (C, N_pad)        complex64
            # training          : (C, N_sym)         complex64
            # constellation     : (M,)               complex64
            # bps_phases_neg    : (B,)               complex64  exp(-j*theta_k)
            # bps_angles        : (B,)               float32    theta_k
            # bps_block_size    : int32              BPS averaging window (≥1)
            # bps_joint_channels: bool               True -> joint metric across C channels
            # W                 : (C, C, T)          complex64  - in-place
            # step_size     : float32
            # n_train       : int32
            # stride        : int
            # store_weights : bool
            # cpr_mode      : int32  1=pll 2=bps
            # pll_mu        : float32
            # pll_beta      : float32
            # symmetry      : int32  - cycle-slip quantum = 2π/symmetry
            # cs_enabled    : bool
            # cs_threshold  : float32
            # pll_phi       : (C,) float64          - in-place
            # pll_freq      : (C,) float64          - in-place
            # cs_buf_x      : (C, H) float64        - in-place
            # cs_buf_y      : (C, H) float64        - in-place
            # cs_buf_ptr    : (C,) int64            - in-place
            # cs_buf_n      : (C,) int64            - in-place
            # cs_stats      : (C, 4) float64  [Sx,Sy,Sxx,Sxy] - in-place
            # bps_prev4     : (C,) float64    - causal 4-fold unwrap state - in-place
            # y_out         : (N_sym, C) complex64
            # e_out         : (N_sym, C) complex64
            # phase_out     : (N_sym, C) float32
            # w_hist_out    : (N_sym or 1, C, C, T) complex64
            C = W.shape[0]
            num_taps = W.shape[2]
            n_sym = y_out.shape[0]
            M = len(constellation)
            B = len(bps_phases_neg)
            H = cs_buf_x.shape[1]
            two_pi = np.float64(2.0 * np.pi)
            quantum = two_pi / np.float64(symmetry)

            X_wins = np.empty((C, num_taps), dtype=np.complex64)
            y_raw = np.empty(C, dtype=np.complex64)
            y_fin = np.empty(C, dtype=np.complex64)
            phi_c = np.empty(C, dtype=np.float64)
            d_sym = np.empty(C, dtype=np.complex64)
            e_clean = np.empty(C, dtype=np.complex64)
            e_eq = np.empty(C, dtype=np.complex64)
            phi_hat_bps = np.zeros(C, dtype=np.float64)
            # BPS: incremental running-sum (avoids O(K·B·M) per-symbol rescan)
            bps_dist_buf = np.zeros((C, bps_block_size, B), dtype=np.float32)
            bps_running_sum = np.zeros((C, B), dtype=np.float64)
            bps_dist_ptr = np.int64(0)

            for idx in range(n_sym):
                sample_idx = idx * stride

                # Window extraction
                for c in range(C):
                    for t in range(num_taps):
                        X_wins[c, t] = x_padded[c, sample_idx + t]

                # Butterfly forward pass: y_raw[i] = Σ_j conj(W[i,j]) · X
                for i in range(C):
                    acc_re = 0.0
                    acc_im = 0.0
                    for j in range(C):
                        for t in range(num_taps):
                            w = W[i, j, t]
                            x = X_wins[j, t]
                            acc_re += np.float64(w.real) * np.float64(
                                x.real
                            ) + np.float64(w.imag) * np.float64(x.imag)
                            acc_im += np.float64(w.real) * np.float64(
                                x.imag
                            ) - np.float64(w.imag) * np.float64(x.real)
                    y_raw[i] = acc_re + 1j * acc_im

                # -- CPR: Phase estimation ----------------------------------

                # BPS: incremental running-sum over a causal window of K symbols.
                # For each new symbol, compute min-dist for all B candidates,
                # subtract the oldest slot from the running sum and add the new
                # one - O(B·M·C) per symbol, independent of window size K.
                # O(1) square-QAM path: snap real/imag to nearest level grid point.
                if cpr_mode == 2:
                    slot = bps_dist_ptr % np.int64(bps_block_size)
                    for i in range(C):
                        for k in range(B):
                            y_rot = y_raw[i] * bps_phases_neg[k]
                            if sq_side > np.int32(0):
                                ir = np.int32(
                                    np.round((y_rot.real - sq_lev_min) / sq_d_grid)
                                )
                                if ir < np.int32(0):
                                    ir = np.int32(0)
                                if ir >= sq_side:
                                    ir = sq_side - np.int32(1)
                                ii = np.int32(
                                    np.round((y_rot.imag - sq_lev_min) / sq_d_grid)
                                )
                                if ii < np.int32(0):
                                    ii = np.int32(0)
                                if ii >= sq_side:
                                    ii = sq_side - np.int32(1)
                                nr = sq_lev_min + np.float32(ir) * sq_d_grid
                                ni = sq_lev_min + np.float32(ii) * sq_d_grid
                                d2_min = (y_rot.real - nr) ** 2 + (y_rot.imag - ni) ** 2
                            else:
                                d2_min = np.float32(1e38)
                                for m in range(M):
                                    dv = y_rot - constellation[m]
                                    d2 = dv.real * dv.real + dv.imag * dv.imag
                                    if d2 < d2_min:
                                        d2_min = d2
                            # Subtract evicted slot, add new slot
                            bps_running_sum[i, k] = (
                                bps_running_sum[i, k]
                                - bps_dist_buf[i, slot, k]
                                + d2_min
                            )
                            bps_dist_buf[i, slot, k] = d2_min
                    bps_dist_ptr = bps_dist_ptr + np.int64(1)

                    if bps_joint_channels:
                        best_k_joint = np.int32(0)
                        min_tot_joint = np.float64(1e300)
                        for k in range(B):
                            metric_k = np.float64(0.0)
                            for i in range(C):
                                metric_k = metric_k + bps_running_sum[i, k]
                            if metric_k < min_tot_joint:
                                min_tot_joint = metric_k
                                best_k_joint = k
                        # Raw argmin in [0, π/2); apply 4-fold causal unwrap
                        raw4 = np.float64(bps_angles[best_k_joint]) * np.float64(4.0)
                        for i in range(C):
                            diff4 = raw4 - bps_prev4[i]
                            diff4 = diff4 - np.float64(2.0 * np.pi) * np.round(
                                diff4 / (np.float64(2.0 * np.pi))
                            )
                            bps_prev4[i] = bps_prev4[i] + diff4
                            phi_hat_bps[i] = bps_prev4[i] / np.float64(4.0)
                    else:
                        for i in range(C):
                            best_k = np.int32(0)
                            min_tot = np.float64(1e300)
                            for k in range(B):
                                if bps_running_sum[i, k] < min_tot:
                                    min_tot = bps_running_sum[i, k]
                                    best_k = k
                            # Raw argmin in [0, π/2); apply 4-fold causal unwrap
                            raw4 = np.float64(bps_angles[best_k]) * np.float64(4.0)
                            diff4 = raw4 - bps_prev4[i]
                            diff4 = diff4 - np.float64(2.0 * np.pi) * np.round(
                                diff4 / (np.float64(2.0 * np.pi))
                            )
                            bps_prev4[i] = bps_prev4[i] + diff4
                            phi_hat_bps[i] = bps_prev4[i] / np.float64(4.0)

                for i in range(C):
                    if cpr_mode == 1:  # PLL: read current integrator state
                        phi_hat = pll_phi[i]
                    else:  # BPS: unwrapped phase estimate
                        phi_hat = phi_hat_bps[i]

                    # -- Cycle-slip correction ------------------------------
                    if cs_enabled:
                        n_b = cs_buf_n[i]
                        ptr = cs_buf_ptr[i]
                        y_b = np.float64(phi_hat)

                        if n_b == 0:
                            phi_corr = phi_hat
                        else:
                            if n_b < np.int64(10):
                                last_pos = (ptr - np.int64(1) + np.int64(H)) % np.int64(
                                    H
                                )
                                phi_expected = cs_buf_y[i, last_pos]
                            else:
                                # cs_stats layout: [0]=Sy, [1]=Sxy (relative coords)
                                Sy = cs_stats[i, 0]
                                Sxy = cs_stats[i, 1]
                                n_f = np.float64(n_b)
                                if n_b < np.int64(H):
                                    Sx_c = (
                                        n_f * (n_f - np.float64(1.0)) / np.float64(2.0)
                                    )
                                    Sxx_c = (
                                        n_f
                                        * (n_f - np.float64(1.0))
                                        * (np.float64(2.0) * n_f - np.float64(1.0))
                                        / np.float64(6.0)
                                    )
                                    denom = n_f * Sxx_c - Sx_c * Sx_c
                                else:
                                    H_f = np.float64(H)
                                    Sx_c = (
                                        H_f * (H_f - np.float64(1.0)) / np.float64(2.0)
                                    )
                                    Sxx_c = (
                                        H_f
                                        * (H_f - np.float64(1.0))
                                        * (np.float64(2.0) * H_f - np.float64(1.0))
                                        / np.float64(6.0)
                                    )
                                    denom = H_f * Sxx_c - Sx_c * Sx_c
                                if np.abs(denom) > np.float64(1e-30):
                                    slope = (n_f * Sxy - Sx_c * Sy) / denom
                                    intercept = (Sy - slope * Sx_c) / n_f
                                else:
                                    slope = np.float64(0.0)
                                    intercept = Sy / n_f
                                phi_expected = slope * n_f + intercept

                            diff = y_b - phi_expected
                            k_slip = np.int64(np.round(diff / quantum))
                            if np.abs(diff) > np.float64(
                                cs_threshold
                            ) and k_slip != np.int64(0):
                                y_b = y_b - np.float64(k_slip) * quantum
                            phi_corr = y_b

                        # Update circular buffer - relative coords, only y needed
                        write_pos = ptr % np.int64(H)
                        if n_b == np.int64(H):
                            old_y = cs_buf_y[i, write_pos]
                            H_f = np.float64(H)
                            old_Sy = cs_stats[i, 0]
                            # Sxy_new = Sxy_old - Sy_old + y_old + (H-1)*y_new
                            cs_stats[i, 1] = (
                                cs_stats[i, 1]
                                - old_Sy
                                + old_y
                                + (H_f - np.float64(1.0)) * phi_corr
                            )
                            cs_stats[i, 0] = old_Sy - old_y + phi_corr
                        else:
                            n_b_f = np.float64(n_b)
                            cs_stats[i, 1] += n_b_f * phi_corr
                            cs_stats[i, 0] += phi_corr
                        cs_buf_y[i, write_pos] = phi_corr
                        cs_buf_ptr[i] = ptr + np.int64(1)
                        if n_b < np.int64(H):
                            cs_buf_n[i] = n_b + np.int64(1)
                    else:
                        phi_corr = np.float64(phi_hat)

                    phi_c[i] = phi_corr
                    phase_out[idx, i] = phi_corr

                    # y_final = y_raw * exp(-j * phi_corr)
                    cos_p = np.cos(phi_corr)
                    sin_p = np.sin(phi_corr)
                    yr = y_raw[i].real * np.float32(cos_p) + y_raw[i].imag * np.float32(
                        sin_p
                    )
                    yi = -y_raw[i].real * np.float32(sin_p) + y_raw[
                        i
                    ].imag * np.float32(cos_p)
                    y_fin[i] = np.complex64(yr + yi * np.complex64(1j))
                    y_out[idx, i] = y_fin[i]

                # -- Slicer and error ---------------------------------------
                for i in range(C):
                    if idx < n_train:
                        d_i = training[i, idx]
                    elif sq_side > np.int32(0):
                        ir = np.int32(
                            np.round((y_fin[i].real - sq_lev_min) / sq_d_grid)
                        )
                        if ir < np.int32(0):
                            ir = np.int32(0)
                        if ir >= sq_side:
                            ir = sq_side - np.int32(1)
                        ii = np.int32(
                            np.round((y_fin[i].imag - sq_lev_min) / sq_d_grid)
                        )
                        if ii < np.int32(0):
                            ii = np.int32(0)
                        if ii >= sq_side:
                            ii = sq_side - np.int32(1)
                        nr = sq_lev_min + np.float32(ir) * sq_d_grid
                        ni = sq_lev_min + np.float32(ii) * sq_d_grid
                        d_i = np.complex64(nr + ni * np.complex64(1j))
                    else:
                        min_dist = np.float32(1e38)
                        min_idx = 0
                        for k in range(M):
                            dv = y_fin[i] - constellation[k]
                            dist = dv.real * dv.real + dv.imag * dv.imag
                            if dist < min_dist:
                                min_dist = dist
                                min_idx = k
                        d_i = constellation[min_idx]
                    d_sym[i] = d_i
                    e_clean[i] = d_i - y_fin[i]
                    e_out[idx, i] = e_clean[i]

                # -- PLL state update ---------------------------------------
                if cpr_mode == 1:
                    if bps_joint_channels:
                        # Average phase error across channels (shared LO)
                        e_ph_sum = np.float32(0.0)
                        for i in range(C):
                            e_ph_sum = e_ph_sum + (
                                y_fin[i].imag * d_sym[i].real
                                - y_fin[i].real * d_sym[i].imag
                            )
                        e_ph_avg = e_ph_sum / np.float32(C)
                        for i in range(C):
                            pll_phi[i] = pll_phi[i] + pll_mu * e_ph_avg + pll_freq[i]
                            pll_freq[i] = pll_freq[i] + pll_beta * e_ph_avg
                    else:
                        for i in range(C):
                            # Cross-product phase detector: Im(y_fin · conj(d))
                            e_ph = (
                                y_fin[i].imag * d_sym[i].real
                                - y_fin[i].real * d_sym[i].imag
                            )
                            pll_phi[i] = (
                                pll_phi[i] + pll_mu * np.float32(e_ph) + pll_freq[i]
                            )
                            pll_freq[i] = pll_freq[i] + pll_beta * np.float32(e_ph)

                # -- De-rotate error and weight update ----------------------
                for i in range(C):
                    # e_eq = e_clean * exp(+j * phi_corr)  (back to pre-rotation plane)
                    cos_p = np.cos(phi_c[i])
                    sin_p = np.sin(phi_c[i])
                    er = e_clean[i].real * np.float32(cos_p) - e_clean[
                        i
                    ].imag * np.float32(sin_p)
                    ei = e_clean[i].real * np.float32(sin_p) + e_clean[
                        i
                    ].imag * np.float32(cos_p)
                    e_eq[i] = np.complex64(er + ei * np.complex64(1j))

                for i in range(C):
                    ce_i = np.conj(e_eq[i])
                    for j in range(C):
                        for t in range(num_taps):
                            W[i, j, t] = W[i, j, t] + step_size * ce_i * X_wins[j, t]

                if store_weights:
                    for i in range(C):
                        for j in range(C):
                            for t in range(num_taps):
                                w_hist_out[idx, i, j, t] = W[i, j, t]

        _NUMBA_KERNELS["lms_cpr"] = lms_cpr_loop
    return _NUMBA_KERNELS["lms_cpr"]


def _get_numba_rls_cpr():
    """JIT-compile and cache the Numba RLS+CPR butterfly loop kernel.

    Combines the Leaky-RLS Riccati update with the same inline CPR tracker
    used by the LMS CPR kernel.

    Returns
    -------
    rls_cpr_loop : numba-compiled callable
    """
    if "rls_cpr" not in _NUMBA_KERNELS:
        numba_mod = _get_numba()
        if numba_mod is None:
            raise ImportError("Numba is required for backend='numba'.")

        @numba_mod.njit(cache=True, fastmath=False, nogil=True)
        def rls_cpr_loop(
            x_padded,
            training,
            constellation,
            bps_phases_neg,
            bps_angles,
            bps_block_size,
            bps_joint_channels,
            W,
            P,
            lam,
            leakage,
            n_train,
            n_update_halt,
            stride,
            store_weights,
            cpr_mode,
            pll_mu,
            pll_beta,
            symmetry,
            cs_enabled,
            cs_threshold,
            pll_phi,
            pll_freq,
            cs_buf_x,
            cs_buf_y,
            cs_buf_ptr,
            cs_buf_n,
            cs_stats,
            bps_prev4,
            y_out,
            e_out,
            phase_out,
            w_hist_out,
            sq_lev_min,
            sq_d_grid,
            sq_side,
        ):
            # Same shapes as rls_loop plus CPR state - see lms_cpr_loop header.
            C = W.shape[0]
            num_taps = W.shape[2]
            n_sym = y_out.shape[0]
            N = C * num_taps
            M = len(constellation)
            B = len(bps_phases_neg)
            H = cs_buf_x.shape[1]
            two_pi = np.float64(2.0 * np.pi)
            quantum = two_pi / np.float64(symmetry)

            x_bar = np.empty(N, dtype=np.complex128)
            Px = np.empty(N, dtype=np.complex128)
            xH_P = np.empty(N, dtype=np.complex128)
            k_gain = np.empty(N, dtype=np.complex128)
            y_raw = np.empty(C, dtype=np.complex64)
            y_fin = np.empty(C, dtype=np.complex64)
            phi_c = np.empty(C, dtype=np.float64)
            d_sym = np.empty(C, dtype=np.complex64)
            e_clean = np.empty(C, dtype=np.complex64)
            e_eq = np.empty(C, dtype=np.complex64)
            phi_hat_bps = np.zeros(C, dtype=np.float64)
            bps_dist_buf = np.zeros((C, bps_block_size, B), dtype=np.float32)
            bps_running_sum = np.zeros((C, B), dtype=np.float64)
            bps_dist_ptr = np.int64(0)

            lam_f64 = np.float64(lam)
            leak_term = np.float32(1.0) - np.float32(leakage)

            for idx in range(n_sym):
                sample_idx = idx * stride

                for j in range(C):
                    for t in range(num_taps):
                        x_bar[j * num_taps + t] = x_padded[j, sample_idx + t]

                # Butterfly forward pass
                for i in range(C):
                    acc_re = 0.0
                    acc_im = 0.0
                    for j in range(C):
                        for t in range(num_taps):
                            w = W[i, j, t]
                            x = x_bar[j * num_taps + t]
                            acc_re += np.float64(w.real) * np.float64(
                                x.real
                            ) + np.float64(w.imag) * np.float64(x.imag)
                            acc_im += np.float64(w.real) * np.float64(
                                x.imag
                            ) - np.float64(w.imag) * np.float64(x.real)
                    y_raw[i] = acc_re + 1j * acc_im

                # -- CPR: Phase estimation ----------------------------------

                # BPS: incremental running-sum (O(B·M·C) per symbol, independent of K)
                # O(1) square-QAM path: snap real/imag to nearest level grid point.
                if cpr_mode == 2:
                    slot = bps_dist_ptr % np.int64(bps_block_size)
                    for i in range(C):
                        for k in range(B):
                            y_rot = y_raw[i] * bps_phases_neg[k]
                            if sq_side > np.int32(0):
                                ir = np.int32(
                                    np.round((y_rot.real - sq_lev_min) / sq_d_grid)
                                )
                                if ir < np.int32(0):
                                    ir = np.int32(0)
                                if ir >= sq_side:
                                    ir = sq_side - np.int32(1)
                                ii = np.int32(
                                    np.round((y_rot.imag - sq_lev_min) / sq_d_grid)
                                )
                                if ii < np.int32(0):
                                    ii = np.int32(0)
                                if ii >= sq_side:
                                    ii = sq_side - np.int32(1)
                                nr = sq_lev_min + np.float32(ir) * sq_d_grid
                                ni = sq_lev_min + np.float32(ii) * sq_d_grid
                                d2_min = (y_rot.real - nr) ** 2 + (y_rot.imag - ni) ** 2
                            else:
                                d2_min = np.float32(1e38)
                                for m in range(M):
                                    dv = y_rot - constellation[m]
                                    d2 = dv.real * dv.real + dv.imag * dv.imag
                                    if d2 < d2_min:
                                        d2_min = d2
                            bps_running_sum[i, k] = (
                                bps_running_sum[i, k]
                                - bps_dist_buf[i, slot, k]
                                + d2_min
                            )
                            bps_dist_buf[i, slot, k] = d2_min
                    bps_dist_ptr = bps_dist_ptr + np.int64(1)

                    if bps_joint_channels:
                        best_k_joint = np.int32(0)
                        min_tot_joint = np.float64(1e300)
                        for k in range(B):
                            metric_k = np.float64(0.0)
                            for i in range(C):
                                metric_k = metric_k + bps_running_sum[i, k]
                            if metric_k < min_tot_joint:
                                min_tot_joint = metric_k
                                best_k_joint = k
                        raw4 = np.float64(bps_angles[best_k_joint]) * np.float64(4.0)
                        for i in range(C):
                            diff4 = raw4 - bps_prev4[i]
                            diff4 = diff4 - np.float64(2.0 * np.pi) * np.round(
                                diff4 / (np.float64(2.0 * np.pi))
                            )
                            bps_prev4[i] = bps_prev4[i] + diff4
                            phi_hat_bps[i] = bps_prev4[i] / np.float64(4.0)
                    else:
                        for i in range(C):
                            best_k = np.int32(0)
                            min_tot = np.float64(1e300)
                            for k in range(B):
                                if bps_running_sum[i, k] < min_tot:
                                    min_tot = bps_running_sum[i, k]
                                    best_k = k
                            raw4 = np.float64(bps_angles[best_k]) * np.float64(4.0)
                            diff4 = raw4 - bps_prev4[i]
                            diff4 = diff4 - np.float64(2.0 * np.pi) * np.round(
                                diff4 / (np.float64(2.0 * np.pi))
                            )
                            bps_prev4[i] = bps_prev4[i] + diff4
                            phi_hat_bps[i] = bps_prev4[i] / np.float64(4.0)

                for i in range(C):
                    if cpr_mode == 1:
                        phi_hat = pll_phi[i]
                    else:
                        phi_hat = phi_hat_bps[i]

                    # -- Cycle-slip correction ------------------------------
                    if cs_enabled:
                        n_b = cs_buf_n[i]
                        ptr = cs_buf_ptr[i]
                        y_b = np.float64(phi_hat)

                        if n_b == 0:
                            phi_corr = phi_hat
                        else:
                            if n_b < np.int64(10):
                                last_pos = (ptr - np.int64(1) + np.int64(H)) % np.int64(
                                    H
                                )
                                phi_expected = cs_buf_y[i, last_pos]
                            else:
                                # cs_stats layout: [0]=Sy, [1]=Sxy (relative coords)
                                Sy = cs_stats[i, 0]
                                Sxy = cs_stats[i, 1]
                                n_f = np.float64(n_b)
                                if n_b < np.int64(H):
                                    Sx_c = (
                                        n_f * (n_f - np.float64(1.0)) / np.float64(2.0)
                                    )
                                    Sxx_c = (
                                        n_f
                                        * (n_f - np.float64(1.0))
                                        * (np.float64(2.0) * n_f - np.float64(1.0))
                                        / np.float64(6.0)
                                    )
                                    denom = n_f * Sxx_c - Sx_c * Sx_c
                                else:
                                    H_f = np.float64(H)
                                    Sx_c = (
                                        H_f * (H_f - np.float64(1.0)) / np.float64(2.0)
                                    )
                                    Sxx_c = (
                                        H_f
                                        * (H_f - np.float64(1.0))
                                        * (np.float64(2.0) * H_f - np.float64(1.0))
                                        / np.float64(6.0)
                                    )
                                    denom = H_f * Sxx_c - Sx_c * Sx_c
                                if np.abs(denom) > np.float64(1e-30):
                                    slope = (n_f * Sxy - Sx_c * Sy) / denom
                                    intercept = (Sy - slope * Sx_c) / n_f
                                else:
                                    slope = np.float64(0.0)
                                    intercept = Sy / n_f
                                phi_expected = slope * n_f + intercept

                            diff = y_b - phi_expected
                            k_slip = np.int64(np.round(diff / quantum))
                            if np.abs(diff) > np.float64(
                                cs_threshold
                            ) and k_slip != np.int64(0):
                                y_b = y_b - np.float64(k_slip) * quantum
                            phi_corr = y_b

                        # Update circular buffer - relative coords, only y needed
                        write_pos = ptr % np.int64(H)
                        if n_b == np.int64(H):
                            old_y = cs_buf_y[i, write_pos]
                            H_f = np.float64(H)
                            old_Sy = cs_stats[i, 0]
                            # Sxy_new = Sxy_old - Sy_old + y_old + (H-1)*y_new
                            cs_stats[i, 1] = (
                                cs_stats[i, 1]
                                - old_Sy
                                + old_y
                                + (H_f - np.float64(1.0)) * phi_corr
                            )
                            cs_stats[i, 0] = old_Sy - old_y + phi_corr
                        else:
                            n_b_f = np.float64(n_b)
                            cs_stats[i, 1] += n_b_f * phi_corr
                            cs_stats[i, 0] += phi_corr
                        cs_buf_y[i, write_pos] = phi_corr
                        cs_buf_ptr[i] = ptr + np.int64(1)
                        if n_b < np.int64(H):
                            cs_buf_n[i] = n_b + np.int64(1)
                    else:
                        phi_corr = np.float64(phi_hat)

                    phi_c[i] = phi_corr
                    phase_out[idx, i] = phi_corr

                    cos_p = np.cos(phi_corr)
                    sin_p = np.sin(phi_corr)
                    yr = y_raw[i].real * np.float32(cos_p) + y_raw[i].imag * np.float32(
                        sin_p
                    )
                    yi = -y_raw[i].real * np.float32(sin_p) + y_raw[
                        i
                    ].imag * np.float32(cos_p)
                    y_fin[i] = np.complex64(yr + yi * np.complex64(1j))
                    y_out[idx, i] = y_fin[i]

                # -- Slicer and error ---------------------------------------
                for i in range(C):
                    if idx < n_train:
                        d_i = training[i, idx]
                    elif sq_side > np.int32(0):
                        ir = np.int32(
                            np.round((y_fin[i].real - sq_lev_min) / sq_d_grid)
                        )
                        if ir < np.int32(0):
                            ir = np.int32(0)
                        if ir >= sq_side:
                            ir = sq_side - np.int32(1)
                        ii = np.int32(
                            np.round((y_fin[i].imag - sq_lev_min) / sq_d_grid)
                        )
                        if ii < np.int32(0):
                            ii = np.int32(0)
                        if ii >= sq_side:
                            ii = sq_side - np.int32(1)
                        nr = sq_lev_min + np.float32(ir) * sq_d_grid
                        ni = sq_lev_min + np.float32(ii) * sq_d_grid
                        d_i = np.complex64(nr + ni * np.complex64(1j))
                    else:
                        min_dist = np.float32(1e38)
                        min_idx = 0
                        for kk in range(M):
                            dv = y_fin[i] - constellation[kk]
                            dist = dv.real * dv.real + dv.imag * dv.imag
                            if dist < min_dist:
                                min_dist = dist
                                min_idx = kk
                        d_i = constellation[min_idx]
                    d_sym[i] = d_i
                    e_clean[i] = d_i - y_fin[i]
                    e_out[idx, i] = e_clean[i]

                # -- PLL state update ---------------------------------------
                if cpr_mode == 1:
                    if bps_joint_channels:
                        e_ph_sum = np.float32(0.0)
                        for i in range(C):
                            e_ph_sum = e_ph_sum + (
                                y_fin[i].imag * d_sym[i].real
                                - y_fin[i].real * d_sym[i].imag
                            )
                        e_ph_avg = e_ph_sum / np.float32(C)
                        for i in range(C):
                            pll_phi[i] = pll_phi[i] + pll_mu * e_ph_avg + pll_freq[i]
                            pll_freq[i] = pll_freq[i] + pll_beta * e_ph_avg
                    else:
                        for i in range(C):
                            e_ph = (
                                y_fin[i].imag * d_sym[i].real
                                - y_fin[i].real * d_sym[i].imag
                            )
                            pll_phi[i] = (
                                pll_phi[i] + pll_mu * np.float32(e_ph) + pll_freq[i]
                            )
                            pll_freq[i] = pll_freq[i] + pll_beta * np.float32(e_ph)

                # -- De-rotate error ----------------------------------------
                for i in range(C):
                    cos_p = np.cos(phi_c[i])
                    sin_p = np.sin(phi_c[i])
                    er = e_clean[i].real * np.float32(cos_p) - e_clean[
                        i
                    ].imag * np.float32(sin_p)
                    ei = e_clean[i].real * np.float32(sin_p) + e_clean[
                        i
                    ].imag * np.float32(cos_p)
                    e_eq[i] = np.complex64(er + ei * np.complex64(1j))

                # -- Kalman gain --------------------------------------------
                for ii in range(N):
                    acc_re = 0.0
                    acc_im = 0.0
                    for jj in range(N):
                        p_val = P[ii, jj]
                        x_val = x_bar[jj]
                        acc_re += p_val.real * x_val.real - p_val.imag * x_val.imag
                        acc_im += p_val.real * x_val.imag + p_val.imag * x_val.real
                    Px[ii] = acc_re + 1j * acc_im
                denom_k = lam_f64
                for jj in range(N):
                    denom_k = denom_k + (np.conj(x_bar[jj]) * Px[jj]).real
                inv_denom = np.float64(1.0) / denom_k
                for ii in range(N):
                    k_gain[ii] = Px[ii] * inv_denom

                if idx < n_update_halt:
                    for i in range(C):
                        ce_i = np.conj(e_eq[i])
                        for j in range(C):
                            for t in range(num_taps):
                                W[i, j, t] = np.complex64(
                                    leak_term * W[i, j, t]
                                    + k_gain[j * num_taps + t] * ce_i
                                )
                    for jj in range(N):
                        xH_P[jj] = np.conj(Px[jj])
                    for ii in range(N):
                        for jj in range(ii, N):
                            val = (P[ii, jj] - k_gain[ii] * xH_P[jj]) / lam_f64
                            if ii == jj:
                                P[ii, ii] = np.complex128(val.real)
                            else:
                                conj_val = (P[jj, ii] - k_gain[jj] * xH_P[ii]) / lam_f64
                                avg = (val + np.conj(conj_val)) * np.float64(0.5)
                                P[ii, jj] = avg
                                P[jj, ii] = np.conj(avg)

                if store_weights:
                    for i in range(C):
                        for j in range(C):
                            for t in range(num_taps):
                                w_hist_out[idx, i, j, t] = W[i, j, t]

        _NUMBA_KERNELS["rls_cpr"] = rls_cpr_loop
    return _NUMBA_KERNELS["rls_cpr"]


def _get_numba_cma():
    """JIT-compile and cache the Numba CMA butterfly loop kernel.

    Returns
    -------
    cma_loop : numba-compiled callable
        See kernel source for argument shapes and semantics.
    """
    if "cma" not in _NUMBA_KERNELS:
        numba_mod = _get_numba()
        if numba_mod is None:
            raise ImportError("Numba is required for backend='numba'.")

        @numba_mod.njit(cache=True, fastmath=True, nogil=True)
        def cma_loop(
            x_padded,
            W,
            step_size,
            r2,
            stride,
            store_weights,
            y_out,
            e_out,
            w_hist_out,
        ):
            # x_padded   : (C, N_pad)          complex64
            # W          : (C, C, num_taps)     complex64 - modified in-place
            # step_size  : float32             - fixed CMA gradient step μ
            # r2         : float32             - Godard radius R² = E[|s|⁴]/E[|s|²]
            # stride     : int                 - sps
            # store_weights : bool
            # y_out      : (N_sym, C)           complex64
            # e_out      : (N_sym, C)           complex64
            # w_hist_out : (N_sym or 1, C, C, T) complex64
            C = W.shape[0]
            num_taps = W.shape[2]
            n_sym = y_out.shape[0]

            X_wins = np.empty((C, num_taps), dtype=np.complex64)
            y = np.empty(C, dtype=np.complex64)
            e = np.empty(C, dtype=np.complex64)

            for idx in range(n_sym):
                sample_idx = idx * stride

                # Window extraction
                for c in range(C):
                    for t in range(num_taps):
                        X_wins[c, t] = x_padded[c, sample_idx + t]

                # Butterfly forward pass
                for i in range(C):
                    acc_re = 0.0
                    acc_im = 0.0
                    for j in range(C):
                        for t in range(num_taps):
                            w = W[i, j, t]
                            x = X_wins[j, t]
                            acc_re += np.float64(w.real) * np.float64(
                                x.real
                            ) + np.float64(w.imag) * np.float64(x.imag)
                            acc_im += np.float64(w.real) * np.float64(
                                x.imag
                            ) - np.float64(w.imag) * np.float64(x.real)
                    y[i] = acc_re + 1j * acc_im
                    y_out[idx, i] = acc_re + 1j * acc_im

                # CMA error: e[i] = y[i] * (|y[i]|² - R²)
                # Real-valued |y|² prevents imaginary leakage from FP noise
                # from causing a parasitic phase rotation via multiplication by y.
                for i in range(C):
                    mod2 = y[i].real * y[i].real + y[i].imag * y[i].imag
                    e[i] = y[i] * np.float32(mod2 - r2)
                    e_out[idx, i] = e[i]

                # Weight update (gradient descent): W -= mu * conj(e[i]) * X_wins[j,t]
                for i in range(C):
                    ce_i = np.conj(e[i])
                    for j in range(C):
                        for t in range(num_taps):
                            W[i, j, t] = W[i, j, t] - step_size * ce_i * X_wins[j, t]

                if store_weights:
                    for i in range(C):
                        for j in range(C):
                            for t in range(num_taps):
                                w_hist_out[idx, i, j, t] = W[i, j, t]

        _NUMBA_KERNELS["cma"] = cma_loop
    return _NUMBA_KERNELS["cma"]


def _get_numba_rde():
    """JIT-compile and cache the Numba RDE butterfly loop kernel.

    RDE (Radius Directed Equalizer) is a CMA variant that selects a
    per-symbol target radius from the set of unique constellation radii
    rather than using a single fixed Godard radius.  This makes it
    converge correctly on multi-ring constellations (16-QAM, 64-QAM).

    Returns
    -------
    rde_loop : numba-compiled callable
        See kernel source for argument shapes and semantics.
    """
    if "rde" not in _NUMBA_KERNELS:
        numba_mod = _get_numba()
        if numba_mod is None:
            raise ImportError("Numba is required for backend='numba'.")

        @numba_mod.njit(cache=True, fastmath=True, nogil=True)
        def rde_loop(
            x_padded,
            W,
            step_size,
            radii,
            stride,
            store_weights,
            y_out,
            e_out,
            w_hist_out,
        ):
            # x_padded    : (C, N_pad)            complex64
            # W           : (C, C, num_taps)       complex64 - modified in-place
            # step_size   : float32               - fixed gradient step μ
            # radii       : (K,)                  float32   - unique |c| radii, sorted ascending
            # stride      : int                   - sps
            # store_weights : bool
            # y_out       : (N_sym, C)             complex64
            # e_out       : (N_sym, C)             complex64
            # w_hist_out  : (N_sym or 1, C, C, T)  complex64
            C = W.shape[0]
            num_taps = W.shape[2]
            n_sym = y_out.shape[0]
            K = radii.shape[0]

            X_wins = np.empty((C, num_taps), dtype=np.complex64)
            y = np.empty(C, dtype=np.complex64)
            e = np.empty(C, dtype=np.complex64)

            for idx in range(n_sym):
                sample_idx = idx * stride

                # Window extraction
                for c in range(C):
                    for t in range(num_taps):
                        X_wins[c, t] = x_padded[c, sample_idx + t]

                # Butterfly forward pass
                for i in range(C):
                    acc_re = 0.0
                    acc_im = 0.0
                    for j in range(C):
                        for t in range(num_taps):
                            w = W[i, j, t]
                            x = X_wins[j, t]
                            acc_re += np.float64(w.real) * np.float64(
                                x.real
                            ) + np.float64(w.imag) * np.float64(x.imag)
                            acc_im += np.float64(w.real) * np.float64(
                                x.imag
                            ) - np.float64(w.imag) * np.float64(x.real)
                    y[i] = acc_re + 1j * acc_im
                    y_out[idx, i] = acc_re + 1j * acc_im

                # RDE error: e[i] = y[i] * (|y[i]|² - R_d²)
                # R_d is the magnitude of the nearest constellation ring.
                # Linear scan over K unique radii is cheap (K ≤ 8 for typical QAM).
                for i in range(C):
                    mod2 = y[i].real * y[i].real + y[i].imag * y[i].imag
                    abs_y = np.sqrt(mod2)

                    best_rd = radii[0]
                    best_dist = abs(abs_y - radii[0])
                    for k in range(1, K):
                        dist_k = abs(abs_y - radii[k])
                        if dist_k < best_dist:
                            best_dist = dist_k
                            best_rd = radii[k]

                    e[i] = y[i] * np.float32(mod2 - best_rd * best_rd)
                    e_out[idx, i] = e[i]

                # Weight update: W -= mu * conj(e[i]) * X_wins[j,t]
                for i in range(C):
                    ce_i = np.conj(e[i])
                    for j in range(C):
                        for t in range(num_taps):
                            W[i, j, t] = W[i, j, t] - step_size * ce_i * X_wins[j, t]

                if store_weights:
                    for i in range(C):
                        for j in range(C):
                            for t in range(num_taps):
                                w_hist_out[idx, i, j, t] = W[i, j, t]

        _NUMBA_KERNELS["rde"] = rde_loop
    return _NUMBA_KERNELS["rde"]


# -----------------------------------------------------------------------------
# HYBRID DA / BLIND KERNELS  (internal - used by cma/rde when pilot_ref supplied)
# -----------------------------------------------------------------------------
#
# These kernels process every symbol with a per-step error-function switch:
#
#   pilot_mask[idx] == 1  ->  DA-LMS error:   e[i] = pilot_ref[i, idx] - y[i]
#   pilot_mask[idx] == 0  ->  blind error:    e[i] = CMA or RDE formula
#
# The weight update rule is identical to the blind case.  This achieves
# phase-resolved convergence at pilot/preamble positions while preserving
# correct blind tracking at data positions - all in a single kernel pass.


def _get_numba_pa_cma():
    """JIT-compile and cache the Numba pilot-aided CMA butterfly loop kernel.

    Hybrid CMA: LMS error at pilot positions (pilot_mask==1), standard
    Godard CMA error at data positions (pilot_mask==0).  "Pilot-Aided" (PA)
    is the standard telecom term for this mixed DA/blind adaptation strategy.

    Returns
    -------
    pa_cma_loop : numba-compiled callable
    """
    if "pa_cma" not in _NUMBA_KERNELS:
        numba_mod = _get_numba()
        if numba_mod is None:
            raise ImportError("Numba is required for backend='numba'.")

        @numba_mod.njit(cache=True, fastmath=True, nogil=True)
        def pa_cma_loop(
            x_padded,
            W,
            step_size,
            r2,
            stride,
            store_weights,
            y_out,
            e_out,
            w_hist_out,
            pilot_ref,
            pilot_mask,
        ):
            # x_padded   : (C, N_pad)          complex64
            # W          : (C, C, num_taps)     complex64 - modified in-place
            # step_size  : float32
            # r2         : float32             - Godard radius R²
            # stride     : int
            # store_weights : bool
            # y_out      : (N_sym, C)           complex64
            # e_out      : (N_sym, C)           complex64
            # w_hist_out : (N_sym or 1, C, C, T) complex64
            # pilot_ref  : (C, N_sym)           complex64 - known symbol; 0 at data positions
            # pilot_mask : (N_sym,)             uint8     - 1 at pilot/preamble, 0 at data
            C = W.shape[0]
            num_taps = W.shape[2]
            n_sym = y_out.shape[0]

            X_wins = np.empty((C, num_taps), dtype=np.complex64)
            y = np.empty(C, dtype=np.complex64)
            e = np.empty(C, dtype=np.complex64)

            for idx in range(n_sym):
                sample_idx = idx * stride

                # Window extraction
                for c in range(C):
                    for t in range(num_taps):
                        X_wins[c, t] = x_padded[c, sample_idx + t]

                # Butterfly forward pass
                for i in range(C):
                    acc_re = 0.0
                    acc_im = 0.0
                    for j in range(C):
                        for t in range(num_taps):
                            w = W[i, j, t]
                            x = X_wins[j, t]
                            acc_re += np.float64(w.real) * np.float64(
                                x.real
                            ) + np.float64(w.imag) * np.float64(x.imag)
                            acc_im += np.float64(w.real) * np.float64(
                                x.imag
                            ) - np.float64(w.imag) * np.float64(x.real)
                    y[i] = acc_re + 1j * acc_im
                    y_out[idx, i] = acc_re + 1j * acc_im

                # Error: DA-LMS at pilots, CMA Godard at data
                for i in range(C):
                    if pilot_mask[idx]:
                        e[i] = (
                            y[i] - pilot_ref[i, idx]
                        )  # inverted to match blind subtractive update
                    else:
                        mod2 = y[i].real * y[i].real + y[i].imag * y[i].imag
                        e[i] = y[i] * np.float32(mod2 - r2)
                    e_out[idx, i] = e[i]

                # Weight update
                for i in range(C):
                    ce_i = np.conj(e[i])
                    for j in range(C):
                        for t in range(num_taps):
                            W[i, j, t] = W[i, j, t] - step_size * ce_i * X_wins[j, t]

                if store_weights:
                    for i in range(C):
                        for j in range(C):
                            for t in range(num_taps):
                                w_hist_out[idx, i, j, t] = W[i, j, t]

        _NUMBA_KERNELS["pa_cma"] = pa_cma_loop
    return _NUMBA_KERNELS["pa_cma"]


def _get_numba_pa_rde():
    """JIT-compile and cache the Numba pilot-aided RDE butterfly loop kernel.

    Hybrid RDE: LMS error at pilot positions (pilot_mask==1), standard
    ring-directed RDE error at data positions (pilot_mask==0).  "Pilot-Aided"
    (PA) is the standard telecom term for this mixed DA/blind adaptation strategy.

    Returns
    -------
    pa_rde_loop : numba-compiled callable
    """
    if "pa_rde" not in _NUMBA_KERNELS:
        numba_mod = _get_numba()
        if numba_mod is None:
            raise ImportError("Numba is required for backend='numba'.")

        @numba_mod.njit(cache=True, fastmath=True, nogil=True)
        def pa_rde_loop(
            x_padded,
            W,
            step_size,
            radii,
            stride,
            store_weights,
            y_out,
            e_out,
            w_hist_out,
            pilot_ref,
            pilot_mask,
        ):
            # x_padded   : (C, N_pad)            complex64
            # W          : (C, C, num_taps)       complex64 - modified in-place
            # step_size  : float32
            # radii      : (K,)                  float32   - unique |c| radii, sorted
            # stride     : int
            # store_weights : bool
            # y_out      : (N_sym, C)             complex64
            # e_out      : (N_sym, C)             complex64
            # w_hist_out : (N_sym or 1, C, C, T)  complex64
            # pilot_ref  : (C, N_sym)             complex64 - 0 at data positions
            # pilot_mask : (N_sym,)               uint8     - 1 at pilot/preamble
            C = W.shape[0]
            num_taps = W.shape[2]
            n_sym = y_out.shape[0]
            K = radii.shape[0]

            X_wins = np.empty((C, num_taps), dtype=np.complex64)
            y = np.empty(C, dtype=np.complex64)
            e = np.empty(C, dtype=np.complex64)

            for idx in range(n_sym):
                sample_idx = idx * stride

                # Window extraction
                for c in range(C):
                    for t in range(num_taps):
                        X_wins[c, t] = x_padded[c, sample_idx + t]

                # Butterfly forward pass
                for i in range(C):
                    acc_re = 0.0
                    acc_im = 0.0
                    for j in range(C):
                        for t in range(num_taps):
                            w = W[i, j, t]
                            x = X_wins[j, t]
                            acc_re += np.float64(w.real) * np.float64(
                                x.real
                            ) + np.float64(w.imag) * np.float64(x.imag)
                            acc_im += np.float64(w.real) * np.float64(
                                x.imag
                            ) - np.float64(w.imag) * np.float64(x.real)
                    y[i] = acc_re + 1j * acc_im
                    y_out[idx, i] = acc_re + 1j * acc_im

                # Error: DA-LMS at pilots, RDE ring-directed at data
                for i in range(C):
                    if pilot_mask[idx]:
                        e[i] = (
                            y[i] - pilot_ref[i, idx]
                        )  # inverted to match blind subtractive update
                    else:
                        mod2 = y[i].real * y[i].real + y[i].imag * y[i].imag
                        abs_y = np.sqrt(mod2)
                        best_rd = radii[0]
                        best_dist = abs(abs_y - radii[0])
                        for k in range(1, K):
                            dist_k = abs(abs_y - radii[k])
                            if dist_k < best_dist:
                                best_dist = dist_k
                                best_rd = radii[k]
                        e[i] = y[i] * np.float32(mod2 - best_rd * best_rd)
                    e_out[idx, i] = e[i]

                # Weight update
                for i in range(C):
                    ce_i = np.conj(e[i])
                    for j in range(C):
                        for t in range(num_taps):
                            W[i, j, t] = W[i, j, t] - step_size * ce_i * X_wins[j, t]

                if store_weights:
                    for i in range(C):
                        for j in range(C):
                            for t in range(num_taps):
                                w_hist_out[idx, i, j, t] = W[i, j, t]

        _NUMBA_KERNELS["pa_rde"] = pa_rde_loop
    return _NUMBA_KERNELS["pa_rde"]


def _get_numba_cs_block():
    """Lazy-compile a Numba JIT kernel for the block_lms cycle-slip correction loop.

    Replaces the Python ``for ci in range(C): for i in range(B)`` loop with
    compiled native code.  D->H/H->D transfers (few KB) still happen, but the
    loop body itself runs at native speed, eliminating the dominant overhead
    when ``cpr_cycle_slip_correction=True`` on GPU.
    """
    if "cs_block" not in _NUMBA_KERNELS:
        numba_mod = _get_numba()
        if numba_mod is None:
            return None

        @numba_mod.njit(cache=True, fastmath=True, nogil=True)
        def cs_block(
            phi_blk,
            phi_corr,
            cs_buf_x,
            cs_buf_y,
            cs_buf_ptr,
            cs_buf_n,
            cs_stats,
            b_start,
            quantum,
            threshold,
            cs_H,
        ):
            """Per-symbol cycle-slip correction for one equalizer block.

            Parameters
            ----------
            phi_blk   : (C, B) float64 - BPS phase before correction (input)
            phi_corr  : (C, B) float64 - corrected phase (output, pre-allocated)
            cs_buf_x  : (C, H) float64 - unused (retained for call-site compat)
            cs_buf_y  : (C, H) float64 - circular buffer of past corrected phases
            cs_buf_ptr: (C,)   int64   - write pointer (monotonically increasing)
            cs_buf_n  : (C,)   int64   - number of valid entries (≤ H)
            cs_stats  : (C, 4) float64 - [0]=Sy, [1]=Sxy (relative coords); [2..3] unused
            b_start   : int            - global symbol index of block start (unused)
            quantum   : float64        - slip quantum (2π / symmetry)
            threshold : float64        - |diff| threshold for declaring a slip
            cs_H      : int            - circular buffer length
            """
            C = phi_blk.shape[0]
            B = phi_blk.shape[1]
            H_f = float(cs_H)
            Sx_full = H_f * (H_f - 1.0) / 2.0
            Sxx_full = H_f * (H_f - 1.0) * (2.0 * H_f - 1.0) / 6.0
            denom_full = H_f * Sxx_full - Sx_full * Sx_full
            for ci in range(C):
                for i in range(B):
                    y_b = phi_blk[ci, i]
                    n_b = cs_buf_n[ci]
                    ptr = cs_buf_ptr[ci]

                    if n_b == 0:
                        phi_expected = y_b
                    elif n_b < 10:
                        last_pos = (ptr - 1 + cs_H) % cs_H
                        phi_expected = cs_buf_y[ci, last_pos]
                    else:
                        sy = cs_stats[ci, 0]
                        sxy = cs_stats[ci, 1]
                        n_f = float(n_b)
                        if n_b < cs_H:
                            Sx_c = n_f * (n_f - 1.0) / 2.0
                            Sxx_c = n_f * (n_f - 1.0) * (2.0 * n_f - 1.0) / 6.0
                            denom = n_f * Sxx_c - Sx_c * Sx_c
                        else:
                            Sx_c = Sx_full
                            denom = denom_full
                        if abs(denom) > 1e-30:
                            slope = (n_f * sxy - Sx_c * sy) / denom
                            intercept = (sy - slope * Sx_c) / n_f
                        else:
                            slope = 0.0
                            intercept = sy / n_f
                        phi_expected = slope * n_f + intercept

                    diff = y_b - phi_expected
                    k_slip = int(round(diff / quantum))
                    if abs(diff) > threshold and k_slip != 0:
                        y_b -= float(k_slip) * quantum
                    phi_corr[ci, i] = y_b

                    # Update circular buffer - relative coords, only y needed
                    write_pos = ptr % cs_H
                    if n_b == cs_H:
                        old_y = cs_buf_y[ci, write_pos]
                        old_sy = cs_stats[ci, 0]
                        # Sxy_new = Sxy_old - Sy_old + y_old + (H-1)*y_new
                        cs_stats[ci, 1] = (
                            cs_stats[ci, 1] - old_sy + old_y + (H_f - 1.0) * y_b
                        )
                        cs_stats[ci, 0] = old_sy - old_y + y_b
                    else:
                        cs_stats[ci, 1] += float(n_b) * y_b
                        cs_stats[ci, 0] += y_b
                    cs_buf_y[ci, write_pos] = y_b
                    cs_buf_ptr[ci] = ptr + 1
                    if n_b < cs_H:
                        cs_buf_n[ci] = n_b + 1

        _NUMBA_KERNELS["cs_block"] = cs_block
    return _NUMBA_KERNELS["cs_block"]
