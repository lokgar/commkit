"""Numba loops of the sequential equalizers (private).

The compiled CPU hot loops behind :func:`commkit.equalization.lms` and
``rls`` (with and without inline carrier-phase recovery), ``cma`` and ``rde``
(blind and pilot-aided) and ``cs_block``, the cycle-slip scan of
``block_lms``.  Imported on first use through the getters of
``_kernels_numba``, so ``import commkit`` does not load Numba.

The kernels share their steps (window, butterfly, slicer, tap update, the
RLS update) as helpers that Numba inlines into each kernel before compiling
it (``inline="always"``), so a helper compiles with its kernel's flags.
Every array passed to an inlined helper is reference-counted once per call,
so the per-symbol carrier recovery of :func:`cpr_loop` (one kernel for LMS
and RLS) stays written out in it, cycle-slip step included: as helpers, the
same steps cost 15-25% on the CPU.  It compiles without ``fastmath`` (the
S-fold unwrap rounds at the ±π boundary and the running sums must not be
reassociated), except for its arithmetic over the taps and the BPS
distances: those run in separate fast-math functions (``*_fm``,
:func:`_bps_distances`), which LLVM inlines with their own flags (an
``inline="always"`` helper would take the kernel's).  The other kernels
compile with ``fastmath``.

Conventions: ``x_padded`` ``(C, N_pad)`` complex64, ``W`` ``(C, C, T)``
complex64 (updated in place), ``y_out`` / ``e_out`` ``(N_sym, C)`` complex64,
``w_hist_out`` ``(N_sym or 1, C, C, T)``; dot products accumulate in float64,
RLS keeps ``P``, its gain and the regressor in complex128.
"""

from typing import Any

import numba
import numpy as np

_fast = numba.njit(cache=True, fastmath=True, nogil=True)
_strict = numba.njit(cache=True, nogil=True)
# A step shared by the kernels, inlined into each before it is compiled.
_step = numba.njit(inline="always", nogil=True)


# -----------------------------------------------------------------------------
# Shared steps
# -----------------------------------------------------------------------------


@_step
def _load_window(x_padded, start, X):
    """``X[c, t] = x_padded[c, start + t]`` for the ``(C, T)`` window."""
    for c in range(X.shape[0]):
        for t in range(X.shape[1]):
            X[c, t] = x_padded[c, start + t]


@_step
def _store_weights(W, w_hist_out, idx):
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            for t in range(W.shape[2]):
                w_hist_out[idx, i, j, t] = W[i, j, t]


@_step
def _butterfly(W, X, y):
    """``y[i] = Σ_j conj(W[i, j]) · X[j]``, accumulated in float64."""
    C = W.shape[0]
    num_taps = W.shape[2]
    for i in range(C):
        acc_re = 0.0
        acc_im = 0.0
        for j in range(C):
            for t in range(num_taps):
                w = W[i, j, t]
                x = X[j, t]
                acc_re += np.float64(w.real) * np.float64(x.real) + np.float64(
                    w.imag
                ) * np.float64(x.imag)
                acc_im += np.float64(w.real) * np.float64(x.imag) - np.float64(
                    w.imag
                ) * np.float64(x.real)
        y[i] = acc_re + 1j * acc_im


@_step
def _slice(z, constellation, sq_lev_min, sq_d_grid, sq_side):
    """Nearest constellation point of ``z``: snapped on a square grid
    (``sq_side > 0``), else by a scan over the points."""
    if sq_side > np.int32(0):
        ir = np.int32(np.round((z.real - sq_lev_min) / sq_d_grid))
        if ir < np.int32(0):
            ir = np.int32(0)
        if ir >= sq_side:
            ir = sq_side - np.int32(1)
        ii = np.int32(np.round((z.imag - sq_lev_min) / sq_d_grid))
        if ii < np.int32(0):
            ii = np.int32(0)
        if ii >= sq_side:
            ii = sq_side - np.int32(1)
        nr = sq_lev_min + np.float32(ir) * sq_d_grid
        ni = sq_lev_min + np.float32(ii) * sq_d_grid
        return np.complex64(nr + ni * np.complex64(1j))
    min_dist = np.float32(1e38)
    min_idx = 0
    for k in range(len(constellation)):
        diff = z - constellation[k]
        dist = diff.real * diff.real + diff.imag * diff.imag
        if dist < min_dist:
            min_dist = dist
            min_idx = k
    return constellation[min_idx]


@_step
def _update_taps(W, e, X, step_size):
    """Stochastic-gradient step ``W[i, j] += μ · conj(e[i]) · X[j]``."""
    for i in range(W.shape[0]):
        ce_i = np.conj(e[i])
        for j in range(W.shape[1]):
            for t in range(W.shape[2]):
                W[i, j, t] = W[i, j, t] + step_size * ce_i * X[j, t]


@_step
def _descend_taps(W, e, X, step_size):
    """Gradient-descent step of the blind equalizers,
    ``W[i, j] -= μ · conj(e[i]) · X[j]``."""
    for i in range(W.shape[0]):
        ce_i = np.conj(e[i])
        for j in range(W.shape[1]):
            for t in range(W.shape[2]):
                W[i, j, t] = W[i, j, t] - step_size * ce_i * X[j, t]


@_step
def _rls_update(W, P, x_bar, e, Px, xH_P, k_gain, lam_f64, leak_term, update):
    """RLS gain ``k = P x / (λ + xᴴ P x)``; with ``update``, the leaky tap
    step and the Riccati update of ``P`` (re-symmetrized in the same pass)."""
    N = x_bar.shape[0]
    num_taps = W.shape[2]
    # Px = P @ x_bar (in place, no heap allocation)
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
    inv_denom = np.float64(1.0) / denom
    for ii in range(N):
        k_gain[ii] = Px[ii] * inv_denom

    if update:
        # Leaky W update: W[i,j,t] = (1-γ)W[i,j,t] + k[j*T+t]*conj(e[i])
        for i in range(W.shape[0]):
            ce_i = np.conj(e[i])
            for j in range(W.shape[1]):
                for t in range(num_taps):
                    W[i, j, t] = np.complex64(
                        leak_term * W[i, j, t] + k_gain[j * num_taps + t] * ce_i
                    )
        # P = P^H, so (x^H P)[j] = conj((Px)[j]): reuse Px.
        for jj in range(N):
            xH_P[jj] = np.conj(Px[jj])
        # Riccati update and Hermitian re-symmetrization in one pass over
        # the upper triangle.
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


# Fast-math steps of the strict cpr_loop: separate compiled functions, so
# that LLVM inlines them into it with their own fast-math flags.
def _fast_copy(step: Any) -> Any:
    # NUMBA_DISABLE_JIT leaves the step a plain function, without py_func.
    return _fast(getattr(step, "py_func", step))


_butterfly_fm = _fast_copy(_butterfly)
_update_taps_fm = _fast_copy(_update_taps)
_rls_update_fm = _fast_copy(_rls_update)


@_fast
def _bps_distances(y, phases_neg, constellation, sq_lev_min, sq_d_grid, sq_side, d2):
    """``d2[k]``: squared distance of ``y`` rotated by ``phases_neg[k]`` to its
    nearest constellation point, snapped on a square grid or scanned."""
    for k in range(phases_neg.shape[0]):
        y_rot = y * phases_neg[k]
        if sq_side > np.int32(0):
            ir = np.int32(np.round((y_rot.real - sq_lev_min) / sq_d_grid))
            if ir < np.int32(0):
                ir = np.int32(0)
            if ir >= sq_side:
                ir = sq_side - np.int32(1)
            ii = np.int32(np.round((y_rot.imag - sq_lev_min) / sq_d_grid))
            if ii < np.int32(0):
                ii = np.int32(0)
            if ii >= sq_side:
                ii = sq_side - np.int32(1)
            nr = sq_lev_min + np.float32(ir) * sq_d_grid
            ni = sq_lev_min + np.float32(ii) * sq_d_grid
            d2[k] = (y_rot.real - nr) ** 2 + (y_rot.imag - ni) ** 2
        else:
            d2_min = np.float32(1e38)
            for m in range(constellation.shape[0]):
                dv = y_rot - constellation[m]
                dd = dv.real * dv.real + dv.imag * dv.imag
                if dd < d2_min:
                    d2_min = dd
            d2[k] = d2_min


@_step
def _cs_step(i, y_b, cs_buf_y, cs_buf_ptr, cs_buf_n, cs_stats, H, quantum, threshold):
    """One cycle-slip decision for channel ``i``; returns the repaired phase.

    The phase is predicted by a least-squares line through the last (up to
    ``H``) repaired phases, in relative coordinates so that ``Sx`` and
    ``Sxx`` are closed-form (``cs_stats[i] = [Sy, Sxy, -, -]``); a deviation
    beyond ``threshold`` is repaired by whole multiples of ``quantum``.  The
    first 10 phases are predicted by the last one.
    """
    n_b = cs_buf_n[i]
    ptr = cs_buf_ptr[i]
    if n_b == 0:
        phi_expected = y_b
    elif n_b < np.int64(10):
        last_pos = (ptr - np.int64(1) + np.int64(H)) % np.int64(H)
        phi_expected = cs_buf_y[i, last_pos]
    else:
        Sy = cs_stats[i, 0]
        Sxy = cs_stats[i, 1]
        n_f = np.float64(n_b)
        if n_b < np.int64(H):
            Sx_c = n_f * (n_f - np.float64(1.0)) / np.float64(2.0)
            Sxx_c = (
                n_f
                * (n_f - np.float64(1.0))
                * (np.float64(2.0) * n_f - np.float64(1.0))
                / np.float64(6.0)
            )
            denom = n_f * Sxx_c - Sx_c * Sx_c
        else:
            H_f = np.float64(H)
            Sx_c = H_f * (H_f - np.float64(1.0)) / np.float64(2.0)
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
    if np.abs(diff) > threshold and k_slip != np.int64(0):
        y_b = y_b - np.float64(k_slip) * quantum

    # Update the circular buffer (relative coordinates: only y is kept).
    write_pos = ptr % np.int64(H)
    if n_b == np.int64(H):
        old_y = cs_buf_y[i, write_pos]
        H_f = np.float64(H)
        old_Sy = cs_stats[i, 0]
        # Sxy_new = Sxy_old - Sy_old + y_old + (H-1)*y_new
        cs_stats[i, 1] = cs_stats[i, 1] - old_Sy + old_y + (H_f - np.float64(1.0)) * y_b
        cs_stats[i, 0] = old_Sy - old_y + y_b
    else:
        cs_stats[i, 1] += np.float64(n_b) * y_b
        cs_stats[i, 0] += y_b
    cs_buf_y[i, write_pos] = y_b
    cs_buf_ptr[i] = ptr + np.int64(1)
    if n_b < np.int64(H):
        cs_buf_n[i] = n_b + np.int64(1)
    return y_b


# -----------------------------------------------------------------------------
# Kernels
# -----------------------------------------------------------------------------


@_fast
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
    """Decision-directed LMS; ``training`` ``(C, N_sym)`` until ``n_train``."""
    C = W.shape[0]
    num_taps = W.shape[2]
    X_wins = np.empty((C, num_taps), dtype=np.complex64)
    y = np.empty(C, dtype=np.complex64)
    e = np.empty(C, dtype=np.complex64)
    for idx in range(y_out.shape[0]):
        _load_window(x_padded, idx * stride, X_wins)
        _butterfly(W, X_wins, y)
        for i in range(C):
            y_out[idx, i] = y[i]
            if idx < n_train:
                d_i = training[i, idx]
            else:
                d_i = _slice(y[i], constellation, sq_lev_min, sq_d_grid, sq_side)
            e[i] = d_i - y[i]
            e_out[idx, i] = e[i]
        _update_taps(W, e, X_wins, step_size)
        if store_weights:
            _store_weights(W, w_hist_out, idx)


@_fast
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
    """Decision-directed RLS.

    ``P`` ``(C·T, C·T)`` complex128 is updated in place, and the taps are
    frozen from symbol ``n_update_halt`` on.
    """
    C = W.shape[0]
    num_taps = W.shape[2]
    N = C * num_taps
    x_bar = np.empty(N, dtype=np.complex128)
    Px = np.empty(N, dtype=np.complex128)
    xH_P = np.empty(N, dtype=np.complex128)
    k_gain = np.empty(N, dtype=np.complex128)
    y = np.empty(C, dtype=np.complex64)
    e = np.empty(C, dtype=np.complex64)
    lam_f64 = np.float64(lam)
    leak_term = np.float32(1.0) - np.float32(leakage)
    X = x_bar.reshape((C, num_taps))  # the regressor as a (C, T) window
    for idx in range(y_out.shape[0]):
        _load_window(x_padded, idx * stride, X)
        _butterfly(W, X, y)
        for i in range(C):
            y_out[idx, i] = y[i]
            if idx < n_train:
                d_i = training[i, idx]
            else:
                d_i = _slice(y[i], constellation, sq_lev_min, sq_d_grid, sq_side)
            e[i] = d_i - y[i]
            e_out[idx, i] = e[i]
        _rls_update(
            W, P, x_bar, e, Px, xH_P, k_gain, lam_f64, leak_term, idx < n_update_halt
        )
        if store_weights:
            _store_weights(W, w_hist_out, idx)


@_strict
def cpr_loop(
    x_padded,
    training,
    constellation,
    bps_phases_neg,
    bps_angles,
    bps_block_size,
    bps_joint_channels,
    W,
    rls,
    step_size,
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
    bps_dist_buf,
    bps_running_sum,
    bps_dist_ptr,
    y_out,
    e_out,
    phase_out,
    w_hist_out,
    sq_lev_min,
    sq_d_grid,
    sq_side,
):
    """LMS or RLS with inline carrier recovery.

    LMS with rls False (step_size), RLS with rls True (P, lam, leakage, the
    taps frozen from n_update_halt).  cpr_mode 1 is a decision-directed
    PLL, 2 a causal BPS anchored by the training symbols, both with
    optional cycle-slip repair.

    The PLL state (pll_phi, pll_freq), the cycle-slip buffers
    (cs_*; cs_buf_x unused) and the BPS window (bps_prev4,
    bps_dist_buf, bps_running_sum, bps_dist_ptr) are updated in
    place for continuation; phase_out (N_sym, C) gets the phase.
    """
    C = W.shape[0]
    num_taps = W.shape[2]
    n_sym = y_out.shape[0]
    M = len(constellation)
    B = len(bps_phases_neg)
    H = cs_buf_x.shape[1]
    two_pi = np.float64(2.0 * np.pi)
    quantum = two_pi / np.float64(symmetry)

    N = C * num_taps
    X_wins = np.empty((C, num_taps), dtype=np.complex64)
    x_bar = np.empty(N, dtype=np.complex128)  # RLS: the regressor
    Px = np.empty(N, dtype=np.complex128)
    xH_P = np.empty(N, dtype=np.complex128)
    k_gain = np.empty(N, dtype=np.complex128)
    X_rls = x_bar.reshape((C, num_taps))
    lam_f64 = np.float64(lam)
    leak_term = np.float32(1.0) - np.float32(leakage)
    y_raw = np.empty(C, dtype=np.complex64)
    y_fin = np.empty(C, dtype=np.complex64)
    phi_c = np.empty(C, dtype=np.float64)
    d_sym = np.empty(C, dtype=np.complex64)
    e_clean = np.empty(C, dtype=np.complex64)
    e_eq = np.empty(C, dtype=np.complex64)
    phi_hat_bps = np.zeros(C, dtype=np.float64)
    da_buf = np.zeros((C, bps_block_size), dtype=np.complex128)
    da_sum = np.zeros(C, dtype=np.complex128)
    d2_new = np.empty(B, dtype=np.float32)
    for idx in range(n_sym):
        sample_idx = idx * stride

        if rls:
            _load_window(x_padded, sample_idx, X_rls)
            _butterfly_fm(W, X_rls, y_raw)
        else:
            _load_window(x_padded, sample_idx, X_wins)
            _butterfly_fm(W, X_wins, y_raw)

        # -- CPR: Phase estimation ----------------------------------

        # BPS: incremental running-sum over a causal window of K symbols.
        # For each new symbol, compute min-dist for all B candidates,
        # subtract the oldest slot from the running sum and add the new
        # one - O(B·M·C) per symbol, independent of window size K.
        # O(1) square-QAM path: snap real/imag to nearest level grid point.
        if cpr_mode == 2:
            slot = bps_dist_ptr[0] % np.int64(bps_block_size)
            for i in range(C):
                _bps_distances(
                    y_raw[i], bps_phases_neg, constellation,
                    sq_lev_min, sq_d_grid, sq_side, d2_new,
                )  # fmt: skip
                for k in range(B):
                    # Subtract evicted slot, add new slot
                    bps_running_sum[i, k] = (
                        bps_running_sum[i, k] - bps_dist_buf[i, slot, k] + d2_new[k]
                    )
                    bps_dist_buf[i, slot, k] = d2_new[k]
            bps_dist_ptr[0] = bps_dist_ptr[0] + np.int64(1)

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
                # Raw argmin in [0, 2π/S); apply S-fold causal unwrap (S = symmetry)
                raw4 = np.float64(bps_angles[best_k_joint]) * np.float64(symmetry)
                for i in range(C):
                    diff4 = raw4 - bps_prev4[i]
                    diff4 = diff4 - np.float64(2.0 * np.pi) * np.round(
                        diff4 / (np.float64(2.0 * np.pi))
                    )
                    bps_prev4[i] = bps_prev4[i] + diff4
                    phi_hat_bps[i] = bps_prev4[i] / np.float64(symmetry)
            else:
                for i in range(C):
                    best_k = np.int32(0)
                    min_tot = np.float64(1e300)
                    for k in range(B):
                        if bps_running_sum[i, k] < min_tot:
                            min_tot = bps_running_sum[i, k]
                            best_k = k
                    # Raw argmin in [0, 2π/S); apply S-fold causal unwrap (S = symmetry)
                    raw4 = np.float64(bps_angles[best_k]) * np.float64(symmetry)
                    diff4 = raw4 - bps_prev4[i]
                    diff4 = diff4 - np.float64(2.0 * np.pi) * np.round(
                        diff4 / (np.float64(2.0 * np.pi))
                    )
                    bps_prev4[i] = bps_prev4[i] + diff4
                    phi_hat_bps[i] = bps_prev4[i] / np.float64(symmetry)

        # Training symbols anchor the BPS phase: while they last, the
        # phase is the angle of sum(y_raw conj(d)) over the last K of
        # them, over the full 2π.  Blind BPS sees only 2π/S, and a
        # blind estimate here is copied into the taps by the training
        # error, so nothing would fix the absolute phase.  prev4 follows
        # the anchor, so the S-fold unwrap continues from its branch
        # once decisions take over.
        if cpr_mode == 2 and idx < n_train:
            da_slot = idx % bps_block_size
            for i in range(C):
                p_da = np.complex128(y_raw[i]) * np.conj(
                    np.complex128(training[i, idx])
                )
                da_sum[i] = da_sum[i] + p_da - da_buf[i, da_slot]
                da_buf[i, da_slot] = p_da
            da_tot = np.complex128(0.0)
            if bps_joint_channels:
                for i in range(C):
                    da_tot = da_tot + da_sum[i]
            for i in range(C):
                if not bps_joint_channels:
                    da_tot = da_sum[i]
                ang = np.arctan2(da_tot.imag, da_tot.real)
                prev = bps_prev4[i] / np.float64(symmetry)
                ang = ang + two_pi * np.round((prev - ang) / two_pi)
                bps_prev4[i] = ang * np.float64(symmetry)
                phi_hat_bps[i] = ang

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
                        last_pos = (ptr - np.int64(1) + np.int64(H)) % np.int64(H)
                        phi_expected = cs_buf_y[i, last_pos]
                    else:
                        # cs_stats layout: [0]=Sy, [1]=Sxy (relative coords)
                        Sy = cs_stats[i, 0]
                        Sxy = cs_stats[i, 1]
                        n_f = np.float64(n_b)
                        if n_b < np.int64(H):
                            Sx_c = n_f * (n_f - np.float64(1.0)) / np.float64(2.0)
                            Sxx_c = (
                                n_f
                                * (n_f - np.float64(1.0))
                                * (np.float64(2.0) * n_f - np.float64(1.0))
                                / np.float64(6.0)
                            )
                            denom = n_f * Sxx_c - Sx_c * Sx_c
                        else:
                            H_f = np.float64(H)
                            Sx_c = H_f * (H_f - np.float64(1.0)) / np.float64(2.0)
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
                    if np.abs(diff) > np.float64(cs_threshold) and k_slip != np.int64(
                        0
                    ):
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
            yr = y_raw[i].real * np.float32(cos_p) + y_raw[i].imag * np.float32(sin_p)
            yi = -y_raw[i].real * np.float32(sin_p) + y_raw[i].imag * np.float32(cos_p)
            y_fin[i] = np.complex64(yr + yi * np.complex64(1j))
            y_out[idx, i] = y_fin[i]

        # -- Slicer and error ---------------------------------------
        for i in range(C):
            if idx < n_train:
                d_i = training[i, idx]
            elif sq_side > np.int32(0):
                ir = np.int32(np.round((y_fin[i].real - sq_lev_min) / sq_d_grid))
                if ir < np.int32(0):
                    ir = np.int32(0)
                if ir >= sq_side:
                    ir = sq_side - np.int32(1)
                ii = np.int32(np.round((y_fin[i].imag - sq_lev_min) / sq_d_grid))
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
                        y_fin[i].imag * d_sym[i].real - y_fin[i].real * d_sym[i].imag
                    )
                e_ph_avg = e_ph_sum / np.float32(C)
                for i in range(C):
                    pll_phi[i] = pll_phi[i] + pll_mu * e_ph_avg + pll_freq[i]
                    pll_freq[i] = pll_freq[i] + pll_beta * e_ph_avg
            else:
                for i in range(C):
                    # Cross-product phase detector: Im(y_fin · conj(d))
                    e_ph = y_fin[i].imag * d_sym[i].real - y_fin[i].real * d_sym[i].imag
                    pll_phi[i] = pll_phi[i] + pll_mu * np.float32(e_ph) + pll_freq[i]
                    pll_freq[i] = pll_freq[i] + pll_beta * np.float32(e_ph)

        # -- De-rotate error and weight update ----------------------
        for i in range(C):
            # e_eq = e_clean * exp(+j * phi_corr)  (back to pre-rotation plane)
            cos_p = np.cos(phi_c[i])
            sin_p = np.sin(phi_c[i])
            er = e_clean[i].real * np.float32(cos_p) - e_clean[i].imag * np.float32(
                sin_p
            )
            ei = e_clean[i].real * np.float32(sin_p) + e_clean[i].imag * np.float32(
                cos_p
            )
            e_eq[i] = np.complex64(er + ei * np.complex64(1j))

        if rls:
            _rls_update_fm(
                W, P, x_bar, e_eq, Px, xH_P, k_gain, lam_f64, leak_term,
                idx < n_update_halt,
            )  # fmt: skip
        else:
            _update_taps_fm(W, e_eq, X_wins, step_size)

        if store_weights:
            _store_weights(W, w_hist_out, idx)


@_fast
def cma_loop(
    x_padded, W, step_size, r2, stride, store_weights, y_out, e_out, w_hist_out,
    use_pilots, pilot_ref, pilot_mask,
):  # fmt: skip
    """CMA with the Godard radius ``r2``, blind or pilot-aided.

    With ``use_pilots``, the symbols where ``pilot_mask`` ``(N_sym,)`` is set
    use the LMS residual against ``pilot_ref`` ``(C, N_sym)`` instead.
    """
    C = W.shape[0]
    num_taps = W.shape[2]
    X_wins = np.empty((C, num_taps), dtype=np.complex64)
    y = np.empty(C, dtype=np.complex64)
    e = np.empty(C, dtype=np.complex64)
    for idx in range(y_out.shape[0]):
        _load_window(x_padded, idx * stride, X_wins)
        _butterfly(W, X_wins, y)
        for i in range(C):
            y_out[idx, i] = y[i]
            if use_pilots and pilot_mask[idx]:
                # The residual inverted, to match the subtractive update.
                e[i] = y[i] - pilot_ref[i, idx]
            else:
                # Real |y|² keeps FP noise from rotating the error through y.
                mod2 = y[i].real * y[i].real + y[i].imag * y[i].imag
                e[i] = y[i] * np.float32(mod2 - r2)
            e_out[idx, i] = e[i]
        _descend_taps(W, e, X_wins, step_size)
        if store_weights:
            _store_weights(W, w_hist_out, idx)


@_fast
def rde_loop(
    x_padded, W, step_size, rings, stride, store_weights, y_out, e_out,
    w_hist_out, use_pilots, pilot_ref, pilot_mask,
):  # fmt: skip
    """RDE towards the nearest ring, blind or pilot-aided as :func:`cma_loop`.

    ``rings`` is the ``(2, K)`` table of ``_common._rde_rings``.
    """
    C = W.shape[0]
    num_taps = W.shape[2]
    K = rings.shape[1]
    X_wins = np.empty((C, num_taps), dtype=np.complex64)
    y = np.empty(C, dtype=np.complex64)
    e = np.empty(C, dtype=np.complex64)
    for idx in range(y_out.shape[0]):
        _load_window(x_padded, idx * stride, X_wins)
        _butterfly(W, X_wins, y)
        for i in range(C):
            y_out[idx, i] = y[i]
            if use_pilots and pilot_mask[idx]:
                e[i] = y[i] - pilot_ref[i, idx]
            else:
                mod2 = y[i].real * y[i].real + y[i].imag * y[i].imag
                k = 0
                for j in range(K - 1):  # branch-free: count the bounds below
                    k += mod2 > rings[1, j]
                e[i] = y[i] * np.float32(mod2 - rings[0, k])
            e_out[idx, i] = e[i]
        _descend_taps(W, e, X_wins, step_size)
        if store_weights:
            _store_weights(W, w_hist_out, idx)


@_fast
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
    """Repair the cycle slips of one equalizer block, symbol by symbol.

    ``phi_blk`` ``(C, B)`` float64 into ``phi_corr``; the buffers as in
    :func:`cpr_loop` (``cs_buf_x`` and ``b_start`` unused).
    """
    for ci in range(phi_blk.shape[0]):
        for i in range(phi_blk.shape[1]):
            phi_corr[ci, i] = _cs_step(
                ci,
                phi_blk[ci, i],
                cs_buf_y,
                cs_buf_ptr,
                cs_buf_n,
                cs_stats,
                cs_H,
                quantum,
                threshold,
            )
