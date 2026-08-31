"""JAX (lax.scan) sequential and block equalizer kernels."""

from __future__ import annotations

import functools
from typing import Any

from ..backend import _get_jax

# -----------------------------------------------------------------------------
# JAX KERNELS - ADAPTIVE EQUALIZER SCANS
# -----------------------------------------------------------------------------
#
# Each factory JIT-compiles a jax.lax.scan kernel on first call and caches
# it in _JITTED_EQ.  Static closure variables are baked into the compiled XLA
# program - changing any of them produces a cache miss and triggers a new
# trace+compile (not a retrace of an existing graph).
#
# Static variables (shared by all three kernels):
#   num_taps : FIR length per polyphase arm - fixes XLA buffer allocation
#   stride   : decimation factor (== sps, typically 2 for T/2-spaced input)
#   num_ch   : MIMO butterfly width C - fixes matrix shapes in XLA IR
#
# Kernel-specific static variable:
#   const_size : constellation size M for LMS/RLS - fixes the slicer argmin
#                shape at trace time so XLA can compile the min-distance
#                search without dynamic dispatch.  CMA is blind (no slicer),
#                so this variable is omitted from its key.
#
# All kernels share the same output convention:
#   y_hat  : (N_sym, C)              complex64 - equalized symbols
#              (transposed by _unpack_result_jax to (C, N_sym) before return)
#   errors : (N_sym, C)              complex64 - complex errors d - y
#   w_hist : (N_sym, C, C, num_taps) complex64 - weight snapshots per symbol
#
# Performance notes:
#   - jax.jit + XLA ahead-of-time compilation eliminates Python overhead for
#     the scan loop; each symbol step is a single XLA op dispatch.
#   - jax.vmap(get_win) vectorises window extraction across C channels.
#   - jnp.einsum('ijt,jt->i') compiles to an optimised GEMV via OpenBLAS
#     on CPU or cuBLAS on GPU.
#   - lax.dynamic_slice emits a single gather op with runtime offset and
#     compile-time slice size - no Python-level indexing overhead.
#   - For GPU: lax.scan serialises the symbol loop on a single CUDA stream,
#     limiting parallelism to the per-step GEMV.  GPU is only beneficial
#     for large MIMO widths (C >> 4) that saturate the cuBLAS kernel.
#     Use backend='numba' for CPU-optimal throughput on typical SISO/2x2.

_JITTED_EQ: dict[tuple[Any, ...], Any] = {}


def _get_jax_lms(
    num_taps, stride, const_size, num_ch, sq_side=0, sq_lev_min=0.0, sq_d_grid=1.0
):
    """JIT-compile and cache the sample-by-sample LMS butterfly scan.

    Static closure variables (baked into the compiled kernel; a new cache
    entry is created - not a retrace - when any of these change):

    num_taps   : FIR filter length per polyphase arm.
    stride     : decimation factor (== sps, typically 2 for T/2-spaced input).
    const_size : constellation size M - fixes the slicer ``argmin`` shape at
                 trace time so XLA can compile it without dynamic dispatch.
    num_ch     : MIMO butterfly width C (number of input/output channels).
    sq_side    : int - 0 means O(M) slicer; >0 enables O(1) square-QAM slicer.
    sq_lev_min, sq_d_grid : float - constellation level grid parameters.

    Returns
    -------
    lms_scan : JIT-compiled callable
        See the inner function for the call signature.
    """
    key = (
        "lms",
        num_taps,
        stride,
        const_size,
        num_ch,
        sq_side,
        float(sq_lev_min),
        float(sq_d_grid),
    )
    if key not in _JITTED_EQ:
        jax, jnp, _ = _get_jax()

        @jax.jit
        def lms_scan(
            x_input, training_padded, constellation, w_init, step_size, n_train
        ):
            # x_input         : (C, N_pad)        complex64
            # training_padded : (C, N_sym)         complex64
            # constellation   : (M,)               complex64 - slicer lookup table
            # w_init          : (C, C, num_taps)   complex64
            # step_size       : scalar float32
            # n_train         : scalar int32

            def step(W, idx):
                sample_idx = idx * stride

                X_wins = jax.lax.dynamic_slice(
                    x_input, (0, sample_idx), (num_ch, num_taps)
                )

                _P = jax.lax.Precision.HIGHEST
                y = jnp.einsum("ijt,jt->i", jnp.conj(W), X_wins, precision=_P)  # (C,)

                def slicer(ch_y):
                    if sq_side > 0:  # static branch at trace time
                        ir = jnp.clip(
                            jnp.round((ch_y.real - sq_lev_min) / sq_d_grid).astype(
                                jnp.int32
                            ),
                            0,
                            sq_side - 1,
                        )
                        ii = jnp.clip(
                            jnp.round((ch_y.imag - sq_lev_min) / sq_d_grid).astype(
                                jnp.int32
                            ),
                            0,
                            sq_side - 1,
                        )
                        nr = sq_lev_min + ir.astype(jnp.float32) * jnp.float32(
                            sq_d_grid
                        )
                        ni = sq_lev_min + ii.astype(jnp.float32) * jnp.float32(
                            sq_d_grid
                        )
                        return jax.lax.complex(nr, ni)
                    else:
                        return constellation[
                            jnp.argmin(jnp.abs(ch_y - constellation) ** 2)
                        ]

                dd = jax.vmap(slicer)(y)  # (C,)
                d = jnp.where(idx < n_train, training_padded[:, idx], dd)

                e = d - y  # (C,)

                W_new = W + step_size * jnp.einsum("i,jt->ijt", jnp.conj(e), X_wins)
                return W_new, (y, e, W_new)

            n_sym = training_padded.shape[1]
            W_final, (y_hat, errors, w_hist) = jax.lax.scan(
                step, w_init, jnp.arange(n_sym)
            )
            return y_hat, errors, W_final, w_hist

        _JITTED_EQ[key] = lms_scan
    return _JITTED_EQ[key]


def _get_jax_rls(
    num_taps, stride, const_size, num_ch, sq_side=0, sq_lev_min=0.0, sq_d_grid=1.0
):
    """JIT-compile and cache the sample-by-sample Leaky-RLS butterfly scan.

    Static closure variables (same semantics as ``_get_jax_lms``):
    num_taps, stride, const_size, num_ch.
    sq_side, sq_lev_min, sq_d_grid: O(1) square-QAM slicer parameters.

    Returns
    -------
    rls_scan : JIT-compiled callable
        See the inner function for the call signature.
    """
    key = (
        "rls",
        num_taps,
        stride,
        const_size,
        num_ch,
        sq_side,
        float(sq_lev_min),
        float(sq_d_grid),
    )
    if key not in _JITTED_EQ:
        jax, jnp, _ = _get_jax()

        @jax.jit
        def rls_scan(
            x_input,
            training_padded,
            constellation,
            w_init,
            P_init,
            lam,
            n_train,
            leakage,
            n_update_halt,
        ):
            # Argument shapes and semantics
            # ------------------------------
            # x_input         : (C, N_pad)              complex64 - padded received samples
            # training_padded : (C, N_sym)               complex64 - reference symbols
            # constellation   : (M,)                    complex64 - slicer lookup table
            # w_init          : (C, C, num_taps)         complex64 - initial butterfly weights
            # P_init          : (C*num_taps, C*num_taps) complex64 - initial inverse
            #                    correlation matrix P = (1/delta) * I
            # lam             : scalar float32           - forgetting factor λ ∈ (0,1]
            # n_train         : scalar int32             - training/DD boundary (dynamic)
            # leakage         : scalar float32           - weight-decay coefficient γ ∈ [0,1):
            #                    W <- (1-γ)W + k⊗ē  each step; P update is unchanged.
            #                    Decays null-subspace weights without inflating P eigenvalues.
            # n_update_halt   : scalar int32             - freeze W and P beyond this index;
            #                    = n_sym - num_taps//2 to avoid zero-padding contamination
            #
            # lax.scan carry  : (W, P)
            #   W : (C, C, num_taps)         butterfly weight matrix
            #   P : (C*num_taps, C*num_taps) inverse input correlation matrix
            # lax.scan xs     : jnp.arange(N_sym)
            # lax.scan output : y_hat   (N_sym, C)              equalized symbols
            #                   errors  (N_sym, C)              complex errors d - y
            #                   w_hist  (N_sym, C, C, num_taps) weight snapshots
            #
            # Per-step equations (idx is the current symbol index):
            #   X_wins = [x_input[c, idx*stride : idx*stride+num_taps] for c]  (C, T)
            #   x_bar  = X_wins.flatten()                                        (C*T,)
            #   y      = einsum('ijt,jt->i', conj(W), X_wins)                   (C,)
            #   d      = training[:, idx] if idx < n_train else slicer(y)        (C,)
            #   e      = d - y                                                   (C,)
            #   Px     = P @ x_bar                                               (C*T,)
            #   k      = Px / (λ + x_bar^H @ Px)                                (C*T,) Kalman gain
            #   W     <- (1-γ)W + k ⊗ conj(e)              (if idx < n_update_halt)
            #   P      = (P - outer(k, x_bar^H P)) / λ     (if idx < n_update_halt)

            _P = jax.lax.Precision.HIGHEST

            def step(carry, idx):
                W, P = carry
                sample_idx = idx * stride

                X_wins = jax.lax.dynamic_slice(
                    x_input, (0, sample_idx), (num_ch, num_taps)
                )

                y = jnp.einsum("ijt,jt->i", jnp.conj(W), X_wins, precision=_P)

                def slicer(ch_y):
                    if sq_side > 0:  # static branch at trace time
                        ir = jnp.clip(
                            jnp.round((ch_y.real - sq_lev_min) / sq_d_grid).astype(
                                jnp.int32
                            ),
                            0,
                            sq_side - 1,
                        )
                        ii = jnp.clip(
                            jnp.round((ch_y.imag - sq_lev_min) / sq_d_grid).astype(
                                jnp.int32
                            ),
                            0,
                            sq_side - 1,
                        )
                        nr = sq_lev_min + ir.astype(jnp.float32) * jnp.float32(
                            sq_d_grid
                        )
                        ni = sq_lev_min + ii.astype(jnp.float32) * jnp.float32(
                            sq_d_grid
                        )
                        return jax.lax.complex(nr, ni)
                    else:
                        return constellation[
                            jnp.argmin(jnp.abs(ch_y - constellation) ** 2)
                        ]

                dd = jax.vmap(slicer)(y)
                d = jnp.where(idx < n_train, training_padded[:, idx], dd)
                e = d - y

                x_bar = X_wins.flatten()  # (C * num_taps,)

                Px = jnp.matmul(P, x_bar, precision=_P)
                denom = lam + jnp.real(jnp.dot(jnp.conj(x_bar), Px, precision=_P))
                k = Px / denom

                def w_update(w_row, err_val):
                    w_flat = w_row.flatten()
                    # Weight decay: suppress null-subspace taps exponentially.
                    # Adding γI directly to P would inflate P eigenvalues, making
                    # R_xx more singular and worsening AWGN amplification.
                    w_flat_new = (1.0 - leakage) * w_flat + k * jnp.conj(err_val)
                    return w_flat_new.reshape(num_ch, num_taps)

                W_upd = jax.vmap(w_update)(W, e).astype(jnp.complex64)
                # Riccati: exploit Hermitian symmetry P = P^H so (x^H P)[j] = conj((Px)[j]).
                # outer(k, conj(Px)) == k ⊗ (x^H P); reuses Px to avoid a second O(N²) mat-vec.
                P_upd = (P - jnp.outer(k, jnp.conj(Px))) / lam
                # Hermitian re-symmetrization: P <- (P + Pᴴ)/2
                # Prevents asymmetry drift from the 1/λ amplification.
                P_upd = 0.5 * (P_upd + jnp.conj(P_upd).T)

                # Early halt: freeze W and P once the sliding window begins
                # overlapping the right zero-padding (last num_taps//2 symbols).
                # The forward pass (y) continues so all output symbols are produced.
                update_ok = idx < n_update_halt
                W_new = jnp.where(update_ok, W_upd, W)
                P_new = jnp.where(update_ok, P_upd, P)

                return (W_new, P_new), (y, e, W_new)

            n_sym = training_padded.shape[1]
            (W_final, _), (y_hat, errors, w_hist) = jax.lax.scan(
                step, (w_init, P_init), jnp.arange(n_sym)
            )
            return y_hat, errors, W_final, w_hist

        _JITTED_EQ[key] = rls_scan
    return _JITTED_EQ[key]


def _get_jax_lms_cpr(
    num_taps,
    stride,
    const_size,
    num_ch,
    cpr_type,
    bps_n,
    bps_block_size,
    bps_joint_channels,
    cs_history_len,
    symmetry=4,
    sq_side=0,
    sq_lev_min=0.0,
    sq_d_grid=1.0,
):
    """JIT-compile and cache the LMS+CPR butterfly scan.

    All CPR parameters are static closure variables (baked into the XLA graph
    at trace time).  A separate cache entry is created for each distinct
    combination of these parameters.

    cpr_type : "pll" or "bps"
    bps_n    : number of BPS test phases (ignored for cpr_type="pll")
    cs_history_len : int - circular buffer depth for cycle-slip correction
    sq_side, sq_lev_min, sq_d_grid : O(1) square-QAM slicer parameters
    """
    key = (
        "lms_cpr",
        num_taps,
        stride,
        const_size,
        num_ch,
        cpr_type,
        bps_n,
        bps_block_size,
        bps_joint_channels,
        cs_history_len,
        int(symmetry),
        sq_side,
        float(sq_lev_min),
        float(sq_d_grid),
    )
    if key not in _JITTED_EQ:
        jax, jnp, _ = _get_jax()

        H = cs_history_len
        KB = bps_block_size  # static closure: BPS window length

        import math as _math

        _quantum_static = jnp.float64(2.0 * _math.pi / symmetry)

        _PREC = jax.lax.Precision.HIGHEST

        @jax.jit
        def lms_cpr_scan(
            x_input,
            training_padded,
            constellation,
            bps_phases_neg,
            bps_angles,
            w_init,
            step_size,
            n_train,
            pll_mu,
            pll_beta,
            cs_threshold,
            cs_enabled,
            pll_phi_init,
            pll_freq_init,
            bps_buf_init,
            bps_buf_ptr_init,
            bps_prev4_init,
            cs_buf_x_init,
            cs_buf_y_init,
            cs_buf_ptr_init,
        ):
            # x_input         : (C, N_pad)       complex64
            # training_padded : (C, N_sym)        complex64
            # constellation   : (M,)              complex64
            # bps_phases_neg  : (B,)              complex64  exp(-j*theta_k)
            # bps_angles      : (B,)              float32    theta_k
            # w_init          : (C, C, T)         complex64
            # step_size       : scalar float32
            # n_train         : scalar int32
            # pll_mu, pll_beta: scalar float64
            # cs_threshold    : scalar float64
            # cs_enabled      : scalar bool
            # pll_phi_init    : (C,) float64  - warm-start PLL integrator (zeros -> cold)
            # pll_freq_init   : (C,) float64  - warm-start PLL frequency  (zeros -> cold)
            # bps_buf_init    : (KB, C) complex64 - warm-start BPS buffer (zeros -> cold)
            # bps_buf_ptr_init: scalar int32  - warm-start BPS buffer pointer
            # bps_prev4_init  : (C,) float64  - warm-start 4-fold unwrap state
            # cs_buf_x_init   : (C, H) float64 - warm-start cycle-slip symbol index
            # cs_buf_y_init   : (C, H) float64 - warm-start cycle-slip phase value
            # cs_buf_ptr_init : (C,)   int32   - warm-start cycle-slip write pointer
            #
            # lax.scan carry:
            #   W             : (C, C, T)         complex64
            #   pll_phi       : (C,)              float64
            #   pll_freq      : (C,)              float64
            #   bps_buf       : (KB, C)           complex64  - y_raw circular buffer
            #   bps_buf_ptr   : scalar int32
            #   bps_prev4     : (C,)              float64   - causal 4-fold unwrap state
            #   cs_buf_x      : (C, H)            float64  - symbol index
            #   cs_buf_y      : (C, H)            float64  - phase value
            #   cs_buf_ptr    : (C,)              int32    - write pointer
            #   bps_d2_slots  : (B, KB, C)        float64  - per-slot BPS sq-dist
            #   bps_metric    : (B, C)            float64  - running sum of slots

            def _bps_d2(rotated):
                # Min squared distance of `rotated` (..., complex) to the
                # constellation, returned float64.  Shared by the per-symbol
                # new-slot update and the warm-start reconstruction so the
                # distance formula has a single source of truth.
                if sq_side > 0:  # static branch at trace time
                    r_idx = jnp.clip(
                        jnp.round((rotated.real - sq_lev_min) / sq_d_grid).astype(
                            jnp.int32
                        ),
                        0,
                        sq_side - 1,
                    )
                    i_idx = jnp.clip(
                        jnp.round((rotated.imag - sq_lev_min) / sq_d_grid).astype(
                            jnp.int32
                        ),
                        0,
                        sq_side - 1,
                    )
                    r_near = sq_lev_min + r_idx.astype(jnp.float32) * jnp.float32(
                        sq_d_grid
                    )
                    i_near = sq_lev_min + i_idx.astype(jnp.float32) * jnp.float32(
                        sq_d_grid
                    )
                    d2 = (rotated.real - r_near) ** 2 + (rotated.imag - i_near) ** 2
                    return d2.astype(jnp.float64)
                return jnp.min(
                    jnp.abs(rotated[..., None] - constellation) ** 2,
                    axis=-1,
                ).astype(jnp.float64)

            def step(carry, idx):
                (
                    W,
                    pll_phi,
                    pll_freq,
                    bps_buf,
                    bps_buf_ptr,
                    bps_prev4,
                    bps_d2_slots,
                    bps_metric,
                    cs_buf_x,
                    cs_buf_y,
                    cs_buf_ptr,
                ) = carry
                sample_idx = idx * stride

                X_wins = jax.lax.dynamic_slice(
                    x_input, (0, sample_idx), (num_ch, num_taps)
                )  # (C, T)

                y_raw = jnp.einsum(
                    "ijt,jt->i",
                    jnp.conj(W).astype(jnp.complex128),
                    X_wins.astype(jnp.complex128),
                    precision=_PREC,
                ).astype(
                    jnp.complex64
                )  # float64 accumulation -> complex64, matches Numba

                # -- Phase estimation (static branch at trace time) --
                if cpr_type == "pll":
                    phi_hat = pll_phi  # (C,)
                    bps_buf_new = bps_buf
                    bps_buf_ptr_new = bps_buf_ptr
                    bps_prev4_new = bps_prev4
                    bps_d2_slots_new = bps_d2_slots
                    bps_metric_new = bps_metric
                else:
                    # Fill BPS circular buffer with current y_raw
                    slot = jnp.int32(bps_buf_ptr % KB)
                    bps_buf_new = jax.lax.dynamic_update_slice(
                        bps_buf,
                        y_raw[None, :],  # (1, C) - update one row
                        (slot, jnp.int32(0)),
                    )  # broadcast doesn't work for transpose, use (C, KB) layout instead
                    bps_buf_ptr_new = bps_buf_ptr + 1

                    # Incremental running sum (O(B*M)/symbol vs O(B*KB*M) for a
                    # full re-sum, matching the Numba bps_running_sum): score the
                    # NEW slot only, add it to the metric and subtract the evicted
                    # slot's stored distance.  Not-yet-filled slots hold 0 (set at
                    # warm-start reconstruction), so the subtraction is exact and
                    # reproduces the old fill-mask semantics.
                    rotated_new = bps_phases_neg[:, None] * y_raw[None, :]  # (B, C)
                    d2_new = _bps_d2(rotated_new)  # (B, C) float64
                    old_d2 = jax.lax.dynamic_slice_in_dim(
                        bps_d2_slots, slot, 1, axis=1
                    )[:, 0, :]  # (B, C)
                    metric = bps_metric + d2_new - old_d2  # (B, C) float64
                    bps_d2_slots_new = jax.lax.dynamic_update_slice_in_dim(
                        bps_d2_slots, d2_new[:, None, :], slot, axis=1
                    )
                    bps_metric_new = metric

                    if bps_joint_channels:
                        # Sum over channels too -> (B,); broadcast winner to all C
                        best_k = jnp.argmin(metric.sum(axis=-1))  # scalar
                        phi_raw = jnp.full(num_ch, bps_angles[best_k])  # (C,)
                    else:
                        best_k = jnp.argmin(metric, axis=0)  # (C,)
                        phi_raw = bps_angles[best_k]  # (C,)

                    # Causal 4-fold phase unwrap: equivalent to np.unwrap(phi*4)/4
                    raw4 = phi_raw.astype(jnp.float64) * jnp.float64(4.0)
                    diff4 = raw4 - bps_prev4
                    two_pi = jnp.float64(2.0 * 3.141592653589793)
                    diff4 = diff4 - jnp.round(diff4 / two_pi) * two_pi
                    bps_prev4_new = bps_prev4 + diff4
                    phi_hat = bps_prev4_new / jnp.float64(4.0)  # (C,) unwrapped float64

                # -- Cycle-slip correction ----------------------------
                def correct_slip_ch(phi_h, buf_x_ch, buf_y_ch, ptr_ch):
                    # buf_x_ch is retained for call-site compat but unused here.
                    fill_cs = jnp.minimum(ptr_ch, H)
                    y_b = phi_h

                    mask = jnp.arange(H) < fill_cs
                    n_f = fill_cs.astype(jnp.float64)

                    # Sx and Sxx are exact closed-form constants - no cancellation.
                    Sx = n_f * (n_f - jnp.float64(1.0)) / jnp.float64(2.0)
                    Sxx = (
                        n_f
                        * (n_f - jnp.float64(1.0))
                        * (jnp.float64(2.0) * n_f - jnp.float64(1.0))
                        / jnp.float64(6.0)
                    )
                    denom = n_f * Sxx - Sx * Sx

                    # Sxy uses relative positions [0, fill_cs-1].  In the circular
                    # buffer the oldest entry is at slot ptr_ch%H and has position 0;
                    # slot j has relative position (j - ptr_ch%H + H) % H for a full
                    # window, or just j for a partial window (entries written 0..fill-1).
                    oldest_slot = ptr_ch % H
                    slots = jnp.arange(H)
                    rel_pos_full = (slots - oldest_slot + H) % H
                    rel_pos_partial = slots
                    rel_pos = jnp.where(
                        fill_cs >= H, rel_pos_full, rel_pos_partial
                    ).astype(jnp.float64)
                    Sy = jnp.where(mask, buf_y_ch, jnp.float64(0.0)).sum()
                    Sxy = jnp.where(mask, rel_pos * buf_y_ch, jnp.float64(0.0)).sum()

                    # Prediction target: one step past the newest entry (relative pos = fill_cs).
                    x_pred = n_f

                    safe_denom = jnp.where(
                        jnp.abs(denom) > 1e-20, denom, jnp.float64(1.0)
                    )
                    slope = jnp.where(
                        fill_cs >= 10,
                        (n_f * Sxy - Sx * Sy) / safe_denom,
                        jnp.float64(0.0),
                    )
                    intercept = jnp.where(
                        fill_cs >= 10,
                        (Sy - slope * Sx) / jnp.maximum(n_f, jnp.float64(1.0)),
                        Sy / jnp.maximum(n_f, jnp.float64(1.0)),
                    )
                    phi_exp_lin = slope * x_pred + intercept

                    last_pos = (ptr_ch - 1 + H) % H
                    phi_last = jax.lax.dynamic_index_in_dim(
                        buf_y_ch, last_pos, keepdims=False
                    )
                    phi_expected = jnp.where(fill_cs >= 10, phi_exp_lin, phi_last)
                    phi_expected = jnp.where(fill_cs == 0, y_b, phi_expected)

                    diff = y_b - phi_expected
                    k_slip = jnp.round(diff / _quantum_static)
                    should_correct = (
                        cs_enabled & (jnp.abs(diff) > cs_threshold) & (k_slip != 0)
                    )
                    phi_corr = jnp.where(
                        should_correct, y_b - k_slip * _quantum_static, y_b
                    )

                    write_pos = ptr_ch % H
                    buf_y_new = jax.lax.dynamic_update_slice(
                        buf_y_ch, phi_corr[None], [write_pos]
                    )
                    ptr_new = ptr_ch + 1
                    return phi_corr, buf_x_ch, buf_y_new, ptr_new

                phi_corr, cs_buf_x_new, cs_buf_y_new, cs_buf_ptr_new = jax.vmap(
                    correct_slip_ch
                )(phi_hat, cs_buf_x, cs_buf_y, cs_buf_ptr)

                # Wrap to [-π, π] before casting to float32 for fast GPU exp
                _two_pi_f64 = jnp.float64(2.0 * 3.141592653589793)
                phi_wrapped = phi_corr - jnp.round(phi_corr / _two_pi_f64) * _two_pi_f64
                _phi32 = phi_wrapped.astype(jnp.float32)
                phasor = jnp.exp(_phi32 * jnp.array(-1j, dtype=jnp.complex64))
                y_fin = y_raw * phasor  # (C,)

                def slicer(ch_y):
                    if sq_side > 0:  # static branch at trace time
                        ir = jnp.clip(
                            jnp.round((ch_y.real - sq_lev_min) / sq_d_grid).astype(
                                jnp.int32
                            ),
                            0,
                            sq_side - 1,
                        )
                        ii = jnp.clip(
                            jnp.round((ch_y.imag - sq_lev_min) / sq_d_grid).astype(
                                jnp.int32
                            ),
                            0,
                            sq_side - 1,
                        )
                        nr = sq_lev_min + ir.astype(jnp.float32) * jnp.float32(
                            sq_d_grid
                        )
                        ni = sq_lev_min + ii.astype(jnp.float32) * jnp.float32(
                            sq_d_grid
                        )
                        return jax.lax.complex(nr, ni)
                    else:
                        return constellation[
                            jnp.argmin(jnp.abs(ch_y - constellation) ** 2)
                        ]

                dd = jax.vmap(slicer)(y_fin)
                d = jnp.where(idx < n_train, training_padded[:, idx], dd)
                e_clean = d - y_fin  # (C,)

                if cpr_type == "pll":
                    e_ph = y_fin.imag * d.real - y_fin.real * d.imag  # (C,)
                    if bps_joint_channels:  # static branch - shared LO
                        e_ph = jnp.full(num_ch, e_ph.mean())
                    pll_phi_new = pll_phi + pll_mu * e_ph + pll_freq
                    pll_freq_new = pll_freq + pll_beta * e_ph
                else:
                    pll_phi_new = pll_phi
                    pll_freq_new = pll_freq

                phasor_inv = jnp.exp(_phi32 * jnp.array(1j, dtype=jnp.complex64))
                e_eq = e_clean * phasor_inv  # (C,)

                W_new = W + step_size * jnp.einsum("i,jt->ijt", jnp.conj(e_eq), X_wins)

                carry_new = (
                    W_new,
                    pll_phi_new,
                    pll_freq_new,
                    bps_buf_new,
                    bps_buf_ptr_new,
                    bps_prev4_new,
                    bps_d2_slots_new,
                    bps_metric_new,
                    cs_buf_x_new,
                    cs_buf_y_new,
                    cs_buf_ptr_new,
                )
                return carry_new, (y_fin, e_clean, W_new, phi_corr)

            n_sym = training_padded.shape[1]
            # Warm-start: reconstruct per-slot BPS distances + running metric from
            # the buffer (single source of truth via _bps_d2).  Cold start (zero
            # buffer, ptr=0) yields an all-zero metric through the fill mask; the
            # mask also stores 0 for unfilled slots so the in-loop eviction stays
            # exact.
            _fill0 = jnp.minimum(bps_buf_ptr_init, KB)
            _rot0 = bps_phases_neg[:, None, None] * bps_buf_init[None, :, :]  # (B,KB,C)
            _mask0 = jnp.arange(KB)[None, :, None] < _fill0
            bps_d2_slots_init = jnp.where(_mask0, _bps_d2(_rot0), jnp.float64(0.0))
            bps_metric_init = bps_d2_slots_init.sum(axis=1)  # (B, C) float64
            init_carry = (
                w_init,
                pll_phi_init,
                pll_freq_init,
                bps_buf_init,
                bps_buf_ptr_init,
                bps_prev4_init,
                bps_d2_slots_init,
                bps_metric_init,
                cs_buf_x_init,
                cs_buf_y_init,
                cs_buf_ptr_init,
            )
            (
                (
                    W_final,
                    pll_phi_f,
                    pll_freq_f,
                    bps_buf_f,
                    bps_buf_ptr_f,
                    bps_prev4_f,
                    _bps_d2_slots_f,
                    _bps_metric_f,
                    cs_buf_x_f,
                    cs_buf_y_f,
                    cs_buf_ptr_f,
                ),
                (y_hat, errors, w_hist, phi_traj),
            ) = jax.lax.scan(step, init_carry, jnp.arange(n_sym))
            return (
                y_hat,
                errors,
                W_final,
                w_hist,
                phi_traj,
                pll_phi_f,
                pll_freq_f,
                bps_buf_f,
                bps_buf_ptr_f,
                bps_prev4_f,
                cs_buf_x_f,
                cs_buf_y_f,
                cs_buf_ptr_f,
            )

        _JITTED_EQ[key] = lms_cpr_scan
    return _JITTED_EQ[key]


def _get_jax_rls_cpr(
    num_taps,
    stride,
    const_size,
    num_ch,
    cpr_type,
    bps_n,
    bps_block_size,
    bps_joint_channels,
    cs_history_len,
    symmetry=4,
    sq_side=0,
    sq_lev_min=0.0,
    sq_d_grid=1.0,
):
    """JIT-compile and cache the RLS+CPR butterfly scan.

    Combines the Leaky-RLS Riccati update with an inline CPR tracker.
    Static parameters are identical to ``_get_jax_lms_cpr``.
    sq_side, sq_lev_min, sq_d_grid : O(1) square-QAM slicer parameters.
    """
    key = (
        "rls_cpr",
        num_taps,
        stride,
        const_size,
        num_ch,
        cpr_type,
        bps_n,
        bps_block_size,
        bps_joint_channels,
        cs_history_len,
        int(symmetry),
        sq_side,
        float(sq_lev_min),
        float(sq_d_grid),
    )
    if key not in _JITTED_EQ:
        jax, jnp, _ = _get_jax()

        H = cs_history_len
        KB = bps_block_size
        import math as _math

        _quantum_static = jnp.float64(2.0 * _math.pi / symmetry)

        _PREC = jax.lax.Precision.HIGHEST

        @jax.jit
        def rls_cpr_scan(
            x_input,
            training_padded,
            constellation,
            bps_phases_neg,
            bps_angles,
            w_init,
            P_init,
            lam,
            n_train,
            leakage,
            n_update_halt,
            pll_mu,
            pll_beta,
            cs_threshold,
            cs_enabled,
            pll_phi_init,
            pll_freq_init,
            bps_buf_init,
            bps_buf_ptr_init,
            bps_prev4_init,
            cs_buf_x_init,
            cs_buf_y_init,
            cs_buf_ptr_init,
        ):
            def _bps_d2(rotated):
                # Min squared distance of `rotated` (..., complex) to the
                # constellation, returned float64.  Shared by the per-symbol
                # new-slot update and the warm-start reconstruction so the
                # distance formula has a single source of truth.
                if sq_side > 0:  # static branch at trace time
                    r_idx = jnp.clip(
                        jnp.round((rotated.real - sq_lev_min) / sq_d_grid).astype(
                            jnp.int32
                        ),
                        0,
                        sq_side - 1,
                    )
                    i_idx = jnp.clip(
                        jnp.round((rotated.imag - sq_lev_min) / sq_d_grid).astype(
                            jnp.int32
                        ),
                        0,
                        sq_side - 1,
                    )
                    r_near = sq_lev_min + r_idx.astype(jnp.float32) * jnp.float32(
                        sq_d_grid
                    )
                    i_near = sq_lev_min + i_idx.astype(jnp.float32) * jnp.float32(
                        sq_d_grid
                    )
                    d2 = (rotated.real - r_near) ** 2 + (rotated.imag - i_near) ** 2
                    return d2.astype(jnp.float64)
                return jnp.min(
                    jnp.abs(rotated[..., None] - constellation) ** 2,
                    axis=-1,
                ).astype(jnp.float64)

            def step(carry, idx):
                (
                    W,
                    P,
                    pll_phi,
                    pll_freq,
                    bps_buf,
                    bps_buf_ptr,
                    bps_prev4,
                    bps_d2_slots,
                    bps_metric,
                    cs_buf_x,
                    cs_buf_y,
                    cs_buf_ptr,
                ) = carry
                sample_idx = idx * stride

                X_wins = jax.lax.dynamic_slice(
                    x_input, (0, sample_idx), (num_ch, num_taps)
                )
                y_raw = jnp.einsum(
                    "ijt,jt->i",
                    jnp.conj(W).astype(jnp.complex128),
                    X_wins.astype(jnp.complex128),
                    precision=_PREC,
                ).astype(
                    jnp.complex64
                )  # float64 accumulation -> complex64, matches Numba

                if cpr_type == "pll":
                    phi_hat = pll_phi
                    bps_buf_new = bps_buf
                    bps_buf_ptr_new = bps_buf_ptr
                    bps_prev4_new = bps_prev4
                    bps_d2_slots_new = bps_d2_slots
                    bps_metric_new = bps_metric
                else:
                    slot = jnp.int32(bps_buf_ptr % KB)
                    bps_buf_new = jax.lax.dynamic_update_slice(
                        bps_buf,
                        y_raw[None, :],
                        (slot, jnp.int32(0)),
                    )
                    bps_buf_ptr_new = bps_buf_ptr + 1

                    # Incremental running sum (O(B*M)/symbol vs O(B*KB*M) for a
                    # full re-sum, matching the Numba bps_running_sum): score the
                    # NEW slot only, add it to the metric and subtract the evicted
                    # slot's stored distance.  Not-yet-filled slots hold 0 (set at
                    # warm-start reconstruction), so the subtraction is exact and
                    # reproduces the old fill-mask semantics.
                    rotated_new = bps_phases_neg[:, None] * y_raw[None, :]  # (B, C)
                    d2_new = _bps_d2(rotated_new)  # (B, C) float64
                    old_d2 = jax.lax.dynamic_slice_in_dim(
                        bps_d2_slots, slot, 1, axis=1
                    )[:, 0, :]  # (B, C)
                    metric = bps_metric + d2_new - old_d2  # (B, C) float64
                    bps_d2_slots_new = jax.lax.dynamic_update_slice_in_dim(
                        bps_d2_slots, d2_new[:, None, :], slot, axis=1
                    )
                    bps_metric_new = metric

                    if bps_joint_channels:
                        best_k = jnp.argmin(metric.sum(axis=-1))
                        phi_raw = jnp.full(num_ch, bps_angles[best_k])
                    else:
                        best_k = jnp.argmin(metric, axis=0)
                        phi_raw = bps_angles[best_k]

                    # Causal 4-fold phase unwrap
                    raw4 = phi_raw.astype(jnp.float64) * jnp.float64(4.0)
                    diff4 = raw4 - bps_prev4
                    two_pi = jnp.float64(2.0 * 3.141592653589793)
                    diff4 = diff4 - jnp.round(diff4 / two_pi) * two_pi
                    bps_prev4_new = bps_prev4 + diff4
                    phi_hat = bps_prev4_new / jnp.float64(4.0)

                def correct_slip_ch(phi_h, buf_x_ch, buf_y_ch, ptr_ch):
                    # buf_x_ch is retained for call-site compat but unused here.
                    fill_cs = jnp.minimum(ptr_ch, H)
                    y_b = phi_h
                    mask = jnp.arange(H) < fill_cs
                    n_f = fill_cs.astype(jnp.float64)

                    # Sx and Sxx are exact closed-form constants - no cancellation.
                    Sx = n_f * (n_f - jnp.float64(1.0)) / jnp.float64(2.0)
                    Sxx = (
                        n_f
                        * (n_f - jnp.float64(1.0))
                        * (jnp.float64(2.0) * n_f - jnp.float64(1.0))
                        / jnp.float64(6.0)
                    )
                    denom = n_f * Sxx - Sx * Sx

                    # Sxy uses relative positions [0, fill_cs-1] derived from the
                    # circular buffer layout (oldest slot = ptr_ch % H -> position 0).
                    oldest_slot = ptr_ch % H
                    slots = jnp.arange(H)
                    rel_pos_full = (slots - oldest_slot + H) % H
                    rel_pos = jnp.where(fill_cs >= H, rel_pos_full, slots).astype(
                        jnp.float64
                    )
                    Sy = jnp.where(mask, buf_y_ch, jnp.float64(0.0)).sum()
                    Sxy = jnp.where(mask, rel_pos * buf_y_ch, jnp.float64(0.0)).sum()

                    # Prediction target: one step past newest (relative pos = fill_cs).
                    x_pred = n_f

                    safe_denom = jnp.where(
                        jnp.abs(denom) > 1e-20, denom, jnp.float64(1.0)
                    )
                    slope = jnp.where(
                        fill_cs >= 10,
                        (n_f * Sxy - Sx * Sy) / safe_denom,
                        jnp.float64(0.0),
                    )
                    intercept = jnp.where(
                        fill_cs >= 10,
                        (Sy - slope * Sx) / jnp.maximum(n_f, jnp.float64(1.0)),
                        Sy / jnp.maximum(n_f, jnp.float64(1.0)),
                    )
                    phi_exp_lin = slope * x_pred + intercept
                    last_pos = (ptr_ch - 1 + H) % H
                    phi_last = jax.lax.dynamic_index_in_dim(
                        buf_y_ch, last_pos, keepdims=False
                    )
                    phi_expected = jnp.where(fill_cs >= 10, phi_exp_lin, phi_last)
                    phi_expected = jnp.where(fill_cs == 0, y_b, phi_expected)
                    diff = y_b - phi_expected
                    k_slip = jnp.round(diff / _quantum_static)
                    should_correct = (
                        cs_enabled & (jnp.abs(diff) > cs_threshold) & (k_slip != 0)
                    )
                    phi_corr = jnp.where(
                        should_correct, y_b - k_slip * _quantum_static, y_b
                    )
                    write_pos = ptr_ch % H
                    buf_y_new = jax.lax.dynamic_update_slice(
                        buf_y_ch, phi_corr[None], [write_pos]
                    )
                    ptr_new = ptr_ch + 1
                    return phi_corr, buf_x_ch, buf_y_new, ptr_new

                phi_corr, cs_buf_x_new, cs_buf_y_new, cs_buf_ptr_new = jax.vmap(
                    correct_slip_ch
                )(phi_hat, cs_buf_x, cs_buf_y, cs_buf_ptr)

                # Wrap to [-π, π] before casting to float32 for fast GPU exp
                _two_pi_f64 = jnp.float64(2.0 * 3.141592653589793)
                phi_wrapped = phi_corr - jnp.round(phi_corr / _two_pi_f64) * _two_pi_f64
                _phi32 = phi_wrapped.astype(jnp.float32)
                phasor = jnp.exp(_phi32 * jnp.array(-1j, dtype=jnp.complex64))
                y_fin = y_raw * phasor

                def slicer(ch_y):
                    if sq_side > 0:  # static branch at trace time
                        ir = jnp.clip(
                            jnp.round((ch_y.real - sq_lev_min) / sq_d_grid).astype(
                                jnp.int32
                            ),
                            0,
                            sq_side - 1,
                        )
                        ii = jnp.clip(
                            jnp.round((ch_y.imag - sq_lev_min) / sq_d_grid).astype(
                                jnp.int32
                            ),
                            0,
                            sq_side - 1,
                        )
                        nr = sq_lev_min + ir.astype(jnp.float32) * jnp.float32(
                            sq_d_grid
                        )
                        ni = sq_lev_min + ii.astype(jnp.float32) * jnp.float32(
                            sq_d_grid
                        )
                        return jax.lax.complex(nr, ni)
                    else:
                        return constellation[
                            jnp.argmin(jnp.abs(ch_y - constellation) ** 2)
                        ]

                dd = jax.vmap(slicer)(y_fin)
                d = jnp.where(idx < n_train, training_padded[:, idx], dd)
                e_clean = d - y_fin

                if cpr_type == "pll":
                    e_ph = y_fin.imag * d.real - y_fin.real * d.imag
                    if bps_joint_channels:  # static branch - shared LO
                        e_ph = jnp.full(num_ch, e_ph.mean())
                    pll_phi_new = pll_phi + pll_mu * e_ph + pll_freq
                    pll_freq_new = pll_freq + pll_beta * e_ph
                else:
                    pll_phi_new = pll_phi
                    pll_freq_new = pll_freq

                phasor_inv = jnp.exp(_phi32 * jnp.array(1j, dtype=jnp.complex64))
                e_eq = e_clean * phasor_inv

                x_bar = X_wins.flatten()
                Px = jnp.matmul(P, x_bar, precision=_PREC)
                denom_k = lam + jnp.real(jnp.dot(jnp.conj(x_bar), Px, precision=_PREC))
                k_gain = Px / denom_k

                def w_update(w_row, err_val):
                    w_flat = w_row.flatten()
                    w_flat_new = (1.0 - leakage) * w_flat + k_gain * jnp.conj(err_val)
                    return w_flat_new.reshape(num_ch, num_taps)

                W_upd = jax.vmap(w_update)(W, e_eq).astype(jnp.complex64)
                P_upd = (P - jnp.outer(k_gain, jnp.conj(Px))) / lam
                P_upd = 0.5 * (P_upd + jnp.conj(P_upd).T)

                update_ok = idx < n_update_halt
                W_new = jnp.where(update_ok, W_upd, W)
                P_new = jnp.where(update_ok, P_upd, P)

                carry_new = (
                    W_new,
                    P_new,
                    pll_phi_new,
                    pll_freq_new,
                    bps_buf_new,
                    bps_buf_ptr_new,
                    bps_prev4_new,
                    bps_d2_slots_new,
                    bps_metric_new,
                    cs_buf_x_new,
                    cs_buf_y_new,
                    cs_buf_ptr_new,
                )
                return carry_new, (y_fin, e_clean, W_new, phi_corr)

            n_sym = training_padded.shape[1]
            # Warm-start: reconstruct per-slot BPS distances + running metric from
            # the buffer (single source of truth via _bps_d2).  Cold start (zero
            # buffer, ptr=0) yields an all-zero metric through the fill mask; the
            # mask also stores 0 for unfilled slots so the in-loop eviction stays
            # exact.
            _fill0 = jnp.minimum(bps_buf_ptr_init, KB)
            _rot0 = bps_phases_neg[:, None, None] * bps_buf_init[None, :, :]  # (B,KB,C)
            _mask0 = jnp.arange(KB)[None, :, None] < _fill0
            bps_d2_slots_init = jnp.where(_mask0, _bps_d2(_rot0), jnp.float64(0.0))
            bps_metric_init = bps_d2_slots_init.sum(axis=1)  # (B, C) float64
            init_carry = (
                w_init,
                P_init,
                pll_phi_init,
                pll_freq_init,
                bps_buf_init,
                bps_buf_ptr_init,
                bps_prev4_init,
                bps_d2_slots_init,
                bps_metric_init,
                cs_buf_x_init,
                cs_buf_y_init,
                cs_buf_ptr_init,
            )
            (
                (
                    W_final,
                    _,
                    pll_phi_f,
                    pll_freq_f,
                    bps_buf_f,
                    bps_buf_ptr_f,
                    bps_prev4_f,
                    _bps_d2_slots_f,
                    _bps_metric_f,
                    cs_buf_x_f,
                    cs_buf_y_f,
                    cs_buf_ptr_f,
                ),
                (y_hat, errors, w_hist, phi_traj),
            ) = jax.lax.scan(step, init_carry, jnp.arange(n_sym))
            return (
                y_hat,
                errors,
                W_final,
                w_hist,
                phi_traj,
                pll_phi_f,
                pll_freq_f,
                bps_buf_f,
                bps_buf_ptr_f,
                bps_prev4_f,
                cs_buf_x_f,
                cs_buf_y_f,
                cs_buf_ptr_f,
            )

        _JITTED_EQ[key] = rls_cpr_scan
    return _JITTED_EQ[key]


def _get_jax_cma(num_taps, stride, num_ch):
    """JIT-compile and cache the sample-by-sample CMA butterfly scan.

    Static closure variables: num_taps, stride, num_ch (same as LMS/RLS).
    No constellation required - CMA is a blind algorithm.

    Returns
    -------
    cma_scan : JIT-compiled callable
        See the inner function for the call signature.
    """
    key = ("cma", num_taps, stride, num_ch)
    if key not in _JITTED_EQ:
        jax, jnp, _ = _get_jax()

        # n_sym (arg 4) is static: lax.scan requires a compile-time iteration
        # count.  JAX's bounded LRU cache retraces when n_sym changes, keeping
        # _JITTED_EQ from growing without bound.
        @functools.partial(jax.jit, static_argnums=(4,))
        def cma_scan(x_input, w_init, step_size, r2, n_sym):
            # Argument shapes and semantics
            # ------------------------------
            # x_input   : (C, N_pad)       complex64 - padded received samples
            # w_init    : (C, C, num_taps) complex64 - initial butterfly weights
            # step_size : scalar float32   - fixed gradient step μ (no NLMS; non-convex surface)
            # r2        : scalar float32   - Godard dispersion radius R² = E[|s|⁴] / E[|s|²]
            # n_sym     : int (static)     - total symbol count; fixes scan iteration count
            #
            # lax.scan carry  : W  (C, C, num_taps)
            # lax.scan xs     : jnp.arange(n_sym)
            # lax.scan output : y_hat   (n_sym, C)
            #                   errors  (n_sym, C)   Godard errors y*(|y|²-R²)
            #                   w_hist  (n_sym, C, C, num_taps)
            #
            # Per-step gradient descent (Godard criterion):
            #   y = einsum('ijt,jt->i', conj(W), X_wins)            (C,)
            #   e = y * (real(y * conj(y)) - R²)                    (C,)  CMA error
            #   W -= μ * einsum('i,jt->ijt', conj(e), X_wins)               gradient step
            # Note: real() is required to prevent imaginary leakage from
            # floating-point noise in |y|² from causing parasitic phase rotation.
            _P = jax.lax.Precision.HIGHEST

            def step(W, idx):
                sample_idx = idx * stride

                X_wins = jax.lax.dynamic_slice(
                    x_input, (0, sample_idx), (num_ch, num_taps)
                )
                y = jnp.einsum("ijt,jt->i", jnp.conj(W), X_wins, precision=_P)

                # CMA error: e_i = y_i * (|y_i|^2 - R2)
                # jnp.real enforces strict real-valued modulus: floating-point noise
                # in y*conj(y) would otherwise inject imaginary components, causing
                # a parasitic phase rotation into the gradient via multiplication by y.
                e = y * (jnp.real(y * jnp.conj(y)) - r2)

                W_new = W - step_size * jnp.einsum("i,jt->ijt", jnp.conj(e), X_wins)
                return W_new, (y, e, W_new)

            W_final, (y_hat, errors, w_hist) = jax.lax.scan(
                step, w_init, jnp.arange(n_sym)
            )
            return y_hat, errors, W_final, w_hist

        _JITTED_EQ[key] = cma_scan
    return _JITTED_EQ[key]


def _get_jax_rde(num_taps, stride, num_radii, num_ch):
    """JIT-compile and cache the sample-by-sample RDE butterfly scan.

    RDE (Radius Directed Equalizer) extends CMA by replacing the single
    Godard radius with per-symbol radius selection from a precomputed set
    of unique constellation magnitudes.  This provides correct blind
    convergence on multi-ring constellations such as 16-QAM and 64-QAM.

    Static closure variables (baked into the compiled kernel):

    num_taps   : FIR filter length per polyphase arm.
    stride     : decimation factor (sps, typically 2 for T/2-spaced input).
    num_radii  : number of unique radii K - fixes the argmin shape at trace
                 time so XLA can compile without dynamic dispatch.
    num_ch     : MIMO butterfly width C.

    Returns
    -------
    rde_scan : JIT-compiled callable
        See the inner function for the call signature.
    """
    key = ("rde", num_taps, stride, num_radii, num_ch)
    if key not in _JITTED_EQ:
        jax, jnp, _ = _get_jax()

        @functools.partial(jax.jit, static_argnums=(4,))
        def rde_scan(x_input, w_init, step_size, radii, n_sym):
            # Argument shapes and semantics
            # ------------------------------
            # x_input   : (C, N_pad)       complex64 - padded received samples
            # w_init    : (C, C, num_taps) complex64 - initial butterfly weights
            # step_size : scalar float32   - fixed gradient step μ
            # radii     : (K,)             float32   - unique constellation radii, sorted
            # n_sym     : int (static)     - total symbol count; fixes scan iteration count
            #
            # lax.scan carry  : W  (C, C, num_taps)
            # lax.scan xs     : jnp.arange(n_sym)
            # lax.scan output : y_hat   (n_sym, C)
            #                   errors  (n_sym, C)   RDE errors y*(|y|²-R_d²)
            #                   w_hist  (n_sym, C, C, num_taps)
            #
            # Per-step RDE gradient:
            #   y     = einsum('ijt,jt->i', conj(W), X_wins)    (C,)
            #   abs_y = sqrt(real(y*conj(y)))                   (C,)
            #   R_d   = radii[argmin(|radii-abs_y|)]            (C,)  nearest ring
            #   e     = y * (real(y*conj(y)) - R_d²)            (C,)
            #   W    -= μ * einsum('i,jt->ijt', conj(e), X_wins)
            _P = jax.lax.Precision.HIGHEST

            def step(W, idx):
                sample_idx = idx * stride

                X_wins = jax.lax.dynamic_slice(
                    x_input, (0, sample_idx), (num_ch, num_taps)
                )
                y = jnp.einsum("ijt,jt->i", jnp.conj(W), X_wins, precision=_P)

                abs_y2 = jnp.real(y * jnp.conj(y))  # (C,) strict real |y|²
                abs_y = jnp.sqrt(abs_y2)  # (C,) |y|

                # (C, K) distance table; argmin over K gives nearest radius index
                dist = jnp.abs(abs_y[:, None] - radii[None, :])  # (C, K)
                rd = radii[jnp.argmin(dist, axis=1)]  # (C,)

                e = y * (abs_y2 - rd**2)

                W_new = W - step_size * jnp.einsum("i,jt->ijt", jnp.conj(e), X_wins)
                return W_new, (y, e, W_new)

            W_final, (y_hat, errors, w_hist) = jax.lax.scan(
                step, w_init, jnp.arange(n_sym)
            )
            return y_hat, errors, W_final, w_hist

        _JITTED_EQ[key] = rde_scan
    return _JITTED_EQ[key]


# -----------------------------------------------------------------------------
# BLOCK-UPDATE EQUALIZERS  (update_mode='block' - time-domain)
# -----------------------------------------------------------------------------
#
# Block-update LMS/CMA/RDE freeze the butterfly weights over a chunk of ``D``
# symbols, accumulate one aggregated gradient, and apply a single weight update
# per chunk.  This replaces the per-symbol weight dependency with one matrix
# product per chunk, which XLA (JAX) and CuPy execute on the wide units - the
# per-symbol ``lax.scan`` is launch-overhead-bound and cannot occupy a GPU.
#
# All variants share the **unified subtractive update**
#     W -= mu * sum_d conj(E[d]) ⊗ X[d]
# (matching the pilot-aided kernels): LMS uses ``E = y - d`` (algebraically
# identical to the additive ``W += mu·conj(d-y)·X``); CMA/RDE use the Godard /
# ring error; pilot positions invert to ``E = y - pilot_ref``.  Per CLAUDE.md
# the two einsums run at ``Precision.HIGHEST`` to force true FP32 (no TF32).


def _jax_block_core(
    jax,
    jnp,
    x_input,
    W_init,
    mu,
    error_fn,
    aux_xs,
    *,
    n_sym,
    D,
    stride,
    num_taps,
    num_ch,
):
    """Run a chunked block-update butterfly scan under an active jit trace.

    Builds the per-symbol strided regressor windows, reshapes into
    ``n_chunks`` chunks of ``D`` symbols (zero-padding the final partial
    chunk and masking its gradient contribution), and scans over chunks with
    frozen weights.  ``error_fn(Y_chunk, aux_chunk) -> E_chunk`` (both
    ``(D, num_ch)``) supplies the variant-specific error; the unified
    subtractive update is applied once per chunk.

    Returns ``(y_hat (n_sym, C), errors (n_sym, C), W_final (C, C, T))``.
    """
    _P = jax.lax.Precision.HIGHEST
    n_chunks = (n_sym + D - 1) // D
    n_pad = n_chunks * D

    # Per-symbol strided window gather -> (n_pad, C, num_taps).  Padded symbols
    # (>= n_sym) index past the real signal; clamp to stay in-bounds - their
    # rows are masked out of the gradient and trimmed from the outputs.
    sym_ids = jnp.arange(n_pad)
    tap_ids = jnp.arange(num_taps)
    idx = sym_ids[:, None] * stride + tap_ids[None, :]  # (n_pad, num_taps)
    idx = jnp.clip(idx, 0, x_input.shape[1] - 1)
    X_all = jnp.transpose(x_input[:, idx], (1, 0, 2))  # (n_pad, C, T)
    X_chunks = X_all.reshape(n_chunks, D, num_ch, num_taps)

    valid = (sym_ids < n_sym).reshape(n_chunks, D)  # (n_chunks, D)

    def body(W, xs):
        X_chunk, valid_chunk, aux = xs
        Y = jnp.einsum("ijt,djt->di", jnp.conj(W), X_chunk, precision=_P)  # (D,C)
        E = error_fn(Y, aux)  # (D, C)
        E_masked = jnp.where(valid_chunk[:, None], E, jnp.zeros_like(E))
        grad = jnp.einsum("di,djt->ijt", jnp.conj(E_masked), X_chunk, precision=_P)
        # Accumulate at HIGHEST precision but keep complex64 weight storage
        # (CLAUDE.md): an x64-enabled session can otherwise promote the error to
        # complex128 and break the scan carry dtype.
        W_new = (W - mu * grad).astype(W.dtype)
        return W_new, (Y, E)

    # unroll a few chunks per scan step to amortise XLA loop-control overhead -
    # the per-chunk matmuls are small, so loop dispatch is a real cost at the
    # default block_len.
    W_final, (Y_chunks, E_chunks) = jax.lax.scan(
        body, W_init, (X_chunks, valid, aux_xs), unroll=4
    )
    y_hat = Y_chunks.reshape(n_pad, num_ch)[:n_sym]
    errors = E_chunks.reshape(n_pad, num_ch)[:n_sym]
    return y_hat, errors, W_final


def _jax_block_pilot_aux(
    jax, jnp, n_sym, D, num_ch, n_chunks, has_pilots, pref, pmask, blind_fn
):
    """Build the per-chunk pilot aux and wrap a blind error with a pilot override.

    ``blind_fn(Y) -> E_blind`` (both ``(D, C)``).  When ``has_pilots`` is True,
    masked positions invert to the LMS residual ``Y - pref`` (subtractive
    update, matching the per-symbol pilot-aided kernels); otherwise the aux is
    a dummy per-chunk index and the blind error is used everywhere.

    Returns ``(aux_xs, error_fn)`` ready for ``_jax_block_core``.
    """
    n_pad = n_chunks * D
    if has_pilots:
        pref_T = jnp.transpose(pref)  # (n_sym, C)
        pref_pad = jnp.zeros((n_pad, num_ch), pref_T.dtype).at[:n_sym].set(pref_T)
        pmask_pad = (
            jnp.zeros((n_pad,), jnp.bool_).at[:n_sym].set(pmask.astype(jnp.bool_))
        )
        aux_xs = (
            pref_pad.reshape(n_chunks, D, num_ch),
            pmask_pad.reshape(n_chunks, D),
        )

        def error_fn(Y, aux):
            pref_c, pmask_c = aux
            return jnp.where(pmask_c[:, None], Y - pref_c, blind_fn(Y))

    else:
        aux_xs = jnp.arange(n_chunks)

        def error_fn(Y, aux):
            return blind_fn(Y)

    return aux_xs, error_fn


def _get_jax_lms_block(
    num_taps,
    stride,
    const_size,
    num_ch,
    n_sym,
    D,
    sq_side=0,
    sq_lev_min=0.0,
    sq_d_grid=1.0,
):
    """JIT-compile and cache the block-update LMS butterfly scan.

    Static closure variables mirror ``_get_jax_lms`` plus ``n_sym`` and the
    update block length ``D`` (both fix the chunk count / scan length at trace
    time).  Training/DD switch is applied elementwise within each chunk.
    """
    key = (
        "lms_block",
        num_taps,
        stride,
        const_size,
        num_ch,
        n_sym,
        D,
        sq_side,
        float(sq_lev_min),
        float(sq_d_grid),
    )
    if key not in _JITTED_EQ:
        jax, jnp, _ = _get_jax()
        n_chunks = (n_sym + D - 1) // D
        n_pad = n_chunks * D

        def _slice_block(Y, constellation):
            if sq_side > 0:  # static branch - O(1) square-QAM slicer
                ir = jnp.clip(
                    jnp.round((Y.real - sq_lev_min) / sq_d_grid).astype(jnp.int32),
                    0,
                    sq_side - 1,
                )
                ii = jnp.clip(
                    jnp.round((Y.imag - sq_lev_min) / sq_d_grid).astype(jnp.int32),
                    0,
                    sq_side - 1,
                )
                nr = sq_lev_min + ir.astype(jnp.float32) * jnp.float32(sq_d_grid)
                ni = sq_lev_min + ii.astype(jnp.float32) * jnp.float32(sq_d_grid)
                return jax.lax.complex(nr, ni)
            d2 = jnp.abs(Y[..., None] - constellation) ** 2  # (D, C, M)
            return constellation[jnp.argmin(d2, axis=-1)]

        @jax.jit
        def run(x_input, training_padded, constellation, W_init, mu, n_train):
            # training_padded: (C, n_sym) -> per-symbol (n_pad, C) chunks
            train_T = jnp.transpose(training_padded)  # (n_sym, C)
            train_pad = jnp.zeros((n_pad, num_ch), train_T.dtype)
            train_pad = train_pad.at[:n_sym].set(train_T)
            train_chunks = train_pad.reshape(n_chunks, D, num_ch)
            gidx_chunks = jnp.arange(n_pad).reshape(n_chunks, D)

            def error_fn(Y, aux):
                train_chunk, gidx = aux  # (D, C), (D,)
                dd = _slice_block(Y, constellation)
                d = jnp.where((gidx < n_train)[:, None], train_chunk, dd)
                return Y - d

            y_hat, errors, W_final = _jax_block_core(
                jax,
                jnp,
                x_input,
                W_init,
                mu,
                error_fn,
                (train_chunks, gidx_chunks),
                n_sym=n_sym,
                D=D,
                stride=stride,
                num_taps=num_taps,
                num_ch=num_ch,
            )
            return y_hat, errors, W_final, jnp.zeros((1,), jnp.complex64)

        _JITTED_EQ[key] = run
    return _JITTED_EQ[key]


def _get_jax_cma_block(num_taps, stride, num_ch, n_sym, D, has_pilots=False):
    """JIT-compile and cache the block-update CMA butterfly scan.

    Blind Godard error per chunk; when ``has_pilots`` the error inverts to the
    LMS residual ``y - pilot_ref`` at masked positions (subtractive update,
    matching ``_get_numba_pa_cma``).
    """
    key = ("cma_block", num_taps, stride, num_ch, n_sym, D, has_pilots)
    if key not in _JITTED_EQ:
        jax, jnp, _ = _get_jax()
        n_chunks = (n_sym + D - 1) // D

        @jax.jit
        def run(x_input, W_init, mu, r2, pref, pmask):
            aux_xs, error_fn = _jax_block_pilot_aux(
                jax,
                jnp,
                n_sym,
                D,
                num_ch,
                n_chunks,
                has_pilots,
                pref,
                pmask,
                lambda Y: Y * (jnp.real(Y * jnp.conj(Y)) - r2),
            )
            y_hat, errors, W_final = _jax_block_core(
                jax,
                jnp,
                x_input,
                W_init,
                mu,
                error_fn,
                aux_xs,
                n_sym=n_sym,
                D=D,
                stride=stride,
                num_taps=num_taps,
                num_ch=num_ch,
            )
            return y_hat, errors, W_final, jnp.zeros((1,), jnp.complex64)

        _JITTED_EQ[key] = run
    return _JITTED_EQ[key]


def _get_jax_rde_block(num_taps, stride, num_radii, num_ch, n_sym, D, has_pilots=False):
    """JIT-compile and cache the block-update RDE butterfly scan.

    Per-symbol nearest-ring radial error per chunk; pilot positions invert to
    the LMS residual as in CMA.
    """
    key = ("rde_block", num_taps, stride, num_radii, num_ch, n_sym, D, has_pilots)
    if key not in _JITTED_EQ:
        jax, jnp, _ = _get_jax()
        n_chunks = (n_sym + D - 1) // D

        @jax.jit
        def run(x_input, W_init, mu, radii, pref, pmask):
            def blind_fn(Y):
                abs_y2 = jnp.real(Y * jnp.conj(Y))  # (D, C)
                abs_y = jnp.sqrt(abs_y2)
                dist = jnp.abs(abs_y[..., None] - radii[None, None, :])  # (D, C, K)
                rd = radii[jnp.argmin(dist, axis=-1)]  # (D, C)
                return Y * (abs_y2 - rd**2)

            aux_xs, error_fn = _jax_block_pilot_aux(
                jax, jnp, n_sym, D, num_ch, n_chunks, has_pilots, pref, pmask, blind_fn
            )
            y_hat, errors, W_final = _jax_block_core(
                jax,
                jnp,
                x_input,
                W_init,
                mu,
                error_fn,
                aux_xs,
                n_sym=n_sym,
                D=D,
                stride=stride,
                num_taps=num_taps,
                num_ch=num_ch,
            )
            return y_hat, errors, W_final, jnp.zeros((1,), jnp.complex64)

        _JITTED_EQ[key] = run
    return _JITTED_EQ[key]


def _get_jax_pa_cma(num_taps: int, stride: int, num_ch: int):
    """JIT-compile and cache the JAX pilot-aided CMA butterfly scan.

    Hybrid CMA scan using ``jax.lax.scan``.  At pilot positions
    (``pilot_mask[t] == True``) the error is the standard LMS residual
    ``pilot_ref[t] - y``; at data positions the Godard CMA error
    ``y * (|y|² - R²)`` is used.  The switch is XLA-branchless via
    ``jnp.where``, keeping the scan body shape-static for efficient
    compilation.

    Parameters
    ----------
    num_taps : int
    stride   : int - samples per symbol (sps)
    num_ch   : int - number of MIMO channels C

    Returns
    -------
    pa_cma_scan : jax.jit-compiled callable
        ``(x_input, w_init, step_size, r2, pilot_ref, pilot_mask, n_sym)
        -> (y_all, e_all, W_final, w_hist)``
        where ``pilot_ref`` has shape ``(n_sym, C)`` and ``pilot_mask``
        has shape ``(n_sym,)`` bool.
    """
    key = ("pa_cma", num_taps, stride, num_ch)
    if key not in _JITTED_EQ:
        jax, jnp, _ = _get_jax()
        assert jax is not None
        assert jnp is not None

        @functools.partial(jax.jit, static_argnums=(6,))
        def pa_cma_scan(
            x_input,  # (C, N_pad)  complex64 - padded received samples
            w_init,  # (C, C, T)   complex64 - initial butterfly weights
            step_size,  # ()          float32
            r2,  # ()          float32 - Godard R² = E[|s|⁴]/E[|s|²]
            pilot_ref,  # (n_sym, C)  complex64 - known symbols; 0 at data
            pilot_mask,  # (n_sym,)    bool - True at pilot/preamble positions
            n_sym,  # int (static) - total symbol count
        ):
            _P = jax.lax.Precision.HIGHEST

            def step(W, xs_t):
                idx, p_ref, p_mask = xs_t  # (): int, (C,): cplx, (): bool
                X_wins = jax.lax.dynamic_slice(
                    x_input, (0, idx * stride), (num_ch, num_taps)
                )  # (C, T)
                y = jnp.einsum("ijt,jt->i", jnp.conj(W), X_wins, precision=_P)  # (C,)

                abs_y2 = jnp.real(y * jnp.conj(y))  # strict real |y|²
                e_blind = y * (abs_y2 - r2)  # Godard CMA
                e_da = (
                    y - p_ref
                )  # pilot LMS (inverted to match blind subtractive update)
                e = jnp.where(p_mask, e_da, e_blind)  # (C,) branchless

                W_new = W - step_size * jnp.einsum("i,jt->ijt", jnp.conj(e), X_wins)
                return W_new, (y, e, W_new)

            xs = (jnp.arange(n_sym), pilot_ref, pilot_mask)
            W_final, (y_all, e_all, wh_all) = jax.lax.scan(step, w_init, xs)
            return y_all, e_all, W_final, wh_all

        _JITTED_EQ[key] = pa_cma_scan  # type: ignore[index]
    return _JITTED_EQ[key]  # type: ignore[index]


def _get_jax_pa_rde(num_taps: int, stride: int, num_radii: int, num_ch: int):
    """JIT-compile and cache the JAX pilot-aided RDE butterfly scan.

    Hybrid RDE scan using ``jax.lax.scan``.  At pilot positions the error
    is the LMS residual ``pilot_ref[t] - y``; at data positions the
    ring-directed RDE error ``y * (|y|² - R_d²)`` is used, where ``R_d``
    is the nearest constellation ring radius.  The switch is branchless.

    Parameters
    ----------
    num_taps  : int
    stride    : int - samples per symbol (sps)
    num_radii : int - number of unique constellation ring radii K
    num_ch    : int - number of MIMO channels C

    Returns
    -------
    pa_rde_scan : jax.jit-compiled callable
        ``(x_input, w_init, step_size, radii, pilot_ref, pilot_mask, n_sym)
        -> (y_all, e_all, W_final, w_hist)``
        where ``pilot_ref`` has shape ``(n_sym, C)`` and ``pilot_mask``
        has shape ``(n_sym,)`` bool.
    """
    key = ("pa_rde", num_taps, stride, num_radii, num_ch)
    if key not in _JITTED_EQ:
        jax, jnp, _ = _get_jax()
        assert jax is not None
        assert jnp is not None

        @functools.partial(jax.jit, static_argnums=(6,))
        def pa_rde_scan(
            x_input,  # (C, N_pad)  complex64 - padded received samples
            w_init,  # (C, C, T)   complex64 - initial butterfly weights
            step_size,  # ()          float32
            radii,  # (K,)        float32 - unique |c| constellation radii, sorted
            pilot_ref,  # (n_sym, C)  complex64 - known symbols; 0 at data
            pilot_mask,  # (n_sym,)    bool - True at pilot/preamble positions
            n_sym,  # int (static) - total symbol count
        ):
            _P = jax.lax.Precision.HIGHEST

            def step(W, xs_t):
                idx, p_ref, p_mask = xs_t  # (): int, (C,): cplx, (): bool
                X_wins = jax.lax.dynamic_slice(
                    x_input, (0, idx * stride), (num_ch, num_taps)
                )  # (C, T)
                y = jnp.einsum("ijt,jt->i", jnp.conj(W), X_wins, precision=_P)  # (C,)

                abs_y2 = jnp.real(y * jnp.conj(y))  # strict real |y|²
                abs_y = jnp.sqrt(abs_y2)  # (C,)
                dist = jnp.abs(abs_y[:, None] - radii[None, :])  # (C, K) broadcast
                rd = radii[jnp.argmin(dist, axis=1)]  # (C,) nearest radius

                e_blind = y * (abs_y2 - rd**2)  # RDE ring-directed
                e_da = (
                    y - p_ref
                )  # pilot LMS (inverted to match blind subtractive update)
                e = jnp.where(p_mask, e_da, e_blind)  # (C,) branchless

                W_new = W - step_size * jnp.einsum("i,jt->ijt", jnp.conj(e), X_wins)
                return W_new, (y, e, W_new)

            xs = (jnp.arange(n_sym), pilot_ref, pilot_mask)
            W_final, (y_all, e_all, wh_all) = jax.lax.scan(step, w_init, xs)
            return y_all, e_all, W_final, wh_all

        _JITTED_EQ[key] = pa_rde_scan  # type: ignore[index]
    return _JITTED_EQ[key]  # type: ignore[index]
