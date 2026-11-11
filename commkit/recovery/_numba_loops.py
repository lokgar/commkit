"""Numba loops of carrier-phase recovery (private).

The compiled CPU kernels of :mod:`commkit.recovery`: the decision-directed
PLL (one loop per channel, or one shared by the channels), the block
metrics of :class:`~commkit.recovery.BPS` (a search over the table, or the
O(1) slicer of square QAM), the Rauch-Tung-Striebel smoother of
:class:`~commkit.recovery.Tikhonov` and the cycle-slip repair of
:func:`~commkit.recovery.correct_cycle_slips`.  Imported on first use
through the ``_get_numba_*`` getters of ``recovery/pll.py``,
``recovery/bps.py``, ``recovery/tikhonov.py`` and
``recovery/corrections.py``, so ``import commkit`` does not load Numba.

Phases, loop state and smoother state are float64; symbols are read in
their own precision and widened as they are read.
"""

import numba
import numpy as np

_fast = numba.njit(cache=True, fastmath=True, nogil=True)
_parallel = numba.njit(cache=True, fastmath=True, nogil=True, parallel=True)


@_parallel
def dd_pll(
    sym,
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
    """Run one decision-directed PLL per channel over the symbols (Numba).

    Parameters
    ----------
    sym : numpy.ndarray
        Received symbols, complex64 or complex128, ``(C, N)``; each is
        widened to float64 as it is read.
    const_r, const_i : numpy.ndarray
        Real and imaginary parts of the constellation, float64, ``(M,)``.
    mu : float
        Proportional (phase) gain: corrects the instantaneous phase error.
    beta : float
        Integral (frequency) gain: tracks a residual frequency drift; 0 for
        a first-order loop.
    phi0 : float
        Initial phase in radians.
    freq0 : float
        Initial frequency correction in radians per symbol.
    is_sq_qam : bool
        The constellation is a square QAM grid: decide by rounding, O(1),
        instead of the O(M) search.
    levels : numpy.ndarray
        Sorted distinct axis levels of the square grid, float64, ``(side,)``;
        unused otherwise.
    d_grid : float
        Grid spacing, ``levels[1] - levels[0]``.
    lev_min : float
        Smallest level, ``levels[0]``.
    side : int
        Points per axis, ``sqrt(M)``.

    Returns
    -------
    numpy.ndarray
        Per-symbol phase ``φ[c, n]``, float64, ``(C, N)``.
    """
    C = sym.shape[0]
    N = sym.shape[1]
    M = len(const_r)
    phase_est = np.empty((C, N), dtype=np.float64)

    for c in numba.prange(C):
        phi = phi0
        freq = freq0
        for n in range(N):
            # Rotate received symbol by current phase estimate:
            # y[n] = s[n] · exp(-jφ[n])
            cos_phi = np.cos(phi)
            sin_phi = np.sin(phi)
            sr = np.float64(sym[c, n].real)
            si = np.float64(sym[c, n].imag)
            yr = sr * cos_phi + si * sin_phi
            yi = -sr * sin_phi + si * cos_phi

            # Hard decision: the nearest constellation point
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
            phase_est[c, n] = phi

            # 2nd-order loop filter (reduces to 1st order when beta=0):
            #   φ[n+1] = φ[n] + μ·e[n] + ν[n]
            #   ν[n]   = ν[n-1] + β·e[n]
            phi = phi + mu * e + freq
            freq = freq + beta * e

    return phase_est


@_fast
def dd_pll_joint(
    sym,
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
    """Run one decision-directed PLL shared by all channels (Numba).

    The phase error is averaged over the channels at each symbol and drives
    one PI loop: the minimum-variance joint estimator for a shared LO.

    Parameters
    ----------
    sym : numpy.ndarray
        Received symbols, complex64 or complex128, ``(C, N)``; each is
        widened to float64 as it is read.
    const_r, const_i : numpy.ndarray
        Real and imaginary parts of the constellation, float64, ``(M,)``.
    mu, beta, phi0, freq0 : float
        Loop gains and initial state, as for ``dd_pll``.
    is_sq_qam : bool
        Decide by rounding on a square QAM grid.
    levels : numpy.ndarray
        Axis levels of the square grid, float64, ``(side,)``.
    d_grid, lev_min : float
        Grid spacing and smallest level.
    side : int
        Points per axis.

    Returns
    -------
    numpy.ndarray
        The shared phase ``φ[n]``, float64, ``(N,)``; the caller copies it to
        every channel.
    """
    C = sym.shape[0]
    N = sym.shape[1]
    M = len(const_r)
    phase_est = np.empty(N, dtype=np.float64)
    phi = phi0
    freq = freq0

    for n in range(N):
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        e_sum = 0.0
        for c in range(C):
            sr = np.float64(sym[c, n].real)
            si = np.float64(sym[c, n].imag)
            yr = sr * cos_phi + si * sin_phi
            yi = -sr * sin_phi + si * cos_phi
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


@_parallel
def bps_table(x_re, x_im, ph_re, ph_im, c_re, c_im, block_size, out):
    """Per block and test phase, the summed distance to the nearest point.

    ``out[c, b, k] = sum_{n in block b} min_m |x[c, n] e^{-j theta_k} - s_m|^2``
    for block ``b`` of channel ``c`` rotated by phase ``k``, the nearest
    point found by a search over the table, in parallel over channels and
    blocks.  Distances are float32 for complex64 input, the block sums
    float64.
    """
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


@_parallel
def bps_grid(x_re, x_im, ph_re, ph_im, lev_min, d_grid, side, block_size, out):
    """Per block and test phase, the summed distance to the nearest point.

    As ``bps_table``, the nearest point found in O(1) by snapping each axis
    to the grid of ``side`` levels from ``lev_min`` in steps of ``d_grid``.
    """
    C = x_re.shape[0]
    n_blocks = out.shape[1]
    B = ph_re.shape[0]
    last = side - 1
    for job in numba.prange(C * n_blocks):
        c = job // n_blocks
        b = job - c * n_blocks
        for k in range(B):
            acc = 0.0
            for n in range(b * block_size, (b + 1) * block_size):
                xr = x_re[c, n] * ph_re[k] - x_im[c, n] * ph_im[k]
                xi = x_re[c, n] * ph_im[k] + x_im[c, n] * ph_re[k]
                ir = min(max(np.round((xr - lev_min) / d_grid), 0), last)
                ii = min(max(np.round((xi - lev_min) / d_grid), 0), last)
                dr = xr - (lev_min + ir * d_grid)
                di = xi - (lev_min + ii * d_grid)
                acc += dr * dr + di * di
            out[c, b, k] = acc


@_fast
def rts_smoother(phi_obs, sigma_p2, sigma_v2):
    """Run the Rauch-Tung-Striebel smoother over the block phases (Numba).

    Parameters
    ----------
    phi_obs : numpy.ndarray
        Observed block phases, float64, ``(B,)``.
    sigma_p2 : float
        Process noise variance per block.
    sigma_v2 : float
        Observation noise variance per block.

    Returns
    -------
    numpy.ndarray
        Smoothed phases, float64, ``(B,)``.
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


@_fast
def cycle_slip(phi_u, symmetry, history_length, threshold):
    """Detect and correct cycle slips by linear extrapolation (Numba).

    Scans each row of block phases ``phi_u`` in order.  Each block is
    extrapolated linearly from up to ``history_length`` past corrected
    blocks; a deviation beyond ``threshold`` is corrected by a multiple
    of ``2π/symmetry``.

    The regression uses the relative coordinates ``[0, W-1]``, so ``Sx`` and
    ``Sxx`` are closed-form constants; only ``Sy`` and ``Sxy`` are kept as
    running state, updated in O(1) per step by a sliding-window identity.

    Parameters
    ----------
    phi_u : numpy.ndarray
        Block phases after the M-fold unwrap, float64, ``(R, B)``;
        modified in place.
    symmetry : int
        Rotational symmetry; the correction quantum is ``2π/symmetry``.
    history_length : int
        Largest number of past corrected blocks in the fit; block ``b`` uses
        ``min(b, history_length)``.
    threshold : float
        Deviation from the extrapolated value, in radians, that declares a
        slip (``π/4`` by default in the callers).

    Returns
    -------
    numpy.ndarray
        The corrected block phases, ``phi_u`` itself.
    """
    two_pi = 2.0 * np.pi
    quantum = two_pi / float(symmetry)
    R, B = phi_u.shape
    W = history_length
    W_f = float(W)

    # Precompute full-window regression constants in relative coords [0, W-1].
    # With relative coords the x-values are always small integers, so Sx and
    # Sxx never grow and there is no catastrophic cancellation regardless of
    # how many total blocks have been processed.
    Sx_full = W_f * (W_f - 1.0) / 2.0
    Sxx_full = W_f * (W_f - 1.0) * (2.0 * W_f - 1.0) / 6.0
    denom_full = W_f * Sxx_full - Sx_full * Sx_full  # W²(W²-1)/12

    # Rows (channels, usually one or two) in series: under prange
    # the scan ran at half speed per row (two rows of 2^16 values:
    # 0.41 ms in parallel, 0.40 ms in series), a gain only from
    # four rows on.
    for r in range(R):
        row = phi_u[r]
        # Only Sy and Sxy need to be maintained as running state.
        buf_y = np.empty(W, dtype=np.float64)
        buf_head = 0  # next write slot (circular)
        n_buf = 0  # valid entries currently in buffer

        Sy = 0.0
        Sxy = 0.0

        for b in range(B):
            y_b = row[b]

            if n_buf == 0:
                # First block: trust it unconditionally (at relative position 0).
                buf_y[0] = y_b
                buf_head = 1
                n_buf = 1
                Sy = y_b
                Sxy = 0.0  # 0 * y_b
                continue

            if n_buf < min(10, W):
                # Constant extrapolation during warmup to avoid cementing false slips.
                phi_pred = buf_y[(buf_head - 1) % W]
            else:
                # Linear extrapolation.  Prediction target is always one step past
                # the newest buffered entry, i.e. relative coordinate = n_buf.
                x_pred = float(n_buf)
                n_f = float(n_buf)
                if n_buf < W:
                    # Partial window: derive exact Sx/Sxx from closed-form sums.
                    Sx_p = n_f * (n_f - 1.0) / 2.0
                    Sxx_p = n_f * (n_f - 1.0) * (2.0 * n_f - 1.0) / 6.0
                    denom = n_f * Sxx_p - Sx_p * Sx_p
                    if abs(denom) > 1e-30:
                        slope = (n_f * Sxy - Sx_p * Sy) / denom
                        intercept = (Sy - slope * Sx_p) / n_f
                    else:
                        slope = 0.0
                        intercept = Sy / n_f
                else:
                    # Full window: use precomputed constants (numerically exact).
                    if denom_full > 1e-30:
                        slope = (W_f * Sxy - Sx_full * Sy) / denom_full
                        intercept = (Sy - slope * Sx_full) / W_f
                    else:
                        slope = 0.0
                        intercept = Sy / W_f
                phi_pred = slope * x_pred + intercept

            diff = y_b - phi_pred
            # Round to nearest correction quantum
            k = round(diff / quantum)
            if abs(diff) > threshold and k != 0:
                row[b] -= float(k) * quantum
                y_b = row[b]

            # Update circular buffer using relative coordinates.
            if n_buf == W:
                # Slide window: evict oldest (relative pos 0), shift all down by 1,
                # add y_b at relative position W-1.
                # Sxy update uses the identity:
                #   Sxy_new = Sxy_old - Sy_old + y_old + (W-1)·y_new
                # (derived by relabelling positions after eviction)
                old_idx = buf_head % W
                y_old = buf_y[old_idx]
                Sxy = Sxy - Sy + y_old + (W_f - 1.0) * y_b  # must precede Sy update
                Sy = Sy - y_old + y_b
                buf_y[old_idx] = y_b
                buf_head += 1
            else:
                # Append at relative position n_buf.
                idx = buf_head % W
                buf_y[idx] = y_b
                Sxy += float(n_buf) * y_b
                Sy += y_b
                buf_head += 1
                n_buf += 1

    return phi_u
