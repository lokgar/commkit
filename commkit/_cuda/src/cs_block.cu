// Per-symbol cycle-slip correction for one block_lms equalizer block.
//
// Faithful port of the Numba kernel `cs_block` (equalization.py): for each
// symbol, predict the expected phase from an online OLS regression over the
// last H corrected phases (relative coordinates, closed-form Sx/Sxx), snap
// the BPS phase by the nearest integer multiple of `quantum` when |diff|
// exceeds `threshold`, then push the corrected phase into the per-channel
// circular history using the O(1) rolling-sum identities:
//
//   full buffer: Sxy_new = Sxy_old - Sy_old + y_old + (H-1) * y_new
//                Sy_new  = Sy_old - y_old + y_new
//   filling:     Sxy += n * y_new ; Sy += y_new
//
// Within a channel the recursion is strictly sequential; channels are
// independent. Total work is C*B trivial scalar iterations per launch, so
// the kernel runs as a single block with one thread per channel - the goal
// is not throughput but keeping the block loop free of host synchronization
// (it replaces a per-block D2H -> CPU Numba -> H2D round trip).
//
// Layout contract (enforced by the Python wrapper):
//   phi_blk    (C, B) float64 - BPS phase before correction (input)
//   phi_corr   (C, B) float64 - corrected phase (output)
//   cs_buf_y   (C, H) float64 - circular buffer of past corrected phases
//   cs_buf_ptr (C,)   int64   - write pointer (monotonically increasing)
//   cs_buf_n   (C,)   int64   - number of valid entries (<= H)
//   cs_stats   (C, 4) float64 - [0]=Sy, [1]=Sxy (relative coords); [2..3] unused
//
// Launch contract: grid = (1, 1, 1), block = (C, 1, 1), C <= 1024.
// All arithmetic is float64, matching the Numba kernel - the GeForce FP64
// throughput cliff is irrelevant at C*B ~ 512 sequential iterations.

__global__ void cs_block(const double* __restrict__ phi_blk,
                         double* __restrict__ phi_corr,
                         double* cs_buf_y,
                         long long* cs_buf_ptr,
                         long long* cs_buf_n,
                         double* cs_stats,
                         const double quantum,
                         const double threshold,
                         const int cs_H,
                         const int B) {
    const int ci = threadIdx.x;  // one thread per channel; blockDim.x == C

    const double H_f = static_cast<double>(cs_H);
    const double Sx_full = H_f * (H_f - 1.0) / 2.0;
    const double Sxx_full = H_f * (H_f - 1.0) * (2.0 * H_f - 1.0) / 6.0;
    const double denom_full = H_f * Sxx_full - Sx_full * Sx_full;

    // Channel-local state in registers; written back once after the loop.
    long long n_b = cs_buf_n[ci];
    long long ptr = cs_buf_ptr[ci];
    double sy = cs_stats[ci * 4 + 0];
    double sxy = cs_stats[ci * 4 + 1];
    double* buf_y = cs_buf_y + static_cast<long long>(ci) * cs_H;
    const double* phi_in = phi_blk + static_cast<long long>(ci) * B;
    double* phi_out = phi_corr + static_cast<long long>(ci) * B;

    for (int i = 0; i < B; ++i) {
        double y_b = phi_in[i];
        double phi_expected;

        if (n_b == 0) {
            phi_expected = y_b;
        } else if (n_b < 10) {
            const long long last_pos = (ptr - 1 + cs_H) % cs_H;
            phi_expected = buf_y[last_pos];
        } else {
            const double n_f = static_cast<double>(n_b);
            double Sx_c, denom;
            if (n_b < cs_H) {
                Sx_c = n_f * (n_f - 1.0) / 2.0;
                const double Sxx_c = n_f * (n_f - 1.0) * (2.0 * n_f - 1.0) / 6.0;
                denom = n_f * Sxx_c - Sx_c * Sx_c;
            } else {
                Sx_c = Sx_full;
                denom = denom_full;
            }
            double slope, intercept;
            if (fabs(denom) > 1e-30) {
                slope = (n_f * sxy - Sx_c * sy) / denom;
                intercept = (sy - slope * Sx_c) / n_f;
            } else {
                slope = 0.0;
                intercept = sy / n_f;
            }
            phi_expected = slope * n_f + intercept;
        }

        const double diff = y_b - phi_expected;
        // llrint = round-half-even, matching Python round() in the Numba kernel.
        const long long k_slip = llrint(diff / quantum);
        if (fabs(diff) > threshold && k_slip != 0) {
            y_b -= static_cast<double>(k_slip) * quantum;
        }
        phi_out[i] = y_b;

        // Update circular buffer - relative coords, only y needed.
        const long long write_pos = ptr % cs_H;
        if (n_b == cs_H) {
            const double old_y = buf_y[write_pos];
            const double old_sy = sy;
            sxy = sxy - old_sy + old_y + (H_f - 1.0) * y_b;
            sy = old_sy - old_y + y_b;
        } else {
            sxy += static_cast<double>(n_b) * y_b;
            sy += y_b;
        }
        buf_y[write_pos] = y_b;
        ptr += 1;
        if (n_b < cs_H) {
            n_b += 1;
        }
    }

    cs_buf_n[ci] = n_b;
    cs_buf_ptr[ci] = ptr;
    cs_stats[ci * 4 + 0] = sy;
    cs_stats[ci * 4 + 1] = sxy;
}
