// Per-symbol cycle-slip correction for one block_lms equalizer block.
//
// Device version of the Numba kernel `cs_block`, its CPU reference: each
// phase y_i is predicted by an OLS line through the last n = min(count, H)
// corrected phases (x = 0 .. n-1, oldest first; the last value alone while
// n < 10), and snapped by the nearest integer multiple k_i of `quantum`
// when |y_i - prediction| exceeds `threshold`.  Corrected values feed the
// later predictions, so the recursion is sequential, but its decisions are
// almost always "no slip".  The kernel therefore speculates:
//
//   1. guess k_i for every symbol of the block (initially 0);
//   2. with y_i = x_i - k_i * quantum, compute all predictions at once from
//      prefix sums of y and j * y over Z = [history, block] (parallel scan);
//   3. the first symbol whose decision differs from its guess is decided
//      correctly (everything before it matched).  Fix it, guess the same k
//      for the rest (a slip persists), and repeat from the next symbol.
//
// One pass per changed decision (a slip, or an isolated outlier costs two)
// instead of one dependent step per symbol.  The OLS sums come from prefix
// differences rather than rolling updates, so they differ from the
// reference only in float64 rounding; the end-of-block state is recomputed
// exactly from the window.  float64 throughout, on values shifted by the
// first window value to keep the prefix sums small.
//
// Layout contract (enforced by the Python wrapper):
//   phi_blk    (C, B)     float64 - BPS phase before correction (input)
//   phi_corr   (C, B)     float64 - corrected phase (output)
//   cs_buf_y   (C, H)     float64 - circular buffer of past corrected phases
//   cs_buf_ptr (C,)       int64   - write pointer (monotonically increasing)
//   cs_buf_n   (C,)       int64   - number of valid entries (<= H)
//   cs_stats   (C, 4)     float64 - [0]=Sy, [1]=Sxy (relative coords); [2..3] unused
//   z          (C, H+B)   float64 - scratch: shifted window values
//   p1, p2     (C, H+B+1) float64 - scratch: prefix sums of z and j * z
//   guess      (C, B)     int32   - scratch: guessed k
//   decided    (C, B)     int32   - scratch: decisions under the guess
//
// Launch contract: grid = (C, 1, 1), block = (CS_THREADS, 1, 1), B >= 1;
// CS_THREADS is a multiple of 32 and at most 1024.

#define CS_THREADS 256

extern "C" __global__ void cs_block(const double* __restrict__ phi_blk,
                                    double* __restrict__ phi_corr,
                                    double* cs_buf_y,
                                    long long* cs_buf_ptr,
                                    long long* cs_buf_n,
                                    double* cs_stats,
                                    double* __restrict__ z_all,
                                    double* __restrict__ p1_all,
                                    double* __restrict__ p2_all,
                                    int* __restrict__ guess_all,
                                    int* __restrict__ decided_all,
                                    const double quantum,
                                    const double threshold,
                                    const int H,
                                    const int B) {
    __shared__ double warp1[CS_THREADS / 32];
    __shared__ double warp2[CS_THREADS / 32];
    __shared__ int first;

    const int c = blockIdx.x;
    const int tid = threadIdx.x;
    const int T = blockDim.x;
    const int stride = H + B;

    const double* x = phi_blk + static_cast<long long>(c) * B;
    double* out = phi_corr + static_cast<long long>(c) * B;
    double* buf = cs_buf_y + static_cast<long long>(c) * H;
    double* z = z_all + static_cast<long long>(c) * stride;
    double* p1 = p1_all + static_cast<long long>(c) * (stride + 1);
    double* p2 = p2_all + static_cast<long long>(c) * (stride + 1);
    int* guess = guess_all + static_cast<long long>(c) * B;
    int* decided = decided_all + static_cast<long long>(c) * B;

    const int n0 = static_cast<int>(cs_buf_n[c]);
    const long long ptr0 = cs_buf_ptr[c];
    const int L = n0 + B;  // Z = [history (n0, oldest first), block (B)]

    // Shift by the oldest window value: the OLS prediction is shift-equivariant.
    const double c0 = (n0 > 0) ? buf[(ptr0 - n0) % H] : x[0];

    for (int j = tid; j < n0; j += T) {
        z[j] = buf[(ptr0 - n0 + j) % H] - c0;
    }
    for (int i = tid; i < B; i += T) {
        guess[i] = 0;
    }
    if (tid == 0) {
        p1[0] = 0.0;
        p2[0] = 0.0;
    }

    // Full window (m = H): the OLS prediction is a fixed linear combination
    // a * Sy + b * Sxy; precomputed, as float64 division is slow on GPUs.
    const double Hd = static_cast<double>(H);
    const double Sx_H = Hd * (Hd - 1.0) / 2.0;
    const double Sxx_H = Hd * (Hd - 1.0) * (2.0 * Hd - 1.0) / 6.0;
    const double den_H = Hd * Sxx_H - Sx_H * Sx_H;
    const bool full_ok = H >= 10 && fabs(den_H) > 1e-30;
    const double b_H = full_ok ? (Hd - Sx_H / Hd) * Hd / den_H : 0.0;
    const double a_H = full_ok ? 1.0 / Hd - b_H * Sx_H / Hd : 0.0;
    const double inv_quantum = 1.0 / quantum;
    const int chunk = (L + T - 1) / T;
    const int j0 = min(tid * chunk, L);
    const int j1 = min(j0 + chunk, L);

    int start = 0;    // first symbol not yet decided
    int rewrite = 0;  // first symbol whose guess changed
    while (true) {
        // y_i = x_i - k_i * quantum where the guess changed.
        for (int i = rewrite + tid; i < B; i += T) {
            z[n0 + i] = (x[i] - static_cast<double>(guess[i]) * quantum) - c0;
        }
        if (tid == 0) {
            first = B;
        }
        __syncthreads();

        // Inclusive prefix sums of z and j * z: chunk sums, scanned across
        // the block with warp shuffles, then each chunk adds its base.
        double s1 = 0.0, s2 = 0.0;
        for (int j = j0; j < j1; ++j) {
            s1 += z[j];
            s2 += static_cast<double>(j) * z[j];
        }
        const int lane = tid & 31;
        const int warp = tid >> 5;
        double v1 = s1, v2 = s2;
        for (int off = 1; off < 32; off <<= 1) {
            const double t1 = __shfl_up_sync(0xffffffffu, v1, off);
            const double t2 = __shfl_up_sync(0xffffffffu, v2, off);
            if (lane >= off) {
                v1 += t1;
                v2 += t2;
            }
        }
        if (lane == 31) {
            warp1[warp] = v1;
            warp2[warp] = v2;
        }
        __syncthreads();
        if (warp == 0) {
            const int nw = T >> 5;
            double w1 = (lane < nw) ? warp1[lane] : 0.0;
            double w2 = (lane < nw) ? warp2[lane] : 0.0;
            for (int off = 1; off < 32; off <<= 1) {
                const double t1 = __shfl_up_sync(0xffffffffu, w1, off);
                const double t2 = __shfl_up_sync(0xffffffffu, w2, off);
                if (lane >= off) {
                    w1 += t1;
                    w2 += t2;
                }
            }
            if (lane < nw) {
                warp1[lane] = w1;  // inclusive over warps
                warp2[lane] = w2;
            }
        }
        __syncthreads();
        double r1 = v1 - s1 + (warp > 0 ? warp1[warp - 1] : 0.0);
        double r2 = v2 - s2 + (warp > 0 ? warp2[warp - 1] : 0.0);
        for (int j = j0; j < j1; ++j) {
            r1 += z[j];
            r2 += static_cast<double>(j) * z[j];
            p1[j + 1] = r1;
            p2[j + 1] = r2;
        }
        __syncthreads();

        // Decisions under the guess; the earliest mismatch wins.
        for (int i = start + tid; i < B; i += T) {
            const int p = n0 + i;               // position of y_i in Z
            const int m = min(n0 + i, H);       // window size before y_i
            const double raw = x[i] - c0;
            double expected;
            if (m == 0) {
                expected = raw;
            } else if (m < 10) {
                expected = z[p - 1];
            } else if (m == H && full_ok) {
                const double sy = p1[p] - p1[p - m];
                const double sxy =
                    (p2[p] - p2[p - m]) - static_cast<double>(p - m) * sy;
                expected = a_H * sy + b_H * sxy;
            } else {
                const double n_f = static_cast<double>(m);
                const double Sx_c = n_f * (n_f - 1.0) / 2.0;
                const double Sxx_c = n_f * (n_f - 1.0) * (2.0 * n_f - 1.0) / 6.0;
                const double denom = n_f * Sxx_c - Sx_c * Sx_c;
                const double sy = p1[p] - p1[p - m];
                const double sxy =
                    (p2[p] - p2[p - m]) - static_cast<double>(p - m) * sy;
                double slope, intercept;
                if (fabs(denom) > 1e-30) {
                    slope = (n_f * sxy - Sx_c * sy) / denom;
                    intercept = (sy - slope * Sx_c) / n_f;
                } else {
                    slope = 0.0;
                    intercept = sy / n_f;
                }
                expected = slope * n_f + intercept;
            }
            const double diff = raw - expected;
            // llrint = round-half-even, matching Python round() in the reference.
            const long long k_slip = llrint(diff * inv_quantum);
            const int d = (fabs(diff) > threshold && k_slip != 0)
                              ? static_cast<int>(k_slip)
                              : 0;
            decided[i] = d;
            if (d != guess[i]) {
                atomicMin(&first, i);
            }
        }
        __syncthreads();

        const int f = first;
        if (f >= B) {
            break;  // every decision matches its guess: z holds the result
        }
        const int kf = decided[f];
        for (int i = f + tid; i < B; i += T) {
            guess[i] = kf;  // decided at f; a slip persists for the rest
        }
        rewrite = f;
        start = f + 1;
        __syncthreads();
    }

    for (int i = tid; i < B; i += T) {
        out[i] = x[i] - static_cast<double>(guess[i]) * quantum;
    }
    // The last min(B, H) corrected values enter the circular buffer.
    const int keep = min(B, H);
    for (int i = B - keep + tid; i < B; i += T) {
        buf[(ptr0 + i) % H] = x[i] - static_cast<double>(guess[i]) * quantum;
    }
    if (tid == 0) {
        // Window statistics, exactly, in the reference's coordinates.
        const int nn = min(L, H);
        const double nn_f = static_cast<double>(nn);
        const double sy = p1[L] - p1[L - nn];
        const double sxy = (p2[L] - p2[L - nn]) - static_cast<double>(L - nn) * sy;
        cs_stats[c * 4 + 0] = sy + nn_f * c0;
        cs_stats[c * 4 + 1] = sxy + c0 * nn_f * (nn_f - 1.0) / 2.0;
        cs_buf_n[c] = nn;
        cs_buf_ptr[c] = ptr0 + B;
    }
}
