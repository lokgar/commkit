// Data-aided BPS phase over the training symbols of one block_lms block.
//
// Device port of `_BlockBps._anchor` (equalization/_block/_dd.py), its CPU
// reference: for each training symbol i the phase is the angle of the sum
// of y conj(d) over a causal K-symbol window (summed over channels when
// joint), unwrapped from the previous phase over 2 pi.  The window
// continues across blocks through the last K-1 products.
//
// One block of threads.  Phase 1: each thread slides the window sum over a
// contiguous chunk of symbols (K + n/T steps) and writes the raw angles;
// float32 products, sums and atan2f (an error of ~1e-6 rad over the window,
// against float64 throughput 1/64 of float32 on consumer GPUs).  Phase 2,
// after a barrier: the float64 unwrap as a parallel prefix sum.  One launch
// per block keeps the block loop free of a host synchronization, which the
// per-block D2H -> NumPy -> H2D round trip costs.
//
// Window bookkeeping: Z = [hist (K-1), p_0 .. p_{n-1}]; the window of
// symbol i is Z[i .. i+K-1].  Products are recomputed from y and d instead
// of stored.
//
// Layout contract (enforced by the Python wrapper):
//   y        (C, n)   complex64 - equalizer output, training columns
//   d        (C, n)   complex64 - training symbols
//   hist_in  (C, K-1) complex64 - previous trailing products
//   hist_out (C, K-1) complex64 - new trailing products (output)
//   total    (n,)     complex64 - scratch for the joint sum (zeroed)
//   offset4  (C,)     float64   - S x phase, read and written
//   prev4    (C,)     float64   - written (= offset4 after the scan)
//   phi      (C, n)   float64   - anchored phase (output; holds the raw
//                                 angles between the two phases)
//
// Launch contract: grid = (1, 1, 1), block = (256, 1, 1), n >= 1.

#include <cupy/complex.cuh>

// Z[j] of channel c: the trailing history, then this block's products.
__device__ inline complex<float> z_at(const complex<float>* y,
                                      const complex<float>* d,
                                      const complex<float>* hist,
                                      int c, int j, int n, int H) {
    if (j < H) {
        return hist[static_cast<long long>(c) * H + j];
    }
    const long long k = static_cast<long long>(c) * n + (j - H);
    return y[k] * conj(d[k]);
}

extern "C" __global__ void bps_anchor(const complex<float>* __restrict__ y,
                                      const complex<float>* __restrict__ d,
                                      const complex<float>* __restrict__ hist_in,
                                      complex<float>* __restrict__ hist_out,
                                      complex<float>* __restrict__ total,
                                      double* offset4,
                                      double* prev4,
                                      double* __restrict__ phi,
                                      const int C,
                                      const int n,
                                      const int K,
                                      const int joint,
                                      const double S) {
    const double two_pi = 6.283185307179586;
    const int H = K - 1;
    const int tid = threadIdx.x;
    const int T = blockDim.x;
    const int chunk = (n + T - 1) / T;
    const int i0 = tid * chunk;
    const int i1 = min(i0 + chunk, n);

    // Phase 1: sliding window sums over this thread's chunk.
    for (int c = 0; c < C && i0 < i1; ++c) {
        complex<float> s(0.0f, 0.0f);
        for (int j = i0; j < i0 + K; ++j) {
            s += z_at(y, d, hist_in, c, j, n, H);
        }
        for (int i = i0; i < i1; ++i) {
            if (i > i0) {
                s += z_at(y, d, hist_in, c, i + K - 1, n, H) -
                     z_at(y, d, hist_in, c, i - 1, n, H);
            }
            if (joint) {
                total[i] += s;
            } else {
                phi[static_cast<long long>(c) * n + i] =
                    static_cast<double>(atan2f(s.imag(), s.real()));
            }
        }
    }
    if (joint) {
        for (int i = i0; i < i1; ++i) {
            const double ang =
                static_cast<double>(atan2f(total[i].imag(), total[i].real()));
            for (int c = 0; c < C; ++c) {
                phi[static_cast<long long>(c) * n + i] = ang;
            }
        }
    }
    for (int k = tid; k < C * H; k += T) {
        const int c = k / H;
        hist_out[k] = z_at(y, d, hist_in, c, n + (k - c * H), n, H);
    }
    __syncthreads();

    // Phase 2: unwrap from the previous phase, as a parallel prefix sum of
    // wrapped differences (phi_i = phi_{i-1} + wrap(ang_i - ang_{i-1}),
    // phi_{-1} = offset4 / S): each thread sums its chunk, thread 0 scans
    // the T chunk totals, and every thread adds its offset.  float64.
    __shared__ double part[256];
    for (int c = 0; c < C; ++c) {
        double* row = phi + static_cast<long long>(c) * n;
        const double start = offset4[c] / S;
        // Read the raw angle before this chunk before anyone overwrites it.
        const double a_before = (i0 == 0) ? start : (i0 < n ? row[i0 - 1] : 0.0);
        __syncthreads();
        double acc = 0.0;
        double a_prev = a_before;
        for (int i = i0; i < i1; ++i) {
            const double ang = row[i];
            const double dlt = ang - a_prev;
            acc += dlt - two_pi * rint(dlt / two_pi);
            a_prev = ang;
            row[i] = acc;  // chunk-local cumulative sum
        }
        part[tid] = acc;
        __syncthreads();
        if (tid == 0) {
            double run = start;
            for (int t = 0; t < T; ++t) {
                const double v = part[t];
                part[t] = run;  // exclusive scan, plus the start phase
                run += v;
            }
        }
        __syncthreads();
        const double base = part[tid];
        for (int i = i0; i < i1; ++i) {
            row[i] += base;
        }
        __syncthreads();
        if (tid == 0) {
            const double last = row[n - 1];
            offset4[c] = last * S;
            prev4[c] = last * S;
        }
        __syncthreads();
    }
}
