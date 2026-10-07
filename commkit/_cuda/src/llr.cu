// Bit log-likelihood ratios, one thread per symbol.
//
// Device port of the Numba kernel in `mapping/llr.py`, its CPU reference:
// per symbol, the metric m(s) = -|x - s|^2 / sigma^2 + log P(s) of every
// point, then per bit the max over the points whose bit is 0 minus the max
// over those whose bit is 1 (max-log), or the log-sum-exp of each set
// (exact; a second pass over the points, scaled by the max).  The per-bit
// maxima and sums stay in registers, so no (N, M) or (N, k, M/2)
// intermediate is formed - the CuPy path is limited by exactly those.
// float32 with explicit literals; the points, priors and labels are read
// through the cache (every thread walks them in the same order).
//
// Layout contract (enforced by the Python wrapper):
//   x       (n,)    complex64 - received symbols
//   pts     (M,)    complex64 - constellation points
//   log_pmf (M,)    float32   - log prior (zeros for uniform)
//   labels  (M,)    int32     - bit label of each point, MSB first
//   out     (n, k)  float32   - LLRs (output)
//
// Launch contract: 1-D grid of 1-D blocks, any size (grid-stride loop);
// 1 <= k <= LLR_MAX_K.

#include <cupy/complex.cuh>

#define LLR_MAX_K 16
#define LLR_INF __int_as_float(0x7f800000)

extern "C" __global__ void llr(const complex<float>* __restrict__ x,
                               const complex<float>* __restrict__ pts,
                               const float* __restrict__ log_pmf,
                               const int* __restrict__ labels,
                               float* __restrict__ out,
                               const int n,
                               const int M,
                               const int k,
                               const float inv_s2,
                               const int exact) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        const float xr = x[i].real();
        const float xi = x[i].imag();

        float p0[LLR_MAX_K];
        float p1[LLR_MAX_K];
#pragma unroll
        for (int b = 0; b < LLR_MAX_K; ++b) {
            p0[b] = -LLR_INF;
            p1[b] = -LLR_INF;
        }
        for (int m = 0; m < M; ++m) {
            const complex<float> s = pts[m];
            const float dr = xr - s.real();
            const float di = xi - s.imag();
            const float met = -(dr * dr + di * di) * inv_s2 + log_pmf[m];
            const int lab = labels[m];
#pragma unroll
            for (int b = 0; b < LLR_MAX_K; ++b) {
                if (b < k) {
                    if ((lab >> (k - 1 - b)) & 1) {
                        p1[b] = fmaxf(p1[b], met);
                    } else {
                        p0[b] = fmaxf(p0[b], met);
                    }
                }
            }
        }

        float* row = out + static_cast<long long>(i) * k;
        if (!exact) {
#pragma unroll
            for (int b = 0; b < LLR_MAX_K; ++b) {
                if (b < k) {
                    row[b] = p0[b] - p1[b];
                }
            }
            continue;
        }

        float s0[LLR_MAX_K];
        float s1[LLR_MAX_K];
#pragma unroll
        for (int b = 0; b < LLR_MAX_K; ++b) {
            s0[b] = 0.0f;
            s1[b] = 0.0f;
        }
        for (int m = 0; m < M; ++m) {
            const complex<float> s = pts[m];
            const float dr = xr - s.real();
            const float di = xi - s.imag();
            const float met = -(dr * dr + di * di) * inv_s2 + log_pmf[m];
            const int lab = labels[m];
#pragma unroll
            for (int b = 0; b < LLR_MAX_K; ++b) {
                if (b < k) {
                    if ((lab >> (k - 1 - b)) & 1) {
                        s1[b] += expf(met - p1[b]);
                    } else {
                        s0[b] += expf(met - p0[b]);
                    }
                }
            }
        }
#pragma unroll
        for (int b = 0; b < LLR_MAX_K; ++b) {
            if (b < k) {
                row[b] = (logf(s0[b]) + p0[b]) - (logf(s1[b]) + p1[b]);
            }
        }
    }
}
