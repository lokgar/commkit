// Bit log-likelihood ratios, one thread per symbol.
//
// Device ports of the Numba kernels in `mapping/llr.py`, their CPU
// references:
//
//   llr            the M metrics m(s) = -|x - s|^2 / sigma^2 + log P(s) of
//                  every point;
//   llr_separable  the 2 sqrt(M) metrics of a separable square lattice, the
//                  real levels of each axis (every bit depends on one axis,
//                  and the other axis's factor cancels from its LLR);
//   llr_separable_maxlog
//                  max-log LLRs of a uniform separable lattice in closed
//                  form, 1 + 2k metrics per axis: the metric of the snapped
//                  level against the better of the two levels just outside
//                  its run of equal bits (see mapping/llr.py).
//
// Per bit, the LLR is the max over the points whose bit is 0 minus the max
// over those whose bit is 1 (max-log), or the difference of the log-sums of
// exp(m) of the two halves (exact).  The exact sums take e = exp(m - peak)
// once per point, with the symbol's peak metric (which cancels in the
// difference), so one exponential per point instead of one per point and
// bit.  A half whose maximum lies so far below the peak that its sum could
// fall below LLR_EXACT_FLOOR (_EXACT_FLOOR in llr.py) is summed relative to
// its own maximum instead, in the same pass: the maxima of the first pass
// tell which halves those are, so no symbol needs a third pass (which sent
// its whole warp through it).  The Numba kernel redoes a half-sum that
// fell below the floor, which agrees to float rounding.  The exponentials
// are the hardware __expf (about 2 ulp): the LLRs move by under 4e-7
// relative to expf, far below the tolerance against the Numba reference,
// and the exact pass runs 1.25-1.5x faster.  The per-bit
// maxima and sums stay in registers, so no (N, M) or (N, k, M/2)
// intermediate is formed.  Each block stages its LLRs in shared memory and
// stores them contiguously (coalesced) instead of k values per thread at
// stride k.  float32 with explicit literals; the tables are read through
// the cache (every thread walks them in the same order).
//
// Layout contract (enforced by the Python wrappers):
//   llr:
//     x       (n,)    complex64 - received symbols
//     pts     (M,)    complex64 - constellation points
//     log_pmf (M,)    float32   - log prior (zeros for uniform)
//     labels  (M,)    int32     - bit label of each point, MSB first
//   llr_separable, per axis a (re, im):
//     lev_a   (L_a,)  float32   - levels of the axis
//     logp_a  (L_a,)  float32   - log marginal prior of the levels
//     lab_a   (L_a,)  int32     - label of each level over the axis's k_a
//                                 bits, MSB first
//     pos_a   (k_a,)  int32     - position of each of those bits in the
//                                 symbol's label
//   llr_separable_maxlog, per axis a: lev_a, lab_a and pos_a as above, and
//     lo_a, hi_a (k_a, L_a) int32 - per bit and level, the nearest level
//                                 below / above with the other bit value,
//                                 -1 for none; L_a >= 2, levels ascending
//                                 and equally spaced
//   out       (n, k)  float32   - LLRs (output)
//
// Launch contract: 1-D grid of 1-D blocks of LLR_THREADS threads, any grid
// size (grid-stride loop over tiles of LLR_THREADS symbols); dynamic shared
// memory LLR_THREADS * llr_stride(k) floats; 1 <= k <= LLR_MAX_K and
// 1 <= k_a <= LLR_MAX_K.

#include <cupy/complex.cuh>

#define LLR_MAX_K 16
#define LLR_THREADS 128
#define LLR_INF __int_as_float(0x7f800000)
#define LLR_EXACT_FLOOR 1e-26f

// Row stride of the shared staging tile: odd, so that thread t's row
// t * stride falls in its own bank.
__device__ __forceinline__ int llr_stride(const int k) { return k | 1; }

struct PointMetric {
    const complex<float>* pts;
    const float* log_pmf;
    float xr;
    float xi;
    float inv_s2;

    __device__ __forceinline__ float operator()(const int m) const {
        const complex<float> s = pts[m];
        const float dr = xr - s.real();
        const float di = xi - s.imag();
        return -(dr * dr + di * di) * inv_s2 + log_pmf[m];
    }
};

struct LevelMetric {
    const float* lev;
    const float* logp;
    float x;
    float inv_s2;

    __device__ __forceinline__ float operator()(const int m) const {
        const float d = x - lev[m];
        return -(d * d) * inv_s2 + logp[m];
    }
};

// LLRs of the kb bits labeled by `labels` over the M points of `met`; bit b
// goes to dst[pos ? pos[b] : b].
template <class Metric>
__device__ __forceinline__ void llrs_of(const Metric& met,
                                        const int M,
                                        const int* __restrict__ labels,
                                        const int kb,
                                        const int* __restrict__ pos,
                                        const int exact,
                                        float* dst) {
    float p0[LLR_MAX_K];
    float p1[LLR_MAX_K];
#pragma unroll
    for (int b = 0; b < LLR_MAX_K; ++b) {
        p0[b] = -LLR_INF;
        p1[b] = -LLR_INF;
    }
    for (int m = 0; m < M; ++m) {
        const float v = met(m);
        const int lab = labels[m];
#pragma unroll
        for (int b = 0; b < LLR_MAX_K; ++b) {
            if (b < kb) {
                if ((lab >> (kb - 1 - b)) & 1) {
                    p1[b] = fmaxf(p1[b], v);
                } else {
                    p0[b] = fmaxf(p0[b], v);
                }
            }
        }
    }
    if (!exact) {
#pragma unroll
        for (int b = 0; b < LLR_MAX_K; ++b) {
            if (b < kb) {
                dst[pos ? pos[b] : b] = p0[b] - p1[b];
            }
        }
        return;
    }

    // Every point lies in one half of bit 0: its two maxima give the peak.
    // Halves whose maximum is more than -log(LLR_EXACT_FLOOR) below it
    // (bit b of weak0/weak1) are summed relative to that maximum.
    const float peak = fmaxf(p0[0], p1[0]);
    const float log_floor = logf(LLR_EXACT_FLOOR);
    unsigned weak0 = 0u;
    unsigned weak1 = 0u;
    float s0[LLR_MAX_K];
    float s1[LLR_MAX_K];
#pragma unroll
    for (int b = 0; b < LLR_MAX_K; ++b) {
        s0[b] = 0.0f;
        s1[b] = 0.0f;
        if (b < kb) {
            weak0 |= (p0[b] - peak < log_floor ? 1u : 0u) << b;
            weak1 |= (p1[b] - peak < log_floor ? 1u : 0u) << b;
        }
    }
    for (int m = 0; m < M; ++m) {
        const float v = met(m);
        const float e = __expf(v - peak);
        const int lab = labels[m];
#pragma unroll
        for (int b = 0; b < LLR_MAX_K; ++b) {
            if (b < kb) {
                if ((lab >> (kb - 1 - b)) & 1) {
                    s1[b] += ((weak1 >> b) & 1u) ? __expf(v - p1[b]) : e;
                } else {
                    s0[b] += ((weak0 >> b) & 1u) ? __expf(v - p0[b]) : e;
                }
            }
        }
    }
#pragma unroll
    for (int b = 0; b < LLR_MAX_K; ++b) {
        if (b < kb) {
            // log of each half-sum, relative to the peak.
            const float l0 = logf(s0[b]) + (((weak0 >> b) & 1u) ? p0[b] - peak : 0.0f);
            const float l1 = logf(s1[b]) + (((weak1 >> b) & 1u) ? p1[b] - peak : 0.0f);
            dst[pos ? pos[b] : b] = l0 - l1;
        }
    }
}

// Copies the tile of `count` symbols staged in `tile` (row stride
// llr_stride(k)) to out[base * k ...], contiguously.
__device__ __forceinline__ void llr_store_tile(const float* tile,
                                               float* __restrict__ out,
                                               const long long base,
                                               const int count,
                                               const int k) {
    const int stride = llr_stride(k);
    const int total = count * k;
    float* dst = out + base * k;
    for (int j = threadIdx.x; j < total; j += blockDim.x) {
        dst[j] = tile[(j / k) * stride + j % k];
    }
}

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
    extern __shared__ float tile[];
    float* row = tile + threadIdx.x * llr_stride(k);
    for (long long base = static_cast<long long>(blockIdx.x) * blockDim.x; base < n;
         base += static_cast<long long>(gridDim.x) * blockDim.x) {
        const long long i = base + threadIdx.x;
        if (i < n) {
            const PointMetric met{pts, log_pmf, x[i].real(), x[i].imag(), inv_s2};
            llrs_of(met, M, labels, k, nullptr, exact, row);
        }
        __syncthreads();
        llr_store_tile(tile, out, base, static_cast<int>(min(static_cast<long long>(blockDim.x), n - base)), k);
        __syncthreads();
    }
}

extern "C" __global__ void llr_separable(const complex<float>* __restrict__ x,
                                         const float* __restrict__ lev_re,
                                         const float* __restrict__ logp_re,
                                         const int* __restrict__ lab_re,
                                         const int* __restrict__ pos_re,
                                         const int L_re,
                                         const int k_re,
                                         const float* __restrict__ lev_im,
                                         const float* __restrict__ logp_im,
                                         const int* __restrict__ lab_im,
                                         const int* __restrict__ pos_im,
                                         const int L_im,
                                         const int k_im,
                                         float* __restrict__ out,
                                         const int n,
                                         const int k,
                                         const float inv_s2,
                                         const int exact) {
    extern __shared__ float tile[];
    float* row = tile + threadIdx.x * llr_stride(k);
    for (long long base = static_cast<long long>(blockIdx.x) * blockDim.x; base < n;
         base += static_cast<long long>(gridDim.x) * blockDim.x) {
        const long long i = base + threadIdx.x;
        if (i < n) {
            const complex<float> xi = x[i];
            const LevelMetric met_re{lev_re, logp_re, xi.real(), inv_s2};
            llrs_of(met_re, L_re, lab_re, k_re, pos_re, exact, row);
            const LevelMetric met_im{lev_im, logp_im, xi.imag(), inv_s2};
            llrs_of(met_im, L_im, lab_im, k_im, pos_im, exact, row);
        }
        __syncthreads();
        llr_store_tile(tile, out, base, static_cast<int>(min(static_cast<long long>(blockDim.x), n - base)), k);
        __syncthreads();
    }
}

// Closed-form max-log LLRs of one axis's bits into row[pos[b]].
__device__ __forceinline__ void axis_maxlog(const float x,
                                            const float* __restrict__ lev,
                                            const int* __restrict__ lab,
                                            const int* __restrict__ lo,
                                            const int* __restrict__ hi,
                                            const int* __restrict__ pos,
                                            const int L,
                                            const int kb,
                                            const float inv_s2,
                                            float* row) {
    int s = __float2int_rn((x - lev[0]) / (lev[1] - lev[0]));
    s = min(max(s, 0), L - 1);
    float t = x - lev[s];
    // Near a midpoint the rounded quotient can land one level off: keep the
    // nearer of s and its neighbor towards x.
    const int s2 = t > 0.0f ? s + 1 : s - 1;
    if (s2 >= 0 && s2 < L) {
        const float t2 = x - lev[s2];
        if (t2 * t2 < t * t) {
            s = s2;
            t = t2;
        }
    }
    const float m_own = -(t * t) * inv_s2;
    const int label = lab[s];
    for (int b = 0; b < kb; ++b) {
        float m_other = -LLR_INF;
        int j = lo[b * L + s];
        if (j >= 0) {
            t = x - lev[j];
            m_other = fmaxf(m_other, -(t * t) * inv_s2);
        }
        j = hi[b * L + s];
        if (j >= 0) {
            t = x - lev[j];
            m_other = fmaxf(m_other, -(t * t) * inv_s2);
        }
        const bool one = (label >> (kb - 1 - b)) & 1;
        row[pos[b]] = one ? m_other - m_own : m_own - m_other;
    }
}

extern "C" __global__ void llr_separable_maxlog(const complex<float>* __restrict__ x,
                                                const float* __restrict__ lev_re,
                                                const int* __restrict__ lab_re,
                                                const int* __restrict__ lo_re,
                                                const int* __restrict__ hi_re,
                                                const int* __restrict__ pos_re,
                                                const int L_re,
                                                const int k_re,
                                                const float* __restrict__ lev_im,
                                                const int* __restrict__ lab_im,
                                                const int* __restrict__ lo_im,
                                                const int* __restrict__ hi_im,
                                                const int* __restrict__ pos_im,
                                                const int L_im,
                                                const int k_im,
                                                float* __restrict__ out,
                                                const int n,
                                                const int k,
                                                const float inv_s2) {
    extern __shared__ float tile[];
    float* row = tile + threadIdx.x * llr_stride(k);
    for (long long base = static_cast<long long>(blockIdx.x) * blockDim.x; base < n;
         base += static_cast<long long>(gridDim.x) * blockDim.x) {
        const long long i = base + threadIdx.x;
        if (i < n) {
            const complex<float> xi = x[i];
            axis_maxlog(xi.real(), lev_re, lab_re, lo_re, hi_re, pos_re, L_re, k_re, inv_s2, row);
            axis_maxlog(xi.imag(), lev_im, lab_im, lo_im, hi_im, pos_im, L_im, k_im, inv_s2, row);
        }
        __syncthreads();
        llr_store_tile(tile, out, base, static_cast<int>(min(static_cast<long long>(blockDim.x), n - base)), k);
        __syncthreads();
    }
}
