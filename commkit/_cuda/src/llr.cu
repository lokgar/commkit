// Bit log-likelihood ratios, one thread per symbol.
//
// Device port of the Numba kernel in `mapping/llr.py`, its CPU reference:
// per symbol, the metric m(s) = -|x - s|^2 / sigma^2 + log P(s) of every
// point.  Per bit, the LLR is the max over the points whose bit is 0 minus the max
// over those whose bit is 1 (max-log), or the difference of the log-sums of
// exp(m) of the two halves (exact).  The exact sums take e = exp(m - peak)
// once per point, with the symbol's peak metric (which cancels in the
// difference), so one exponential per point instead of one per point and
// bit; a half-sum below LLR_EXACT_FLOOR (_EXACT_FLOOR in llr.py) is redone
// relative to the half's own maximum, as in the Numba kernel.  The per-bit
// maxima and sums stay in registers, so no (N, M) or (N, k, M/2)
// intermediate is formed.  Each block stages its LLRs in shared memory and
// stores them contiguously (coalesced) instead of k values per thread at
// stride k.  float32 with explicit literals; the tables are read through
// the cache (every thread walks them in the same order).
//
// Layout contract (enforced by the Python wrapper):
//   x       (n,)    complex64 - received symbols
//   pts     (M,)    complex64 - constellation points
//   log_pmf (M,)    float32   - log prior (zeros for uniform)
//   labels  (M,)    int32     - bit label of each point, MSB first
//   out     (n, k)  float32   - LLRs (output)
//
// Launch contract: 1-D grid of 1-D blocks of LLR_THREADS threads, any grid
// size (grid-stride loop over tiles of LLR_THREADS symbols); dynamic shared
// memory LLR_THREADS * llr_stride(k) floats; 1 <= k <= LLR_MAX_K.

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

// LLRs of the kb bits labeled by `labels` over the M points of `met`; bit b
// goes to dst[b].
template <class Metric>
__device__ __forceinline__ void llrs_of(const Metric& met,
                                        const int M,
                                        const int* __restrict__ labels,
                                        const int kb,
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
                dst[b] = p0[b] - p1[b];
            }
        }
        return;
    }

    // Every point lies in one half of bit 0: its two maxima give the peak.
    const float peak = fmaxf(p0[0], p1[0]);
    float s0[LLR_MAX_K];
    float s1[LLR_MAX_K];
#pragma unroll
    for (int b = 0; b < LLR_MAX_K; ++b) {
        s0[b] = 0.0f;
        s1[b] = 0.0f;
    }
    for (int m = 0; m < M; ++m) {
        const float e = expf(met(m) - peak);
        const int lab = labels[m];
#pragma unroll
        for (int b = 0; b < LLR_MAX_K; ++b) {
            if (b < kb) {
                if ((lab >> (kb - 1 - b)) & 1) {
                    s1[b] += e;
                } else {
                    s0[b] += e;
                }
            }
        }
    }

    // Halves that underflowed relative to the peak: bit b of under0/under1.
    unsigned under0 = 0u;
    unsigned under1 = 0u;
#pragma unroll
    for (int b = 0; b < LLR_MAX_K; ++b) {
        if (b < kb) {
            if (!(s0[b] >= LLR_EXACT_FLOOR)) {
                under0 |= 1u << b;
                s0[b] = 0.0f;
            }
            if (!(s1[b] >= LLR_EXACT_FLOOR)) {
                under1 |= 1u << b;
                s1[b] = 0.0f;
            }
        }
    }
    if (under0 | under1) {
        for (int m = 0; m < M; ++m) {
            const float v = met(m);
            const int lab = labels[m];
#pragma unroll
            for (int b = 0; b < LLR_MAX_K; ++b) {
                if (b < kb) {
                    if ((lab >> (kb - 1 - b)) & 1) {
                        if ((under1 >> b) & 1u) {
                            s1[b] += expf(v - p1[b]);
                        }
                    } else if ((under0 >> b) & 1u) {
                        s0[b] += expf(v - p0[b]);
                    }
                }
            }
        }
    }
#pragma unroll
    for (int b = 0; b < LLR_MAX_K; ++b) {
        if (b < kb) {
            // log of each half-sum, relative to the peak.
            const float l0 = logf(s0[b]) + (((under0 >> b) & 1u) ? p0[b] - peak : 0.0f);
            const float l1 = logf(s1[b]) + (((under1 >> b) & 1u) ? p1[b] - peak : 0.0f);
            dst[b] = l0 - l1;
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
            llrs_of(met, M, labels, k, exact, row);
        }
        __syncthreads();
        llr_store_tile(tile, out, base, static_cast<int>(min(static_cast<long long>(blockDim.x), n - base)), k);
        __syncthreads();
    }
}
