// Fused minimum-squared-distance kernel for Blind Phase Search (BPS) and
// decision-directed slicing.
//
// Computes, in a single pass over the input symbols:
//
//     min_d2[p, c, n] = min_m | x[c, n] * phasor[p] - const[m] |^2
//
// replacing the materialized (.., P/B, M) candidate-distance tensors of the
// array-module implementations. The input symbols are read once and only the
// (P, C, N) float32 minima (plus optional int32 argmin indices) are written,
// removing the elementwise-chain DRAM traffic that dominates the xp path.
//
// Template modes
//   MODE = MODE_TABLE : general constellation search over const[0..M-1],
//                       cooperatively staged in shared memory.
//   MODE = MODE_GRID  : square-QAM O(1) nearest point by per-component
//                       rounding onto the uniform level grid
//                       lev_min + k * d_grid, k in [0, side-1].
//   RETURN_ARGMIN     : additionally writes the index of the nearest point
//                       (TABLE: table index m; GRID: re_idx * side + im_idx).
//
// Layout contract (enforced by the Python wrapper):
//   x        (C, N) complex64, C-contiguous, time on the last axis
//   phasor   (P,)   complex64, P <= 128
//   constel  (M,)   complex64, M <= 1024 (TABLE mode only)
//   min_d2   (P, C, N) float32
//   argmin   (P, C, N) int32   (RETURN_ARGMIN only)
//
// Launch contract: block = (BLOCK_N, 1, 1); grid = (ceil(N/BLOCK_N), P, C);
// dynamic shared memory = M * sizeof(complex<float>) in TABLE mode, 0 in GRID.
// Arithmetic is FP32 throughout, matching the float32 metric of the xp path.

#include <cupy/complex.cuh>

#define MODE_TABLE 0
#define MODE_GRID 1

template <int MODE, bool RETURN_ARGMIN>
__global__ void bps_min_d2(const complex<float>* __restrict__ x,
                           const complex<float>* __restrict__ phasor,
                           const complex<float>* __restrict__ constel,
                           const float lev_min,
                           const float d_grid,
                           const int side,
                           const int M,
                           const long long N,
                           float* __restrict__ min_d2,
                           int* __restrict__ argmin_out) {
    extern __shared__ complex<float> s_const[];

    const int p = blockIdx.y;
    const int c = blockIdx.z;

    if (MODE == MODE_TABLE) {
        for (int m = threadIdx.x; m < M; m += blockDim.x) {
            s_const[m] = constel[m];
        }
        __syncthreads();
    }

    const long long n =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (n >= N) {
        return;
    }

    // Coalesced in n; the per-block phasor load is an L1-cached broadcast.
    const complex<float> xr = x[static_cast<long long>(c) * N + n] * phasor[p];
    const float re = xr.real();
    const float im = xr.imag();

    float best = 3.402823466e+38f;  // FLT_MAX
    int best_m = 0;

    if (MODE == MODE_TABLE) {
        // Warp-uniform m => shared-memory broadcast, no bank conflicts.
        for (int m = 0; m < M; ++m) {
            const float dr = re - s_const[m].real();
            const float di = im - s_const[m].imag();
            const float d2 = fmaf(dr, dr, di * di);
            if (d2 < best) {
                best = d2;
                best_m = m;
            }
        }
    } else {  // MODE_GRID
        const float hi = static_cast<float>(side - 1);
        const float ri = fminf(fmaxf(roundf((re - lev_min) / d_grid), 0.0f), hi);
        const float ii = fminf(fmaxf(roundf((im - lev_min) / d_grid), 0.0f), hi);
        const float dr = re - fmaf(ri, d_grid, lev_min);
        const float di = im - fmaf(ii, d_grid, lev_min);
        best = fmaf(dr, dr, di * di);
        if (RETURN_ARGMIN) {
            best_m = static_cast<int>(ri) * side + static_cast<int>(ii);
        }
    }

    const long long out_idx =
        (static_cast<long long>(p) * gridDim.z + c) * N + n;
    min_d2[out_idx] = best;
    if (RETURN_ARGMIN) {
        argmin_out[out_idx] = best_m;
    }
}
