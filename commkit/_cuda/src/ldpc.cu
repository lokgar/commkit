// Layered belief-propagation decoding of LDPC codes, one thread block per
// codeword.
//
// Device version of the Numba decoder `ldpc_layered` (coding/ldpc.py), its
// CPU reference.  The parity checks are stored row by row (CSR) in layer
// order: the rows of a layer share no variable, so the threads of a block
// update the rows of one layer in parallel without races, and a barrier
// separates the layers.  Each row makes two passes over its edges, as the
// CPU decoder: the first gathers Q = APP - R_old (min1, min2, argmin and the
// sign parity for min-sum; the sum of phi(|Q|) for sum-product), the second
// recomputes Q and writes R_new and APP = Q + R_new.  Rows and layers run in
// the same order as on the CPU, so min-sum gives the same values bit for bit
// (explicit _rn intrinsics keep the compiler from contracting into FMAs);
// sum-product differs by the rounding of logf/tanhf.
//
// Early stop: after each iteration (and before the first) the parities of
// the hard decisions are reduced with __syncthreads_or; a codeword whose
// checks are all satisfied stops.  `ok` is uniform across the block, so the
// loop and its barriers are too.
//
// Layout contract (enforced by the Python wrapper):
//   app        (B, N) float32 - channel LLRs in, a-posteriori LLRs out
//   msg        (B, E) float32 - scratch: check-to-variable messages
//   row_ptr    (m + 1,) int32 - CSR row offsets, rows in layer order
//   cols       (E,) int32     - variable of each edge
//   layer_ptr  (L + 1,) int32 - first row of each layer
//   iters      (B,) int32     - iterations run (out)
//   valid      (B,) uint8     - all checks satisfied (out)
//
// Launch contract: grid = (B, 1, 1), block = (LDPC_THREADS, 1, 1), B >= 1;
// every row has at least two edges.

#define LDPC_THREADS 256
#define PHI_MIN 1e-7f
#define PHI_MAX 30.0f

__device__ __forceinline__ float phi(float x) {
    x = fminf(fmaxf(x, PHI_MIN), PHI_MAX);
    return -logf(tanhf(0.5f * x));
}

__device__ __forceinline__ void min_sum_row(float* a, float* r, const int* cols,
                                            int e0, int e1, float scale,
                                            float offset) {
    float min1 = __int_as_float(0x7f800000), min2 = min1;  // +inf
    int idx = -1, neg = 0;
    for (int e = e0; e < e1; ++e) {
        const float q = __fsub_rn(a[cols[e]], r[e]);
        neg ^= (q < 0.0f);
        const float mag = fabsf(q);
        if (mag < min1) {
            min2 = min1;
            min1 = mag;
            idx = e;
        } else if (mag < min2) {
            min2 = mag;
        }
    }
    const float m1 = fmaxf(__fsub_rn(__fmul_rn(scale, min1), offset), 0.0f);
    const float m2 = fmaxf(__fsub_rn(__fmul_rn(scale, min2), offset), 0.0f);
    for (int e = e0; e < e1; ++e) {
        const int c = cols[e];
        const float q = __fsub_rn(a[c], r[e]);
        const float mag = (e == idx) ? m2 : m1;
        const float rn = (neg ^ (q < 0.0f)) ? -mag : mag;
        a[c] = __fadd_rn(q, rn);
        r[e] = rn;
    }
}

__device__ __forceinline__ void sum_product_row(float* a, float* r,
                                                const int* cols, int e0, int e1) {
    float total = 0.0f;
    int neg = 0;
    for (int e = e0; e < e1; ++e) {
        const float q = __fsub_rn(a[cols[e]], r[e]);
        neg ^= (q < 0.0f);
        total = __fadd_rn(total, phi(fabsf(q)));
    }
    for (int e = e0; e < e1; ++e) {
        const int c = cols[e];
        const float q = __fsub_rn(a[c], r[e]);
        const float mag = phi(__fsub_rn(total, phi(fabsf(q))));
        const float rn = (neg ^ (q < 0.0f)) ? -mag : mag;
        a[c] = __fadd_rn(q, rn);
        r[e] = rn;
    }
}

// True in every thread when all checks of the codeword are satisfied.
__device__ __forceinline__ bool satisfied(const float* a, const int* row_ptr,
                                          const int* cols, int m) {
    int bad = 0;
    for (int row = threadIdx.x; row < m; row += blockDim.x) {
        int parity = 0;
        for (int e = row_ptr[row]; e < row_ptr[row + 1]; ++e) {
            parity ^= (a[cols[e]] < 0.0f);
        }
        bad |= parity;
    }
    return !__syncthreads_or(bad);
}

template <bool SUM_PRODUCT>
__global__ void ldpc_layered(float* __restrict__ app,
                             float* __restrict__ msg,
                             const int* __restrict__ row_ptr,
                             const int* __restrict__ cols,
                             const int* __restrict__ layer_ptr,
                             const int num_layers,
                             const int m,
                             const int N,
                             const int E,
                             const float scale,
                             const float offset,
                             const int max_iterations,
                             const int early_stop,
                             int* __restrict__ iters,
                             unsigned char* __restrict__ valid) {
    const long long b = blockIdx.x;
    float* a = app + b * N;
    float* r = msg + b * E;
    for (int e = threadIdx.x; e < E; e += blockDim.x) {
        r[e] = 0.0f;
    }
    __syncthreads();

    int it = 0;
    bool ok = early_stop && satisfied(a, row_ptr, cols, m);
    while (!ok && it < max_iterations) {
        for (int layer = 0; layer < num_layers; ++layer) {
            const int end = layer_ptr[layer + 1];
            for (int row = layer_ptr[layer] + threadIdx.x; row < end;
                 row += blockDim.x) {
                if (SUM_PRODUCT) {
                    sum_product_row(a, r, cols, row_ptr[row], row_ptr[row + 1]);
                } else {
                    min_sum_row(a, r, cols, row_ptr[row], row_ptr[row + 1], scale,
                                offset);
                }
            }
            __syncthreads();
        }
        ++it;
        if (early_stop) {
            ok = satisfied(a, row_ptr, cols, m);
        }
    }
    if (!early_stop) {
        ok = satisfied(a, row_ptr, cols, m);
    }
    if (threadIdx.x == 0) {
        iters[b] = it;
        valid[b] = ok ? 1 : 0;
    }
}
