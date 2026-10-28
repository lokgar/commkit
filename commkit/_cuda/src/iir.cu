// IIR filtering in second-order sections, block-parallel over the record.
//
// The device port of the Numba kernel `_sos_loop` in `filtering/linear.py`,
// its CPU reference: transposed direct form II, the form of
// scipy.signal.sosfilt, through all S sections per sample.  The samples are
// read and written in their own type T (float, double, complex<float>,
// complex<double>); the state and the arithmetic are in double precision
// (U = double or complex<double>), because single-precision sections are
// unstable at low cutoffs (poles near z = 1).
//
// A linear recursion over the record has no parallel work per step, but its
// state after a block is linear in the state before it:
//
//   z_out = A · z_in + z_end,
//
// with A the zero-input transition of the cascade over the L samples of a
// block (2S x 2S, real, computed on the host) and z_end the state the block
// reaches from zero.  The carry is itself split in two levels, over groups
// of G blocks, so that no thread walks the whole record:
//
//   iir_block_state  one thread per (channel, block): z_end of each block;
//   iir_group_carry  one thread per (channel, group): from zero through the
//                    group, the local start state of each block and the
//                    group's end state;
//   iir_carry        one thread per channel: the start state of each group,
//                    from zi, with the transition over a group;
//   iir_block_out    one thread per (channel, block): its start state, the
//                    local one plus P[j] times its group's (P[j] the
//                    transition over j blocks), then the block again,
//                    writing the output.
//
// The record is read twice and written once in its own type; there is no
// double-precision copy of it.
//
// Layout contract (enforced by the Python wrapper):
//   sos   (S, 6)          double - b0 b1 b2 a0 a1 a2 per section, a0 == 1
//   x, y  (C, N)          T      - C-contiguous
//   zi    (C, S, 2)       U      - initial state of each channel
//   A     (2S, 2S)        double - state transition over L samples, row
//                                  major, state index 2k + i
//   AG    (2S, 2S)        double - the same over G blocks (L * G samples)
//   P     (G, 2S, 2S)     double - the same over j blocks, j = 0 .. G-1
//   zend, zloc (C, nb, S, 2) U   - per-block states, nb = ceil(N / L)
//   zgend, zgin (C, ng, S, 2) U  - per-group states, ng = ceil(nb / G)
// Launch contract: 1-D grids of 1-D blocks; iir_block_state and
// iir_block_out cover C * nb threads, iir_group_carry C * ng, iir_carry C;
// S <= IIR_MAX_S.

#include <cupy/complex.cuh>

#define IIR_MAX_S 16

template <typename U>
__device__ __forceinline__ U iir_promote(const float v) { return U(double(v)); }
template <typename U>
__device__ __forceinline__ U iir_promote(const double v) { return U(v); }
template <typename U>
__device__ __forceinline__ U iir_promote(const complex<float> v) {
    return U(double(v.real()), double(v.imag()));
}
template <typename U>
__device__ __forceinline__ U iir_promote(const complex<double> v) { return U(v); }

template <typename T>
__device__ __forceinline__ T iir_demote(const double v) { return T(v); }
template <typename T>
__device__ __forceinline__ T iir_demote(const complex<double> v) {
    return T(v.real(), v.imag());
}

// One sample through the S sections, as `_sos_loop`.
template <typename U>
__device__ __forceinline__ U iir_step(const double* __restrict__ sos,
                                      const int S,
                                      U v,
                                      U* z0,
                                      U* z1) {
    for (int k = 0; k < S; ++k) {
        const double* c = sos + 6 * k;
        const U out = c[0] * v + z0[k];
        z0[k] = c[1] * v - c[4] * out + z1[k];
        z1[k] = c[2] * v - c[5] * out;
        v = out;
    }
    return v;
}

template <typename T, typename U>
__global__ void iir_block_state(const double* __restrict__ sos,
                                const int S,
                                const T* __restrict__ x,
                                const long long N,
                                const int L,
                                const int nb,
                                const int C,
                                U* __restrict__ zend) {
    const long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (tid >= (long long)C * nb) return;
    const int c = (int)(tid / nb);
    const int b = (int)(tid % nb);
    U z0[IIR_MAX_S];
    U z1[IIR_MAX_S];
    for (int k = 0; k < S; ++k) {
        z0[k] = U(0.0);
        z1[k] = U(0.0);
    }
    const long long start = (long long)b * L;
    const long long stop = min(start + L, N);
    const T* row = x + (long long)c * N;
    for (long long n = start; n < stop; ++n) {
        iir_step(sos, S, iir_promote<U>(row[n]), z0, z1);
    }
    U* dst = zend + tid * 2 * S;
    for (int k = 0; k < S; ++k) {
        dst[2 * k] = z0[k];
        dst[2 * k + 1] = z1[k];
    }
}

template <typename U>
__global__ void iir_group_carry(const double* __restrict__ A,
                                const int S,
                                const U* __restrict__ zend,
                                const int nb,
                                const int G,
                                const int ng,
                                const int C,
                                U* __restrict__ zloc,
                                U* __restrict__ zgend) {
    const long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (tid >= (long long)C * ng) return;
    const int c = (int)(tid / ng);
    const int g = (int)(tid % ng);
    const int D = 2 * S;
    U z[2 * IIR_MAX_S];
    U t[2 * IIR_MAX_S];
    for (int j = 0; j < D; ++j) z[j] = U(0.0);
    const int stop = min((g + 1) * G, nb);
    for (int b = g * G; b < stop; ++b) {
        const long long off = ((long long)c * nb + b) * D;
        for (int j = 0; j < D; ++j) zloc[off + j] = z[j];
        // z <- A z + zend[c, b]
        for (int i = 0; i < D; ++i) {
            U acc = zend[off + i];
            for (int j = 0; j < D; ++j) acc += A[i * D + j] * z[j];
            t[i] = acc;
        }
        for (int j = 0; j < D; ++j) z[j] = t[j];
    }
    for (int j = 0; j < D; ++j) zgend[tid * D + j] = z[j];
}

template <typename U>
__global__ void iir_carry(const double* __restrict__ A,
                          const int S,
                          const U* __restrict__ zi,
                          const U* __restrict__ zend,
                          const int nb,
                          const int C,
                          U* __restrict__ zin) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const int D = 2 * S;
    U z[2 * IIR_MAX_S];
    U t[2 * IIR_MAX_S];
    for (int j = 0; j < D; ++j) z[j] = zi[(long long)c * D + j];
    for (int b = 0; b < nb; ++b) {
        const long long off = ((long long)c * nb + b) * D;
        for (int j = 0; j < D; ++j) zin[off + j] = z[j];
        // z <- A z + zend[c, b]
        for (int i = 0; i < D; ++i) {
            U acc = zend[off + i];
            for (int j = 0; j < D; ++j) acc += A[i * D + j] * z[j];
            t[i] = acc;
        }
        for (int j = 0; j < D; ++j) z[j] = t[j];
    }
}

template <typename T, typename U>
__global__ void iir_block_out(const double* __restrict__ sos,
                              const int S,
                              const T* __restrict__ x,
                              const long long N,
                              const int L,
                              const int nb,
                              const int C,
                              const double* __restrict__ P,
                              const int G,
                              const int ng,
                              const U* __restrict__ zloc,
                              const U* __restrict__ zgin,
                              T* __restrict__ y) {
    const long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (tid >= (long long)C * nb) return;
    const int c = (int)(tid / nb);
    const int b = (int)(tid % nb);
    const int D = 2 * S;
    // Start state: the local one plus P[j] times the group's start state.
    const double* Pj = P + (long long)(b % G) * D * D;
    const U* zl = zloc + tid * D;
    const U* zg = zgin + ((long long)c * ng + b / G) * D;
    U z0[IIR_MAX_S];
    U z1[IIR_MAX_S];
    for (int k = 0; k < S; ++k) {
        U a0 = zl[2 * k];
        U a1 = zl[2 * k + 1];
        for (int j = 0; j < D; ++j) {
            a0 += Pj[(2 * k) * D + j] * zg[j];
            a1 += Pj[(2 * k + 1) * D + j] * zg[j];
        }
        z0[k] = a0;
        z1[k] = a1;
    }
    const long long start = (long long)b * L;
    const long long stop = min(start + L, N);
    const T* row = x + (long long)c * N;
    T* out = y + (long long)c * N;
    for (long long n = start; n < stop; ++n) {
        out[n] = iir_demote<T>(iir_step(sos, S, iir_promote<U>(row[n]), z0, z1));
    }
}
