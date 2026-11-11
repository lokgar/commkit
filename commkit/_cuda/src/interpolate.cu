// Windowed-sinc interpolation at arbitrary instants, one thread per instant.
//
// The device port of the Numba kernels `interp_at` and `interp_piecewise`
// in `commkit/_numba_interpolation.py`, their CPU reference: a sinc of
// 2 * HALF taps under the 4-term Blackman-Harris window
//
//   w(u) = a0 - a1 cos φ + a2 cos 2φ - a3 cos 3φ,  φ = π (u / HALF + 1),
//
// for a tap at distance u = t - μ from the instant, written through
// cos 2φ and cos 3φ in terms of cos φ (Chebyshev), and with
// sin(π(t - μ)) = -(-1)^t sin(πμ), taken on the nearer side of 1.
// Samples outside 0 .. N - 1 count as zero.
//
// The instant is formed in double precision (the knots' interpolation, or
// the index plus the fraction), so its accuracy does not fall with the
// record length; the taps and the sums are in the samples' real type R
// (float for complex64, as the transform keeps the input's precision), the
// window's cosine by cospi per tap, so no recurrence error accumulates in
// single precision.  Channels sharing the instants (MODE_AT, or one knot
// row) reuse the thread's taps.
//
// Layout contract (enforced by the Python wrapper):
//   x      (C, N)     T       - C-contiguous samples
//   y      (C, M)     T       - output
//   MODE_AT:        base (M,) int64, frac (M,) R in [0, 1)
//   MODE_PIECEWISE: knots (rows, K) double, rows 1 or C, M = sub * (K - 1);
//                   instant m = k * sub + j at
//                   knots[r, k] + j * ((knots[r, k + 1] - knots[r, k]) / sub)
// Launch contract: a 1-D grid of 1-D blocks covering M threads.

#include <cupy/complex.cuh>

#define MODE_AT 0
#define MODE_PIECEWISE 1

__device__ __forceinline__ float interp_sinpi(const float v) { return sinpif(v); }
__device__ __forceinline__ double interp_sinpi(const double v) { return sinpi(v); }
__device__ __forceinline__ float interp_cospi(const float v) { return cospif(v); }
__device__ __forceinline__ double interp_cospi(const double v) { return cospi(v); }

template <typename T, typename R>
struct interp_value {
    // Real samples: one accumulator.
    __device__ static __forceinline__ void add(R& re, R&, const T v, const R h) {
        re += h * v;
    }
    __device__ static __forceinline__ T make(const R re, const R) { return T(re); }
};

template <typename R>
struct interp_value<complex<R>, R> {
    __device__ static __forceinline__ void add(R& re, R& im, const complex<R> v,
                                               const R h) {
        re += h * v.real();
        im += h * v.imag();
    }
    __device__ static __forceinline__ complex<R> make(const R re, const R im) {
        return complex<R>(re, im);
    }
};

// The 2 * HALF taps of the fraction mu in [0, 1): h[t + HALF - 1] for the
// sample at offset t = 1 - HALF .. HALF from the instant's index.
template <typename R, int HALF>
__device__ __forceinline__ void interp_taps(const R mu, R* h) {
    const R a0 = R(0.35875), a1 = R(0.48829), a2 = R(0.14128), a3 = R(0.01168);
    if (mu == R(0)) {
#pragma unroll
        for (int i = 0; i < 2 * HALF; ++i) h[i] = R(0);
        h[HALF - 1] = R(1);
        return;
    }
    const R s = interp_sinpi(mu < R(1) - mu ? mu : R(1) - mu);
    const R pi = R(3.14159265358979323846);
#pragma unroll
    for (int i = 0; i < 2 * HALF; ++i) {
        const int t = i + 1 - HALF;
        const R u = R(t) - mu;
        const R c1 = interp_cospi(u / R(HALF) + R(1));
        const R window =
            (a0 - a2) - (a1 - R(3) * a3) * c1 + c1 * c1 * (R(2) * a2 - R(4) * a3 * c1);
        const R neg_alt = (t & 1) ? R(1) : R(-1);  // -(-1)^t
        h[i] = neg_alt * s / (pi * u) * window;
    }
}

template <typename T, typename R, int HALF>
__device__ __forceinline__ T interp_apply(const T* __restrict__ row, const long long n,
                                          const long long i, const R* h) {
    R re = R(0), im = R(0);
    const long long first = i + 1 - HALF;
    if (first >= 0 && i + HALF < n) {
#pragma unroll
        for (int t = 0; t < 2 * HALF; ++t)
            interp_value<T, R>::add(re, im, row[first + t], h[t]);
    } else {
#pragma unroll
        for (int t = 0; t < 2 * HALF; ++t) {
            const long long k = first + t;
            if (k >= 0 && k < n) interp_value<T, R>::add(re, im, row[k], h[t]);
        }
    }
    return interp_value<T, R>::make(re, im);
}

// Index and fraction of the instant p (double); a fraction that rounds to 1
// in R moves to the next sample.
template <typename R>
__device__ __forceinline__ void interp_split(const double p, long long& i, R& mu) {
    const double whole = floor(p);
    i = static_cast<long long>(whole);
    mu = static_cast<R>(p - whole);
    if (mu >= R(1)) {
        i += 1;
        mu = R(0);
    }
}

template <typename T, typename R, int HALF, int MODE>
__global__ void interpolate(const T* __restrict__ x, T* __restrict__ y,
                            const long long C, const long long N, const long long M,
                            const long long* __restrict__ base,
                            const R* __restrict__ frac,
                            const double* __restrict__ knots, const long long rows,
                            const long long K, const long long sub) {
    const long long m = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (m >= M) return;
    R h[2 * HALF];
    long long i = 0;
    R mu = R(0);
    if (MODE == MODE_AT) {
        i = base[m];
        mu = frac[m];
        if (mu >= R(1)) {
            i += 1;
            mu = R(0);
        }
        interp_taps<R, HALF>(mu, h);
        for (long long c = 0; c < C; ++c)
            y[c * M + m] = interp_apply<T, R, HALF>(x + c * N, N, i, h);
        return;
    }
    const long long k = m / sub;
    const long long j = m - k * sub;
    for (long long c = 0; c < C; ++c) {
        if (c == 0 || rows > 1) {
            const double* kr = knots + (rows > 1 ? c : 0) * K;
            const double a = kr[k];
            const double p = a + double(j) * ((kr[k + 1] - a) / double(sub));
            interp_split<R>(p, i, mu);
            interp_taps<R, HALF>(mu, h);
        }
        y[c * M + m] = interp_apply<T, R, HALF>(x + c * N, N, i, h);
    }
}
