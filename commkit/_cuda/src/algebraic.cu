// Algebraic decoding of BCH and Reed-Solomon codes from their syndromes,
// one thread per codeword.
//
// Device version of the Numba decoder `algebraic_correct`
// (coding/_numba_loops.py), its CPU reference, step for step: the
// Berlekamp-Massey recursion, a Chien search over the positions that stops
// at the locator's degree, and for Reed-Solomon Forney's formula at the
// roots.  A word is corrected in place only when all its steps succeed, so
// both give the same words and flags.  The field arithmetic is integer
// table lookups, exact on both.
//
// The per-codeword polynomials live in a global workspace laid out as
// (slot, B), so the threads of a warp access consecutive addresses; any t
// fits.  The Chien search advances each term's exponent by the position's
// log instead of reducing a product modulo q - 1.
//
// Layout contract (enforced by the Python wrapper):
//   s             (B, 2t) int64   - syndromes S_j as field elements
//   exp_t         (2q,) int64     - antilog table, doubled
//   log_t         (q,) int64      - log table (log_t[0] unused)
//   locator_logs  (n,) int64      - log of each position's locator X_p
//   work          (10 t + 3, B) int64 - scratch
//   words         (B, n m) int8   - hard words in, corrected words out
//                 (m = 1 for BCH: a position is a bit)
//   valid         (B,) uint8      - decoding succeeded (out)
//
// Launch contract: grid = (ceil(B / ALG_THREADS), 1, 1),
// block = (ALG_THREADS, 1, 1), B >= 1.

#define ALG_THREADS 128

typedef long long i64;

__device__ __forceinline__ i64 gf_mul(i64 a, i64 b, const i64* exp_t, const i64* log_t) {
    return (a == 0 || b == 0) ? 0 : exp_t[log_t[a] + log_t[b]];
}

// sum_j coef[j * B] alpha^(j x_log) over j < count, with 0 <= x_log < q1.
__device__ __forceinline__ i64 gf_eval(const i64* coef, int count, i64 x_log, i64 q1,
                                       int B, const i64* exp_t, const i64* log_t) {
    i64 acc = 0, power = 0;
    for (int j = 0; j < count; ++j) {
        const i64 c = coef[(i64)j * B];
        if (c != 0) acc ^= exp_t[log_t[c] + power];
        power += x_log;
        if (power >= q1) power -= q1;
    }
    return acc;
}

extern "C" __global__ void algebraic_correct(
    const i64* __restrict__ s, const i64* __restrict__ exp_t,
    const i64* __restrict__ log_t, const i64* __restrict__ locator_logs,
    i64* __restrict__ work, signed char* __restrict__ words,
    unsigned char* __restrict__ valid, int B, int n, int t, int first_root, int m,
    int binary, i64 q1) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= B) return;
    const int width = 2 * t + 1;
    const i64* syn = s + (i64)row * 2 * t;
    i64* locator = work + row;
    i64* previous = locator + (i64)width * B;
    i64* updated = previous + (i64)width * B;
    i64* omega = updated + (i64)width * B;
    i64* derivative = omega + (i64)t * B;
    i64* positions = derivative + (i64)t * B;
    i64* values = positions + (i64)t * B;
#define AT(a, i) a[(i64)(i) * B]
    for (int c = 0; c < width; ++c) {
        AT(locator, c) = 0;
        AT(previous, c) = 0;
    }
    AT(locator, 0) = 1;
    AT(previous, 0) = 1;
    int length = 0, shift = 1;
    i64 last = 1;
    for (int r = 0; r < 2 * t; ++r) {
        i64 d = 0;
        for (int i = 0; i <= r; ++i) d ^= gf_mul(AT(locator, i), syn[r - i], exp_t, log_t);
        const bool grow = d != 0 && 2 * length <= r;
        if (d != 0) {
            const i64 coef = exp_t[log_t[d] - log_t[last] + q1];
            for (int c = 0; c < width; ++c) {
                const i64 shifted = c >= shift ? AT(previous, c - shift) : 0;
                AT(updated, c) = AT(locator, c) ^ gf_mul(coef, shifted, exp_t, log_t);
            }
            for (int c = 0; c < width; ++c) {
                if (grow) AT(previous, c) = AT(locator, c);
                AT(locator, c) = AT(updated, c);
            }
        }
        if (grow) {
            last = d;
            length = r + 1 - length;
            shift = 1;
        } else {
            shift += 1;
        }
    }
    int degree = 0;
    for (int c = 0; c < width; ++c)
        if (AT(locator, c) != 0) degree = c;
    valid[row] = 0;
    if (degree != length || length > t) return;
    if (degree == 0) {
        valid[row] = 1;
        return;
    }
    // Chien search: a locator of degree L has at most L roots.
    int found = 0;
    for (int p = 0; p < n && found < degree; ++p) {
        const i64 x = locator_logs[p];
        if (gf_eval(locator, degree + 1, x == 0 ? 0 : q1 - x, q1, B, exp_t, log_t) == 0)
            AT(positions, found++) = p;
    }
    if (found != degree) return;
    signed char* word = words + (i64)row * n * m;
    if (binary) {
        for (int e = 0; e < degree; ++e) word[AT(positions, e)] ^= 1;
        valid[row] = 1;
        return;
    }
    // Forney: e_p = X_p^(1-b) Omega(X_p^-1) / Lambda'(X_p^-1), both of
    // degree < t, Omega = S Lambda mod x^2t and Lambda' the odd terms.
    for (int k = 0; k < t; ++k) {
        i64 o = 0;
        for (int i = 0; i <= k; ++i) o ^= gf_mul(AT(locator, i), syn[k - i], exp_t, log_t);
        AT(omega, k) = o;
        AT(derivative, k) = (k % 2 == 0) ? AT(locator, k + 1) : 0;
    }
    for (int e = 0; e < degree; ++e) {
        const i64 x = locator_logs[AT(positions, e)];
        const i64 xinv = x == 0 ? 0 : q1 - x;
        const i64 num = gf_eval(omega, t, xinv, q1, B, exp_t, log_t);
        const i64 den = gf_eval(derivative, t, xinv, q1, B, exp_t, log_t);
        if (num == 0 || den == 0) return;
        i64 xpow = ((i64)(1 - first_root) * x) % q1;
        if (xpow < 0) xpow += q1;
        AT(values, e) = exp_t[(log_t[num] - log_t[den] + q1 + xpow) % q1];
    }
    for (int e = 0; e < degree; ++e) {
        const i64 base = AT(positions, e) * m;
        const i64 v = AT(values, e);
        for (int b = 0; b < m; ++b) word[base + b] ^= (signed char)((v >> (m - 1 - b)) & 1);
    }
    valid[row] = 1;
#undef AT
}
