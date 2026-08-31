// Trivial templated kernel validating the _cuda compile/specialize/launch
// infrastructure (compiler.py + get_kernel). Not used by any DSP path.

template <typename T>
__global__ void selftest_scale(const T* __restrict__ x,
                               T* __restrict__ y,
                               T alpha,
                               long long n) {
    const long long i =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = alpha * x[i];
    }
}
