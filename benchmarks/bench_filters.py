"""Benchmarks: IIR filtering and the moving-average smoother.

``bench_iir_filter`` runs a 4th-order low-pass Butterworth at a low cutoff
(1% of the sampling rate, poles near z = 1) on dual-channel complex64; on the
GPU it times the double-precision ``cupyx.scipy.signal`` path.
``bench_moving_average_valid`` smooths a two-channel float32 phase record.
``bench_filter_response`` evaluates the response and group delay of
low-pass FIRs on the default 4096-point grid (FFTs) and on an arbitrary
grid of as many points (Horner evaluation).
"""

import pytest
from workloads import FS, bps_workload, phase_workload

from commkit.filtering import butterworth_sos, filter_response, fir_taps, iir_filter
from commkit.smoothing import moving_average

ROUNDS = dict(rounds=3, warmup_rounds=1, iterations=1)
N = 1 << 20


@pytest.mark.parametrize("zero_phase", [False, True], ids=["causal", "zero-phase"])
def bench_iir_filter(benchmark, backend_device, xp, sync, zero_phase):
    x = xp.asarray(bps_workload(order=16, n_sym=N, num_ch=2))
    sos = butterworth_sos(sampling_rate=FS, cutoff=0.01 * FS, order=4)

    def run():
        out = iir_filter(x, sos, zero_phase=zero_phase)
        sync()
        return out

    benchmark.pedantic(run, **ROUNDS)


@pytest.mark.parametrize("window", [16, 257])
def bench_moving_average_valid(benchmark, backend_device, xp, sync, window):
    x = xp.asarray(phase_workload(N if backend_device == "gpu" else N // 4))

    def run():
        out = moving_average(x, window=window, mode="valid")
        sync()
        return out

    benchmark.pedantic(run, **ROUNDS)


@pytest.mark.parametrize("grid", ["dft", "custom"])
@pytest.mark.parametrize("num_taps", [31, 257, 4097])
def bench_filter_response(benchmark, backend_device, xp, sync, num_taps, grid):
    taps = fir_taps(sampling_rate=FS, num_taps=num_taps, cutoff=0.1 * FS)
    taps = xp.asarray(taps, dtype=xp.float32)
    frequencies = xp.linspace(-0.5, 0.5, 4096) if grid == "custom" else None

    def run():
        result = filter_response(taps, nfft=4096, frequencies=frequencies)
        sync()
        return result

    benchmark.pedantic(run, **ROUNDS)
