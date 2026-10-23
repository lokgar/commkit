"""GPU paths of the split-step solver (commkit._fiber).

The fused CuPy kernels and the per-span CUDA graph are checked against the
plain CuPy stages and against the CPU; the physics is tested through the
public functions (impairments/channel/test_channel_nonlinear.py,
test_filtering.py).
"""

import logging

import numpy as np
import pytest

from commkit import _fiber
from commkit.backend import to_device
from commkit.filtering import correct_fiber_propagation
from commkit.impairments import Fiber, apply_fiber_propagation

FS = 64e9


def _field(num_channels, n=1 << 14, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((num_channels, n)) + 1j * rng.standard_normal(
        (num_channels, n)
    )
    spectrum = np.fft.fft(x, axis=-1)
    spectrum[:, np.abs(np.fft.fftfreq(n)) > 0.25] = 0.0
    return np.fft.ifft(spectrum, axis=-1).astype(np.complex64)


def _rel(a, b):
    a, b = np.asarray(to_device(a, "cpu")), np.asarray(to_device(b, "cpu"))
    return float(np.sqrt(np.mean(np.abs(a - b) ** 2) / np.mean(np.abs(b) ** 2)))


def _plain_ops(monkeypatch):
    """Force the unfused CuPy stages (the fused path's reference)."""
    monkeypatch.setattr(
        _fiber, "_make_ops", lambda x, xp, *args: _fiber._Ops(xp, x.dtype, *args)
    )


CASES = {
    "1pol": (1, Fiber(), None),
    "2pol": (2, Fiber(dispersion_slope_ps_nm2_km=0.057), None),
    "pmd+ase": (2, Fiber(pmd_ps_sqrt_km=0.1), 5.0),
}


@pytest.mark.gpu_only
@pytest.mark.parametrize("case", list(CASES))
def test_fused_kernels_match_plain_stages(xp, monkeypatch, caplog, case):
    """Six spans: the PMD-free cases replay a captured span from the third
    span on; noise and waveplates come from the same seed in both runs."""
    num_channels, fiber, nf = CASES[case]
    x = xp.asarray(_field(num_channels))
    if num_channels == 1:
        x = x[0]
    kw = dict(
        fiber=fiber,
        span_length_km=80.0,
        num_spans=6,
        launch_power_dbm=6.0,
        steps_per_span=20,
        noise_figure_db=nf,
        rng=5,
        sampling_rate=FS,
    )
    with caplog.at_level(logging.WARNING, logger="commkit"):
        fused = apply_fiber_propagation(x, **kw)
    assert "fused CUDA kernels unavailable" not in caplog.text
    assert "capture failed" not in caplog.text
    _plain_ops(monkeypatch)
    plain = apply_fiber_propagation(x, **kw)
    assert _rel(fused, plain) < 1e-4


@pytest.mark.gpu_only
def test_fused_dbp_matches_plain_stages(xp, monkeypatch):
    x = xp.asarray(_field(2))
    kw = dict(
        fiber=Fiber(),
        span_length_km=80.0,
        num_spans=5,
        launch_power_dbm=6.0,
        steps_per_span=10,
        sampling_rate=FS,
    )
    fused = correct_fiber_propagation(x, **kw)
    _plain_ops(monkeypatch)
    assert _rel(fused, correct_fiber_propagation(x, **kw)) < 1e-4


@pytest.mark.gpu_only
def test_gpu_matches_cpu(xp):
    """Noiseless and PMD-free, the GPU field equals the CPU field up to
    float32 rounding (the noise realizations differ between devices)."""
    x = _field(2)
    kw = dict(
        fiber=Fiber(),
        span_length_km=80.0,
        num_spans=4,
        launch_power_dbm=6.0,
        steps_per_span=20,
        sampling_rate=FS,
    )
    gpu = apply_fiber_propagation(xp.asarray(x), **kw)
    assert _rel(gpu, apply_fiber_propagation(x, **kw)) < 1e-4


# -----------------------------------------------------------------------------
# CPU passes against the plain stages (oracle; also gives the Numba kernels
# real coverage under NUMBA_DISABLE_JIT)
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_numba_passes_match_the_plain_stages(dtype, monkeypatch):
    monkeypatch.setattr(_fiber, "_HYBRID_KERR_MIN", 0)  # Numba Kerr at any size
    rng = np.random.default_rng(3)
    n = 512
    x = (rng.standard_normal((2, n)) + 1j * rng.standard_normal((2, n))).astype(dtype)
    phase_km = rng.uniform(-3.0, 3.0, n)
    omega = 2.0 * np.pi * np.fft.fftfreq(n, d=1.0 / FS)
    waveplates = _fiber._haar_unitaries(rng, (2, 3))
    args = (np, np.dtype(dtype), phase_km, omega, 2e-13, waveplates)
    cpu, plain = _fiber._CpuOps(*args), _fiber._Ops(*args)
    cpu.set_span(1)
    plain.set_span(1)
    tol = 1e-5 if dtype == np.complex64 else 1e-12
    for section in (None, 2):
        np.testing.assert_allclose(
            cpu.linear(x.copy(), 0.7, section),
            plain.linear(x.copy(), 0.7, section),
            rtol=tol,
            atol=tol,
        )
    for k in (0.3, -1.2):
        np.testing.assert_allclose(
            cpu.nonlinear(x.copy(), k), plain.nonlinear(x.copy(), k), rtol=tol, atol=tol
        )
        np.testing.assert_allclose(
            cpu.nonlinear(x[:1].copy(), k),
            plain.nonlinear(x[:1].copy(), k),
            rtol=tol,
            atol=tol,
        )


def test_dbp_leaves_pmd_alone():
    """A receiver cannot know the waveplates: DBP of a PMD fiber is DBP of
    the same fiber without PMD."""
    x = _field(2, n=2048)
    kw = dict(
        span_length_km=80.0,
        num_spans=2,
        launch_power_dbm=4.0,
        steps_per_span=5,
        sampling_rate=FS,
    )
    a = correct_fiber_propagation(x, fiber=Fiber(pmd_ps_sqrt_km=0.2), **kw)
    b = correct_fiber_propagation(x, fiber=Fiber(), **kw)
    np.testing.assert_array_equal(a, b)


def test_real_input_propagates_as_a_complex_field():
    x = _field(1, n=2048)[0].real
    y = apply_fiber_propagation(
        x,
        fiber=Fiber(),
        span_length_km=40.0,
        num_spans=1,
        launch_power_dbm=3.0,
        steps_per_span=5,
        sampling_rate=FS,
    )
    ref = apply_fiber_propagation(
        x.astype(np.complex64),
        fiber=Fiber(),
        span_length_km=40.0,
        num_spans=1,
        launch_power_dbm=3.0,
        steps_per_span=5,
        sampling_rate=FS,
    )
    assert y.dtype == np.complex64
    np.testing.assert_allclose(y, ref, atol=1e-6)


@pytest.mark.parametrize(
    ("x", "power", "match"),
    [
        (np.zeros((2, 64), np.complex64), 0.0, "no power"),
        (np.ones((2, 64), np.complex64), float("nan"), "launch_power_dbm"),
    ],
    ids=["silent", "nan-power"],
)
def test_launch_power_errors(x, power, match):
    with pytest.raises(ValueError, match=match):
        apply_fiber_propagation(
            x,
            fiber=Fiber(),
            span_length_km=10.0,
            num_spans=1,
            launch_power_dbm=power,
            sampling_rate=FS,
        )


@pytest.mark.gpu_only
def test_complex128_on_the_gpu_matches_the_cpu(xp):
    """complex128 takes the plain CuPy stages (the fused kernels are
    complex64) and keeps double precision end to end."""
    x = _field(2, n=4096).astype(np.complex128)
    kw = dict(
        fiber=Fiber(pmd_ps_sqrt_km=0.1),
        span_length_km=80.0,
        num_spans=3,
        launch_power_dbm=6.0,
        steps_per_span=20,
        rng=2,
        sampling_rate=FS,
    )
    gpu = apply_fiber_propagation(xp.asarray(x), **kw)
    assert gpu.dtype == xp.complex128
    assert _rel(gpu, apply_fiber_propagation(x, **kw)) < 1e-10
