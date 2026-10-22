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
        _fiber, "_make_ops", lambda x, xp, phi, omega: _fiber._Ops(xp, phi, omega)
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
