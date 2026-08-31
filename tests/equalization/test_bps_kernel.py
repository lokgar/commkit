"""Tests for the fused bps_min_d2 CUDA kernel and its call-site integration.

Kernel-level tests compare against a float64 NumPy reference; end-to-end
tests compare the public BPS / block_lms outputs computed with the kernel
against the same call with the kernel disabled (xp fallback), which also
exercises the fallback contract.
"""

import numpy as np
import pytest

from commkit import _cuda, recovery
from commkit.equalization import block_lms
from commkit.mapping import gray_constellation


def _skip_unless_kernel(backend_device):
    if backend_device != "gpu":
        pytest.skip("requires a CUDA device")
    if not _cuda.is_available():
        pytest.skip("CUDA device below compute capability 7.0")


@pytest.fixture
def no_kernel(monkeypatch):
    """Disable the fused kernel so call sites take the xp fallback path."""

    def disable():
        monkeypatch.setattr(_cuda, "get_kernel", lambda *a, **k: None)

    return disable


def _reference_d2(x, phasor, const):
    """Full (P, C, N, M) candidate-distance tensor in float64."""
    xr = x[None, :, :].astype(np.complex128) * phasor[:, None, None].astype(
        np.complex128
    )
    return np.abs(xr[..., None] - const[None, None, None, :].astype(np.complex128)) ** 2


# -- Kernel-level correctness --------------------------------------------------


def test_table_min_d2_matches_reference(backend_device, xp):
    _skip_unless_kernel(backend_device)
    rng = np.random.RandomState(11)
    C, N, P = 2, 4096, 64
    x = (rng.randn(C, N) + 1j * rng.randn(C, N)).astype(np.complex64)
    angles = np.linspace(0.0, np.pi / 2.0, P, endpoint=False)
    phasor = np.exp(-1j * angles).astype(np.complex64)
    const = gray_constellation("qam", 128).astype(np.complex64)

    ref = _reference_d2(x, phasor, const).min(axis=-1)

    kern = _cuda.get_kernel("bps_min_d2", mode="table")
    assert kern is not None
    out = kern(xp.asarray(x), xp.asarray(phasor), constellation=xp.asarray(const))

    assert out.shape == (P, C, N) and out.dtype == xp.float32
    np.testing.assert_allclose(xp.asnumpy(out), ref, rtol=1e-5, atol=1e-7)


def test_table_argmin_matches_reference(backend_device, xp):
    _skip_unless_kernel(backend_device)
    rng = np.random.RandomState(12)
    C, N, P = 2, 4096, 16
    x = (rng.randn(C, N) + 1j * rng.randn(C, N)).astype(np.complex64)
    phasor = np.exp(-1j * np.linspace(0.0, np.pi / 2.0, P, endpoint=False)).astype(
        np.complex64
    )
    const = gray_constellation("qam", 128).astype(np.complex64)

    d2 = _reference_d2(x, phasor, const)
    ref_idx = d2.argmin(axis=-1)
    ref_min = d2.min(axis=-1)

    kern = _cuda.get_kernel("bps_min_d2", mode="table", return_argmin=True)
    assert kern is not None
    out, idx = kern(xp.asarray(x), xp.asarray(phasor), constellation=xp.asarray(const))
    idx_np = xp.asnumpy(idx)

    # ≥ 99.9 % index agreement
    assert float(np.mean(idx_np == ref_idx)) >= 0.999
    # ... and tie-tolerant: the distance at every returned index is the minimum.
    d_at_idx = np.take_along_axis(d2, idx_np[..., None], axis=-1)[..., 0]
    np.testing.assert_allclose(d_at_idx, ref_min, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(xp.asnumpy(out), ref_min, rtol=1e-5, atol=1e-7)


def test_grid_min_d2_matches_reference(backend_device, xp):
    _skip_unless_kernel(backend_device)
    rng = np.random.RandomState(13)
    C, N, P = 2, 4096, 64
    x = (rng.randn(C, N) + 1j * rng.randn(C, N)).astype(np.complex64)
    phasor = np.exp(-1j * np.linspace(0.0, np.pi / 2.0, P, endpoint=False)).astype(
        np.complex64
    )

    const = gray_constellation("qam", 64)
    levels = np.sort(np.unique(const.real))
    lev_min = float(levels[0])
    d_grid = float(levels[1] - levels[0])
    side = len(levels)

    xr = x[None, :, :].astype(np.complex128) * phasor[:, None, None].astype(
        np.complex128
    )
    ri = np.clip(np.round((xr.real - lev_min) / d_grid), 0, side - 1)
    ii = np.clip(np.round((xr.imag - lev_min) / d_grid), 0, side - 1)
    ref = (xr.real - (lev_min + ri * d_grid)) ** 2 + (
        xr.imag - (lev_min + ii * d_grid)
    ) ** 2

    kern = _cuda.get_kernel("bps_min_d2", mode="grid")
    assert kern is not None
    out = kern(
        xp.asarray(x), xp.asarray(phasor), lev_min=lev_min, d_grid=d_grid, side=side
    )
    np.testing.assert_allclose(xp.asnumpy(out), ref, rtol=1e-5, atol=1e-7)


def test_grid_mode_rejects_argmin():
    with pytest.raises(KeyError):
        _cuda.get_kernel("not_bps_min_d2")
    # GRID + argmin is an invalid spec: the factory raises, get_kernel
    # translates it into the None fallback (warn-once path).
    assert _cuda.get_kernel("bps_min_d2", mode="grid", return_argmin=True) is None


# -- End-to-end equivalence: kernel vs xp fallback -----------------------------


def _phase_noise_symbols(order, C=2, N=20_000, seed=7):
    rng = np.random.RandomState(seed)
    const = gray_constellation("qam", order)
    syms = const[rng.randint(0, order, (C, N))].astype(np.complex64)
    phase = np.cumsum(rng.randn(C, N) * 0.01, axis=1)
    x = syms * np.exp(1j * phase).astype(np.complex64)
    x += (rng.randn(C, N) + 1j * rng.randn(C, N)).astype(np.complex64) * 0.05
    return x


@pytest.mark.parametrize("order", [128, 16])  # TABLE and GRID call sites
def test_recovery_bps_kernel_matches_fallback(backend_device, xp, no_kernel, order):
    _skip_unless_kernel(backend_device)
    x = xp.asarray(_phase_noise_symbols(order))

    phi_kern = recovery.recover_carrier_phase_bps(
        x, "qam", order, num_test_phases=64, block_size=32
    )
    no_kernel()
    phi_fall = recovery.recover_carrier_phase_bps(
        x, "qam", order, num_test_phases=64, block_size=32
    )
    np.testing.assert_allclose(
        xp.asnumpy(phi_kern), xp.asnumpy(phi_fall), rtol=1e-6, atol=1e-9
    )


def _equalizer_input(order, C=2, n_sym=6000, n_train=1500, seed=3):
    rng = np.random.RandomState(seed)
    const = gray_constellation("qam", order)
    syms = const[rng.randint(0, order, (C, n_sym))].astype(np.complex64)
    x = np.stack(
        [np.convolve(syms[c], [0.05, 1.0, -0.08], mode="same") for c in range(C)]
    )
    x = np.repeat(x, 2, axis=1)  # 2 sps
    phase = np.cumsum(rng.randn(C, x.shape[1]) * 0.002, axis=1)
    x = (x * np.exp(1j * phase)).astype(np.complex64)
    x += (rng.randn(*x.shape) + 1j * rng.randn(*x.shape)).astype(np.complex64) * 0.02
    return x, syms[:, :n_train]


@pytest.mark.parametrize(
    "order,cpr_kwargs",
    [
        (128, dict(cpr_type="bps")),  # TABLE inline BPS + TABLE+argmin DD slicer
        (16, dict(cpr_type="bps")),  # GRID inline BPS
        (128, dict()),  # DD slicer only (no CPR)
        (128, dict(cpr_type="bps", cpr_cycle_slip_correction=True)),
    ],
)
def test_block_lms_kernel_matches_fallback(
    backend_device, xp, no_kernel, order, cpr_kwargs
):
    _skip_unless_kernel(backend_device)
    x, train = _equalizer_input(order)

    def run():
        return block_lms(
            xp.asarray(x),
            xp.asarray(train),
            num_taps=11,
            sps=2,
            modulation="qam",
            order=order,
            block_size=128,
            **cpr_kwargs,
        )

    r_kern = run()
    no_kernel()
    r_fall = run()

    np.testing.assert_allclose(
        xp.asnumpy(xp.asarray(r_kern.y_hat)),
        xp.asnumpy(xp.asarray(r_fall.y_hat)),
        rtol=1e-6,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        xp.asnumpy(xp.asarray(r_kern.error)),
        xp.asnumpy(xp.asarray(r_fall.error)),
        rtol=1e-6,
        atol=1e-7,
    )
    if r_kern.phase_trajectory is not None:
        np.testing.assert_allclose(
            xp.asnumpy(xp.asarray(r_kern.phase_trajectory)),
            xp.asnumpy(xp.asarray(r_fall.phase_trajectory)),
            rtol=1e-6,
            atol=1e-9,
        )


# -- Fallback conditions -------------------------------------------------------


def test_recovery_complex128_does_not_consult_kernel(backend_device, xp, monkeypatch):
    _skip_unless_kernel(backend_device)

    def _fail(*a, **k):
        raise AssertionError("get_kernel must not be consulted for complex128 input")

    monkeypatch.setattr(_cuda, "get_kernel", _fail)
    x = xp.asarray(_phase_noise_symbols(16).astype(np.complex128))
    phi = recovery.recover_carrier_phase_bps(
        x, "qam", 16, num_test_phases=32, block_size=32
    )
    assert bool(xp.all(xp.isfinite(phi)))


def test_recovery_bps_runs_on_cpu_without_kernel(backend_device, xp):
    if backend_device != "cpu":
        pytest.skip("CPU-leg fallback test")
    x = xp.asarray(_phase_noise_symbols(128, N=4000))
    phi = recovery.recover_carrier_phase_bps(
        x, "qam", 128, num_test_phases=32, block_size=32
    )
    assert phi.shape == x.shape
    assert bool(np.all(np.isfinite(phi)))
