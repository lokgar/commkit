"""Private interpolator (commkit._interpolation): the Numba kernel and the
CUDA kernel against a plain-Python windowed sinc, both forms of instants,
the record's edges and real input; the CUDA kernel against the Numba one."""

import numpy as np
import pytest

from commkit import _interpolation as interp
from tests.common.conversions import to_numpy
from tests.common.reference_impl import windowed_sinc_reference


def _data(xp, dtype, shape=(2, 300), seed=3):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    if np.dtype(dtype).kind == "f":
        x = x.real
    return xp.asarray(x.astype(dtype))


TOL = {"complex64": 2e-6, "float32": 2e-6, "complex128": 1e-12, "float64": 1e-12}


@pytest.mark.parametrize("dtype", ["complex64", "complex128", "float32", "float64"])
def test_at_instants_matches_the_reference(xp, dtype):
    """Arbitrary instants, including samples, both edges (zero-extended)
    and beyond them, agree with the reference to the dtype's rounding (one
    tap sum of order 1)."""
    x = _data(xp, dtype)
    pos = np.concatenate(
        [np.random.default_rng(1).uniform(-5.0, 304.0, 200), [0.0, 17.0, 299.0]]
    )
    base = np.floor(pos).astype(np.int64)
    real = np.dtype(dtype).type(0).real.dtype
    out = interp.interpolate_at(
        x, xp.asarray(base), xp.asarray((pos - base).astype(real))
    )
    assert out.dtype == x.dtype and out.shape == (2, pos.size)
    ref = windowed_sinc_reference(to_numpy(x), pos, interp.HALF_WIDTH)
    if np.dtype(dtype).kind == "f":
        ref = ref.real
    np.testing.assert_allclose(to_numpy(out), ref, atol=TOL[dtype])


@pytest.mark.parametrize("shared", [True, False])
def test_piecewise_matches_the_reference(xp, shared):
    """Instants linear between knots, shared by the channels or one row of
    knots each, agree with the reference at the same positions."""
    x = _data(xp, "complex128")
    knots = np.array([[-1.5, 40.25, 120.0, 301.0], [0.0, 90.5, 200.0, 280.75]])
    if shared:
        knots = knots[:1]
    sub = 7
    out = to_numpy(interp.interpolate_piecewise(x, knots, sub))
    assert out.shape == (2, sub * (knots.shape[1] - 1))
    j = np.arange(sub) / sub
    for c in range(2):
        r = 0 if shared else c
        k = knots[r]
        pos = (k[:-1, None] + j[None, :] * np.diff(k)[:, None]).ravel()
        ref = windowed_sinc_reference(to_numpy(x)[c : c + 1], pos, interp.HALF_WIDTH)
        np.testing.assert_allclose(out[c], ref[0], atol=1e-12)


def test_a_fraction_that_rounds_to_one(xp):
    """An instant just below a sample whose fraction rounds to 1 in float32
    is that sample, not NaN (the sinc's 0/0 at the next tap)."""
    x = _data(xp, "complex64")
    knots = np.array([[17.0 - 1e-9, 18.0 - 1e-9]])
    out = to_numpy(interp.interpolate_piecewise(x, knots, 1))
    np.testing.assert_allclose(out[:, 0], to_numpy(x)[:, 17], atol=2e-6)


def test_no_instants(xp):
    """No knots intervals: an empty result, without a zero-size launch."""
    x = _data(xp, "complex64")
    assert interp.interpolate_piecewise(x, np.array([[3.0]]), 4).shape == (2, 0)
    empty = xp.zeros(0, dtype=xp.int64)
    out = interp.interpolate_at(x, empty, xp.zeros(0, dtype=xp.float32))
    assert out.shape == (2, 0)


@pytest.mark.gpu_only
@pytest.mark.parametrize("dtype", ["complex64", "float32", "complex128"])
@pytest.mark.parametrize("shared", [True, False])
def test_cuda_kernel_matches_the_numba_kernel(xp, dtype, shared):
    """2^20 samples resampled at 1000 ppm (instants up to 2^20 formed in
    float64): the CUDA kernel, with its taps in the samples' precision,
    agrees with the Numba kernel's float64 taps to single-precision
    rounding of a tap sum of unit-power data (measured 2.4e-6).  In double precision
    the difference is the Numba kernel's: fast-math forms its instants near
    10^6 a few ulp off (measured 2e-9; the CUDA kernel matches the
    plain-Python reference to 5e-16)."""
    x = _data(np, dtype, shape=(2, 1 << 20), seed=4)
    n_out = (1 << 20) - 1100
    knots = np.array([[0.0, n_out * 1.001], [0.0, n_out * 0.9995]])
    if shared:
        knots = knots[:1]
    cpu = interp.interpolate_piecewise(x, knots, n_out)
    gpu = to_numpy(interp.interpolate_piecewise(xp.asarray(x), knots, n_out))
    tol = 1e-8 if dtype == "complex128" else 1e-5
    np.testing.assert_allclose(gpu, cpu, rtol=0, atol=tol)
