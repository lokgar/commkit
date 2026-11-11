"""Private interpolator (commkit._interpolation): the CPU kernel and the GPU
gather against a plain-Python windowed sinc, both forms of instants, the
record's edges and real input."""

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


TOL = {"complex64": 2e-6, "float32": 2e-6, "complex128": 1e-12}


@pytest.mark.parametrize("dtype", ["complex64", "complex128", "float32"])
def test_at_instants_matches_the_reference(xp, dtype):
    """Arbitrary instants, including samples, both edges (zero-extended)
    and a margin beyond them, agree with the reference to the dtype's
    rounding (one tap sum of order 1)."""
    x = _data(xp, dtype)
    pos = np.concatenate(
        [np.random.default_rng(1).uniform(-5.0, 304.0, 200), [0.0, 17.0, 299.0]]
    )
    base = np.floor(pos).astype(np.int64)
    real = np.dtype(dtype).type(0).real.dtype
    out = interp.interpolate_at(
        x, xp.asarray(base), xp.asarray((pos - base).astype(real)), margin=6
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
