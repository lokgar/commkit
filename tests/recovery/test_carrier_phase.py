"""estimate_carrier_phase: the result class of each symbol-rate method."""

import dataclasses

import numpy as np
import pytest

from commkit import recovery
from commkit.mapping import Constellation
from tests.common.conversions import to_numpy

C16 = Constellation.qam(16)


def _symbols(xp, n=2048, phase=0.2, seed=0):
    rng = np.random.default_rng(seed)
    tx = C16.points[rng.integers(0, 16, (2, n))]
    noise = 0.02 * (rng.standard_normal(tx.shape) + 1j * rng.standard_normal(tx.shape))
    rx = (tx * np.exp(1j * phase) + noise).astype(np.complex64)
    return xp.asarray(rx), tx


@pytest.mark.parametrize(
    ("method", "result"),
    [
        (recovery.BPS(), recovery.BlockPhaseEstimate),
        (recovery.ViterbiViterbi(), recovery.BlockPhaseEstimate),
        (recovery.PLL(), recovery.CarrierPhaseEstimate),
        ("pilots", recovery.PilotInterpolationEstimate),
        ("constant", recovery.CarrierPhaseEstimate),
    ],
    ids=["bps", "vv", "pll", "pilots", "constant"],
)
def test_each_method_returns_its_result_class(xp, method, result):
    """The result holds exactly its method's fields, none of them None, and
    the phase it reports is the applied one (0.2 rad, up to the
    constellation's quarter-turn symmetry for the blind methods)."""
    rx, tx = _symbols(xp)
    if method == "pilots":
        idx = np.arange(0, rx.shape[-1], 16)
        method = recovery.PilotInterpolation(indices=idx, values=xp.asarray(tx[:, idx]))
    elif method == "constant":
        method = recovery.ConstantPhase(symbols=xp.asarray(tx))
    est = recovery.estimate_carrier_phase(rx, method, constellation=C16)
    assert type(est) is result
    assert all(getattr(est, f.name) is not None for f in dataclasses.fields(est))
    phase = to_numpy(est.value)
    residual = np.angle(np.exp(1j * 4 * (np.median(phase, axis=-1) - 0.2))) / 4
    assert np.all(np.abs(residual) < 0.02)
