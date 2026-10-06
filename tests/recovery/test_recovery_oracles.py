"""Carrier-phase recovery checked against plain-Python reference implementations.

Each test runs the public estimator and the matching oracle from
``tests.common.reference_impl`` on the same input and compares the per-symbol
phase trajectories.  Inputs are complex128 so that the library's metrics are
computed in float64, like the oracles: arg-min ties then resolve identically.
"""

import numpy as np
import pytest

from commkit.mapping import Constellation
from commkit.recovery import (
    BPS,
    PLL,
    ViterbiViterbi,
    correct_cycle_slips,
    estimate_carrier_phase,
)
from tests.common.reference_impl import (
    bps_reference,
    cycle_slip_reference,
    pll_reference,
    viterbi_viterbi_reference,
)

N_SYM = 2048
ATOL = 1e-9
# The PLL's square-QAM fast slicer builds decided points from float32 grid
# constants (``square_qam_slicer_params``), so its trajectory differs from the
# float64 oracle by up to ~3e-7 rad; cross-QAM uses the exact table and stays
# within ATOL.
PLL_ATOL = 1e-6


def _phase_noise_input(modulation: str, order: int, num_ch: int, seed: int = 3):
    """Symbols with a Wiener phase walk (plus a static offset) and 25 dB AWGN.

    Returns ``(C, N_SYM)`` complex128, or ``(N_SYM,)`` for ``num_ch == 1``.
    """
    rng = np.random.default_rng(seed)
    const = getattr(Constellation, modulation)(order).points
    syms = const[rng.integers(0, order, (num_ch, N_SYM))]
    walk = np.cumsum(rng.normal(0.0, 0.02, (num_ch, N_SYM)), axis=-1) + 0.3
    noise_std = np.sqrt(10 ** (-25 / 10) / 2)
    noise = noise_std * (
        rng.standard_normal((num_ch, N_SYM)) + 1j * rng.standard_normal((num_ch, N_SYM))
    )
    x = (syms * np.exp(1j * walk) + noise).astype(np.complex128)
    return x[0] if num_ch == 1 else x


@pytest.mark.parametrize(
    ("modulation", "order", "block_size"), [("psk", 4, 16), ("qam", 16, 32)]
)
@pytest.mark.parametrize(
    ("num_ch", "joint"),
    [(1, False), (2, False), (2, True)],
    ids=["siso", "mimo", "joint"],
)
def test_viterbi_viterbi_matches_oracle(modulation, order, block_size, num_ch, joint):
    x = _phase_noise_input(modulation, order, num_ch)
    phi = estimate_carrier_phase(
        x,
        ViterbiViterbi(block_size=block_size, joint_channels=joint),
        constellation=getattr(Constellation, modulation)(order),
    ).value
    ref = viterbi_viterbi_reference(
        x,
        modulation=modulation,
        order=order,
        block_size=block_size,
        joint_channels=joint,
    )
    np.testing.assert_allclose(np.atleast_2d(phi), ref, rtol=0, atol=ATOL)


@pytest.mark.parametrize(
    ("num_ch", "joint"),
    [(1, False), (2, False), (2, True)],
    ids=["siso", "mimo", "joint"],
)
@pytest.mark.parametrize("order", [16, 32], ids=["square", "cross"])
def test_bps_matches_oracle(num_ch, joint, order):
    x = _phase_noise_input("qam", order, num_ch)
    phi = estimate_carrier_phase(
        x,
        BPS(test_phases=32, block_size=16, joint_channels=joint),
        constellation=Constellation.qam(order),
    ).value
    ref = bps_reference(
        x,
        Constellation.qam(order).points,
        num_test_phases=32,
        block_size=16,
        joint_channels=joint,
    )
    np.testing.assert_allclose(np.atleast_2d(phi), ref, rtol=0, atol=ATOL)


@pytest.mark.parametrize("num_ch", [1, 2])
@pytest.mark.parametrize("beta", [0.0, 1e-4], ids=["1st-order", "2nd-order"])
@pytest.mark.parametrize("order", [16, 32], ids=["square", "cross"])
def test_pll_matches_oracle(num_ch, beta, order):
    x = _phase_noise_input("qam", order, num_ch)
    phi = estimate_carrier_phase(
        x, PLL(mu=2e-2, beta=beta), constellation=Constellation.qam(order)
    ).value
    ref = pll_reference(x, Constellation.qam(order).points, mu=2e-2, beta=beta)
    np.testing.assert_allclose(np.atleast_2d(phi), ref, rtol=0, atol=PLL_ATOL)


@pytest.mark.parametrize("history_length", [5, 50])
def test_cycle_slips_match_oracle(history_length):
    """A drifting trajectory with injected pi/2 slips and estimator noise."""
    rng = np.random.default_rng(11)
    n = 400
    phi = 0.002 * np.arange(n) + rng.normal(0.0, 0.05, n)
    for start, k in [(60, 1), (150, -1), (260, 2), (330, 1)]:
        phi[start:] += k * np.pi / 2
    ref = cycle_slip_reference(
        phi, symmetry=4, history_length=history_length, threshold=np.pi / 4
    )
    out = correct_cycle_slips(phi.copy(), 4, history_length, np.pi / 4)
    np.testing.assert_allclose(out, ref, rtol=0, atol=ATOL)
    # The oracle itself must remove the injected slips (sanity check on the
    # test signal, independent of the library).
    assert np.max(np.abs(ref - 0.002 * np.arange(n))) < 0.5
