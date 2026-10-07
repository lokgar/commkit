"""Sequential equalizers checked against plain-Python reference implementations.

Each test runs the public equalizer and the matching oracle from
``tests.common.reference_impl`` on the same small input and compares outputs,
errors and final taps.  The oracles derive normalization, padding and initial
taps from the documented contract, so these tests also guard the preparation
code around the kernels.  Inputs use high SNR so that decision-directed slicing
cannot flip a decision through float32-versus-float64 rounding.
"""

import numpy as np
import pytest

from commkit.equalization import cma, lms, rde, rls
from commkit.mapping import Constellation
from tests.common.reference_impl import (
    cma_reference,
    lms_reference,
    rde_reference,
    rls_reference,
)

SPS = 2
NUM_TAPS = 7
N_SYM = 300
RTOL = 1e-4
ATOL = 1e-5


def _isi_input(order: int, num_ch: int, seed: int = 7):
    """Dual-rate QAM through a short ISI channel (+ 2x2 mixing) at 35 dB SNR.

    Returns ``(samples (C, N_SYM*SPS) complex64 or (N,) for C=1,
    symbols (C, N_SYM))``.
    """
    rng = np.random.default_rng(seed)
    const = Constellation.qam(order).points
    syms = const[rng.integers(0, order, (num_ch, N_SYM))]
    x = np.repeat(syms, SPS, axis=-1)
    h = np.array([0.08, 1.0, 0.15j])
    x = np.stack([np.convolve(ch, h, mode="same") for ch in x])
    if num_ch == 2:
        th = np.deg2rad(20.0)
        x = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]]) @ x
    noise_std = np.sqrt(np.mean(np.abs(x) ** 2) * 10 ** (-35 / 10) / 2)
    x = x + noise_std * (
        rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape)
    )
    x = x.astype(np.complex64)
    return (x[0] if num_ch == 1 else x), syms


def _assert_matches(result, ref, num_ch: int):
    """Compare an EqualizerResult with an oracle dict (SISO results squeezed)."""
    y = np.atleast_2d(np.asarray(result.y_hat))
    e = np.atleast_2d(np.asarray(result.error))
    w = np.asarray(result.weights)
    if num_ch == 1:
        w = w[None, None, :]
    np.testing.assert_allclose(y, ref["y"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(e, ref["e"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(w, ref["w"], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("num_ch", [1, 2])
@pytest.mark.parametrize("n_train", [N_SYM, 100])
def test_lms_matches_oracle(num_ch, n_train):
    samples, syms = _isi_input(16, num_ch)
    training = syms[..., :n_train] if num_ch == 2 else syms[0, :n_train]
    res = lms(
        samples,
        training,
        num_taps=NUM_TAPS,
        sps=SPS,
        step_size=1e-2,
        constellation=Constellation.qam(16),
    )
    ref = lms_reference(
        samples,
        training,
        Constellation.qam(16).points,
        num_taps=NUM_TAPS,
        sps=SPS,
        step_size=1e-2,
    )
    _assert_matches(res, ref, num_ch)


@pytest.mark.parametrize("num_ch", [1, 2])
@pytest.mark.parametrize("leakage", [0.0, 1e-3])
def test_rls_matches_oracle(num_ch, leakage):
    samples, syms = _isi_input(16, num_ch)
    training = syms[..., :100] if num_ch == 2 else syms[0, :100]
    res = rls(
        samples,
        training,
        num_taps=NUM_TAPS,
        sps=SPS,
        forgetting_factor=0.99,
        delta=0.01,
        leakage=leakage,
        constellation=Constellation.qam(16),
    )
    ref = rls_reference(
        samples,
        training,
        Constellation.qam(16).points,
        num_taps=NUM_TAPS,
        sps=SPS,
        forgetting_factor=0.99,
        delta=0.01,
        leakage=leakage,
    )
    assert res.tail_trim == NUM_TAPS // 2
    _assert_matches(res, ref, num_ch)


@pytest.mark.parametrize("num_ch", [1, 2])
def test_cma_matches_oracle(num_ch):
    samples, _ = _isi_input(4, num_ch)
    const = Constellation.qam(4).points
    r2 = float(np.mean(np.abs(const) ** 4) / np.mean(np.abs(const) ** 2))
    res = cma(
        samples,
        num_taps=NUM_TAPS,
        sps=SPS,
        step_size=1e-3,
        constellation=Constellation.qam(4),
    )
    ref = cma_reference(samples, num_taps=NUM_TAPS, sps=SPS, step_size=1e-3, r2=r2)
    _assert_matches(res, ref, num_ch)


@pytest.mark.parametrize("num_ch", [1, 2])
def test_rde_matches_oracle(num_ch):
    samples, _ = _isi_input(16, num_ch)
    radii = np.unique(np.round(np.abs(Constellation.qam(16).points), 6))
    res = rde(
        samples,
        num_taps=NUM_TAPS,
        sps=SPS,
        step_size=1e-3,
        constellation=Constellation.qam(16),
    )
    ref = rde_reference(
        samples, num_taps=NUM_TAPS, sps=SPS, step_size=1e-3, radii=radii
    )
    _assert_matches(res, ref, num_ch)


# -----------------------------------------------------------------------------
# Inline carrier phase recovery
# -----------------------------------------------------------------------------

CPR_CASES = [
    ("pll", False, False),
    ("pll", True, False),
    ("pll", False, True),
    ("bps", False, False),
    ("bps", True, False),
    ("bps", False, True),
]


def _phase_rotated(samples, rate=2e-3, offset=0.4):
    """A phase ramp the inline CPR has to track (the taps alone cannot)."""
    n = np.arange(samples.shape[-1])
    return (samples * np.exp(1j * (offset + rate * n / SPS))).astype(np.complex64)


def _cpr_objects(kind, joint, slips, num_ch):
    from commkit.recovery import BPS, PLL, CycleSlip
    from tests.common.reference_impl import _InlineCpr

    cycle_slip = CycleSlip(history=20) if slips else None
    const = Constellation.qam(16).points
    if kind == "pll":
        obj = PLL(mu=2e-2, beta=2e-4, joint_channels=joint, cycle_slip=cycle_slip)
        ref = _InlineCpr(
            "pll",
            num_ch,
            const,
            mu=np.float32(2e-2),
            beta=np.float32(2e-4),
            joint=joint,
            history=20 if slips else None,
        )
    else:
        obj = BPS(
            test_phases=16, block_size=8, joint_channels=joint, cycle_slip=cycle_slip
        )
        ref = _InlineCpr(
            "bps",
            num_ch,
            const,
            test_phases=16,
            window=8,
            joint=joint,
            history=20 if slips else None,
        )
    return obj, ref


def _assert_cpr_matches(result, ref, num_ch):
    _assert_matches(result, ref, num_ch)
    phi = np.atleast_2d(np.asarray(result.phase_trajectory))
    np.testing.assert_allclose(phi, ref["phi"], rtol=0, atol=1e-5)


@pytest.mark.parametrize("num_ch", [1, 2])
@pytest.mark.parametrize(("kind", "joint", "slips"), CPR_CASES)
def test_lms_inline_cpr_matches_oracle(num_ch, kind, joint, slips):
    from tests.common.reference_impl import lms_cpr_reference

    samples, syms = _isi_input(16, num_ch)
    samples = _phase_rotated(samples)
    training = syms[..., :100] if num_ch == 2 else syms[0, :100]
    obj, ref_cpr = _cpr_objects(kind, joint, slips, num_ch)
    res = lms(
        samples,
        training,
        num_taps=NUM_TAPS,
        sps=SPS,
        step_size=1e-2,
        constellation=Constellation.qam(16),
        cpr=obj,
    )
    ref = lms_cpr_reference(
        samples,
        training,
        Constellation.qam(16).points,
        num_taps=NUM_TAPS,
        sps=SPS,
        step_size=1e-2,
        cpr=ref_cpr,
    )
    _assert_cpr_matches(res, ref, num_ch)


@pytest.mark.parametrize("num_ch", [1, 2])
@pytest.mark.parametrize(("kind", "joint", "slips"), CPR_CASES)
def test_rls_inline_cpr_matches_oracle(num_ch, kind, joint, slips):
    from tests.common.reference_impl import rls_cpr_reference

    samples, syms = _isi_input(16, num_ch)
    samples = _phase_rotated(samples)
    training = syms[..., :100] if num_ch == 2 else syms[0, :100]
    obj, ref_cpr = _cpr_objects(kind, joint, slips, num_ch)
    res = rls(
        samples,
        training,
        num_taps=NUM_TAPS,
        sps=SPS,
        forgetting_factor=0.99,
        delta=0.01,
        constellation=Constellation.qam(16),
        cpr=obj,
    )
    ref = rls_cpr_reference(
        samples,
        training,
        Constellation.qam(16).points,
        num_taps=NUM_TAPS,
        sps=SPS,
        forgetting_factor=0.99,
        delta=0.01,
        cpr=ref_cpr,
    )
    _assert_cpr_matches(res, ref, num_ch)


@pytest.mark.parametrize(
    "constellation", [Constellation.psk(8), Constellation.psk(2)], ids=["8psk", "bpsk"]
)
def test_lms_inline_bps_follows_symmetry(constellation):
    """The candidates span ``2π/S`` and the unwrap is ``S``-fold, ``S`` the
    constellation's rotational symmetry (8 for 8-PSK, 2 for BPSK)."""
    from commkit.recovery import BPS, CycleSlip
    from tests.common.reference_impl import _InlineCpr, lms_cpr_reference

    points = constellation.points
    rng = np.random.default_rng(5)
    syms = points[rng.integers(0, points.size, N_SYM)]
    x = np.convolve(np.repeat(syms, SPS), [0.08, 1.0, 0.15j], mode="same")
    x = x + 0.03 * (rng.standard_normal(x.size) + 1j * rng.standard_normal(x.size))
    samples = _phase_rotated(x.astype(np.complex64))
    S = int(constellation.rotational_symmetry)
    kw = dict(num_taps=NUM_TAPS, sps=SPS, step_size=1e-2)
    res = lms(
        samples,
        syms[:100],
        **kw,
        constellation=constellation,
        cpr=BPS(test_phases=16, block_size=8, cycle_slip=CycleSlip(history=20)),
    )
    ref_cpr = _InlineCpr(
        "bps", 1, points, test_phases=16, window=8, symmetry=S, history=20
    )
    ref = lms_cpr_reference(samples, syms[:100], points, **kw, cpr=ref_cpr)
    _assert_cpr_matches(res, ref, 1)
