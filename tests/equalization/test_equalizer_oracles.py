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
        modulation="qam",
        order=16,
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
        modulation="qam",
        order=16,
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
        samples, num_taps=NUM_TAPS, sps=SPS, step_size=1e-3, modulation="qam", order=4
    )
    ref = cma_reference(samples, num_taps=NUM_TAPS, sps=SPS, step_size=1e-3, r2=r2)
    _assert_matches(res, ref, num_ch)


@pytest.mark.parametrize("num_ch", [1, 2])
def test_rde_matches_oracle(num_ch):
    samples, _ = _isi_input(16, num_ch)
    radii = np.unique(np.round(np.abs(Constellation.qam(16).points), 6))
    res = rde(
        samples, num_taps=NUM_TAPS, sps=SPS, step_size=1e-3, modulation="qam", order=16
    )
    ref = rde_reference(
        samples, num_taps=NUM_TAPS, sps=SPS, step_size=1e-3, radii=radii
    )
    _assert_matches(res, ref, num_ch)
