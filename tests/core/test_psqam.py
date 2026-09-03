"""Tests for Probabilistic Shaping QAM (PS-QAM)."""

import numpy as np
import pytest

from commkit import generate_psqam, generate_qam, mapping, metrics, multirate
from commkit.backend import to_device
from commkit.impairments import apply_awgn
from commkit.mapping import (
    compute_llr,
    constellation_power,
    gray_constellation,
    maxwell_boltzmann,
    optimal_nu,
    ps_entropy,
    rescale_ps_symbols,
    sample_ps_symbols,
)

# ---------------------------------------------------------------------------
# maxwell_boltzmann
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("order,nu", [(16, 0.5), (64, 0.2), (256, 0.1), (16, 2.0)])
def test_maxwell_boltzmann_sums_to_one(order, nu):
    pmf = maxwell_boltzmann(order, nu)
    assert pmf.shape == (order,)
    assert np.isclose(pmf.sum(), 1.0, atol=1e-12)
    assert np.all(pmf >= 0)


@pytest.mark.parametrize("order", [16, 64, 256])
def test_maxwell_boltzmann_uniform_at_zero(order):
    pmf = maxwell_boltzmann(order, 0.0)
    expected = np.full(order, 1.0 / order)
    np.testing.assert_allclose(pmf, expected, atol=1e-14)


@pytest.mark.parametrize("order", [16, 64])
def test_maxwell_boltzmann_inner_higher_probability(order):
    """Inner constellation points must have higher probability than outer ones."""
    pmf = maxwell_boltzmann(order, nu=0.5)
    const = gray_constellation("qam", order)
    energies = np.abs(const) ** 2
    # PMF must be monotonically decreasing with energy (up to floating-point ties)
    # Check: if energy[i] < energy[j], then pmf[i] >= pmf[j]
    for i in range(order):
        for j in range(order):
            if energies[i] < energies[j] - 1e-6:
                assert pmf[i] >= pmf[j], (
                    f"Expected pmf[{i}]={pmf[i]:.4f} >= pmf[{j}]={pmf[j]:.4f} "
                    f"(energies {energies[i]:.3f} < {energies[j]:.3f})"
                )


# ---------------------------------------------------------------------------
# ps_entropy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("order", [16, 64, 256])
def test_ps_entropy_uniform(order):
    h = ps_entropy(order, nu=0.0)
    assert np.isclose(h, np.log2(order), atol=1e-10)


@pytest.mark.parametrize("order", [16, 64])
def test_ps_entropy_decreasing_with_nu(order):
    nus = [0.0, 0.1, 0.5, 1.0, 2.0]
    entropies = [ps_entropy(order, nu) for nu in nus]
    for a, b in zip(entropies, entropies[1:]):
        assert a >= b, f"Entropy should decrease with nu: {entropies}"


# ---------------------------------------------------------------------------
# optimal_nu
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "order,target",
    [
        (16, 3.5),
        (16, 3.9),
        (64, 5.0),
        (64, 5.8),
        (256, 7.0),
    ],
)
def test_optimal_nu_recovers_entropy(order, target):
    nu, achieved = optimal_nu(order, target)
    assert nu >= 0
    assert abs(achieved - target) < 1e-6, (
        f"Achieved entropy {achieved:.8f} != target {target}"
    )


def test_optimal_nu_at_max_entropy_returns_zero():
    order = 16
    nu, achieved = optimal_nu(order, np.log2(order))
    assert nu == 0.0
    assert np.isclose(achieved, np.log2(order), atol=1e-8)


def test_optimal_nu_invalid_entropy():
    with pytest.raises(ValueError):
        optimal_nu(16, 0.0)
    with pytest.raises(ValueError):
        optimal_nu(16, 5.0)  # > log2(16) = 4


# ---------------------------------------------------------------------------
# sample_ps_symbols
# ---------------------------------------------------------------------------


def test_sample_ps_symbols_all_on_constellation():
    order = 16
    pmf = maxwell_boltzmann(order, nu=0.5)
    const = gray_constellation("qam", order).astype(np.complex64)
    symbols = sample_ps_symbols(5000, order, pmf, seed=0)

    assert symbols.shape == (5000,)
    for sym in symbols:
        dists = np.abs(const - sym)
        assert dists.min() < 1e-5, f"Symbol {sym} not on constellation"


def test_sample_ps_symbols_empirical_distribution():
    """Empirical frequencies should approximate the target PMF."""
    order = 16
    nu = 0.8
    pmf = maxwell_boltzmann(order, nu)
    const = gray_constellation("qam", order).astype(np.complex64)
    N = 100_000
    symbols = sample_ps_symbols(N, order, pmf, seed=42)

    # Count frequency per constellation point
    counts = np.zeros(order)
    for m, point in enumerate(const):
        counts[m] = np.sum(np.abs(symbols - point) < 1e-5)
    empirical = counts / N

    np.testing.assert_allclose(empirical, pmf, atol=0.01)


def test_sample_ps_symbols_seed_reproducibility():
    pmf = maxwell_boltzmann(64, nu=0.3)
    s1 = sample_ps_symbols(1000, 64, pmf, seed=7)
    s2 = sample_ps_symbols(1000, 64, pmf, seed=7)
    np.testing.assert_array_equal(s1, s2)


# ---------------------------------------------------------------------------
# generate_psqam factory
# ---------------------------------------------------------------------------


def test_psqam_source_fields_set():
    sig = generate_psqam(500, sps=2, symbol_rate=32e9, order=16, nu=0.5)
    assert sig.source_bits is not None
    assert sig.source_symbols is not None
    assert sig.ps_pmf is not None
    assert sig.mod_scheme == "PS-QAM"
    assert sig.mod_order == 16


def test_psqam_via_entropy():
    target = 3.5
    sig = generate_psqam(1000, sps=2, symbol_rate=32e9, order=16, entropy=target)
    pmf = np.asarray(sig.ps_pmf)
    nz = pmf > 0
    achieved = float(-np.sum(pmf[nz] * np.log2(pmf[nz])))
    assert abs(achieved - target) < 1e-5


def test_psqam_requires_exactly_one_of_nu_entropy():
    with pytest.raises(ValueError):
        generate_psqam(100, sps=2, symbol_rate=1e9, order=16)  # neither
    with pytest.raises(ValueError):
        generate_psqam(
            100, sps=2, symbol_rate=1e9, order=16, nu=0.3, entropy=3.5
        )  # both


def test_psqam_lower_average_energy_than_uniform():
    """PS-QAM symbols must have lower average energy than uniform at same order."""
    order = 64
    nu = 0.3
    sig = generate_psqam(
        10_000, sps=1, symbol_rate=32e9, order=order, nu=nu, pulse_shape="none"
    )
    src = to_device(sig.source_symbols, "cpu")
    avg_energy_ps = float(np.mean(np.abs(src) ** 2))

    # Uniform QAM constellation is normalised to E[|s|²] = 1
    assert avg_energy_ps < 1.0, (
        f"PS-QAM avg energy {avg_energy_ps:.4f} should be < 1.0 (uniform)"
    )


def test_psqam_source_bits_match_symbols():
    """Hard-demapping source_symbols should recover source_bits exactly."""
    from commkit.mapping import demap_symbols_hard

    sig = generate_psqam(
        2000, sps=1, symbol_rate=32e9, order=16, nu=0.5, pulse_shape="none"
    )
    src_sym = to_device(sig.source_symbols, "cpu")
    src_bits = to_device(sig.source_bits, "cpu")
    recovered_bits = demap_symbols_hard(src_sym, "qam", 16)
    np.testing.assert_array_equal(src_bits, recovered_bits)


def test_psqam_ber_computable():
    """BER should be computable end-to-end (source_bits is not None)."""
    sig = generate_psqam(
        5000, sps=1, symbol_rate=32e9, order=16, nu=0.5, pulse_shape="none"
    )
    noisy = apply_awgn(sig.samples, esn0_db=20.0, sps=1)
    sig.samples = noisy
    sig = multirate.resolve_symbols(sig)
    sig = mapping.demap_symbols_hard(sig)
    ber_val = metrics.ber(sig.resolved_bits, sig.source_bits)
    assert 0.0 <= ber_val <= 1.0


# ---------------------------------------------------------------------------
# metrics.mi with PMF
# ---------------------------------------------------------------------------


def test_mi_uniform_pmf_matches_none():
    """Passing explicit uniform PMF must give the same result as pmf=None."""
    order = 16
    sig = generate_qam(5000, sps=1, symbol_rate=32e9, order=order, pulse_shape="none")
    noisy = apply_awgn(sig.samples, esn0_db=15.0, sps=1)

    mi_none = metrics.mi(noisy, "qam", order, noise_var=10 ** (-15.0 / 10))
    pmf_uniform = np.full(order, 1.0 / order)
    mi_uniform = metrics.mi(
        noisy, "qam", order, noise_var=10 ** (-15.0 / 10), pmf=pmf_uniform
    )

    assert abs(mi_none - mi_uniform) < 1e-6


def test_mi_ps_bounded_by_entropy():
    """PS-QAM MI must not exceed H(X)."""
    order = 64
    nu = 0.4
    pmf = maxwell_boltzmann(order, nu)
    nz = pmf > 0
    h_x = float(-np.sum(pmf[nz] * np.log2(pmf[nz])))

    sig = generate_psqam(
        10_000, sps=1, symbol_rate=32e9, order=order, nu=nu, pulse_shape="none"
    )
    noisy = apply_awgn(sig.samples, esn0_db=25.0, sps=1)
    mi_val = metrics.mi(noisy, "qam", order, noise_var=10 ** (-25.0 / 10), pmf=pmf)

    assert mi_val <= h_x + 1e-6, f"MI {mi_val:.4f} exceeds H(X) {h_x:.4f}"
    assert mi_val >= 0.0


# ---------------------------------------------------------------------------
# compute_llr with PMF
# ---------------------------------------------------------------------------


def test_compute_llr_uniform_pmf_matches_none():
    """log_pmf = zeros (uniform) must produce the same LLRs as pmf=None."""
    order = 16
    sig = generate_qam(200, sps=1, symbol_rate=32e9, order=order, pulse_shape="none")
    noisy = apply_awgn(sig.samples, esn0_db=12.0, sps=1)
    noise_var = 10 ** (-12.0 / 10)

    llr_none = compute_llr(
        noisy, "qam", order, noise_var, method="exact", output="numpy"
    )
    pmf_uniform = np.full(order, 1.0 / order)
    llr_uniform = compute_llr(
        noisy, "qam", order, noise_var, method="exact", pmf=pmf_uniform, output="numpy"
    )

    np.testing.assert_allclose(llr_none, llr_uniform, atol=1e-4)


def test_compute_llr_ps_shifts_toward_inner_points():
    """PS LLRs should favour inner-point bits more than uniform LLRs."""
    order = 16
    nu = 1.0
    pmf = maxwell_boltzmann(order, nu)
    const = gray_constellation("qam", order).astype(np.complex64)

    # Transmit the innermost symbol (index with smallest |s|)
    inner_idx = int(np.argmin(np.abs(const)))
    tx_sym = np.array([const[inner_idx]] * 100, dtype=np.complex64)
    noise_var = 0.1
    rx = (
        tx_sym
        + np.random.default_rng(0)
        .normal(0, np.sqrt(noise_var / 2), (100, 2))
        .view(np.complex128)
        .astype(np.complex64)
        .ravel()
    )

    llr_none = compute_llr(rx, "qam", order, noise_var, method="exact", output="numpy")
    llr_ps = compute_llr(
        rx, "qam", order, noise_var, method="exact", pmf=pmf, output="numpy"
    )

    # The magnitudes of PS LLRs should be >= uniform (higher confidence for inner pts)
    assert np.mean(np.abs(llr_ps)) >= np.mean(np.abs(llr_none)) * 0.95


# ---------------------------------------------------------------------------
# rescale_ps_symbols - shared PS-QAM rescale core (metrics/mapping.bits)
# ---------------------------------------------------------------------------


def test_rescale_ps_symbols_uniform_is_noop():
    """pmf=None must return rx unchanged (identity, not just equal)."""
    rx = np.array([1 + 1j, -1 - 1j], dtype=np.complex64)
    result = rescale_ps_symbols(rx, np, "qam", 16, None)
    assert result is rx


def test_rescale_ps_symbols_matches_manual_sqrt_e_ps():
    """The shared helper must match the sqrt(E_PS) rescale it replaced inline."""
    order = 16
    nu = 1.0
    pmf = maxwell_boltzmann(order, nu)
    const = gray_constellation("qam", order)
    e_ps = constellation_power(const, pmf)
    assert e_ps < 1.0 - 1e-6  # shaping must actually lower the average power

    rng = np.random.default_rng(1)
    rx = (const / np.sqrt(e_ps))[rng.integers(0, order, size=50)].astype(np.complex64)

    result = rescale_ps_symbols(rx, np, "qam", order, pmf)
    expected = rx * np.sqrt(e_ps).astype(np.float32)
    np.testing.assert_allclose(result, expected, rtol=1e-5)
