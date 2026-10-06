"""Tests for Probabilistic Shaping QAM (PS-QAM)."""

from typing import Any

import numpy as np
import pytest

from commkit import generate_psqam, generate_qam, mapping, metrics, multirate
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
from tests.common.conversions import to_numpy


class TestMaxwellBoltzmann:
    """Tests for Maxwell-Boltzmann probability mass function computation."""

    @pytest.mark.parametrize("order,nu", [(16, 0.5), (64, 0.2), (256, 0.1), (16, 2.0)])
    def test_sums_to_one(self, order: int, nu: float) -> None:
        """Verify PMF sums to 1 and has non-negative probabilities."""
        pmf = maxwell_boltzmann(order, nu)
        assert pmf.shape == (order,)
        assert np.isclose(pmf.sum(), 1.0, atol=1e-12)
        np.testing.assert_array_equal(pmf >= 0, True)

    @pytest.mark.parametrize("order", [16, 64, 256])
    def test_uniform_at_zero(self, order: int, xpt: Any) -> None:
        """Zero temperature factor (nu=0) yields a discrete uniform distribution."""
        pmf = maxwell_boltzmann(order, 0.0)
        expected = np.full(order, 1.0 / order)
        xpt.assert_allclose(pmf, expected, atol=1e-14)

    @pytest.mark.parametrize("order", [16, 64])
    def test_inner_higher_probability(self, order: int) -> None:
        """Inner constellation points must have higher probability than outer ones."""
        pmf = maxwell_boltzmann(order, nu=0.5)
        const = gray_constellation("qam", order)
        energies = np.abs(const) ** 2
        for i in range(order):
            for j in range(order):
                if energies[i] < energies[j] - 1e-6:
                    assert pmf[i] >= pmf[j], (
                        f"Expected pmf[{i}]={pmf[i]:.4f} >= pmf[{j}]={pmf[j]:.4f} "
                        f"(energies {energies[i]:.3f} < {energies[j]:.3f})"
                    )


class TestPSEntropy:
    """Tests for constellation entropy under probabilistic shaping."""

    @pytest.mark.parametrize("order", [16, 64, 256])
    def test_entropy_uniform(self, order: int) -> None:
        """nu=0 must achieve theoretical maximum entropy log2(M)."""
        h = ps_entropy(order, nu=0.0)
        assert np.isclose(h, np.log2(order), atol=1e-10)

    @pytest.mark.parametrize("order", [16, 64])
    def test_entropy_decreasing_with_nu(self, order: int) -> None:
        """Entropy must monotonically decrease with increasing shaping parameter nu."""
        nus = [0.0, 0.1, 0.5, 1.0, 2.0]
        entropies = [ps_entropy(order, nu) for nu in nus]
        for a, b in zip(entropies, entropies[1:], strict=False):
            assert a >= b, f"Entropy should decrease with nu: {entropies}"


class TestOptimalNu:
    """Tests for numeric solver recovering optimal nu for target entropy."""

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
    def test_optimal_nu_recovers_entropy(self, order: int, target: float) -> None:
        """optimal_nu must find a parameter achieving target entropy within 1e-6."""
        nu, achieved = optimal_nu(order, target)
        assert nu >= 0
        assert abs(achieved - target) < 1e-6

    def test_optimal_nu_at_max_entropy_returns_zero(self) -> None:
        """Targeting maximum entropy log2(M) returns nu=0."""
        order = 16
        nu, achieved = optimal_nu(order, np.log2(order))
        assert nu == 0.0
        assert np.isclose(achieved, np.log2(order), atol=1e-8)

    def test_optimal_nu_invalid_entropy(self) -> None:
        """Out-of-range target entropy raises ValueError."""
        with pytest.raises(ValueError):
            optimal_nu(16, 0.0)
        with pytest.raises(ValueError):
            optimal_nu(16, 5.0)


class TestSamplePSSymbols:
    """Tests for sampling symbols according to a shaped PMF."""

    def test_sample_symbols_all_on_constellation(self) -> None:
        """Sampled symbols must all match constellation grid points."""
        order = 16
        pmf = maxwell_boltzmann(order, nu=0.5)
        const = gray_constellation("qam", order).astype(np.complex64)
        symbols = sample_ps_symbols(5000, order, pmf, seed=0)

        assert symbols.shape == (5000,)
        for sym in symbols:
            dists = np.abs(const - sym)
            assert dists.min() < 1e-5, f"Symbol {sym} not on constellation"

    def test_sample_symbols_empirical_distribution(self, xpt: Any) -> None:
        """Empirical frequencies must approximate the target PMF."""
        order = 16
        nu = 0.8
        pmf = maxwell_boltzmann(order, nu)
        const = gray_constellation("qam", order).astype(np.complex64)
        n = 100_000
        symbols = sample_ps_symbols(n, order, pmf, seed=42)

        counts = np.zeros(order)
        for m, point in enumerate(const):
            counts[m] = np.sum(np.abs(symbols - point) < 1e-5)
        empirical = counts / n

        xpt.assert_allclose(empirical, pmf, atol=0.01)

    def test_sample_symbols_seed_reproducibility(self, xpt: Any) -> None:
        """Deterministic sampling with identical seed."""
        pmf = maxwell_boltzmann(64, nu=0.3)
        s1 = sample_ps_symbols(1000, 64, pmf, seed=7)
        s2 = sample_ps_symbols(1000, 64, pmf, seed=7)
        xpt.assert_array_equal(s1, s2)


class TestGeneratePSQAM:
    """Tests for the high-level generate_psqam waveform factory."""

    def test_psqam_source_fields_set(self) -> None:
        """Signal container fields and metadata are correctly populated."""
        sig = generate_psqam(500, sps=2, symbol_rate=32e9, order=16, nu=0.5)
        assert sig.source_bits is not None
        assert sig.source_symbols is not None
        assert sig.ps_pmf is not None
        assert sig.mod_scheme == "PS-QAM"
        assert sig.mod_order == 16

    def test_psqam_via_entropy(self) -> None:
        """Specifying target entropy produces matching PMF."""
        target = 3.5
        sig = generate_psqam(1000, sps=2, symbol_rate=32e9, order=16, entropy=target)
        pmf = np.asarray(sig.ps_pmf)
        nz = pmf > 0
        achieved = float(-np.sum(pmf[nz] * np.log2(pmf[nz])))
        assert abs(achieved - target) < 1e-5

    def test_psqam_validation_nu_entropy(self) -> None:
        """generate_psqam requires exactly one of nu or entropy."""
        with pytest.raises(ValueError):
            generate_psqam(100, sps=2, symbol_rate=1e9, order=16)
        with pytest.raises(ValueError):
            generate_psqam(100, sps=2, symbol_rate=1e9, order=16, nu=0.3, entropy=3.5)

    def test_psqam_lower_average_energy_than_uniform(self) -> None:
        """PS-QAM symbols must have lower average energy than uniform constellation."""
        order = 64
        nu = 0.3
        sig = generate_psqam(
            10_000, sps=1, symbol_rate=32e9, order=order, nu=nu, pulse_shape="none"
        )
        src = to_numpy(sig.source_symbols)
        avg_energy_ps = float(np.mean(np.abs(src) ** 2))
        assert avg_energy_ps < 1.0

    def test_psqam_source_bits_match_symbols(self, xp: Any, xpt: Any) -> None:
        """Hard-demapping source_symbols recovers source_bits across backends."""
        from commkit.mapping import demap_symbols_hard

        sig = generate_psqam(
            2000, sps=1, symbol_rate=32e9, order=16, nu=0.5, pulse_shape="none"
        )
        src_sym = xp.asarray(sig.source_symbols)
        src_bits = xp.asarray(sig.source_bits)
        recovered_bits = demap_symbols_hard(src_sym, "qam", 16)
        xpt.assert_array_equal(src_bits, recovered_bits)

    def test_psqam_ber_computable(self, xp: Any) -> None:
        """BER is computable end-to-end with resolved symbols and bits."""
        sig = generate_psqam(
            5000, sps=1, symbol_rate=32e9, order=16, nu=0.5, pulse_shape="none"
        )
        noisy = apply_awgn(xp.asarray(sig.samples), esn0_db=20.0, sps=1)
        sig = sig.replace(samples=noisy)
        sig = multirate.resolve_symbols(sig)
        sig = mapping.demap_symbols_hard(sig)
        ber_val = metrics.ber(sig.resolved_bits, sig.source_bits)
        assert 0.0 <= ber_val <= 1.0


class TestPSQAMMetricsAndDemapping:
    """Tests for mutual information, LLR calculation, and symbol rescaling with PMF."""

    def test_mi_uniform_pmf_matches_none(self) -> None:
        """Passing explicit uniform PMF gives identical mutual information to pmf=None."""
        order = 16
        sig = generate_qam(
            5000, sps=1, symbol_rate=32e9, order=order, pulse_shape="none"
        )
        noisy = apply_awgn(sig.samples, esn0_db=15.0, sps=1)

        mi_none = metrics.mi(noisy, "qam", order, noise_var=10 ** (-15.0 / 10))
        pmf_uniform = np.full(order, 1.0 / order)
        mi_uniform = metrics.mi(
            noisy, "qam", order, noise_var=10 ** (-15.0 / 10), pmf=pmf_uniform
        )
        assert abs(mi_none - mi_uniform) < 1e-6

    def test_mi_ps_bounded_by_entropy(self) -> None:
        """PS-QAM mutual information must not exceed constellation entropy H(X)."""
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

        assert mi_val <= h_x + 1e-6
        assert mi_val >= 0.0

    def test_compute_llr_uniform_pmf_matches_none(self, xpt: Any) -> None:
        """Uniform PMF produces identical LLRs to pmf=None."""
        order = 16
        sig = generate_qam(
            200, sps=1, symbol_rate=32e9, order=order, pulse_shape="none"
        )
        noisy = apply_awgn(sig.samples, esn0_db=12.0, sps=1)
        noise_var = 10 ** (-12.0 / 10)

        llr_none = compute_llr(noisy, "qam", order, noise_var, method="exact")
        pmf_uniform = np.full(order, 1.0 / order)
        llr_uniform = compute_llr(
            noisy,
            "qam",
            order,
            noise_var,
            method="exact",
            pmf=pmf_uniform,
        )
        xpt.assert_allclose(llr_none, llr_uniform, atol=1e-4)

    def test_compute_llr_ps_shifts_toward_inner_points(self) -> None:
        """Shaped PMF gives higher confidence magnitudes to inner points."""
        order = 16
        nu = 1.0
        pmf = maxwell_boltzmann(order, nu)
        const = gray_constellation("qam", order).astype(np.complex64)

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

        llr_none = compute_llr(rx, "qam", order, noise_var, method="exact")
        llr_ps = compute_llr(rx, "qam", order, noise_var, method="exact", pmf=pmf)
        assert np.mean(np.abs(llr_ps)) >= np.mean(np.abs(llr_none)) * 0.95

    def test_rescale_ps_symbols_uniform_is_noop(self) -> None:
        """pmf=None returns symbol array unchanged (identity)."""
        rx = np.array([1 + 1j, -1 - 1j], dtype=np.complex64)
        result = rescale_ps_symbols(rx, np, "qam", 16, None)
        assert result is rx

    def test_rescale_ps_symbols_matches_manual_sqrt_e_ps(self, xpt: Any) -> None:
        """Shared rescale helper matches manual sqrt(E_PS) normalisation."""
        order = 16
        nu = 1.0
        pmf = maxwell_boltzmann(order, nu)
        const = gray_constellation("qam", order)
        e_ps = constellation_power(const, pmf)
        assert e_ps < 1.0 - 1e-6

        rng = np.random.default_rng(1)
        rx = (const / np.sqrt(e_ps))[rng.integers(0, order, size=50)].astype(
            np.complex64
        )

        result = rescale_ps_symbols(rx, np, "qam", order, pmf)
        expected = rx * np.sqrt(e_ps).astype(np.float32)
        xpt.assert_allclose(result, expected, rtol=1e-5)
