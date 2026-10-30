"""Tests for the EXIT analysis of LDPC codes (ldpc_exit_curves, ldpc_threshold
in coding/ldpc.py)."""

import logging
from typing import Any

import numpy as np
import pytest
from scipy import integrate

from commkit.coding import (
    ExitCurve,
    LDPCCode,
    LinearBlockCode,
    SumProduct,
    decoder_exit_curve,
    exit_chart,
    ldpc_exit_curves,
    ldpc_threshold,
)


def _qc(z: int, shifts: np.ndarray, punctured: int = 0) -> LDPCCode:
    """Quasi-cyclic code: one z x z circulant per entry of ``shifts``."""
    rows, cols = [], []
    for i, row in enumerate(shifts):
        for j, s in enumerate(row):
            rows += [i * z + t for t in range(z)]
            cols += [j * z + (t + s) % z for t in range(z)]
    m, n = shifts.shape[0] * z, shifts.shape[1] * z
    return LDPCCode(np.array(rows), np.array(cols), m, n, punctured=punctured)


REGULAR_3_6 = _qc(100, np.random.default_rng(0).integers(0, 100, (3, 6)))


def _bpsk_mi(ebn0_db: float, rate: float) -> float:
    """Information of the BPSK-AWGN channel LLRs, by numerical integration.

    The LLR is N(mu, 2 mu) with mu = 4 R Eb/N0 given bit 0, and the
    information 1 - E[log2(1 + exp(-L))].
    """
    s2 = 8 * rate * 10 ** (ebn0_db / 10)
    mu, s = s2 / 2, np.sqrt(s2)

    def pdf(v: float) -> float:
        return np.exp(-((v - mu) ** 2) / (2 * s2)) / np.sqrt(2 * np.pi * s2)

    loss = integrate.quad(
        lambda v: pdf(v) * np.log2(1 + np.exp(-v)), mu - 40 * s, mu + 40 * s
    )
    return 1.0 - loss[0]


class TestCurves:
    def test_without_a_priori_the_variable_nodes_pass_the_channel(self) -> None:
        """At I_A = 0 a variable node's extrinsic output is its channel LLR."""
        for mi in (0.2, 0.5, 0.9):
            variable, _ = ldpc_exit_curves(REGULAR_3_6, channel_mi=mi)
            assert variable.extrinsic[0] == pytest.approx(mi, abs=2e-3)

    def test_punctured_variables_get_no_channel(self) -> None:
        """At I_A = 0, the punctured edges carry no information: the output is
        the channel information times the unpunctured edge fraction."""
        shifts = np.random.default_rng(1).integers(0, 50, (3, 6))
        code = _qc(50, shifts, punctured=50)
        variable, _ = ldpc_exit_curves(code, channel_mi=0.5)
        edges = np.isin(code.variables, np.arange(code.n))
        assert variable.extrinsic[0] == pytest.approx(0.5 * edges.mean(), abs=2e-3)

    def test_degree_two_checks_are_the_identity(self) -> None:
        """A check of two bits passes the other bit's information unchanged."""
        code = _qc(40, np.array([[0, 3]]))  # every row holds two ones
        _, check = ldpc_exit_curves(code, channel_mi=0.7)
        np.testing.assert_allclose(check.extrinsic, check.a_priori, atol=2e-3)

    def test_endpoints_and_labels(self) -> None:
        variable, check = ldpc_exit_curves(REGULAR_3_6, channel_mi=0.5, points=11)
        assert isinstance(variable, ExitCurve) and variable.a_priori.shape == (11,)
        assert check.extrinsic[0] == pytest.approx(0.0, abs=1e-6)
        assert check.extrinsic[-1] == pytest.approx(1.0, abs=1e-6)
        assert variable.extrinsic[-1] == pytest.approx(1.0, abs=1e-6)
        assert np.all(np.diff(variable.extrinsic) >= 0)
        assert np.all(np.diff(check.extrinsic) >= 0)
        assert variable.label == "variable nodes, I_ch = 0.500"
        assert check.label == "check nodes"

    def test_measured_decoder_curve_of_the_whole_code(self) -> None:
        """The decoder of the whole code, measured, gains information from
        the a-priori LLRs alone once they carry more than its rate."""
        curve = decoder_exit_curve(
            REGULAR_3_6,
            method=SumProduct(early_stop=False),
            points=5,
            num_blocks=40,
            rng=0,
        )
        rate = REGULAR_3_6.k / REGULAR_3_6.n
        assert curve.extrinsic[0] == pytest.approx(0.0, abs=0.01)
        above = curve.a_priori > rate + 0.1
        assert np.all(curve.extrinsic[above] > 0.95)

    def test_early_stopping_is_reported(self, caplog: Any) -> None:
        """With early stopping, a valid a-priori codeword is returned as it
        is: the extrinsic part is zero at I_A = 1, and a warning says so."""
        with caplog.at_level(logging.WARNING, logger="commkit"):
            curve = decoder_exit_curve(
                REGULAR_3_6, method=SumProduct(), points=5, num_blocks=40, rng=0
            )
        assert curve.extrinsic[-1] < 0.05
        (warning,) = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert "40 of 40 codewords" in warning and "early_stop=False" in warning


class TestThreshold:
    def test_regular_3_6(self, caplog: Any) -> None:
        """The (3, 6) ensemble's density-evolution threshold is 1.11 dB Eb/N0;
        the Gaussian approximation lands within 0.1 dB of it."""
        rate = REGULAR_3_6.k / REGULAR_3_6.n
        with caplog.at_level(logging.INFO, logger="commkit"):
            thr = ldpc_threshold(REGULAR_3_6)
        assert _bpsk_mi(1.01, rate) < thr < _bpsk_mi(1.21, rate)
        assert "Belief-propagation threshold" in caplog.records[-1].message

    def test_chart_converges_above_the_threshold_only(self) -> None:
        thr = ldpc_threshold(REGULAR_3_6)
        above = exit_chart(*ldpc_exit_curves(REGULAR_3_6, channel_mi=thr + 0.005))
        below = exit_chart(*ldpc_exit_curves(REGULAR_3_6, channel_mi=thr - 0.005))
        assert above.converges and not below.converges
        assert above.trajectory[-1, 0] >= 0.99
        assert below.trajectory[-1, 0] < 0.99

    def test_trajectory_steps_between_the_curves(self) -> None:
        """Each step goes up to the variable curve and across to the check
        curve, both computed independently of the chart."""
        variable, check = ldpc_exit_curves(REGULAR_3_6, channel_mi=0.6)
        steps = exit_chart(variable, check).trajectory[1:]
        up, across = steps[0::2], steps[1::2]
        np.testing.assert_allclose(
            up[:, 1], np.interp(up[:, 0], variable.a_priori, variable.extrinsic)
        )
        np.testing.assert_allclose(
            across[:, 0], np.interp(across[:, 1], check.a_priori, check.extrinsic)
        )

    def test_degree_one_variable_does_not_close_the_tunnel(self) -> None:
        """DVB-S2's last parity bit has degree 1, which caps the variable
        curve just below 1.  The threshold of the rate-1/2 short code stays
        where its frames start to decode in simulation (notebook 07: an NGMI
        of 0.49-0.50 for QPSK, the information per bit)."""
        code = LDPCCode.dvbs2("1/2", "short")
        assert np.bincount(code.variables).min() == 1
        assert 0.49 < ldpc_threshold(code) < 0.505

    def test_dvbs2_threshold_near_its_rate(self) -> None:
        """A capacity-approaching code opens its tunnel above, but near, an
        information equal to its rate."""
        code = LDPCCode.dvbs2("1/2", "normal")
        assert code.k / code.n < ldpc_threshold(code) < code.k / code.n + 0.05


class TestValidation:
    def test_errors(self) -> None:
        with pytest.raises(TypeError, match="LDPCCode"):
            ldpc_exit_curves(LinearBlockCode.hamming(3), channel_mi=0.5)
        with pytest.raises(TypeError, match="LDPCCode"):
            ldpc_threshold(LinearBlockCode.hamming(3))
        for mi in (0.0, 1.0, -0.1):
            with pytest.raises(ValueError, match="channel_mi"):
                ldpc_exit_curves(REGULAR_3_6, channel_mi=mi)
        with pytest.raises(ValueError, match="points"):
            ldpc_exit_curves(REGULAR_3_6, channel_mi=0.5, points=1)
