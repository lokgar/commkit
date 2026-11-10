"""Closed-form AWGN performance: awgn_ser, awgn_ber, awgn_mi, awgn_gmi.

Checked against independent closed forms in tests/common/theory.py and
against the library's measured metrics on simulated AWGN."""

import logging
import math

import numpy as np
import pytest

import commkit as ck
from commkit.mapping import Constellation
from commkit.metrics import awgn, awgn_ber, awgn_gmi, awgn_mi, awgn_ser
from tests.common import theory

ESN0_DB = [0.0, 6.0, 12.0, 18.0]


def _lin(db):
    return 10 ** (db / 10)


class TestErrorRatesAgainstClosedForms:
    """Exact cases equal the textbook formulas to rounding (rtol 1e-9)."""

    @pytest.mark.parametrize("esn0_db", ESN0_DB)
    def test_square_qam(self, esn0_db):
        e = _lin(esn0_db)
        assert awgn_ber(Constellation.qam(16), esn0_db=esn0_db) == pytest.approx(
            theory.qam16_gray_ber(e), rel=1e-9
        )
        for m in (16, 64, 256):
            c = Constellation.qam(m)
            assert awgn_ser(c, esn0_db=esn0_db) == pytest.approx(
                theory.qam_ser(m, e), rel=1e-9
            )
            assert awgn_ber(c, esn0_db=esn0_db) == pytest.approx(
                theory.qam_gray_ber(m, e), rel=1e-9
            )

    @pytest.mark.parametrize("esn0_db", ESN0_DB)
    @pytest.mark.parametrize("unipolar", [False, True])
    @pytest.mark.parametrize("order", [2, 4, 8])
    def test_pam(self, esn0_db, unipolar, order):
        """Real points see the in-phase half of the noise, N0/2."""
        c = Constellation.pam(order, unipolar=unipolar)
        assert awgn_ber(c, esn0_db=esn0_db) == pytest.approx(
            theory.pam_gray_ber(order, _lin(esn0_db), unipolar=unipolar), rel=1e-9
        )

    @pytest.mark.parametrize("esn0_db", ESN0_DB)
    def test_bpsk_and_qpsk(self, esn0_db):
        """QPSK on the axes is separable after a 45° rotation, BPSK is one
        dimensional: both exact."""
        e = _lin(esn0_db)
        assert awgn_ber(Constellation.psk(2), esn0_db=esn0_db) == pytest.approx(
            theory.bpsk_ber(e), rel=1e-6
        )
        assert awgn_ber(Constellation.psk(4), esn0_db=esn0_db) == pytest.approx(
            theory.qpsk_ber(e), rel=1e-6
        )

    def test_union_bound_against_exact_psk(self):
        """8-PSK has no separable form: its SER is the union bound over all
        pairs, above Craig's exact value and within 1 % of it once the SER is
        below 1e-3 (measured 0.1 % at 18 dB)."""
        c = Constellation.psk(8)
        for esn0_db in (6.0, 12.0, 18.0, 22.0):
            exact = theory.psk_ser(8, _lin(esn0_db))
            bound = awgn_ser(c, esn0_db=esn0_db)
            assert bound >= exact * (1 - 1e-9)
            if exact < 1e-3:
                assert bound < 1.01 * exact

    def test_logs_the_method(self, caplog):
        with caplog.at_level(logging.DEBUG, logger="commkit"):
            awgn_ber(Constellation.psk(4), esn0_db=10.0)
            awgn_ber(Constellation.psk(8), esn0_db=10.0)
        assert "exact" in caplog.text
        assert "union bound" in caplog.text


class TestAgainstSimulation:
    """The closed forms against the library's metrics on simulated AWGN."""

    @pytest.mark.parametrize(
        ("constellation", "esn0_db"),
        [
            (Constellation.qam(64).shaped(entropy=5.0), 16.0),
            (Constellation.pam(4).shaped(entropy=1.6), 10.0),
            (Constellation.psk(8), 14.0),
        ],
        ids=["qam64-shaped", "pam4-shaped", "psk8"],
    )
    def test_error_rates_and_gmi(self, constellation, esn0_db):
        """SER, BER over 2^18 symbols within 5 binomial standard deviations
        of the closed form (the union bound of 8-PSK is 0.6 % above the
        exact SER at 14 dB, inside that); GMI with exact LLRs within 0.01
        bit (Monte Carlo standard error about 0.002)."""
        n = 2**18
        tx = ck.generate(constellation, num_symbols=n, symbol_rate=1e9, rng=11)
        rx = ck.impairments.apply_awgn(tx, esn0_db=esn0_db, rng=12)
        k = constellation.bits_per_symbol
        p_ser = awgn_ser(constellation, esn0_db=esn0_db)
        p_ber = awgn_ber(constellation, esn0_db=esn0_db)
        assert abs(ck.metrics.ser(rx) - p_ser) < 5 * math.sqrt(p_ser / n) + 1e-4
        assert abs(ck.metrics.ber(rx) - p_ber) < 5 * math.sqrt(p_ber / (n * k)) + 1e-4
        measured = ck.metrics.gmi(rx, esn0_db=esn0_db, method="exact")
        assert abs(measured - awgn_gmi(constellation, esn0_db=esn0_db)) < 0.01

    def test_mi(self):
        """MI of shaped 16-QAM against the library's Monte Carlo mi
        (standard error about 0.002 bit over 2^17 symbols)."""
        c = Constellation.qam(16).shaped(entropy=3.5)
        tx = ck.generate(c, num_symbols=2**17, symbol_rate=1e9, rng=13)
        rx = ck.impairments.apply_awgn(tx, esn0_db=8.0, rng=14)
        assert abs(ck.metrics.mi(rx, esn0_db=8.0) - awgn_mi(c, esn0_db=8.0)) < 0.01


class TestInformation:
    @pytest.mark.parametrize("esn0_db", [-5.0, 5.0, 15.0])
    def test_mi_against_planar_quadrature(self, esn0_db):
        """MI of 16-QAM per axis against the independent 2-D quadrature
        (120 nodes per axis), and 8-PSK in the plane: agreement 1e-7."""
        for c in (Constellation.qam(16), Constellation.psk(8)):
            nv = c.power() / _lin(esn0_db)
            expected = theory.awgn_mi(c.points, nv, nodes=120)
            assert awgn_mi(c, esn0_db=esn0_db) == pytest.approx(expected, abs=1e-7)

    @pytest.mark.parametrize(
        "constellation",
        [
            Constellation.qam(16),
            Constellation.qam(64).shaped(entropy=5.0),
            Constellation.psk(8),
            Constellation.pam(4, unipolar=True),
        ],
        ids=["qam16", "qam64-shaped", "psk8", "pam4-unipolar"],
    )
    def test_ordering_and_limits(self, constellation):
        """GMI ≤ MI ≤ H(X); both tend to H(X) at high SNR.  At low SNR MI
        tends to 0 and GMI to H(X) - Σ_b H(B_b), 0 for a uniform prior and
        negative for a shaped one, whose label bits are dependent (-0.147
        at -30 dB for 64-QAM at 5 bits)."""
        esn0 = np.array([-30.0, -5.0, 5.0, 15.0, 40.0])
        mi = awgn_mi(constellation, esn0_db=esn0)
        gmi = awgn_gmi(constellation, esn0_db=esn0)
        h = constellation.entropy
        assert np.all(gmi <= mi + 1e-9) and np.all(mi <= h + 1e-9)
        assert mi[-1] == pytest.approx(h, abs=1e-6)
        assert gmi[-1] == pytest.approx(h, abs=1e-6)
        assert mi[0] < 0.01
        pmf = (
            constellation.pmf
            if constellation.pmf is not None
            else np.full(constellation.order, 1 / constellation.order)
        )
        p1 = pmf @ constellation.bit_labels  # P(bit b = 1)
        bit_entropy = -np.sum(p1 * np.log2(p1) + (1 - p1) * np.log2(1 - p1))
        assert gmi[0] == pytest.approx(h - bit_entropy, abs=0.01)

    def test_quadrature_converged(self, monkeypatch):
        """Doubling the quadrature order changes no rate by more than 1e-6
        bit (measured 3e-7)."""
        cases = [Constellation.qam(16), Constellation.psk(8), Constellation.pam(8)]
        esn0 = np.arange(-5.0, 31.0, 5.0)
        base = [(awgn_mi(c, esn0_db=esn0), awgn_gmi(c, esn0_db=esn0)) for c in cases]
        monkeypatch.setattr(awgn, "_ORDER_1D", 2 * awgn._ORDER_1D)
        monkeypatch.setattr(awgn, "_ORDER_2D", 2 * awgn._ORDER_2D)
        for c, (mi, gmi) in zip(cases, base, strict=True):
            np.testing.assert_allclose(awgn_mi(c, esn0_db=esn0), mi, atol=1e-6)
            np.testing.assert_allclose(awgn_gmi(c, esn0_db=esn0), gmi, atol=1e-6)


class TestInterface:
    def test_scalar_and_array(self):
        c = Constellation.qam(16)
        assert isinstance(awgn_ber(c, esn0_db=10.0), float)
        out = awgn_ser(c, esn0_db=[[5.0, 10.0], [15.0, 20.0]])
        assert isinstance(out, np.ndarray) and out.shape == (2, 2)
        assert np.all(np.diff(out.ravel()) < 0)

    def test_validation(self):
        with pytest.raises(TypeError, match="Constellation"):
            awgn_ber("qam16", esn0_db=10.0)
        with pytest.raises(ValueError, match="finite"):
            awgn_mi(Constellation.qam(4), esn0_db=float("nan"))
