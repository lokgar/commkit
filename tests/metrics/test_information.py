"""Tests for the metrics' information rates (metrics/information.py): MI, GMI and NGMI."""

from typing import Any

import numpy as np
import pytest

from commkit import generate, metrics
from commkit.backend import to_device
from commkit.impairments import apply_awgn
from commkit.mapping import Constellation, compute_llr, map_bits
from commkit.metrics import information


class TestInformationMetrics:
    """Tests for Mutual Information (MI) and Generalized Mutual Information (GMI)."""

    def test_gmi_high_snr_approaches_log2m(self, xp: Any) -> None:
        """At infinite SNR (perfect LLRs), GMI -> log2(M)."""
        k = 4
        M = 16
        N = 200
        rng = np.random.default_rng(42)
        bits = rng.integers(0, 2, N * k).astype("int32")
        c = Constellation.qam(M)
        symbols = map_bits(xp.asarray(bits), constellation=c)

        llrs = compute_llr(symbols, noise_var=1e-6, constellation=c)
        gmi_val = metrics.gmi(llrs, bits, constellation=c)
        assert gmi_val > np.log2(M) - 0.05

    def test_gmi_low_snr_approaches_zero(self, xp: Any) -> None:
        """At very low SNR, LLRs collapse to zero -> GMI -> 0."""
        k = 4
        M = 16
        N = 500
        rng = np.random.default_rng(7)
        bits = rng.integers(0, 2, N * k).astype("int32")
        c = Constellation.qam(M)
        symbols = map_bits(xp.asarray(bits), constellation=c)

        llrs = compute_llr(symbols, noise_var=1e6, constellation=c)
        gmi_val = metrics.gmi(llrs, bits, constellation=c)
        assert gmi_val < 0.2

    def test_gmi_multichannel(self, xp: Any) -> None:
        """(C, N k) LLRs give one GMI per channel."""
        rng = np.random.default_rng(3)
        bits = rng.integers(0, 2, (2, 200)).astype("int32")
        c = Constellation.qam(4)
        llrs = compute_llr(
            map_bits(xp.asarray(bits), constellation=c), noise_var=0.1, constellation=c
        )
        out = metrics.gmi(llrs, bits, constellation=c)
        assert isinstance(out, np.ndarray)
        assert out.shape == (2,)
        assert out[0] == pytest.approx(metrics.gmi(llrs[0], bits[0], constellation=c))

    def test_gmi_shape_mismatch_raises(self, xp: Any) -> None:
        """gmi() raises ValueError when llrs and the bits differ in shape."""
        llrs = np.array([1.0, -1.0, 2.0, 0.5])
        bits = np.array([0, 1])
        with pytest.raises(ValueError, match="shape mismatch"):
            metrics.gmi(llrs, bits, constellation=Constellation.qam(4))

    def test_gmi_requires_constellation(self, xp: Any) -> None:
        """k comes from the constellation; an LLR array alone does not give it."""
        with pytest.raises(ValueError, match="constellation"):
            metrics.gmi(np.zeros(4), np.zeros(4))

    def test_gmi_noise_var_with_llrs_raises(self, xp: Any) -> None:
        """noise_var is for Signal input; LLRs already contain it."""
        with pytest.raises(ValueError, match="noise_var"):
            metrics.gmi(
                np.zeros(4),
                np.zeros(4),
                constellation=Constellation.qam(4),
                noise_var=0.1,
            )

    def test_gmi_bounded_by_k(self, xp: Any) -> None:
        """GMI in [0, k]."""
        k = 2
        N = 100
        rng = np.random.default_rng(55)
        bits = rng.integers(0, 2, N * k).astype("int32")
        c = Constellation.qam(4)
        symbols = map_bits(xp.asarray(bits), constellation=c)
        llrs = compute_llr(symbols, noise_var=0.1, constellation=c)
        gmi_val = metrics.gmi(llrs, bits, constellation=c)
        assert 0.0 <= gmi_val <= np.log2(4)

    def test_mi_high_snr_approaches_log2m(self, xp: Any) -> None:
        """At high SNR, MI -> log2(M)."""
        M = 16
        c = Constellation.qam(M)
        rng = np.random.default_rng(42)
        symbols = c.points[rng.integers(0, M, 500)]

        mi_val = metrics.mi(xp.asarray(symbols), noise_var=1e-8, constellation=c)
        assert mi_val > np.log2(M) - 0.1

    def test_mi_never_exceeds_log2m(self, xp: Any) -> None:
        """MI <= log2(M) always (capacity bound)."""
        M = 4
        c = Constellation.qam(M)
        rng = np.random.default_rng(7)
        symbols = c.points[rng.integers(0, M, 200)]

        for noise_var in [1e-4, 0.1, 1.0, 10.0]:
            mi_val = metrics.mi(
                xp.asarray(symbols), noise_var=noise_var, constellation=c
            )
            assert mi_val <= np.log2(M) + 1e-6

    def test_mi_multichannel(self, xp: Any) -> None:
        """(C, N) symbols give one MI per channel (1.x pooled the channels)."""
        c = Constellation.qam(16)
        rng = np.random.default_rng(4)
        symbols = c.points[rng.integers(0, 16, (2, 300))]
        symbols[1] += 0.3 * rng.standard_normal(300)
        out = metrics.mi(xp.asarray(symbols), noise_var=0.05, constellation=c)
        assert out.shape == (2,)
        assert out[0] == pytest.approx(
            metrics.mi(xp.asarray(symbols[0]), noise_var=0.05, constellation=c)
        )
        assert out[1] < out[0]

    def test_mi_does_not_depend_on_the_chunks(self, xp: Any, monkeypatch) -> None:
        """Chunks of 7 symbols (a partial last one) give the one-chunk MI, for
        three channels of 256-QAM."""
        c = Constellation.qam(256)
        rng = np.random.default_rng(5)
        symbols = c.points[rng.integers(0, 256, (3, 200))]
        symbols = symbols + 0.05 * (
            rng.standard_normal((3, 200)) + 1j * rng.standard_normal((3, 200))
        )
        x = xp.asarray(symbols.astype(np.complex64))
        whole = metrics.mi(x, noise_var=0.005, constellation=c)
        monkeypatch.setattr(information, "_MI_ELEMENTS", 3 * 256 * 7)
        np.testing.assert_allclose(
            metrics.mi(x, noise_var=0.005, constellation=c), whole, rtol=1e-12
        )

    def test_mi_decreases_with_noise(self, xp: Any) -> None:
        """MI should decrease as noise increases."""
        M = 16
        c = Constellation.qam(M)
        rng = np.random.default_rng(99)
        symbols = xp.asarray(c.points[rng.integers(0, M, 500)])

        mi_low_noise = metrics.mi(symbols, noise_var=0.01, constellation=c)
        mi_high_noise = metrics.mi(symbols, noise_var=1.0, constellation=c)
        assert mi_low_noise > mi_high_noise


class TestGMIShaped:
    """GMI is the bit-metric decoding rate H(X) - sum_b H(B_b | Y)."""

    @staticmethod
    def _received(c: Constellation, esn0_db: float, n: int = 20000, seed: int = 0):
        rng = np.random.default_rng(seed)
        nv = 10 ** (-esn0_db / 10)
        idx = rng.choice(c.order, n, p=c.pmf)
        noise = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        rx = c.points[idx] + np.sqrt(nv / 2) * noise
        return rx.astype(np.complex64), c.bit_labels[idx].reshape(-1), nv

    @staticmethod
    def _bmd_rate(c: Constellation, rx, bits, nv) -> float:
        """Independent oracle: exact bitwise posteriors from the points, the
        labels and the prior, in float64."""
        log_joint = np.log(c.pmf) - np.abs(rx[:, None] - c.points) ** 2 / nv
        log_joint -= log_joint.max(axis=1, keepdims=True)
        joint = np.exp(log_joint)
        joint /= joint.sum(axis=1, keepdims=True)  # P(s | y)
        sent = bits.reshape(rx.size, -1)
        h_cond = 0.0
        for b in range(c.bits_per_symbol):
            p_one = joint[:, c.bit_labels[:, b] == 1].sum(axis=1)
            p_sent = np.where(sent[:, b] == 1, p_one, 1 - p_one)
            h_cond += -np.mean(np.log2(np.maximum(p_sent, 1e-300)))
        return c.entropy - h_cond

    @pytest.mark.parametrize(
        ("c", "esn0_db"),
        [
            (Constellation.qam(16).shaped(entropy=3.3), 20.0),
            (Constellation.qam(64).shaped(nu=0.075), 0.0),
            (Constellation.qam(64).shaped(nu=0.075), 10.0),
            (Constellation.qam(256).shaped(entropy=7.0), 15.0),
        ],
        ids=["16qam-20dB", "64qam-0dB", "64qam-10dB", "256qam-15dB"],
    )
    def test_shaped_gmi_matches_bmd_oracle_and_bounds(self, c, esn0_db, xp) -> None:
        """GMI equals the independently computed BMD rate and obeys
        GMI <= MI <= H(X)."""
        rx, bits, nv = self._received(c, esn0_db)
        llrs = compute_llr(
            xp.asarray(rx), noise_var=nv, constellation=c, method="exact"
        )
        g = metrics.gmi(llrs, xp.asarray(bits), constellation=c)
        m = metrics.mi(xp.asarray(rx), noise_var=nv, constellation=c)
        assert g == pytest.approx(self._bmd_rate(c, rx, bits, nv), abs=2e-4)
        assert g <= m + 1e-3
        assert m <= c.entropy + 1e-9

    def test_matched_payload_keeps_the_rates(self, xp) -> None:
        """Shaped symbols from generate (a CCDM per record) and symbols drawn
        i.i.d. from the pmf give the same GMI and MI within Monte Carlo
        error: the matcher makes every symbol come from bits without moving
        the rate metrics."""
        c = Constellation.qam(64).shaped(entropy=5.0)
        nv = 10**-1.2
        n = 100_000
        matched = generate(c, num_symbols=n, symbol_rate=1e9, rng=0).reference.symbols
        drawn = c.points[np.random.default_rng(1).choice(64, n, p=c.pmf)].astype(
            np.complex64
        )
        rates = []
        for i, sym in enumerate((matched, drawn)):
            noise = np.random.default_rng(2 + i).standard_normal((2, n)) * np.sqrt(
                nv / 2
            )
            rx = xp.asarray((sym + noise[0] + 1j * noise[1]).astype(np.complex64))
            bits = c.bit_labels[c._nearest(sym)].reshape(-1)
            llrs = compute_llr(rx, noise_var=nv, constellation=c)
            rates.append(
                (
                    metrics.gmi(llrs, xp.asarray(bits), constellation=c),
                    metrics.mi(rx, noise_var=nv, constellation=c),
                )
            )
        assert rates[0][0] == pytest.approx(rates[1][0], abs=0.02)
        assert rates[0][1] == pytest.approx(rates[1][1], abs=0.02)

    def test_uniform_gmi_unchanged(self, xp) -> None:
        """Uniform: H(X) = k, so the rate is the usual k - sum_b E[...]."""
        c = Constellation.qam(16)
        rng = np.random.default_rng(1)
        bits = rng.integers(0, 2, 4000).astype(np.int8)
        rx = (c.map(bits) + 0.2 * rng.standard_normal(1000)).astype(np.complex64)
        llrs = compute_llr(xp.asarray(rx), noise_var=0.08, constellation=c)
        x = -np.asarray(to_device(llrs, "cpu"), np.float64) * (1 - 2.0 * bits)
        sp = (np.log1p(np.exp(-np.abs(x))) + np.maximum(0, x)) / np.log(2)
        expected = 4 * (1 - np.mean(sp))
        # float32 per element, float64 sum: equal to rounding.
        got = metrics.gmi(llrs, xp.asarray(bits), constellation=c)
        assert got == pytest.approx(expected, rel=1e-7)


class TestNGMI:
    """NGMI = 1 - (H(X) - GMI) / m, the soft-FEC threshold metric."""

    @pytest.mark.parametrize(
        "c", [Constellation.qam(16), Constellation.qam(64).shaped(entropy=5.0)]
    )
    def test_relation_to_gmi(self, xp: Any, c: Constellation) -> None:
        sig = apply_awgn(
            generate(c, num_symbols=4000, symbol_rate=1e9, rng=0), esn0_db=9.0, rng=1
        )
        sig = sig.to("gpu" if xp is not np else "cpu")
        nv = 10**-0.9
        g = metrics.gmi(sig, noise_var=nv)
        n = metrics.ngmi(sig, noise_var=nv)
        m = c.bits_per_symbol
        assert n == pytest.approx(1 - (c.entropy - g) / m, abs=1e-12)
        assert 0 < n < 1

    def test_error_free_llrs_give_one(self, xp: Any) -> None:
        c = Constellation.qam(16)
        bits = np.random.default_rng(2).integers(0, 2, 4000).astype(np.int8)
        llrs = compute_llr(
            map_bits(xp.asarray(bits), constellation=c), noise_var=1e-4, constellation=c
        )
        assert metrics.ngmi(llrs, xp.asarray(bits), constellation=c) == pytest.approx(
            1.0, abs=1e-9
        )

    def test_errors(self, xp: Any) -> None:
        llrs = xp.zeros(8, dtype=xp.float32)
        with pytest.raises(ValueError, match="noise_var"):
            metrics.ngmi(
                llrs, xp.zeros(8), constellation=Constellation.qam(4), noise_var=1.0
            )
