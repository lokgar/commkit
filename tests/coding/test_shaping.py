"""Tests for distribution matching: CCDM and PAS (coding/shaping.py)."""

import itertools
from typing import Any

import numpy as np
import pytest

from commkit import generate, mapping, metrics
from commkit.backend import to_device
from commkit.coding import (
    CCDM,
    PAS,
    LDPCCode,
    LinearBlockCode,
    SumProduct,
    decode,
    encode,
)
from commkit.coding.shaping import _composition, _get_numba_ccdm
from commkit.impairments import apply_awgn
from commkit.mapping import Constellation
from tests.common.coding import random_bits

AMPLITUDES = Constellation(np.array([1.0, 3.0, 5.0, 7.0]), pmf=[0.4, 0.3, 0.2, 0.1])
PS64 = Constellation.qam(64).shaped(entropy=5.0)


def _kl(c: np.ndarray, p: np.ndarray) -> float:
    q = c / c.sum()
    nz = q > 0
    return float(np.sum(q[nz] * np.log(q[nz] / p[nz])))


class TestComposition:
    @pytest.mark.parametrize("n", [5, 9, 16])
    def test_minimizes_the_divergence(self, n: int) -> None:
        """Against every composition of n points."""
        p = np.asarray(AMPLITUDES.pmf)
        best = min(
            (c for c in itertools.product(range(n + 1), repeat=4) if sum(c) == n),
            key=lambda c: _kl(np.array(c, float), p),
        )
        got = _composition(p, n)
        assert got.sum() == n
        assert _kl(got.astype(float), p) == pytest.approx(_kl(np.array(best, float), p))


class TestCCDM:
    @pytest.mark.parametrize("n", [3, 4, 6, 8, 11])
    @pytest.mark.parametrize(
        "pmf", [[0.4, 0.3, 0.2, 0.1], [0.25] * 4, [0.7, 0.2, 0.05, 0.05]]
    )
    def test_exhaustive_round_trip(self, n: int, pmf) -> None:
        """Every k-bit input: exact composition, distinct sequences, inverse."""
        cc = CCDM(Constellation(AMPLITUDES.points, pmf=pmf), block_length=n)
        u = np.array(list(itertools.product([0, 1], repeat=cc.k)), dtype=np.int8)
        sym = cc._match(u)
        for row in sym:
            np.testing.assert_array_equal(np.bincount(row, minlength=4), cc.composition)
        assert len({tuple(r) for r in sym}) == len(sym)
        back, valid = cc._dematch(sym)
        np.testing.assert_array_equal(back, u)
        assert valid.all()

    @pytest.mark.parametrize("n", [3, 4, 6, 8])
    @pytest.mark.parametrize(
        "pmf", [[0.4, 0.3, 0.2, 0.1], [0.25] * 4, [0.7, 0.2, 0.05, 0.05]]
    )
    def test_valid_exactly_on_the_image(self, n: int, pmf) -> None:
        """Of all sequences of the composition (more than the 2^k payloads),
        exactly those the matcher emits are valid."""
        cc = CCDM(Constellation(AMPLITUDES.points, pmf=pmf), block_length=n)
        u = np.array(list(itertools.product([0, 1], repeat=cc.k)), dtype=np.int8)
        image = {tuple(r) for r in cc._match(u)}
        every = np.array(list(itertools.product(range(4), repeat=n)), dtype=np.int64)
        every = every[
            (np.stack([(every == i).sum(axis=1) for i in range(4)], axis=1)
             == cc.composition).all(axis=1)
        ]  # fmt: skip
        assert len(every) > len(image)
        _, valid = cc._dematch(every)
        np.testing.assert_array_equal(valid, [tuple(r) in image for r in every])

    def test_two_points_of_four_sequences(self) -> None:
        """Two of each of two points: six sequences, four payloads (k = 2)."""
        cc = CCDM(Constellation.pam(2), block_length=4)
        assert cc.k == 2
        every = np.array(
            [r for r in itertools.product(range(2), repeat=4) if sum(r) == 2]
        )
        _, valid = cc._dematch(every)
        assert valid.sum() == 4

    def test_round_trip_at_operating_size(self, xp: Any, xpt: Any) -> None:
        cc = CCDM(PS64, block_length=20_000)
        u = random_bits((3, cc.k), xp, seed=1)
        labels = encode(u, code=cc)
        assert labels.shape == (3, 20_000 * 6)
        assert type(labels).__module__.split(".")[0] == xp.__name__
        symbols = PS64._nearest(PS64.map(labels[0]))
        np.testing.assert_array_equal(
            np.bincount(to_device(symbols, "cpu"), minlength=64), cc.composition
        )
        res = decode(labels, code=cc)
        xpt.assert_array_equal(res.bits, u)
        assert bool(res.valid.all())

    def test_rate_approaches_entropy(self) -> None:
        """Rate loss H - k/n is positive and O(log n / n)."""
        losses = []
        for n in (100, 1_000, 10_000, 100_000):
            cc = CCDM(AMPLITUDES, block_length=n)
            loss = AMPLITUDES.entropy - cc.k / n
            assert 0 < loss < 4 * np.log2(n) / n
            losses.append(loss)
        assert np.all(np.diff(losses) < 0)

    def test_invalid_composition_is_flagged(self, xp: Any) -> None:
        cc = CCDM(AMPLITUDES, block_length=50)
        labels = encode(random_bits(cc.k, xp, seed=2), code=cc)
        labels[:2] = 1 - labels[:2]  # the first point changes kind
        assert not bool(decode(labels, code=cc).valid.all())

    def test_dematcher_guards_its_buffer(self) -> None:
        """The dematcher flags a block whose interval needs more fractional
        bits than its buffer holds, instead of writing past it."""
        cc = CCDM(AMPLITUDES, block_length=50)
        symbols = cc._match(to_device(random_bits(cc.k, np, seed=3), "cpu")[None])
        bits = np.empty((1, cc.k), dtype=np.int8)
        valid = np.empty(1, dtype=np.bool_)
        dematch = _get_numba_ccdm("dematch")
        dematch(symbols, cc.composition, cc._buffer_bits, bits, valid)
        assert valid[0]
        dematch(symbols, cc.composition, cc.k + 1, bits, valid)
        assert not valid[0]

    def test_validation(self) -> None:
        with pytest.raises(TypeError, match="Constellation"):
            CCDM("qam", block_length=10)
        with pytest.raises(ValueError, match="block_length"):
            CCDM(AMPLITUDES, block_length=0)
        with pytest.raises(ValueError, match="no information bit"):
            CCDM(AMPLITUDES, block_length=2)
        assert CCDM(AMPLITUDES, block_length=10) == CCDM(AMPLITUDES, block_length=10)
        assert "block_length=10" in repr(CCDM(AMPLITUDES, block_length=10))


class TestPAS:
    def test_layout(self) -> None:
        pas = PAS(LDPCCode.dvbs2("3/4"), PS64)
        n_real = 64800 // 3
        assert pas.n == 64800
        assert pas.matcher.block_length == n_real
        assert pas.k == pas.matcher.k + 48600 - 2 * n_real
        # Signs per real dimension, then the two amplitude bits.
        perm = pas._interleaver.permutation.reshape(n_real, 3)
        np.testing.assert_array_equal(perm[:, 1:].ravel(), np.arange(2 * n_real))
        assert set(perm[:, 0]) == set(range(2 * n_real, 64800))

    @pytest.mark.parametrize(
        ("c", "code"),
        [
            (PS64, LDPCCode.dvbs2("3/4", "short")),
            (Constellation.pam(8).shaped(nu=0.05), LDPCCode.dvbs2("3/4", "short")),
            # Rate 1/2 = (m - 1) / m: every sign is a parity bit.
            (
                Constellation.qam(16).shaped(entropy=3.5),
                LinearBlockCode.hamming(3, extended=True),
            ),
        ],
        ids=["64qam-ldpc", "8pam-ldpc", "16qam-hamming"],
    )
    def test_round_trip(self, xp: Any, xpt: Any, c, code) -> None:
        pas = PAS(code, c)
        u = random_bits((2, 2 * pas.k), xp, seed=3)
        labels = encode(u, code=pas)
        res = decode(labels, code=pas)
        xpt.assert_array_equal(res.bits, u)
        assert bool(res.valid.all())

    def test_statistics_follow_the_pmf(self) -> None:
        """Amplitudes from the matcher, signs uniform: the empirical point
        distribution is the shaped pmf."""
        pas = PAS(LDPCCode.dvbs2("3/4"), PS64)
        sig = generate(PS64, 10800 * 6, symbol_rate=1e9, code=pas, rng=4)
        idx = PS64._nearest(sig.reference.symbols)
        freq = np.bincount(idx, minlength=64) / idx.size
        np.testing.assert_allclose(freq, PS64.pmf, atol=2.5e-3)

    def test_decodes_above_the_gmi_threshold(self, backend_device: str) -> None:
        """DVB-S2 rate 3/4 with PS-64-QAM (entropy 5): rate 3.50 bit/symbol.
        It must fail where the GMI is below the rate, and decodes 1 dB above
        that point, the gap Böcherer, Steiner and Schulte (IEEE Trans.
        Commun., 2015) report for DVB-S2 codes."""
        pas = PAS(LDPCCode.dvbs2("3/4"), PS64)
        rate = pas.k / (pas.n / 6)
        sig = generate(
            PS64, 10800 * 6, symbol_rate=1e9, code=pas, rng=5, device=backend_device
        )
        for esn0_db, decodes in ((9.75, False), (11.25, True)):
            rx = apply_awgn(sig, esn0_db=esn0_db, rng=6)
            nv = 10 ** (-esn0_db / 10)
            assert (metrics.gmi(rx, noise_var=nv) > rate) == decodes
            res = decode(
                mapping.compute_llr(rx, noise_var=nv), code=pas, method=SumProduct()
            )
            fer = metrics.fer(res.bits, sig.reference.info_bits, block_length=pas.k)
            assert fer == (0.0 if decodes else 1.0)

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="below"):
            PAS(LDPCCode.dvbs2("1/2"), PS64)
        with pytest.raises(ValueError, match="sign"):
            PAS(LDPCCode.dvbs2("3/4"), Constellation.pam(8, unipolar=True))
        with pytest.raises(ValueError, match="square QAM"):
            PAS(LDPCCode.dvbs2("3/4"), Constellation.qam(32))
        with pytest.raises(ValueError, match="multiple"):
            PAS(LinearBlockCode.hamming(3), PS64)
        mixed = LinearBlockCode(generator=[[1, 1, 0, 1, 0, 0], [0, 1, 1, 1, 1, 1],
                                           [1, 0, 0, 0, 0, 1], [0, 0, 1, 1, 0, 1],
                                           [1, 1, 1, 0, 0, 0]])  # fmt: skip
        with pytest.raises(TypeError, match="not systematic"):
            PAS(mixed, Constellation.pam(8).shaped(nu=0.05))
        with pytest.raises(TypeError, match="Constellation"):
            PAS(LDPCCode.dvbs2("3/4"), "qam")
        assert "PAS(" in repr(PAS(LDPCCode.dvbs2("3/4"), PS64))


class TestGenerateMatched:
    def test_ccdm_payload(self, backend_device: str, xp: Any) -> None:
        cc = CCDM(PS64, block_length=1000)
        sig = generate(
            PS64,
            4000,
            symbol_rate=1e9,
            code=cc,
            num_channels=2,
            rng=7,
            device=backend_device,
        )
        ref = sig.reference
        assert ref.code_block == (6000, cc.k)
        assert ref.info_bits.shape == (2, 4 * cc.k)
        assert isinstance(ref.bits, xp.ndarray)
        cpu = generate(PS64, 4000, symbol_rate=1e9, code=cc, num_channels=2, rng=7)
        np.testing.assert_array_equal(to_device(ref.bits, "cpu"), cpu.reference.bits)
        for block in to_device(ref.symbols, "cpu").reshape(8, 1000):
            np.testing.assert_array_equal(
                np.bincount(PS64._nearest(block), minlength=64), cc.composition
            )

    def test_invalid(self) -> None:
        with pytest.raises(ValueError, match="PAS"):
            generate(PS64, 10800, symbol_rate=1.0, code=LDPCCode.dvbs2("3/4"))
        with pytest.raises(ValueError, match="another constellation"):
            generate(
                Constellation.qam(64), 1000, symbol_rate=1.0,
                code=CCDM(PS64, block_length=1000),
            )  # fmt: skip
