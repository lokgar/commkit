"""Tests for LDPC codes and their layered decoders (coding/ldpc.py)."""

import logging
from itertools import product
from typing import Any

import numpy as np
import pytest
import scipy.sparse

from commkit import generate, mapping, metrics
from commkit.backend import to_device
from commkit.coding import LDPCCode, MinSum, SumProduct, decode, encode
from commkit.coding import ldpc as ldpc_module
from commkit.impairments import apply_awgn
from tests.common.coding import random_bits
from tests.common.reference_impl import ldpc_layered_reference

# A small quasi-cyclic code with an invertible parity part (lifting 7).
QC_BASE = np.array(
    [
        [0, 2, -1, 5, 0, -1, -1, -1],
        [3, -1, 4, 0, 1, 0, -1, -1],
        [-1, 6, 1, 2, -1, 3, 0, -1],
        [1, 0, 3, -1, 2, -1, 4, 0],
    ]
)  # parity part block lower triangular with identities on the diagonal
# Tanner graph without cycles: bitwise MAP is exact BP.
TREE = [[0, 1, 2], [2, 3, 4], [4, 5, 6], [1, 7]]


def _h(rows: list[list[int]], n: int) -> np.ndarray:
    h = np.zeros((len(rows), n), dtype=np.int8)
    for i, r in enumerate(rows):
        h[i, r] = 1
    return h


def _syndrome(code: LDPCCode, words: np.ndarray) -> np.ndarray:
    return (code.parity_check().astype(np.int64) @ words.T.astype(np.int64)) % 2


def _rows_in_schedule(code: LDPCCode) -> list[list[int]]:
    row_ptr, cols, _ = code._schedule
    return [cols[row_ptr[i] : row_ptr[i + 1]].tolist() for i in range(row_ptr.size - 1)]


class TestConstruction:
    def test_from_parity_check_dense_and_sparse(self) -> None:
        h = _h(TREE, 8)
        a = LDPCCode.from_parity_check(h)
        b = LDPCCode.from_parity_check(scipy.sparse.csr_matrix(h))
        assert a == b and hash(a) == hash(b)
        assert (a.n, a.k, a.num_checks) == (8, 4, 4)
        np.testing.assert_array_equal(a.parity_check().toarray(), h)
        assert repr(a) == "LDPCCode(n=8, k=4)"

    def test_quasi_cyclic_circulants(self) -> None:
        code = LDPCCode.quasi_cyclic(QC_BASE, 7)
        h = code.parity_check().toarray()
        assert h.shape == (28, 56)
        for (i, j), s in np.ndenumerate(QC_BASE):
            block = h[7 * i : 7 * i + 7, 7 * j : 7 * j + 7]
            if s < 0:
                assert not block.any()
            else:
                np.testing.assert_array_equal(block, np.roll(np.eye(7), s, axis=1))

    def test_warns_on_4_cycles(self, caplog) -> None:
        with caplog.at_level(logging.WARNING, logger="commkit"):
            LDPCCode.from_parity_check(_h([[0, 1, 2], [0, 1, 3]], 5))
        assert "4-cycles" in caplog.text
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="commkit"):
            LDPCCode.from_parity_check(_h(TREE, 8))
        assert "4-cycles" not in caplog.text

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="only 0 and 1"):
            LDPCCode.from_parity_check([[2, 1, 0]])
        with pytest.raises(ValueError, match="at least two"):
            LDPCCode.from_parity_check(_h([[0, 1], [2]], 4))
        with pytest.raises(ValueError, match="duplicate"):
            LDPCCode([0, 0, 1, 1], [0, 0, 1, 2], 2, 4)
        with pytest.raises(ValueError, match="out of range"):
            LDPCCode([0, 0], [0, 9], 1, 4)
        with pytest.raises(ValueError, match="num_checks"):
            LDPCCode([0, 0], [0, 1], 2, 2)
        with pytest.raises(ValueError, match="equal-length"):
            LDPCCode([0, 0], [0], 1, 4)
        with pytest.raises(ValueError, match=">= 0"):
            LDPCCode.from_parity_check(_h(TREE, 8), punctured=-1)
        with pytest.raises(ValueError, match="parity bits"):
            LDPCCode.from_parity_check(_h(TREE, 8), punctured=5)
        with pytest.raises(ValueError, match="lifting"):
            LDPCCode.quasi_cyclic(QC_BASE, 0)
        with pytest.raises(ValueError, match="shifts"):
            LDPCCode.quasi_cyclic(QC_BASE - 2, 7)
        with pytest.raises(ValueError, match="rate"):
            LDPCCode.dvbs2("9/10", "short")
        with pytest.raises(ValueError, match="frame"):
            LDPCCode.dvbs2("1/2", "medium")

    def test_methods_validation(self) -> None:
        with pytest.raises(ValueError, match="scale"):
            MinSum(scale=0.0)
        with pytest.raises(ValueError, match="offset"):
            MinSum(offset=-1.0)
        with pytest.raises(ValueError, match="max_iterations"):
            SumProduct(max_iterations=0)
        with pytest.raises(ValueError, match="max_iterations"):
            MinSum(max_iterations=True)

    def test_layers_share_no_bits(self) -> None:
        for code in (LDPCCode.quasi_cyclic(QC_BASE, 7), LDPCCode.dvbs2("3/5", "short")):
            row_ptr, cols, layer_ptr = code._schedule
            assert row_ptr[-1] == code.checks.size
            for layer in range(layer_ptr.size - 1):
                bits = cols[row_ptr[layer_ptr[layer]] : row_ptr[layer_ptr[layer + 1]]]
                assert np.unique(bits).size == bits.size
            # Same checks as H, reordered.
            got = sorted(tuple(r) for r in _rows_in_schedule(code))
            h = code.parity_check()
            assert got == sorted(
                tuple(h[i].indices.tolist()) for i in range(h.shape[0])
            )


class TestEncoding:
    @pytest.mark.parametrize(
        ("code", "kind"),
        [
            (LDPCCode.dvbs2("1/2", "short"), "accumulate"),
            (LDPCCode.quasi_cyclic(QC_BASE, 7), "inverse"),
            (LDPCCode.from_parity_check(_h(TREE + [[0, 3, 5, 7]], 8)), "dense"),
        ],
        ids=["accumulate", "inverse", "dense"],
    )
    def test_codewords_satisfy_parity(self, xp: Any, xpt: Any, code, kind) -> None:
        assert code._encoder.kind == kind
        u = random_bits((6, code.k), xp, seed=1)
        c = encode(u.reshape(-1), code=code).reshape(6, -1)
        assert not _syndrome(code, to_device(c, "cpu")).any()
        xpt.assert_array_equal(c[:, code._encoder.info], u)  # systematic
        res = decode(c.reshape(-1), code=code)
        xpt.assert_array_equal(res.bits, u.reshape(-1))
        assert bool(res.valid.all()) and bool((res.iterations == 0).all())

    def test_dense_fallback_size_limit(self, monkeypatch) -> None:
        monkeypatch.setattr(ldpc_module, "_MAX_INVERSE", 1)
        monkeypatch.setattr(ldpc_module, "_MAX_DENSE", 10)
        with pytest.raises(ValueError, match="too large"):
            LDPCCode.quasi_cyclic(QC_BASE, 7)

    def test_punctured_bits_are_erasures(self, xp: Any, xpt: Any) -> None:
        """PON: the last 384 parity bits are never sent; noiseless decoding
        recovers them from the checks."""
        code = LDPCCode.pon()
        u = random_bits(2 * code.k, xp, seed=2)
        c = encode(u, code=code)
        assert c.shape == (2 * 17280,)
        res = decode(c, code=code)
        xpt.assert_array_equal(res.bits, u)
        assert bool(res.valid.all())
        assert res.llrs.shape == (2 * 17280,)


class TestDecoding:
    @pytest.mark.parametrize(
        "method",
        [
            MinSum(scale=0.75, offset=0.1, max_iterations=3, early_stop=False),
            MinSum(scale=1.0, max_iterations=3, early_stop=False),
            SumProduct(max_iterations=3, early_stop=False),
        ],
        ids=["offset", "plain", "sum-product"],
    )
    def test_matches_textbook_reference(self, xp: Any, method) -> None:
        """Numba (and the kernel, on the GPU) against plain float64 layered BP
        in the same row order."""
        code = LDPCCode.quasi_cyclic(QC_BASE, 7)
        llrs = (
            1.0 + 1.5 * np.random.default_rng(3).standard_normal((3, code.n))
        ).astype(np.float32)
        got = to_device(decode(xp.asarray(llrs), code=code, method=method).llrs, "cpu")
        sp = isinstance(method, SumProduct)
        for b in range(3):
            ref = ldpc_layered_reference(
                llrs[b],
                _rows_in_schedule(code),
                sum_product=sp,
                scale=getattr(method, "scale", 1.0),
                offset=getattr(method, "offset", 0.0),
                iterations=3,
            )
            np.testing.assert_allclose(
                got[b], ref, atol=5e-3 if sp else 1e-4, rtol=1e-4
            )

    def test_cycle_free_graph_is_exact(self, xp: Any) -> None:
        """On a tree, sum-product gives the bitwise MAP LLRs and min-sum
        (scale 1) the max-log MAP LLRs."""
        code = LDPCCode.from_parity_check(_h(TREE, 8))
        words = np.array(
            [
                c
                for c in product([0, 1], repeat=8)
                if not _syndrome(code, np.array([c])).any()
            ]
        )
        llrs = np.random.default_rng(4).standard_normal((5, 8)).astype(np.float32)
        metric = -(llrs.astype(np.float64) @ words.T)  # log P(y | c) up to a constant
        exact = np.empty_like(llrs, dtype=np.float64)
        maxlog = np.empty_like(exact)
        for i in range(8):
            zero, one = metric[:, words[:, i] == 0], metric[:, words[:, i] == 1]
            exact[:, i] = np.logaddexp.reduce(zero, axis=1) - np.logaddexp.reduce(
                one, axis=1
            )
            maxlog[:, i] = zero.max(axis=1) - one.max(axis=1)
        x = xp.asarray(llrs)
        sp = decode(
            x, code=code, method=SumProduct(max_iterations=10, early_stop=False)
        )
        ms = decode(
            x, code=code, method=MinSum(scale=1.0, max_iterations=10, early_stop=False)
        )
        np.testing.assert_allclose(
            to_device(sp.llrs, "cpu").reshape(5, 8), exact, atol=2e-5
        )
        np.testing.assert_allclose(
            to_device(ms.llrs, "cpu").reshape(5, 8), maxlog, atol=2e-5
        )

    def test_failed_codewords_report_invalid(self, xp: Any) -> None:
        code = LDPCCode.dvbs2("1/2", "short")
        llrs = xp.asarray(
            np.random.default_rng(5).standard_normal(2 * code.n), dtype=xp.float32
        )
        res = decode(llrs, code=code, method=MinSum(max_iterations=4))
        assert not bool(res.valid.any())
        assert bool((res.iterations == 4).all())

    @pytest.mark.parametrize(
        "method", [MinSum(), SumProduct()], ids=["min-sum", "sum-product"]
    )
    def test_dvbs2_waterfall(self, backend_device: str, method) -> None:
        """QPSK, rate 1/2, normal frames: ETSI EN 302 307-1 Table 13 gives
        Es/N0 = 1.00 dB at a packet error rate of 1e-7 (50 iterations, with
        the outer BCH code).  0.3 dB above it every codeword decodes; 0.6 dB
        below it most fail."""
        code = LDPCCode.dvbs2("1/2")
        c = mapping.Constellation.qam(4)
        for esn0_db, check in ((1.3, lambda f: f == 0.0), (0.4, lambda f: f > 0.5)):
            sig = generate(
                c, 32400 * 12, symbol_rate=1e9, code=code, rng=6, device=backend_device
            )
            rx = apply_awgn(sig, esn0_db=esn0_db, rng=7)
            llrs = mapping.compute_llr(rx, noise_var=10 ** (-esn0_db / 10))
            res = decode(llrs, code=code, method=method)
            fer = metrics.fer(res.bits, sig.reference.info_bits, block_length=code.k)
            assert check(fer), (esn0_db, fer)

    def test_pon_corrects_its_threshold(self, backend_device: str) -> None:
        """LDPC(17280, 14592) soft decoding clears a pre-FEC BER near 2e-2."""
        code = LDPCCode.pon()
        sig = generate(
            mapping.Constellation.qam(4),
            8640 * 8,
            symbol_rate=1e9,
            code=code,
            rng=8,
            device=backend_device,
        )
        rx = apply_awgn(sig, esn0_db=6.0, rng=9)
        assert 0.02 < metrics.ber(rx) < 0.025
        res = decode(mapping.compute_llr(rx, noise_var=10**-0.6), code=code)
        assert metrics.ber(res.bits, sig.reference.info_bits) == 0.0
        assert bool(res.valid.all())
