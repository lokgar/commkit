"""Tests for polar codes (coding/polar.py)."""

import itertools
from typing import Any

import numpy as np
import pytest

from commkit import generate, mapping, metrics
from commkit.backend import to_device
from commkit.coding import CRC, SC, SCL, PolarCode, decode, encode
from commkit.coding.polar import _NR_SEQUENCE
from commkit.impairments import apply_awgn
from tests.common.coding import random_bits
from tests.common.reference_impl import polar_sc_reference


def _generator(n: int) -> np.ndarray:
    g = np.array([[1]])
    while g.shape[0] < n:
        g = np.kron(g, np.array([[1, 0], [1, 1]]))
    return g


class TestConstruction:
    def test_nr_sequence(self) -> None:
        """TS 38.212 Table 5.3.1.2-1: a permutation of 0 ... 1023."""
        np.testing.assert_array_equal(np.sort(_NR_SEQUENCE), np.arange(1024))
        assert _NR_SEQUENCE[:10].tolist() == [0, 1, 2, 4, 8, 16, 32, 3, 5, 64]
        assert _NR_SEQUENCE[-1] == 1023

    def test_nr_information_set(self) -> None:
        code = PolarCode.nr(40, 128, crc="11")
        assert (code.n, code.k) == (128, 40)
        reliable = _NR_SEQUENCE[_NR_SEQUENCE < 128][-51:]
        np.testing.assert_array_equal(code.info_positions, np.sort(reliable))
        assert code._frozen.sum() == 128 - 51

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="power of two"):
            PolarCode(12, [0, 1])
        with pytest.raises(ValueError, match="distinct"):
            PolarCode(8, [1, 1])
        with pytest.raises(ValueError, match="integers in"):
            PolarCode(8, [9])
        with pytest.raises(ValueError, match="information positions"):
            PolarCode(8, [5, 6, 7], crc=CRC.nr("6", 2))
        with pytest.raises(TypeError, match="CRC"):
            PolarCode(8, [7], crc="6")
        with pytest.raises(ValueError, match="n must be"):
            PolarCode.nr(10, 100)
        with pytest.raises(ValueError, match="fewer"):
            PolarCode.nr(30, 32, crc="6")
        with pytest.raises(ValueError, match="1-D"):
            PolarCode.from_frozen(np.ones((2, 4)))
        with pytest.raises(ValueError, match="list_size"):
            SCL(0)
        assert repr(PolarCode.nr(20, 64, crc="6")) == "PolarCode(n=64, k=20, crc=6)"


class TestEncoding:
    def test_matches_the_kronecker_power(self, xp: Any) -> None:
        code = PolarCode.nr(20, 64)
        u_info = random_bits((5, 20), xp, seed=1)
        x = to_device(encode(u_info, code=code), "cpu")
        u = np.zeros((5, 64), dtype=int)
        u[:, code.info_positions] = to_device(u_info, "cpu")
        np.testing.assert_array_equal(x, (u @ _generator(64)) % 2)

    def test_crc_is_appended(self, xp: Any, xpt: Any) -> None:
        code = PolarCode.nr(30, 64, crc="6")
        u = random_bits((4, 30), xp, seed=2)
        x = encode(u, code=code)
        res = decode(x, code=code)
        xpt.assert_array_equal(res.bits, u)
        assert bool(res.valid.all())


class TestDecoding:
    @pytest.mark.parametrize("n", [32, 64])
    def test_sc_matches_the_recursive_reference(self, xp: Any, n: int) -> None:
        code = PolarCode.nr(n // 2, n)
        llrs = (
            np.random.default_rng(n).standard_normal((50, n)).astype(np.float32) + 0.5
        )
        got = to_device(decode(xp.asarray(llrs), code=code, method=SC()).bits, "cpu")
        for row in range(50):
            u, _ = polar_sc_reference(llrs[row], code._frozen)
            np.testing.assert_array_equal(
                got.reshape(50, -1)[row], u[code.info_positions]
            )
        one = to_device(decode(xp.asarray(llrs), code=code, method=SCL(1)).bits, "cpu")
        np.testing.assert_array_equal(one, got)

    def test_full_list_is_maximum_likelihood(self, xp: Any) -> None:
        """With every path kept, SCL returns the best-correlating codeword."""
        code = PolarCode.from_frozen([1, 1, 1, 0, 1, 0, 0, 0])
        words = np.array(
            [to_device(encode(np.array(w, np.int8), code=code), "cpu")
             for w in itertools.product([0, 1], repeat=code.k)]
        )  # fmt: skip
        llrs = np.random.default_rng(3).standard_normal((300, 8)).astype(np.float32)
        best = np.argmax(llrs @ (1 - 2 * words).T, axis=1)
        ml = np.array(list(itertools.product([0, 1], repeat=code.k)))[best]
        got = decode(xp.asarray(llrs), code=code, method=SCL(16)).bits
        np.testing.assert_array_equal(to_device(got, "cpu").reshape(300, -1), ml)

    def test_list_and_crc_help(self, backend_device: str) -> None:
        code = PolarCode.nr(64, 128, crc="11")
        sig = generate(
            mapping.Constellation.qam(4), 64 * 300, symbol_rate=1e9, code=code, rng=4,
            device=backend_device,
        )  # fmt: skip
        rx = apply_awgn(sig, esn0_db=1.5, rng=5)
        llrs = mapping.compute_llr(rx, noise_var=10**-0.15)
        fer = {}
        for name, method in (("sc", SC()), ("scl", SCL(8))):
            res = decode(llrs, code=code, method=method)
            assert (
                type(res.bits).__module__.split(".")[0]
                == type(llrs).__module__.split(".")[0]
            )
            fer[name] = metrics.fer(
                res.bits, sig.reference.info_bits, block_length=code.k
            )
        assert fer["scl"] < 0.6 * fer["sc"]
        assert 0 < fer["scl"] < 0.5
