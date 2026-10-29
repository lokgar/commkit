"""Tests for BCH and Reed-Solomon codes (coding/algebraic.py, coding/_galois.py)."""

from itertools import combinations
from math import comb
from typing import Any

import numpy as np
import pytest

from commkit.backend import to_device
from commkit.coding import BCHCode, ReedSolomonCode, decode, encode
from commkit.coding._galois import PRIMITIVE, _Field
from commkit.coding._gf2 import matmul_mod2
from commkit.coding.algebraic import _correct, _correct_lockstep, _xor_reduce
from commkit.mapping.gray import _pack
from tests.common.coding import random_bits


def _clmul_mod(a: int, b: int, prim: int, m: int) -> int:
    """Schoolbook product in GF(2^m): carry-less multiply, reduce by prim."""
    out = 0
    for i in range(m):
        if (b >> i) & 1:
            out ^= a << i
    for i in range(2 * m - 2, m - 1, -1):
        if (out >> i) & 1:
            out ^= prim << (i - m)
    return out


def _poly_eval(field: _Field, coeffs: list[int], x: int) -> int:
    """Horner, highest degree first."""
    acc = 0
    for c in coeffs:
        acc = field.mul_scalar(acc, x) ^ c
    return acc


def _symbols(bits: Any, m: int) -> list[int]:
    b = to_device(bits, "cpu").reshape(-1, m)
    return [int("".join(map(str, row)), 2) for row in b]


def _corrupt(words: Any, rng: np.random.Generator, counts: Any, m: int) -> Any:
    """Add ``counts[r]`` nonzero symbol errors (m-bit symbols) to row r."""
    w = to_device(words, "cpu").copy()
    rows, n = w.shape[0], w.shape[1] // m
    w = w.reshape(rows, n, m)
    for r in range(rows):
        for p in rng.choice(n, int(counts[r]), replace=False):
            value = int(rng.integers(1, 2**m))
            w[r, p] ^= ((value >> np.arange(m - 1, -1, -1)) & 1).astype(np.int8)
    return w.reshape(rows, -1)


class TestField:
    @pytest.mark.parametrize("m", sorted(PRIMITIVE))
    def test_tables_follow_the_primitive_polynomial(self, m: int) -> None:
        f = _Field(m, PRIMITIVE[m])
        exp, log = f._tables
        q = 1 << m
        assert np.unique(exp[: q - 1]).size == q - 1  # alpha has order q - 1
        assert f.alpha(m) == PRIMITIVE[m] ^ q  # alpha^m reduced by p(x)
        rng = np.random.default_rng(m)
        a, b = rng.integers(0, q, 200), rng.integers(0, q, 200)
        expected = [
            _clmul_mod(int(x), int(y), PRIMITIVE[m], m)
            for x, y in zip(a, b, strict=True)
        ]
        np.testing.assert_array_equal(f.mul(a, b), expected)
        nz = b != 0
        np.testing.assert_array_equal(f.mul(f.div(a[nz], b[nz]), b[nz]), a[nz])

    def test_evaluation_matrix_is_multiplication(self) -> None:
        f = _Field(8, 0x11D)
        powers = np.array([[0, 1, 5], [3, 7, 200]])  # two coefficients, three points
        mat = f.evaluation_matrix(powers)
        a = np.array([0x53, 0xCA])
        got = (f.bits(a).reshape(-1) @ mat) % 2
        want = [
            f.mul_scalar(int(a[0]), f.alpha(int(powers[0, i])))
            ^ f.mul_scalar(int(a[1]), f.alpha(int(powers[1, i])))
            for i in range(3)
        ]
        np.testing.assert_array_equal(got.reshape(3, 8), f.bits(want))

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="degree"):
            _Field(4, 0xB)
        with pytest.raises(ValueError, match="not primitive"):
            _Field(4, 0x1F)  # x^4 + x^3 + x^2 + x + 1 has order 5
        with pytest.raises(ValueError, match="m must be"):
            _Field(1, 0x3)


RS_CODES = [
    ReedSolomonCode(15, 11, m=4),
    ReedSolomonCode(15, 9, m=4, first_root=1),
    ReedSolomonCode.g709(),
    ReedSolomonCode.kr4(),
    ReedSolomonCode.kp4(),
    ReedSolomonCode.epon10g(),
    ReedSolomonCode.xgspon(),
]


class TestReedSolomon:
    @pytest.mark.parametrize(
        ("code", "nkm"),
        [
            (ReedSolomonCode.g709(), (255, 239, 8)),
            (ReedSolomonCode.kr4(), (528, 514, 10)),
            (ReedSolomonCode.kp4(), (544, 514, 10)),
            (ReedSolomonCode.epon10g(), (255, 223, 8)),
            (ReedSolomonCode.xgspon(), (248, 216, 8)),
        ],
        ids=str,
    )
    def test_presets(self, code, nkm) -> None:
        n, k, m = nkm
        assert (code.symbols, code.info_symbols, code.m) == nkm
        assert (code.n, code.k, code.t) == (n * m, k * m, (n - k) // 2)

    @pytest.mark.parametrize("code", RS_CODES, ids=str)
    def test_codewords_vanish_at_the_generator_roots(self, code) -> None:
        f = code.field
        g = [int(v) for v in code.generator_polynomial]
        for j in range(2 * code.t):
            assert _poly_eval(f, g, f.alpha(code.first_root + j)) == 0
        word = _symbols(encode(random_bits(code.k, np, seed=1), code=code), code.m)
        for j in range(2 * code.t):
            assert _poly_eval(f, word, f.alpha(code.first_root + j)) == 0

    def test_corrects_every_double_error(self, xp: Any, xpt: Any) -> None:
        """RS(15, 11), t = 2: all position pairs, random nonzero values."""
        code = ReedSolomonCode(15, 11, m=4)
        pairs = list(combinations(range(15), 2))
        u = random_bits((len(pairs), code.k), xp, seed=2)
        c = to_device(encode(u, code=code), "cpu").reshape(len(pairs), 15, 4)
        rng = np.random.default_rng(3)
        for row, (i, j) in enumerate(pairs):
            c[row, i] ^= (
                (int(rng.integers(1, 16)) >> np.arange(3, -1, -1)) & 1
            ).astype(np.int8)
            c[row, j] ^= (
                (int(rng.integers(1, 16)) >> np.arange(3, -1, -1)) & 1
            ).astype(np.int8)
        res = decode(xp.asarray(c.reshape(len(pairs), -1)), code=code)
        xpt.assert_array_equal(res.bits, u)
        assert bool(res.valid.all())

    @pytest.mark.parametrize("code", RS_CODES[2:], ids=str)
    def test_corrects_up_to_t(self, xp: Any, xpt: Any, code) -> None:
        rng = np.random.default_rng(4)
        u = random_bits((64, code.k), xp, seed=5)
        received = _corrupt(
            encode(u, code=code), rng, rng.integers(0, code.t + 1, 64), code.m
        )
        res = decode(xp.asarray(received), code=code)
        xpt.assert_array_equal(res.bits, u)
        assert bool(res.valid.all())

    @pytest.mark.parametrize("code", [RS_CODES[0], ReedSolomonCode.kp4()], ids=str)
    def test_beyond_t_is_detected_or_bounded(self, xp: Any, code) -> None:
        """t + 1 errors: reported invalid, or decoded to a codeword within
        distance t of the received word (bounded-distance decoding)."""
        rng = np.random.default_rng(6)
        u = random_bits((200, code.k), xp, seed=7)
        received = _corrupt(encode(u, code=code), rng, np.full(200, code.t + 1), code.m)
        res = decode(xp.asarray(received), code=code)
        valid = to_device(res.valid, "cpu").ravel()
        recoded = to_device(encode(res.bits, code=code), "cpu")
        diff = (recoded != received).reshape(200, -1, code.m).any(axis=2).sum(axis=1)
        assert np.all(diff[valid] <= code.t)
        assert valid.mean() < (0.9 if code.t == 2 else 0.01)  # rare for large t

    def test_fer_follows_the_binomial_tail(self, backend_device: str) -> None:
        """KP4 fails exactly when a codeword has more than 15 symbol errors:
        the FER at a symbol error rate p is the binomial tail, which at the
        clause 119 threshold (pre-FEC BER 2.4e-4) is about 1e-12."""
        code = ReedSolomonCode.kp4()
        xp = np if backend_device == "cpu" else __import__("cupy")
        rng = np.random.default_rng(8)
        p_bit = 4e-3
        rows = 400
        u = random_bits((rows, code.k), xp, seed=9)
        c = to_device(encode(u, code=code), "cpu")
        flips = rng.random(c.shape) < p_bit
        res = decode(xp.asarray(c ^ flips), code=code)
        fer = float(to_device(res.bits != u, "cpu").any(axis=1).mean())
        p_sym = 1 - (1 - p_bit) ** 10

        def tail(p: float) -> float:
            return sum(
                comb(544, i) * p**i * (1 - p) ** (544 - i) for i in range(16, 545)
            )

        assert fer == pytest.approx(tail(p_sym), abs=4 * np.sqrt(tail(p_sym) / rows))
        assert 1e-13 < tail(1 - (1 - 2.4e-4) ** 10) < 1e-11

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="n - k even"):
            ReedSolomonCode(15, 10, m=4)
        with pytest.raises(ValueError, match="2\\^m - 1"):
            ReedSolomonCode(16, 10, m=4)
        with pytest.raises(ValueError, match="primitive"):
            ReedSolomonCode(15, 11, m=20)
        assert repr(ReedSolomonCode.kp4()) == "ReedSolomonCode(544, 514, m=10)"


class TestBCH:
    @pytest.mark.parametrize(
        ("m", "t", "k"),
        [
            (4, 1, 11),
            (4, 2, 7),
            (4, 3, 5),
            (5, 2, 21),
            (6, 2, 51),
            (7, 2, 113),
            (10, 3, 993),
        ],
    )
    def test_dimensions(self, m: int, t: int, k: int) -> None:
        """The (n, k) of the narrow-sense primitive BCH tables."""
        code = BCHCode(m, t)
        assert (code.n, code.k) == (2**m - 1, k)

    def test_generator_divides_x_n_plus_1(self) -> None:
        code = BCHCode(6, 3)
        g = code._generator
        x = (1 << 63) | 1  # x^63 + 1
        # Polynomial division over GF(2).
        deg = g.bit_length() - 1
        while x.bit_length() - 1 >= deg:
            x ^= g << (x.bit_length() - 1 - deg)
        assert x == 0

    def test_corrects_every_pattern_up_to_t(self, xp: Any, xpt: Any) -> None:
        code = BCHCode(4, 2)  # (15, 7)
        patterns = [()] + [(i,) for i in range(15)] + list(combinations(range(15), 2))
        u = random_bits((len(patterns), code.k), xp, seed=10)
        c = to_device(encode(u, code=code), "cpu")
        for row, pat in enumerate(patterns):
            c[row, list(pat)] ^= 1
        res = decode(xp.asarray(c), code=code)
        xpt.assert_array_equal(res.bits, u)
        assert bool(res.valid.all())

    def test_shortened(self, xp: Any, xpt: Any) -> None:
        code = BCHCode(10, 3, length=500)
        assert (code.n, code.k) == (500, 470)
        rng = np.random.default_rng(11)
        u = random_bits((50, code.k), xp, seed=12)
        received = _corrupt(encode(u, code=code), rng, rng.integers(0, 4, 50), 1)
        xpt.assert_array_equal(decode(xp.asarray(received), code=code).bits, u)

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="t must be"):
            BCHCode(4, 0)
        with pytest.raises(ValueError, match="no information"):
            BCHCode(4, 8)
        with pytest.raises(ValueError, match="length"):
            BCHCode(4, 2, length=8)
        with pytest.raises(ValueError, match="primitive"):
            BCHCode(20, 2)
        assert repr(BCHCode(4, 2)) == "BCHCode(n=15, k=7, t=2)"
        np.testing.assert_array_equal(
            BCHCode(4, 1).generator_polynomial, [1, 0, 0, 1, 1]
        )


class TestXorReduce:
    @pytest.mark.parametrize("m", [1, 4, 8, 10, 16])
    @pytest.mark.parametrize("shape", [(5, 7, 3), (5, 0, 3), (1, 1, 1)])
    def test_matches_the_parity_of_each_bit(self, xp: Any, xpt: Any, m, shape) -> None:
        """The field sum: each bit of the result is the parity of that bit."""
        x = np.random.default_rng(m).integers(0, 1 << m, size=shape, dtype=np.int64)
        for axis in range(3):
            bits = (x[..., None] >> np.arange(m)) & 1
            expected = ((bits.sum(axis=axis) & 1) << np.arange(m)).sum(axis=-1)
            xpt.assert_array_equal(
                _xor_reduce(xp.asarray(x), axis), xp.asarray(expected)
            )


class TestCpuKernel:
    @pytest.mark.parametrize(
        "code",
        [
            ReedSolomonCode(15, 9, m=4),
            ReedSolomonCode(31, 21, m=5, first_root=3),
            ReedSolomonCode.kp4(),
            BCHCode(m=6, t=3),
            BCHCode(m=10, t=3, length=1000),
        ],
        ids=repr,
    )
    def test_matches_the_lockstep_path(self, code) -> None:
        """The Numba decoder equals the lockstep one, up to and beyond t."""
        rng = np.random.default_rng(7)
        m = 1 if code.binary else code.m
        rows = 200
        words = encode(random_bits(rows * code.k, np, seed=3), code=code).reshape(
            rows, -1
        )
        counts = rng.integers(0, 2 * code.t + 3, rows)
        hard = _corrupt(words, rng, counts, m)
        syn_t = code._decoder_tables[0]
        s = _pack(matmul_mod2(hard, syn_t).reshape(rows, 2 * code.t, code.field.m))
        got, ok = _correct(code, hard)
        want, want_ok = _correct_lockstep(code, hard, s)
        np.testing.assert_array_equal(ok, want_ok)
        np.testing.assert_array_equal(got, want)
        assert ok[counts <= code.t].all() and not ok.all()
