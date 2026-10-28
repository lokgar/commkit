"""Tests for GF(2) linear algebra and matmul_mod2 (coding/_gf2.py)."""

from typing import Any

import numpy as np
import pytest

from commkit.coding import _gf2


def _rank_reference(m: np.ndarray) -> int:
    """Rank over GF(2) by plain elimination on Python integers (row bitmasks)."""
    rows = [int("".join(map(str, r)), 2) for r in m.astype(int)]
    rank = 0
    for bit in reversed(range(m.shape[1])):
        pivot = next((r for r in rows if r >> bit & 1), None)
        if pivot is None:
            continue
        rows.remove(pivot)
        rows = [r ^ pivot if r >> bit & 1 else r for r in rows]
        rank += 1
    return rank


def _random(shape: tuple[int, int], seed: int, density: float = 0.5) -> np.ndarray:
    return (np.random.default_rng(seed).random(shape) < density).astype(np.int8)


class TestRowReduce:
    @pytest.mark.parametrize(
        ("shape", "seed"), [((5, 9), 0), ((12, 12), 1), ((30, 70), 2), ((70, 30), 3)]
    )
    def test_rank_matches_plain_elimination(self, shape, seed) -> None:
        m = _random(shape, seed, density=0.3)
        assert _gf2.rank(m) == _rank_reference(m)

    @pytest.mark.parametrize("reverse", [False, True])
    def test_reduced_form_spans_the_rows(self, reverse: bool) -> None:
        m = _random((10, 140), 4)  # more than two 64-bit words per row
        m[7] = m[2] ^ m[5]  # one dependent row
        rref, pivots = _gf2.row_reduce(m, reverse=reverse)
        assert rref.shape == (9, 140)
        np.testing.assert_array_equal(rref[:, pivots], np.eye(9, dtype=np.int8))
        # Same row space: stacking either on the other adds no rank.
        assert _gf2.rank(np.concatenate([rref, m])) == 9
        # Pivots: the greedy independent columns from the chosen side.
        expected: list[int] = []
        for col in range(139, -1, -1) if reverse else range(140):
            if _rank_reference(m[:, [*expected, col]]) > len(expected):
                expected.append(col)
        np.testing.assert_array_equal(pivots, expected)

    def test_rejects_non_matrix(self) -> None:
        with pytest.raises(ValueError, match="matrix"):
            _gf2.row_reduce(np.zeros(4))


class TestNullSpace:
    @pytest.mark.parametrize("reverse", [False, True])
    def test_basis(self, reverse: bool) -> None:
        m = _random((6, 20), 5)
        basis, free = _gf2.null_space(m, reverse=reverse)
        assert basis.shape == (20 - _gf2.rank(m), 20)
        assert not np.any((m.astype(int) @ basis.T.astype(int)) % 2)
        np.testing.assert_array_equal(basis[:, free], np.eye(free.size, dtype=np.int8))
        assert _gf2.rank(basis) == basis.shape[0]

    def test_reverse_frees_the_leftmost_columns(self) -> None:
        a = _random((4, 7), 6)
        h = np.concatenate([a, np.eye(4, dtype=np.int8)], axis=1)
        _, free = _gf2.null_space(h, reverse=True)
        np.testing.assert_array_equal(free, np.arange(7))


class TestInverse:
    def test_inverse(self) -> None:
        rng = np.random.default_rng(7)
        while True:
            m = rng.integers(0, 2, (9, 9)).astype(np.int8)
            if _gf2.rank(m) == 9:
                break
        inv = _gf2.inverse(m)
        np.testing.assert_array_equal((m.astype(int) @ inv) % 2, np.eye(9))

    def test_singular_raises(self) -> None:
        m = np.array([[1, 1], [1, 1]], dtype=np.int8)
        with pytest.raises(ValueError, match="singular"):
            _gf2.inverse(m)
        with pytest.raises(ValueError, match="square"):
            _gf2.inverse(np.ones((2, 3)))


class TestMatmulMod2:
    @pytest.mark.parametrize("shape", [(17,), (5, 17), (2, 3, 17)])
    def test_matches_integer_product(self, xp: Any, xpt: Any, shape) -> None:
        rng = np.random.default_rng(8)
        a = rng.integers(0, 2, shape).astype(np.int8)
        b = rng.integers(0, 2, (17, 11)).astype(np.int8)
        out = _gf2.matmul_mod2(xp.asarray(a), xp.asarray(b))
        assert out.dtype == np.int8
        assert type(out).__module__.split(".")[0] == xp.__name__
        xpt.assert_array_equal(out, xp.asarray((a.astype(int) @ b) % 2))

    def test_chunks_agree(self, xp: Any, xpt: Any, monkeypatch) -> None:
        a = xp.asarray(_random((300, 40), 9))
        b = xp.asarray(_random((40, 25), 10))
        whole = _gf2.matmul_mod2(a, b)
        monkeypatch.setattr(_gf2, "_CHUNK_ELEMENTS", 200)  # 5 rows per chunk
        xpt.assert_array_equal(_gf2.matmul_mod2(a, b), whole)

    def test_errors(self, xp: Any, monkeypatch) -> None:
        with pytest.raises(ValueError, match="cannot multiply"):
            _gf2.matmul_mod2(xp.zeros((2, 3)), xp.zeros((4, 2)))
        monkeypatch.setattr(_gf2, "_MAX_INNER", 3)
        with pytest.raises(ValueError, match="2\\^24"):
            _gf2.matmul_mod2(xp.zeros((2, 3)), xp.zeros((3, 2)))
