"""Tests for interleavers (coding/interleaving.py)."""

from typing import Any

import numpy as np
import pytest

from commkit.backend import to_device
from commkit.coding import Interleaver, deinterleave, interleave


def test_block_reads_columns(xp: Any, xpt: Any) -> None:
    """Written row by row into 2 x 3, read column by column."""
    out = interleave(xp.arange(12), interleaver=Interleaver.block(2, 3))
    xpt.assert_array_equal(out, xp.asarray([0, 3, 1, 4, 2, 5, 6, 9, 7, 10, 8, 11]))


@pytest.mark.parametrize("dtype", [np.int8, np.float32, np.complex64])
def test_round_trip(xp: Any, xpt: Any, dtype) -> None:
    pi = Interleaver.random(16, rng=0)
    x = xp.asarray(np.random.default_rng(1).standard_normal((3, 48)).astype(dtype))
    y = interleave(x, interleaver=pi)
    assert y.dtype == x.dtype and y.shape == x.shape
    xpt.assert_array_equal(deinterleave(y, interleaver=pi), x)
    xpt.assert_array_equal(y[1, 16:32], x[1, 16:32][xp.asarray(pi.permutation)])


def test_random_is_seeded() -> None:
    assert Interleaver.random(50, rng=3) == Interleaver.random(50, rng=3)
    assert Interleaver.random(50, rng=3) != Interleaver.random(50, rng=4)
    np.testing.assert_array_equal(
        np.sort(Interleaver.random(50, rng=3).permutation), np.arange(50)
    )


def test_spreads_bursts() -> None:
    """A burst of `rows` adjacent errors lands in `rows` different rows."""
    pi = Interleaver.block(4, 25)
    burst = np.zeros(100, dtype=np.int8)
    burst[40:44] = 1
    spread = to_device(deinterleave(burst, interleaver=pi), "cpu")
    gaps = np.diff(np.flatnonzero(spread))
    assert np.all(gaps >= 25)


def test_validation(xp: Any) -> None:
    with pytest.raises(ValueError, match="each of"):
        Interleaver([0, 0, 1])
    with pytest.raises(ValueError, match="1-D integer"):
        Interleaver([0.0, 1.0])
    with pytest.raises(ValueError, match="rows"):
        Interleaver.block(0, 3)
    with pytest.raises(ValueError, match="length"):
        Interleaver.random(0)
    with pytest.raises(ValueError, match="multiple"):
        interleave(xp.zeros(7), interleaver=Interleaver.block(2, 2))
    with pytest.raises(ValueError, match="shape"):
        interleave(xp.zeros((2, 2, 4)), interleaver=Interleaver.block(2, 2))
    with pytest.raises(TypeError, match="Interleaver"):
        interleave(xp.zeros(4), interleaver=[1, 0, 2, 3])
    assert repr(Interleaver.block(2, 2)) == "Interleaver(length=4)"
