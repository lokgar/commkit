"""Structure of the standard LDPC codes against their standards."""

import numpy as np
import pytest

from commkit.coding import LDPCCode, encode

# ETSI EN 302 307-1, Table 5a (normal) and 5b (short): k of the LDPC code.
K_NORMAL = {"1/4": 16200, "1/3": 21600, "2/5": 25920, "1/2": 32400, "3/5": 38880,
            "2/3": 43200, "3/4": 48600, "4/5": 51840, "5/6": 54000, "8/9": 57600,
            "9/10": 58320}  # fmt: skip
K_SHORT = {"1/4": 3240, "1/3": 5400, "2/5": 6480, "1/2": 7200, "3/5": 9720,
           "2/3": 10800, "3/4": 11880, "4/5": 12600, "5/6": 13320, "8/9": 14400}  # fmt: skip


@pytest.mark.parametrize(
    ("frame", "rate"),
    [("normal", r) for r in K_NORMAL] + [("short", r) for r in K_SHORT],
)
def test_dvbs2_dimensions_and_accumulator(frame: str, rate: str) -> None:
    code = LDPCCode.dvbs2(rate, frame)
    n = 64800 if frame == "normal" else 16200
    k = (K_NORMAL if frame == "normal" else K_SHORT)[rate]
    assert (code.n, code.k, code.num_checks) == (n, k, n - k)
    assert code._encoder.kind == "accumulate"
    degree = np.bincount(code.variables, minlength=n)
    # Parity bits: degree 2, the last 1 (clause 5.3.2).
    np.testing.assert_array_equal(degree[k:-1], 2)
    assert degree[-1] == 1
    # Information bits: two degrees, in whole groups of 360 (Table 7a/7b).
    info = degree[:k].reshape(-1, 360)
    assert np.all(info == info[:, :1])
    assert np.unique(info).size <= 2


@pytest.mark.parametrize(
    ("rate", "high", "count"), [("1/2", 8, 12960), ("3/4", 12, 5400)]
)
def test_dvbs2_degree_distribution(rate: str, high: int, count: int) -> None:
    """ETSI EN 302 307-1, Table 7a: bits of the high degree; the rest have 3."""
    code = LDPCCode.dvbs2(rate)
    degree = np.bincount(code.variables, minlength=64800)[: code.k]
    assert np.count_nonzero(degree == high) == count
    assert np.count_nonzero(degree == 3) == code.k - count


def test_dvbs2_codewords() -> None:
    code = LDPCCode.dvbs2("2/3", "short")
    u = np.random.default_rng(0).integers(0, 2, (3, code.k)).astype(np.int8)
    c = encode(u, code=code).astype(np.int64)
    assert not ((code.parity_check().astype(np.int64) @ c.T) % 2).any()


def test_pon_structure() -> None:
    """ITU-T G.9804.2 Annex B.1: 12 x 69 circulants of 256, 384 punctured."""
    code = LDPCCode.pon()
    assert (code.num_variables, code.num_checks, code.punctured) == (17664, 3072, 384)
    assert (code.n, code.k) == (17280, 14592)
    assert code.rate == pytest.approx(0.8444, abs=1e-4)
    degree = np.bincount(code.variables, minlength=17664).reshape(69, 256)
    assert np.all(degree == degree[:, :1])
    np.testing.assert_array_equal(degree[:, 0], [6] * 17 + [3] * 50 + [12, 11])
    rows = np.bincount(code.checks, minlength=3072).reshape(12, 256)
    assert np.all(rows == rows[:, :1])
