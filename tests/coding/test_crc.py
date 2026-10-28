"""Tests for cyclic redundancy checks (coding/crc.py)."""

from typing import Any

import numpy as np
import pytest

from commkit.backend import to_device
from commkit.coding import CRC, decode, encode
from tests.common.coding import random_bits

ASCII_123456789 = np.unpackbits(np.frombuffer(b"123456789", np.uint8)).astype(np.int8)


def _value(bits: Any) -> int:
    return int("".join(str(int(b)) for b in to_device(bits, "cpu")), 2)


@pytest.mark.parametrize(
    ("crc", "check"),
    [
        (CRC.ethernet(72), 0xCBF43926),  # CRC-32 (IEEE 802.3)
        (CRC.nr("16", 72), 0x31C3),  # CRC-16/XMODEM
        (CRC.nr("24A", 72), 0xCDE703),  # CRC-24/LTE-A
        (CRC.nr("24B", 72), 0x23EF52),  # CRC-24/LTE-B
        (CRC(0x1021, 16, 72, init=0xFFFF), 0x29B1),  # CRC-16/CCITT-FALSE
        (CRC(0x1021, 16, 72, reflect_in=True, reflect_out=True), 0x2189),  # KERMIT
    ],
    ids=["crc32", "nr16", "nr24a", "nr24b", "ccitt-false", "kermit"],
)
def test_standard_check_values(xp: Any, crc: CRC, check: int) -> None:
    """The check value of the ASCII string "123456789" of each catalogued CRC."""
    assert _value(crc.checksum(xp.asarray(ASCII_123456789))) == check


def test_matches_polynomial_division() -> None:
    """Zero init, no reflection: the remainder of u(x) x^r divided by g(x)."""
    crc = CRC.nr("11", 40)
    g = [1] + [(0x621 >> (10 - j)) & 1 for j in range(11)]
    u = np.random.default_rng(0).integers(0, 2, 40)
    reg = list(u) + [0] * 11
    for i in range(40):
        if reg[i]:
            for j in range(12):
                reg[i + j] ^= g[j]
    np.testing.assert_array_equal(crc.checksum(u.astype(np.int8)), reg[40:])


def test_encode_appends_and_decode_checks(xp: Any, xpt: Any) -> None:
    crc = CRC.nr("24C", 100)
    u = random_bits((3, 100), xp, seed=1)
    c = encode(u, code=crc)
    assert c.shape == (3, 124)
    xpt.assert_array_equal(c[:, :100], u)
    res = decode(c, code=crc)
    xpt.assert_array_equal(res.bits, u)
    assert bool(res.valid.all())
    c[1, 7] ^= 1  # any single error is detected
    c[2, 110] ^= 1
    res = decode(c, code=crc)
    assert to_device(res.valid, "cpu").ravel().tolist() == [True, False, False]


def test_validation() -> None:
    with pytest.raises(ValueError, match="degree"):
        CRC(0x3, 0, 8)
    with pytest.raises(ValueError, match="block_length"):
        CRC(0x3, 2, 0)
    with pytest.raises(ValueError, match="polynomial"):
        CRC(0x10, 2, 8)
    with pytest.raises(ValueError, match="multiple of 8"):
        CRC.ethernet(12)
    with pytest.raises(ValueError, match="name"):
        CRC.nr("32", 8)
    assert repr(CRC.nr("6", 10)) == "CRC(0x21, degree=6, k=10)"
    assert CRC.nr("6", 10).rate == pytest.approx(10 / 16)
