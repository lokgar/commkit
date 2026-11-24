"""Tests for the encode/decode verbs and DecodeResult (coding/base.py)."""

import dataclasses
import logging
import os
import pickle
import subprocess
import sys
import warnings
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from commkit import Constellation, PrecisionWarning, backend
from commkit.coding import (
    BCJR,
    CCDM,
    CRC,
    SC,
    SCL,
    BCHCode,
    Chase,
    Concatenated,
    ConvolutionalCode,
    Interleaver,
    LDPCCode,
    LinearBlockCode,
    MaximumLikelihood,
    MinSum,
    PolarCode,
    Punctured,
    ReedSolomonCode,
    Shortened,
    SumProduct,
    Syndrome,
    TurboCode,
    Viterbi,
    base,
    decode,
    encode,
)
from tests.common.coding import random_bits

H74 = LinearBlockCode.hamming(3)


class TestEncode:
    def test_layout_per_channel(self, xp: Any, xpt: Any) -> None:
        """(C, B*k) -> (C, B*n); each channel's blocks encode on their own."""
        u = random_bits((3, 4 * 5), xp)
        c = encode(u, code=H74)
        assert c.shape == (3, 7 * 5)
        assert c.dtype == np.int8
        for ch in range(3):
            xpt.assert_array_equal(c[ch], encode(u[ch], code=H74))

    def test_bool_input(self, xp: Any, xpt: Any) -> None:
        u = random_bits(8, xp)
        xpt.assert_array_equal(encode(u.astype(bool), code=H74), encode(u, code=H74))

    def test_empty(self, xp: Any) -> None:
        out = encode(xp.zeros((2, 0), dtype=np.int8), code=H74)
        assert out.shape == (2, 0)

    def test_errors(self, xp: Any) -> None:
        with pytest.raises(TypeError, match="code must be a code object"):
            encode(random_bits(8, xp), code="hamming")
        with pytest.raises(ValueError, match="multiple of the block length 4"):
            encode(random_bits(6, xp), code=H74)
        with pytest.raises(TypeError, match="integers or bool"):
            encode(xp.zeros(8, dtype=xp.float32), code=H74)
        with pytest.raises(ValueError):
            encode(xp.zeros((2, 2, 4), dtype=np.int8), code=H74)

    @pytest.mark.parametrize(
        "bits",
        [[2, 0, 0, 0], [0, -1, 0, 0], [0, 0, 256, 0]],
        ids=["two", "minus-one", "wraps-to-zero-in-int8"],
    )
    def test_nonbinary_bits(self, xp: Any, bits: list[int]) -> None:
        """Values other than 0 and 1 raise instead of encoding another payload."""
        with pytest.raises(ValueError, match="0 or 1"):
            encode(xp.asarray(bits, dtype=xp.int16), code=H74)

    @pytest.mark.parametrize("dtype", ["int8", "int64", "uint8"])
    def test_nonbinary_bits_any_dtype_and_stride(self, xp: Any, dtype: str) -> None:
        """The check reads every integer dtype and strided input."""
        words = xp.zeros((2, 8), dtype=dtype)
        words[1, 2] = 255 if dtype == "uint8" else -1
        assert encode(words[0], code=H74).shape == (14,)
        with pytest.raises(ValueError, match="0 or 1"):
            encode(words[:, 2], code=LinearBlockCode.hamming(3, length=5))


class TestDecode:
    def test_result_layout(self, xp: Any) -> None:
        llrs = xp.ones((2, 7 * 3), dtype=xp.float32)
        res = decode(llrs, code=H74)
        assert res.bits.shape == (2, 12) and res.bits.dtype == np.int8
        assert res.valid.shape == (2, 3) and res.valid.dtype == bool
        assert res.iterations.shape == (2, 3) and res.iterations.dtype == np.int32
        assert res.llrs is None
        assert res.checked is False  # Hamming(7, 4) is perfect
        res1 = decode(llrs[0], code=H74)
        assert res1.bits.shape == (12,) and res1.valid.shape == (3,)

    def test_hard_input_is_the_sign(self, xp: Any, xpt: Any) -> None:
        """Integer bits decode as LLRs of sign (1 - 2b): ML on hard input is
        minimum Hamming distance."""
        bits = random_bits(7 * 40, xp, seed=1)
        llrs = (1.0 - 2.0 * bits).astype(xp.float32)
        for method in (Syndrome(), MaximumLikelihood()):
            a = decode(bits, code=H74, method=method)
            b = decode(llrs, code=H74, method=method)
            xpt.assert_array_equal(a.bits, b.bits)

    def test_double_warns(self, xp: Any) -> None:
        with pytest.warns(PrecisionWarning):
            decode(xp.ones(7, dtype=xp.float64), code=H74)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            decode(xp.ones(7, dtype=xp.float32), code=H74)

    def test_empty(self, xp: Any) -> None:
        res = decode(xp.zeros((2, 0), dtype=xp.float32), code=H74)
        assert res.bits.shape == (2, 0) and res.valid.shape == (2, 0)

    def test_errors(self, xp: Any) -> None:
        llrs = xp.ones(7, dtype=xp.float32)
        with pytest.raises(TypeError, match="code must be a code object"):
            decode(llrs, code=None)
        with pytest.raises(TypeError, match="does not decode"):
            decode(llrs, code=H74, method="syndrome")
        with pytest.raises(ValueError, match="multiple of the block length 7"):
            decode(xp.ones(8, dtype=xp.float32), code=H74)
        with pytest.raises(TypeError, match="real"):
            decode(xp.ones(7, dtype=xp.complex64), code=H74)


class TestDecodeLogging:
    def test_logs_failed_checks(self, xp: Any, caplog: Any) -> None:
        """One INFO line per call with the blocks whose own check failed."""
        h16 = LinearBlockCode.hamming(4, extended=True)
        llrs = xp.ones(3 * 16, dtype=xp.float32)
        llrs[[0, 1]] = -1.0  # two errors: detected, not correctable
        with caplog.at_level(logging.INFO, logger="commkit"):
            decode(llrs, code=h16, method=Syndrome(max_weight=1))
        lines = [r.message for r in caplog.records if r.levelno == logging.INFO]
        assert len(lines) == 1
        assert "Decoded 3 block(s)" in lines[0]
        assert "1 failed the decoder's check." in lines[0]

    def test_logs_no_check(self, xp: Any, caplog: Any) -> None:
        """A decoder without a success check does not report zero failures."""
        with caplog.at_level(logging.INFO, logger="commkit"):
            decode(xp.ones(7, dtype=xp.float32), code=H74, method=MaximumLikelihood())
        (line,) = [r.message for r in caplog.records if r.levelno == logging.INFO]
        assert "the decoder has no success check." in line

    def test_no_op_when_info_disabled(self, xp: Any, caplog: Any) -> None:
        """No log line and no host transfer when INFO is off."""
        with (
            caplog.at_level(logging.WARNING, logger="commkit"),
            patch.object(base, "to_device") as transfer,
        ):
            decode(xp.ones(7, dtype=xp.float32), code=H74)
        transfer.assert_not_called()
        assert not caplog.records


H16 = LinearBlockCode.hamming(4, extended=True)  # (16, 11), distance 4
LDPC74 = LDPCCode.from_parity_check(H74.parity_check)

# (code, method, checked): whether each decoder's valid can be False.
CHECKED_CASES = [
    (H74, Syndrome(), False),  # perfect code: every syndrome has a leader
    (H74, Syndrome(max_weight=1), False),
    (H16, Syndrome(), False),  # complete decoding
    (H16, Syndrome(max_weight=1), True),  # detects double errors
    (H16, MaximumLikelihood(), False),
    (H16, Chase(2, hard=Syndrome(max_weight=1)), True),
    (H74, Chase(2), False),
    (BCHCode(4, 2), None, True),
    (BCHCode(4, 2), Chase(2), True),
    (ReedSolomonCode.kp4(), None, True),
    (LDPCCode.dvbs2("1/2", frame="short"), MinSum(), True),
    (LDPCCode.dvbs2("1/2", frame="short"), SumProduct(), True),
    (PolarCode.nr(k=32, n=64), SC(), False),
    (PolarCode.nr(k=32, n=64, crc="11"), SCL(4), True),
    (ConvolutionalCode.k7(32), Viterbi(), False),
    (ConvolutionalCode.k7(32), BCJR(), False),
    (TurboCode.lte(40), None, False),
    (CRC.nr("6", info_bits=10), None, True),
    (CCDM(Constellation.pam(2), num_symbols=4), None, True),
    (Punctured(H16, [15]), Syndrome(max_weight=1), True),
    (Shortened(H16, 2), Syndrome(), False),
    (Concatenated.ieee_802_3dj(), None, True),
    # valid is the outer decoder's, so a checked inner code does not count.
    (Concatenated(H74, CRC.nr("6", info_bits=7)), None, False),
]


def _case_id(case: Any) -> str:
    code, method, _ = case
    return f"{code!r}-{type(method).__name__}"


class TestChecked:
    @pytest.mark.parametrize("case", CHECKED_CASES, ids=_case_id)
    def test_flag(self, case: Any, xp: Any) -> None:
        """Set per code and decoder, also for empty input (nothing decoded)."""
        code, method, expected = case
        res = decode(xp.zeros(0, dtype=xp.float32), code=code, method=method)
        assert res.checked is expected
        assert base._checked(code, method) is expected

    @pytest.mark.parametrize(
        "case",
        [
            c
            for c in CHECKED_CASES
            if c[0].n <= 1000
            and not isinstance(c[0], CCDM)
            # A test word flipping an odd number of bits always has a single
            # error's syndrome, so this Chase never fails.
            and not (c[0] == H16 and isinstance(c[1], Chase))
        ],
        ids=_case_id,
    )
    def test_valid_follows_the_flag(self, case: Any, xp: Any) -> None:
        """LLRs of random words (not codewords): an unchecked decoder reports
        every block valid; a checked one flags some of the 64."""
        code, method, expected = case
        rng = np.random.default_rng(3)
        llrs = xp.asarray(rng.normal(0.0, 1.0, 64 * code.n).astype(np.float32))
        res = decode(llrs, code=code, method=method)
        assert res.checked is expected
        flagged = int(xp.count_nonzero(~res.valid))
        assert flagged > 0 if expected else flagged == 0


class TestCodeValue:
    def test_equality_and_hash(self) -> None:
        a = LinearBlockCode.hamming(4)
        b = LinearBlockCode(generator=a.generator.copy())
        assert a == b and hash(a) == hash(b)
        assert a != LinearBlockCode.hamming(4, extended=True)
        assert a != "code"
        assert a.rate == pytest.approx(11 / 15)

    def test_arrays_are_read_only(self) -> None:
        with pytest.raises(ValueError):
            H74.generator[0, 0] = 1

    @pytest.mark.parametrize(
        "build",
        [
            lambda a: LinearBlockCode(generator=a(H74.generator)),
            lambda a: LinearBlockCode.from_parity_check(a(H74.parity_check)),
            lambda a: LDPCCode(a(LDPC74.checks), a(LDPC74.variables), 3, 7),
            lambda a: LDPCCode.from_parity_check(
                a(
                    np.array(
                        [[1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 1, 0], [1, 0, 0, 0, 1, 1]]
                    )
                )
            ),
            lambda a: LDPCCode.quasi_cyclic(
                a(np.array([[0, 0, 1, -1], [-1, 1, 0, 0]])), lifting=5
            ),
            lambda a: PolarCode(8, a(np.array([3, 5, 6, 7]))),
            lambda a: PolarCode.from_frozen(
                a(np.array([True, True, True, False, True, False, False, False]))
            ),
            lambda a: Punctured(H74, a(np.array([6]))),
            lambda a: Interleaver(a(np.array([2, 0, 3, 1]))),
            lambda a: Interleaver(a(np.arange(512)[::-1])),
        ],
        ids=[
            "linear",
            "linear-from-h",
            "ldpc",
            "ldpc-from-h",
            "ldpc-qc",
            "polar",
            "polar-from-frozen",
            "punctured",
            "interleaver",
            "interleaver-512",
        ],
    )
    def test_device_arrays_become_host_copies(self, xp: Any, build: Any) -> None:
        """Array fields given on the device equal the host-built object and
        are read-only NumPy copies; a NumPy argument stays writable."""
        on_device = build(xp.asarray)
        on_host = build(np.array)
        assert on_device == on_host
        for f in dataclasses.fields(on_device):
            value = getattr(on_device, f.name)
            if hasattr(value, "shape"):
                assert isinstance(value, np.ndarray) and not value.flags.writeable
        mine = np.array(H74.generator)
        LinearBlockCode(generator=mine)
        assert mine.flags.writeable

    def test_pickle_leaves_the_hash_behind(self) -> None:
        """A hash of bytes is valid in its own process only."""
        code = LinearBlockCode.hamming(4)
        hash(code)
        copy = pickle.loads(pickle.dumps(code))
        assert "_hash" not in copy.__dict__
        assert copy == code and hash(copy) == hash(code)

    def test_pickled_code_hashes_like_a_new_one_in_another_process(self) -> None:
        script = (
            "import pickle, sys; from commkit.coding import LinearBlockCode; "
            "c = pickle.loads(sys.stdin.buffer.read()); "
            "print(hash(c) == hash(LinearBlockCode.hamming(4)))"
        )
        code = LinearBlockCode.hamming(4)
        hash(code)
        env = {**os.environ, "PYTHONHASHSEED": "12345"}
        out = subprocess.run(
            [sys.executable, "-c", script], input=pickle.dumps(code),
            capture_output=True, env=env, check=True,
        )  # fmt: skip
        assert out.stdout.strip() == b"True"

    def test_equal_codes_share_device_tables(self, xp: Any) -> None:
        a = LinearBlockCode.hamming(5)
        b = LinearBlockCode(generator=a.generator.copy())
        encode(random_bits(26, xp), code=a)
        tables = dict(backend._DEVICE_TABLES[a])
        encode(random_bits(26, xp), code=b)
        assert backend._DEVICE_TABLES[b].keys() == tables.keys()
        for key, value in tables.items():
            assert backend._DEVICE_TABLES[b][key] is value


class TestUpfrontValidation:
    """Composite codes validate wrapped decoder methods upfront (even on empty
    input), and code constructors reject bool/float integer arguments."""

    def test_composite_codes_reject_bad_inner_method_on_empty(self, xp: Any) -> None:
        from commkit.coding import PAS, StageDecoders
        from commkit.coding.product import _Repeat

        empty = xp.zeros(0, dtype=xp.float32)
        pas = PAS(
            LinearBlockCode.hamming(4, extended=True),
            Constellation.pam(4).shaped(nu=0.2),
        )
        concat = Concatenated(H74, H16)
        rep = _Repeat(H74, 2)
        for code, bad in (
            (pas, "garbage"),
            (Punctured(H74, [6]), "garbage"),
            (Shortened(H74, 1), "garbage"),
            (rep, "garbage"),
            (concat, StageDecoders(inner="garbage")),
            (concat, StageDecoders(outer="garbage")),
            (concat, "garbage"),
        ):
            with pytest.raises(TypeError, match="does not decode"):
                decode(empty, code=code, method=bad)

    def test_strict_int_and_construction_validation(self) -> None:
        with pytest.raises(TypeError, match="integer"):
            ReedSolomonCode(True, 1, m=3)
        with pytest.raises(TypeError, match="integer"):
            ReedSolomonCode(7.0, 5, m=3)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            # Non-primitive polynomial must fail on construction via self.field.
            ReedSolomonCode(7, 5, m=3, primitive=0b1111)
        with pytest.raises(TypeError, match="integer"):
            BCHCode(4, True)
        with pytest.raises(TypeError, match="integer"):
            BCHCode(4.0, 1)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="n must be"):
            PolarCode.nr(k=16, n=32.0)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="k must be"):
            PolarCode.nr(k=True, n=32)
        with pytest.raises(ValueError, match="2-D"):
            LDPCCode.from_parity_check(np.array([1, 0, 1]))
        with pytest.raises(TypeError, match="lifting"):
            LDPCCode.quasi_cyclic(np.array([[0, 1], [1, 0]]), lifting=True)
        with pytest.raises(ValueError, match="integer"):
            LDPCCode.quasi_cyclic(np.array([[0.5, 1.0], [1.0, 0.0]]), lifting=4)
        with pytest.raises(TypeError, match="polynomial"):
            CRC(True, 8, 16)
        with pytest.raises(TypeError, match="generators"):
            ConvolutionalCode((0o5, True), 3, 10)
        with pytest.raises(TypeError, match="generators"):
            ConvolutionalCode((5.5, 7), 3, 10)  # type: ignore[arg-type]
