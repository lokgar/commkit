"""Tests for the error-rate curves of codes (coding/performance.py)."""

from typing import Any

import numpy as np
import pytest

from commkit.coding import (
    CCDM,
    PAS,
    ErrorRateCurve,
    LDPCCode,
    LinearBlockCode,
    MinSum,
    Syndrome,
    error_rate_curve,
    performance,
)
from commkit.mapping import Constellation
from tests.common import scoring, theory


class TestErrorRateCurve:
    """error_rate_curve: Monte Carlo counts against closed forms, the stop
    rule and the outcome split."""

    UNCODED = LinearBlockCode(generator=np.eye(8, dtype=int))  # n = k = 8

    @pytest.mark.parametrize(
        ("constellation", "theory_ber"),
        [
            (Constellation.pam(2), theory.bpsk_ber),
            (Constellation.qam(4), theory.qpsk_ber),
            (Constellation.qam(16), theory.qam16_gray_ber),
        ],
        ids=["bpsk", "qpsk", "16qam"],
    )
    def test_uncoded_ber_is_theory(
        self, backend_device: str, constellation: Constellation, theory_ber: Any
    ) -> None:
        """With n = k and no parity the decoder passes the hard decisions
        on: BER and pre-FEC BER both equal the AWGN closed form, within 4
        sigma of 8000 blocks (64000 bits).  BPSK sees the in-phase noise
        (N0 / 2): 0.079 at 0 dB, where QPSK has 0.159."""
        esn0_db = np.array([0.0, 4.0, 8.0])
        curve = error_rate_curve(
            self.UNCODED,
            constellation=constellation,
            esn0_db=esn0_db,
            min_block_errors=10**6,
            max_blocks=8000,
            rng=1,
            device=backend_device,
        )
        np.testing.assert_array_equal(curve.num_blocks, 8000)
        np.testing.assert_array_equal(curve.bit_errors, curve.pre_fec_bit_errors)
        for e, ber in zip(esn0_db, curve.ber, strict=True):
            p = theory_ber(10 ** (e / 10))
            assert abs(ber - p) < scoring.binomial_margin(p, 64000), (e, ber, p)

    def test_hamming_hard_decision_fer(self, backend_device: str) -> None:
        """Hamming(7, 4) with syndrome decoding corrects exactly the single
        errors: FER = 1 - (1-p)^7 - 7p(1-p)^6 at the QPSK bit error rate p
        (0.0097 at 6 dB, 0.30 at 0 dB), within 4 sigma of 20000 blocks.  An
        uncorrecting decoder would give 1 - (1-p)^4 on the information bits
        (0.089 at 6 dB)."""
        esn0_db = np.array([0.0, 6.0])
        curve = error_rate_curve(
            LinearBlockCode.hamming(3),
            constellation=Constellation.qam(4),
            esn0_db=esn0_db,
            min_block_errors=10**6,
            max_blocks=20_000,
            rng=2,
            device=backend_device,
        )
        for e, fer in zip(esn0_db, curve.fer, strict=True):
            f = theory.bounded_distance_fer(theory.qpsk_ber(10 ** (e / 10)), 7, 1)
            assert abs(fer - f) < scoring.binomial_margin(f, 20_000), (e, fer, f)
        # Perfect code, complete decoding: no failure can be detected.
        assert not curve.checked
        np.testing.assert_array_equal(curve.detected_failures, 0)
        np.testing.assert_array_equal(curve.undetected_errors, curve.block_errors)
        np.testing.assert_array_equal(curve.total_iterations, curve.num_blocks)

    def test_stop_rule(self, monkeypatch: Any) -> None:
        """Batches of 10 blocks: at -10 dB (FER near 1) the first batch
        reaches 5 block errors and the point stops there; at 20 dB (no
        errors) the point stops at max_blocks, the last batch cut to it."""
        monkeypatch.setattr(performance, "_BATCH_BITS", 70)
        curve = error_rate_curve(
            LinearBlockCode.hamming(3),
            constellation=Constellation.qam(4),
            esn0_db=[-10.0, 20.0],
            min_block_errors=5,
            max_blocks=25,
            rng=3,
        )
        np.testing.assert_array_equal(curve.num_blocks, [10, 25])
        assert curve.block_errors[0] >= 5 and curve.block_errors[1] == 0

    def test_outcome_split(self) -> None:
        """Extended Hamming(16, 11) with Syndrome(max_weight=1) detects the
        double errors: at 3 dB some blocks are flagged, flagged blocks are
        not counted as undetected, and every block in error is flagged or
        undetected."""
        curve = error_rate_curve(
            LinearBlockCode.hamming(4, extended=True),
            constellation=Constellation.qam(4),
            esn0_db=[3.0],
            method=Syndrome(max_weight=1),
            min_block_errors=200,
            rng=4,
        )
        assert curve.checked
        assert curve.detected_failures[0] > 0
        assert curve.undetected_errors[0] <= curve.block_errors[0]
        assert (
            curve.block_errors[0]
            <= curve.detected_failures[0] + curve.undetected_errors[0]
        )
        assert curve.bit_errors[0] >= curve.block_errors[0]

    def test_shaped_with_pas(self) -> None:
        """PAS onto probabilistically shaped 64-QAM decodes every block at
        16 dB, and none at 6 dB; Eb/N0 counts the information bits per
        symbol, k/n * 6."""
        ps64 = Constellation.qam(64).shaped(entropy=5.0)
        code = PAS(LDPCCode.dvbs2("3/4", "short"), ps64)
        curve = error_rate_curve(
            code,
            constellation=ps64,
            esn0_db=[6.0, 16.0],
            min_block_errors=4,
            max_blocks=4,
            rng=5,
        )
        np.testing.assert_array_equal(curve.block_errors, [4, 0])
        np.testing.assert_allclose(
            curve.ebn0_db, curve.esn0_db - 10 * np.log10(code.k / code.n * 6)
        )

    def test_reproducible(self) -> None:
        kwargs = dict(
            constellation=Constellation.qam(4), esn0_db=[2.0], max_blocks=500, rng=6
        )
        a = error_rate_curve(LinearBlockCode.hamming(3), **kwargs)
        b = error_rate_curve(LinearBlockCode.hamming(3), **kwargs)
        np.testing.assert_array_equal(a.bit_errors, b.bit_errors)
        np.testing.assert_array_equal(a.pre_fec_bit_errors, b.pre_fec_bit_errors)

    def test_validation(self) -> None:
        h = LinearBlockCode.hamming(3)
        q = Constellation.qam(4)
        with pytest.raises(TypeError, match="code must be a code object"):
            error_rate_curve("hamming", constellation=q, esn0_db=[0.0])
        with pytest.raises(TypeError, match="does not decode"):
            error_rate_curve(h, constellation=q, esn0_db=[0.0], method=MinSum())
        with pytest.raises(TypeError, match="Constellation"):
            error_rate_curve(h, constellation="qpsk", esn0_db=[0.0])
        with pytest.raises(ValueError, match="needs a distribution matcher"):
            error_rate_curve(
                h, constellation=Constellation.qam(64).shaped(entropy=5.0), esn0_db=[0]
            )
        with pytest.raises(ValueError, match="another constellation"):
            error_rate_curve(
                CCDM(Constellation.pam(4).shaped(entropy=1.5), num_symbols=8),
                constellation=Constellation.pam(4),
                esn0_db=[0.0],
            )
        with pytest.raises(ValueError, match="esn0_db"):
            error_rate_curve(h, constellation=q, esn0_db=[])
        with pytest.raises(ValueError, match="min_block_errors"):
            error_rate_curve(h, constellation=q, esn0_db=[0], min_block_errors=0)
        with pytest.raises(ValueError, match="max_blocks"):
            error_rate_curve(h, constellation=q, esn0_db=[0], max_blocks=1.5)
        with pytest.raises(ValueError, match="device"):
            error_rate_curve(h, constellation=q, esn0_db=[0], device="tpu")

    def test_curve_value(self) -> None:
        """Counts are read-only int64 host copies; the rates follow from them."""
        curve = ErrorRateCurve(
            esn0_db=[1.0, 2.0],
            num_blocks=[10, 20],
            bit_errors=[8, 4],
            block_errors=[5, 2],
            pre_fec_bit_errors=[70, 35],
            detected_failures=[3, 1],
            undetected_errors=[2, 1],
            total_iterations=[100, 120],
            k=4,
            n=7,
            bits_per_symbol=2,
            label="test",
        )
        assert curve.num_blocks.dtype == np.int64
        assert not curve.bit_errors.flags.writeable
        np.testing.assert_allclose(curve.ber, [8 / 40, 4 / 80])
        np.testing.assert_allclose(curve.fer, [0.5, 0.1])
        np.testing.assert_allclose(curve.pre_fec_ber, [70 / 70, 35 / 140])
        np.testing.assert_allclose(curve.mean_iterations, [10.0, 6.0])
        np.testing.assert_allclose(curve.ebn0_db, [1.0, 2.0] - 10 * np.log10(8 / 7))
        with pytest.raises(ValueError, match="shape of esn0_db"):
            ErrorRateCurve(
                [1.0], [1, 2], [0], [0], [0], [0], [0], [1], k=1, n=1, bits_per_symbol=1
            )
        with pytest.raises(ValueError, match="num_blocks must be >= 1"):
            ErrorRateCurve(
                [1.0], [0], [0], [0], [0], [0], [0], [0], k=1, n=1, bits_per_symbol=1
            )
        with pytest.raises(ValueError, match="counts >= 0"):
            ErrorRateCurve(
                [1.0], [1], [-1], [0], [0], [0], [0], [1], k=1, n=1, bits_per_symbol=1
            )
