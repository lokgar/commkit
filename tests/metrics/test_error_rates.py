"""Tests for the metrics' error rates (metrics/error_rates.py): BER, FER and SER."""

from typing import Any

import numpy as np
import pytest

from commkit import generate, metrics, multirate
from commkit.coding import LinearBlockCode, decode
from commkit.impairments import apply_awgn
from commkit.mapping import Constellation, map_bits
from tests.common import scoring, theory
from tests.common.coding import coded_awgn


class TestBitErrorRate:
    """Tests for Bit Error Rate (BER) computation."""

    def test_ber_no_errors(self, xp: Any) -> None:
        """BER = 0 for identical bit sequences."""
        bits = xp.array([0, 1, 0, 1, 1, 0, 0, 1])
        ber_val = metrics.ber(bits, bits)
        assert ber_val == 0.0

    def test_ber_known_errors(self, xp: Any) -> None:
        """BER calculation with known error count."""
        bits_tx = xp.array([0, 1, 0, 1, 1, 0, 0, 1])
        bits_rx = xp.array([1, 1, 0, 1, 0, 0, 0, 1])  # 2 errors (positions 0 and 4)
        ber_val = metrics.ber(bits_rx, bits_tx)
        expected_ber = 2 / 8  # 0.25
        assert abs(ber_val - expected_ber) < 1e-10

    def test_ber_all_errors(self, xp: Any) -> None:
        """BER = 1 when all bits are flipped."""
        bits_tx = xp.array([0, 0, 0, 0])
        bits_rx = xp.array([1, 1, 1, 1])
        ber_val = metrics.ber(bits_rx, bits_tx)
        assert ber_val == 1.0

    def test_ber_empty_raises(self, xp: Any) -> None:
        """Nothing to measure raises; it is never reported as zero errors."""
        with pytest.raises(ValueError, match="nothing to measure"):
            metrics.ber(xp.array([]), xp.array([]))

    def test_ber_skip_symbols(self, xp: Any) -> None:
        """num_skip_symbols drops k bits per symbol (k from the constellation)."""
        tx = xp.array([0, 1, 0, 1, 1, 0, 0, 1])
        rx = xp.array([1, 1, 0, 1, 1, 0, 0, 0])  # errors in symbols 0 and 3
        c = Constellation.qam(4)
        assert metrics.ber(rx, tx, constellation=c, num_skip_symbols=1) == 1 / 6
        with pytest.raises(ValueError, match="constellation"):
            metrics.ber(rx, tx, num_skip_symbols=1)

    def test_ber_length_mismatch(self, xp: Any) -> None:
        """Verify error on bit length mismatch."""
        with pytest.raises(ValueError, match="shape mismatch"):
            metrics.ber(xp.array([1, 0]), xp.array([1, 0, 1]))

    def test_ber_multichannel(self, xp: Any) -> None:
        """Verify multi-channel BER returns per-channel error rates."""
        # 2 channels, each with 10 bits
        tx = xp.array([[0, 1, 0, 1, 1, 0, 0, 1, 0, 1], [1, 0, 1, 0, 0, 1, 1, 0, 1, 0]])
        # Channel 0: 1 error at position 0, Channel 1: 2 errors at positions 0,1
        rx = xp.array([[1, 1, 0, 1, 1, 0, 0, 1, 0, 1], [0, 1, 1, 0, 0, 1, 1, 0, 1, 0]])

        ber_values = metrics.ber(rx, tx)
        assert ber_values.shape == (2,)
        assert float(ber_values[0]) == pytest.approx(1 / 10)
        assert float(ber_values[1]) == pytest.approx(2 / 10)


class TestSymbolErrorRate:
    """Tests for Symbol Error Rate (SER) computation."""

    def test_ser_perfect_signal(self, xp: Any) -> None:
        """SER = 0 when rx equals tx exactly."""
        rng = np.random.default_rng(0)
        bits = rng.integers(0, 2, 800).astype("int32")
        tx = map_bits(xp.asarray(bits), constellation=Constellation.qam(16))
        assert metrics.ser(tx, tx, constellation=Constellation.qam(16)) == 0.0

    @pytest.mark.parametrize(
        ("order", "esn0_db"), [(4, 6.0), (4, 9.0), (16, 12.0), (64, 18.0)]
    )
    def test_ser_and_ber_follow_awgn_theory(
        self, backend_device: str, order: int, esn0_db: float
    ) -> None:
        """generate -> apply_awgn -> ser/ber of square QAM at Es/N0 match the
        closed forms: SER 1 - (1 - 2(1 - 1/√M) Q(sqrt(3 Es/N0/(M-1))))², and
        for Gray QPSK BER Q(sqrt(Es/N0)), within 4 binomial standard
        deviations over 2e5 symbols.  This checks the symbol energy, the
        noise scale, the demapper and the metric together."""
        n = 200_000
        sig = generate(
            Constellation.qam(order),
            num_symbols=n,
            symbol_rate=1e9,
            rng=1,
            device=backend_device,
        )
        rx = apply_awgn(sig, esn0_db=esn0_db, rng=2)
        esn0 = 10 ** (esn0_db / 10)
        p_s = theory.qam_ser(order, esn0)
        assert metrics.ser(rx) == pytest.approx(
            p_s, abs=scoring.binomial_margin(p_s, n)
        )
        if order == 4:
            p_b = theory.qpsk_ber(esn0)
            assert metrics.ber(rx) == pytest.approx(
                p_b, abs=scoring.binomial_margin(p_b, 2 * n)
            )

    @pytest.mark.parametrize(("order", "esn0_db"), [(4, 3.0), (16, 10.0)])
    def test_mi_follows_awgn_theory(self, xp: Any, order: int, esn0_db: float) -> None:
        """MI at mid SNR equals the Gauss-Hermite integral of the AWGN mutual
        information (QPSK 3 dB: 1.70 bit; 16-QAM 10 dB: 2.97 bit) within
        0.01 bit (2e5 symbols)."""
        c = Constellation.qam(order)
        sig = apply_awgn(
            generate(c, num_symbols=200_000, symbol_rate=1e9, rng=4),
            esn0_db=esn0_db,
            rng=5,
        )
        nv = 10 ** (-esn0_db / 10)
        mi = metrics.mi(xp.asarray(sig.samples), noise_var=nv, constellation=c)
        assert mi == pytest.approx(theory.awgn_mi(c.points, nv), abs=0.01)

    def test_ser_multichannel(self, xp: Any) -> None:
        """SER returns array (N_ch,) for 2D input."""
        rng = np.random.default_rng(2)
        bits = rng.integers(0, 2, 400).astype("int32")
        tx_row = map_bits(xp.asarray(bits), constellation=Constellation.qam(4))
        tx = xp.stack([tx_row, tx_row])

        result = metrics.ser(tx, tx, constellation=Constellation.qam(4))
        assert result.shape == (2,)
        assert float(result[0]) == 0.0
        assert float(result[1]) == 0.0

    def test_ser_shaped_clean_signal_is_zero(self, xp: Any) -> None:
        """A noiseless shaped signal has no symbol errors: received and
        reference symbols are both decided at unit power."""
        c = Constellation.qam(64).shaped(nu=0.075)  # E_PS ~ 0.31 on the 1.x grid
        sig = generate(c, num_symbols=2000, symbol_rate=1e9, rng=1).to(
            "gpu" if xp is not np else "cpu"
        )
        assert metrics.ser(multirate.decimate_to_symbol_rate(sig)) == 0.0

    def test_ser_shape_mismatch_raises(self, xp: Any) -> None:
        """SER raises ValueError on shape mismatch."""
        with pytest.raises(ValueError, match="shape mismatch"):
            metrics.ser(xp.zeros(10), xp.zeros(11), constellation=Constellation.qam(4))


class TestFrameErrorRate:
    """fer: the fraction of blocks with at least one wrong bit."""

    def test_known_counts(self, xp: Any) -> None:
        tx = xp.zeros((2, 12), dtype=xp.int8)
        rx = tx.copy()
        rx[0, 0] = 1  # block 0 of channel 0
        rx[0, 1] = 1  # same block: still one block error
        rx[1, 11] = 1  # last block of channel 1
        rx[1, 5] = 1  # block 1 of channel 1
        out = metrics.fer(rx, tx, block_length=4)
        np.testing.assert_allclose(out, [1 / 3, 2 / 3])
        assert metrics.fer(rx[0], tx[0], block_length=4) == pytest.approx(1 / 3)
        assert metrics.fer(tx[0], tx[0], block_length=12) == 0.0

    def test_errors(self, xp: Any) -> None:
        bits = xp.zeros(8, dtype=xp.int8)
        with pytest.raises(ValueError, match="multiple of block_length"):
            metrics.fer(bits, bits, block_length=3)
        with pytest.raises(ValueError, match="nothing to measure"):
            metrics.fer(bits[:0], bits[:0], block_length=4)
        with pytest.raises(ValueError, match="shape mismatch"):
            metrics.fer(bits, bits[:4], block_length=4)
        with pytest.raises(ValueError, match="block_length"):
            metrics.fer(bits, bits, block_length=0)
        with pytest.raises(ValueError, match="block_length"):
            metrics.fer(bits, bits, block_length=2.0)


class TestPostFEC:
    """The coded chain end to end: generate -> AWGN -> LLRs -> decode -> metrics."""

    def test_hamming_chain(self, backend_device: str) -> None:
        code = LinearBlockCode.hamming(3)
        rx, llrs = coded_awgn(
            code,
            Constellation.qam(16),
            num_symbols=7000,
            esn0_db=13.0,
            backend_device=backend_device,
            num_channels=2,
        )
        ref = rx.reference
        res = decode(llrs, code=code)
        pre = metrics.ber(rx)
        post = metrics.ber(res.bits, ref.info_bits)
        fer = metrics.fer(res.bits, ref.info_bits, block_length=ref.code_block[1])
        assert np.all(post < pre / 2)
        assert np.all(fer > post) and np.all(fer < 1)
        # A DecodeResult with the Signal or its Reference: info_bits and k
        # come from the reference.
        np.testing.assert_array_equal(metrics.ber(res, rx), post)
        np.testing.assert_array_equal(metrics.ber(res, ref), post)
        np.testing.assert_array_equal(metrics.fer(res, rx), fer)
        np.testing.assert_array_equal(metrics.fer(res, ref, block_length=4), fer)
        with pytest.raises(ValueError, match="disagrees"):
            metrics.fer(res, rx, block_length=7)
        with pytest.raises(ValueError, match="give block_length"):
            metrics.fer(res.bits, ref.info_bits)
        with pytest.raises(ValueError, match="needs the transmitted Signal"):
            metrics.ber(res)
        with pytest.raises(ValueError, match="num_skip_symbols"):
            metrics.ber(res, rx, num_skip_symbols=2)

    def test_estimate_noise_variance(self, xp: Any) -> None:
        """σ² on the constellation's scale, 10^(-SNR/10), near 10^(-Es/N0/10)."""
        sig = apply_awgn(
            generate(Constellation.qam(16), num_symbols=20_000, symbol_rate=1e9, rng=3),
            esn0_db=12.0,
            rng=4,
        ).to("gpu" if xp is not np else "cpu")
        nv = metrics.estimate_noise_variance(sig)
        assert isinstance(nv, float)
        assert nv == pytest.approx(10 ** (-metrics.snr(sig) / 10), rel=1e-9)
        assert nv == pytest.approx(10**-1.2, rel=0.05)

    def test_uncoded_post_fec_equals_pre_fec(self, xp: Any) -> None:
        """info_bits is bits: the post-FEC call measures the same BER."""
        sig = apply_awgn(
            generate(Constellation.qam(16), num_symbols=2000, symbol_rate=1e9, rng=3),
            esn0_db=10.0,
            rng=4,
        ).to("gpu" if xp is not np else "cpu")
        hard = sig.constellation.demap(sig.samples)
        assert metrics.ber(sig) == metrics.ber(hard, sig.reference.info_bits)
