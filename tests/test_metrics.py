"""Tests for performance metrics module."""

from typing import Any

import numpy as np
import pytest

from commkit import generate, metrics, multirate
from commkit.backend import to_device
from commkit.impairments import apply_awgn
from commkit.mapping import Constellation, compute_llr, map_bits


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


class TestErrorVectorMagnitude:
    """Tests for Error Vector Magnitude (EVM) in data-aided and blind modes."""

    def test_evm_perfect_signal(self, xp: Any) -> None:
        """EVM should be 0% for identical tx/rx symbols."""
        symbols = xp.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j])
        evm_pct = metrics.evm(symbols, symbols)
        assert evm_pct < 1e-10

    def test_evm_with_known_error(self, xp: Any) -> None:
        """EVM with known error magnitude."""
        tx = xp.array([1.0 + 0j, 0.0 + 1j, -1.0 + 0j, 0.0 - 1j])
        rx = tx + 0.1
        evm_pct = metrics.evm(rx, tx)
        assert abs(evm_pct - 10.0) < 1.0

    def test_evm_near_zero_ref(self, xp: Any) -> None:
        """Verify EVM behavior when reference signal is near zero."""
        tx = xp.zeros(10)
        rx = xp.ones(10)
        assert metrics.evm(rx, tx) == float("inf")

    def test_evm_shape_mismatch(self, xp: Any) -> None:
        """Verify error on shape mismatch in evm."""
        with pytest.raises(ValueError, match="shape mismatch"):
            metrics.evm(xp.zeros(10), xp.zeros(11))

    def test_evm_array_handling(self, xp: Any, xpt: Any) -> None:
        """Verify evm multichannel array handling."""
        rx = xp.array([[1.0, 1.0], [1.0, 0.8]])
        tx = xp.array([[1.0, 1.0], [1.1, 1.1]])

        ep = metrics.evm(rx, tx)
        assert isinstance(ep, np.ndarray)
        assert ep.shape == (2,)
        assert ep[0] == 0
        assert ep[1] > 0

        # Low power mask for array
        tx_zero = xp.zeros((2, 2))
        np.testing.assert_array_equal(metrics.evm(rx, tx_zero), float("inf"))

    def test_evm_normalized_scaling(self, xp: Any) -> None:
        """Identical shape with scalar scale factor normalizes to 0% EVM."""
        ref = xp.array([1.0, 1.0])
        rx = xp.array([1.1, 1.1])
        assert metrics.evm(rx, ref) < 1e-5

    def test_evm_blind_perfect_signal(self, xp: Any) -> None:
        """Blind EVM should be near 0% when rx sits exactly on constellation points."""
        const = xp.asarray(Constellation.qam(16).points)
        rx = xp.tile(const, 32)
        pct = metrics.evm(rx, blind=True, constellation=Constellation.qam(16))
        assert pct < 1e-6

    def test_evm_blind_decreases_with_snr(self, xp: Any) -> None:
        """Blind EVM at high SNR should be lower than at low SNR."""
        const = np.asarray(Constellation.qam(16).points)
        rng = np.random.default_rng(7)
        tx = const[rng.integers(0, 16, 2000)]

        rx_high = apply_awgn(xp.asarray(tx), esn0_db=30.0, sps=1)
        rx_low = apply_awgn(xp.asarray(tx), esn0_db=10.0, sps=1)

        c16 = Constellation.qam(16)
        pct_high = metrics.evm(rx_high, blind=True, constellation=c16)
        pct_low = metrics.evm(rx_low, blind=True, constellation=c16)
        assert pct_high < pct_low

    def test_evm_blind_vs_data_aided_converge(self, xp: Any) -> None:
        """At high SNR blind and data-aided EVM should agree closely."""
        rng = np.random.default_rng(42)
        bits = rng.integers(0, 2, 4000).astype("int32")
        tx = map_bits(xp.asarray(bits), constellation=Constellation.qam(16))
        rx = apply_awgn(tx, esn0_db=30.0, sps=1)

        pct_da = metrics.evm(rx, tx)
        pct_bl = metrics.evm(rx, blind=True, constellation=Constellation.qam(16))
        assert abs(pct_da - pct_bl) < 0.5

    def test_evm_blind_shaped_matches_data_aided(self, xp: Any) -> None:
        """At high SNR blind EVM equals data-aided EVM on a shaped
        constellation too: both are relative to its average power."""
        c = Constellation.qam(64).shaped(nu=0.075)
        sig = generate(c, 20000, symbol_rate=1e9, rng=2)
        rx = apply_awgn(xp.asarray(sig.samples), esn0_db=35.0, sps=1, rng=3)
        sig = multirate.decimate_to_symbol_rate(sig.replace(samples=rx))
        pct_da = metrics.evm(sig)
        pct_bl = metrics.evm(sig, blind=True)
        assert pct_bl == pytest.approx(pct_da, rel=0.02)

    def test_evm_blind_multichannel(self, xp: Any, xpt: Any) -> None:
        """Blind EVM returns array of shape (N_ch,) for MIMO input."""
        const = xp.asarray(Constellation.qam(4).points)
        rng = np.random.default_rng(1)
        rx = xp.stack([const[rng.integers(0, 4, 200)] for _ in range(3)])

        pct = metrics.evm(rx, blind=True, constellation=Constellation.qam(4))
        assert pct.shape == (3,)
        np.testing.assert_allclose(pct, 0.0, atol=1e-6)

    def test_evm_blind_missing_constellation_raises(self, xp: Any) -> None:
        """Blind EVM without a constellation raises ValueError."""
        with pytest.raises(ValueError, match="constellation"):
            metrics.evm(xp.zeros(10), blind=True)

    def test_evm_data_aided_missing_reference_raises(self, xp: Any) -> None:
        """Data-aided EVM without a reference raises ValueError."""
        with pytest.raises(ValueError, match="reference"):
            metrics.evm(xp.zeros(10))

    def test_evm_blind_with_reference_raises(self, xp: Any) -> None:
        """blind=True and a reference contradict each other."""
        with pytest.raises(ValueError, match="blind"):
            metrics.evm(xp.ones(10), xp.ones(10), blind=True)

    def test_evm_skip_symbols(self, xp: Any) -> None:
        """num_skip_symbols leaves out leading symbols; skipping all raises."""
        tx = xp.ones(20, dtype=xp.complex64)
        rx = tx.copy()
        rx[:5] = -1  # wrong only during "training"
        assert metrics.evm(rx, tx, num_skip_symbols=5) < 1e-6
        with pytest.raises(ValueError, match="nothing to measure"):
            metrics.evm(rx, tx, num_skip_symbols=20)


class TestSignalToNoiseRatio:
    """Tests for Signal-to-Noise Ratio (SNR) estimation."""

    def test_snr_matches_applied(self, xp: Any) -> None:
        """SNR estimate should approximately match applied AWGN level."""
        bits = np.random.default_rng(42).integers(0, 2, 20000, dtype="int8")
        symbols = Constellation.qam(4).map(bits)
        target_snr_db = 20.0
        noisy = apply_awgn(symbols, esn0_db=target_snr_db, sps=1)

        estimated_snr = metrics.snr(noisy, symbols)
        assert abs(estimated_snr - target_snr_db) < 1.0

    def test_snr_high_snr(self, xp: Any) -> None:
        """Very high SNR should return very high estimate."""
        symbols = xp.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j])
        snr_db = metrics.snr(symbols, symbols)
        assert snr_db > 100

    def test_snr_shape_mismatch(self, xp: Any) -> None:
        """Verify error on shape mismatch in snr."""
        with pytest.raises(ValueError, match="shape mismatch"):
            metrics.snr(xp.zeros(10), xp.zeros(11))

    def test_snr_divide_by_zero(self, xp: Any) -> None:
        """SNR with all zeros reference and received returns inf."""
        zero = xp.zeros(10)
        snr_val = metrics.snr(zero, zero)
        assert snr_val == float("inf")

    def test_snr_scalar_low_power(self, xp: Any) -> None:
        """Verify snr returns -inf when reference power is zero."""
        rx = xp.ones(10)
        tx = xp.zeros(10)
        assert metrics.snr(rx, tx) == float("-inf")

    def test_snr_array_low_power(self, xp: Any, xpt: Any) -> None:
        """Verify snr array handling returns -inf per channel when reference power is zero."""
        rx = xp.ones((2, 10))
        tx = xp.zeros((2, 10))
        res = metrics.snr(rx, tx)
        np.testing.assert_array_equal(res, float("-inf"))

        # Mixed case
        tx_mixed = xp.array([xp.ones(10), xp.zeros(10)])
        res = metrics.snr(rx, tx_mixed)
        assert res[0] == float("inf")
        assert res[1] == float("-inf")


class TestSymbolErrorRate:
    """Tests for Symbol Error Rate (SER) computation."""

    def test_ser_perfect_signal(self, xp: Any) -> None:
        """SER = 0 when rx equals tx exactly."""
        rng = np.random.default_rng(0)
        bits = rng.integers(0, 2, 800).astype("int32")
        tx = map_bits(xp.asarray(bits), constellation=Constellation.qam(16))
        assert metrics.ser(tx, tx, constellation=Constellation.qam(16)) == 0.0

    def test_ser_high_snr_near_zero(self, xp: Any) -> None:
        """SER should be negligible at very high SNR."""
        rng = np.random.default_rng(1)
        bits = rng.integers(0, 2, 2000).astype("int32")
        tx = map_bits(xp.asarray(bits), constellation=Constellation.qam(4))
        rx = apply_awgn(tx, esn0_db=40.0, sps=1)
        assert metrics.ser(rx, tx, constellation=Constellation.qam(4)) < 1e-3

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
        sig = generate(c, 2000, symbol_rate=1e9, rng=1).to(
            "gpu" if xp is not np else "cpu"
        )
        assert metrics.ser(multirate.decimate_to_symbol_rate(sig)) == 0.0

    def test_ser_shape_mismatch_raises(self, xp: Any) -> None:
        """SER raises ValueError on shape mismatch."""
        with pytest.raises(ValueError, match="shape mismatch"):
            metrics.ser(xp.zeros(10), xp.zeros(11), constellation=Constellation.qam(4))


class TestInformationMetrics:
    """Tests for Mutual Information (MI) and Generalized Mutual Information (GMI)."""

    def test_gmi_high_snr_approaches_log2m(self, xp: Any) -> None:
        """At infinite SNR (perfect LLRs), GMI -> log2(M)."""
        k = 4
        M = 16
        N = 200
        rng = np.random.default_rng(42)
        bits = rng.integers(0, 2, N * k).astype("int32")
        c = Constellation.qam(M)
        symbols = map_bits(xp.asarray(bits), constellation=c)

        llrs = compute_llr(symbols, noise_var=1e-6, constellation=c)
        gmi_val = metrics.gmi(llrs, bits, constellation=c)
        assert gmi_val > np.log2(M) - 0.05

    def test_gmi_low_snr_approaches_zero(self, xp: Any) -> None:
        """At very low SNR, LLRs collapse to zero -> GMI -> 0."""
        k = 4
        M = 16
        N = 500
        rng = np.random.default_rng(7)
        bits = rng.integers(0, 2, N * k).astype("int32")
        c = Constellation.qam(M)
        symbols = map_bits(xp.asarray(bits), constellation=c)

        llrs = compute_llr(symbols, noise_var=1e6, constellation=c)
        gmi_val = metrics.gmi(llrs, bits, constellation=c)
        assert gmi_val < 0.2

    def test_gmi_returns_scalar_float(self, xp: Any) -> None:
        """gmi() of 1-D LLRs returns a Python float."""
        bits = np.array([0, 1, 1, 0, 0, 1, 1, 0], dtype="int32")
        c = Constellation.qam(4)
        symbols = map_bits(xp.asarray(bits), constellation=c)
        llrs = compute_llr(symbols, noise_var=0.1, constellation=c)
        gmi_val = metrics.gmi(llrs, bits, constellation=c)
        assert isinstance(gmi_val, float)

    def test_gmi_multichannel(self, xp: Any) -> None:
        """(C, N k) LLRs give one GMI per channel."""
        rng = np.random.default_rng(3)
        bits = rng.integers(0, 2, (2, 200)).astype("int32")
        c = Constellation.qam(4)
        llrs = compute_llr(
            map_bits(xp.asarray(bits), constellation=c), noise_var=0.1, constellation=c
        )
        out = metrics.gmi(llrs, bits, constellation=c)
        assert isinstance(out, np.ndarray)
        assert out.shape == (2,)
        assert out[0] == pytest.approx(metrics.gmi(llrs[0], bits[0], constellation=c))

    def test_gmi_shape_mismatch_raises(self, xp: Any) -> None:
        """gmi() raises ValueError when llrs and the bits differ in shape."""
        llrs = np.array([1.0, -1.0, 2.0, 0.5])
        bits = np.array([0, 1])
        with pytest.raises(ValueError, match="shape mismatch"):
            metrics.gmi(llrs, bits, constellation=Constellation.qam(4))

    def test_gmi_requires_constellation(self, xp: Any) -> None:
        """k comes from the constellation; an LLR array alone does not give it."""
        with pytest.raises(ValueError, match="constellation"):
            metrics.gmi(np.zeros(4), np.zeros(4))

    def test_gmi_noise_var_with_llrs_raises(self, xp: Any) -> None:
        """noise_var is for Signal input; LLRs already contain it."""
        with pytest.raises(ValueError, match="noise_var"):
            metrics.gmi(
                np.zeros(4),
                np.zeros(4),
                constellation=Constellation.qam(4),
                noise_var=0.1,
            )

    def test_gmi_bounded_by_k(self, xp: Any) -> None:
        """GMI in [0, k]."""
        k = 2
        N = 100
        rng = np.random.default_rng(55)
        bits = rng.integers(0, 2, N * k).astype("int32")
        c = Constellation.qam(4)
        symbols = map_bits(xp.asarray(bits), constellation=c)
        llrs = compute_llr(symbols, noise_var=0.1, constellation=c)
        gmi_val = metrics.gmi(llrs, bits, constellation=c)
        assert 0.0 <= gmi_val <= np.log2(4)

    def test_mi_high_snr_approaches_log2m(self, xp: Any) -> None:
        """At high SNR, MI -> log2(M)."""
        M = 16
        c = Constellation.qam(M)
        rng = np.random.default_rng(42)
        symbols = c.points[rng.integers(0, M, 500)]

        mi_val = metrics.mi(xp.asarray(symbols), noise_var=1e-8, constellation=c)
        assert mi_val > np.log2(M) - 0.1

    def test_mi_never_exceeds_log2m(self, xp: Any) -> None:
        """MI <= log2(M) always (capacity bound)."""
        M = 4
        c = Constellation.qam(M)
        rng = np.random.default_rng(7)
        symbols = c.points[rng.integers(0, M, 200)]

        for noise_var in [1e-4, 0.1, 1.0, 10.0]:
            mi_val = metrics.mi(
                xp.asarray(symbols), noise_var=noise_var, constellation=c
            )
            assert mi_val <= np.log2(M) + 1e-6

    def test_mi_returns_scalar_float(self, xp: Any) -> None:
        """mi() of (N,) symbols returns a Python float."""
        c = Constellation.qam(4)
        mi_val = metrics.mi(xp.asarray(c.points[:10]), noise_var=0.1, constellation=c)
        assert isinstance(mi_val, float)

    def test_mi_multichannel(self, xp: Any) -> None:
        """(C, N) symbols give one MI per channel (1.x pooled the channels)."""
        c = Constellation.qam(16)
        rng = np.random.default_rng(4)
        symbols = c.points[rng.integers(0, 16, (2, 300))]
        symbols[1] += 0.3 * rng.standard_normal(300)
        out = metrics.mi(xp.asarray(symbols), noise_var=0.05, constellation=c)
        assert out.shape == (2,)
        assert out[0] == pytest.approx(
            metrics.mi(xp.asarray(symbols[0]), noise_var=0.05, constellation=c)
        )
        assert out[1] < out[0]

    def test_mi_decreases_with_noise(self, xp: Any) -> None:
        """MI should decrease as noise increases."""
        M = 16
        c = Constellation.qam(M)
        rng = np.random.default_rng(99)
        symbols = xp.asarray(c.points[rng.integers(0, M, 500)])

        mi_low_noise = metrics.mi(symbols, noise_var=0.01, constellation=c)
        mi_high_noise = metrics.mi(symbols, noise_var=1.0, constellation=c)
        assert mi_low_noise > mi_high_noise


class TestSignalMetricsIntegration:
    """Metrics of a 1-SPS Signal against its reference."""

    @staticmethod
    def _sig(c: Constellation, n: int, xp: Any, **kw: Any):
        sig = generate(c, n, symbol_rate=1e6, **kw)
        return multirate.decimate_to_symbol_rate(
            sig.to("gpu" if xp is not np else "cpu")
        )

    def test_signal_evm(self, xp: Any) -> None:
        """evm(sig) uses reference.symbols."""
        assert metrics.evm(self._sig(Constellation.qam(4), 100, xp)) < 1e-4

    def test_signal_ber(self, xp: Any) -> None:
        """ber(sig) hard-decides the samples against reference.bits."""
        assert metrics.ber(self._sig(Constellation.qam(4), 100, xp)) == 0.0

    def test_signal_evm_blind(self, xp: Any) -> None:
        """evm(sig, blind=True) near zero for a clean signal."""
        sig = self._sig(Constellation.qam(16), 2000, xp)
        assert metrics.evm(sig, blind=True) < 3.0

    def test_signal_ser(self, xp: Any) -> None:
        """ser(sig) is 0 for a clean signal."""
        assert metrics.ser(self._sig(Constellation.qam(16), 200, xp)) == 0.0

    def test_signal_gmi_and_mi(self, xp: Any) -> None:
        """gmi(sig)/mi(sig) compute from the samples with the Signal's
        constellation; equal to the array path."""
        c = Constellation.qam(16)
        sig = self._sig(c, 500, xp)
        rx = sig.replace(samples=apply_awgn(sig.samples, esn0_db=10.0, sps=1, rng=1))
        nv = 0.1
        llrs = compute_llr(rx.samples, noise_var=nv, constellation=c)
        assert metrics.gmi(rx, noise_var=nv) == metrics.gmi(
            llrs, rx.reference.bits, constellation=c
        )
        assert metrics.mi(rx, noise_var=nv) == metrics.mi(
            rx.samples, noise_var=nv, constellation=c
        )

    def test_signal_multichannel_returns_host_array(self, xp: Any) -> None:
        """(C, N) Signals give host (C,) arrays."""
        sig = self._sig(Constellation.qam(4), 100, xp, num_channels=2)
        for out in (metrics.evm(sig), metrics.ser(sig), metrics.ber(sig)):
            assert isinstance(out, np.ndarray)
            assert out.shape == (2,)

    def test_explicit_reference_wins(self, xp: Any) -> None:
        """An explicit reference replaces the Signal's."""
        sig = self._sig(Constellation.qam(4), 100, xp)
        assert metrics.ser(sig, -sig.reference.symbols) == 1.0

    def test_oversampled_signal_raises(self, xp: Any) -> None:
        """Metrics need one sample per symbol; they never decimate."""
        sig = generate(Constellation.qam(4), 100, symbol_rate=1e6, sps=2)
        with pytest.raises(ValueError, match="one sample per symbol"):
            metrics.evm(sig)

    def test_frame_signal_raises(self, xp: Any) -> None:
        """A frame Signal mixes segments: extract the payload first."""
        from commkit.core import SingleCarrierFrame

        sig = SingleCarrierFrame(payload_len=100).to_signal(sps=1, symbol_rate=1e6)
        with pytest.raises(ValueError, match="extract_payload"):
            metrics.ser(sig)

    def test_length_mismatch_raises(self, xp: Any) -> None:
        """The reference is never clamped to the received length."""
        sig = self._sig(Constellation.qam(4), 100, xp)
        with pytest.raises(ValueError, match="shape mismatch"):
            metrics.snr(sig, sig.reference.symbols[:-1])

    def test_missing_reference_raises(self, xp: Any) -> None:
        """A Signal without a reference needs reference= (or blind=True)."""
        sig = self._sig(Constellation.qam(4), 100, xp)
        sig = sig.replace(reference=None)
        with pytest.raises(ValueError, match="reference"):
            metrics.evm(sig)


class TestGMIShaped:
    """GMI is the bit-metric decoding rate H(X) - sum_b H(B_b | Y) (3.8e)."""

    @staticmethod
    def _received(c: Constellation, esn0_db: float, n: int = 20000, seed: int = 0):
        rng = np.random.default_rng(seed)
        nv = 10 ** (-esn0_db / 10)
        idx = rng.choice(c.order, n, p=c.pmf)
        noise = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        rx = c.points[idx] + np.sqrt(nv / 2) * noise
        return rx.astype(np.complex64), c.bit_labels[idx].reshape(-1), nv

    @staticmethod
    def _bmd_rate(c: Constellation, rx, bits, nv) -> float:
        """Independent oracle: exact bitwise posteriors from the points, the
        labels and the prior, in float64."""
        log_joint = np.log(c.pmf) - np.abs(rx[:, None] - c.points) ** 2 / nv
        log_joint -= log_joint.max(axis=1, keepdims=True)
        joint = np.exp(log_joint)
        joint /= joint.sum(axis=1, keepdims=True)  # P(s | y)
        sent = bits.reshape(rx.size, -1)
        h_cond = 0.0
        for b in range(c.bits_per_symbol):
            p_one = joint[:, c.bit_labels[:, b] == 1].sum(axis=1)
            p_sent = np.where(sent[:, b] == 1, p_one, 1 - p_one)
            h_cond += -np.mean(np.log2(np.maximum(p_sent, 1e-300)))
        return c.entropy - h_cond

    @pytest.mark.parametrize(
        ("c", "esn0_db"),
        [
            (Constellation.qam(16).shaped(entropy=3.3), 20.0),
            (Constellation.qam(64).shaped(nu=0.075), 0.0),
            (Constellation.qam(64).shaped(nu=0.075), 10.0),
            (Constellation.qam(256).shaped(entropy=7.0), 15.0),
        ],
        ids=["16qam-20dB", "64qam-0dB", "64qam-10dB", "256qam-15dB"],
    )
    def test_shaped_gmi_matches_bmd_oracle_and_bounds(self, c, esn0_db, xp) -> None:
        """GMI equals the independently computed BMD rate and obeys
        GMI <= MI <= H(X)."""
        rx, bits, nv = self._received(c, esn0_db)
        llrs = compute_llr(
            xp.asarray(rx), noise_var=nv, constellation=c, method="exact"
        )
        g = metrics.gmi(llrs, xp.asarray(bits), constellation=c)
        m = metrics.mi(xp.asarray(rx), noise_var=nv, constellation=c)
        assert g == pytest.approx(self._bmd_rate(c, rx, bits, nv), abs=2e-4)
        assert g <= m + 1e-3
        assert m <= c.entropy + 1e-9

    def test_uniform_gmi_unchanged(self, xp) -> None:
        """Uniform: H(X) = k, so the rate is the usual k - sum_b E[...]."""
        c = Constellation.qam(16)
        rng = np.random.default_rng(1)
        bits = rng.integers(0, 2, 4000).astype(np.int8)
        rx = c.map(bits) + 0.2 * rng.standard_normal(1000)
        llrs = compute_llr(xp.asarray(rx), noise_var=0.08, constellation=c)
        x = -np.asarray(to_device(llrs, "cpu"), np.float64) * (1 - 2.0 * bits)
        sp = (np.log1p(np.exp(-np.abs(x))) + np.maximum(0, x)) / np.log(2)
        expected = 4 * (1 - np.mean(sp))
        assert metrics.gmi(llrs, xp.asarray(bits), constellation=c) == expected
