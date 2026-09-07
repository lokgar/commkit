"""Tests for performance metrics module."""

from typing import Any

import numpy as np
import pytest

from commkit import generate_qam, mapping, metrics, multirate
from commkit.helpers import generate_symbols
from commkit.impairments import apply_awgn
from commkit.mapping import compute_llr, gray_constellation, map_bits


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

    def test_ber_empty(self, xp: Any) -> None:
        """Verify BER with empty arrays."""
        assert metrics.ber(xp.array([]), xp.array([])) == 0.0

    def test_ber_length_mismatch(self, xp: Any) -> None:
        """Verify error on bit length mismatch."""
        with pytest.raises(ValueError, match="Shape mismatch"):
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
        evm_pct, evm_db = metrics.evm(symbols, symbols)
        assert evm_pct < 1e-10
        assert evm_db < -200

    def test_evm_with_known_error(self, xp: Any) -> None:
        """EVM with known error magnitude."""
        tx = xp.array([1.0 + 0j, 0.0 + 1j, -1.0 + 0j, 0.0 - 1j])
        rx = tx + 0.1
        evm_pct, _ = metrics.evm(rx, tx)
        assert abs(evm_pct - 10.0) < 1.0

    def test_evm_near_zero_ref(self, xp: Any) -> None:
        """Verify EVM behavior when reference signal is near zero."""
        tx = xp.zeros(10)
        rx = xp.ones(10)
        pct, db = metrics.evm(rx, tx)
        assert pct == float("inf")
        assert db == float("inf")

    def test_evm_shape_mismatch(self, xp: Any) -> None:
        """Verify error on shape mismatch in evm."""
        with pytest.raises(ValueError, match="Shape mismatch"):
            metrics.evm(xp.zeros(10), xp.zeros(11))

    def test_evm_array_handling(self, xp: Any, xpt: Any) -> None:
        """Verify evm multichannel array handling."""
        rx = xp.array([[1.0, 1.0], [1.0, 0.8]])
        tx = xp.array([[1.0, 1.0], [1.1, 1.1]])

        ep, edb = metrics.evm(rx, tx)
        assert ep.shape == (2,)
        assert ep[0] == 0
        assert ep[1] > 0

        # Low power mask for array
        tx_zero = xp.zeros((2, 2))
        ep_z, edb_z = metrics.evm(rx, tx_zero)
        xpt.assert_array_equal(ep_z, float("inf"))
        xpt.assert_array_equal(edb_z, float("inf"))

    def test_evm_normalized_scaling(self, xp: Any) -> None:
        """Identical shape with scalar scale factor normalizes to 0% EVM."""
        ref = xp.array([1.0, 1.0])
        rx = xp.array([1.1, 1.1])
        evm_pct, _ = metrics.evm(rx, ref)
        assert evm_pct < 1e-5

    def test_evm_blind_perfect_signal(self, xp: Any) -> None:
        """Blind EVM should be near 0% when rx sits exactly on constellation points."""
        const = xp.asarray(gray_constellation("qam", 16))
        rx = xp.tile(const, 32)
        pct, db = metrics.evm(rx, mode="blind", modulation="qam", order=16)
        assert pct < 1e-6

    def test_evm_blind_decreases_with_snr(self, xp: Any) -> None:
        """Blind EVM at high SNR should be lower than at low SNR."""
        const = np.asarray(gray_constellation("qam", 16))
        rng = np.random.default_rng(7)
        tx = const[rng.integers(0, 16, 2000)]

        rx_high = apply_awgn(xp.asarray(tx), esn0_db=30.0, sps=1)
        rx_low = apply_awgn(xp.asarray(tx), esn0_db=10.0, sps=1)

        pct_high, _ = metrics.evm(rx_high, mode="blind", modulation="qam", order=16)
        pct_low, _ = metrics.evm(rx_low, mode="blind", modulation="qam", order=16)
        assert pct_high < pct_low

    def test_evm_blind_vs_data_aided_converge(self, xp: Any) -> None:
        """At high SNR blind and data-aided EVM should agree closely."""
        rng = np.random.default_rng(42)
        bits = rng.integers(0, 2, 4000).astype("int32")
        tx = map_bits(xp.asarray(bits), "qam", 16)
        rx = apply_awgn(tx, esn0_db=30.0, sps=1)

        pct_da, _ = metrics.evm(rx, tx)
        pct_bl, _ = metrics.evm(rx, mode="blind", modulation="qam", order=16)
        assert abs(pct_da - pct_bl) < 0.5

    def test_evm_blind_multichannel(self, xp: Any, xpt: Any) -> None:
        """Blind EVM returns array of shape (N_ch,) for MIMO input."""
        const = xp.asarray(gray_constellation("qam", 4))
        rng = np.random.default_rng(1)
        rx = xp.stack([const[rng.integers(0, 4, 200)] for _ in range(3)])

        pct, db = metrics.evm(rx, mode="blind", modulation="qam", order=4)
        assert pct.shape == (3,)
        xpt.assert_allclose(pct, 0.0, atol=1e-6)

    def test_evm_blind_missing_args_raises(self, xp: Any) -> None:
        """Blind mode without modulation/order raises ValueError."""
        with pytest.raises(ValueError, match="modulation and order"):
            metrics.evm(xp.zeros(10), mode="blind")

    def test_evm_data_aided_missing_tx_raises(self, xp: Any) -> None:
        """data_aided mode without tx_symbols raises ValueError."""
        with pytest.raises(ValueError, match="tx_symbols"):
            metrics.evm(xp.zeros(10))

    def test_evm_unknown_mode_raises(self, xp: Any) -> None:
        """Unknown mode string raises ValueError."""
        with pytest.raises(ValueError, match="Unknown mode"):
            metrics.evm(xp.zeros(10), xp.zeros(10), mode="magic")


class TestSignalToNoiseRatio:
    """Tests for Signal-to-Noise Ratio (SNR) estimation."""

    def test_snr_matches_applied(self, xp: Any) -> None:
        """SNR estimate should approximately match applied AWGN level."""
        symbols = generate_symbols(10000, "qam", 4, seed=42)
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
        with pytest.raises(ValueError, match="Shape mismatch"):
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
        xpt.assert_array_equal(res, float("-inf"))

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
        tx = map_bits(xp.asarray(bits), "qam", 16)
        assert metrics.ser(tx, tx, "qam", 16) == 0.0

    def test_ser_high_snr_near_zero(self, xp: Any) -> None:
        """SER should be negligible at very high SNR."""
        rng = np.random.default_rng(1)
        bits = rng.integers(0, 2, 2000).astype("int32")
        tx = map_bits(xp.asarray(bits), "qam", 4)
        rx = apply_awgn(tx, esn0_db=40.0, sps=1)
        assert metrics.ser(rx, tx, "qam", 4) < 1e-3

    def test_ser_multichannel(self, xp: Any) -> None:
        """SER returns array (N_ch,) for 2D input."""
        rng = np.random.default_rng(2)
        bits = rng.integers(0, 2, 400).astype("int32")
        tx_row = map_bits(xp.asarray(bits), "qam", 4)
        tx = xp.stack([tx_row, tx_row])

        result = metrics.ser(tx, tx, "qam", 4)
        assert result.shape == (2,)
        assert float(result[0]) == 0.0
        assert float(result[1]) == 0.0

    def test_ser_shape_mismatch_raises(self, xp: Any) -> None:
        """SER raises ValueError on shape mismatch."""
        with pytest.raises(ValueError, match="Shape mismatch"):
            metrics.ser(xp.zeros(10), xp.zeros(11), "qam", 4)


class TestInformationMetrics:
    """Tests for Mutual Information (MI) and Generalized Mutual Information (GMI)."""

    def test_gmi_high_snr_approaches_log2m(self, xp: Any) -> None:
        """At infinite SNR (perfect LLRs), GMI -> log2(M)."""
        k = 4
        M = 16
        N = 200
        rng = np.random.default_rng(42)
        bits = rng.integers(0, 2, N * k).astype("int32")
        symbols = map_bits(xp.asarray(bits), "qam", M)

        llrs = compute_llr(symbols, "qam", M, noise_var=1e-6, output="numpy").reshape(
            N, k
        )
        gmi_val = metrics.gmi(llrs, bits.reshape(N, k))
        assert gmi_val > np.log2(M) - 0.05

    def test_gmi_low_snr_approaches_zero(self, xp: Any) -> None:
        """At very low SNR, LLRs collapse to zero -> GMI -> 0."""
        k = 4
        M = 16
        N = 500
        rng = np.random.default_rng(7)
        bits = rng.integers(0, 2, N * k).astype("int32")
        symbols = map_bits(xp.asarray(bits), "qam", M)

        llrs = compute_llr(symbols, "qam", M, noise_var=1e6, output="numpy").reshape(
            N, k
        )
        gmi_val = metrics.gmi(llrs, bits.reshape(N, k))
        assert gmi_val < 0.2

    def test_gmi_flat_input_returns_per_bit(self, xp: Any) -> None:
        """Flat 1D input: gmi() treats k=1 and returns per-bit GMI in [0, 1]."""
        bits = np.array([0, 1, 1, 0, 0, 1, 1, 0], dtype="int32")
        symbols = map_bits(xp.asarray(bits), "qam", 4)
        llrs = compute_llr(symbols, "qam", 4, noise_var=0.1, output="numpy")
        gmi_val = metrics.gmi(llrs, bits)
        assert 0.0 <= gmi_val <= 1.0

    def test_gmi_returns_scalar_float(self, xp: Any) -> None:
        """gmi() must return a Python float."""
        k = 2
        N = 4
        bits = np.array([0, 1, 1, 0, 0, 1, 1, 0], dtype="int32")
        symbols = map_bits(xp.asarray(bits), "qam", 4)
        llrs = compute_llr(symbols, "qam", 4, noise_var=0.1, output="numpy").reshape(
            N, k
        )
        gmi_val = metrics.gmi(llrs, bits.reshape(N, k))
        assert isinstance(gmi_val, float)

    def test_gmi_shape_mismatch_raises(self, xp: Any) -> None:
        """gmi() should raise ValueError when llrs and tx_bits have different sizes."""
        llrs = np.array([1.0, -1.0, 2.0])
        bits = np.array([0, 1])
        with pytest.raises(ValueError, match="same number of elements"):
            metrics.gmi(llrs, bits)

    def test_gmi_2d_bounded_by_log2m(self, xp: Any) -> None:
        """For (N, k) input, GMI in [0, k]."""
        k = 2
        N = 100
        rng = np.random.default_rng(55)
        bits = rng.integers(0, 2, N * k).astype("int32")
        symbols = map_bits(xp.asarray(bits), "qam", 4)
        llrs = compute_llr(symbols, "qam", 4, noise_var=0.1, output="numpy").reshape(
            N, k
        )
        bits_2d = bits.reshape(N, k)
        gmi_val = metrics.gmi(llrs, bits_2d)
        assert 0.0 <= gmi_val <= np.log2(4)

    def test_mi_high_snr_approaches_log2m(self, xp: Any) -> None:
        """At high SNR, MI -> log2(M)."""
        M = 16
        const = gray_constellation("qam", M)
        rng = np.random.default_rng(42)
        symbols = const[rng.integers(0, M, 500)]

        mi_val = metrics.mi(xp.asarray(symbols), "qam", M, noise_var=1e-8)
        assert mi_val > np.log2(M) - 0.1

    def test_mi_never_exceeds_log2m(self, xp: Any) -> None:
        """MI <= log2(M) always (capacity bound)."""
        M = 4
        const = gray_constellation("qam", M)
        rng = np.random.default_rng(7)
        symbols = const[rng.integers(0, M, 200)]

        for noise_var in [1e-4, 0.1, 1.0, 10.0]:
            mi_val = metrics.mi(xp.asarray(symbols), "qam", M, noise_var=noise_var)
            assert mi_val <= np.log2(M) + 1e-6

    def test_mi_returns_scalar_float(self, xp: Any) -> None:
        """mi() must return a Python float."""
        M = 4
        const = gray_constellation("qam", M)
        symbols = const[:10]
        mi_val = metrics.mi(xp.asarray(symbols), "qam", M, noise_var=0.1)
        assert isinstance(mi_val, float)

    def test_mi_decreases_with_noise(self, xp: Any) -> None:
        """MI should decrease as noise increases."""
        M = 16
        const = gray_constellation("qam", M)
        rng = np.random.default_rng(99)
        symbols = const[rng.integers(0, M, 500)]

        mi_low_noise = metrics.mi(xp.asarray(symbols), "qam", M, noise_var=0.01)
        mi_high_noise = metrics.mi(xp.asarray(symbols), "qam", M, noise_var=1.0)
        assert mi_low_noise > mi_high_noise


class TestSignalMetricsIntegration:
    """Tests for Signal container integration with metrics."""

    def test_signal_evm_method(self, xp: Any) -> None:
        """Test Signal.evm() method using source_symbols as reference."""
        sig = generate_qam(
            order=4, num_symbols=100, sps=1, symbol_rate=1e6, pulse_shape="none"
        )
        sig = multirate.resolve_symbols(sig)
        evm_pct, evm_db = metrics.evm(sig)
        assert evm_pct < 1e-4

    def test_signal_ber_method(self, xp: Any) -> None:
        """Test Signal.ber() method using source_bits as reference."""
        sig = generate_qam(
            order=4, num_symbols=100, sps=1, symbol_rate=1e6, pulse_shape="none"
        )
        sig = multirate.resolve_symbols(sig)
        sig = mapping.demap_symbols_hard(sig)
        ber_val = metrics.ber(sig)
        assert ber_val == 0.0

    def test_signal_demap_hard(self, xp: Any, xpt: Any) -> None:
        """Test Signal.demap_symbols_hard() hard decision matches source_bits."""
        sig = generate_qam(
            order=4, num_symbols=50, sps=1, symbol_rate=1e6, pulse_shape="none"
        )
        sig = multirate.resolve_symbols(sig)
        sig = mapping.demap_symbols_hard(sig)
        xpt.assert_array_equal(sig.resolved_bits.flatten(), sig.source_bits.flatten())

    def test_signal_evm_blind(self, xp: Any) -> None:
        """Signal.evm(mode='blind') returns near-zero EVM for a clean signal."""
        sig = generate_qam(
            order=16, num_symbols=2000, sps=1, symbol_rate=1e6, pulse_shape="none"
        )
        sig = multirate.resolve_symbols(sig)
        pct, db = metrics.evm(sig, mode="blind")
        assert pct < 3.0

    def test_signal_ser_method(self, xp: Any) -> None:
        """Signal.ser() returns 0 for a clean signal."""
        sig = generate_qam(
            order=16, num_symbols=200, sps=1, symbol_rate=1e6, pulse_shape="none"
        )
        sig = multirate.resolve_symbols(sig)
        assert metrics.ser(sig) == 0.0
