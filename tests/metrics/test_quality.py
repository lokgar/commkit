"""Tests for the metrics' quality (metrics/quality.py): EVM, SNR and the noise variance."""

from typing import Any

import numpy as np
import pytest

from commkit import generate, metrics, multirate
from commkit.impairments import apply_awgn
from commkit.mapping import Constellation, map_bits


class TestErrorVectorMagnitude:
    """Tests for Error Vector Magnitude (EVM) in data-aided and blind modes."""

    def test_evm_perfect_signal(self, xp: Any) -> None:
        """EVM should be 0% for identical tx/rx symbols."""
        symbols = xp.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j])
        evm_pct = metrics.evm(symbols, symbols)
        assert evm_pct < 1e-10

    def test_evm_with_known_error(self, xp: Any) -> None:
        """A constant error 0.1 on unit-power points is exactly 10 % EVM."""
        tx = xp.array([1.0 + 0j, 0.0 + 1j, -1.0 + 0j, 0.0 - 1j])
        assert metrics.evm(tx + 0.1, tx) == pytest.approx(10.0, rel=1e-6)

    @pytest.mark.allow_warnings
    def test_evm_near_zero_ref(self, xp: Any) -> None:
        """Verify EVM behavior when reference signal is near zero."""
        tx = xp.zeros(10)
        rx = xp.ones(10)
        assert metrics.evm(rx, tx) == float("inf")

    def test_evm_shape_mismatch(self, xp: Any) -> None:
        """Verify error on shape mismatch in evm."""
        with pytest.raises(ValueError, match="shape mismatch"):
            metrics.evm(xp.zeros(10), xp.zeros(11))

    @pytest.mark.allow_warnings
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
        """The SNR of 20 dB AWGN on 20000 QPSK symbols is estimated within
        0.15 dB: the noise-power estimate has a relative spread of
        1/sqrt(N) = 0.7 %, 0.03 dB."""
        bits = np.random.default_rng(42).integers(0, 2, 40000, dtype="int8")
        symbols = xp.asarray(Constellation.qam(4).map(bits))
        noisy = apply_awgn(symbols, esn0_db=20.0, sps=1, rng=3)
        assert metrics.snr(noisy, symbols) == pytest.approx(20.0, abs=0.15)

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
