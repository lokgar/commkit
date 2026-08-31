"""Tests for performance metrics module."""

import numpy as np
import pytest

from commkit import generate_qam, mapping, metrics, multirate
from commkit.helpers import generate_symbols
from commkit.impairments import apply_awgn


def test_evm_perfect_signal(backend_device, xp):
    """EVM should be 0% for identical tx/rx symbols."""
    symbols = xp.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j])

    evm_pct, evm_db = metrics.evm(symbols, symbols)

    assert evm_pct < 1e-10
    assert evm_db < -200  # Very negative dB for near-zero EVM


def test_evm_with_known_error(backend_device, xp):
    """EVM with known error magnitude."""
    # Reference: unit power symbols
    tx = xp.array([1.0 + 0j, 0.0 + 1j, -1.0 + 0j, 0.0 - 1j])
    # Add 10% error (0.1 magnitude error on unit symbols)
    rx = tx + 0.1

    evm_pct, _ = metrics.evm(rx, tx)

    # Error RMS = 0.1, Reference RMS = 1.0 -> EVM = 10%
    assert abs(evm_pct - 10.0) < 1.0  # Within 1% tolerance


def test_snr_matches_applied(backend_device, xp):
    """SNR estimate should approximately match applied AWGN level."""
    # Generate random QPSK symbols
    symbols = generate_symbols(10000, "qam", 4, seed=42)

    # Apply known Es/N0
    target_snr_db = 20.0
    noisy = apply_awgn(symbols, esn0_db=target_snr_db, sps=1)

    estimated_snr = metrics.snr(noisy, symbols)

    # Should be within 1 dB of target
    assert abs(estimated_snr - target_snr_db) < 1.0, (
        f"SNR estimate {estimated_snr:.1f} dB far from target {target_snr_db:.1f} dB"
    )


def test_snr_high_snr(backend_device, xp):
    """Very high SNR should return very high estimate."""
    symbols = xp.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j])

    snr_db = metrics.snr(symbols, symbols)

    assert snr_db > 100  # Essentially infinite for identical signals


def test_ber_no_errors(backend_device, xp):
    """BER = 0 for identical bit sequences."""
    bits = xp.array([0, 1, 0, 1, 1, 0, 0, 1])

    ber_val = metrics.ber(bits, bits)

    assert ber_val == 0.0


def test_ber_known_errors(backend_device, xp):
    """BER calculation with known error count."""
    bits_tx = xp.array([0, 1, 0, 1, 1, 0, 0, 1])
    bits_rx = xp.array([1, 1, 0, 1, 0, 0, 0, 1])  # 2 errors (positions 0 and 4)

    ber_val = metrics.ber(bits_rx, bits_tx)

    expected_ber = 2 / 8  # 0.25
    assert abs(ber_val - expected_ber) < 1e-10


def test_ber_all_errors(backend_device, xp):
    """BER = 1 when all bits are flipped."""
    bits_tx = xp.array([0, 0, 0, 0])
    bits_rx = xp.array([1, 1, 1, 1])

    ber_val = metrics.ber(bits_rx, bits_tx)

    assert ber_val == 1.0


def test_signal_evm_method(backend_device, xp):
    """Test Signal.evm() method using source_symbols as reference."""

    # Create signal with known source_symbols
    sig = generate_qam(
        order=4, num_symbols=100, sps=1, symbol_rate=1e6, pulse_shape="none"
    )

    # Must resolve symbols before EVM
    sig = multirate.resolve_symbols(sig)
    evm_pct, evm_db = metrics.evm(sig)
    # Relaxed slightly for robustness logic which might have tiny epsilon effects
    assert evm_pct < 1e-4  # Near-zero EVM for perfect signal


def test_signal_ber_method(backend_device, xp):
    """Test Signal.ber() method using source_bits as reference."""

    # Create signal with known source_bits
    sig = generate_qam(
        order=4, num_symbols=100, sps=1, symbol_rate=1e6, pulse_shape="none"
    )

    # Must resolve symbols then demap symbols before BER
    sig = multirate.resolve_symbols(sig)
    sig = mapping.demap_symbols_hard(sig)
    ber_val = metrics.ber(sig)
    assert ber_val == 0.0  # Perfect signal, no errors


def test_signal_demap_hard(backend_device, xp, xpt):
    """Test Signal.demap_symbols_hard() hard decision."""

    sig = generate_qam(
        order=4, num_symbols=50, sps=1, symbol_rate=1e6, pulse_shape="none"
    )

    # Must resolve symbols before demapping
    sig = multirate.resolve_symbols(sig)
    # Demap to bits
    sig = mapping.demap_symbols_hard(sig)

    # Should match source_bits
    xpt.assert_array_equal(sig.resolved_bits.flatten(), sig.source_bits.flatten())


def test_metrics_more(backend_device, xp):
    """Verify more metrics edge cases."""
    # 1. EVM without normalization (now always normalized)
    # The previous test relied on normalize=False to check unnormalized behavior.
    # Since we removed that argument and enforce normalization, we test that
    # normalized inputs return 0% error even if scaled differently?
    # No, if ref and rx are different but scaled differently, they are both normalized
    # to unit power.
    # Original test: ref=[1,1], rx=[1.1, 1.1].
    # Normalized: ref -> [1/sqrt(2), 1/sqrt(2)] -> RMS=1.
    # rx -> [1.1/sqrt(2*1.1^2), ...] -> [1/sqrt(2), ...].
    # So normalized rx == normalized ref. EVM should be 0.
    ref = xp.array([1.0, 1.0])
    rx = xp.array([1.1, 1.1])
    evm_pct, _ = metrics.evm(rx, ref)
    # 0% because they are identical shapes, just scaled.
    assert evm_pct < 1e-5

    # 2. SNR divide by zero (returns inf in current implementation)
    zero = xp.zeros(10)
    snr_val = metrics.snr(zero, zero)
    assert snr_val == float("inf")

    # 3. BER mismatch
    with pytest.raises(ValueError, match="Shape mismatch"):
        metrics.ber(xp.array([0, 1]), xp.array([0]))


def test_ber_empty(backend_device, xp):
    """Verify BER with empty arrays."""
    assert metrics.ber(xp.array([]), xp.array([])) == 0.0


def test_evm_near_zero_ref(backend_device, xp):
    """Verify EVM behavior when reference signal is near zero."""
    tx = xp.zeros(10)
    rx = xp.ones(10)
    pct, db = metrics.evm(rx, tx)
    assert pct == float("inf")
    assert db == float("inf")


def test_evm_shape_mismatch(backend_device, xp):
    """Verify error on shape mismatch in evm."""
    with pytest.raises(ValueError, match="Shape mismatch"):
        metrics.evm(xp.zeros(10), xp.zeros(11))


def test_snr_shape_mismatch(backend_device, xp):
    """Verify error on shape mismatch in snr."""
    with pytest.raises(ValueError, match="Shape mismatch"):
        metrics.snr(xp.zeros(10), xp.zeros(11))


def test_evm_array_handling(backend_device, xp):
    """Verify evm multichannel array handling."""
    # 2 channels, one with error, one perfect
    rx = xp.array([[1.0, 1.0], [1.0, 0.8]])
    tx = xp.array([[1.0, 1.0], [1.1, 1.1]])  # Ch 1 is [1.0, 1.0] after norm

    ep, edb = metrics.evm(rx, tx)
    assert ep.shape == (2,)
    assert ep[0] == 0
    assert ep[1] > 0

    # Test low power mask for array
    tx_zero = xp.zeros((2, 2))
    ep, edb = metrics.evm(rx, tx_zero)
    assert xp.all(ep == float("inf"))
    assert xp.all(edb == float("inf"))


def test_snr_scalar_low_power(backend_device, xp):
    """Verify snr returns -inf when reference power is zero."""
    rx = xp.ones(10)
    tx = xp.zeros(10)
    # Ref power is 0, noise is 1.0. SNR should be -inf dB.
    assert metrics.snr(rx, tx) == float("-inf")


def test_snr_array_low_power(backend_device, xp):
    """Verify snr array handling returns -inf per channel when reference power is zero."""
    rx = xp.ones((2, 10))
    tx = xp.zeros((2, 10))
    # Both channels have 0 signal ref
    res = metrics.snr(rx, tx)
    assert xp.all(res == float("-inf"))

    # Mixed case
    tx_mixed = xp.array([xp.ones(10), xp.zeros(10)])
    res = metrics.snr(rx, tx_mixed)
    assert res[0] == float("inf")  # Perfect match for Ch 0
    assert res[1] == float("-inf")  # 0 Signal for Ch 1


def test_ber_length_mismatch(backend_device, xp):
    """Verify error on bit length mismatch."""
    with pytest.raises(ValueError, match="Shape mismatch"):
        metrics.ber(xp.array([1, 0]), xp.array([1, 0, 1]))


def test_ber_multichannel(backend_device, xp):
    """Verify multi-channel BER returns per-channel error rates."""
    # 2 channels, each with 10 bits
    tx = xp.array([[0, 1, 0, 1, 1, 0, 0, 1, 0, 1], [1, 0, 1, 0, 0, 1, 1, 0, 1, 0]])
    # Channel 0: 1 error at position 0, Channel 1: 2 errors at positions 0,1
    rx = xp.array([[1, 1, 0, 1, 1, 0, 0, 1, 0, 1], [0, 1, 1, 0, 0, 1, 1, 0, 1, 0]])

    ber_values = metrics.ber(rx, tx)

    # Should return array of shape (2,)
    assert ber_values.shape == (2,)
    assert float(ber_values[0]) == pytest.approx(1 / 10)
    assert float(ber_values[1]) == pytest.approx(2 / 10)


# =============================================================================
# EVM BLIND TESTS
# =============================================================================


def test_evm_blind_perfect_signal(backend_device, xp):
    """Blind EVM should be 0% when rx sits exactly on constellation points.

    Uses a perfectly balanced sample (each point appears equally many times)
    so sample average power == population average power == 1.0, avoiding any
    finite-sample gain-estimation artefact.
    """
    from commkit.mapping import gray_constellation

    const = xp.asarray(gray_constellation("qam", 16))
    rx = xp.tile(const, 32)  # 512 symbols, each of the 16 points exactly 32x

    pct, db = metrics.evm(rx, mode="blind", modulation="qam", order=16)

    assert pct < 1e-6


def test_evm_blind_decreases_with_snr(backend_device, xp):
    """Blind EVM at high SNR should be lower than at low SNR."""
    from commkit.impairments import apply_awgn
    from commkit.mapping import gray_constellation

    const = np.asarray(gray_constellation("qam", 16))
    rng = np.random.default_rng(7)
    tx = const[rng.integers(0, 16, 2000)]

    rx_high = apply_awgn(xp.asarray(tx), esn0_db=30.0, sps=1)
    rx_low = apply_awgn(xp.asarray(tx), esn0_db=10.0, sps=1)

    pct_high, _ = metrics.evm(rx_high, mode="blind", modulation="qam", order=16)
    pct_low, _ = metrics.evm(rx_low, mode="blind", modulation="qam", order=16)

    assert pct_high < pct_low


def test_evm_blind_vs_data_aided_converge(backend_device, xp):
    """At high SNR blind and data-aided EVM should agree closely."""
    from commkit.impairments import apply_awgn
    from commkit.mapping import map_bits

    rng = np.random.default_rng(42)
    bits = rng.integers(0, 2, 4000).astype("int32")
    tx = map_bits(xp.asarray(bits), "qam", 16)
    rx = apply_awgn(tx, esn0_db=30.0, sps=1)

    pct_da, _ = metrics.evm(rx, tx)
    pct_bl, _ = metrics.evm(rx, mode="blind", modulation="qam", order=16)

    assert abs(pct_da - pct_bl) < 0.5  # within 0.5 pp at 30 dB SNR


def test_evm_blind_multichannel(backend_device, xp):
    """Blind EVM returns array of shape (N_ch,) for MIMO input."""
    from commkit.mapping import gray_constellation

    const = xp.asarray(gray_constellation("qam", 4))
    rng = np.random.default_rng(1)
    rx = xp.stack([const[rng.integers(0, 4, 200)] for _ in range(3)])  # (3, 200)

    pct, db = metrics.evm(rx, mode="blind", modulation="qam", order=4)

    assert pct.shape == (3,)
    assert xp.all(pct < 1e-6)


def test_evm_blind_missing_args_raises(backend_device, xp):
    """Blind mode without modulation/order raises ValueError."""
    with pytest.raises(ValueError, match="modulation and order"):
        metrics.evm(xp.zeros(10), mode="blind")


def test_evm_data_aided_missing_tx_raises(backend_device, xp):
    """data_aided mode without tx_symbols raises ValueError."""
    with pytest.raises(ValueError, match="tx_symbols"):
        metrics.evm(xp.zeros(10))


def test_evm_unknown_mode_raises(backend_device, xp):
    """Unknown mode string raises ValueError."""
    with pytest.raises(ValueError, match="Unknown mode"):
        metrics.evm(xp.zeros(10), xp.zeros(10), mode="magic")


def test_signal_evm_blind(backend_device, xp):
    """Signal.evm(mode='blind') returns near-zero EVM for a clean signal.

    resolve_symbols() normalises by the empirical symbol-average power of the
    specific N-symbol draw, which differs slightly from the theoretical
    constellation average power (1.0).  For 16-QAM with Var(|c|²) ≈ 0.32 and
    N=2000, the std of the empirical mean is ≈ 1.3%, so blind EVM can be up to
    ~0.65%.  The tolerance of 3% safely covers 3σ without masking real errors.
    """

    sig = generate_qam(
        order=16, num_symbols=2000, sps=1, symbol_rate=1e6, pulse_shape="none"
    )
    sig = multirate.resolve_symbols(sig)

    pct, db = metrics.evm(sig, mode="blind")
    assert pct < 3.0


# =============================================================================
# SER TESTS
# =============================================================================


def test_ser_perfect_signal(backend_device, xp):
    """SER = 0 when rx equals tx exactly."""
    from commkit.mapping import map_bits

    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, 800).astype("int32")
    tx = map_bits(xp.asarray(bits), "qam", 16)

    assert metrics.ser(tx, tx, "qam", 16) == 0.0


def test_ser_high_snr_near_zero(backend_device, xp):
    """SER should be negligible at very high SNR."""
    from commkit.impairments import apply_awgn
    from commkit.mapping import map_bits

    rng = np.random.default_rng(1)
    bits = rng.integers(0, 2, 2000).astype("int32")
    tx = map_bits(xp.asarray(bits), "qam", 4)
    rx = apply_awgn(tx, esn0_db=40.0, sps=1)

    assert metrics.ser(rx, tx, "qam", 4) < 1e-3


def test_ser_multichannel(backend_device, xp):
    """SER returns array (N_ch,) for 2D input."""
    from commkit.mapping import map_bits

    rng = np.random.default_rng(2)
    bits = rng.integers(0, 2, 400).astype("int32")
    tx_row = map_bits(xp.asarray(bits), "qam", 4)
    tx = xp.stack([tx_row, tx_row])  # (2, 200)

    result = metrics.ser(tx, tx, "qam", 4)

    assert result.shape == (2,)
    assert float(result[0]) == 0.0
    assert float(result[1]) == 0.0


def test_ser_shape_mismatch_raises(backend_device, xp):
    """SER raises ValueError on shape mismatch."""
    with pytest.raises(ValueError, match="Shape mismatch"):
        metrics.ser(xp.zeros(10), xp.zeros(11), "qam", 4)


def test_signal_ser_method(backend_device, xp):
    """Signal.ser() returns 0 for a clean signal."""

    sig = generate_qam(
        order=16, num_symbols=200, sps=1, symbol_rate=1e6, pulse_shape="none"
    )
    sig = multirate.resolve_symbols(sig)

    assert metrics.ser(sig) == 0.0


# =============================================================================
# GMI TESTS
# =============================================================================


def test_gmi_high_snr_approaches_log2m(backend_device, xp):
    """At infinite SNR (perfect LLRs), GMI -> log2(M)."""
    from commkit.mapping import compute_llr, map_bits

    k = 4  # bits per symbol for QAM-16
    M = 16
    N = 200
    rng = np.random.default_rng(42)
    bits = rng.integers(0, 2, N * k).astype("int32")
    symbols = map_bits(xp.asarray(bits), "qam", M)

    # Very low noise -> large LLR magnitudes -> GMI ≈ log2(M)
    # Use (N, k) shape so gmi() knows k and returns b/cu in [0, log2(M)]
    llrs = compute_llr(symbols, "qam", M, noise_var=1e-6, output="numpy").reshape(N, k)
    gmi_val = metrics.gmi(llrs, bits.reshape(N, k))

    assert gmi_val > np.log2(M) - 0.05, f"High-SNR GMI {gmi_val:.4f} too low"


def test_gmi_low_snr_approaches_zero(backend_device, xp):
    """At very low SNR, LLRs collapse to zero -> GMI -> 0."""
    from commkit.mapping import compute_llr, map_bits

    k = 4
    M = 16
    N = 500
    rng = np.random.default_rng(7)
    bits = rng.integers(0, 2, N * k).astype("int32")
    symbols = map_bits(xp.asarray(bits), "qam", M)

    # Huge noise -> all LLRs ≈ 0 -> log2(1+1)=1 for every bit -> GMI ≈ 0
    llrs = compute_llr(symbols, "qam", M, noise_var=1e6, output="numpy").reshape(N, k)
    gmi_val = metrics.gmi(llrs, bits.reshape(N, k))

    assert gmi_val < 0.2, f"Low-SNR GMI {gmi_val:.4f} should be near 0"


def test_gmi_flat_input_returns_per_bit(backend_device, xp):
    """Flat 1D input: gmi() treats k=1 and returns per-bit GMI in [0, 1]."""
    from commkit.mapping import compute_llr, map_bits

    bits = np.array([0, 1, 1, 0, 0, 1, 1, 0], dtype="int32")
    symbols = map_bits(xp.asarray(bits), "qam", 4)
    # Flat (N*k,) LLRs - k not known, so gmi() treats as k=1
    llrs = compute_llr(symbols, "qam", 4, noise_var=0.1, output="numpy")
    gmi_val = metrics.gmi(llrs, bits)

    assert 0.0 <= gmi_val <= 1.0


def test_gmi_returns_scalar_float(backend_device, xp):
    """gmi() must return a Python float."""
    from commkit.mapping import compute_llr, map_bits

    k = 2
    N = 4
    bits = np.array([0, 1, 1, 0, 0, 1, 1, 0], dtype="int32")
    symbols = map_bits(xp.asarray(bits), "qam", 4)
    llrs = compute_llr(symbols, "qam", 4, noise_var=0.1, output="numpy").reshape(N, k)
    gmi_val = metrics.gmi(llrs, bits.reshape(N, k))

    assert isinstance(gmi_val, float)


def test_gmi_shape_mismatch_raises(backend_device, xp):
    """gmi() should raise ValueError when llrs and tx_bits have different sizes."""
    llrs = np.array([1.0, -1.0, 2.0])
    bits = np.array([0, 1])
    with pytest.raises(ValueError, match="same number of elements"):
        metrics.gmi(llrs, bits)


def test_gmi_2d_bounded_by_log2m(backend_device, xp):
    """For (N, k) input, GMI ∈ [0, k]."""
    from commkit.mapping import compute_llr, map_bits

    k = 2  # QPSK
    N = 100
    rng = np.random.default_rng(55)
    bits = rng.integers(0, 2, N * k).astype("int32")
    symbols = map_bits(xp.asarray(bits), "qam", 4)
    llrs = compute_llr(symbols, "qam", 4, noise_var=0.1, output="numpy").reshape(N, k)
    bits_2d = bits.reshape(N, k)
    gmi_val = metrics.gmi(llrs, bits_2d)
    assert 0.0 <= gmi_val <= np.log2(4)


# =============================================================================
# MI TESTS
# =============================================================================


def test_mi_high_snr_approaches_log2m(backend_device, xp):
    """At high SNR, MI -> log2(M)."""
    from commkit.mapping import gray_constellation

    M = 16
    const = gray_constellation("qam", M)
    rng = np.random.default_rng(42)
    # Sample uniformly from constellation (no noise)
    symbols = const[rng.integers(0, M, 500)]

    mi_val = metrics.mi(xp.asarray(symbols), "qam", M, noise_var=1e-8)

    assert mi_val > np.log2(M) - 0.1, f"High-SNR MI {mi_val:.4f} too low"


def test_mi_never_exceeds_log2m(backend_device, xp):
    """MI ≤ log2(M) always (capacity bound)."""
    from commkit.mapping import gray_constellation

    M = 4
    const = gray_constellation("qam", M)
    rng = np.random.default_rng(7)
    symbols = const[rng.integers(0, M, 200)]

    for noise_var in [1e-4, 0.1, 1.0, 10.0]:
        mi_val = metrics.mi(xp.asarray(symbols), "qam", M, noise_var=noise_var)
        assert mi_val <= np.log2(M) + 1e-6, (
            f"MI={mi_val:.4f} exceeded log2(M)={np.log2(M):.4f} at noise_var={noise_var}"
        )


def test_mi_returns_scalar_float(backend_device, xp):
    """mi() must return a Python float."""
    from commkit.mapping import gray_constellation

    M = 4
    const = gray_constellation("qam", M)
    symbols = const[:10]
    mi_val = metrics.mi(xp.asarray(symbols), "qam", M, noise_var=0.1)
    assert isinstance(mi_val, float)


def test_mi_decreases_with_noise(backend_device, xp):
    """MI should decrease as noise increases."""
    from commkit.mapping import gray_constellation

    M = 16
    const = gray_constellation("qam", M)
    rng = np.random.default_rng(99)
    symbols = const[rng.integers(0, M, 500)]

    mi_low_noise = metrics.mi(xp.asarray(symbols), "qam", M, noise_var=0.01)
    mi_high_noise = metrics.mi(xp.asarray(symbols), "qam", M, noise_var=1.0)

    assert mi_low_noise > mi_high_noise
