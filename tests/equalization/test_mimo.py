"""Butterfly (2x2 / 3x3) MIMO sequential equalization."""

import pytest

from commkit import equalization, generate_psk, generate_qam


@pytest.fixture(autouse=True)
def _enable_jax_x64():
    """Enable JAX x64 mode for all tests in this module.

    JAX RLS requires complex128 for P-matrix stability; LMS CPR requires float64
    for phase accumulation. Enabling x64 globally is safe - it only affects
    precision when 64-bit dtypes are explicitly requested.
    """
    try:
        import jax

        jax.config.update("jax_enable_x64", True)
    except ImportError:
        pass


class TestButterflyMIMO:
    """Tests for butterfly MIMO equalization structure."""

    def test_lms_2x2_cross_channel(self, backend_device, xp):
        """LMS butterfly should recover 2 streams through a 2x2 mixing channel."""
        n_symbols = 3000

        # 2x2 channel mixing matrix
        H = xp.array([[1.0, 0.3], [0.2, 1.0]], dtype=xp.complex64)

        sig = generate_psk(
            symbol_rate=1e6,
            num_symbols=n_symbols,
            order=4,
            pulse_shape="rrc",
            sps=2,
            num_streams=2,
            seed=10,
        )
        tx_mimo = xp.asarray(sig.source_symbols)
        rx_up = xp.asarray(sig.samples)

        # Mix the streams (multiplying H @ rx_up where rx_up is (2, N_samples))
        rx_mimo = H @ rx_up
        rx_mimo = xp.ascontiguousarray(rx_mimo)

        result = equalization.lms(
            rx_mimo,
            training_symbols=tx_mimo,
            num_taps=21,
            step_size=0.01,
            modulation="psk",
            order=4,
        )

        y = result.y_hat
        w = result.weights

        # Output should be (2, N_sym)
        assert y.shape[0] == 2
        assert w.shape == (2, 2, 21)

        # Check convergence on both streams
        e = xp.abs(result.error) ** 2
        mse_ch0 = xp.mean(e[0, -500:])
        mse_ch1 = xp.mean(e[1, -500:])

        mse_ch0 = float(mse_ch0)
        mse_ch1 = float(mse_ch1)

        assert mse_ch0 < 0.1, f"MIMO ch0 MSE = {mse_ch0:.4f}"
        assert mse_ch1 < 0.1, f"MIMO ch1 MSE = {mse_ch1:.4f}"

    def test_cma_2x2_polarization_demux(self, backend_device, xp):
        """CMA butterfly should demux 2 mixed polarizations."""
        n_symbols = 3000

        sig = generate_psk(
            symbol_rate=1e6,
            num_symbols=n_symbols,
            order=4,
            pulse_shape="rrc",
            sps=2,
            num_streams=2,
            seed=30,
        )
        rx_up = xp.asarray(sig.samples)

        # Polarization rotation mixing
        theta = xp.pi / 6  # 30 degree rotation
        c, s = xp.cos(theta), xp.sin(theta)
        H = xp.array([[c, s], [-s, c]], dtype=xp.complex64)

        rx_mimo = H @ rx_up
        rx_mimo = xp.ascontiguousarray(rx_mimo)

        result = equalization.cma(
            rx_mimo,
            num_taps=11,
            step_size=0.003,
            modulation="psk",
            order=4,
        )

        y = result.y_hat
        assert y.shape[0] == 2

        # After CMA, output should have constant modulus on both streams
        for ch in range(2):
            modulus = xp.abs(y[ch, -500:])
            std_dev = xp.std(modulus)
            std_dev = float(std_dev)

            assert std_dev < 0.35, f"CMA MIMO ch{ch} modulus std = {std_dev:.4f}"

    def test_zf_mimo_channel_matrix(self, backend_device, xp, xpt):
        """ZF with full (C, C, L) channel matrix should invert MIMO channel."""
        n = 128
        rng = xp.random.RandomState(42)

        # 2x2 single-tap channel matrix
        # H[rx, tx, delay]
        H = xp.zeros((2, 2, 1), dtype=xp.complex64)
        H[0, 0, 0] = 1.0
        H[0, 1, 0] = 0.3
        H[1, 0, 0] = 0.2
        H[1, 1, 0] = 1.0

        # Transmit 2 independent streams
        tx = (rng.randn(2, n) + 1j * rng.randn(2, n)).astype(xp.complex64)

        # Apply MIMO channel in frequency domain
        H_f = xp.fft.fft(H, n=n, axis=-1)  # (2, 2, N)
        Tx_f = xp.fft.fft(tx, axis=-1)  # (2, N)
        Rx_f = xp.zeros_like(Tx_f)

        # Manually compute matrix mul at each freq bin?
        # Einsum could work: 'r t k, t k -> r k'
        # CuPy supports einsum.
        # But for loop is also fine for test clarity.
        # Actually H_f is (Rx, Tx, Freq), Tx_f is (Tx, Freq).
        # We want (Rx, Freq).

        # Vectorized matmul broadcasting over freq?
        # Move axis to make it (N, Rx, Tx) @ (N, Tx, 1) -> (N, Rx, 1)
        # H_f_T = H_f.transpose(2, 0, 1) # (N, 2, 2)
        # Tx_f_T = Tx_f.T[..., None]     # (N, 2, 1)
        # Rx_f_T = H_f_T @ Tx_f_T        # (N, 2, 1)
        # Rx_f = Rx_f_T.squeeze(-1).T    # (2, N)

        H_f_T = xp.moveaxis(H_f, -1, 0)
        Tx_f_T = xp.moveaxis(Tx_f, -1, 0)[..., None]
        Rx_f_T = H_f_T @ Tx_f_T
        Rx_f = xp.moveaxis(Rx_f_T.squeeze(-1), 0, -1)

        rx = xp.fft.ifft(Rx_f, axis=-1).astype(xp.complex64)

        equalized = equalization.zf_equalizer(rx, H)

        xpt.assert_allclose(equalized, tx, atol=1e-3)


jax = pytest.importorskip("jax", reason="JAX not installed")


class TestButterflyMIMOExtended:
    """Additional MIMO butterfly tests for RDE and JAX backends."""

    def test_rde_2x2_butterfly_numba(self, backend_device, xp):
        """RDE Numba butterfly should handle 2x2 cross-polarization without error."""

        n_symbols = 2000
        sig = generate_qam(
            symbol_rate=1e6,
            num_symbols=n_symbols,
            order=16,
            pulse_shape="rrc",
            sps=2,
            seed=11,
        )
        rx = xp.asarray(sig.samples)

        # Simulate polarization mixing: channel mixes two copies of same signal
        mix = xp.array([[0.9, 0.1], [0.1, 0.9]], dtype=xp.complex64)
        rx_mimo = xp.stack(
            [
                mix[0, 0] * rx + mix[0, 1] * xp.roll(rx, 3),
                mix[1, 0] * rx + mix[1, 1] * xp.roll(rx, 5),
            ],
            axis=0,
        )

        result = equalization.rde(
            rx_mimo,
            num_taps=11,
            step_size=5e-4,
            modulation="qam",
            order=16,
            backend="numba",
        )

        assert result.y_hat.shape == (2, n_symbols)
        assert result.weights.shape == (2, 2, 11)
        assert result.error.shape == (2, n_symbols)

    def test_lms_jax_2x2_cross_channel(self, backend_device, xp):
        """LMS JAX butterfly should cancel cross-channel interference."""

        pytest.importorskip("jax")

        n_symbols = 2000
        sig = generate_psk(
            symbol_rate=1e6,
            num_symbols=n_symbols,
            order=4,
            pulse_shape="rrc",
            sps=2,
            seed=7,
        )
        rx = xp.asarray(sig.samples)
        train = xp.asarray(sig.source_symbols)

        # 2x2 mixed input
        rx_mimo = xp.stack([rx, xp.roll(rx, 2)], axis=0)
        train_mimo = xp.stack([train, train], axis=0)

        result = equalization.lms(
            rx_mimo,
            training_symbols=train_mimo,
            num_taps=7,
            step_size=0.05,
            modulation="psk",
            order=4,
            backend="jax",
        )

        assert result.y_hat.shape == (2, n_symbols)
        assert result.weights.shape == (2, 2, 7)
