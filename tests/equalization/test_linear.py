"""Linear equalizers: zero-forcing / MMSE (zf_equalizer, apply_taps)."""

import pytest

from commkit import equalization
from commkit.core import Signal


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


class TestZFEqualizer:
    """Tests for the Zero-Forcing / MMSE block equalizer."""

    def test_identity_channel(self, backend_device, xp, xpt):
        """ZF should be a no-op for a unit impulse channel."""
        n = 128
        channel = xp.array([1.0 + 0j], dtype=xp.complex64)

        rng = xp.random.RandomState(0)
        symbols = (rng.randn(n) + 1j * rng.randn(n)).astype(xp.complex64)

        equalized = equalization.zf_equalizer(symbols, channel)

        xpt.assert_allclose(equalized, symbols, atol=1e-5)

    def test_simple_channel_inversion(self, backend_device, xp, xpt):
        """ZF should invert a simple 2-tap channel."""
        n = 256
        channel = xp.array([1.0, 0.5], dtype=xp.complex64)

        # Generate known signal and apply channel via FFT
        rng = xp.random.RandomState(42)
        tx = (rng.randn(n) + 1j * rng.randn(n)).astype(xp.complex64)

        # Apply Channel
        rx = xp.convolve(tx, channel, mode="full")[:n]

        equalized = equalization.zf_equalizer(rx, channel)

        xpt.assert_allclose(equalized, tx, atol=1e-4)

    def test_mmse_better_than_zf_in_noise(self, backend_device, xp):
        """MMSE should have lower MSE than ZF with a noisy spectral null."""
        n = 512
        channel = xp.array([1.0, -0.9, 0.1], dtype=xp.complex64)

        rng = xp.random.RandomState(42)
        tx = (rng.randn(n) + 1j * rng.randn(n)).astype(xp.complex64)

        rx_full = xp.convolve(tx, channel, mode="full")
        rx = rx_full[:n]

        # Add Noise
        noise = 0.1 * (rng.randn(n) + 1j * rng.randn(n)).astype(xp.complex64)
        rx_noisy = rx + noise

        zf_out = equalization.zf_equalizer(rx_noisy, channel)
        mmse_out = equalization.zf_equalizer(rx_noisy, channel, noise_variance=0.01)

        mse_zf = xp.mean(xp.abs(zf_out - tx) ** 2)
        mse_mmse = xp.mean(xp.abs(mmse_out - tx) ** 2)

        mse_zf = float(mse_zf)
        mse_mmse = float(mse_mmse)

        assert mse_mmse < mse_zf, (
            f"MMSE ({mse_mmse:.4f}) not better than ZF ({mse_zf:.4f})"
        )

    def test_siso_output_shape(self, backend_device, xp):
        """ZF SISO output should match input shape."""
        n = 64
        rx = xp.ones(n, dtype=xp.complex64)
        h = xp.array([1.0 + 0j], dtype=xp.complex64)

        out = equalization.zf_equalizer(rx, h)
        assert out.shape == (n,)

    def test_mimo_per_channel(self, backend_device, xp):
        """ZF with SISO channel on MIMO input should equalize per-channel."""
        n = 64
        rx = xp.ones((2, n), dtype=xp.complex64)
        h = xp.array([1.0 + 0j], dtype=xp.complex64)

        out = equalization.zf_equalizer(rx, h)
        assert out.shape == (2, n)

    def test_mmse_multi_block_two_sided_channel(self, backend_device, xp):
        """MMSE must correctly handle IIR two-sided response across multiple blocks.

        Channel h=[0.2, 1.0, 0.2] is symmetric with minimum frequency response 0.6
        (no spectral null). Its MMSE inverse is IIR and two-sided. The 50% overlap-save
        architecture discards N_fft/4 samples from each end of every IFFT block,
        rejecting both the causal and anti-causal circular transients. This test
        uses N >> N_fft/2 (N_fft=1024, B=512) to exercise multiple block boundaries.
        """
        N = 2048  # 4 blocks: ceil(2048/512)=4
        channel = xp.array([0.2, 1.0, 0.2], dtype=xp.complex64)
        # h=[0.2,1,0.2]: H(e^jω) = e^{-jω}(1 + 0.4cos(ω)) >= 0.6 - no null

        rng = xp.random.RandomState(7)
        tx = (rng.randn(N) + 1j * rng.randn(N)).astype(xp.complex64)

        rx_full = xp.convolve(tx, channel, mode="full")
        rx = rx_full[:N]

        noise = 0.05 * (rng.randn(N) + 1j * rng.randn(N)).astype(xp.complex64)
        rx_noisy = rx + noise

        out = equalization.zf_equalizer(rx_noisy, channel, noise_variance=0.0025)

        # Skip edges affected by convolution truncation (mode="full"[:N])
        interior = slice(10, N - 10)
        mse = xp.mean(xp.abs(out[interior] - tx[interior]) ** 2)
        mse = float(mse)

        # At σ²=0.0025, worst-case MMSE MSE ≈ σ²/min(|H|²) ≈ 0.0025/0.36 ≈ 0.007
        assert mse < 0.05, f"MMSE multi-block IIR test failed: interior MSE={mse:.4f}"

    def test_signal_input_returns_signal(self, backend_device, xp, xpt):
        """Signal input returns a Signal with the equalized samples."""
        n = 128
        channel = xp.array([1.0 + 0j], dtype=xp.complex64)
        data = xp.ones(n, dtype=xp.complex64)
        sig = Signal(samples=data, sampling_rate=1.0, symbol_rate=1.0)

        out_sig = equalization.zf_equalizer(sig, channel)
        out_arr = equalization.zf_equalizer(data, channel)

        assert isinstance(out_sig, Signal)
        xpt.assert_allclose(out_sig.samples, out_arr)


jax = pytest.importorskip("jax", reason="JAX not installed")


class TestZF3x3:
    """Tests for the ZF/MMSE equalizer with 3+ channel MIMO (uses linalg.inv)."""

    def test_zf_3x3_identity(self, backend_device, xp, xpt):
        """ZF on 3x3 identity channel should be a no-op (linalg.inv path)."""
        n = 256
        rng = xp.random.RandomState(0)
        symbols = (rng.randn(3, n) + 1j * rng.randn(3, n)).astype(xp.complex64)

        # 3x3 identity impulse response
        channel = xp.zeros((3, 3, 1), dtype=xp.complex64)
        for i in range(3):
            channel[i, i, 0] = 1.0

        equalized = equalization.zf_equalizer(symbols, channel)

        assert equalized.shape == (3, n)
        xpt.assert_allclose(equalized, symbols, atol=1e-4)

    def test_zf_3x3_inversion(self, backend_device, xp, xpt):
        """ZF should invert a non-trivial 3x3 single-tap MIMO channel."""
        n = 256
        rng = xp.random.RandomState(42)
        tx = (rng.randn(3, n) + 1j * rng.randn(3, n)).astype(xp.complex64)

        # Single-tap 3x3 mixing matrix
        H = xp.array(
            [[[1.0], [0.3], [0.0]], [[0.0], [1.0], [0.2]], [[0.1], [0.0], [1.0]]],
            dtype=xp.complex64,
        )

        # Apply channel: rx[i] = sum_j H[i,j] * tx[j]
        rx = xp.zeros_like(tx)
        for i in range(3):
            for j in range(3):
                rx[i] += H[i, j, 0] * tx[j]

        equalized = equalization.zf_equalizer(rx, H, noise_variance=0.0)

        assert equalized.shape == (3, n)
        xpt.assert_allclose(equalized, tx, atol=1e-3)

    def test_mmse_3x3_with_noise(self, backend_device, xp):
        """MMSE 3x3 should run without error and return correct shape."""
        n = 128
        channel = xp.zeros((3, 3, 2), dtype=xp.complex64)
        for i in range(3):
            channel[i, i, 0] = 1.0
            channel[i, i, 1] = 0.2

        rng = xp.random.RandomState(0)
        rx = (rng.randn(3, n) + 1j * rng.randn(3, n)).astype(xp.complex64)

        equalized = equalization.zf_equalizer(rx, channel, noise_variance=0.01)

        assert equalized.shape == (3, n)


class TestEstimateTransferFunction:
    """Welch-H1 data-aided (MIMO) channel estimator."""

    def test_flat_gain_siso(self, backend_device, xp, xpt):
        """A flat complex-gain channel y=g·x -> B(f) ≈ g at every bin."""
        rng = xp.random.RandomState(0)
        n = 1 << 16
        x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(xp.complex64)
        g = 2.0 - 1.0j
        y = (g * x).astype(xp.complex64)
        B = equalization.estimate_transfer_function(x, y, n_fft=256, reg=1e-3)
        assert B.shape == (256,)
        xpt.assert_allclose(B, xp.full_like(B, g), rtol=2e-2, atol=2e-2)

    def test_instantaneous_mimo_mix(self, backend_device, xp, xpt):
        """A memoryless 2x2 mixing matrix -> B(f) ≈ M at every bin (incl. crosstalk)."""
        rng = xp.random.RandomState(1)
        n = 1 << 16
        x = (rng.standard_normal((2, n)) + 1j * rng.standard_normal((2, n))).astype(
            xp.complex64
        )
        M = xp.asarray([[1.0, 0.3j], [-0.2, 0.9]], dtype=xp.complex64)
        y = (M @ x).astype(xp.complex64)
        B = equalization.estimate_transfer_function(x, y, n_fft=256, reg=1e-3)
        assert B.shape == (256, 2, 2)
        xpt.assert_allclose(B[64], M, rtol=3e-2, atol=3e-2)

    def test_siso_memory_frequency_response(self, backend_device, xp, xpt):
        """A short FIR channel -> B(f) ≈ H(f) = fft(h)."""
        rng = xp.random.RandomState(2)
        n = 1 << 16
        x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(xp.complex64)
        h = xp.asarray([1.0, 0.5, -0.25], dtype=xp.complex64)
        y = xp.convolve(x, h)[:n].astype(xp.complex64)
        B = equalization.estimate_transfer_function(x, y, n_fft=256, reg=1e-3)
        Hf = xp.fft.fft(h, n=256)
        xpt.assert_allclose(B, Hf, rtol=0.1, atol=0.05)

    def test_taps_form_centered_impulse(self, backend_device, xp, xpt):
        """num_taps returns a centred, Hann-tapered impulse response (delta for flat)."""
        rng = xp.random.RandomState(3)
        n = 1 << 16
        x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(xp.complex64)
        y = (1.5 * x).astype(xp.complex64)
        taps = equalization.estimate_transfer_function(
            x, y, n_fft=256, reg=1e-3, num_taps=65
        )
        assert taps.shape == (65,)
        peak = int(xp.argmax(xp.abs(taps)))
        assert peak == 32  # centre of 65 taps
        xpt.assert_allclose(taps[peak], xp.asarray(1.5 + 0j), rtol=0.1, atol=0.1)

    def test_mimo_taps_shape_for_zf_equalizer(self, backend_device, xp):
        """MIMO taps come back as (C, C, L) - the layout zf_equalizer consumes."""
        rng = xp.random.RandomState(4)
        n = 1 << 14
        x = (rng.standard_normal((2, n)) + 1j * rng.standard_normal((2, n))).astype(
            xp.complex64
        )
        M = xp.asarray([[1.0, 0.1], [0.0, 1.0]], dtype=xp.complex64)
        y = (M @ x).astype(xp.complex64)
        taps = equalization.estimate_transfer_function(x, y, n_fft=128, num_taps=63)
        assert taps.shape == (2, 2, 63)
        # taps drop straight into zf_equalizer without shape juggling
        out = equalization.zf_equalizer(y, taps, noise_variance=1e-2)
        assert out.shape == (2, n)
