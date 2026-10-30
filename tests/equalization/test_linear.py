"""Linear equalizers: zero-forcing / MMSE (zf_equalizer, apply_taps)."""

import numpy as np
import pytest

from commkit import equalization
from commkit.core import Signal
from tests.common import theory
from tests.common.conversions import to_numpy


def _peak_lag(xp, y, x):
    """Lag in samples of the circular cross-correlation peak of ``y`` against ``x``."""
    c = xp.fft.ifft(xp.fft.fft(y) * xp.conj(xp.fft.fft(x)))
    k = int(xp.argmax(xp.abs(c)))
    return k if k < c.size // 2 else k - c.size


class TestZFEqualizer:
    """Tests for the Zero-Forcing / MMSE block equalizer."""

    def test_identity_channel(self, xp, xpt):
        """ZF should be a no-op for a unit impulse channel."""
        n = 128
        channel = xp.array([1.0 + 0j], dtype=xp.complex64)

        rng = xp.random.RandomState(0)
        symbols = (rng.randn(n) + 1j * rng.randn(n)).astype(xp.complex64)

        equalized = equalization.zf_equalizer(symbols, channel)

        xpt.assert_allclose(equalized, symbols, atol=1e-5)

    def test_simple_channel_inversion(self, xp, xpt):
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

    def test_center_inverts_noncausal_channel(self, xp, xpt):
        """center=c marks lag 0 at index c: taps before it are precursors."""
        n = 256
        channel = xp.array([0.25, 1.0, 0.4j], dtype=xp.complex64)
        rng = xp.random.RandomState(7)
        tx = (rng.randn(n) + 1j * rng.randn(n)).astype(xp.complex64)

        # rx[k] = 0.25 tx[k + 1] + tx[k] + 0.4j tx[k - 1]
        rx = xp.convolve(tx, channel, mode="full")[1 : n + 1]

        equalized = equalization.zf_equalizer(rx, channel, center_tap=1)

        # The last sample lacks the precursor of tx[n], so skip the edges.
        xpt.assert_allclose(equalized[8:-8], tx[8:-8], atol=1e-4)

    @pytest.mark.parametrize("estimate_dtype", ["float64", "complex128"])
    def test_samples_set_the_precision(self, xp, xpt, estimate_dtype):
        """A double-precision estimate does not promote complex64 samples,
        SISO or MIMO (whose 2x2 path runs its own inversion)."""
        rng = xp.random.RandomState(3)
        tx = (rng.randn(2, 256) + 1j * rng.randn(2, 256)).astype(xp.complex64)
        siso = xp.asarray([1.0, 0.5], dtype=estimate_dtype)
        rx = xp.convolve(tx[0], siso.astype(xp.complex64), mode="full")[:256]
        out = equalization.zf_equalizer(rx, siso)
        assert out.dtype == xp.complex64
        xpt.assert_allclose(out, tx[0], atol=1e-4)
        mimo = xp.zeros((2, 2, 2), dtype=estimate_dtype)
        mimo[0, 0, 0] = mimo[1, 1, 0] = 1.0
        mimo[0, 1, 1] = mimo[1, 0, 1] = 0.3
        assert equalization.zf_equalizer(tx, mimo).dtype == xp.complex64

    def test_batches_do_not_change_the_output(self, xp, xpt, monkeypatch):
        """MIMO ZF over batches of 3 blocks equals the one-batch result."""
        import commkit._overlap_save as ols

        rng = xp.random.RandomState(4)
        tx = (rng.randn(2, 10_000) + 1j * rng.randn(2, 10_000)).astype(xp.complex64)
        h = xp.zeros((2, 2, 3), dtype=xp.complex64)
        h[0, 0, 0] = h[1, 1, 0] = 1.0
        h[0, 1, 2] = h[1, 0, 1] = 0.3j
        whole = equalization.zf_equalizer(tx, h)
        monkeypatch.setattr(ols, "_OLS_ELEMENTS", 2 * 1024 * 3)
        xpt.assert_allclose(equalization.zf_equalizer(tx, h), whole, atol=1e-5)

    @pytest.mark.parametrize("center", [-1, 3])
    def test_center_out_of_range_raises(self, xp, center):
        channel = xp.array([0.25, 1.0, 0.4j], dtype=xp.complex64)
        samples = xp.ones(64, dtype=xp.complex64)
        with pytest.raises(ValueError, match="center"):
            equalization.zf_equalizer(samples, channel, center_tap=center)

    @pytest.mark.parametrize(
        "taps", [[1.0, -0.9, 0.1], [0.2, 1.0, 0.2]], ids=["near-null", "two-sided"]
    )
    def test_mse_matches_theory(self, xp, taps):
        """On a unit-power white input with white noise σ², the equalized
        MSE is the theoretical per-bin value: MMSE ``mean σ²/(|H|² + σ²)``,
        ZF ``mean σ²/|H|²``, within 10 % (2e4 samples, several overlap-save
        blocks).  So MMSE beats ZF, by a factor ~5 near the spectral null."""
        n, sigma2 = 20000, 0.01
        rng = np.random.default_rng(42)
        tx = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
        noise = np.sqrt(sigma2 / 2) * (
            rng.standard_normal(n) + 1j * rng.standard_normal(n)
        )
        rx = (np.convolve(tx, taps)[:n] + noise).astype(np.complex64)
        h_fft = np.fft.fft(taps, 4096)
        interior = slice(100, n - 100)
        for nv, expected in (
            (sigma2, theory.mmse_mse(h_fft, sigma2, 1.0)),
            (0.0, float(np.mean(sigma2 / np.abs(h_fft) ** 2))),
        ):
            out = to_numpy(
                equalization.zf_equalizer(
                    xp.asarray(rx),
                    xp.asarray(np.asarray(taps, np.complex64)),
                    noise_var=nv,
                )
            )
            mse = float(np.mean(np.abs(out[interior] - tx[interior]) ** 2))
            assert mse == pytest.approx(expected, rel=0.1), (nv, mse, expected)

    def test_siso_output_shape(self, xp):
        """ZF SISO output should match input shape."""
        n = 64
        rx = xp.ones(n, dtype=xp.complex64)
        h = xp.array([1.0 + 0j], dtype=xp.complex64)

        out = equalization.zf_equalizer(rx, h)
        assert out.shape == (n,)

    def test_mimo_per_channel(self, xp):
        """ZF with SISO channel on MIMO input should equalize per-channel."""
        n = 64
        rx = xp.ones((2, n), dtype=xp.complex64)
        h = xp.array([1.0 + 0j], dtype=xp.complex64)

        out = equalization.zf_equalizer(rx, h)
        assert out.shape == (2, n)

    def test_signal_input_returns_signal(self, xp, xpt):
        """Signal input returns a Signal with the equalized samples."""
        n = 128
        channel = xp.array([1.0 + 0j], dtype=xp.complex64)
        data = xp.ones(n, dtype=xp.complex64)
        sig = Signal(samples=data, sampling_rate=1.0, symbol_rate=1.0)

        out_sig = equalization.zf_equalizer(sig, channel)
        out_arr = equalization.zf_equalizer(data, channel)

        assert isinstance(out_sig, Signal)
        xpt.assert_allclose(out_sig.samples, out_arr)


class TestApplyTaps:
    """Tests for apply_taps (frozen-weight inference pass)."""

    def test_signal_input_returns_signal_at_symbol_rate(self, xp, xpt):
        """Signal input: sps is taken from the signal and sampling_rate is
        set to symbol_rate on the returned Signal."""
        n = 256
        sps = 2
        weights = xp.array([1.0 + 0j], dtype=xp.complex64)  # single-tap identity
        data = xp.ones(n, dtype=xp.complex64)
        sig = Signal(samples=data, sampling_rate=sps * 1e6, symbol_rate=1e6)

        out_sig = equalization.apply_taps(sig, weights, normalize=False)
        out_arr = equalization.apply_taps(data, weights, sps=sps, normalize=False)

        assert isinstance(out_sig, Signal)
        assert out_sig.sampling_rate == 1e6
        xpt.assert_allclose(out_sig.samples, out_arr)


class TestZF3x3:
    """Tests for the ZF/MMSE equalizer with 3+ channel MIMO (uses linalg.inv)."""

    def test_zf_3x3_identity(self, xp, xpt):
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

    def test_zf_3x3_inversion(self, xp, xpt):
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

        equalized = equalization.zf_equalizer(rx, H, noise_var=0.0)

        assert equalized.shape == (3, n)
        xpt.assert_allclose(equalized, tx, atol=1e-3)

    def test_mmse_3x3_with_noise(self, xp):
        """MMSE 3x3 should run without error and return correct shape."""
        n = 128
        channel = xp.zeros((3, 3, 2), dtype=xp.complex64)
        for i in range(3):
            channel[i, i, 0] = 1.0
            channel[i, i, 1] = 0.2

        rng = xp.random.RandomState(0)
        rx = (rng.randn(3, n) + 1j * rng.randn(3, n)).astype(xp.complex64)

        equalized = equalization.zf_equalizer(rx, channel, noise_var=0.01)

        assert equalized.shape == (3, n)


class TestEstimateTransferFunction:
    """Welch-H1 data-aided (MIMO) channel estimator."""

    def test_flat_gain_siso(self, xp, xpt):
        """A flat complex-gain channel y=g·x -> B(f) ≈ g at every bin."""
        rng = xp.random.RandomState(0)
        n = 1 << 16
        x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(xp.complex64)
        g = 2.0 - 1.0j
        y = (g * x).astype(xp.complex64)
        B = equalization.estimate_transfer_function(y, x, nfft=256, regularization=1e-3)
        assert B.shape == (256,)
        xpt.assert_allclose(B, xp.full_like(B, g), rtol=2e-2, atol=2e-2)

    def test_instantaneous_mimo_mix(self, xp, xpt):
        """A memoryless 2x2 mixing matrix -> B(f) ≈ M at every bin (incl. crosstalk)."""
        rng = xp.random.RandomState(1)
        n = 1 << 16
        x = (rng.standard_normal((2, n)) + 1j * rng.standard_normal((2, n))).astype(
            xp.complex64
        )
        M = xp.asarray([[1.0, 0.3j], [-0.2, 0.9]], dtype=xp.complex64)
        y = (M @ x).astype(xp.complex64)
        B = equalization.estimate_transfer_function(y, x, nfft=256, regularization=1e-3)
        assert B.shape == (256, 2, 2)
        xpt.assert_allclose(B[64], M, rtol=3e-2, atol=3e-2)

    def test_siso_memory_frequency_response(self, xp, xpt):
        """A short FIR channel -> B(f) ≈ H(f) = fft(h)."""
        rng = xp.random.RandomState(2)
        n = 1 << 16
        x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(xp.complex64)
        h = xp.asarray([1.0, 0.5, -0.25], dtype=xp.complex64)
        y = xp.convolve(x, h)[:n].astype(xp.complex64)
        B = equalization.estimate_transfer_function(y, x, nfft=256, regularization=1e-3)
        Hf = xp.fft.fft(h, n=256)
        xpt.assert_allclose(B, Hf, rtol=0.1, atol=0.05)

    def test_taps_form_centered_impulse(self, xp, xpt):
        """num_taps returns a centred, Hann-tapered impulse response (delta for flat)."""
        rng = xp.random.RandomState(3)
        n = 1 << 16
        x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(xp.complex64)
        y = (1.5 * x).astype(xp.complex64)
        taps = equalization.estimate_transfer_function(
            y, x, nfft=256, regularization=1e-3, num_taps=65
        )
        assert taps.shape == (65,)
        peak = int(xp.argmax(xp.abs(taps)))
        assert peak == 32  # centre of 65 taps
        xpt.assert_allclose(taps[peak], xp.asarray(1.5 + 0j), rtol=0.1, atol=0.1)

    def test_even_num_taps_returns_exactly_num_taps(self, xp, xpt):
        """An even num_taps returns num_taps taps with an untapered lag 0 at L // 2."""
        rng = xp.random.RandomState(3)
        n = 1 << 16
        x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(xp.complex64)
        y = (1.5 * x).astype(xp.complex64)
        taps = equalization.estimate_transfer_function(
            y, x, nfft=256, regularization=1e-3, num_taps=16
        )
        assert taps.shape == (16,)
        assert int(xp.argmax(xp.abs(taps))) == 8
        xpt.assert_allclose(taps[8], xp.asarray(1.5 + 0j), rtol=0.01, atol=0.01)

    @pytest.mark.parametrize("num_taps", [15, 16, 31])
    def test_taps_compose_with_zf_equalizer(self, xp, num_taps):
        """zf_equalizer(rx, taps, center_tap=L // 2) reproduces tx at lag 0."""
        rng = xp.random.RandomState(5)
        n = 1 << 15
        tx = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / 2**0.5
        tx = tx.astype(xp.complex64)
        h = xp.asarray([0.25, 1.0, 0.4j], dtype=xp.complex64)
        rx = xp.convolve(tx, h)[:n].astype(xp.complex64)

        taps = equalization.estimate_transfer_function(rx, tx, num_taps=num_taps)
        out = equalization.zf_equalizer(rx, taps, center_tap=num_taps // 2)

        assert _peak_lag(xp, out, tx) == 0
        # The Hann taper attenuates the off-centre taps, so the inversion is
        # approximate: about -20 dB at 15 taps, -30 dB at 31.
        err = float(xp.mean(xp.abs(out[64:-64] - tx[64:-64]) ** 2))
        assert err < 10 ** (-1.5)

    def test_mimo_taps_compose_with_zf_equalizer(self, xp, xpt):
        """MIMO taps come back as (C, C, L) and invert the channel with center."""
        rng = xp.random.RandomState(4)
        n = 1 << 14
        x = (rng.standard_normal((2, n)) + 1j * rng.standard_normal((2, n))).astype(
            xp.complex64
        )
        M = xp.asarray([[1.0, 0.1], [0.0, 1.0]], dtype=xp.complex64)
        y = (M @ x).astype(xp.complex64)
        taps = equalization.estimate_transfer_function(y, x, nfft=128, num_taps=63)
        assert taps.shape == (2, 2, 63)
        out = equalization.zf_equalizer(y, taps, noise_var=1e-2, center_tap=31)
        assert out.shape == (2, n)
        xpt.assert_allclose(out[:, 64:-64], x[:, 64:-64], atol=0.1)

    def test_signal_input_unwraps_both_arguments(self, xp, xpt):
        """Signal input for either/both arguments unwraps to .samples; output
        stays a raw array (frequency response is not signal-domain data)."""
        rng = xp.random.RandomState(0)
        n = 1 << 14
        x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(xp.complex64)
        g = 2.0 - 1.0j
        y = (g * x).astype(xp.complex64)
        ref_sig = Signal(samples=x, sampling_rate=1.0, symbol_rate=1.0)
        rx_sig = Signal(samples=y, sampling_rate=1.0, symbol_rate=1.0)

        B_sig = equalization.estimate_transfer_function(
            rx_sig, ref_sig, nfft=256, regularization=1e-3
        )
        B_arr = equalization.estimate_transfer_function(
            y, x, nfft=256, regularization=1e-3
        )

        assert not isinstance(B_sig, Signal)
        xpt.assert_allclose(B_sig, B_arr)
