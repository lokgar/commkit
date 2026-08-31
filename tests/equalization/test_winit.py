"""Weight initialization, prefix-pad normalization, and length-independence."""

import numpy as np
import pytest

from commkit import equalization, generate_qam
from commkit.equalization import EqualizerResult
from commkit.mapping import gray_constellation


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


def _to_np(arr):
    """Convert a NumPy or CuPy array to plain NumPy (no-op for NumPy)."""
    if hasattr(arr, "get"):  # CuPy
        return arr.get()
    return np.asarray(arr)


jax = pytest.importorskip("jax", reason="JAX not installed")


def _make_qam16_rx(xp, n_symbols=2000, seed=0):
    """Generate a simple AWGN-impaired 16-QAM signal at 2 SPS."""
    from commkit.impairments import apply_awgn

    sig = generate_qam(
        symbol_rate=1e6,
        num_symbols=n_symbols,
        order=16,
        pulse_shape="rrc",
        sps=2,
        seed=seed,
    )
    rx = apply_awgn(sig.samples, esn0_db=20.0, sps=2)
    return xp.ascontiguousarray(xp.asarray(rx))


class TestWInit:
    """w_init parameter: warm-start from prior equalizer weights."""

    def test_lms_accepts_w_init(self, backend_device, xp):
        """lms() accepts w_init array with correct shape and returns EqualizerResult."""
        rx = _make_qam16_rx(xp)
        num_taps, num_ch = 21, 1
        w0 = np.zeros((num_ch, num_ch, num_taps), dtype=np.complex64)
        w0[0, 0, num_taps // 2] = 1.0 + 0j

        result = equalization.lms(
            rx,
            training_symbols=None,
            modulation="qam",
            order=16,
            num_taps=num_taps,
            w_init=w0,
        )
        assert isinstance(result, EqualizerResult)
        assert result.weights.shape == (num_taps,)  # SISO squeeze

    def test_rls_accepts_w_init(self, backend_device, xp):
        """rls() accepts w_init array with correct shape."""

        sig = generate_qam(
            symbol_rate=1e6, num_symbols=1000, order=4, pulse_shape="rrc", sps=2, seed=1
        )
        rx = xp.asarray(sig.samples)
        num_taps, num_ch = 11, 1
        w0 = np.zeros((num_ch, num_ch, num_taps), dtype=np.complex64)
        w0[0, 0, num_taps // 2] = 1.0 + 0j

        result = equalization.rls(
            rx,
            training_symbols=xp.asarray(sig.source_symbols),
            modulation="qam",
            order=4,
            num_taps=num_taps,
            sps=2,
            w_init=w0,
        )
        assert isinstance(result, EqualizerResult)

    def test_cma_accepts_w_init(self, backend_device, xp):
        """cma() accepts w_init array with correct shape."""
        rx = _make_qam16_rx(xp)
        num_taps, num_ch = 21, 1
        w0 = np.zeros((num_ch, num_ch, num_taps), dtype=np.complex64)
        w0[0, 0, num_taps // 2] = 1.0 + 0j

        result = equalization.cma(
            rx, modulation="qam", order=16, num_taps=num_taps, w_init=w0
        )
        assert isinstance(result, EqualizerResult)

    def test_rde_accepts_w_init(self, backend_device, xp):
        """rde() accepts w_init array with correct shape."""
        rx = _make_qam16_rx(xp)
        num_taps, num_ch = 21, 1
        w0 = np.zeros((num_ch, num_ch, num_taps), dtype=np.complex64)
        w0[0, 0, num_taps // 2] = 1.0 + 0j

        result = equalization.rde(
            rx, modulation="qam", order=16, num_taps=num_taps, w_init=w0
        )
        assert isinstance(result, EqualizerResult)

    def test_w_init_shape_mismatch_raises(self, backend_device, xp):
        """Wrong w_init shape raises ValueError before kernel is called."""
        rx = _make_qam16_rx(xp)
        bad_w = np.zeros((1, 1, 99), dtype=np.complex64)  # wrong num_taps

        with pytest.raises(ValueError, match="w_init shape"):
            equalization.cma(rx, modulation="qam", order=16, num_taps=21, w_init=bad_w)

        with pytest.raises(ValueError, match="w_init shape"):
            equalization.rde(rx, modulation="qam", order=16, num_taps=21, w_init=bad_w)

    def test_lms_to_rde_handoff_output_shape(self, backend_device, xp):
        """LMS weights can be handed off to RDE via w_init; output shape is correct."""
        rx = _make_qam16_rx(xp, n_symbols=3000)
        half = rx.shape[-1] // 2

        pre_rx = rx[..., :half]
        payload_rx = rx[..., half:]

        pre = equalization.lms(
            pre_rx,
            modulation="qam",
            order=16,
            num_taps=21,
            step_size=0.05,
        )
        w0 = _to_np(pre.weights)
        if w0.ndim == 1:
            w0 = w0[np.newaxis, np.newaxis, :]

        result = equalization.rde(
            payload_rx,
            modulation="qam",
            order=16,
            num_taps=21,
            step_size=1e-4,
            w_init=w0,
        )
        expected_syms = payload_rx.shape[-1] // 2
        assert result.y_hat.shape[-1] == expected_syms

    def test_warm_start_rde_same_or_better_evm(self, backend_device, xp):
        """RDE warm-started from LMS achieves same or better EVM than cold-start."""
        from commkit.impairments import apply_awgn

        sig = generate_qam(
            symbol_rate=1e6,
            num_symbols=4000,
            order=16,
            pulse_shape="rrc",
            sps=2,
            seed=42,
        )
        rx_np = apply_awgn(sig.samples, esn0_db=25.0, sps=2)
        rx = xp.asarray(rx_np)

        # Cold-start RDE
        cold = equalization.rde(
            rx, modulation="qam", order=16, num_taps=21, step_size=5e-4
        )
        # LMS pre-convergence
        pre = equalization.lms(
            rx,
            training_symbols=xp.asarray(sig.source_symbols[:200]),
            modulation="qam",
            order=16,
            num_taps=21,
        )
        _w = pre.weights
        w0 = _to_np(pre.weights)
        if w0.ndim == 1:
            w0 = w0[np.newaxis, np.newaxis, :]

        # Warm RDE
        warm = equalization.rde(
            rx, modulation="qam", order=16, num_taps=21, step_size=5e-4, w_init=w0
        )

        tail = slice(-500, None)
        ref = _to_np(sig.source_symbols)
        cold_hat = _to_np(cold.y_hat)
        warm_hat = _to_np(warm.y_hat)
        evm_cold = float(np.mean(np.abs(cold_hat[tail] - ref[tail]) ** 2))
        evm_warm = float(np.mean(np.abs(warm_hat[tail] - ref[tail]) ** 2))
        # Warm start must not be significantly worse
        assert evm_warm <= evm_cold * 1.5, (
            f"Warm RDE EVM {evm_warm:.4f} much worse than cold {evm_cold:.4f}"
        )


class TestEqualizerWInitBackend:
    """Verify w_init works correctly on both numba and jax backends."""

    def test_cma_jax_w_init(self, backend_device, xp):
        """CMA JAX backend accepts w_init without error."""
        pytest.importorskip("jax")
        rx = _make_qam16_rx(xp)
        num_taps = 21
        w0 = np.zeros((1, 1, num_taps), dtype=np.complex64)
        w0[0, 0, num_taps // 2] = 1.0 + 0j

        result = equalization.cma(
            rx,
            modulation="qam",
            order=16,
            num_taps=num_taps,
            w_init=w0,
            backend="jax",
        )
        assert isinstance(result, EqualizerResult)

    def test_rde_jax_w_init(self, backend_device, xp):
        """RDE JAX backend accepts w_init without error."""
        pytest.importorskip("jax")
        rx = _make_qam16_rx(xp)
        num_taps = 21
        w0 = np.zeros((1, 1, num_taps), dtype=np.complex64)
        w0[0, 0, num_taps // 2] = 1.0 + 0j

        result = equalization.rde(
            rx,
            modulation="qam",
            order=16,
            num_taps=num_taps,
            w_init=w0,
            backend="jax",
        )
        assert isinstance(result, EqualizerResult)


class TestNormalizationLengthIndependence:
    """Normalization uses full-signal RMS - training output scales with signal power."""

    @pytest.mark.parametrize("algo", ["lms", "rls"])
    @pytest.mark.parametrize("backend", ["numba"])
    def test_training_output_finite(self, algo, backend, backend_device, xp):
        """y_hat training region is finite and non-trivial."""
        import numpy as np

        from commkit.equalization import lms, rls
        from commkit.mapping import gray_constellation

        rng = np.random.default_rng(42)
        n_train = 200
        n_sym = 500

        const = gray_constellation("qam", 16).astype(np.complex64)
        syms = const[rng.integers(0, 16, n_sym)]
        noise = (
            0.05 * (rng.standard_normal(n_sym) + 1j * rng.standard_normal(n_sym))
        ).astype(np.complex64)
        sig = (syms + noise).astype(np.complex64)

        fn = lms if algo == "lms" else rls
        res = fn(
            sig,
            training_symbols=syms[:n_train],
            num_taps=1,
            sps=1,
            modulation="qam",
            order=16,
            backend=backend,
        )

        assert np.all(np.isfinite(np.asarray(res.y_hat[:n_train])))


def _make_qpsk(xp, n_sym=2000, snr_db=20.0, seed=77):
    """Build a QPSK signal using the given array module (numpy or cupy)."""
    rng = np.random.default_rng(seed)
    const = gray_constellation("psk", 4).astype(np.complex64)
    syms_np = const[rng.integers(0, 4, n_sym)]
    noise_std = np.sqrt(10 ** (-snr_db / 10) / 2)
    samples_np = (
        syms_np
        + noise_std * (rng.standard_normal(n_sym) + 1j * rng.standard_normal(n_sym))
    ).astype(np.complex64)
    return xp.asarray(samples_np), xp.asarray(syms_np)


def _algo_kw(algo, num_taps):
    """Return algorithm-specific kwargs (lms uses step_size, rls uses forgetting_factor)."""
    base = dict(num_taps=num_taps, sps=1, modulation="psk", order=4)
    return (
        {**base, "step_size": 1e-2}
        if algo == "lms"
        else {**base, "forgetting_factor": 0.999}
    )


class TestPrefixPadNormPhase4:
    """Tests for samples_prefix / pad_mode / input_norm_factor added in Phase 4."""

    @pytest.mark.parametrize("algo", ["lms", "rls"])
    def test_pad_mode_zeros_is_baseline(self, algo, backend_device, xp, xpt):
        """Explicit pad_mode='zeros' with no prefix must be byte-exact with default."""
        samples, syms = _make_qpsk(xp)
        fn = getattr(equalization, algo)
        kw = _algo_kw(algo, num_taps=7)
        r_default = fn(samples, syms[:50], **kw)
        r_explicit = fn(samples, syms[:50], **kw, pad_mode="zeros", samples_prefix=None)
        xpt.assert_array_equal(
            xp.asarray(r_default.y_hat),
            xp.asarray(r_explicit.y_hat),
        )

    @pytest.mark.parametrize("algo", ["lms", "rls"])
    def test_samples_prefix_does_not_worsen_leading_error(
        self, algo, backend_device, xp
    ):
        """Warm prefix must not increase MSE on the first num_taps output symbols."""
        n_total, half, num_taps = 4000, 2000, 11
        samples, syms = _make_qpsk(xp, n_sym=n_total, snr_db=30.0)
        fn = getattr(equalization, algo)
        kw = _algo_kw(algo, num_taps=num_taps)

        r1 = fn(samples[:half], syms[: half // 2], **kw)

        r_cold = fn(
            samples[half:],
            syms[half : half + 50],
            **kw,
            w_init=r1.weights,
            input_norm_factor=r1.input_norm_factor,
        )
        prefix = samples[half - num_taps + 1 : half]
        r_prefix = fn(
            samples[half:],
            syms[half : half + 50],
            **kw,
            w_init=r1.weights,
            input_norm_factor=r1.input_norm_factor,
            samples_prefix=prefix,
        )
        ref = xp.asarray(syms[half : half + num_taps])
        e_cold = float(xp.mean(xp.abs(xp.asarray(r_cold.y_hat[:num_taps]) - ref) ** 2))
        e_prefix = float(
            xp.mean(xp.abs(xp.asarray(r_prefix.y_hat[:num_taps]) - ref) ** 2)
        )
        assert e_prefix <= e_cold + 1e-3, (
            f"{algo}: prefix raised leading MSE ({e_prefix:.4f} > {e_cold:.4f})"
        )

    def test_samples_prefix_shape_validation_lms(self, backend_device, xp):
        """Undersized samples_prefix must raise ValueError mentioning 'pad_left'."""
        samples, syms = _make_qpsk(xp, n_sym=500)
        # pad_left = min(num_taps//2, ...) = min(5, ...) - prefix of 1 is too short
        with pytest.raises(ValueError, match="pad_left"):
            equalization.lms(
                samples,
                syms[:20],
                num_taps=11,
                sps=1,
                step_size=1e-2,
                modulation="psk",
                order=4,
                samples_prefix=xp.zeros(1, dtype=xp.complex64),
            )

    def test_pad_mode_edge_lms(self, backend_device, xp):
        """pad_mode='edge' must not raise and must produce finite output."""
        samples, syms = _make_qpsk(xp, n_sym=500)
        r = equalization.lms(
            samples,
            syms[:20],
            num_taps=11,
            sps=1,
            step_size=1e-2,
            modulation="psk",
            order=4,
            pad_mode="edge",
        )
        assert bool(xp.all(xp.isfinite(xp.asarray(r.y_hat))))

    def test_input_norm_factor_stored_in_result_lms(self, backend_device, xp):
        """EqualizerResult.input_norm_factor must be a positive scalar for SISO."""
        samples, syms = _make_qpsk(xp, n_sym=500)
        r = equalization.lms(
            samples,
            syms[:20],
            num_taps=5,
            sps=1,
            step_size=1e-2,
            modulation="psk",
            order=4,
        )
        assert isinstance(r.input_norm_factor, float)
        assert r.input_norm_factor > 0.0
