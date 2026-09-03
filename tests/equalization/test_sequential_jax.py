"""Sequential equalizers on the JAX backend (incl. import-error branches)."""

import sys
from unittest.mock import patch

import pytest

from commkit import equalization, generate_psk, generate_qam
from commkit.equalization import EqualizerResult


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


jax = pytest.importorskip("jax", reason="JAX not installed")


class TestJAXBackend:
    """Tests for the JAX (lax.scan) backend on all adaptive algorithms.

    These tests cover the JAX kernel factories (_get_jax_lms, _get_jax_rls,
    _get_jax_cma, _get_jax_rde) and all JAX branches in the public API
    functions (lms, rls, cma, rde with backend='jax').
    """

    def _make_qpsk_rx(self, xp, n_symbols=1000, seed=0):
        """Helper: generate RRC-shaped QPSK at 2 SPS through mild ISI."""

        channel = xp.array([0.1, 1.0, 0.15], dtype=xp.complex64)
        sig = generate_psk(
            symbol_rate=1e6,
            num_symbols=n_symbols,
            order=4,
            pulse_shape="rrc",
            sps=2,
            seed=seed,
        )
        rx = xp.convolve(xp.asarray(sig.samples), channel, mode="same")
        return xp.ascontiguousarray(rx), sig

    def _make_qam16_rx(self, xp, n_symbols=2000, seed=0):
        """Helper: generate RRC-shaped 16-QAM at 2 SPS through mild ISI."""

        channel = xp.array([0.1, 1.0, 0.15], dtype=xp.complex64)
        sig = generate_qam(
            symbol_rate=1e6,
            num_symbols=n_symbols,
            order=16,
            pulse_shape="rrc",
            sps=2,
            seed=seed,
        )
        rx = xp.convolve(xp.asarray(sig.samples), channel, mode="same")
        return xp.ascontiguousarray(rx), sig

    # ---- LMS JAX ----

    def test_lms_jax_siso_convergence(self, backend_device, xp):
        """LMS JAX backend should converge on QPSK and return EqualizerResult."""
        rx, sig = self._make_qpsk_rx(xp)
        train = xp.asarray(sig.source_symbols)

        result = equalization.lms(
            rx,
            training_symbols=train,
            num_taps=11,
            step_size=0.05,
            modulation="psk",
            order=4,
            backend="jax",
        )

        assert isinstance(result, EqualizerResult)
        assert result.y_hat.ndim == 1
        assert result.weights.shape == (11,)
        assert result.weights_history is None

    def test_lms_jax_store_weights(self, backend_device, xp):
        """LMS JAX backend should populate weights_history when requested."""
        rx, sig = self._make_qpsk_rx(xp, n_symbols=500)
        train = xp.asarray(sig.source_symbols)
        n_sym = rx.shape[0] // 2

        result = equalization.lms(
            rx,
            training_symbols=train,
            num_taps=7,
            step_size=0.05,
            modulation="psk",
            order=4,
            backend="jax",
            store_weights=True,
        )

        assert result.weights_history is not None
        assert result.weights_history.shape == (n_sym, 7)

    def test_lms_jax_constellation_from_training(self, backend_device, xp):
        """LMS JAX backend should infer constellation from training when no modulation given."""
        rx, sig = self._make_qpsk_rx(xp, n_symbols=500)
        train = xp.asarray(sig.source_symbols)

        # Pass training only, no modulation/order - constellation inferred from train
        result = equalization.lms(
            rx,
            training_symbols=train,
            num_taps=7,
            step_size=0.05,
            backend="jax",
            # intentionally omit modulation/order
        )

        assert isinstance(result, EqualizerResult)
        assert result.y_hat.shape[0] > 0

    def test_lms_jax_num_train_symbols(self, backend_device, xp):
        """LMS JAX: pre-sliced training_symbols limits DA phase length."""
        rx, sig = self._make_qpsk_rx(xp, n_symbols=1000)
        train = xp.asarray(sig.source_symbols[:50])

        result = equalization.lms(
            rx,
            training_symbols=train,
            num_taps=11,
            step_size=0.05,
            modulation="psk",
            order=4,
            backend="jax",
        )

        assert isinstance(result, EqualizerResult)
        assert result.num_train_symbols <= 50

    def test_lms_jax_mimo(self, backend_device, xp):
        """LMS JAX butterfly should handle 2x2 MIMO input."""

        n_symbols = 1000
        sig = generate_psk(
            symbol_rate=1e6,
            num_symbols=n_symbols,
            order=4,
            pulse_shape="rrc",
            sps=2,
            seed=3,
        )
        rx1 = xp.asarray(sig.samples)
        rx2 = xp.roll(rx1, 1)
        rx_mimo = xp.stack([rx1, rx2], axis=0)  # (2, N)
        train = xp.stack([xp.asarray(sig.source_symbols)] * 2, axis=0)

        result = equalization.lms(
            rx_mimo,
            training_symbols=train,
            num_taps=7,
            step_size=0.05,
            modulation="psk",
            order=4,
            backend="jax",
        )

        assert result.y_hat.shape == (2, n_symbols)
        assert result.weights.shape == (2, 2, 7)

    # ---- RLS JAX ----

    def test_rls_jax_siso_convergence(self, backend_device, xp):
        """RLS JAX backend should converge and return correct shapes."""
        rx, sig = self._make_qpsk_rx(xp)
        train = xp.asarray(sig.source_symbols)

        result = equalization.rls(
            rx,
            training_symbols=train,
            sps=1,
            num_taps=7,
            modulation="psk",
            order=4,
            backend="jax",
        )

        assert isinstance(result, EqualizerResult)
        assert result.y_hat.ndim == 1
        assert result.weights.shape == (7,)

    def test_rls_jax_with_leakage(self, backend_device, xp):
        """RLS JAX backend should run without error with leakage > 0."""
        rx, sig = self._make_qpsk_rx(xp)
        train = xp.asarray(sig.source_symbols)

        result = equalization.rls(
            rx,
            training_symbols=train,
            sps=1,
            num_taps=7,
            modulation="psk",
            order=4,
            backend="jax",
            leakage=1e-4,
        )

        assert isinstance(result, EqualizerResult)
        assert result.y_hat.shape[0] > 0

    def test_rls_jax_store_weights(self, backend_device, xp):
        """RLS JAX backend should store weight trajectory when requested."""
        rx, sig = self._make_qpsk_rx(xp, n_symbols=500)
        train = xp.asarray(sig.source_symbols)

        result = equalization.rls(
            rx,
            training_symbols=train,
            sps=1,
            num_taps=7,
            modulation="psk",
            order=4,
            backend="jax",
            store_weights=True,
        )

        assert result.weights_history is not None

    # ---- CMA JAX ----

    def test_cma_jax_siso_convergence(self, backend_device, xp):
        """CMA JAX backend should converge for QPSK."""
        rx, _ = self._make_qpsk_rx(xp)

        result = equalization.cma(
            rx,
            num_taps=11,
            step_size=0.005,
            modulation="psk",
            order=4,
            backend="jax",
        )

        assert isinstance(result, EqualizerResult)
        assert result.y_hat.ndim == 1
        assert result.weights.shape == (11,)

    def test_cma_jax_store_weights(self, backend_device, xp):
        """CMA JAX backend should populate weights_history when requested."""
        rx, _ = self._make_qpsk_rx(xp, n_symbols=400)
        n_sym = rx.shape[0] // 2

        result = equalization.cma(
            rx,
            num_taps=7,
            step_size=0.005,
            modulation="psk",
            order=4,
            backend="jax",
            store_weights=True,
        )

        assert result.weights_history is not None
        assert result.weights_history.shape == (n_sym, 7)

    def test_cma_jax_default_r2(self, backend_device, xp):
        """CMA JAX backend should work with default R²=1.0 (no modulation given)."""
        rx, _ = self._make_qpsk_rx(xp, n_symbols=400)

        result = equalization.cma(rx, num_taps=7, step_size=0.005, backend="jax")

        assert isinstance(result, EqualizerResult)
        assert result.y_hat.shape[0] > 0

    # ---- RDE JAX ----

    def test_rde_jax_siso_convergence(self, backend_device, xp):
        """RDE JAX backend should converge on QPSK."""
        rx, _ = self._make_qpsk_rx(xp)

        result = equalization.rde(
            rx,
            num_taps=11,
            step_size=0.005,
            modulation="psk",
            order=4,
            backend="jax",
        )

        assert isinstance(result, EqualizerResult)
        assert result.y_hat.ndim == 1
        assert result.weights.shape == (11,)

    def test_rde_jax_16qam(self, backend_device, xp):
        """RDE JAX backend should converge on 16-QAM."""
        rx, _ = self._make_qam16_rx(xp)

        result = equalization.rde(
            rx,
            num_taps=11,
            step_size=5e-4,
            modulation="qam",
            order=16,
            backend="jax",
        )

        assert isinstance(result, EqualizerResult)
        assert result.y_hat.ndim == 1

    def test_rde_jax_store_weights(self, backend_device, xp):
        """RDE JAX backend should populate weights_history when requested."""
        rx, _ = self._make_qpsk_rx(xp, n_symbols=400)
        n_sym = rx.shape[0] // 2

        result = equalization.rde(
            rx,
            num_taps=7,
            step_size=0.005,
            modulation="psk",
            order=4,
            backend="jax",
            store_weights=True,
        )

        assert result.weights_history is not None
        assert result.weights_history.shape == (n_sym, 7)

    def test_rde_jax_no_modulation_unit_radius(self, backend_device, xp):
        """RDE JAX backend with no modulation should fall back to unit radius (≡ CMA)."""
        rx, _ = self._make_qpsk_rx(xp, n_symbols=400)

        result = equalization.rde(rx, num_taps=7, step_size=0.005, backend="jax")

        assert isinstance(result, EqualizerResult)
        assert result.y_hat.shape[0] > 0

    def test_rde_jax_mimo(self, backend_device, xp):
        """RDE JAX butterfly should handle 2x2 MIMO input."""

        n_symbols = 1000
        sig = generate_qam(
            symbol_rate=1e6,
            num_symbols=n_symbols,
            order=16,
            pulse_shape="rrc",
            sps=2,
            seed=5,
        )
        rx1 = xp.asarray(sig.samples)
        rx_mimo = xp.stack([rx1, xp.roll(rx1, 2)], axis=0)

        result = equalization.rde(
            rx_mimo,
            num_taps=7,
            step_size=5e-4,
            modulation="qam",
            order=16,
            backend="jax",
        )

        assert result.y_hat.shape == (2, n_symbols)
        assert result.weights.shape == (2, 2, 7)

    def test_kernel_cache_reuse(self, backend_device, xp):
        """Calling the same JAX algorithm twice with identical parameters must reuse the cache."""
        from commkit.equalization import _JITTED_EQ

        rx, sig = self._make_qpsk_rx(xp, n_symbols=300)
        train = xp.asarray(sig.source_symbols)

        equalization.lms(
            rx,
            training_symbols=train,
            num_taps=5,
            modulation="psk",
            order=4,
            backend="jax",
        )
        key_after_first = set(_JITTED_EQ.keys())
        equalization.lms(
            rx,
            training_symbols=train,
            num_taps=5,
            modulation="psk",
            order=4,
            backend="jax",
        )
        key_after_second = set(_JITTED_EQ.keys())

        # No new keys should be added on the second identical call
        new_keys = key_after_second - key_after_first
        assert len(new_keys) == 0, f"Unexpected new cache entries: {new_keys}"


@pytest.mark.skipif(
    "jax" not in sys.modules and not pytest.importorskip("jax", reason="skip"),
    reason="JAX required",
)
class TestRLSJAXConstellationFromTraining:
    """RLS JAX derives constellation from training symbols when no modulation is given."""

    def test_rls_jax_constellation_from_training(self, backend_device, xp):
        """RLS JAX with training only (no modulation) infers constellation from training."""
        pytest.importorskip("jax")

        n_symbols = 800
        sig = generate_psk(
            symbol_rate=1e6,
            num_symbols=n_symbols,
            order=4,
            pulse_shape="rrc",
            sps=2,
            seed=9,
        )
        channel = xp.array([0.1, 1.0, 0.15], dtype=xp.complex64)
        rx = xp.convolve(xp.asarray(sig.samples), channel, mode="same")
        rx = xp.ascontiguousarray(rx)
        train = xp.asarray(sig.source_symbols)

        result = equalization.rls(
            rx,
            training_symbols=train,
            sps=2,
            num_taps=7,
            backend="jax",
            # No modulation/order -> constellation from training
        )

        assert isinstance(result, EqualizerResult)
        assert result.y_hat.ndim == 1


class TestImportErrorBranches:
    """Tests for ImportError branches when Numba/JAX are unavailable (uses mocking)."""

    def _make_rx(self, xp, n_symbols=400):

        sig = generate_psk(
            symbol_rate=1e6,
            num_symbols=n_symbols,
            order=4,
            pulse_shape="rrc",
            sps=2,
            seed=0,
        )
        return xp.ascontiguousarray(xp.asarray(sig.samples)), sig

    def test_lms_jax_not_installed(self, backend_device, xp):
        """LMS raises ImportError when backend='jax' but JAX is not available."""
        rx, _ = self._make_rx(xp)
        with patch(
            "commkit.equalization.sequential._dd._get_jax",
            return_value=(None, None, None),
        ):
            with pytest.raises(ImportError, match="JAX is required"):
                equalization.lms(rx, modulation="psk", order=4, backend="jax")

    def test_rls_numba_not_installed(self, backend_device, xp):
        """RLS raises ImportError when backend='numba' but Numba is not available."""
        rx, sig = self._make_rx(xp)
        train = xp.asarray(sig.source_symbols)
        with patch("commkit.equalization.sequential._dd._get_numba", return_value=None):
            with pytest.raises(ImportError, match="Numba is required"):
                equalization.rls(
                    rx,
                    training_symbols=train,
                    sps=2,
                    modulation="psk",
                    order=4,
                    backend="numba",
                )

    def test_rls_jax_not_installed(self, backend_device, xp):
        """RLS raises ImportError when backend='jax' but JAX is not available."""
        rx, sig = self._make_rx(xp)
        train = xp.asarray(sig.source_symbols)
        with patch(
            "commkit.equalization.sequential._dd._get_jax",
            return_value=(None, None, None),
        ):
            with pytest.raises(ImportError, match="JAX is required"):
                equalization.rls(
                    rx,
                    training_symbols=train,
                    sps=2,
                    modulation="psk",
                    order=4,
                    backend="jax",
                )

    def test_cma_numba_not_installed(self, backend_device, xp):
        """CMA raises ImportError when backend='numba' but Numba is not available."""
        rx, _ = self._make_rx(xp)
        with patch(
            "commkit.equalization.sequential._blind._get_numba", return_value=None
        ):
            with pytest.raises(ImportError, match="Numba is required"):
                equalization.cma(rx, modulation="psk", order=4, backend="numba")

    def test_cma_jax_not_installed(self, backend_device, xp):
        """CMA raises ImportError when backend='jax' but JAX is not available."""
        rx, _ = self._make_rx(xp)
        with patch(
            "commkit.equalization.sequential._blind._get_jax",
            return_value=(None, None, None),
        ):
            with pytest.raises(ImportError, match="JAX is required"):
                equalization.cma(rx, modulation="psk", order=4, backend="jax")

    def test_rde_numba_not_installed(self, backend_device, xp):
        """RDE raises ImportError when backend='numba' but Numba is not available."""
        rx, _ = self._make_rx(xp)
        with patch(
            "commkit.equalization.sequential._blind._get_numba", return_value=None
        ):
            with pytest.raises(ImportError, match="Numba is required"):
                equalization.rde(rx, modulation="qam", order=16, backend="numba")

    def test_rde_jax_not_installed(self, backend_device, xp):
        """RDE raises ImportError when backend='jax' but JAX is not available."""
        rx, _ = self._make_rx(xp)
        with patch(
            "commkit.equalization.sequential._blind._get_jax",
            return_value=(None, None, None),
        ):
            with pytest.raises(ImportError, match="JAX is required"):
                equalization.rde(rx, modulation="qam", order=16, backend="jax")


class TestLMSJAXPureDD:
    """LMS JAX backend without training symbols runs pure decision-directed mode."""

    def test_lms_jax_pure_dd_no_training(self, backend_device, xp):
        """LMS JAX with modulation but no training_symbols runs in pure decision-directed mode from the start."""
        pytest.importorskip("jax")

        sig = generate_psk(
            symbol_rate=1e6, num_symbols=800, order=4, pulse_shape="rrc", sps=2, seed=42
        )
        rx = xp.ascontiguousarray(xp.asarray(sig.samples))

        # No training_symbols -> _prepare_training_jax receives None -> else branch
        result = equalization.lms(
            rx,
            num_taps=11,
            step_size=0.01,
            modulation="psk",
            order=4,
            backend="jax",
            # deliberately no training_symbols
        )

        assert isinstance(result, EqualizerResult)
        assert result.y_hat.ndim == 1
