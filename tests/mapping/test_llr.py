"""Tests for soft-decision (LLR) demapping."""

from typing import Any

import numpy as np
import pytest

from commkit import mapping
from commkit.core import Signal


class TestComputeLLRCore:
    """Tests for algorithmic accuracy, exact vs max-log approximations, and MIMO support."""

    def test_compute_llr_sign_correctness(self, xp: Any, xpt: Any) -> None:
        """LLR sign should match hard decision at high SNR (noiseless symbols)."""
        modulation = "qam"
        order = 16

        bits = xp.array(
            [0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0], dtype=xp.int32
        )
        symbols = mapping.map_bits(bits, modulation, order)

        llrs = mapping.compute_llr(
            symbols, modulation, order, noise_var=1e-6, method="maxlog", output="numpy"
        )
        hard_from_llr = (np.asarray(llrs) < 0).astype("int32")
        xpt.assert_array_equal(xp.asarray(hard_from_llr), bits)

    def test_compute_llr_roundtrip(self, xp: Any, xpt: Any) -> None:
        """Hard decision from LLR matches direct hard demapping at high SNR."""
        modulation = "psk"
        order = 8

        bits = xp.array([0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0], dtype=xp.int32)
        symbols = mapping.map_bits(bits, modulation, order)

        llrs = mapping.compute_llr(
            symbols, modulation, order, noise_var=1e-6, output="numpy"
        )
        hard_from_llr = (np.asarray(llrs) < 0).astype("int32")
        hard_direct = mapping.demap_symbols_hard(symbols, modulation, order)

        xpt.assert_array_equal(xp.asarray(hard_from_llr), hard_direct)

    def test_compute_llr_exact_vs_maxlog(self, xp: Any) -> None:
        """Exact and max-log methods agree on sign and remain close in magnitude."""
        modulation = "qam"
        order = 4

        bits = xp.array([0, 0, 0, 1, 1, 0, 1, 1], dtype=xp.int32)
        symbols = mapping.map_bits(bits, modulation, order)

        llrs_maxlog = np.asarray(
            mapping.compute_llr(
                symbols,
                modulation,
                order,
                noise_var=0.01,
                method="maxlog",
                output="numpy",
            )
        )
        llrs_exact = np.asarray(
            mapping.compute_llr(
                symbols,
                modulation,
                order,
                noise_var=0.01,
                method="exact",
                output="numpy",
            )
        )

        assert np.array_equal(np.sign(llrs_maxlog), np.sign(llrs_exact))
        ratio = np.abs(llrs_maxlog) / (np.abs(llrs_exact) + 1e-10)
        np.testing.assert_array_equal((ratio > 0.5) & (ratio < 2.0), True)

    def test_compute_llr_mimo_shape(self, xp: Any) -> None:
        """compute_llr preserves multi-channel MIMO structure."""
        modulation = "qam"
        order = 4

        bits = xp.zeros(16, dtype=xp.int32)
        symbols = mapping.map_bits(bits, modulation, order).reshape(2, 4)

        llrs = mapping.compute_llr(symbols, modulation, order, noise_var=0.1)
        assert llrs.shape == (2, 8)

    def test_compute_llr_methods_agree(self, xp: Any) -> None:
        """Verify maxlog and exact methods agree on sign for 16-QAM."""
        modulation = "qam"
        order = 16
        bits = xp.array([0, 0, 1, 1, 0, 1, 0, 1], dtype=xp.int32)
        symbols = mapping.map_bits(bits, modulation, order)
        noise_var = 0.1

        llrs_maxlog = np.asarray(
            mapping.compute_llr(
                symbols, modulation, order, noise_var, method="maxlog", output="numpy"
            )
        )
        llrs_exact = np.asarray(
            mapping.compute_llr(
                symbols, modulation, order, noise_var, method="exact", output="numpy"
            )
        )
        assert np.array_equal(np.sign(llrs_exact), np.sign(llrs_maxlog))

    def test_compute_llr_real_jax_symbols(self) -> None:
        """compute_llr with real-valued JAX symbols (PAM) casts constellation to float32."""
        jax = pytest.importorskip("jax")
        import jax.numpy as jnp

        bits = np.array([0, 0, 0, 1, 1, 0, 1, 1], dtype="int32")
        symbols_np = mapping.map_bits(bits, "pam", 4)
        symbols_jax = jnp.asarray(symbols_np)
        assert not jnp.iscomplexobj(symbols_jax)

        llrs = mapping.compute_llr(symbols_jax, "pam", 4, noise_var=0.1)
        assert isinstance(llrs, jax.Array)
        assert llrs.shape == (len(bits),)

    def test_compute_llr_validation(self, xp: Any) -> None:
        """Invalid modulation order or method raises ValueError."""
        with pytest.raises(ValueError, match="Order must be a power of 2"):
            mapping.compute_llr(xp.ones(1), "qam", 6, 0.1)

        with pytest.raises(ValueError, match="Unknown method"):
            mapping.compute_llr(xp.ones(1), "qam", 4, 0.1, method="magic")


class TestComputeLLRDifferentiability:
    """Tests for autodiff gradients and JAX evaluation consistency."""

    def test_compute_llr_gradient(self) -> None:
        """jax.grad through LLRs w.r.t. input symbols produces finite gradients."""
        jax = pytest.importorskip("jax")
        import jax.numpy as jnp

        symbols_jax = jnp.array([0.7 + 0.7j, -0.7 - 0.7j], dtype="complex64")

        def loss_fn(syms: Any) -> Any:
            llrs = mapping.compute_llr(syms, "qam", 4, 0.1, method="maxlog")
            return jnp.sum(llrs**2)

        grad = jax.grad(loss_fn)(symbols_jax)
        assert grad.shape == symbols_jax.shape
        assert jnp.all(jnp.isfinite(grad))
        assert not jnp.all(grad == 0)

    def test_compute_llr_numpy_vs_jax_input_agree(self, xpt: Any) -> None:
        """NumPy and JAX inputs produce numerically identical LLRs."""
        pytest.importorskip("jax")
        import jax.numpy as jnp

        bits = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1], dtype="int32")
        symbols_np = mapping.map_bits(bits, "psk", 8)

        for method in ("maxlog", "exact"):
            llrs_from_np = mapping.compute_llr(
                symbols_np, "psk", 8, 0.05, method=method
            )
            llrs_from_jax = mapping.compute_llr(
                jnp.asarray(symbols_np), "psk", 8, 0.05, method=method
            )
            xpt.assert_allclose(
                np.asarray(llrs_from_np), np.asarray(llrs_from_jax), atol=1e-5
            )


class TestComputeLLROutputModes:
    """Tests for output argument handling: 'jax', 'numpy', 'input'."""

    def test_compute_llr_output_jax_is_default(self) -> None:
        """output='jax' (default) returns jax.Array."""
        jax = pytest.importorskip("jax")
        bits = np.array([0, 0, 1, 1, 0, 1, 0, 1], dtype="int32")
        symbols = mapping.map_bits(bits, "qam", 16)
        llrs = mapping.compute_llr(symbols, "qam", 16, noise_var=0.1, output="jax")
        assert isinstance(llrs, jax.Array)

    def test_compute_llr_output_numpy_returns_numpy(self) -> None:
        """output='numpy' returns numpy.ndarray regardless of input backend."""
        pytest.importorskip("jax")
        bits = np.array([0, 0, 1, 1, 0, 1, 0, 1], dtype="int32")
        symbols = mapping.map_bits(bits, "qam", 16)
        llrs = mapping.compute_llr(symbols, "qam", 16, noise_var=0.1, output="numpy")
        assert isinstance(llrs, np.ndarray)

    def test_compute_llr_output_numpy_from_jax_input(self) -> None:
        """output='numpy' from JAX input returns numpy.ndarray."""
        pytest.importorskip("jax")
        import jax.numpy as jnp

        bits = np.array([0, 1, 0, 1], dtype="int32")
        symbols_jax = jnp.asarray(mapping.map_bits(bits, "qam", 4))
        llrs = mapping.compute_llr(symbols_jax, "qam", 4, noise_var=0.1, output="numpy")
        assert isinstance(llrs, np.ndarray)

    def test_compute_llr_output_input_preserves_type(self) -> None:
        """output='input' returns same container type as input."""
        jax = pytest.importorskip("jax")
        import jax.numpy as jnp

        # NumPy in -> NumPy out
        bits = np.array([0, 0, 1, 1, 0, 1, 0, 1], dtype="int32")
        syms_np = mapping.map_bits(bits, "qam", 16)
        out_np = mapping.compute_llr(syms_np, "qam", 16, noise_var=0.1, output="input")
        assert isinstance(out_np, np.ndarray)

        # JAX in -> JAX out
        syms_jax = jnp.asarray(syms_np)
        out_jax = mapping.compute_llr(
            syms_jax, "qam", 16, noise_var=0.1, output="input"
        )
        assert isinstance(out_jax, jax.Array)

    def test_compute_llr_output_invalid_raises(self) -> None:
        """Invalid output destination raises ValueError."""
        pytest.importorskip("jax")
        bits = np.array([0, 1], dtype="int32")
        symbols = mapping.map_bits(bits, "qam", 4)
        with pytest.raises(ValueError, match="output"):
            mapping.compute_llr(symbols, "qam", 4, noise_var=0.1, output="cuda")

    def test_compute_llr_output_numpy_values_match_jax(self, xpt: Any) -> None:
        """output='numpy' produces identical values to output='jax'."""
        pytest.importorskip("jax")
        bits = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0], dtype="int32")
        symbols = mapping.map_bits(bits, "qam", 16)
        llrs_jax = mapping.compute_llr(symbols, "qam", 16, noise_var=0.1, output="jax")
        llrs_np = mapping.compute_llr(symbols, "qam", 16, noise_var=0.1, output="numpy")
        xpt.assert_allclose(np.asarray(llrs_jax), llrs_np, atol=1e-6)


class TestComputeLLRSignalIntegration:
    """Tests for compute_llr when supplied with a Signal object."""

    def test_compute_llr_signal_input_uses_metadata(self, xpt: Any) -> None:
        """Signal input resolves symbols and modulation parameters from metadata."""
        pytest.importorskip("jax")
        bits = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0], dtype="int32")
        symbols = mapping.map_bits(bits, "qam", 16)
        sig = Signal(
            samples=symbols,
            sampling_rate=1.0,
            symbol_rate=1.0,
            mod_scheme="qam",
            mod_order=16,
        )
        sig.resolved_symbols = symbols

        llrs_sig = mapping.compute_llr(sig, noise_var=1e-6, output="numpy")
        llrs_arr = mapping.compute_llr(
            symbols, "qam", 16, noise_var=1e-6, output="numpy"
        )
        xpt.assert_allclose(llrs_sig, llrs_arr)

    def test_compute_llr_signal_input_raises_without_resolved(self) -> None:
        """Signal input missing resolved_symbols raises ValueError."""
        pytest.importorskip("jax")
        bits = np.array([0, 1, 0, 1], dtype="int32")
        symbols = mapping.map_bits(bits, "qam", 4)
        sig = Signal(samples=symbols, sampling_rate=1.0, symbol_rate=1.0)

        with pytest.raises(ValueError, match="resolved symbols"):
            mapping.compute_llr(sig, noise_var=0.1)
