"""Tests for soft-decision (LLR) demapping."""

import numpy as np
import pytest

from commkit import mapping
from commkit.core import Signal


def test_compute_llr_sign_correctness(xp, xpt):
    """LLR sign should match hard decision at high SNR (noiseless symbols)."""
    modulation = "qam"
    order = 16

    bits = xp.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0], dtype="int32")
    symbols = mapping.map_bits(bits, modulation, order)

    llrs = mapping.compute_llr(
        symbols, modulation, order, noise_var=1e-6, method="maxlog"
    )

    hard_from_llr = (np.asarray(llrs) < 0).astype("int32")
    xpt.assert_array_equal(hard_from_llr, bits)


def test_compute_llr_roundtrip(xp, xpt):
    """Hard decision from LLR should match direct hard demapping at high SNR."""
    modulation = "psk"
    order = 8

    bits = xp.array([0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0], dtype="int32")
    symbols = mapping.map_bits(bits, modulation, order)

    llrs = mapping.compute_llr(symbols, modulation, order, noise_var=1e-6)
    hard_from_llr = (np.asarray(llrs) < 0).astype("int32")
    hard_direct = mapping.demap_symbols_hard(symbols, modulation, order)

    xpt.assert_array_equal(hard_from_llr, hard_direct)


def test_compute_llr_exact_vs_maxlog(xp):
    """Exact and max-log methods should agree on sign and be close in magnitude."""
    modulation = "qam"
    order = 4

    bits = xp.array([0, 0, 0, 1, 1, 0, 1, 1], dtype="int32")
    symbols = mapping.map_bits(bits, modulation, order)

    llrs_maxlog = np.asarray(
        mapping.compute_llr(symbols, modulation, order, noise_var=0.01, method="maxlog")
    )
    llrs_exact = np.asarray(
        mapping.compute_llr(symbols, modulation, order, noise_var=0.01, method="exact")
    )

    assert np.array_equal(np.sign(llrs_maxlog), np.sign(llrs_exact))

    # Max-log is an approximation; values should be within a reasonable ratio
    ratio = np.abs(llrs_maxlog) / (np.abs(llrs_exact) + 1e-10)
    assert np.all(ratio > 0.5) and np.all(ratio < 2.0)


def test_compute_llr_mimo_shape(xp):
    """compute_llr should preserve MIMO channel structure."""
    modulation = "qam"
    order = 4

    bits = xp.zeros(16, dtype="int32")
    symbols = mapping.map_bits(bits, modulation, order).reshape(2, 4)

    llrs = mapping.compute_llr(symbols, modulation, order, noise_var=0.1)

    assert llrs.shape == (2, 8)


def test_compute_llr_methods_agree(xp):
    """Verify maxlog and exact methods agree on sign for 16-QAM."""
    modulation = "qam"
    order = 16
    bits = xp.array([0, 0, 1, 1, 0, 1, 0, 1], dtype="int32")
    symbols = mapping.map_bits(bits, modulation, order)
    noise_var = 0.1

    llrs_maxlog = np.asarray(
        mapping.compute_llr(symbols, modulation, order, noise_var, method="maxlog")
    )
    llrs_exact = np.asarray(
        mapping.compute_llr(symbols, modulation, order, noise_var, method="exact")
    )

    assert np.array_equal(np.sign(llrs_exact), np.sign(llrs_maxlog))

    with pytest.raises(ValueError, match="Unknown method"):
        mapping.compute_llr(symbols, modulation, order, noise_var, method="unknown")


def test_compute_llr_invalid_order(xp):
    """Verify error for non-power-of-2 order."""
    with pytest.raises(ValueError, match="Order must be a power of 2"):
        mapping.compute_llr(xp.ones(1), "qam", 6, 0.1)


def test_compute_llr_invalid_method(xp):
    """Verify error for unknown method."""
    with pytest.raises(ValueError, match="Unknown method"):
        mapping.compute_llr(xp.ones(1), "qam", 4, 0.1, method="magic")


# === compute_llr always-JAX output and differentiability tests ===


def test_compute_llr_numpy_input_returns_jax():
    """NumPy input should return a JAX array."""
    jax = pytest.importorskip("jax")

    bits = np.array([0, 0, 1, 1, 0, 1, 0, 1], dtype="int32")
    symbols = mapping.map_bits(bits, "qam", 16)

    llrs = mapping.compute_llr(symbols, "qam", 16, noise_var=0.1)

    assert isinstance(llrs, jax.Array), (
        "compute_llr must return jax.Array for NumPy input"
    )


def test_compute_llr_cupy_input_returns_jax():
    """CuPy input should return a JAX array."""
    jax = pytest.importorskip("jax")
    cp = pytest.importorskip("cupy")

    bits = cp.array([0, 0, 1, 1, 0, 1, 0, 1], dtype="int32")
    symbols = mapping.map_bits(bits, "qam", 16)

    llrs = mapping.compute_llr(symbols, "qam", 16, noise_var=0.1)

    assert isinstance(llrs, jax.Array), (
        "compute_llr must return jax.Array for CuPy input"
    )


def test_compute_llr_jax_input_returns_jax():
    """JAX array input should return a JAX array (maxlog)."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    bits_np = np.array([0, 0, 1, 1, 0, 1, 0, 1], dtype="int32")
    symbols_jax = jnp.asarray(mapping.map_bits(bits_np, "qam", 16))

    llrs = mapping.compute_llr(symbols_jax, "qam", 16, noise_var=0.1, method="maxlog")

    assert isinstance(llrs, jax.Array)
    hard = (np.asarray(llrs) < 0).astype("int32")
    assert np.array_equal(hard, bits_np)


def test_compute_llr_jax_exact_returns_jax():
    """JAX array input should return a JAX array (exact)."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    bits_np = np.array([0, 1, 0, 1, 1, 0, 1, 0], dtype="int32")
    symbols_jax = jnp.asarray(mapping.map_bits(bits_np, "qam", 16))

    llrs = mapping.compute_llr(symbols_jax, "qam", 16, noise_var=0.01, method="exact")

    assert isinstance(llrs, jax.Array)
    hard = (np.asarray(llrs) < 0).astype("int32")
    assert np.array_equal(hard, bits_np)


def test_compute_llr_gradient():
    """jax.grad through LLRs w.r.t. input symbols should produce finite gradients."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    symbols_jax = jnp.array([0.7 + 0.7j, -0.7 - 0.7j], dtype="complex64")

    def loss_fn(syms):
        llrs = mapping.compute_llr(syms, "qam", 4, 0.1, method="maxlog")
        return jnp.sum(llrs**2)

    grad = jax.grad(loss_fn)(symbols_jax)

    assert grad.shape == symbols_jax.shape
    assert jnp.all(jnp.isfinite(grad))
    assert not jnp.all(grad == 0)


def test_compute_llr_numpy_vs_jax_input_agree():
    """NumPy and JAX inputs should produce numerically identical LLRs."""
    pytest.importorskip("jax")
    import jax.numpy as jnp

    bits = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1], dtype="int32")
    symbols_np = mapping.map_bits(bits, "psk", 8)

    for method in ("maxlog", "exact"):
        llrs_from_np = mapping.compute_llr(symbols_np, "psk", 8, 0.05, method=method)
        llrs_from_jax = mapping.compute_llr(
            jnp.asarray(symbols_np), "psk", 8, 0.05, method=method
        )
        np.testing.assert_allclose(
            np.asarray(llrs_from_np), np.asarray(llrs_from_jax), atol=1e-5
        )


def test_compute_llr_real_jax_symbols():
    """compute_llr with real-valued JAX symbols (PAM) should cast the constellation to float32."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    # PAM-4 uses real-valued symbols
    bits = np.array([0, 0, 0, 1, 1, 0, 1, 1], dtype="int32")
    symbols_np = mapping.map_bits(bits, "pam", 4)
    # PAM symbols are real; wrap in JAX array
    symbols_jax = jnp.asarray(symbols_np)
    assert not jnp.iscomplexobj(symbols_jax), "PAM symbols should be real-valued"

    llrs = mapping.compute_llr(symbols_jax, "pam", 4, noise_var=0.1)
    assert isinstance(llrs, jax.Array)
    assert llrs.shape == (len(bits),)


# === output parameter tests ===


def test_compute_llr_output_jax_is_default():
    """output='jax' (default) should still return jax.Array - backward compat."""
    jax = pytest.importorskip("jax")

    bits = np.array([0, 0, 1, 1, 0, 1, 0, 1], dtype="int32")
    symbols = mapping.map_bits(bits, "qam", 16)
    llrs = mapping.compute_llr(symbols, "qam", 16, noise_var=0.1, output="jax")
    assert isinstance(llrs, jax.Array)


def test_compute_llr_output_numpy_returns_numpy():
    """output='numpy' should return a plain numpy.ndarray regardless of input backend."""
    pytest.importorskip("jax")

    bits = np.array([0, 0, 1, 1, 0, 1, 0, 1], dtype="int32")
    symbols = mapping.map_bits(bits, "qam", 16)
    llrs = mapping.compute_llr(symbols, "qam", 16, noise_var=0.1, output="numpy")
    assert isinstance(llrs, np.ndarray)


def test_compute_llr_output_numpy_from_jax_input():
    """output='numpy' from a JAX input should still return numpy.ndarray."""
    pytest.importorskip("jax")
    import jax.numpy as jnp

    bits = np.array([0, 1, 0, 1], dtype="int32")
    symbols_jax = jnp.asarray(mapping.map_bits(bits, "qam", 4))
    llrs = mapping.compute_llr(symbols_jax, "qam", 4, noise_var=0.1, output="numpy")
    assert isinstance(llrs, np.ndarray)


def test_compute_llr_output_input_numpy_returns_numpy():
    """output='input' with NumPy input should return numpy.ndarray."""
    pytest.importorskip("jax")

    bits = np.array([0, 0, 1, 1, 0, 1, 0, 1], dtype="int32")
    symbols_np = mapping.map_bits(bits, "qam", 16)
    llrs = mapping.compute_llr(symbols_np, "qam", 16, noise_var=0.1, output="input")
    assert isinstance(llrs, np.ndarray)


def test_compute_llr_output_input_jax_returns_jax():
    """output='input' with JAX input should return jax.Array."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    bits = np.array([0, 1, 1, 0], dtype="int32")
    symbols_jax = jnp.asarray(mapping.map_bits(bits, "qam", 4))
    llrs = mapping.compute_llr(symbols_jax, "qam", 4, noise_var=0.1, output="input")
    assert isinstance(llrs, jax.Array)


def test_compute_llr_output_invalid_raises():
    """Unknown output value should raise ValueError."""
    pytest.importorskip("jax")

    bits = np.array([0, 1], dtype="int32")
    symbols = mapping.map_bits(bits, "qam", 4)
    with pytest.raises(ValueError, match="output"):
        mapping.compute_llr(symbols, "qam", 4, noise_var=0.1, output="cuda")


def test_compute_llr_output_numpy_values_match_jax():
    """output='numpy' should produce numerically identical values to output='jax'."""
    pytest.importorskip("jax")

    bits = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0], dtype="int32")
    symbols = mapping.map_bits(bits, "qam", 16)
    llrs_jax = mapping.compute_llr(symbols, "qam", 16, noise_var=0.1, output="jax")
    llrs_np = mapping.compute_llr(symbols, "qam", 16, noise_var=0.1, output="numpy")
    np.testing.assert_allclose(np.asarray(llrs_jax), llrs_np, atol=1e-6)


def test_compute_llr_signal_input_uses_metadata():
    """Signal input: resolved_symbols/modulation/order come from the signal."""
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
    llrs_arr = mapping.compute_llr(symbols, "qam", 16, noise_var=1e-6, output="numpy")

    np.testing.assert_allclose(llrs_sig, llrs_arr)


def test_compute_llr_signal_input_raises_without_resolved():
    """Signal input without resolved_symbols raises ValueError."""
    pytest.importorskip("jax")

    bits = np.array([0, 1, 0, 1], dtype="int32")
    symbols = mapping.map_bits(bits, "qam", 4)
    sig = Signal(samples=symbols, sampling_rate=1.0, symbol_rate=1.0)

    with pytest.raises(ValueError, match="resolved symbols"):
        mapping.compute_llr(sig, noise_var=0.1)
