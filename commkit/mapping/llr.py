"""
Soft-decision demapping (log-likelihood ratios).

JAX JIT-compiled max-log and exact LLR kernels, with optional probabilistic-
shaping prior.  Sign convention: positive LLR -> bit 0 more likely.
"""

from typing import Any

import numpy as np

from ..backend import ArrayType, _get_jax, dispatch, is_jax_array, to_jax
from ..core.signal import Signal
from ..helpers import unwrap_signal
from ..logger import logger
from .gray import gray_constellation

__all__ = ["compute_llr"]

# Lazy cache for JIT-compiled soft demapping kernels
_JITTED_SOFT_DEMAP: dict[str, Any] = {}


def _get_jitted_soft_demap():
    """
    Returns JIT-compiled maxlog and exact LLR computation functions.

    Functions are defined and compiled lazily on first call to avoid
    importing JAX at module load time.
    """
    if not _JITTED_SOFT_DEMAP:
        jax, jnp, _ = _get_jax()
        if jax is None:
            raise ImportError(
                "JAX is required for soft demapping. Install with: pip install jax"
            )

        @jax.jit
        def maxlog(symbols, constellation, bits_table_t, sigma_sq, log_pmf):
            """Max-log LLR with PS prior.

            symbols (N,), constellation (M,), bits_table_t (k, M), log_pmf (M,).
            Uniform case: pass log_pmf = jnp.zeros(M) - constant offset cancels.
            PS case: log_pmf = log P(sₘ).

            Effective metric: eff_m = d_m/σ² - log P(sₘ)
            LLR_k = min_{b=1} eff - min_{b=0} eff
            """
            distances_sq = (
                jnp.abs(symbols[:, None] - constellation[None, :]) ** 2
            )  # (N, M)
            eff = distances_sq / sigma_sq - log_pmf[None, :]  # (N, M)

            def bit_llr(bit_row):  # (M,)
                d0 = jnp.where(bit_row == 0, eff, jnp.inf)  # (N, M)
                d1 = jnp.where(bit_row == 1, eff, jnp.inf)  # (N, M)
                return jnp.min(d1, axis=1) - jnp.min(d0, axis=1)  # (N,)

            return jax.vmap(bit_llr)(bits_table_t).T  # (k, N) -> (N, k)

        @jax.jit
        def exact(symbols, constellation, bits_table_t, sigma_sq, log_pmf):
            """Exact LLR with PS prior via log-sum-exp.

            symbols (N,), constellation (M,), bits_table_t (k, M), log_pmf (M,).
            Uniform case: pass log_pmf = jnp.zeros(M).
            PS case: log_pmf = log P(sₘ).

            log_terms_m = log P(sₘ) - d_m/σ²
            LLR_k = LSE_{b=0}(log_terms) - LSE_{b=1}(log_terms)
            """
            distances_sq = (
                jnp.abs(symbols[:, None] - constellation[None, :]) ** 2
            )  # (N, M)
            log_terms = log_pmf[None, :] - distances_sq / sigma_sq  # (N, M)

            def bit_llr(bit_row):  # (M,)
                e0 = jnp.where(bit_row == 0, log_terms, -jnp.inf)  # (N, M)
                e1 = jnp.where(bit_row == 1, log_terms, -jnp.inf)  # (N, M)
                return jax.scipy.special.logsumexp(
                    e0, axis=1
                ) - jax.scipy.special.logsumexp(e1, axis=1)  # (N,)

            return jax.vmap(bit_llr)(bits_table_t).T  # (k, N) -> (N, k)

        _JITTED_SOFT_DEMAP["maxlog"] = maxlog
        _JITTED_SOFT_DEMAP["exact"] = exact

    return _JITTED_SOFT_DEMAP["maxlog"], _JITTED_SOFT_DEMAP["exact"]


def compute_llr(
    symbols: ArrayType | Signal,
    modulation: str | None = None,
    order: int | None = None,
    noise_var: float | None = None,
    method: str = "maxlog",
    unipolar: bool = False,
    output: str = "jax",
    pmf: np.ndarray | None = None,
) -> ArrayType:
    """
    Compute Log-Likelihood Ratios (LLRs) for soft-decision decoding.

    Positive LLR -> bit 0 more likely; negative -> bit 1; magnitude = confidence.
    JAX JIT-compiled with ``jax.vmap`` over bit positions; fully differentiable.

    Parameters
    ----------
    symbols : array_like or Signal
        Received noisy symbols. Shape: (..., N_symbols). NumPy, CuPy, or JAX.
        A :class:`Signal` supplies ``resolved_symbols`` and defaults
        ``modulation``/``order``/``pmf`` from its metadata when not given
        explicitly.
    modulation : {"psk", "qam", "ask"}, optional
        Modulation type.  Required for array input; for :class:`Signal`
        input, used only as a fallback when the signal's ``mod_scheme`` is
        unset.
    order : int, optional
        Modulation order.  Required for array input; for :class:`Signal`
        input, used only as a fallback when the signal's ``mod_order`` is
        unset.
    noise_var : float
        Complex noise variance sigma^2 referenced to the normalised
        constellation (unit avg power).  sigma^2 = 10^(-EsN0_dB / 10).
    method : {"maxlog", "exact"}, default "maxlog"
        LLR algorithm. ``"maxlog"`` is faster; ``"exact"`` uses log-sum-exp.
    unipolar : bool, default False
        Use unipolar constellation for ASK/PAM.
    output : {"jax", "input", "numpy"}, default "jax"
        Output array type.  ``"jax"`` preserves differentiability;
        ``"input"`` matches the input backend; ``"numpy"`` forces NumPy.
    pmf : np.ndarray, optional
        Symbol PMF of shape ``(order,)`` for PS-QAM.  Pass
        ``maxwell_boltzmann(order, nu)`` to incorporate the non-uniform prior.
        ``None`` assumes uniform prior.  For :class:`Signal` input, used
        only as a fallback when the signal's ``ps_pmf`` is unset.

    Returns
    -------
    array_like
        LLR values. Shape: (..., N_symbols * log2(order)).
        Array type determined by ``output``.

    Notes
    -----
    Max-Log: LLR_k ≈ (1/sigma^2) * (min_{S_1^k} |r-s|^2 - min_{S_0^k} |r-s|^2).
    Exact: LLR_k = log sum_{S_0^k} exp(-|r-s|^2/sigma^2) - log sum_{S_1^k} ...

    For PS-QAM, ``symbols`` must be on the same scale as
    ``gray_constellation`` (unit avg power).  After
    ``resolve_symbols`` the receiver renormalises;
    use ``gmi`` instead for correct scale.
    """
    x, sig = unwrap_signal(symbols, field="resolved_symbols")
    if sig is not None:
        if x is None:
            raise ValueError(
                "No resolved symbols available. Call resolve_symbols(sig) first."
            )
        mod = sig.mod_scheme
        if mod is None:
            mod = modulation
            if mod is not None:
                logger.warning(
                    "compute_llr(): Signal has no mod_scheme set; falling "
                    "back to supplied modulation=%r.",
                    mod,
                )
        ord_ = sig.mod_order
        if ord_ is None:
            ord_ = order
            if ord_ is not None:
                logger.warning(
                    "compute_llr(): Signal has no mod_order set; falling "
                    "back to supplied order=%r.",
                    ord_,
                )
        eff_pmf = sig.ps_pmf
        if eff_pmf is None:
            eff_pmf = pmf
            if eff_pmf is not None:
                logger.warning(
                    "compute_llr(): Signal has no ps_pmf set; falling back "
                    "to supplied pmf."
                )
        return compute_llr(
            x,
            mod,
            ord_,
            noise_var,
            method=method,
            unipolar=unipolar,
            output=output,
            pmf=eff_pmf,
        )

    if modulation is None or order is None:
        raise ValueError("compute_llr() requires modulation and order for array input.")
    if noise_var is None:
        raise ValueError("compute_llr() requires noise_var.")
    symbols = x

    logger.debug(
        "Computing LLRs for %s %s-level (method=%s, output=%s).",
        modulation.upper(),
        order,
        method,
        output,
    )

    k = int(np.log2(order))
    if 2**k != order:
        raise ValueError(f"Order must be a power of 2, got {order}")
    if method not in ("maxlog", "exact"):
        raise ValueError(f"Unknown method: {method}. Use 'maxlog' or 'exact'.")
    if output not in ("jax", "input", "numpy"):
        raise ValueError(f"Unknown output: {output!r}. Use 'jax', 'input', or 'numpy'.")

    # Convert to JAX if not already
    jax, jnp, _ = _get_jax()
    if jax is None:
        raise ImportError(
            "JAX is required for LLR computation. Install with: pip install jax"
        )

    # Build constellation and bits table on CPU once - shared by both paths.
    is_complex = symbols.dtype.kind == "c"
    const = gray_constellation(modulation, order, unipolar=unipolar).astype(
        "complex64" if is_complex else "float32"
    )
    bits_table_np = (
        (
            (
                np.arange(order, dtype="int32")[:, None]
                >> np.arange(k - 1, -1, -1, dtype="int32")
            )
            & 1
        )
        .astype("int32")
        .T
    )
    sigma_np = np.float32(max(noise_var, 1e-20))

    # Build log_pmf: zeros = uniform (constant offset cancels in LLR difference).
    if pmf is not None:
        log_pmf_np = np.log(np.clip(np.asarray(pmf, dtype=np.float32), 1e-40, None))
    else:
        log_pmf_np = np.zeros(order, dtype=np.float32)

    # JAX path
    jax_module, jnp, _ = _get_jax()
    if is_jax_array(symbols):
        assert jnp is not None
        if hasattr(symbols, "shape"):
            original_shape = symbols.shape
        else:
            # Fallback for JAX tracers or odd objects
            original_shape = jnp.shape(symbols)

        jax_symbols_flat = symbols.flatten()
        constellation_jax = jnp.asarray(const)
        bits_table_t_jax = jnp.asarray(bits_table_np)
        sigma_sq = jnp.asarray(sigma_np)
        log_pmf_jax = jnp.asarray(log_pmf_np)

    # NumPy/CuPy path
    else:
        symbols, xp, _ = dispatch(symbols)
        original_shape = symbols.shape

        jax_symbols_flat = to_jax(
            symbols, dtype="complex64" if is_complex else "float32"
        ).flatten()
        device = jax_symbols_flat.device

        assert jax_module is not None
        assert jnp is not None
        # device_put accepts NumPy arrays directly - no intermediate jnp.asarray needed
        constellation_jax = jax_module.device_put(const, device)
        bits_table_t_jax = jax_module.device_put(bits_table_np, device)
        sigma_sq = jax_module.device_put(jnp.asarray(sigma_np), device)
        log_pmf_jax = jax_module.device_put(log_pmf_np, device)

    # Compute LLRs via JIT-compiled kernels
    maxlog_fn, exact_fn = _get_jitted_soft_demap()
    if method == "maxlog":
        llrs = maxlog_fn(
            jax_symbols_flat, constellation_jax, bits_table_t_jax, sigma_sq, log_pmf_jax
        )
    else:
        llrs = exact_fn(
            jax_symbols_flat, constellation_jax, bits_table_t_jax, sigma_sq, log_pmf_jax
        )

    # Reshape to match input structure
    flat_llrs = llrs.flatten()
    if len(original_shape) > 1:
        new_shape = list(original_shape)
        new_shape[-1] = new_shape[-1] * k
        flat_llrs = flat_llrs.reshape(new_shape)

    # Convert output to the requested backend
    if output == "jax":
        return flat_llrs
    elif output == "numpy":
        return np.asarray(flat_llrs)
    else:  # output == "input"
        if is_jax_array(symbols):
            return flat_llrs  # already JAX
        # xp is NumPy or CuPy - convert via NumPy intermediate
        return xp.asarray(np.asarray(flat_llrs))
