"""Shared builders of coded test data, through the public API."""

from typing import Any

import numpy as np

from commkit import generate
from commkit.core import Signal
from commkit.impairments import apply_awgn
from commkit.mapping import Constellation, compute_llr


def coded_awgn(
    code: Any,
    constellation: Constellation,
    *,
    num_symbols: int,
    esn0_db: float,
    backend_device: str,
    num_channels: int = 1,
    seed: int = 0,
) -> tuple[Signal, Any]:
    """A coded payload over AWGN at one sample per symbol, and its LLRs.

    Returns the received Signal (its reference holds the coded bits and the
    information bits) and float32 LLRs on ``backend_device``.
    """
    sig = generate(
        constellation,
        num_symbols,
        symbol_rate=1e9,
        num_channels=num_channels,
        code=code,
        rng=seed,
        device=backend_device,
    )
    rx = apply_awgn(sig, esn0_db=esn0_db, rng=seed + 1)
    llrs = compute_llr(rx, noise_var=10 ** (-esn0_db / 10))
    return rx, llrs


def random_bits(shape: Any, xp: Any, seed: int = 0) -> Any:
    """int8 random bits of ``shape`` on ``xp``'s device."""
    rng = np.random.default_rng(seed)
    return xp.asarray(rng.integers(0, 2, shape).astype(np.int8))
