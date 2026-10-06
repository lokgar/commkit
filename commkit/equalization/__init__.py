"""
Adaptive and block channel equalization.

This package provides sequential (LMS, RLS, CMA, RDE) and block / frequency-domain
equalizers with optional carrier-phase recovery, plus linear (zero-forcing) and
pilot-tone polarization-demultiplexing routines. Sequential equalizers run as
Numba loops on the CPU; block / frequency-domain equalizers run on the device
the input lives on (NumPy or CuPy).

The public API is unchanged from when this was a single module:
``from commkit.equalization import lms, rls, cma, rde, ...`` continues to work.
"""

from __future__ import annotations

# ``_get_numba`` and ``_check_rls_divergence`` are re-exported for tests that
# reach package internals through this namespace (removed in the equalization
# module pass). F401 is silenced for this re-export hub in pyproject.toml.
from ._block import block_lms
from ._kernels_numba import _get_numba
from .blind import block_cma, block_rde, build_pilot_ref
from .linear import apply_taps, estimate_transfer_function, zf_equalizer
from .polarization import (
    apply_interpolated_matrix,
    demultiplex_polarization_tones_dynamic,
    demultiplex_polarization_tones_static,
)
from .result import CPRState, EqualizerResult
from .sequential import _check_rls_divergence, cma, lms, rde, rls

__all__ = [
    "CPRState",
    "EqualizerResult",
    "apply_interpolated_matrix",
    "apply_taps",
    "block_cma",
    "block_lms",
    "block_rde",
    "build_pilot_ref",
    "cma",
    "demultiplex_polarization_tones_dynamic",
    "demultiplex_polarization_tones_static",
    "estimate_transfer_function",
    "lms",
    "rde",
    "rls",
    "zf_equalizer",
]
