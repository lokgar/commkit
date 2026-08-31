"""
`commkit` is a high-performance library for simulating and analyzing
digital communication systems. It provides a unified API for generating,
transforming, and assessing signals across diverse computational backends
(CPU, GPU, and JAX).

Main Features
-------------
- **Signal Abstractions**: Unified `Signal` and `SingleCarrierFrame` containers.
- **Modulation**: Support for PAM, PSK, and QAM (NRZ/RZ) with Gray coding.
- **Impairments**: Simulation of AWGN, Phase Noise, and Frequency Offset.
- **Synchronization**: Time and frequency synchronization algorithms.
- **Execution backends**: Transparent NumPy, CuPy, and JAX support.
"""

import warnings

__version__ = "1.0.0"

# Leaf modules import ``Signal`` from ``.core.signal`` at top level for
# Signal/array dispatch.  ``core.signal`` is a leaf (it imports no sibling
# domain modules), so the first leaf module imported below pulls ``core`` in
# without a cycle, regardless of statement order here.
from . import (
    analysis,
    equalization,
    frequency,
    impairments,
    metrics,
    recovery,
    spectral,
    timing,
)
from .core import (
    Preamble,
    Signal,
    SingleCarrierFrame,
    generate,
    generate_pam,
    generate_psk,
    generate_psqam,
    generate_qam,
)
from .io import load_npz, save_npz
from .logger import set_log_level
from .plotting import apply_default_theme

# Filter the specific warning message using a regular expression match
warnings.filterwarnings("ignore", message=".*cupyx.jit.rawkernel is experimental.*")

__all__ = [
    "Preamble",
    "Signal",
    "SingleCarrierFrame",
    "__version__",
    "analysis",
    "equalization",
    "frequency",
    "generate",
    "generate_pam",
    "generate_psk",
    "generate_psqam",
    "generate_qam",
    "impairments",
    "load_npz",
    "metrics",
    "recovery",
    "save_npz",
    "set_log_level",
    "spectral",
    "timing",
]

apply_default_theme()
