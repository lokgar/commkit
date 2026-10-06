"""
`commkit` is a high-performance library for simulating and analyzing
digital communication systems. It provides a unified API for generating,
transforming, and assessing signals on NumPy (CPU) and CuPy (GPU) data.

Main Features
-------------
- **Signal Abstractions**: Unified `Signal` and `SingleCarrierFrame` containers.
- **Modulation**: Support for PAM, PSK, and QAM (NRZ/RZ) with Gray coding.
- **Impairments**: Simulation of AWGN, Phase Noise, and Frequency Offset.
- **Synchronization**: Time and frequency synchronization algorithms.
- **Execution backends**: the device follows the data (NumPy or CuPy).

Importing commkit has no side effects: it does not configure logging or
Matplotlib, change warning filters, or touch the GPU.  Subpackages such as
``commkit.plotting`` are loaded on first access.
"""

import importlib
from typing import Any

__version__ = "1.1.0"

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
from .mapping import Constellation

# Loaded on first attribute access (PEP 562), so ``import commkit`` stays cheap
# and does not import Matplotlib, SciPy-heavy modules, or CuPy until needed.
_SUBMODULES = frozenset(
    {
        "analysis",
        "backend",
        "coding",
        "equalization",
        "filtering",
        "frequency",
        "impairments",
        "mapping",
        "metrics",
        "multirate",
        "plotting",
        "recovery",
        "smoothing",
        "spectral",
        "timing",
    }
)


# Value objects re-exported at top level from modules that are loaded lazily.
_LAZY_NAMES = {
    "RRC": "filtering",
    "RC": "filtering",
    "Gaussian": "filtering",
    "Rect": "filtering",
    "SmoothRect": "filtering",
}


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        return importlib.import_module(f".{name}", __name__)
    if name in _LAZY_NAMES:
        module = importlib.import_module(f".{_LAZY_NAMES[name]}", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _SUBMODULES | set(_LAZY_NAMES))


__all__ = [
    "Constellation",
    "Gaussian",
    "RC",
    "RRC",
    "Rect",
    "SmoothRect",
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
