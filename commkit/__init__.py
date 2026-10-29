"""Digital-communications research library for NumPy (CPU) and CuPy (GPU).

CommKit covers the signal chain of coherent and IM/DD links: waveform
generation (:mod:`commkit.core`, :mod:`commkit.mapping`), filters and
resampling (:mod:`commkit.filtering`, :mod:`commkit.multirate`,
:mod:`commkit.spectral`), channel impairments (:mod:`commkit.impairments`),
synchronization and carrier recovery (:mod:`commkit.timing`,
:mod:`commkit.frequency`, :mod:`commkit.recovery`), equalization
(:mod:`commkit.equalization`), channel coding (:mod:`commkit.coding`),
metrics (:mod:`commkit.metrics`), laser and phase characterization
(:mod:`commkit.analysis`) and plots of their results (:mod:`commkit.plotting`).

Data is an array or a :class:`Signal`, which carries the samples with their
facts (sampling and symbol rate), their description (constellation, pulse)
and the transmitted reference.  Algorithms are plain functions; a
modulation, a pulse or a sub-algorithm is described by a frozen value object
written in the call.  Every function runs on the device of its input.

The top level re-exports :class:`Signal`, :class:`Reference`,
:func:`generate`, :class:`SingleCarrierFrame`, :class:`Preamble`,
:class:`Constellation`, the pulses (:class:`RRC`, :class:`RC`,
:class:`Gaussian`, :class:`Rect`, :class:`SmoothRect`), :func:`save_npz`,
:func:`load_npz`, :func:`set_log_level` and :class:`PrecisionWarning`.  The
subpackages and modules are imported on first attribute access
(``commkit.plotting``, ...).

Importing commkit has no side effects: it does not configure logging or
Matplotlib, change warning filters, or touch the GPU.
"""

import importlib
from typing import Any

__version__ = "3.1.1"

from .backend import PrecisionWarning
from .core import (
    Preamble,
    Reference,
    Signal,
    SingleCarrierFrame,
    generate,
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
        "math",
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
    "Reference",
    "Rect",
    "SmoothRect",
    "Preamble",
    "PrecisionWarning",
    "Signal",
    "SingleCarrierFrame",
    "__version__",
    "analysis",
    "coding",
    "equalization",
    "frequency",
    "generate",
    "impairments",
    "load_npz",
    "metrics",
    "recovery",
    "save_npz",
    "set_log_level",
    "spectral",
    "timing",
]
