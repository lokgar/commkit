"""Hand-written CUDA kernels for CuPy input (private).

Raw CUDA C++ kernels (under ``src/``) and their lazy compilation through
:class:`cupy.RawModule`.  :func:`get_kernel` is the one entry point: it
returns a launchable wrapper for a kernel, and callers use it only on CuPy
input, so CuPy already works.  A kernel that fails to compile or load
raises: there is no slower path to fall back to, since the kernels are
tested on a GPU before every release.

The launch wrappers live by family in :mod:`.bps`, :mod:`.llr`,
:mod:`.coding`, :mod:`.filters` and :mod:`.selftest` and check their
arguments with :func:`.compiler.check_arrays`; this module holds the
registry.

Importing this package (or ``commkit``) never touches NVRTC: kernel sources
are read and compiled on the first ``get_kernel()`` call for a given
specialization and cached in-process afterwards.  Cross-process reuse comes
from CuPy's on-disk kernel cache (``~/.cupy/kernel_cache``).
"""

from collections.abc import Callable
from typing import Any

from .bps import _bps_anchor_factory, _bps_min_d2_factory, _cs_block_factory
from .coding import _algebraic_correct_factory, _ldpc_layered_factory
from .filters import _iir_factory, _interpolate_factory
from .llr import (
    _llr_factory,
    _llr_separable_factory,
    _llr_separable_maxlog_factory,
)
from .selftest import _selftest_scale_factory


def get_kernel(name: str, **spec: Any) -> Callable:
    """Return a launchable kernel wrapper.

    Parameters
    ----------
    name : str
        Registered kernel name (see ``_KERNEL_FACTORIES``).
    **spec
        Kernel-specific specialization options (e.g. mode or dtype),
        forwarded to the kernel's wrapper factory.

    Returns
    -------
    callable
        A wrapper that handles grid/block computation and input
        validation. Call sites pass CuPy arrays and plain Python scalars
        only.

    Raises
    ------
    KeyError
        If `name` is not a registered kernel.
    ValueError
        If ``spec`` is not a valid specialization of the kernel.
    """
    factory = _KERNEL_FACTORIES.get(name)
    if factory is None:
        raise KeyError(
            f"Unknown CUDA kernel {name!r}; registered: {sorted(_KERNEL_FACTORIES)}"
        )
    return factory(**spec)


#: Kernel name -> wrapper factory.
_KERNEL_FACTORIES: dict[str, Callable[..., Callable]] = {
    "selftest_scale": _selftest_scale_factory,
    "bps_min_d2": _bps_min_d2_factory,
    "cs_block": _cs_block_factory,
    "bps_anchor": _bps_anchor_factory,
    "llr": _llr_factory,
    "llr_separable": _llr_separable_factory,
    "llr_separable_maxlog": _llr_separable_maxlog_factory,
    "iir": _iir_factory,
    "ldpc_layered": _ldpc_layered_factory,
    "algebraic_correct": _algebraic_correct_factory,
    "interpolate": _interpolate_factory,
}
