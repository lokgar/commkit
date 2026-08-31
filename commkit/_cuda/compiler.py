"""
Lazy CUDA kernel compilation with in-process caching.

Kernel sources are real ``.cu`` files under ``commkit/_cuda/src/``
(shipped as package data) and are compiled via :class:`cupy.RawModule`
with C++17 and templates for mode/dtype specialization. ``name_expressions``
provides mangled-name resolution for the template instantiations.

Compilation is lazy: nothing here imports CuPy or touches NVRTC until
``get_raw_kernel()`` is called. Compiled kernels are cached per
``(source_name, options, name_expression)`` for the process lifetime;
cross-process reuse comes from CuPy's on-disk NVRTC cache.
"""

from importlib import resources
from typing import Any

DEFAULT_OPTIONS: tuple[str, ...] = ("-std=c++17", "--use_fast_math")

_SOURCE_CACHE: dict[str, str] = {}
# (source_name, options, name_expression) -> (RawModule, RawKernel).
# The module is cached alongside the kernel to keep the loaded CUmodule
# alive for the process lifetime.
_KERNEL_CACHE: dict[tuple[str, tuple[str, ...], str], tuple[Any, Any]] = {}


def read_source(source_name: str) -> str:
    """Loads a CUDA C++ source file from the package's ``src/`` directory.

    Parameters
    ----------
    source_name : str
        Base name of the source file, without the ``.cu`` extension
        (e.g. ``"bps_min_d2"``).

    Returns
    -------
    str
        The file contents.
    """
    src = _SOURCE_CACHE.get(source_name)
    if src is None:
        src = (
            resources.files("commkit._cuda")
            .joinpath(f"src/{source_name}.cu")
            .read_text(encoding="utf-8")
        )
        _SOURCE_CACHE[source_name] = src
    return src


def get_raw_kernel(
    source_name: str,
    name_expression: str,
    options: tuple[str, ...] = DEFAULT_OPTIONS,
) -> Any:
    """Compiles (or retrieves from cache) one kernel specialization.

    Parameters
    ----------
    source_name : str
        Base name of the ``.cu`` file under ``src/``.
    name_expression : str
        C++ name expression for the kernel instantiation,
        e.g. ``"bps_min_d2<TABLE>"``.
    options : tuple of str, default ``DEFAULT_OPTIONS``
        NVRTC compile options.

    Returns
    -------
    cupy.RawKernel
        The launchable kernel for `name_expression`.
    """
    key = (source_name, options, name_expression)
    cached = _KERNEL_CACHE.get(key)
    if cached is not None:
        return cached[1]

    import cupy as cp

    module = cp.RawModule(
        code=read_source(source_name),
        options=options,
        name_expressions=[name_expression],
    )
    kernel = module.get_function(name_expression)
    _KERNEL_CACHE[key] = (module, kernel)
    return kernel
