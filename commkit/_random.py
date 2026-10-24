"""Random sources (private).

Every random function takes ``rng: int | numpy.random.Generator | None``
(SciPy SPEC 7), which :func:`as_generator` turns into a Generator.  What is
transmitted (bits, symbols) is drawn on the host from that Generator and
transferred, so a seed gives the same payload on every device.  What the
channel adds (AWGN, phase noise, ASE) is drawn on the data's device by
:func:`standard_normal` and :func:`complex_normal`: on the GPU from a CuPy
Generator seeded from the host Generator, so CPU and GPU realizations differ
while their statistics agree.  The CuPy Generator uses the counter-based
Philox bit generator, whose setup is negligible (XORWOW initializes
per-thread state and costs about 0.8 ms per call).
"""

from types import ModuleType

import numpy as np

from .backend import ArrayType

RNG = int | np.random.Generator | None


def as_generator(rng: RNG) -> np.random.Generator:
    """The host Generator for ``rng`` (a Generator is returned as is)."""
    return np.random.default_rng(rng)


def standard_normal(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    *,
    dtype: type | np.dtype,
    xp: ModuleType,
) -> ArrayType:
    """N(0, 1) samples of ``shape`` and real ``dtype`` on ``xp``'s device."""
    if xp is np:
        return rng.standard_normal(shape, dtype=dtype)
    seed = int(rng.integers(2**63))
    device_rng = xp.random.Generator(xp.random.Philox4x3210(seed))
    return device_rng.standard_normal(shape, dtype=dtype)


def complex_normal(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    *,
    std: float | ArrayType,
    dtype: type | np.dtype,
    xp: ModuleType,
) -> ArrayType:
    """Circular complex Gaussian noise ``std * (a + j b)``, ``a, b ~ N(0, 1)``,
    of ``shape`` on ``xp``'s device; ``dtype`` is the real component type.

    ``E[|n|^2] = 2 std^2``: pass ``std = sqrt(P / 2)`` for noise power ``P``.
    """
    n = standard_normal(rng, (2, *shape), dtype=dtype, xp=xp)
    return std * (n[0] + 1j * n[1])
