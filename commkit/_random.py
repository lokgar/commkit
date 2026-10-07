"""Random sources (private).

Every random function takes ``rng: int | numpy.random.Generator | None``
(SciPy SPEC 7).  What is transmitted (bits, symbols) is drawn on the host
from that Generator and transferred, so a seed gives the same payload on
every device.  What the channel adds (AWGN, phase noise) is drawn on the
data's device: on the GPU from a CuPy Generator seeded from the host
Generator, so CPU and GPU realizations differ while their statistics agree.  It uses the counter-based Philox
bit generator, whose setup is negligible (XORWOW initializes per-thread state
and costs about 0.8 ms per call).
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
