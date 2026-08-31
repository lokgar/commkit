"""
The :class:`Constellation` value object.

A small immutable container bundling the three things that are otherwise
passed around as loose tuples/arrays at every CMA/RDE radius and GMI/MI call
site: the constellation ``points``, their natural-binary ``bit_labels``, and an
optional probabilistic-shaping ``pmf``.  ``Constellation.gray(...)`` is the
canonical constructor (delegating to :func:`gray_constellation`), and
``power()`` / ``map()`` / ``demap()`` / ``llr()`` are thin wrappers over the
existing free functions so a single object carries geometry + labels + prior
together.

The loose-array forms (``gray_constellation``, ``map_bits``,
``demap_symbols_hard``, ``compute_llr``, ``constellation_power``) remain the
public surface; this object is an additive convenience layered on top.
"""

from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

from ..backend import ArrayType
from .bits import demap_symbols_hard, map_bits
from .gray import gray_constellation
from .llr import compute_llr
from .shaping import constellation_power

__all__ = ["Constellation"]


@dataclass(frozen=True, eq=False)
class Constellation:
    """Immutable constellation: points, Gray bit-labels, and optional PS pmf.

    Attributes
    ----------
    points : np.ndarray
        Constellation points, shape ``(M,)``, indexed by natural-binary symbol
        value (``points[i]`` is the symbol for integer ``i``).  Complex for
        PSK/QAM, real for ASK/PAM.
    bit_labels : np.ndarray
        Natural-binary bit pattern per symbol index, shape ``(M, k)`` where
        ``k = log2(M)``, dtype ``int8`` (MSB first).
    modulation : str
        Modulation scheme (``"psk"``, ``"qam"``, ``"ask"``, ``"pam"``).
    order : int
        Modulation order ``M``.
    unipolar : bool
        Whether the grid is the unipolar ASK/PAM variant.
    pmf : np.ndarray or None
        Optional Maxwell-Boltzmann shaping prior, shape ``(M,)``, aligned with
        ``points``.  ``None`` means uniform.
    """

    points: np.ndarray
    bit_labels: np.ndarray
    modulation: str
    order: int
    unipolar: bool = False
    pmf: np.ndarray | None = None

    @property
    def bits_per_symbol(self) -> int:
        """Number of bits per symbol, ``log2(order)``."""
        return int(self.bit_labels.shape[1])

    @classmethod
    def gray(
        cls,
        modulation: str,
        order: int,
        *,
        normalize: bool = True,
        unipolar: bool = False,
        pmf: np.ndarray | None = None,
    ) -> "Constellation":
        """Build a Gray-mapped constellation (cached for the ``pmf=None`` base).

        Parameters mirror :func:`gray_constellation`.  ``pmf`` attaches a
        probabilistic-shaping prior without rebuilding the geometry.
        """
        base = _gray_base(modulation, order, normalize, unipolar)
        if pmf is None:
            return base
        return replace(base, pmf=np.asarray(pmf, dtype=np.float64))

    def power(self) -> float:
        """pmf-weighted average symbol power ``E[|s|^2]`` (uniform if no pmf)."""
        return constellation_power(self.points, self.pmf)

    def map(self, bits: ArrayType) -> ArrayType:
        """Map a flat bit sequence to symbols (see :func:`map_bits`)."""
        return map_bits(bits, self.modulation, self.order, unipolar=self.unipolar)

    def demap(self, symbols: ArrayType) -> ArrayType:
        """Hard-decision demap symbols to bits (see :func:`demap_symbols_hard`).

        Carries this constellation's ``pmf`` through for PS-QAM rescaling.
        """
        return demap_symbols_hard(
            symbols,
            self.modulation,
            self.order,
            unipolar=self.unipolar,
            pmf=self.pmf,
        )

    def llr(
        self,
        symbols: ArrayType,
        noise_var: float,
        *,
        method: str = "maxlog",
        output: str = "jax",
    ) -> ArrayType:
        """Soft-decision LLRs (see :func:`compute_llr`), carrying this ``pmf``."""
        return compute_llr(
            symbols,
            self.modulation,
            self.order,
            noise_var,
            method=method,
            unipolar=self.unipolar,
            output=output,
            pmf=self.pmf,
        )


@lru_cache(maxsize=128)
def _gray_base(
    modulation: str, order: int, normalize: bool, unipolar: bool
) -> Constellation:
    """Cached Gray constellation (geometry + labels), without a shaping pmf."""
    points = gray_constellation(
        modulation, order, normalize=normalize, unipolar=unipolar
    )
    k = int(np.log2(order))
    bit_labels = (
        (
            np.arange(order, dtype="int32")[:, None]
            >> np.arange(k - 1, -1, -1, dtype="int32")
        )
        & 1
    ).astype(np.int8)
    return Constellation(
        points=points,
        bit_labels=bit_labels,
        modulation=modulation,
        order=order,
        unipolar=unipolar,
        pmf=None,
    )
