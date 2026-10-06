"""
The :class:`Constellation` value object.

A constellation is described by its ``points``, the ``bit_labels`` assigned to
each point, and an optional probabilistic-shaping prior ``pmf``.  It is the
single way to describe a modulation in CommKit 2.0: functions take a
``Constellation``, never a ``modulation`` string plus an ``order``.

Build one with a factory (Gray labelled, unit average power)::

    Constellation.qam(16)
    Constellation.psk(8)
    Constellation.pam(4, unipolar=True)
    Constellation.qam(64).shaped(entropy=5.0)

or from arbitrary points (natural-binary labels unless given)::

    Constellation(points, bit_labels=labels)

Instances are immutable, hashable, compare by value, and hold read-only
NumPy arrays.  They live on the host; methods that take data run on the
data's device.
"""

from dataclasses import dataclass, replace
from functools import cache, lru_cache

import numpy as np

from ..backend import ArrayType, dispatch
from .gray import _gray_points, _nearest_index, _unpack_bits
from .llr import _llr
from .shaping import _constellation_power

__all__ = ["Constellation"]

_FAMILIES = ("qam", "psk", "pam")


@dataclass(frozen=True, eq=False, repr=False)
class Constellation:
    """Immutable constellation: points, bit labels and an optional shaping pmf.

    Parameters
    ----------
    points : array_like
        Constellation points, shape ``(M,)`` with ``M = 2**k``.  Complex for
        two-dimensional constellations, real for PAM.  Stored as
        ``complex128`` or ``float64``.  Points must be distinct.
    bit_labels : array_like, optional
        Bit pattern of each point, shape ``(M, k)``, MSB first, values 0/1.
        Every pattern must occur exactly once.  ``None`` assigns the natural
        binary labels (``points[i]`` carries the bits of ``i``).  The factories
        assign Gray labels this way, because their points are ordered by label.
    pmf : array_like, optional
        Prior probability of each point, shape ``(M,)``, non-negative and
        summing to 1.  ``None`` means uniform.  Use :meth:`shaped` to build a
        Maxwell-Boltzmann prior.
    family : {"qam", "psk", "pam"}, optional
        Informational label set by the factories and shown by ``repr``.
        ``None`` for arbitrary points.  Algorithms never rely on it, and it
        does not take part in equality.

    Notes
    -----
    The factories return unit average power, ``E[|s|^2] = 1`` under the
    ``pmf``.  A shaped constellation is rescaled to keep that property, so
    received symbols normalized to unit power lie on ``points`` directly.
    """

    points: np.ndarray
    bit_labels: np.ndarray = None  # type: ignore[assignment]  # None -> natural binary
    pmf: np.ndarray | None = None
    family: str | None = None

    def __post_init__(self) -> None:
        points = _host_array(self.points, "points")
        if points.ndim != 1 or points.size < 2:
            raise ValueError(
                f"points must have shape (M,) with M >= 2, got {points.shape}."
            )
        points = (
            points.astype(np.complex128)
            if np.iscomplexobj(points)
            else points.astype(np.float64)
        )
        if not np.all(np.isfinite(points)):
            raise ValueError("points must be finite.")
        order = points.size
        if np.unique(points).size != order:
            raise ValueError("points must be distinct.")

        k = int(np.log2(order))
        if 2**k != order:
            raise ValueError(f"The number of points must be a power of 2, got {order}.")
        if self.bit_labels is None:
            labels = _unpack_bits(np.arange(order, dtype=np.int32), k)
        else:
            labels = _host_array(self.bit_labels, "bit_labels")
            if labels.shape != (order, k):
                raise ValueError(
                    f"bit_labels must have shape ({order}, {k}), got {labels.shape}."
                )
            if not np.all((labels == 0) | (labels == 1)):
                raise ValueError("bit_labels must contain only 0 and 1.")
            labels = labels.astype(np.int8)
            if np.unique(_pack(labels)).size != order:
                raise ValueError("bit_labels must assign every bit pattern once.")

        pmf = None
        if self.pmf is not None:
            pmf = _host_array(self.pmf, "pmf").astype(np.float64)
            if pmf.shape != (order,):
                raise ValueError(f"pmf must have shape ({order},), got {pmf.shape}.")
            if not np.all(np.isfinite(pmf)) or np.any(pmf < 0):
                raise ValueError("pmf must be finite and non-negative.")
            total = pmf.sum()
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"pmf must sum to 1, got {total:.8g}.")
            pmf = pmf / total

        if self.family is not None and self.family not in _FAMILIES:
            raise ValueError(f"family must be one of {_FAMILIES} or None.")

        for name, value in (("points", points), ("bit_labels", labels), ("pmf", pmf)):
            if value is not None:
                value.setflags(write=False)
            object.__setattr__(self, name, value)

    # -- factories -----------------------------------------------------------

    @classmethod
    def qam(cls, order: int) -> "Constellation":
        """Gray-labelled M-QAM at unit average power.

        Square QAM for even ``k = log2(M)``, rectangular 8-QAM, and cross QAM
        for other odd ``k`` (32, 128, ...).
        """
        return _named("qam", order, False)

    @classmethod
    def psk(cls, order: int) -> "Constellation":
        """Gray-labelled M-PSK on the unit circle."""
        return _named("psk", order, False)

    @classmethod
    def pam(cls, order: int, *, unipolar: bool = False) -> "Constellation":
        """Gray-labelled M-PAM (real) at unit average power.

        ``unipolar=True`` gives non-negative levels, which is M-ASK as used in
        intensity modulation (``pam(2, unipolar=True)`` is on-off keying).
        """
        return _named("pam", order, unipolar)

    def shaped(
        self, *, nu: float | None = None, entropy: float | None = None
    ) -> "Constellation":
        r"""Return a copy with a Maxwell-Boltzmann prior, rescaled to unit power.

        ``P(s) ∝ exp(-nu |s'|^2)``, where ``s'`` are the points scaled to a
        minimum distance of 2 (the integer grid for QAM and PAM, so ``nu``
        matches the literature and :func:`maxwell_boltzmann`).  Give exactly
        one of ``nu`` (``>= 0``) or ``entropy`` (bits per symbol, in
        ``(0, log2(M)]``).

        The points are rescaled so that the pmf-weighted power is 1; labels and
        point order are unchanged.
        """
        if (nu is None) == (entropy is None):
            raise ValueError("shaped() takes exactly one of nu or entropy.")
        energies = self._grid_energies()
        if np.ptp(energies) < 1e-9:
            raise ValueError(
                "shaped() needs points of different energy; every point of "
                "this constellation has the same energy."
            )
        if entropy is not None:
            nu = _nu_for_entropy(energies, float(entropy))
        assert nu is not None
        if nu < 0:
            raise ValueError(f"nu must be >= 0, got {nu}.")
        pmf = _mb_pmf(energies, float(nu))
        scale = 1.0 / np.sqrt(np.dot(pmf, np.abs(self.points) ** 2))
        return replace(self, points=self.points * scale, pmf=pmf)

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
        """Bridge for unmigrated modules; removed in 3.2.

        Unlike :meth:`shaped`, a ``pmf`` is attached *without* rescaling, which
        is the 1.x PS convention (``E[|s|^2] < 1`` under the pmf).
        """
        base = _gray_base(modulation, order, normalize, unipolar)
        if pmf is None:
            return base
        return replace(base, pmf=np.asarray(pmf, dtype=np.float64))

    # -- properties ----------------------------------------------------------

    @property
    def order(self) -> int:
        """Number of points ``M``."""
        return int(self.points.size)

    @property
    def bits_per_symbol(self) -> int:
        """Bits per symbol, ``k = log2(M)``."""
        return int(self.bit_labels.shape[1])

    @property
    def is_complex(self) -> bool:
        """Whether the points are complex (two-dimensional)."""
        return bool(np.iscomplexobj(self.points))

    @property
    def unipolar(self) -> bool:
        """Whether the constellation is real with non-negative points."""
        return not self.is_complex and bool(np.all(self.points >= 0))

    @property
    def entropy(self) -> float:
        """Entropy of the prior in bits per symbol (``log2(M)`` if uniform)."""
        if self.pmf is None:
            return float(self.bits_per_symbol)
        p = self.pmf[self.pmf > 0]
        return float(-np.sum(p * np.log2(p)))

    # -- operations ----------------------------------------------------------

    def power(self) -> float:
        """Average symbol power ``E[|s|^2]`` under the pmf (uniform if none)."""
        return _constellation_power(self.points, self.pmf)

    def map(self, bits: ArrayType) -> ArrayType:
        """Map bits to symbols on the bits' device.

        ``bits`` has shape ``(..., n)`` with ``n`` a multiple of ``k``; groups
        of ``k`` bits (MSB first) become one symbol.  Returns shape
        ``(..., n / k)``, ``complex64`` (or ``float32`` for real points).
        """
        bits, xp, _ = dispatch(bits)
        k = self.bits_per_symbol
        if bits.shape[-1] % k:
            raise ValueError(
                f"bits: the last axis ({bits.shape[-1]}) must be a multiple of "
                f"bits_per_symbol ({k})."
            )
        groups = bits.reshape(*bits.shape[:-1], -1, k).astype(xp.int32)
        weights = xp.asarray(1 << np.arange(k - 1, -1, -1, dtype=np.int32))
        index_of_label = xp.asarray(self._index_of_label())
        points = xp.asarray(self.points.astype(self._storage_dtype()))
        return points[index_of_label[(groups * weights).sum(axis=-1)]]

    def demap(self, symbols: ArrayType) -> ArrayType:
        """Hard decisions: bits of the nearest point, on the symbols' device.

        ``symbols`` has shape ``(..., N)``; returns ``int8`` bits of shape
        ``(..., N * k)``.  Symbols are compared with ``points`` as they are, so
        they must be on the constellation's scale.
        """
        symbols, xp, _ = dispatch(symbols)
        idx = _nearest_index(
            symbols.reshape(-1), self.points.astype(self._storage_dtype())
        )
        bits = xp.asarray(self.bit_labels)[idx]
        return bits.reshape(*symbols.shape[:-1], -1)

    def llr(
        self, symbols: ArrayType, *, noise_var: float, method: str = "maxlog"
    ) -> ArrayType:
        """Bit LLRs (positive means bit 0 is more likely), with this prior.

        ``noise_var`` is the complex noise variance on the constellation's
        scale.  ``method`` is ``"maxlog"`` or ``"exact"`` (log-sum-exp).
        Returns ``float32`` of shape ``(..., N * k)`` on the symbols' device.
        """
        return _llr(symbols, self.points, self.bit_labels, self.pmf, noise_var, method)

    # -- value semantics -----------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Constellation):
            return NotImplemented
        return (
            np.array_equal(self.points, other.points)
            and np.array_equal(self.bit_labels, other.bit_labels)
            and (
                (self.pmf is None and other.pmf is None)
                or (
                    self.pmf is not None
                    and other.pmf is not None
                    and np.array_equal(self.pmf, other.pmf)
                )
            )
        )

    def __hash__(self) -> int:
        # ``+ 0.0`` maps -0.0 to 0.0, which array_equal treats as equal.
        pmf = None if self.pmf is None else (self.pmf + 0.0).tobytes()
        return hash(((self.points + 0.0).tobytes(), self.bit_labels.tobytes(), pmf))

    def __repr__(self) -> str:
        name = self.family.upper() if self.family else "custom"
        if self.family == "pam" and self.unipolar:
            name += ", unipolar"
        shaped = "" if self.pmf is None else f", shaped H={self.entropy:.3f} bits"
        return f"Constellation({self.order}-{name}{shaped})"

    # -- internals -----------------------------------------------------------

    def _storage_dtype(self) -> type:
        return np.complex64 if self.is_complex else np.float32

    def _index_of_label(self) -> np.ndarray:
        """Point index for each packed label value, shape ``(M,)``."""
        index = np.empty(self.order, dtype=np.int64)
        index[_pack(self.bit_labels)] = np.arange(self.order)
        return index

    def _grid_energies(self) -> np.ndarray:
        """Point energies on the grid scaled to a minimum distance of 2."""
        d = np.abs(self.points[:, None] - self.points[None, :])
        d_min = d[~np.eye(self.order, dtype=bool)].min()
        return np.abs(self.points * (2.0 / d_min)) ** 2


def _host_array(value, name: str) -> np.ndarray:
    """``value`` as a NumPy array; device arrays are rejected by name."""
    if type(value).__module__.split(".")[0] == "cupy":
        raise TypeError(f"{name} must be a host (NumPy) array; use .get() first.")
    return np.array(value, copy=True)


def _pack(labels: np.ndarray) -> np.ndarray:
    """MSB-first bit rows to integers."""
    k = labels.shape[1]
    return labels.astype(np.int64) @ (1 << np.arange(k - 1, -1, -1, dtype=np.int64))


def _mb_pmf(energies: np.ndarray, nu: float) -> np.ndarray:
    log_p = -nu * energies
    p = np.exp(log_p - log_p.max())
    return p / p.sum()


def _nu_for_entropy(energies: np.ndarray, entropy: float) -> float:
    from scipy.optimize import brentq

    max_h = np.log2(energies.size)
    if not 0 < entropy <= max_h:
        raise ValueError(f"entropy must be in (0, {max_h:g}] bits, got {entropy}.")
    if np.isclose(entropy, max_h, atol=1e-8):
        return 0.0

    def gap(nu: float) -> float:
        p = _mb_pmf(energies, nu)
        p = p[p > 0]
        return float(-np.sum(p * np.log2(p))) - entropy

    nu_hi = 0.01
    while gap(nu_hi) > 0:
        nu_hi *= 10.0
    return float(brentq(gap, 0.0, nu_hi, xtol=1e-12, rtol=1e-12))


@cache
def _named(family: str, order: int, unipolar: bool) -> Constellation:
    if not isinstance(order, int | np.integer) or order < 2:
        raise ValueError(f"order must be an integer >= 2, got {order!r}.")
    points = _gray_points(family, int(order), unipolar=unipolar)
    return Constellation(points, family=family)


@lru_cache(maxsize=128)
def _gray_base(
    modulation: str, order: int, normalize: bool, unipolar: bool
) -> Constellation:
    """Cached 1.x Gray constellation for :meth:`Constellation.gray`."""
    points = _gray_points(modulation, order, normalize=normalize, unipolar=unipolar)
    mod = modulation.lower()
    family = next(
        (f for f in ("qam", "psk") if f in mod),
        "pam" if ("ask" in mod or "pam" in mod) else None,
    )
    return Constellation(points, family=family)
