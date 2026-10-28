"""Interleaving: :func:`interleave` and :func:`deinterleave` with an :class:`Interleaver`.

An interleaver permutes each block of ``length`` bits (or LLRs, or any
values) along the last axis: ``out[j] = x[permutation[j]]``.  Build one
from an explicit permutation, as a row-in column-out block interleaver
(:meth:`Interleaver.block`, which spreads bursts of up to ``rows`` errors)
or as a uniformly random permutation (:meth:`Interleaver.random`).

The permutation is a host array, gathered on the data's device with an
index cached per object and device.  Interleavers separate the bursts of
a channel or of an inner decoder from an outer code, and define the
constituent permutation of turbo codes and the bit layout of PAS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .._array import validate_array
from ..backend import ArrayType, _device_tables, get_array_module
from .base import _Value

__all__ = ["Interleaver", "deinterleave", "interleave"]


@dataclass(frozen=True, eq=False)
class Interleaver(_Value):
    """Interleaver given by an explicit permutation.

    Parameters
    ----------
    permutation : array_like
        ``(L,)``, a permutation of ``0 ... L - 1``: output position ``j``
        takes input position ``permutation[j]``.
    """

    permutation: np.ndarray

    def __post_init__(self) -> None:
        p = np.asarray(self.permutation)
        if p.ndim != 1 or p.size == 0 or p.dtype.kind not in "iu":
            raise ValueError("permutation must be a non-empty 1-D integer array.")
        if not np.array_equal(np.sort(p), np.arange(p.size)):
            raise ValueError("permutation must contain each of 0 ... L - 1 once.")
        p = p.astype(np.int64)
        p.setflags(write=False)
        object.__setattr__(self, "permutation", p)

    @property
    def length(self) -> int:
        """Values per block."""
        return int(self.permutation.size)

    def __repr__(self) -> str:
        return f"Interleaver(length={self.length})"

    @classmethod
    def block(cls, rows: int, columns: int) -> Interleaver:
        """Row-in, column-out block interleaver of ``rows x columns`` values.

        Parameters
        ----------
        rows, columns : int
            Array shape; a block holds ``rows * columns`` values.

        Returns
        -------
        Interleaver
            Values written row by row and read column by column: adjacent
            output values come from input values ``columns`` apart.
        """
        for name, v in (("rows", rows), ("columns", columns)):
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ValueError(f"{name} must be an integer >= 1, got {v!r}.")
        return cls(np.arange(rows * columns).reshape(rows, columns).T.ravel())

    @classmethod
    def random(
        cls, length: int, *, rng: int | np.random.Generator | None = None
    ) -> Interleaver:
        """Uniformly random interleaver of ``length`` values.

        Parameters
        ----------
        length : int
            Values per block.
        rng : int, numpy.random.Generator or None
            Seed or generator (SciPy SPEC 7); drawn on the host.

        Returns
        -------
        Interleaver
        """
        if isinstance(length, bool) or not isinstance(length, int) or length < 1:
            raise ValueError(f"length must be an integer >= 1, got {length!r}.")
        return cls(np.random.default_rng(rng).permutation(length))


def _permute(x: Any, interleaver: Any, inverse: bool, name: str) -> ArrayType:
    if not isinstance(interleaver, Interleaver):
        raise TypeError(
            f"{name}: interleaver must be a Interleaver, got "
            f"{type(interleaver).__name__}."
        )
    arr = validate_array(x, name="x")
    xp = get_array_module(arr)
    if arr.ndim not in (1, 2):
        raise ValueError(f"{name}: x must have shape (N,) or (C, N), got {arr.shape}.")
    length = interleaver.length
    if arr.shape[-1] % length:
        raise ValueError(
            f"{name}: the last axis of x ({arr.shape[-1]}) must be a multiple of the "
            f"interleaver length {length}."
        )

    def build() -> tuple[np.ndarray, ...]:
        p = interleaver.permutation
        return (p, np.argsort(p))

    forward, backward = _device_tables(interleaver, "permutation", xp, build)
    blocks = arr.reshape(*arr.shape[:-1], -1, length)
    return blocks[..., backward if inverse else forward].reshape(arr.shape)


def interleave(x: ArrayType, *, interleaver: Interleaver) -> ArrayType:
    """Interleave blocks of values along the last axis.

    Parameters
    ----------
    x : array_like
        Bits, LLRs or any values, ``(N,)`` or ``(C, N)`` with ``N`` a
        multiple of the interleaver length.
    interleaver : Interleaver
        The permutation.

    Returns
    -------
    ndarray
        ``out[..., b L + j] = x[..., b L + permutation[j]]``, same shape,
        dtype and device.

    Raises
    ------
    ValueError
        If ``N`` is not a multiple of the interleaver length.

    See Also
    --------
    deinterleave : The inverse.
    """
    return _permute(x, interleaver, False, "interleave()")


def deinterleave(x: ArrayType, *, interleaver: Interleaver) -> ArrayType:
    """Undo :func:`interleave`: ``deinterleave(interleave(x)) == x``.

    Parameters
    ----------
    x : array_like
        Interleaved values, ``(N,)`` or ``(C, N)``.
    interleaver : Interleaver
        The interleaver that was applied.

    Returns
    -------
    ndarray
        Same shape, dtype and device.

    Raises
    ------
    ValueError
        If ``N`` is not a multiple of the interleaver length.
    """
    return _permute(x, interleaver, True, "deinterleave()")
