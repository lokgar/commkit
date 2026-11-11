"""Square-lattice structure of a constellation (private).

:func:`square_grid` (the points form a full square lattice with the same
levels on both axes), :func:`separable_axes` (that lattice factors into
per-axis levels, labels and prior) and :func:`slicer_params` (the float32
parameters of the compiled O(1) square-lattice slicers).  A
:class:`~commkit.mapping.Constellation` computes each once through
``find_square_grid`` and ``find_separable_axes`` and caches it; the fast
paths read it here: LLRs and decisions in ``mapping``, the PLL and BPS of
``recovery``, the equalizers' slicers, the AWGN bounds of ``metrics`` and
``PAS`` in ``coding``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np

if TYPE_CHECKING:
    from .constellation import Constellation


class Axis(NamedTuple):
    """One axis of a separable constellation (see :func:`separable_axes`)."""

    levels: np.ndarray  # (L,) real levels on this axis
    labels: np.ndarray  # (L, k_axis) labels of this axis's bits
    pmf: np.ndarray | None  # (L,) marginal prior
    bits: np.ndarray  # (k_axis,) positions of these bits in the symbol label


def square_grid(
    constellation: Constellation,
) -> tuple[float, float, int, np.ndarray] | None:
    """``(lev_min, step, side, lut)`` if the points form a full square lattice.

    ``lut[i * side + q]`` is the index of the point at level ``i`` on the
    real axis and ``q`` on the imaginary axis.  ``None`` otherwise (PSK,
    PAM, cross and 8-QAM, arbitrary points).
    """
    return constellation._square_grid


def separable_axes(constellation: Constellation) -> tuple[Axis, Axis] | None:
    """Per-axis factors of a separable square lattice, or ``None``.

    The real and imaginary axes of a square lattice whose labels and prior
    factor into per-axis ones (Gray QAM, shaped or not).  Every bit then
    depends on one axis only, and the other axis's factor cancels from its
    LLR.
    """
    return constellation._axes


def slicer_params(
    constellation: Constellation | None,
) -> tuple[int, np.float32, np.float32]:
    """``(side, lev_min, step)`` for the compiled O(1) square-lattice slicers.

    ``side = 0`` (and placeholder levels) unless the points form a full
    square lattice with the same levels on both axes
    (:func:`square_grid`); the slicers then search all
    ``M`` points.  ``None`` (no decisions needed) also gives ``side = 0``.
    """
    grid = None if constellation is None else square_grid(constellation)
    if grid is None:
        return 0, np.float32(0.0), np.float32(1.0)
    lev_min, step, side, _ = grid
    return side, np.float32(lev_min), np.float32(step)


def find_square_grid(
    constellation: Constellation,
) -> tuple[float, float, int, np.ndarray] | None:
    """Compute :func:`square_grid` (cached by the constellation)."""
    side = int(round(constellation.order**0.5))
    if not constellation.is_complex or side * side != constellation.order or side < 2:
        return None
    re, im = constellation.points.real, constellation.points.imag
    lev_min = float(re.min())
    step = (float(re.max()) - lev_min) / (side - 1)
    if step <= 0 or abs(float(im.min()) - lev_min) > 1e-9 * step:
        return None
    gi, gq = (re - lev_min) / step, (im - lev_min) / step
    ri, rq = np.round(gi), np.round(gq)
    if max(np.abs(gi - ri).max(), np.abs(gq - rq).max()) > 1e-9:
        return None
    cell = ri.astype(np.int64) * side + rq.astype(np.int64)
    if cell.min() < 0 or cell.max() >= constellation.order:
        return None
    if np.unique(cell).size != constellation.order:
        return None
    lut = np.empty(constellation.order, dtype=np.int64)
    lut[cell] = np.arange(constellation.order)
    lut.setflags(write=False)
    return lev_min, step, side, lut


def find_separable_axes(constellation: Constellation) -> tuple[Axis, Axis] | None:
    """Compute :func:`separable_axes` (cached by the constellation)."""
    grid = square_grid(constellation)
    if grid is None:
        return None
    _, _, side, lut = grid
    labels = constellation.bit_labels[lut].reshape(side, side, -1)  # [i, q, bit]
    on_re = np.all(labels == labels[:, :1, :], axis=(0, 1))  # constant in q
    on_im = np.all(labels == labels[:1, :, :], axis=(0, 1))  # constant in i
    if not np.all(on_re ^ on_im):
        return None
    bits_re, bits_im = np.flatnonzero(on_re), np.flatnonzero(on_im)
    if 2**bits_re.size != side or 2**bits_im.size != side:
        return None
    points = constellation.points[lut].reshape(side, side)
    pmf_re = pmf_im = None
    if constellation.pmf is not None:
        pmf = constellation.pmf[lut].reshape(side, side)
        pmf_re, pmf_im = pmf.sum(axis=1), pmf.sum(axis=0)
        if not np.allclose(pmf, np.outer(pmf_re, pmf_im), rtol=1e-9, atol=0.0):
            return None
    return (
        Axis(points[:, 0].real, labels[:, 0][:, bits_re], pmf_re, bits_re),
        Axis(points[0, :].imag, labels[0, :][:, bits_im], pmf_im, bits_im),
    )
