"""Closed-form performance over the AWGN channel.

:func:`awgn_ser`, :func:`awgn_ber`, :func:`awgn_mi` and :func:`awgn_gmi`
give the symbol and bit error rates, the mutual information and the
bit-metric decoding rate of a :class:`~commkit.mapping.Constellation` at an
Es/N0, in the conventions of the measured metrics (:func:`ser`, :func:`ber`,
:func:`mi`, :func:`gmi`): ``Es`` is the constellation's average energy under
its prior, complex points see complex noise of variance ``N0`` and real
points its in-phase half, and the error rates are those of nearest-point
decisions (:meth:`~commkit.mapping.Constellation.demap`).

How each is computed depends on the geometry:

* **one-dimensional** constellations (PAM, bipolar or unipolar, BPSK, any
  points on a line through the origin) and **separable** ones (square QAM
  whose labels and prior factor into per-axis ones, Gray QAM shaped or not,
  QPSK in any 45°-multiple orientation): exact, per axis, from the
  probabilities of the decision intervals;
* **other** constellations (PSK above 4 points, cross QAM, arbitrary
  points): the error rates are union bounds over all pairs, clipped at 1,
  upper bounds that are tight at low error rates.

The information rates are expectations over the noise by Gauss-Hermite
quadrature: 128 nodes per axis for one-dimensional and separable
constellations, a 128 × 128 product rule for the others.  Doubling either
changes no rate by more than 3e-7 bit (measured from -5 to 30 dB on PAM,
QAM, PSK, cross and star constellations).
"""

import logging
import math
from typing import Any

import numpy as np
from scipy.special import erfc

from ..logger import logger
from ..mapping import Constellation

__all__ = ["awgn_ber", "awgn_gmi", "awgn_mi", "awgn_ser"]

_ORDER_1D = 128
_ORDER_2D = 128
_COLLINEAR_TOL = 1e-9
# Bound on the (sent, candidate, node) arrays of the plane quadrature.
_MAX_ELEMENTS = 1 << 22


def _q(x: np.ndarray) -> np.ndarray:
    return 0.5 * erfc(x / math.sqrt(2.0))


def _esn0(esn0_db: Any, name: str) -> tuple[np.ndarray, bool]:
    arr = np.asarray(esn0_db, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name}: esn0_db must be finite.")
    return arr.reshape(-1), arr.ndim == 0


def _out(values: np.ndarray, scalar: bool, shape: tuple) -> float | np.ndarray:
    return float(values[0]) if scalar else values.reshape(shape)


class _Geometry:
    """The axes of a constellation for the exact computations, or none.

    ``axes`` is a list of ``(levels, labels, pmf, bit_positions)``, one entry
    for a one-dimensional constellation and two for a separable one; the
    noise per axis has variance ``N0 / 2``.  ``points`` are the points for
    the general computations.
    """

    def __init__(self, c: Constellation) -> None:
        self.c = c
        self.k = c.bits_per_symbol
        self.pmf = c.pmf if c.pmf is not None else np.full(c.order, 1.0 / c.order)
        self.points = c.points.astype(np.complex128)
        self.axes = self._axes()

    def _axes(self) -> list | None:
        c = self.c
        all_bits = np.arange(self.k)
        line = _line_angle(self.points)
        if line is not None:
            levels = (self.points * np.exp(-1j * line)).real
            return [(levels, c.bit_labels, self.pmf, all_bits)]
        for angle in (0.0, math.pi / 4):
            rotated = c if angle == 0.0 else _rotated(c, angle)
            axes = rotated._axes
            if axes is not None:
                return [
                    (
                        a.levels,
                        a.labels,
                        a.pmf
                        if a.pmf is not None
                        else np.full(a.levels.size, 1 / a.levels.size),
                        a.bits,
                    )
                    for a in axes
                ]
        return None


def _line_angle(points: np.ndarray) -> float | None:
    """Angle of the line through the origin holding every point, or None."""
    scale = float(np.max(np.abs(points)))
    ref = points[np.argmax(np.abs(points))]
    angle = float(np.angle(ref))
    off = (points * np.exp(-1j * angle)).imag
    if np.max(np.abs(off)) <= _COLLINEAR_TOL * scale:
        return angle
    return None


def _rotated(c: Constellation, angle: float) -> Constellation:
    pts = c.points.astype(np.complex128) * np.exp(-1j * angle)
    # Snap the rotated lattice's rounding so the grid test sees exact levels.
    pts = np.round(pts.real, 12) + 1j * np.round(pts.imag, 12)
    return Constellation(points=pts, bit_labels=c.bit_labels, pmf=c.pmf)


def _interval_probabilities(levels: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """``P[e, i, j]``: probability that level ``i`` sent is decided as ``j``
    (nearest level), noise std ``sigma[e]``."""
    order = np.argsort(levels)
    a = levels[order]
    bounds = np.concatenate(([-np.inf], (a[1:] + a[:-1]) / 2, [np.inf]))
    s = sigma[:, None, None]
    upper = (bounds[None, None, 1:] - a[None, :, None]) / s
    lower = (bounds[None, None, :-1] - a[None, :, None]) / s
    p_sorted = _q(lower) - _q(upper)  # P(lower < n < upper)
    inv = np.argsort(order)
    return p_sorted[:, inv][:, :, inv]


def _n0(geom: _Geometry, esn0: np.ndarray) -> np.ndarray:
    return geom.c.power() / 10 ** (esn0 / 10)


def _exact_rates(geom: _Geometry, esn0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact SER and BER for one-dimensional and separable constellations."""
    assert geom.axes is not None
    sigma = np.sqrt(_n0(geom, esn0) / 2)
    correct = np.ones_like(esn0)
    bit_errors = np.zeros_like(esn0)
    for levels, labels, pmf, _bits in geom.axes:
        p = _interval_probabilities(levels, sigma)
        hamming = np.sum(labels[:, None, :] != labels[None, :, :], axis=-1)
        correct = correct * np.einsum("i,eii->e", pmf, p)
        bit_errors = bit_errors + np.einsum("i,eij,ij->e", pmf, p, hamming)
    return 1.0 - correct, bit_errors / geom.k


def _union_rates(geom: _Geometry, esn0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Union bounds on SER and BER over all pairs of points."""
    sigma = np.sqrt(_n0(geom, esn0) / 2)
    d = np.abs(geom.points[:, None] - geom.points[None, :])
    labels = geom.c.bit_labels
    hamming = np.sum(labels[:, None, :] != labels[None, :, :], axis=-1)
    off = ~np.eye(geom.c.order, dtype=bool)
    pair = np.where(off[None], _q(d[None] / (2 * sigma[:, None, None])), 0.0)
    ser = np.einsum("i,eij->e", geom.pmf, pair)
    ber = np.einsum("i,eij,ij->e", geom.pmf, pair, hamming) / geom.k
    return np.minimum(ser, 1.0), np.minimum(ber, 1.0)


def _rates(constellation: Constellation, esn0_db: Any, name: str) -> tuple:
    if not isinstance(constellation, Constellation):
        raise TypeError(
            f"{name}: constellation must be a Constellation, got "
            f"{type(constellation).__name__}."
        )
    esn0, scalar = _esn0(esn0_db, name)
    geom = _Geometry(constellation)
    if geom.axes is not None:
        logger.debug("%s exact, %d axis/axes.", name, len(geom.axes))
        ser, ber = _exact_rates(geom, esn0)
    else:
        logger.debug("%s union bound over all %d pairs.", name, constellation.order)
        ser, ber = _union_rates(geom, esn0)
    return ser, ber, scalar, np.shape(esn0_db)


def awgn_ser(constellation: Constellation, *, esn0_db: Any) -> float | np.ndarray:
    """Symbol error rate of nearest-point decisions over AWGN.

    Exact for one-dimensional and separable constellations, a union bound
    over all pairs otherwise (see the module notes).

    Parameters
    ----------
    constellation : Constellation
        Points, labels and prior; ``Es`` is its average energy under the
        prior.
    esn0_db : float or array_like
        Es/N0 of the channel in dB.

    Returns
    -------
    float or numpy.ndarray
        The SER: a float for a scalar ``esn0_db``, an array of its shape
        otherwise.

    Raises
    ------
    TypeError
        If ``constellation`` is not a :class:`~commkit.mapping.Constellation`.
    ValueError
        If ``esn0_db`` is not finite.

    Notes
    -----
    Per axis of ``L`` levels ``a_i`` with prior ``p_i``, noise of variance
    ``N0/2``, decided at the midpoints ``b_j`` between sorted levels:

    .. math::

        P(j \\mid i) = Q\\Big(\\frac{b_{j-1} - a_i}{\\sigma}\\Big)
                    - Q\\Big(\\frac{b_j - a_i}{\\sigma}\\Big),
        \\qquad
        \\mathrm{SER} = 1 - \\prod_{\\text{axes}} \\sum_i p_i P(i \\mid i).

    The union bound is
    ``Σ_i p_i Σ_{j≠i} Q(|x_i - x_j| / (2σ))``, ``σ² = N0/2``.

    Examples
    --------
    >>> awgn_ser(Constellation.qam(16), esn0_db=15.0)
    """
    ser, _, scalar, shape = _rates(constellation, esn0_db, "awgn_ser()")
    return _out(ser, scalar, shape)


def awgn_ber(constellation: Constellation, *, esn0_db: Any) -> float | np.ndarray:
    """Bit error rate of nearest-point decisions over AWGN.

    The expected number of label bits that differ between the sent and the
    decided point, over ``k``.  Exact for one-dimensional and separable
    constellations, a union bound over all pairs otherwise (see the module
    notes).

    Parameters
    ----------
    constellation : Constellation
        Points, labels and prior; ``Es`` is its average energy under the
        prior.
    esn0_db : float or array_like
        Es/N0 of the channel in dB.

    Returns
    -------
    float or numpy.ndarray
        The BER: a float for a scalar ``esn0_db``, an array of its shape
        otherwise.

    Raises
    ------
    TypeError
        If ``constellation`` is not a :class:`~commkit.mapping.Constellation`.
    ValueError
        If ``esn0_db`` is not finite.

    Notes
    -----
    With the interval probabilities of :func:`awgn_ser` and the Hamming
    distances ``d_H`` of the labels,
    ``BER = Σ_axes Σ_i Σ_j p_i P(j | i) d_H(i, j) / k``; the union bound is
    ``Σ_i p_i Σ_{j≠i} d_H(i, j) Q(|x_i - x_j| / (2σ)) / k``.

    Examples
    --------
    >>> awgn_ber(Constellation.pam(4, unipolar=True), esn0_db=18.0)
    """
    _, ber, scalar, shape = _rates(constellation, esn0_db, "awgn_ber()")
    return _out(ber, scalar, shape)


def _nodes(order: int, n0: float) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Hermite nodes for noise of variance ``N0/2`` and their weights
    (summing to 1)."""
    t, w = np.polynomial.hermite.hermgauss(order)
    return math.sqrt(n0) * t, w / math.sqrt(math.pi)


def _axis_terms(levels: np.ndarray, n0: float) -> np.ndarray:
    """``-(|x_i - x_j + n|² - |n|²)/N0`` on one axis, ``(L, L, K)``."""
    n, _ = _nodes(_ORDER_1D, n0)
    d = levels[:, None, None] - levels[None, :, None]
    return -(d * d + 2.0 * d * n[None, None, :]) / n0


def _plane_rate(
    points: np.ndarray, pmf: np.ndarray, labels: np.ndarray | None, n0: float
) -> float:
    """MI (``labels=None``) or the GMI loss in the plane, by the product
    rule, chunked over the sent points to bound the ``(i, j, node)`` array."""
    n, w = _nodes(_ORDER_2D, n0)
    noise = (n[:, None] + 1j * n[None, :]).ravel()
    weights = (w[:, None] * w[None, :]).ravel()
    m = points.size
    chunk = max(1, _MAX_ELEMENTS // (m * noise.size))
    total = 0.0
    for start in range(0, m, chunk):
        rows = slice(start, min(start + chunk, m))
        d = points[rows, None, None] - points[None, :, None]
        terms = -(np.abs(d) ** 2 + 2.0 * (d * np.conj(noise)).real) / n0
        if labels is None:
            total += _mi_terms(terms, weights, pmf, pmf[rows])
        else:
            total += _gmi_loss(terms, weights, pmf, labels, rows)
    return total


def _class_sums(
    terms: np.ndarray, pmf: np.ndarray, labels: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """``log Σ_j p_j e^{terms_ij}`` over all candidates and, per bit and
    bit value, over the candidates carrying it: ``(i, K)`` and
    ``(i, K, k, 2)``.  One exponential after subtracting the maximum over
    ``j`` (at most about 230 above the sent point's own term, 0, for
    Gauss-Hermite nodes up to 15, so the sent point's class never
    underflows), then one matrix product over ``j``."""
    top = terms.max(axis=1)  # (i, K)
    e = np.exp(terms - top[:, None, :])
    if labels is None:
        weights = pmf[:, None]
    else:
        bits = labels.astype(bool)
        weights = np.concatenate(
            (pmf[:, None], pmf[:, None] * ~bits, pmf[:, None] * bits), axis=1
        )  # (j, 1 + 2k): all, then bit value 0 per bit, then value 1
    # The class of the bit value the sent point does not carry can underflow
    # to 0 at high SNR; its log (-inf) is never used.
    with np.errstate(divide="ignore"):
        sums = np.log(np.matmul(e.transpose(0, 2, 1), weights)) + top[:, :, None]
    return sums[:, :, 0], sums[:, :, 1:]


def _mi_terms(
    terms: np.ndarray, weights: np.ndarray, pmf: np.ndarray, sent: np.ndarray
) -> float:
    """``-Σ_i p_i E[log2 Σ_j p_j exp(terms_ij)]`` over the sent points of
    ``terms`` (prior ``sent``)."""
    total, _ = _class_sums(terms, pmf, None)
    return float(-np.sum(sent[:, None] * weights[None, :] * total) / math.log(2.0))


def _gmi_loss(
    terms: np.ndarray,
    weights: np.ndarray,
    pmf: np.ndarray,
    labels: np.ndarray,
    rows: slice = slice(None),
) -> float:
    """``Σ_b H(B_b | Y)`` in bits over the sent points ``rows`` of ``terms``:
    per sent point ``i`` and bit ``b``,
    ``E[log2(Σ_j p_j e_j / Σ_{j: c_jb = c_ib} p_j e_j)]``."""
    total, by_value = _class_sums(terms, pmf, labels)
    k = labels.shape[1]
    sent = labels[rows].astype(np.int64)  # (i, k)
    idx = sent * k + np.arange(k)[None, :]  # value 0 block, then value 1
    own = np.take_along_axis(by_value, idx[:, None, :], axis=2)  # (i, K, k)
    # The sent point's own term is 0, so its class sum is at least p_i; at
    # nodes far out (weights below 1e-90) the shared maximum can underflow
    # it, and the bound keeps the product with the weight finite.
    with np.errstate(divide="ignore"):
        floor = np.log(pmf[rows])
    own = np.maximum(own, floor[:, None, None])
    loss = np.sum(
        pmf[rows, None, None] * weights[None, :, None] * (total[:, :, None] - own)
    )
    return float(loss / math.log(2.0))


def _information(
    constellation: Constellation, esn0_db: Any, name: str, generalized: bool
) -> float | np.ndarray:
    if not isinstance(constellation, Constellation):
        raise TypeError(
            f"{name}: constellation must be a Constellation, got "
            f"{type(constellation).__name__}."
        )
    esn0, scalar = _esn0(esn0_db, name)
    geom = _Geometry(constellation)
    out = np.empty_like(esn0)
    logger.debug(
        "%s Gauss-Hermite, %s.",
        name,
        f"{_ORDER_1D} nodes per axis"
        if geom.axes is not None
        else f"{_ORDER_2D}² nodes",
    )
    for e, n0 in enumerate(_n0(geom, esn0)):
        if geom.axes is not None:
            value = 0.0
            for levels, labels, pmf, _bits in geom.axes:
                terms = _axis_terms(levels, float(n0))
                weights = _nodes(_ORDER_1D, float(n0))[1]
                if generalized:
                    value += _entropy(pmf) - _gmi_loss(terms, weights, pmf, labels)
                else:
                    value += _mi_terms(terms, weights, pmf, pmf)
        elif generalized:
            value = constellation.entropy - _plane_rate(
                geom.points, geom.pmf, constellation.bit_labels, float(n0)
            )
        else:
            value = _plane_rate(geom.points, geom.pmf, None, float(n0))
        out[e] = value
    if logger.isEnabledFor(logging.DEBUG) and scalar:
        logger.debug("%s %.6f bits/symbol.", name, float(out[0]))
    return _out(out, scalar, np.shape(esn0_db))


def _entropy(pmf: np.ndarray) -> float:
    p = pmf[pmf > 0]
    return float(-np.sum(p * np.log2(p)))


def awgn_mi(constellation: Constellation, *, esn0_db: Any) -> float | np.ndarray:
    """Mutual information between the points and the AWGN channel's output.

    The rate of a receiver that uses the symbol likelihoods, at most
    ``H(X)`` (``log2 M`` for a uniform prior).

    Parameters
    ----------
    constellation : Constellation
        Points and prior; ``Es`` is its average energy under the prior.
        Complex points see complex noise of variance ``N0``, real points
        real noise of variance ``N0/2``.
    esn0_db : float or array_like
        Es/N0 of the channel in dB.

    Returns
    -------
    float or numpy.ndarray
        MI in bits per symbol: a float for a scalar ``esn0_db``, an array of
        its shape otherwise.

    Raises
    ------
    TypeError
        If ``constellation`` is not a :class:`~commkit.mapping.Constellation`.
    ValueError
        If ``esn0_db`` is not finite.

    Notes
    -----
    With ``y = x_i + n``:

    .. math::

        I(X; Y) = -\\sum_i p_i\\,
            \\mathbb{E}_n\\Big[\\log_2 \\sum_j p_j
            e^{-(|x_i - x_j + n|^2 - |n|^2)/N_0}\\Big],

    the expectation by Gauss-Hermite quadrature (module notes).  For a
    separable constellation the axes are independent and their rates add.

    Examples
    --------
    >>> awgn_mi(Constellation.qam(16), esn0_db=10.0)
    """
    return _information(constellation, esn0_db, "awgn_mi()", generalized=False)


def awgn_gmi(constellation: Constellation, *, esn0_db: Any) -> float | np.ndarray:
    """Generalized mutual information (bit-metric decoding rate) over AWGN.

    ``H(X) - Σ_b H(B_b | Y)`` with exact bit LLRs including the prior, the
    definition of :func:`gmi`.

    Parameters
    ----------
    constellation : Constellation
        Points, labels and prior; ``Es`` is its average energy under the
        prior.  Complex points see complex noise of variance ``N0``, real
        points real noise of variance ``N0/2``.
    esn0_db : float or array_like
        Es/N0 of the channel in dB.

    Returns
    -------
    float or numpy.ndarray
        GMI in bits per symbol: a float for a scalar ``esn0_db``, an array
        of its shape otherwise.

    Raises
    ------
    TypeError
        If ``constellation`` is not a :class:`~commkit.mapping.Constellation`.
    ValueError
        If ``esn0_db`` is not finite.

    Notes
    -----
    .. math::

        \\mathrm{GMI} = H(X) - \\sum_b \\sum_i p_i\\,
            \\mathbb{E}_n\\Big[\\log_2
            \\frac{\\sum_j p_j e_{ij}}{\\sum_{j:\\, c_{jb} = c_{ib}} p_j e_{ij}}\\Big],
        \\quad e_{ij} = e^{-(|x_i - x_j + n|^2 - |n|^2)/N_0},

    by Gauss-Hermite quadrature (module notes).  Bits of a separable
    constellation depend on one axis each, so the axes' rates add.  At low
    SNR the GMI tends to ``H(X) - Σ_b H(B_b)``: 0 for a uniform prior, and
    negative for a shaped one, whose label bits are dependent.

    Examples
    --------
    >>> awgn_gmi(Constellation.qam(64).shaped(entropy=5.0), esn0_db=12.0)
    """
    return _information(constellation, esn0_db, "awgn_gmi()", generalized=True)
