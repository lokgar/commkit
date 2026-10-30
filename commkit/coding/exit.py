"""EXIT charts: :class:`ExitCurve`, :class:`ExitChart` and their sources.

An extrinsic-information transfer (EXIT) curve describes one soft-in
soft-out component of an iterative receiver: the mutual information per bit
of its extrinsic output LLRs, ``I_E``, as a function of the information of
its a-priori input LLRs, ``I_A``.  Two components that exchange extrinsic
information through an interleaver form an :class:`ExitChart`
(:func:`exit_chart`): the first curve against the second with its axes
swapped, and the decoding trajectory between them.

Sources of curves:

- :func:`decoder_exit_curve`, measured by Monte Carlo through
  :func:`~commkit.coding.decode`, for any code whose decoder returns
  a-posteriori LLRs (:class:`~commkit.coding.LDPCCode`,
  :class:`~commkit.coding.ConvolutionalCode` with
  :class:`~commkit.coding.BCJR`, :class:`~commkit.coding.Chase`, and the
  codes built from them);
- :func:`demapper_exit_curve`, measured by Monte Carlo, for the demapper of
  a :class:`~commkit.mapping.Constellation` on AWGN that takes a-priori
  LLRs on the bits of each label (iterative demapping, BICM-ID);
- :func:`ldpc_exit_curves`, in closed form, for the variable and check
  nodes of an LDPC code, with its threshold, :func:`ldpc_threshold`.  This
  is the only family with closed-form curves: they follow from the degrees
  of the code's graph.  Every other code is measured through its decoder.

Conventions.  Mutual information is per bit, in ``[0, 1]``.  A-priori LLRs
are drawn consistent Gaussian, ``N(±σ²/2, σ²)`` (positive for bit 0), whose
information is ``J(σ)``.  Measured information comes from histograms of the
LLRs given their bits, so it holds for LLRs that are not true
log-likelihood ratios (max-log, min-sum, Chase-Pyndiah outputs).  Monte
Carlo draws follow the library's rule: transmitted bits and symbols on the
host from ``rng``, a-priori LLRs and channel noise on ``device``.  Curves
and charts are host arrays.

Limitations, common to all EXIT analysis:

- The chart predicts iterative decoding with long interleavers, where the
  exchanged LLRs are independent and Gaussian-like.  Short blocks and
  real a-priori LLRs (which are not Gaussian) converge later than
  predicted.
- A tunnel is open when every iteration gains information up to an
  information of 0.99; the curves are interpolated linearly between their
  points, and a measured tunnel that is nearly closed is within the Monte
  Carlo error (about ``1/sqrt(bits per point)``) of closing.
- Measured curves have no threshold: sweep the channel and chart each
  point.
- Not covered: decoders that return only bits (turbo, polar SC and SCL,
  algebraic, syndrome and maximum-likelihood decoding) have no measured
  curve; the turbo code's constituent decoders are not exposed, so its
  inner exchange cannot be charted; a demapper on a channel other than
  AWGN, or with shaped symbols, needs its own model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import cache
from typing import Any

import numpy as np

from .. import _random
from ..backend import ArrayType, _module_for, dispatch, to_device
from ..logger import logger
from .base import _decode_blocks, _encode_blocks, _resolve_method
from .ldpc import LDPCCode

__all__ = [
    "ExitChart",
    "ExitCurve",
    "decoder_exit_curve",
    "demapper_exit_curve",
    "exit_chart",
    "ldpc_exit_curves",
    "ldpc_threshold",
]

# Information at which the iteration counts as converged.  Above it, the curves
# of an LDPC code with degree-1 variable nodes meet below (1, 1): such a node
# never gets more than its channel information (the last parity bit of DVB-S2).
_CONVERGED = 0.99

# Grid on which a chart checks its tunnel, and the trajectory's step limit.
_TUNNEL_GRID = np.linspace(0.0, _CONVERGED, 2000)
_MAX_STEPS = 10_000

# Bits per point of a measured curve when the caller gives no size.
_DEFAULT_BITS = 1 << 17

# Histogram of the information estimate: bins over the central 99.9 % of |L|;
# the rest falls into the outer bins with its sign.
_BINS = 200

# Largest a-priori information drawn: J⁻¹(1) is infinite.
_MAX_A_PRIORI = 1.0 - 1e-6

# Elements per chunk of the demapper's (symbols x points) metrics.
_DEMAP_CHUNK = 1 << 22


@dataclass(frozen=True, eq=False)
class ExitCurve:
    """The transfer curve of one soft-in soft-out component.

    Parameters
    ----------
    a_priori : array_like
        Information per bit of the a-priori LLRs, ``(G,)``, increasing, in
        ``[0, 1]``.
    extrinsic : array_like
        Information per bit of the extrinsic LLRs at each ``a_priori``,
        ``(G,)``, in ``[0, 1]``.
    label : str, default ""
        What the curve describes, for legends and log messages.

    Notes
    -----
    The arrays are stored as read-only float64 host copies.  Build curves
    from your own measurements to chart them with :func:`exit_chart`.
    """

    a_priori: np.ndarray
    extrinsic: np.ndarray
    label: str = ""

    def __post_init__(self) -> None:
        a = np.array(to_device(self.a_priori, "cpu"), dtype=np.float64)
        e = np.array(to_device(self.extrinsic, "cpu"), dtype=np.float64)
        if a.ndim != 1 or a.shape != e.shape or a.size < 2:
            raise ValueError(
                "ExitCurve: a_priori and extrinsic must be 1-D of one length >= 2, "
                f"got {a.shape} and {e.shape}."
            )
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(e))):
            raise ValueError("ExitCurve: a_priori and extrinsic must be finite.")
        if a[0] < 0 or a[-1] > 1 or np.any(np.diff(a) <= 0):
            raise ValueError("ExitCurve: a_priori must increase within [0, 1].")
        if e.min() < -1e-9 or e.max() > 1 + 1e-9:
            raise ValueError("ExitCurve: extrinsic must lie in [0, 1].")
        for name, value in (("a_priori", a), ("extrinsic", np.clip(e, 0.0, 1.0))):
            value.setflags(write=False)
            object.__setattr__(self, name, value)


@dataclass(frozen=True, eq=False)
class ExitChart:
    """Two EXIT curves and the decoding trajectory between them.

    Attributes
    ----------
    first : ExitCurve
        Drawn as ``(a_priori, extrinsic)``: its output is the second
        curve's input.
    second : ExitCurve
        Drawn with its axes swapped, ``(extrinsic, a_priori)``.
    converges : bool
        Whether the tunnel between the curves is open: one exchange through
        both components gains information everywhere up to an information
        of 0.99 at the first component's input.
    trajectory : numpy.ndarray
        The decoding trajectory, ``(T, 2)``: from ``(0, 0)``, one step up to
        the first curve and one across to the second per iteration, until
        the first component's input reaches 0.99 or stops growing.
    iterations : int
        Exchanges in the trajectory: to convergence, or to where it stalls.
    narrowest : float
        The first component's input information where the tunnel is
        narrowest if it is open, or where it first closes if it is not: the
        crossing of the curves, which the trajectory approaches and which
        sets the information the receiver ends with.
    """

    first: ExitCurve
    second: ExitCurve
    converges: bool = field(init=False)
    trajectory: np.ndarray = field(init=False)
    iterations: int = field(init=False)
    narrowest: float = field(init=False)

    def __post_init__(self) -> None:
        for name in ("first", "second"):
            curve = getattr(self, name)
            if not isinstance(curve, ExitCurve):
                raise TypeError(
                    f"ExitChart: {name} must be an ExitCurve, got "
                    f"{type(curve).__name__}."
                )
            if curve.a_priori[0] > 0 or curve.a_priori[-1] < _CONVERGED:
                raise ValueError(
                    f"ExitChart: the {name} curve must span a-priori information "
                    f"from 0 to at least {_CONVERGED}, got "
                    f"[{curve.a_priori[0]:.3g}, {curve.a_priori[-1]:.3g}]."
                )
        gain = self._exchange(_TUNNEL_GRID) - _TUNNEL_GRID
        closed = gain <= 0
        k = int(np.argmax(closed)) if closed.any() else int(np.argmin(gain))
        path = [(0.0, 0.0)]
        x = 0.0
        for _ in range(_MAX_STEPS):
            y = float(np.interp(x, self.first.a_priori, self.first.extrinsic))
            x_next = float(np.interp(y, self.second.a_priori, self.second.extrinsic))
            path += [(x, y), (x_next, y)]
            if x_next >= _CONVERGED or x_next - x < 1e-6:
                break
            x = x_next
        trajectory = np.asarray(path)
        trajectory.setflags(write=False)
        object.__setattr__(self, "converges", not bool(closed.any()))
        object.__setattr__(self, "trajectory", trajectory)
        object.__setattr__(self, "iterations", (len(path) - 1) // 2)
        object.__setattr__(self, "narrowest", float(_TUNNEL_GRID[k]))

    def _exchange(self, x: Any) -> Any:
        """The first component's input after one exchange through both."""
        y = np.interp(x, self.first.a_priori, self.first.extrinsic)
        return np.interp(y, self.second.a_priori, self.second.extrinsic)


def exit_chart(first: ExitCurve, second: ExitCurve) -> ExitChart:
    """EXIT chart of two components that exchange extrinsic information.

    Parameters
    ----------
    first : ExitCurve
        The component whose output feeds the second: the demapper of
        iterative demapping, the variable nodes of an LDPC code, the inner
        code of a serial concatenation.
    second : ExitCurve
        The component whose output feeds back into the first: the decoder,
        the check nodes, the outer code.

    Returns
    -------
    ExitChart
        The curves, whether the tunnel is open, and the trajectory.

    Raises
    ------
    TypeError
        If either argument is not an :class:`ExitCurve`.
    ValueError
        If a curve does not span a-priori information from 0 to 0.99.

    See Also
    --------
    commkit.plotting.plot_exit_chart : Draws the result.

    Notes
    -----
    The first component's output must be the second's input and the reverse,
    with an interleaver in between, as in iterative demapping and decoding
    (demapper, decoder), serially concatenated codes (inner, outer), or the
    two node types of an LDPC code (variable, check).  Parallel
    concatenations (turbo codes), where both components see the channel,
    chart each component against the other's swapped curve the same way.
    """
    chart = ExitChart(first, second)
    names = (
        chart.first.label or "the first curve",
        chart.second.label or "the second curve",
    )
    if chart.converges:
        gap = float(chart._exchange(chart.narrowest) - chart.narrowest)
        logger.info(
            "EXIT chart of %s against %s: tunnel open, narrowest at I_A = %.3f "
            "(gain %.4f per iteration); converges in %d iteration(s).",
            *names,
            chart.narrowest,
            gap,
            chart.iterations,
        )
    else:
        logger.info(
            "EXIT chart of %s against %s: tunnel closed at I_A = %.3f; the "
            "trajectory stalls there after %d iteration(s), with the second "
            "component's output at %.3f.",
            *names,
            chart.narrowest,
            chart.iterations,
            chart.trajectory[-1, 0],
        )
    return chart


def decoder_exit_curve(
    code: Any,
    *,
    method: Any = None,
    points: int = 11,
    num_blocks: int | None = None,
    rng: int | np.random.Generator | None = None,
    device: str = "cpu",
) -> ExitCurve:
    """Measure the EXIT curve of a decoder by Monte Carlo.

    For each a-priori information ``I_A`` on a grid from 0 to 1, consistent
    Gaussian LLRs of information ``I_A`` on the transmitted bits of random
    codewords are decoded; the extrinsic LLRs are the decoder's a-posteriori
    LLRs minus its input, and ``I_E`` their measured information.  This is
    the curve of the code as the outer component of a serial concatenation
    or of iterative demapping (the a-priori LLRs are its only input).

    Parameters
    ----------
    code : code object
        Any code of :mod:`commkit.coding` whose decoder returns a-posteriori
        LLRs (``DecodeResult.llrs``).
    method : decoder object, optional
        The decoder; ``None`` takes the code's default, as
        :func:`~commkit.coding.decode` does.
    points : int, default 11
        Points of the a-priori grid, from 0 to 1 inclusive.
    num_blocks : int, optional
        Codewords per point; ``None`` takes enough for 2^17 bits per point.
        The information's Monte Carlo error is about
        ``1/sqrt(num_blocks * n)``.
    rng : int or numpy.random.Generator, optional
        Seed or Generator of the information bits (drawn on the host) and
        the a-priori LLRs (drawn on ``device``).
    device : {"cpu", "gpu"}, default "cpu"
        Where the codewords are encoded and decoded.

    Returns
    -------
    ExitCurve
        ``I_E`` against ``I_A`` on the transmitted bits.

    Raises
    ------
    TypeError
        If ``code`` is not a code object, ``method`` does not decode it, or
        the decoder gives no a-posteriori LLRs.
    ValueError
        If ``points < 2``, ``num_blocks < 1`` or ``device`` is not
        ``"cpu"`` or ``"gpu"``.

    See Also
    --------
    exit_chart : Charts the curve against another.
    demapper_exit_curve : The curve of the component the decoder iterates with.

    Notes
    -----
    Limitations:

    - Decoders that return only bits have no curve: turbo
      (:class:`~commkit.coding.TurboIterations`), polar (SC, SCL),
      algebraic and syndrome decoding, maximum likelihood.
    - The extrinsic part is the output minus the input, which assumes the
      decoder returns a-posteriori LLRs that include its input (as LDPC
      belief propagation, BCJR and Chase-Pyndiah do).
    - An iterative decoder (LDPC) runs all its own iterations at every
      point: the curve is that of the whole decoder, not of one iteration.
    - A decoder that stops early on a satisfied check stops before its
      first iteration when the a-priori LLRs alone already form a valid
      codeword (at high ``I_A``); its output then equals its input and the
      extrinsic part is zero.  Pass ``MinSum(early_stop=False)`` or
      ``SumProduct(early_stop=False)``; a warning names the points affected.
    - Max-log, min-sum and Chase outputs are not true LLRs.  Their
      information is measured correctly, but a component that receives them
      as if they were (a demapper) loses part of it; exact decoders
      (``BCJR(method="exact")``, ``SumProduct``) avoid that.
    - For an exact a-posteriori decoder the area under the curve is
      ``1 - k / n``, which the log message reports as a check.
    """
    method = _resolve_method(code, method, "decoder_exit_curve()")
    _check_points(points, "decoder_exit_curve()")
    xp = _module_for(_check_device(device, "decoder_exit_curve()"))
    blocks = (
        max(1, math.ceil(_DEFAULT_BITS / code.n)) if num_blocks is None else num_blocks
    )
    if isinstance(blocks, bool) or not isinstance(blocks, int) or blocks < 1:
        raise ValueError(
            f"decoder_exit_curve(): num_blocks must be an integer >= 1, got {blocks}."
        )
    gen = _random.as_generator(rng)
    info = gen.integers(0, 2, (blocks, code.k), dtype=np.int8)
    bits = _encode_blocks(code, to_device(info, device))
    sign = (1 - 2 * bits).astype(np.float32)
    grid = np.linspace(0.0, 1.0, points)
    label = f"{method!r} on {code!r}"
    logger.debug(
        "Measuring the decoder EXIT curve of %s: %d points, %d codewords (%d bits) "
        "each, on %s.",
        label,
        points,
        blocks,
        bits.size,
        device,
    )
    extrinsic = np.empty(points)
    stopped = np.zeros(points, dtype=np.int64)  # blocks decoded in 0 iterations
    for i, ia in enumerate(grid):
        a_priori = _a_priori_llrs(sign, float(ia), gen, xp)
        res = _decode_blocks(code, a_priori, method)
        stopped[i] = int(xp.count_nonzero(res.iterations == 0))
        if res.llrs is None:
            raise TypeError(
                f"decoder_exit_curve(): {method!r} returns no a-posteriori LLRs, so "
                "it has no EXIT curve; use a soft-output decoder (MinSum or "
                "SumProduct for LDPC codes, BCJR, Chase)."
            )
        extrinsic[i] = _information(res.llrs - a_priori, bits)
        logger.debug("  I_A = %.3f: I_E = %.4f", ia, extrinsic[i])
    if stopped.any():
        first = int(np.argmax(stopped > 0))
        logger.warning(
            "decoder_exit_curve(): %r stopped before its first iteration in %d of "
            "%d codewords from I_A = %.2f on: their input was already a valid "
            "codeword, so their extrinsic output is zero and the curve drops "
            "there.  Disable early stopping (e.g. SumProduct(early_stop=False)).",
            method,
            int(stopped[-1]),
            blocks,
            grid[first],
        )
    curve = ExitCurve(grid, extrinsic, label)
    area = float(np.trapezoid(curve.extrinsic, grid))
    logger.info(
        "Decoder EXIT curve of %s: I_E = %.3f at I_A = 0 to %.3f at I_A = 1, area "
        "%.3f (1 - rate = %.3f for an exact a-posteriori decoder); %d bits per point.",
        label,
        curve.extrinsic[0],
        curve.extrinsic[-1],
        area,
        1.0 - code.k / code.n,
        bits.size,
    )
    return curve


def demapper_exit_curve(
    constellation: Any,
    *,
    noise_var: float,
    points: int = 11,
    num_symbols: int = 1 << 14,
    method: str = "exact",
    rng: int | np.random.Generator | None = None,
    device: str = "cpu",
) -> ExitCurve:
    """Measure the EXIT curve of a demapper on AWGN by Monte Carlo.

    Uniform random symbols pass an AWGN channel of variance ``noise_var``.
    For each a-priori information ``I_A``, consistent Gaussian a-priori LLRs
    of information ``I_A`` are drawn on every bit, and the demapper computes
    each bit's extrinsic LLR from the received symbol and the a-priori LLRs
    of the label's other bits; ``I_E`` is their measured information.

    Parameters
    ----------
    constellation : Constellation
        The constellation and its bit labels; uniform (not shaped).
    noise_var : float
        Noise variance per complex symbol, on the constellation's scale, as
        :func:`~commkit.mapping.compute_llr` takes it.
    points : int, default 11
        Points of the a-priori grid, from 0 to 1 inclusive.
    num_symbols : int, default 16384
        Symbols per point; ``num_symbols * bits_per_symbol`` bits.
    method : {"exact", "maxlog"}, default "exact"
        Log-sum-exp over the points of each bit value, or only the largest
        term (the max-log approximation of a receiver).
    rng : int or numpy.random.Generator, optional
        Seed or Generator of the symbols (drawn on the host), and of the
        noise and the a-priori LLRs (drawn on ``device``).
    device : {"cpu", "gpu"}, default "cpu"
        Where the channel and the demapper run.

    Returns
    -------
    ExitCurve
        ``I_E`` against ``I_A``, averaged over the bits of the label.

    Raises
    ------
    TypeError
        If ``constellation`` is not a :class:`~commkit.mapping.Constellation`.
    ValueError
        If the constellation is shaped, ``noise_var <= 0``, ``points < 2``,
        ``num_symbols < 1``, ``method`` is unknown or ``device`` is not
        ``"cpu"`` or ``"gpu"``.

    See Also
    --------
    decoder_exit_curve : The curve of the decoder the demapper iterates with.
    exit_chart : Charts the two.

    Notes
    -----
    At ``I_A = 0`` the curve is the information per bit of the demapper's
    plain LLRs, the GMI divided by the bits per symbol
    (:func:`commkit.metrics.gmi`).  Its rise towards ``I_A = 1`` is what
    iterative demapping can gain: none for Gray-labelled QPSK (flat), little
    for Gray-labelled QAM, more for set-partitioning or natural labels.

    Limitations:

    - AWGN only, at one noise variance; other channels need their own
      demapper model.
    - Uniform symbols: with a shaped constellation the bits of a label are
      not independent, which the a-priori model assumes.
    - The demapper computes each symbol's distances to all ``M`` points,
      ``num_symbols * M`` per point of the curve, in chunks; it does not use
      the compiled LLR kernels, whose speed is unaffected.
    """
    from ..mapping import Constellation

    name = "demapper_exit_curve()"
    if not isinstance(constellation, Constellation):
        raise TypeError(
            f"{name}: constellation must be a Constellation, got "
            f"{type(constellation).__name__}."
        )
    if constellation.pmf is not None:
        raise ValueError(
            f"{name}: the constellation is shaped; the demapper's a-priori model "
            "assumes independent bits, which holds for uniform symbols only."
        )
    nv = float(noise_var)
    if not nv > 0:
        raise ValueError(f"{name}: noise_var must be positive, got {noise_var}.")
    _check_points(points, name)
    if (
        isinstance(num_symbols, bool)
        or not isinstance(num_symbols, int)
        or num_symbols < 1
    ):
        raise ValueError(
            f"{name}: num_symbols must be an integer >= 1, got {num_symbols}."
        )
    if method not in ("exact", "maxlog"):
        raise ValueError(f'{name}: method must be "exact" or "maxlog", got {method!r}.')
    xp = _module_for(_check_device(device, name))

    gen = _random.as_generator(rng)
    index = to_device(gen.integers(0, constellation.order, num_symbols), device)
    pts = to_device(constellation.points.astype(np.complex64), device)
    labels = to_device(constellation.bit_labels.astype(np.int8), device)
    noise = _random.complex_normal(
        gen, (num_symbols,), std=math.sqrt(nv / 2), dtype=np.float32, xp=xp
    )
    received = (pts[index] + noise).astype(np.complex64)
    bits = labels[index]
    sign = (1 - 2 * bits).astype(np.float32)
    es = float(np.mean(np.abs(constellation.points) ** 2))
    label = f"{constellation!r} demapper, Es/N0 = {10 * math.log10(es / nv):.1f} dB"
    if method == "maxlog":
        label += ", max-log"
    logger.debug(
        "Measuring the demapper EXIT curve of %s: %d points, %d symbols each, on %s.",
        label,
        points,
        num_symbols,
        device,
    )
    grid = np.linspace(0.0, 1.0, points)
    extrinsic = np.empty(points)
    for i, ia in enumerate(grid):
        a_priori = _a_priori_llrs(sign, float(ia), gen, xp)
        llrs = _demap_extrinsic(received, pts, labels, a_priori, nv, method, xp)
        extrinsic[i] = _information(llrs, bits)
        logger.debug("  I_A = %.3f: I_E = %.4f", ia, extrinsic[i])
    curve = ExitCurve(grid, extrinsic, label)
    logger.info(
        "Demapper EXIT curve of %s: I_E = %.3f without a-priori information (the "
        "information per bit of its LLRs) to %.3f with the other bits known; "
        "%d bits per point.",
        label,
        curve.extrinsic[0],
        curve.extrinsic[-1],
        bits.size,
    )
    return curve


# -----------------------------------------------------------------------------
# LDPC codes: the curves of the nodes in closed form (Gaussian approximation)
# -----------------------------------------------------------------------------

# The curves are float64 host arrays of grid size times the number of degrees,
# not sample-rate data, so the precision audit's reports of them are expected.


def ldpc_exit_curves(
    code: LDPCCode, *, channel_mi: float, points: int = 1001
) -> tuple[ExitCurve, ExitCurve]:
    r"""EXIT curves of the variable and check nodes of an LDPC code, in closed form.

    With ``J(σ)`` the information of a consistent Gaussian LLR of standard
    deviation ``σ``, ``λ_d`` and ``ρ_d`` the fractions of edges at variable
    and check nodes of degree ``d``, and ``σ_ch = J⁻¹(channel_mi)`` (0 for
    punctured variables):

    .. math::

        I_{E,V}(I_A) &= \sum_d \lambda_d\,
            J\!\left(\sqrt{(d-1)\,J^{-1}(I_A)^2 + \sigma_{ch}^2}\right) \\
        I_{E,C}(I_A) &= \sum_d \rho_d\,
            \left[1 - J\!\left(\sqrt{d-1}\,J^{-1}(1 - I_A)\right)\right]

    Parameters
    ----------
    code : LDPCCode
        The code; only the degrees of its parity-check matrix and its
        punctured bits enter.
    channel_mi : float
        Mutual information per bit between the transmitted bits and their
        channel LLRs, in ``(0, 1)``.  For BPSK on AWGN at ``Es/N0`` it is
        ``J(sqrt(8 Es/N0))``; for a BICM link with uniform symbols, the GMI
        divided by the bits per symbol, which is the NGMI
        (:func:`commkit.metrics.ngmi`).
    points : int, default 1001
        Points of the a-priori grid of each curve, from 0 to 1.

    Returns
    -------
    variable : ExitCurve
        The variable nodes, which see the channel: the first curve of
        :func:`exit_chart`.
    check : ExitCurve
        The check nodes: the second curve.

    Raises
    ------
    TypeError
        If ``code`` is not an :class:`LDPCCode`.
    ValueError
        If ``channel_mi`` is not in ``(0, 1)`` or ``points < 2``.

    See Also
    --------
    ldpc_threshold : The channel information at which their tunnel opens.
    decoder_exit_curve : The measured curve of the whole decoder.

    Notes
    -----
    Limitations of the Gaussian approximation:

    - It predicts sum-product decoding of a long code with a cycle-free
      graph: within about 0.1 dB of the exact density-evolution threshold
      on AWGN (1.1 dB ``Eb/N0`` for the regular (3, 6) ensemble).  Min-sum
      decoding and short codes need more.
    - The channel enters only through its information per bit: the LLRs of
      the bits of a QAM label are not Gaussian, and their spread across bit
      levels is averaged out.
    - ``J`` is integrated numerically: the curves of a capacity-approaching
      code run so close together that the closed-form fits of ``J``
      (within 1e-3) move its threshold by 0.02 bit.

    References
    ----------
    S. ten Brink, G. Kramer, A. Ashikhmin, "Design of low-density
    parity-check codes for modulation and detection," IEEE Trans. Commun.
    52(4), 2004.

    Examples
    --------
    >>> code = LDPCCode.dvbs2("1/2", "short")
    >>> chart = exit_chart(*ldpc_exit_curves(code, channel_mi=0.52))
    >>> chart.converges
    True
    """
    name = "ldpc_exit_curves()"
    _check_ldpc(code, name)
    mi = float(channel_mi)
    if not 0.0 < mi < 1.0:
        raise ValueError(f"{name}: channel_mi must be in (0, 1), got {mi}.")
    if isinstance(points, bool) or not isinstance(points, int) or points < 2:
        raise ValueError(f"{name}: points must be an integer >= 2, got {points}.")
    var_groups, check_groups = _edge_degrees(code)
    grid = np.linspace(0.0, 1.0, points)
    logger.info(
        "EXIT curves of %r (Gaussian approximation) at channel information %.4f "
        "per bit: variable-node degrees %s, check-node degrees %s, %d punctured "
        "bit(s).",
        code,
        mi,
        _degree_list(var_groups[0]),
        _degree_list(check_groups[0]),
        code.punctured,
    )
    return (
        ExitCurve(
            grid,
            _variable_curve(grid, float(_j_inv(mi)), var_groups),
            f"variable nodes, I_ch = {mi:.3f}",
        ),
        ExitCurve(grid, _check_curve(grid, check_groups), "check nodes"),
    )


def ldpc_threshold(code: LDPCCode) -> float:
    """Belief-propagation threshold of an LDPC code in the Gaussian approximation.

    The smallest channel information per bit at which the tunnel between
    the curves of :func:`ldpc_exit_curves` opens, found by bisection to
    1e-6.

    Parameters
    ----------
    code : LDPCCode
        The code.

    Returns
    -------
    float
        The threshold, in ``(0, 1)``, above the code rate; NaN if the
        tunnel never opens.

    Raises
    ------
    TypeError
        If ``code`` is not an :class:`LDPCCode`.

    See Also
    --------
    ldpc_exit_curves : The curves, and the limitations of the approximation.

    Notes
    -----
    The tunnel is checked on the closed-form curves up to an information of
    0.99, as :func:`exit_chart` checks it.  Above 0.99 the remaining
    iterations are few, and the curves of a code with degree-1 variable
    nodes (the last parity bit of DVB-S2) meet below ``(1, 1)``: such a node
    never gets more than its channel information, which leaves one
    unreliable bit, not a failed decoding.  For the DVB-S2 rate-1/2 short
    code the threshold, 0.498, is where sum-product frames of QPSK start to
    decode (an NGMI of about 0.5).
    """
    _check_ldpc(code, "ldpc_threshold()")
    var_groups, check_groups = _edge_degrees(code)
    lo, hi = 0.0, 1.0 - 1e-9
    if not _tunnel_open(float(_j_inv(hi)), var_groups, check_groups):
        logger.warning(
            "%r: the EXIT tunnel never opens in the Gaussian approximation.", code
        )
        return math.nan
    while hi - lo > 1e-6:
        mid = 0.5 * (lo + hi)
        if _tunnel_open(float(_j_inv(mid)), var_groups, check_groups):
            hi = mid
        else:
            lo = mid
    logger.info(
        "Belief-propagation threshold of %r (Gaussian approximation): channel "
        "information %.4f per bit, %.4f above the rate %.4f.",
        code,
        hi,
        hi - code.k / code.n,
        code.k / code.n,
    )
    return hi


def _check_ldpc(code: Any, name: str) -> None:
    if not isinstance(code, LDPCCode):
        raise TypeError(f"{name}: code must be an LDPCCode, got {type(code).__name__}.")


def _degree_list(others: np.ndarray) -> str:
    """Node degrees from the per-group ``degree - 1``, as ``"2, 3, 8"``."""
    return ", ".join(str(int(d) + 1) for d in np.unique(others))


# (degree - 1, channel) per group of variable-node edges with their weight,
# and (degree - 1) per group of check-node edges with their weight.
_Groups = tuple[np.ndarray, np.ndarray, np.ndarray]


def _edge_degrees(code: LDPCCode) -> tuple[_Groups, tuple[np.ndarray, np.ndarray]]:
    """Edge-perspective degree distributions of ``code``'s graph."""
    var_deg = np.bincount(code.variables, minlength=code.num_variables)
    chk_deg = np.bincount(code.checks, minlength=code.num_checks)
    edge_var = var_deg[code.variables]
    has_channel = code.variables < code.n  # punctured bits get no channel LLR
    keys, counts = np.unique(
        np.stack([edge_var, has_channel.astype(np.int64)]), axis=1, return_counts=True
    )
    var_groups = (
        (keys[0] - 1).astype(np.float64),
        keys[1].astype(bool),
        counts / code.variables.size,
    )
    degrees, counts = np.unique(chk_deg[code.checks], return_counts=True)
    check_groups = ((degrees - 1).astype(np.float64), counts / code.checks.size)
    return var_groups, check_groups


def _variable_curve(info: np.ndarray, sigma_ch: float, groups: _Groups) -> np.ndarray:
    """Extrinsic information of the variable nodes at a-priori ``info``."""
    others, channel, weight = groups
    a = _j_inv(info)[..., None] ** 2
    sigma2 = others * a + np.where(channel, sigma_ch**2, 0.0)
    return np.sum(weight * _j(np.sqrt(sigma2)), axis=-1)


def _check_curve(info: np.ndarray, groups: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Extrinsic information of the check nodes at a-priori ``info``."""
    others, weight = groups
    a = _j_inv(1.0 - np.asarray(info, dtype=np.float64))[..., None]
    return np.sum(weight * (1.0 - _j(np.sqrt(others) * a)), axis=-1)


def _tunnel_open(sigma_ch: float, var_groups: _Groups, check_groups: tuple) -> bool:
    """Whether one iteration gains information everywhere below 0.99."""
    after = _check_curve(
        _variable_curve(_TUNNEL_GRID, sigma_ch, var_groups), check_groups
    )
    return bool(np.all(after > _TUNNEL_GRID))


# -----------------------------------------------------------------------------
# Information of consistent Gaussian LLRs, and measured information
# -----------------------------------------------------------------------------


@cache
def _j_table() -> tuple[np.ndarray, np.ndarray]:
    """``(σ, J(σ))`` on a grid up to where ``J`` reaches 1 in float64.

    ``J(σ) = 1 - E[log2(1 + exp(-L))]`` for ``L ~ N(σ²/2, σ²)``, by the
    trapezoidal rule in the standard-normal variable (within 1e-13 of
    adaptive quadrature); interpolating the table is within 1e-6.
    """
    sigma = np.linspace(0.0, 16.0, 3201)
    z = np.linspace(-9.0, 9.0, 361)
    weight = np.exp(-0.5 * z * z) * (z[1] - z[0]) / math.sqrt(2.0 * math.pi)
    weight[[0, -1]] *= 0.5
    llr = (0.5 * sigma**2)[:, None] + sigma[:, None] * z  # given bit 0
    info = 1.0 - (np.logaddexp(0.0, -llr) / math.log(2.0)) @ weight
    info[0] = 0.0
    stop = int(np.argmax(np.diff(info) <= 0)) or info.size - 1
    sigma, info = sigma[: stop + 1], info[: stop + 1]
    sigma.setflags(write=False)
    info.setflags(write=False)
    return sigma, info


def _j(sigma: Any) -> np.ndarray:
    """Information of a consistent Gaussian LLR of standard deviation ``sigma``."""
    s, info = _j_table()
    return np.interp(sigma, s, info)


def _j_inv(info: Any) -> np.ndarray:
    """``σ`` of the consistent Gaussian LLR carrying ``info``."""
    s, table = _j_table()
    return np.interp(info, table, s)


def _information(llrs: ArrayType, bits: ArrayType) -> float:
    """Mutual information per bit between LLRs and their bits, from histograms.

    ``I = ½ Σ_b Σ_bins p(L | b) log2(2 p(L | b) / (p(L | 0) + p(L | 1)))``
    over 200 bins spanning the central 99.9 % of ``|L|`` (the rest in the
    outer bins), so it measures LLRs whatever their scaling, with the
    Miller-Madow correction of the histogram entropies (the plain estimate
    reads 0.003 bit high at 2^15 bits, the corrected one within 0.001).
    The histograms are computed on the data's device; the bin counts are
    transferred.
    """
    x, xp, _ = dispatch(llrs)
    x = x.reshape(-1)
    b = bits.reshape(-1)
    lim = float(xp.percentile(xp.abs(x), 99.9)) or 1.0
    x = xp.clip(x, -lim, lim)
    edges = xp.linspace(-lim, lim, _BINS + 1)
    h0 = to_device(xp.histogram(x[b == 0], bins=edges)[0], "cpu").astype(np.float64)
    h1 = to_device(xp.histogram(x[b == 1], bins=edges)[0], "cpu").astype(np.float64)
    n0, n1 = h0.sum(), h1.sum()
    if not n0 or not n1:
        raise ValueError("EXIT measurement needs bits of both values.")
    p0, p1 = h0 / n0, h1 / n1
    total = 0.0
    for p, q in ((p0, p1), (p1, p0)):
        k = p > 0
        total += 0.5 * float(np.sum(p[k] * np.log2(2.0 * p[k] / (p[k] + q[k]))))
    # Miller-Madow: each entropy's estimate is low by (bins used - 1) / 2N nats.
    used = np.count_nonzero(h0 + h1)
    total += (
        (used - 1) / (2.0 * (n0 + n1))
        - 0.5 * (np.count_nonzero(h0) - 1) / (2.0 * n0)
        - 0.5 * (np.count_nonzero(h1) - 1) / (2.0 * n1)
    ) / math.log(2.0)
    return min(max(total, 0.0), 1.0)


def _a_priori_llrs(sign: ArrayType, info: float, gen: Any, xp: Any) -> ArrayType:
    """Consistent Gaussian LLRs of information ``info``, float32, on the device."""
    s = float(_j_inv(min(info, _MAX_A_PRIORI)))
    noise = _random.standard_normal(gen, sign.shape, dtype=np.float32, xp=xp)
    return sign * (0.5 * s * s) + s * noise


def _demap_extrinsic(
    received: ArrayType,
    points: ArrayType,
    labels: ArrayType,
    a_priori: ArrayType,
    noise_var: float,
    method: str,
    xp: Any,
) -> ArrayType:
    """Extrinsic demapper LLRs ``(N, m)``, float32, chunked over the symbols."""
    n, m = a_priori.shape
    sign = (1 - 2 * labels).astype(np.float32)  # (M, m)
    groups = [
        (xp.nonzero(labels[:, k] == 0)[0], xp.nonzero(labels[:, k] == 1)[0])
        for k in range(m)
    ]
    out = xp.empty((n, m), dtype=np.float32)
    rows = max(1, _DEMAP_CHUNK // points.size)
    for r0 in range(0, n, rows):
        y = received[r0 : r0 + rows]
        half = 0.5 * a_priori[r0 : r0 + rows]  # (r, m)
        metric = -(xp.abs(y[:, None] - points[None, :]) ** 2) / noise_var
        metric = metric + half @ sign.T  # a-priori log-probability of each label
        for k, (zero, one) in enumerate(groups):
            own = metric - half[:, k : k + 1] * sign[None, :, k]
            out[r0 : r0 + rows, k] = _reduce(own[:, zero], method, xp) - _reduce(
                own[:, one], method, xp
            )
    return out


def _reduce(metric: ArrayType, method: str, xp: Any) -> ArrayType:
    """Log-sum-exp over the last axis, or its maximum (max-log)."""
    top = metric.max(axis=1)
    if method == "maxlog":
        return top
    return top + xp.log(xp.exp(metric - top[:, None]).sum(axis=1))


def _check_points(points: Any, name: str) -> None:
    if isinstance(points, bool) or not isinstance(points, int) or points < 2:
        raise ValueError(f"{name}: points must be an integer >= 2, got {points}.")


def _check_device(device: Any, name: str) -> str:
    """Validate ``device`` as the factories do (``core`` is above this layer)."""
    if isinstance(device, str) and device.lower() in ("cpu", "gpu"):
        return device.lower()
    raise ValueError(f'{name} requires device to be "cpu" or "gpu"; got {device!r}.')
