"""Fiber compensation: chromatic dispersion and digital backpropagation.

:func:`correct_chromatic_dispersion` is the inverse of
:func:`commkit.impairments.apply_chromatic_dispersion` and
:func:`correct_fiber_propagation` (digital backpropagation) the inverse of
:func:`commkit.impairments.apply_fiber_propagation`.
:func:`dispersion_compensation_taps` designs the FIR filter of the former.

**Records.**  Both treat the samples as a window of an endless stream by
default, as an ADC delivers it, and process it as a receiver does, in
bounded memory: chromatic dispersion as an FIR filter applied by
overlap-save, backpropagation in overlapping blocks.  Within half the
dispersion's memory of either end of a capture no method can be right,
since the history needed lies outside the capture; here those samples are
transients of a zero history.  ``periodic=True`` treats the record as one
period of a periodic sequence (an AWG pattern, or a record dispersed by
the circular channel functions of :mod:`commkit.impairments`) and filters
it whole in the frequency domain with the exact response: exact
everywhere on such a record.  On a capture it leaves the same end
samples wrong (the other end wraps in) and is otherwise exact, at the
cost of transforming the whole record at once.

**Dispersion.**  The receiver knows the accumulated dispersion of the link,
``dispersion_ps_nm`` (``D·L``, measured or from the link plan), not the
fiber's ``D`` and length apart; only their product enters
``β₂L = -D·L·λ² / (2π c)``.
"""

import math
from typing import Any

import numpy as np

from .._array import restore_1d
from .._dispersion import (
    apply_dispersion,
    beta2_length,
    beta3_length,
    compensation_taps,
    spread_samples,
)
from .._fiber import Fiber, check_link, propagate, to_watts
from ..backend import ArrayType, dispatch
from ..core._signal_adapter import S, adapt_signal
from ..logger import logger
from .linear import ols_fir_filter

__all__ = [
    "correct_chromatic_dispersion",
    "correct_fiber_propagation",
    "dispersion_compensation_taps",
]

# -----------------------------------------------------------------------------
# CHROMATIC DISPERSION (Signal-aware)
# -----------------------------------------------------------------------------


def correct_chromatic_dispersion(
    samples: S,
    *,
    dispersion_ps_nm: float,
    center_wavelength_nm: float | None = None,
    sampling_rate: float | None = None,
    periodic: bool = False,
    bandwidth: float | None = None,
    num_taps: int | None = None,
) -> S:
    """Compensate chromatic dispersion: the inverse fiber response.

    Filters with ``H(ω) = exp(+j β₂L ω² / 2)``, ``β₂L = -D·L·λ² / (2π c)``,
    which undoes the dispersion of a link of accumulated dispersion
    ``D·L``.  By default the samples are a window of a stream (a capture):
    the filter is the FIR of :func:`dispersion_compensation_taps`, applied by
    overlap-save (:func:`ols_fir_filter`).  With ``periodic=True`` the
    record is one period of a periodic sequence and the response multiplies
    its whole spectrum, the exact inverse of
    :func:`commkit.impairments.apply_chromatic_dispersion`.

    Parameters
    ----------
    samples : array_like or Signal
        Complex baseband samples, ``(N,)`` or ``(C, N)``.
    dispersion_ps_nm : float
        Accumulated dispersion of the link ``D·L`` in ps/nm (standard fiber:
        17 per km at 1550 nm, so 1360 for 80 km).  The value of the fiber,
        not its negative: the function inverts it.
    center_wavelength_nm : float, optional
        Carrier wavelength in nm.  Taken from an optical Signal's
        carrier (``domain="OPT"``, ``c / center_frequency``); required for
        array input and Signals of another domain.  A value that disagrees
        with the optical Signal raises.
    sampling_rate : float, optional
        Sampling rate in Hz.  Taken from the Signal; required for array
        input.  A value that disagrees with the Signal raises.
    periodic : bool, default False
        Treat ``samples`` as one period of a periodic sequence and filter
        the whole record in the frequency domain with the exact response:
        exact everywhere on a periodic record (one AWG period, or one
        dispersed circularly in simulation).  On a capture the samples
        within half the spread of either end are wrong (the other end wraps
        in), as they are transients of the FIR, and the rest exact; the
        whole record is transformed at once.
    bandwidth : float, optional
        FIR only: two-sided bandwidth in Hz over which the filter is
        fitted, the signal's occupied bandwidth (see
        :func:`dispersion_compensation_taps`).  ``None`` fits the whole
        sampled band.
    num_taps : int, optional
        FIR only: number of taps; ``None`` takes the default of
        :func:`dispersion_compensation_taps`.

    Returns
    -------
    ndarray or Signal
        Compensated samples, same shape, dtype and device as the input.
        Without ``periodic`` the first and last ``(num_taps - 1) // 2``
        samples are filter transients (on a capture they lack the history
        that lies outside it, whatever the method).

    Raises
    ------
    ValueError
        If ``bandwidth`` or ``num_taps`` comes with ``periodic=True``.

    See Also
    --------
    dispersion_compensation_taps : The FIR taps, to inspect or apply.

    Notes
    -----
    The FIR is centred, so it adds no delay and the reference stays
    aligned.  Its default length is 1.25 times the dispersion's
    group-delay spread over the fitted band, about -45 dB of residual
    against the exact response on a signal inside that band; fitting the
    occupied band rather than the whole sampled band needs about half the
    taps (see :func:`dispersion_compensation_taps`).

    Away from the ends of a capture the whole-record filter of
    ``periodic=True`` is more precise still, as its response is exact:
    measured on a window of an 80 km, 2-sps record, both are wrong over the
    first and last 16 samples, the FIR then holds -45 dB, and the
    whole-record filter reaches -30 dB over the next 16, -68 dB beyond 50
    and -110 dB deep inside.  On a record that fits in memory it is a valid
    choice too; the FIR is the receiver's form, in bounded memory, with taps
    to inspect.

    Examples
    --------
    >>> y = correct_chromatic_dispersion(  # a capture after 80 km of SSMF
    ...     capture, dispersion_ps_nm=1360.0, bandwidth=35.2e9)
    >>> y = correct_chromatic_dispersion(  # a circularly simulated record
    ...     sig, dispersion_ps_nm=1360.0, periodic=True)
    """
    signal_adapter = adapt_signal(
        samples, function_name="correct_chromatic_dispersion()"
    )
    sampling_rate = signal_adapter.resolve_fact("sampling_rate", sampling_rate)
    center_wavelength_nm = signal_adapter.resolve_wavelength_nm(center_wavelength_nm)
    if periodic:
        if bandwidth is not None or num_taps is not None:
            raise ValueError(
                "correct_chromatic_dispersion(): bandwidth and num_taps design "
                "the FIR of a capture; periodic=True filters the whole record "
                "exactly and takes neither."
            )
        logger.info(
            "Correcting CD of %s ps/nm (λ=%s nm) over the whole periodic record.",
            dispersion_ps_nm,
            center_wavelength_nm,
        )
        result = apply_dispersion(
            signal_adapter.array,
            sampling_rate=sampling_rate,
            dispersion_ps_nm=dispersion_ps_nm,
            center_wavelength_nm=center_wavelength_nm,
            inverse=True,
        )
        return signal_adapter.wrap_samples(result)
    taps = dispersion_compensation_taps(
        dispersion_ps_nm=dispersion_ps_nm,
        center_wavelength_nm=center_wavelength_nm,
        sampling_rate=sampling_rate,
        bandwidth=bandwidth,
        num_taps=num_taps,
    )
    logger.info(
        "Correcting CD of %s ps/nm (λ=%s nm) with a %d-tap FIR.",
        dispersion_ps_nm,
        center_wavelength_nm,
        taps.size,
    )
    x, _, _ = dispatch(signal_adapter.array)
    return signal_adapter.wrap_samples(ols_fir_filter(x, taps))


#: Default length over the spread: a filter exactly as long as the spread
#: cannot hold the group delay at the band edges (residual near -32 dB on a
#: 2-sps RRC signal), one 25 % longer reaches about -45 dB at any length.
#: Below a few samples of spread the sinc tails of the band-limited response
#: dominate, hence the minimum.
_TAPS_MARGIN = 1.25
_MIN_TAPS = 7


def dispersion_compensation_taps(
    *,
    dispersion_ps_nm: float,
    center_wavelength_nm: float,
    sampling_rate: float,
    bandwidth: float | None = None,
    num_taps: int | None = None,
) -> np.ndarray:
    """Design FIR taps that compensate chromatic dispersion.

    The FIR filter of :func:`correct_chromatic_dispersion`: the
    least-squares fit of the inverse response ``exp(+j β₂L ω² / 2)`` over
    the band ``|f| <= bandwidth / 2`` (Eghbali et al. 2014).  Over the whole
    sampled band the fit is the band-limited impulse response truncated to
    ``num_taps``.  Apply the taps with :func:`ols_fir_filter` (long filters)
    or :func:`fir_filter`, both of which center them; inspect them with
    :func:`filter_response` and :func:`commkit.plotting.plot_filter_response`.

    Parameters
    ----------
    dispersion_ps_nm : float
        Accumulated dispersion of the link ``D·L`` in ps/nm.
    center_wavelength_nm : float
        Carrier wavelength in nm.
    sampling_rate : float
        Sampling rate of the samples the taps will filter, in Hz.
    bandwidth : float, optional
        Two-sided bandwidth in Hz over which the response is fitted, in
        ``(0, sampling_rate]``: the occupied bandwidth of the signal
        (``(1 + rolloff) · symbol_rate`` for an RRC pulse).  ``None`` fits
        the whole sampled band.  Outside the band the response is
        unconstrained, apart from a small regularization that bounds its
        gain, so a narrower band needs fewer taps for the same accuracy.
    num_taps : int, optional
        Number of taps, at least 1.  ``None`` takes 1.25 times the
        group-delay spread over the band, ``2π |β₂L| · bandwidth ·
        sampling_rate`` samples, rounded up to an odd number and at least 7
        (the sinc tails of a short link's response).

    Returns
    -------
    numpy.ndarray
        Host ``complex128`` taps, ``(num_taps,)``, centred on tap
        ``(num_taps - 1) // 2``.

    Raises
    ------
    ValueError
        If ``sampling_rate`` is not positive, ``bandwidth`` is outside
        ``(0, sampling_rate]`` or ``num_taps`` is below 1.

    See Also
    --------
    correct_chromatic_dispersion : Designs and applies these taps.

    Notes
    -----
    The spread is the dispersion's memory: a shorter filter cannot hold the
    group delay ``β₂L ω`` at the band edges.  Measured against the exact
    response on a 32 GBd, 2-samples-per-symbol RRC signal (rolloff 0.1)
    over 20 to 4000 km of standard fiber, the default length leaves a
    residual of -42 to -50 dB when fitted over the occupied band
    (35.2 GHz), and -27 dB (20 km) to -68 dB (4000 km) over the whole band,
    where the truncation ripple spreads into the occupied band.  A filter
    exactly as long as the spread over the occupied band leaves about
    -32 dB.  Sweep ``num_taps`` against a metric of the compensated signal
    to choose a length for another signal.

    References
    ----------
    .. [1] S. J. Savory, "Digital filters for coherent optical receivers,"
       Opt. Express 16(2), 804-817, 2008.
    .. [2] A. Eghbali, H. Johansson, O. Gustafsson and S. J. Savory,
       "Optimal least-squares FIR digital filters for compensation of
       chromatic dispersion in digital coherent optical receivers,"
       J. Lightw. Technol. 32(8), 1449-1456, 2014.

    Examples
    --------
    >>> taps = dispersion_compensation_taps(
    ...     dispersion_ps_nm=1360.0, center_wavelength_nm=1550.0,
    ...     sampling_rate=64e9, bandwidth=35.2e9)
    >>> y = ols_fir_filter(x, taps)
    """
    if not sampling_rate > 0:
        raise ValueError(f"sampling_rate must be positive, got {sampling_rate!r}.")
    band = sampling_rate if bandwidth is None else float(bandwidth)
    if not 0 < band <= sampling_rate:
        raise ValueError(
            f"bandwidth must be in (0, sampling_rate={sampling_rate!r}], "
            f"got {bandwidth!r}."
        )
    b2l = beta2_length(
        dispersion_ps_nm=dispersion_ps_nm, center_wavelength_nm=center_wavelength_nm
    )
    if num_taps is None:
        spread = spread_samples(b2l, sampling_rate=sampling_rate, bandwidth=band)
        num_taps = max(_MIN_TAPS, 2 * math.ceil((_TAPS_MARGIN * spread - 1) / 2) + 1)
    elif int(num_taps) != num_taps or num_taps < 1:
        raise ValueError(f"num_taps must be an integer >= 1, got {num_taps!r}.")
    logger.debug(
        "Designing CD compensation: β₂L=%.3e s², bandwidth=%s Hz, taps=%s.",
        b2l,
        band,
        num_taps,
    )
    return compensation_taps(
        b2l, sampling_rate=sampling_rate, bandwidth=band, num_taps=int(num_taps)
    )


def correct_fiber_propagation(
    samples: S,
    *,
    fiber: Fiber,
    span_length_km: float,
    num_spans: int,
    launch_power_dbm: float,
    center_wavelength_nm: float | None = None,
    steps_per_span: int = 100,
    nonlinear_scale: float = 1.0,
    sampling_rate: float | None = None,
    periodic: bool = False,
) -> S:
    """Undo a nonlinear fiber link by digital backpropagation (DBP).

    Runs the split-step solver of
    :func:`commkit.impairments.apply_fiber_propagation` backward: the spans
    in reverse order and every step inverted (negated dispersion and Kerr
    phase).  The solver works on the loss-normalized field ``u = A e^{αz/2}``,
    on which an amplifier that restores the span loss is the identity, so
    neither the loss nor the gain is applied as a factor: the loss enters
    only through the Kerr phase of each step, ``γ (e^{-αz_{k-1}} -
    e^{-αz_k}) / α`` times the power, which DBP negates.  With the link's own
    ``steps_per_span`` and ``nonlinear_scale=1`` this is the exact inverse
    of a noiseless, PMD-free link, up to rounding: the symmetric split step
    is time-reversible.  That is the default.

    Fewer steps (a common low-complexity choice is one per span) make DBP an
    approximation, and ``nonlinear_scale`` must then be tuned below 1 for
    the best signal quality.  Measured on 4 x 80 km of standard fiber at
    32 GBd, 6 dBm, NF 5 dB (distortion -16 dB after dispersion compensation
    alone): 10 steps per span with ξ = 1 reach the ASE floor (-30 dB); one
    step per span gains 2 dB with ξ = 0.5 but nothing with ξ = 1.

    By default the samples are a window of a stream (a capture): they are
    backpropagated in overlapping blocks, each with a guard of 1.25 times
    half the link's dispersion memory on both sides, and the guards are
    discarded (overlap-save), so the working memory is bounded by the block.
    The dispersion's light cone bounds both the linear and the nonlinear
    interaction, so every kept sample sees all of its past and future within
    the link (measured: the blocks match the whole record to -71 dB).  With
    ``periodic=True`` the record is one period of a periodic sequence and is
    backpropagated whole, the exact inverse of
    :func:`commkit.impairments.apply_fiber_propagation` (which propagates
    its record circularly).  On a capture either way leaves half the link's
    memory at both ends wrong, since the history needed lies outside it.

    Parameters
    ----------
    samples : array_like or Signal
        Received field, ``(N,)`` or ``(2, N)``, in any units: its average
        power (summed over the polarizations) is taken as the launch power,
        so the output of :func:`apply_fiber_propagation` or a receiver
        normalized to unit power both work.
    fiber : Fiber
        The link's fiber.  Its PMD is not undone (a realization is random
        and unknown to the receiver); leave it to the adaptive MIMO
        equalizer.
    span_length_km : float
        Length of every span in km.
    num_spans : int
        Number of spans to backpropagate.
    launch_power_dbm : float
        The link's launch power in dBm, summed over the polarizations.
    center_wavelength_nm : float, optional
        Carrier wavelength in nm.  Taken from an optical Signal's
        carrier (``domain="OPT"``, ``c / center_frequency``); required for
        array input and Signals of another domain.  A value that disagrees
        with the optical Signal raises.
    steps_per_span : int, default 100
        Backward steps per span (the forward model's default).
    nonlinear_scale : float, default 1.0
        Factor ξ on the backward Kerr phase; 1 inverts the Kerr effect.
    sampling_rate : float, optional
        Sampling rate in Hz.  Taken from the Signal; required for array
        input.  A value that disagrees with the Signal raises.
    periodic : bool, default False
        Treat ``samples`` as one period of a periodic sequence and
        backpropagate the whole record at once: exact for a circularly
        simulated record; on a capture the ends are wrong (the other end
        wraps in), as they are transients of the blocks.

    Returns
    -------
    ndarray or Signal
        The backpropagated field in the input's units, complex, on the
        input's device.  A :class:`Signal` returns a new :class:`Signal`.
        Without ``periodic`` the first and last guard (half the link's
        dispersion memory, times 1.25) are transients.

    See Also
    --------
    correct_chromatic_dispersion : the linear (``γ = 0``) special case.

    Notes
    -----
    DBP cannot undo the nonlinear interaction of the signal with ASE added
    along the link, nor random PMD: both bound its gain on a real link.

    Examples
    --------
    >>> from commkit.impairments import Fiber
    >>> rx = correct_fiber_propagation(
    ...     rx, fiber=Fiber(), span_length_km=80.0, num_spans=10,
    ...     launch_power_dbm=2.0)
    """
    name = "correct_fiber_propagation()"
    signal_adapter = adapt_signal(samples, function_name=name)
    sampling_rate = signal_adapter.resolve_fact("sampling_rate", sampling_rate)
    center_wavelength_nm = signal_adapter.resolve_wavelength_nm(center_wavelength_nm)
    link = check_link(
        fiber=fiber,
        span_length_km=span_length_km,
        num_spans=num_spans,
        steps_per_span=steps_per_span,
        center_wavelength_nm=center_wavelength_nm,
        function_name=name,
    )
    if not np.isfinite(nonlinear_scale):
        raise ValueError(
            f"{name}: nonlinear_scale must be finite, got {nonlinear_scale!r}."
        )
    x, xp, _ = dispatch(signal_adapter.array)
    field, was_1d, scale = to_watts(x, xp, launch_power_dbm, name)
    logger.info(
        "DBP: %d x %.4g km, %.3g dBm, %d steps/span, xi=%.3g.",
        link.num_spans,
        link.span_length_km,
        launch_power_dbm,
        link.steps_per_span,
        nonlinear_scale,
    )

    def backpropagate(x: ArrayType) -> ArrayType:
        return propagate(
            x,
            xp,
            link=link,
            sampling_rate=sampling_rate,
            direction=-1,
            nonlinear_scale=float(nonlinear_scale),
        )

    if periodic:
        out = backpropagate(field)
    else:
        out = _overlap_save_blocks(
            field, xp, guard=_link_guard(link, sampling_rate), run=backpropagate
        )
    out *= out.real.dtype.type(1.0 / scale)
    return signal_adapter.wrap_samples(restore_1d(was_1d, out))


#: Smallest block of the overlap-save backpropagation; blocks are at least
#: eight guards long, so at most a quarter of each is discarded.
_MIN_BLOCK = 4096


def _link_guard(link: Any, sampling_rate: float) -> int:
    """Guard in samples on each side of a DBP block.

    1.25 times half the spread of the link's group delay
    ``τ(ω) = β₂L ω + β₃L ω²/2`` over the sampled band, plus a few samples
    for the band-limited tails.
    """
    fiber = link.fiber
    total_ps_nm = fiber.dispersion_ps_nm_km * link.span_length_km * link.num_spans
    kw = {
        "dispersion_ps_nm": total_ps_nm,
        "center_wavelength_nm": link.center_wavelength_nm,
    }
    b2 = beta2_length(**kw)
    slope = fiber.dispersion_slope_ps_nm2_km
    b3 = (
        0.0
        if slope is None
        else beta3_length(
            dispersion_slope_ps_nm2=slope * link.span_length_km * link.num_spans,
            **kw,
        )
    )
    omega = np.linspace(-np.pi, np.pi, 1025) * sampling_rate
    tau = b2 * omega + 0.5 * b3 * omega**2
    spread = float(tau.max() - tau.min()) * sampling_rate
    return int(math.ceil(_TAPS_MARGIN * spread / 2.0)) + 8


def _overlap_save_blocks(
    field: ArrayType, xp: Any, *, guard: int, run: Any
) -> ArrayType:
    """Apply ``run`` to overlapping blocks of ``field`` and keep their cores.

    ``field`` is ``(C, N)``; the blocks, ``(C, B, L)`` with ``L`` a power of
    two of at least eight guards, overlap by two guards and start one guard
    before the record (zero-padded), so the kept cores of ``L - 2·guard``
    samples tile it.  ``run`` processes all blocks at once.
    """
    C, N = field.shape
    length = max(_MIN_BLOCK, 1 << (8 * guard - 1).bit_length())
    hop = length - 2 * guard
    num_blocks = -(-N // hop)
    padded = xp.zeros((C, num_blocks * hop + 2 * guard), dtype=field.dtype)
    padded[:, guard : guard + N] = field
    index = xp.arange(num_blocks)[:, None] * hop + xp.arange(length)[None, :]
    logger.debug(
        "DBP in %d blocks of %d samples (guard %d).", num_blocks, length, guard
    )
    out = run(padded[:, index])  # (C, B, L)
    return out[:, :, guard : guard + hop].reshape(C, num_blocks * hop)[:, :N]
