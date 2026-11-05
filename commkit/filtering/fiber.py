"""Fiber compensation: chromatic dispersion and digital backpropagation.

:func:`correct_chromatic_dispersion` is the inverse of
:func:`commkit.impairments.apply_chromatic_dispersion`, applied to the whole
record in the frequency domain; :func:`dispersion_compensation_taps` designs
its FIR form.  :func:`correct_fiber_propagation` (digital backpropagation) is
the inverse of :func:`commkit.impairments.apply_fiber_propagation`.
"""

import logging
import math

import numpy as np

from .._array import restore_1d
from .._dispersion import (
    apply_dispersion,
    beta2_length,
    compensation_taps,
    spread_samples,
)
from .._fiber import Fiber, check_link, propagate, to_watts
from ..backend import dispatch
from ..core._signal_adapter import S, adapt_signal
from ..logger import logger

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
    dispersion_ps_nm_km: float,
    fiber_length_km: float,
    center_wavelength_nm: float | None = None,
    sampling_rate: float | None = None,
) -> S:
    """Compensate chromatic dispersion electronically: the inverse fiber response.

    Multiplies the spectrum by ``H(ω) = exp(+j β₂ L ω² / 2)`` with
    ``β₂ = -D λ² / (2π c)``, which undoes
    :func:`commkit.impairments.apply_chromatic_dispersion` exactly.  There
    are no taps to choose: the response is applied on the FFT grid of the
    whole record, exact at every frequency (see Notes).

    Parameters
    ----------
    samples : array_like or Signal
        Complex baseband samples, ``(N,)`` or ``(C, N)``.
    dispersion_ps_nm_km : float
        Fiber dispersion parameter D in ps / (nm km) (SMF-28: 17 at 1550 nm).
    fiber_length_km : float
        Fiber length in km.
    center_wavelength_nm : float, optional
        Carrier wavelength in nm.  Taken from an optical Signal's
        carrier (``domain="OPT"``, ``c / center_frequency``); required for
        array input and Signals of another domain.  A value that disagrees
        with the optical Signal raises.
    sampling_rate : float, optional
        Sampling rate in Hz.  Taken from the Signal; required for array
        input.  A value that disagrees with the Signal raises.

    Returns
    -------
    ndarray or Signal
        Compensated samples, same shape, dtype and device as the input.

    See Also
    --------
    dispersion_compensation_taps : The same compensation as FIR taps.

    Notes
    -----
    One FFT of the record, a multiply and an inverse FFT: a circular filter
    as long as the record.  The record is treated as one period, so the
    dispersed tails of its last symbols wrap onto its first and the reverse;
    the samples within half the spread of either end are compensated from
    the other end.  That is exact for a record that was dispersed circularly
    (by :func:`commkit.impairments.apply_chromatic_dispersion`) and wrong
    there for a capture.  The dispersion's memory, the group-delay spread
    ``2π |β₂ L| fs`` over the sampled band, is logged at INFO in samples.
    For an FIR or a block filter of a given length (as a receiver applies
    it), design the taps with :func:`dispersion_compensation_taps` and apply
    them with :func:`ols_fir_filter`.

    Examples
    --------
    >>> sig = correct_chromatic_dispersion(  # optical Signal: its carrier
    ...     sig, dispersion_ps_nm_km=17.0, fiber_length_km=80.0)
    """
    signal_adapter = adapt_signal(
        samples, function_name="correct_chromatic_dispersion()"
    )
    sampling_rate = signal_adapter.resolve_fact("sampling_rate", sampling_rate)
    center_wavelength_nm = signal_adapter.resolve_wavelength_nm(center_wavelength_nm)
    if logger.isEnabledFor(logging.INFO):
        b2l = beta2_length(
            dispersion_ps_nm_km=dispersion_ps_nm_km,
            fiber_length_km=fiber_length_km,
            center_wavelength_nm=center_wavelength_nm,
        )
        logger.info(
            "Correcting CD (D=%s ps/nm/km, L=%s km, λ=%s nm): spread %.1f samples "
            "over the sampled band, applied in the frequency domain.",
            dispersion_ps_nm_km,
            fiber_length_km,
            center_wavelength_nm,
            spread_samples(b2l, sampling_rate=sampling_rate, bandwidth=sampling_rate),
        )
    result = apply_dispersion(
        signal_adapter.array,
        sampling_rate=sampling_rate,
        dispersion_ps_nm_km=dispersion_ps_nm_km,
        fiber_length_km=fiber_length_km,
        center_wavelength_nm=center_wavelength_nm,
        inverse=True,
    )
    return signal_adapter.wrap_samples(result)


#: Default length over the spread: a filter exactly as long as the spread
#: cannot hold the group delay at the band edges (residual near -32 dB on a
#: 2-sps RRC signal), one 25 % longer reaches about -45 dB at any length.
#: Below a few samples of spread the sinc tails of the band-limited response
#: dominate, hence the minimum.
_TAPS_MARGIN = 1.25
_MIN_TAPS = 7


def dispersion_compensation_taps(
    *,
    dispersion_ps_nm_km: float,
    fiber_length_km: float,
    center_wavelength_nm: float,
    sampling_rate: float,
    bandwidth: float | None = None,
    num_taps: int | None = None,
) -> np.ndarray:
    """Design FIR taps that compensate chromatic dispersion.

    The FIR form of :func:`correct_chromatic_dispersion`: the least-squares
    fit of its response ``exp(+j β₂ L ω² / 2)`` over the band
    ``|f| <= bandwidth / 2`` (Eghbali et al. 2014).  Over the whole sampled
    band the fit is the band-limited impulse response truncated to
    ``num_taps``.  Apply the taps with :func:`ols_fir_filter` (long filters)
    or :func:`fir_filter`, both of which center them; inspect them with
    :func:`filter_response` and :func:`commkit.plotting.plot_filter_response`.

    Parameters
    ----------
    dispersion_ps_nm_km : float
        Fiber dispersion parameter D in ps / (nm km).
    fiber_length_km : float
        Fiber length in km.
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
        group-delay spread over the band, ``2π |β₂ L| · bandwidth ·
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
    correct_chromatic_dispersion : The exact compensation of a whole record.

    Notes
    -----
    The spread is the dispersion's memory: a shorter filter cannot hold the
    group delay ``β₂ L ω`` at the band edges.  Measured against
    :func:`correct_chromatic_dispersion` on a 32 GBd, 2-samples-per-symbol
    RRC signal (rolloff 0.1) over 20 to 4000 km of standard fiber, the
    default length leaves a residual of -42 to -50 dB when fitted over the
    occupied band (35.2 GHz), and -27 dB (20 km) to -68 dB (4000 km) over
    the whole band, where the truncation ripple spreads into the occupied
    band.  A filter exactly as long as the spread over the occupied band
    leaves about -32 dB.  Sweep ``num_taps`` against a metric of the
    compensated signal to choose a length for another signal.

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
    ...     dispersion_ps_nm_km=17.0, fiber_length_km=80.0,
    ...     center_wavelength_nm=1550.0, sampling_rate=64e9, bandwidth=35.2e9)
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
        dispersion_ps_nm_km=dispersion_ps_nm_km,
        fiber_length_km=fiber_length_km,
        center_wavelength_nm=center_wavelength_nm,
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

    Returns
    -------
    ndarray or Signal
        The backpropagated field in the input's units, complex, on the
        input's device.  A :class:`Signal` returns a new :class:`Signal`.

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
    out = propagate(
        field,
        xp,
        link=link,
        sampling_rate=sampling_rate,
        direction=-1,
        nonlinear_scale=float(nonlinear_scale),
    )
    out *= out.real.dtype.type(1.0 / scale)
    return signal_adapter.wrap_samples(restore_1d(was_1d, out))
