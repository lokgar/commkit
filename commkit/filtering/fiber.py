"""Fiber compensation: chromatic dispersion and digital backpropagation.

:func:`correct_chromatic_dispersion` is the inverse of
:func:`commkit.impairments.apply_chromatic_dispersion` and
:func:`correct_fiber_propagation` (digital backpropagation) the inverse of
:func:`commkit.impairments.apply_fiber_propagation`.
:func:`dispersion_compensation_taps` designs the FIR filter of the former.

**Methods.**  Each compensation has two implementations of the same
inverse, chosen by ``method``; both are valid on a capture and on a
simulated record:

- the receiver's form, the default: chromatic dispersion as an FIR filter
  applied by overlap-save (``method="fir"``), backpropagation in
  overlapping blocks (``method="blocks"``).  The working memory is bounded
  whatever the record length, and the FIR's taps can be designed and
  inspected (:func:`dispersion_compensation_taps`).
- ``method="whole_record"``: one pass over the whole record in the
  frequency domain, with the exact response.  The record is transformed at
  once, and treated as one period of a periodic sequence.

**Captures and periodic records.**  The channel functions of
:mod:`commkit.impairments` disperse their record circularly, as if an AWG
played it in a loop, and ``"whole_record"`` inverts such a record exactly
at every sample, the FIR to its design accuracy (about -45 dB).  A capture
is a window of an endless stream: within half the dispersion's memory of
either end, the history the inverse needs lies outside the capture, so
those samples are wrong whatever the method (filter transients, or the
other end wrapped in); trim them.  Beyond them both methods are right, and
``"whole_record"`` is the more precise, its response being exact.  So on a
capture that fits in memory ``"whole_record"`` is a correct choice; the
default suits streams of any length and studies of the filter.

**Dispersion.**  The receiver knows the accumulated dispersion of the link,
``dispersion_ps_nm`` (``D·L``, measured or from the link plan), not the
fiber's ``D`` and length apart; only their product enters
``β₂L = -D·L·λ² / (2π c)``.
"""

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .. import _fft
from .._array import as_2d, restore_1d
from .._constants import SPEED_OF_LIGHT as _SPEED_OF_LIGHT
from .._dispersion import (
    apply_dispersion,
    beta2_length,
    beta3_length,
    compensation_taps,
    spread_samples,
)
from .._fiber import Fiber, check_link, propagate, to_watts
from ..backend import ArrayType, dispatch, to_device
from ..core._signal_adapter import S, adapt_signal
from ..logger import logger
from .linear import ols_fir_filter

__all__ = [
    "ChromaticDispersionEstimate",
    "ClockToneScan",
    "correct_chromatic_dispersion",
    "correct_fiber_propagation",
    "dispersion_compensation_taps",
    "estimate_chromatic_dispersion",
]

# -----------------------------------------------------------------------------
# CHROMATIC DISPERSION ESTIMATION
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ClockToneScan:
    """Blind chromatic-dispersion estimation from the clock tone (Godard).

    A linearly modulated signal is cyclostationary at its symbol rate: the
    spectral correlation ``P(f) = E[Y(f) Y*(f - R_s)]`` is non-zero where
    the spectrum and its copy shifted by ``R_s`` overlap (the excess band of
    the pulse) and carries the clock tone.  Dispersion multiplies it by
    ``exp(-j β₂L Ω ω)`` (``Ω = 2π R_s``), a phase linear in frequency, which
    a compensation of ``β₂L`` removes, maximizing ``|Σ_f P(f)|``.  As a
    function of the trial ``β₂L`` that sum is a Fourier transform of
    ``P(f)``, so all candidates are scanned by one zero-padded inverse FFT,
    whose estimate is the centroid of the peak's lobe above half its
    maximum.  The correlation at
    ``-R_s`` adds a second, conjugate scan, and the channels add in power,
    so the tone's fading on one polarization (PMD) is covered by the other.

    Parameters
    ----------
    search_range : (float, float), default (-40000.0, 40000.0)
        Candidates of the accumulated dispersion ``D·L`` in ps/nm, as
        ``(start, stop)``.  The default spans about ±2350 km of standard
        fiber.  The FFT blocks are sized from it: a power of two of at least
        eight times the dispersion spread of its larger end, and 1024
        samples, so a wider range needs a longer record.

    Notes
    -----
    **Accuracy.**  The peak's width is about ``1 / (rolloff · R_s · Ω)`` in
    ``β₂L``, some 1200 ps/nm at 32 GBd with rolloff 0.1, so the estimate
    is the centre of the peak, not its maximum.  Measured on 2^17 samples of
    dual-polarization 16-QAM at 32 GBd, 2 samples per symbol, 8500 ps/nm,
    15 dB Es/N0, over 12 noise realizations: a standard deviation of
    5 ps/nm (largest error 12) at rolloff 0.1 and 1 ps/nm at rolloff 0.35,
    without bias.  With 12 dB Es/N0, 10 ps of DGD and a 1 GHz frequency
    offset, from 0 to ±34000 ps/nm: within 62 ps/nm at rolloff 0.1 and
    7 ps/nm at rolloff 0.35 (CPU and GPU noise realizations).  That is a few
    km of standard fiber at most, a residual a short adaptive equalizer
    absorbs.

    **Limitations.**

    * The signal must have an excess band: the sampled band must hold
      ``(1 + rolloff) · R_s`` and the rolloff must be above zero.
    * A differential group delay near half a symbol at 45° fades the tone
      on both polarizations; the scan then finds no clear peak.
    * The frequency offset and the laser phase noise do not matter (the
      correlation is formed within each block), so the estimate can run
      before frequency and carrier recovery, as CDC does.
    """

    search_range: tuple[float, float] = (-40000.0, 40000.0)

    def __post_init__(self) -> None:
        lo, hi = (float(v) for v in self.search_range)
        if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
            raise ValueError(
                f"search_range must be a finite (start, stop) pair with start < "
                f"stop, got {self.search_range!r}."
            )
        object.__setattr__(self, "search_range", (lo, hi))


@dataclass(frozen=True)
class ChromaticDispersionEstimate:
    """Result of :func:`estimate_chromatic_dispersion`.

    The constructor takes the same fields.

    Attributes
    ----------
    value : ndarray
        Accumulated dispersion ``D·L`` in ps/nm, on the input's device: 0-d
        for ``(N,)`` input, ``(C,)`` for ``(C, N)``, every channel holding the
        same joint estimate (the channels share the fiber).
    method : ClockToneScan
        The method that produced the estimate.
    candidates : numpy.ndarray
        Scanned accumulated dispersions in ps/nm, ascending, ``(K,)``.
    metric : numpy.ndarray
        The scan's metric at ``candidates``, normalized to a peak of 1,
        ``(K,)``.
    """

    value: ArrayType
    method: ClockToneScan
    candidates: np.ndarray
    metric: np.ndarray


#: Smallest FFT block of the clock-tone scan, and the scan's zero padding.
_MIN_SCAN_BLOCK = 1024
_SCAN_PADDING = 16


def _ps_nm_to_beta2_length(ps_nm: float, wavelength_nm: float) -> float:
    return beta2_length(dispersion_ps_nm=ps_nm, center_wavelength_nm=wavelength_nm)


def _clock_tone(
    x: ArrayType,
    method: ClockToneScan,
    *,
    sampling_rate: float,
    symbol_rate: float,
    wavelength_nm: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """The clock-tone scan of ``(C, N)`` samples: estimate, candidates, metric."""
    x, xp, _ = dispatch(x)
    num_ch, n = x.shape
    lo, hi = method.search_range
    b_max = max(abs(_ps_nm_to_beta2_length(v, wavelength_nm)) for v in (lo, hi))
    spread = spread_samples(b_max, sampling_rate=sampling_rate, bandwidth=sampling_rate)
    nfft = max(
        _MIN_SCAN_BLOCK, 1 << max(int(math.ceil(8.0 * spread)) - 1, 1).bit_length()
    )
    if n < nfft:
        raise ValueError(
            f"estimate_chromatic_dispersion(): ClockToneScan needs blocks of {nfft} "
            f"samples for search_range={method.search_range!r}; the record has "
            f"{n}.  Narrow the range or pass a longer record."
        )
    m = int(round(symbol_rate / sampling_rate * nfft))
    if not 0 < m < nfft:
        raise ValueError(
            "estimate_chromatic_dispersion(): ClockToneScan needs a symbol rate "
            f"below the sampling rate; got symbol_rate={symbol_rate!r}, "
            f"sampling_rate={sampling_rate!r}."
        )
    num_blocks = n // nfft
    blocks = x[:, : num_blocks * nfft].reshape(num_ch, num_blocks, nfft)
    Y = xp.fft.fftshift(_fft.fft(blocks, axis=-1), axes=-1)
    # Correlations at +R_s and -R_s, summed over the blocks, on unwrapped
    # bin pairs only: a circular shift would fold in the other cyclic
    # frequency, whose scan peaks at the opposite dispersion.
    plus = xp.zeros((num_ch, nfft), dtype=Y.dtype)
    minus = xp.zeros((num_ch, nfft), dtype=Y.dtype)
    plus[:, m:] = xp.sum(Y[..., m:] * xp.conj(Y[..., :-m]), axis=1)
    minus[:, :-m] = xp.sum(Y[..., :-m] * xp.conj(Y[..., m:]), axis=1)
    size = nfft * _SCAN_PADDING
    k = xp.arange(size, dtype=xp.float64)
    # Bin k sits at frequency (k - nfft/2) fs/nfft; undo that offset.
    shift = xp.exp(-1j * np.pi * k * nfft / size)
    scan_plus = _fft.ifft(plus, n=size, axis=-1) * shift
    scan_minus = _fft.fft(minus, n=size, axis=-1) * xp.conj(shift)
    metric = xp.sum(xp.abs(scan_plus) ** 2 + xp.abs(scan_minus) ** 2, axis=0)
    metric_cpu = np.asarray(to_device(metric, "cpu"), dtype=np.float64)
    lag = np.arange(size)
    lag[lag >= size // 2] -= size
    tau = lag * nfft / (size * sampling_rate)  # s: β₂L·Ω
    omega_s = 2.0 * np.pi * m * sampling_rate / nfft
    beta = tau / omega_s
    lam = wavelength_nm * 1e-9
    ps_nm = -beta * 2.0 * np.pi * _SPEED_OF_LIGHT / lam**2 / 1e-3
    order = np.argsort(ps_nm)
    ps_nm, metric_cpu = ps_nm[order], metric_cpu[order]
    inside = np.flatnonzero((ps_nm >= lo) & (ps_nm <= hi))
    if inside.size < 3:
        raise ValueError(
            f"estimate_chromatic_dispersion(): search_range={method.search_range!r} "
            "holds fewer than 3 candidates; widen it."
        )
    candidates, values = ps_nm[inside], metric_cpu[inside]
    # The centroid of the main lobe above half its maximum: the peak is
    # broad (about 1 / (rolloff R_s Ω) in β₂L) and noise ripples its top, so
    # its centre of mass is far steadier than its maximum (measured at
    # rolloff 0.1: ±5 ps/nm against ±66 for the maximum).
    i = int(np.argmax(values))
    half = 0.5 * values[i]
    a, b = i, i
    while a > 0 and values[a - 1] > half:
        a -= 1
    while b < values.size - 1 and values[b + 1] > half:
        b += 1
    weight = values[a : b + 1] - half
    estimate = float(
        np.sum(candidates[a : b + 1] * weight) / np.sum(weight)
        if np.sum(weight) > 0
        else candidates[i]
    )
    strongest = float(ps_nm[int(np.argmax(metric_cpu))])
    if not lo <= strongest <= hi:
        logger.warning(
            "estimate_chromatic_dispersion(): the strongest clock-tone peak, at "
            "%.0f ps/nm, lies outside search_range=%s; the estimate %.0f ps/nm "
            "is a side lobe. Widen the range.",
            strongest,
            method.search_range,
            estimate,
        )
    peak = float(values.max())
    return estimate, candidates, values / peak if peak > 0 else values


def estimate_chromatic_dispersion(
    samples: ArrayType | Any,
    method: ClockToneScan,
    *,
    center_wavelength_nm: float | None = None,
    sampling_rate: float | None = None,
    symbol_rate: float | None = None,
) -> ChromaticDispersionEstimate:
    """Estimate the accumulated chromatic dispersion of a received signal.

    Blind: needs no reference, no carrier or frequency recovery, only the
    symbol rate.  The estimate is the link's accumulated dispersion ``D·L``
    in ps/nm, the value :func:`correct_chromatic_dispersion` takes (pass
    the estimate itself as its ``how``).  All channels are used jointly,
    since they share the fiber.

    Parameters
    ----------
    samples : array_like or Signal
        Complex baseband samples, ``(N,)`` or ``(C, N)``, at two or more
        samples per symbol in practice (the band must hold the pulse's
        excess band).
    method : ClockToneScan
        The estimator and its search range.
    center_wavelength_nm : float, optional
        Carrier wavelength in nm.  Taken from an optical Signal's carrier;
        required for array input and Signals of another domain.
    sampling_rate : float, optional
        Sampling rate in Hz.  Taken from the Signal; required for array
        input.  A value that disagrees with the Signal raises.
    symbol_rate : float, optional
        Symbol rate in Hz.  Taken from the Signal; required for array input.
        A value that disagrees with the Signal raises.

    Returns
    -------
    ChromaticDispersionEstimate
        The estimate in ps/nm and the scan (candidates and metric).

    Raises
    ------
    TypeError
        If ``method`` is not a :class:`ClockToneScan`.
    ValueError
        If the record is shorter than one block of the scan, or the search
        range holds too few candidates.

    Examples
    --------
    >>> est = estimate_chromatic_dispersion(capture, ClockToneScan())
    >>> y = correct_chromatic_dispersion(capture, est)
    >>> taps = dispersion_compensation_taps(  # the filter it applied
    ...     dispersion_ps_nm=float(est.value), center_wavelength_nm=1550.0,
    ...     sampling_rate=64e9)
    """
    if not isinstance(method, ClockToneScan):
        raise TypeError(
            "estimate_chromatic_dispersion(): method must be a ClockToneScan, got "
            f"{type(method).__name__}."
        )
    signal_adapter = adapt_signal(
        samples, function_name="estimate_chromatic_dispersion()"
    )
    fs = float(signal_adapter.resolve_fact("sampling_rate", sampling_rate))
    rs = float(signal_adapter.resolve_fact("symbol_rate", symbol_rate))
    lam = signal_adapter.resolve_wavelength_nm(center_wavelength_nm)
    x, xp, _ = dispatch(signal_adapter.array)
    x2, was_1d = as_2d(x, name="samples")
    value, candidates, metric = _clock_tone(
        x2, method, sampling_rate=fs, symbol_rate=rs, wavelength_nm=lam
    )
    logger.info(
        "Estimated chromatic dispersion: %.1f ps/nm (clock tone, λ=%s nm).",
        value,
        lam,
    )
    out = xp.full((x2.shape[0],), value, dtype=xp.float64)
    return ChromaticDispersionEstimate(
        value=out[0] if was_1d else out,
        method=method,
        candidates=candidates,
        metric=metric,
    )


def _dispersion_value(value: Any, function_name: str) -> float:
    """One accumulated dispersion in ps/nm from a number or an estimate's
    value (0-d, or ``(C,)`` of one joint estimate)."""
    v = np.asarray(to_device(value, "cpu") if hasattr(value, "shape") else value)
    v = v.astype(np.float64).ravel()
    if (
        v.size == 0
        or not np.all(np.isfinite(v))
        or np.ptp(v) > 1e-9 * max(1.0, float(np.max(np.abs(v))))
    ):
        raise ValueError(
            f"{function_name}: dispersion_ps_nm must be one finite value (or an "
            f"estimate's joint value), got {value!r}."
        )
    return float(v[0])


# -----------------------------------------------------------------------------
# CHROMATIC DISPERSION (Signal-aware)
# -----------------------------------------------------------------------------


def correct_chromatic_dispersion(
    samples: S,
    how: ChromaticDispersionEstimate | ClockToneScan | None = None,
    *,
    dispersion_ps_nm: float | None = None,
    center_wavelength_nm: float | None = None,
    sampling_rate: float | None = None,
    method: str = "fir",
    bandwidth: float | None = None,
    num_taps: int | None = None,
) -> S:
    """Compensate chromatic dispersion: the inverse fiber response.

    Filters with ``H(ω) = exp(+j β₂L ω² / 2)``, ``β₂L = -D·L·λ² / (2π c)``,
    which undoes the dispersion of a link of accumulated dispersion
    ``D·L``, either as the FIR filter of :func:`dispersion_compensation_taps`
    applied by overlap-save (the default, as a receiver does) or by
    multiplying the whole record's spectrum by the exact response.  Both
    work on a capture and on a simulated record; see Notes for the choice.

    Parameters
    ----------
    samples : array_like or Signal
        Complex baseband samples, ``(N,)`` or ``(C, N)``.
    how : ChromaticDispersionEstimate or ClockToneScan, optional
        An estimate of :func:`estimate_chromatic_dispersion`, or a method
        (estimated first, with the Signal's facts; estimate first for array
        input).  Give ``how`` or ``dispersion_ps_nm``, not both.
    dispersion_ps_nm : float, optional
        Known accumulated dispersion of the link ``D·L`` in ps/nm (standard
        fiber: 17 per km at 1550 nm, so 1360 for 80 km).  The value of the
        fiber, not its negative: the function inverts it.
    center_wavelength_nm : float, optional
        Carrier wavelength in nm.  Taken from an optical Signal's
        carrier (``domain="OPT"``, ``c / center_frequency``); required for
        array input and Signals of another domain.  A value that disagrees
        with the optical Signal raises.
    sampling_rate : float, optional
        Sampling rate in Hz.  Taken from the Signal; required for array
        input.  A value that disagrees with the Signal raises.
    method : {"fir", "whole_record"}, default "fir"
        ``"fir"``: the FIR of :func:`dispersion_compensation_taps`, applied
        by overlap-save (:func:`ols_fir_filter`), in bounded memory.
        ``"whole_record"``: the exact response on the FFT grid of the whole
        record, which is transformed at once and treated as one period.
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
        On a capture the first and last half spread (about
        ``(num_taps - 1) // 2`` samples) are wrong with either method, see
        Notes.

    Raises
    ------
    ValueError
        If neither or both of ``how`` and ``dispersion_ps_nm`` are given,
        ``method`` is unknown, or ``bandwidth`` or ``num_taps`` comes with
        ``method="whole_record"``.
    TypeError
        If ``how`` is not an estimate or a :class:`ClockToneScan`.

    See Also
    --------
    estimate_chromatic_dispersion : The accumulated dispersion of a capture.
    dispersion_compensation_taps : The FIR taps, to inspect or apply.

    Notes
    -----
    **Choosing a method.**

    * A record from :func:`commkit.impairments.apply_chromatic_dispersion`
      (or one AWG period) is circular: ``"whole_record"`` inverts it
      exactly at every sample, the FIR to about -45 dB.
    * A capture, a window of an endless stream: within half the spread of
      either end the history the inverse needs lies outside the capture, so
      those samples are wrong whatever the method (FIR transients, or the
      other end wrapped in by ``"whole_record"``).  Trim them.  Beyond them
      both are right and ``"whole_record"`` is the more precise: measured on
      a window of an 80 km, 2-sps record, both are wrong over the first and
      last 16 samples, the FIR then holds -45 dB, ``"whole_record"`` -30 dB
      over the next 16, -68 dB beyond 50 and -110 dB deep inside.
    * So ``"whole_record"`` is a correct choice for a capture that fits in
      memory; ``"fir"`` is the receiver's form, for streams of any length
      in bounded memory and for studying the filter.

    The FIR is centred, so it adds no delay and the reference stays aligned.
    Its default length is 1.25 times the dispersion's group-delay spread
    over the fitted band, about -45 dB of residual against the exact
    response on a signal inside that band; fitting the occupied band rather
    than the whole sampled band needs about half the taps.  To see the
    filter that is applied, design the same taps with
    :func:`dispersion_compensation_taps` and plot them with
    :func:`commkit.plotting.plot_filter_response`.

    Examples
    --------
    >>> y = correct_chromatic_dispersion(  # a capture after 80 km of SSMF
    ...     capture, dispersion_ps_nm=1360.0, bandwidth=35.2e9)
    >>> y = correct_chromatic_dispersion(capture, ClockToneScan())  # unknown D·L
    >>> y = correct_chromatic_dispersion(  # a circularly simulated record
    ...     sig, dispersion_ps_nm=1360.0, method="whole_record")
    """
    signal_adapter = adapt_signal(
        samples, function_name="correct_chromatic_dispersion()"
    )
    sampling_rate = signal_adapter.resolve_fact("sampling_rate", sampling_rate)
    center_wavelength_nm = signal_adapter.resolve_wavelength_nm(center_wavelength_nm)
    name = "correct_chromatic_dispersion()"
    if (how is None) == (dispersion_ps_nm is None):
        raise ValueError(
            f"{name}: give the dispersion as an estimate or method (how) or as "
            "dispersion_ps_nm, exactly one of them."
        )
    if isinstance(how, ClockToneScan):
        how = estimate_chromatic_dispersion(
            samples,
            how,
            center_wavelength_nm=center_wavelength_nm,
            sampling_rate=sampling_rate,
        )
    if how is not None:
        if not isinstance(how, ChromaticDispersionEstimate):
            raise TypeError(
                f"{name}: how must be a ChromaticDispersionEstimate or a "
                f"ClockToneScan, got {type(how).__name__}."
            )
    raw: Any = dispersion_ps_nm if how is None else how.value
    dispersion_ps_nm = _dispersion_value(raw, name)
    if method not in ("fir", "whole_record"):
        raise ValueError(
            "correct_chromatic_dispersion(): method must be 'fir' or "
            f"'whole_record', got {method!r}."
        )
    if method == "whole_record":
        if bandwidth is not None or num_taps is not None:
            raise ValueError(
                "correct_chromatic_dispersion(): bandwidth and num_taps design "
                "the FIR; method='whole_record' applies the exact response and "
                "takes neither."
            )
        logger.info(
            "Correcting CD of %s ps/nm (λ=%s nm) over the whole record.",
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
    dispersion_ps_nm : float or array_like
        Accumulated dispersion of the link ``D·L`` in ps/nm, or the
        ``value`` of a :class:`ChromaticDispersionEstimate`.
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
    dispersion_ps_nm = _dispersion_value(
        dispersion_ps_nm, "dispersion_compensation_taps()"
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
    method: str = "blocks",
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

    ``method="blocks"`` (the default, as a receiver does) backpropagates in
    overlapping blocks, each with a guard of 1.25 times half the link's
    dispersion memory on both sides, and discards the guards
    (overlap-save), so the working memory is bounded by the block.  The
    dispersion's light cone bounds both the linear and the nonlinear
    interaction, so every kept sample sees all of its past and future within
    the link (measured: the blocks match the whole record to -71 dB).
    ``method="whole_record"`` backpropagates the whole record at once, as
    one period: the exact inverse of
    :func:`commkit.impairments.apply_fiber_propagation`, which propagates
    its record circularly.  On a capture both are wrong within half the
    link's memory of either end, whose history lies outside the capture, and
    right beyond it (see :func:`correct_chromatic_dispersion` for the same
    choice in the linear case).

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
    method : {"blocks", "whole_record"}, default "blocks"
        ``"blocks"``: overlapping blocks in bounded memory.
        ``"whole_record"``: the whole record at once, as one period (exact
        on a circularly simulated record).

    Returns
    -------
    ndarray or Signal
        The backpropagated field in the input's units, complex, on the
        input's device.  A :class:`Signal` returns a new :class:`Signal`.
        On a capture the first and last half of the link's dispersion
        memory are wrong with either method.

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
    if method not in ("blocks", "whole_record"):
        raise ValueError(
            f"{name}: method must be 'blocks' or 'whole_record', got {method!r}."
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

    if method == "whole_record":
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
