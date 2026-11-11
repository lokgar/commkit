"""Sampling-clock recovery.

:func:`estimate_sampling_clock_offset` measures the offset of the receiver's
sampling clock from the transmitter's symbol clock (a
:class:`SamplingClockOffsetEstimate`), and :func:`correct_sampling_clock_offset`
resamples the signal onto the transmitter's symbol grid, from an estimate or
by tracking the strobes with a :class:`Gardner` loop.

Convention, as :func:`~commkit.impairments.apply_sampling_clock_offset`: an
offset ``ε`` (in ppm) means the receiver takes ``sps · (1 + ε)`` samples per
transmitted symbol.  Timing is measured in symbol periods; the timing
trajectory ``τ_k`` is the strobe of symbol ``k`` minus ``k``, so its slope is
``ε``.
"""

import logging
import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .._array import as_2d, restore_1d, warn_if_double
from ..backend import ArrayType, dispatch, to_device
from ..core._signal_adapter import S, adapt_signal, require_integer_sps
from ..core.signal import Signal
from ..logger import logger
from ..multirate import _interpolate, _stretched_instants

__all__ = [
    "Gardner",
    "SamplingClockOffsetEstimate",
    "correct_sampling_clock_offset",
    "estimate_sampling_clock_offset",
]

# Open-loop calibration: detector means at _CAL_OFFSETS strobe offsets
# across one symbol, over at most _CAL_SYMBOLS symbols.
_CAL_OFFSETS = 8
_CAL_SYMBOLS = 2048
# Below this first-harmonic amplitude of the S-curve (unit-power input) the
# detector sees no clock tone.
_MIN_CURVE_AMPLITUDE = 1e-3
# Cycle-slip check: the timing may not move by more than _SLIP_THRESHOLD
# symbol periods, beyond the record's mean drift, within _SLIP_WINDOW loop
# time constants 1/(B·T).  The loop's own jitter stays below 0.5 symbol over
# that window up to the bandwidths where slips begin; a slip moves it by 1.
_SLIP_WINDOW = 8.0
_SLIP_THRESHOLD = 0.75


def _get_numba_gardner() -> Any:
    """The tracking loop."""
    from . import _numba_loops

    return _numba_loops.gardner_loop


def _get_numba_ted_curve() -> Any:
    """The open-loop detector means."""
    from . import _numba_loops

    return _numba_loops.ted_curve


@dataclass(frozen=True)
class Gardner:
    """Gardner clock recovery: a timing-error detector in a second-order loop.

    The detector compares the samples at two successive symbol instants
    ``y`` with the one halfway between them ``m``, interpolated from the
    input (16-tap windowed sinc) at the strobes the loop sets.  Its output
    ``e_k`` drives the strobe spacing ``sps · (1 + ν + μ·e_k)`` one symbol
    later, with the integrator ``ν += β·e_k`` absorbing the clock offset.

    Parameters
    ----------
    normalized_bandwidth : float, default 1e-3
        Nominal one-sided loop bandwidth ``B·T`` in ``(0, 0.5)``, a fraction
        of the symbol rate.  Gives critically damped gains ``μ = 4BT / K_d``
        and ``β = 4(BT)² / K_d``, with the detector gain ``K_d`` measured on
        the input (see Notes): ``ζ = 1`` and ``ω_n T = 2BT``, whose noise
        bandwidth is ``1.25·B·T`` (``B·T`` is exact for a first-order loop
        only).  Used when ``proportional_gain`` is not given.
        The detector's self-noise sets the timing jitter: 0.01 symbol rms at
        ``1e-3`` for 16-QAM (classical detector, rolloff 0.1), 0.006 at
        ``1e-4``.  Too wide a loop slips whole symbols: the classical
        detector at a rolloff of 0.1 from ``1e-2`` on, the power detector
        at 0.1 from ``2e-2`` and at 0.5 from ``3e-3`` (16-QAM, 20 dB).
    proportional_gain : float, optional
        Raw proportional gain ``μ`` on the detector output of the input
        scaled to unit power; overrides ``normalized_bandwidth``.
    integral_gain : float, optional
        Raw integral gain ``β``; requires ``proportional_gain``.  Defaults
        to 0 (a first-order loop, which keeps a timing lag ``ε / μK_d``
        under a clock offset) with ``proportional_gain``.
    max_offset_ppm : float, default 1000.0
        Largest clock offset the loop tracks, in ppm, in ``(0, 90000)``:
        the integrator ``ν`` is held within ``±max_offset_ppm · 1e-6``.
        Crystal clocks are within about 100 ppm.  The bound keeps a loop
        that has lost lock from wandering off in frequency, which it
        otherwise does without limit (see Notes).
    joint_channels : bool, default False
        MIMO: one loop driven by the mean detector output of all channels,
        for channels sampled by one clock.
    method : {"classical", "power"}, default "classical"
        The detector.  ``"classical"`` is Gardner's
        ``e = Re{m_k · conj(y_{k-1} - y_k)}``, whose gain falls in proportion
        to the pulse's excess bandwidth (0.28 at a rolloff of 0.1 against
        1.55 at 0.5, 16-QAM through a matched RRC filter).  ``"power"`` applies
        it to the instantaneous power, ``e = -|m_k|² (|y_{k-1}|² - |y_k|²)``,
        and keeps its gain at small rolloffs (1.75 at 0.1).  It needs a
        constellation with more than one ring of power, so the classical
        detector suits QPSK.

    Notes
    -----
    Both detectors have a sinusoidal S-curve in the timing error.  Before the
    loop starts, the detector means at eight strobe offsets across one
    symbol (over the first 2048 symbols, open loop) are fitted with
    ``-A sin(2π(τ - τ_0))``: ``K_d = 2πA`` per symbol period scales the
    gains, and ``τ_0`` starts the loop at the right strobe phase, so there
    is no acquisition transient.  Input whose fitted amplitude is below 1e-3
    (no clock tone, as for an unmodulated carrier) raises.  The classical
    amplitude falls to 0.006 at a zero rolloff (16-QAM, from the truncated
    pulse) while the power detector's stays at 0.34.  The strobe spacing is
    held within ±10 % of ``sps``.

    A loop that slips leaves the S-curve's stable point, and while its
    timing runs through the symbols the mean detector output is zero:
    nothing pulls the integrator back, and its noise walks it away.
    Unbounded, a classical loop at ``B·T = 1e-2`` on a 0.1-rolloff signal
    reached tens of thousands of ppm on a 50 ppm clock; held within
    ``max_offset_ppm`` it keeps returning to lock.  Two checks log a WARNING after the loop: the
    timing moving by more than 0.75 symbol within ``8 / (B·T)`` symbols,
    beyond the record's mean drift (a cycle slip, after which symbol ``k``
    of the output is no longer the ``k``-th transmitted one), and an
    estimated offset within 5 % of ``max_offset_ppm`` (a larger offset,
    which the proportional path follows only with a static timing lag, or
    a loop that lost lock).  ``B·T`` is ``μ K_d / 4`` for raw gains.

    Numba-compiled on the CPU; GPU input makes one host round trip.  The
    loop runs in complex64 whatever the input (double input emits
    :class:`~commkit.PrecisionWarning`); the interpolation of
    :func:`correct_sampling_clock_offset` keeps the input's precision.

    References
    ----------
    F. M. Gardner, "A BPSK/QPSK timing-error detector for sampled
    receivers," IEEE Trans. Commun., vol. 34, no. 5, pp. 423-429, 1986.
    """

    normalized_bandwidth: float = 1e-3
    proportional_gain: float | None = None
    integral_gain: float | None = None
    max_offset_ppm: float = 1000.0
    joint_channels: bool = False
    method: Literal["classical", "power"] = "classical"

    def __post_init__(self) -> None:
        bt = self.normalized_bandwidth
        if not (0.0 < bt < 0.5):
            raise ValueError(
                f"Gardner: normalized_bandwidth must be in (0, 0.5) (a fraction "
                f"of the symbol rate), got {bt}."
            )
        if not (0.0 < self.max_offset_ppm < 9e4):
            raise ValueError(
                "Gardner: max_offset_ppm must be in (0, 90000), got "
                f"{self.max_offset_ppm}."
            )
        if self.integral_gain is not None and self.proportional_gain is None:
            raise ValueError("Gardner: integral_gain requires proportional_gain.")
        for name in ("proportional_gain", "integral_gain"):
            value = getattr(self, name)
            if value is not None and not value >= 0.0:
                raise ValueError(f"Gardner: {name} must be non-negative, got {value}.")
        if self.method not in ("classical", "power"):
            raise ValueError(
                f"Gardner: method must be 'classical' or 'power', got {self.method!r}."
            )

    def gains(self, detector_gain: float) -> tuple[float, float]:
        """The loop gains ``(μ, β)`` for a detector gain ``K_d``."""
        if self.proportional_gain is not None:
            return self.proportional_gain, self.integral_gain or 0.0
        bt = self.normalized_bandwidth
        return 4.0 * bt / detector_gain, 4.0 * bt * bt / detector_gain


@dataclass(frozen=True)
class SamplingClockOffsetEstimate:
    """Result of :func:`estimate_sampling_clock_offset`; the constructor
    takes the same fields.

    Per-channel fields follow the rank rule (0-d for ``(N,)`` input, ``(C,)``
    for ``(C, N)``; one value per channel also for a joint loop) and stay on
    the input's device.

    Attributes
    ----------
    value : ndarray
        Clock offset ``ε`` in ppm: the receiver takes ``sps · (1 + ε)``
        samples per transmitted symbol.
    trajectory : ndarray
        Timing ``τ_k`` of the strobe of symbol ``k`` in symbol periods
        (the strobe's position in samples over ``sps``, minus ``k``),
        float64, ``(K,)`` or ``(C, K)``.
    detector_gain : ndarray
        Measured detector gain ``K_d`` per symbol period (see
        :class:`Gardner`).
    """

    value: ArrayType
    trajectory: ArrayType
    detector_gain: ArrayType


def _track(
    x: np.ndarray, sps: int, how: Gardner
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the loop on host data ``(C, N)``: strobe positions ``(C, K)`` in
    unpadded samples (rows repeated for a joint loop), the detector gain
    and the detector outputs."""
    C, n = x.shape
    if n < 4 * sps:
        raise ValueError(
            f"Gardner clock recovery needs at least 4 symbols, got {n} samples."
        )
    warn_if_double(x, "Gardner clock recovery")
    rms = np.sqrt(np.mean(np.abs(x) ** 2, axis=-1, dtype=np.float64))
    if np.any(rms == 0.0):
        raise ValueError("Gardner clock recovery: a channel is all zeros.")
    pad = 2 * sps + 10
    xn = np.zeros((C, n + 2 * pad), dtype=np.complex64)
    xn[:, pad : pad + n] = x / rms[:, None].astype(np.float32)
    power = how.method == "power"

    num_cal = min(_CAL_SYMBOLS, n // sps - 3)
    offsets = np.arange(_CAL_OFFSETS) / _CAL_OFFSETS
    curves = _get_numba_ted_curve()(xn, pad, float(sps), offsets, num_cal, power)
    if how.joint_channels:
        curves = curves.mean(axis=0, keepdims=True)
    # e(τ) = a cos 2πτ + b sin 2πτ = -A sin(2π(τ - τ0))
    a = 2.0 * np.mean(curves * np.cos(2 * np.pi * offsets), axis=-1)
    b = 2.0 * np.mean(curves * np.sin(2 * np.pi * offsets), axis=-1)
    amplitude = np.hypot(a, b)
    if np.any(amplitude < _MIN_CURVE_AMPLITUDE):
        raise ValueError(
            f"Gardner(method={how.method!r}): the detector sees no clock tone "
            f"(S-curve amplitude {float(amplitude.min()):.1e}), as for an "
            "unmodulated carrier."
        )
    tau0 = np.arctan2(a, -b) / (2 * np.pi)
    tau0 = (tau0 + 0.5) % 1.0 - 0.5  # the strobe of symbol 0, in [-0.5, 0.5)
    kd = 2 * np.pi * amplitude

    rows = curves.shape[0]
    gains = np.array([how.gains(float(g)) for g in kd])
    pos, err, count = _get_numba_gardner()(
        xn,
        pad,
        float(sps),
        pad + tau0 * sps,
        np.ascontiguousarray(gains[:, 0]),
        np.ascontiguousarray(gains[:, 1]),
        how.max_offset_ppm * 1e-6,
        power,
        rows != C,
        n,
        int(n / sps * 1.1) + 4,
    )
    k = int(count.min())
    pos = pos[:, :k] - pad
    err = err[:, :k]
    _check_lock(pos / sps - np.arange(k), gains[:, 0] * kd / 4.0, how)
    if rows != C:
        pos = np.repeat(pos, C, axis=0)
        err = np.repeat(err, C, axis=0)
        kd = np.repeat(kd, C)
    return pos, kd, err


def _first_slip(trajectory: np.ndarray, bandwidth: np.ndarray) -> np.ndarray:
    """Per row, the first symbol where the timing moves by more than
    _SLIP_THRESHOLD within _SLIP_WINDOW / bandwidth symbols beyond the
    row's median drift over that window; -1 where it never does."""
    k = trajectory.shape[-1]
    first = np.full(trajectory.shape[0], -1, dtype=np.int64)
    for r, bt in enumerate(bandwidth):
        if not bt > 0.0:
            continue
        span = int(max(64.0, _SLIP_WINDOW / bt))
        if span >= k:
            continue
        moved = trajectory[r, span:] - trajectory[r, :-span]
        bad = np.abs(moved - np.median(moved)) > _SLIP_THRESHOLD
        if bad.any():
            first[r] = int(np.argmax(bad))
    return first


def _check_lock(trajectory: np.ndarray, bandwidth: np.ndarray, how: Gardner) -> None:
    """Log a WARNING for cycle slips and for an offset at the loop's bound."""
    first = _first_slip(trajectory, bandwidth)
    if np.any(first >= 0):
        logger.warning(
            "Gardner clock recovery slipped whole symbols (first near symbol "
            "%d): after a slip, output symbols no longer match the transmitted "
            "ones in order.  Use a smaller normalized_bandwidth (B·T = %s).",
            int(first[first >= 0].min()),
            ", ".join(f"{bt:.1e}" for bt in np.unique(bandwidth)),
        )
    ppm = _offset_ppm(trajectory)
    if np.any(np.abs(ppm) > 0.95 * how.max_offset_ppm):
        logger.warning(
            "Gardner clock recovery: the offset %s ppm is at the loop's bound "
            "max_offset_ppm=%g: the loop lost lock, or the clock is further off and "
            "the timing lags; raise max_offset_ppm.",
            np.array2string(ppm, precision=1),
            how.max_offset_ppm,
        )


def _offset_ppm(trajectory: np.ndarray) -> np.ndarray:
    """Least-squares slope of the trajectory over its last three quarters,
    in ppm."""
    k = trajectory.shape[-1]
    t = trajectory[:, k // 4 :]
    idx = np.arange(k // 4, k, dtype=np.float64)
    idx -= idx.mean()
    slope = (t - t.mean(axis=-1, keepdims=True)) @ idx / (idx @ idx)
    return slope * 1e6


def _gardner_estimate(
    x: ArrayType, xp: Any, sps: int, how: Gardner
) -> SamplingClockOffsetEstimate:
    pos, kd, _ = _track(to_device(x, "cpu"), sps, how)
    trajectory = pos / sps - np.arange(pos.shape[-1])
    return SamplingClockOffsetEstimate(
        value=_offset_ppm(trajectory), trajectory=trajectory, detector_gain=kd
    )


_ESTIMATORS = {Gardner: _gardner_estimate}


def estimate_sampling_clock_offset(
    samples: ArrayType | Signal, method: Gardner, *, sps: int | None = None
) -> SamplingClockOffsetEstimate:
    """Estimate the offset of the sampling clock from the symbol clock.

    Tracks the symbol strobes with the loop of ``method`` and fits a
    straight line to the timing trajectory over its last three quarters;
    the slope is the offset.

    Parameters
    ----------
    samples : array_like or Signal
        Matched-filter output, ``(N,)`` or ``(C, N)``, at an even integer
        number of samples per symbol.
    method : Gardner
        The clock-recovery loop.
    sps : int, optional
        Samples per symbol, even.  Taken from the Signal; required for array
        input.  A value that disagrees with the Signal raises.

    Returns
    -------
    SamplingClockOffsetEstimate
        The offset in ppm, the timing trajectory and the detector gain, on
        the input's device.

    Raises
    ------
    TypeError
        If ``method`` is not a :class:`Gardner`.
    ValueError
        If ``sps`` is missing, not an even integer or conflicts with the
        Signal, or the detector sees no clock tone.

    Examples
    --------
    >>> est = estimate_sampling_clock_offset(matched_filter(rx), Gardner())
    >>> float(est.value)  # ppm
    """
    estimate_fn = _ESTIMATORS.get(type(method))
    if estimate_fn is None:
        raise TypeError(
            "estimate_sampling_clock_offset(): method must be Gardner(), got "
            f"{type(method).__name__}."
        )
    signal_adapter = adapt_signal(
        samples, function_name="estimate_sampling_clock_offset()"
    )
    sps_int = _resolve_even_sps(signal_adapter, sps, "estimate_sampling_clock_offset()")
    x, xp, _ = dispatch(signal_adapter.array)
    x, was_1d = as_2d(x, name="samples")
    est = estimate_fn(x, xp, sps_int, method)
    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "Sampling-clock offset: %s ppm (Gardner, %s detector).",
            np.array2string(np.asarray(est.value), precision=3),
            method.method,
        )
    return SamplingClockOffsetEstimate(
        value=restore_1d(was_1d, xp.asarray(est.value)),
        trajectory=restore_1d(was_1d, xp.asarray(est.trajectory)),
        detector_gain=restore_1d(was_1d, xp.asarray(est.detector_gain)),
    )


def correct_sampling_clock_offset(
    samples: S,
    how: Gardner | SamplingClockOffsetEstimate,
    *,
    sps: int | None = None,
) -> S:
    """Resample a signal onto the transmitter's symbol clock.

    With a :class:`Gardner`, tracks the strobes and samples the waveform at
    ``sps`` evenly spaced instants per tracked symbol, starting at its
    strobe: the clock offset, its drift and the static timing phase are
    removed, and symbol ``k`` of the output is the ``k``-th symbol of the
    record.  With a :class:`SamplingClockOffsetEstimate` (possibly of an
    earlier capture), resamples by the constant factor ``1 + ε`` of its
    ``value`` only: the drift is removed and a static timing phase remains,
    for the equalizer.

    The waveform between samples comes from a 48-tap windowed-sinc
    interpolator in the input's precision (error below -119 dB for content
    up to 0.4 cycles per sample, as in
    :func:`~commkit.impairments.apply_sampling_clock_offset`).

    Parameters
    ----------
    samples : array_like or Signal
        Matched-filter output, ``(N,)`` or ``(C, N)``, at an even integer
        number of samples per symbol.
    how : Gardner or SamplingClockOffsetEstimate
        The loop, or the offset to remove (0-d, or ``(C,)`` per channel).
    sps : int, optional
        Samples per symbol, even.  Taken from the Signal; required for array
        input.  A value that disagrees with the Signal raises.

    Returns
    -------
    ndarray or Signal
        The resampled samples at ``sps``, same dtype and device; channels
        are trimmed to a common length.  A Signal keeps the reference
        symbols of the symbol periods present.

    Raises
    ------
    TypeError
        If ``how`` is neither.
    ValueError
        As :func:`estimate_sampling_clock_offset`.

    Notes
    -----
    The loop resolves timing within a symbol: a delay of whole symbols, as
    of a frame start, is left to :func:`estimate_timing`.

    Examples
    --------
    >>> y = correct_sampling_clock_offset(matched_filter(rx), Gardner())
    >>> y = correct_sampling_clock_offset(
    ...     next_capture, estimate_sampling_clock_offset(rx, Gardner())
    ... )
    """
    if not isinstance(how, (Gardner, SamplingClockOffsetEstimate)):
        raise TypeError(
            "correct_sampling_clock_offset(): how must be Gardner() or a "
            f"SamplingClockOffsetEstimate, got {type(how).__name__}."
        )
    function_name = "correct_sampling_clock_offset()"
    signal_adapter = adapt_signal(samples, function_name=function_name)
    sps_int = _resolve_even_sps(signal_adapter, sps, function_name)
    x, xp, _ = dispatch(signal_adapter.array)
    x, was_1d = as_2d(x, name="samples")
    real = x.real.dtype
    if isinstance(how, Gardner):
        pos, _, _ = _track(to_device(x, "cpu"), sps_int, how)
        logger.info(
            "Sampling clock recovered (Gardner, %s detector): %d symbols.",
            how.method,
            pos.shape[-1],
        )
        y = _resample_on_strobes(x, xp, pos, sps_int, real)
    else:
        ppm = np.atleast_1d(np.asarray(to_device(how.value, "cpu"), dtype=np.float64))
        if ppm.size not in (1, x.shape[0]):
            raise ValueError(
                f"{function_name}: the estimate has {ppm.size} values for "
                f"{x.shape[0]} channels."
            )
        ppm = np.broadcast_to(ppm, (x.shape[0],))
        logger.info(
            "Removing a sampling-clock offset of %s ppm.",
            np.array2string(ppm, precision=3),
        )
        y = _resample_static(x, xp, ppm, real)
    out = restore_1d(was_1d, y)
    sig = signal_adapter.signal
    if sig is not None and sig.reference is not None:
        num_symbols = int(out.shape[-1] // sps_int)
        return signal_adapter.wrap_samples(
            out, reference=sig.reference.head(num_symbols)
        )
    return signal_adapter.wrap_samples(out)


def _resolve_even_sps(signal_adapter: Any, sps: int | None, function_name: str) -> int:
    sps_value = signal_adapter.resolve_fact("sps", sps)
    sps_int = require_integer_sps(sps_value, function_name=function_name)
    if sps_int < 2 or sps_int % 2:
        raise ValueError(
            f"{function_name}: Gardner clock recovery needs an even sps "
            f"(a sample halfway between symbols), got {sps_int}."
        )
    return sps_int


def _resample_on_strobes(
    x: ArrayType, xp: Any, pos: np.ndarray, sps: int, real: Any
) -> ArrayType:
    """``sps`` instants per strobe interval, ``P_k + j·(P_{k+1} - P_k)/sps``,
    per channel; the instants are split into an integer index and a
    fraction in ``real`` precision.  The first strobe lies at most half a
    symbol before sample 0 and the last instants at most a symbol after the
    record's end: ``margin=sps`` covers both."""
    C, k = pos.shape
    step = np.diff(pos, axis=-1, append=2 * pos[:, -1:] - pos[:, -2:-1])
    base_k = np.floor(pos)
    frac_k = (pos - base_k).astype(np.float32)  # symbol-rate, in [0, 1)
    out = xp.empty((C, k * sps), dtype=x.dtype)
    j = xp.arange(sps, dtype=real)
    for c in range(C):
        q = (
            xp.asarray(frac_k[c], dtype=real)[:, None]
            + j[None, :] * xp.asarray(step[c] / sps, dtype=real)[:, None]
        )
        whole = xp.floor(q)
        base = (
            xp.asarray(base_k[c], dtype=xp.int64)[:, None] + whole.astype(xp.int64)
        ).ravel()
        frac = (q - whole).ravel()
        out[c] = _interpolate(x[c : c + 1], base, frac, margin=sps)[0]
    return out


def _resample_static(x: ArrayType, xp: Any, ppm: np.ndarray, real: Any) -> ArrayType:
    """Output ``n`` at input ``n · (1 + ε)``, per channel, trimmed to the
    shortest."""
    n_in = x.shape[-1]
    rates = 1.0 + ppm * 1e-6
    n_out = min(math.floor((n_in - 1) / r) + 1 for r in rates)
    out = xp.empty((x.shape[0], n_out), dtype=x.dtype)
    for c, rate in enumerate(rates):
        base, frac = _stretched_instants(n_out, float(rate), real, xp)
        out[c] = _interpolate(x[c : c + 1], base, frac)[0]
    return out
