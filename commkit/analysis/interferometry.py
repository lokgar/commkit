r"""Delayed self-heterodyne / self-homodyne (DSH) laser characterization.

Estimate a CW laser's FM-noise PSD and linewidth from the digitized *beat* of
a delayed self-interference measurement: the laser under test is split in two,
one arm is delayed by ``τ_d`` (a fiber spool), the other is optionally
frequency-shifted by ``f_shift`` (an AOM), and the arms are recombined on a
photodetector.  The beat at ``f_shift`` carries the **differential** laser
phase

    Δφ(t) = φ(t) - φ(t - τ_d),

i.e. the interferometer converts the laser's absolute phase noise - which is
unobservable without a second, better laser - into a measurable quantity by
using the laser itself, delayed toward or beyond its own coherence time, as
the reference.

The AOM shift and the receiver are independent choices, and ``dsh_phase``
dispatches on the input *dtype* rather than on a named variant, so three
combinations are supported:

* **heterodyne, single photodetector** - ``f_shift ≠ 0``, *real* samples;
  the analytic signal is formed internally (Hilbert), so the whole beat
  lineshape must sit inside ``(0, f_s/2)``.
* **heterodyne, IQ receiver** - ``f_shift ≠ 0`` with a 90°-hybrid (coherent)
  front-end giving *complex* samples, used directly: no Hilbert step, no
  ``(0, f_s/2)`` restriction (the line may sit anywhere in ``±f_s/2``), and
  receiver DC / hybrid-image spurs land ``f_shift`` / ``2·f_shift`` away
  from the line instead of on top of it.
* **homodyne, IQ receiver** - ``f_shift = 0``, complex samples.  Real-valued
  homodyne detection (single photodiode, ``cos Δφ`` only) is not invertible
  to phase and is rejected.

All functions take ``(N,)`` or ``(C, N)`` records with time on the last
axis; the ``C`` channels are **independent captures** processed as a batch -
per-channel carrier removal, PSDs, and linewidths, no joint/MIMO processing
- e.g. the two outputs of a polarization-diverse receiver, several lasers,
or repeated captures stacked for a single GPU pass.

The DSH estimators are methods of
:func:`~commkit.analysis.linewidth.estimate_linewidth`:

* ``DshFmPsd`` - deconvolve the beat FM-noise PSD by the interferometer
  response ``4 sin²(πfτ_d)`` to recover the laser FM-noise PSD; works in both
  the coherent (short-delay) and incoherent regimes.
* ``DshIncrement`` - lag-slope of the differential-phase increment variance;
  the AWGN-immune Wiener-linewidth estimate (DSH analogue of
  ``IncrementSlope``).
* ``DshLorentzian`` - classic spectral width of the beat line; valid only in
  the incoherent regime ``τ_d ≫ τ_c = 1/(πΔν)``.

This module holds the front end they share: the forward model ``dsh_beat``
and the phase extraction ``dsh_phase``.
"""

import numpy as np

from .._array import as_2d, restore_1d, to_report_scalar
from ..backend import ArrayType, dispatch
from ..core._signal_adapter import adapt_signal
from ..core.signal import Signal
from ..frequency import correct_frequency_offset
from ..logger import logger
from ..math import _remove_linear_trend

__all__ = ["dsh_beat", "dsh_phase"]


# -----------------------------------------------------------------------------
# dsh_beat  - array-only forward model (synthesizes a beat from a phase
#             trajectory for validation; no Signal to unwrap).
# dsh_phase - Signal-aware: takes the real captured beat record.
# -----------------------------------------------------------------------------


def _analytic_beat(samples):
    """Complex analytic beat ``(C, N)`` from raw DSH samples (Hilbert if real)."""
    z, xp, sp = dispatch(samples)
    z2, was_1d = as_2d(z, name="samples")
    if xp.iscomplexobj(z2):
        return z2.astype(xp.complex128, copy=False), was_1d, xp
    return sp.signal.hilbert(z2.astype(xp.float64), axis=-1), was_1d, xp


def dsh_beat(
    phi: ArrayType,
    *,
    sampling_rate: float,
    delay: float,
    f_shift: float = 0.0,
) -> tuple[ArrayType, ArrayType]:
    r"""Ideal interferometer beat for a laser phase trajectory (forward model).

    Deterministic synthesis counterpart of ``dsh_phase`` - the model that the
    estimators in this module invert.  The laser field ``E(t) = exp(jφ(t))``
    interferes with its own delayed replica; the detector output is the field
    product

        z(t) = E(t) · E*(t - τ_d) · exp(j 2π f_shift t)
             = exp(j (2π f_shift t + Δφ(t))),   Δφ(t) = φ(t) - φ(t - τ_d),

    with ``τ_d`` rounded to the nearest whole sample.  ``z`` is what an IQ
    (90°-hybrid) receiver records at any ``f_shift`` - ``f_shift = 0`` is the
    self-*homodyne* case, ``f_shift ≠ 0`` the AOM self-*heterodyne* case; a
    single-photodetector heterodyne receiver records ``z.real`` instead.

    Deliberately *not* included - chain separately to build a full
    measurement:

    * the phase trajectory itself: ``impairments.generate_phase_noise``;
    * detection noise: ``impairments.apply_awgn`` on the returned beat.

    Parameters
    ----------
    phi : array_like
        Laser phase trajectory in radians, ``(N,)`` or ``(C, N)``
        (``float64`` recommended; see ``generate_phase_noise``).
    sampling_rate : float
        Sampling rate in Hz.
    delay : float
        Interferometer delay ``τ_d`` in seconds (≈ 4.9 µs per km of SMF).
        Rounded to ``m = round(delay · f_s)`` samples; must satisfy
        ``1 ≤ m < N``.
    f_shift : float, default 0.0
        AOM frequency shift in Hz (0 = homodyne).

    Returns
    -------
    z : array_like
        Unit-amplitude complex beat, ``(..., N - m)``, ``complex128``, same
        backend as ``phi``.
    delta_phi : array_like
        The true differential phase Δφ, ``(..., N - m)`` - ground truth for
        validating ``dsh_phase`` and the DSH linewidth estimates.
    """
    x, xp, _ = dispatch(phi)

    m = int(round(delay * sampling_rate))
    n = x.shape[-1]
    if not 1 <= m < n:
        raise ValueError(
            f"delay of {m} samples must be in [1, {n}) for a length-{n} trajectory."
        )
    logger.info(
        "DSH beat: delay %s samples (%.3g µs), f_shift %.4g Hz.",
        m,
        m / sampling_rate * 1e6,
        f_shift,
    )

    delta_phi = x[..., m:] - x[..., :-m]
    beat_phase = delta_phi
    if f_shift != 0.0:
        t = xp.arange(n - m, dtype=xp.float64) / sampling_rate
        beat_phase = delta_phi + 2.0 * np.pi * f_shift * t
    return xp.exp(1j * beat_phase), delta_phi


def dsh_phase(
    samples: ArrayType | Signal,
    *,
    sampling_rate: float | None = None,
    f_shift: float | None = None,
) -> tuple[ArrayType, float | np.ndarray]:
    r"""Unwrapped differential laser phase Δφ(t) = φ(t) - φ(t-τ_d) from the beat.

    Removes the beat carrier (the AOM shift plus any receiver frequency
    offset) and unwraps the remaining angle in ``float64``.  Real inputs are
    made analytic with a Hilbert transform first; complex (IQ) inputs are
    used directly at any carrier - both the self-*homodyne* capture
    (``f_shift = 0`` with a 90° hybrid) and the *heterodyne* IQ capture
    (AOM + coherent receiver) work transparently, free of the Hilbert
    step's band restrictions.

    Parameters
    ----------
    samples : array_like or Signal
        Beat record, ``(N,)`` or ``(C, N)``.  Real (single photodetector,
        heterodyne) or complex (IQ front-end).  A :class:`Signal` supplies
        ``sampling_rate``.
    sampling_rate : float, optional
        Sampling rate in Hz (a fact).  Taken from a Signal; required for array
        input.  A value that disagrees with the Signal raises.
    f_shift : float, optional
        Known beat carrier in Hz (AOM frequency).  If None, the mean beat
        frequency is estimated per channel in two stages - coarse Kay
        (lag-1-autocorrelation) estimate, then exact least-squares slope
        removal on the unwrapped phase - and removed.

    Returns
    -------
    delta_phi : array_like
        Unwrapped differential phase in radians (``float64``), same layout and
        backend as the input.
    f_shift_hz : float or ndarray
        The removed carrier frequency (estimated or as passed), float (SISO)
        or ``(C,)`` array.

    Notes
    -----
    **Limitations.**

    * A perfectly removed carrier is *not* required downstream: a constant
      residual offset drops out of the increment variances and of the
      (detrended) FM-noise PSD.  Carrier removal mainly keeps ``delta_phi``
      flat for inspection/plotting.
    * The estimated carrier is the record's best-fit mean beat frequency - it
      absorbs the mean laser drift over the capture into ``f_shift_hz`` and
      *linearly detrends* ``delta_phi``.  Pass the known AOM frequency to keep
      drift visible in ``delta_phi``.
    * **Real input**: the analytic-signal step requires the whole beat
      lineshape inside ``(0, f_s/2)`` - i.e. ``f_shift`` larger than the beat
      half-bandwidth and below Nyquist by the same margin; spectral folding
      corrupts the phase silently.  Real input with ``f_shift = 0`` is
      rejected (see module docstring).
    * ``unwrap`` needs the per-sample phase step below π: keep the beat SNR
      moderate (≳ 10 dB in the beat bandwidth) and the sampling rate well
      above the beat linewidth; slips appear as ±2π staircase jumps in
      ``delta_phi``.
    * Everything the interferometer adds - fiber acoustic/thermal noise in
      the delay arm, AOM RF-synthesizer phase noise - is indistinguishable
      from laser phase noise here and adds to the low-frequency PSD.
    """
    signal_adapter = adapt_signal(samples, function_name="dsh_phase()")
    samples = signal_adapter.array
    sampling_rate = signal_adapter.resolve_fact("sampling_rate", sampling_rate)

    z, xp, _ = dispatch(samples)
    fs = float(sampling_rate)

    if not xp.iscomplexobj(z) and f_shift is not None and float(f_shift) == 0.0:
        raise ValueError(
            "Real-valued samples with f_shift=0 (self-homodyne on a single "
            "photodetector) observe cos(Δφ) only and cannot be inverted to "
            "phase. Use an AOM shift (heterodyne) or an IQ front-end."
        )
    z2, was_1d, xp = _analytic_beat(samples)

    estimate = f_shift is None
    if f_shift is None:
        # Coarse stage - Kay estimator: phase of the lag-1 autocorrelation,
        # wrap-immune, one reduction per channel.  (This is the M=1
        # "generic blind" special case of
        # ``frequency.MengaliMorelli``, which
        # generalizes it to a multi-lag MVUE combination for lower coarse-
        # stage variance.  Not used here: with the fine LS-slope stage below
        # already fitting the *entire* unwrapped record - the classic
        # optimal two-stage tone-frequency estimator, asymptotically
        # equivalent to the Cramér-Rao bound - the coarse stage only has to
        # be accurate enough that ``unwrap`` does not slip; multi-lag
        # averaging would not improve the final ``f_hat`` and costs two
        # padded FFTs plus a Numba-JIT bootstrap dependency for no payoff.)
        # Under strong phase noise the Kay estimate is biased by the sine
        # nonlinearity (kHz-scale here), which the fine stage corrects.
        acc = xp.sum(z2[:, 1:] * xp.conj(z2[:, :-1]), axis=-1)
        f_hat = xp.angle(acc) * (fs / (2.0 * np.pi))  # (C,)
    else:
        f_hat = xp.full(z2.shape[0], float(f_shift), dtype=xp.float64)

    z_bb = correct_frequency_offset(z2, f_hat, sampling_rate=fs)
    dphi = xp.unwrap(xp.angle(z_bb), axis=-1)

    if estimate:
        # Fine stage: remove the per-channel least-squares slope of the
        # unwrapped phase (the exact linear-ramp / mean-frequency component the
        # Kay stage leaves behind).  Skipped when f_shift is user-supplied so
        # genuine drift stays visible.
        dphi, slope = _remove_linear_trend(dphi)  # slope in rad/sample
        f_hat = f_hat + slope * (fs / (2.0 * np.pi))

    f_used = to_report_scalar(f_hat)
    return restore_1d(was_1d, dphi), f_used
