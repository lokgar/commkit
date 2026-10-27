r"""Delayed self-heterodyne / self-homodyne (DSH) laser characterization.

A DSH measurement gives a CW laser's FM-noise PSD and linewidth from the
digitized beat of a delayed self-interference: the laser under test is split in two,
one arm is delayed by ``τ_d`` (a fiber spool), the other is optionally
frequency-shifted by ``f_shift`` (an AOM), and the arms are recombined on a
photodetector.  The beat at ``f_shift`` carries the differential laser
phase

    Δφ(t) = φ(t) - φ(t - τ_d),

so the interferometer turns the laser's absolute phase noise, which cannot
be observed without a second, better laser, into a measurable quantity: the
laser itself, delayed towards or beyond its own coherence time, is the
reference.

The AOM shift and the receiver are independent choices, and ``dsh_phase``
dispatches on the dtype of the input rather than on a named variant, so
three combinations are supported:

* heterodyne with a single photodetector: ``f_shift ≠ 0`` and real samples;
  the analytic signal is formed internally (Hilbert transform), so the
  whole beat lineshape must sit inside ``(0, f_s/2)``.
* heterodyne with an IQ receiver: ``f_shift ≠ 0`` with a 90°-hybrid
  (coherent) front end whose complex samples are used directly.  There is
  no Hilbert step and no ``(0, f_s/2)`` restriction (the line may sit
  anywhere in ``±f_s/2``), and the receiver's DC and hybrid-image spurs land
  ``f_shift`` and ``2·f_shift`` away from the line instead of on it.
* homodyne with an IQ receiver: ``f_shift = 0`` and complex samples.  Real
  homodyne detection (a single photodiode, ``cos Δφ`` only) cannot be
  inverted to a phase and is rejected.

All functions take ``(N,)`` or ``(C, N)`` records with time on the last
axis.  By default the ``C`` channels are independent captures processed as a
batch (carrier removal, PSDs and linewidths per channel), e.g. several
lasers, or repeated captures stacked for one GPU pass.  With
``combine_polarizations=True`` the channels are instead the polarization
outputs of one beat and are combined into a single record (see below).

Polarization diversity.  The two interferometer arms reach the receiver
in different, slowly drifting states of polarization (SOP): the delay spool's
birefringence wanders with temperature and vibration.  Only the projection
of one arm's field onto the other's polarization beats, so a
single-polarization receiver sees the beat amplitude fade (completely when the two
SOPs become orthogonal), and during a fade the phase is pure noise and
unwraps with slips.  A polarization-diverse (dual-pol) coherent receiver
records the beat in two orthogonal polarizations,

    z_p(t) = a_p(t)·s(t) + n_p(t),     p = x, y,

where ``s = exp(j(2π f_shift t + Δφ))`` is the beat both channels share,
``a = (a_x, a_y)`` the slowly varying Jones vector it arrives in
(``|a_x|² + |a_y|²`` is the total beat power, which does not fade), and
``n_p`` independent receiver noise.  The optimal combination for the phase
of ``s`` is maximum-ratio combining, ``y = aᴴz / ‖a‖ = ‖a‖·s + noise``:
each channel weighted by its own amplitude and co-phased.  ``a`` is
estimated from the coherency matrix ``R = ⟨z zᴴ⟩ = a aᴴ + σ² I``, averaged
over short blocks (≈ 1 µs, at least 256 samples): its dominant eigenvector
is ``a/‖a‖`` up to a phase.  That phase is arbitrary in each block, so the
eigenvectors are parallel-transported (each block's phase is chosen to match
the previous block) and interpolated to the sample rate before
``y = vᴴz``.  The combined record has the SNR of the total beat power and no
fading.

Limitations: the channels must be sampled synchronously (deskew them
first); SOP changes faster than a few blocks are not tracked; and a SOP that
traces a closed loop on the Poincaré sphere adds the geometric
(Pancharatnam) phase, half the enclosed solid angle, to ``y``: a slow phase
at the SOP's time scale, far below the frequencies the linewidth estimators
use.

The DSH estimators are methods of
:func:`~commkit.analysis.linewidth.estimate_linewidth`:

* ``DshFmPsd``: fits white FM, flicker FM and detection noise to the beat's
  FM-noise PSD, through the interferometer response as a Welch estimate sees
  it; it works in the coherent (short-delay) and the incoherent regime and,
  with its default segments, needs the delay only roughly.
* ``DshIncrement``: the lag slope of the variance of the differential-phase
  increments, a Wiener-linewidth estimate unaffected by AWGN (the DSH
  analogue of ``IncrementSlope``).
* ``DshLorentzian``: the usual spectral width of the beat line, valid only
  in the incoherent regime ``τ_d ≫ τ_c = 1/(πΔν)``.

This module holds the front end they share: the forward model
:func:`dsh_beat` and the phase extraction :func:`dsh_phase`.
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


def _analytic_beat(
    samples, combine_polarizations: bool = False, sampling_rate: float = 0.0
):
    """Complex analytic beat ``(C, N)`` from raw DSH samples (Hilbert if real).

    With ``combine_polarizations`` the channels are combined into one,
    ``(1, N)``, and reported as 1-D; ``sampling_rate`` sets the combiner's
    block length.
    """
    z, xp, sp = dispatch(samples)
    z2, was_1d = as_2d(z, name="samples")
    if xp.iscomplexobj(z2):
        z2 = z2.astype(xp.complex128, copy=False)
    else:
        # A photodiode's DC term is not part of the beat, which lies inside
        # (0, f_s/2); left in, it adds a constant phasor to the analytic
        # signal and corrupts the beat phase.
        z2 = z2.astype(xp.float64)
        z2 = sp.signal.hilbert(z2 - xp.mean(z2, axis=-1, keepdims=True), axis=-1)
    if combine_polarizations:
        if z2.shape[0] < 2:
            raise ValueError(
                "combine_polarizations=True needs the polarization channels as "
                f"a (C, N) record with C >= 2, got {z2.shape[0]} channel(s)."
            )
        return _combine_polarizations(z2, sampling_rate, xp), True, xp
    return z2, was_1d, xp


def _combine_polarizations(z2, sampling_rate: float, xp):
    """Maximum-ratio combination of the polarization channels of one beat.

    ``(C, N)`` complex in, ``(1, N)`` out: block coherency matrices, their
    dominant eigenvectors (the Jones vector up to a phase), parallel
    transport of that phase, linear interpolation to the sample rate and
    ``y = vᴴz``.  See the module docstring.
    """
    c, n = z2.shape
    block = min(max(256, int(round(1e-6 * sampling_rate))), n)
    g = max(n // block, 1)
    zb = z2[:, : g * block].reshape(c, g, block)
    coherency = xp.einsum("pgk,qgk->gpq", zb, xp.conj(zb))  # (G, C, C)
    # Average a few neighboring blocks: noise in R averages down while the
    # SOP is still tracked block by block.
    kernel = 5 if g >= 5 else 1
    if kernel > 1:
        pad = xp.concatenate(
            [coherency[:1].repeat(2, 0), coherency, coherency[-1:].repeat(2, 0)], 0
        )
        coherency = sum(pad[i : i + g] for i in range(kernel)) / kernel
    _, vecs = xp.linalg.eigh(coherency)
    v = vecs[:, :, -1]  # (G, C): dominant eigenvector per block
    # Parallel transport: rotate each block's vector into phase with the
    # previous one, so the weights (and the combined phase) stay continuous.
    step = xp.angle(xp.sum(xp.conj(v[:-1]) * v[1:], axis=-1))
    turn = xp.concatenate([xp.zeros(1), xp.cumsum(step)])
    v = v * xp.exp(-1j * turn)[:, None]

    centers = (xp.arange(g, dtype=xp.float64) + 0.5) * block
    t = xp.arange(n, dtype=xp.float64)
    y = xp.zeros(n, dtype=xp.complex128)
    for p in range(c):
        w = xp.interp(t, centers, v[:, p].real) + 1j * xp.interp(
            t, centers, v[:, p].imag
        )
        y += xp.conj(w) * z2[p]
    return y[None, :]


def dsh_beat(
    phi: ArrayType,
    *,
    sampling_rate: float,
    delay: float,
    f_shift: float = 0.0,
) -> tuple[ArrayType, ArrayType]:
    r"""Ideal interferometer beat of a laser phase trajectory (the forward model).

    The deterministic counterpart of :func:`dsh_phase`: the model that the
    estimators of this module invert.  The laser field ``E(t) = exp(jφ(t))``
    interferes with its own delayed replica; the detector output is the field
    product

        z(t) = E(t) · E*(t - τ_d) · exp(j 2π f_shift t)
             = exp(j (2π f_shift t + Δφ(t))),   Δφ(t) = φ(t) - φ(t - τ_d),

    with ``τ_d`` rounded to a whole number of samples.  ``z`` is what an IQ
    (90°-hybrid) receiver records at any ``f_shift``: ``f_shift = 0`` is the
    self-homodyne case, ``f_shift ≠ 0`` the self-heterodyne case with an AOM;
    a single-photodetector heterodyne receiver records ``z.real`` instead.

    Two parts of a measurement are left out on purpose, to be chained
    separately:

    * the phase trajectory itself: :func:`commkit.impairments.generate_phase_noise`;
    * detection noise: :func:`commkit.impairments.apply_awgn` on the
      returned beat.

    Parameters
    ----------
    phi : array_like
        Laser phase trajectory in radians, ``(N,)`` or ``(C, N)``, preferably
        float64 (as :func:`~commkit.impairments.generate_phase_noise` returns).
    sampling_rate : float
        Sampling rate in Hz.
    delay : float
        Interferometer delay ``τ_d`` in seconds (about 4.9 µs per km of
        standard fiber), rounded to ``m = round(delay · f_s)`` samples with
        ``1 ≤ m < N``.
    f_shift : float, default 0.0
        AOM frequency shift in Hz; 0 for homodyne.

    Returns
    -------
    z : ndarray
        Unit-amplitude complex beat, ``(..., N - m)``, ``complex128``, on the
        device of ``phi``.
    delta_phi : ndarray
        The true differential phase ``Δφ``, ``(..., N - m)``: the ground
        truth for validating :func:`dsh_phase` and the DSH linewidth
        estimates.
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
    combine_polarizations: bool = False,
) -> tuple[ArrayType, float | np.ndarray]:
    r"""Unwrapped differential laser phase ``Δφ(t) = φ(t) - φ(t - τ_d)`` of a beat.

    Removes the beat carrier (the AOM shift plus any receiver frequency
    offset) and unwraps the remaining angle in float64.  Real input is made
    analytic with a Hilbert transform first; complex (IQ) input is used
    directly at any carrier, so both the self-homodyne capture
    (``f_shift = 0`` with a 90° hybrid) and the heterodyne IQ capture (AOM
    and coherent receiver) work without the band restrictions of the Hilbert
    step.

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
        Known beat carrier in Hz (the AOM frequency).  ``None`` estimates the
        mean beat frequency of each channel and removes it, in two stages: a
        coarse Kay estimate (from the lag-1 autocorrelation), then the exact
        least-squares slope of the unwrapped phase.
    combine_polarizations : bool, default False
        Treat the ``C ≥ 2`` channels as the polarization outputs of one beat
        and combine them (maximum-ratio, see the module docstring) into a
        single differential phase that does not fade.

    Returns
    -------
    delta_phi : ndarray
        Unwrapped differential phase in radians (float64), same layout and
        device as the input (``(N,)`` when the polarizations are combined).
    f_shift_hz : float or ndarray
        The removed carrier frequency (estimated or as passed): a float
        (SISO) or a ``(C,)`` array.

    Notes
    -----
    **Limitations.**

    * The carrier need not be removed perfectly: a constant residual offset
      drops out of the increment variances and of the (detrended) FM-noise
      PSD.  Removing it mainly keeps ``delta_phi`` flat for inspection and
      plots.
    * The estimated carrier is the record's best-fit mean beat frequency: it
      absorbs the mean laser drift over the capture into ``f_shift_hz`` and
      removes the linear trend of ``delta_phi``.  Pass the known AOM
      frequency to keep the drift in ``delta_phi``.
    * Real input: the analytic-signal step needs the whole beat lineshape
      inside ``(0, f_s/2)``, that is ``f_shift`` larger than the beat's
      half-bandwidth and below Nyquist by the same margin; spectral folding
      corrupts the phase without a warning.  Real input with
      ``f_shift = 0`` is rejected (see the module docstring).
    * ``unwrap`` needs the per-sample phase step below π: keep the beat SNR
      moderate (≳ 10 dB in the beat bandwidth) and the sampling rate well
      above the beat linewidth; slips appear as ±2π staircase jumps in
      ``delta_phi``.
    * Everything the interferometer adds (acoustic and thermal noise of the
      delay fiber, phase noise of the AOM's RF synthesizer) cannot be told
      apart from laser phase noise here and adds to the low-frequency PSD.
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
    z2, was_1d, xp = _analytic_beat(samples, combine_polarizations, fs)

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

    f_used = to_report_scalar(f_hat, was_1d=was_1d)
    return restore_1d(was_1d, dphi), f_used
