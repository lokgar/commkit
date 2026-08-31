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

Estimators (see ``linewidth_dsh``):

* ``"fm_psd"`` - deconvolve the beat FM-noise PSD by the interferometer
  response ``4 sin²(πfτ_d)`` to recover the laser FM-noise PSD; works in both
  the coherent (short-delay) and incoherent regimes.
* ``"increment"`` - lag-slope of the differential-phase increment variance;
  the AWGN-immune Wiener-linewidth estimate (DSH analogue of
  ``linewidth_increment``).
* ``"lorentzian"`` - classic spectral width of the beat line; valid only in
  the incoherent regime ``τ_d ≫ τ_c = 1/(πΔν)``.
"""

import numpy as np

from ..backend import ArrayType, dispatch, to_device
from ..frequency import correct_static_frequency_offset
from ..logger import logger
from ..spectral import welch_psd
from ._common import _as_2d, _plateau_mask, _scalar_or_array, _welch_median_bias
from .linewidth import fm_noise_psd

__all__ = ["dsh_beat", "dsh_fm_noise_psd", "dsh_phase", "linewidth_dsh"]


def _analytic_beat(samples):
    """Complex analytic beat ``(C, N)`` from raw DSH samples (Hilbert if real)."""
    z, xp, sp = dispatch(samples)
    z2, was_1d = _as_2d(z)
    if xp.iscomplexobj(z2):
        return z2.astype(xp.complex128, copy=False), was_1d, xp
    return sp.signal.hilbert(z2.astype(xp.float64), axis=-1), was_1d, xp


def dsh_beat(
    phi: ArrayType,
    sampling_rate: float,
    delay: float,
    *,
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
        validating the ``dsh_phase`` / ``linewidth_dsh`` estimates.
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
    samples: ArrayType,
    sampling_rate: float,
    *,
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
    samples : array_like
        Beat record, ``(N,)`` or ``(C, N)``.  Real (single photodetector,
        heterodyne) or complex (IQ front-end).
    sampling_rate : float
        Sampling rate in Hz.
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
    z, xp, sp = dispatch(samples)
    z2, was_1d = _as_2d(z)
    fs = float(sampling_rate)

    if not xp.iscomplexobj(z2):
        if f_shift is not None and float(f_shift) == 0.0:
            raise ValueError(
                "Real-valued samples with f_shift=0 (self-homodyne on a single "
                "photodetector) observe cos(Δφ) only and cannot be inverted to "
                "phase. Use an AOM shift (heterodyne) or an IQ front-end."
            )
        z2 = sp.signal.hilbert(z2.astype(xp.float64), axis=-1)
    else:
        z2 = z2.astype(xp.complex128, copy=False)

    estimate = f_shift is None
    if f_shift is None:
        # Coarse stage - Kay estimator: phase of the lag-1 autocorrelation,
        # wrap-immune, one reduction per channel.  (This is the M=1
        # "generic blind" special case of
        # ``frequency.estimate_frequency_offset_mengali_morelli``, which
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

    n = z2.shape[-1]
    n_idx = xp.arange(n, dtype=xp.float64)
    z_bb = correct_static_frequency_offset(z2, fs, f_hat)
    dphi = xp.unwrap(xp.angle(z_bb), axis=-1)

    if estimate:
        # Fine stage: remove the per-channel least-squares slope of the
        # unwrapped phase (the exact linear-ramp / mean-frequency component the
        # Kay stage leaves behind).  Skipped when f_shift is user-supplied so
        # genuine drift stays visible.
        nc = n_idx - 0.5 * (n - 1)
        slope = (dphi @ nc) / (n * (n * n - 1.0) / 12.0)  # (C,) rad/sample
        dphi = dphi - slope[:, None] * nc[None, :]
        f_hat = f_hat + slope * (fs / (2.0 * np.pi))

    f_used = _scalar_or_array(to_device(f_hat, "cpu"))
    return (dphi[0] if was_1d else dphi), f_used


def dsh_fm_noise_psd(
    delta_phi: ArrayType,
    sampling_rate: float,
    delay: float,
    *,
    nperseg: int | None = None,
    notch_guard: float = 0.1,
    bias_correction: bool = True,
    debug_plot: bool = False,
) -> tuple[ArrayType, ArrayType, ArrayType]:
    r"""Laser FM-noise PSD from the differential phase (notch-guarded deconvolution).

    The interferometer maps the laser phase PSD through
    ``|1 - e^{-j2πfτ_d}|² = 4 sin²(πfτ_d)``, so the *beat* FM-noise PSD relates
    to the *laser* FM-noise PSD as

        S_f,beat(f) = 4 sin²(πfτ_d) · S_f,laser(f).

    This function computes ``S_f,beat`` from ``delta_phi`` (via
    ``fm_noise_psd``) and divides the response back out.  Bins near the
    response notches ``f = k/τ_d`` (including DC) are unrecoverable - they are
    returned as NaN and flagged in ``valid``.

    For ``f ≪ 1/τ_d`` the response reduces to ``(2πfτ_d)²``: the interferometer
    acts as a frequency discriminator with a known gain, which is why the
    method still works when the delay is far *shorter* than the coherence time
    (where the Lorentzian-fit method fails).

    Parameters
    ----------
    delta_phi : array_like
        Unwrapped differential phase from ``dsh_phase`` (radians), ``(N,)`` or
        ``(C, N)``.
    sampling_rate : float
        Sampling rate of ``delta_phi`` in Hz.
    delay : float
        Interferometer differential delay τ_d in seconds (fiber: τ_d ≈ n·L/c ≈
        4.9 µs per km of SMF).
    nperseg : int, optional
        Welch segment length (see ``fm_noise_psd``).
    notch_guard : float, default 0.1
        Bins with ``sin²(πfτ_d)`` below this threshold are masked (NaN).  The
        default keeps ≈ 80 % of every response lobe.
    bias_correction : bool, default True
        Forwarded to ``fm_noise_psd`` (first-difference droop).
    debug_plot : bool, default False
        If True, plot the deconvolved PSD (``frequency_noise_psd``; masked
        bins appear as gaps).

    Returns
    -------
    f : array_like
        One-sided frequency axis in Hz.
    S_f : array_like
        Laser FM-noise PSD in Hz²/Hz (NaN at masked bins), ``(nfreq,)`` or
        ``(C, nfreq)``.
    valid : array_like
        Boolean mask ``(nfreq,)`` of trustworthy bins.

    All three stay on the input backend (no host transfer) - this is the
    composable building block; ``linewidth_dsh`` is the summary layer that
    returns host NumPy for reporting.

    See Also
    --------
    allan_deviation : feed it ``delta_phi/(2π·delay)`` - the interferometer's
        discriminator output - for the laser frequency stability at averaging
        times ``τ ≫ delay`` (shorter τ are low-passed by the τ_d window).

    Notes
    -----
    **Limitations.**

    * Even inside the guard band the deconvolution *amplifies* estimation
      noise by ``1/(4 sin²)`` - near-notch bins are noisier than mid-lobe
      bins.  Prefer median-based summaries (``linewidth_dsh`` does).
    * Additive detector noise on the beat produces an ``f²`` tail in
      ``S_f,beat`` that deconvolution maps into every lobe; restrict analysis
      to the first lobe (``f < 1/τ_d``) unless the beat SNR is very high.
    * The delay must be known accurately: FM-PSD levels scale as ``1/τ_d²``
      below the first notch, so a delay error maps 1:1 (x2) into the
      linewidth.  Calibrate τ_d from the measured notch spacing ``1/τ_d`` if
      in doubt.
    * **Long (decoherence) delays need resolution**: the deconvolution is
      only valid when the Welch bin is much narrower than the response period
      ``1/τ_d`` - i.e. ``nperseg ≳ 8·f_s·τ_d``.  A warning is logged
      otherwise (bins that average across notches bias the PSD low).
    """
    td = float(delay)
    if td <= 0.0:
        raise ValueError(f"delay={delay} must be positive (seconds).")

    f, S_beat = fm_noise_psd(
        delta_phi,
        float(sampling_rate),
        nperseg=nperseg,
        bias_correction=bias_correction,
    )
    _, xp, _ = dispatch(f)

    # The deconvolution samples the response at bin centers; if a Welch bin
    # spans a sizable fraction of the response period 1/τ_d (long decoherence
    # spools), each bin *averages* across lobes and notches and the result is
    # biased low.  Require ≥ 4 bins per period; recommend ≥ 8.
    bin_hz = float(f[1])
    if bin_hz > 0.25 / td:
        rec = 1 << int(np.ceil(np.log2(8.0 * float(sampling_rate) * td)))
        logger.warning(
            "dsh_fm_noise_psd: Welch bin (%.3g Hz) exceeds a quarter of the "
            "interferometer response period 1/τ_d = %.3g Hz - bins average "
            "across notches and the deconvolved PSD is biased low. Increase "
            "nperseg to ≳ %d (≥ 8 bins per period).",
            bin_hz,
            1.0 / td,
            rec,
        )

    s2 = xp.sin(np.pi * f * td) ** 2  # interferometer response / 4
    valid = s2 >= float(notch_guard)
    S_laser = xp.where(valid, S_beat / xp.maximum(4.0 * s2, 1e-300), xp.nan)

    if debug_plot:
        from .. import plotting as _plotting

        _plotting.plot_frequency_noise_psd(
            f, S_laser, show=True, title="DSH laser FM-noise PSD (deconvolved)"
        )

    return f, S_laser, valid


def _lorentzian_widths(f, p, level_lin):
    """Full width of a spectral line ``level_lin`` (linear ratio) below its peak.

    Walks outward from the peak bin to the nearest below-threshold bin on each
    side and interpolates the crossing in log-power.  Returns NaN when the
    threshold is never crossed (e.g. it sits below the noise floor).

    Host-side NumPy by design: the caller hands it an ``nperseg``-sized Welch
    spectrum already brought to the host with a single transfer, and the
    crossing walk is scalar, data-dependent branching that a GPU cannot help
    with.
    """
    i_pk = int(np.argmax(p))
    thr = p[i_pk] / level_lin

    left = np.nonzero(p[:i_pk] < thr)[0]
    right = np.nonzero(p[i_pk + 1 :] < thr)[0]
    if left.size == 0 or right.size == 0:
        return np.nan

    i0 = left[-1]  # crossing between i0 and i0+1
    i1 = i_pk + 1 + right[0]  # crossing between i1-1 and i1
    l0, l1 = np.log(p[i0]), np.log(p[i0 + 1])
    f_lo = f[i0] + (f[i0 + 1] - f[i0]) * (np.log(thr) - l0) / (l1 - l0)
    r0, r1 = np.log(p[i1 - 1]), np.log(p[i1])
    f_hi = f[i1 - 1] + (f[i1] - f[i1 - 1]) * (np.log(thr) - r0) / (r1 - r0)
    return float(f_hi - f_lo)


def linewidth_dsh(
    samples: ArrayType,
    sampling_rate: float,
    delay: float,
    *,
    f_shift: float | None = None,
    method: str = "fm_psd",
    lags: tuple[int, ...] | None = None,
    nperseg: int | None = None,
    f_min: float | None = None,
    f_max: float | None = None,
    notch_guard: float = 0.1,
    level_db: float = 20.0,
    debug_plot: bool = False,
) -> dict[str, object]:
    r"""Laser linewidth from a delayed self-heterodyne / self-homodyne beat.

    Three estimators with complementary validity regions (``τ_c = 1/(πΔν)`` is
    the laser coherence time):

    * ``method="fm_psd"`` (default) - ``dsh_phase`` -> ``dsh_fm_noise_psd`` ->
      white-FM floor ``Δν = π · median(S_f,laser)`` over the valid band.
      Works for **any** delay (coherent or incoherent regime); the notch
      structure, not the regime, sets the usable band.
    * ``method="increment"`` - variance of the lag-``ℓ`` increments of the
      *measured* differential phase.  For lag ``a = ℓ/f_s ≤ τ_d`` the two
      Wiener increments are disjoint, so

          Var[Δφ(t) - Δφ(t-a)] = 4π·Δν·a + 2σ_w²,

      and a straight-line fit vs ``a`` gives ``Δν = slope/(4π)`` with the beat
      angle noise cancelling into the intercept - no SNR estimate needed
      (same trick as ``linewidth_increment``).  White-FM (Wiener) assumption.
    * ``method="lorentzian"`` - the textbook approach: in the incoherent
      regime (``τ_d ≫ τ_c``) the beat line is Lorentzian with FWHM ``2Δν``.
      The width is measured ``level_db`` below the peak and converted via
      ``Δν = W_L / (2·√(10^{L/10} - 1))`` (-> ``W₂₀/(2√99)`` for the customary
      -20 dB width, which suppresses the Gaussian 1/f-noise core that
      contaminates the -3 dB width).

    Parameters
    ----------
    samples : array_like
        Beat record, ``(N,)`` or ``(C, N)``, real (heterodyne photocurrent) or
        complex (IQ).
    sampling_rate : float
        Sampling rate in Hz.
    delay : float
        Interferometer differential delay τ_d in seconds.
    f_shift : float, optional
        Known AOM/beat carrier in Hz; estimated if None (see ``dsh_phase``).
        Unused by ``method="lorentzian"``.
    method : {"fm_psd", "increment", "lorentzian"}, default "fm_psd"
        Estimator, as above.
    lags : tuple of int, optional
        Increment lags in *samples* for ``method="increment"``.  Default:
        five lags up to ``a_max = min(0.5·τ_d·f_s, N/2000)`` - inside the
        disjoint window ``ℓ ≤ τ_d·f_s`` *and* small enough that the variance
        estimator keeps many independent averages when the delay is long
        (decoherence spools).
    nperseg : int, optional
        Welch segment length for the PSD-based methods.
    f_min, f_max : float, optional
        Manual analysis fence for ``method="fm_psd"``.  When **both** are
        None (default) the plateau is **auto-detected**: the white-FM floor
        is located as the minimum of octave-band medians over all valid bins
        and the median runs over every bin within 3x that floor - spanning
        as many interferometer lobes as the detection-noise knee allows, and
        automatically excluding a rising low-frequency (drift/flicker)
        region.  Passing either bound switches to the literal fence
        (``f_min`` -> 0, ``f_max`` -> first notch ``1/τ_d`` when the other is
        omitted), with the median over *all* valid bins inside.
    notch_guard : float, default 0.1
        Notch mask threshold for ``method="fm_psd"`` (``dsh_fm_noise_psd``).
    level_db : float, default 20.0
        Depth below the peak at which the ``lorentzian`` width is measured.
    debug_plot : bool, default False
        Per-method diagnostic plot: the deconvolved FM-noise PSD with the
        fitted white-FM floor (``fm_psd``), the increment-variance fit
        (``increment``), or the beat spectrum with the measured width
        contours (``lorentzian``).

    Returns
    -------
    dict
        Always ``{'linewidth', 'method', ...}`` with linewidths as floats
        (SISO) or ``(C,)`` arrays.  Extra keys per method:

        * ``fm_psd`` - ``f``, ``S_f`` (NaN-masked laser FM PSD), ``valid``,
          ``f_shift``, ``used`` (boolean mask of the bins under the accepted
          plateau region; in auto mode the level is the median of
          *per-log-cell medians* of these bins - one vote per cell, not per
          bin - while a manual fence medians the raw bins directly),
          ``band`` (frequency extent of ``used``), ``n_segments`` (Welch
          segment count ``K`` behind every PSD bin).
        * ``increment`` - ``awgn_var`` (fitted intercept; equals the beat
          angle-noise ``2σ_w²`` only when read with *small* lags - at the
          large default lags the phase-noise term dominates every point and
          the intercept is a noisy extrapolation), ``dphi_var`` (total
          ``Var[Δφ] ≈ 2πΔν·τ_d + σ_w²``), ``lags``, ``f_shift``.
        * ``lorentzian`` - ``linewidth_3db`` (half-power width / 2, the
          *effective* linewidth incl. 1/f broadening), ``lineshape_ratio``
          (``W₂₀/W₃``: ≈ 9.95 pure Lorentzian, ≈ 2.6 pure Gaussian),
          ``coherence_factor`` (``τ_d/τ_c = π·Δν·τ_d``), ``f``, ``psd``,
          ``f_peak``.

        As a *summary* function the dict holds host NumPy only - floats plus
        plot-sized spectra (≤ ``nperseg`` bins, one device->host transfer).
        To keep sample-rate results on the input backend, use ``dsh_phase``
        and ``dsh_fm_noise_psd`` directly.

    Notes
    -----
    **90°-hybrid (IQ) capture - which method?**  The detection scheme, the
    AOM shift, and the delay regime are independent choices: a 90° hybrid
    (phase-diversity receiver) delivers the complex beat directly - with an
    AOM (heterodyne IQ, line at ``f_shift``) or without one (homodyne, line
    at 0 Hz; the AOM becomes optional) - and pairs equally well with a short
    delay or a long *decoherence* spool (``τ_d ≫ τ_c``).  Pick the estimator
    by the coherence factor alone - incoherent regime: all three methods
    apply; coherent regime: ``fm_psd`` / ``increment``.  Two hybrid-specific
    caveats, at their worst in the *homodyne* case where both spurs sit
    exactly at the line center: photodiode DC offsets - calibrate them out
    (arms blocked) *especially* before ``lorentzian``, where a DC spur
    mid-line hijacks the peak and narrows the measured widths; and hybrid
    amplitude/phase imbalance, which creates a conjugate image of the beat -
    correct it first (``impairments.compensate_iq_imbalance_gram_schmidt`` /
    ``..._lowdin``).  An AOM shift moves the line ``f_shift`` away from the
    DC spur and ``2·f_shift`` away from its own image, so heterodyne IQ
    captures are considerably more forgiving of both.

    **Long-term stability**: for an Allan-deviation view, feed the
    discriminator output ``delta_phi/(2π·delay)`` (from ``dsh_phase``) into
    ``allan_deviation`` - valid for averaging times ``τ ≫ delay``.

    **Physical limitations (all methods).**

    * The measurement floor is set by the interferometer, not the laser:
      fiber acoustic/thermal noise (delay arm as a microphone), AOM driver
      phase noise, and polarization fading (use a Faraday mirror or scrambler)
      all add to the apparent laser noise.
    * τ_d must be known (fiber: ``τ_d ≈ 4.9 µs/km``); errors propagate
      linearly (``increment``) or quadratically via the discriminator gain
      (``fm_psd`` below the first notch).

    **Method-specific.**

    * ``lorentzian`` requires ``coherence_factor ≳ 6`` (rule of thumb): below
      that the spectrum develops a coherent carrier spike plus fringes at
      ``1/τ_d`` spacing and the width no longer reads 2Δν - a warning is
      logged.  It also needs the -``level_db`` contour above the noise floor
      (peak dynamic range ≳ ``level_db`` + 10 dB) and ``≳ 10`` Welch bins
      across the line.  1/f noise makes the -3 dB width grow with observation
      time (Gaussian core); the deep-width estimate is the intrinsic
      (Lorentzian/white-FM) linewidth.
    * ``increment`` assumes white FM; flicker bends ``Var(a)`` super-linear
      and biases Δν high (see ``linewidth_increment`` notes).  Slow drift is
      harmless up to linear frequency chirp (constant term after
      differencing), quadratic beyond.
    * ``fm_psd`` reports the white-FM *floor*.  The default auto-detection
      finds the plateau as the PSD's minimum-level region, so a 1/f rise at
      low f and the detection-noise f² tail at high f are excluded without
      manual fencing; it falls back to the first lobe (with a warning) when
      no plateau is found.  For **real** captures the eligible band is
      additionally capped at the receiver's FM detection bandwidth
      ``min(f_shift, f_s/2 - f_shift)`` - beyond it the beat carries no
      sidebands and the deconvolved PSD reads fake-low.  If the *entire*
      usable band is 1/f-dominated (very long τ_d, quiet laser), the
      reported floor is the lowest resolved noise level - inspect the PSD
      before quoting it as Δν.  Welch bins are χ²-distributed, so a raw
      median floor reads ``≈ 1 - 1/(3K)`` *below* the true level for ``K``
      averaged segments; the reported linewidth divides that analytic
      factor back out (relevant for the long ``nperseg`` a dense notch
      comb demands - keep the record ≳ 5·nperseg so ``K ≳ 10``).  The
      plotted log-binned median curve is *not* corrected, so at small
      ``K`` the ``Δν/π`` floor guide sits visibly above it.
    """
    fs = float(sampling_rate)
    td = float(delay)
    if td <= 0.0:
        raise ValueError(f"delay={delay} must be positive (seconds).")
    m_samp = td * fs
    if m_samp < 1.0:
        raise ValueError(
            f"delay·sampling_rate = {m_samp:.3g} < 1 sample - the differential "
            "delay is unresolvable at this sampling rate."
        )

    if method == "fm_psd":
        dphi, f_hat = dsh_phase(samples, fs, f_shift=f_shift)
        f, S_l, valid = dsh_fm_noise_psd(
            dphi, fs, td, nperseg=nperseg, notch_guard=notch_guard
        )
        # Summary layer: one transfer, host-side plateau search + median
        # (plot-sized spectra - see the package backend policy).
        f_cpu = np.asarray(to_device(f, "cpu"), dtype=np.float64)
        S_cpu = np.asarray(to_device(S_l, "cpu"), dtype=np.float64)
        valid_cpu = np.asarray(to_device(valid, "cpu"), dtype=bool)
        S2 = S_cpu[None, :] if S_cpu.ndim == 1 else S_cpu
        n_ch = S2.shape[0]

        # Welch bins are χ²-distributed, so every median-based floor below
        # reads the true level low by median(χ²_ν)/ν ≈ 1 - 1/(3K); the factor
        # is divided back out of the linewidth at the end.
        nps_used = int(round(fs / float(f_cpu[1])))
        m_med, k_seg = _welch_median_bias(dphi.shape[-1] - 1, nps_used)
        if k_seg < 4:
            logger.warning(
                "linewidth_dsh(fm_psd): only %d Welch segment(s) at "
                "nperseg=%d - the plateau median is noisy and the χ²-median "
                "correction (÷%.3f) is asymptotic; extend the record (aim "
                "for ≥ 5·nperseg samples).",
                k_seg,
                nps_used,
                m_med,
            )
        else:
            logger.info(
                "linewidth_dsh(fm_psd): %d Welch segments - χ²-median floor "
                "bias corrected by ÷%.4f.",
                k_seg,
                m_med,
            )

        auto_band = f_min is None and f_max is None
        if auto_band:
            base = valid_cpu & (f_cpu > 0)
            # Real (single-photodiode) captures carry FM sidebands only out
            # to the beat carrier's distance from the band edges: offsets
            # beyond min(f_shift, f_nyq - f_shift) have no physical support
            # after the analytic-signal step and read as fake-low PSD - cap
            # the eligible band there.  Complex IQ captures have no such
            # limit (full ±Nyquist).
            x_in, xp_in, _ = dispatch(samples)
            if not xp_in.iscomplexobj(x_in):
                fh = float(np.min(np.asarray(f_hat)))
                f_cap = min(fh, fs / 2.0 - fh)
                base = base & (f_cpu <= f_cap)
            base2 = np.broadcast_to(base, S2.shape)
            used2, levels = _plateau_mask(f_cpu, S2, base2)
            lw = np.pi * levels
            # Channels where no plateau was found fall back to the first lobe.
            for c in range(n_ch):
                if not np.isfinite(lw[c]):
                    logger.warning(
                        "linewidth_dsh(fm_psd): no plateau detected "
                        "(channel %d) - falling back to the first-lobe band "
                        "[0, 1/τ_d]. Inspect the PSD before quoting Δν.",
                        c,
                    )
                    used2[c] = valid_cpu & (f_cpu > 0) & (f_cpu <= 1.0 / td)
                    if used2[c].any():
                        lw[c] = np.pi * np.median(S2[c][used2[c]])
        else:
            fmin = 0.0 if f_min is None else float(f_min)
            fmax = (1.0 / td) if f_max is None else float(f_max)
            used2 = np.broadcast_to(
                valid_cpu & (f_cpu >= fmin) & (f_cpu <= fmax), S2.shape
            )
            if used2.any(axis=-1).all():
                lw = np.array([np.pi * np.median(S2[c][used2[c]]) for c in range(n_ch)])

        if not used2.any(axis=-1).all():
            raise ValueError(
                "No valid FM-PSD bins in the analysis band - widen "
                "[f_min, f_max], lower notch_guard, or increase nperseg."
            )
        lw = lw / m_med
        f_used = f_cpu[used2.any(axis=0)]
        band_used = (float(f_used[0]), float(f_used[-1]))

        used_cpu = used2[0] if S_cpu.ndim == 1 else used2
        result = {
            "linewidth": _scalar_or_array(lw),
            "f": f_cpu,
            "S_f": S_cpu,
            "valid": valid_cpu,
            "used": used_cpu,
            "band": band_used,
            "n_segments": k_seg,
            "f_shift": f_hat,
            "method": method,
        }
        if debug_plot:
            from .. import plotting as _plotting

            _plotting.plot_frequency_noise_psd(
                f_cpu,
                S_cpu,
                floor=result["linewidth"],
                band=band_used,
                used=used_cpu,
                show=True,
                title="DSH laser FM-noise PSD (deconvolved)",
            )
        return result

    if method == "increment":
        dphi, f_hat = dsh_phase(samples, fs, f_shift=f_shift)
        d2, _ = _as_2d(dphi)
        _, xp, _ = dispatch(d2)

        m_int = int(round(m_samp))
        if lags is None:
            # Largest default lag: half the delay (disjoint-increment bound),
            # further capped by the record length - the Var estimator's
            # correlation support scales with the lag, so for long decoherence
            # spools small lags give far more independent averages (measured:
            # ~3x lower spread at τ_d·f_s = 6·10⁴, N = 2·10⁶) while the AWGN
            # intercept still cancels in the fit.
            a_max = max(1.0, min(0.5 * m_samp, d2.shape[-1] / 2000.0))
            ls = np.unique(
                np.maximum(
                    1,
                    np.round(a_max * np.array([0.2, 0.4, 0.6, 0.8, 1.0])).astype(int),
                )
            )
        else:
            ls = np.unique(np.asarray([int(lag) for lag in lags]))
            if ls.size and ls[0] < 1:
                raise ValueError("lags must be positive sample counts.")
        if ls.size < 2:
            raise ValueError(
                f"method='increment' needs ≥ 2 distinct lags (delay spans only "
                f"{m_int} samples - increase the sampling rate or pass lags)."
            )
        if int(ls[-1]) > m_int:
            logger.warning(
                "linewidth_dsh: max lag %d exceeds the delay (%d samples); the "
                "Wiener increments overlap and Var(a) is no longer linear - "
                "the fit will be biased low.",
                int(ls[-1]),
                m_int,
            )

        var_l = xp.stack(
            [xp.var(d2[:, lag:] - d2[:, :-lag], axis=-1) for lag in ls], axis=0
        )  # (n_lag, C)
        dphi_var = xp.var(d2, axis=-1)
        # Tiny (n_lag, C) fit input - one D2H transfer + host polyfit.
        var_cpu = np.asarray(to_device(var_l, "cpu"), dtype=np.float64)
        a_sec = ls.astype(np.float64) / fs
        coeffs = np.polyfit(a_sec, var_cpu, 1)  # (2, C): [slope, intercept]
        slope, intercept = coeffs[0], coeffs[1]
        lw_cpu = np.maximum(slope, 0.0) / (4.0 * np.pi)

        if debug_plot:
            from .. import plotting as _plotting

            _plotting.plot_increment_variance(
                a_sec,
                var_cpu.T,
                slope=slope,
                intercept=intercept,
                show=True,
                title="DSH differential-phase increment variance",
            )

        return {
            "linewidth": _scalar_or_array(lw_cpu),
            "awgn_var": _scalar_or_array(intercept),
            "dphi_var": _scalar_or_array(to_device(dphi_var, "cpu")),
            "lags": ls,
            "f_shift": f_hat,
            "method": method,
        }

    if method == "lorentzian":
        z2, was_1d, xp = _analytic_beat(samples)
        n = z2.shape[-1]
        npseg = int(min(max(n // 8, 256), 1 << 14)) if nperseg is None else nperseg
        npseg = min(npseg, n)
        f, P = welch_psd(
            z2, sampling_rate=fs, nperseg=npseg, return_onesided=False, axis=-1
        )
        # From here on the work is scalar peak/width searching on an
        # nperseg-sized spectrum - host-side NumPy on purpose.
        f_cpu = np.asarray(to_device(f, "cpu"), dtype=np.float64)
        P_cpu = np.asarray(to_device(P, "cpu"), dtype=np.float64)
        P2 = P_cpu[None, :] if P_cpu.ndim == 1 else P_cpu
        c = P2.shape[0]

        from scipy.ndimage import uniform_filter1d

        r_deep = 10.0 ** (float(level_db) / 10.0)
        dnu_deep = np.full(c, np.nan)
        dnu_3db = np.full(c, np.nan)
        ratio = np.full(c, np.nan)
        f_peak = np.full(c, np.nan)
        bin_hz = f_cpu[1] - f_cpu[0]
        for ch in range(c):
            p = P2[ch]
            # Pass 1 - rough half-power width on the raw spectrum, only to
            # size the smoothing window.  The raw argmax bin rides the upward
            # Welch fluctuations (max over many ±1/√K bins), which biases the
            # peak high and every width low; smoothing over ≈ FWHM/5 removes
            # that bias at < 3 % lineshape droop.
            w3_rough = _lorentzian_widths(f_cpu, p, 2.0)
            if np.isfinite(w3_rough):
                w_bins = min(int(w3_rough / (5.0 * bin_hz)) | 1, 101)
                if w_bins >= 3:
                    p = uniform_filter1d(p, w_bins, mode="nearest")

            i_pk = int(np.argmax(p))
            f_peak[ch] = f_cpu[i_pk]
            dyn_db = 10.0 * np.log10(p[i_pk] / np.median(p))
            if dyn_db < float(level_db) + 10.0:
                logger.warning(
                    "linewidth_dsh: beat peak only %.1f dB above the PSD floor "
                    "(channel %d) - the -%g dB contour is noise-limited; "
                    "increase averaging or lower level_db.",
                    dyn_db,
                    ch,
                    float(level_db),
                )
            w3 = _lorentzian_widths(f_cpu, p, 2.0)  # half-power width
            w_deep = _lorentzian_widths(f_cpu, p, r_deep)
            if np.isfinite(w3) and w3 < 6.0 * bin_hz:
                logger.warning(
                    "linewidth_dsh: half-power width spans < 6 Welch bins "
                    "(channel %d) - increase nperseg for a resolved line.",
                    ch,
                )
            dnu_3db[ch] = w3 / 2.0
            dnu_deep[ch] = w_deep / (2.0 * np.sqrt(r_deep - 1.0))
            if np.isfinite(w3) and np.isfinite(w_deep) and w3 > 0.0:
                ratio[ch] = w_deep / w3

        coh = np.pi * dnu_deep * td  # τ_d / τ_c
        if np.any(np.isfinite(coh) & (coh < 6.0)):
            logger.warning(
                "linewidth_dsh: τ_d/τ_c = %s < 6 - coherent-regime fringes; the "
                "Lorentzian width is unreliable. Use method='fm_psd' or "
                "'increment'.",
                np.array2string(coh, precision=2),
            )

        if debug_plot:
            from .. import plotting as _plotting

            _plotting.plot_dsh_beat_psd(
                f_cpu,
                P2[0] if was_1d else P2,
                f_peak=f_peak,
                linewidth=dnu_deep,
                linewidth_3db=dnu_3db,
                level_db=level_db,
                show=True,
            )

        return {
            "linewidth": _scalar_or_array(dnu_deep),
            "linewidth_3db": _scalar_or_array(dnu_3db),
            "lineshape_ratio": _scalar_or_array(ratio),
            "coherence_factor": _scalar_or_array(coh),
            "f": f_cpu,
            "psd": P2[0] if was_1d else P2,
            "f_peak": _scalar_or_array(f_peak),
            "method": method,
        }

    raise ValueError(
        f"Unknown method {method!r} (use 'fm_psd', 'increment' or 'lorentzian')."
    )
