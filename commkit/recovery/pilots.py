"""Pilot-symbol and pilot-tone aided carrier phase recovery."""

import logging

import numpy as np

from ..backend import ArrayType, dispatch, to_device
from ..logger import logger
from .corrections import correct_cycle_slips


def recover_carrier_phase_pilot_symbols(
    symbols: ArrayType,
    pilot_indices: ArrayType,
    pilot_values: ArrayType,
    interpolation: str = "linear",
    joint_channels: bool = False,
    cycle_slip_correction: bool = False,
    cycle_slip_history: int = 100,
    cycle_slip_threshold: float = np.pi / 4,
    debug_plot: bool = False,
) -> ArrayType:
    """
    Carrier phase recovery using known pilot symbols.

    Computes the phase error at each pilot position, unwraps the pilot
    phase sequence, and interpolates across the full symbol grid.

    Parameters
    ----------
    symbols : array_like
        Received 1-SPS complex symbols. Shape: (N,) or (C, N).
    pilot_indices : array_like of int
        Indices of pilot symbols within the frame, in increasing order.
        Shape: (P,).
    pilot_values : array_like
        Known transmitted pilot constellation points.
        Shape: (P,) for shared pilots (broadcast to all MIMO channels),
        or (C, P) for per-channel pilots.
    interpolation : {'linear', 'cubic'}, default 'linear'
        Interpolation method between pilot positions.  Both modes loop over
        MIMO channels (``xp.interp`` and ``CubicSpline`` are 1D-only);
        C is typically 1-4 so the overhead is negligible.  ``'cubic'`` uses
        ``CubicSpline`` (CPU) or
        ``CubicSpline`` (GPU) with natural
        boundary conditions (zero second derivative at endpoints) and
        constant-hold extrapolation outside the pilot span.
    joint_channels : bool, default False
        For MIMO inputs (C > 1): if ``True``, perform coherent complex
        averaging of ``r_pilot * conj(s_pilot)`` across all channels before
        calling ``angle()``.  This avoids wrap-around artefacts that arise
        when averaging phases directly, and reduces variance by ~√C for
        shared-LO systems.  The resulting single phase trajectory is broadcast
        to all C output rows.  Has no effect for SISO (C = 1).
    cycle_slip_correction : bool, default False
        If ``True``, apply ``correct_cycle_slips`` to the unwrapped pilot
        phase sequence before interpolation, with ``symmetry=1`` (correction
        quantum ``2π``) to detect and fix wrap-around errors introduced by
        ``xp.unwrap`` at large inter-pilot gaps.
    cycle_slip_history : int, default 100
        ``history_length`` passed to ``correct_cycle_slips``.
    cycle_slip_threshold : float, default π/4
        ``threshold`` passed to ``correct_cycle_slips`` (radians).
    debug_plot : bool, default False
        If ``True``, opens a diagnostic figure showing the unwrapped pilot
        phase sequence and the interpolated phase trajectory.

    Returns
    -------
    array_like
        Per-symbol phase estimate in radians. Shape matches ``symbols``.
        Same backend as input.

    Notes
    -----
    Phase at each pilot: phi_hat[k] = angle(r[k] * conj(s[k])).  Linear
    interpolation constant-holds at the boundaries; cubic uses natural spline
    with constant-hold extrapolation.  Single-carrier only.
    """
    symbols, xp, _ = dispatch(symbols)
    was_1d = symbols.ndim == 1
    if was_1d:
        symbols = symbols[None, :]
    C, N = symbols.shape

    pilot_indices_np = to_device(pilot_indices, "cpu").astype(np.intp)
    pilot_indices_xp = xp.asarray(pilot_indices, dtype=xp.float64)
    pilot_values_xp = xp.asarray(pilot_values)
    P = len(pilot_indices_np)

    # Broadcast shared pilots (P,) -> (C, P) for all channels
    if pilot_values_xp.ndim == 1:
        pilot_values_xp = xp.broadcast_to(pilot_values_xp[None, :], (C, P))

    # Phase at each pilot position: angle(r_pilot · conj(s_pilot))
    r_pilots = symbols[:, pilot_indices_np]  # (C, P)

    if joint_channels and C > 1:
        # Coherent complex averaging before angle() - avoids wrap-around artefacts
        # that arise from averaging phases directly (e.g. antipodal channels).
        z_joint = xp.mean(r_pilots * xp.conj(pilot_values_xp), axis=0)  # (P,)
        phi_joint_u = xp.unwrap(xp.angle(z_joint).astype(xp.float64))  # (P,)
        if cycle_slip_correction:
            phi_joint_np = to_device(phi_joint_u, "cpu")
            phi_joint_np = correct_cycle_slips(
                phi_joint_np,
                symmetry=1,
                history_length=cycle_slip_history,
                threshold=cycle_slip_threshold,
            )
            phi_joint_u = xp.asarray(phi_joint_np)
        # Broadcast to (C, P) - read-only, downstream code only reads phi_pilots_u[ch]
        phi_pilots_u = xp.broadcast_to(phi_joint_u[None, :], (C, P))
    else:
        phi_pilots = xp.angle(r_pilots * xp.conj(pilot_values_xp))  # (C, P)
        # Unwrap along the pilot axis in float64 (cp.unwrap preserves input dtype;
        # casting before avoids precision loss in the discontinuity test for float32 input)
        phi_pilots_u = xp.unwrap(phi_pilots.astype(xp.float64), axis=-1)  # (C, P)
        if cycle_slip_correction:
            phi_pilots_u_np = to_device(phi_pilots_u, "cpu")
            for ch in range(C):
                phi_pilots_u_np[ch] = correct_cycle_slips(
                    phi_pilots_u_np[ch],
                    symmetry=1,
                    history_length=cycle_slip_history,
                    threshold=cycle_slip_threshold,
                )
            phi_pilots_u = xp.asarray(phi_pilots_u_np)

    all_positions = xp.arange(N, dtype=xp.float64)

    if interpolation == "linear":
        # xp.interp handles non-uniform pilot spacing natively, is boundary-safe
        # (extrapolates with first/last pilot value), and avoids the divide-by-zero
        # guards that the searchsorted form required.  Loop over C channels because
        # xp.interp is 1D-only; overhead is negligible for typical C = 1-4.
        phi_full = xp.empty((C, N), dtype=xp.float64)
        for ch in range(C):
            phi_full[ch] = xp.interp(all_positions, pilot_indices_xp, phi_pilots_u[ch])

    elif interpolation == "cubic":
        # CubicSpline is inherently per-channel (1D y input); loop is unavoidable.
        # Both scipy (CPU) and cupyx.scipy (GPU) share the same API.
        phi_full = xp.empty((C, N), dtype=xp.float64)
        if xp is not np:
            from cupyx.scipy.interpolate import CubicSpline
        else:
            from scipy.interpolate import CubicSpline

        for ch in range(C):
            phi_ch = phi_pilots_u[ch]  # already float64
            cs = CubicSpline(pilot_indices_xp, phi_ch, bc_type="natural")
            # Evaluate the spline only within the pilot span; constant-hold outside.
            first_idx = int(pilot_indices_np[0])
            last_idx = int(pilot_indices_np[-1])
            phi_full[ch, first_idx : last_idx + 1] = cs(
                all_positions[first_idx : last_idx + 1]
            )
            if first_idx > 0:
                phi_full[ch, :first_idx] = phi_ch[0]
            if last_idx < N - 1:
                phi_full[ch, last_idx + 1 :] = phi_ch[-1]

    else:
        raise ValueError(
            f"Unknown interpolation method: {interpolation!r}. "
            "Choose 'linear' or 'cubic'."
        )

    # Host copy of the trajectory is needed only for the INFO summary and the
    # optional debug plot; skip the transfer + reductions otherwise (the device
    # phi_full drives the actual correction and is what gets returned).
    _want_log = logger.isEnabledFor(logging.INFO)
    if _want_log or debug_plot:
        phi_full_np = to_device(phi_full, "cpu")
    if _want_log:
        phi_mean_deg = float(np.mean(phi_full_np)) * 180.0 / np.pi
        phi_std_deg = float(np.std(phi_full_np)) * 180.0 / np.pi
        logger.info(
            f"CPR (pilot-aided, {interpolation}): phase mean={phi_mean_deg:.2f}°, "
            f"std={phi_std_deg:.2f}° [P={P} pilots, C={C}]"
        )

    if debug_plot:
        from .. import plotting as _plotting

        phi_pilots_u_np = to_device(phi_pilots_u, "cpu")
        _plotting.plot_pilot_phase_estimate(
            pilot_indices=pilot_indices_np,
            phi_pilots_u=phi_pilots_u_np,
            phi_full=phi_full_np,
            show=True,
            title="CPR - Pilot-Aided Phase",
        )

    if was_1d:
        return phi_full[0]
    return phi_full


def _extract_pilot_phasor(
    samples: ArrayType,
    sampling_rate: float,
    tone_frequency: float,
    bandwidth: float,
    xp,
    search_band: float | None = None,
    refine_tone: bool = True,
    window: str | tuple = "tukey",
    X: ArrayType | None = None,
    return_window: bool = False,
) -> tuple[ArrayType, np.ndarray, np.ndarray, np.ndarray, ArrayType | None, ArrayType]:
    """Isolate a CW pilot tone and return its carrier-stripped complex phasor.

    Shared core of the pilot-tone CPR functions
    (``recover_carrier_phase_pilot_tone``, ``recover_carrier_phase_pilot_tones``):
    refine the per-channel tone centre, extract it with a zero-phase spectral
    window (FFT -> window -> IFFT, sample-aligned), and strip the nominal carrier
    so a residual frequency offset survives as a slow phase ramp.

    Everything runs in the signal's working precision (complex64 for complex64
    input) - the ±π-safe part of the pipeline is the float64 promotion of the
    *angle* before unwrap, which the callers already perform (CLAUDE.md), not
    double-precision spectra.  Tone refinement reuses the extraction FFT
    (device-side log-parabolic fit; one host transfer) instead of running a
    zero-padded full-record FFT per channel, and the window/noise statistics
    touch only the ``O(bandwidth/df)`` occupied bins instead of dense (C, N)
    arrays.

    Parameters
    ----------
    samples : (C, N) array
        Oversampled complex samples, already 2-D (caller handles the 1-D case).
    xp : module
        The dispatched array module for ``samples`` (numpy/cupy).
    X : (C, N) array, optional
        Precomputed ``xp.fft.fft(samples, axis=-1)`` in working precision.
        Pass it when extracting several tones from the same record so the
        record is transformed once (``recover_carrier_phase_pilot_tones``).
    return_window : bool, default False
        Build and return the dense (C, N) extraction window ``W`` (diagnostics
        only); when ``False`` the ``W`` slot in the return tuple is ``None``.
    (others) : see ``recover_carrier_phase_pilot_tone``.

    Returns
    -------
    phasor : (C, N) complex, working precision
        ``z(n) ≈ A·e^{jθ(n)}`` per channel (carrier-frequency stripped).
    f_centers : (C,) float64
        Detected per-channel tone centre [Hz].
    sig_power : (C,) float64
        In-band tone power ``|A|²`` within the tracking window, in the same
        units as ``mean(|z|²)`` (for SNR weights).
    noise_power : (C,) float64
        Additive-noise power ``σ²`` within the tracking window, same units.
    W : (C, N) float64 or None
        The extraction window (only if ``return_window=True``).
    X : (C, N) complex, working precision
        The full FFT (reusable for further tones / diagnostics).
    """
    from ..frequency import _refine_tones_from_spectrum, correct_static_frequency_offset

    C, N = samples.shape
    df = sampling_rate / N
    if search_band is None:
        search_band = bandwidth

    # 1) One FFT in working precision (nfft = N keeps the IFFT sample-aligned).
    if X is None:
        xw = (
            samples
            if samples.dtype == xp.complex128
            else samples.astype(xp.complex64, copy=False)
        )
        X = xp.fft.fft(xw, axis=-1)  # (C, N)
    real_dtype = xp.float64 if X.dtype == xp.complex128 else xp.float32

    # 2) Per-channel tone centre.  Refinement absorbs a frequency offset that
    #    has dragged the tone away from nominal, so the window stays centred on
    #    it.  Runs on the shared spectrum - no extra FFTs, one host transfer.
    if refine_tone:
        f_centers = _refine_tones_from_spectrum(
            X,
            sampling_rate,
            [float(tone_frequency)] * C,
            search_band,
            rows=range(C),
        )
    else:
        f_centers = np.full(C, float(tone_frequency), dtype=np.float64)

    # 3) Zero-phase extraction window placed circularly at each channel's tone bin.
    from scipy.signal import get_window

    half = int(bandwidth // df)  # bins from centre to band edge
    n_win = 2 * half + 1
    try:
        win_cpu = np.asarray(get_window(window, n_win, fftbins=False), dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Invalid window {window!r}: {exc}. Pass any scipy.signal.get_window "
            "spec, e.g. 'tukey', ('tukey', 0.3), 'boxcar', ('gaussian', 50)."
        ) from exc
    k_centers = np.round(f_centers / df).astype(np.int64) % N  # (C,) centre bins
    idx_np = (k_centers[:, None] + np.arange(-half, half + 1)[None, :]) % N
    idx = xp.asarray(idx_np)  # (C, n_win) circular in-band bins
    rows = xp.arange(C)[:, None]
    Xb = X[rows, idx]  # (C, n_win) gathered in-band spectrum

    # Per-channel in-band tone power and additive-noise power, via Parseval
    # (mean|z|² = ΣΣ|X·W|²/N²).  The noise floor is the median |X|² of a guard
    # band one window-width outside the passband (local, so a neighbouring tone
    # on the other channel does not inflate it).  Batched over channels on the
    # gathered bins only - no dense (C, N) window/power arrays, one transfer.
    win64 = xp.asarray(win_cpu)  # (n_win,) float64 on device
    pow_b = xp.abs(Xb).astype(xp.float64) ** 2  # (C, n_win)
    guard_off = np.arange(half + 1, half + 1 + n_win)
    guard_np = (
        np.concatenate(
            [
                (k_centers[:, None] + guard_off[None, :]),
                (k_centers[:, None] - guard_off),
            ],
            axis=1,
        )
        % N
    )  # (C, 2·n_win)
    pow_guard = xp.abs(X[rows, xp.asarray(guard_np)]).astype(xp.float64) ** 2
    floor_psd = xp.median(pow_guard, axis=-1)  # (C,)
    noise_dev = floor_psd * (n_win / (N * N))
    win_tot = xp.sum(pow_b * win64**2, axis=-1)  # (C,)
    sig_dev = xp.maximum(win_tot / (N * N) - noise_dev, 1e-30)
    stats = to_device(xp.stack([sig_dev, noise_dev]), "cpu")  # one transfer
    sig_power, noise_power = stats[0], stats[1]

    # 4) Windowed band -> time domain: scatter the weighted bins into an
    #    otherwise-zero spectrum (equivalent to the dense X·W, without the
    #    full-record multiply) and IFFT.
    Xw = xp.zeros((C, N), dtype=X.dtype)
    Xw[rows, idx] = Xb * win64.astype(real_dtype)[None, :]
    tone_t = xp.fft.ifft(Xw, axis=-1)  # (C, N) working precision

    # 5) Strip the *nominal* carrier so a residual frequency offset survives as
    #    a phase ramp - exact (non-quantized) complex mixing, same primitive
    #    the FOE correctors use (float64 phase ramp, wrapped, then cast to
    #    tone_t's working precision).
    phasor = correct_static_frequency_offset(tone_t, sampling_rate, tone_frequency)

    W = None
    if return_window:  # dense window for diagnostics/plots only
        W = xp.zeros((C, N), dtype=xp.float64)
        W[rows, idx] = win64[None, :]
    return phasor, f_centers, sig_power, noise_power, W, X


def recover_carrier_phase_pilot_tone(
    samples: ArrayType,
    sampling_rate: float,
    tone_frequency: float,
    bandwidth: float,
    search_band: float | None = None,
    refine_tone: bool = True,
    window: str | tuple = "tukey",
    remove_frequency_offset: bool = True,
    joint_channels: bool = False,
    debug_plot: bool = False,
) -> ArrayType:
    r"""
    Carrier phase recovery from a continuous-wave (CW) pilot tone.

    Reads the common carrier phase straight off a pilot tone added at the
    transmitter (see ``add_pilot_tone``).  Because
    the tone shares the data's local oscillator and channel, its phase equals
    the common phase theta[n] = 2*pi*delta_f*n/f_s + phi_PN[n]
    + phi_0 - the carrier **frequency offset and phase noise jointly**.  No
    symbol decisions are required, so there is no M-th-power noise enhancement
    and the estimate tracks fast phase noise sample-by-sample.

    The tone is isolated with a **zero-phase** spectral window (FFT -> window ->
    IFFT), so there is no group-delay misalignment between the recovered phase
    and the samples.

    Note: operates on the **oversampled waveform, before matched filtering and
    decimation**.  The tone lives in a guard band that the matched filter would
    otherwise remove.  Apply the returned phase to the same oversampled
    ``samples`` with ``correct_carrier_phase``, then run matched filtering /
    decimation and any residual 1-sps CPR.

    Parameters
    ----------
    samples : array_like
        Oversampled complex samples (``sps > 1``). Shape: ``(N,)`` or
        ``(C, N)``.  Same rate as used for ``add_pilot_tone``.
    sampling_rate : float
        Sampling rate f_s in Hz.
    tone_frequency : float
        Nominal pilot-tone frequency f_p in Hz (as added at the TX).
        The recovered phase is referenced to **this** carrier, so any carrier
        frequency offset remains in the phase ramp when
        ``remove_frequency_offset=False`` is *not* set (see below).
    bandwidth : float
        Half-width B of the spectral extraction window in Hz - the
        **tracking bandwidth**.  Must be wide enough to pass the phase-noise
        sidebands (``B ≳ a few x linewidth``) yet narrow enough to reject the
        data band (``B`` smaller than the tone-to-signal-edge guard).  See the
        guide at the end of this docstring.
    search_band : float, optional
        Half-width in Hz of the peak-search window handed to
        ``find_bias_tone`` when ``refine_tone=True``.
        The actual tone peak is sought within
        ``[f_p - search_band, f_p + search_band]``; this bounds how far
        a frequency offset may have dragged the tone from nominal.  Defaults
        to ``bandwidth``.  Enlarge it (independently of ``bandwidth``)
        when the offset can exceed ``B`` but keep it inside the guard so the
        data band never wins the argmax.
    refine_tone : bool, default True
        If ``True``, locate the actual per-channel tone frequency with
        ``find_bias_tone`` and centre the extraction
        window there.  Essential when a frequency offset may shift the tone by
        more than ``B`` (otherwise the tone falls outside a window centred at
        nominal).  If ``False``, the window is centred at ``tone_frequency``.
    window : str or tuple, default 'tukey'
        Spectral window applied over the passband |f - f_centre| <= B.
        Any spec accepted by ``get_window`` (e.g. ``'tukey'``,
        ``'boxcar'``, ``('gaussian', std)``).  Tukey (default) gives a flat top
        with tapered edges for suppressed ringing.
    remove_frequency_offset : bool, default True
        If ``True`` (default), the recovered phase **retains** the linear ramp
        from any residual carrier frequency offset, so applying it corrects
        frequency offset and phase noise together.  If ``False``, the
        least-squares linear trend is subtracted per channel, leaving only the
        phase-noise fluctuation (use when the frequency offset is handled by a
        separate stage).
    joint_channels : bool, default False
        For MIMO inputs (C > 1): if ``True``, coherently sum the extracted
        tone phasors across channels before taking the angle (shared-LO,
        ~√C variance reduction).  The single trajectory is broadcast to all
        rows.  No effect for SISO.
    debug_plot : bool, default False
        If ``True``, open the dedicated diagnostic figure
        (``pilot_tone_phase_estimate``): the tone
        spectrum with the extraction window overlaid, and the recovered phase.

    Returns
    -------
    array_like
        Per-sample phase estimate theta_hat[n] in radians.  Shape
        matches ``samples``; same backend.  Apply with
        ``correct_carrier_phase``.

    Notes
    -----
    Pipeline: FFT -> (optional) refine tone centre -> zero-phase window extraction
    -> strip nominal carrier -> unwrap(angle) in float64.

    ``bandwidth`` B trades phase-noise tracking bandwidth against tone SNR.
    Lower bound: B ≳ 3-5 * linewidth (pass all phase-noise sidebands).
    Upper bound: B below the guard between the tone and the signal band edge.
    Place the tone at |f_p| > (1+beta)*R_s/2 + B and keep |f_p| + B < f_s/2.
    """
    if bandwidth <= 0.0:
        raise ValueError(f"bandwidth must be > 0, got {bandwidth}.")
    if not (-sampling_rate / 2.0 < tone_frequency < sampling_rate / 2.0):
        raise ValueError(f"tone_frequency={tone_frequency} must lie in (-fs/2, fs/2).")

    samples, xp, _ = dispatch(samples)
    was_1d = samples.ndim == 1
    if was_1d:
        samples = samples[None, :]  # (1, N)
    C, N = samples.shape

    df = sampling_rate / N
    if bandwidth < df:
        logger.warning(
            f"CPR (pilot-tone): bandwidth={bandwidth:.3g} Hz is below the FFT "
            f"resolution df=fs/N={df:.3g} Hz; the extraction window may capture too few "
            f"bins. Increase bandwidth or the record length N."
        )

    # Isolate the tone and strip the nominal carrier (shared core).
    phasor, f_centers, _, _, W, X = _extract_pilot_phasor(
        samples,
        sampling_rate,
        tone_frequency,
        bandwidth,
        xp,
        search_band=search_band,
        refine_tone=refine_tone,
        window=window,
        return_window=debug_plot,
    )
    n = xp.arange(N, dtype=xp.float64)  # for the residual-FOE detrend below

    # 5) Phase extraction + unwrap in float64.
    if joint_channels and C > 1:
        z_joint = xp.sum(phasor, axis=0)  # (N,) coherent sum
        theta_joint = xp.unwrap(xp.angle(z_joint).astype(xp.float64))  # (N,)
        theta = xp.broadcast_to(theta_joint[None, :], (C, N)).copy()
    else:
        theta = xp.unwrap(xp.angle(phasor).astype(xp.float64), axis=-1)  # (C, N)

    if not remove_frequency_offset:
        # Subtract the per-channel least-squares linear trend (residual FOE),
        # preserving the mean phase; leaves only the phase-noise fluctuation.
        nc = n - xp.mean(n)
        denom = xp.sum(nc * nc)
        theta_c = theta - xp.mean(theta, axis=-1, keepdims=True)
        slope = xp.sum(theta_c * nc[None, :], axis=-1, keepdims=True) / denom  # (C, 1)
        theta = theta - slope * nc[None, :]

    # Host copy of theta is needed only for the INFO summary and the optional
    # debug plot; skip the transfer + reductions otherwise (the device theta is
    # what gets returned and applied).
    _want_log = logger.isEnabledFor(logging.INFO)
    if _want_log or debug_plot:
        theta_np = to_device(theta, "cpu")
    if _want_log:
        phi_mean_deg = float(np.mean(theta_np)) * 180.0 / np.pi
        phi_std_deg = float(np.std(theta_np)) * 180.0 / np.pi
        mode_str = "joint" if (joint_channels and C > 1) else "independent"
        logger.info(
            f"CPR (pilot-tone, {window}, {mode_str}): phase mean={phi_mean_deg:.2f}°, "
            f"std={phi_std_deg:.2f}° [f_p={tone_frequency:.3g} Hz, B={bandwidth:.3g} Hz, "
            f"refine={refine_tone}, remove_foe={remove_frequency_offset}, C={C}]"
        )

    if debug_plot:
        from .. import plotting as _plotting

        _plotting.plot_pilot_tone_phase_estimate(
            freqs=np.fft.fftfreq(N, d=1.0 / sampling_rate),
            mag_spectrum=to_device(xp.abs(X), "cpu"),
            window=to_device(W, "cpu"),
            f_tones=f_centers,
            theta=theta_np,
            tone_frequency=float(tone_frequency),
            bandwidth=float(bandwidth),
            show=True,
        )

    if was_1d:
        return theta[0]
    return theta


def _lowpass_fft(z: ArrayType, sampling_rate: float, cutoff: float, xp) -> ArrayType:
    """Zero-phase brick-wall low-pass of a complex stream (FFT -> mask -> IFFT).

    Used to isolate the **slow** inter-tone differential phasor; zero-phase so
    the recovered ``δ(n)`` is lag-free (it is far inside the passband anyway).
    """
    N = z.shape[-1]
    freqs = xp.fft.fftfreq(N, d=1.0 / sampling_rate)
    mask = (xp.abs(freqs) <= cutoff).astype(z.real.dtype)
    return xp.fft.ifft(xp.fft.fft(z, axis=-1) * mask, axis=-1)


def recover_carrier_phase_pilot_tones(
    samples: ArrayType,
    sampling_rate: float,
    tone_frequencies,
    bandwidth: float,
    differential_bandwidth: float = 5e3,
    search_band: float | None = None,
    per_tone_channel: list | None = None,
    snr_gate_db: float = 3.0,
    coherence_gate: float = 0.3,
    refine_tone: bool = True,
    window: str | tuple = "tukey",
    return_diagnostics: bool = False,
    debug_plot: bool = False,
):
    r"""
    Common carrier-phase recovery from two (or more) CW pilot tones via
    SNR-weighted maximal-ratio combining with slow inter-tone tracking.

    For a shared-laser/shared-LO dual-pol link the carrier (beat) phase phi[n]
    is common-mode across both polarizations, and every pilot rides the same
    phi[n].  Combining K tones lowers the residual phase noise by up to sqrt(K)
    over a single tone, directly reducing the excess noise it converts into
    (xi_phi ~ V_A * Var(d_phi)).  The static inter-tone offset theta_k - theta_0
    is constant back-to-back but drifts over fiber (SOP rotation acting on the
    orthogonally-launched pilots), so it is tracked, not calibrated: the product
    z_k * conj(z_0) cancels the common phi[n] exactly, leaving only the slow
    differential, which a narrow low-pass isolates.

    The whole combine collapses to one expression,

        z_comb[n] = sum_k  z_k[n] * conj(c_k[n]) / sigma_k^2,
        c_k[n]    = LPF( z_k[n] * conj(z_0[n]) ),

    where conj(c_k) carries both the magnitude weight |A_k||A_0| (so fades are
    down-weighted per-sample) and the de-rotation exp(-j*delta_k); dividing by
    the additive-noise power sigma_k^2 makes it true MRC.  A tone whose SNR or
    differential coherence falls below the gates is dropped, so the combine
    degrades gracefully to single-tone in a deep fade.

    Operates on the oversampled waveform, before matched filtering, exactly like
    ``recover_carrier_phase_pilot_tone``.  Best run *after* the polarization
    demux (each pilot isolated on its own output, ``per_tone_channel=[0, 1]``);
    pre-demux it falls back to a joint-across-channels sum per tone.

    Parameters
    ----------
    samples : (N,) or (C, N) array
        Oversampled complex samples (``sps > 1``).
    sampling_rate : float
        Sampling rate in Hz (of *these* samples - pass the post-resample rate if
        the demux/resample ran first).
    tone_frequencies : sequence of float
        Nominal pilot-tone frequencies in Hz (length ``K``).
    bandwidth : float
        Half-width of the per-tone extraction window in Hz (the common-phase
        tracking bandwidth); see ``recover_carrier_phase_pilot_tone``.
    differential_bandwidth : float, default 5e3
        Low-pass cut-off in Hz for the slow inter-tone differential delta_k[n].
        Choose above the SOP drift rate (so delta is not lagged) and far below
        the phase-noise band (kHz-scale is typical).  Set it from the knee of the
        ``angle(z_k·conj(z_0))`` spectrum.
    search_band : float, optional
        Peak-search half-width handed to ``find_bias_tone``; defaults to
        ``bandwidth``.
    per_tone_channel : list of int, optional
        Channel each tone is read from (post-demux isolation, e.g. ``[0, 1]``).
        If ``None``, each tone's phasor is the coherent sum across all channels
        (pre-demux joint combine).
    snr_gate_db : float, default 3.0
        A non-reference tone is dropped if its in-band SNR is below this.
    coherence_gate : float, default 0.3
        A non-reference tone is dropped if its differential coherence
        ``mean|c_k| / sqrt(S_k·S_ref)`` is below this (deep fade / lost lock).
    refine_tone, window : see ``recover_carrier_phase_pilot_tone``.
    return_diagnostics : bool, default False
        If ``True``, also return a dict with ``delta`` (per-tone delta_k[n]),
        ``snr_db``, ``ref`` (reference-tone index) and ``used`` (combined tone
        indices).
    debug_plot : bool, default False
        If ``True``, plot the per-tone differential phase and the combined track.

    Returns
    -------
    array_like
        Per-sample common phase estimate phi_hat[n], shape matching ``samples``
        (one track broadcast to all rows).  Apply with
        ``correct_carrier_phase``.  If ``return_diagnostics``, returns
        ``(phi, diagnostics)``.
    """
    if bandwidth <= 0.0:
        raise ValueError(f"bandwidth must be > 0, got {bandwidth}.")
    tone_frequencies = list(tone_frequencies)
    K = len(tone_frequencies)
    if K < 1:
        raise ValueError("tone_frequencies must contain at least one frequency.")

    samples, xp, _ = dispatch(samples)
    was_1d = samples.ndim == 1
    if was_1d:
        samples = samples[None, :]
    C, N = samples.shape
    if per_tone_channel is not None and len(per_tone_channel) != K:
        raise ValueError(
            f"per_tone_channel must have one entry per tone (len {K}), "
            f"got {len(per_tone_channel)}."
        )

    # 1) Extract each tone's scalar phasor stream z_k(n) and its (S_k, σ_k²).
    # One shared working-precision FFT of the record serves every tone's
    # refinement and extraction (K windowed IFFTs remain, nothing else scales
    # with K·N).
    xw = (
        samples
        if samples.dtype == xp.complex128
        else samples.astype(xp.complex64, copy=False)
    )
    X = xp.fft.fft(xw, axis=-1)  # (C, N)
    z_tones, sig, noise, f_centers = [], [], [], []
    for k, f_k in enumerate(tone_frequencies):
        ph, fc, s_c, n_c, _, _ = _extract_pilot_phasor(
            samples,
            sampling_rate,
            f_k,
            bandwidth,
            xp,
            search_band=search_band,
            refine_tone=refine_tone,
            window=window,
            X=X,
        )
        if per_tone_channel is None:
            z_k = xp.sum(ph, axis=0)  # joint across channels (pre-demux)
            s_k, n_k = float(np.sum(s_c)), float(np.sum(n_c))
        else:
            ch = int(per_tone_channel[k])
            z_k, s_k, n_k = ph[ch], float(s_c[ch]), float(n_c[ch])
        z_tones.append(z_k)
        sig.append(s_k)
        noise.append(max(n_k, 1e-30))
        f_centers.append(fc)

    # 2) Reference = highest-SNR tone; build the slow differential phasors c_k.
    snr = np.array([s / nz for s, nz in zip(sig, noise)], dtype=np.float64)
    ref = int(np.argmax(snr))
    z_ref = z_tones[ref]
    snr_gate = 10.0 ** (snr_gate_db / 10.0)

    z_comb = xp.zeros(N, dtype=z_ref.dtype)
    delta_diag, used = [], []
    want_diag = return_diagnostics or debug_plot  # per-tone δ_k needed either way
    for k in range(K):
        if k == ref:
            # Self-product: LPF(|z|²) ≈ |A|² + σ²; subtract the floor so the
            # reference weight is the true |A|² (its phase is 0 => no de-rotation).
            c_k = _lowpass_fft(
                (xp.abs(z_ref) ** 2).astype(z_ref.dtype),
                sampling_rate,
                differential_bandwidth,
                xp,
            )
            c_k = xp.clip(xp.real(c_k) - noise[ref], 1e-30, None).astype(z_ref.dtype)
            coh = 1.0
        else:
            # Cross-product: the two tones' noises are independent => the LPF
            # rejects them, so c_k ≈ A_k A_0* - carries |A_k||A_0| and e^{jδ_k}.
            c_k = _lowpass_fft(
                z_tones[k] * xp.conj(z_ref),
                sampling_rate,
                differential_bandwidth,
                xp,
            )
            coh = float(xp.mean(xp.abs(c_k))) / np.sqrt(max(sig[k] * sig[ref], 1e-30))
            if snr[k] < snr_gate or coh < coherence_gate:
                logger.info(
                    f"CPR (pilot-tones): tone {k} dropped "
                    f"(SNR={10 * np.log10(snr[k]):.1f} dB, coherence={coh:.2f})."
                )
                if want_diag:
                    delta_diag.append(
                        to_device(xp.angle(c_k).astype(xp.float64), "cpu")
                    )
                continue
        contrib = z_tones[k] * xp.conj(c_k)
        contrib /= noise[k]
        z_comb += contrib
        used.append(k)
        if want_diag:
            delta_diag.append(to_device(xp.angle(c_k).astype(xp.float64), "cpu"))

    # 3) Common phase = angle of the combined phasor, unwrapped in float64.
    phi = xp.unwrap(xp.angle(z_comb).astype(xp.float64))  # (N,)
    phi_full = xp.broadcast_to(phi[None, :], (C, N)).copy()

    # Host copy of phi is needed only for the INFO summary and the optional
    # debug plot; skip the transfer otherwise (phi_full drives the correction).
    _want_log = logger.isEnabledFor(logging.INFO)
    if _want_log or debug_plot:
        phi_np = to_device(phi, "cpu")
    if _want_log:
        logger.info(
            f"CPR (pilot-tones, MRC): phase std={float(np.std(phi_np)) * 180 / np.pi:.2f}°, "
            f"[K={K}, used={used}, ref={ref}, B={bandwidth:.3g} Hz, "
            f"diff_B={differential_bandwidth:.3g} Hz, C={C}]"
        )

    if debug_plot:
        from .. import plotting as _plotting

        _plotting.plot_pilot_tones_phase_estimate(
            delta=delta_diag,
            phi=phi_np,
            ref=ref,
            used=used,
            show=True,
        )

    phi_out = phi_full[0] if was_1d else phi_full
    if return_diagnostics:
        diagnostics = {
            "delta": delta_diag,
            "snr_db": 10.0 * np.log10(snr),
            "ref": ref,
            "used": used,
            "f_centers": [np.asarray(fc) for fc in f_centers],
        }
        return phi_out, diagnostics
    return phi_out
