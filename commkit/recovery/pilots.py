"""Pilot-symbol and pilot-tone aided carrier phase recovery."""

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .._array import broadcast_channels
from ..backend import ArrayType, dispatch, to_device
from ..logger import logger
from ..math import _remove_linear_trend
from ._common import _Context, _Phase
from .corrections import CycleSlip, _log_phase_summary, _repair_slips


@dataclass(frozen=True, eq=False)
class PilotAided:
    """
    Phase from known pilot symbols, interpolated between pilots.

    The phase at each pilot is ``∠(r_k·s_k*)``; the pilot phases are
    unwrapped and interpolated over the whole record.  Single-carrier only.

    Parameters
    ----------
    indices : array_like of int
        Symbol indices of the pilots, strictly increasing, ``(P,)``.
    values : array_like
        Transmitted pilot symbols, ``(P,)`` shared or ``(C, P)`` per channel.
    interpolation : {"linear", "cubic"}, default "linear"
        ``"linear"`` holds the first and last pilot phase outside the pilot
        span; ``"cubic"`` is a natural cubic spline inside the span with the
        same constant hold outside.
    joint_channels : bool, default False
        MIMO: average ``r·s*`` coherently across channels before the angle
        (no wrap-around artefacts) and give every channel the one trajectory.
    cycle_slip : CycleSlip, optional
        Repair ``2π`` wraps in the unwrapped pilot phases (large pilot gaps)
        before interpolation.
    """

    indices: np.ndarray
    values: np.ndarray
    interpolation: Literal["linear", "cubic"] = "linear"
    joint_channels: bool = False
    cycle_slip: CycleSlip | None = None

    def __post_init__(self) -> None:
        indices = np.array(to_device(self.indices, "cpu")).astype(np.intp)
        values = np.array(to_device(self.values, "cpu"))
        if indices.ndim != 1 or indices.size < 1:
            raise ValueError("indices must be 1-D with at least one pilot.")
        if np.any(np.diff(indices) <= 0):
            raise ValueError("indices must be strictly increasing.")
        if values.shape[-1] != indices.size or values.ndim not in (1, 2):
            raise ValueError(
                f"values must have shape (P,) or (C, P) with P={indices.size}, "
                f"got {values.shape}."
            )
        if self.interpolation not in ("linear", "cubic"):
            raise ValueError(
                f"Unknown interpolation method: {self.interpolation!r}. "
                "Choose 'linear' or 'cubic'."
            )
        for name, arr in (("indices", indices), ("values", values)):
            arr.setflags(write=False)
            object.__setattr__(self, name, arr)


def _pilot_aided(symbols: ArrayType, method: PilotAided, ctx: _Context) -> _Phase:
    """Pilot-aided phase of ``(C, N)`` symbols."""
    symbols, xp, _ = dispatch(symbols)
    C, N = symbols.shape
    interpolation = method.interpolation

    pilot_indices_np = method.indices
    pilot_indices_xp = xp.asarray(pilot_indices_np, dtype=xp.float64)
    P = len(pilot_indices_np)
    if pilot_indices_np[-1] >= N:
        raise ValueError(
            f"Pilot index {int(pilot_indices_np[-1])} is outside the {N} symbols."
        )

    # Broadcast shared pilots (P,) -> (C, P) for all channels
    pilot_values_xp = broadcast_channels(
        xp.asarray(method.values), C, xp, name="pilot_values"
    )

    # Phase at each pilot position: angle(r_pilot · conj(s_pilot))
    r_pilots = symbols[:, pilot_indices_np]  # (C, P)

    joint = method.joint_channels and C > 1
    if joint:
        # Coherent complex averaging before angle() - avoids wrap-around artefacts
        # that arise from averaging phases directly (e.g. antipodal channels).
        z_joint = xp.mean(r_pilots * xp.conj(pilot_values_xp), axis=0)  # (P,)
        phi_pilots_u = xp.unwrap(xp.angle(z_joint).astype(xp.float64))[None, :]
    else:
        phi_pilots = xp.angle(r_pilots * xp.conj(pilot_values_xp))  # (C, P)
        # Unwrap along the pilot axis in float64 (cp.unwrap preserves input dtype;
        # casting before avoids precision loss in the discontinuity test for float32 input)
        phi_pilots_u = xp.unwrap(phi_pilots.astype(xp.float64), axis=-1)  # (C, P)
    # Correction quantum 2π: wrap-around errors of the unwrap at large gaps.
    phi_pilots_u = _repair_slips(phi_pilots_u, xp, method.cycle_slip, 1)
    R = phi_pilots_u.shape[0]

    all_positions = xp.arange(N, dtype=xp.float64)
    phi_full = xp.empty((R, N), dtype=xp.float64)
    if interpolation == "linear":
        # xp.interp handles non-uniform pilot spacing natively and holds the
        # first/last pilot value outside the span.  1D-only: loop over rows.
        for ch in range(R):
            phi_full[ch] = xp.interp(all_positions, pilot_indices_xp, phi_pilots_u[ch])
    else:
        # CubicSpline is inherently per-channel (1D y input); loop is unavoidable.
        # Both scipy (CPU) and cupyx.scipy (GPU) share the same API.
        if xp is not np:
            from cupyx.scipy.interpolate import CubicSpline
        else:
            from scipy.interpolate import CubicSpline

        first_idx = int(pilot_indices_np[0])
        last_idx = int(pilot_indices_np[-1])
        for ch in range(R):
            phi_ch = phi_pilots_u[ch]  # already float64
            cs = CubicSpline(pilot_indices_xp, phi_ch, bc_type="natural")
            # Evaluate the spline only within the pilot span; constant-hold outside.
            phi_full[ch, first_idx : last_idx + 1] = cs(
                all_positions[first_idx : last_idx + 1]
            )
            if first_idx > 0:
                phi_full[ch, :first_idx] = phi_ch[0]
            if last_idx < N - 1:
                phi_full[ch, last_idx + 1 :] = phi_ch[-1]
    if joint:
        phi_full = xp.broadcast_to(phi_full, (C, N)).copy()
        phi_pilots_u = xp.broadcast_to(phi_pilots_u, (C, P)).copy()

    _log_phase_summary(
        phi_full,
        "CPR (pilot-aided, %s)",
        (interpolation,),
        "[P=%s pilots, C=%s]",
        (P, C),
    )
    return _Phase(
        phase=phi_full, pilot_indices=pilot_indices_np, pilot_phase=phi_pilots_u
    )


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

    Shared core of the ``PilotTone`` and ``PilotTones`` estimators:
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
        record is transformed once (``PilotTones``).
    return_window : bool, default False
        Build and return the dense (C, N) extraction window ``W`` (diagnostics
        only); when ``False`` the ``W`` slot in the return tuple is ``None``.
    (others) : see ``PilotTone``.

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
    from ..frequency import _refine_tones_from_spectrum, correct_frequency_offset

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
    phasor = correct_frequency_offset(
        tone_t, tone_frequency, sampling_rate=sampling_rate
    )

    W = None
    if return_window:  # dense window for diagnostics/plots only
        W = xp.zeros((C, N), dtype=xp.float64)
        W[rows, idx] = win64[None, :]
    return phasor, f_centers, sig_power, noise_power, W, X


@dataclass(frozen=True)
class PilotTone:
    r"""
    Common carrier phase from a continuous-wave pilot tone.

    The tone (see ``spectral.add_pilot_tone``) shares the data's oscillator
    and channel, so its phase is the common phase
    ``θ[n] = 2π·Δf·n/f_s + φ_PN[n] + φ_0``: frequency offset and phase noise
    together, with no decisions and no M-th power noise enhancement.  It is
    isolated with a zero-phase spectral window (FFT, window, IFFT), so the
    phase is not delayed against the samples.

    Works on the oversampled waveform before matched filtering (the tone
    sits in a guard band the matched filter removes).  Correct the same
    samples, then matched-filter, decimate and run any residual 1-SPS CPR.

    Parameters
    ----------
    frequency : float
        Nominal tone frequency f_p in Hz, in ``(-f_s/2, f_s/2)``.  The phase
        is referenced to this carrier.
    bandwidth : float
        Half-width B of the extraction window in Hz, the tracking
        bandwidth: above a few linewidths, below the tone-to-data guard.
    search_band : float, optional
        Half-width in Hz of the peak search when ``refine=True``; defaults
        to ``bandwidth``.  Keep it inside the guard so the data band never
        wins the search.
    refine : bool, default True
        Centre the window on the measured per-channel tone peak (needed when
        a frequency offset can move the tone by more than B).
    window : str or tuple, default "tukey"
        Window over the passband, any ``scipy.signal.get_window`` spec.
    remove_frequency_offset : bool, default True
        Keep the linear phase ramp of a residual frequency offset in the
        estimate, so that correcting removes offset and phase noise
        together.  ``False`` subtracts the per-channel least-squares trend
        and leaves only the phase-noise fluctuation.
    joint_channels : bool, default False
        MIMO: sum the tone phasors across channels before the angle (shared
        LO) and give every channel the one trajectory.

    Notes
    -----
    Place the tone at ``|f_p| > (1+β)·R_s/2 + B`` and keep
    ``|f_p| + B < f_s/2``; choose ``B ≳ 3-5 × linewidth``.
    """

    frequency: float
    bandwidth: float
    search_band: float | None = None
    refine: bool = True
    window: str | tuple = "tukey"
    remove_frequency_offset: bool = True
    joint_channels: bool = False

    def __post_init__(self) -> None:
        if not self.bandwidth > 0.0:
            raise ValueError(f"bandwidth must be > 0, got {self.bandwidth}.")


def _pilot_tone(samples: ArrayType, method: PilotTone, ctx: _Context) -> _Phase:
    """Pilot-tone phase of ``(C, N)`` samples."""
    sampling_rate = ctx.need_sampling_rate(method)
    tone_frequency = method.frequency
    bandwidth = method.bandwidth
    if not (-sampling_rate / 2.0 < tone_frequency < sampling_rate / 2.0):
        raise ValueError(f"tone frequency {tone_frequency} must lie in (-fs/2, fs/2).")

    samples, xp, _ = dispatch(samples)
    C, N = samples.shape

    df = sampling_rate / N
    if bandwidth < df:
        logger.warning(
            "CPR (pilot-tone): bandwidth=%.3g Hz is below the FFT "
            "resolution df=fs/N=%.3g Hz; the extraction window may capture "
            "too few bins. Increase bandwidth or the record length N.",
            bandwidth,
            df,
        )

    # Isolate the tone and strip the nominal carrier (shared core).
    phasor, f_centers, _, _, _, _ = _extract_pilot_phasor(
        samples,
        sampling_rate,
        tone_frequency,
        bandwidth,
        xp,
        search_band=method.search_band,
        refine_tone=method.refine,
        window=method.window,
    )

    # Phase extraction + unwrap in float64.
    joint = method.joint_channels and C > 1
    if joint:
        z_joint = xp.sum(phasor, axis=0)  # (N,) coherent sum
        theta_joint = xp.unwrap(xp.angle(z_joint).astype(xp.float64))  # (N,)
        theta = xp.broadcast_to(theta_joint[None, :], (C, N)).copy()
    else:
        theta = xp.unwrap(xp.angle(phasor).astype(xp.float64), axis=-1)  # (C, N)

    if not method.remove_frequency_offset:
        # Subtract the per-channel least-squares linear trend (residual FOE),
        # preserving the mean phase; leaves only the phase-noise fluctuation.
        theta, _ = _remove_linear_trend(theta)

    _log_phase_summary(
        theta,
        "CPR (pilot-tone, %s, %s)",
        (method.window, "joint" if joint else "independent"),
        "[f_p=%.3g Hz, B=%.3g Hz, refine=%s, remove_foe=%s, C=%s]",
        (tone_frequency, bandwidth, method.refine, method.remove_frequency_offset, C),
    )
    return _Phase(phase=theta, tone_frequencies=np.asarray(f_centers))


def _lowpass_fft(z: ArrayType, sampling_rate: float, cutoff: float, xp) -> ArrayType:
    """Zero-phase brick-wall low-pass of a complex stream (FFT -> mask -> IFFT).

    Used to isolate the **slow** inter-tone differential phasor; zero-phase so
    the recovered ``δ(n)`` is lag-free (it is far inside the passband anyway).
    """
    N = z.shape[-1]
    freqs = xp.fft.fftfreq(N, d=1.0 / sampling_rate)
    mask = (xp.abs(freqs) <= cutoff).astype(z.real.dtype)
    return xp.fft.ifft(xp.fft.fft(z, axis=-1) * mask, axis=-1)


@dataclass(frozen=True)
class PilotTones:
    r"""
    Common carrier phase from two or more pilot tones, combined by MRC.

    For a shared-laser dual-polarization link every pilot rides the same
    common phase φ[n]; combining K tones lowers the residual phase noise by
    up to √K over one tone.  The static inter-tone offset drifts over fiber
    (SOP rotation), so it is tracked: ``z_k·conj(z_0)`` cancels φ[n] and a
    narrow low-pass keeps the slow differential.  The combine is

        z_comb[n] = Σ_k z_k[n]·conj(c_k[n]) / σ_k²,
        c_k[n]    = LPF(z_k[n]·conj(z_0[n])),

    where ``conj(c_k)`` carries the magnitude weight and the de-rotation and
    ``σ_k²`` is the tone's noise power.  The reference is the strongest
    tone; a tone below the SNR or coherence gate is dropped, so a deep fade
    degrades gracefully to one tone.

    Like :class:`PilotTone`, works on the oversampled waveform before
    matched filtering, best after polarization demultiplexing.

    Parameters
    ----------
    frequencies : sequence of float
        Nominal tone frequencies in Hz (K values).
    bandwidth : float
        Half-width of each tone's extraction window in Hz.
    differential_bandwidth : float, default 5e3
        Low-pass cut-off in Hz for the inter-tone differential: above the
        SOP drift rate, far below the phase-noise band.
    search_band : float, optional
        Peak-search half-width in Hz; defaults to ``bandwidth``.
    per_tone_channel : sequence of int, optional
        Channel each tone is read from after demultiplexing (e.g.
        ``(0, 1)``).  ``None`` sums each tone across channels.
    snr_gate_db : float, default 3.0
        A non-reference tone below this in-band SNR is dropped.
    coherence_gate : float, default 0.3
        A non-reference tone whose differential coherence
        ``mean|c_k| / √(S_k·S_ref)`` is below this is dropped.
    refine, window
        As for :class:`PilotTone`.
    """

    frequencies: tuple[float, ...]
    bandwidth: float
    differential_bandwidth: float = 5e3
    search_band: float | None = None
    per_tone_channel: tuple[int, ...] | None = None
    snr_gate_db: float = 3.0
    coherence_gate: float = 0.3
    refine: bool = True
    window: str | tuple = "tukey"

    def __post_init__(self) -> None:
        frequencies = tuple(float(f) for f in self.frequencies)
        object.__setattr__(self, "frequencies", frequencies)
        if len(frequencies) < 1:
            raise ValueError("frequencies must contain at least one frequency.")
        if not self.bandwidth > 0.0:
            raise ValueError(f"bandwidth must be > 0, got {self.bandwidth}.")
        if self.per_tone_channel is not None:
            channels = tuple(int(c) for c in self.per_tone_channel)
            object.__setattr__(self, "per_tone_channel", channels)
            if len(channels) != len(frequencies):
                raise ValueError(
                    "per_tone_channel must have one entry per tone (len "
                    f"{len(frequencies)}), got {len(channels)}."
                )


def _pilot_tones(samples: ArrayType, method: PilotTones, ctx: _Context) -> _Phase:
    """MRC pilot-tones phase of ``(C, N)`` samples."""
    sampling_rate = ctx.need_sampling_rate(method)
    tone_frequencies = list(method.frequencies)
    bandwidth = method.bandwidth
    differential_bandwidth = method.differential_bandwidth
    per_tone_channel = method.per_tone_channel
    K = len(tone_frequencies)

    samples, xp, _ = dispatch(samples)
    C, N = samples.shape

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
            search_band=method.search_band,
            refine_tone=method.refine,
            window=method.window,
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
    snr_gate = 10.0 ** (method.snr_gate_db / 10.0)

    z_comb = xp.zeros(N, dtype=z_ref.dtype)
    delta, used = [], []
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
            if snr[k] < snr_gate or coh < method.coherence_gate:
                logger.info(
                    "CPR (pilot-tones): tone %s dropped (SNR=%.1f dB, coherence=%.2f).",
                    k,
                    10 * np.log10(snr[k]),
                    coh,
                )
                delta.append(xp.angle(c_k).astype(xp.float64))
                continue
        contrib = z_tones[k] * xp.conj(c_k)
        contrib /= noise[k]
        z_comb += contrib
        used.append(k)
        delta.append(xp.angle(c_k).astype(xp.float64))

    # 3) Common phase = angle of the combined phasor, unwrapped in float64.
    phi = xp.unwrap(xp.angle(z_comb).astype(xp.float64))  # (N,)
    phi_full = xp.broadcast_to(phi[None, :], (C, N)).copy()

    # Host copy of phi is needed only for the INFO summary; skip the transfer
    # otherwise (phi_full drives the correction).
    if logger.isEnabledFor(logging.INFO):
        phi_np = to_device(phi, "cpu")
        logger.info(
            "CPR (pilot-tones, MRC): phase std=%.2f°, [K=%s, used=%s, "
            "ref=%s, B=%.3g Hz, diff_B=%.3g Hz, C=%s]",
            float(np.std(phi_np)) * 180 / np.pi,
            K,
            used,
            ref,
            bandwidth,
            differential_bandwidth,
            C,
        )

    return _Phase(
        phase=phi_full,
        tone_frequencies=np.stack([np.asarray(fc) for fc in f_centers], axis=-1),
        tone_snr_db=10.0 * np.log10(snr),
        differential_phase=xp.stack(delta),
        reference_tone=ref,
        used_tones=tuple(used),
    )
