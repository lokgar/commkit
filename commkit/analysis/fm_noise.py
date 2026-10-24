"""FM-noise power spectral densities of phase records.

:func:`fm_noise_psd` differentiates a phase trajectory to the instantaneous
frequency and returns its Welch PSD; :func:`dsh_fm_noise_psd` does the same
for a delayed self-heterodyne differential phase and divides the
interferometer response back out (a :class:`DshFmNoisePsd`).  Both stay on
the input's device; the linewidth estimators in
:mod:`commkit.analysis.linewidth` summarize them.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from .. import _fft
from .._array import as_2d, restore_1d
from ..backend import ArrayType, dispatch
from ..spectral import welch_psd
from ._common import _resolve_nperseg

__all__ = ["DshFmNoisePsd", "dsh_fm_noise_psd", "fm_noise_psd"]


@dataclass(frozen=True, eq=False)
class DshFmNoisePsd:
    """Laser FM-noise PSD recovered from a DSH differential phase.

    All three stay on the input backend.

    Attributes
    ----------
    f : array_like
        One-sided frequency axis in Hz.
    S_f : array_like
        Laser FM-noise PSD in Hz²/Hz (NaN at masked bins), ``(nfreq,)`` or
        ``(C, nfreq)``.
    valid : array_like
        Boolean mask ``(nfreq,)`` of trustworthy bins (away from the
        interferometer's transfer-function notches).
    """

    f: Any
    S_f: Any
    valid: Any


def fm_noise_psd(
    phi: ArrayType,
    *,
    sampling_rate: float,
    nperseg: int | None = None,
    detrend: str | bool = "constant",
    bias_correction: bool = False,
) -> tuple[ArrayType, ArrayType]:
    r"""One-sided frequency-noise PSD S_f(f) [Hz²/Hz] from the phase.

    Differentiates the phase to the instantaneous frequency
    ``f_inst = diff(phi)/(2π·T)`` (Hz, ``T = 1/sampling_rate``) and estimates its one-sided PSD via
    Welch's method (``welch_psd``).  Distinct
    impairments occupy distinct regions of S_f(f):

    * **white-FM** (linewidth): flat plateau at ``S_f = Δν/π``,
    * **drift / flicker**: steep ``1/f`` (and steeper) rise at low ``f``,
    * **AWGN** angle noise: white phase noise -> ``S_f ∝ f²`` rise at high ``f``.

    Parameters
    ----------
    phi : array_like
        Unwrapped carrier phase (radians), ``(N,)`` or ``(C, N)``.
    sampling_rate : float
        Sampling rate of ``phi`` in Hz (the symbol rate for a per-symbol
        trajectory).
    nperseg : int, optional
        Welch segment length.  Defaults to ``min(N//8, 4096)`` (clipped ≥ 256).
    detrend : str or bool, default "constant"
        Per-segment detrend passed to Welch; ``"constant"`` removes the mean
        residual frequency offset.
    bias_correction : bool, default False
        Divide by ``sinc²(f/f_s)``, the first-difference roll-off of a
        *band-limited* frequency.  Leave it off for laser phase noise: white
        FM is not band-limited, and its sampled first difference is already
        exactly white (see Notes).

    Returns
    -------
    f : ndarray
        One-sided frequency axis in Hz (length ``nperseg//2 + 1``).
    S_f : ndarray
        Frequency-noise PSD in Hz²/Hz, shape ``(nfreq,)`` or ``(C, nfreq)``.
        Both stay on the input backend (compute layer - no host transfer;
        ``estimate_linewidth`` is the reporting layer).

    Notes
    -----
    **What the first difference measures.**  Sampling the phase at ``t = nT``
    and differencing gives the frequency *averaged* over each sample
    interval,

        φ[n] - φ[n-1] = 2π ∫_{(n-1)T}^{nT} f(t) dt,

    i.e. ``f(t)`` through a ``T``-long boxcar (response ``sinc(fT)``) and then
    sampled, which folds every band ``f + k·f_s`` onto ``f``:

        S_est(f) = Σ_k S_f(f + k·f_s) · sinc²((f + k·f_s)·T).

    Two cases follow:

    * **White FM (the Lorentzian linewidth)**: ``S_f = Δν/π`` at all
      frequencies, and ``Σ_k sinc²(x + k) = 1`` for every ``x``, so the folded
      copies exactly fill the droop and ``S_est = Δν/π`` flat up to Nyquist.
      Equivalently, a sampled Wiener phase has independent Gaussian
      increments.  No correction is needed, and dividing by ``sinc²`` would
      tilt the plateau up by as much as ``π²/4 ≈ 2.47`` at Nyquist.
    * **Band-limited FM** (well inside ``±f_s/2``, e.g. slow drift or a
      modulation tone): only ``k = 0`` contributes and the estimate droops by
      ``sinc²(fT)``, -3.9 dB at Nyquist.  ``bias_correction=True`` undoes it.

    Additive white *phase* noise ``σ²`` per sample (detection noise on the
    beat angle) is a third case: it is already discrete, and its difference
    gives the one-sided ``S_est(f) = (2σ²·f_s/π²)·sin²(πf/f_s)``, which is
    ``≈ (2σ²/f_s)·f²`` well below Nyquist.

    **Limitations.**

    * Frequency resolution is ``sampling_rate/nperseg``; noise processes slower than the
      segment length (drift, flicker below the first bin) alias into the
      lowest bins and are *not* resolved - extend the capture, not
      ``nperseg``, to see them.
    * Welch averaging trades variance for resolution: with ``K`` segments the
      per-bin relative std is ``≈ 1/√K``.  The default ``N//8`` with 50 %
      overlap gives ``K ≈ 15``.
    * ``detrend="constant"`` removes the *mean* frequency per segment; a
      residual frequency ramp within a segment still leaks into the lowest
      bins.
    """
    p, xp, _ = dispatch(phi)
    p2, was_1d = as_2d(p, name="phi")
    dt = 1.0 / float(sampling_rate)

    f_inst = xp.diff(p2.astype(xp.float64), axis=-1) / (2.0 * np.pi * dt)
    n = f_inst.shape[-1]
    nperseg = _resolve_nperseg(n, nperseg, cap=4096)

    f, S_f = welch_psd(
        f_inst,
        sampling_rate=float(sampling_rate),
        nperseg=nperseg,
        detrend=detrend,
        return_onesided=True,
    )
    if bias_correction:
        # S_f,est = S_f,true · sinc²(fT); undo the diff-differentiator droop.
        S_f = S_f / (xp.sinc(f * dt) ** 2)
    S_out = restore_1d(was_1d, S_f)

    return f, S_out


def _hann_overlap(lag: float, nperseg: int) -> float:
    """Normalized autocorrelation ``ρ_w`` of the Welch (periodic Hann) window.

    ``ρ_w(ℓ) = Σ_n w[n]·w[n+ℓ] / Σ_n w[n]²`` at a lag of ``ℓ`` samples,
    linearly interpolated for a fractional lag.  It is 1 at lag 0 and exactly
    0 from ``ℓ = nperseg`` on, where the shifted window no longer overlaps.
    """
    lag = abs(float(lag))
    if lag >= nperseg:
        return 0.0
    n = np.arange(nperseg)
    w = 0.5 - 0.5 * np.cos(2.0 * np.pi * n / nperseg)
    acf = _fft.irfft(np.abs(_fft.rfft(w, 2 * nperseg)) ** 2)[: nperseg + 1]
    acf[nperseg] = 0.0
    k = int(lag)
    frac = lag - k
    return float(((1.0 - frac) * acf[k] + frac * acf[k + 1]) / acf[0])


def _interferometer_response(
    f: ArrayType, *, delay: float, sampling_rate: float, nperseg: int
) -> ArrayType:
    r"""Mean Welch response of the beat FM PSD to the laser FM PSD.

    The beat frequency ``f_beat(t) = f(t) - f(t - τ_d)`` passes the laser
    frequency through ``|1 - e^{-j2πfτ_d}|² = 2 - 2 cos(2πfτ_d)``.  A Welch
    bin is not a point sample of that response: its expectation is the
    response convolved with the window's spectral kernel, which multiplies
    the oscillating term by the window autocorrelation at the delay,

        E[S_beat(f)] = (2 - 2 ρ_w(τ_d)·cos 2πfτ_d) · S_laser(f)

    for an ``S_laser`` that is smooth over one bin.  Long segments
    (``ρ_w → 1``) resolve the ``4 sin²(πfτ_d)`` fringes; segments no longer
    than the delay (``ρ_w = 0``) see the flat factor 2, because the two
    copies of the laser frequency inside one segment never overlap.
    """
    rho = _hann_overlap(float(delay) * float(sampling_rate), nperseg)
    _, xp, _ = dispatch(f)
    return 2.0 - 2.0 * rho * xp.cos(2.0 * np.pi * f * float(delay))


def _dsh_nperseg(n: int, delay_samples: float, nperseg: int | None) -> int:
    """Default DSH Welch segment: the longest power of two within the delay.

    Segments no longer than ``τ_d`` make the interferometer response flat
    (``ρ_w = 0``, see ``_interferometer_response``), so the estimate needs no
    notches and no accurate delay, and a long spool still yields many
    segments.  Clipped to ``[256, 16384]`` and to the record; an explicit
    ``nperseg`` is honoured (clipped to the record).
    """
    if nperseg is not None:
        return int(min(int(nperseg), n))
    within = 1 << int(np.floor(np.log2(max(delay_samples, 1.0))))
    return int(min(max(within, 256), 1 << 14, n))


def dsh_fm_noise_psd(
    delta_phi: ArrayType,
    *,
    sampling_rate: float,
    delay: float,
    nperseg: int | None = None,
    notch_guard: float = 0.1,
    bias_correction: bool = False,
) -> DshFmNoisePsd:
    r"""Laser FM-noise PSD from the DSH differential phase.

    **Physics.**  The beat carries the differential phase
    ``Δφ(t) = φ(t) - φ(t - τ_d)``, so its instantaneous frequency is the laser
    frequency minus its own delayed copy, ``f_beat(t) = f(t) - f(t - τ_d)``.
    That is a linear filter with response ``1 - e^{-j2πfτ_d}``:

        S_f,beat(f) = 4 sin²(πfτ_d) · S_f,laser(f) = (2 - 2 cos 2πfτ_d) · S_f,laser(f).

    The response has notches at ``f = k/τ_d`` (frequencies whose period fits
    the delay a whole number of times, so the two arms cancel) and peaks of 4
    halfway between.  For ``f ≪ 1/τ_d`` it is ``(2πfτ_d)²``: a frequency
    discriminator with a known gain, which is why the method also works when
    the delay is far *shorter* than the coherence time.

    **What a Welch estimate sees.**  A Welch bin averages the spectrum over
    the window's spectral kernel, which turns ``cos 2πfτ_d`` into
    ``ρ_w(τ_d)·cos 2πfτ_d``, with ``ρ_w`` the window's normalized
    autocorrelation at a lag of ``τ_d``:

        E[S_f,beat(f)] = (2 - 2 ρ_w(τ_d)·cos 2πfτ_d) · S_f,laser(f).

    This function divides by that response.  With segments much longer than
    the delay (``ρ_w ≈ 1``) it is the classic ``4 sin²`` deconvolution, and
    bins near the notches are unrecoverable (NaN, flagged in ``valid``).
    With segments no longer than the delay (``ρ_w = 0``, the default) the two
    delayed copies of the laser frequency never share a segment, the
    response is the flat factor 2 and every bin is valid: no notches, and the
    delay only needs to be known well enough to choose the segment.

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
        Welch segment length.  Defaults to the longest power of two within
        the delay (``ρ_w = 0``), clipped to ``[256, 16384]``.
    notch_guard : float, default 0.1
        Bins whose response is below ``4·notch_guard`` (``sin²(πfτ_d) <
        notch_guard`` for long segments) are masked (NaN).  The default keeps
        about 80 % of every response lobe; with ``ρ_w = 0`` nothing is masked.
    bias_correction : bool, default False
        Forwarded to ``fm_noise_psd``; leave it off for laser phase noise.

    Returns
    -------
    DshFmNoisePsd
        ``f``, ``S_f`` and ``valid``, all on the input backend (no host
        transfer) - this is the composable building block;
        ``estimate_linewidth(x, DshFmPsd(...))`` is the summary layer that
        returns host NumPy for reporting.

    See Also
    --------
    allan_deviation : feed it ``delta_phi/(2π·delay)`` - the interferometer's
        discriminator output - for the laser frequency stability at averaging
        times ``τ ≫ delay`` (shorter τ are low-passed by the τ_d window).

    Notes
    -----
    **Limitations.**

    * Additive detection noise on the beat adds white phase noise, an
      ``∝ sin²(πf/f_s)`` (≈ ``f²``) term that is *not* shaped by the
      interferometer; after the division it rises in every lobe.
      ``DshFmPsd`` fits it as a separate term instead of dividing it.
    * Near the notches (long segments only) the division amplifies the
      estimation noise by ``1/(4 sin²)``.
    * With long segments the result depends on τ_d through the notch
      positions: notch ``k`` moves by ``k·Δτ/τ_d`` of a lobe for a delay error
      ``Δτ``, so high lobes need τ_d to a fraction of ``1/k``.  The default
      segments avoid this.
    * The response formula assumes ``S_f,laser`` smooth over one bin; the
      lowest bins of a steep (flicker, drift) spectrum are approximate.
    """
    td = float(delay)
    if td <= 0.0:
        raise ValueError(f"delay={delay} must be positive (seconds).")

    dphi_arr, _, _ = dispatch(delta_phi)
    fs = float(sampling_rate)
    nps = _dsh_nperseg(dphi_arr.shape[-1] - 1, td * fs, nperseg)
    f, S_beat = fm_noise_psd(
        dphi_arr, sampling_rate=fs, nperseg=nps, bias_correction=bias_correction
    )
    _, xp, _ = dispatch(f)

    response = _interferometer_response(f, delay=td, sampling_rate=fs, nperseg=nps)
    valid = response >= 4.0 * float(notch_guard)
    S_laser = xp.where(valid, S_beat / xp.maximum(response, 1e-300), xp.nan)

    return DshFmNoisePsd(f=f, S_f=S_laser, valid=valid)
