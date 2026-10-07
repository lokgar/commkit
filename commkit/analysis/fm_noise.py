"""FM-noise power spectral densities of phase records.

``fm_noise_psd`` differentiates a phase trajectory to the instantaneous
frequency and returns its Welch PSD; ``dsh_fm_noise_psd`` does the same for a
delayed self-heterodyne differential phase and divides the interferometer
response back out.  Both stay on the input backend; the linewidth estimators
in :mod:`commkit.analysis.linewidth` summarize them.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from .._array import as_2d, restore_1d
from ..backend import ArrayType, dispatch
from ..logger import logger
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
    symbol_rate: float,
    nperseg: int | None = None,
    detrend: str | bool = "constant",
    bias_correction: bool = True,
) -> tuple[ArrayType, ArrayType]:
    r"""One-sided frequency-noise PSD S_f(f) [Hz²/Hz] from the phase.

    Differentiates the phase to the instantaneous frequency
    ``f_inst = diff(phi)/(2π·T_sym)`` (Hz) and estimates its one-sided PSD via
    Welch's method (``welch_psd``).  Distinct
    impairments occupy distinct regions of S_f(f):

    * **white-FM** (linewidth): flat plateau at ``S_f = Δν/π``,
    * **drift / flicker**: steep ``1/f`` (and steeper) rise at low ``f``,
    * **AWGN** angle noise: white phase noise -> ``S_f ∝ f²`` rise at high ``f``.

    Parameters
    ----------
    phi : array_like
        Unwrapped carrier phase (radians), ``(N,)`` or ``(C, N)``.
    symbol_rate : float
        Symbol rate in Baud (sampling rate of ``phi``).
    nperseg : int, optional
        Welch segment length.  Defaults to ``min(N//8, 4096)`` (clipped ≥ 256).
    detrend : str or bool, default "constant"
        Per-segment detrend passed to Welch; ``"constant"`` removes the mean
        residual frequency offset.
    bias_correction : bool, default True
        Undo the first-difference roll-off (see Notes).

    Returns
    -------
    f : array_like
        One-sided frequency axis in Hz (length ``nperseg//2 + 1``).
    S_f : array_like
        Frequency-noise PSD in Hz²/Hz, shape ``(nfreq,)`` or ``(C, nfreq)``.
        Both stay on the input backend (compute layer - no host transfer;
        ``estimate_linewidth`` is the reporting layer).

    Notes
    -----
    The first difference is not an ideal differentiator: its magnitude
    response is ``|2 sin(πfT)|`` versus the ideal ``2πfT``, so the raw
    estimate is ``S_f,true(f) · sinc²(fT)`` - a -3.9 dB droop at Nyquist
    (``R/2``).  With ``bias_correction=True`` the PSD is divided by
    ``sinc²(fT)`` so the white-FM plateau and the AWGN ``f²`` tail keep their
    analytic levels all the way to Nyquist.

    **Limitations.**

    * Frequency resolution is ``R/nperseg``; noise processes slower than the
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
    t_sym = 1.0 / float(symbol_rate)

    f_inst = xp.diff(p2.astype(xp.float64), axis=-1) / (2.0 * np.pi * t_sym)
    n = f_inst.shape[-1]
    nperseg = _resolve_nperseg(n, nperseg, cap=4096)

    f, S_f = welch_psd(
        f_inst,
        sampling_rate=float(symbol_rate),
        nperseg=nperseg,
        detrend=detrend,
        return_onesided=True,
    )
    if bias_correction:
        # S_f,est = S_f,true · sinc²(fT); undo the diff-differentiator droop.
        S_f = S_f / (xp.sinc(f * t_sym) ** 2)
    S_out = restore_1d(was_1d, S_f)

    return f, S_out


def dsh_fm_noise_psd(
    delta_phi: ArrayType,
    *,
    sampling_rate: float,
    delay: float,
    nperseg: int | None = None,
    notch_guard: float = 0.1,
    bias_correction: bool = True,
) -> DshFmNoisePsd:
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

    * Even inside the guard band the deconvolution *amplifies* estimation
      noise by ``1/(4 sin²)`` - near-notch bins are noisier than mid-lobe
      bins.  Prefer median-based summaries (``DshFmPsd`` uses one).
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

    dphi_arr, _, _ = dispatch(delta_phi)
    f, S_beat = fm_noise_psd(
        dphi_arr,
        symbol_rate=float(sampling_rate),
        nperseg=nperseg,
        bias_correction=bias_correction,
    )
    _, xp, _ = dispatch(f)

    # The deconvolution samples the response at bin centers; if a Welch bin
    # spans a sizable fraction of the response period 1/τ_d (long decoherence
    # spools), each bin *averages* across lobes and notches and the result is
    # biased low.  Require ≥ 4 bins per period; recommend ≥ 8.
    # f[1] = f_s / nperseg by construction - reproduce it from the resolver
    # instead of syncing a scalar back off the device frequency axis.
    bin_hz = float(sampling_rate) / _resolve_nperseg(
        dphi_arr.shape[-1] - 1, nperseg, cap=4096
    )
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

    return DshFmNoisePsd(f=f, S_f=S_laser, valid=valid)
