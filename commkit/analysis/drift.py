"""Drift / phase-noise separation and residual frequency-wander metrics."""

import numpy as np

from ..backend import ArrayType, dispatch, to_device
from ..helpers import as_2d, restore_1d, to_report_scalar

__all__ = ["frequency_drift_metrics", "separate_drift_phase_noise"]


def separate_drift_phase_noise(
    phi: ArrayType,
    symbol_rate: float,
    *,
    cutoff: float,
    method: str = "butterworth",
    order: int = 4,
    debug_plot: bool = False,
) -> tuple[ArrayType, ArrayType]:
    r"""Split a phase trajectory into slow drift and fast phase-noise residual.

    Applies a **zero-phase** low-pass (default 4th-order Butterworth in
    second-order-sections form via ``sosfiltfilt``, numerically stable at the
    very low normalized cutoffs typical here) at ``cutoff`` to obtain the
    drift; the residual
    ``pn = phi - drift`` carries the phase noise + AWGN.  Zero-phase filtering
    avoids the group-delay bias of a causal filter and the spectral leakage of
    a boxcar moving average.

    The split is a modelling choice: too low a cutoff lets fast drift leak into
    ``pn`` (inflating the linewidth); too high a cutoff absorbs genuine
    low-frequency phase noise into ``drift``.  Because the single-symbol
    increment used downstream is itself a high-pass, the *increment-variance*
    linewidth is only weakly sensitive to this cutoff - but discard the filter
    edge transients (``edge_trim`` on the metric functions) regardless.

    Parameters
    ----------
    phi : array_like
        Unwrapped carrier phase (radians), ``(N,)`` or ``(C, N)``.  Sampled at
        the symbol rate (one value per symbol).
    symbol_rate : float
        Symbol rate in Baud; the effective sampling rate of ``phi``.
    cutoff : float
        Low-pass cutoff in Hz separating drift (below) from phase noise
        (above).  Must satisfy ``0 < cutoff < symbol_rate / 2``.
    method : {"butterworth", "savgol", "boxcar"}, default "butterworth"
        Low-pass implementation.  ``"savgol"`` is a polynomial (Savitzky-Golay)
        detrend; ``"boxcar"`` is the crude moving average (provided for
        comparison only).
    order : int, default 4
        Butterworth order, Savitzky-Golay polynomial order, or - reinterpreted
        - ignored for the boxcar.
    debug_plot : bool, default False
        If True, plot the phase trajectory with the drift overlaid
        (``carrier_phase_decomposition``).

    Returns
    -------
    drift_phase : array_like
        Low-frequency drift component, same shape/backend/dtype as ``phi``.
    pn_phase : array_like
        High-frequency phase-noise + AWGN residual ``phi - drift``.

    Notes
    -----
    **Limitations.**

    * The drift/phase-noise dichotomy is *spectral*, not physical: a laser's
      1/f (flicker) FM noise straddles any cutoff, so part of it lands in
      ``drift`` and part in ``pn`` no matter where the cutoff is placed.
      Quote the cutoff alongside any derived metric.
    * ``"savgol"`` sizes its window to ≈ one cutoff period, which is only a
      rough equivalent-noise-bandwidth match to the Butterworth response -
      treat its cutoff as approximate.
    * ``"boxcar"`` has -13 dB sidelobes (sinc response) that leak drift into
      ``pn``; it is provided for comparison only.
    * ``sosfiltfilt`` extends the signal internally, but the first/last
      ``~0.5·fs/cutoff`` samples of ``drift`` remain transient-contaminated -
      trim them via ``edge_trim`` in the downstream metric functions
      (``edge_trim ≈ 0.5·symbol_rate/cutoff``).
    """
    phi_arr, xp, sp = dispatch(phi)
    fs = float(symbol_rate)
    nyq = 0.5 * fs
    if not (0.0 < cutoff < nyq):
        raise ValueError(
            f"cutoff={cutoff} must lie in (0, symbol_rate/2={nyq}). "
            "phi is sampled at the symbol rate."
        )

    phi2, was_1d = as_2d(phi_arr, name="phi")
    in_dtype = phi2.dtype

    if method == "butterworth":
        # SOS form is numerically stable at the very low normalized cutoffs
        # typical here (cutoff ≪ symbol_rate => poles bunch near z=1).
        sos = sp.signal.butter(order, cutoff / nyq, btype="low", output="sos")
        if xp.__name__ == "cupy":
            sos = xp.asarray(sos)
        drift = sp.signal.sosfiltfilt(sos, phi2.astype(xp.float64), axis=-1)
    elif method == "savgol":
        # Window ≈ one cutoff period (odd, > polyorder).
        win = int(round(fs / cutoff)) | 1
        win = max(win, order + 2 + (order % 2 == 0))
        win = min(win, phi2.shape[-1] - (1 - phi2.shape[-1] % 2))
        drift = sp.signal.savgol_filter(phi2.astype(xp.float64), win, order, axis=-1)
    elif method == "boxcar":
        # uniform_filter1d is vectorized over channels on both backends and
        # its "nearest" edge mode avoids the zero-padding bias of
        # convolve(mode="same"), which drags the drift estimate toward zero
        # over the first/last window.
        w = max(1, int(round(fs / cutoff)))
        drift = sp.ndimage.uniform_filter1d(
            phi2.astype(xp.float64), w, axis=-1, mode="nearest"
        )
    else:
        raise ValueError(f"Unknown method {method!r}.")

    pn = phi2 - drift
    drift, pn = restore_1d(was_1d, drift, pn)

    drift = drift.astype(in_dtype, copy=False)
    pn = pn.astype(in_dtype, copy=False)

    if debug_plot:
        from .. import plotting as _plotting

        _plotting.plot_carrier_phase_decomposition(
            phi, drift, symbol_rate=symbol_rate, show=True
        )

    return drift, pn


def frequency_drift_metrics(
    drift_phase: ArrayType,
    symbol_rate: float,
    *,
    edge_trim: int = 0,
    amp_ref: float | None = None,
    debug_plot: bool = False,
) -> dict[str, float | np.ndarray]:
    r"""Residual frequency-wander statistics from a smoothed phase ramp.

    The instantaneous residual frequency offset is the phase slope
    ``df = diff(drift) / (2π T_sym)`` in Hz.  Report the std (typical wander)
    and the peak-to-peak (worst-case spin the CPR must follow).

    Relate to the BPS tracking limit: a residual ``δf`` rotates the phase by
    ``2π·δf·T_sym`` per symbol, so over a window of ``K`` symbols the
    intra-window rotation must stay below the QAM quarter-symmetry ``π/4`` ->
    ``δf_max ≈ 1/(8·K·T_sym)``.  A larger BPS window tracks *less* drift.

    Parameters
    ----------
    drift_phase : array_like
        Drift phase component (radians), ``(N,)`` or ``(C, N)``.
    symbol_rate : float
        Symbol rate in Baud.
    edge_trim : int, default 0
        Number of samples to discard from each end before differencing
        (removes low-pass filter transients).
    amp_ref : float, optional
        Reference wander amplitude (Hz) drawn as ``±amp_ref`` guides when
        ``debug_plot=True`` (e.g. an injected amplitude in a simulation).
    debug_plot : bool, default False
        If True, plot the residual frequency vs time
        (``frequency_drift``).

    Returns
    -------
    dict
        ``{'df', 'std', 'pp', 'max_abs'}``.  ``df`` is the
        per-symbol residual frequency array; the rest are floats (SISO) or
        per-channel arrays (MIMO).

    Notes
    -----
    The first difference under-reads a spectral component at ``f`` by
    ``sinc(f/R)`` relative to a true derivative.  Because ``drift_phase`` is
    low-passed (``cutoff ≪ R``), this bias is negligible here - it only
    matters when differencing broadband phase (see ``fm_noise_psd``, which
    corrects for it).
    """
    d, xp, _ = dispatch(drift_phase)
    d2, was_1d = as_2d(d, name="drift_phase")
    if edge_trim > 0:
        d2 = d2[:, edge_trim:-edge_trim]

    t_sym = 1.0 / float(symbol_rate)
    df = xp.diff(d2.astype(xp.float64), axis=-1) / (2.0 * np.pi * t_sym)

    # One D2H transfer for all three summaries instead of three syncs.
    stats = to_device(
        xp.stack(
            [
                xp.std(df, axis=-1),
                xp.max(df, axis=-1) - xp.min(df, axis=-1),
                xp.max(xp.abs(df), axis=-1),
            ]
        ),
        "cpu",
    )
    std, pp, max_abs = stats[0], stats[1], stats[2]

    df_out = restore_1d(was_1d, df)

    if debug_plot:
        from .. import plotting as _plotting

        _plotting.plot_frequency_drift(
            df_out, symbol_rate=symbol_rate, amp_ref=amp_ref, show=True
        )

    return {
        "df": df_out,
        "std": to_report_scalar(std),
        "pp": to_report_scalar(pp),
        "max_abs": to_report_scalar(max_abs),
    }
