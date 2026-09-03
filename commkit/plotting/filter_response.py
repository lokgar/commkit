"""
Filter design visualization.

Plots the impulse/frequency/group-delay response of a filter, in any of the
representations produced elsewhere in the package: FIR taps
(:mod:`commkit.filtering`'s ``rrc_taps``/``lowpass_taps``/...), an ``(b, a)``
IIR transfer function, or the second-order-sections (``sos``) form produced
by :func:`commkit.filtering.iir_zero_phase_filter`. This module visualizes
filter-design artifacts only - it has nothing to do with
:mod:`commkit.smoothing`'s diagnostic smoothers, which have no meaningful
frequency response to plot.
"""

from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from ..backend import dispatch, to_device
from ..logger import logger
from .theme import _grid_figsize, _set_eng_formatter


def plot_filter_response(
    system: Any,
    sps: float = 1.0,
    sampling_rate: float | None = None,
    ax: Any | None = None,
    n_impulse: int = 500,
    show: bool = False,
) -> tuple[Any, tuple[Any, Any, Any, Any]] | None:
    """
    Plots the impulse, magnitude, phase, and group-delay response of a filter.

    A 2x2 diagnostic panel - ``(impulse, magnitude), (phase, group_delay)`` -
    covering the standard filter-design characterization set. Accepts any of
    three filter representations via ``system``:

    * a 1-D array - FIR taps (``b = system``, ``a = [1.0]``); the impulse
      response is ``system`` itself, exact and finite.
    * a 2-D array with trailing dimension 6 - second-order sections (SOS),
      the representation :func:`commkit.filtering.iir_zero_phase_filter`
      designs internally.
    * a length-2 ``(b, a)`` tuple/list - a general IIR transfer function.

    For the SOS/IIR forms the true impulse response is infinite, so it's
    approximated by filtering a unit impulse of length ``n_impulse`` and
    truncating - a standard diagnostic approximation, not an exact result.

    Parameters
    ----------
    system : array_like or (b, a) tuple
        Filter representation, see above.
    sps : float, default 1.0
        Samples per symbol for time/group-delay axis normalization, used
        when ``sampling_rate`` is not given.
    sampling_rate : float, optional
        Sampling rate in Hz. When given, the impulse/group-delay axes are in
        seconds and the frequency axes are in Hz with automatic SI scaling
        (Hz/kHz/MHz/GHz). When omitted, axes stay in symbol-period /
        cycles-per-sample units (backward-compatible with FIR-only usage).
    ax : array_like, optional
        Exactly 4 axes to plot on (a 2x2 array or a flat sequence of 4).
        Any other count logs a warning and a fresh 2x2 figure is created
        instead.
    n_impulse : int, default 500
        Impulse-response truncation length for SOS/IIR input. Ignored for
        FIR (1-D array) input, whose impulse response is exact.
    show : bool, default False
        If True, calls ``plt.show()``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    axes : tuple of matplotlib.axes.Axes
        The (impulse, magnitude, phase, group_delay) axes.
    """
    # --- Detect the filter representation and dispatch on its backend ---
    if isinstance(system, (tuple, list)) and len(system) == 2:
        kind = "ba"
        b_raw, a_raw = system
        b, xp, sp = dispatch(b_raw)
        a = xp.asarray(a_raw)
    else:
        arr, xp, sp = dispatch(system)
        if arr.ndim == 2 and arr.shape[-1] == 6:
            kind = "sos"
            sos = arr
        else:
            kind = "fir"
            taps = arr

    if ax is None:
        logger.debug("Generating filter response plot.")
        fig, raw_axes = plt.subplots(2, 2, figsize=_grid_figsize(2, 2))
        ax1, ax2, ax3, ax4 = raw_axes.flat
    else:
        flat_ax = list(np.asarray(ax).flat) if hasattr(ax, "__len__") else [ax]
        if len(flat_ax) == 4:
            fig = flat_ax[0].figure
            ax1, ax2, ax3, ax4 = flat_ax
        else:
            logger.warning("filter_response requires 4 axes. Creating new figure.")
            fig, raw_axes = plt.subplots(2, 2, figsize=_grid_figsize(2, 2))
            ax1, ax2, ax3, ax4 = raw_axes.flat

    # --- 1. Impulse response ---
    if kind == "fir":
        h_imp = taps
        num_imp = len(h_imp)
        n_axis = xp.arange(num_imp) - (num_imp - 1) / 2  # centered (linear-phase FIR)
    else:
        impulse = xp.zeros(n_impulse, dtype=xp.float64)
        impulse[0] = 1.0
        h_imp = (
            sp.signal.sosfilt(sos, impulse)
            if kind == "sos"
            else sp.signal.lfilter(b, a, impulse)
        )
        n_axis = xp.arange(n_impulse)  # causal, starts at 0

    if sampling_rate is not None:
        t = n_axis / float(sampling_rate)
    else:
        t = n_axis / sps

    t_cpu = to_device(t, "cpu")
    h_imp_cpu = to_device(h_imp, "cpu")

    if xp.iscomplexobj(h_imp):
        ax1.plot(t_cpu, h_imp_cpu.real, label="Real", color="C0")
        ax1.plot(t_cpu, h_imp_cpu.imag, label="Imag", color="C1")
        ax1.legend()
    else:
        ax1.plot(t_cpu, h_imp_cpu, color="C0")

    ax1.set_title("Impulse Response")
    if sampling_rate is not None:
        ax1.set_xlabel("Time [s]")
    else:
        ax1.set_xlabel("Time [Symbol Periods]")
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))

        def t_formatter(x, pos):
            if np.isclose(x, 0):
                return "0"
            return f"{int(x)}T" if float(x).is_integer() else f"{x}T"

        ax1.xaxis.set_major_formatter(ticker.FuncFormatter(t_formatter))
    ax1.set_ylabel("Amplitude")

    # --- 2/3. Frequency response (magnitude + phase) ---
    if kind == "fir":
        w, h_resp = sp.signal.freqz(taps, worN=2048)
    elif kind == "sos":
        w, h_resp = sp.signal.sosfreqz(sos, worN=2048)
    else:
        w, h_resp = sp.signal.freqz(b, a, worN=2048)

    mag = 20 * xp.log10(xp.abs(h_resp) + 1e-12)
    angles = xp.unwrap(xp.angle(h_resp))

    if sampling_rate is not None:
        freqs = w / (2 * xp.pi) * float(sampling_rate)
        freqs_disp = to_device(freqs, "cpu")
        freq_label = "Frequency [Hz]"
    else:
        freqs_disp = to_device(w / (2 * xp.pi), "cpu")  # cycles/sample
        freq_label = "Frequency [Cycles/Sample]"

    mag_cpu = to_device(mag, "cpu")
    angles_cpu = to_device(angles, "cpu")

    ax2.plot(freqs_disp, mag_cpu, color="C2")
    ax2.set_ylabel("Magnitude [dB]")
    ax2.set_title("Frequency Response (Magnitude)")
    ax2.set_xlabel(freq_label)
    ax2.set_xlim(freqs_disp[0], freqs_disp[-1])
    if sampling_rate is not None:
        _set_eng_formatter(ax2, "x", "Hz")

    ax3.plot(freqs_disp, angles_cpu, color="C3")
    ax3.set_ylabel("Phase [rad]")
    ax3.set_title("Frequency Response (Phase)")
    ax3.set_xlabel(freq_label)
    ax3.set_xlim(freqs_disp[0], freqs_disp[-1])
    if sampling_rate is not None:
        _set_eng_formatter(ax3, "x", "Hz")

    # --- 4. Group delay ---
    if kind == "fir":
        _, gd = sp.signal.group_delay((taps, [1.0]), w=w)
    elif kind == "sos":
        b_tf, a_tf = sp.signal.sos2tf(sos)
        _, gd = sp.signal.group_delay((b_tf, a_tf), w=w)
    else:
        _, gd = sp.signal.group_delay((b, a), w=w)

    if sampling_rate is not None:
        gd_disp = to_device(gd / float(sampling_rate), "cpu")
        gd_label = "Group Delay [s]"
    else:
        gd_disp = to_device(gd / sps, "cpu")
        gd_label = "Group Delay [Symbol Periods]"

    ax4.plot(freqs_disp, gd_disp, color="C4")
    ax4.set_ylabel(gd_label)
    ax4.set_title("Group Delay")
    ax4.set_xlabel(freq_label)
    ax4.set_xlim(freqs_disp[0], freqs_disp[-1])
    if sampling_rate is not None:
        _set_eng_formatter(ax4, "x", "Hz")

    if show:
        plt.show()
        return None
    return fig, (ax1, ax2, ax3, ax4)
