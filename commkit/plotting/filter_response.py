"""Filter responses: :func:`plot_filter_response`.

Plots the impulse, step, magnitude, phase and group-delay responses of a
filter in any of the forms the library designs: FIR taps
(``RRC(0.1).taps(sps)``, :func:`~commkit.filtering.fir_taps`, ...), a
``(b, a)`` transfer function, or second-order sections
(:func:`commkit.filtering.butterworth_sos` and the other ``*_sos`` designs).
The responses come from :func:`~commkit.filtering.filter_response`,
:func:`~commkit.filtering.impulse_response` and
:func:`~commkit.filtering.step_response`.

Units: seconds and hertz with a ``sampling_rate``; symbol periods and
symbol rates with ``sps`` alone; samples and cycles per sample with
neither.

Time reference: an FIR filter is drawn as :func:`~commkit.filtering.fir_filter`
applies it.  By default that is aligned on tap ``(L - 1) // 2``: the time axes
start at ``-(L - 1) // 2`` and the phase and group delay are measured from that
tap, so a linear-phase filter of odd length has zero group delay.
``causal=True`` gives the causal convolution, with its delay of
``(L - 1) // 2`` samples.  IIR filters are always causal.

``relative=True`` removes the taps' arbitrary scale for comparisons: the
impulse response relative to its peak magnitude, the step response and the
magnitude relative to the filter's peak gain (0 dB at the maximum of
``|H(f)|`` on the plotted grid, which also holds for high-pass and
band-pass filters, whose DC gain is zero). Phase and group delay do not
depend on a positive gain and are drawn unchanged.

The phase is continuous: the jumps of π where the amplitude changes sign
(at the stopband zeros of a linear-phase filter) are not drawn, as the
group delay, ``-dφ/dω``, does not see them either.

Autoscaled frequency panels span at least 0.1 dB, 0.01 rad or 0.01 samples,
so a flat response (a linear-phase filter's group delay, an all-pass
magnitude) draws as a flat line instead of its rounding.
"""

from typing import Any

import matplotlib.ticker as ticker
import numpy as np

from .._array import require_finite, require_int
from ..backend import dispatch, to_device
from ..filtering import filter_response, impulse_response, step_response
from .theme import _finish, _guide, _panel_grid, _select, _set_eng_formatter

__all__ = ["plot_filter_response"]

_GUIDE_3DB = "commkit-3db-guide"
_GUIDE_NYQUIST = "commkit-nyquist-guide"
_GUIDE_FINAL = "commkit-final-value-guide"
_SYMBOL_INSTANTS = "commkit-symbol-instants"


def plot_filter_response(
    system: Any,
    *,
    sps: float | None = None,
    sampling_rate: float | None = None,
    causal: bool = False,
    ax: Any | None = None,
    num_impulse_samples: int | None = None,
    quantity: str | None = None,
    nfft: int = 2048,
    frequencies: Any | None = None,
    relative: bool = False,
    annotations: bool = False,
    label: str | None = None,
    legend: bool | None = None,
    title: str | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    show: bool = False,
) -> tuple[Any, Any] | None:
    """Plot a digital filter's response.

    Parameters
    ----------
    system : array_like or (b, a) tuple
        FIR taps, second-order sections, or transfer coefficients.
    sps : float, optional
        Samples per symbol.  Without a sampling rate, time and group delay
        are in symbol periods and frequency in symbol rates; with one, it
        only places the symbol guides of ``annotations``.  ``None`` counts
        samples and cycles per sample.
    sampling_rate : float, optional
        Sampling rate in Hz, for time in seconds and frequency in hertz.
    causal : bool, default False
        Draw an FIR filter as ``fir_filter(..., causal=causal)`` applies it:
        aligned on tap ``(L - 1) // 2`` by default, the time axes starting
        at ``-(L - 1) // 2`` and the phase and group delay measured from
        that tap; ``True`` for the causal convolution from tap 0.  IIR
        filters are always causal.
    ax : matplotlib.axes.Axes or array_like of Axes, optional
        One axis for a selected quantity; four for the default dashboard.
    num_impulse_samples : int, optional
        Number of samples for impulse and step plots. ``None`` derives the
        length from the filter: all taps for an FIR impulse response, the
        taps plus a settling margin for an FIR step response, and the decay
        of the slowest pole for an IIR filter. An explicit integer is always
        honoured, counted from tap 0.
    quantity : {"impulse", "step", "magnitude", "phase", "group_delay"}, optional
        A single response to plot. ``None`` plots the four-panel dashboard.
    nfft : int, default 2048
        Number of points in the frequency response, from DC to Nyquist for
        real coefficients and over ``[-fs/2, fs/2)`` for complex ones (see
        :func:`~commkit.filtering.filter_response`).
    frequencies : array_like, optional
        Frequency grid on the system's device, in Hz with a sampling rate,
        otherwise in cycles per sample. ``None`` uses the uniform
        ``nfft``-point grid.
    relative : bool, default False
        Draw the impulse response relative to its peak magnitude, and the
        step response and magnitude relative to the peak gain, the maximum
        of ``|H(f)|`` on the frequency grid, so that the magnitude peaks at
        0 dB. Phase and group delay are unchanged. An all-zero filter stays
        zero.
    annotations : bool, default False
        Draw guides, dashed in the theme's accent color and once per place
        on an axis: on the magnitude, a line 3 dB below its peak (at -3 dB
        with ``relative``, shared by the overlays) and, with ``sps``, one at
        half the symbol rate, the Nyquist frequency of the symbols, about
        which a Nyquist pulse's spectrum is symmetric (an RC is at -6 dB
        there, an RRC at -3 dB); on the impulse response, with an integer
        ``sps`` above 1, markers in the curve's color at the symbol instants,
        every ``sps`` samples from the centre tap of an FIR filter (from tap
        0 for an IIR filter), where a Nyquist pulse has its zero crossings;
        on the step response, a line at its final value. ``False`` leaves
        only the responses.
    label : str, optional
        Label for the plotted response, for comparisons on the same axes.
    legend : bool, optional
        Show (``True``) or hide (``False``) the legend; ``None`` keeps the
        default, a legend only where the plot labels its parts.
    title : str, optional
        Title for a single selected quantity; ``None`` uses its default title.
    xlim, ylim : tuple of float, optional
        Limits for a single selected quantity, in the displayed axis units.
    show : bool, default False
        Show the figure and return ``None``.

    Returns
    -------
    (fig, ax) or None
        Figure and one axis for a selected quantity, or four axes for the
        default dashboard; ``None`` with ``show=True``.
    """
    _select(
        quantity,
        ("impulse", "step", "magnitude", "phase", "group_delay"),
        title=title,
        xlim=xlim,
        ylim=ylim,
    )
    if sps is not None:
        require_finite(sps, "plot_filter_response(): sps", low=0.0)
    if num_impulse_samples is not None:
        require_int(num_impulse_samples, "num_impulse_samples", low=1)
    if quantity is None:
        selected: tuple[str, ...] = ("impulse", "magnitude", "phase", "group_delay")
        fig, grid = _panel_grid(ax, 2, 2)
    else:
        selected = (quantity,)
        fig, grid = _panel_grid(ax, 1, 1)
    axes = list(grid.flat)

    if isinstance(system, tuple):
        is_fir = dispatch(system[1])[0].size == 1
        num_taps = dispatch(system[0])[0].size
    else:
        coefficients = dispatch(system)[0]
        is_fir = coefficients.ndim == 1
        num_taps = coefficients.shape[-1]
    centre = (num_taps - 1) // 2 if is_fir else 0  # the pulse's centre tap
    origin = 0 if causal or not is_fir else centre  # sample drawn at t = 0
    # Seconds, symbol periods or samples per sample; hertz, symbol rates or
    # cycles per sample per cycle per sample.
    time_unit = sampling_rate if sampling_rate is not None else (sps or 1.0)
    freq_scale = 1.0 if sampling_rate is not None else (sps or 1.0)
    time_label, freq_label = (
        ("Time [s]", "Frequency [Hz]")
        if sampling_rate is not None
        else ("Time [Symbol Periods]", "Frequency [Symbol Rates]")
        if sps is not None
        else ("Time [Samples]", "Frequency [Cycles/Sample]")
    )

    response = None
    gain = 1.0  # divides the step response and the magnitude
    if any(q in selected for q in ("magnitude", "phase", "group_delay")) or (
        relative and "step" in selected
    ):
        response = filter_response(
            system, nfft=nfft, sampling_rate=sampling_rate, frequencies=frequencies
        )
        if relative:
            xp = dispatch(response.response)[1]
            peak = float(xp.max(xp.abs(response.response)))
            gain = peak if peak > 0 else 1.0
    for q, axis in zip(selected, axes, strict=True):
        if q in ("impulse", "step"):
            y = (
                impulse_response(system, num_samples=num_impulse_samples)
                if q == "impulse"
                else step_response(system, num_samples=num_impulse_samples)
            )
            count = y.size
            n = np.arange(count)
            t = (n - origin) / time_unit
            y = to_device(y, "cpu")
            if relative and q == "impulse":
                peak = float(np.max(np.abs(y)))
                y = y / peak if peak > 0 else y
            elif q == "step":
                y = y / gain
            parts = (
                [(y.real, f"{label} I" if label else "Real"),
                 (y.imag, f"{label} Q" if label else "Imag")]
                if np.iscomplexobj(y)
                else [(y, label)]
            )  # fmt: skip
            for part, part_label in parts:
                (line,) = axis.plot(t, part, label=part_label)
                if annotations and q == "impulse":
                    _mark_symbol_instants(axis, t, part, n, centre, sps, line)
                elif annotations and q == "step":
                    _guide(axis, float(part[-1]), gid=_GUIDE_FINAL)
            axis.set_xlabel(time_label)
            axis.set_ylabel("Relative Amplitude" if relative else "Amplitude")
            if sampling_rate is not None:
                _set_eng_formatter(axis, "x")
            elif count > 1 and t[-1] - t[0] >= 2:
                axis.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            default_title = "Impulse Response" if q == "impulse" else "Step Response"
        else:
            assert response is not None
            f_host = to_device(response.frequencies, "cpu")
            h = to_device(response.response, "cpu")
            # Cycles per sample, for the phase of the time reference.
            f_cycles = f_host / sampling_rate if sampling_rate is not None else f_host
            if q == "magnitude":
                y = 20 * np.log10(np.abs(h) / gain + 1e-12)
                ylabel, default_title = (
                    "Relative Magnitude [dB]" if relative else "Magnitude [dB]",
                    "Frequency Response (Magnitude)",
                )
            elif q == "phase":
                aligned = h * np.exp(2j * np.pi * origin * f_cycles)
                # Unwrapping twice the phase and halving it removes the
                # jumps of π where the amplitude changes sign (the stopband
                # zeros of a linear-phase filter), which np.unwrap would
                # turn up or down by rounding; true wraps of 2π go as well.
                y = np.unwrap(2 * np.angle(aligned).astype(np.float64)) / 2
                ylabel, default_title = "Phase [rad]", "Frequency Response (Phase)"
            else:
                delay = to_device(response.group_delay, "cpu").astype(np.float64)
                # Seconds with a sampling rate, otherwise samples.
                y = delay - origin / (sampling_rate or 1.0)
                if sampling_rate is None:
                    y = y / (sps or 1.0)
                ylabel = (
                    "Group Delay [s]"
                    if sampling_rate is not None
                    else "Group Delay [Symbol Periods]"
                    if sps is not None
                    else "Group Delay [Samples]"
                )
                default_title = "Group Delay"
            x = f_host * freq_scale
            axis.plot(x, y, label=label)
            # One sample in the displayed delay unit.
            sample = 1 / (sampling_rate if sampling_rate is not None else sps or 1.0)
            _span_floor(
                axis, x, y, {"magnitude": 0.1, "phase": 0.01}.get(q, 0.01 * sample)
            )
            if annotations and q == "magnitude":
                _guide(axis, float(np.max(y)) - 3.0, gid=_GUIDE_3DB)
                if sps is not None:
                    half_rate = (
                        sampling_rate / (2 * sps) if sampling_rate is not None else 0.5
                    )
                    for edge in (-half_rate, half_rate):
                        if x[0] <= edge <= x[-1]:
                            _guide(axis, edge, gid=_GUIDE_NYQUIST, vertical=True)
            if quantity is None:
                axis.set_xlim(x[0], x[-1])
            axis.set_xlabel(freq_label)
            axis.set_ylabel(ylabel)
            if sampling_rate is not None:
                _set_eng_formatter(axis, "x")
                if q == "group_delay":
                    _set_eng_formatter(axis, "y")
        axis.set_title(title if title is not None else default_title)
        if legend is None and q in ("impulse", "step") and np.iscomplexobj(y):
            axis.legend()
    return _finish(
        (fig, tuple(axes) if quantity is None else axes[0]),
        show,
        xlim=xlim,
        ylim=ylim,
        legend=legend,
    )


def _mark_symbol_instants(
    axis: Any,
    t: np.ndarray,
    y: np.ndarray,
    n: np.ndarray,
    centre: int,
    sps: float | None,
    line: Any,
) -> None:
    """Markers on the samples ``centre + k * sps`` of an impulse response,
    in ``line``'s color; nothing unless ``sps`` is an integer above 1."""
    if sps is None or sps <= 1 or float(sps) != int(sps):
        return
    on = (n - centre) % int(sps) == 0
    axis.plot(
        t[on],
        y[on],
        "o",
        color=line.get_color(),
        linestyle="none",
        gid=_SYMBOL_INSTANTS,
    )


def _span_floor(axis: Any, x: np.ndarray, y: np.ndarray, floor: float) -> None:
    """Widen the data limits of ``axis`` to at least ``floor`` around ``y``.

    Autoscaling a curve that is flat up to rounding (a constant group delay,
    an all-pass magnitude) would zoom into the rounding.  The extra extent
    goes into the data limits, so it holds through later overlays and
    redraws, and an explicit ``ylim`` still wins.
    """
    finite = y[np.isfinite(y)]
    if not finite.size:
        return
    lo, hi = float(finite.min()), float(finite.max())
    if hi - lo >= floor:
        return
    centre = (lo + hi) / 2
    axis.update_datalim(
        [(float(x[0]), centre - floor / 2), (float(x[0]), centre + floor / 2)],
        updatex=False,
    )
    axis.autoscale_view(scalex=False)
