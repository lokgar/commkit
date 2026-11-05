"""Filter responses: :func:`plot_filter_response`.

Plots the impulse, step, magnitude, phase and group-delay responses of a
filter in any of the forms the library designs: FIR taps
(``RRC(0.1).taps(sps)``, :func:`~commkit.filtering.fir_taps`, ...), a
``(b, a)`` transfer function, or second-order sections
(:func:`commkit.filtering.butterworth_sos` and the other ``*_sos`` designs).
The responses come from :func:`~commkit.filtering.filter_response`,
:func:`~commkit.filtering.impulse_response` and
:func:`~commkit.filtering.step_response`.
"""

from typing import Any

import matplotlib.ticker as ticker
import numpy as np

from ..backend import dispatch, to_device
from ..filtering import filter_response, impulse_response, step_response
from .theme import _finish, _panel_grid, _select, _set_eng_formatter

__all__ = ["plot_filter_response"]


def plot_filter_response(
    system: Any,
    *,
    sps: float = 1.0,
    sampling_rate: float | None = None,
    ax: Any | None = None,
    num_impulse_samples: int = 500,
    quantity: str | None = None,
    nfft: int = 2048,
    frequencies: Any | None = None,
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
    sps : float, default 1.0
        Samples per symbol, used for time and group delay without a sampling rate.
    sampling_rate : float, optional
        Sampling rate in Hz. Without it, frequency is in cycles per sample.
    ax : matplotlib.axes.Axes or array_like of Axes, optional
        One axis for a selected quantity; four for the default dashboard.
    num_impulse_samples : int, default 500
        Number of samples for IIR impulse and step responses. The default
        dashboard displays all FIR taps as its impulse response.
    quantity : {"impulse", "step", "magnitude", "phase", "group_delay"}, optional
        A single response to plot. ``None`` plots the four-panel dashboard.
    nfft : int, default 2048
        Number of points in the frequency response, from DC to Nyquist for
        real coefficients and over ``[-fs/2, fs/2)`` for complex ones (see
        :func:`~commkit.filtering.filter_response`).
    frequencies : array_like, optional
        Frequency grid on the system's device, in Hz or cycles per sample.
        ``None`` uses the uniform ``nfft``-point grid.
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
    if sps <= 0:
        raise ValueError("sps must be positive")
    if quantity is None:
        selected: tuple[str, ...] = ("impulse", "magnitude", "phase", "group_delay")
        fig, grid = _panel_grid(ax, 2, 2)
    else:
        selected = (quantity,)
        fig, grid = _panel_grid(ax, 1, 1)
    axes = list(grid.flat)

    response = None
    if any(q in selected for q in ("magnitude", "phase", "group_delay")):
        response = filter_response(
            system, nfft=nfft, sampling_rate=sampling_rate, frequencies=frequencies
        )
    raw = system[0] if isinstance(system, tuple) else system
    coeff, _, _ = dispatch(raw)
    is_fir = not isinstance(system, tuple) and coeff.ndim == 1
    for q, axis in zip(selected, axes, strict=True):
        if q in ("impulse", "step"):
            count = (
                coeff.size
                if q == "impulse" and is_fir and quantity is None
                else num_impulse_samples
            )
            y = (
                impulse_response(system, num_samples=count)
                if q == "impulse"
                else step_response(system, num_samples=count)
            )
            t = np.arange(count, dtype=float)
            if q == "impulse" and is_fir and quantity is None:
                t -= (count - 1) / 2
            t /= sampling_rate if sampling_rate is not None else sps
            y = to_device(y, "cpu")
            if np.iscomplexobj(y):
                axis.plot(t, y.real, label=f"{label} I" if label else "Real")
                axis.plot(t, y.imag, label=f"{label} Q" if label else "Imag")
            else:
                axis.plot(t, y, label=label)
            axis.set_xlabel(
                "Time [s]" if sampling_rate is not None else "Time [Symbol Periods]"
            )
            axis.set_ylabel("Amplitude")
            if sampling_rate is not None:
                _set_eng_formatter(axis, "x")
            else:
                axis.xaxis.set_major_locator(ticker.MultipleLocator(1))
            default_title = "Impulse Response" if q == "impulse" else "Step Response"
        else:
            assert response is not None
            f_host = to_device(response.frequencies, "cpu")
            h = to_device(response.response, "cpu")
            if q == "magnitude":
                y = 20 * np.log10(np.abs(h) + 1e-12)
                ylabel, default_title = (
                    "Magnitude [dB]",
                    "Frequency Response (Magnitude)",
                )
            elif q == "phase":
                y = np.unwrap(np.angle(h).astype(np.float64))
                ylabel, default_title = "Phase [rad]", "Frequency Response (Phase)"
            else:
                y = to_device(response.group_delay, "cpu")
                if sampling_rate is None:
                    y = y / sps
                ylabel = (
                    "Group Delay [s]"
                    if sampling_rate is not None
                    else "Group Delay [Symbol Periods]"
                )
                default_title = "Group Delay"
            axis.plot(f_host, y, label=label)
            if quantity is None:
                axis.set_xlim(f_host[0], f_host[-1])
            axis.set_xlabel(
                "Frequency [Hz]"
                if sampling_rate is not None
                else "Frequency [Cycles/Sample]"
            )
            axis.set_ylabel(ylabel)
            if sampling_rate is not None:
                _set_eng_formatter(axis, "x")
                if q == "group_delay":
                    _set_eng_formatter(axis, "y")
        axis.set_title(title if title is not None else default_title)
        if xlim is not None:
            axis.set_xlim(xlim)
        if ylim is not None:
            axis.set_ylim(ylim)
        if legend or (
            legend is None
            and quantity is None
            and q == "impulse"
            and np.iscomplexobj(y)
        ):
            axis.legend()
    return _finish((fig, tuple(axes) if quantity is None else axes[0]), show)
