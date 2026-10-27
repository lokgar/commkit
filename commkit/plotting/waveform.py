"""Time-domain waveform plots: :func:`plot_time_domain`."""

from typing import Any

import numpy as np

from ..backend import dispatch, to_device
from ..core._signal_adapter import adapt_signal
from ..logger import logger
from .theme import (
    _channel_axes,
    _finish,
    _get_axis,
    _set_eng_formatter,
)

__all__ = ["plot_time_domain"]


def plot_time_domain(
    samples: Any,
    *,
    sampling_rate: float | None = None,
    start_symbol: int = 0,
    num_symbols: int | None = None,
    sps: float | None = None,
    max_points: int | None = 10_000,
    ax: Any | None = None,
    title: str | None = "Waveform",
    show: bool = False,
    **kwargs: Any,
) -> tuple[Any, Any] | None:
    """Plot a signal against time.

    Complex samples are drawn as their in-phase (I) and quadrature (Q)
    components; the time axis uses SI prefixes.  ``(C, N)`` input gets one
    panel per channel.

    Parameters
    ----------
    samples : array_like or Signal
        Input samples, ``(N,)`` or ``(C, N)``.
    sampling_rate : float, optional
        Sampling rate in Hz.  Taken from the Signal; required for array
        input.
    start_symbol : int, default 0
        First symbol (or sample, without ``sps``) drawn.
    num_symbols : int, optional
        Number of symbol periods drawn; needs ``sps``.
    sps : float, optional
        Samples per symbol (a fact): taken from a Signal; for arrays, needed
        to count in symbols, otherwise ``start_symbol`` counts samples.
    max_points : int or None, default 10000
        A longer view is drawn as its min/max envelope: the samples are cut
        into ``max_points // 2`` buckets and each bucket is drawn as its
        minimum and maximum, computed on the input's device before the
        transfer.  At screen resolution this looks like the full trace.
        ``None`` draws every sample.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; ``None`` creates a figure.
    title : str or None, default "Waveform"
        Title of the figure; ``None`` for none.
    show : bool, default False
        Call ``plt.show()`` and return ``None``.
    **kwargs : Any
        Passed to ``ax.plot``.

    Returns
    -------
    (fig, ax) or None
        The figure and the Axes (an array of Axes for several panels);
        ``None`` with ``show=True``, which shows the figure instead.
    """
    signal_adapter = adapt_signal(samples, function_name="plot_time_domain()")
    x = signal_adapter.array
    sampling_rate = signal_adapter.resolve_fact("sampling_rate", sampling_rate)
    if signal_adapter.signal is not None:
        sps = signal_adapter.resolve_fact("sps", sps)
    return _plot_time_domain(
        x,
        sampling_rate=sampling_rate,
        start_symbol=start_symbol,
        num_symbols=num_symbols,
        sps=sps,
        max_points=max_points,
        ax=ax,
        title=title,
        show=show,
        **kwargs,
    )


def _plot_time_domain(
    samples: Any,
    *,
    sampling_rate: Any,
    start_symbol: Any,
    num_symbols: Any,
    sps: Any,
    max_points: int | None,
    ax: Any,
    title: Any,
    show: Any,
    **kwargs: Any,
) -> tuple[Any, Any] | None:
    """Render array data for :func:`plot_time_domain` (the public boundary resolves the Signal)."""
    logger.debug("Generating time-domain plot.")

    samples, xp, _ = dispatch(samples)

    # Handle Multichannel
    # Convention: (Channels, Time)
    if samples.ndim > 1:
        num_channels = samples.shape[0]

        fig, axes, targets = _channel_axes(num_channels, ax)

        for i in range(num_channels):
            channel_samples = samples[i]

            target_ax = targets[i]

            ch_title = f"{title} (Ch {i})" if title else f"Channel {i}"

            _plot_time_domain(
                channel_samples,
                sampling_rate=sampling_rate,
                start_symbol=start_symbol,
                num_symbols=num_symbols,
                sps=sps,
                max_points=max_points,
                ax=target_ax,
                title=ch_title,
                show=False,
                **kwargs,
            )

        return _finish((fig, axes), show)

    # --- 1D Logic ---

    fig, ax = _get_axis(ax)

    start_idx = int(start_symbol * sps) if sps is not None else int(start_symbol)

    # Slice on the device: only the viewed samples are transferred.
    if num_symbols is not None and sps is not None:
        limit = start_idx + int(num_symbols * sps)
        if limit > samples.shape[-1]:
            limit = samples.shape[-1]
            logger.warning(
                "Limit exceeds number of symbols. Plotting up to last symbol."
            )
        view = samples[start_idx:limit]
    else:
        view = samples[start_idx:]

    parts = (
        [("I", view.real), ("Q", view.imag)]
        if xp.iscomplexobj(view)
        else [(None, view)]
    )
    for label, part in parts:
        t, y = _envelope(part, sampling_rate, max_points, xp)
        ax.plot(t, y, label=label, **kwargs)
    if len(parts) > 1:
        ax.legend()
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    _set_eng_formatter(ax, "x", "s")
    if title is not None:
        ax.set_title(title)

    return _finish((fig, ax), show)


def _envelope(
    y: Any, sampling_rate: float, max_points: int | None, xp: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Host time axis and values to draw.

    The samples, or their min/max envelope when there are more than
    ``max_points``.

    About ``max_points // 2`` buckets of equal width (the last one shorter)
    each contribute their minimum and maximum at the bucket's start time,
    reduced on the device, so no sample is left out of the envelope.
    """
    n = int(y.shape[-1])
    if max_points is None or n <= max_points:
        return np.arange(n) / sampling_rate, np.asarray(to_device(y, "cpu"))
    width = -(-n // max(1, max_points // 2))  # ceil
    full = n // width
    blocks = y[: full * width].reshape(full, width)
    lo, hi = [blocks.min(axis=1)], [blocks.max(axis=1)]
    if full * width < n:
        tail = y[full * width :]
        lo.append(tail.min()[None])
        hi.append(tail.max()[None])
    env = xp.stack([xp.concatenate(lo), xp.concatenate(hi)], axis=1).ravel()
    starts = np.arange(env.shape[0] // 2) * width
    return np.repeat(starts, 2) / sampling_rate, np.asarray(to_device(env, "cpu"))
