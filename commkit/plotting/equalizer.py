"""Equalizer plots.

:func:`plot_equalizer_result` draws an
:class:`~commkit.equalization.EqualizerResult` (convergence, taps and the
carrier phase of inline CPR); :func:`plot_zf_equalizer_response` draws the
channel, equalizer and combined responses of a ZF / MMSE equalizer.
"""

from typing import Any

import numpy as np

from .. import _fft
from ..backend import get_array_module, to_device
from ..smoothing import moving_average
from .theme import (
    _decimate_minmax,
    _finish,
    _panel_grid,
    _select,
    _series_label,
    _set_eng_formatter,
)

__all__ = ["plot_equalizer_result", "plot_zf_equalizer_response"]


def plot_equalizer_result(
    result,
    *,
    smoothing: int = 50,
    annotations: bool = True,
    quantity: str | None = None,
    label: str | None = None,
    legend: bool | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    ax=None,
    title: str | None = None,
    show: bool = False,
) -> tuple[Any, Any] | None:
    """Plot the convergence, taps and inline-CPR phase of an equalizer result.

    Panels: the smoothed MSE convergence curve, the tap weights and, when
    the equalizer ran with ``cpr``, the recovered phase trajectory.

    Parameters
    ----------
    result : EqualizerResult
        The equalizer's result (its ``error``, ``weights`` and, if any,
        ``phase_trajectory``).
    smoothing : int, default 50
        Moving-average window of the MSE convergence curve, in symbols.
    annotations : bool, default True
        Mark the end of training on the convergence panel.  ``False`` leaves
        only the measured curves, for overlays of several results.
    quantity : {"convergence", "taps", "phase"}, optional
        Plot only that panel. ``None`` draws the complete dashboard. A phase
        trajectory is required for ``"phase"``.
    label : str, optional
        Label for a selected panel, for comparisons on one axis.
    legend : bool, optional
        Show or hide legends; ``None`` preserves the default.
    xlim, ylim : tuple of float, optional
        Axis limits for a selected panel.
    ax : matplotlib.axes.Axes or array_like of Axes, optional
        One axis for a selected quantity; 2 or 3 for the dashboard. ``None``
        creates a figure.
    title : str, optional
        Title of the figure.
    show : bool, default False
        Call ``plt.show()`` and return ``None``.

    Returns
    -------
    (fig, axes) or None
        Figure and one axis for a selected quantity, or the dashboard axes;
        ``None`` with ``show=True``.
    """
    error = result.error
    phase_traj = getattr(result, "phase_trajectory", None)
    available = ("convergence", "taps") + (("phase",) if phase_traj is not None else ())
    if quantity == "phase" and phase_traj is None:
        raise ValueError("quantity='phase' requires a phase_trajectory")
    panels = _select(
        quantity, ("convergence", "taps", "phase"), label=label, xlim=xlim, ylim=ylim
    )
    if quantity is None:
        panels = available
    is_mimo = error.ndim == 2
    fig, grid = _panel_grid(ax, 1, len(panels))
    axes = grid[0]

    if "convergence" in panels:
        # --- Panel 1: MSE convergence ---
        ax_conv = axes[panels.index("convergence")]

        def _smooth_mse(mse, smoothing):
            """Return host (x_coords, smoothed_mse_db) in raw-symbol-index space.

            The smoothing window is clipped to at most n//3 so the curve always
            spans at least 2/3 of the data range even for short sequences.
            """
            xp = get_array_module(mse)
            n = len(mse)
            effective = max(1, min(smoothing, n // 3))
            if effective > 1 and n > effective:
                mse_smooth = moving_average(mse, window=effective, mode="valid")
                # Element k of mode="valid" output averages mse[k : k+effective].
                # Place it at the center of that window so the x-axis is in
                # actual symbol-index space, not smoothed-bin-index space.
                x = xp.arange(len(mse_smooth)) + (effective - 1) / 2.0
            else:
                mse_smooth = mse
                x = xp.arange(n, dtype=xp.float64)
            x, y = _decimate_minmax(x, 10 * xp.log10(mse_smooth + 1e-30))
            return to_device(x, "cpu"), to_device(y, "cpu")

        if is_mimo:
            num_ch = error.shape[0]
            for ch in range(num_ch):
                mse = abs(error[ch]) ** 2
                x_smooth, mse_db = _smooth_mse(mse, smoothing)
                ax_conv.plot(x_smooth, mse_db, label=_series_label(label, f"Ch {ch}"))
            ax_conv.legend()
        else:
            mse = abs(error) ** 2
            x_smooth, mse_db = _smooth_mse(mse, smoothing)
            ax_conv.plot(x_smooth, mse_db, label=label)

        n_train = getattr(result, "num_train_symbols", 0)
        if annotations and n_train and n_train > 0:
            ax_conv.axvline(
                n_train,
                color="white",
                linestyle="--",
                label=f"DD start ({n_train})",
            )
            ax_conv.legend()

        ax_conv.set_xlabel("Symbol Index")
        ax_conv.set_ylabel("MSE [dB]")
        ax_conv.set_title("Convergence")

    if "taps" in panels:
        # --- Panel 2: Final tap weights ---
        ax_taps = axes[panels.index("taps")]
        weights = to_device(result.weights, "cpu")

        if is_mimo:
            # Butterfly weights: (C, C, num_taps) - plot magnitude of each row
            num_ch = weights.shape[0]
            num_taps = weights.shape[2]
            tap_idx = np.arange(num_taps) - (num_taps // 2)
            for i in range(num_ch):
                for j in range(num_ch):
                    ax_taps.plot(
                        tap_idx,
                        np.abs(weights[i, j]),
                        marker="o",
                        label=_series_label(label, f"w[{i},{j}]"),
                    )
            ax_taps.legend(ncol=2)
        else:
            # SISO: (num_taps,) - stem plot
            num_taps = weights.shape[0]
            tap_idx = np.arange(num_taps) - (num_taps // 2)
            ax_taps.stem(tap_idx, np.abs(weights), label=label)

        ax_taps.set_xlabel("Tap Index")
        ax_taps.set_ylabel("|w|")
        ax_taps.set_title("Tap Weights")

    if "phase" in panels and phase_traj is not None:
        # --- Panel 3: Phase trajectory (CPR only) ---
        ax_phase = axes[panels.index("phase")]
        xp = get_array_module(phase_traj)
        phi = phase_traj * (180.0 / np.pi)
        rows = phi[None] if phi.ndim == 1 else phi
        idx = xp.arange(rows.shape[-1], dtype=xp.float64)
        for ch in range(rows.shape[0]):
            n_d, p_d = _decimate_minmax(idx, rows[ch])
            ax_phase.plot(
                to_device(n_d, "cpu"),
                to_device(p_d, "cpu"),
                label=_series_label(label, f"Ch {ch}" if phi.ndim > 1 else None),
            )
        if phi.ndim > 1:
            ax_phase.legend()
        ax_phase.set_xlabel("Symbol Index")
        ax_phase.set_ylabel("Phase [deg]")
        ax_phase.set_title("CPR Phase Trajectory")

    if quantity is None:
        if title is not None:
            fig.suptitle(title)
        return _finish((fig, axes), show, legend=legend)
    if title is not None:
        axes[0].set_title(title)
    return _finish((fig, axes[0]), show, xlim=xlim, ylim=ylim, legend=legend)


# -----------------------------------------------------------------------------
# SYNCHRONIZATION DIAGNOSTICS
# -----------------------------------------------------------------------------


def plot_zf_equalizer_response(
    channel_estimate,
    *,
    noise_var: float = 0.0,
    nfft: int = 1024,
    sampling_rate: float = 1.0,
    quantity: str | None = None,
    label: str | None = None,
    legend: bool | None = None,
    ax=None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    show: bool = False,
    title: str = "ZF/MMSE Equalizer Response",
) -> tuple[Any, Any] | None:
    """Plot the channel, ZF / MMSE equalizer and combined frequency responses.

    Three panels per channel path, in dB:

    * ``|H(f)|``: the channel;
    * ``|W(f)|``: the equalizer :func:`commkit.equalization.zf_equalizer`
      applies (``H* / (|H|² + σ²)`` for SISO, the regularized matrix inverse
      for MIMO), measured by its response to an impulse on each input;
    * ``|W(f)·H(f)|``: the combination, a matrix product per bin for MIMO
      (≈ 0 dB on the diagonal for ZF, a soft mask for MMSE).

    Parameters
    ----------
    channel_estimate : array_like
        Causal channel impulse response (lag 0 at index 0), ``(L,)`` for
        SISO or ``(C, C, L)`` for MIMO, ``[i, j]`` from input ``j`` to
        output ``i``.
    noise_var : float, default 0.0
        Noise variance ``σ²`` of the MMSE regularization; ``0.0`` gives ZF.
    nfft : int, default 1024
        FFT size of the frequency responses.
    sampling_rate : float, default 1.0
        Sampling rate in Hz, for the frequency axis.
    quantity : {"channel", "equalizer", "combined"}, optional
        Plot only that response. ``None`` draws all three panels per path.
    label : str, optional
        Label for a selected response, for comparisons on one axis.
    legend : bool, optional
        Show (``True``) or hide (``False``) the legends; ``None`` draws none.
    ax : matplotlib.axes.Axes or array_like of Axes, optional
        One axis per channel path for a selected response, or three per path
        (``(paths, 3)``) for all of them; ``None`` creates a figure.
    xlim, ylim : tuple of float, optional
        Limits applied to the frequency-response panels.
    show : bool, default False
        Call ``plt.show()`` and return ``None``.
    title : str, default "ZF/MMSE Equalizer Response"
        Title of the figure.

    Returns
    -------
    (fig, axes) or None
        The figure and its axes; ``None`` with ``show=True``, which
        shows the figure instead.
    """
    names = {
        "channel": "Channel |H(f)|",
        "equalizer": "Equalizer |W(f)|",
        "combined": "Combined |H·W(f)|",
    }
    panels = _select(quantity, tuple(names), label=label)
    from ..equalization import zf_equalizer

    h = to_device(channel_estimate, "cpu")
    if h.ndim not in (1, 3):
        raise ValueError("channel_estimate must be 1-D or 3-D")
    siso = h.ndim == 1
    paths = (
        [("", h)]
        if siso
        else [
            (f"[rx{i}$\\rightarrow$tx{j}]", h[i, j])
            for i in range(h.shape[0])
            for j in range(h.shape[1])
        ]
    )

    freqs = np.fft.fftfreq(nfft, d=1.0 / sampling_rate)
    sort_idx = np.argsort(freqs)
    f_sorted = freqs[sort_idx]

    # The equalizer's impulse response from input j: an impulse at the
    # record's centre (room for the anti-causal tail), rolled back to lag 0.
    hm = h[None, None, :] if siso else h
    c = hm.shape[0]
    w = np.empty((c, c, nfft), dtype=np.complex64)
    for j in range(c):
        impulse = np.zeros((c, nfft), dtype=np.complex64)
        impulse[j, nfft // 2] = 1.0
        y = zf_equalizer(impulse[0] if siso else impulse, h, noise_var=noise_var)
        w[:, j] = np.roll(np.reshape(y, (c, nfft)), -(nfft // 2), axis=-1)
    Hf = np.moveaxis(_fft.fft(hm, n=nfft, axis=-1), -1, 0)  # (nfft, C, C)
    Wf = np.moveaxis(_fft.fft(w, axis=-1), -1, 0)
    Gf = Wf @ Hf
    spectra = {"channel": Hf, "equalizer": Wf, "combined": Gf}

    fig, axes = _panel_grid(ax, len(paths), len(panels))
    for k, ((row_label, _), ax_row) in enumerate(zip(paths, axes, strict=True)):
        i, j = divmod(k, c)
        responses = {n: s[sort_idx, i, j] for n, s in spectra.items()}
        for name, axi in zip(panels, ax_row, strict=True):
            axi.plot(
                f_sorted,
                20 * np.log10(np.abs(responses[name]) + 1e-12),
                label=label,
            )
            if quantity is None:
                axi.set_title(f"{row_label} {names[name]}".strip())
            else:
                axi.set_title(f"{title} {row_label}".strip())
            _set_eng_formatter(axi, "x")
            axi.set_xlabel("Frequency [Hz]")
            axi.set_ylabel("Magnitude [dB]")

    if quantity is None:
        fig.suptitle(title)
        result = axes[0] if siso else axes
    else:
        result = axes[0, 0] if siso else axes[:, 0]
    return _finish((fig, result), show, xlim=xlim, ylim=ylim, legend=legend)


# -----------------------------------------------------------------------------
# CARRIER-PHASE CHARACTERIZATION DIAGNOSTICS
# -----------------------------------------------------------------------------
