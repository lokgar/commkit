"""Equalizer plots.

:func:`plot_equalizer_result` draws an
:class:`~commkit.equalization.EqualizerResult` (convergence, taps and the
carrier phase of inline CPR); :func:`plot_zf_equalizer_response` draws the
channel, equalizer and combined responses of a ZF / MMSE equalizer.
"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .. import _fft
from ..backend import get_array_module, to_device
from ..smoothing import moving_average
from .theme import _decimate_minmax, _grid_figsize, _set_eng_formatter

__all__ = ["plot_equalizer_result", "plot_zf_equalizer_response"]


def plot_equalizer_result(
    result,
    *,
    smoothing: int = 50,
    ax=None,
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
    ax : array_like of matplotlib.axes.Axes, optional
        Axes to draw on: 2 without a ``phase_trajectory``, 3 with one; ``None``
        creates a figure.
    show : bool, default False
        Call ``plt.show()`` and return ``None``.

    Returns
    -------
    (fig, axes) or None
        The figure and its axes; ``None`` with ``show=True``, which
        shows the figure instead.
    """
    # Reduced on the result's device: the curves are smoothed and cut to
    # their min/max envelope there, and only the plotted points move.
    error = result.error
    weights = to_device(result.weights, "cpu")
    phase_traj = getattr(result, "phase_trajectory", None)

    is_mimo = error.ndim == 2
    n_panels = 3 if phase_traj is not None else 2

    if ax is None:
        fig, axes = plt.subplots(1, n_panels, figsize=_grid_figsize(1, n_panels))
    else:
        axes = np.asarray(ax).flatten()[:n_panels]
        fig = axes[0].figure

    # --- Panel 1: MSE convergence ---
    ax_conv = axes[0]

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
            ax_conv.plot(x_smooth, mse_db, label=f"Ch {ch}")
        ax_conv.legend()
    else:
        mse = abs(error) ** 2
        x_smooth, mse_db = _smooth_mse(mse, smoothing)
        ax_conv.plot(x_smooth, mse_db)

    n_train = getattr(result, "num_train_symbols", 0)
    if n_train and n_train > 0:
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

    # --- Panel 2: Final tap weights ---
    ax_taps = axes[1]

    if is_mimo:
        # Butterfly weights: (C, C, num_taps) - plot magnitude of each row
        num_ch = weights.shape[0]
        num_taps = weights.shape[2]
        tap_idx = np.arange(num_taps) - (num_taps // 2)
        for i in range(num_ch):
            for j in range(num_ch):
                label = f"w[{i},{j}]"
                ax_taps.plot(
                    tap_idx,
                    np.abs(weights[i, j]),
                    marker="o",
                    markersize=3,
                    label=label,
                )
        ax_taps.legend(ncol=2)
    else:
        # SISO: (num_taps,) - stem plot
        num_taps = weights.shape[0]
        tap_idx = np.arange(num_taps) - (num_taps // 2)
        markerline, _, _ = ax_taps.stem(tap_idx, np.abs(weights))
        markerline.set_markersize(4)

    ax_taps.set_xlabel("Tap Index")
    ax_taps.set_ylabel("|w|")
    ax_taps.set_title("Tap Weights")

    # --- Panel 3: Phase trajectory (CPR only) ---
    if phase_traj is not None:
        ax_phase = axes[2]
        xp = get_array_module(phase_traj)
        phi = phase_traj * (180.0 / np.pi)
        rows = phi[None] if phi.ndim == 1 else phi
        idx = xp.arange(rows.shape[-1], dtype=xp.float64)
        for ch in range(rows.shape[0]):
            n_d, p_d = _decimate_minmax(idx, rows[ch])
            ax_phase.plot(
                to_device(n_d, "cpu"),
                to_device(p_d, "cpu"),
                label=f"Ch {ch}" if phi.ndim > 1 else None,
            )
        if phi.ndim > 1:
            ax_phase.legend()
        ax_phase.set_xlabel("Symbol Index")
        ax_phase.set_ylabel("Phase [deg]")
        ax_phase.set_title("CPR Phase Trajectory")

    if show:
        plt.show()
        return None
    return fig, axes


# -----------------------------------------------------------------------------
# SYNCHRONIZATION DIAGNOSTICS
# -----------------------------------------------------------------------------


def plot_zf_equalizer_response(
    channel_estimate,
    *,
    noise_variance: float = 0.0,
    nfft: int = 1024,
    sampling_rate: float = 1.0,
    ax=None,
    show: bool = False,
    title: str = "ZF/MMSE Equalizer Response",
) -> tuple[Any, Any] | None:
    """Plot the channel, ZF / MMSE equalizer and combined frequency responses.

    Three panels per channel path, in dB:

    * ``|H(f)|``: the channel;
    * ``|W(f)|``: the equalizer, ``W = H* / (|H|² + σ²)``;
    * ``|H(f)·W(f)|``: the combination (≈ 0 dB for ZF, a soft mask for
      MMSE).

    Parameters
    ----------
    channel_estimate : array_like
        Channel impulse response, ``(L,)`` for SISO or ``(C_rx, C_tx, L)``
        for MIMO.
    noise_variance : float, default 0.0
        Noise variance ``σ²`` of the MMSE regularization; ``0.0`` gives ZF.
    nfft : int, default 1024
        FFT size of the frequency responses.
    sampling_rate : float, default 1.0
        Sampling rate in Hz, for the frequency axis.
    ax : array_like of matplotlib.axes.Axes, optional
        Axes to draw on: 3 for SISO, ``(C_rx * C_tx, 3)`` for MIMO; ``None``
        creates a figure.
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
    h = to_device(channel_estimate, "cpu")
    reg = max(noise_variance, 1e-12)
    siso = h.ndim == 1

    freqs = np.fft.fftfreq(nfft, d=1.0 / sampling_rate)
    sort_idx = np.argsort(freqs)
    f_sorted = freqs[sort_idx]

    def _draw_triplet(H_f, ax_row, row_label=""):
        H_s = H_f[sort_idx]
        W_s = np.conj(H_s) / (np.abs(H_s) ** 2 + reg)
        HW_s = H_s * W_s
        for axi, data, lbl, col in zip(
            ax_row,
            [H_s, W_s, HW_s],
            ["Channel |H(f)|", "Equalizer |W(f)|", "Combined |H·W(f)|"],
            ["C0", "C1", "C2"],
        ):
            axi.plot(f_sorted, 20 * np.log10(np.abs(data) + 1e-12), color=col)
            prefix = f"{row_label} " if row_label else ""
            axi.set_title(f"{prefix}{lbl}")
            _set_eng_formatter(axi, "x", "Hz")
            axi.set_xlabel("Frequency [Hz]")
            axi.set_ylabel("Magnitude [dB]")

    if siso:
        H_f = _fft.fft(h, n=nfft)
        if ax is None:
            fig, axes = plt.subplots(1, 3, figsize=_grid_figsize(1, 3))
        else:
            axes = list(ax) if hasattr(ax, "__len__") else [ax]
            fig = axes[0].figure
        _draw_triplet(H_f, axes)
        fig.suptitle(title)
        if show:
            plt.show()
            return None
        return fig, axes

    C_rx, C_tx = h.shape[0], h.shape[1]
    n_rows = C_rx * C_tx
    if ax is None:
        fig, raw_axes = plt.subplots(
            n_rows, 3, figsize=_grid_figsize(n_rows, 3), squeeze=False
        )
        axes = [list(raw_axes[r]) for r in range(n_rows)]
    else:
        if hasattr(ax, "__len__") and hasattr(ax[0], "__len__"):
            axes = [list(row) for row in ax]
        else:
            flat = list(ax)
            axes = [flat[r * 3 : (r + 1) * 3] for r in range(n_rows)]
        fig = axes[0][0].figure

    for i in range(C_rx):
        for j in range(C_tx):
            row = i * C_tx + j
            H_f = _fft.fft(h[i, j], n=nfft)
            _draw_triplet(H_f, axes[row], row_label=f"[rx{i}$\\rightarrow$tx{j}]")

    fig.suptitle(title)
    if show:
        plt.show()
        return None
    return fig, axes


# -----------------------------------------------------------------------------
# CARRIER-PHASE CHARACTERIZATION DIAGNOSTICS
# -----------------------------------------------------------------------------
