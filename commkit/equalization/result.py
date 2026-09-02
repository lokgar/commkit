"""Equalizer result containers and CPR state."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ..backend import ArrayType, to_device
from ..logger import logger

# -----------------------------------------------------------------------------
# RESULT CONTAINER
# -----------------------------------------------------------------------------


@dataclass
class CPRState:
    """Carrier-phase-recovery state for warm-starting across equalizer calls.

    All arrays are CPU NumPy regardless of the equalizer backend - CPR state is
    small and must survive device resets.  Do not convert to CuPy.

    Used/produced by ``lms()``, ``rls()``, and ``block_lms()`` when
    ``cpr_type`` is not None.  Pass as ``cpr_state=result.cpr_state`` to the
    next call to continue phase tracking without a re-lock transient.
    """

    # PLL state (lms / rls with cpr_type='pll' or 'bps')
    pll_phi: np.ndarray | None = None  # (C,) float64
    pll_freq: np.ndarray | None = None  # (C,) float64

    # BPS cross-block unwrap state (block_lms with cpr_type='bps')
    bps_prev4: np.ndarray | None = None  # (C,) float64
    bps_offset4: np.ndarray | None = None  # (C,) float64
    bps_d2_hist: np.ndarray | None = None  # (P, C, K-1) float32 - CPU copy

    # Cycle-slip regression state (all CPR modes)
    cs_buf_x: np.ndarray | None = None  # (C, H) float64
    cs_buf_y: np.ndarray | None = None  # (C, H) float64
    cs_buf_ptr: np.ndarray | None = None  # (C,) int64
    cs_buf_n: np.ndarray | None = None  # (C,) int64
    cs_stats: np.ndarray | None = None  # (C, 4) float64

    # JAX-specific BPS buffer state (JAX backend only; None for Numba)
    jax_bps_buf: np.ndarray | None = None  # (KB, C) complex64
    jax_bps_buf_ptr: int | None = None  # scalar int32

    # Identity tags - used to validate shape compatibility on warm-start
    cpr_type: str | None = None
    num_ch: int = 0
    symmetry: int = 4
    bps_P: int = 0  # number of BPS test phases
    bps_K: int = 0  # BPS block size
    cs_H: int = 0  # cycle-slip history length


@dataclass
class EqualizerResult:
    """Container for equalizer outputs.

    Attributes
    ----------
    y_hat : ArrayType
        Equalized symbol sequence.  Shape: ``(N_sym,)`` SISO or
        ``(C, N_sym)`` MIMO.
    weights : ArrayType
        Final tap weight vector. Shape: ``(num_taps,)`` for SISO
        or ``(C, C, num_taps)`` for MIMO butterfly.
    error : ArrayType
        Error signal history. Shape: ``(N_sym,)`` or ``(C, N_sym)``.
    weights_history : ArrayType or None
        Tap weight evolution over time. Only populated when
        ``store_weights=True``.
    num_train_symbols : int
        Number of data-aided training symbols consumed (LMS/RLS).  Used to
        discard the DA-trained transient before computing steady-state metrics
        like EVM, SNR, and BER.
    input_norm_factor : float or np.ndarray
        Normalization factor(s) ``rms(samples, axis=-1) * sqrt(sps)`` applied
        per-channel by ``_normalize_inputs`` before equalization.  For SISO
        this is a plain ``float``; for MIMO it is a 1-D ``np.ndarray`` of
        shape ``(C,)`` - one factor per input stream.
        Stored so callers can apply the post-hoc power correction needed when
        passing a different capture (e.g. vacuum noise) through the same
        frozen taps without disturbing the per-channel ratio:

        Example::

            α = signal_result.input_norm_factor  # float or (C,) array
            β = noise_result.input_norm_factor
            P_noise_corrected = (
                np.mean(np.abs(noise_result.y_hat) ** 2, axis=-1) * (β / α) ** 2
            )

        ``P_signal / P_noise_corrected`` is then the physically meaningful
        per-channel signal-to-noise power ratio preserved through the DSP chain.
        At ``sps=1`` this factor equals the plain per-channel RMS of the input symbols.
    tail_trim : int
        Number of symbols trimmed from the tail of ``y_hat`` to remove the
        zero-padding contamination zone.  Non-zero only for RLS (equals
        ``num_taps // 2``).  If non-zero, trim reference arrays to match::

            source_symbols = source_symbols[..., :-result.tail_trim]
            source_bits    = source_bits[..., :-result.tail_trim * bits_per_symbol]
    phase_trajectory : np.ndarray or None
        Per-symbol phase estimates produced by the inline CPR stage, in
        radians.  ``None`` when ``cpr_type=None``.

        Shape: ``(N_sym,)`` for SISO, ``(C, N_sym)`` for MIMO butterfly.

        The values are the instantaneous phase corrections *applied* to each
        symbol before the hard decision and weight update, i.e.
        ``y_hat[n] = (W^H x[n]) · exp(-j · phase_trajectory[n])``.
        Useful for post-hoc phase-noise analysis, cycle-slip diagnostics, and
        as a warm-start phase estimate for a subsequent CPR stage.
    """

    y_hat: ArrayType
    weights: ArrayType
    error: ArrayType
    weights_history: ArrayType | None = None
    num_train_symbols: int = 0
    input_norm_factor: float | np.ndarray = 1.0
    tail_trim: int = 0
    phase_trajectory: ArrayType | None = None
    cpr_state: CPRState | None = None


def _log_equalizer_exit(
    result: EqualizerResult,
    name: str,
    debug_plot: bool = False,
    check_convergence: bool = False,
    plot_smoothing: int = 50,
) -> EqualizerResult:
    """Log exit MSE and optionally show a debug plot for an EqualizerResult."""
    if result.error is not None:
        n_sym = result.error.shape[-1]  # time axis; (N_sym,) or (C, N_sym)
        _want_log = logger.isEnabledFor(logging.INFO)
        # The convergence check emits a WARNING (normally always enabled), but
        # only runs when explicitly requested and the signal is long enough.
        want_conv = check_convergence and n_sym >= 20

        # Skip the whole MSE computation - and its device->host transfer - when
        # nothing will consume the result.  When it is needed, transfer only the
        # tail (and head, for convergence) windows rather than the full error
        # array: ≤100 samples/channel instead of N.
        if _want_log or want_conv:
            window = max(1, min(100, n_sym))
            tail = to_device(result.error[..., -window:], "cpu")

            if tail.ndim == 1:  # SISO
                mse_final = float(np.mean(np.abs(tail) ** 2))
                mse_db = 10.0 * np.log10(mse_final + 1e-30)
                if _want_log:
                    logger.info(
                        "%s: exit MSE=%.1f dB (final %s symbols)", name, mse_db, window
                    )
            else:  # MIMO: per-channel MSE; keep the mean for the convergence check
                per_ch_mse = np.mean(np.abs(tail) ** 2, axis=-1)  # (C,)
                if _want_log:
                    parts = ", ".join(
                        f"ch{c}={10.0 * np.log10(m + 1e-30):.1f}"
                        for c, m in enumerate(per_ch_mse)
                    )
                    logger.info(
                        "%s: exit MSE (final %s symbols): %s dB", name, window, parts
                    )
                mse_final = float(np.mean(per_ch_mse))
                mse_db = 10.0 * np.log10(mse_final + 1e-30)

            if want_conv:
                init_window = max(1, min(100, n_sym // 10))
                head = to_device(result.error[..., :init_window], "cpu")
                mse_init = float(np.mean(np.abs(head) ** 2))
                if mse_init > 0 and mse_final > mse_init * 0.9:
                    logger.warning(
                        "%s: convergence may be poor - final MSE (%.1f dB) "
                        "not significantly below initial MSE (%.1f dB). "
                        "Consider reducing step_size or increasing signal length.",
                        name,
                        mse_db,
                        10.0 * np.log10(mse_init + 1e-30),
                    )

    if debug_plot:
        from .. import plotting as _plotting  # lazy import avoids circular dep

        _plotting.plot_equalizer_result(result, smoothing=plot_smoothing)

    return result
