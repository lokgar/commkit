"""Equalizer result containers and CPR state."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ..backend import ArrayType, to_device
from ..logger import logger

if TYPE_CHECKING:
    from ..core.signal import Signal

# -----------------------------------------------------------------------------
# RESULT CONTAINER
# -----------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class EqualizerState:
    """Where an adaptive equalizer stopped; pass as ``state=`` to continue.

    The state is taken after the last output symbol whose filter window lies
    entirely inside the data seen so far (a block boundary for the block
    equalizers).  It carries the weights, the input normalization, the input
    from that point on, the inline CPR state and, for RLS, the inverse
    correlation matrix.  The last ``overlap`` symbols of the result that
    produced it were computed with a zero-padded window; the next call
    recomputes them, so stitching ``y1[..., :-overlap]`` (all of ``y1`` when
    ``overlap == 0``) with the next output gives exactly the output of one
    uninterrupted call.

    Training symbols and pilots of a continued call start at its first
    output symbol.  A state only continues the equalizer and configuration
    that produced it; anything else raises.

    Attributes
    ----------
    equalizer : str
        Name of the equalizer that produced the state.
    num_taps, sps, block_size : int
        Its configuration (``block_size`` is 0 for the sequential ones).
    cpr : PLL, BPS or None
        Its inline carrier phase recovery.
    weights : numpy.ndarray
        ``(C, C, num_taps)`` complex64 butterfly weights.
    input_norm_factor : float or numpy.ndarray
        Normalization applied to the input (``(C,)`` for MIMO).
    pending : numpy.ndarray
        ``(C, L)`` complex64 normalized input from the resume point on.
    lead : int
        Samples of ``pending`` before the first resumed symbol's position.
    overlap : int
        Trailing output symbols of the previous result that the next call
        recomputes.
    inverse_correlation : numpy.ndarray or None
        RLS only: ``(C·T, C·T)`` complex128 inverse correlation matrix.
    """

    equalizer: str
    num_taps: int
    sps: int
    block_size: int
    cpr: Any
    weights: np.ndarray
    input_norm_factor: float | np.ndarray
    pending: np.ndarray
    lead: int
    overlap: int
    inverse_correlation: np.ndarray | None = None
    carrier: Any = None  # inline CPR arrays (private layout per engine)

    @property
    def num_channels(self) -> int:
        """Number of channels ``C``."""
        return int(self.weights.shape[0])


def _check_state(
    state: EqualizerState | None,
    *,
    equalizer: str,
    num_taps: int,
    sps: int,
    num_ch: int,
    block_size: int = 0,
    cpr: Any = None,
    initial_taps: Any = None,
    center_tap: int | None = None,
) -> None:
    """Raise unless ``state`` continues this exact equalizer configuration."""
    if state is None:
        return
    if not isinstance(state, EqualizerState):
        raise TypeError(
            f"{equalizer}(): state must be an EqualizerState (result.state), got "
            f"{type(state).__name__}."
        )
    if initial_taps is not None:
        raise ValueError(
            f"{equalizer}(): give initial_taps or state, not both (the state "
            "carries the weights)."
        )
    if center_tap is not None:
        raise ValueError(
            f"{equalizer}(): center_tap is fixed by the state that is continued."
        )
    expected = {
        "equalizer": equalizer,
        "num_taps": num_taps,
        "sps": sps,
        "block_size": block_size,
        "number of channels": num_ch,
    }
    actual = {
        "equalizer": state.equalizer,
        "num_taps": state.num_taps,
        "sps": state.sps,
        "block_size": state.block_size,
        "number of channels": state.num_channels,
    }
    for key, value in expected.items():
        if actual[key] != value:
            raise ValueError(
                f"{equalizer}(): the state was made with {key}={actual[key]!r}, "
                f"this call has {value!r}."
            )
    if state.cpr != cpr:
        raise ValueError(
            f"{equalizer}(): the state was made with cpr={state.cpr!r}, this "
            f"call has cpr={cpr!r}."
        )


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
        radians.  ``None`` without ``cpr``.
    state : EqualizerState or None
        Continuation state; pass as ``state=`` to the next call.  ``None``
        for a run that does not continue (``apply_taps``).

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
    state: EqualizerState | None = None


def _log_equalizer_exit(
    result: EqualizerResult,
    name: str,
    check_convergence: bool = False,
) -> EqualizerResult:
    """Log the exit MSE of an EqualizerResult (INFO level)."""
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

    return result


def _attach_equalized_signal(
    result: EqualizerResult, signal: Signal | None
) -> EqualizerResult:
    """Attach an array result to its originating Signal at symbol rate."""
    if signal is not None:
        result.y_hat = signal.replace_samples(
            result.y_hat, sampling_rate=signal.symbol_rate
        )
    return result
