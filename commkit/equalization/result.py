"""Equalizer results.

:class:`EqualizerResult` is what every adaptive equalizer returns; its
``state``, an :class:`EqualizerState`, continues a later call.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ..backend import ArrayType, to_device
from ..logger import logger

__all__ = ["EqualizerResult", "EqualizerState"]

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
    carrier : object
        The inline CPR's arrays, in the private layout of each engine.
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


# (state made by, continued by): equalizer changes a state may carry over.
_HANDOVER = {("cma", "rde"), ("block_cma", "block_rde")}


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
    """Raise unless ``state`` continues this exact equalizer configuration.

    The one allowed change of equalizer is the blind handover from CMA to RDE
    (``_HANDOVER``): the two share their engine and state layout, and CMA
    pre-convergence is how RDE is started on multi-ring QAM.
    """
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
    handover = (state.equalizer, equalizer) in _HANDOVER
    actual = {
        "equalizer": equalizer if handover else state.equalizer,
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
    """Output of an adaptive or block equalizer.

    Arrays stay on the input's device.

    Attributes
    ----------
    y_hat : ndarray
        Equalized symbols, always an array (also for Signal input),
        ``(N_sym,)`` SISO or ``(C, N_sym)`` MIMO.
    weights : ndarray
        Final tap weights, ``(num_taps,)`` SISO or ``(C, C, num_taps)`` for
        the MIMO butterfly.
    error : ndarray
        Error signal, ``(N_sym,)`` or ``(C, N_sym)``.
    weights_history : ndarray or None
        Tap weights over time, only with ``store_weights=True``.
    num_train_symbols : int
        Training symbols consumed in the data-aided phase (LMS/RLS); skip
        them before steady-state metrics such as EVM, SNR and BER.
    input_norm_factor : float or numpy.ndarray
        Factor ``rms(samples, axis=-1) * sqrt(sps)`` by which each channel's
        input was divided before equalization: a ``float`` for SISO, ``(C,)``
        for MIMO (at ``sps=1``, the plain RMS of the input symbols).  It
        restores absolute power when a different capture (e.g. vacuum noise)
        is passed through the same frozen taps::

            α = signal_result.input_norm_factor  # float or (C,) array
            β = noise_result.input_norm_factor
            P_noise_corrected = (
                np.mean(np.abs(noise_result.y_hat) ** 2, axis=-1) * (β / α) ** 2
            )

        ``P_signal / P_noise_corrected`` is then the per-channel signal to
        noise power ratio through the DSP chain.
    tail_trim : int
        Symbols trimmed from the tail of ``y_hat`` (the zero-padded window);
        non-zero only for RLS, where it is ``num_taps // 2``.
        ``result.signal`` carries the reference trimmed to match; with array
        input, trim the reference arrays::

            symbols = symbols[..., :-result.tail_trim]
            bits = bits[..., :-result.tail_trim * bits_per_symbol]
    phase_trajectory : ndarray or None
        Phase in radians that the inline CPR applied to each symbol before
        the decision and the weight update,
        ``y_hat[n] = (Wᴴ x[n]) · exp(-j · phase_trajectory[n])``;
        ``(N_sym,)`` or ``(C, N_sym)``, ``None`` without ``cpr``.  For
        phase-noise analysis, cycle-slip diagnostics, or to start a later
        CPR stage.
    state : EqualizerState or None
        Continuation state; pass as ``state=`` to the next call.
    signal : Signal or None
        For Signal input, the output as a 1-SPS Signal (``sampling_rate =
        symbol_rate``) whose reference is cut to the output symbols.  After
        a continued call it starts at the call's own first symbol, past the
        ``overlap`` recomputed for the previous chunk.  ``None`` for array
        input.
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
    signal: Signal | None = None


def _log_equalizer_exit(result: EqualizerResult, name: str) -> EqualizerResult:
    """Log the exit MSE of an EqualizerResult (INFO level)."""
    # Skip the device->host transfer when nothing will consume the result, and
    # transfer only the tail window (<= 100 samples per channel) when it will.
    if result.error is None or not logger.isEnabledFor(logging.INFO):
        return result
    n_sym = result.error.shape[-1]  # time axis; (N_sym,) or (C, N_sym)
    window = max(1, min(100, n_sym))
    tail = to_device(result.error[..., -window:], "cpu")
    if tail.ndim == 1:  # SISO
        mse_db = 10.0 * np.log10(float(np.mean(np.abs(tail) ** 2)) + 1e-30)
        logger.info("%s: exit MSE=%.1f dB (final %s symbols)", name, mse_db, window)
    else:  # MIMO: per-channel MSE
        per_ch_mse = np.mean(np.abs(tail) ** 2, axis=-1)  # (C,)
        parts = ", ".join(
            f"ch{c}={10.0 * np.log10(m + 1e-30):.1f}" for c, m in enumerate(per_ch_mse)
        )
        logger.info("%s: exit MSE (final %s symbols): %s dB", name, window, parts)
    return result


def _attach_equalized_signal(
    result: EqualizerResult,
    signal: Signal | None,
    state: EqualizerState | None = None,
) -> EqualizerResult:
    """Attach the 1-SPS output Signal for Signal input.

    The Signal holds the output symbols that belong to ``signal``'s own
    samples - after the ``state.overlap`` symbols a continued call recomputes
    for the previous chunk - with the reference cut to them (RLS drops its
    tail), at ``sampling_rate = symbol_rate``.  ``y_hat`` stays an array.
    """
    if signal is None:
        return result
    leading = 0 if state is None else state.overlap
    y = result.y_hat[..., leading:]
    reference = signal.reference
    if reference is not None:
        reference = reference.head(y.shape[-1])
    result.signal = signal.replace(
        samples=y, sampling_rate=signal.symbol_rate, reference=reference
    )
    return result
