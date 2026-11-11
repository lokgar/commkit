"""Carrier-phase estimation and correction.

:func:`estimate_carrier_phase` and :func:`correct_carrier_phase`.  The method
object selects the algorithm; each lives next to its kernel in this package and
is dispatched through ``_ESTIMATORS``.  The estimate is a
:class:`CarrierPhaseEstimate`.
"""

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from .._array import as_2d, restore_1d
from ..backend import ArrayType, dispatch
from ..core._signal_adapter import S, SignalAdapter, adapt_signal
from ..core.signal import Signal
from ..logger import logger
from ..math import _derotate
from ._common import _Context, _Phase
from .bps import BPS, _bps
from .corrections import ConstantPhase, _data_aided
from .pilots import (
    PilotInterpolation,
    PilotTone,
    PilotTones,
    _pilot_aided,
    _pilot_tone,
    _pilot_tones,
)
from .pll import PLL, _pll
from .tikhonov import Tikhonov, _tikhonov
from .viterbi_viterbi import ViterbiViterbi, _viterbi_viterbi

__all__ = [
    "BlockPhaseEstimate",
    "CarrierPhaseEstimate",
    "MultiTonePhaseEstimate",
    "PilotPhaseEstimate",
    "TonePhaseEstimate",
    "correct_carrier_phase",
    "estimate_carrier_phase",
]

CarrierPhaseMethod = (
    BPS
    | ViterbiViterbi
    | Tikhonov
    | PLL
    | PilotInterpolation
    | PilotTone
    | PilotTones
    | ConstantPhase
)

_ESTIMATORS: dict[type, Callable[[ArrayType, Any, _Context], _Phase]] = {
    BPS: _bps,
    ViterbiViterbi: _viterbi_viterbi,
    Tikhonov: _tikhonov,
    PLL: _pll,
    PilotInterpolation: _pilot_aided,
    PilotTone: _pilot_tone,
    PilotTones: _pilot_tones,
    ConstantPhase: _data_aided,
}

# Methods that read one symbol per sample (the tone methods read the
# oversampled waveform).
_SYMBOL_RATE = (BPS, ViterbiViterbi, Tikhonov, PLL, PilotInterpolation, ConstantPhase)


@dataclass(frozen=True)
class CarrierPhaseEstimate:
    """Result of :func:`estimate_carrier_phase`: the result of ``PLL`` and
    ``ConstantPhase``, and the field every method gives.

    The constructor takes the same fields.  The other methods return a
    subclass with their diagnostics: :class:`BlockPhaseEstimate` (BPS,
    Viterbi-Viterbi, Tikhonov), :class:`PilotPhaseEstimate`
    (``PilotInterpolation``), :class:`TonePhaseEstimate` (``PilotTone``) and
    :class:`MultiTonePhaseEstimate` (``PilotTones``).  Per-channel fields
    drop their channel axis for ``(N,)`` input and stay on the input's
    device.

    Attributes
    ----------
    value : ndarray
        Phase trajectory in radians, ``float64``, the shape of the samples.
        :func:`correct_carrier_phase` rotates by ``exp(-j·value)``.
    """

    value: ArrayType


@dataclass(frozen=True)
class BlockPhaseEstimate(CarrierPhaseEstimate):
    """Result of the block methods: BPS, Viterbi-Viterbi, Tikhonov.

    The constructor takes the same fields.

    Attributes
    ----------
    value : ndarray
        Phase trajectory in radians, as in :class:`CarrierPhaseEstimate`.
    block_centers : numpy.ndarray
        Block center positions in symbols, ``(B,)``.
    block_phase : ndarray
        The block phases after unwrapping, smoothing and cycle-slip repair,
        ``(B,)`` or ``(C, B)``.
    """

    block_centers: np.ndarray
    block_phase: ArrayType


@dataclass(frozen=True)
class PilotPhaseEstimate(CarrierPhaseEstimate):
    """Result of ``PilotInterpolation``.

    The constructor takes the same fields.

    Attributes
    ----------
    value : ndarray
        Phase trajectory in radians, as in :class:`CarrierPhaseEstimate`.
    pilot_indices : numpy.ndarray
        The pilot symbol indices, ``(P,)``.
    pilot_phase : ndarray
        Unwrapped pilot phases, ``(P,)`` or ``(C, P)``.
    """

    pilot_indices: np.ndarray
    pilot_phase: ArrayType


@dataclass(frozen=True)
class TonePhaseEstimate(CarrierPhaseEstimate):
    """Result of ``PilotTone``.

    The constructor takes the same fields.

    Attributes
    ----------
    value : ndarray
        Phase trajectory in radians, as in :class:`CarrierPhaseEstimate`.
    tone_frequencies : numpy.ndarray
        Measured tone centers in Hz on the host: per channel (0-d or
        ``(C,)``) for ``PilotTone``, per channel and tone (``(K,)`` or
        ``(C, K)``) for ``PilotTones``.
    """

    tone_frequencies: np.ndarray


@dataclass(frozen=True)
class MultiTonePhaseEstimate(TonePhaseEstimate):
    """Result of ``PilotTones``.

    The constructor takes the same fields: those of
    :class:`TonePhaseEstimate`, then the attributes below.

    Attributes
    ----------
    tone_snr_db : numpy.ndarray
        In-band SNR of each tone in dB, ``(K,)``.
    differential_phase : ndarray
        Tracked inter-tone phase ``δ_k[n]`` in radians, ``(K, N)`` (zero for
        the reference tone).
    reference_tone : int
        Index of the reference (strongest) tone.
    used_tones : tuple of int
        Tones that passed the gates and were combined.
    """

    tone_snr_db: np.ndarray
    differential_phase: ArrayType
    reference_tone: int
    used_tones: tuple[int, ...]


#: Result class by method type (CarrierPhaseEstimate for the rest).
_RESULTS: dict[type, type[CarrierPhaseEstimate]] = {
    BPS: BlockPhaseEstimate,
    ViterbiViterbi: BlockPhaseEstimate,
    Tikhonov: BlockPhaseEstimate,
    PilotInterpolation: PilotPhaseEstimate,
    PilotTone: TonePhaseEstimate,
    PilotTones: MultiTonePhaseEstimate,
}
#: Fields with a channel axis, dropped for (N,) input.
_PER_CHANNEL = ("value", "block_phase", "pilot_phase", "tone_frequencies")


def estimate_carrier_phase(
    samples: ArrayType | Signal,
    method: CarrierPhaseMethod,
    *,
    sampling_rate: float | None = None,
    constellation: Any = None,
) -> CarrierPhaseEstimate:
    """Estimate the carrier phase trajectory.

    Parameters
    ----------
    samples : array_like or Signal
        ``(N,)`` or ``(C, N)``: symbols at one sample per symbol for
        ``BPS``, ``ViterbiViterbi``, ``Tikhonov``, ``PLL``, ``PilotInterpolation``
        and ``ConstantPhase``; the oversampled waveform for ``PilotTone`` and
        ``PilotTones``.
    method : BPS, ViterbiViterbi, Tikhonov, PLL, PilotInterpolation, PilotTone, PilotTones or ConstantPhase
        Estimation method.
    sampling_rate : float, optional
        Sampling rate in Hz, needed by the tone methods and by ``Tikhonov``
        (where it is the symbol rate, to relate the linewidth to the symbol
        period).  Taken from the Signal; required for array input.  A value
        that disagrees with the Signal raises.
    constellation : Constellation, optional
        Decision constellation of the blind and decision-directed methods.
        Defaults to the Signal's ``constellation``.

    Returns
    -------
    CarrierPhaseEstimate
        The phase trajectory plus the method's diagnostics.

    Examples
    --------
    >>> est = estimate_carrier_phase(y, BPS(test_phases=64))
    >>> y = correct_carrier_phase(y, est)
    >>> y = correct_carrier_phase(y, ViterbiViterbi(block_size=64))
    """
    signal_adapter = adapt_signal(samples, function_name="estimate_carrier_phase()")
    return _estimate_carrier_phase(
        signal_adapter, method, sampling_rate=sampling_rate, constellation=constellation
    )


def _estimate_carrier_phase(
    signal_adapter: SignalAdapter[Any],
    method: CarrierPhaseMethod,
    *,
    sampling_rate: float | None,
    constellation: Any,
) -> CarrierPhaseEstimate:
    """:func:`estimate_carrier_phase` on an adapted input (unwrapped once)."""
    estimator = _ESTIMATORS.get(type(method))
    if estimator is None:
        raise TypeError(
            "estimate_carrier_phase(): method must be BPS, ViterbiViterbi, "
            "Tikhonov, PLL, PilotInterpolation, PilotTone, PilotTones or ConstantPhase, "
            f"got {type(method).__name__}."
        )
    sig = signal_adapter.signal
    fs = None
    if sig is not None or sampling_rate is not None:
        fs = float(signal_adapter.resolve_fact("sampling_rate", sampling_rate))
    symbol_rate = isinstance(method, _SYMBOL_RATE)
    if sig is not None and symbol_rate and not np.isclose(sig.sps, 1.0, rtol=1e-9):
        raise ValueError(
            f"estimate_carrier_phase(): {type(method).__name__} needs one "
            f"sample per symbol; the Signal has sps={sig.sps:g}."
        )
    reference = getattr(method, "symbols", None)
    if reference is None and sig is not None and sig.reference is not None:
        reference = sig.reference.symbols
    ctx = _Context(
        constellation=signal_adapter.resolve_choice("constellation", constellation),
        sampling_rate=fs,
        reference=reference,
    )

    x, _, _ = dispatch(signal_adapter.array)
    x, was_1d = as_2d(x, name="samples")
    res = estimator(x, method, ctx)

    cls = _RESULTS.get(type(method), CarrierPhaseEstimate)
    fields = {}
    for f in dataclasses.fields(cls):
        a = res.phase if f.name == "value" else getattr(res, f.name)
        fields[f.name] = restore_1d(was_1d, a) if f.name in _PER_CHANNEL else a
    return cls(**fields)


def correct_carrier_phase(
    samples: S,
    how: CarrierPhaseEstimate | CarrierPhaseMethod | float | ArrayType,
    *,
    sampling_rate: float | None = None,
    constellation: Any = None,
) -> S:
    """Remove a carrier phase: ``y[n] = x[n]·exp(-j·φ[n])``.

    Parameters
    ----------
    samples : array_like or Signal
        Samples, ``(N,)`` or ``(C, N)``.
    how : CarrierPhaseEstimate, method object, float or array_like
        An estimate, a method (estimated first; see
        :func:`estimate_carrier_phase`), or the phase in radians: a scalar,
        a trajectory, or anything that broadcasts against the samples.
    sampling_rate, constellation
        Passed to :func:`estimate_carrier_phase` when ``how`` is a method.

    Returns
    -------
    ndarray or Signal
        Corrected samples, same shape, dtype and device.

    Notes
    -----
    The phase is wrapped to ``[-π, π]`` in float64 (a standalone trajectory
    is unbounded), then rotated in the samples' precision (a float32 phasor
    for complex64, float64 for complex128).
    """
    signal_adapter = adapt_signal(samples, function_name="correct_carrier_phase()")
    if isinstance(how, CarrierPhaseMethod):
        how = _estimate_carrier_phase(
            signal_adapter,
            how,
            sampling_rate=sampling_rate,
            constellation=constellation,
        )
    elif sampling_rate is not None:
        signal_adapter.resolve_fact("sampling_rate", sampling_rate)
    phase = how.value if isinstance(how, CarrierPhaseEstimate) else how

    x, xp, _ = dispatch(signal_adapter.array)
    logger.debug("Applying carrier phase correction: shape=%s", x.shape)
    return signal_adapter.wrap_samples(_derotate(x, phase))
