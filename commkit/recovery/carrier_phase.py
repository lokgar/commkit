"""Carrier phase: ``estimate_carrier_phase`` and ``correct_carrier_phase``.

The method object selects the algorithm (D16); each lives next to its kernel
in this package and is dispatched through ``_ESTIMATORS``.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from .._array import as_2d, restore_1d
from ..backend import ArrayType, dispatch
from ..core._signal_adapter import S, adapt_signal
from ..core.signal import Signal
from ..logger import logger
from ._common import _Context, _Phase
from .bps import BPS, _bps
from .corrections import DataAided, _data_aided
from .pilots import (
    PilotAided,
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
    "CarrierPhaseEstimate",
    "correct_carrier_phase",
    "estimate_carrier_phase",
]

CarrierPhaseMethod = (
    BPS
    | ViterbiViterbi
    | Tikhonov
    | PLL
    | PilotAided
    | PilotTone
    | PilotTones
    | DataAided
)

_ESTIMATORS: dict[type, Callable[[ArrayType, Any, _Context], _Phase]] = {
    BPS: _bps,
    ViterbiViterbi: _viterbi_viterbi,
    Tikhonov: _tikhonov,
    PLL: _pll,
    PilotAided: _pilot_aided,
    PilotTone: _pilot_tone,
    PilotTones: _pilot_tones,
    DataAided: _data_aided,
}

# Methods that read one symbol per sample (the tone methods read the
# oversampled waveform).
_SYMBOL_RATE = (BPS, ViterbiViterbi, Tikhonov, PLL, PilotAided, DataAided)


@dataclass(frozen=True)
class CarrierPhaseEstimate:
    """Result of :func:`estimate_carrier_phase`.

    Per-channel fields drop their channel axis for ``(N,)`` input and stay on
    the input's device; method-specific fields are ``None`` for other
    methods.

    Attributes
    ----------
    value : array_like
        Phase trajectory in radians, ``float64``, the shape of the samples.
        :func:`correct_carrier_phase` rotates by ``exp(-j·value)``.
    block_centers : numpy.ndarray or None
        Block methods (BPS, Viterbi-Viterbi, Tikhonov): block centre
        positions in symbols, ``(B,)``.
    block_phase : array_like or None
        Block methods: the block phases after unwrapping, smoothing and
        cycle-slip repair, ``(B,)`` or ``(C, B)``.
    pilot_indices : numpy.ndarray or None
        ``PilotAided``: the pilot symbol indices, ``(P,)``.
    pilot_phase : array_like or None
        ``PilotAided``: unwrapped pilot phases, ``(P,)`` or ``(C, P)``.
    tone_frequencies : numpy.ndarray or None
        Tone methods: measured tone centres in Hz on the host; ``PilotTone``
        per channel (0-d or ``(C,)``), ``PilotTones`` per channel and tone
        (``(K,)`` or ``(C, K)``).
    tone_snr_db : numpy.ndarray or None
        ``PilotTones``: in-band SNR of each tone in dB, ``(K,)``.
    differential_phase : array_like or None
        ``PilotTones``: tracked inter-tone phase ``δ_k[n]`` in radians,
        ``(K, N)`` (zero for the reference tone).
    reference_tone : int or None
        ``PilotTones``: index of the reference (strongest) tone.
    used_tones : tuple of int or None
        ``PilotTones``: tones that passed the gates and were combined.
    """

    value: ArrayType
    block_centers: np.ndarray | None = None
    block_phase: ArrayType | None = None
    pilot_indices: np.ndarray | None = None
    pilot_phase: ArrayType | None = None
    tone_frequencies: np.ndarray | None = None
    tone_snr_db: np.ndarray | None = None
    differential_phase: ArrayType | None = None
    reference_tone: int | None = None
    used_tones: tuple[int, ...] | None = None


def estimate_carrier_phase(
    samples: ArrayType | Signal,
    method: CarrierPhaseMethod,
    *,
    sampling_rate: float | None = None,
    constellation: Any = None,
) -> CarrierPhaseEstimate:
    """
    Estimate the carrier phase trajectory.

    Parameters
    ----------
    samples : array_like or Signal
        ``(N,)`` or ``(C, N)``: symbols at one sample per symbol for
        ``BPS``, ``ViterbiViterbi``, ``Tikhonov``, ``PLL``, ``PilotAided``
        and ``DataAided``; the oversampled waveform for ``PilotTone`` and
        ``PilotTones``.
    method : BPS, ViterbiViterbi, Tikhonov, PLL, PilotAided, PilotTone, PilotTones or DataAided
        Estimation method.
    sampling_rate : float, optional
        Sampling rate in Hz, needed by the tone methods.  Taken from the
        Signal; required for array input.  A value that disagrees with the
        Signal raises.
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
    estimator = _ESTIMATORS.get(type(method))
    if estimator is None:
        raise TypeError(
            "estimate_carrier_phase(): method must be BPS, ViterbiViterbi, "
            "Tikhonov, PLL, PilotAided, PilotTone, PilotTones or DataAided, "
            f"got {type(method).__name__}."
        )
    signal_adapter = adapt_signal(samples, function_name="estimate_carrier_phase()")
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

    def per_channel(a: Any) -> Any:
        return None if a is None else restore_1d(was_1d, a)

    return CarrierPhaseEstimate(
        value=per_channel(res.phase),
        block_centers=res.block_centers,
        block_phase=per_channel(res.block_phase),
        pilot_indices=res.pilot_indices,
        pilot_phase=per_channel(res.pilot_phase),
        tone_frequencies=per_channel(res.tone_frequencies),
        tone_snr_db=res.tone_snr_db,
        differential_phase=res.differential_phase,
        reference_tone=res.reference_tone,
        used_tones=res.used_tones,
    )


_PHASE_ROTATE_KERNEL: dict = {}


def _get_cupy_phase_rotate() -> Any:
    """Compile and cache the fused CuPy phase-rotation kernel.

    Fuses the whole ``s · exp(-j·wrap(φ))`` chain - float64 wrap, float32
    sin/cos, complex multiply - into a single elementwise kernel: one read of
    the symbols, one read of the phase, one write, instead of the ~7 separate
    full-record kernel passes and temporaries of the ufunc chain.
    """
    if "k" not in _PHASE_ROTATE_KERNEL:
        import cupy as cp

        _PHASE_ROTATE_KERNEL["k"] = cp.ElementwiseKernel(
            "complex64 s, float64 phi",
            "complex64 out",
            """
            double w = phi - rint(phi * 0.15915494309189535) * 6.283185307179586;
            float sw, cw;
            sincosf((float)w, &sw, &cw);
            out = s * complex<float>(cw, -sw);
            """,
            "commkit_phase_rotate",
        )
    return _PHASE_ROTATE_KERNEL["k"]


def correct_carrier_phase(
    samples: S,
    how: CarrierPhaseEstimate | CarrierPhaseMethod | float | ArrayType,
    *,
    sampling_rate: float | None = None,
    constellation: Any = None,
) -> S:
    """
    Remove a carrier phase: ``y[n] = x[n]·exp(-j·φ[n])``.

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
    array_like or Signal
        Corrected samples, same shape, dtype and device.

    Notes
    -----
    The phase is wrapped to ``[-π, π]`` in float64 (a standalone trajectory
    is unbounded), then rotated with a float32 phasor.
    """
    signal_adapter = adapt_signal(samples, function_name="correct_carrier_phase()")
    if isinstance(how, CarrierPhaseMethod):
        how = estimate_carrier_phase(
            samples, how, sampling_rate=sampling_rate, constellation=constellation
        )
    elif sampling_rate is not None:
        signal_adapter.resolve_fact("sampling_rate", sampling_rate)
    phase = how.value if isinstance(how, CarrierPhaseEstimate) else how

    x, xp, _ = dispatch(signal_adapter.array)
    logger.debug("Applying carrier phase correction: shape=%s", x.shape)
    phase_f64 = xp.asarray(phase, dtype=xp.float64)
    if xp is not np and x.dtype == xp.complex64:
        # GPU fast path: single fused kernel (broadcasts (N,) phase over (C, N)).
        return signal_adapter.wrap_samples(_get_cupy_phase_rotate()(x, phase_f64))
    two_pi = 2.0 * xp.pi
    phase_wrapped = (phase_f64 - xp.round(phase_f64 / two_pi) * two_pi).astype(
        xp.float32
    )
    phasor = xp.exp(-1j * phase_wrapped)
    if phasor.dtype != x.dtype:
        phasor = phasor.astype(x.dtype)
    return signal_adapter.wrap_samples(x * phasor)
