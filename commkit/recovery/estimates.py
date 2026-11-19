"""Carrier-phase estimates.

:class:`CarrierPhaseEstimate`, the result of ``PLL`` and ``ConstantPhase``,
and the subclasses holding the other methods' diagnostics:
:class:`BlockPhaseEstimate` (BPS, Viterbi-Viterbi, Tikhonov),
:class:`PilotInterpolationEstimate`, :class:`PilotToneEstimate` and
:class:`PilotTonesEstimate`.  Each method builds its estimate with a leading
channel axis; :func:`~commkit.recovery.estimate_carrier_phase` drops it for
``(N,)`` input.
"""

from dataclasses import dataclass

import numpy as np

from ..backend import ArrayType

__all__ = [
    "BlockPhaseEstimate",
    "CarrierPhaseEstimate",
    "PilotInterpolationEstimate",
    "PilotToneEstimate",
    "PilotTonesEstimate",
]


@dataclass(frozen=True)
class CarrierPhaseEstimate:
    """Result of :func:`estimate_carrier_phase`: the result of ``PLL`` and
    ``ConstantPhase``, and the field every method gives.

    The constructor takes the same fields.  The other methods return a
    subclass with their diagnostics: :class:`BlockPhaseEstimate` (BPS,
    Viterbi-Viterbi, Tikhonov), :class:`PilotInterpolationEstimate`
    (``PilotInterpolation``), :class:`PilotToneEstimate` (``PilotTone``) and
    :class:`PilotTonesEstimate` (``PilotTones``).  Per-channel fields
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
class PilotInterpolationEstimate(CarrierPhaseEstimate):
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
class PilotToneEstimate(CarrierPhaseEstimate):
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
class PilotTonesEstimate(PilotToneEstimate):
    """Result of ``PilotTones``.

    The constructor takes the same fields: those of
    :class:`PilotToneEstimate`, then the attributes below.

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
