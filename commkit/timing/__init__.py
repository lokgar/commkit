"""Timing synchronization.

- :mod:`~commkit.timing.synchronization`: frame and delay synchronization:
  :func:`estimate_timing` finds the integer and fractional timing offset by
  cross-correlation (:func:`cross_correlate`) with a template or a
  :class:`~commkit.core.Preamble` (a :class:`TimingEstimate`), and
  :func:`correct_timing` removes it; :func:`peak_offset` and
  :func:`apply_delay` measure and apply sub-sample delays; the
  sequences :func:`barker_sequence` and :func:`zadoff_chu_sequence` build
  preambles.
- :mod:`~commkit.timing.clock_recovery`: sampling-clock recovery:
  :func:`estimate_sampling_clock_offset` (a
  :class:`SamplingClockOffsetEstimate`) and
  :func:`correct_sampling_clock_offset` with a :class:`Gardner` loop.
"""

from .clock_recovery import (
    Gardner,
    SamplingClockOffsetEstimate,
    correct_sampling_clock_offset,
    estimate_sampling_clock_offset,
)
from .synchronization import (
    TimingEstimate,
    apply_delay,
    barker_sequence,
    correct_timing,
    cross_correlate,
    estimate_timing,
    peak_offset,
    zadoff_chu_sequence,
)

__all__ = [
    "Gardner",
    "SamplingClockOffsetEstimate",
    "TimingEstimate",
    "apply_delay",
    "barker_sequence",
    "correct_sampling_clock_offset",
    "correct_timing",
    "cross_correlate",
    "estimate_sampling_clock_offset",
    "estimate_timing",
    "peak_offset",
    "zadoff_chu_sequence",
]
