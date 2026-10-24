"""Laser and carrier-phase characterization.

Post-processing of recovered signals and laser captures, as opposed to the
DSP stages that produce them:

- :func:`carrier_phase_trajectory`: the data-aided unwrapped carrier phase
  that the other analyses consume.
- :func:`separate_drift_phase_noise` and :func:`frequency_drift`: slow
  frequency drift against fast phase noise.
- :func:`fm_noise_psd` and :func:`dsh_fm_noise_psd`: FM-noise spectra.
- :func:`allan_deviation`: overlapping Allan deviation of the frequency.
- :func:`estimate_linewidth`: the laser linewidth, with one method object per
  estimator (:class:`IncrementSlope`, :class:`IncrementSubtract`,
  :class:`BetaSeparation` on phase trajectories; :class:`DshFmPsd`,
  :class:`DshIncrement`, :class:`DshLorentzian` on delayed self-heterodyne
  beats, whose front end is :func:`dsh_beat` and :func:`dsh_phase`).

Sample-rate outputs (``carrier_phase_trajectory``,
``separate_drift_phase_noise``, ``frequency_drift(...).df``, ``dsh_phase``,
the FM-noise PSDs) stay on the input's device, so they chain without
transfers.  Summaries (:class:`LinewidthEstimate`, :class:`AllanDeviation`,
the :class:`FrequencyDrift` statistics) are reporting-layer fits: they
reduce on the device, transfer plot-sized arrays once and return host floats
(SISO) or ``(C,)`` arrays (MIMO).  Host-side NumPy is otherwise limited to
metadata (lag lists, segment sizes, tau grids) and the small fits on those
reductions.
"""

from .allan import AllanDeviation, allan_deviation
from .drift import FrequencyDrift, frequency_drift, separate_drift_phase_noise
from .fm_noise import DshFmNoisePsd, dsh_fm_noise_psd, fm_noise_psd
from .interferometry import dsh_beat, dsh_phase
from .linewidth import (
    BetaSeparation,
    DshFmPsd,
    DshIncrement,
    DshLorentzian,
    IncrementSlope,
    IncrementSubtract,
    LinewidthEstimate,
    estimate_linewidth,
)
from .trajectory import carrier_phase_trajectory

__all__ = [
    "AllanDeviation",
    "BetaSeparation",
    "DshFmNoisePsd",
    "DshFmPsd",
    "DshIncrement",
    "DshLorentzian",
    "FrequencyDrift",
    "IncrementSlope",
    "IncrementSubtract",
    "LinewidthEstimate",
    "allan_deviation",
    "carrier_phase_trajectory",
    "dsh_beat",
    "dsh_fm_noise_psd",
    "dsh_phase",
    "estimate_linewidth",
    "fm_noise_psd",
    "frequency_drift",
    "separate_drift_phase_noise",
]
