"""
Signal analysis and characterization.

Post-processing and diagnostic routines that operate on recovered signals to
quantify their properties, as opposed to the DSP stages that *produce* those
signals (synchronization, equalization, recovery, ...).  Functions here are
grouped by the property they characterize; new analyses can be added as
independent groups without disturbing the others.

Backend policy
--------------
Every function dispatches on the input array type (NumPy -> CPU, CuPy -> GPU)
and all sample-rate work - phase extraction, unwrapping, filtering, Welch
PSDs, variance reductions - runs on that backend.  Host-side NumPy inside this
package is deliberate and limited to three cases:

* **metadata**: lag lists, Welch segment sizing, tau grids - scalar shapes,
  never data;
* **tiny post-reduction fits**: e.g. ``np.polyfit`` on an ``(n_lag, C)``
  variance matrix after a single device->host transfer - cheaper than a device
  least-squares launch;
* **report packaging**: summaries (``estimate_linewidth``,
  ``allan_deviation``, ``frequency_drift_metrics`` scalars) return Python
  floats and *plot-sized* NumPy arrays (Welch/Allan grids, ≤ ``nperseg``
  bins) after one transfer, because their consumers are prints and plots.

Inputs are NumPy or CuPy arrays; arrays from other frameworks raise
``TypeError`` (convert them explicitly through DLPack).

Linewidth follows the verb rules of the rest of the library: one
``estimate_linewidth(x, method)`` whose method object (``IncrementSlope``,
``BetaSeparation``, ``DshFmPsd``, ...) chooses the estimator and returns a
``LinewidthEstimate``.  The other analyses are plain computations.  Results
with more than one value are frozen dataclasses (``AllanDeviation``,
``FrequencyDrift``, ...).  Sample-rate arrays returned to the caller
(``carrier_phase_trajectory``, ``separate_drift_phase_noise``,
``frequency_drift_metrics(...).df``,
``dsh_phase``, ``fm_noise_psd``, ``dsh_fm_noise_psd``) always stay on the
input backend - chain them without paying transfers.
"""

from .allan import AllanDeviation, allan_deviation
from .drift import FrequencyDrift, frequency_drift_metrics, separate_drift_phase_noise
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
    "frequency_drift_metrics",
    "separate_drift_phase_noise",
]
