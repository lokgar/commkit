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
* **report packaging**: summary functions (``linewidth_*``,
  ``allan_deviation``, ``frequency_drift_metrics`` scalars) return Python
  floats and *plot-sized* NumPy arrays (Welch/Allan grids, ≤ ``nperseg``
  bins) after one transfer, because their consumers are prints and plots.

JAX arrays are **not** an input backend here (nor anywhere behind
``backend.dispatch``): they report NumPy and are silently pulled to the host.
Convert at the boundary with ``backend.from_jax`` before calling in.

Sample-rate arrays returned to the caller (``carrier_phase_trajectory``,
``separate_drift_phase_noise``, ``frequency_drift_metrics['df']``,
``dsh_phase``, ``fm_noise_psd``, ``dsh_fm_noise_psd``) always stay on the
input backend - chain them without paying transfers.
"""

from .allan import allan_deviation
from .drift import frequency_drift_metrics, separate_drift_phase_noise
from .interferometry import dsh_beat, dsh_fm_noise_psd, dsh_phase, linewidth_dsh
from .linewidth import (
    fm_noise_psd,
    linewidth_beta_separation,
    linewidth_increment,
)
from .trajectory import carrier_phase_trajectory

__all__ = [
    "allan_deviation",
    "carrier_phase_trajectory",
    "dsh_beat",
    "dsh_fm_noise_psd",
    "dsh_phase",
    "fm_noise_psd",
    "frequency_drift_metrics",
    "linewidth_beta_separation",
    "linewidth_dsh",
    "linewidth_increment",
    "separate_drift_phase_noise",
]
