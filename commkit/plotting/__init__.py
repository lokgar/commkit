"""Plots of signals and of the library's results (Matplotlib).

Every public function but :func:`apply_default_theme` is a ``plot_*`` function
that draws a Signal, an array or a result object (an estimate, an
:class:`~commkit.equalization.EqualizerResult`, an analysis summary). A plot
that needs a calculation calls the corresponding public compute function;
device arrays are reduced on the device and only what is drawn is transferred.
Tick labels use SI prefixes.
:func:`apply_default_theme` sets the library's Matplotlib style, dark or
light; importing the package changes no Matplotlib settings. The plot
functions take colors, fonts and sizes from ``matplotlib.rcParams``, so they
follow that theme or any other style.

- :mod:`~commkit.plotting.waveform`, :mod:`~commkit.plotting.eye`,
  :mod:`~commkit.plotting.constellation`, :mod:`~commkit.plotting.spectral`:
  the signal itself.
- :mod:`~commkit.plotting.filter_response`,
  :mod:`~commkit.plotting.equalizer`: filters and equalizers.
- :mod:`~commkit.plotting.timing`, :mod:`~commkit.plotting.frequency`,
  :mod:`~commkit.plotting.recovery`: timing, frequency-offset and
  carrier-phase estimates (``plot_<quantity>_estimate``, which draws the
  diagnostics of the estimate's method, chosen by its class).
- :mod:`~commkit.plotting.coding`: error-rate curves, LLR distributions,
  parity-check matrices and EXIT charts.
- :mod:`~commkit.plotting.analysis`: carrier-phase decomposition, drift,
  Allan deviation and linewidth.
"""

from __future__ import annotations

from .analysis import (
    plot_allan_deviation,
    plot_carrier_phase_characterization,
    plot_carrier_phase_decomposition,
    plot_frequency_drift,
    plot_linewidth_estimate,
)
from .coding import (
    plot_error_rate,
    plot_exit_chart,
    plot_llr_distribution,
    plot_parity_check,
)
from .constellation import plot_constellation, plot_ideal_constellation
from .equalizer import plot_channel_correction, plot_equalizer_result
from .eye import plot_eye_diagram
from .filter_response import plot_filter_response
from .frequency import plot_frequency_offset_estimate
from .recovery import plot_carrier_phase_estimate, plot_carrier_phase_trajectory
from .spectral import plot_psd, plot_spectrogram
from .theme import apply_default_theme
from .timing import plot_timing_estimate
from .waveform import plot_time_domain

__all__ = [
    "apply_default_theme",
    "plot_allan_deviation",
    "plot_carrier_phase_characterization",
    "plot_carrier_phase_decomposition",
    "plot_carrier_phase_estimate",
    "plot_carrier_phase_trajectory",
    "plot_channel_correction",
    "plot_constellation",
    "plot_equalizer_result",
    "plot_error_rate",
    "plot_exit_chart",
    "plot_eye_diagram",
    "plot_filter_response",
    "plot_frequency_drift",
    "plot_frequency_offset_estimate",
    "plot_ideal_constellation",
    "plot_linewidth_estimate",
    "plot_llr_distribution",
    "plot_parity_check",
    "plot_psd",
    "plot_spectrogram",
    "plot_time_domain",
    "plot_timing_estimate",
]
