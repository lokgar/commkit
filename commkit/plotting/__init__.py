"""Plots of signals and of the library's results (Matplotlib).

Every public function is a ``plot_*`` that draws a Signal, an array or a
result object (an estimate, an :class:`~commkit.equalization.EqualizerResult`,
an analysis summary) and never computes the result itself; device arrays are
reduced on the device and only what is drawn is transferred.  Axes use SI
prefixes.  :func:`apply_default_theme` sets the library's Matplotlib style;
importing the package changes no Matplotlib settings.

- :mod:`~commkit.plotting.waveform`, :mod:`~commkit.plotting.eye`,
  :mod:`~commkit.plotting.constellation`, :mod:`~commkit.plotting.spectral`:
  the signal itself.
- :mod:`~commkit.plotting.filter_response`,
  :mod:`~commkit.plotting.equalizer`: filters and equalizers.
- :mod:`~commkit.plotting.sync`: timing, frequency-offset and carrier-phase
  estimates.
- :mod:`~commkit.plotting.analysis`: drift, Allan deviation and linewidth.
"""

from __future__ import annotations

from .analysis import (
    plot_allan_deviation,
    plot_carrier_phase_characterization,
    plot_dsh_beat_psd,
    plot_frequency_drift,
    plot_frequency_noise_psd,
    plot_increment_variance,
)
from .constellation import plot_constellation, plot_ideal_constellation
from .equalizer import (
    plot_equalizer_result,
    plot_zf_equalizer_response,
)
from .eye import plot_eye_diagram
from .filter_response import plot_filter_response
from .spectral import plot_psd, plot_spectrogram
from .sync import (
    plot_carrier_phase_decomposition,
    plot_carrier_phase_trajectory,
    plot_frequency_offset_blockwise_result,
    plot_frequency_offset_spectrum,
    plot_mm_autocorrelation,
    plot_pilot_phase_estimate,
    plot_pilot_tone_phase_estimate,
    plot_pilot_tones_phase_estimate,
    plot_timing_correlation,
)
from .theme import apply_default_theme
from .waveform import plot_time_domain

__all__ = [
    "apply_default_theme",
    "plot_allan_deviation",
    "plot_carrier_phase_characterization",
    "plot_carrier_phase_decomposition",
    "plot_carrier_phase_trajectory",
    "plot_constellation",
    "plot_dsh_beat_psd",
    "plot_equalizer_result",
    "plot_eye_diagram",
    "plot_filter_response",
    "plot_frequency_drift",
    "plot_frequency_noise_psd",
    "plot_frequency_offset_blockwise_result",
    "plot_frequency_offset_spectrum",
    "plot_ideal_constellation",
    "plot_increment_variance",
    "plot_mm_autocorrelation",
    "plot_pilot_phase_estimate",
    "plot_pilot_tone_phase_estimate",
    "plot_pilot_tones_phase_estimate",
    "plot_psd",
    "plot_spectrogram",
    "plot_time_domain",
    "plot_timing_correlation",
    "plot_zf_equalizer_response",
]
