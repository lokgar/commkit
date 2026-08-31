"""
Signal visualization and publication-quality plotting tools.

This package provides high-level plotting functions optimized for
communication signals. It leverages Matplotlib to produce high-density,
professional diagrams with automatic SI scaling and backend-agnostic data
handling.

The public API is unchanged from when this was a single module:
``from commkit.plotting import plot_constellation, plot_psd, ...`` continues
to work.
"""

from __future__ import annotations

# Re-exported so ``patch("commkit.plotting.logger...")`` and similar
# attribute access on the package namespace keep working.
from ..logger import logger
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
    plot_filter_response,
    plot_zf_equalizer_response,
)

# Private helpers re-exported for tests that reach package internals through
# this namespace. F401 is silenced for this re-export hub in pyproject.toml.
from .eye import _plot_eye_traces, plot_eye_diagram
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
from .theme import _create_subplot_grid, apply_default_theme
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
