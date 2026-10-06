"""API contract registry: one row per public callable, generic checks per row.

Every public function and class has a row in ``ROWS``. The row records what
kind of API it is and how many leading *data* arguments it takes. Rows for
processing functions also carry a ``call`` builder, so behavioral checks can
invoke them.

The checks encode the 2.0 API rules in AGENTS.md section 4:

- ``signature``: every parameter after the data arguments is keyword-only.
- ``array_roundtrip``: a waveform transform returns an array on the input's
  device, with the input's rank.
- ``signal_roundtrip``: a Signal-aware transform returns a new Signal and
  leaves the input Signal unchanged.
- ``no_mutation``: input arrays are not modified.
- ``foreign_array``: array types other than NumPy/CuPy raise ``TypeError``.
- ``fact_conflict``: a Signal fact passed again with a different value raises
  ``ValueError``.
- ``trajectory``: per-symbol estimates keep the input's shape and device.
- ``estimate_rank``: per-channel estimates reduce the time axis, giving 0-d for
  ``(N,)`` and ``(C,)`` for ``(C, N)``, and stay on the input device.
- ``metric_host``: metrics return a ``float`` for 1-D input and an
  ``np.ndarray (C,)`` for 2-D input.
- ``equalizer_result``: ``EqualizerResult.y_hat`` is an array on the input
  device, and ``result.signal`` is a Signal for Signal input.

``LEGACY`` (at the end of this file) lists, per function, the rules the code
does not follow yet. Those checks run as ``xfail(strict=True)``, and an entry
ending in ``@gpu`` or ``@cpu`` applies to one device only. A strict xfail that
starts passing fails the suite, so the module pass that fixes a rule must
delete its entry. ``LEGACY`` is the live migration checklist.

Rows of kind ``REMOVE`` are scheduled for deletion and are not checked. Once a
function is deleted, ``test_rows_resolve`` fails until its row is deleted too.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np
import pytest

import commkit
from commkit.backend import to_device
from commkit.core import Signal
from commkit.equalization import EqualizerResult
from commkit.frequency import MthPower
from commkit.impairments import Lowdin
from commkit.mapping import Constellation
from commkit.recovery import BPS

# -----------------------------------------------------------------------------
# Kinds and checks
# -----------------------------------------------------------------------------

TRANSFORM = "transform"  # waveform/symbols in, same-length out
RATE_CHANGE = "rate_change"  # waveform in, resampled or trimmed waveform out
TRAJECTORY = "trajectory"  # per-symbol estimate (e.g. carrier phase)
ESTIMATE = "estimate"  # per-channel estimate (array, or an Estimate with .value)
METRIC = "metric"  # reporting-layer figure of merit
EQUALIZER = "equalizer"  # returns EqualizerResult
MULTI = "multi"  # returns several values (2.0: a frozen dataclass)
DESIGN = "design"  # builds coefficients/sequences/values from parameters
SYNTHESIS = "synthesis"  # generates signals or data from parameters
PLOT = "plot"
VALUE = "value"  # classes / value objects
HELPER = "helper"  # array/math utilities (moving out of helpers.py)
INFRA = "infra"  # backend, logging, io
REMOVE = "remove"  # scheduled for deletion in 2.0

BEHAVIORAL = {TRANSFORM, RATE_CHANGE, TRAJECTORY, ESTIMATE, METRIC, EQUALIZER, MULTI}

SIG = "signature"
ARR = "array_roundtrip"
SGN = "signal_roundtrip"
MUT = "no_mutation"
FOR = "foreign_array"
FCT = "fact_conflict"
TRJ = "trajectory"
RNK = "estimate_rank"
MET = "metric_host"
EQR = "equalizer_result"

# Placeholder modules for planned work are kept on purpose and not registered.
PLACEHOLDERS = ("commkit.coding", "commkit.impairments.channel.nonlinear")

Builder = Callable[["Inputs", Any], tuple[tuple, dict]]


@dataclass(frozen=True)
class Row:
    target: str  # defining module + qualname
    kind: str
    data: int = 1  # leading parameters that may stay positional
    call: Builder | None = None
    fact: tuple[str, Any] | None = None  # (kwarg, conflicting value)
    dims: tuple[int, ...] = (1, 2)  # accepted input ranks: (N,) and/or (C, N)
    symbols: bool = False  # primary input is 1-SPS symbols, not a waveform

    @property
    def obj(self) -> Any:
        module, _, name = self.target.rpartition(".")
        return getattr(importlib.import_module(module), name)

    @property
    def signal_aware(self) -> bool:
        """The first parameter accepts a Signal: ``ArrayType | Signal`` or the
        transform TypeVar ``S`` (bound to ``np.ndarray | Signal``)."""
        params = list(inspect.signature(self.obj).parameters.values())
        if not params:
            return False
        annotation = params[0].annotation
        if annotation == "S":  # postponed annotations
            return True
        if isinstance(annotation, TypeVar):
            annotation = annotation.__bound__
        return "Signal" in str(annotation)

    def __str__(self) -> str:
        return self.target.removeprefix("commkit.")


# -----------------------------------------------------------------------------
# Shared inputs
# -----------------------------------------------------------------------------

SPS = 2
N_SYM = 512
RS = 1e9
FS = RS * SPS
QAM16 = {"modulation": "qam", "order": 16}
C16 = Constellation.qam(16)


class Inputs:
    """Deterministic 16-QAM test data on the active backend."""

    def __init__(self, xp: Any):
        rng = np.random.default_rng(0)
        const = Constellation.qam(16).points.astype(np.complex64)
        sym = const[rng.integers(0, 16, (2, N_SYM))]
        wave = np.repeat(sym, SPS, axis=-1)
        noise = rng.standard_normal(wave.shape) + 1j * rng.standard_normal(wave.shape)
        self.xp = xp
        self.sym2 = xp.asarray(sym.astype(np.complex64))
        self.sym1 = self.sym2[0].copy()
        self.wave2 = xp.asarray((wave + 0.02 * noise).astype(np.complex64))
        self.wave1 = self.wave2[0].copy()
        self.taps = xp.asarray(np.array([0.1, 1.0, 0.1], dtype=np.float32))
        self.bits1 = xp.asarray(rng.integers(0, 2, N_SYM * 4).astype(np.int8))
        self.bits2 = xp.asarray(rng.integers(0, 2, (2, N_SYM * 4)).astype(np.int8))

    def primary(self, row: Row, ndim: int) -> Any:
        """The row's primary array input with ``ndim`` dimensions."""
        if row.symbols:
            return self.sym1 if ndim == 1 else self.sym2
        return self.wave1 if ndim == 1 else self.wave2

    def primary_signal(self, row: Row, ndim: int) -> Signal:
        """The row's primary input as a Signal (1-SPS for symbol inputs)."""
        sps = 1 if row.symbols else SPS
        return Signal(
            samples=self.primary(row, ndim),
            sampling_rate=RS * sps,
            symbol_rate=RS,
            constellation=Constellation.qam(16),
        )


def _a(*args: Any, **kwargs: Any) -> tuple[tuple, dict]:
    return args, kwargs


def _arr(x: Any) -> Any:
    return x.samples if isinstance(x, Signal) else x


def _ref(c: Inputs, x: Any) -> Any:
    """Known symbols with the channel count of ``x``."""
    return c.sym1 if _arr(x).ndim == 1 else c.sym2


def _bits(c: Inputs, x: Any) -> Any:
    return c.bits1 if _arr(x).ndim == 1 else c.bits2


def _identity_taps(c: Inputs, x: Any) -> Any:
    """Centre-tap identity: (T,) for SISO, (C, C, T) for MIMO."""
    if _arr(x).ndim == 1:
        return c.xp.asarray(np.array([0, 0, 1, 0, 0], np.complex64))
    w = np.zeros((2, 2, 5), np.complex64)
    w[0, 0, 2] = w[1, 1, 2] = 1
    return c.xp.asarray(w)


def _pilots(x: Any) -> dict:
    return {
        "pilot_indices": np.arange(0, N_SYM, 8),
        "pilot_values": to_device(_arr(x), "cpu")[..., ::8],
    }


def L(*checks: str) -> frozenset[str]:
    return frozenset(checks)


CD = {"dispersion_ps_nm_km": 17, "fiber_length_km": 1, "center_wavelength_nm": 1550}
FS_CONFLICT = ("sampling_rate", 3 * FS)
SPS_CONFLICT = ("sps", 4)

# -----------------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------------

ROWS: list[Row] = [
    # --- analysis (array-only: inputs are derived quantities) ---------------
    Row("commkit.analysis.allan.allan_deviation", MULTI),
    Row("commkit.analysis.drift.frequency_drift_metrics", MULTI),
    Row("commkit.analysis.drift.separate_drift_phase_noise", MULTI),
    Row("commkit.analysis.interferometry.dsh_beat", DESIGN),
    Row("commkit.analysis.interferometry.dsh_fm_noise_psd", MULTI),
    Row("commkit.analysis.interferometry.dsh_phase", MULTI),
    Row("commkit.analysis.interferometry.linewidth_dsh", MULTI),
    Row("commkit.analysis.linewidth.fm_noise_psd", MULTI),
    Row("commkit.analysis.linewidth.linewidth_beta_separation", MULTI),
    Row("commkit.analysis.linewidth.linewidth_increment", MULTI),
    Row("commkit.analysis.trajectory.carrier_phase_trajectory", TRAJECTORY, data=2),
    # --- backend / logging / io --------------------------------------------
    Row("commkit.backend.dispatch", INFRA),
    Row("commkit.backend.get_array_module", INFRA),
    Row("commkit.backend.is_cupy_available", INFRA, data=0),
    Row("commkit.backend.to_device", INFRA, data=2),
    Row("commkit.logger.set_log_level", INFRA),
    Row("commkit.io.load_npz", INFRA),
    Row("commkit.io.save_npz", INFRA, data=2),
    # --- core ---------------------------------------------------------------
    Row("commkit.core.frame.Preamble", VALUE),
    Row("commkit.core.signal.Reference", VALUE),
    Row("commkit.core.frame.SingleCarrierFrame", VALUE),
    Row("commkit.core.signal.Signal", VALUE),
    Row("commkit.core.generation.expand", DESIGN),
    Row("commkit.core.generation.shape_pulse", DESIGN),
    # constellation and num_symbols are positional (plan 2.6).
    Row("commkit.core.generation.generate", SYNTHESIS, data=2),
    # --- equalization -------------------------------------------------------
    Row(
        "commkit.equalization.sequential._dd.lms",
        EQUALIZER,
        data=2,
        call=lambda c, x: _a(x, _ref(c, x)[..., :200], num_taps=5, sps=SPS, **QAM16),
        fact=SPS_CONFLICT,
    ),
    Row(
        "commkit.equalization.sequential._dd.rls",
        EQUALIZER,
        data=2,
        call=lambda c, x: _a(x, _ref(c, x)[..., :200], num_taps=5, sps=SPS, **QAM16),
        fact=SPS_CONFLICT,
    ),
    Row(
        "commkit.equalization.sequential._blind.cma",
        EQUALIZER,
        call=lambda c, x: _a(x, num_taps=5, sps=SPS, **QAM16),
        fact=SPS_CONFLICT,
    ),
    Row(
        "commkit.equalization.sequential._blind.rde",
        EQUALIZER,
        call=lambda c, x: _a(x, num_taps=5, sps=SPS, **QAM16),
        fact=SPS_CONFLICT,
    ),
    Row("commkit.equalization.sequential._dd._check_rls_divergence", REMOVE),
    Row(
        "commkit.equalization._block._dd.block_lms",
        EQUALIZER,
        data=2,
        call=lambda c, x: _a(
            x, _ref(c, x)[..., :256], num_taps=5, sps=SPS, block_size=64, **QAM16
        ),
        fact=SPS_CONFLICT,
    ),
    Row(
        "commkit.equalization.blind.block_cma",
        EQUALIZER,
        call=lambda c, x: _a(x, num_taps=5, sps=SPS, block_size=64, **QAM16),
        fact=SPS_CONFLICT,
    ),
    Row(
        "commkit.equalization.blind.block_rde",
        EQUALIZER,
        call=lambda c, x: _a(x, num_taps=5, sps=SPS, block_size=64, **QAM16),
        fact=SPS_CONFLICT,
    ),
    Row("commkit.equalization.blind.build_pilot_ref", DESIGN, data=2),
    Row(
        "commkit.equalization.linear.apply_taps",
        RATE_CHANGE,
        data=2,
        call=lambda c, x: _a(x, _identity_taps(c, x), sps=SPS),
        fact=SPS_CONFLICT,
    ),
    Row("commkit.equalization.linear.estimate_transfer_function", MULTI, data=2),
    Row(
        "commkit.equalization.linear.zf_equalizer",
        TRANSFORM,
        data=2,
        call=lambda c, x: _a(x, c.taps),
        dims=(1,),
    ),
    Row(
        "commkit.equalization.polarization.apply_interpolated_matrix", TRANSFORM, data=3
    ),
    Row(
        "commkit.equalization.polarization.demultiplex_polarization_tones_dynamic",
        MULTI,
    ),
    Row(
        "commkit.equalization.polarization.demultiplex_polarization_tones_static",
        MULTI,
    ),
    Row("commkit.equalization.result.CPRState", VALUE),
    Row("commkit.equalization.result.EqualizerResult", VALUE),
    # --- filtering ----------------------------------------------------------
    Row("commkit.filtering.Gaussian", VALUE),
    Row("commkit.filtering.Pulse", VALUE),
    Row("commkit.filtering.RC", VALUE),
    Row("commkit.filtering.RRC", VALUE),
    Row("commkit.filtering.Rect", VALUE),
    Row("commkit.filtering.SmoothRect", VALUE),
    Row("commkit.filtering.bessel_sos", DESIGN, data=0),
    Row("commkit.filtering.butterworth_sos", DESIGN, data=0),
    Row("commkit.filtering.chebyshev1_sos", DESIGN, data=0),
    Row("commkit.filtering.chebyshev2_sos", DESIGN, data=0),
    Row("commkit.filtering.elliptic_sos", DESIGN, data=0),
    Row("commkit.filtering.fir_taps", DESIGN, data=0),
    Row("commkit.filtering.gaussian_taps", DESIGN, data=0),
    Row("commkit.filtering.rc_taps", DESIGN, data=0),
    Row("commkit.filtering.rect_taps", DESIGN, data=0),
    Row("commkit.filtering.rrc_taps", DESIGN, data=0),
    Row("commkit.filtering.smoothrect_taps", DESIGN, data=0),
    Row(
        "commkit.filtering.fir_filter",
        TRANSFORM,
        data=2,
        call=lambda c, x: _a(x, c.taps),
    ),
    Row(
        "commkit.filtering.ols_fir_filter",
        TRANSFORM,
        data=2,
        call=lambda c, x: _a(x, c.taps),
    ),
    Row(
        "commkit.filtering.iir_filter",
        TRANSFORM,
        data=2,
        call=lambda c, x: _a(
            x, c.xp.asarray(np.array([[0.2, 0.2, 0, 1, -0.6, 0]], np.float64))
        ),
    ),
    Row(
        "commkit.filtering.matched_filter",
        TRANSFORM,
        call=lambda c, x: _a(x, pulse=c.taps),
    ),
    Row(
        "commkit.filtering.correct_chromatic_dispersion",
        TRANSFORM,
        call=lambda c, x: _a(x, sampling_rate=FS, **CD),
        fact=FS_CONFLICT,
    ),
    # --- frequency ----------------------------------------------------------
    Row("commkit.frequency.BiasTone", VALUE),
    Row("commkit.frequency.FrequencyOffsetEstimate", VALUE),
    Row("commkit.frequency.MengaliMorelli", VALUE),
    Row("commkit.frequency.MthPower", VALUE),
    Row("commkit.frequency.PilotSymbols", VALUE),
    Row(
        "commkit.frequency.correct_frequency_offset",
        TRANSFORM,
        data=2,
        call=lambda c, x: _a(x, 1e6, sampling_rate=FS),
        fact=FS_CONFLICT,
    ),
    Row(
        "commkit.frequency.estimate_frequency_offset",
        ESTIMATE,
        data=2,
        call=lambda c, x: _a(x, MthPower(), sampling_rate=FS, constellation=C16),
        fact=FS_CONFLICT,
    ),
    # --- math -------------------------------------------------------------
    Row("commkit.math.db_to_linear", HELPER),
    Row("commkit.math.linear_to_db", HELPER),
    Row("commkit.math.normalize", HELPER),
    Row("commkit.math.rms", HELPER),
    # --- helpers (dissolved into owning modules in the module passes) -------
    Row("commkit.helpers.linear_trend_slope", HELPER),
    Row("commkit.helpers.remove_linear_trend", HELPER),
    # --- impairments --------------------------------------------------------
    Row(
        "commkit.impairments.channel.linear.apply_chromatic_dispersion",
        TRANSFORM,
        call=lambda c, x: _a(x, sampling_rate=FS, **CD),
        fact=FS_CONFLICT,
    ),
    Row(
        "commkit.impairments.channel.linear.apply_pmd",
        TRANSFORM,
        call=lambda c, x: _a(x, sampling_rate=FS, dgd=1e-12, theta=0.3),
        fact=FS_CONFLICT,
        dims=(2,),
    ),
    Row(
        "commkit.impairments.channel.linear.apply_polarization_mixing",
        TRANSFORM,
        call=lambda c, x: _a(x, theta=0.3),
        dims=(2,),
    ),
    Row(
        "commkit.impairments.frontend.apply_iq_imbalance",
        TRANSFORM,
        call=lambda c, x: _a(x, amplitude_imbalance_db=0.5, phase_imbalance_deg=3),
    ),
    Row("commkit.impairments.frontend.GramSchmidt", VALUE),
    Row("commkit.impairments.frontend.Lowdin", VALUE),
    Row(
        "commkit.impairments.frontend.correct_iq_imbalance",
        TRANSFORM,
        data=2,
        call=lambda c, x: _a(x, Lowdin()),
    ),
    Row(
        "commkit.impairments.noise.apply_awgn",
        TRANSFORM,
        call=lambda c, x: _a(x, sps=SPS, esn0_db=20, rng=1),
        fact=SPS_CONFLICT,
    ),
    Row(
        "commkit.impairments.source.apply_phase_noise",
        TRANSFORM,
        call=lambda c, x: _a(x, sampling_rate=FS, linewidth=1e5, rng=1),
        fact=FS_CONFLICT,
    ),
    Row("commkit.impairments.source.generate_phase_noise", SYNTHESIS, data=0),
    # --- mapping ------------------------------------------------------------
    Row(
        "commkit.mapping.bits.demap_symbols_hard",
        MULTI,
        call=lambda c, x: _a(x, constellation=C16),
        symbols=True,
    ),
    Row("commkit.mapping.bits.map_bits", DESIGN),
    Row("commkit.mapping.constellation.Constellation", VALUE),
    Row("commkit.mapping.gray.gray_code", DESIGN),
    Row("commkit.mapping.gray.gray_to_binary", DESIGN),
    Row(
        "commkit.mapping.llr.compute_llr",
        MULTI,
        call=lambda c, x: _a(x, noise_var=0.1, constellation=C16),
        symbols=True,
    ),
    Row("commkit.mapping.shaping.maxwell_boltzmann", DESIGN),
    Row("commkit.mapping.shaping.optimal_nu", DESIGN),
    # --- metrics (Signal semantics are redefined in pass 3.8) ----------------
    Row(
        "commkit.metrics.ber",
        METRIC,
        data=2,
        call=lambda c, x: _a(_bits(c, x), _bits(c, x)),
        symbols=True,
    ),
    Row(
        "commkit.metrics.evm",
        METRIC,
        data=2,
        call=lambda c, x: _a(x * 1.01, _ref(c, x)),
        symbols=True,
    ),
    Row(
        "commkit.metrics.snr",
        METRIC,
        data=2,
        call=lambda c, x: _a(x * 1.01, _ref(c, x)),
        symbols=True,
    ),
    Row(
        "commkit.metrics.ser",
        METRIC,
        data=2,
        call=lambda c, x: _a(x * 1.01, _ref(c, x), **QAM16),
        symbols=True,
    ),
    Row("commkit.metrics.gmi", METRIC, data=2),
    Row(
        "commkit.metrics.mi",
        METRIC,
        call=lambda c, x: _a(x, noise_var=0.1, **QAM16),
        symbols=True,
    ),
    # --- multirate ----------------------------------------------------------
    Row(
        "commkit.multirate.decimate",
        RATE_CHANGE,
        call=lambda c, x: _a(x, factor=2),
    ),
    Row(
        "commkit.multirate.decimate_to_symbol_rate",
        RATE_CHANGE,
        call=lambda c, x: _a(x, sps=SPS),
        fact=SPS_CONFLICT,
    ),
    Row(
        "commkit.multirate.resample",
        RATE_CHANGE,
        call=lambda c, x: _a(x, sps_in=SPS, sps_out=3),
        fact=("sps_in", 4),
    ),
    Row(
        "commkit.multirate.resolve_symbols",
        RATE_CHANGE,
        call=lambda c, x: _a(x, sps=SPS),
        fact=SPS_CONFLICT,
    ),
    Row(
        "commkit.multirate.upsample",
        RATE_CHANGE,
        call=lambda c, x: _a(x, factor=2),
    ),
    # --- plotting (consume data; never produce it) ---------------------------
    Row("commkit.plotting.analysis.plot_allan_deviation", PLOT, data=2),
    Row("commkit.plotting.analysis.plot_carrier_phase_characterization", PLOT),
    Row("commkit.plotting.analysis.plot_dsh_beat_psd", PLOT, data=2),
    Row("commkit.plotting.analysis.plot_frequency_drift", PLOT),
    Row("commkit.plotting.analysis.plot_frequency_noise_psd", PLOT, data=2),
    Row("commkit.plotting.analysis.plot_increment_variance", PLOT, data=2),
    Row("commkit.plotting.constellation.plot_constellation", PLOT),
    Row("commkit.plotting.constellation.plot_ideal_constellation", PLOT),
    Row("commkit.plotting.equalizer.plot_equalizer_result", PLOT),
    Row("commkit.plotting.equalizer.plot_zf_equalizer_response", PLOT),
    Row("commkit.plotting.eye.plot_eye_diagram", PLOT),
    Row("commkit.plotting.filter_response.plot_filter_response", PLOT),
    Row("commkit.plotting.spectral.plot_psd", PLOT),
    Row("commkit.plotting.spectral.plot_spectrogram", PLOT),
    Row("commkit.plotting.sync.plot_carrier_phase_decomposition", PLOT, data=2),
    Row("commkit.plotting.sync.plot_carrier_phase_trajectory", PLOT),
    Row("commkit.plotting.sync.plot_frequency_offset_blockwise_result", PLOT, data=2),
    Row("commkit.plotting.sync.plot_frequency_offset_spectrum", PLOT, data=2),
    Row("commkit.plotting.sync.plot_mm_autocorrelation", PLOT),
    Row("commkit.plotting.sync.plot_pilot_phase_estimate", PLOT, data=3),
    Row("commkit.plotting.sync.plot_pilot_tone_phase_estimate", PLOT, data=2),
    Row("commkit.plotting.sync.plot_pilot_tones_phase_estimate", PLOT, data=2),
    Row("commkit.plotting.sync.plot_timing_correlation", PLOT),
    Row("commkit.plotting.theme.apply_default_theme", PLOT, data=0),
    Row("commkit.plotting.waveform.plot_time_domain", PLOT),
    # --- recovery -----------------------------------------------------------
    Row("commkit.recovery.bps.BPS", VALUE),
    Row("commkit.recovery.carrier_phase.CarrierPhaseEstimate", VALUE),
    Row(
        "commkit.recovery.carrier_phase.correct_carrier_phase",
        TRANSFORM,
        data=2,
        call=lambda c, x: _a(x, 0.1),
        symbols=True,
    ),
    Row(
        "commkit.recovery.carrier_phase.estimate_carrier_phase",
        TRAJECTORY,
        data=2,
        call=lambda c, x: _a(x, BPS(), constellation=C16),
        fact=("sampling_rate", 3 * RS),
        symbols=True,
    ),
    Row("commkit.recovery.corrections.CycleSlip", VALUE),
    Row("commkit.recovery.corrections.DataAided", VALUE),
    Row("commkit.recovery.pilots.PilotAided", VALUE),
    Row("commkit.recovery.pilots.PilotTone", VALUE),
    Row("commkit.recovery.pilots.PilotTones", VALUE),
    Row("commkit.recovery.pll.PLL", VALUE),
    Row("commkit.recovery.tikhonov.Tikhonov", VALUE),
    Row("commkit.recovery.viterbi_viterbi.ViterbiViterbi", VALUE),
    Row("commkit.recovery.corrections.correct_cycle_slips", DESIGN),
    Row(
        "commkit.recovery.corrections.resolve_channel_permutation",
        TRANSFORM,
        data=2,
        call=lambda c, x: _a(x, _ref(c, x)),
        symbols=True,
    ),
    Row(
        "commkit.recovery.corrections.resolve_phase_ambiguity",
        TRANSFORM,
        data=2,
        call=lambda c, x: _a(x, _ref(c, x), **QAM16),
        symbols=True,
    ),
    Row("commkit.recovery.corrections.smooth_phase_wiener", TRAJECTORY),
    # --- smoothing ----------------------------------------------------------
    Row("commkit.smoothing.moving_average", HELPER),
    Row("commkit.smoothing.savgol_smooth", HELPER),
    Row("commkit.smoothing.smooth_density_2d", HELPER),
    # --- spectral -----------------------------------------------------------
    Row(
        "commkit.spectral.add_pilot_tone",
        TRANSFORM,
        call=lambda c, x: _a(x, sampling_rate=FS, frequency=1e8, power_ratio_db=-10),
        fact=FS_CONFLICT,
    ),
    Row(
        "commkit.spectral.shift_frequency",
        TRANSFORM,
        call=lambda c, x: _a(x, frequency=1e7, sampling_rate=FS),
        fact=FS_CONFLICT,
    ),
    Row("commkit.spectral.grid_frequency", DESIGN, data=1),
    Row("commkit.spectral.Spectrogram", VALUE),
    Row(
        "commkit.spectral.spectrogram",
        MULTI,
        call=lambda c, x: _a(x, sampling_rate=FS, nperseg=64),
        fact=FS_CONFLICT,
    ),
    Row(
        "commkit.spectral.welch_psd",
        MULTI,
        call=lambda c, x: _a(x, sampling_rate=FS, nperseg=64),
        fact=FS_CONFLICT,
    ),
    # --- timing -------------------------------------------------------------
    Row("commkit._sequences.barker_sequence", DESIGN),
    Row("commkit._sequences.zadoff_chu_sequence", DESIGN),
    Row("commkit.timing.cross_correlate_fft", HELPER, data=2),
    Row(
        "commkit.timing.correct_timing",
        RATE_CHANGE,
        data=2,
        call=lambda c, x: _a(x, 3),
    ),
    Row("commkit.timing.estimate_fractional_delay", ESTIMATE, data=2),
    Row("commkit.timing.TimingEstimate", VALUE),
    Row(
        "commkit.timing.estimate_timing",
        ESTIMATE,
        call=lambda c, x: _a(x, template=c.wave2[0, :64], threshold=1.0),
    ),
    Row(
        "commkit.timing.fft_fractional_delay",
        TRANSFORM,
        call=lambda c, x: _a(x, delay=0.25),
    ),
]

ROW_BY_TARGET = {r.target: r for r in ROWS}


def _rows(kinds: set[str], *, predicate: Callable[[Row], bool] | None = None) -> list:
    return [
        pytest.param(r, id=str(r))
        for r in ROWS
        if r.kind in kinds
        and r.call is not None
        and (predicate is None or predicate(r))
    ]


def _expect(request: pytest.FixtureRequest, row: Row, check: str, device: str) -> None:
    """Mark the running check as a strict xfail when ``LEGACY`` lists it."""
    legacy = LEGACY.get(row.target, frozenset())
    if check in legacy or f"{check}@{device}" in legacy:
        request.applymarker(
            pytest.mark.xfail(strict=True, reason=f"2.0 rule not yet met: {check}")
        )


# -----------------------------------------------------------------------------
# Registry completeness
# -----------------------------------------------------------------------------


def _public_callables() -> dict[str, Any]:
    """Public callables: a module's ``__all__``, else its own public functions."""
    found: dict[str, Any] = {}
    for info in pkgutil.walk_packages(commkit.__path__, "commkit."):
        parts = info.name.split(".")[1:]
        if any(p.startswith("_") for p in parts) or info.name.startswith(PLACEHOLDERS):
            continue
        mod = importlib.import_module(info.name)
        names = getattr(mod, "__all__", None)
        if names is None:
            names = [
                n
                for n, o in vars(mod).items()
                if not n.startswith("_")
                and (inspect.isfunction(o) or inspect.isclass(o))
                and o.__module__ == info.name
            ]
        for n in names:
            obj = getattr(mod, n)
            if inspect.ismodule(obj) or not callable(obj):
                continue
            found[f"{obj.__module__}.{obj.__qualname__}"] = obj
    return found


def test_every_public_callable_is_registered():
    missing = sorted(set(_public_callables()) - set(ROW_BY_TARGET))
    assert not missing, (
        "Public callables missing from tests/test_api_contracts.py ROWS "
        f"(add a row with its kind and data-argument count): {missing}"
    )


def test_rows_resolve():
    """Every row names an existing object; delete rows of removed functions."""
    stale = []
    for r in ROWS:
        try:
            _ = r.obj
        except (ImportError, AttributeError):
            stale.append(r.target)
    assert not stale, f"Registry rows for objects that no longer exist: {stale}"


def test_rows_are_unique():
    assert len(ROW_BY_TARGET) == len(ROWS)


def test_legacy_entries_are_valid():
    checks = {SIG, ARR, SGN, MUT, FOR, FCT, TRJ, RNK, MET, EQR}
    for target, entries in LEGACY.items():
        assert target in ROW_BY_TARGET, f"LEGACY entry for unregistered {target}"
        for e in entries:
            assert e.partition("@")[0] in checks, f"{target}: unknown check {e!r}"


@pytest.mark.parametrize(
    "row", [pytest.param(r, id=str(r)) for r in ROWS if r.call is not None]
)
def test_call_builder_smoke(row: Row, xp):
    """Each builder's plain array call works, so check failures are real."""
    c = Inputs(xp)
    for ndim in row.dims:
        row.call(c, c.primary(row, ndim))  # builder itself
        args, kwargs = row.call(c, c.primary(row, ndim))
        row.obj(*args, **kwargs)


# -----------------------------------------------------------------------------
# Signature rule
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "row",
    [
        pytest.param(r, id=str(r))
        for r in ROWS
        if r.kind not in (VALUE, REMOVE) and inspect.isfunction(r.obj)
    ],
)
def test_signature_keyword_only(row: Row, request):
    _expect(request, row, SIG, "any")
    params = [
        p
        for p in inspect.signature(row.obj).parameters.values()
        if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ]
    positional = [p.name for p in params[row.data :] if p.kind != p.KEYWORD_ONLY]
    assert not positional, f"must be keyword-only: {positional}"


# -----------------------------------------------------------------------------
# Behavioral rules
# -----------------------------------------------------------------------------


class _ForeignArray:
    """An array from another framework: array protocols, but not NumPy/CuPy."""

    def __init__(self, data: Any):
        self._a = np.asarray(to_device(data, "cpu"))
        self.shape, self.dtype, self.ndim = self._a.shape, self._a.dtype, self._a.ndim

    def __array__(self, dtype=None, copy=None):
        return self._a if dtype is None else self._a.astype(dtype)

    def __dlpack__(self, **kwargs):
        return self._a.__dlpack__(**kwargs)

    def __dlpack_device__(self):
        return self._a.__dlpack_device__()

    def __len__(self):
        return len(self._a)


def _call(row: Row, c: Inputs, x: Any) -> Any:
    args, kwargs = row.call(c, x)
    return row.obj(*args, **kwargs)


def _module(a: Any) -> str:
    return type(a).__module__.split(".")[0]


@pytest.mark.parametrize("row", _rows({TRANSFORM, RATE_CHANGE}))
def test_array_roundtrip(row: Row, xp, backend_device, request):
    _expect(request, row, ARR, backend_device)
    c = Inputs(xp)
    for ndim in row.dims:
        out = _call(row, c, c.primary(row, ndim))
        assert _module(out) == xp.__name__, f"returned {type(out)}"
        assert out.ndim == ndim


@pytest.mark.parametrize(
    "row",
    _rows({TRANSFORM, RATE_CHANGE, TRAJECTORY}, predicate=lambda r: r.signal_aware),
)
def test_signal_roundtrip(row: Row, xp, backend_device, request):
    """Transforms return a new Signal; estimates return their documented type."""
    _expect(request, row, SGN, backend_device)
    c = Inputs(xp)
    sig = c.primary_signal(row, row.dims[0])
    before = to_device(sig.samples, "cpu").copy()
    out = _call(row, c, sig)
    if row.kind == TRAJECTORY:
        out = getattr(out, "value", out)  # CarrierPhaseEstimate
        assert _module(out) == xp.__name__, f"returned {type(out)}"
    else:
        assert isinstance(out, Signal), f"returned {type(out)}"
        assert out is not sig
    np.testing.assert_array_equal(to_device(sig.samples, "cpu"), before)


@pytest.mark.parametrize("row", _rows(BEHAVIORAL))
def test_no_mutation(row: Row, xp, backend_device, request):
    _expect(request, row, MUT, backend_device)
    c = Inputs(xp)
    snapshot = {
        k: to_device(v, "cpu").copy() for k, v in vars(c).items() if hasattr(v, "shape")
    }
    for ndim in row.dims:
        _call(row, c, c.primary(row, ndim))
    for k, v in snapshot.items():
        np.testing.assert_array_equal(to_device(getattr(c, k), "cpu"), v, err_msg=k)


@pytest.mark.parametrize("row", _rows(BEHAVIORAL))
def test_foreign_array_rejected(row: Row, xp, backend_device, request):
    _expect(request, row, FOR, backend_device)
    c = Inputs(xp)
    args, kwargs = row.call(c, c.primary(row, row.dims[0]))
    args = (_ForeignArray(args[0]), *args[1:])
    with pytest.raises(TypeError):
        row.obj(*args, **kwargs)


@pytest.mark.parametrize(
    "row",
    _rows(BEHAVIORAL, predicate=lambda r: r.fact is not None and r.signal_aware),
)
def test_fact_conflict_raises(row: Row, xp, backend_device, request):
    _expect(request, row, FCT, backend_device)
    c = Inputs(xp)
    args, kwargs = row.call(c, c.primary_signal(row, row.dims[0]))
    name, wrong = row.fact
    kwargs[name] = wrong
    with pytest.raises(ValueError):
        row.obj(*args, **kwargs)


@pytest.mark.parametrize("row", _rows({TRAJECTORY}))
def test_trajectory_shape_and_device(row: Row, xp, backend_device, request):
    _expect(request, row, TRJ, backend_device)
    c = Inputs(xp)
    for ndim in row.dims:
        x = c.primary(row, ndim)
        out = _call(row, c, x)
        out = getattr(out, "value", out)  # CarrierPhaseEstimate
        assert _module(out) == xp.__name__, f"returned {type(out)}"
        assert out.shape == x.shape


@pytest.mark.parametrize("row", _rows({ESTIMATE}))
def test_estimate_rank_rule(row: Row, xp, backend_device, request):
    _expect(request, row, RNK, backend_device)
    c = Inputs(xp)
    for ndim in row.dims:
        out = _call(row, c, c.primary(row, ndim))
        out = getattr(out, "value", out)  # <Quantity>Estimate dataclasses
        assert _module(out) == xp.__name__, f"returned {type(out)}"
        assert out.shape == (() if ndim == 1 else (2,))


@pytest.mark.parametrize("row", _rows({METRIC}))
def test_metric_host_values(row: Row, xp, backend_device, request):
    _expect(request, row, MET, backend_device)
    c = Inputs(xp)
    for ndim in row.dims:
        out = _call(row, c, c.primary(row, ndim))
        if ndim == 1:
            assert type(out) is float, f"returned {type(out)}"
        else:
            assert isinstance(out, np.ndarray), f"returned {type(out)}"
            assert out.shape == (2,)


@pytest.mark.parametrize("row", _rows({EQUALIZER}))
def test_equalizer_result(row: Row, xp, backend_device, request):
    _expect(request, row, EQR, backend_device)
    c = Inputs(xp)
    res = _call(row, c, c.wave1)
    assert isinstance(res, EqualizerResult)
    assert _module(res.y_hat) == xp.__name__
    res_sig = _call(row, c, c.primary_signal(row, 1))
    assert _module(res_sig.y_hat) == xp.__name__, f"y_hat is {type(res_sig.y_hat)}"
    assert isinstance(getattr(res_sig, "signal", None), Signal)


# -----------------------------------------------------------------------------
# Migration checklist: 2.0 rules not met yet (strict xfail)
# -----------------------------------------------------------------------------

LEGACY: dict[str, frozenset[str]] = {
    "commkit.analysis.allan.allan_deviation": L(SIG),
    "commkit.analysis.drift.frequency_drift_metrics": L(SIG),
    "commkit.analysis.drift.separate_drift_phase_noise": L(SIG),
    "commkit.analysis.interferometry.dsh_beat": L(SIG),
    "commkit.analysis.interferometry.dsh_fm_noise_psd": L(SIG),
    "commkit.analysis.interferometry.dsh_phase": L(SIG),
    "commkit.analysis.interferometry.linewidth_dsh": L(SIG),
    "commkit.analysis.linewidth.fm_noise_psd": L(SIG),
    "commkit.analysis.linewidth.linewidth_beta_separation": L(SIG),
    "commkit.analysis.linewidth.linewidth_increment": L(SIG),
    "commkit.equalization._block._dd.block_lms": L(SIG, FCT, EQR),
    "commkit.equalization.blind.block_cma": L(SIG, FCT, EQR),
    "commkit.equalization.blind.block_rde": L(SIG, FCT, EQR),
    "commkit.equalization.blind.build_pilot_ref": L(SIG),
    "commkit.equalization.linear.apply_taps": L(SIG, FCT),
    "commkit.equalization.linear.zf_equalizer": L(SIG),
    "commkit.equalization.polarization.demultiplex_polarization_tones_dynamic": L(SIG),
    "commkit.equalization.polarization.demultiplex_polarization_tones_static": L(SIG),
    "commkit.equalization.sequential._blind.cma": L(SIG, FCT, EQR),
    "commkit.equalization.sequential._blind.rde": L(SIG, FCT, EQR),
    "commkit.equalization.sequential._dd.lms": L(SIG, FCT, EQR),
    "commkit.equalization.sequential._dd.rls": L(SIG, FCT, EQR),
    "commkit.metrics.ber": L(f"{MET}@gpu"),
    "commkit.metrics.evm": L(MET),
    "commkit.metrics.mi": L(SIG, MET),
    "commkit.metrics.ser": L(SIG, f"{MET}@gpu"),
    "commkit.metrics.snr": L(f"{MET}@gpu"),
    "commkit.plotting.constellation.plot_constellation": L(SIG),
    "commkit.plotting.constellation.plot_ideal_constellation": L(SIG),
    "commkit.plotting.equalizer.plot_equalizer_result": L(SIG),
    "commkit.plotting.equalizer.plot_zf_equalizer_response": L(SIG),
    "commkit.plotting.eye.plot_eye_diagram": L(SIG),
    "commkit.plotting.filter_response.plot_filter_response": L(SIG),
    "commkit.plotting.spectral.plot_psd": L(SIG),
    "commkit.plotting.spectral.plot_spectrogram": L(SIG),
    "commkit.plotting.sync.plot_carrier_phase_trajectory": L(SIG),
    "commkit.plotting.sync.plot_frequency_offset_blockwise_result": L(SIG),
    "commkit.plotting.sync.plot_frequency_offset_spectrum": L(SIG),
    "commkit.plotting.sync.plot_mm_autocorrelation": L(SIG),
    "commkit.plotting.sync.plot_pilot_phase_estimate": L(SIG),
    "commkit.plotting.sync.plot_pilot_tone_phase_estimate": L(SIG),
    "commkit.plotting.sync.plot_pilot_tones_phase_estimate": L(SIG),
    "commkit.plotting.sync.plot_timing_correlation": L(SIG),
    "commkit.plotting.waveform.plot_time_domain": L(SIG),
    "commkit.recovery.corrections.correct_cycle_slips": L(SIG),
    "commkit.recovery.corrections.resolve_channel_permutation": L(SGN),
    "commkit.recovery.corrections.resolve_phase_ambiguity": L(SIG, SGN),
    "commkit.recovery.corrections.smooth_phase_wiener": L(SIG),
}
