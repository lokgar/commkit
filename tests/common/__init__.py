"""Shared test fixtures, signal generators, metrics, and conversion utilities."""

from tests.common.conversions import ensure_jax_x64, to_numpy
from tests.common.kernel_utils import (
    reference_bps_d2,
    reference_cs_block,
    reset_warned_kernels,
    skip_unless_kernel_available,
    slip_workload,
)
from tests.common.metrics import (
    calc_dispersion,
    calc_freq_response,
    calc_mse_db,
    calc_rms_phase_error,
    calc_tail_mse_db,
)
from tests.common.signals import (
    apply_phase_ramp,
    make_adapter_test_signal,
    make_ambiguous_qam16,
    make_dsh_beat,
    make_isi_distorted_signal,
    make_test_frame_signal,
    make_test_mimo_samples,
    make_test_mimo_signal,
    make_test_psk_samples,
    make_test_psk_signal,
    make_test_qam_samples,
    make_test_qam_signal,
    make_test_symbols,
    make_wiener_phase,
)

__all__ = [
    "apply_phase_ramp",
    "calc_dispersion",
    "calc_freq_response",
    "calc_mse_db",
    "calc_rms_phase_error",
    "calc_tail_mse_db",
    "ensure_jax_x64",
    "make_adapter_test_signal",
    "make_ambiguous_qam16",
    "make_dsh_beat",
    "make_isi_distorted_signal",
    "make_test_frame_signal",
    "make_test_mimo_samples",
    "make_test_mimo_signal",
    "make_test_psk_samples",
    "make_test_psk_signal",
    "make_test_qam_samples",
    "make_test_qam_signal",
    "make_test_symbols",
    "make_wiener_phase",
    "reference_bps_d2",
    "reference_cs_block",
    "reset_warned_kernels",
    "skip_unless_kernel_available",
    "slip_workload",
    "to_numpy",
]
