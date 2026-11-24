"""pytest plugin: find host-device transfers inside commkit (audit tool).

    uv run pytest --device=gpu -p tests.plugins.transfer_trace \
        --ignore=tests/test_examples.py

Every copy of at least 512 elements between the GPU and the host that commkit
makes during the run is recorded with the commkit function that made it, by
:mod:`tests.common.transfers` (which defines the copies traced).

The terminal summary lists every site with its lines and largest copy, ``!``
marking a site that is neither in ``_DOCUMENTED`` nor a plot-sized
(``<= _PLOT_MAX`` elements) D2H copy in ``plotting/``; such a site fails the
run.  The documented kinds are those of AGENTS.md: the sequential recursions' round trip, the
equalizers' continuation state (host copies), the reporting layer (metrics,
analysis summaries, plotting), explicit device moves and the transmitted
payload, and host values (reference symbols, pilots, tables built from host
parameters) moved to the data's device.  Documented sites that the run did not
reach are listed too, as a hint that an entry is stale (meaningful on the full
suite only).
"""

import collections
import sys
from typing import Any

import pytest

from tests.common.transfers import install

# Plot-sized ceiling for D2H copies inside plotting/: the images are bounded
# independently of the record (500x500 density and eye grids, a spectrogram
# of at most 1000 rows of 256 bins), so a full-record pull stands out.
_PLOT_MAX = 300_000
_TOOL = sys.monitoring.PROFILER_ID + 1  # 3: free for third-party tools

_RECURSION = "sequential recursion on the CPU (Numba): one round trip"
_STATE = "continuation state: host copies"
_REPORT = "reporting layer: fit on a plot-sized reduction"
_HOST_VALUE = "host value moved to the data's device"
_PAYLOAD = "transmitted payload drawn on the host"

# "path/to/module.py::qualname" -> why the transfer is deliberate.
_DOCUMENTED: dict[str, str] = {
    "_array.py::host_array": "value object: host copy of an array field",
    "analysis/linewidth/phase.py::_beta_separation": _REPORT,
    "analysis/linewidth/phase.py::_increment_subtract": _HOST_VALUE
    + " (reference symbols)",
    "analysis/linewidth/dsh.py::_dsh_fm_psd": _REPORT,
    "analysis/linewidth/dsh.py::_dsh_lorentzian": _REPORT,
    "analysis/linewidth/dsh.py::_not_a_beat": _REPORT,
    "backend.py::_device_tables": _HOST_VALUE
    + " (tables of a value object: once per object and device)",
    "coding/convolutional.py::_decode": _RECURSION + " (Viterbi, BCJR)",
    "coding/exit.py::decoder_exit_curve": _PAYLOAD + " (the codewords' bits)",
    "coding/exit.py::demapper_exit_curve": _PAYLOAD + " (the labels' bits)",
    "coding/polar.py::_decode": _RECURSION + " (successive-cancellation list decoding)",
    "coding/shaping.py::_dematch_labels": _RECURSION + " (the dematcher)",
    "coding/shaping.py::_encode_ccdm": _RECURSION
    + " (the matcher; generate runs it on the host)",
    "coding/turbo.py::_decode": _RECURSION + " (turbo decoding)",
    "coding/turbo.py::_encode": _RECURSION + " (recursive systematic encoding)",
    "core/frame.py::SingleCarrierFrame.to_signal": _PAYLOAD
    + " (the frame's body symbols)",
    "core/frame.py::extract_payload": _HOST_VALUE + " (the payload reference)",
    "core/generation.py::generate": _PAYLOAD,
    "core/signal.py::Reference.to": "explicit device move",
    "core/signal.py::Signal.to": "explicit device move",
    "equalization/_block/_blind.py::_block_fdaf_blind": _HOST_VALUE + " (pilots)",
    "equalization/_block/_dd.py::_BlockBps.snapshot": _STATE,
    "equalization/_block/_dd.py::block_lms": _STATE + " (cycle-slip buffers)",
    "equalization/_block/_engine.py::_block_state": _STATE,
    "equalization/_block/_engine.py::_prepare_block": _HOST_VALUE
    + " (training symbols, pilot mask)",
    "equalization/_common.py::_unpack_result": _RECURSION,
    "equalization/polarization.py::_jones_at_grid_points": _HOST_VALUE + " (taps)",
    "equalization/_sequential/_setup.py::_assemble_sequential": _RECURSION,
    "equalization/_sequential/_setup.py::_prepare_sequential": _RECURSION,
    "io.py::save_npz": "explicit device move (samples to the host to store them)",
    "equalization/fiber/estimation.py::_quadratic_phase": _HOST_VALUE
    + " (the known symbols)",
    "equalization/fiber/estimation.py::_quadratic_phase_pass": _REPORT
    + " (the differential phase across the band, fitted on the host)",
    "equalization/fiber/estimation.py::_clock_tone": _REPORT
    + " (the dispersion scan, searched for its peak on the host)",
    "filtering/linear.py::_sosfilt": _RECURSION
    + " (IIR above the 16-section GPU kernel limit)",
    "frequency.py::_mengali_morelli": _RECURSION + " (bootstrap over the lags)",
    "frequency.py::_pilot_slope": _HOST_VALUE + " (pilot symbols and indices)",
    "_cuda/filters.py::_iir_factory.<locals>.launch": _HOST_VALUE
    + " (the IIR state transitions)",
    "metrics/_common.py::_paired": _HOST_VALUE + " (reference symbols)",
    "coding/performance.py::error_rate_curve": _PAYLOAD + " (per batch of blocks)",
    "recovery/corrections.py::_data_aided": _HOST_VALUE + " (reference symbols)",
    "recovery/corrections.py::correct_cycle_slips": _RECURSION,
    "recovery/pilots.py::_extract_pilot_phasor": _HOST_VALUE
    + " (SciPy window, bin tables)",
    "recovery/pll.py::_pll": _RECURSION,
    "timing/clock_recovery.py::_gardner_estimate": _RECURSION,
    "timing/clock_recovery.py::correct_sampling_clock_offset": _RECURSION,
    "timing/clock_recovery.py::estimate_sampling_clock_offset": _RECURSION,
    "_interpolation.py::interpolate_piecewise": _RECURSION
    + " (the strobe positions, symbol rate, as knots of the resampling)",
    "recovery/tikhonov.py::_tikhonov": _RECURSION + " (RTS block-phase smoother)",
}

_HITS: collections.Counter[tuple[str, str]] = collections.Counter()
_MAX_SIZE: collections.defaultdict[tuple[str, str], int] = collections.defaultdict(int)
_LINES: collections.defaultdict[tuple[str, str], set[int]] = collections.defaultdict(
    set
)


def _documented(kind: str, site: str) -> bool:
    if site in _DOCUMENTED:
        return True
    if site.startswith("plotting/"):
        return kind == "D2H" and _MAX_SIZE[(kind, site)] <= _PLOT_MAX
    return False


def _record(kind: str, size: int, site: str, line: int) -> None:
    key = (kind, site)
    _HITS[key] += 1
    _MAX_SIZE[key] = max(_MAX_SIZE[key], size)
    _LINES[key].add(line)


def pytest_configure(config: Any) -> None:
    install(_TOOL, "commkit-transfer-trace", _record)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if any(not _documented(kind, site) for kind, site in _HITS):
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_terminal_summary(terminalreporter: Any) -> None:
    terminalreporter.section("host-device transfers in commkit")
    if not _HITS:
        terminalreporter.write_line("none")
    for (kind, site), count in sorted(_HITS.items(), key=lambda kv: kv[0][1]):
        mark = " " if _documented(kind, site) else "!"
        lines = ",".join(map(str, sorted(_LINES[(kind, site)])))
        peak = _MAX_SIZE[(kind, site)]
        terminalreporter.write_line(
            f"{mark} {count:6d}  {kind} {site} (line {lines}, max {peak})"
        )
    unseen = sorted(set(_DOCUMENTED) - {site for _, site in _HITS})
    if unseen:
        terminalreporter.write_line("documented but not reached:")
        for site in unseen:
            terminalreporter.write_line(f"          {site}")
