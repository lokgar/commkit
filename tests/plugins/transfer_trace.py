"""pytest plugin: find host-device transfers inside commkit (audit tool).

    uv run pytest --device=gpu -p tests.plugins.transfer_trace \
        --ignore=tests/test_examples.py

Every copy of at least 512 elements between the GPU and the host that commkit
makes during the run is recorded with the commkit function that made it:

- ``D2H``: ``cupy.ndarray.get``, directly or through ``cupy.asnumpy`` or
  :func:`commkit.backend.to_device`;
- ``H2D``: ``cupy.asarray`` / ``cupy.array`` of a NumPy array, or
  ``cupy.ndarray.set``.

A transfer is charged to the innermost commkit frame outside ``backend`` (so a
:func:`~commkit.backend.to_device` call is charged to its caller); one with no
commkit frame on the stack (the tests' own) is ignored.  The terminal summary
lists every site with its lines, ``!`` marking a site that is neither in
``_DOCUMENTED`` nor in ``plotting``; such a site fails the run.  The documented
kinds are those of AGENTS.md: the sequential recursions' round trip, the
equalizers' continuation state (host copies), the reporting layer (metrics,
analysis summaries, plotting), explicit device moves and the transmitted
payload, and host values (reference symbols, pilots, tables built from host
parameters) moved to the data's device.  Documented sites that the run did not
reach are listed too, as a hint that an entry is stale (meaningful on the full
suite only).  Calls are traced with ``sys.monitoring``, since
``cupy.ndarray.get`` cannot be patched.
"""

import collections
import importlib.util
import os
import sys
from types import CodeType, FrameType
from typing import Any

import numpy as np
import pytest

_MIN = 512
_TOOL = sys.monitoring.PROFILER_ID + 1  # 3: free for third-party tools

_RECURSION = "sequential recursion on the CPU (Numba): one round trip"
_STATE = "continuation state: host copies"
_REPORT = "reporting layer: fit on a plot-sized reduction"
_HOST_VALUE = "host value moved to the data's device"
_PAYLOAD = "transmitted payload drawn on the host"

# "path/to/module.py::qualname" -> why the transfer is deliberate.
_DOCUMENTED: dict[str, str] = {
    "analysis/linewidth/phase.py::_beta_separation": _REPORT,
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
    "equalization/_common.py::_unpack_result_numpy": _RECURSION,
    "equalization/polarization.py::_jones_at_grid_points": _HOST_VALUE + " (taps)",
    "equalization/_sequential/_setup.py::_assemble_sequential": _RECURSION,
    "equalization/_sequential/_setup.py::_prepare_sequential": _RECURSION,
    "io.py::save_npz": "explicit device move (samples to the host to store them)",
    "filtering/fiber.py::_quadratic_phase": _HOST_VALUE + " (the known symbols)",
    "filtering/fiber.py::_quadratic_phase_pass": _REPORT
    + " (the differential phase across the band, fitted on the host)",
    "filtering/fiber.py::_clock_tone": _REPORT
    + " (the dispersion scan, searched for its peak on the host)",
    "frequency.py::_mengali_morelli": _RECURSION + " (bootstrap over the lags)",
    "_cuda/__init__.py::_iir_factory.<locals>.launch": _HOST_VALUE
    + " (the IIR state transitions)",
    "metrics/_common.py::_paired": _HOST_VALUE + " (reference symbols)",
    "coding/performance.py::error_rate_curve": _PAYLOAD + " (per batch of blocks)",
    "recovery/corrections.py::ConstantPhase.__post_init__": "value object: host copy",
    "recovery/corrections.py::_data_aided": _HOST_VALUE + " (reference symbols)",
    "recovery/corrections.py::correct_cycle_slips": _RECURSION,
    "recovery/pilots.py::_extract_pilot_phasor": _HOST_VALUE
    + " (SciPy window, bin tables)",
    "recovery/pll.py::_pll": _RECURSION,
    "timing/clock_recovery.py::_gardner_estimate": _RECURSION,
    "timing/clock_recovery.py::correct_sampling_clock_offset": _RECURSION,
    "timing/clock_recovery.py::estimate_sampling_clock_offset": _RECURSION,
    "timing/clock_recovery.py::_resample_on_strobes": _RECURSION,
    "recovery/tikhonov.py::_tikhonov": _RECURSION + " (RTS block-phase smoother)",
}

_HITS: collections.Counter[tuple[str, str]] = collections.Counter()
_LINES: collections.defaultdict[tuple[str, str], set[int]] = collections.defaultdict(
    set
)


def _package_dir() -> str:
    """The directory of the commkit package, found without importing it.

    Sites are matched against it, not against a ``commkit/commkit/`` path
    fragment, so the audit works in a checkout of any name (a worktree).
    """
    spec = importlib.util.find_spec("commkit")
    assert spec is not None and spec.origin is not None, "commkit is not importable"
    return os.path.dirname(os.path.realpath(spec.origin)).replace("\\", "/") + "/"


_PKG = _package_dir()


def _relative(path: str) -> str | None:
    """``path`` relative to the commkit package, or None outside it."""
    path = path.replace("\\", "/")
    return path[len(_PKG) :] if path.startswith(_PKG) else None


def _documented(site: str) -> bool:
    return site in _DOCUMENTED or site.startswith("plotting/")


def _site() -> tuple[str, int] | None:
    frame: FrameType | None = sys._getframe(2)
    while frame is not None:
        path = frame.f_code.co_filename.replace("\\", "/")
        if path.endswith("/commkit/backend.py") and frame.f_code.co_qualname.startswith(
            "_device_tables"
        ):
            return "backend.py::_device_tables", frame.f_lineno
        if "/site-packages/cupy" in path or path.endswith("/commkit/backend.py"):
            frame = frame.f_back
            continue
        rel = _relative(path)
        if rel is None:
            return None
        return f"{rel}::{frame.f_code.co_qualname}", frame.f_lineno
    return None


def _install() -> None:
    import cupy as cp

    get, set_ = cp.ndarray.get, cp.ndarray.set
    uploads = (cp.asarray, cp.array)
    disable = sys.monitoring.DISABLE

    def on_call(code: CodeType, offset: int, f: Any, arg0: Any) -> Any:
        # ``x.get()`` passes the method descriptor and x; ``h = x.get; h()``
        # passes the bound method and its first argument.
        bound = getattr(f, "__self__", None)
        method = getattr(f, "__name__", None) if isinstance(bound, cp.ndarray) else None
        kind, size = None, 0
        if f is get or f is set_:
            kind, size = ("D2H" if f is get else "H2D"), getattr(arg0, "size", 0)
        elif method in ("get", "set"):
            kind, size = ("D2H" if method == "get" else "H2D"), bound.size
        elif any(f is u for u in uploads):
            if isinstance(arg0, np.ndarray):
                kind, size = "H2D", arg0.size
        elif _relative(code.co_filename) is None:
            # The callee at a call site outside commkit does not change, so
            # stop watching it.  Inside commkit it may (``xp.asarray`` with
            # xp = numpy first, cupy later), so keep watching.
            return disable
        if kind and size >= _MIN:
            found = _site()
            if found:
                _HITS[(kind, found[0])] += 1
                _LINES[(kind, found[0])].add(found[1])
        return None

    sys.monitoring.use_tool_id(_TOOL, "commkit-transfer-trace")
    sys.monitoring.register_callback(_TOOL, sys.monitoring.events.CALL, on_call)
    sys.monitoring.set_events(_TOOL, sys.monitoring.events.CALL)


def pytest_configure(config: Any) -> None:
    _install()


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if any(not _documented(site) for _, site in _HITS):
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_terminal_summary(terminalreporter: Any) -> None:
    terminalreporter.section("host-device transfers in commkit")
    if not _HITS:
        terminalreporter.write_line("none")
    for (kind, site), count in sorted(_HITS.items(), key=lambda kv: kv[0][1]):
        mark = " " if _documented(site) else "!"
        lines = ",".join(map(str, sorted(_LINES[(kind, site)])))
        terminalreporter.write_line(f"{mark} {count:6d}  {kind} {site} (line {lines})")
    unseen = sorted(set(_DOCUMENTED) - {site for _, site in _HITS})
    if unseen:
        terminalreporter.write_line("documented but not reached:")
        for site in unseen:
            terminalreporter.write_line(f"          {site}")
