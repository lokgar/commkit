# CLAUDE.md

This file provides guidance and reference commands for developers and AI agents (such as Claude Code) working in the **CommKit** repository.

---

## 1. Project Overview

CommKit is a Python library for high-performance digital communications research. It provides a unified `Signal` abstraction over CPU (NumPy), GPU (CuPy), and JAX backends, with automatic runtime dispatch based on where data resides.

---

## 2. Command Reference

All environment and script executions **must** use `uv` (Astral's Python package manager). Python 3.12+ is required.

### Environment Management

```bash
# Sync environment (core dependencies only)
uv sync

# Sync environment with all extras (includes local GPU development packages)
uv sync --all-extras

# Re-resolve and upgrade all packages in the lockfile to their latest matched versions
uv lock --upgrade

# Add a dependency / dev dependency
uv add <package>
uv add --dev <package>

# Run a Python script or console command
uv run <script.py>
```

### Testing (pytest)

```bash
# Run tests on all backends - CPU and GPU (project default via addopts)
uv run pytest

# Run tests on CPU only
uv run pytest --device=cpu

# Run tests on GPU only (requires CuPy + CUDA)
uv run pytest --device=gpu

# Run a single test file (or a whole mirrored subpackage)
uv run pytest tests/core/test_signal.py
uv run pytest tests/equalization/

# Run a specific test case
uv run pytest tests/core/test_signal.py::test_signal_initialization -v

# Run with test coverage
uv run pytest --cov=commkit
```

### Benchmarking (pytest-benchmark)

The `benchmarks/` suite is **explicit-only**: the default `uv run pytest` collects `tests/` only (`testpaths` in `pyproject.toml`), so benchmarks never slow down a normal test run.

```bash
# Run the full benchmark suite (CPU and GPU)
uv run pytest benchmarks/ --benchmark-only --device=all

# Run one benchmark file / one benchmark
uv run pytest benchmarks/bench_bps.py --benchmark-only --device=gpu
uv run pytest "benchmarks/bench_equalizers.py::bench_lms" --benchmark-only --device=all

# Save a named baseline (commit the JSON under benchmarks/baselines/)
uv run pytest benchmarks/ --benchmark-only --device=all \
    --benchmark-save=<label> --benchmark-storage=file://benchmarks/baselines

# Compare the current code against a saved baseline (e.g. 0001_main)
uv run pytest benchmarks/ --benchmark-only --device=all \
    --benchmark-compare=0001 --benchmark-storage=file://benchmarks/baselines
```

See **Section 5 - Benchmark Suite** for what each file measures and how to read the results.

### Linting & Formatting

```bash
# Run Ruff linter
uv run ruff check .

# Format code with Ruff
uv run ruff format .

# Run static type checking
uv run mypy commkit/
```

> **CI gates on `ruff format --check .`, `ruff check .`, *and*
> `mypy commkit/` (see `.github/workflows/ci.yml`) - not just the linter.**
> `ruff check` and `ruff format` are independent: the linter passing does
> **not** mean the file is formatted, and neither implies the type check
> passes. Before committing/pushing **any** edited or newly added file -
> especially code appended programmatically (e.g. `cat >>`), which bypasses
> editor auto-format - run the same gate CI runs:
>
> ```bash
> uv run ruff format .        # apply formatting (or `--check .` to verify only)
> uv run ruff check . --fix   # lint + autofix
> uv run mypy commkit/     # static type check (library only, like CI)
> uv run pytest               # CPU+GPU tests (CI runs --device=cpu)
> ```
>
> A formatting-only diff (line wrapping, trailing commas) still fails CI - run
> `ruff format` last so nothing slips through.

### Version Management & Release Workflow

Since `bump-my-version` is defined in the project's development dependencies, always use `uv run bump-my-version` to run it directly from your local virtual environment (it is faster and avoids re-downloading packages compared to `uvx`).

#### Release Step-by-Step Guide

1. **Verify tests & package build correctness**:
   Ensure all tests are passing and the library compiles cleanly:

   ```bash
   uv run pytest
   uv build
   ```

2. **Commit your active code changes**:
   Make sure all your actual development features or bugfixes are committed in Git first:

   ```bash
   git add .
   git commit -m "feat: add digital transceiver enhancement"
   ```

3. **Bump the version locally**:
   Because `[tool.bumpversion]` in `pyproject.toml` is configured with `commit = true` and `tag = true` by default, running the bump command **automatically** updates version strings, commits the version changes to Git, and generates the Git release tag in a single atomic transaction:

   ```bash
   uv run bump-my-version bump patch   # e.g., 3.4.1 -> 3.4.2
   uv run bump-my-version bump minor   # e.g., 3.4.1 -> 3.5.0
   uv run bump-my-version bump major   # e.g., 3.4.1 -> 4.0.0
   ```

4. **Push the release commits and tags to GitHub**:
   Push the feature commits, version bump commit, and release tags to your origin repository:

   ```bash
   git push origin main --tags
   ```

*Note: You only need to run `uv sync --all-extras` during release preparation if you explicitly added or modified package dependencies in `pyproject.toml`.*

---

## 3. DSP & Coding Guidelines

### Multi-Backend Dispatching

Always utilize the `backend.dispatch(samples)` helper in DSP functions. It dynamically returns the raw device array, the corresponding array module (`xp`), and the signal processing module (`sp`) based on the location of the data:

```python
from commkit.backend import dispatch

def my_dsp_function(samples):
    x, xp, sp = dispatch(samples)
    # xp is numpy or cupy; sp is scipy or cupyx.scipy
    return xp.fft.fft(x)  # transparent CPU/GPU execution
```

### Data Types & Precision

To maximize GPU throughput and minimize memory footprint, CommKit utilizes mixed-precision layouts. Adhere strictly to the following dtype conventions:

* **Default Storage**: Use `complex64` (`np.complex64` / `cp.complex64`) for raw IQ samples and `float32` (`np.float32` / `cp.float32`) for real-valued signals.
* **Filter Dot-Product Accumulators**: In sequential adaptive filtering loops (LMS, CMA), inputs and weights are stored in `complex64` to save bandwidth. However, inside the hot loop, intermediate multiply-accumulate operations (dot-products for `y_out` and gradient weight adjustments) **must** promote variables to double precision (`float64` / `complex128`) to prevent catastrophic round-off cancellation over long symbol durations.
* **RLS Matrix Inversions**: The Recursive Least Squares (RLS) algorithm is highly sensitive to numerical accumulation. The inverse correlation matrix $P$, Kalman gain vector $k$, and regressor buffers must be maintained in **double precision** (`complex128` / `float64`) throughout the sequential loop. Updating $P$ in single-precision `complex64` quickly leads to loss of its positive-definite Hermitian properties, resulting in catastrophic filter divergence.
* **JAX TensorFloat-32 (TF32) Mitigation**: On modern NVIDIA GPUs (Ampere+), JAX defaults to TensorFloat-32 (TF32) for fast matrix multiplications, which truncates the mantissa from 23 bits (FP32) to 10 bits. For sequential gradient-accumulation loops like LMS/RLS weight updates, TF32 truncation causes severe drift. You **must** explicitly specify `Precision.HIGHEST` for JAX matrix products in equalizers to force true FP32.
* **Phase Unwrapping & Kalman Smoothers (CPR)**: In carrier phase recovery (e.g., Viterbi-Viterbi, BPS, pilot-aided), phase angle arrays must be promoted to double precision (`float64`) before calling `xp.unwrap()`. Unwrapping is extremely sensitive to $\pm\pi/M$ boundaries; single-precision `float32` rounding error can trigger spurious quadrant wrap-around slips. Tikhonov Kalman smoothers must also compute block transitions in `float64` to prevent underflow in small noise covariance variables.

### Signal Normalisation Invariant

To prevent scaling issues across cascaded DSP operations, CommKit maintains a strict power normalisation invariant:

* **Symbol power representation**: A signal oversampled at `sps` has average sample energy `E[|x|²] = 1 / sps`.
* **Symbol-rate representation**: A signal at `sps=1` has average sample energy `E[|x|²] = 1`.
* Any new DSP blocks (e.g. filters, upsamplers, decimators) that alter the rate **must** apply the exact deterministic gain corrections (e.g., `sps_before / sps_after` scaling) to preserve this invariant.

### Reproducibility & Randomness

Never call a module-level global RNG (`np.random.normal`, `xp.random.rand`, ...)
directly inside library code. Every function performing stochastic modeling
(noise injection, impairments, `generate_*`) must accept an optional
`seed: int | None = None` parameter and use one of two generation patterns,
chosen by data volume:

1. **Device-identical (preferred)** - generate with NumPy's `default_rng` on
   the CPU and transfer with `to_device(...)`, so a given seed produces
   bit-identical data on CPU and GPU (`generate_bits`, `generate_phase_noise`).
   Use for bit/symbol sequences and trajectories whose size is modest.

2. **On-device** - only when the noise is as large as the signal itself and a
   host round-trip would dominate (`apply_awgn`):

   ```python
   rng = xp.random.RandomState(seed) if seed is not None else xp.random
   noise = rng.normal(0, std, samples.shape)
   ```

   Streams then differ between CPU and GPU for the same seed.

**Seed-stability policy:** a seed guarantees reproducibility *within* a
library version only - RNG internals may improve between versions (e.g.
`apply_phase_noise` moved from backend `RandomState` to pattern 1 in the
`generate_*` refactor). Never build tests or stored baselines on exact noise
realizations; assert statistics instead.

### Performance JIT Compilation

* **Numba**: Use `@numba.njit(cache=True, fastmath=True, nogil=True)` for serial loops (like sequential LMS/RLS adaptive updates on CPU). Keep kernels compiled lazily and cached. For single-stream sequential equalization, Numba on CPU is the **fastest existing backend** - measured 150-200x faster than the per-symbol JAX scan on GPU (see `benchmarks/`).
* **JAX**: `jax.lax.scan` compiles sequential weight updates, but per-symbol scans on GPU are dominated by per-step XLA overhead and are slow for single streams. Reserve the JAX path for differentiability or batched workloads; GPU-side throughput improvements should use block/chunked formulations (`update_mode='block'`, the `block_lms` FDAF engine, and the `block_cma`/`block_rde` siblings).

### Host-Device Synchronization Hygiene

Inside library code, **never extract a scalar from a possibly-GPU array inside a loop** (`float(x[ch])`, `int(x[ch])`, `.item()`) - each extraction forces a full GPU pipeline flush. Instead:

* Compute the full per-channel vector on device, transfer it **once** with `to_device(vec, "cpu")`, then loop over the host copy.
* Prefer on-device gathers (`xp.take_along_axis`) over Python list comprehensions with indexed scalars.
* Per-channel **diagnostic logging** must be gated: wrap the transfer + loop in `if logger.isEnabledFor(logging.INFO):` so disabled logging costs zero syncs (see `metrics.py` for the canonical pattern).
* Bound large broadcast intermediates: distance-matrix style `(N, M)` allocations should be chunked over N with an on-device accumulator (see `metrics.mi`).

### Array Shapes

* **SISO**: 1-D array: `(N_samples,)`
* **MIMO**: 2-D array: `(N_channels, N_samples)` - **time is always on the last axis**.

Never hand-roll the promote/squeeze idiom.  `commkit/helpers.py` owns it, so the
validation is identical everywhere (0-d and 3-D inputs raise a shape error
naming the offending argument instead of silently reaching a confusing
broadcast failure downstream):

```python
from commkit.helpers import as_2d, broadcast_channels, require_channels, restore_1d

def my_dsp_function(samples, ref_symbols):
    x, xp, _ = dispatch(samples)
    x2, was_1d = as_2d(x, name="samples")             # (N,) -> (1, N)
    ref = broadcast_channels(xp.asarray(ref_symbols),  # (L,) / (1, L) -> (C, L)
                             x2.shape[0], xp, name="ref_symbols")
    out = ...                                          # channel-batched body
    return restore_1d(was_1d, out)                     # squeeze back for SISO
```

* `as_2d` / `restore_1d` - the promote/squeeze pair.  `restore_1d` takes several
  outputs at once: `drift, pn = restore_1d(was_1d, drift, pn)`.
* `broadcast_channels` - a shared reference across channels, with the
  channel-count check the bare `ref[None, :]` promotion lacks.  Returns a
  **read-only** broadcast view; `.copy()` before writing.
* `require_channels` - for entry points defined only at a fixed channel count
  (dual-pol channel models); pass `description=` to keep domain wording in the
  error.
* `to_report_scalar` - the reporting-layer counterpart: collapses a `(C,)`
  metric to a Python float for SISO, transferring from device if needed.
* `linear_trend_slope` / `remove_linear_trend` - the per-channel least-squares
  slope shared by the pilot-tone and DSH detrend stages; keeps `Σ(x-x̄)²` on
  device rather than syncing it back as a float.

Note these are plain indexing/broadcast helpers, valid on NumPy, CuPy **and**
JAX arrays - unlike `dispatch`, which recognizes NumPy/CuPy only and will
silently pull a JAX array to the host.

### Signal-Awareness

Any new DSP function whose primary argument is genuinely **signal-representable
data** (raw IQ samples, or a field a `Signal` actually carries, like
`resolved_symbols`) should accept a `Signal` alongside a raw array, transparently:
array in → array out, `Signal` in → `Signal` out.  Use the same idiom
everywhere, defined once in `commkit/helpers.py`:

```python
from commkit.helpers import unwrap_signal, rewrap_signal

def fir_filter(samples, taps, axis=-1):
    x, sig = unwrap_signal(samples)          # x: array; sig: Signal | None
    if sig is not None:
        return rewrap_signal(sig, fir_filter(x, taps, axis=-1))
    ... existing array-only body, unchanged ...
```

* `unwrap_signal(x, *, field="samples")` - returns `(array, signal_or_None)`.
  Most functions read `.samples`; a few (hard-decision demapping, phase-rotation
  correction on `resolved_symbols`) pass `field=` to read a different attribute.
* `rewrap_signal(sig, array, **metadata)` - `sig=None` passes `array` through
  unchanged; otherwise delegates to `sig.with_samples()`, which shallow-copies
  metadata, replaces `.samples`, invalidates resolved caches, and applies any
  `**metadata` kwargs via assignment validation (e.g.
  `sampling_rate=sig.symbol_rate` after decimating to symbol rate).

**Metadata priority: the `Signal`'s own value always wins.** When a function
takes both a `Signal` and a scalar metadata parameter that duplicates one of
its fields (`sampling_rate`, `sps`, `mod_scheme`, `mod_order`, `ps_pmf`,
...), the `Signal`'s field - never the supplied argument - is what the
Signal-branch call actually uses. A caller passing a conflicting value on a
`Signal` is a bug to catch, not a request to honor: silently letting the
supplied value win would let a signal at 2 GSa/s be equalized with a stale
`sps=2` nobody meant to apply to it.

There is no shared helper for this - both cases are short enough to write
inline in the `if sig is not None:` branch, and a generic helper covering
both the "always wins" and "falls back with a warning" behaviors in one call
signature ends up harder to read at the call site than the few extra lines.

* **Required fields** (`sampling_rate`, `symbol_rate`, and the derived `sps`)
  are never `None` on a `Signal` - pydantic enforces it - so the Signal
  branch references `sig.sampling_rate`/`sig.sps` directly and drops the
  supplied argument entirely.  There is no case where the Signal is missing
  the field, but the caller can still pass a stale/conflicting value by
  mistake, so warn whenever one was supplied at all:

  ```python
  x, sig = unwrap_signal(samples)
  if sig is not None:
      # sig.sampling_rate is required, so it always wins over a supplied
      # sampling_rate - see CLAUDE.md, "Signal-Awareness".
      if sampling_rate is not None:
          logger.warning(
              "my_dsp_function(): ignoring supplied sampling_rate=%r for "
              "Signal input; using the signal's own sampling_rate=%r instead.",
              sampling_rate,
              sig.sampling_rate,
          )
      return my_dsp_function(x, sig.sampling_rate, ...)
  ```

* **Optional fields** (`mod_scheme`, `mod_order`, `ps_pmf`, ...) can
  legitimately be unset on a `Signal` (e.g. an unshaped QAM signal has
  `ps_pmf=None`). Read the Signal's field first; fall back to the supplied
  argument - logging a `logger.warning` - only when the Signal genuinely
  lacks it *and* a fallback was supplied. Passing neither stays silent (the
  common case), so the log isn't spammed on every call to an unshaped signal:

  ```python
  mod = sig.mod_scheme
  if mod is None:
      mod = modulation
      if mod is not None:
          logger.warning(
              "my_dsp_function(): Signal has no mod_scheme set; falling "
              "back to supplied modulation=%r.", mod,
          )
  ```

**Metadata rules for functions that change rate or domain:**

| Situation | What to pass `rewrap_signal` |
| --- | --- |
| Shape/rate unchanged (most `apply_*`/`correct_*`) | `rewrap_signal(sig, result)` - no metadata |
| Decimates to one sample/symbol (equalizers) | `sampling_rate=sig.symbol_rate` |
| Upsamples by an integer factor | `sampling_rate=sig.sampling_rate * factor` |
| Resamples to an explicit target rate | `sampling_rate=<the target rate param>` |
| Output field differs from the input field (`resolve_symbols`, `demap_symbols_hard`) | Skip `rewrap_signal`; `sig.clone()` + `setattr` by hand - the one sanctioned exception |
| Returns a scalar/dict/tuple in a different domain (frequency/tau/PSD bins, phase or frequency *estimates*) | `unwrap_signal` only, on the input - never wrap the output |

**Not every array parameter is signal-representable.** A function that
consumes a *derived* quantity - a phase or frequency trajectory, a
correlation array, PSD bins, an `EqualizerResult`, filter taps - one or more
stages downstream of the original capture has no sound field to unwrap from;
forcing Signal-awareness there is backwards, since there is no `Signal` yet
(or any more) to preserve metadata from.  `smooth_phase_wiener`,
`estimate_fractional_delay`, `allan_deviation`, and every `plot_*` function
in `plotting/{analysis,equalizer,sync}.py` are examples: they take arrays a
`Signal` would never hold as `.samples`, so they stay plain-array functions.
Likewise, a function whose only job is to build the data a `Signal` gets
constructed *from* is a synthesis primitive, not a transform on existing
`Signal` data - same exclusion as `generate_*`.  `core/generation.py`'s
`shape_pulse`/`expand` (TX symbol -> waveform) and
`equalization.build_pilot_ref` (sparse pilots -> dense equalizer reference)
are the current examples; note that `shape_pulse`/`expand` live in
`core/generation.py` rather than `filtering.py`/`multirate.py` for exactly
this reason - they build the samples a `Signal` gets constructed from, not
transform an existing one.

### Naming Conventions

* **Verb prefixes for processing functions.** Recovery/correction routines follow a
  fixed verb vocabulary so the call site reads as a pipeline:
  * `estimate_*` - measure an impairment without altering the signal
    (returns the estimate, e.g. `estimate_carrier_frequency_offset`).
  * `correct_*` - apply a (possibly externally supplied) correction
    (e.g. `correct_carrier_phase`, `correct_cycle_slips`).
  * `recover_*` - the combined estimate-then-correct convenience entry point
    (e.g. `recover_carrier_phase_bps`).
  * `resolve_*` - disambiguate a discrete/structural unknown
    (e.g. `resolve_phase_ambiguity`, `resolve_channel_permutation`).
* **`generate_*` for synthesis.** Any function that synthesizes a new signal,
  sequence, or noise process from *parameters* rather than from an input array
  (stochastic, accepts `seed`) takes a `generate_` prefix: `generate_qam`,
  `generate_psk`, `generate_pam`, `generate_psqam`, `generate_bits`,
  `generate_symbols`, `generate_phase_noise`, plus the generic
  `generate(modulation=...)` engine.  Deterministic transforms of existing
  arrays keep the `apply_*` / compute-noun conventions (e.g.
  `analysis.dsh_beat(phi, ...)` is a compute function - no randomness, no
  seed - even though it synthesizes a waveform from a phase trajectory).
* **Compute vs. plot.** A computation keeps the plain noun
  (`analysis.carrier_phase_trajectory`, `analysis.allan_deviation`,
  `spectral.spectrogram`). **Every** public function in `plotting` takes a
  `plot_` prefix (`plot_constellation`, `plot_eye_diagram`, `plot_psd`,
  `plot_carrier_phase_trajectory`, `plot_allan_deviation`, `plot_spectrogram`,
  ...) - the only exception is the non-plot theme helper `apply_default_theme`.
  This makes the layer obvious at the call site and removes every same-name
  collision between the `analysis`/`spectral`/`timing`/`frequency` compute
  modules and `plotting`. Never add a bare-noun plot function that shadows a
  compute function.

### Design-vs-Apply Separation

For any DSP building block with distinct "design" and "apply" phases - a
coefficient/parameter generator, and a step that consumes those coefficients
against data - keep the two as separate functions rather than one function
that does both internally.  This lets a caller design once and apply many
times (e.g. reuse the same filter across a batch of signals) and keeps each
function's signature focused.  Established examples in `filtering.py`:

* FIR taps generators (`rrc_taps`, `rc_taps`, `gaussian_taps`, `fir_taps`
  (lowpass/highpass/bandpass/bandstop via `btype=`), ...) produce coefficient
  arrays; `fir_filter(samples, taps)` applies them.
* IIR SOS generators (`butterworth_sos`, `chebyshev1_sos`, `chebyshev2_sos`,
  `elliptic_sos`, `bessel_sos`) produce second-order-section coefficients;
  `iir_filter(samples, sos)` applies them.

New DSP building blocks with a design/apply split should follow this same
two-function pattern rather than folding both steps into one.

`filtering.py` vs. `smoothing.py`: `filtering.py` holds real signal-chain
filters (the design/apply pairs above, matched filtering, Overlap-Save) with
an actual frequency response and causality claim.  `smoothing.py` holds
diagnostic/plotting-only smoothers (`moving_average`, `savgol_smooth`,
`smooth_density_2d`) - non-causal, no frequency-response meaning, used only
to make a plotted curve or a robust estimate less noisy.  A routine that
removes/passes a frequency band as part of the signal chain itself belongs in
`filtering.py`; a routine that only exists to smooth something for display or
a peak search belongs in `smoothing.py`.

---

## 4. Testing Conventions

* **Parametrization**: Test cases must utilize `backend_device` and `xp` fixtures from `conftest.py` to automatically validate code correctness on both CPU and GPU backends.
* **Assertions**: Standard `numpy.testing` assertions raise `TypeError` when evaluated on GPU arrays. Always use the `xpt` helper assertion module. Use `xp.asarray(expected)` to cast expectation variables to the active backend, and cast reductions to standard Python scalars before comparison:

  ```python
  from commkit.testing import xpt
  # ...
  xpt.assert_allclose(result, expected, rtol=1e-5)
  assert float(xp.mean(xp.abs(result))) > 0.0
  ```

* **Layout mirrors the source tree.** Tests for a subpackage live in the
  matching test subpackage and, as far as practical, **one test file maps to one
  source module**:
  * `tests/equalization/` <-> `commkit/equalization/` - `test_sequential.py`
    (lms/rls/cma/rde, Numba), `test_sequential_jax.py`, `test_mimo.py`,
    `test_winit.py`, `test_linear.py` (zf/MMSE), `test_polarization.py`,
    `test_blind.py` + `test_block_update.py` (block_cma/block_rde),
    `test_block.py` (block_lms / FDAF), `test_cpr.py`, and the CUDA-kernel tests
    `test_bps_kernel.py` / `test_cs_kernel.py`.  `sequential.py` and `_block.py`
    are themselves subpackages internally (`sequential/_dd.py` + `_blind.py`;
    `_block/_seqmode.py` + `_dd.py` + `_blind.py`) per the module-splitting
    trigger below - the test files above are unaffected since they exercise
    the public `commkit.equalization` surface, not these internal module
    paths (a handful of `patch()`/`monkeypatch` targets in `test_block.py` /
    `test_sequential_jax.py` do reach into the internal paths and were
    updated when the split landed).
  * `tests/recovery/` <-> `commkit/recovery/` - `test_viterbi_viterbi.py`,
    `test_bps.py`, `test_pilots.py`, `test_tikhonov.py`, `test_pll.py`,
    `test_corrections.py` (also covers `recovery/_common.py`'s shared
    `_vv_block_phase` block-phase estimator and `_log_phase_summary` CPR
    diagnostic logger), plus `test_joint.py` for the cross-algorithm
    joint-channel consistency checks.
  * `tests/core/` <-> `commkit/core/` - `test_signal.py`, `test_signal_mimo.py`,
    `test_frame.py` (`Preamble`/`SingleCarrierFrame`), `test_psqam.py` (generation).
  * `tests/analysis/` <-> `commkit/analysis/` - `test_trajectory.py`,
    `test_drift.py`, `test_linewidth.py`, `test_allan.py`,
    `test_interferometry.py` (DSH laser characterization).
  * `tests/impairments/` <-> `commkit/impairments/` - `test_noise.py`,
    `test_source.py`, `test_frontend.py`, and `channel/test_channel_linear.py`
    (the file basename is `test_channel_linear` rather than `test_linear`
    because pytest's default prepend import-mode requires globally-unique test
    basenames and `tests/equalization/test_linear.py` already exists).
  * `tests/mapping/` <-> `commkit/mapping/` - `test_gray.py`, `test_bits.py`,
    `test_llr.py`, `test_constellation.py` (the `Constellation` value object).
    Probabilistic-shaping tests live in `tests/core/test_psqam.py`.
  * Flat modules (`filtering`, `metrics`, `spectral`, `timing`, `frequency`,
    `smoothing`, ...) keep a single top-level `tests/test_<module>.py`.

  When a single module's test file grows unwieldy, split it by *concern* within
  the same subpackage (e.g. sequential vs. JAX vs. MIMO) rather than letting one
  multi-thousand-line file accumulate.

* **Module-splitting trigger (when to promote a flat module to a package).**
  Split a flat module into a subpackage when it crosses **~1,000 LOC** *and*
  contains **≥2 clearly separable concerns** (distinct physical/mathematical
  domains, or estimate-vs-correct workflows) that don't share much state. Size
  alone is not the trigger - cohesive large modules can stay flat - and neither
  is multiple concerns in a small file. When splitting, the package
  `__init__.py` **must re-export exactly today's public names** so the import
  surface (`from commkit.X import Y`) stays byte-identical; internals move,
  user imports don't break. Mirror the split in `tests/` per the rule above.

---

## 5. Benchmark Suite

The `benchmarks/` directory tracks the performance of the GPU-relevant hot paths. Baselines are committed under `benchmarks/baselines/` so any optimization PR can be gated quantitatively (run -> compare -> quote the delta).

### What each file measures

| File | Functions | What the numbers show |
| --- | --- | --- |
| `bench_bps.py` | `recover_carrier_phase_bps` | Square-QAM O(1) fast path vs. the non-square `(CHUNK, B, M)` distance-tensor path (16-QAM vs. 128-cross). Includes the GPU-only gate workload `128cross / N=1e6 / C=2` for the fused-kernel work. |
| `bench_block_lms.py` | `block_lms` | Frequency-domain equalizer with CPR off / `bps` / `bps + cycle-slip`. Three legs: `bench_block_lms` (block_size=256, fully trained) is the launch-overhead-bound eager stress case; `bench_block_lms_large` (block_size=2048) is the recommended large-block operating point and guards block-size-scaling regressions; `bench_block_lms_dd` (block_size=256, short training prefix) is the realistic decision-directed steady state and the only leg that exercises the CUDA-graph path (graph captures full DD blocks only) - a graph regression / silent fallback shows up here as a jump back toward the eager ~800 ms. |
| `bench_equalizers.py` | `lms`, `cma`, `rls` | Sequential equalizers across `numba` and `jax` backends, 50k symbols (20k for RLS, symbol-spaced). |
| `bench_sync_misc.py` | Viterbi-Viterbi + cycle slips, `resolve_phase_ambiguity`, `evm` | Host-sync hygiene targets in recovery/metrics. |

### How to read the IDs

Benchmark IDs encode `[<input-device>-<equalizer-backend>]`:

* `[cpu-numba]` - NumPy input, Numba CPU loop (the reference).
* `[gpu-numba]` - **CuPy input** with `backend='numba'`: measures the documented D2H -> CPU loop -> H2D round trip (the Δ vs. `[cpu-numba]` is the transfer + sync cost, ~1-3 ms per 50k symbols).
* `[gpu-jax]` - JAX `lax.scan` running on the GPU.

### Methodology rules

* Timed bodies must end with the `sync` fixture call - GPU wall time without a stream sync measures kernel *launches*, not execution.
* Every benchmark does one warmup round so Numba/NVRTC/XLA compilation and CuPy pool growth are excluded.
* Workloads come from `benchmarks/workloads.py` with **fixed seeds** - never inline ad-hoc signal generation, or baselines stop being comparable.
* `benchmarks/benchutils.py` provides `CudaEventTimer` (pure device time) and `nvtx_range` (annotate stages for `nsys profile` - used to count D2H transfers per function).
* **Trust deltas, not single runs**: the default 3 rounds are noisy (±20-40% has been observed under ambient load). Before believing a regression, re-run the benchmark in isolation or do a controlled A/B against the prior revision (`git checkout <rev> -- <file>` + a fixed-seed timing script with ≥7 repetitions).
* Library logging is set to WARNING in `benchmarks/conftest.py` - benchmark numbers exclude diagnostic-logging costs by design (and INFO-gated diagnostics are skipped entirely).
