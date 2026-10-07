# AGENTS.md

Guide for anyone, human or coding agent, changing the **CommKit** repository.

The code is being migrated to 2.0 on the `v2` branch following
[`REFACTORING_PLAN.md`](REFACTORING_PLAN.md). The rules below describe the 2.0
target. Code that does not follow them yet is listed in
[Migration status](#9-migration-status). Never copy a pattern listed there into
new code.

---

## 1. What CommKit is

A Python library for digital-communications research: simulation, receiver DSP,
metrics, plotting and experimental captures, on NumPy (CPU) and CuPy (GPU) data.
It is **not differentiable**. A separate JAX library, *commax*, will hold the
blocks that must sit inside a gradient. commax depends on commkit, never the
reverse, and takes its conventions (`Constellation`, bit labels, pulse taps,
frame layouts) from commkit.

Users are researchers and engineers. Design for them:

- **Algorithms are plain functions.** Data is a `Signal` or an array. Things
  that *describe* a modulation, a pulse or an embedded sub-algorithm are small
  frozen value objects written inline in the call:

  ```python
  res = lms(rx, num_taps=21, step_size=1e-3, cpr=BPS(test_phases=64))
  ```

- **Everything is expressed in Python.** No configuration files, registries,
  global switches or multi-step setup.
- **We deliberately do not build** pipeline or receiver objects, DSP methods on
  `Signal`, string shorthands for constellations, or public "config objects per
  algorithm".

---

## 2. Commands

Always use `uv`. Python 3.12+.

```bash
uv sync --all-extras                     # environment incl. GPU packages
uv run pytest                            # CPU + GPU tests (default --device=all)
uv run pytest --device=cpu               # what CI runs
uv run pytest tests/equalization/ -k lms # subset
uv run pytest --cov=commkit              # coverage
NUMBA_DISABLE_JIT=1 uv run pytest tests/equalization tests/recovery \
    tests/test_frequency.py --device=cpu --cov=commkit   # real kernel coverage
```

**Commit gate.** CI fails on any of these, so run all of them before every
commit, with `ruff format` last:

```bash
uv run ruff check . --fix
uv run mypy commkit/
uv run pytest
uv run ruff format .
```

**Benchmarks** are explicit-only; `uv run pytest` never collects them. A
commit that changes a public signature also smoke-runs them once, without
timing, so their call sites cannot rot unnoticed:

```bash
uv run pytest benchmarks/ --benchmark-disable --device=all
```

Timed runs:

```bash
uv run pytest benchmarks/bench_bps.py --benchmark-only --device=all
uv run pytest benchmarks/ --benchmark-only --device=all \
    --benchmark-compare=0002 --benchmark-storage=file://benchmarks/baselines
```

**Release:**

1. Tests pass and `uv build` succeeds.
2. Commit the work.
3. `uv run bump-my-version bump {patch|minor|major}`. This commits and tags
   automatically.
4. `git push origin main --tags`.

---

## 3. Architecture

Dependencies point downward only:

| Layer | Modules | May import |
| --- | --- | --- |
| Infrastructure | `backend`, `logger`, `_cuda` | NumPy, CuPy |
| Array and math helpers | `_array` (shape and validation), `math` (power, normalization, dB, linear trend) | infrastructure |
| Value objects | `mapping.Constellation`, pulse classes, `recovery` CPR configs | layers above |
| Core | `core` (`Signal`, frames, generation), `io` | layers above |
| DSP | `mapping`, `filtering`, `multirate`, `spectral`, `smoothing`, `impairments`, `timing`, `frequency`, `recovery`, `equalization`, `metrics`, `analysis` | layers above, and lower DSP modules |
| Presentation | `plotting` | everything; **nothing imports plotting** |

- **Responsibilities:**
  - `filtering` holds signal-chain filters with a real frequency response.
  - `smoothing` holds display and estimation smoothers only.
  - `analysis` holds laser and phase characterization.
  - `metrics` is the reporting layer.
- **Splitting modules:** promote a flat module to a package only when it
  exceeds about 1000 lines *and* has two or more separable concerns. The
  package `__init__` re-exports the public names. Never create a `utils/`
  package or one-function files.
- **Placeholder modules are kept on purpose.** `commkit.coding` and
  `commkit.impairments.channel.nonlinear` are reminders of planned work. Never
  delete them or treat them as dead code. They are excluded from coverage and
  from the contract registry, and the README labels them "planned, not
  implemented".

---

## 4. API rules

### Signal

- `Signal` holds: samples (NumPy or CuPy, `(N,)` or `(C, N)`, time on the last
  axis), the **facts** `sampling_rate`, `symbol_rate` and `center_frequency`, the
  **description** `constellation` and `pulse`, and the ground truth
  `reference` (transmitted symbols and bits) plus the `frame` layout.
- It never holds pipeline results.
- Construction does no hidden work: no mapping, no normalization, no device
  move, no shape guessing. Factories (`generate`, frames) do that work.
- `sig.to("gpu")` and `sig.replace(...)` return new Signals. Arrays are shared,
  not copied; use `clone()` for an independent copy.
- **Alignment invariant:** reference symbol *k* corresponds to symbol period *k*
  of the samples. Any function that drops or shifts symbol periods slices the
  reference to match. Metrics raise on mismatch; they never guess.

### Function signatures

- `f(data, other_data, *, params)`: everything after the data arguments is
  keyword-only. A function without data arguments may keep its single primary
  parameter positional (`barker_sequence(13)`, `gray_code(5)`); with several
  parameters, all are keyword-only (`rrc_taps(sps=2, rolloff=0.1, span=10)`).
  The data-argument count is recorded per function in the contract registry.
- **Signal awareness.** A function whose main argument is waveform data (or a
  field a Signal really carries) accepts an array or a Signal. It unwraps once
  with `adapt_signal()` (in `core/_signal_adapter.py`), works on arrays, and
  returns through `wrap_samples()`. Never re-enter the public function
  recursively.
  - Functions on *derived* quantities (phase trajectories, correlations, taps,
    PSD bins, results) and synthesis primitives stay array-only.
- **Metadata rule: facts versus choices.**
  - Facts (`sampling_rate`, `symbol_rate`, and so `sps`) come from the Signal.
    A conflicting argument raises `ValueError`; an equal one is fine.
  - Choices (decision `constellation`, `noise_var`, `pulse`) default to the
    Signal's value, and an explicit argument wins.
  - A missing fact for array input raises. Neither case warns.
- Integer-SPS algorithms validate with `require_integer_sps()` before any cast.
  Never truncate 1.5 SPS to 1.
- **Value objects are frozen dataclasses** with read-only NumPy arrays:
  - `Constellation`: `.qam(M)`, `.psk(M)`, `.pam(M, unipolar=)`, or arbitrary
    `points`; `.shaped(nu=|entropy=)` keeps unit power. Functions accept the
    object only, never strings.
  - Pulses: `RRC`, `RC`, `Gaussian`, `Rect`, `SmoothRect`. A raw taps array is
    accepted wherever a pulse is.
  - Algorithm objects, one per estimation method (`BPS`, `PLL`,
    `ViterbiViterbi`, `MthPower`, `CycleSlip`, ...). The same object is used
    standalone (`correct_carrier_phase(y, BPS())`) and nested
    (`lms(..., cpr=BPS())`). It lives next to its kernel and validates its
    parameters on construction. Class names are unique across the package.
  - Equalizer continuation uses `state=result.state`.
- **Randomness:** `rng: int | np.random.Generator | None` (SciPy SPEC 7).
  - Small data (bits, symbols, trajectories): generate on the host with the
    Generator, then transfer. This gives the same data on CPU and GPU.
  - Signal-sized noise: generate on the device with a CuPy generator seeded
    from it.
  - Never use a global RNG.
- **Types:** a transform is written `def f(samples: S, ...) -> S` with the
  TypeVar `S` from `core/_signal_adapter.py` (bound to `np.ndarray | Signal`),
  so a Signal gives a Signal and an array an array. Do not reassign the
  `samples` parameter; name the unwrapped array `x`.
- **Device follows the data.** There are no `backend=` or `device=` arguments
  and no global switches. Unsupported array types (JAX, PyTorch) raise
  `TypeError`.
- **No `debug_plot`.** Numerical code never imports plotting. Plot functions
  consume results or recompute through public compute functions.
- **Design and apply are separate functions**, for example `rrc_taps` and
  `fir_filter`, or `butterworth_sos` and `iir_filter`.

### Returns

- **Transforms:** array in gives array out on the same device; Signal in gives
  a new Signal out (update `sampling_rate` when the rate changes).
- **Estimators** reduce over the time axis by input rank: `(N,)` gives 0-d and
  `(C, N)` gives `(C,)`. Results stay on the input device.
- **Metrics** (`evm`, `snr`, `ber`, `ser`, `gmi`, `mi`) return host values: a
  `float` for 1-D input and `np.ndarray (C,)` for 2-D input. They **raise** on
  empty selections; they never return `None` or `0.0` for "nothing measured".
  Metrics never silently align, rotate or permute against the reference.
- **More than one value** is returned as a frozen dataclass with named fields.
  No dicts, and no tuples longer than two.
- **Equalizers** return `EqualizerResult`. `y_hat` is always an array,
  `signal` holds the 1-SPS Signal for Signal input, `state` is the
  continuation state, and diagnostics are returned as data.

### Naming

| Prefix | Meaning |
| --- | --- |
| `apply_*` | add an impairment or apply a model |
| `estimate_<quantity>(x, method)` | measure, never change the data; returns a frozen `<Quantity>Estimate` (value plus diagnostics) |
| `correct_<quantity>(x, how)` | remove the impairment; returns only the corrected data. `how` is an estimate or an algorithm object (which estimates first) |
| `resolve_*` | pick among discrete candidates (π/2 rotation, channel permutation); nothing else |
| `generate` | synthesis from parameters, takes `rng` |
| `plot_*` | every public plotting function (only `apply_default_theme` is exempt) |

- The algorithm is chosen by an object, never by the function name or a
  string: `estimate_carrier_phase(y, BPS())`, not
  `recover_carrier_phase_bps(y)`. The verb function dispatches on the object's
  type through a private table in its module. There is no `recover_*` or
  `compensate_*`.
- Computations use plain nouns (`allan_deviation`).
- **Glossary**, one name each:
  - `num_taps`, `step_size`, `sps`, `constellation`, `pulse`, `rng`,
    `num_symbols`, `num_channels`, `sampling_rate`, `symbol_rate`.
  - Units are in the docstring. Use `_db` and `_hz` suffixes where the unit is
    ambiguous.

### Errors and logging

- Wrong input raises, with a message naming the argument.
- `logger` is for diagnostics only. Per-channel logging that needs a
  device-to-host transfer is wrapped in
  `if logger.isEnabledFor(logging.INFO):`.
- The library never configures handlers, Matplotlib or warning filters at
  import time.

---

## 5. Numerics

- **Storage dtypes:** `complex64` for IQ samples and `float32` for real
  signals.
- **Accumulation precision:**
  - LMS/CMA/RDE: inputs and weights are `complex64`, but dot products and
    gradient updates in the hot loop are accumulated in `complex128`.
  - RLS keeps `P`, the gain `k` and the regressor buffers in `complex128`
    throughout. Single precision loses the Hermitian positive-definite
    property and the filter diverges.
  - Carrier phase: promote angles to `float64` before `xp.unwrap()`, because
    float32 rounding causes spurious quadrant slips. Tikhonov/Kalman block
    transitions also run in `float64`.
- **Normalization invariant:** at `sps` samples per symbol, `E[|x|²] = 1/sps`;
  at symbol rate, `E[|x|²] = 1`. Any rate-changing block applies the exact
  gain correction.
- **Shapes:** SISO is `(N,)` and MIMO is `(C, N)`. Use `as_2d`, `restore_1d`,
  `broadcast_channels` and `require_channels`; never hand-roll the
  promote/squeeze idiom.
- **Reproducibility:** a seed is reproducible within a library version only.
  Tests and baselines assert statistics, never exact noise realizations.

---

## 6. Performance

| Algorithm structure | Implementation | Device |
| --- | --- | --- |
| Vectorized: filters, resampling, spectra, impairments, metrics, LLR, frequency-domain equalizers | NumPy/CuPy via `x, xp, sp = dispatch(samples)` | follows the input |
| Sequential recursions: LMS/RLS/CMA/RDE, PLL, Tikhonov, cycle slips | Numba `@njit(cache=True, fastmath=True, nogil=True)` | CPU; GPU input does one round trip to the host and back |
| Hot spots limited by intermediates, launches or round trips | CUDA C++ via CuPy `RawModule` (`commkit/_cuda`) | GPU |

**Host-sync hygiene:**

- Never extract scalars from a possibly-GPU array inside a loop (`float(x[c])`,
  `.item()`). Compute the vector on the device, transfer it once with
  `to_device(v, "cpu")`, then loop on the host.
- Prefer on-device gathers (`xp.take_along_axis`).
- Chunk `(N, M)` intermediates over N with an on-device accumulator.
- Reduce on the device before transferring, including for plotting.
- Pass whole records to compiled code. There are no Python loops over symbols
  or small blocks, except inside a CUDA graph.

**Kernel policy:**

- A `.cu` kernel needs a benchmark showing that CuPy is limited by
  intermediates, launch count or round trips.
- It optimizes a CPU reference (Numba or NumPy) and is tested against it with
  `--device=all`.
- Wrappers check dtype, contiguity, shape and size before launch.
- `float32` with explicit literals.
- `--use_fast_math` only per kernel, after an accuracy check.
- Short elementwise chains use `ElementwiseKernel` or `cupy.fuse`.
- Sequential recursions are never ported to the GPU; one record has no
  parallel work per step.

---

## 7. Tests

- **Fixtures:** use `backend_device`, `xp` and `xpt` from `tests/conftest.py`.
  Assert with `xpt.assert_allclose(...)`, because `numpy.testing` raises on
  CuPy arrays. Cast expectations with `xp.asarray`, and reductions with
  `float(...)`.
- **Layout mirrors the source:** `tests/<subpackage>/test_<module>.py`, and
  `tests/test_<module>.py` for flat modules. Test basenames must be globally
  unique (pytest prepend import mode), hence names like
  `channel/test_channel_linear.py`. Split a large test file by concern.
- **Shared builders** live in `tests/common/` and use the public API.
- **Contract registry** (`tests/test_api_contracts.py`): every public function
  has a row. Generic checks run per row: array in/out, Signal in/out, device
  preserved, input not mutated, keyword-only parameters, `TypeError` on
  unsupported arrays, fact-conflict errors, return rules.
  - Rules a module does not follow yet are marked `xfail(strict=True)`; a
    module's migration removes its marks.
  - Do not duplicate these checks in per-module tests.
- **Oracles:** sequential kernels are tested against plain-Python reference
  implementations in `tests/common/reference_impl.py`. CUDA kernels are tested
  against their CPU reference.
- **Mathematical meaning:** test documented definitions, such as a pulse
  width, a dispersion sign, or noise-variance scaling, with independently
  derived expectations, not only shapes and energies.
- **API changes never change expected values.** A commit that changes a
  signature only changes how tests call the code. Numerical changes land in
  their own commit with independent validation.
- **Coverage** must not drop below the CI floor. Numba kernel coverage is
  measured by a separate `NUMBA_DISABLE_JIT=1` job.

---

## 8. Benchmarks

`benchmarks/` tracks the GPU-relevant hot paths. Baselines are committed under
`benchmarks/baselines/`. The reference for the 2.0 work is `0002` (`pre_v2`,
recorded in plan commit 0.4) on the reference machine: RTX 4070 Ti, Ryzen 7
7800X3D, WSL2.

- **IDs** are `[cpu]` or `[gpu]`, meaning the input device.
- **Timing:**
  - Timed bodies end with the `sync` fixture; otherwise GPU timings measure
    launches, not execution.
  - Each benchmark has one warmup round, so compilation and pool growth are
    excluded.
- **Workloads** come from `benchmarks/workloads.py` with fixed seeds; never
  generate data inline.
- **Tools:** `benchmarks/benchutils.py` provides `CudaEventTimer` and
  `nvtx_range` for `nsys` profiling.
- **Logging** is set to WARNING in `benchmarks/conftest.py`.
- **Trust deltas, not single runs.** A ±20-40% swing has been seen under load.
  Re-run a suspected regression in isolation, or do a controlled A/B with at
  least 7 repetitions.
- A commit touching a hot path quotes its delta against `0002`. Tolerance is
  5% on CPU and 10% on GPU.

---

## 9. Migration status

These are the legacy patterns that remain. Each line names the commit in
`REFACTORING_PLAN.md` that removes it. Delete a line when its commit lands, and
delete this section in commit 4.2.

The machine-checked, per-function version of this list is the `LEGACY` table in
`tests/test_api_contracts.py`. Its entries run as strict xfails, so fixing a
rule without deleting its entry fails the suite.

| Legacy pattern still in the code | Removed by |
| --- | --- |
| Read-only `Signal` bridge properties (`mod_*`, `ps_pmf`, `source_*`, pulse fields, `signal_type`) used by unmigrated modules | Module passes, deleted in 4.1 |
| `resolve_required` / `resolve_optional` (Signal wins, with a warning) | 2.5, then each module pass |
| `generate_qam/psk/pam/psqam`, `seed=` with `RandomState` | 2.6, then each module pass |
| `modulation=` / `order=` / `unipolar=` / `pmf=` parameters; positional parameters; `float \| ndarray` and dict returns; metrics returning `None` | Module passes 3.1-3.10 |
| Test-only private exports in the `plotting` `__init__` | 3.10 |
| `--use_fast_math` as the global CUDA default | Step 5 |
