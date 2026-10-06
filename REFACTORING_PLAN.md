# CommKit 2.0 refactoring plan

Supersedes and replaces the former `ARCHITECTURE_MODULARITY_PLAN.md` and
`commkit-roadmap.md`, both deleted; their still-valid content is folded in
below. Phases 0-2 of the modularity plan (Signal pipeline characterization,
copy-efficient replacement, the `SignalAdapter` boundary; commits `5f026b9`,
`757c98a`, `be344e9`, `f492f91`, `b210ed7`) are complete and are the starting
point for this plan.

Baseline at plan time (2026-10-06, `6087d41` + working tree): CPU suite
**1173 passed in 17 s**, line coverage **78 %** (Numba kernels report 9 %
because compiled code is not traced; see §5). Benchmark baseline
`0001_commkit_baseline` (2026-09-02, `33f7122`).

There are no external users, so 2.0 is a clean break: no deprecation shims,
no forwarding exports, no migration tables. A short CHANGELOG entry suffices.

---

## 1. Goals

1. **Comfortable for researchers and engineers.** One obvious way to do each
   thing, readable call sites, visible mathematical parameters, everything
   expressed in Python code. No configuration files, registries, or setup steps.
2. **Proper Python.** Typed frozen value objects, keyword-only parameters,
   predictable return types, errors instead of warnings for wrong input, and no
   import side effects or global switches.
3. **Maintainable.** Every concern has one owner, there is one CPU reference per
   algorithm, and the surface is small enough to test as a contract.
4. **Fast.** Execution follows the algorithm's structure (§2.7). Every
   performance claim is backed by `benchmarks/`.

---

## 2. Target design

### 2.1 The model in one picture

```text
plain arrays (NumPy / CuPy)          value objects (frozen, inline, no I/O)
        \                            Constellation, RRC/RC/Gaussian/Rect pulses,
         \                           PLL / BPS / CycleSlip, EqualizerState
          v                                   |
       Signal  = samples + facts + description + reference truth
          |
          v
   functions (estimate_* / correct_* / recover_* / apply_* / generate ...)
          |                \
          v                 v
   Signal or array      typed result dataclasses (EqualizerResult, ...)
                            |
                            v
                  plotting (consumes data, never produces it)
```

Algorithms are **functions**. Things that *describe* a modulation, a pulse, or
an embedded sub-algorithm are **small frozen value objects** written inline in
the call. Data is a **`Signal` or an array**. There is no pipeline object,
receiver class, experiment runner, or configuration file.

The 2.0 workflow this design targets:

```python
import commkit as ck
from commkit import Constellation, RRC
from commkit.recovery import BPS, CycleSlip

tx = ck.generate(Constellation.qam(16), num_symbols=2**16, symbol_rate=32e9,
                 sps=2, pulse=RRC(rolloff=0.1), rng=1)
rx = ck.impairments.apply_phase_noise(tx, linewidth=100e3, rng=2)
rx = ck.impairments.apply_awgn(rx, esn0_db=18, rng=3).to("gpu")  # explicit move
rx = ck.filtering.matched_filter(rx)                  # pulse taken from rx.pulse

res = ck.equalization.lms(
    rx, num_taps=21, step_size=1e-3,
    training_symbols=rx.reference.symbols[..., :2000],
    cpr=BPS(test_phases=64, cycle_slip=CycleSlip(history=100)),
)
y = res.signal                                        # 1-SPS Signal, reference aligned
print(ck.metrics.evm(y), ck.metrics.ber(y))           # host floats

res2 = ck.equalization.lms(rx_next, num_taps=21, step_size=1e-3,
                           state=res.state)           # continuation
```

### 2.2 `Signal`: waveform plus description, no pipeline state

`Signal` becomes a `@dataclass(frozen=True, slots=True)`. Validation happens in
`__post_init__`, and updates go through a validating `sig.replace(...)` that is
built on `dataclasses.replace`. This replaces Pydantic. Pydantic adds nothing for
`Any`-typed array fields, and it is the reason the type checker sees `Any`
everywhere.

| Field | Meaning |
| --- | --- |
| `samples` | NumPy/CuPy array, `(N,)` or `(C, N)`, time last |
| `sampling_rate`, `symbol_rate` | **Facts** (Hz); `sps` is a derived property |
| `constellation: Constellation \| None` | Replaces `mod_scheme`, `mod_order`, `mod_unipolar`, `ps_pmf`, `ps_nu` |
| `pulse: Pulse \| None` | Replaces `pulse_shape`, `filter_span`, `rrc_rolloff`, `rc_rolloff`, `duty_cycle`, `rise_time` |
| `reference: Reference \| None` | Known transmitted `symbols` / `bits` (the ground truth) |
| `frame: FrameLayout \| None` | Segment layout aligned with `reference` |
| `center_frequency` | Physical fact (Hz) |

Removed:

- `resolved_symbols` and `resolved_bits`. These are pipeline state stored on the
  input. Metrics compute decisions on demand from a 1-SPS Signal. This deletes
  the whole cache-invalidation policy.
- `signal_type`, `spectral_domain`, `physical_domain`,
  `digital_frequency_offset`. Nothing reads them.
- `pilot_tone_*`, unless the polarization demultiplexer starts using them as its
  defaults. The rule is that a field must be read by at least one algorithm.

Construction does **no hidden work**:

- no bit-to-symbol mapping and no normalization (factories do that),
- no GPU placement (`.to("gpu")` is the only path),
- no shape guessing (a `(N, C)`-looking input raises instead of being
  transposed).

`print_info()` becomes `_repr_html_` and `__repr__`, so notebooks display a
Signal automatically.

**Alignment invariant:** `reference` symbol *k* corresponds to symbol period *k*
of `samples`. The few functions that drop or shift symbol periods (equalizer
trimming, timing correction, payload extraction) slice `reference` to match.
Metrics raise if lengths disagree; they never guess an alignment.

### 2.3 Value objects

All value objects are frozen dataclasses with read-only array fields, validated
on construction, and plain NumPy inside, so commax can produce them.

- **`Constellation`** (`commkit.mapping`, re-exported at top level). Built with
  `Constellation(points, bit_labels=None, pmf=None)` for arbitrary or learned
  constellations, or with `.qam(M)`, `.psk(M)`, `.pam(M, unipolar=)`, `.ask(M)`.
  Shaping uses `.qam(64).shaped(nu=...)` or `.shaped(entropy=...)`. Functions
  accept **objects only, no strings**. It replaces `modulation` / `order` /
  `unipolar` / `pmf` across all 24 modules.
- **Pulses** (`commkit.filtering`, re-exported): `RRC(rolloff, span)`,
  `RC(...)`, `Gaussian(...)`, `Rect(duty_cycle, rise_time)`, `SmoothRect(...)`,
  each with `.taps(sps)`. Wherever a pulse is accepted, a raw taps array is
  accepted too.
- **CPR sub-algorithms** (`commkit.recovery`): `PLL(bandwidth, mu=None,
  beta=None)`, `BPS(test_phases=64, block_size=32, joint_channels=False,
  cycle_slip=None)`, `CycleSlip(history=100, threshold=pi/4)`.
  - Equalizers take `cpr=`.
  - The standalone `recover_carrier_phase_*` functions keep their own parameters
    flat and take only the nested `cycle_slip=` object.
  - There is no generic `recover_carrier_phase(symbols, cpr)` dispatcher.
- **`EqualizerState`**. One `state=` argument (taken from `result.state`) replaces
  the four continuation parameters `w_init`, `samples_prefix`,
  `input_norm_factor` and `cpr_state` (about 250 references). User-supplied
  initial taps remain a plain `initial_taps=` argument.

No config files and no YAML: `io.py` stores metadata as a JSON string. That
removes PyYAML and `allow_pickle=True`.

### 2.4 Function signature rules

- `def f(data, other_data, *, params...)`: everything after the data arguments
  is **keyword-only**.
- Verb prefixes stay as they are (`estimate_` / `correct_` / `recover_` /
  `resolve_` / `apply_`; `generate` for synthesis; `plot_` in plotting).
- Randomness uses `rng: int | np.random.Generator | None` (SciPy SPEC 7)
  everywhere. A CuPy generator for large on-device noise is seeded from it.
- No `backend=`, `device=`, `update_mode=` or `debug_plot=` arguments, and no
  global switches (`use_cpu_only` is removed).
- A one-time **naming glossary** is applied in 2.0. Examples: `num_taps`,
  `step_size`, `sps`, `constellation`, `rng`, unit suffixes `_db` / `_hz`
  where ambiguous. The glossary lives in `AGENTS.md`.
- **Types:** transforms use one constrained TypeVar,
  `S = TypeVar("S", np.ndarray, cp.ndarray, Signal)`, written
  `def fir_filter(x: S, taps) -> S`. This gives exact editor types without
  per-function overloads. `ArrayType` becomes `np.ndarray | cp.ndarray`, with no
  `Any`.

### 2.5 Metadata rule: facts versus choices

> **The Signal describes the data; your arguments describe what you want done.**

- **Facts** about the samples (`sampling_rate`, `symbol_rate`, hence `sps`,
  `center_frequency`) come from the Signal.
  - Passing the same value is allowed.
  - Passing a **conflicting value raises `ValueError`**, because a conflict means
    the call or the metadata is wrong, and continuing would produce wrong output
    with plausible-looking metadata.
- **Choices** (the decision `constellation`, `noise_var`, `pulse` used by a
  matched filter) default to the Signal's value. An explicit argument wins
  silently, because it is deliberate. Using a QPSK partition for CPR on 16-QAM
  is one example.
- **Array input:** a missing fact raises.
- No warnings in either case. Warnings scroll past unnoticed in notebooks.

This replaces both "Signal always wins with a warning" (the current code) and
"explicit always wins" (the roadmap). The `SignalAdapter` keeps owning it, as
`resolve_fact()` and `resolve_choice()`.

### 2.6 Return rules

- **Transforms:** array in gives array out on the same device; Signal in gives a
  new Signal out.
- **Estimators** reduce over time following the input's rank: `(N,)` gives 0-d
  and `(C, N)` gives `(C,)`. Results stay on the input device, and the shape
  never depends on the channel count.
- **Metrics** (`evm`, `snr`, `ber`, `ser`, `gmi`, `mi`) are the reporting layer.
  They return host values: a `float` for 1-D input and an `np.ndarray (C,)` for
  2-D input. They **raise** on empty selections or frame-level Signals; they
  never return `None` or `0.0`.
- **More than one value** comes back as a frozen dataclass with named fields:
  no dicts, and no tuples longer than two elements. This affects
  `analysis.drift`, `analysis.linewidth`, `metrics.evm` and others.
- **Equalizers** return `EqualizerResult`:
  - `y_hat` is always an array,
  - `signal` is the 1-SPS Signal when the input was a Signal,
  - `state` is the continuation state,
  - plus diagnostics as data.
- **Errors versus logging:** wrong input raises. `logger` is for diagnostics
  only and is gated by `isEnabledFor`. Library code never uses `warnings.warn`
  for user mistakes.

### 2.7 Execution model (from the roadmap, trimmed)

| Algorithm structure | Implementation | Device |
| --- | --- | --- |
| Vectorized: filtering, resampling, spectra, impairments, metrics, LLR, frequency-domain equalizers | NumPy / CuPy through `dispatch()` | follows the input |
| Sequential recursions: LMS/RLS/CMA/RDE, PLL, Tikhonov, cycle slips | Numba `@njit` | CPU; GPU input does one round trip to the host and back |
| Hot spots where CuPy is limited by intermediates or launches | CUDA C++ via `RawModule`, tested against the CPU reference | GPU |

- **The device follows the data.** JAX, PyTorch and other array types raise
  `TypeError`. Interoperability is documented through DLPack only.
- **JAX is removed.** The 2026-09-02 baseline shows Numba beating every JAX path:
  - sequential LMS: 8 ms on Numba against 608 ms on GPU JAX,
  - time-domain block LMS: 14 ms on CPU JAX, 3.7 s on GPU through NumPy/CuPy.

  Time-domain block mode is therefore removed as well. `block_lms`, `block_cma`
  and `block_rde` (frequency-domain) stay as separate functions.
- **`--use_fast_math` per kernel,** after an accuracy check, instead of as the
  global default.
- **Dependencies:**
  - core: numpy, scipy, numba, matplotlib,
  - `gpu` extra: cupy,
  - removed: jax, pydantic, pyyaml.
- **Imports:** `import commkit` has no side effects. It sets no theme, adds no
  log handler, changes no global warning filter, and makes no GPU allocation.
  Subpackages load lazily through module `__getattr__`, so `ck.plotting` still
  works without an explicit import.

**Kernel policy:**

- A `.cu` kernel is added only when `benchmarks/` shows the CuPy path is limited
  by intermediate memory, launch count or host round trips.
- Each kernel optimizes a CPU reference and is tested against it; it is never a
  separate algorithm.
- Python wrappers check dtype, contiguity, shape and size limits before launch,
  because kernels do not check bounds.
- Kernels use `float32` by default, with explicit literals.
- Short elementwise chains use `ElementwiseKernel` or `cupy.fuse`, not `.cu`
  files.

**Why CUDA C++ through CuPy and not numba-cuda:**

- numba-cuda adds Python dispatch to every launch, which dominates kernels that
  run once per block.
- `RawKernel` launches run on CuPy's current stream, so the block equalizers'
  CUDA-graph capture records them.
- Numba types float literals as float64, which is slow on GPUs with weak FP64.
- C++ templates and CuPy's on-disk compile cache are already in use.

The price is one CPU twin per kernel, which is acceptable while kernels are
few.

**Performance guidelines:**

- Pass whole records to compiled code. There are no Python loops over symbols
  or small blocks, unless they are captured in a CUDA graph.
- Avoid `(N, M)` and `(N, C, C)` intermediates where a kernel can loop
  internally; otherwise chunk over N with an on-device accumulator.
- Reduce on the device before transferring (binning, decimation, slicing),
  including for plotting.
- Storage is `complex64`/`float32`. Accumulators are promoted to double where
  `AGENTS.md` requires it: LMS/CMA dot products, all of RLS, and phase
  unwrapping.

### 2.8 Relationship to commax

commax is a separate JAX library containing only blocks that must be
differentiated: end-to-end learning, geometric and probabilistic shaping,
learned DSP, and model fitting through a receiver chain.

- commax depends on commkit, never the reverse.
- commax takes its conventions from commkit (`Constellation`, bit labels, pulse
  taps, frame layouts). This is why the value objects are plain NumPy and
  constructible from arbitrary arrays.
- commkit functions are the reference implementations, and commax blocks are
  tested against them.
- Trained parameters (points, taps, a PMF) come back as plain arrays and are
  validated in commkit's simulation chain and on captures.
- Data crosses between the libraries through DLPack.

A function goes into commax only if it must be differentiated; everything else
is implemented once, in commkit.

### 2.9 What we deliberately do not build

Pipeline or receiver objects, config files or registries, a public "config
object per algorithm", methods for DSP on `Signal`, string shorthands for
constellations, or a generic Array-API protocol hierarchy.

**Placeholder modules are kept.** `commkit.coding` and
`commkit.impairments.channel.nonlinear` are not implemented yet, but they stay
as reminders of planned work. Do not delete them, and do not count them as dead
code. They are excluded from the contract registry and the coverage floor. The
README must label them "planned, not implemented".

---

## 3. Decision record

| # | Decision | Status |
| --- | --- | --- |
| D1 | Remove JAX entirely; commax will own differentiable code | Decided |
| D2 | `block_*` (frequency-domain) stay separate functions; time-domain block mode removed | Decided |
| D3 | Metadata rule: facts versus choices (§2.5) | Decided |
| D4 | `Signal`, `Frame`, `Preamble` become frozen dataclasses; Pydantic removed | Decided |
| D5 | Single 2.0 break, no shims | Decided |
| D6 | `Constellation` objects only | Decided |
| D7 | No config files; value objects are written inline | Decided |
| D8 | `resolved_*` caches removed; `reference` + `frame` layout on Signal | Decided |
| D9 | `Pulse` value objects replace six pulse fields | Decided |
| D10 | `state=` replaces four continuation parameters | Decided |
| D11 | One `generate(constellation, ...)` replaces `generate_qam/psk/pam/psqam` | Decided |
| D12 | Return rules (§2.6): rank rule for estimators, host values for metrics | Decided |
| D13 | Lazy subpackage loading at top level | Decided |
| D14 | `AGENTS.md` is the only agent guide; there is no `CLAUDE.md` | Decided |
| D15 | Placeholder modules (`coding`, `channel.nonlinear`) are kept as reminders; never delete them | Decided |

All decisions were confirmed on 2026-10-06. Reopening one requires a recorded
reason in this table.

---

## 4. Work plan

### Ordering principles

1. **Shrink first.** Delete JAX, block mode, `debug_plot`, global switches and
   dead fields *before* restructuring anything, so no code is refactored and then
   deleted.
2. **Touch each module once.** Rather than separate package-wide passes for
   `Constellation`, `rng`, keyword-only, naming, returns, metadata, typing,
   helper ownership and tests, each module gets **one pass** that applies the
   full checklist below.
3. **Bottom-up.** A module is migrated after everything it calls, so callers
   always target the new API.
4. **Temporary bridge.** While modules are migrating, `Signal` carries read-only
   compatibility properties (for example `mod_scheme` derived from
   `constellation`). They are deleted in step 4. This keeps every commit green
   without touching modules twice.
5. **Never change expected numerical values in a commit that changes the API.**
   Test diffs in migration commits are call-site changes only. Intentional
   numerical changes land separately with independent validation.

Sizes: S = under a day, M = 1-3 days, L = more than 3 days.

### Workflow

- **Branch.** All work happens on a long-lived `v2` branch. Open one draft pull
  request from `v2` into `main` at the start, so CI runs on every push. `main`
  stays at 1.1.0 until the release.
- **Commit gate.** Every commit passes the following before it is made:

  ```bash
  uv run ruff format . && uv run ruff check . --fix && uv run mypy commkit/
  uv run pytest              # CPU + GPU, about 50 s on the reference machine
  ```

  Commits that touch hot paths also run the affected benchmark file against
  `0002_pre_v2` and quote the delta in the commit message.
- **Reference machine.** RTX 4070 Ti, Ryzen 7 7800X3D, WSL2. CuPy works there
  and the full CPU + GPU suite passes (2284 tests in 50 s at plan time). All
  baselines and GPU checks are recorded on it.
- **Just-in-time detail.** Steps 0-2 below are listed commit by commit. Before
  starting each Step 3 module pass, write its commit list into this plan,
  following the template in Step 3. Interfaces fixed in Step 2 decide those
  details, so writing them earlier would be guesswork.
- **Progress tracking.** Tick each commit's box here, in the same commit.

### Step 0: safety net and tooling

There are no library changes in this step, only documentation, tests,
benchmarks and CI.

- [x] **0.1 `docs: add 2.0 refactoring plan`.** Commit `REFACTORING_PLAN.md` and
  open the draft pull request.
- [x] **0.2 `docs: rewrite AGENTS.md for 2.0; drop CLAUDE.md`.** Write it to the
  §7 outline. The "migration status" section lists every 2.0 rule the code
  does not follow yet, each pointing to the commit that will fix it. Done when
  no section describes JAX, and no section contradicts §2 except as a
  migration-status item.
- [x] **0.3 `bench: add LLR and import-time benchmarks`.**
  - `benchmarks/bench_llr.py` covers `compute_llr` (maxlog and exact) and `gmi`
    on 16-, 64- and 256-QAM with N = 1e6 symbols, on `[cpu]` and `[gpu]`. Its
    workload goes in `workloads.py`.
  - `benchmarks/bench_import.py` measures `import commkit` in a fresh
    subprocess: wall time and the number of modules in `sys.modules`.
- [x] **0.4 `bench: record 0002_pre_v2 baseline`.** Run
  `uv run pytest benchmarks/ --benchmark-only --device=all --benchmark-save=pre_v2 --benchmark-storage=file://benchmarks/baselines`
  on the reference machine, then delete `0001`. The commit message records the
  hardware and the commit hash.
- [ ] **0.5 `test: add pure-Python reference oracles for sequential equalizers`.**
  - `tests/common/reference_impl.py` gets plain-loop LMS, RLS, CMA and RDE in
    SISO and 2x2 butterfly form, written from the textbook equations with
    float64/complex128 throughout.
  - `tests/equalization/test_oracles.py` compares the Numba kernels to them on
    N ≈ 300 symbols. It checks outputs, final weights and errors at rtol 1e-5.
- [ ] **0.6 `test: add reference oracles for PLL, BPS, Viterbi-Viterbi and cycle-slip correction`.**
  Same pattern, in `tests/recovery/test_oracles.py`.
- [ ] **0.7 `ci: measure Numba kernel coverage and add a coverage floor`.**
  - A new CI job runs `tests/equalization tests/recovery` with
    `NUMBA_DISABLE_JIT=1 --cov`. Kernels that cannot run without JIT are
    listed in `AGENTS.md`, not forced.
  - The main job gets `--cov-fail-under=78`.
  - Placeholder modules are omitted from coverage (`[tool.coverage.run] omit`).
- [ ] **0.8 `test: add API contract registry`.**
  - `tests/test_api_contracts.py` holds one row per public function (§5.2).
  - Checks that hold today must pass.
  - Rules that only hold in 2.0 (keyword-only parameters, `TypeError` on
    unsupported arrays, raising on fact conflicts, return rules) are recorded
    per function as `xfail(strict=True)`. A module pass therefore *must*
    remove its xfail marks: a strict xfail that starts passing fails the suite.
    The remaining marks are the live migration checklist.
  - The meta-test fails if any public `__all__` name is missing from the
    registry.

### Step 1: deletions and global hygiene

Tests for removed features are deleted; the remaining tests change only at call
sites. The order avoids conflicts, because 1.4-1.6 all edit
`equalization/sequential/_dd.py`.

- [ ] **1.1 `refactor(io)!: store metadata as JSON, load without pickle`.**
  - `save_npz` writes the metadata as a 0-d unicode array of JSON text;
    `load_npz` uses `allow_pickle=False`.
  - PyYAML is removed from the dependencies.
  - The test checks that loading a file containing an object array raises.
- [ ] **1.2 `refactor!: explicit device placement`.**
  - Remove `use_cpu_only`, the automatic GPU move in `Signal.model_post_init`,
    and `load_npz(device="auto")`, which now defaults to `"cpu"`.
  - `Signal.to()` returns a new Signal instead of modifying itself.
  - `tests/conftest.py`, `benchmarks/conftest.py`, `test_backend.py` and
    `test_cuda_infra.py` stop toggling the flag.
  - GPU tests build their inputs with `xp` or call `.to("gpu")` explicitly.
  - These three must land together: without the flag, CPU tests would
    otherwise move Signals to the GPU.
- [ ] **1.3 `refactor!: no side effects on import`.**
  - No `apply_default_theme()` call on import.
  - The logger gets a `NullHandler`. `set_log_level()` keeps working and
    attaches the colour handler on request.
  - The global `warnings.filterwarnings` call becomes a local
    `catch_warnings` where CuPy JIT is used.
  - The CuPy functional probe (`cp.arange(1)`) runs lazily on first GPU use
    and is cached.
  - Subpackages are loaded lazily through module `__getattr__`.
  - Process-level tests (§5.4) are added. The commit quotes `bench_import`
    before and after.
- [ ] **1.4 `refactor!: remove debug_plot`.**
  - Covers all 20 modules and the 4 test files that use it.
  - For each plot that needed internal data, check that a public compute
    function provides it (for example timing correlation, the FOE spectrum);
    add such a function only if none exists.
- [ ] **1.5 `refactor(equalization)!: remove time-domain block update mode`.**
  - Removes `update_mode`, `block_len`, `_block/_seqmode.py`'s `xp` and JAX
    block runners, `tests/equalization/test_block_update.py`, and the
    `bench_lms_block` legs.
  - `block_lms`, `block_cma` and `block_rde` (frequency-domain) are untouched.
- [ ] **1.6 `refactor(equalization)!: Numba-only sequential equalizers`.**
  - Removes the `backend=` and `device=` parameters, the JAX branches in
    `sequential/_dd.py`, `sequential/_blind.py` and `_common.py`,
    `_kernels_jax.py`, `test_sequential_jax.py`, and the JAX legs in
    `bench_equalizers.py`.
  - Removes `CPRState.jax_bps_*`.
  - The 0.5 oracles must still pass unchanged.
- [ ] **1.7 `refactor(mapping)!: NumPy/CuPy LLR`.**
  - `compute_llr` is rewritten as maxlog/exact over chunks of N with an
    on-device accumulator, following `metrics.mi`. The `output=` parameter is
    removed.
  - Expected values in `test_llr.py` stay unchanged; only the `output=`
    arguments are dropped.
  - Compare against `bench_llr` in 0002. If the GPU is more than 2x slower than
    the old JAX path, the follow-up fused kernel becomes the next commit.
- [ ] **1.8 `refactor!: remove JAX`.**
  - Removes `backend._get_jax`, `to_jax` / `from_jax` / `is_jax_array`,
    `Signal.export_samples_to_jax` / `update_samples_from_jax`, and the JAX
    note in `analysis/__init__`.
  - In tests: the `jax` fixture, the session-wide `_ensure_jax_precision`
    (which silently ran every JAX test in float64, a precision users never
    got), and `tests/common/conversions.ensure_jax_x64`.
  - Removes the dependency and the extras.
  - `dispatch()` raises `TypeError` for anything that is not NumPy or CuPy.
  - Done when `grep -ri jax commkit tests benchmarks pyproject.toml` finds
    nothing.

### Step 2: core foundations

- [ ] **2.1 `feat(mapping): Constellation value object for 2.0`.** This is
  additive; the old functions keep working.
  - Read-only arrays, plus the constructor `Constellation(points,
    bit_labels=None, pmf=None)`.
  - `.qam/.psk/.pam/.ask` factories and `.shaped(nu=...|entropy=...)`.
- [ ] **2.2 `feat(filtering): pulse value objects`.** This is additive.
  - `RRC`, `RC`, `Gaussian`, `Rect` and `SmoothRect` with `.taps(sps)`. They
    delegate to the existing `*_taps` functions, and tests compare them to
    those functions.
- [ ] **2.3 `refactor(core)!: Signal, Preamble and SingleCarrierFrame become frozen dataclasses`.**
  - A container swap with the *same fields*; Pydantic is removed.
  - Validation moves to `__post_init__`, and `sig.replace(**changes)`
    replaces `replace_samples`.
  - The 8 in-place assignments in the library and 59 in tests are rewritten
    to use `replace`.
  - `_repr_html_` replaces `print_info`.
- [ ] **2.4 `refactor(core)!: Signal 2.0 fields with compatibility bridge`.**
  - Adds `constellation`, `pulse`, `reference` and `frame` (a layout
    snapshot) and removes the dead fields (§2.2).
  - Construction does no implicit mapping, normalization or shape guessing;
    the factories do that work.
  - Read-only bridge properties (`mod_scheme`, `mod_order`, `source_symbols`,
    ...) keep unmigrated modules working.
  - `resolved_*` stays, marked bridge-only, until 3.8.
- [ ] **2.5 `refactor(core): facts/choices metadata resolution and SPS tolerance`.**
  - Adds `resolve_fact()` and `resolve_choice()` to the adapter, plus the
    near-integer SPS rule.
  - The old `resolve_required` / `resolve_optional` stay until the last module
    pass.
- [ ] **2.6 `refactor(core)!: single generate()`.**
  - `generate(constellation, num_symbols, *, symbol_rate, sps, pulse, num_channels, rng)`
    always generates on the CPU.
  - Absorbs `generate_qam/psk/pam/psqam` and the generation helpers from
    `helpers.py`.
  - `SingleCarrierFrame` layout methods are testable without generating data,
    and the effective `payload_len` is explicit.
  - `tests/common/signals.py` builders move onto `generate`.

### Step 3: module passes (bottom-up)

**Per-module checklist.** Every item is done in the same pass:

- [ ] `constellation=` / `pulse=` / `rng=` replace the old parameters
- [ ] Keyword-only parameters, glossary names, units in docstrings
- [ ] Facts/choices resolution through the adapter
- [ ] §2.6 return rules; results become frozen dataclasses
- [ ] Helpers owned by this module move in from `helpers.py`
- [ ] TypeVar signatures, and the module added to the strict mypy override list
- [ ] Tests migrated: call sites only, plus a contract-registry entry, plus
      definition-level checks where today's tests are only structural
- [ ] Benchmarks touching the module are compared to `0002_pre_v2`

**Commit template for each pass** (expanded into concrete commits in this plan
before the pass starts):

1. `refactor(<module>): take ownership of <helpers>`. A pure move out of
   `helpers.py`, with imports updated. No behavior change.
2. `refactor(<module>)!: 2.0 signatures`. Applies the checklist above, migrates
   the test call sites, removes the module's strict-xfail marks in the contract
   registry, and deletes the Signal/array parity tests the registry now covers.
3. Optionally, `test(<module>): definition-level checks`. Only where today's
   tests are structural (shape or energy) rather than mathematical.
4. Optionally, `perf(<module>): ...`. Only when commit 2's benchmark comparison
   shows a regression above the tolerance.

The equalization pass (3.7) gets more commits:

- prepare/run/assemble decomposition with unchanged behavior,
- then `cpr=`,
- then `state=` together with the chunked-equivalence tests,
- then `result.signal`.

| Order | Pass | Notes | Size |
| --- | --- | --- | --- |
| 3.1 | `backend`, array helpers | `_array.py` (`as_2d`, `restore_1d`, `broadcast_channels`, ...), `rms` / `normalize` / dB helpers into `commkit.math`, `format_si` into a small display module. No `utils/` package and no one-function files. Array helpers import nothing from `core`, plotting or DSP modules, which breaks the `core -> helpers -> core` cycle. | S |
| 3.2 | `mapping` | gray, bits, llr, shaping all take `Constellation` | M |
| 3.3 | `filtering`, `multirate`, `spectral`, `smoothing` | Pulses (add a definition-level check that the Gaussian `duty_cycle` equals the FWHM); overlap-save into private `_overlap_save.py`; `resolve_symbols` becomes plain decimation to a 1-SPS Signal. Chromatic dispersion moves into private `_dispersion.py`, which owns the D/wavelength/length to beta2·L conversion, the frequency grid, and both forward and inverse transfer functions, with an explicit sign. It gets independent sign and unit tests, because a round trip alone hides errors shared by both directions. | M |
| 3.4 | `impairments` | `rng`; uses `_dispersion.py` | S |
| 3.5 | `timing`, `frequency` | `cross_correlate_fft`, peak interpolation and `zc_mimo_root` move here; timing correction slices `reference` | M |
| 3.6 | `recovery` | `PLL` / `BPS` / `CycleSlip` objects; PLL gain resolution moves into `recovery/_common.py` | M |
| 3.7 | `equalization` | API plus the internal decomposition (old Phase 3), done once: validate, then `_prepare()`, then Numba or NumPy/CuPy runner, then `_assemble_result()`. Adds `state=`, `cpr=` and `result.signal`. Includes chunked-versus-uninterrupted equivalence tests for `state=`. See the equalizer safety rules below. | L |
| 3.8 | `metrics` | Host return values, raise on empty input, `reference`-based Signal path, payload extraction (`extract_payload(sig)` using `frame`), a units and scaling table. See the metrics contract below. | M |
| 3.9 | `analysis` | Typed result dataclasses instead of dicts; trend fitting lives here | S |
| 3.10 | `plotting` | Consumes the new results and Signals; recomputes through public functions; no numerical module imports matplotlib (tested) | M |

**Equalizer safety rules (3.7):**

- Keep the dtype rules: complex128 accumulation in LMS/CMA, and float64 for all
  RLS state.
- Do not merge RLS and LMS state representations for symmetry.
- Parity checks cover outputs, final weights, phase trajectories, CPR state and
  training counts, not only `y_hat`.
- Invalid combinations are rejected before any compilation, allocation or
  transfer.
- Remove the test-only exports from `equalization/__init__.py` (`_get_numba`,
  `_check_rls_divergence`, ...). Tests import from the owning module and patch
  where a name is looked up. The same applies to plotting.
- Timing reports distinguish compilation, steady-state execution and transfers.

**Metrics contract (3.8):**

- An empty selection after training removal or trimming raises. It is never
  reported as zero errors.
- The units table defines sample power, symbol energy, Es/N0, EVM, SNR, and
  complex noise variance. It distinguishes per-complex-sample from
  per-quadrature variance and documents SPS and PS-QAM scaling. Units are
  expressed through argument names, not a units object.
- Metrics never silently align, permute or rotate the output against the
  reference, because that would mask receiver defects. Synchronization,
  ambiguity resolution and channel permutation stay explicit steps.

**Frame rules (2.2 and 3.8):**

- Payload extraction uses known alignment only. Unknown alignment after
  arbitrary transforms raises a useful error rather than producing a plausible
  wrong metric.
- `frame` on a Signal is an immutable layout snapshot, not live mutable state.
- The same seed produces the same layout and data.

### Step 4: close-out

| Task | Size |
| --- | --- |
| 4.1 Delete the compatibility bridge and `helpers.py`; check that no module imports a removed name. | S |
| 4.2 Final `AGENTS.md` pass (drop the migration-status section); README rewritten around §2.1. | S |
| 4.3 Examples: the new `qam_receiver_quickstart.py` plus the five existing examples migrated; all run as a CI smoke test at small sizes. | M |
| 4.4 Full GPU suite and full benchmark run; save `0003_v2_0`; CHANGELOG entry; bump to 2.0.0. | S |

### Step 5: performance follow-ups (independent, any time after step 3.6)

- A Numba CPU path for non-square BPS (1256 ms on CPU against 26 ms on GPU in
  the baseline).
- Per-kernel `--use_fast_math` with accuracy tests.
- `block_lms` at small block sizes on GPU (242 ms on GPU against 47 ms on CPU):
  profile the launch overhead.

---

## 5. Tests

### 5.1 Principles

- Fixtures (`backend_device`, `xp`, `xpt`) and the mirrored layout stay.
- Shared test builders in `tests/common/` use the public 2.0 API (`generate`,
  `Constellation`) instead of hand-assembling signals with `gray_constellation`
  and `normalize`.
- Tests assert statistics and definitions, never noise realizations. The `rng`
  migration therefore needs no golden data.
- Per-module Signal-vs-array parity tests are **deleted** once the contract
  registry covers that function. This is the main deduplication.

### 5.2 Contract registry (`tests/test_api_contracts.py`)

One table lists each public function with an input builder, its kind
(`transform`, `rate_change`, `estimate`, `metric`) and minimal keyword
arguments. Parametrized checks run for every row:

- array in gives array out; Signal in gives a new Signal out with the expected
  rate and alignment;
- the device is preserved (GPU); the input is not mutated;
- parameters after the data arguments are keyword-only (`inspect.signature`);
- unsupported array types raise `TypeError`; a conflicting fact raises
  `ValueError`; a choice override is honoured;
- estimators follow the rank rule; metrics return host values.

A meta-test fails if any name in a public `__all__` is missing from the
registry, so new functions are forced to follow the rules.

### 5.3 Oracles and kernels

- Each Numba kernel is tested against the pure-Python oracle (commits 0.5 and 0.6) on small
  inputs with tight tolerances.
- Each `.cu` kernel is tested against the Numba/NumPy CPU reference
  (`--device=all`).
- The kernel coverage job (commit 0.7) measures the kernels' line coverage.

### 5.4 Process-level tests (fresh subprocess)

- `import commkit` leaves `rcParams`, logging handlers and warning filters
  unchanged and allocates nothing on the GPU.
- Importing numerical subpackages does not load matplotlib.
- Constructing a Signal from NumPy input never touches CuPy.

### 5.5 Coverage

The floor is the current total. Kernel coverage is reported separately. Coverage
must not drop in any module pass, because removed code takes its tests with it.

---

## 6. Benchmarks

- **Baselines:**
  - `0002_pre_v2` (commit 0.4) is the comparison point for the whole refactor.
  - `0003_v2_0` (task 4.4) is the release baseline.
  - `0001` is deleted after 0002 exists; its IDs no longer exist.
- **IDs** become `[cpu]` / `[gpu]` (the input device). The `-numba` / `-jax` /
  `-xp` legs disappear with D1 and D2.
- **New:** `bench_llr.py`, `bench_import.py`, and a Signal-construction leg in
  `bench_signal_pipeline.py`. **Removed:** the JAX and time-domain block legs.
- **Workloads** move to the new API in the pass that changes the API they use.
  Seeds stay fixed. The data changes only where `rng` replaces `RandomState`,
  which does not matter for timing.
- **Gate:** each pass touching a hot path quotes its delta against 0002 in the
  commit message. A regression above 5 % (CPU) or 10 % (GPU) is re-run in
  isolation per the methodology rules before it counts.

---

## 7. `AGENTS.md` rewrite (commit 0.2)

`AGENTS.md` is the single guide for all coding agents, including Claude Code;
there is no `CLAUDE.md`. The current file is 532 lines and nearly identical to
the old `CLAUDE.md`.

**What is stale or misplaced:**

- the JAX sections;
- the incomplete test-file listing (no `test_generation.py`);
- the benchmark table, which is missing `bench_block_blind` and
  `bench_signal_pipeline`;
- a roughly 100-line Signal-adapter tutorial that belongs in the adapter
  docstring;
- historical measurements presented as rules.

**Target: about 250 lines of rules, not inventory.**

1. What CommKit is, how it relates to commax, and the philosophy (§2.1, §2.8, §2.9)
2. Commands: uv, tests, benchmarks, and the CI gate (format, lint, mypy, tests)
3. Architecture: module responsibilities, the allowed dependency direction, and the placeholder modules that must not be deleted (D15)
4. API rules: Signal, value objects, signatures, the facts/choices rule, return
   rules, the naming glossary, errors versus logging
5. Numerics: dtypes and precision (including complex128 accumulation and RLS in
   float64), the normalization invariant, RNG
6. Performance: the execution table, host-sync hygiene, the kernel policy
7. Tests: fixtures, the contract registry, oracles, mirrored layout, and the rule
   that API commits never change expected values
8. Benchmarks: methodology and baselines
9. Migration status (temporary; removed in task 4.2)

---

## 8. Done criteria for 2.0

- The CPU suite passes on CI (3.12-3.14, locked and lowest-direct).
- The GPU suite and full benchmark run pass on the reference machine.
- Coverage is at or above the floor.
- Mypy is strict for `core`, `mapping` and `equalization`, and the default gate
  passes for everything else.
- No `jax`, `pydantic` or `yaml` imports remain. There is no `helpers.py`, no
  `debug_plot`, and no `backend=` / `device=` / `update_mode=` parameters.
- Every public function is in the contract registry.
- The quickstart and all examples run in CI.
- Benchmarks are within tolerance of 0002, apart from improvements and the
  documented removals.

---

## 9. Dropped from the previous documents

| Item | Source | Why dropped |
| --- | --- | --- |
| Two-tier NumPy/CuPy + JAX model, JAX examples, TF32 `Precision.HIGHEST` rule | Modularity plan 4, 8 | JAX removed (D1) |
| Three-backend equalizer runners, kernel file layout, capability table | Modularity plan 3 | Two backends remain; the combinations collapse |
| Separate phases for metadata conflicts, SPS tolerance, typing, helpers, OLS | Modularity plan 2, 5, 8 | Folded into the per-module checklist |
| Cache-invalidation policy, ownership follow-ups | Modularity plan 1 | Caches removed (D8); ownership docs done |
| Deprecation policy, forwarding exports, migration tables | Modularity plan | No users (D5) |
| `commkit/dsp/` and `commkit/physics/` packages | Modularity plan 5, 7 | One private module each is enough |
| Phase 4B as its own phase | Modularity plan 4 | Part of 2.2 |
| `update=Sequential() \| Block() \| FrequencyDomain()` | Roadmap 4.3 | Block mode has no fast backend; FDAF stays separate (D2) |
| `recover_carrier_phase(symbols, cpr)` dispatcher | Roadmap 4.2 | Named functions are clearer; objects are used only when nested |
| "Explicit argument always wins" | Roadmap 4.5 | Replaced by facts/choices (D3) |
| "Per-channel results always `(C,)`" | Roadmap 4.6 | Replaced by the rank rule plus host metrics (D12) |
| `Generator`-only randomness | Roadmap 4.7 | SPEC 7 `rng` also accepts an int seed |
| PyYAML and Pydantic as core dependencies | Roadmap 5 | Removed (1.1, D4) |

## 10. Deferred (not part of 2.0)

commax itself; FEC (`coding`); nonlinear channel models; OFDM; batched GPU
kernels for sequential equalizers across realizations (only if a benchmark
justifies them); property-based tests.
