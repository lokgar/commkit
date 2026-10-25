# AGENTS.md

Guide for anyone, human or coding agent, changing the **CommKit** repository.

The rules below describe the code as it is. New code follows them; a change that breaks one changes this file in the same commit.

---

## 1. What CommKit is

A Python library for digital-communications research: simulation, receiver DSP, metrics, plotting and experimental captures, on NumPy (CPU) and CuPy (GPU) data. It is **not differentiable**. A separate JAX library, *commax*, will hold the blocks that must sit inside a gradient. commax depends on commkit, never the reverse, and takes its conventions (`Constellation`, bit labels, pulse taps, frame layouts) from commkit.

Users are researchers and engineers. Design for them:

- **Algorithms are plain functions.** Data is a `Signal` or an array. Things that *describe* a modulation, a pulse or an embedded sub-algorithm are small frozen value objects written inline in the call:

  ```python
  res = lms(rx, num_taps=21, step_size=1e-3, cpr=BPS(test_phases=64))
  ```

- **Everything is expressed in Python.** No configuration files, registries, global switches or multi-step setup.
- **We deliberately do not build** pipeline or receiver objects, DSP methods on `Signal`, string shorthands for constellations, or public "config objects per algorithm".
- **Scope is the reference model and its inverse**, not a catalogue of research variants: the fiber channel and standard digital backpropagation (`correct_fiber_propagation`, with `steps_per_span` and `nonlinear_scale`), like chromatic dispersion and its compensation. Reduced-complexity nonlinearity compensation (filtered or enhanced split step, perturbation-based) is left to user code built on these, and learned DBP, which needs gradients, belongs in commax.

---

## 2. Commands

Always use `uv`. Python 3.12+.

```bash
uv sync --all-extras                     # environment incl. GPU packages
uv run nbstripout --install              # once per clone: strip notebook outputs on commit
uv run pytest                            # CPU + GPU tests (default --device=all)
uv run pytest --device=cpu               # what CI runs
uv run pytest tests/equalization/ -k lms # subset
uv run pytest --cov=commkit              # coverage
NUMBA_DISABLE_JIT=1 uv run pytest tests/equalization tests/recovery \
    tests/test_frequency.py tests/test_fiber.py --device=cpu --cov=commkit   # real kernel coverage
uv run pytest --device=cpu -p tests.plugins.precision_trace \
    --ignore=tests/test_examples.py      # precision audit (see Numerics); the suite passes under it
```

**Commit gate.** CI fails on any of these, so run all of them before every commit, with `ruff format` last:

```bash
uv run ruff check . --fix
uv run mypy commkit/
uv run pytest
uv run ruff format .
```

**Benchmarks** are explicit-only; `uv run pytest` never collects them. A commit that changes a public signature also smoke-runs them once, without timing, so their call sites cannot rot unnoticed:

```bash
uv run pytest benchmarks/ --benchmark-disable --device=all
```

Timed runs:

```bash
uv run pytest benchmarks/bench_bps.py --benchmark-only --device=all
uv run pytest benchmarks/ --benchmark-only --device=all \
    --benchmark-compare=0003 --benchmark-storage=file://benchmarks/baselines
uv run python benchmarks/record_baseline.py NAME   # record a new baseline
```

**Release:**

1. On `main`: the CPU and GPU suites pass with no failures or skips (`uv run pytest -rs`), `uv build` succeeds and `uv run --with twine twine check dist/*` passes.
2. `uv run bump-my-version bump {patch|minor|major}`. This commits the new version and tags it `vX.Y.Z`.
3. `git push origin main --tags`.
4. Publish a GitHub Release for the tag (`gh release create vX.Y.Z --generate-notes`). Publishing the release runs `.github/workflows/publish.yml`, which builds and uploads to PyPI through Trusted Publishing; pushing a tag alone publishes nothing.

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
- **Splitting modules:** promote a flat module to a package only when it exceeds about 1000 lines *and* has two or more separable concerns. The package `__init__` re-exports the public names. Never create a `utils/` package or one-function files.
- **Placeholder modules are kept on purpose.** `commkit.coding` is a reminder of planned work. Never delete it or treat it as dead code. It is excluded from coverage and from the contract registry, and the README labels it "planned, not implemented".
- **Shared physics lives in one private module**, used by the channel and by its compensation: `_dispersion` (β₂, β₃, the dispersion response) and `_fiber` (`Fiber` and the split-step solver behind `apply_fiber_propagation` and `correct_fiber_propagation`).

---

## 4. API rules

### Signal

- `Signal` holds: samples (NumPy or CuPy, `(N,)` or `(C, N)`, time on the last axis), the **facts** `sampling_rate`, `symbol_rate` and `center_frequency`, the **description** `constellation` and `pulse`, and the ground truth `reference` (transmitted symbols and bits) plus the `frame` layout.
- It never holds pipeline results.
- Construction does no hidden work: no mapping, no normalization, no device move, no shape guessing. Factories (`generate`, frames) do that work.
- `sig.to("gpu")` and `sig.replace(...)` return new Signals. Arrays are shared, not copied; use `clone()` for an independent copy.
- **Alignment invariant:** reference symbol *k* corresponds to symbol period *k* of the samples. Any function that drops or shifts symbol periods slices the reference to match. Metrics raise on mismatch; they never guess.

### Function signatures

- `f(data, other_data, *, params)`: everything after the data arguments is keyword-only. A function without data arguments may keep its single primary parameter positional (`barker_sequence(13)`, `gray_code(5)`); with several parameters, all are keyword-only (`rrc_taps(sps=2, rolloff=0.1, span=10)`). The data-argument count is recorded per function in the contract registry.
- **Signal awareness.** A function whose main argument is waveform data (or a field a Signal really carries) accepts an array or a Signal. It unwraps once with `adapt_signal()` (in `core/_signal_adapter.py`), works on arrays, and returns through `wrap_samples()`. Never re-enter the public function recursively.
  - Functions on *derived* quantities (phase trajectories, correlations, taps, PSD bins, results) and synthesis primitives stay array-only.
- **Metadata rule: facts versus choices.**
  - Facts (`sampling_rate`, `symbol_rate`, and so `sps`) come from the Signal. A conflicting argument raises `ValueError`; an equal one is fine.
  - Choices (decision `constellation`, `noise_var`, `pulse`) default to the Signal's value, and an explicit argument wins.
  - A missing fact for array input raises. Neither case warns.
- Integer-SPS algorithms validate with `require_integer_sps()` before any cast. Never truncate 1.5 SPS to 1.
- **Value objects are frozen dataclasses** with read-only NumPy arrays:
  - `Constellation`: `.qam(M)`, `.psk(M)`, `.pam(M, unipolar=)`, or arbitrary `points`; `.shaped(nu=|entropy=)` keeps unit power. Functions accept the object only, never strings.
  - Pulses: `RRC`, `RC`, `Gaussian`, `Rect`, `SmoothRect`. A raw taps array is accepted wherever a pulse is.
  - Algorithm objects, one per estimation method (`BPS`, `PLL`, `ViterbiViterbi`, `MthPower`, `CycleSlip`, ...). The same object is used standalone (`correct_carrier_phase(y, BPS())`) and nested (`lms(..., cpr=BPS())`). It lives next to its kernel and validates its parameters on construction. Class names are unique across the package.
  - Equalizer continuation uses `state=result.state`.
- **Randomness:** `rng: int | np.random.Generator | None` (SciPy SPEC 7).
  - What is transmitted (bits, symbols, pilots): draw on the host with the Generator, then transfer. A seed gives the same payload on CPU and GPU.
  - What the channel adds (AWGN, phase noise, ASE): draw on the data's device through `_random.standard_normal` / `complex_normal`. Realizations differ between devices; statistics agree, and tests assert statistics.
  - Random parameters of a realization (the PMD waveplates of `Fiber`) are few and drawn on the host, so a seed gives the same fiber on every device.
  - Never use a global RNG.
- **Types:** a transform is written `def f(samples: S, ...) -> S` with the TypeVar `S` from `core/_signal_adapter.py` (bound to `np.ndarray | Signal`), so a Signal gives a Signal and an array an array. Do not reassign the `samples` parameter; name the unwrapped array `x`.
- **Device follows the data.** There are no `backend=` arguments and no global switches. Unsupported array types (JAX, PyTorch) raise `TypeError`.
  - Only factories with no input data to follow (`generate`, `generate_phase_noise`, `Preamble`/`SingleCarrierFrame.to_signal`, `load_npz`) take `device: str = "cpu"`, validated with `require_device()`. They draw randomness as the rule above says and do the vectorized work (mapping, shaping) on `device`. The default is never "GPU if available".
- **No `debug_plot`.** Numerical code never imports plotting. Plot functions consume results or recompute through public compute functions.
- **Design and apply are separate functions**, for example `rrc_taps` and `fir_filter`, or `butterworth_sos` and `iir_filter`.

### Returns

- **Transforms:** array in gives array out on the same device; Signal in gives a new Signal out (update `sampling_rate` when the rate changes).
- **Estimators** reduce over the time axis by input rank: `(N,)` gives 0-d and `(C, N)` gives `(C,)`. Results stay on the input device.
- **Analysis summaries** (`estimate_linewidth`, `allan_deviation`, the `frequency_drift` statistics) are reporting-layer fits on plot-sized reductions. Like the metrics, they return host floats or `(C,)` arrays. Sample-rate outputs (trajectories, PSDs, `dsh_phase`) stay on the input device.
- **Metrics** (`evm`, `snr`, `ber`, `ser`, `gmi`, `mi`) return host values: a `float` for 1-D input and `np.ndarray (C,)` for 2-D input. They **raise** on empty selections; they never return `None` or `0.0` for "nothing measured". Metrics never silently align, rotate or permute against the reference.
- **More than one value** is returned as a frozen dataclass with named fields. No dicts, and no tuples longer than two.
- **Equalizers** return `EqualizerResult`. `y_hat` is always an array, `signal` holds the 1-SPS Signal for Signal input, `state` is the continuation state, and diagnostics are returned as data.

### Naming

| Prefix | Meaning |
| --- | --- |
| `apply_*` | add an impairment or apply a model |
| `estimate_<quantity>(x, method)` | measure, never change the data; returns a frozen `<Quantity>Estimate` (value plus diagnostics) |
| `correct_<quantity>(x, how)` | remove the impairment; returns only the corrected data. `how` is an estimate or an algorithm object (which estimates first) |
| `resolve_*` | pick among discrete candidates (π/2 rotation, channel permutation); nothing else |
| `generate` | synthesis from parameters, takes `rng` |
| `plot_*` | every public plotting function (only `apply_default_theme` is exempt) |

- The algorithm is chosen by an object, never by the function name or a string: `estimate_carrier_phase(y, BPS())`, not `recover_carrier_phase_bps(y)`. The verb function dispatches on the object's type through a private table in its module. There is no `recover_*` or `compensate_*`.
- Computations use plain nouns (`allan_deviation`).
- **Glossary**, one name each:
  - `num_taps`, `step_size`, `sps`, `constellation`, `pulse`, `rng`, `num_symbols`, `num_channels`, `sampling_rate`, `symbol_rate`.
  - Units are in the docstring. Use `_db` and `_hz` suffixes where the unit is ambiguous.

### Docstrings

Docstrings describe the code as it is; a change in behavior updates them in the same commit. NumPy format (numpydoc), sections in its order: Parameters, Returns, Raises, Warns, See Also, Notes, References, Examples.

- **Summary line** on the line of the opening quotes (`"""Summary.`), one line, ending with a period.
  - Functions: imperative mood (`Apply ...`, `Estimate ...`, `Plot ...`), or a noun phrase for a function named after the quantity it returns (`evm`, `allan_deviation`, `welch_psd`).
  - Classes and modules: a noun phrase.
- **Module docstrings** say what the module holds, by public name (`:func:`, `:class:`), then the conventions particular to it (a sign convention, a scaling, a file layout). No marketing adjectives ("high-performance", "publication-quality"), no history of earlier layouts, and no restatement of the library-wide rules in this file. A private module's summary ends with "(private)" and the body names its users.
- **Types:** inputs are `array_like` (NumPy or CuPy), `array_like or Signal` for Signal-aware arguments; returned arrays and result fields are `ndarray` (NumPy or CuPy, on the input's device), `ndarray or Signal` for transforms, and `numpy.ndarray` where the array is always on the host. Shapes go in the description, in double backticks: ``(N,)``, ``(C, N)``.
- **Plots** return `(fig, ax) or None` (`None` with `show=True`).
- **Defaults:** `name : type, default value`; `name : type, optional` when the default is `None`, with the description saying what `None` does.
- Units go in the description; inline code in double backticks; references to library names through `:func:` / `:class:` / `:mod:` / `:meth:`.

### Errors and logging

- Wrong input raises, with a message naming the argument.
- `logger` is for diagnostics only. Per-channel logging that needs a device-to-host transfer is wrapped in `if logger.isEnabledFor(logging.INFO):`.
- The library never configures handlers, Matplotlib or warning filters at import time.

---

## 5. Numerics

- **Storage dtypes:** `complex64` for IQ samples and `float32` for real signals; `generate`, frames and `load_npz` produce them, so the GPU's single-precision path is the default.
- **Precision contract.** Every function computes in a predictable precision, and none promotes single-precision data on its own:
  - **Transforms keep the input's precision**: complex64 in, complex64 out and computed throughout (filters, impairments, the fiber solver and DBP, resampling, delays, carrier-phase correction); complex128 in, complex128 out.
  - **Fixed-precision blocks** compute in single precision whatever the input: the equalizers in complex64 (with complex128 accumulation inside, below), `compute_llr` / `Constellation.llr` in float32. Double input to them emits `commkit.PrecisionWarning` (`_array.warn_if_double`, attributed to the caller's line; Python shows it once per call site). Users who want strictness turn it into an error with `warnings.simplefilter("error", commkit.PrecisionWarning)`; the test suite does (`pytest_configure` in `tests/conftest.py`; never in pyproject's `filterwarnings`, which imports commkit before coverage starts and drops every module-level line from the report), so a library path that feeds double data to these blocks fails CI. Training and reference symbols are ideal values and are not checked.
  - **Deliberate float64**, documented where it happens, always on phases, small arrays or accumulators, never on sample-rate data that is then cast back: phases before `unwrap`, phase trajectories and phase ramps (wrapped in float64, rotated in the data's precision), frequency grids and estimates, the recursions' state (PLL, Tikhonov, RLS `P`), the IIR filter's sections and state (poles near z = 1; on the CPU the samples keep their precision), and sums: the metrics compute per element in the input's precision and accumulate in float64 (`gmi` / `mi` in single precision moved by under 4e-8 bit, far below the Monte Carlo error).
  - **One known exception:** `iir_filter` on the GPU filters single-precision samples in double precision and casts the result back, because `cupyx.scipy.signal`'s block-parallel filter computes in one dtype for samples and sections, and single-precision sections are unstable at low cutoffs. See the roadmap (Performance).
  - **Audit with `tests/plugins/precision_trace.py`** after touching numerics: it lists every float64/complex128 array of sample-rate size that a NumPy path of the library creates from single-precision data or from scratch, with its source line. Every reported site must be a deliberate use from the list above (or a test passing complex128 on purpose). Output dtypes alone do not show this: a function that computes in complex128 and casts back returns complex64.
  - complex128 pays only where single-precision rounding accumulates over many FFT passes, as in the fiber solver (6e-4 after 10^4 steps in complex64, 1e-12 in complex128); there the GPU runs its plain CuPy stages, not the fused complex64 kernels.
  - **The NEP 50 trap**: under NumPy 2 a NumPy scalar is not weak, so `np.float64(a) * x64` (any `np.sqrt`, `np.pi * ...` result, or a float64 array such as `xp.fft.fftfreq`) promotes complex64 data to complex128. Combine single-precision data only with Python scalars or with arrays cast to its precision; rotate with `math._derotate` (float64 phase wrap, single-precision phasor, one fused kernel on the GPU).
- **Accumulation precision:**
  - LMS/CMA/RDE: inputs and weights are `complex64`, but dot products and gradient updates in the hot loop are accumulated in `complex128`.
  - RLS keeps `P`, the gain `k` and the regressor buffers in `complex128` throughout. Single precision loses the Hermitian positive-definite property and the filter diverges.
  - Carrier phase: promote angles to `float64` before `xp.unwrap()`, because float32 rounding causes spurious quadrant slips. Tikhonov/Kalman block transitions also run in `float64`.
- **Normalization invariant:** at `sps` samples per symbol, `E[|x|²] = 1/sps`; at symbol rate, `E[|x|²] = 1`. Any rate-changing block applies the exact gain correction.
- **Shapes:** SISO is `(N,)` and MIMO is `(C, N)`. Use `as_2d`, `restore_1d`, `broadcast_channels` and `require_channels`; never hand-roll the promote/squeeze idiom.
- **Reproducibility:** a seed is reproducible within a library version only. Tests and baselines assert statistics, never exact noise realizations.

---

## 6. Performance

| Algorithm structure | Implementation | Device |
| --- | --- | --- |
| Vectorized: filters, resampling, spectra, impairments, metrics, LLR, frequency-domain equalizers | NumPy/CuPy via `x, xp, sp = dispatch(samples)` | follows the input |
| Sequential recursions: LMS/RLS/CMA/RDE, PLL, Tikhonov, cycle slips | Numba `@njit(cache=True, fastmath=True, nogil=True)` | CPU; GPU input does one round trip to the host and back |
| Linear recursions: IIR filtering | CPU: Numba kernel without `fastmath`, so double input gives SciPy's result bit for bit. GPU: `cupyx.scipy.signal`, a block-parallel filter | follows the input |
| Hot spots limited by intermediates, launches or round trips | CUDA C++ via CuPy `RawModule` (`commkit/_cuda`) | GPU |
| Elementwise chains with many temporaries, where a paired A/B shows the gain end to end (the PMD stage of `_fiber`: 0.61x on 4 cores, but not on the reference machine; see the roadmap) | Numba `@njit(parallel=True)` in one pass, real arithmetic (a `1j` literal promotes complex64 to complex128) | CPU |

**Host-sync hygiene:**

- Never extract scalars from a possibly-GPU array inside a loop (`float(x[c])`, `.item()`). Compute the vector on the device, transfer it once with `to_device(v, "cpu")`, then loop on the host.
- Prefer on-device gathers (`xp.take_along_axis`).
- Every FFT goes through `commkit._fft` (`fft`, `ifft`, `rfft`, `irfft`), never `np.fft` or `xp.fft` directly: on the CPU it runs `scipy.fft` on every core and keeps complex64 in single precision (NumPy 2 computes it in double, about 3x slower per core); on the GPU it is `cupy.fft`. Frequency grids (`xp.fft.fftfreq`) are exempt.
- Frequency responses (dispersion, delays, Jones matrices) are computed at float64 phase accuracy and cast to the spectrum's precision before the multiply, so complex64 data is never promoted.
- Chunk `(N, M)` intermediates over N with an on-device accumulator.
- Reduce on the device before transferring, including for plotting.
- Pass whole records to compiled code. There are no Python loops over symbols or small blocks, except inside a CUDA graph.
- Repeated GPU work with fixed buffers (the block equalizers' blocks, the fiber solver's spans) is replayed from CUDA graphs through `_cuda.graphs.GraphRunner`, the one driver: warm once, capture on the second run, replay, fall back to eager on a failed capture. Its stream is reused per device and thread, because CuPy's pool caches freed blocks per stream and a fresh stream per call grows the pool every call.

**Kernel policy:**

- A `.cu` kernel needs a benchmark showing that CuPy is limited by intermediates, launch count or round trips.
- It optimizes a CPU reference (Numba or NumPy) and is tested against it with `--device=all`.
- Wrappers check dtype, contiguity, shape and size before launch.
- `float32` with explicit literals.
- `--use_fast_math` only per kernel (through `options`), after an accuracy check; the compiler default does not pass it.
- Short elementwise chains use `ElementwiseKernel` or `cupy.fuse`.
- Sequential recursions over a whole record are never ported to the GPU; one record has no parallel work per step. Linear recursions (IIR filters) are the exception: blocks of the record are filtered in parallel from a zero state and then corrected by the state carried in from the previous block, as `cupyx.scipy.signal` does. A short per-block scan may be, as one launch, when it replaces a host round trip per block (`cs_block`, `bps_anchor`); keep its parallel part parallel (float64 is 1/64 of float32 throughput on consumer GPUs).

---

## 7. Tests

- **Fixtures:** use `backend_device`, `xp` and `xpt` from `tests/conftest.py`. Assert with `xpt.assert_allclose(...)`, because `numpy.testing` raises on CuPy arrays. Cast expectations with `xp.asarray`, and reductions with `float(...)`.
- **Layout mirrors the source:** `tests/<subpackage>/test_<module>.py`, and `tests/test_<module>.py` for flat modules. Test basenames must be globally unique (pytest prepend import mode), hence names like `channel/test_channel_linear.py`. Split a large test file by concern.
- **Shared builders** live in `tests/common/` and use the public API.
- **Contract registry** (`tests/test_api_contracts.py`): every public function has a row. Generic checks run per row: array in/out, Signal in/out, device preserved, input not mutated, keyword-only parameters, `TypeError` on unsupported arrays, fact-conflict errors, return rules.
  - Do not duplicate these checks in per-module tests.
- **Oracles:** sequential kernels are tested against plain-Python reference implementations in `tests/common/reference_impl.py`. CUDA kernels are tested against their CPU reference.
- **Mathematical meaning:** test documented definitions, such as a pulse width, a dispersion sign, or noise-variance scaling, with independently derived expectations, not only shapes and energies.
- **API changes never change expected values.** A commit that changes a signature only changes how tests call the code. Numerical changes land in their own commit with independent validation.
- **Examples** are Jupyter notebooks in `examples/`, committed without outputs: `.gitattributes` routes them through the `nbstripout` filter, which each clone enables once with `uv run nbstripout --install` (git does not ship filter configuration). `tests/test_examples.py` executes each one, so an API change updates the notebooks in the same commit.
  - They form a course: one topic per notebook, named `NN-lowercase-words.ipynb`, numbered in teaching order (signals and DSP, channel and metrics, receiver algorithms, fiber systems, characterization). Each opens with what it covers, links the notebooks it builds on and ends with a link to the next. `examples/README.md` indexes them; a new notebook takes its place in the order and updates the index and the neighbors' links.
  - Every claim in the text matches what the cells print: run the notebook and read its numbers before committing it. Keep each under about 20 s on the CPU.
  - Plots of hand-computed arrays go through `to_device(x, "cpu")`, so the notebooks run unchanged on the GPU.
- **Coverage** must not drop below the CI floor. Numba kernel coverage is measured by a separate `NUMBA_DISABLE_JIT=1` job.

---

## 8. Benchmarks

`benchmarks/` tracks the GPU-relevant hot paths. Baselines are committed under `benchmarks/baselines/`. The current reference is `0004` (`fiber_dbp`) on the reference machine: RTX 4070 Ti, Ryzen 7 7800X3D, WSL2. It adds the fiber and DBP benchmarks to `0003` (`v2_0`, the 2.0 state); `0002` (`pre_v2`) is the 1.x state. `0002` ran the CPR equalizer and Viterbi-Viterbi benchmarks on workloads that did not converge, so compare those only from `0003` on.

- **Workloads must converge.** A workload the algorithm cannot handle times a failure mode (slip storms, divergence), not the operating point. Every equalizer benchmark asserts its symbol error rate with `benchutils.assert_converged`.
- **Baselines** are recorded with `benchmarks/record_baseline.py`: each file in its own process, best of three passes. In one full-suite process, small GPU benchmarks after the large equalizer workloads ran 2-12x slower than alone, so never record a baseline from a single `pytest benchmarks/` run.

- **IDs** are `[cpu]` or `[gpu]`, meaning the input device.
- **Timing:**
  - Timed bodies end with the `sync` fixture; otherwise GPU timings measure launches, not execution.
  - Each benchmark has one warmup round, so compilation and pool growth are excluded.
- **Workloads** come from `benchmarks/workloads.py` with fixed seeds; never generate data inline.
- **Tools:** `benchmarks/benchutils.py` provides `CudaEventTimer` and `nvtx_range` for `nsys` profiling.
- **Logging** is set to WARNING in `benchmarks/conftest.py`.
- **Trust deltas, not single runs.** A ±20-40% swing has been seen under load. Re-run a suspected regression in isolation, or do a controlled A/B with at least 7 repetitions, alternating old and new per repetition and comparing the paired ratios. When even unchanged benchmarks move, time the changed part itself with `CudaEventTimer` and quote it against the step it sits in.
- A commit touching a hot path quotes its delta against `0004`. Tolerance is 5% on CPU and 10% on GPU.

---

## 9. Roadmap and open items

What is not done yet, in suggested order. Unlike the sections above, this is a to-do list, not a description of the code: remove an item in the commit that resolves it, and add one when a known gap is left open.

### Switch to Python 3.15 (blocked on CuPy)

Goal: require Python 3.15 and nothing older, mainly for explicit lazy imports (PEP 810, the `lazy` keyword). State checked on 2026-10-23 with uv 0.13.5 and CPython 3.15.1:

- **Blocker: CuPy has no 3.15 wheels.** `cupy-cuda13x` 14.2.0 ships cp310-cp314 only, on PyPI and on `pip.cupy.dev/pre`. Support is tracked in cupy/cupy#10259, milestone v14.3.0 (unreleased; the 14.3.0.dev0 docs already list 3.15). Switching before then would leave `commkit[gpu]` uninstallable and the GPU suite unrunnable.
- **Second blocker: `jupyter_client` fails to import on 3.15.** `jupyter_client/jsonutil.py` (line 32) calls `strptime` with a day of month and no year, which 3.15 rejects (cpython#70647). `tests/test_examples.py` then fails at collection, and the notebooks cannot run. Wait for a fixed release; do not work around it in commkit.
- **Ready:** wheel-only installs on 3.15 work for NumPy 2.5.4, SciPy 1.18.1, Numba 0.68.0 (llvmlite 0.50.0), Matplotlib 3.11.2, mypy 2.4.0, ruff, pytest, pytest-cov, pytest-benchmark, nbclient, ipykernel, nbstripout and bump-my-version. The CPU suite has not been run on 3.15 beyond the collection error above.

Steps, once CuPy 14.3 (with cp315 wheels for `cupy-cuda13x`) and a fixed `jupyter_client` are out. Keep the commits separate: the version switch alone first, then each refactor, so a failure points at one change.

1. **Validate before changing anything**: in a 3.15 venv, install the current code and run the full CPU+GPU suite (`-rs`, notebooks included), the benchmark smoke run, the `NUMBA_DISABLE_JIT` job and the precision audit.
2. **Switch the version floor**:
   - `pyproject.toml`: `requires-python = ">=3.15"`, classifiers 3.15 only, ruff `target-version = "py315"` (if ruff supports it), raise the dependency minimums to the first releases with 3.15 wheels (`cupy-cuda13x[ctk]>=14.3`, `numba>=0.68`, `numpy>=2.5`, `scipy>=1.18`; recheck each).
   - `ci.yml`: test matrix `python-version: ["3.15"]` (keep both `locked` and `lowest-direct`); the static-analysis, kernel-coverage and notebook jobs on 3.15. `publish.yml`: build on 3.15.
   - `uv lock`; README install requirements; §2 here ("Python 3.12+").
   - Benchmarks: record a new baseline on 3.15 (it lands in `benchmarks/baselines/Linux-CPython-3.15-64bit/`), compare it with `0004` by a paired A/B of 3.14 against 3.15 on unchanged code, and make it the reference.
3. **Lazy imports (PEP 810)**. Confirm the rules on the interpreter first. As specified, `lazy` is allowed only for module-level imports: not inside functions, class bodies or `try` blocks, and not with `*`.
   - About 60 function-local imports exist to keep `import commkit` light or optional: `import cupy as cp` (11 sites), `import numba` (9), `from . import compiler` in `_cuda` (5), and local imports that break cycles (`Constellation`, `Signal`, `normalize`). Move each to a module-level `lazy import` / `lazy from`. Concentrated in `_cuda/__init__.py`, `backend.py`, `io.py`, `recovery/`, `core/`, `plotting/`.
   - CuPy stays optional: a lazy import of a missing module fails only when used, so every use must stay behind the existing availability checks (`is_cupy_available`, `dispatch`). `backend.py`'s probing needs care: decide per site whether the probe imports eagerly inside `try`.
   - Numba: `@njit` at module level compiles lazily already; check that a lazy `numba` import does not make every first call pay the import at an odd moment (time with `benchmarks/bench_import.py`).
   - The six `TYPE_CHECKING` blocks can become `lazy from` imports where the name is also useful at run time; keep them where it is not.
   - Measure `import commkit` before and after with `bench_import.py`, and keep `tests/test_import_hygiene.py` passing; extend it to assert that `import commkit` loads neither CuPy nor Matplotlib, if lazy imports make that true.
4. **Other cleanups that the 3.15 floor allows** (each its own commit; verify each against What's New in 3.14 and 3.15):
   - Drop `from __future__ import annotations` (25 modules): annotations are evaluated lazily by default since 3.14 (PEP 649/749). Check that nothing reads `__annotations__` as strings.
   - UTF-8 mode is the default from 3.15 (PEP 686): check `io` and any `open()` without `encoding=`.
   - `strptime` with a day and no year now raises (the `jupyter_client` failure above); grep commkit for date parsing.
   - The 3.15 sampling profiler (PEP 799) is a candidate tool for the CPU hot paths next to `nsys`; mention it in §8 if it proves useful.

### Planned modules

1. **Channel coding** (`coding`, a placeholder package). Resolve the "Before channel coding" items below first, so the decoder is built on soft demapping that is already at its final speed on both devices. Start with one soft decoder (LDPC) that consumes `compute_llr` output (positive LLR = bit 0) and reports post-FEC BER next to `gmi`, then add encoders. The package stays out of `commkit/__init__.py` until a real encode/decode entry point exists. Decoders are iterative message passing over many codewords: batch codewords on the GPU from the start.

### Nonlinear fiber channel: future features

Each is an addition that leaves the current API unchanged.

- **Batched realizations**: an extra leading axis `(B, 2, N)` (launch-power sweeps, Monte Carlo over seeds) so cuFFT batches them and short records fill the GPU.
- **Adaptive step rules**: a `step_rule=` object (maximum nonlinear phase per step, local-error method) next to the equal `steps_per_span`, which measured 3x more accurate than logarithmic steps on a 32 GBd, 80 km span.
- **WDM helpers**: combining frequency-shifted channels into one wideband field and selecting one after propagation; the solver already handles the wide field (with `dispersion_slope_ps_nm2_km` set).
- **Distributed (Raman) amplification and per-span fiber types** in one call (chaining calls already works: every call speaks launch-relative units).
- **cuFFT callbacks** fusing the linear stage into the transforms (CuPy's support is experimental and compiles with nvcc at run time).
- **Self-steepening and Raman scattering** for ultra-wideband fields.

### Before channel coding: mapping and soft demapping

The decoder's input path. The GPU items need a CUDA device to write, test (`--device=all`, against the Numba reference) and time.

- **GPU LLRs of separable QAM.** `Constellation.llr` takes the per-axis path (2·sqrt(M) metrics instead of M) only on the CPU; on the GPU, 256-QAM evaluates 256 metrics per symbol instead of 32. Port `llr_separable` to a CUDA kernel.
- **Non-square hard decisions on the GPU.** `gray._nearest_index` (used by `Constellation.demap`, `evm(blind=True)` and `ser` for PSK, cross QAM and arbitrary points) loops in Python over 4096-symbol chunks: about 250 iterations of several launches per 10^6 symbols. On the CPU that chunk and `abs` are the fastest NumPy form (measured); on the GPU, time it against a larger element-budget chunk or the nearest-point mode of `bps_min_d2`.
- **Closed-form max-log LLRs of Gray QAM** (optional, CPU and GPU): per bit, the nearest point of each half is found by snapping, O(k) instead of O(sqrt(M)). Expected gain under 2x on the CPU; measure before adopting.

### Performance

- **Single-precision IIR filtering on the GPU.** `iir_filter` on the GPU converts complex64 samples to complex128 for `cupyx.scipy.signal` and back: two double-size copies of the record, against the precision contract. A block-parallel CUDA kernel that reads and writes the samples in their dtype and keeps the sections, block carries and corrections in float64 removes both. Test it against the Numba reference with `--device=all` and time it against the CuPy path.
- **Block equalizers at small `block_size` on the GPU**: a fixed cost per block; at 256 the GPU only ties the CPU without carrier recovery.
- **The fiber solver's Numba passes lose end to end on the reference machine.** On 4 cores (where they were tuned) the hybrid Kerr stage won from `_fiber._HYBRID_KERR_MIN = 2^18` and the trig-free Jones (PMD) pass won end to end (0.61x). On the reference machine (8 cores, 16 threads), whole propagations with NumPy stages and multi-core `scipy.fft` are faster: hybrid Kerr 1.74x slower at 2^18, 1.32x at 2^19, 1.12x at 2^20; Numba Jones 1.35x at 2^14, 2.19x at 2^16, 1.54x at 2^18, even at 2^20. The cause is contention: Numba's OpenMP workers spin after each pass and take the cores from the FFT workers (`OMP_WAIT_POLICY=passive` cuts the Kerr loss at 2^18 to 1.09x; one Numba thread makes Jones even with NumPy). The stage timed alone hides this. Options: choose the passes by core count, run them single-threaded, or keep NumPy stages only; decide by end-to-end A/Bs on both machines.
- **A fully fused Numba Kerr stage** may beat the hybrid one where Numba has SVML (vector sin/cos); neither machine measured so far has it (`numba.core.config.USING_SVML` is False), so it is untimed.
- **`cs_block` on phases with frequent slip decisions**: one speculation pass per changed decision (about 1 pass per block on a realistic link, more on pathological input). Outside a CUDA graph the Python wrapper (validation, scratch allocation) costs about 18 us per launch.
