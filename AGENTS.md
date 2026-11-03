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
- **Scope is the reference model and its inverse**, not a catalogue of research variants: the fiber channel and standard digital backpropagation (`correct_fiber_propagation`, with `steps_per_span` and `nonlinear_scale`), like chromatic dispersion and its compensation. Reduced-complexity nonlinearity compensation (filtered or enhanced split step, perturbation-based) is left to user code built on these; learned DBP, which needs gradients, belongs in commax.

---

## 2. Commands

Always use `uv`. Python 3.12+.

```bash
uv sync --all-extras                     # environment incl. GPU packages
uv run nbstripout --install              # once per clone: strip notebook outputs on commit
uv run pytest                            # CPU + GPU tests (default --device=all)
uv run pytest --device=cpu               # what CI runs
uv run pytest --run-slow                 # also the long statistical validation (marker slow)
uv run pytest tests/equalization/ -k lms # subset
uv run pytest --cov=commkit              # coverage
uv run pytest --device=cpu -p tests.plugins.precision_trace \
    --ignore=tests/test_examples.py      # precision audit (§5); the suite passes under it
uv run pytest --device=gpu -p tests.plugins.transfer_trace \
    --ignore=tests/test_examples.py      # transfer audit (§6); fails on an undocumented transfer
scripts/sanitize.sh                      # kernels under compute-sanitizer (§6); setup in the script
```

Numba kernel coverage runs the kernel tests with `NUMBA_DISABLE_JIT=1` and reports only the kernel modules; the test selection and the `--include` list are in the `kernel-coverage` job of `.github/workflows/ci.yml`.

**Commit gate.** CI fails on any of these, so run all of them before every commit, with `ruff format` last:

```bash
uv run ruff check . --fix
uv run mypy commkit/
uv run pytest
uv run ruff format .
```

**Benchmarks** are explicit-only; `uv run pytest` never collects them. A commit that changes a public signature smoke-runs them once, untimed, so their call sites cannot rot:

```bash
uv run pytest benchmarks/ --benchmark-disable --device=all
uv run pytest benchmarks/bench_bps.py --benchmark-only --device=all   # timed
uv run pytest benchmarks/ --benchmark-only --device=all \
    --benchmark-compare=0008 --benchmark-storage=file://benchmarks/baselines
uv run python benchmarks/record_baseline.py NAME   # record a new baseline (§8)
```

---

## 3. Architecture

Dependencies point downward only:

| Layer | Modules | May import |
| --- | --- | --- |
| Infrastructure | `backend`, `logger`, `_cuda` | NumPy, CuPy |
| Array and math helpers | `_array` (shape and validation), `math` (power, normalization, dB, linear trend) | infrastructure |
| Value objects | `mapping.Constellation`, pulse classes, `recovery` CPR configs, `coding` (code objects with their `encode` / `decode`; array-only, never imports `core`) | layers above |
| Core | `core` (`Signal`, frames, generation), `io` | layers above |
| DSP | `mapping`, `filtering`, `multirate`, `spectral`, `smoothing`, `impairments`, `timing`, `frequency`, `recovery`, `equalization`, `metrics`, `analysis` | layers above, and lower DSP modules |
| Presentation | `plotting` | everything; **nothing imports plotting** (no `debug_plot`: plot functions consume results or recompute through public functions) |

- **Responsibilities:** `filtering` holds signal-chain filters with a real frequency response; `smoothing` display and estimation smoothers only; `analysis` laser and phase characterization; `metrics` is the reporting layer.
- **Codes are storable:** `io` stores a code as its class and constructor fields, so those fields are numbers, strings, tuples, host arrays, constellations or other codes.
- **Splitting modules:** promote a flat module to a package only when it exceeds about 1000 lines *and* has two or more separable concerns. Never create a `utils/` package or one-function files.
- **One documented home per public name.** A package `__init__` re-exports its subpackages' public names, and users and documentation import them from the package (`commkit.impairments.Fiber`), not from the subpackage that defines them.
- **Shared physics lives in one private module**, used by the channel and by its compensation: `_dispersion` (β₂, β₃, the dispersion response) and `_fiber` (`Fiber` and the split-step solver behind `apply_fiber_propagation` and `correct_fiber_propagation`).

---

## 4. API rules

### Signal

- `Signal` holds:
  - samples: NumPy or CuPy, `(N,)` or `(C, N)`, time on the last axis;
  - the **facts** `sampling_rate`, `symbol_rate`, `center_frequency` and `domain` (`"DIG"`, `"RF"` or `"OPT"`; only RF and optical signals have a carrier, and an optical carrier gives the derived fact `center_wavelength_nm`);
  - the **description** `constellation` and `pulse`;
  - the ground truth `reference`: transmitted symbols, the bits on the air `bits`, the source bits `info_bits` (the same array when nothing is coded or matched) and the `code` between them, plus the `frame` layout.
- It never holds pipeline results.
- Construction does no hidden work: no mapping, no normalization, no device move, no shape guessing. Factories (`generate`, frames) do that work.
- `sig.to("gpu")` and `sig.replace(...)` return new Signals sharing the arrays; `clone()` copies.
- **Alignment invariant:** reference symbol *k* corresponds to symbol period *k* of the samples. A function that drops or shifts symbol periods slices the reference to match. Metrics raise on mismatch; they never guess.

### Function signatures

- `f(data, other_data, *, params)`: everything after the data arguments is keyword-only. A function without data arguments may keep a single primary parameter positional (`barker_sequence(13)`); with several, all are keyword-only. The contract registry records each function's data-argument count.
- **Signal awareness.** A function whose main argument is waveform data (or a field a Signal carries) accepts an array or a Signal: it unwraps once with `adapt_signal()` (`core/_signal_adapter.py`), works on arrays, and returns through `wrap_samples()`, never by re-entering the public function. Functions on *derived* quantities (phase trajectories, correlations, taps, PSD bins, results) and synthesis primitives stay array-only.
- **Types:** a transform is `def f(samples: S, ...) -> S` with the TypeVar `S` from `core/_signal_adapter.py`. Do not reassign `samples`; name the unwrapped array `x`.
- **Facts versus choices.** Facts (`sampling_rate`, `symbol_rate`, hence `sps`) come from the Signal; a conflicting argument raises `ValueError`, an equal one is fine, and a missing fact for array input raises. Choices (decision `constellation`, `noise_var`, `pulse`) default to the Signal's value; an explicit argument wins. Neither case warns.
- Integer-SPS algorithms validate with `require_integer_sps()` before any cast; never truncate 1.5 SPS to 1.
- **Shapes:** SISO is `(N,)`, MIMO `(C, N)`. Use `as_2d`, `restore_1d`, `broadcast_channels` and `require_channels`; never hand-roll the promote/squeeze idiom.
- **Value objects are frozen dataclasses** with read-only NumPy arrays:
  - `Constellation`: `.qam(M)`, `.psk(M)`, `.pam(M, unipolar=)`, or arbitrary `points`; `.shaped(nu=|entropy=)` keeps unit power. Functions accept the object only, never strings.
  - Pulses: `RRC`, `RC`, `Gaussian`, `Rect`, `SmoothRect`. A raw taps array is accepted wherever a pulse is.
  - Algorithm objects, one per method (`BPS`, `PLL`, `ViterbiViterbi`, `MthPower`, `CycleSlip`, ...), used standalone (`correct_carrier_phase(y, BPS())`) and nested (`lms(..., cpr=BPS())`). Each lives next to its kernel and validates its parameters on construction. Class names are unique across the package.
- **Design and apply are separate functions**: `RRC(...).taps(sps)` and `fir_filter`, `butterworth_sos` and `iir_filter`.
- **Randomness:** `rng: int | np.random.Generator | None` (SciPy SPEC 7); never a global RNG.
  - What is transmitted (bits, symbols, pilots) and the few random parameters of a realization (the PMD waveplates of `Fiber`) are drawn on the host, so a seed gives the same payload and fiber on every device.
  - What the channel adds (AWGN, phase noise, ASE) is drawn on the data's device through `_random.standard_normal` / `complex_normal`; realizations differ between devices, statistics agree.
  - A seed is reproducible within a library version only. Tests and baselines assert statistics, never exact realizations.
- **Device follows the data.** No `backend=` arguments, no global switches. Unsupported array types (JAX, PyTorch) raise `TypeError`.
  - Only factories with no input data to follow take `device: str = "cpu"` (never "GPU if available"): `generate`, `generate_phase_noise`, `Preamble`/`SingleCarrierFrame.to_signal`, `load_npz`, `decoder_exit_curve`, `demapper_exit_curve`, `error_rate_curve`. They validate it with `require_device()` (in `coding`, by the same check in `coding.exit`), draw randomness as above and do the vectorized work on `device`.

### Returns

- **Transforms:** array in gives array out on the same device; Signal in gives a new Signal (with `sampling_rate` updated when the rate changes).
- **Estimators** reduce over the time axis: `(N,)` gives 0-d, `(C, N)` gives `(C,)`, on the input device.
- **Metrics** (`evm`, `snr`, `ber`, `fer`, `ser`, `gmi`, `ngmi`, `mi`) and **analysis summaries** (`estimate_linewidth`, `allan_deviation`, the `frequency_drift` statistics, which fit plot-sized reductions) return host values: a `float` for 1-D input, `np.ndarray (C,)` for 2-D. Sample-rate outputs (trajectories, PSDs, `dsh_phase`) stay on the input device.
- Metrics **raise** on empty selections, never return `None` or `0.0` for "nothing measured", and never silently align, rotate or permute against the reference.
- **More than one value** is a frozen dataclass with named fields. No dicts, no tuples longer than two.
- **Equalizers** return `EqualizerResult`: `y_hat` is always an array, `signal` the 1-SPS Signal for Signal input, `state` the continuation state (`state=result.state`), and diagnostics as data.

### Naming

| Prefix | Meaning |
| --- | --- |
| `apply_*` | add an impairment or apply a model |
| `estimate_<quantity>(x, method)` | measure, never change the data; returns a frozen `<Quantity>Estimate` (value plus diagnostics) |
| `correct_<quantity>(x, how)` | remove the impairment; returns only the corrected data. `how` is an estimate or an algorithm object (which estimates first) |
| `resolve_*` | pick among discrete candidates (π/2 rotation, channel permutation); nothing else |
| `generate` | synthesis from parameters, takes `rng` |
| `encode` / `decode` | information bits to a code's transmitted bits, and LLRs (or hard decisions) back to a `DecodeResult`; dispatched on the code object, the decoder chosen by a `method` object |
| `plot_*` | every public plotting function (only `apply_default_theme` is exempt) |

- The algorithm is chosen by an object, never by the function name or a string: `estimate_carrier_phase(y, BPS())`, not `recover_carrier_phase_bps(y)`. The verb function dispatches on the object's type through a private table in its module. There is no `recover_*` or `compensate_*`.
- Computations use plain nouns (`allan_deviation`).
- **Glossary**, one name each:
  - `num_taps`, `step_size`, `sps`, `constellation`, `pulse`, `rng`, `num_symbols`, `num_channels` (never streams), `sampling_rate`, `symbol_rate`, `noise_var`, `nfft`, `regularization`, `max_iterations`.
  - A size names its unit: `num_symbols`, `num_blocks` (codewords of a FEC code), `info_bits` (a code's information length where it is a parameter), `blocks_per_frame`, `num_variables`, `num_segments`. In `coding`, `length` and `n` mean the codeword length in bits and `k` the information length, nothing else.
  - A seeded part of a factory takes `<part>_rng` (`payload_rng`), with the semantics of `rng`.
  - `method=` strings name only a numerical flavor of one algorithm (`method="maxlog" | "exact"`); algorithms are objects.
  - Values are in SI units without a suffix (`linewidth` in Hz, `drift_rate` in rad/s, `taus` in s). A result holding one quantity in two units suffixes the SI one (`lags` in samples, `lag_s`). A conventional non-SI unit goes into the name: `_db`, `_dbm`, `_deg`, `_nm`, `_km`, `_ps_nm_km`. Normalized quantities say so (`normalized_bandwidth`). No second name for one quantity in another unit (no `center_frequency_ghz`); `center_wavelength_nm` is a different, derived quantity.

### Docstrings

Docstrings describe the code as it is; a change in behavior updates them in the same commit. NumPy format (numpydoc), sections in its order: Parameters, Returns, Raises, Warns, See Also, Notes, References, Examples.

- **Summary line** on the line of the opening quotes, one line, ending with a period. Functions: imperative mood (`Apply ...`), or a noun phrase for a function named after what it returns (`evm`, `welch_psd`). Classes and modules: a noun phrase.
- **Module docstrings** name what the module holds (`:func:`, `:class:`), then the conventions particular to it (a sign convention, a scaling, a file layout). No marketing adjectives, no history of earlier layouts, no restatement of this file. A private module's summary ends with "(private)" and the body names its users.
- **Types:** inputs are `array_like`, or `array_like or Signal` where Signal-aware; returned arrays and fields are `ndarray` (on the input's device), `ndarray or Signal` for transforms, `numpy.ndarray` where always on the host. Shapes go in the description in double backticks: ``(C, N)``.
- **Plots** return `(fig, ax) or None`, multi-panel plots `(fig, axes) or None` (`None` with `show=True`).
- **Defaults:** `name : type, default value`; `name : type, optional` when the default is `None`, with the description saying what `None` does.
- Units go in the description; inline code in double backticks; library names through `:func:` / `:class:` / `:mod:` / `:meth:`.

### Errors and logging

- Wrong input raises, with a message naming the argument.
- **No fallbacks for a broken environment.** Code never catches a failure of CuPy, a CUDA kernel (compile, load, launch), a graph capture or another dependency to run a different implementation; the failure raises to the caller. A path kept "just in case" is code no test or user runs. (Dispatch on the *input* is allowed; see §6 Kernel policy.)
- `logger` is for diagnostics only. Per-channel logging that needs a device-to-host transfer is wrapped in `if logger.isEnabledFor(logging.INFO):`.
- **Levels:** `DEBUG` traces what a call is about to do (method, parameters, sizes). `INFO` is one summary per public call of what a user would print next (the values found, the parameters applied, an equalizer's exit MSE, a decoder's failed checks, what `generate` or `io` produced). `WARNING` flags a returned but suspect result (no convergence, a fit outside its valid range, near-zero power, a code with 4-cycles). Value objects and pure helpers do not log.
- `warnings.warn` is used only for `PrecisionWarning` (§5). The library never configures handlers, Matplotlib or warning filters at import time.

---

## 5. Numerics

- **Storage dtypes:** `complex64` for IQ samples, `float32` for real signals; `generate`, frames and `load_npz` produce them, so single precision is the default path.
- **Precision contract.** Every function computes in a predictable precision, and none promotes single-precision data on its own:
  - **Transforms keep the input's precision** throughout (filters, impairments, the fiber solver and DBP, resampling, delays, carrier-phase correction): complex64 in and out, complex128 in and out.
  - **Fixed-precision blocks** compute in single precision whatever the input: the equalizers in complex64, `compute_llr` / `Constellation.llr` in float32. Double input emits `commkit.PrecisionWarning` (`_array.warn_if_double`, attributed to the caller's line). The test suite turns it into an error (`pytest_configure` in `tests/conftest.py`; not pyproject's `filterwarnings`, which imports commkit before coverage starts), so a library path feeding double data to these blocks fails CI. Training and reference symbols are not checked.
  - **Deliberate float64 / complex128**, documented where it happens, only on phases, small arrays or accumulators, never on sample-rate data that is then cast back:
    - phases before `xp.unwrap()` (float32 rounding causes spurious quadrant slips), phase trajectories and ramps (wrapped in float64, rotated in the data's precision), frequency grids and estimates;
    - recursion state: PLL, Tikhonov/Kalman block transitions, the IIR filter's sections and state (poles near z = 1);
    - RLS keeps `P`, the gain `k` and the regressor buffers in complex128 (in single precision `P` loses positive definiteness and the filter diverges);
    - LMS/CMA/RDE accumulate dot products and gradient updates in complex128 around complex64 inputs and weights; the block equalizers stay complex64, as their tap sums run inside the FFTs;
    - sums: metrics compute per element in the input's precision and accumulate in float64.
  - complex128 pays only where single-precision rounding accumulates over many FFT passes, as in the fiber solver (6e-4 after 10^4 steps in complex64, 1e-12 in complex128); there the GPU runs its plain CuPy stages, not the fused complex64 kernels.
- **The NEP 50 trap:** under NumPy 2 a NumPy scalar is not weak, so `np.float64(a) * x64` (any `np.sqrt`, `np.pi * ...` result, or a float64 array such as `xp.fft.fftfreq`) promotes complex64 to complex128. Combine single-precision data only with Python scalars or arrays cast to its precision. Frequency responses (dispersion, delays, Jones matrices) are computed at float64 phase accuracy and cast to the spectrum's precision before the multiply; rotate with `math._derotate`.
- **Audit with `tests/plugins/precision_trace.py`** after touching numerics. It lists every float64/complex128 array of sample-rate size that a NumPy path creates from single-precision, integer or bool data or from scratch, with its source line; output dtypes alone hide a compute-in-double-and-cast-back. Every reported site must be one of the deliberate uses above.
- **Normalization invariant:** at `sps` samples per symbol, `E[|x|²] = 1/sps`; at symbol rate, `E[|x|²] = 1`. Any rate-changing block applies the exact gain correction.

---

## 6. Performance

| Algorithm structure | Implementation | Device |
| --- | --- | --- |
| Vectorized: filters, resampling, spectra, impairments, metrics, frequency-domain equalizers | NumPy/CuPy via `x, xp, sp = dispatch(samples)` | follows the input |
| Per-symbol searches over a small table: LLRs (all points, or per axis for separable QAM; max-log without a prior in closed form), hard decisions without a square lattice | CPU: Numba `parallel=True` over blocks of symbols (LLRs), chunked NumPy (decisions). GPU: one thread per symbol (`llr`, `llr_separable`, `llr_separable_maxlog`, `bps_min_d2`'s nearest-point mode); decisions on double input or above 1024 points use the chunked search on both devices; LLRs above 16 bits per symbol raise on the GPU | follows the input |
| Sequential recursions: LMS/RLS/CMA/RDE, PLL, Tikhonov, cycle slips, the Mengali-Morelli bootstrap, per-codeword decoders on trellises and trees (CCDM, polar SC/SCL, Viterbi, BCJR, turbo, recursive systematic encoding) | Numba `@njit(cache=True, fastmath=True, nogil=True)`, independent rows in `prange`. The equalizers' kernel with inline carrier recovery compiles strict (unwrap, running sums) and calls fast-math tap and distance loops as separate functions (an `inline="always"` helper takes its caller's flags) | CPU; GPU input does one round trip |
| Linear recursions: IIR filtering | CPU: Numba without `fastmath`, so double input matches SciPy bit for bit. GPU: `iir` kernel: each block's state from a zero start in parallel, states carried across blocks in two levels (groups, then across groups), blocks filtered again, in double precision | follows the input |
| Binary linear algebra of codes: encoding, syndromes, ML-decoding correlations on the GPU, Chase's test patterns | one float32 GEMM per chunk (`coding._gf2.matmul_mod2`), exact below an inner dimension of 2^24, parity in place (CPU) or by one elementwise kernel writing int8 (GPU); small tables cached per code and device (`backend._device_tables`) | follows the input |
| Algebraic decoding (BCH, Reed-Solomon): Berlekamp-Massey, Chien search to the locator's degree, Forney | CPU: Numba `parallel=True` over codewords. GPU: `algebraic_correct`, one thread per codeword in integer table arithmetic, so both agree exactly (a lockstep CuPy version was 12x slower) | follows the input |
| Layered LDPC belief propagation (min-sum, sum-product) | CPU: Numba `parallel=True` over codewords, all iterations inside, no fast-math. GPU: `ldpc_layered`, one block per codeword, rows of a layer in parallel, early stop per codeword; same row order, so min-sum agrees bit for bit (a CuPy version was 7-23x slower) | follows the input |
| Maximum-likelihood decoding: the correlation with all `2^k` codewords, the decision and the per-bit maxima of the max-log soft output | CPU: one Numba pass, `parallel=True` over rows, branch-free selects under fast-math without the no-NaN and no-Inf flags (a GEMM then a separate pass ran 3x slower, its threads competing with BLAS's). GPU: the GEMM per chunk, the per-bit maxima by a CuPy `ReductionKernel` that broadcasts over the codewords instead of building `(rows, codewords, n)` | follows the input |
| Monte Carlo analysis (EXIT curves, error-rate curves) | the existing encoders, decoders and demapper and vectorized NumPy/CuPy, through the private block functions (no INFO line per batch); closed-form parts are host NumPy on small grids. No kernel of its own | follows `device` |
| Elementwise chains between multi-core FFTs (fiber solver's Kerr and PMD stages) | NumPy: a Numba `parallel=True` pass competes with `scipy.fft`'s threads and measured slower end to end; adopt one only on a paired end-to-end A/B | CPU |
| Other hot spots limited by intermediates, launches or round trips | CUDA C++ via CuPy `RawModule` (`commkit/_cuda/src`) | GPU |

**Numba kernels live at module level** in a private module that a getter imports on first use (`equalization/_numba_loops.py`, `coding/_numba_loops.py`, `mapping/_numba_llr.py`), so `import commkit` does not load Numba and `cache=True` works across processes. A kernel defined inside a function that calls another kernel defined there recompiles in every process (Numba's cache key misses the captured dispatcher); a getter may build a closure that captures only constants (`_get_numba_sos`).

**Host-sync hygiene:**

- Never extract scalars from a possibly-GPU array inside a loop (`float(x[c])`, `.item()`). Compute the vector on the device, transfer it once with `to_device(v, "cpu")`, then loop on the host.
- Reduce on the device before transferring, including for plotting; prefer on-device gathers (`xp.take_along_axis`); chunk `(N, M)` intermediates over N with an on-device accumulator.
- Build grids, windows and index tables on the data's device (`xp.fft.fftfreq`, `xp.hanning`, `xp.arange`).
- Every FFT goes through `commkit._fft` (`fft`, `ifft`, `rfft`, `irfft`), never `np.fft` / `xp.fft` (frequency grids excepted): on the CPU it runs multi-threaded `scipy.fft` and keeps complex64 in single precision (NumPy 2 computes it in double); on the GPU it is `cupy.fft`.
- Pass whole records to compiled code. No Python loops over symbols or small blocks, except inside a CUDA graph.
- Repeated GPU work with fixed buffers (the block equalizers' blocks, the fiber solver's spans) is replayed from CUDA graphs through `_cuda.graphs.GraphRunner`: warm once, capture on the second run, replay; a failed capture raises. Per-replay offsets come from a device index the graph advances, so the host launches one graph per block. Its stream is reused per device and thread, because CuPy's pool caches freed blocks per stream.
- **Audit with `tests/plugins/transfer_trace.py`** after touching a GPU path; the contract registry cannot see a silent round trip through the host. The plugin records every sample-rate host-device copy during a `--device=gpu` run and fails on a site missing from its documented list (the sequential recursions' round trip, equalizer continuation state, the reporting layer, explicit device moves, the transmitted payload, host values such as reference symbols or SciPy windows moved to the device). A new site is fixed or added to the list with its reason.

**Kernel policy:**

- A `.cu` kernel needs a benchmark showing that CuPy is limited by intermediates, launch count or round trips. It optimizes a CPU reference (Numba or NumPy) and is tested against it with `--device=all`.
- `_cuda.get_kernel` returns the launch wrapper or raises (compile, load, bad specialization); it is called only on CuPy input.
- Wrappers check dtype, contiguity, shape and size before launch and never launch a zero-size grid: empty input returns an empty result, or raises where the caller never has one (`cs_block`, `bps_anchor`). Grid dimensions y and z stop at 65535; a kernel mapping channels or rows there walks the rest in a grid-stride loop (`bps_min_d2`).
- **Dispatch on input limits:** a kernel's documented limits (dtype, number of points, phases, channels) may route to the general implementation where it exists anyway (the CPU's), saying so where it happens. A GPU-only path serving only inputs beyond a kernel's limit is not written; such input raises (`compute_llr` above 16 bits per symbol).
- `float32` with explicit literals. `--use_fast_math` only per kernel (through `options`), after an accuracy check.
- Short elementwise chains use `ElementwiseKernel` (named `commkit_*`) or `cupy.fuse`.
- Sequential recursions over a whole record are never ported to the GPU (IIR, above, is the linear exception). A short per-block scan may be, as one launch, when it replaces a host round trip per block (`cs_block`, which speculates its slip decisions in parallel; `bps_anchor`); keep its parallel part parallel, as float64 runs at 1/64 of float32 on consumer GPUs.
- A new or changed kernel passes `scripts/sanitize.sh` (memcheck, racecheck, synccheck, initcheck) before release; add its tests and kernel-name prefix to the script.

---

## 7. Tests

- **Fixtures:** use `backend_device`, `xp` and `xpt` from `tests/conftest.py`. Assert with `xpt.assert_allclose(...)` (`numpy.testing` raises on CuPy arrays); cast expectations with `xp.asarray` and reductions with `float(...)`.
- **No configuration skips.** The GPU legs skip only when CuPy is not installed, and fail instead under an explicit `--device=gpu`; a broken CuPy fails under any setting. Tests outside the selected run are deselected at collection (`gpu_only`, `cpu_only`, `slow` without `--run-slow`), so a run with CuPy reports no skips. A new test never calls `pytest.skip` for a configuration.
- **Layout mirrors the source:** `tests/<subpackage>/test_<module>.py`, `tests/test_<module>.py` for flat modules. Basenames are globally unique (prepend import mode), hence `channel/test_channel_linear.py`. Split a large file by concern. Shared builders live in `tests/common/` and use the public API.
- **Contract registry** (`tests/test_api_contracts.py`): every public function has a row, checked generically for array and Signal in/out, device preserved, input not mutated, keyword-only parameters, `TypeError` on unsupported arrays, fact conflicts and return rules. Do not repeat these checks per module.
- **Oracles:** sequential kernels are tested against plain-Python references in `tests/common/reference_impl.py`, CUDA kernels against their CPU reference. Kernel tests skip only on the CPU parametrization (`skip_unless_gpu`), so a kernel that fails to compile fails its tests.
- **Mathematical meaning:** test documented definitions (a pulse width, a dispersion sign, noise-variance scaling) against independently derived expectations, not only shapes and energies.
- **A test of what an algorithm achieves** (an equalizer converges, carrier recovery tracks, a decoder corrects, an estimator is calibrated) must fail on a plausibly wrong implementation:
  - **Score against the truth**: the transmitted reference or the true phase, delay or offset, never the receiver's own decisions or a sibling estimate; constellation symmetry hides a wrong quadrant, a slip or a duplicated stream. Helpers are in `tests/common/scoring.py` (`ser`, `blind_ser`, `phase_rms`, `block_rotations`, `stream_correlation`).
  - **Resolve only the legitimate ambiguities**, each once over the whole scored interval, never per segment: one rotation per stream (by the constellation's symmetry, or any angle for a phase-blind equalizer), one bijective channel permutation for blind MIMO, a documented fixed delay.
  - **A baseline must fail**: the same test without the block it credits (step size 0, `cpr=None`, the corrector off) asserts failure.
  - **Tolerances are derived** (AWGN theory, a Cramér–Rao bound, a Wiener or loop-noise prediction, a binomial interval), and the docstring states the derivation and the measured value. `tests/common/theory.py` holds the closed forms, in NumPy and SciPy only.
  - **Real channels**: convergence tests use `make_channel_signal` (RRC pulses, ISI, fractional delay, polarization rotation, DGD, AWGN). The repeat-upsampled builders serve plumbing tests only.
  - **Break the code to check the test**: monkeypatch a plausible wrong result (zeros, a factor of two, the feature off) and confirm the test fails.
  - A strengthened test that exposes a library fault is marked `pytest.mark.xfail(strict=True, reason=...)` with the measured numbers and listed in §10; the fix lands in its own commit, which removes the mark.
- **Monte Carlo tests** state what their sample bounds (0 failures in 12 frames bounds the FER below 0.22 at 95 %) and check the operating point (the pre-FEC BER against theory). The default run uses a smoke size; a `slow` parametrization repeats it with enough trials to mean something.
- **End-to-end chains** in `tests/test_receiver_chain.py` run several impairments and their receiver, scored against the source bits or AWGN theory, each with one correction left out as a failing baseline.
- **Warnings are failures**: a WARNING from the `commkit` logger fails the test unless it is marked `allow_warnings` (it provokes a suspect result on purpose) or asserts the log through `caplog`. A WARNING that reports what the caller asked for is a log-level bug, fixed in the library.
- **API changes never change expected values.** A signature change only changes how tests call the code; numerical changes land in their own commit with independent validation.
- **Examples** are Jupyter notebooks in `examples/`, committed without outputs (the `nbstripout` filter, §2). `tests/test_examples.py` executes each one, so an API change updates the notebooks in the same commit.
  - They form a course: one topic per notebook, `NN-lowercase-words.ipynb`, in teaching order (signals and DSP, channel and metrics, coding, receiver algorithms, fiber systems, characterization). Each opens with what it covers and links the notebooks it builds on, and ends with a link to the next. A new notebook takes its place in the order and updates `examples/README.md` and its neighbors' links.
  - Every claim in the text matches what the cells print: run the notebook and read its numbers before committing. Keep each under about 20 s on the CPU.
  - Plots of hand-computed arrays go through `to_device(x, "cpu")`, so the notebooks run unchanged on the GPU.
- **Coverage** must not drop below the CI floors: 78 % for the package, 85 % for the Numba kernel modules (the `NUMBA_DISABLE_JIT=1` job, §2).

---

## 8. Benchmarks

`benchmarks/` tracks the GPU-relevant hot paths. Baselines are committed under `benchmarks/baselines/`. The reference is `0008` (`v3_1`, the 3.1 state) on the reference machine: RTX 4070 Ti, Ryzen 7 7800X3D, WSL2. Earlier baselines lack later benchmark files; `0002` ran the CPR equalizer and Viterbi-Viterbi benchmarks on workloads that did not converge, so compare those only from `0003` on.

- **Workloads must converge**, or the benchmark times a failure mode (slip storms, divergence). Every equalizer benchmark asserts its symbol error rate with `benchutils.assert_converged`.
- **Workloads** come from `benchmarks/workloads.py` with fixed seeds; never generate data inline.
- **Baselines** are recorded with `benchmarks/record_baseline.py`: each file in its own process, best of `--passes` (default three; raise it when small benchmarks keep outliers, as `0008` did with ten). Never record from a single `pytest benchmarks/` run: small GPU benchmarks after the large equalizer workloads ran several times slower than alone.
- **IDs** are `[cpu]` or `[gpu]`, meaning the input device.
- **Timing:** timed bodies end with the `sync` fixture, or GPU timings measure launches. Each benchmark has one warmup round (compilation, pool growth), and a session fixture warms Numba's thread pool before the first.
- **Tools:** `benchmarks/benchutils.py` provides `CudaEventTimer` and `nvtx_range` for `nsys`. Logging is set to WARNING in `benchmarks/conftest.py`.
- **Trust deltas, not single runs.** Swings of ±20-40 % have been seen under load. Re-run a suspected regression in isolation, or do a paired A/B of at least 7 repetitions, alternating old and new. When even unchanged benchmarks move, time the changed part with `CudaEventTimer` and quote it against the step it sits in.
- A commit touching a hot path quotes its delta against `0008`. Tolerance is 5 % on CPU and 10 % on GPU.

---

## 9. Release

1. On `main`: the CPU and GPU suites pass with no failures or skips, slow validation included (`uv run pytest -rs --run-slow`; strict xfails, if any, are listed in §10); `scripts/sanitize.sh` is clean if a kernel changed; `uv build` succeeds and `uv run --with twine twine check dist/*` passes.
2. `CHANGELOG.md`: a `## [X.Y.Z] - YYYY-MM-DD` section written for users. Changes that may need code updates come first, with how to migrate, then Added, Performance (measured, as a ratio of new to old time), Fixed and Development. Add its compare link at the bottom and commit.
3. `uv run bump-my-version bump {patch|minor|major}` commits the new version and tags `vX.Y.Z`.
4. `git push origin main --tags`. The ruleset on `main` requires pull requests; repository admins bypass it for these two commits.
5. `gh release create vX.Y.Z --title vX.Y.Z --notes-file notes.md` with the changelog section as notes. Publishing the release runs `.github/workflows/publish.yml`, which uploads to PyPI through Trusted Publishing; a pushed tag alone publishes nothing.

---

## 10. Roadmap and open items

What is not done yet, in suggested order. Unlike the sections above, this is a to-do list: remove an item in the commit that resolves it, and add one when a known gap is left open.

### Switch to Python 3.15 (blocked)

Goal: require Python 3.15 only, mainly for explicit lazy imports (PEP 810). State checked on 2026-10-23 (uv 0.13.5, CPython 3.15.1):

- **Blocker: CuPy has no 3.15 wheels.** `cupy-cuda13x` 14.2.0 ships cp310-cp314 only; support is tracked in cupy/cupy#10259, milestone v14.3.0.
- **Blocker: `jupyter_client` fails to import on 3.15**: `jsonutil.py` calls `strptime` with a day and no year, which 3.15 rejects (cpython#70647), so `tests/test_examples.py` fails at collection. Wait for a fixed release; do not work around it in commkit.
- **Ready:** wheels on 3.15 exist for NumPy 2.5.4, SciPy 1.18.1, Numba 0.68.0, Matplotlib 3.11.2, mypy 2.4.0 and the dev tools. The CPU suite has not been run on 3.15.

Steps, each its own commit (the version switch first, then each refactor):

1. **Validate unchanged code** in a 3.15 venv: full CPU+GPU suite with notebooks (`-rs`), benchmark smoke run, kernel-coverage job, precision audit.
2. **Switch the floor:** `requires-python = ">=3.15"`, classifiers, ruff `target-version`, dependency minimums to the first 3.15 wheels (`cupy-cuda13x[ctk]>=14.3`, `numba>=0.68`, `numpy>=2.5`, `scipy>=1.18`; recheck each); `ci.yml` matrix and jobs and `publish.yml` on 3.15 (keep `locked` and `lowest-direct`); `uv lock`; README and §2. Record a 3.15 baseline, compare it with `0008` by a paired A/B on unchanged code, and make it the reference.
3. **Lazy imports.** `lazy` is module-level only (not in functions, classes or `try`, not with `*`); confirm on the interpreter.
   - Move the function-local imports (`import cupy as cp`, `import numba`, `from . import compiler` in `_cuda`, cycle-breaking imports of `Constellation`, `Signal`, `normalize`) to `lazy import` / `lazy from`; find them with `grep -rnE '^\s{4,}(import|from) ' commkit`.
   - CuPy stays optional: every use stays behind `is_cupy_available` / `dispatch`; decide per site whether `backend.py`'s probes stay eager inside `try`.
   - Check that a lazy `numba` does not move the import cost to an odd first call, and turn `TYPE_CHECKING` blocks into `lazy from` where the name is useful at run time.
   - Measure with `benchmarks/bench_import.py`; keep `tests/test_import_hygiene.py` passing and extend it to assert that `import commkit` loads neither CuPy nor Matplotlib, if that becomes true.
4. **Other cleanups** (verify each against What's New in 3.14 and 3.15): drop `from __future__ import annotations` (PEP 649/749; check nothing reads `__annotations__` as strings); check `open()` without `encoding=` under default UTF-8 mode (PEP 686); grep for `strptime` with a day and no year; try the sampling profiler (PEP 799) on the CPU hot paths and mention it in §8 if useful.

### Channel coding

1. **oFEC** (OpenROADM, ITU-T G.709.3) is not implemented: its block structure is specified in OpenROADM MSA documents that were not available in a form to implement from. It would build on `BCHCode` (extended BCH(256, 239)), soft-output `Chase` and the staircase decoder's windowed iteration.
2. **NR rate matching for polar codes** (sub-block interleaving, circular buffer, puncturing and shortening with their frozen-set rules, parity-check bits) and **LTE rate matching for turbo codes** are not included: `PolarCode` and `TurboCode` are the mother codes; `Punctured` and `Shortened` cover generic rate matching.

### Real-valued channels

- **`apply_awgn` and `compute_llr` disagree on real constellations.** `apply_awgn` gives real samples real noise of the full variance `σ²`, while `compute_llr` / `Constellation.llr` score `-|d|²/σ²`, the circular-noise model whose real part carries `σ²/2`. PAM LLRs computed with the `noise_var` that `apply_awgn` used are twice too confident, in any IM/DD link built from the two (notebook 21's PAM4 link was, until it moved to `error_rate_curve`). `error_rate_curve` uses the circular model (`N0/2` per real dimension, BPSK `BER = Q(sqrt(2 Es/N0))`). Pick one convention for real symbols, document it in both functions and in the metrics units table, and land it in its own commit with the tests' expected values rederived.

### Test suite

- **Standard tables against the standards.** The NR polar reliability sequence (TS 38.212 Table 5.3.1.2-1), the LTE QPP interleaver parameters (TS 36.212 Table 5.1.3-3) and the DVB-S2 and PON parity-check address tables are tested against commkit's own tables. They should be compared in full with fixtures under `tests/data/standards/` transcribed from the specifications, each with a provenance header (document, version, table). ETSI's download pages refuse scripted access; the documents have to be supplied.

### Nonlinear fiber channel

Each is an addition that leaves the current API unchanged.

- **Batched realizations**: a leading axis `(B, 2, N)` (launch-power sweeps, Monte Carlo over seeds) so cuFFT batches them and short records fill the GPU.
- **Adaptive step rules**: a `step_rule=` object (maximum nonlinear phase per step, local-error method) next to the equal `steps_per_span`, which measured 3x more accurate than logarithmic steps on a 32 GBd, 80 km span.
- **WDM helpers**: combining frequency-shifted channels into one wideband field and selecting one after propagation; the solver already handles the wide field (with `dispersion_slope_ps_nm2_km` set).
- **Distributed (Raman) amplification and per-span fiber types** in one call (chaining calls already works: every call speaks launch-relative units).
- **cuFFT callbacks** fusing the linear stage into the transforms (CuPy's support is experimental and compiles with nvcc at run time).
- **Self-steepening and Raman scattering** for ultra-wideband fields.
