# CommKit

**Digital-communications research kit for Python, on CPU and GPU.**

![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)
![Backends](https://img.shields.io/badge/backend-NumPy%20%7C%20CuPy-orange)
![License](https://img.shields.io/badge/license-MIT-green)
![CUDA](https://img.shields.io/badge/CUDA-13.x-76B900?logo=nvidia)

CommKit covers the receiver chain of coherent and IM/DD links (waveform
generation, channel impairments, synchronization, carrier recovery, adaptive
equalization, metrics, laser characterization and plotting) on NumPy or CuPy
arrays. Computation runs on the device the data lives on.

---

## The model

```text
plain arrays (NumPy / CuPy)          value objects (frozen, written inline)
        \                            Constellation, RRC / RC / Gaussian / Rect,
         \                           BPS / PLL / CycleSlip, MthPower, ...
          v                                   |
       Signal  = samples + facts + description + reference
          |
          v
   functions: generate, apply_*, estimate_*, correct_*, resolve_*, ...
          |                 \
          v                  v
   Signal or array      typed results (EqualizerResult, *Estimate, ...)
                             |
                             v
                   plotting (draws results, never computes them)
```

- **Algorithms are plain functions.** Data is a `Signal` or an array.
- **Things that describe** a modulation, a pulse or a sub-algorithm are small
  frozen objects written in the call: `lms(rx, cpr=BPS(test_phases=64))`.
  The same object works standalone: `correct_carrier_phase(y, BPS())`.
- **A `Signal` carries facts and ground truth.** Facts are `sampling_rate`,
  `symbol_rate` and `center_frequency`. The description is `constellation`
  and `pulse`. The ground truth is `reference` (transmitted symbols and
  bits). Functions take facts from the Signal and raise on a conflicting
  argument. Choices such as the decision constellation default to the
  Signal's and can be overridden.
- **Results with several values are frozen dataclasses** with named fields.
  Metrics return host floats: one per channel for `(C, N)` input.
- **The device follows the data.** Processing functions have no `backend=`
  or `device=` argument. Only factories, which have no input data to
  follow, take `device=`: `generate(..., device="gpu")`. Move existing data
  with `sig.to("gpu")`.
- There is no pipeline object, receiver class or configuration file: the
  orchestration stays in your script.

## Quickstart

```python
import commkit as ck
from commkit import RRC, Constellation
from commkit.recovery import BPS, CycleSlip

tx = ck.generate(Constellation.qam(16), num_symbols=2**16, symbol_rate=32e9,
                 sps=2, pulse=RRC(rolloff=0.1), rng=1, device="gpu")
rx = ck.impairments.apply_phase_noise(tx, linewidth=100e3, rng=2)
rx = ck.impairments.apply_awgn(rx, esn0_db=18, rng=3)   # on the GPU, like tx
rx = ck.filtering.matched_filter(rx)                     # pulse from rx

cpr = BPS(test_phases=64, cycle_slip=CycleSlip(history=100))
res = ck.equalization.lms(
    rx, num_taps=21, step_size=1e-3,
    training_symbols=rx.reference.symbols[..., :2000], cpr=cpr,
)
y = res.signal                    # 1-SPS Signal, reference aligned
print(f"EVM {ck.metrics.evm(y, num_skip_symbols=2000):.1f} %, "
      f"BER {ck.metrics.ber(y, num_skip_symbols=2000):.1e}")
# EVM 12.8 %, BER 2.5e-04

# The next record continues from the converged taps and CPR state.
res2 = ck.equalization.lms(rx, num_taps=21, step_size=1e-3, cpr=cpr, state=res.state)
```

Without CuPy, pass `device="cpu"` (the default): the same code runs on the
CPU.

## CPU and GPU

CommKit never picks a device for you: data stays where you put it, and each
function runs on the device of its input. A few habits get the speed out of
a GPU:

- **Put data on the GPU once, at the source.** Build it there with
  `generate(..., device="gpu")`, `frame.to_signal(..., device="gpu")` or
  `load_npz(path, device="gpu")`; move a capture with `sig.to("gpu")`. Do not
  move data back and forth between stages. Building on the GPU is faster
  than `generate(...).to("gpu")`: about 26 ms instead of 370 ms for 4M
  symbols, because mapping and pulse shaping run there.
- **You do not need to move results back.** Metrics (`evm`, `ber`, ...) and
  analysis summaries return host floats, and plots reduce on the device and
  transfer only what they draw.
- **Know what runs where.**

  | Work | GPU input |
  | --- | --- |
  | Generation, filters, resampling, spectra, impairments, BPS, LLRs, metrics, analysis | runs on the GPU |
  | Frequency-domain equalizers `block_lms`, `block_cma`, `block_rde` | runs on the GPU (CUDA graphs) |
  | Sequential equalizers `lms`, `rls`, `cma`, `rde`; `PLL`, `Tikhonov`, cycle-slip correction | runs on the CPU (Numba): one copy to the host and back, no speedup |

  A sample-by-sample recursion has no parallel work per step. For long
  records on the GPU, prefer the block equalizers with `block_size=1024` or
  more. Each block costs a fixed overhead, so at 256 the GPU is slower than
  the CPU, and at 2048 it is about 4x faster.
- **Short records do not pay off.** Launch overhead dominates below roughly
  10⁴-10⁵ samples, where the CPU is as fast.
- **Batch channels and records.** Pass `(C, N)` arrays rather than looping
  over channels in Python: one call does the work of C.
- **Keep scalars on the device inside loops.** `float(x)`, `x.item()` and
  `if x > 0:` on a CuPy array wait for the GPU and copy. Collect values in an
  array and transfer once.
- **The first call compiles.** Numba and CUDA kernels compile on first use
  and are cached on disk. Warm up once before timing anything.
- **Seeds and devices.** The same `rng` gives the same bits and symbols on
  both devices. Channel noise (AWGN, phase noise) is drawn on the device, so
  its realization differs between CPU and GPU while its statistics match.
  For the exact same phase-noise trajectory on both, draw it on the CPU with
  `generate_phase_noise(...)` and move it.

---

## Modules

| Module | Contents |
| --- | --- |
| [`commkit.core`](https://github.com/lokgar/commkit/tree/main/commkit/core) | `Signal`, `Reference`, `generate`, `SingleCarrierFrame` and `Preamble` (pilots, guard intervals, MIMO streams), `extract_payload`. |
| [`commkit.mapping`](https://github.com/lokgar/commkit/tree/main/commkit/mapping) | `Constellation` (QAM, PSK, PAM, arbitrary points; Gray labels; probabilistic shaping with `.shaped()`), bit mapping, hard demapping, max-log and exact LLRs. |
| [`commkit.filtering`](https://github.com/lokgar/commkit/blob/main/commkit/filtering.py) | Pulses (`RRC`, `RC`, `Gaussian`, `Rect`, `SmoothRect`), FIR and IIR design (Butterworth, Chebyshev I/II, elliptic, Bessel), `fir_filter`, `iir_filter`, `matched_filter`, overlap-save, chromatic-dispersion compensation. |
| [`commkit.multirate`](https://github.com/lokgar/commkit/blob/main/commkit/multirate.py) | `resample` (fractional), `decimate`, `upsample`, `decimate_to_symbol_rate`. |
| [`commkit.spectral`](https://github.com/lokgar/commkit/blob/main/commkit/spectral.py) | Welch PSD, spectrograms, frequency shifting, pilot tones. |
| [`commkit.impairments`](https://github.com/lokgar/commkit/tree/main/commkit/impairments) | AWGN, phase noise, IQ imbalance (with Löwdin and Gram-Schmidt correction), chromatic dispersion, PMD and polarization mixing. Nonlinear channel models: **planned, not implemented**. |
| [`commkit.timing`](https://github.com/lokgar/commkit/blob/main/commkit/timing.py) | Barker and Zadoff-Chu sequences, `estimate_timing` / `correct_timing`, fractional delay estimation and correction. |
| [`commkit.frequency`](https://github.com/lokgar/commkit/blob/main/commkit/frequency.py) | `estimate_frequency_offset` / `correct_frequency_offset` with `MthPower`, `MengaliMorelli`, `PilotSymbols` and `BiasTone`; static and blockwise. |
| [`commkit.recovery`](https://github.com/lokgar/commkit/tree/main/commkit/recovery) | `estimate_carrier_phase` / `correct_carrier_phase` with `ViterbiViterbi`, `BPS`, `PLL`, `Tikhonov`, `DataAided`, `PilotAided`, `PilotTone(s)`; cycle-slip correction; `resolve_phase_ambiguity` and `resolve_channel_permutation`. |
| [`commkit.equalization`](https://github.com/lokgar/commkit/tree/main/commkit/equalization) | Sequential `lms`, `rls`, `cma`, `rde` (Numba) and frequency-domain `block_lms`, `block_cma`, `block_rde` (CuPy, CUDA graphs), butterfly MIMO, inline carrier recovery (`cpr=PLL()` / `BPS()`), continuation with `state=`, `zf_equalizer`, polarization-tone demultiplexing. |
| [`commkit.metrics`](https://github.com/lokgar/commkit/blob/main/commkit/metrics.py) | `evm`, `snr`, `ber`, `ser`, `gmi`, `mi`, including shaped constellations; host values per channel. |
| [`commkit.analysis`](https://github.com/lokgar/commkit/tree/main/commkit/analysis) | Laser and carrier-phase characterization: `estimate_linewidth` (increment slope, β-separation, delayed self-heterodyne FM-PSD / increment / Lorentzian), FM-noise PSDs, drift separation, Allan deviation. |
| [`commkit.math`](https://github.com/lokgar/commkit/blob/main/commkit/math.py) | `rms`, `normalize`, dB conversions. |
| [`commkit.smoothing`](https://github.com/lokgar/commkit/blob/main/commkit/smoothing.py) | Display and estimation smoothers (moving average, Savitzky-Golay, 2-D density). Signal-chain filters are in `filtering`. |
| [`commkit.io`](https://github.com/lokgar/commkit/blob/main/commkit/io.py) | `save_npz` / `load_npz` for Signals. |
| [`commkit.plotting`](https://github.com/lokgar/commkit/tree/main/commkit/plotting) | Constellations, eye diagrams, spectra, filter responses, equalizer convergence, and synchronization and laser diagnostics that draw the estimates. Imported on first use. |
| [`commkit.coding`](https://github.com/lokgar/commkit/tree/main/commkit/coding) | Channel coding and FEC: **planned, not implemented**. |

Importing `commkit` has no side effects: it configures no logging, Matplotlib
or warning filters, and does not touch the GPU.

---

## Installation

**Requires Python 3.12+.**

```bash
pip install commkit                 # CPU
pip install "commkit[gpu]"          # CuPy with the CUDA 13 toolkit libraries
pip install "commkit[notebook]"     # to run the example notebooks
pip install "commkit[full]"         # everything
```

With [`uv`](https://github.com/astral-sh/uv), use `uv pip install` in place of
`pip install`. Extras combine, e.g. `commkit[gpu,notebook]`.

> [!NOTE]
> **WSL2 and CUDA.** With NVIDIA drivers and CUDA installed on Windows, there
> is no need to install CUDA inside WSL2. To let the Python CUDA packages find
> the bundled NVIDIA libraries, add to `~/.bashrc`:
>
> ```bash
> export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(echo $HOME/commkit/.venv/lib/python3.*/site-packages/nvidia/cu13/lib)
> ```
>
> *(This assumes the repository is cloned to `$HOME/commkit`; adjust the path
> otherwise.)*

## Examples

Jupyter notebooks in [`examples/`](https://github.com/lokgar/commkit/tree/main/examples) (install the `notebook` extra
and run `jupyter lab examples`):

- [`qam_receiver_quickstart`](https://github.com/lokgar/commkit/blob/main/examples/qam_receiver_quickstart.ipynb) - the
  quickstart above, cell by cell, with the constellation, spectrum and
  equalizer plots;
- [`carrier_phase_analysis`](https://github.com/lokgar/commkit/blob/main/examples/carrier_phase_analysis.ipynb) - drift,
  linewidth and Allan deviation of a recovered carrier phase;
- [`laser_linewidth_dsh`](https://github.com/lokgar/commkit/blob/main/examples/laser_linewidth_dsh.ipynb) and
  [`laser_linewidth_homodyne_iq`](https://github.com/lokgar/commkit/blob/main/examples/laser_linewidth_homodyne_iq.ipynb)
  - laser linewidth from delayed self-heterodyne and homodyne IQ captures;
- `measurement_laser_linewidth_*` - lean templates for real captures.

The notebooks are committed without outputs and run in CI.

## Development

```bash
git clone https://github.com/lokgar/commkit.git
cd commkit
uv sync --all-extras
uv run nbstripout --install      # once per clone: notebooks are committed without outputs

uv run pytest                    # CPU and GPU tests
uv run pytest --device=cpu       # what CI runs
uv run ruff check . && uv run mypy commkit/
```

Contributor and coding-agent guidance (architecture, API rules, numerics,
performance and test conventions) is in [AGENTS.md](https://github.com/lokgar/commkit/blob/main/AGENTS.md).

## License

[MIT](https://github.com/lokgar/commkit/blob/main/LICENSE).
