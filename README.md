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
- **The device follows the data.** There are no `backend=` or `device=`
  arguments; move data explicitly with `sig.to("gpu")`.
- There is no pipeline object, receiver class or configuration file: the
  orchestration stays in your script.

## Quickstart

```python
import commkit as ck
from commkit import RRC, Constellation
from commkit.recovery import BPS, CycleSlip

tx = ck.generate(Constellation.qam(16), num_symbols=2**16, symbol_rate=32e9,
                 sps=2, pulse=RRC(rolloff=0.1), rng=1)
rx = ck.impairments.apply_phase_noise(tx, linewidth=100e3, rng=2)
rx = ck.impairments.apply_awgn(rx, esn0_db=18, rng=3).to("gpu")  # explicit move
rx = ck.filtering.matched_filter(rx)                              # pulse from rx

cpr = BPS(test_phases=64, cycle_slip=CycleSlip(history=100))
res = ck.equalization.lms(
    rx, num_taps=21, step_size=1e-3,
    training_symbols=rx.reference.symbols[..., :2000], cpr=cpr,
)
y = res.signal                    # 1-SPS Signal, reference aligned
print(f"EVM {ck.metrics.evm(y, num_skip_symbols=2000):.1f} %, "
      f"BER {ck.metrics.ber(y, num_skip_symbols=2000):.1e}")
# EVM 12.9 %, BER 2.6e-04

# The next record continues from the converged taps and CPR state.
res2 = ck.equalization.lms(rx, num_taps=21, step_size=1e-3, cpr=cpr, state=res.state)
```

Without CuPy, drop `.to("gpu")`: the same code runs on the CPU.

---

## Modules

| Module | Contents |
| --- | --- |
| [`commkit.core`](commkit/core) | `Signal`, `Reference`, `generate`, `SingleCarrierFrame` and `Preamble` (pilots, guard intervals, MIMO streams), `extract_payload`. |
| [`commkit.mapping`](commkit/mapping) | `Constellation` (QAM, PSK, PAM, arbitrary points; Gray labels; probabilistic shaping with `.shaped()`), bit mapping, hard demapping, max-log and exact LLRs. |
| [`commkit.filtering`](commkit/filtering.py) | Pulses (`RRC`, `RC`, `Gaussian`, `Rect`, `SmoothRect`), FIR and IIR design (Butterworth, Chebyshev I/II, elliptic, Bessel), `fir_filter`, `iir_filter`, `matched_filter`, overlap-save, chromatic-dispersion compensation. |
| [`commkit.multirate`](commkit/multirate.py) | `resample` (fractional), `decimate`, `upsample`, `decimate_to_symbol_rate`. |
| [`commkit.spectral`](commkit/spectral.py) | Welch PSD, spectrograms, frequency shifting, pilot tones. |
| [`commkit.impairments`](commkit/impairments) | AWGN, phase noise, IQ imbalance (with Löwdin and Gram-Schmidt correction), chromatic dispersion, PMD and polarization mixing. Nonlinear channel models: **planned, not implemented**. |
| [`commkit.timing`](commkit/timing.py) | Barker and Zadoff-Chu sequences, `estimate_timing` / `correct_timing`, fractional delay estimation and correction. |
| [`commkit.frequency`](commkit/frequency.py) | `estimate_frequency_offset` / `correct_frequency_offset` with `MthPower`, `MengaliMorelli`, `PilotSymbols` and `BiasTone`; static and blockwise. |
| [`commkit.recovery`](commkit/recovery) | `estimate_carrier_phase` / `correct_carrier_phase` with `ViterbiViterbi`, `BPS`, `PLL`, `Tikhonov`, `DataAided`, `PilotAided`, `PilotTone(s)`; cycle-slip correction; `resolve_phase_ambiguity` and `resolve_channel_permutation`. |
| [`commkit.equalization`](commkit/equalization) | Sequential `lms`, `rls`, `cma`, `rde` (Numba) and frequency-domain `block_lms`, `block_cma`, `block_rde` (CuPy, CUDA graphs), butterfly MIMO, inline carrier recovery (`cpr=PLL()` / `BPS()`), continuation with `state=`, `zf_equalizer`, polarization-tone demultiplexing. |
| [`commkit.metrics`](commkit/metrics.py) | `evm`, `snr`, `ber`, `ser`, `gmi`, `mi`, including shaped constellations; host values per channel. |
| [`commkit.analysis`](commkit/analysis) | Laser and carrier-phase characterization: `estimate_linewidth` (increment slope, β-separation, delayed self-heterodyne FM-PSD / increment / Lorentzian), FM-noise PSDs, drift separation, Allan deviation. |
| [`commkit.math`](commkit/math.py) | `rms`, `normalize`, dB conversions. |
| [`commkit.smoothing`](commkit/smoothing.py) | Display and estimation smoothers (moving average, Savitzky-Golay, 2-D density). Signal-chain filters are in `filtering`. |
| [`commkit.io`](commkit/io.py) | `save_npz` / `load_npz` for Signals. |
| [`commkit.plotting`](commkit/plotting) | Constellations, eye diagrams, spectra, filter responses, equalizer convergence, and synchronization and laser diagnostics that draw the estimates. Imported on first use. |
| [`commkit.coding`](commkit/coding) | Channel coding and FEC: **planned, not implemented**. |

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

The [`examples/`](examples) directory holds runnable scripts (also usable as
notebooks): laser linewidth from delayed self-heterodyne and homodyne IQ
captures, and carrier-phase analysis of a recovered signal.

## Development

```bash
git clone https://github.com/lokgar/commkit.git
cd commkit
uv sync --all-extras

uv run pytest                    # CPU and GPU tests
uv run pytest --device=cpu       # what CI runs
uv run ruff check . && uv run mypy commkit/
```

Contributor and coding-agent guidance (architecture, API rules, numerics,
performance and test conventions) is in [AGENTS.md](AGENTS.md).

## License

[MIT](LICENSE).
