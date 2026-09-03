# CommKit

**High-performance digital communications research kit for Python.**

![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)
![Backends](https://img.shields.io/badge/backend-NumPy%20%7C%20CuPy%20%7C%20JAX-orange)
![License](https://img.shields.io/badge/license-MIT-green)
![CUDA](https://img.shields.io/badge/CUDA-13.x-76B900?logo=nvidia)

---

CommKit is a Python library for digital communications research that treats hardware as a first-class concern. A single `Signal` object carries IQ samples, physical metadata, and modulation context - with DSP operations dispatching automatically to NumPy, CuPy, or JAX based on data location.

---

## Why CommKit?

- **One object, complete context:** Sampling rate, symbol rate, modulation format, and pulse shape travel with the signal through the processing pipeline.
- **Backend-transparent DSP:** `dispatch()` resolves NumPy, CuPy, or SciPy modules at runtime. The same code executes seamlessly on CPU or GPU.
- **Functional pipelines:** DSP functions accept and return a `Signal` directly (`sig = fir_filter(sig, taps)`; `sig = resample(sig, sps_out=2)`), so pipelines compose without a monolithic `Signal` wrapper API - `sig.to("gpu")` moves data across backends, the rest is plain function composition.
- **JAX escape hatch:** Zero-copy DLPack export on GPU allows direct application of JAX transforms (gradients, `vmap`, `scan`) without leaving the research loop.

---

## Modules & Features

| Module | Key Capabilities & Features |
| --- | --- |
| [`commkit.core`](commkit/core) | `Signal` container (IQ samples + metadata), `SingleCarrierFrame`, `Preamble`, and symbol/frame factories (PAM, PSK, QAM, PS-QAM). |
| [`commkit.backend`](commkit/backend.py) | Hardware abstraction layer (`dispatch`, `to_device`, `to_jax`, `from_jax`), placement management, and backend execution (NumPy, CuPy, JAX). |
| [`commkit.mapping`](commkit/mapping) | Gray-coded constellations, symbol mapping, hard demapping, soft LLR computation (max-log and exact log-sum-exp via JAX JIT), and probabilistic shaping (Maxwell-Boltzmann). |
| [`commkit.filtering`](commkit/filtering.py) | Pulse shaping (RRC, RC, Gaussian, Smooth-Rectangle), FIR tap generators, IIR SOS filter design (Butterworth, Chebyshev I/II, elliptic, Bessel) and application, matched filtering, and Overlap-Save. |
| [`commkit.multirate`](commkit/multirate.py) | Fractional and integer sample rate conversion (`resample`, `decimate`, `upsample`, `decimate_to_symbol_rate`). |
| [`commkit.timing`](commkit/timing.py) | Preamble generation (Barker, Zadoff-Chu), cross-correlation timing delay estimation, and frame alignment. |
| [`commkit.frequency`](commkit/frequency.py) | Carrier frequency offset estimation (FOE via M-th power, Mengali-Morelli, pilot-symbol, bias-tone) and static/blockwise time-varying FOE correction. |
| [`commkit.recovery`](commkit/recovery) | Carrier phase recovery (CPR via Viterbi-Viterbi, BPS, DD-PLL, MAP Tikhonov-RTS, pilot-symbol/pilot-tone), cycle-slip detection/correction, and phase/channel-permutation ambiguity resolution. |
| [`commkit.equalization`](commkit/equalization) | Sequential (`lms`, `rls`, `cma`, `rde`) and frequency-domain block (`block_lms`, `block_cma`, `block_rde`) adaptive equalizers, `zf_equalizer`, butterfly MIMO topology support, and polarization-tone demultiplexing, with Numba JIT and JAX execution backends. |
| [`commkit.impairments`](commkit/impairments) | Channel impairments simulation: AWGN (with SPS correction), PMD (differential group delay, Jones matrix), phase noise, IQ imbalance (application + Löwdin/Gram-Schmidt compensation), and chromatic dispersion. |
| [`commkit.coding`](commkit/coding) | **Planned, not yet implemented** - scaffold-only placeholders reserving the layout for channel coding / FEC primitives (BCH, Convolutional, CRC, Galois field arithmetic, Hamming, Interleaving, LDPC, Polar, Rate matching, Reed-Solomon, Turbo codes). |
| [`commkit.metrics`](commkit/metrics.py) | System performance evaluation: EVM, SNR, BER, SER, and capacity metrics (GMI, MI) with PS-QAM support. |
| [`commkit.analysis`](commkit/analysis) | Laser phase and linewidth characterization: DSH, homodyne IQ, zero-phase drift detrending, AWGN-free lag-slope linewidth fit, Di Domenico $\beta$-separation line FWHM, and Allan deviation. |
| [`commkit.spectral`](commkit/spectral.py) | Welch PSD estimation, spectrograms, and frequency shifting with bin-quantized mixing. |
| [`commkit.smoothing`](commkit/smoothing.py) | Diagnostic/plotting-only smoothers (moving average, Savitzky-Golay, 2-D density smoothing) - not signal-chain filters; see `commkit.filtering` for those. |
| [`commkit.io`](commkit/io.py) | Signal persistence and disk serialization (`load_npz`, `save_npz`). |
| [`commkit.plotting`](commkit/plotting) | Visualization tools for constellations, eye diagrams, PSDs/spectrograms, time-domain signals, filter responses, equalizer convergence, and sync/CPR diagnostics (timing correlation, FOE spectra, carrier-phase trajectories). |
| [`commkit.helpers`](commkit/helpers.py) | General DSP helpers: random bit/symbol generators, array normalization, RMS calculation, dB<->linear conversion, and SI prefix formatting. |

---

## Installation & Usage

**Requires Python 3.12+** and [`uv`](https://github.com/astral-sh/uv).

### Core Installation (CPU)

```bash
# Using uv (Recommended)
uv pip install commkit

# Or with standard pip
pip install commkit
```

### GPU Support

To install with CUDA acceleration (includes JAX CUDA 13 and CuPy stacks):

```bash
# Using uv
uv pip install "commkit[gpu]"

# Or with standard pip
pip install "commkit[gpu]"
```

> [!NOTE]
> **WSL2 CUDA Configuration:**  
> When NVIDIA drivers and CUDA are properly installed on Windows, there is no need to install CUDA inside WSL2. However, to allow Python CUDA packages inside WSL2 to locate the bundled NVIDIA shared libraries, add the following line to your `~/.bashrc`:
>
> ```bash
> export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(echo $HOME/commkit/.venv/lib/python3.*/site-packages/nvidia/cu13/lib)
> ```
>
> *(Assumes `commkit` is cloned in `$HOME/commkit`. If located elsewhere, replace `$HOME/commkit` with `<path-to-repo>`. `python3.*/` matches any Python version automatically).*

### Notebook Support

To run the example notebooks and enable the rich HTML `Signal.print_info()` table (falls back to plain text without it):

```bash
# Using uv
uv pip install "commkit[notebook]"

# Or with standard pip
pip install "commkit[notebook]"
```

Extras can be combined, e.g. `commkit[gpu,notebook]`, or install everything at once with `commkit[full]`.

### Development Installation

```bash
git clone https://github.com/lokgar/commkit.git
cd commkit

# Sync core environment
uv sync

# Sync environment with all extras (including GPU and notebook packages)
uv sync --all-extras
```

---

## Examples & Notebooks

For complete usage scripts and DSP examples, explore the [`examples/`](examples) directory.

---

## Running Tests

```bash
# CPU test suite
uv run pytest --device=cpu

# GPU test suite (requires CuPy + CUDA)
uv run pytest --device=gpu

# Run all test suites
uv run pytest --device=all
```

---

## License

This project is licensed under the [MIT License](LICENSE).
