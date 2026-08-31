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
- **Method chaining:** High-level fluent API (`sig.to("gpu").fir_filter(taps).resample(sps_out=2)...`) for rapid prototyping.
- **JAX escape hatch:** Zero-copy DLPack export on GPU allows direct application of JAX transforms (gradients, `vmap`, `scan`) without leaving the research loop.

---

## Modules & Features

| Module | Key Capabilities & Features |
| --- | --- |
| [`commkit.core`](commkit/core) | `Signal` container (IQ samples + metadata), `SingleCarrierFrame`, `Preamble`, and symbol/frame factories (PAM, PSK, QAM). |
| [`commkit.backend`](commkit/backend.py) | Hardware abstraction layer (`dispatch`, `to_device`, `to_jax`, `from_jax`), placement management, and backend execution (NumPy, CuPy, JAX). |
| [`commkit.mapping`](commkit/mapping) | Gray-coded constellations, symbol mapping, hard demapping, soft LLR computation (max-log and exact log-sum-exp via JAX JIT), and GMI. |
| [`commkit.filtering`](commkit/filtering.py) | Pulse shaping (RRC, RC, Gaussian, Smooth-Rectangle), tap generators, matched filtering, Overlap-Save, and polyphase multirate filtering. |
| [`commkit.multirate`](commkit/multirate.py) | Fractional and integer sample rate conversion (`resample`, `decimate`, `upsample`, `decimate_to_symbol_rate`). |
| [`commkit.timing`](commkit/timing.py) | Preamble generation (Barker, Zadoff-Chu), cross-correlation timing delay estimation, and frame alignment. |
| [`commkit.frequency`](commkit/frequency.py) | Carrier frequency offset estimation (FOE via M-th power, Mengali-Morelli, Jacobsen) and phase-locked FOE correction. |
| [`commkit.recovery`](commkit/recovery) | Carrier phase recovery (CPR via Viterbi-Viterbi, BPS, DD-PLL) and cycle-slip detection/correction. |
| [`commkit.equalization`](commkit/equalization) | Adaptive equalizers (`lms`, `rls`, `cma`, `rde`, `zf`), butterfly MIMO topology support, with Numba JIT and JAX execution backends. |
| [`commkit.impairments`](commkit/impairments) | Channel impairments simulation: AWGN (with SPS correction), PMD (differential group delay, Jones matrix), phase noise, IQ imbalance, and chromatic dispersion. |
| [`commkit.coding`](commkit/coding) | Channel coding and FEC primitives (BCH, Convolutional, CRC, Galois field arithmetic, Hamming, Interleaving, LDPC, Polar, Rate matching, Reed-Solomon, Turbo codes). |
| [`commkit.metrics`](commkit/metrics.py) | System performance evaluation: EVM (%, dB), data-aided SNR, and BER estimation. |
| [`commkit.analysis`](commkit/analysis) | Laser phase and linewidth characterization: DSH, homodyne IQ, zero-phase drift detrending, AWGN-free lag-slope linewidth fit, Di Domenico $\beta$-separation line FWHM, and Allan deviation. |
| [`commkit.spectral`](commkit/spectral.py) | Welch PSD estimation and frequency shifting with bin-quantized mixing. |
| [`commkit.io`](commkit/io.py) | Signal persistence and disk serialization (`load_npz`, `save_npz`). |
| [`commkit.plotting`](commkit/plotting) | Visualization tools for constellations, eye diagrams, PSDs, time-domain signals, filter responses, and equalizer convergence. |
| [`commkit.helpers`](commkit/helpers.py) | General DSP helpers: random bit/symbol generators, array normalization, RMS calculation, and SI prefix formatting. |

---

## Installation & Usage

**Requires Python 3.12+** and [`uv`](https://github.com/astral-sh/uv).

### Core Installation (CPU)

```bash
# Using uv (Recommended)
uv pip install git+https://github.com/lokgar/commkit.git

# Or with standard pip
pip install git+https://github.com/lokgar/commkit.git
```

### GPU Support

To install with CUDA acceleration (includes JAX CUDA 13 and CuPy stacks):

```bash
# Using uv
uv pip install "commkit[gpu] @ git+https://github.com/lokgar/commkit.git"

# Or with standard pip
pip install "commkit[gpu] @ git+https://github.com/lokgar/commkit.git"
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

### Development Installation

```bash
git clone https://github.com/lokgar/commkit.git
cd commkit

# Sync core environment
uv sync

# Sync environment with all extras (including GPU packages)
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
