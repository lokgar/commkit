# CommKit examples

A course in notebooks, from the `Signal` object to a complete coherent
transceiver and laser measurements.  Each notebook covers one topic, builds
on the ones before it, and links to the next.  Read them in order the first
time; later, jump to the topic you need.

Install the `notebook` extra and start Jupyter in this folder:

```bash
uv sync --extra notebook
uv run jupyter lab examples
```

The notebooks run on the GPU when CuPy works and on the CPU otherwise.  They
are committed without outputs, and `tests/test_examples.py` runs every one.

## I. Signals and DSP fundamentals

| # | notebook | topics |
|---|---|---|
| 00 | [Getting started](00-getting-started.ipynb) | `Signal` (facts, description, reference), `generate`, devices, functions that return new data, files, logging |
| 01 | [Constellations and bit mapping](01-constellations-and-bit-mapping.ipynb) | QAM, PSK, PAM; Gray labels; mapping and hard decisions; rotational symmetry; probabilistic shaping |
| 02 | [Spectral analysis](02-spectral-analysis.ipynb) | Welch PSD (resolution and variance), absolute levels, frequency shifts, pilot tones, spectrograms |
| 03 | [Digital filters](03-digital-filters.ipynb) | FIR window design, overlap-save, IIR families and group delay, causal and zero-phase filtering, smoothing |
| 04 | [Pulse shaping and matched filtering](04-pulse-shaping-and-matched-filtering.ipynb) | pulses, the Nyquist criterion, eye diagrams, the matched filter, roll-off and timing, pulse length |
| 05 | [Multirate processing and resampling](05-multirate-and-resampling.ipynb) | upsampling, decimation and aliasing, the symbol-rate slicer, rational resampling, periodic AWG records |

## II. Channel and performance

| # | notebook | topics |
|---|---|---|
| 06 | [AWGN and performance metrics](06-awgn-and-performance-metrics.ipynb) | Es/N0 at any oversampling, SNR and EVM, BER against theory, LLRs, GMI and MI against capacity |
| 07 | [Laser phase noise](07-laser-phase-noise.ipynb) | the Wiener model, the Lorentzian line, `Δν·T`, a shared LO, flicker noise and the FM-noise PSD |
| 08 | [Front-end impairments](08-front-end-impairments.ipynb) | IQ imbalance and its correction, DC offsets, electrical bandwidth, carrier frequency offset |

## III. Receiver algorithms

| # | notebook | topics |
|---|---|---|
| 09 | [Frame synchronization and timing](09-frame-synchronization-and-timing.ipynb) | Barker and Zadoff-Chu preambles, frames, `estimate_timing` / `correct_timing`, payload extraction |
| 10 | [Frequency-offset estimation](10-frequency-offset-estimation.ipynb) | M-th power and Mengali-Morelli, lock range, accuracy, pilot symbols, pilot tones, drift tracking |
| 11 | [Blind carrier-phase recovery](11-blind-carrier-phase-recovery.ipynb) | Viterbi-Viterbi, BPS, block size, Tikhonov, the 90° ambiguity, cycle slips, joint polarizations |
| 12 | [Pilot-aided carrier recovery](12-pilot-aided-carrier-recovery.ipynb) | pilot symbols, data-aided rotation, the decision-directed PLL, pilot tones |
| 13 | [Linear channel equalization](13-linear-channel-equalization.ipynb) | ISI channels, zero forcing, MMSE, channel sounding |
| 14 | [Adaptive equalization](14-adaptive-equalization.ipynb) | LMS training and decisions, step size, RLS, block LMS, inline CPR, continuation with `state=` |
| 15 | [Blind equalization](15-blind-equalization.ipynb) | CMA and RDE, CMA pre-convergence handed over to RDE |
| 16 | [Polarization multiplexing](16-polarization-multiplexing.ipynb) | Jones matrices, the 2x2 butterfly, PMD, SOP drift, channel permutation |

## IV. Optical fiber systems

| # | notebook | topics |
|---|---|---|
| 17 | [Chromatic dispersion](17-chromatic-dispersion.ipynb) | the dispersion spread, the all-pass response, compensation, residual dispersion |
| 18 | [Nonlinear fiber and backpropagation](18-nonlinear-fiber-and-backpropagation.ipynb) | the Kerr effect, the optimum launch power, digital backpropagation |
| 19 | [End-to-end coherent transceiver](19-coherent-transceiver-end-to-end.ipynb) | every impairment and the full receiver DSP chain in one link, with its SNR budget |

## V. Laser and carrier-phase characterization

| # | notebook | topics |
|---|---|---|
| 20 | [Carrier-phase characterization](20-carrier-phase-characterization.ipynb) | data-aided phase, drift and phase noise, linewidth estimators, Allan deviation |
| 21 | [Self-heterodyne linewidth](21-laser-linewidth-self-heterodyne.ipynb) | the delayed self-heterodyne setup and its three linewidth estimators |
| 22 | [Short delays, flicker and fading](22-laser-linewidth-short-delay-and-flicker.ipynb) | the coherent regime, flicker FM noise, Allan deviation, polarization fading |
| 23 | [Self-homodyne IQ linewidth](23-laser-linewidth-self-homodyne-iq.ipynb) | the AOM-free setup with a 90° hybrid, front-end calibration |
| 24 | [Template: self-heterodyne](24-measurement-template-self-heterodyne.ipynb) | a lean notebook for real captures |
| 25 | [Template: self-homodyne IQ](25-measurement-template-self-homodyne-iq.ipynb) | a lean notebook for real captures |
