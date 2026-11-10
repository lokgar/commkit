# CommKit examples

A course in notebooks, from the `Signal` object to a complete coherent transceiver and laser measurements. Each notebook covers one topic, builds on the ones before it, and links to the next. Read them in order the first time; later, jump to the topic you need.

Install the `notebook` extra and start Jupyter in this folder:

```bash
uv sync --extra notebook
uv run jupyter lab examples
```

The notebooks run on the GPU when CuPy works and on the CPU otherwise. They are committed without outputs, and `tests/test_examples.py` runs every one.

## I. Signals and DSP fundamentals

| # | notebook | topics |
| --- | --- | --- |
| 00 | [Getting started](00-getting-started.ipynb) | `Signal` (facts, description, reference), `generate`, devices, functions that return new data, files, logging |
| 01 | [Constellations and bit mapping](01-constellations-and-bit-mapping.ipynb) | QAM, PSK, PAM; Gray labels; mapping and hard decisions; rotational symmetry; probabilistic shaping |
| 02 | [Spectral analysis](02-spectral-analysis.ipynb) | Welch PSD (resolution and variance), absolute levels, frequency shifts, pilot tones, spectrograms |
| 03 | [Digital filters](03-digital-filters.ipynb) | FIR window design, overlap-save, IIR families and group delay, causal and zero-phase filtering, smoothing |
| 04 | [Pulse shaping and matched filtering](04-pulse-shaping-and-matched-filtering.ipynb) | pulses, the Nyquist criterion, eye diagrams, the matched filter, roll-off and timing, pulse length |
| 05 | [Multirate processing and resampling](05-multirate-and-resampling.ipynb) | upsampling, decimation and aliasing, the symbol-rate slicer, rational resampling, periodic AWG records |

## II. Channel and performance

| # | notebook | topics |
| --- | --- | --- |
| 06 | [AWGN and performance metrics](06-awgn-and-performance-metrics.ipynb) | Es/N0 at any oversampling, Es/N0, SNR and noise variance for real and complex samples, SNR and EVM, BER against theory, LLRs and their consistency, GMI and MI against capacity |
| 07 | [Channel coding](07-channel-coding.ipynb) | `encode` / `decode`, coded `generate`, post-FEC BER and FER, hard and soft decoding, error-rate curves and the decoding outcome, Reed-Solomon, LDPC, NGMI as a threshold, EXIT charts, polar, convolutional and turbo codes against Eb/N0, block length |
| 08 | [Probabilistic shaping and coded modulation](08-probabilistic-shaping-and-coded-modulation.ipynb) | the CCDM, shaped `generate`, probabilistic amplitude shaping with LDPC, the shaping gain at the decoder |
| 09 | [Laser phase noise](09-laser-phase-noise.ipynb) | the Wiener model, the Lorentzian line, `Δν·T`, a shared LO, flicker noise and the FM-noise PSD |
| 10 | [Front-end impairments](10-front-end-impairments.ipynb) | IQ imbalance and its correction, DC offsets, electrical bandwidth, carrier frequency offset |

## III. Receiver algorithms

| # | notebook | topics |
| --- | --- | --- |
| 11 | [Frame synchronization and timing](11-frame-synchronization-and-timing.ipynb) | Barker and Zadoff-Chu preambles, frames, `estimate_timing` / `correct_timing`, payload extraction |
| 12 | [Sampling-clock recovery](12-sampling-clock-recovery.ipynb) | the clock offset and symbol drift, the Gardner detector and its power form, tracking and resampling, a known offset |
| 13 | [Frequency-offset estimation](13-frequency-offset-estimation.ipynb) | M-th power and Mengali-Morelli, lock range, accuracy, pilot symbols, pilot tones, drift tracking |
| 14 | [Blind carrier-phase recovery](14-blind-carrier-phase-recovery.ipynb) | Viterbi-Viterbi, BPS, block size, Tikhonov, the 90° ambiguity, cycle slips, joint polarizations |
| 15 | [Pilot-aided carrier recovery](15-pilot-aided-carrier-recovery.ipynb) | pilot symbols, data-aided rotation, the decision-directed PLL, pilot tones |
| 16 | [Linear channel equalization](16-linear-channel-equalization.ipynb) | ISI channels, zero forcing, MMSE, channel sounding |
| 17 | [Adaptive equalization](17-adaptive-equalization.ipynb) | LMS training and decisions, step size, RLS, block LMS, inline CPR, continuation with `state=` |
| 18 | [Blind equalization](18-blind-equalization.ipynb) | CMA and RDE, CMA pre-convergence handed over to RDE |
| 19 | [Polarization multiplexing](19-polarization-multiplexing.ipynb) | Jones matrices, the 2x2 butterfly, PMD, SOP drift, channel permutation |

## IV. Optical fiber systems

| # | notebook | topics |
| --- | --- | --- |
| 20 | [Chromatic dispersion](20-chromatic-dispersion.ipynb) | the dispersion spread, the all-pass response, compensation of captures and simulated records, blind estimation, the compensating FIR and its length, residual dispersion |
| 21 | [Nonlinear fiber and backpropagation](21-nonlinear-fiber-and-backpropagation.ipynb) | the Kerr effect, the optimum launch power, digital backpropagation |
| 22 | [FEC in optical standards](22-fec-in-optical-standards.ipynb) | KP4 and IEEE 802.3dj, G.709 RS and the staircase code, the 400ZR C-FEC, PON RS and LDPC, their thresholds against overhead |
| 23 | [End-to-end coherent transceiver](23-coherent-transceiver-end-to-end.ipynb) | every impairment and the full receiver DSP chain in one link, an LDPC-coded payload, the SNR budget |

## V. Laser and carrier-phase characterization

| # | notebook | topics |
| --- | --- | --- |
| 24 | [Carrier-phase characterization](24-carrier-phase-characterization.ipynb) | data-aided phase, drift and phase noise, linewidth estimators, Allan deviation |
| 25 | [Self-heterodyne linewidth](25-laser-linewidth-self-heterodyne.ipynb) | the delayed self-heterodyne setup and its three linewidth estimators |
| 26 | [Short delays, flicker and fading](26-laser-linewidth-short-delay-and-flicker.ipynb) | the coherent regime, flicker FM noise, Allan deviation, polarization fading |
| 27 | [Self-homodyne IQ linewidth](27-laser-linewidth-self-homodyne-iq.ipynb) | the AOM-free setup with a 90° hybrid, front-end calibration |
| 28 | [Template: self-heterodyne](28-measurement-template-self-heterodyne.ipynb) | a lean notebook for real captures |
| 29 | [Template: self-homodyne IQ](29-measurement-template-self-homodyne-iq.ipynb) | a lean notebook for real captures |
