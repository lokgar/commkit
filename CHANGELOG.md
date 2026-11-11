# Changelog

Notable changes to CommKit, newest first. Performance figures are from the reference machine (RTX 4070 Ti, Ryzen 7 7800X3D, WSL2) unless stated otherwise, as a ratio of new to old time (below 1 is faster).

## [Unreleased]

### Changes that may need code updates

- Factories take their secondary parameters by keyword: `LDPCCode.dvbs2(rate, frame=...)`, `LDPCCode.quasi_cyclic(base, lifting=...)`, `CRC.nr(name, info_bits=...)` and `Interleaver.block(rows=..., columns=...)`. Add the keywords to existing calls.
- `SingleCarrierFrame` needs a payload length: give `payload_num_symbols=` or, with a `payload_code`, `payload_blocks=`. It no longer defaults to 1000 symbols.
- `SingleCarrierFrame(pilot_seed=)` is `pilot_rng=`, with the semantics of `payload_rng` (an integer or a Generator, kept as the integer seed). Files saved with `pilot_seed` still load.
- `decoder_exit_curve`, `demapper_exit_curve` and `ldpc_exit_curves` take `num_points=` instead of `points=`.
- `demapper_exit_curve(noise_var=)` on a real constellation (PAM) is the variance of the real noise, as for `compute_llr`; it was the variance per complex symbol, twice the real noise. Halve `noise_var` in existing PAM calls to keep their curves, or pass the new `esn0_db=`.
- `resample(sps_out=)` raises when `sps_out / sps_in` has no fraction with terms up to 1000; such ratios designed filters of up to millions of taps. Give `up` and `down` explicitly.
- `zadoff_chu_sequence` raises on a root that shares a factor with the length, and the per-channel roots of a MIMO ZC `Preamble` skip such roots: on a composite length they gave full-height autocorrelation sidelobes. Preambles of prime length are unchanged; a MIMO preamble of length 64 now uses the roots 1, 3, 5, 7 instead of 1, 2, 3, 4.
- `estimate_iq_imbalance`, `apply_iq_imbalance` and `correct_iq_imbalance` raise `ValueError` on real samples instead of returning NaN, the input unchanged or a NumPy error.
- `SingleCarrierFrame.get_structure_map` returns a `FrameStructure` with the fields `pilots`, `payload`, `preamble` and `guard` (read-only, `None` where the map leaves a segment out) instead of a dict: `layout["pilots"]` becomes `layout.pilots`.
- `ser` and `evm` raise when the reference symbols are not points of the decision constellation (for `evm`, when one is known: the argument or the Signal's), for example after remapping the samples without the reference; they reported a SER of 0.75 and an EVM of 138 % for unipolar PAM-4 shifted to bipolar levels. A reference rescaled to its measured power still passes.
- The adaptive equalizers raise when `training_symbols` is longer than the record; the excess was dropped with a WARNING.
- Logging: WARNINGs that only restated the call's own parameters are DEBUG (equalizer `sps` and short `num_taps`, `rls` at `sps > 1`, small `BPS`, Viterbi-Viterbi and Tikhonov blocks on QAM, M-th power above symmetry 4, a narrow `PilotTone` band, real input to `plot_constellation`); the snapping of `shift_frequency` and `add_pilot_tone` is in their INFO line. Metrics, `estimate_timing`, `correct_timing`, `rls`, `ConstantPhase` and `resolve_phase_ambiguity` log one INFO line per call, the channels in a list. Filters that matched those messages need updating.

### Added

- `demapper_exit_curve(esn0_db=)` as an alternative to `noise_var`.
- `impairments.apply_frequency_offset`: a carrier frequency offset at any value (not snapped to the FFT grid like `shift_frequency`), per channel if needed, the exact inverse of `correct_frequency_offset`.
- A WARNING when the decisions of `lms`, `block_lms` or `rls` collapse onto a few points (blind unipolar PAM-4 below about 8 dB Es/N0 decides every symbol as the lowest level).

### Performance

- `iir_filter` and the per-channel `PLL` run their channels in parallel on the CPU.
- `apply_sampling_clock_offset`, `apply_sampling_jitter` and the resampling of `correct_sampling_clock_offset` interpolate through a Numba kernel on the CPU, parallel over the output: 0.07 and 0.11 of the former time (clock offset and jitter, 2 x 2^20 complex64 samples). The Gardner loop's interpolator steps its window by a rotation instead of a cosine per tap. `correct_sampling_clock_offset` with `Gardner` takes 0.21 of the former time on the CPU and 0.62 on GPU input (2^18 symbols at 2 sps).
- Turbo decoding stops a block once its two decoders take the same hard decision on every information bit, and `DecodeResult.iterations` reports the iterations run. LTE K = 6144 at Eb/N0 = 1 dB takes 4.1 iterations of the 8 allowed, K = 128 at 3 dB 1.7; 32 blocks of K = 1024 at Es/N0 = 0 dB (the benchmark) decode in 0.56 of the former time (max-log) and 0.59 (log-MAP), and in the waterfall (K = 6144, 0.5 dB) in 0.97. The FER is unchanged from K = 40 to 6144 (2^15 frames at K = 128 and 2 dB: 121 failures against 115).
- `correct_fiber_propagation` with `method="blocks"` backpropagates its blocks in batches of 2^16 samples per polarization on the CPU, which stay in cache: 0.64 of the former time (2 x 2^20 samples, 4 x 80 km, 10 steps per span), 0.85 at 2^23 samples. GPU input is unchanged.

### Fixed

- CMA on a constellation that is not at unit power and has an explicit `pmf` used the Godard radius of the unit-power set (1.32 instead of 5.28 for 16-QAM at power 4).
- `evm(blind=True)` divides the error by the power of the decided points: 16-QAM at power 4 read 20 % where the data-aided EVM is 10 %.
- `correct_polarization` with a tracked `JonesTrack` keeps complex128 input in double precision; it computed in complex64 and cast back.
- Gardner clock recovery emits `PrecisionWarning` on double input, as the loop computes in complex64.
- `apply_sampling_clock_offset` and `correct_sampling_clock_offset` keep their -119 dB interpolation accuracy at any record length. They formed the sampling instants in float32, whose rounding grew with the accumulated drift: -84 dB at the end of 2^20 complex64 samples at 2000 ppm, -68 dB at 2^23 samples and 1000 ppm. Complex128 input to the Gardner resampling also had its fractions rounded to float32.
- Gardner clock recovery calibrates its detector gain in blocks short against the clock drift and starts its integrator at the measured offset. A single fit over 2048 symbols scaled the gain by `sinc(2048 ε)`, so the loop ran at twice the requested bandwidth at 300 ppm and slipped from about 450 ppm (classical detector, rolloff 0.1); the gain now stays within a few percent of its zero-offset value up to `max_offset_ppm`, and offsets of 900 ppm are measured within 1 ppm by both detectors.
- `correct_fiber_propagation` with `method="blocks"` bounds its working memory, as documented: it propagated every block of the record at once, with an index array as large as the record. Peak host memory on 2 x 2^23 complex64 samples: 2.5 times the input (320 MiB), against 5.2 times before and 4 times for `"whole_record"`.
- `BetaSeparation` reads whether a bin lies above the β-line from its fitted model, and from the bin itself only where the bin exceeds the model beyond the fit's spur screen (drift, a modulation line). Thresholding the noisy Welch bins kept upward fluctuations near the crossover: on white FM with 70 bins below it the mean read 1.03 of the linewidth, now 0.99, and the spread of one record halves (0.04 of the linewidth at 7 bins, from 0.12). A WARNING fires below 10 bins, where it fired below about 3; the docstring states the measured spread.
- `apply_fiber_propagation` and `correct_fiber_propagation` on complex64 GPU input form each stage's dispersion phase in float64, reduced modulo 2π, before the fused kernels take it in float32; the kernels scaled a float32 phase of 1 km, whose rounding repeated in every span. Linear propagation over 20 x 80 km at 1 THz against complex128: -101 dB, from -60 dB (the CPU's complex64: -107 dB). At 64 GS/s the change is below 1 dB.
- `MengaliMorelli` keeps its autocorrelation in the data's precision; a float64 lag array promoted it to complex128.
- `plot_frequency_offset_blockwise_result` and two frequency axes label their unit with an SI prefix chosen from the data, instead of hand-scaled kHz or plain Hz.
- Error messages of `dispersion_compensation_taps`, the dispersion estimators, `PLL`, `BPS` and `Tikhonov` name the function or class.
- Docstrings: `Gaussian(fwhm=)` is the amplitude FWHM; the loop bandwidth of `Gardner` and `PLL` is nominal (the second-order loop's noise bandwidth is `1.25·B·T`); a frame's Es/N0 under `apply_awgn` is the frame average, not the payload's.
- `import commkit` no longer exposes `Any` and `importlib`.
- `dsh_phase`, `estimate_linewidth` with the DSH methods and `dsh_beat` keep single-precision input in single precision, widening only the phase to float64 before the unwrap; they computed the beat in complex128. Complex64 and complex128 runs agree to below 1e-6 rad per sample.
- `plot_zf_equalizer_response` plots the equalizer `zf_equalizer` applies; for MIMO it showed a scalar inverse per path instead of the matrix inverse, and the combined panel is now the matrix product `W·H`.

## [3.7.1] - 2026-11-10

Gardner clock recovery no longer runs away when its loop is too wide: the integrator is bounded to plausible clock offsets, and slips are reported. The notebooks on clock recovery, direct and coherent detection and the ADC now introduce their topics with figures.

### Added

- `timing.Gardner(max_offset_ppm=1000.0)`: the largest clock offset the loop tracks. The loop's integrator is held within it.
- A WARNING when Gardner clock recovery slips whole symbols (the timing moves by more than 0.75 symbol within `8 / (B·T)` symbols), after which output symbols no longer match the transmitted ones in order. A second WARNING when the estimated offset sits at `max_offset_ppm`.

### Performance

- Gardner clock recovery checks its trajectory for slips after the loop: 1.01 to 1.04 of the former time (paired runs, 2^17 symbols).

### Fixed

- A Gardner loop too wide for its detector's noise, such as the classical detector at `normalized_bandwidth=1e-2` on a 0.1-rolloff signal, no longer runs away in frequency without notice. Once it slipped, nothing pulled its integrator back, and it reached tens of thousands of ppm on a 50 ppm clock, with thousands of extra symbols in the corrected output. The bounded loop keeps returning to lock and reports the slips.

### Development

- Notebooks 12 (sampling-clock recovery), 22 (direct and self-coherent detection, now also coherent detection and its LO gain) and the skew and ADC sections of 10 explain their topics with figures: the drifting eye, how the Gardner detector tells early from late, loop bandwidth and cycle slips, PAM-4 eyes and sensitivity, thermal against shot noise, laser RIN, balanced photodiodes, quantization and jitter.

## [3.7.0] - 2026-11-10

The receiver's analog front end and the detector, at absolute power. The ADC clock, its jitter and quantizer, IQ skew and the photodiode join the impairments, with Gardner clock recovery as the inverse of the clock; coherent and square-law detection turn received and LO powers into the noise they give. Closed-form AWGN error rates and information rates sit next to the measured ones. Blind decision-directed equalizers no longer lock falsely on unipolar PAM and dense QAM, because their input scaling hands the slicer the right gain.

### Changes that may need code updates

- The adaptive equalizers (`lms`, `rls`, `cma`, `rde`, `block_lms`, `block_cma`, `block_rde`, `apply_taps(normalize=True)`) divide each input channel by its RMS instead of `rms * sqrt(sps)`. The old scale applied the transmit convention `E|x|^2 = 1/sps` at the slicer and handed it a gain of `1/sqrt(sps)` (0.71 at 2 sps), below the decision margins of unipolar PAM-4 (0.83) and 64-QAM (0.86), where blind decision-directed adaptation settled at a shrunk false solution (BER 0.25 and 0.20). After a matched filter the symbol instants now reach the slicer within `beta/4` of unit power at any timing phase, and blind adaptation reaches the trained BER. Consequences:
  - an explicit `step_size` of the sequential equalizers acts `sps` times stronger (the regressor power is 1): halve it at 2 sps for the former behaviour;
  - the block equalizers' default steps are divided by `sps`, so the defaults behave as before; explicit block steps act `sps` times stronger, and their stability ceilings read `step_size * block_size * sps`;
  - `EqualizerResult.input_norm_factor` and `EqualizerState.input_norm_factor` are the plain per-channel RMS; the docstring gives the exact rebuild of `y_hat` from the raw samples, the taps and this factor.
- `commkit.timing` is a package (`synchronization`, `clock_recovery`). Import from `commkit.timing` and nothing changes.
- The example notebooks are renumbered: the new 12 (sampling-clock recovery) and 22 (direct and self-coherent detection) move the former 12-28 to 13-30.

### Added

- `timing.estimate_sampling_clock_offset` and `timing.correct_sampling_clock_offset` with `timing.Gardner(normalized_bandwidth=, proportional_gain=, integral_gain=, joint_channels=, method="classical" | "power")`: Gardner clock recovery in one Numba loop. The offset in ppm comes from the slope of the tracked timing trajectory (`SamplingClockOffsetEstimate`); the correction resamples along the strobes, or by a constant factor from an earlier capture's estimate. An open-loop fit of the S-curve sets the gains from the loop bandwidth and starts the loop at the right phase; the power detector keeps its gain at small rolloffs. Measured within 0.06 ppm on 2^17 symbols at 20 dB; after recovery the SNR is the AWGN's plus the jitter's ISI, as predicted.
- `impairments.apply_sampling_clock_offset(offset_ppm=)`: an ADC clock off its nominal rate, by band-limited resampling (48-tap windowed sinc, error below -119 dB up to 0.4 cycles per sample).
- `impairments.apply_sampling_jitter(jitter=)` and `impairments.apply_quantization(num_bits=, full_scale=, enob=)`: sampling at jittered instants (SNR `-20 log10(2 pi f sigma)` of a tone) and a mid-rise quantizer with clipping and the converter's lumped noise.
- `apply_iq_imbalance(skew=)`: a time skew of Q against I, the frequency-dependent part of IQ imbalance (image `tan^2(pi f tau)`), which the memoryless corrections cannot remove.
- `metrics.awgn_ser`, `awgn_ber`, `awgn_mi`, `awgn_gmi(constellation, *, esn0_db)`: closed-form performance over AWGN in the conventions of the measured metrics. Error rates are exact for one-dimensional and separable constellations (PAM bipolar and unipolar, square QAM including shaped, BPSK, QPSK) and union bounds otherwise; MI and GMI by Gauss-Hermite quadrature, converged to 3e-7 bit.
- `impairments.apply_coherent_detection(received_power_dbm=, lo_power_dbm=, responsivity=, thermal_noise_density=, cmrr_db=, lo_rin_db_hz=, balanced=)`: the shot and thermal noise of a phase-diversity receiver from physical parameters (shot-noise limit `Es/N0 = eta n_p`), the common-mode leak of the signal's own power and of the LO's RIN through a finite CMRR, and single-ended detection.
- `impairments.apply_photodetection(received_power_dbm=, ...)`: the square law in the input's power units with signal-dependent shot noise, thermal noise, dark current, saturation and `sum_channels`; real output, and a Signal becomes digital baseband. `impairments.apply_intensity_noise(rin_db_hz=)`: laser RIN as a real amplitude factor.
- A WARNING when the power of an equalizer's decided symbols differs from the decision constellation's by more than 1.5 dB: a false lock, or a constellation that does not match the signal.
- Notebooks: 12 sampling-clock recovery, 22 direct and self-coherent detection; notebook 10 covers IQ skew and the ADC, notebook 06 uses the closed forms.

### Fixed

- `upsample`, `decimate` and `resample` promoted float32 to float64 (complex64 to complex128) on the GPU, because `cupyx` designs its polyphase filter in float64; the GPU now applies the CPU's filter in the data's precision.
- Equalizer docstrings: `weights_history` row `n` of the sequential equalizers holds the taps after the update at symbol `n`; an explicit `constellation` applies to the call only and the returned Signal keeps the input's description; real input is equalized as complex64. `Constellation.pam` states the unipolar energy penalty `2(2M - 1)/(M + 1)` (3.01 dB for OOK, 4.47 dB for PAM-4).

### Development

- `AGENTS.md`: the absolute-power rule (samples are dimensionless, a Signal holds no power, physical functions take `<where>_power_dbm` and return samples in the input's units), the `_ppm` and `_db_hz` suffixes, text-only commits skip mypy and pytest.
- Tests: a GN-model oracle for the split-step solver (first-order NLI with the coherent span factor, within 0.5 dB and the `P^3` law), the scaling chain of every equalizer rebuilt from raw samples, an IM-DD receiver chain with blind LMS on unipolar PAM-4, a plain-Python oracle for the Gardner loop.
- `benchmarks/bench_gardner.py` (no baseline recorded yet).

## [3.6.1] - 2026-11-05

### Added

- `plot_frequency_noise_psd` shades the white-FM plateau: the fit bins where the fitted model stays within 10 % of the floor `Δν/π`, where white FM dominates and pins the linewidth down. It is a visual check only; `Δν` still comes from the model fitted over every bin in `used`, not from an average over the plateau.

## [3.6.0] - 2026-11-05

Chromatic dispersion for real captures. The compensation takes the link's accumulated dispersion and works as a receiver does, an FIR filter (or overlap-save blocks for backpropagation) in bounded memory, with the exact whole-record path as an option that is valid on captures too; three estimators measure an unknown dispersion, blind or from known symbols. The laser-characterization chain and its notebooks were corrected wherever a plot or a number did not mean what it said, method names that were generic now say what they do, and plot axes carry their SI prefix in the label.

### Changes that may need code updates

- `correct_chromatic_dispersion` takes the link's accumulated dispersion, `dispersion_ps_nm=` (D·L in ps/nm), instead of `dispersion_ps_nm_km=` and `fiber_length_km=`: a receiver knows the total, and only the product enters the physics. Write `dispersion_ps_nm=17 * 80` for 80 km of standard fiber. `dispersion_compensation_taps` takes the same argument.
- `correct_chromatic_dispersion(method=)` and `correct_fiber_propagation(method=)` choose between two implementations of the same inverse. The default is the receiver's form, in bounded memory on a record of any length: CD by the FIR of `dispersion_compensation_taps` applied by overlap-save (`bandwidth=` and `num_taps=` choose the filter), DBP in overlapping blocks guarded by the link's dispersion memory, all blocks in one batched pass. `method="whole_record"` is the former path, the exact response over the whole record treated as one period; pass it to reproduce earlier results on simulated records. On a capture both are right except within half the dispersion memory of its ends, which no method can recover; `"whole_record"` is the more precise elsewhere, the default suits streams and studies of the filter.
- Method objects whose names were generic, or named one thing two ways, say what they do: `recovery.ConstantPhase` replaces `DataAided` (every known-symbol method is data-aided; this one fits one constant phase per channel), `recovery.PilotInterpolation` replaces `PilotAided` and `frequency.PilotSlope` replaces `PilotSymbols` (the same pilots, interpolated as a phase or fitted as a slope), `frequency.SpectralPeak` replaces `BiasTone` (the tone `recovery.PilotTone` names too; its mechanism is the spectral peak, the counterpart of `MthPower`), `analysis.DshFmFit` replaces `DshFmPsd` (one word from the result class `DshFmNoisePsd`), and `coding.StageDecoders` replaces `Stages`. Rename at the call; nothing else changes.
- `commkit.impairments.channel` is `commkit.impairments.fiber_channel`, leaving room for other channel models. Import from `commkit.impairments` and nothing changes.
- `filter_response` and `plot_filter_response` evaluate complex coefficients over the whole band `[-fs/2, fs/2)` by default, since their response differs at negative frequencies; real coefficients keep the grid from DC to Nyquist. Pass `frequencies=` for the old grid.
- `BetaSeparation`'s white-FM floor (`linewidth_floor`) is a Whittle fit of white FM, flicker FM and the detection noise's `f²` tail, the model of `DshFmFit`, instead of the median of an auto-detected plateau, and the estimate now carries `flicker`, `angle_noise_var` and `fm_psd_model`. Without a fence the fit runs from the plateau's lower edge to Nyquist. The default segment is the largest power of two within an eighth of the record (it was capped at 4096 samples), so the β-area can resolve the white-FM crossover; pass `nperseg=` for the old segment.
- Plot axes in SI units carry one SI prefix in the axis label (`Frequency [GHz]`, `Time [ns]`) with plain numbers on the ticks, chosen anew at every draw, so zooming or new limits rescale both; ticks no longer repeat the prefix (`10 G`). Log axes keep a prefix per tick. Limits stay in SI units (`xlim=(0, 2e9)`).
- `plot_allan_deviation` draws no `τ^(-1/2)` guide by default (`reference_slopes=False`): anchored at the first point, it was meaningless wherever that point is not white-FM limited.

### Added

- `filtering.dispersion_compensation_taps` designs the FIR filter of `correct_chromatic_dispersion`: the least-squares fit of the inverse fiber response over a band (the signal's occupied bandwidth, or the whole sampled band), exact through Fresnel integrals. Its default length is 1.25 times the group-delay spread over the band, which reaches about -45 dB of residual against the frequency-domain compensation; fitting only the occupied band needs about a third of the taps of a fit over the whole band. Notebook 19 shows the taps, their response and the SNR against the length.
- `filtering.estimate_chromatic_dispersion` with `ClockToneScan` measures a capture's accumulated dispersion blind, before frequency and carrier recovery: Godard's clock-tone spectral correlation, every candidate scanned by one inverse FFT, the estimate the centroid of the peak (a standard deviation of 5 ps/nm at rolloff 0.1 and 1 ps/nm at 0.35 on 2^17 samples at 15 dB; within 62 ps/nm with 10 ps of DGD and a 1 GHz offset). It returns a `ChromaticDispersionEstimate` with the scan, which `correct_chromatic_dispersion(x, estimate)` takes as its `how` (or `correct_chromatic_dispersion(x, ClockToneScan())` to estimate first), and whose value designs the taps to study with `dispersion_compensation_taps(dispersion_ps_nm=estimate.value, ...)`.
- `estimate_chromatic_dispersion` has two more methods: `FourthMomentScan`, blind and without an excess band (Nyquist pulses), which scores each candidate compensation by the normalized fourth moment of the field (within 11 ps/nm at rolloff 0 and 0.1; about a second on the CPU, 0.05 s on the GPU), and `QuadraticPhaseFit`, data-aided from the Signal's reference or `symbols=`, which fits the quadratic phase of the channel response measured against the known symbols (within 12 ps/nm to ±34000 ps/nm at 6 dB Es/N0, with phase noise and PMD) and returns the measured group delay. `ClockToneScan` warns when the Signal's pulse has a rolloff below 0.05.

### Fixed

- `BetaSeparation`'s floor read the foot of the AWGN `f²` tail as plateau: 11 % high on the 32 GBd, 1 MHz, 20 dB trajectory of notebook 23 at the default segment (now 0.999 MHz). When the Welch bins cannot resolve the white-FM crossover with the β-line, a warning says so and names the segment needed; the β-area used to return 0 silently.
- `DshFmPsd` on a real (single-photodiode) beat reports `fm_psd` and `fm_psd_model` as NaN beyond the beat's support, `min(f_shift, f_s/2 - f_shift)`, where the bins read several times low; `valid` marks it. The fit already excluded them.
- `plot_frequency_noise_psd` draws the fitted model through the same log-binned reduction as a dense PSD (a fringe-resolving DSH model swung at every notch and was drawn as a filled band), keeps the β-line from setting the vertical range, and no longer breaks log-binned curves at empty low-frequency cells. `plot_dsh_beat_psd` zooms on the line, twice the deep width on each side. `plot_carrier_phase_decomposition` labels the total phase next to the drift.
- The laser-characterization notebooks (23 to 28) report what their plots show: beat-spectrum limits in Hz (they were MHz on a Hz axis, an empty plot), the β-area resolved and explained as the observed width next to the intrinsic floor, the Allan deviation of the unfiltered frequency with its white-PM, white-FM and wander regions, the flicker corner on segments that resolve it, the record-to-record scatter of `DshIncrement` and of the Lorentzian widths, and verdicts in the measurement templates computed from the numbers instead of printed unconditionally.
- The fiber channel functions document that they treat their record as one period (circular), which the receiver inverts exactly with `method="whole_record"`.

### Development

- The precision audit caught the clock-tone scan's frequency-shift phasor built in complex128, which promoted the complex64 scan; it is built in the data's precision.
- The transfer audit lists the dispersion estimators' host-side fits and the data-aided fit's known symbols; CD compensation and block DBP add no transfers.
- `AGENTS.md` states the plotting convention for SI units (the base unit in the label, the prefix chosen at draw time) and names the fiber-channel test files.

## [3.5.0] - 2026-11-05

One name per concept, finished. A review of the API and the module layout renamed the arguments and result fields that broke the library's naming rules, split or merged the modules that broke its size rules, and fixed the two audit tools that could pass without checking anything. Numerical results do not change.

### Changes that may need code updates

Every rename fails loudly with `TypeError` (or `AttributeError` for a field); none changes a result.

- `generate` takes `num_symbols` by keyword only, like every other parameter after the constellation: write `generate(c, num_symbols=N, ...)` instead of `generate(c, N, ...)`.
- `metrics.fer(info_bits=)` replaces `block_length=`, the code's information length as everywhere else in the coding API.
- Result fields of the laser analyses have descriptive names:
  - `LinewidthEstimate`: `frequencies`, `fm_psd`, `fm_psd_model`, `increment_var`, `delta_phi_var` and `peak_frequency` replace `f`, `S_f`, `S_f_model`, `var`, `dphi_var` and `f_peak`;
  - `DshFmNoisePsd`: `frequencies` and `fm_psd` replace `f` and `S_f`;
  - `FrequencyDrift`: `peak_to_peak` replaces `pp`.
- `analysis.carrier_phase_trajectory(symbols, ...)` and `equalization.estimate_transfer_function(samples, ...)` rename their first argument (was `y_eq` and `received`); positional calls are unaffected.
- `recovery.PLL(proportional_gain=, integral_gain=)` replaces `mu=` and `beta=`, the raw loop gains. `normalized_bandwidth` is unchanged.
- `plotting.plot_error_rate(x_axis="ebn0")` replaces `ebn0=True`, as `plot_psd(x_axis=)` chooses its axis; another value raises `ValueError`.
- `plot_frequency_drift(reference_amplitude=)` and `plot_carrier_phase_characterization(reference_amplitude=)` replace `amp_ref=`.
- `plot_constellation` draws no grid.
- Modules moved; import from the packages as documented and nothing changes. `commkit.impairments.channel` is one module (it was the subpackage `channel.linear` / `channel.nonlinear`); `commkit.plotting.sync` is split into `plotting.timing`, `plotting.frequency` and `plotting.recovery`, and `plot_carrier_phase_decomposition` lives in `plotting.analysis`; `Preamble` is defined in `commkit.core.preamble`; `commkit.equalization.sequential` and `commkit.equalization.blind` are private (`_sequential`, `_block`).

### Fixed

- `ErrorRateCurve` and `ExitCurve` document their fields under Attributes, as every other result class does.

### Development

- The transfer audit (`tests/plugins/transfer_trace.py`) failed on 3.4.0 for three legitimate copies it did not list (`SingleCarrierFrame.to_signal`, `extract_payload`, `save_npz`); they are documented now. Both audit plugins found commkit by the path fragment `commkit/commkit/`, so in a checkout of another name (a git worktree) they recorded nothing and passed; they now match the installed package's directory.
- The Numba kernel getters of the equalizers sit in the modules that call them, as in `coding` and `mapping`, and `_kernels_numba.py` is gone.
- AGENTS.md holds only the rules for changing the code; it gains glossary entries for output scaling (`normalize`, `correct_power`, `*_normalization`) and search windows (`search_range`, `search_band`), and the rule that result classes document their fields under Attributes. The README carries the new logo.

## [3.4.0] - 2026-11-03

Error-rate curves for every code, decoders that say what they checked, and one noise convention for real and complex samples.

### Changes that may need code updates

- Real-valued samples (PAM, intensity modulation) now follow one noise convention, so their LLRs and information rates are right. Complex samples are unaffected.
  - `apply_awgn(esn0_db=)` adds half the noise variance to real samples: the in-phase half of the channel's noise. A constellation's error rates at an Es/N0 no longer depend on whether its samples are real or complex, and BPSK has `BER = Q(sqrt(2 Es/N0))`. For a power ratio in the samples, as measurement code means it, use the new `apply_awgn(snr_db=)`; it adds what `esn0_db` used to add to real samples at one sample per symbol.
  - `compute_llr`, `Constellation.llr`, `mi`, `gmi` and `ngmi` use the likelihood of real noise for real symbols. With the `noise_var` that `estimate_noise_variance` measures, real-symbol LLRs used to be twice too confident.
- `DecodeResult` has a new field, `checked`: whether `valid` comes from a check that can fail. Code that builds a `DecodeResult` itself must pass it. For decoders without a check, `valid` stays `True` everywhere, as before.

### Added

- `coding.error_rate_curve(code, constellation=, esn0_db=, ...)` measures a code's BER, FER and pre-FEC BER over AWGN by Monte Carlo. Each point stops after `min_block_errors` block errors or `max_blocks` blocks. It returns an `ErrorRateCurve` of counts, which you can also build from your own simulation loop.
- `plotting.plot_error_rate(curves)`: BER and FER against Es/N0 or Eb/N0, with 95 % intervals on the FER and the pre-FEC BER for reference, plus a panel splitting failures into detected and undetected ones.
- Soft output from more decoders: `TurboIterations` (a-posteriori LLRs of the last iteration), `MaximumLikelihood` (max-log) and `Concatenated` (the inner decoder's LLRs). `decoder_exit_curve` now covers turbo and maximum-likelihood decoding.
- `Chase(..., soft=False)` and `MaximumLikelihood(soft=False)` skip the soft output when only the decisions matter (`llrs` is `None`, the bits are the same).
- `compute_llr`, `mi`, `gmi` and `ngmi` take `esn0_db=` instead of `noise_var`, and convert it for real or complex symbols.
- `plot_llr_distribution(res, sig)` takes a `DecodeResult` and the transmitted Signal or Reference, as `ber` does.
- The example notebooks draw with the library's plot functions wherever one exists, and show spectra, constellations and eye diagrams before and after each block. Notebook 06 explains Es/N0, SNR and noise variance for real and complex samples; notebook 07 measures error-rate curves and the decoding outcome, compares the polar, convolutional and turbo codes against Eb/N0 and the turbo code at three block sizes; notebook 21 charts the optical standards' thresholds against their overhead.

### Performance

- `MaximumLikelihood` searches the codewords in one Numba pass on the CPU. For 16384 Hamming(16, 11) words (paired A/B against 3.3.0, 7 runs), decisions alone (`soft=False`) take 0.62x the time on the CPU and 1.0x on the GPU; with the new soft output, 1.55x and 4.3x.
- `Chase(soft=False)`: 0.44x the time of the default on the CPU and 0.74x on the GPU (400ZR Hamming(128, 119), 4 test positions, 4096 words).
- Turbo decoding is unchanged (1.00), now with its soft output.

### Development

- `metrics` is a package (`quality`, `error_rates`, `information`); every name is still imported from `commkit.metrics`.
- The transfer audit lists the payload moves of the EXIT and error-rate curves, which it flagged as undocumented; the kernel-coverage job runs the maximum-likelihood tests.

## [3.3.0] - 2026-11-03

Numerical filter responses and plots built for comparisons. `filter_response`, `impulse_response` and `step_response` evaluate any FIR, SOS or `(b, a)` filter on its device; the multi-panel diagnostics draw one selected panel, and every plot can overlay several results on one axis. The example notebooks explain what each result shows.

### Changes that may need code updates

- `plot_filter_response` raises `ValueError` when `ax` holds other than four axes for the dashboard (or one for a selected `quantity=`), as the other multi-panel plots do; it used to log a warning and open a new figure. Pass the right number of axes, or none.

### Added

- `filter_response` (returning a `FilterResponse` of frequencies, complex response and group delay), `impulse_response` and `step_response` compute the responses of FIR taps, second-order sections or a `(b, a)` pair on the coefficients' device. The group delay of SOS filters is summed section by section, so high orders keep their accuracy; real IIR time responses run through `iir_filter`, in double-precision state whatever the coefficients' precision.
- Multi-panel diagnostics (`plot_filter_response`, `plot_equalizer_result`, `plot_zf_equalizer_response`, `plot_llr_distribution`, `plot_parity_check`, `plot_timing_correlation`, `plot_mm_autocorrelation`, `plot_frequency_offset_blockwise_result`, `plot_pilot_phase_estimate`, `plot_pilot_tone_phase_estimate`, `plot_pilot_tones_phase_estimate`) take `quantity=` to draw one panel on one axis.
- Comparisons on one axis: the single-panel plots and a selected `quantity=` take `label=`, `xlim=`, `ylim=` and `legend=` (`True` shows, `False` hides, `None` keeps the plot's default), and the plots with fits, guides or markers take `annotations=False` to draw only the measured curves of an overlay. `plot_eye_diagram(quantity="i" | "q")` draws one component; `plot_carrier_phase_characterization` accepts existing axes.
- The example notebooks explain what each receiver result shows, their subplot grids scale the theme's figure size, and notebooks 13 and 16 overlay filter, smoothing, shaping and equalizer comparisons on one axis.

### Development

- The transfer audit documents the Tikhonov smoother's host round trip, which it flagged as undocumented.
- `bench_filter_response` times responses on the DFT grid and on an arbitrary grid at 31, 257 and 4097 taps.

## [3.2.1] - 2026-11-02

Fixes found by a rework of the test suite, which now checks what each algorithm achieves against the transmitted truth and theory. Five numerical defaults or models were wrong and change results (PLL cycle-slip repair, Tikhonov smoothing, `block_cma`'s step, `ols_fir_filter`'s alignment of even taps, `MthPower` near 0 Hz); no signature changes. Plots take their sizes from the theme.

### Changes that alter results

- **`Tikhonov` uses the actual variance of a Viterbi-Viterbi block phase** as its observation variance, computed from the constellation and `snr_db`: the noise through the M-th power plus the data self-noise of QAM. It used `1/(M²·SNR·N)`, too small by `M²/2` for PSK and by three to four orders of magnitude for QAM, so the smoother followed the raw block phases: on 16-QAM it now lowers the Viterbi-Viterbi error (static phase, 30 dB, 16-symbol blocks: 0.101 rad before and after smoothing, now 0.048 rad).
- **`PLL(cycle_slip=CycleSlip())` defaults to a history of `8/μ` symbols** (twice the loop memory, between 10 and 640), standalone and inline in `lms` / `rls`; it used 640 symbols, as the block methods do. A PLL trajectory is smoothed by the loop, and the long linear fit lagged the phase noise enough to declare false slips: 16-QAM, `2π·Δν·T = 2π·1e-4`, 16 dB, `B·T = 2e-2`, four records of 2·10⁴ symbols: 10 rotation changes against the transmitted symbols with the repair, 3 without; now 3 with it. The repair does not undo the loop's own slips, which drift over about `1/(B·T)` symbols; the `CycleSlip` docstring says so. An explicit `history` is unchanged.
- **`block_cma` defaults to `step_size=min(1e-3, 0.5 / block_size)`**, as `block_rde` does; it used 2e-4. On QPSK, where CMA and RDE are the same algorithm, the old default left the worst channel at 2.3 times the AWGN symbol error rate over the second half of 8·10⁴ symbols (dual polarization through rotation and DGD, Es/N0 10 dB; now 1.3 times) and at 16 times on 2·10⁴ symbols (now 2 times); on 16-QAM at 18 dB the SER of 2·10⁴-symbol records falls from 0.13 to 0.008. Above 2500-symbol blocks the new default is the smaller step, inside the stability ceiling that 2e-4 crossed (16-QAM, 4096-symbol blocks: SER 0.02 before, 0.004 now). Pass `step_size=2e-4` for the previous behavior.
- **`ols_fir_filter` centres even-length taps on tap `(L - 1) // 2`**, as `fir_filter` (SciPy's `mode="same"`) does and as its docstring promised; it used `L // 2`, one sample later. Odd-length taps and `causal=True` are unchanged. The docstrings of both now state the centre tap; they said `L // 2`.

- **Plots take line widths, marker sizes and fonts from the theme.** The plot functions no longer set their own line widths and marker sizes, so `apply_default_theme` and the user's rcParams control them; the theme's text is larger (16 pt, legends 14 pt, titles 18 pt) with longer ticks. SI-prefix tick labels show the prefix only (`10 k`, not `10 kHz`), as the axis labels already carry the unit.

### Fixed

- **`MthPower` estimates offsets near 0 Hz.** It zeroed bin 0 of the M-th power spectrum and, for constellations without a constant envelope, subtracted the mean of `x^M`, which for QAM is the tone itself: an offset with `|M Δf|` under half a bin was never found (QPSK, 4096 symbols at 1 MS/s: 0 Hz read as -82 kHz, 10 Hz as 30.5 Hz). Both steps are removed; at other offsets the estimates do not change, also with a DC offset of -20 dB in the input.
- Messages that report what the caller asked for log at INFO, not WARNING, so a WARNING from commkit always means a suspect result: the block equalizers' advice to move CPU data to the GPU, RLS's tail trim, `correct_timing(mode="slice")` shortening the record and the plotting theme's font fallback. The advice to use more taps for fractionally spaced equalization is no longer logged at `sps=1`, where it does not apply.
- `rls()`: `weights_history` has one row per returned symbol, so it matches `y_hat` after the tail trim (it kept the `num_taps // 2` trimmed rows). A run whose inverse correlation matrix loses positive definiteness (a forgetting factor far below 1) raises the documented `RuntimeError` instead of a bare `ZeroDivisionError` from the kernel, and the message now advises raising the forgetting factor toward 1, not lowering it.

### Development

- The test suite checks what the code achieves, not how it is written. Equalizers converge on RRC records through ISI, fractional delays, polarization rotation and DGD, scored against the transmitted symbols with only the legitimate rotation and permutation removed, and each "improves" test also runs a baseline that must fail; carrier recovery is scored against the true phase; estimators and metrics against Cramér–Rao bounds and AWGN theory; Chase and shortened-code decoding against brute force. New end-to-end receiver chains (coded single carrier, dual polarization, 80 km of fiber) each show that leaving out one correction breaks them. Monte Carlo tests state what their frame counts bound and gain `--run-slow` validation runs. A WARNING log from commkit fails the test. Tests that only repeated the contract registry's shape and device checks are removed.
- The library faults these tests found are fixed above: the PLL's cycle-slip repair adding slips, Tikhonov's noise model on QAM, the dead zone of `MthPower` around 0 Hz, the alignment of even-length taps (`ols_fir_filter` against `fir_filter`) and RLS's `weights_history` length.
- `benchutils.assert_converged` checks the symbol error rate of the worst channel (it compared bit errors before), and the 2048-tap block RDE benchmark uses a workload that converges.

## [3.2.0] - 2026-10-30

EXIT analysis and coding diagnostics. EXIT charts for any soft-in soft-out component, measured through the existing decoders or the demapper, or in closed form for LDPC codes with their threshold; plots of LLR distributions, parity-check matrices and EXIT charts; and INFO logging in the modules where it was missing. Nothing changes for existing code.

### Added

- **EXIT analysis** in `commkit.coding.exit`. `ExitCurve` is the transfer curve of one soft-in soft-out component and `exit_chart(first, second)` the `ExitChart` of two, with the decoding trajectory, whether the tunnel is open and where it closes. Curves come from `decoder_exit_curve` (measured through `decode` for any code whose decoder returns a-posteriori LLRs: LDPC, BCJR, Chase), `demapper_exit_curve` (measured, for iterative demapping on AWGN) and `ldpc_exit_curves` (the variable and check nodes of an LDPC code in closed form), with `ldpc_threshold`, the belief-propagation threshold in the Gaussian approximation (0.498 for the DVB-S2 rate-1/2 short code, the NGMI at which simulated QPSK frames start to decode). Measured curves run on the chosen device; the docstrings list the limitations (Gaussian a-priori LLRs, long interleavers, no curve for decoders without soft output such as turbo and polar, early stopping).
- **Coding plots**: `plot_llr_distribution(llrs, bits)` draws the LLR densities given the transmitted bit and their consistency, `log p(L | 0) / p(L | 1)` against `L`, which shows LLRs from a wrong `noise_var` or demapper model; `plot_parity_check(code)` draws the nonzero entries and node degrees of a parity-check matrix; `plot_exit_chart(chart)` draws an `ExitChart` and marks where a closed tunnel stalls.
- INFO logging where it was missing: `decode` reports the blocks whose own check failed and the iterations used, `estimate_iq_imbalance` its estimate, and `SingleCarrierFrame.to_signal`, `save_npz` and `load_npz` what they built, saved or loaded. Notebook 00 shows how to turn it on.
- Notebook 06 checks LLR consistency; notebook 07 predicts the LDPC threshold with an EXIT chart and charts iterative demapping with Gray and natural labels.

### Development

- AGENTS.md states the logging levels: DEBUG for what a call is about to do, INFO for one summary per public call, WARNING for a returned but suspect result; value objects do not log, and `warnings.warn` is kept for `PrecisionWarning`.

## [3.1.3] - 2026-10-30

Additive noise without a scaling knob. `apply_awgn` either measures the signal or takes an absolute noise variance, and the OSNR moves to its own function, `apply_ase`.

### Changes that may need code updates

- **`apply_awgn(signal_power=)` is removed.** Its uses each have a direct replacement:
  - silence or zero padding around the signal: nothing to pass, as Es is now measured as energy per symbol (below);
  - a fixed noise floor, noise referenced to the transmitter, or noise on a dark capture: `noise_var=`, the absolute noise variance per sample. For unit-energy symbols, `noise_var = 10 ** (-esn0_db / 10)` at any `sps`.
- **`apply_awgn(osnr_db=, reference_bandwidth=, noise_modes=)` is `apply_ase(osnr_db=, reference_bandwidth=, noise_modes=)`.** Same arguments and noise.

### Changes that alter results

- **`apply_awgn(esn0_db=)` measures Es as energy per symbol**: each channel's energy over the number of symbols the record carries (a Signal's frame length without a zero guard, else its reference's symbol count, else `N / sps`). Zero padding and silence around a frame no longer lower the noise level; without silence the noise is as before. `apply_ase` measures the signal power the same way.

### Added

- `apply_awgn(noise_var=)`, an absolute noise variance per sample, scalar or `(C,)`; exactly one of `esn0_db` and `noise_var` is given.
- `apply_awgn(per_channel=False)` gives every channel one noise level, from the mean symbol energy over channels, as identical receivers with one noise floor; the default gives each channel the requested Es/N0.
- Notebook 06 shows relative (`esn0_db`) against absolute (`noise_var`) noise.

## [3.1.2] - 2026-10-29

`apply_awgn` without hidden assumptions about what a channel is: Es/N0 holds per channel, and the OSNR counts the ASE of the modes the medium guides instead of assuming the two polarizations of a single-mode fiber.

### Changes that alter results

- **`apply_awgn(esn0_db=)` gives every channel the requested Es/N0.** The default noise level comes from each channel's own mean power; it used to come from the mean over all channels, so with unequal channel powers a stronger channel got a better Es/N0 than requested and a weaker one a worse. Equal powers give the same result as before. Pass a scalar `signal_power` for the old behavior, one absolute noise level shared by every channel.
- **`apply_awgn(osnr_db=)` counts the ASE of `noise_modes` modes, `max(C, 2)` by default.** One or two channels are unchanged (both polarizations of a single-mode fiber). More channels, such as the modes of a few-mode fiber or the cores of a multicore fiber, each count as a mode; they used to share the ASE of two polarizations, which put C/2 times too much noise on each.

### Added

- `apply_awgn(noise_modes=)` for a medium that guides modes without signal, e.g. `noise_modes=6` for four channels in a three-mode fiber.
- `apply_awgn(signal_power=)` accepts one power per channel, `(C,)`.

## [3.1.1] - 2026-10-29

API consistency. One name per concept and a unit in every size, defaults that model the physics, physical units where the rest of the API has them, and features that use what the Signal and the frame already carry: the optical carrier for dispersion, fiber and OSNR noise, the frame's pilots for the pilot-aided algorithms, the code of the reference for post-FEC metrics. Most changes are renames that fail loudly with `TypeError`; the four default changes and the speed of light alter numbers and are listed first.

### Changes that alter results

- **`iir_filter` is causal by default** (`zero_phase=False`): a DAC, driver or photodiode bandwidth has group delay and an unsquared magnitude response. The old default filtered forward and backward, squaring the response. Pass `zero_phase=True` for offline zero-phase filtering.
- **`correct_timing` zero-fills by default** (`mode="zero"`); `mode="circular"` wrapped the end of a capture onto its start, right only for a periodic record.
- **`SingleCarrierFrame` draws a fresh payload by default** (`payload_rng=None`), so a Monte Carlo loop over frames no longer repeats one payload. Pass `payload_rng=<int>` for a fixed payload; the frame keeps the seed it used, so copies, `replace()` and saved frames rebuild the same one. The pilots keep their fixed `pilot_seed`.
- **`SingleCarrierFrame.to_signal` and `Preamble.to_signal` require `symbol_rate`** and default to `sps=1`, as `generate` does; they used to default to 1 MBd at 4 samples per symbol.
- **One exact speed of light** (299 792 458 m/s): dispersion used 2.998e8, which moves chromatic-dispersion results by 7e-5 relative.

### Changes that may need code updates

- **Code sizes name their unit**: `CRC`, `ConvolutionalCode` and `TurboCode` (and `CRC.nr`, `CRC.ethernet`, `ConvolutionalCode.k7`, `TurboCode.lte`) take `info_bits` instead of `block_length`; `CCDM` takes `num_symbols`; `StaircaseCode`, `StaircaseCode.g709_2` and `Concatenated.cfec` take `blocks_per_frame` instead of `num_blocks` (which `generate` uses for codewords); `LDPCCode`'s mother length is `num_variables`; `PolarCode.nr(k=, n=)` is keyword-only. The Reed-Solomon `length` property (in code symbols) is gone: read `symbols` or `n`.
- **`num_streams` is `num_channels`** in `SingleCarrierFrame`, `Preamble` and `Signal`.
- **One name per concept**:
  - `noise_var` in `zf_equalizer` and `plot_zf_equalizer_response` (was `noise_variance`); `LinewidthEstimate.awgn_var`, a phase term in rad², is `angle_noise_var`.
  - `nfft` in `estimate_transfer_function` (was `n_fft`) and `ols_fir_filter` (was `fft_size`).
  - `regularization` in `estimate_transfer_function` (was `reg`) and `rls` (was `delta`).
  - `zf_equalizer(center_tap=)` (was `center`); `ols_fir_filter(causal=False)` replaces `center=True`.
  - `num_train_symbols` and `num_impulse_samples` in the equalizer plots (were `n_train`, `n_impulse`).
  - `HardIterations(max_iterations=)` (was `iterations`, though it stops early); `BCJR` and `TurboIterations` take `method="maxlog" | "exact"` like `compute_llr` (was `exact=False | True`).
  - `barker_sequence` and `zadoff_chu_sequence` take `num_symbols` like `Preamble` (was `length`).
  - `AllanDeviation.taus` matches `allan_deviation(taus=)` (was `tau_s`); `LinewidthEstimate.num_segments` (was `n_segments`).
- **`estimate_transfer_function(received, reference)`**: the data under test first, as everywhere else (was `reference, received`).
- **Physical units**: `apply_polarization_mixing(drift_rate=)` in rad/s with the sampling rate from the Signal (was `drift_rad_per_sample`); `Tikhonov(linewidth=)` in Hz like `apply_phase_noise`, the symbol period taken from its one-sample-per-symbol input (was `linewidth_symbol_periods`); `PLL(normalized_bandwidth=)` (was `bandwidth`), so a bandwidth in Hz cannot be passed by mistake.
- **One entry point per concept**: the pulse objects (`RRC`, `RC`, `Gaussian`, `Rect`, `SmoothRect`) are the only way to build pulse taps; `rrc_taps`, `rc_taps`, `gaussian_taps`, `rect_taps` and `smoothrect_taps` are private, as their defaults disagreed with the objects'. `demap_symbols_hard` is `demap_symbols`, the twin of `Constellation.demap`.
- **Polarization demultiplexing on pilot tones is an estimate / correct pair**: `estimate_polarization_tones(samples, tone_frequencies=, track_bandwidth=)` returns a `JonesTrack` (one unmixing matrix by default, or matrices tracked over time), and `correct_polarization(samples, track)` applies it. This replaces `demultiplex_polarization_tones_static` / `_dynamic`, whose return type changed with `return_matrix` and `apply`, and the public `apply_interpolated_matrix`.
- **The optical functions read the carrier**: `apply_chromatic_dispersion`, `correct_chromatic_dispersion`, `apply_fiber_propagation` and `correct_fiber_propagation` take `center_wavelength_nm` from an optical Signal's `center_frequency`; an explicit value must agree. Arrays and other Signals must pass it: the fiber functions no longer assume 1550 nm silently.
- **Pilots for the blind equalizers are one object**: `EqualizerPilots(values, mask, gain_db)` replaces the `pilot_ref`, `pilot_mask` and `pilot_gain_db` arguments of `cma`, `rde`, `block_cma` and `block_rde` (now `pilots=`) and the `build_pilot_ref` helper.

### Added

- **`Signal.center_wavelength_nm`**, the wavelength of an optical Signal's carrier.
- **`apply_awgn(osnr_db=, reference_bandwidth=12.5e9)`** adds ASE to an optical Signal at an OSNR, counted over both polarizations in the reference bandwidth, so `Es/N0 = 2 B_ref / (p R_s) · OSNR` whatever the oversampling.
- **`EqualizerPilots.from_frame`, `PilotAided.from_frame` and `PilotSymbols.from_frame`** take the pilot positions, symbols and gain from a `SingleCarrierFrame`.
- **Post-FEC metrics from a `DecodeResult`**: `ber(decoded, sig)` and `fer(decoded, sig)` compare with the reference's `info_bits`, and `fer` takes the block length from the code.
- **`estimate_noise_variance`**, the data-aided noise variance that `compute_llr`, `gmi` and `mi` take, without the round trip through `snr`.
- **`estimate_iq_imbalance`** measures the amplitude (dB) and phase (degrees) of an IQ imbalance blindly, per channel, as an `IQImbalanceEstimate`; `correct_iq_imbalance` also takes the estimate and inverts the model exactly.
- `SingleCarrierFrame.pilot_mask` and `payload_mask`, the symbol masks over the whole frame.

### Fixed

- `apply_awgn` rejects a non-positive `reference_bandwidth`.
- `zadoff_chu_sequence`'s length error names the argument.

## [3.1.0] - 2026-10-29

Review fixes and the completion of the coding release. Coded records are sized in codewords (`generate(num_blocks=)`, `SingleCarrierFrame(payload_blocks=)`) or padded to any length, the reference carries its code, `save_npz` stores everything a Signal holds (codes included), and `Signal.domain` is back. Reed-Solomon and BCH decoding run in per-codeword kernels (KP4 23 times faster on the GPU, 16 on the CPU), and the remaining decoders, LLRs and several DSP paths got faster. Correctness fixes cover bit validation in `encode`, staircase and CCDM validity, complex FIR taps, ZF precision and timing correction. Frame lengths in symbols are renamed `*_num_symbols`; see the first section.

### Changes that may need code updates

- **The `gpu` extra requires CuPy 14.2** (`cupy-cuda13x>=14.2`), the first release with sparse arrays (`cupyx.scipy.sparse.csr_array`), which the LDPC encoder now uses on the GPU.
- **Lengths in symbols are named `*_num_symbols` throughout the frame**, as in `generate` and `SingleCarrierFrame.num_symbols`: `SingleCarrierFrame(payload_len=, guard_len=, pilot_block_len=)` are now `payload_num_symbols`, `guard_num_symbols` and `pilot_block_num_symbols`, and `Preamble(length=)` is `Preamble(num_symbols=)`. Rename the arguments and attribute reads.
- **`encode` rejects bits other than 0 and 1** with `ValueError`. Before, linear codes silently encoded another payload (`[2, 0, 0, 0]` gave the all-zero codeword), a value of 256 wrapped to 0, and the turbo encoder indexed its trellis with the value.
- **`TurboCode` and `LDPCCode` reject non-integer parameters** with `TypeError` (`TurboCode(40.0)` was accepted; float LDPC indices were truncated into a different code).
- **`fir_filter` and `ols_fir_filter` raise for complex taps on real samples.** The imaginary part was dropped (the taps `[1j]` gave zeros). Pass complex samples for a complex output.
- **`correct_timing(mode="slice")` raises for offsets outside `[0, N)`**: a negative offset kept only the last samples. Use `mode="zero"` or `"circular"` for negative offsets.
- **`LDPCCode.parity_check()` returns a `scipy.sparse.csr_array`** instead of a `csr_matrix`, as SciPy plans to deprecate its matrix interface. `@` is unchanged, but `*` is now elementwise and indexing a row gives a 1-D array; use `@` for products, or `scipy.sparse.csr_matrix(code.parity_check())` for the old type. `LDPCCode.from_parity_check` accepts either.
- **`gray_code`, `gray_to_binary` return read-only arrays.** They are cached, so writing to one changed every later call's result; copy before modifying.

### Added

- **`Reference.code`**, the code that turned `info_bits` into `bits` (the FEC code, or the matcher `generate` picks for a shaped constellation), so `decode(llrs, code=sig.reference.code)` needs nothing else; `code_block` is taken from it. `generate` and `extract_payload` set it, and the Signal summary names it.
- **`save_npz` stores everything a Signal holds**: a frame's `payload_code` and a reference's `code` (a FEC code, distribution matcher or their composition) as their class and constructor fields, with their arrays (parity checks, information positions, ...) as `frame_payload_code.*` and `reference_code.*` entries, and the frame's pilot bits. Still loaded with `allow_pickle=False`; only the classes of `commkit.coding` are rebuilt. A frame with a payload code used to raise.
- **`Signal.domain`: `"DIG"` (default), `"RF"` or `"OPT"`**, where the samples live. An RF or optical Signal carries its `center_frequency` (a digital one keeps 0, and setting a carrier on it raises); the summary shows the domain and the carrier (with its wavelength for an optical signal), `save_npz`/`load_npz` round-trip it, and `plot_psd`/`plot_spectrogram` take it from the Signal, so an optical spectrum is drawn at its physical frequencies or wavelengths without arguments.
- **`generate(..., code=, num_blocks=)` sizes a coded record in code blocks** (codewords of a FEC code) instead of symbols, so `generate(QPSK, symbol_rate=..., code=code, num_blocks=100)` replaces `num_codewords * code.n // bits_per_symbol`. Give either `num_symbols` or `num_blocks`.
- **`pad=True` fits a coded record to any length**, such as an arbitrary waveform generator's record: `generate(..., num_symbols=N, code=code, pad=True)` carries as many whole codewords as fit, then random uncoded filler bits; `SingleCarrierFrame(payload_pad=True)` does the same for a frame's payload. Not with a distribution matcher, whose composition the filler would change.
- `Reference.num_blocks`, the whole code blocks per channel: decode them with `decode(llrs[..., : ref.num_blocks * code.n], code=code)`.
- **`SingleCarrierFrame(payload_blocks=)` sizes a coded payload in codewords** instead of `payload_num_symbols` (which it then sets); the codewords must fill whole symbols and pilot periods, and the error names the nearest valid counts. `payload_code` takes distribution matchers and PAS too, with the shaped payload constellation they match onto.
- `SingleCarrierFrame.num_symbols`, the frame length in symbols (preamble, body, guard): `to_signal` gives `num_symbols * sps` samples per stream. A `payload_num_symbols` that does not fill whole codewords now names the nearest valid lengths, pilot periods included.
- The Signal summary (`print(sig)`) shows the code block, the blocks per channel with any filler, and the information rate of a coded reference, and a frame's payload code.
- `generate`'s docstring explains coded payloads (blocks, sizing, decoding) and that a shaped constellation without a code goes through one CCDM block of the whole record, the ideal-matcher limit, with how to model a transmitter's shorter matcher.

### Performance

Paired A/B of alternating processes against the previous commit of each change:

- Reed-Solomon decoding: KP4, 2048 hard-decision codewords at BER 1e-3, CPU 410 to 26 ms (direct XOR reduction 0.80, Forney's formula at the error positions only 0.42, hard input converted in place 0.79, a per-codeword Numba decoder from the syndromes 0.27), GPU 41 to 1.8 ms (a CUDA kernel with one thread per codeword in place of Berlekamp-Massey's 450 launches, 0.080).
- BCH decoding through the same per-codeword decoders: Chase-II of BCH(1000, 970) 0.26 on the CPU and 0.22 on the GPU, staircase decoding 0.35 and 0.14.
- Chase decoding: 0.53 on the CPU, 0.30-0.51 on the GPU (no LLR round trip or re-encoding, no 4-D flip mask, larger GPU chunks).
- LDPC sum-product decoding on the CPU: DVB-S2 1/2, 64 codewords, 0.70 (`phi(|Q|)` kept from a row's first pass instead of computed again).
- Products of bit arrays modulo 2 (encoding, syndromes): encoding 2^17 OIF 400ZR codewords 0.85 on the CPU and 0.64 on the GPU (no int32 temporaries for the parity).
- CCDM matching: 8 blocks of 20000 PS-64-QAM symbols, matching 0.76 and dematching 0.84 (one division per point and symbol; the dematcher's interval start in 32-bit limbs instead of one byte per bit).
- Polar decoding: SC of N = 1024 0.32, SCL(8) 0.70, SCL(32) 0.68 (paths copied only when they change slot).
- Tail-biting Viterbi: K = 7, 0.035 (all start states scored in one pass); decision storage a quarter of before at every termination.
- Turbo decoding: LTE K = 1024, max-log 0.82, exact log-MAP 0.65.
- Cached code tables: interleaving 2^20 values with one `Interleaver` 0.19 on the CPU, 0.013 on the GPU (identity shortcut in the code objects' equality, which every cached lookup called).
- `mi`: one exponential per element, 256-QAM 0.82 on the CPU and 0.33 on the GPU, 16-QAM 0.52 on the GPU.
- LMS/RLS with inline carrier recovery: 2 x 50000 16-QAM symbols, PLL 0.95, BPS 0.97 (each symbol's phasor computed once).
- `Constellation.map` / `demap` on the GPU: about 0.65 per call for 64-QAM and 8-PSK, 4096 to 2^20 symbols (tables cached per device; the hash cached and an identity shortcut in the equality the cache calls).
- LLRs on the CPU (`compute_llr`, `Constellation.llr`): 0.07-0.26 for 4096 symbols (tables cached per constellation; 256-QAM's closed-form tables were most of the call), 0.84 for 2^20 (no copies of the real and imaginary parts).
- `correct_frequency_offset` with a static offset: 2 x 2^22 samples, 0.83 on the CPU and 0.69 on the GPU (one float64 ramp instead of three, and no host sync for an offset on the GPU).
- `plot_equalizer_result` of 2 x 10^6 symbols: 0.08 on both devices (curves smoothed and cut to their min/max envelope on the result's device; about 4000 points per line are transferred and drawn instead of every symbol).
- White-FM plateau detection of the linewidth estimators: 0.26 for 2^19 PSD bins, 0.91 for 4097 (each log-spaced cell read as a slice instead of a scan over all bins).
- `apply_pmd`: 2 x 2^22 samples, 0.91 on the CPU and 0.90 on the GPU (the second phasor is the conjugate of the first).
- `block_lms` with BPS: 2 x 100000 16-QAM symbols, block 256, 0.97 on the CPU and 0.86 on the GPU (the phasor from a real cos and sin).

### Fixed

- **A shaped `Constellation` rebuilt from its fields is equal to the original.** The constructor divided the pmf by its sum every time, which moved an already normalized pmf by an ulp, so a shaped constellation loaded with `load_npz` could compare unequal to the saved one. A pmf that sums to 1 within rounding is now kept as given.
- **Staircase `valid` reflected an earlier state of the frame**: a block corrected through the next one kept its failed flag, so `HardIterations(1)` could return the right payload with `valid=False`. Success is now checked on the final frame.
- **CCDM (and so PAS) `valid` accepted sequences no payload maps to**: a composition has more sequences than the 2^k payloads, and all were valid. Only the matcher's image is valid now.
- **`zf_equalizer` with a float64 channel estimate returned complex128** for complex64 samples; it now computes in the samples' precision.
- **`correct_timing(mode="zero")`** repeated each channel's first sample for a negative per-channel offset instead of zero-filling, and returned zeros for a scalar offset of 0.
- **`mi` and `ols_fir_filter` / `zf_equalizer` bound their working memory.** `mi`'s intermediates reached GiBs for many channels of high-order QAM (1 GiB of differences for two channels of 1024-QAM); they are now capped at 2^24 elements. Overlap-save transformed the whole record at once, against its docstring, about 8 times the input; it now runs in batches of blocks (3.5 times the input on long records; about 5% slower on the CPU for 2^25 samples).
- **A code object unpickled in another process missed its cached device tables**: it carried the hash of the first process (hashes of bytes differ between processes), so it compared equal to the same code but hashed differently. The cached hash is no longer pickled.
- `load_npz` closes its archive; `ber`, `fer` and `ser` transfer their error counts once.

### Development

- Benchmarks for polar, Viterbi, turbo, Chase and CCDM decoding in `bench_coding.py`; `ldpc_workload` is now `coded_workload`.
- Baseline `0008` (`v3_1`) is the new benchmark reference, the first with those benchmarks, best of ten passes. Against `0007` the geometric mean of the time ratios is 0.88 (0.92 without coding); no benchmark is slower beyond the GPU's run-to-run swing (the slowest, 1.08, on a 0.26 ms GPU call). KP4 decoding takes 1.8 ms on the GPU and 26 ms on the CPU, against 41 and 410 ms.
- The algebraic decoders' kernel (`algebraic`) joins the sanitizer pass of AGENTS §2, and their Numba decoder the `NUMBA_DISABLE_JIT` coverage job.

## [3.0.0] - 2026-10-28

The channel-coding release. `commkit.coding` grows from an empty scaffold into a full FEC toolbox: one `encode` / `decode` pair dispatched on code objects covers linear block, LDPC, BCH and Reed-Solomon, polar, convolutional and turbo codes, the FEC structures of the optical standards (KP4, the 802.3dj and 400ZR concatenated codes, the G.709.2 staircase code, PON LDPC), CRCs, interleavers, rate matching and probabilistic amplitude shaping. `generate` and frames produce coded payloads, the reference keeps the source bits next to the bits on the air, and three new notebooks teach it. Shaped `generate` now goes through a distribution matcher, and the example notebooks are renumbered; see the first section.

### Changes that may need code updates

- **Shaped `generate` goes through a distribution matcher.** A shaped constellation used to draw i.i.d. symbols from its pmf and take their labels as the "bits", so the bits described a stream that was never sent. Now random bits pass through `CCDM(constellation, block_length=num_symbols)` per channel: every symbol comes from bits, each record holds every point exactly as often as the pmf (quantized to `1/num_symbols`) says, and `reference.info_bits` holds the matcher's input. For the same seed the symbols differ from 2.x; GMI and MI agree within Monte Carlo error. A shaped record must be long enough to carry a bit (a few symbols). Frames inherit this, and `save_npz` stores a shaped frame's source bits.
- **`Reference` has two more fields**, `info_bits` and `code_block`. `bits` are the bits on the air; `info_bits` are the source bits before coding. Without coding `info_bits` is `bits` itself and `code_block` is `(1, 1)`, so existing code that reads `reference.bits` is unaffected. Code that builds a `Reference` positionally keeps working.
- **The placeholder modules of `commkit.coding` are gone.** 2.x shipped empty scaffold modules (`hamming`, `bch`, `reed_solomon`, `galois`) that held only docstrings. The codes now live in `linear` (Hamming and other linear block codes), `algebraic` (BCH and Reed-Solomon) and the private `_galois`; import the public names from `commkit.coding` itself (`from commkit.coding import BCHCode, ReedSolomonCode`), which is the supported path. `commkit.coding` is also reachable as `ck.coding` after `import commkit as ck`.
- **The example notebooks are renumbered** to fit the coding chapters: 07 Channel coding and 08 Probabilistic shaping and coded modulation join Part II, 21 FEC in optical standards joins Part IV. Old 07-18 are now 09-20, old 19 (the end-to-end transceiver) is 22, and old 20-25 (characterization and templates) are 23-28. Update bookmarks and links to the notebooks.

### Added

- **`commkit.coding`**, the channel-coding module: `encode(bits, code=...)` and `decode(llrs, code=..., method=...)`, dispatched on a code object, returning a `DecodeResult` (decoded bits, per-block success check and iterations).
  - `LinearBlockCode` from a generator or parity-check matrix, and `LinearBlockCode.hamming(r, extended=, length=)` for Hamming codes, including the inner codes of 400ZR, Hamming(128,119), and of IEEE 802.3dj, Hamming(128,120).
  - Decoders `Syndrome(max_weight=)` (coset leaders; `max_weight=1` makes an extended Hamming code correct one error and detect two) and `MaximumLikelihood()` (exhaustive, for small codes).
- **LDPC codes:** `LDPCCode.from_parity_check(H)`, `LDPCCode.quasi_cyclic(base, lifting)`, `LDPCCode.dvbs2(rate, frame)` (all DVB-S2 rates, normal and short frames) and `LDPCCode.pon()` (LDPC(17280, 14592) of IEEE 802.3ca and ITU-T G.9804.2). Decoders `MinSum(scale=, offset=)` and `SumProduct()`, layered, with early stopping per codeword, on the CPU (Numba) and the GPU (a CUDA kernel, about 10x the CPU on 64 DVB-S2 codewords). Encoding is systematic: a sparse product and a cumulative XOR for DVB-S2, a sparse product and a GEMM with the inverted parity part for PON.
- **Algebraic codes:** `ReedSolomonCode(symbols, info_symbols, m=)` with the presets `g709()` (RS(255,239)), `kr4()` (RS(528,514)), `kp4()` (RS(544,514)), `epon10g()` (RS(255,223)) and `xgspon()` (RS(248,216)), and binary `BCHCode(m, t, length=)`, decoded by `BerlekampMassey()` (Chien search, Forney) on both devices; a codeword with more than `t` errors is reported `valid=False` or decoded to a codeword within distance `t`. Encoding, syndromes and the locator evaluations are GEMMs over GF(2): KP4 decodes 2048 codewords in 0.45 s on the CPU and 89 ms on the GPU.
- **Soft decoding of block codes and optical FEC structures:** `Chase(test_positions=, hard=, beta=)`, Chase-II decoding with Pyndiah's soft output for linear block and BCH codes; `LinearBlockCode.oif400zr()`, the double-extended Hamming(128,119) of OIF 400ZR exactly as its clause 10.5 defines it; `StaircaseCode.g709_2(num_blocks)`, the 512 x 510 staircase code of ITU-T G.709.2 (its double-extended BCH(1022,990) component and column permutation as specified), decoded by `HardIterations()`; and `Concatenated(outer, inner, interleaver)` with the presets `cfec()` (400ZR C-FEC) and `ieee_802_3dj()` (KP4 + Hamming(128,120)), decoded by `Stages(inner=, outer=)`. Measured thresholds match the standards: the staircase code decodes at an input BER of 4.5e-3, the C-FEC at 1.2e-2.
- **Polar codes:** `PolarCode.nr(k, n, crc=)` with the information set of the 5G NR reliability sequence (TS 38.212 Table 5.3.1.2-1) and an optional NR CRC, or `PolarCode.from_frozen(mask)`; decoders `SC()` and `SCL(list_size)` (CRC-aided when the code has a CRC). Mother code only (no NR rate matching).
- **Convolutional and turbo codes:** `ConvolutionalCode(generators, constraint_length, block_length, termination=)` and `ConvolutionalCode.k7()` (133, 171), terminated or tail-biting, decoded by `Viterbi()` (maximum likelihood) or `BCJR(exact=)` (a-posteriori LLRs); `TurboCode.lte(K)` for the 188 LTE block sizes (QPP interleaver of TS 36.212 Table 5.1.3-3, trellis termination as clause 5.1.3.2.2), decoded by `TurboIterations(iterations, exact=)`.
- **Bit-level building blocks:** `CRC` as a code (`CRC.nr` for the 5G NR and LTE polynomials, `CRC.ethernet` for the IEEE 802.3 CRC-32, or any Rocksoft-model CRC), whose `decode` flags blocks that fail the check; `interleave` / `deinterleave` with an `Interleaver` (explicit, `Interleaver.block`, `Interleaver.random`); and the rate-matching codes `Punctured(code, positions)` and `Shortened(code, num_bits)`, decoded with the inner code's decoders.
- **Distribution matching:** `CCDM(constellation, block_length=)`, a constant-composition matcher (an exact 64-bit range coder, invertible for every input, blocks up to 2^30 points, 0.35 s for 10^6 points of 64-QAM on one core), and `PAS(code, constellation)`, probabilistic amplitude shaping with a systematic FEC code supplying the signs. Both are codes: `generate(..., code=PAS(LDPCCode.dvbs2("3/4"), Constellation.qam(64).shaped(entropy=5)))` builds a coded PS payload, and `decode` runs the FEC decoder and the dematcher.
- **Coded payloads:** `generate(..., code=)` and `SingleCarrierFrame(payload_code=)` encode random information bits before mapping. The reference keeps both bit streams.
- **`metrics.fer`**, the block error rate, and **`metrics.ngmi`**, the normalized GMI used as the soft-decision FEC threshold. Post-FEC BER is `ber(res.bits, reference.info_bits)`.
- **`save_npz` / `load_npz`** store a frame's payload source bits and code block (`frame_payload_info_bits`), so a shaped frame round-trips; a frame with a `payload_code` raises, as code objects are not serialized (store `extract_payload(signal)` instead).
- **Notebooks:** [07 · Channel coding](examples/07-channel-coding.ipynb) (codes as objects, hard and soft decoding, Reed-Solomon, LDPC waterfalls, NGMI as a threshold, polar, convolutional and turbo codes), [08 · Probabilistic shaping and coded modulation](examples/08-probabilistic-shaping-and-coded-modulation.ipynb) (the CCDM, the shaping gain, PAS with LDPC against uniform 16-QAM) and [21 · FEC in optical standards](examples/21-fec-in-optical-standards.ipynb) (the codes of IEEE 802.3, G.709, 400ZR and PON at their thresholds). The end-to-end transceiver (22) carries an LDPC-coded payload and reports post-FEC BER and FER.

### Fixed

- **`compute_llr` compiled its CPU kernels in every process.** They are declared with Numba's disk cache, but their kernels called helpers built in the same function, which Numba cannot cache, so the first call of every Python process spent about 3 s compiling. They now load from the cache (about 0.2 s, Numba import included); the results are unchanged.

### Development

- `benchmarks/bench_coding.py` times LDPC decoding (min-sum on both devices) and KP4 decoding; the `ldpc` kernel is in the sanitizer pass of AGENTS §2, and the coding tests of the Numba decoders and the CCDM run in the `NUMBA_DISABLE_JIT` coverage job.
- Baseline `0007` (`v3_0`) is the new benchmark reference, the first with the coding benchmarks: min-sum decoding of DVB-S2 codewords takes 16 ms on the GPU against 133 ms on the CPU, KP4 decoding 41 ms against 410 ms. The benchmarks without coding match `0006` (geometric mean of the ratios 0.97; the slowest moves reproduced within 10% in isolation, or were within the GPU's run-to-run swing on a 0.1 ms call).
- A session fixture keeps Numba's thread pool busy for half a second before the first benchmark: kernels now loaded from the cache reached the timed rounds while the OpenMP workers were still coming up, which made the first parallel benchmark of a file 4-8x slower.
- The Numba kernels of the coding module and of `compute_llr` live in private modules (`coding/_numba_loops.py`, `mapping/_numba_llr.py`), so Numba's disk cache keeps them; the coding decoders compiled again in every process too, costing up to 6 s on their first call.
- Every helper in the package has a docstring, with one-line summaries throughout.
- `backend._device_tables` caches small per-object tables (generators, interleavers, field tables) on each device, shared by the coding module and the earlier users; its upload is a documented site of the transfer audit.

## [2.3.0] - 2026-10-28

A performance and robustness release. GPU block equalizers and the CPU decision-directed equalizers with inline carrier recovery run substantially faster. `iir_filter` keeps single precision on both devices. Max-log LLRs of uniform square QAM have a closed form. The CUDA code was reviewed end to end. A few changes need code updates; see the first section.

### Changes that may need code updates

- **`build_pilot_ref`** takes `num_symbols=` and `num_channels=` instead of `n_sym=` and `num_ch=`, the library's glossary names. Rename the keywords at call sites.
- **`EqualizerResult` is a frozen dataclass**, like every other multi-value result. Assigning to a field raises; build a modified copy with `dataclasses.replace(result, field=value)`.
- **GPU failures raise.** A CUDA kernel that fails to compile or load, or a CUDA-graph capture that fails, used to log one warning and fall back to a slower path. It now raises, because it means a broken install or a bug.
- **`SingleCarrierFrame` validates its pilots on construction.** `pilot_pattern="block"` without `pilot_block_len` raises; it used to silently build a frame with no pilots. A pilot period that does not tile the body also raises at construction instead of on first use.
- **`moving_average(mode="valid")`** raises on a record shorter than the window, instead of returning NumPy's swapped convolution.
- **Numerical results move at rounding level** in a few places, from reordered or fused arithmetic. Seeds reproduce results within one version only. The largest moves:
  - fiber propagation on the CPU: about 3e-6 relative;
  - `lms` with inline carrier recovery: up to 5e-6 relative;
  - `correct_iq_imbalance`: 5e-7 relative in complex64;
  - `rde`: 2.5e-7 relative.

  Everything else changed in this release is bit-identical or within float32 rounding of before.

### Added

- **Single-precision `iir_filter` on both devices.**
  - CPU: a Numba kernel that reads and writes the samples in their own dtype and keeps the sections and the state in double precision. Double input still matches SciPy bit for bit.
  - GPU: a block-parallel CUDA kernel. Blocks are filtered in parallel and the state is carried across them in two levels, in double precision.
  - There is no longer a double-precision copy of the record: complex64 is 0.32–0.41x of the previous GPU path.
- **Closed-form max-log LLRs** for uniform separable QAM, `compute_llr`'s default case. Each bit's LLR comes from the nearest level of each bit value, so an axis costs 1 + 2k metrics instead of sqrt(M). It works for any per-axis labeling, not only Gray, and gives the same values as before: bit for bit on the CPU, within 3e-7 on the GPU.
  - GPU, every order: 0.87x at 16-QAM, 0.74x at 256-QAM.
  - CPU, from 256-QAM (where it starts to pay): 0.65x at 256-QAM, 0.40x at 1024-QAM, 0.23x at 4096-QAM.
  - Exact LLRs and shaped constellations keep the per-level computation.
- `TimingEstimate` is listed in `commkit.timing.__all__`, like the other estimate classes.
- `pad_mode="reflect"` is documented for the equalizers and `apply_taps`; they already accepted it.

### Performance

Equalizers:

- **Block equalizers on the GPU:** each block's window load, training staging and output stores now run inside the replayed CUDA graph, so the host launches one graph per block. Baseline `0006` against `0005`:
  - `block_lms` at `block_size=256`: 0.36x without carrier recovery, 0.78x with BPS, 0.65x with BPS and cycle slips;
  - `block_cma`: 0.50x;
  - `block_rde`: 0.66x.

  At 256 symbols per block the GPU now runs `block_lms` about 2x faster than the CPU without carrier recovery, and 5x faster with BPS.
- **`block_lms` per block:** one phasor per block and one divergence check after the loop. The cycle-slip kernel binds its buffers once per run, cutting the per-block cost from 17 µs to 10–14 µs. On the GPU, cycle-slip correction runs for any channel count, where above 1024 channels it went through the host every block.
- **`lms` and `rls` with inline carrier recovery (CPU):** the tap and distance loops compile with fast math while the phase unwrap stays strict. RLS with BPS 0.63x, LMS with PLL 0.87x.
- **`rde` and `block_rde`:** the ring is decided by squared midpoints, with no square root. On the GPU this is one fused kernel: 112 µs → 10 µs per 2 × 1024-symbol block.

Carrier and frequency recovery:

- **BPS of square QAM on the CPU:** one Numba pass instead of a chunked NumPy loop: 0.39x.
- **PLL:** reads the samples in their own precision, with no complex128 copy of the record: 0.95–0.97x.
- **Blockwise `MengaliMorelli`:** one Numba call for all blocks instead of one per block. CPU 0.85x, GPU 0.62x.
- **Blockwise `correct_frequency_offset` on the GPU:** builds its phase trajectory on the GPU, where it used to make a round trip through the host.

Mapping and LLRs:

- **Exact LLRs on the GPU:** one exponential per point instead of one per point and bit. With the third pass over the points also gone, 128-cross QAM is 0.79x of baseline `0005`.
- **Square QAM on the GPU:** demapped per axis, 2·sqrt(M) metrics instead of M.
- **Hard decisions of PSK and cross QAM on the GPU:** one kernel launch instead of a Python loop over chunks.
- **`Constellation.map`:** packs the bits in one pass. On a 4-core machine: 16-QAM 33 → 5.8 ms, 256-QAM 55 → 10.5 ms, 2^20 symbols.
- **`Constellation.rotational_symmetry`:** 1024-QAM in 0.026 s instead of 2.2 s.

Other functions:

- **`correct_iq_imbalance`:** all channels at once from closed-form 2×2 statistics, CPU about 0.45x.
- **`evm` and `snr`:** measure the reference power once: 0.74–0.78x.
- **`moving_average(mode="valid")`:** one pass over all channels. CPU 0.11–0.15x, GPU 0.21–0.71x.
- **`estimate_timing` with several templates:** transforms the signal once: CPU 0.62–0.72x.
- **`plot_eye_diagram` histograms:** CPU 0.33x, GPU 0.55x.
- **Fiber propagation on the CPU:** NumPy stages replace the Numba passes, which competed with the multithreaded FFTs (0.56–0.92x for 2^14 to 2^20 samples). On the GPU, the plain CuPy stages that complex128 input runs are 0.53x for DBP.
- **Fewer GPU copies:** frequency grids, windows and the polarization demultiplexer's grid are built on the GPU instead of being copied there on every call.

### Fixed

- `plot_eye_diagram` ignored `vmin` and `vmax`.
- BPS on the GPU failed with `CUDA_ERROR_INVALID_VALUE` for more than 65535 channels.
- Single-precision input was computed in double precision in BPS distances, `block_rde`'s ring distances, the block equalizers' forward filter and `iir_filter`, against the library's precision contract.

### Development

- **Test plugins:**
  - `tests/plugins/transfer_trace.py` audits host-device copies in a GPU run.
  - `tests/plugins/precision_trace.py` now follows single-precision data through the FFTs.
- **CI:** the JIT-disabled kernel-coverage job measures the equalizers' Numba kernels again, and strict mypy covers every module of the `filtering` package.
- **Benchmarks:** new benchmarks for the IIR filter, moving average, PLL, blockwise Mengali-Morelli and IQ correction. Baseline `0006` is the new reference.
- **Code organization:** `filtering` and `analysis.linewidth` are packages, and every public name imports from the same place as before. The sequential equalizers share their Numba steps, CMA and RDE share one body, and duplicated helpers are merged.
- **Docs:** docstrings follow one style throughout. AGENTS.md documents the CUDA review rules and how to run `compute-sanitizer`.

## [2.2.0] - 2026-10-23

See the [GitHub release](https://github.com/lokgar/commkit/releases/tag/v2.2.0) and the [changes since 2.1.0](https://github.com/lokgar/commkit/compare/v2.1.0...v2.2.0).

## [2.1.0] - 2026-10-21

See the [GitHub release](https://github.com/lokgar/commkit/releases/tag/v2.1.0) and the [changes since 2.0.0](https://github.com/lokgar/commkit/compare/v2.0.0...v2.1.0).

## [2.0.0] - 2026-10-07

See the [GitHub release](https://github.com/lokgar/commkit/releases/tag/v2.0.0) and the [changes since 1.1.0](https://github.com/lokgar/commkit/compare/v1.1.0...v2.0.0).

## [1.1.0] - 2026-09-03

See the [GitHub release](https://github.com/lokgar/commkit/releases/tag/v1.1.0) and the [changes since 1.0.0](https://github.com/lokgar/commkit/compare/v1.0.0...v1.1.0).

## [1.0.0] - 2026-08-31

First release. See the [GitHub release](https://github.com/lokgar/commkit/releases/tag/v1.0.0).

[Unreleased]: https://github.com/lokgar/commkit/compare/v3.7.1...HEAD
[3.7.1]: https://github.com/lokgar/commkit/compare/v3.7.0...v3.7.1
[3.7.0]: https://github.com/lokgar/commkit/compare/v3.6.1...v3.7.0
[3.6.1]: https://github.com/lokgar/commkit/compare/v3.6.0...v3.6.1
[3.6.0]: https://github.com/lokgar/commkit/compare/v3.5.0...v3.6.0
[3.5.0]: https://github.com/lokgar/commkit/compare/v3.4.0...v3.5.0
[3.4.0]: https://github.com/lokgar/commkit/compare/v3.3.0...v3.4.0
[3.3.0]: https://github.com/lokgar/commkit/compare/v3.2.1...v3.3.0
[3.2.1]: https://github.com/lokgar/commkit/compare/v3.2.0...v3.2.1
[3.2.0]: https://github.com/lokgar/commkit/compare/v3.1.3...v3.2.0
[3.1.3]: https://github.com/lokgar/commkit/compare/v3.1.2...v3.1.3
[3.1.2]: https://github.com/lokgar/commkit/compare/v3.1.1...v3.1.2
[3.1.1]: https://github.com/lokgar/commkit/compare/v3.1.0...v3.1.1
[3.1.0]: https://github.com/lokgar/commkit/compare/v3.0.0...v3.1.0
[3.0.0]: https://github.com/lokgar/commkit/compare/v2.3.0...v3.0.0
[2.3.0]: https://github.com/lokgar/commkit/compare/v2.2.0...v2.3.0
[2.2.0]: https://github.com/lokgar/commkit/compare/v2.1.0...v2.2.0
[2.1.0]: https://github.com/lokgar/commkit/compare/v2.0.0...v2.1.0
[2.0.0]: https://github.com/lokgar/commkit/compare/v1.1.0...v2.0.0
[1.1.0]: https://github.com/lokgar/commkit/compare/v1.0.0...v1.1.0
