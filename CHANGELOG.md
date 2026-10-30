# Changelog

Notable changes to CommKit, newest first. Performance figures are from the reference machine (RTX 4070 Ti, Ryzen 7 7800X3D, WSL2) unless stated otherwise, as a ratio of new to old time (below 1 is faster).

## [Unreleased]

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

[Unreleased]: https://github.com/lokgar/commkit/compare/v3.2.0...HEAD
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
