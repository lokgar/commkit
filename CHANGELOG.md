# Changelog

Notable changes to CommKit, newest first. Performance figures are from the reference machine (RTX 4070 Ti, Ryzen 7 7800X3D, WSL2) unless stated otherwise, as a ratio of new to old time (below 1 is faster).

## [Unreleased]

### Changes that may need code updates

- **`encode` rejects bits other than 0 and 1** with `ValueError`. Before, linear codes silently encoded another payload (`[2, 0, 0, 0]` gave the all-zero codeword), a value of 256 wrapped to 0, and the turbo encoder indexed its trellis with the value.
- **`TurboCode` and `LDPCCode` reject non-integer parameters** with `TypeError` (`TurboCode(40.0)` was accepted; float LDPC indices were truncated into a different code).
- **`fir_filter` and `ols_fir_filter` raise for complex taps on real samples.** The imaginary part was dropped (the taps `[1j]` gave zeros). Pass complex samples for a complex output.
- **`correct_timing(mode="slice")` raises for offsets outside `[0, N)`**: a negative offset kept only the last samples. Use `mode="zero"` or `"circular"` for negative offsets.
- **`gray_code`, `gray_to_binary` return read-only arrays.** They are cached, so writing to one changed every later call's result; copy before modifying.

### Performance

Paired A/B of alternating processes against the previous commit of each change:

- Reed-Solomon decoding: KP4, 2048 hard-decision codewords at BER 1e-3, CPU 413 to about 110 ms (direct XOR reduction 0.80, Forney's formula at the error positions only 0.42, hard input converted in place 0.79), GPU 42 to 25 ms.
- Chase decoding: 0.53 on the CPU, 0.30-0.51 on the GPU (no LLR round trip or re-encoding, no 4-D flip mask, larger GPU chunks).
- Polar decoding: SC of N = 1024 0.32, SCL(8) 0.70, SCL(32) 0.68 (paths copied only when they change slot).
- Tail-biting Viterbi: K = 7, 0.035 (all start states scored in one pass); decision storage a quarter of before at every termination.
- Turbo decoding: LTE K = 1024, max-log 0.82, exact log-MAP 0.65.
- Cached code tables: interleaving 2^20 values with one `Interleaver` 0.19 on the CPU, 0.013 on the GPU (identity shortcut in the code objects' equality, which every cached lookup called).
- `mi`: one exponential per element, 256-QAM 0.82 on the CPU and 0.33 on the GPU, 16-QAM 0.52 on the GPU.

### Fixed

- **Staircase `valid` reflected an earlier state of the frame**: a block corrected through the next one kept its failed flag, so `HardIterations(1)` could return the right payload with `valid=False`. Success is now checked on the final frame.
- **CCDM (and so PAS) `valid` accepted sequences no payload maps to**: a composition has more sequences than the 2^k payloads, and all were valid. Only the matcher's image is valid now.
- **`zf_equalizer` with a float64 channel estimate returned complex128** for complex64 samples; it now computes in the samples' precision.
- **`correct_timing(mode="zero")`** repeated each channel's first sample for a negative per-channel offset instead of zero-filling, and returned zeros for a scalar offset of 0.
- **`mi` and `ols_fir_filter` / `zf_equalizer` bound their working memory.** `mi`'s intermediates reached GiBs for many channels of high-order QAM (1 GiB of differences for two channels of 1024-QAM); they are now capped at 2^24 elements. Overlap-save transformed the whole record at once, against its docstring, about 8 times the input; it now runs in batches of blocks (3.5 times the input on long records; about 5% slower on the CPU for 2^25 samples).
- `load_npz` closes its archive; `ber`, `fer` and `ser` transfer their error counts once.

### Development

- Benchmarks for polar, Viterbi, turbo, Chase and CCDM decoding in `bench_coding.py`; `ldpc_workload` is now `coded_workload`.

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

[Unreleased]: https://github.com/lokgar/commkit/compare/v3.0.0...HEAD
[3.0.0]: https://github.com/lokgar/commkit/compare/v2.3.0...v3.0.0
[2.3.0]: https://github.com/lokgar/commkit/compare/v2.2.0...v2.3.0
[2.2.0]: https://github.com/lokgar/commkit/compare/v2.1.0...v2.2.0
[2.1.0]: https://github.com/lokgar/commkit/compare/v2.0.0...v2.1.0
[2.0.0]: https://github.com/lokgar/commkit/compare/v1.1.0...v2.0.0
[1.1.0]: https://github.com/lokgar/commkit/compare/v1.0.0...v1.1.0
