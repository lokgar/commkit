# Changelog

Notable changes to CommKit, newest first. Performance figures are from the reference machine (RTX 4070 Ti, Ryzen 7 7800X3D, WSL2) unless stated otherwise, as a ratio of new to old time (below 1 is faster).

## [Unreleased]

### Changes that may need code updates

- **`Reference` has two more fields**, `info_bits` and `code_block`. `bits` are the bits on the air; `info_bits` are the source bits before coding. Without coding `info_bits` is `bits` itself and `code_block` is `(1, 1)`, so existing code that reads `reference.bits` is unaffected. Code that builds a `Reference` positionally keeps working.

### Added

- **`commkit.coding`**, the channel-coding module: `encode(bits, code=...)` and `decode(llrs, code=..., method=...)`, dispatched on a code object, returning a `DecodeResult` (decoded bits, per-block success check and iterations).
  - `LinearBlockCode` from a generator or parity-check matrix, and `LinearBlockCode.hamming(r, extended=, length=)` for Hamming codes, including the inner codes of 400ZR, Hamming(128,119), and of IEEE 802.3dj, Hamming(128,120).
  - Decoders `Syndrome(max_weight=)` (coset leaders; `max_weight=1` makes an extended Hamming code correct one error and detect two) and `MaximumLikelihood()` (exhaustive, for small codes).
- **LDPC codes:** `LDPCCode.from_parity_check(H)`, `LDPCCode.quasi_cyclic(base, lifting)`, `LDPCCode.dvbs2(rate, frame)` (all DVB-S2 rates, normal and short frames) and `LDPCCode.pon()` (LDPC(17280, 14592) of IEEE 802.3ca and ITU-T G.9804.2). Decoders `MinSum(scale=, offset=)` and `SumProduct()`, layered, with early stopping per codeword, on the CPU (Numba) and the GPU (a CUDA kernel, about 10x the CPU on 64 DVB-S2 codewords). Encoding is systematic: a sparse product and a cumulative XOR for DVB-S2, a sparse product and a GEMM with the inverted parity part for PON.
- **Bit-level building blocks:** `CRC` as a code (`CRC.nr` for the 5G NR and LTE polynomials, `CRC.ethernet` for the IEEE 802.3 CRC-32, or any Rocksoft-model CRC), whose `decode` flags blocks that fail the check; `interleave` / `deinterleave` with an `Interleaver` (explicit, `Interleaver.block`, `Interleaver.random`); and the rate-matching codes `Punctured(code, positions)` and `Shortened(code, num_bits)`, decoded with the inner code's decoders.
- **Distribution matching:** `CCDM(constellation, block_length=)`, a constant-composition matcher (an exact 64-bit range coder, invertible for every input, blocks up to 2^30 points, 0.35 s for 10^6 points of 64-QAM on one core), and `PAS(code, constellation)`, probabilistic amplitude shaping with a systematic FEC code supplying the signs. Both are codes: `generate(..., code=PAS(LDPCCode.dvbs2("3/4"), Constellation.qam(64).shaped(entropy=5)))` builds a coded PS payload, and `decode` runs the FEC decoder and the dematcher.
- **Coded payloads:** `generate(..., code=)` and `SingleCarrierFrame(payload_code=)` encode random information bits before mapping. The reference keeps both bit streams.
- **`metrics.fer`**, the block error rate, and **`metrics.ngmi`**, the normalized GMI used as the soft-decision FEC threshold. Post-FEC BER is `ber(res.bits, reference.info_bits)`.

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

[Unreleased]: https://github.com/lokgar/commkit/compare/v2.3.0...HEAD
[2.3.0]: https://github.com/lokgar/commkit/compare/v2.2.0...v2.3.0
[2.2.0]: https://github.com/lokgar/commkit/compare/v2.1.0...v2.2.0
[2.1.0]: https://github.com/lokgar/commkit/compare/v2.0.0...v2.1.0
[2.0.0]: https://github.com/lokgar/commkit/compare/v1.1.0...v2.0.0
[1.1.0]: https://github.com/lokgar/commkit/compare/v1.0.0...v1.1.0
