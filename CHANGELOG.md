# Changelog

## 2.0.0

A redesign of the API around one model: plain functions on a `Signal` or an
array, with small frozen objects for constellations, pulses and algorithm
choices. Most public signatures changed; the numerical fixes below change
results. See the README for the model and a runnable quickstart.

### The model

- **`Signal`** is a frozen dataclass: `samples`, the facts `sampling_rate`,
  `symbol_rate` and `center_frequency`, the description `constellation` and
  `pulse`, the ground truth `reference` (`Reference(symbols, bits)`) and an
  optional `frame` layout. It holds no pipeline results. `sig.replace(...)`
  and `sig.to("gpu")` return new Signals.
- **Facts and choices.** Functions read facts (rates, sps) from a Signal and
  raise on a conflicting argument; choices (decision constellation, pulse,
  noise variance) default to the Signal's and an explicit argument wins.
- **Value objects** describe what 1.x passed as strings and loose keywords:
  `Constellation.qam(16)` / `.psk()` / `.pam()` / `.shaped(nu=)`, pulses
  `RRC`, `RC`, `Gaussian`, `Rect`, `SmoothRect`, and one object per algorithm
  (`BPS`, `PLL`, `ViterbiViterbi`, `CycleSlip`, `MthPower`, `MengaliMorelli`,
  `PilotSymbols`, `IncrementSlope`, `DshFmPsd`, ...).
- **Verbs.** `estimate_<quantity>(x, method)` returns a frozen
  `<Quantity>Estimate`; `correct_<quantity>(x, how)` returns the corrected
  data; `resolve_*` picks among discrete candidates.
- **Results** with several values are frozen dataclasses with named fields.
  Metrics return host floats, one per channel for `(C, N)` input.
- **The device follows the data.** No `backend=`, `device=` or global device
  switch; `sig.to("gpu")` moves data explicitly. Importing `commkit` has no
  side effects.
- **Keyword-only parameters** after the data arguments, everywhere.

### Breaking changes (1.x -> 2.0)

| 1.x | 2.0 |
| --- | --- |
| `generate_pam(...)`, `generate_psk(...)`, `generate_qam(...)`, `generate_psqam(...)` | `generate(Constellation.qam(16), num_symbols, symbol_rate=, sps=, pulse=RRC(0.1), rng=)`; shaping via `Constellation.qam(64).shaped(nu=)` |
| `modulation="qam", order=16` | `constellation=Constellation.qam(16)` |
| `sig.source_symbols`, `sig.source_bits` | `sig.reference.symbols`, `sig.reference.bits` |
| `sig.mod_scheme`, `mod_order`, `ps_pmf`, `pulse_shape`, `rrc_rolloff`, ... | `sig.constellation` (`.family`, `.order`, `.pmf`), `sig.pulse` |
| `sig.resolved_symbols`, `replace_samples` | removed; use `sig.replace(samples=...)` |
| `seed=` | `rng: int \| Generator \| None`; `None` no longer uses the global RNG |
| `lms(..., cpr_type="bps", cpr_bps_test_phases=64, ...)` (nine `cpr_*` keywords) | `lms(..., cpr=BPS(test_phases=64, cycle_slip=CycleSlip()))` |
| `cpr_state=`, `input_norm_factor=`, `samples_prefix=` | `state=result.state` |
| `lms(..., backend="jax")`, `update_mode="block"` | removed: sequential equalizers are Numba; use `block_lms` for frequency-domain blocks |
| `recover_carrier_phase_bps(y)`, `recover_carrier_phase_pll(y)`, `recover_carrier_phase_pilot_tone(y)`, ... | `correct_carrier_phase(y, BPS())` / `estimate_carrier_phase(y, PilotTone(...))` |
| `estimate_frequency_offset_mth_power(...)`, `correct_static_frequency_offset`, `correct_frequency_offset_blockwise` | `estimate_frequency_offset(x, MthPower(power=4))`, `correct_frequency_offset(x, estimate)` |
| `compensate_iq_imbalance_lowdin(x)`, `compensate_chromatic_dispersion` | `correct_iq_imbalance(x, Lowdin())`, `correct_chromatic_dispersion` |
| `evm(..., mode="blind")`, `num_train_symbols=` | `evm(..., blind=True)`, `num_skip_symbols=` |
| `linewidth_increment(..., method="slope")`, `linewidth_beta_separation`, `linewidth_dsh(..., method=)` | `estimate_linewidth(x, IncrementSlope())`, `BetaSeparation()`, `DshFmPsd()` / `DshIncrement()` / `DshLorentzian()`, returning a `LinewidthEstimate` |
| `frequency_drift_metrics`, `symbol_rate=` for phase records | `frequency_drift`, `sampling_rate=` |
| analysis results as dicts | `AllanDeviation`, `FrequencyDrift`, `LinewidthEstimate`, `DshFmNoisePsd` |
| sync and analysis plots taking loose arrays | plots take the estimate or result they draw |
| `debug_plot=True` | removed; plot the returned result |
| `commkit.helpers` | `commkit.math` (`rms`, `normalize`, dB conversions); private helpers moved |
| JAX, PyYAML | removed from the dependencies; `load_npz` no longer unpickles |

Metrics: `evm`, `snr`, `ber`, `ser`, `gmi` and `mi` take
`(data, reference=None, *, constellation=, num_skip_symbols=, ...)`, read the
reference from a Signal, return host values, and raise instead of returning
`None` or `0.0` when nothing is measured. EVM is in percent.

### Fixed (results change)

- **Inline BPS in `lms`, `rls` and `block_lms`** failed on ordinary
  receivers (16-QAM, 2 sps, 21 taps: over 100 % EVM): the blind phase during
  training was copied into the taps and never anchored. Training symbols now
  anchor the phase (data-aided over the BPS window); 13 % EVM at 18 dB.
- **BPS and cycle-slip repair follow the constellation's rotational
  symmetry** (standalone and inline): BPSK phases beyond pi/2 were
  unreachable and 8-PSK slips of pi/4 were never repaired.
- **`block_lms` cycle-slip carry** used a factor 4 instead of the symmetry:
  8-PSK blocks started pi/8 off after a corrected slip.
- **Metrics scale by the data-aided gain** `|<r s*>| / <|s|^2>` instead of
  unit total power, which shrank the symbols by `1/sqrt(1 + 1/SNR)`: SNR at
  -10 dB read -1.45 dB.
- **GMI of shaped constellations** is the bit-metric decoding rate
  `H(X) - sum_b H(B_b | Y)`; 1.x overstated it by `k - H(X)`.
- **SER and blind EVM of shaped constellations** decided on the wrong grid.
- **The reference holds exact constellation points**; training symbols and
  pilots are taken on the unit-power constellation's scale.
- **Joint Viterbi-Viterbi** weighted channels by amplitude^M; channels are
  normalized first.
- **PLL slicer grid** in float64.
- **`block_lms`** with host training symbols and GPU samples crashed.
- **`plot_equalizer_result`** printed Matplotlib's property table.

### Added

- `extract_payload`, `estimate_timing` / `correct_timing` with a
  `TimingEstimate`, `estimate_linewidth`, `resolve_channel_permutation`.
- `state=` continuation for every equalizer; `result.signal` (1-SPS Signal
  with the aligned reference).
- A warning when a DSH linewidth method receives what looks like a phase
  trajectory.
- `plot_time_domain(max_points=)`.
- Example notebooks, including `qam_receiver_quickstart.ipynb`, executed in
  CI.

### Performance

- LLRs on NumPy/CuPy: GPU 0.6-0.9x the 1.x time; CPU 1.4x slower than the
  JAX path for 16-QAM exact and 256-QAM (a follow-up).
- `import commkit` 6x faster (plotting loads on first use).
- `apply_awgn` 2-3x faster (Philox on the GPU).
- Plots: constellation 3-5x, eye diagrams 5-19x, long time-domain views 9x,
  spectrograms 5x faster.
- Equalizers at parity with 1.x on CPU; GPU within measurement noise.
