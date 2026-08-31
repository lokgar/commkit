# %% [markdown]
# # Carrier-phase characterization of a coherent transmission
#
# **Goal**: given the output of a coherent receiver - equalized symbols with
# the carrier-phase recovery (CPR) **disabled** - quantify what the combined
# TX + LO laser pair is doing: residual frequency wander (drift), Wiener
# linewidth, FM-noise spectrum, and the Allan deviation of the frequency.
#
# This is the `commkit.analysis` chain:
#
# ```
# y_eq (CPR off), ref symbols
#     | carrier_phase_trajectory      -> φ[n]        (data-aided, slip-free)
#     | separate_drift_phase_noise    -> drift + pn  (spectral split @ cutoff)
#     + frequency_drift_metrics       -> wander std / peak-to-peak
#     + linewidth_increment           -> Δν from Var(Δφ_k) slope
#     + linewidth_beta_separation     -> Δν from the FM-noise PSD
#     + allan_deviation               -> noise-type classification
# ```
#
# **Why data-aided?**  A blind phase estimator (BPS, Viterbi-Viterbi) adds its
# own estimator noise and - worse - cycle slips, which destroy a linewidth
# estimate.  With *known* symbols, `angle(y·conj(d))` cancels the modulation
# exactly, so the trajectory carries only carrier phase + AWGN angle noise and
# never slips.  Use a training sequence, a decided-and-verified payload, or a
# loopback measurement.

# %% [markdown]
# ## 0. Backend and reproducibility
#
# CommKit dispatches on the array type: build the arrays with CuPy and the
# whole chain below runs on the GPU (useful - these captures are long); build
# them with NumPy and it runs on the CPU.  Hand-made matplotlib overlays need
# an explicit device->host transfer (`to_device(x, "cpu")`); the library plot
# functions handle that internally.
#
# All randomness flows through *seeded library generators* (`generate_psk`,
# `generate_phase_noise`, `apply_awgn`) - no module-level RNG.  Every cell is
# individually re-runnable with identical results, and CPU and GPU runs
# produce bit-identical data (generation happens on the CPU and transfers).

# %%
import matplotlib.pyplot as plt
import numpy as np

try:
    import cupy as xp  # arrays (and all analysis) on the GPU
except ImportError:
    xp = np  # CPU fallback - everything below is backend-agnostic

from commkit import analysis, generate_psk, plotting
from commkit.backend import to_device
from commkit.impairments import apply_awgn, generate_phase_noise


def cpu(a):
    """Device->host helper for hand-made matplotlib overlays."""
    return to_device(a, "cpu")


plotting.apply_default_theme()

# %% [markdown]
# ## 1. Simulate the receiver output - step by step
#
# 32 GBaud QPSK with three impairments on top of the (assumed perfectly
# equalized) symbols:
#
# * **QPSK reference sequence** `d` - `generate_psk` at 1 sample/symbol with
#   pulse shaping off yields the unit-energy symbols directly;
# * **Wiener phase noise** - combined beat linewidth `Δν = 1 MHz` (TX + LO
#   DFB pair): `generate_phase_noise` draws the per-symbol random walk with
#   increments `N(0, 2πΔν·T_sym)` *and returns the trajectory*, so the ground
#   truth stays available for every overlay and check below (this is why the
#   rotation is applied explicitly here rather than inside
#   `impairments.apply_phase_noise`);
# * **sinusoidal frequency wander** - ±4 MHz at 100 kHz, standing in for
#   residual FOE error / laser frequency dither (timescales compressed so a
#   few periods fit in this short record); deterministic, so built inline:
#   the phase is the integral of the instantaneous frequency,
#   `φ_drift = 2π·Σ δf·T`;
# * **AWGN** at 20 dB SNR (`impairments.apply_awgn`; at 1 sps, `esn0_db` is
#   the per-symbol SNR).
#
# In a real pipeline `y_eq` comes from your equalizer with CPR off (e.g.
# `equalization.apply_taps(...)`), and `d` is the known sequence aligned to
# it.

# %%
R = 32e9  # symbol rate [Baud]
T = 1.0 / R
N = 1 << 20  # symbols (≈ 33 µs)
DNU_TRUE = 1e6  # combined linewidth [Hz]
SNR_DB = 20.0
WANDER_AMP = 4e6  # frequency wander amplitude [Hz]
WANDER_FREQ = 100e3  # wander rate [Hz]

t = xp.arange(N) * T

# Known QPSK sequence (the data-aided reference), unit symbol energy:
d = generate_psk(N, sps=1, symbol_rate=R, order=4, pulse_shape="none", seed=7).samples

# Carrier phase = laser random walk (ground truth in hand) + injected wander:
phi_pn = generate_phase_noise(N, R, linewidth=DNU_TRUE, seed=17)
df_wander = WANDER_AMP * xp.sin(2.0 * np.pi * WANDER_FREQ * t)
phi_drift = 2.0 * np.pi * xp.cumsum(df_wander) * T
phi_true = phi_pn + phi_drift

# Rotate the symbols by the carrier phase, then add detection noise:
y = apply_awgn(d * xp.exp(1j * phi_true), sps=1, esn0_db=SNR_DB, seed=8)

# %% [markdown]
# ## 2. Data-aided phase trajectory
#
# `carrier_phase_trajectory` computes `unwrap(angle(y·conj(d)))` in float64.
# For dual-pol captures it also resolves the 0<->1 channel permutation
# (`channel_pairing="auto"`); it does **not** resolve a time misalignment -
# align first, or the product turns noise-like and everything downstream
# inflates.
#
# Sanity requirement: the per-symbol phase step `2πΔf·T + Δφ_pn + Δφ_awgn`
# must stay below π or `unwrap` itself slips - at 32 GBaud that bounds the
# uncompensated frequency offset to ≪ 16 GHz (not a constraint in practice)
# and the SNR to ≳ 5 dB.

# %%
phi = analysis.carrier_phase_trajectory(y, d)

fig, ax = plt.subplots()
step = 256
t_us = cpu(t) * 1e6
ax.plot(t_us[::step], cpu(phi)[::step], label=r"Recovered $\phi$")
ax.plot(
    t_us[::step],
    cpu(phi_true - phi_true[0] + phi[0])[::step],
    "--",
    label=r"True $\phi$ (offset)",
)
ax.set_xlabel("Time [µs]")
ax.set_ylabel("Carrier phase [rad]")
ax.set_title("Data-aided carrier phase: random walk + sinusoidal drift")
ax.legend()
plt.show()

# %% [markdown]
# ## 3. Split drift from phase noise
#
# The split is **a modelling choice, not physics**: a zero-phase low-pass at
# `cutoff` defines everything slower as "drift" and everything faster as
# "phase noise".  Report the cutoff together with any number derived from the
# parts.  Guidance:
#
# * cutoff well above the wander rate (here 100 kHz) so the drift component
#   captures it completely;
# * cutoff well below `Δν`-scale dynamics so the split does not eat genuine
#   phase noise - the increment-variance linewidth is *weakly* sensitive to
#   this (its per-symbol difference is itself a high-pass), but the PSD's
#   low-frequency decade is not;
# * discard ~half a cutoff period at each edge (filter transients) - the
#   metric functions take `edge_trim`.

# %%
CUTOFF = 2e6  # Hz
drift, pn = analysis.separate_drift_phase_noise(phi, R, cutoff=CUTOFF)
edge = int(round(0.5 * R / CUTOFF))

dm = analysis.frequency_drift_metrics(drift, R, edge_trim=edge)

# The drift component also carries the laser's own white-FM content below the
# cutoff - variance (Δν/π)·cutoff - which adds in quadrature to the injected
# wander (and dominates the peak excursions):
leak_std = np.sqrt(DNU_TRUE / np.pi * CUTOFF)
std_pred = np.sqrt(WANDER_AMP**2 / 2.0 + leak_std**2)
print(
    f"wander std     = {dm['std'] / 1e6:.2f} MHz   "
    f"(injected A/√2 = {WANDER_AMP / np.sqrt(2) / 1e6:.2f}, "
    f"+ white-FM leakage -> {std_pred / 1e6:.2f})"
)
print(
    f"wander pk-pk   = {dm['pp'] / 1e6:.2f} MHz   "
    f"(injected 2A = {2 * WANDER_AMP / 1e6:.2f}; excess = leakage tails)"
)
print(f"wander |max|   = {dm['max_abs'] / 1e6:.2f} MHz")

fig, ax = plt.subplots()
tt = (np.arange(cpu(dm["df"]).size) + edge) * T * 1e6
ax.plot(tt[::step], cpu(dm["df"])[::step] / 1e6, label=r"Recovered $\delta f(t)$")
ax.plot(t_us[::step], cpu(df_wander)[::step] / 1e6, "--", label="Injected wander")
ax.set_xlabel("Time [µs]")
ax.set_ylabel("Residual frequency [MHz]")
ax.set_title(
    f"Frequency wander from the drift phase slope (cutoff {CUTOFF / 1e6:.0f} MHz)"
)
ax.legend()
plt.show()

# %% [markdown]
# **Meaning of the numbers**: `std` is the typical residual spin your CPR has
# to follow; `pp`/`max_abs` are the worst case.  Relate them to the BPS
# tracking limit - a window of `K` symbols tolerates roughly
# `δf_max ≈ 1/(8·K·T_sym)` before the intra-window rotation exceeds the QAM
# quarter-symmetry:

# %%
for K in (32, 64, 128):
    print(
        f"BPS window K = {K:3d}:  δf_max ≈ {1.0 / (8 * K * T) / 1e6:8.1f} MHz   "
        f"({'OK' if 1.0 / (8 * K * T) > dm['max_abs'] else 'TOO SLOW'} for this wander)"
    )

# %% [markdown]
# ## 4. Linewidth from the increment variance
#
# For a Wiener phase, `Var(φ[n] - φ[n-k]) = 2πΔν·T_sym·k + 2σ_φ²`: linear in
# the lag `k`, with the AWGN angle noise in the intercept.  Fitting the line
# over a few lags therefore gives an **AWGN-free** linewidth - no SNR estimate
# needed (`method="slope"`, the default).  `method="subtract"` is the
# single-lag textbook version: it needs the noise variance explicitly and
# over-subtracts if you hand it a total-residual SNR - kept for comparison.
#
# `debug_plot=True` draws the measured points with the fitted line - the
# intercept should sit at `σ_n²` for unit-power QPSK at this SNR.
#
# **Assumption**: white-FM (Wiener) noise.  Flicker bends `Var(k)` upward and
# the fitted "linewidth" becomes an effective, lag-timescale number - check
# the FM PSD in §5 for a clean plateau before quoting it.

# %%
lw_inc = analysis.linewidth_increment(
    pn, R, method="slope", edge_trim=edge, debug_plot=True
)
print(
    f"linewidth (slope)   = {lw_inc['linewidth'] / 1e6:.3f} MHz  "
    f"(truth {DNU_TRUE / 1e6:.1f})"
)
print(
    f"fitted intercept    = {lw_inc['awgn_var']:.4f} rad²  "
    f"(σ_n² = {10.0 ** (-SNR_DB / 10.0):.4f} for unit-power QPSK)"
)

# %% [markdown]
# ## 5. Linewidth from the FM-noise PSD (β-separation)
#
# `fm_noise_psd` differentiates the phase and Welch-estimates the PSD of the
# instantaneous frequency, `S_f(f)` in Hz²/Hz.  Each impairment lives in its
# own region:
#
# * **drift / flicker** -> steep rise at low `f`;
# * **white FM (linewidth)** -> flat plateau at `S_f = Δν/π`;
# * **AWGN** -> white *phase* noise -> `S_f ∝ f²` rise toward Nyquist,
#   crossing the plateau at `f_knee = √(Δν/(2π σ_φ² T_sym))`.
#
# Two estimators come out of this PSD:
#
# * **β-area** (`linewidth`): integrate `S_f` over the **region where it
#   exceeds** the Di Domenico β-separation line `(8 ln2/π²)·f` - a union of
#   intervals gated by the line, *not* the `[f_min, f_max]` window (that is
#   only an outer fence); `Δν = √(8 ln2 · A)`.  The result carries the exact
#   integrated region as the `above` mask, and the plot below fills it in
#   red between the line and the PSD.  For pure white FM the plateau only
#   exceeds the line below `f_c = πΔν/(8 ln2) ≈ 0.57·Δν` - for our 1 MHz
#   laser that is 566 kHz, **below the Welch resolution** of a 33 µs record
#   at practical `nperseg` (32 GBaud / 2¹⁵ ≈ 1 MHz per bin).  The area
#   method therefore returns ~0 here - and the plot shows an *empty* red
#   region, which is the honest statement: it needs records long enough to
#   resolve decades *below* `f_c` (its home turf is slow-noise-dominated
#   sources, where most linewidth area sits at low `f`).
# * **white-FM floor** (`linewidth_floor` = π·median of in-band `S_f`): reads
#   the plateau directly - the robust choice at high symbol rates.  Its dials:
#   `f_min` above the drift region, `f_max` **below the AWGN knee** - put the
#   band's top at ≲ f_knee/3 or the `f²` tail inflates the median (a factor-2
#   error is easy; try it).

# %%
sigma_phi2 = 10.0 ** (-SNR_DB / 10.0) / 2.0
f_knee = np.sqrt(DNU_TRUE / (2.0 * np.pi * sigma_phi2 * T))
print(
    f"predicted AWGN knee ≈ {f_knee / 1e9:.2f} GHz -> use f_max ≈ {f_knee / 3e6:.0f} MHz"
)

lw_beta = analysis.linewidth_beta_separation(
    phi, R, nperseg=1 << 15, f_min=20e6, f_max=f_knee / 3.0
)
plotting.plot_frequency_noise_psd(
    lw_beta["f"],
    lw_beta["S_f"],
    beta_line=lw_beta["beta_line"],
    floor=lw_beta["linewidth_floor"],
    band=lw_beta["band"],
    above=lw_beta["above"],  # the *actual* β-integration region (S_f > β-line)
)
plt.show()

print(
    f"linewidth (β-area)  = {lw_beta['linewidth'] / 1e6:.3f} MHz  "
    "(≈0: crossover f_c unresolved, see text)"
)
print(
    f"linewidth (floor)   = {lw_beta['linewidth_floor'] / 1e6:.3f} MHz  "
    f"(truth {DNU_TRUE / 1e6:.1f})"
)

# What happens if f_max ignores the knee: the f² tail joins the median.
lw_bad = analysis.linewidth_beta_separation(
    phi, R, nperseg=1 << 15, f_min=20e6, f_max=2e9
)
print(
    f"floor with f_max=2 GHz (above knee/3): "
    f"{lw_bad['linewidth_floor'] / 1e6:.3f} MHz - AWGN-inflated"
)

# %% [markdown]
# ## 6. Allan deviation - which noise dominates at which timescale?
#
# The overlapping Allan deviation of the residual frequency classifies the
# dominant process by its log-log slope: white-FM `τ^{-1/2}`, flicker-FM
# `τ^0`, random-walk-FM `τ^{+1/2}`, linear drift `τ^{+1}`.  A *periodic*
# wander produces the characteristic scalloped bumps peaking near
# `τ ≈ 0.37/f_wander` - do not read those as random noise.
#
# Caveats: values are in Hz (not fractional frequency); the last decade of τ
# averages only a handful of samples and is statistically fragile; white-PM
# vs flicker-PM cannot be distinguished (needs the modified Allan deviation).

# %%
allan = analysis.allan_deviation(dm["df"], R, n_taus=40)
_, ax = plotting.plot_allan_deviation(allan["tau_s"], allan["adev"])
ax.axvline(0.37 / WANDER_FREQ, color="gray", ls=":")
plt.show()

# %% [markdown]
# ## 7. The dashboard
#
# The pieces above compose into the 2x2 overview figure - assemble the report
# dict from the individual results (this keeps orchestration in *your* script,
# where the cutoffs and bands are visible, not hidden in a library wrapper).

# %%
report = {
    "phi": phi,
    "drift": drift,
    "drift_metrics": dm,
    "linewidth_beta": lw_beta,
    "allan": allan,
}
plotting.plot_carrier_phase_characterization(
    report,
    symbol_rate=R,
    drift_cutoff=CUTOFF,
    band=(20e6, f_knee / 3.0),
    amp_ref=WANDER_AMP,
)
plt.show()

print(f"increment linewidth : {lw_inc['linewidth'] / 1e6:.3f} MHz")
print(f"β-separation floor  : {lw_beta['linewidth_floor'] / 1e6:.3f} MHz")
print(f"wander std          : {dm['std'] / 1e6:.2f} MHz")

# %% [markdown]
# ## 8. Limitations checklist
#
# * **Alignment**: `y_eq` and `d` must be symbol-aligned; the auto pairing
#   only fixes dual-pol permutation, not delay.
# * **Unwrap slips**: need SNR ≳ 5 dB and per-symbol phase steps < π; below
#   that the trajectory itself is unreliable.
# * **The drift/PN split is spectral, not physical** - quote the cutoff.
#   Flicker noise straddles it and partially lands on both sides.
# * **Increment linewidth** assumes white FM; it reports an effective value
#   under flicker.  Sub-100 kHz linewidths at GBaud rates drown under the
#   AWGN intercept - prefer the PSD floor there.
# * **β-separation** depends on `f_min` (observation time) for non-white
#   sources and on `f_max` staying below the AWGN knee.  The Welch segment
#   sets the lowest resolvable frequency (`R/nperseg`) - slow lab drift needs
#   longer records, not larger `nperseg`.
# * **Allan deviation** here is in Hz and drift-component only; periodic
#   wander scallops it.
# * All functions accept `(C, N)` dual-pol arrays (time on the last axis) and
#   run on CPU or GPU transparently - this notebook already does (§0).
