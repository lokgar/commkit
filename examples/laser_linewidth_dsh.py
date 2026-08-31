# %% [markdown]
# # Laser linewidth from a delayed self-heterodyne (DSH) measurement
#
# **Goal**: characterize the phase noise of a CW laser - its FM-noise PSD and
# its (Lorentzian) linewidth - from a *digitized interferometric beat*, using
# `commkit.analysis.dsh_phase`, `dsh_fm_noise_psd`, and `linewidth_dsh`.
#
# ## The physical setup
#
# A laser's absolute phase noise cannot be observed directly: you would need a
# second, much better laser to beat against.  The delayed self-interference
# trick manufactures a reference **out of the laser itself**:
#
# ```
#                     +-------- fiber delay τ_d --------+
#   laser --> 50/50 --+                                 +--> 50/50 --> PD --> ADC
#                     +-- AOM (shift f_shift) ----------+
# ```
#
# * One arm is delayed by `τ_d = n·L/c` (≈ 4.9 µs per km of SMF).
# * The other arm is frequency-shifted by an AOM at `f_shift` (typically
#   40-200 MHz) so the beat lands away from DC - **self-heterodyne**: a single
#   photodetector suffices and the ADC records *real* samples.
# * The photodetector sees the interference of the laser with a delayed copy
#   of itself; the beat carries the **differential phase**
#
#   $$\Delta\varphi(t) = \varphi(t) - \varphi(t-\tau_d).$$
#
# This notebook covers this classic **AOM + single-photodetector receiver**
# (real samples).  The receiver is an independent choice, though: the same
# AOM setup read out with a coherent (90°-hybrid IQ) receiver delivers
# *complex* samples that every function below accepts directly - the Hilbert
# step and its Nyquist-margin constraint (checklist item 3) simply drop out.
# The AOM-free variant - the hybrid with `f_shift = 0` (self-homodyne) - has
# its own dedicated walkthrough in `laser_linewidth_homodyne_iq.py`.
#
# Whether the differential phase has "decohered" is governed by the ratio of
# the delay to the **coherence time** `τ_c = 1/(π·Δν)`:
#
# | regime | condition | beat spectrum |
# |---|---|---|
# | **incoherent** | `τ_d ≫ τ_c` | smooth Lorentzian of FWHM `2Δν` |
# | **coherent** | `τ_d ≲ τ_c` | carrier spike + fringes spaced `1/τ_d` |
#
# The **main path of this notebook is the classic long-delay (incoherent)
# measurement** - for the 100 kHz-1 MHz-class lasers of coherent transmission,
# `τ_d ≫ τ_c` costs only a few tens of km of fiber, and all three estimators
# apply.  The coherent short-delay regime is covered as a sidebar (§7): it is
# the *necessary* choice for kHz-class lasers, where the decoherence spool
# would grow to hundreds of km (loss, acoustics) - not the default.

# %% [markdown]
# ## 0. Backend and reproducibility
#
# CommKit dispatches on the array type: build the arrays with CuPy and
# every `analysis` call below runs on the GPU; build them with NumPy and it
# runs on the CPU.  Only hand-made matplotlib overlays need an explicit
# device->host transfer (`to_device(x, "cpu")`) - the library plot functions
# handle that internally.
#
# All randomness flows through *seeded library generators*
# (`generate_phase_noise`, `apply_awgn`) - no module-level RNG.  Every cell
# is therefore individually re-runnable with identical results, and CPU and
# GPU runs produce bit-identical data (trajectories are generated on the CPU
# and transferred).

# %%
import matplotlib.pyplot as plt
import numpy as np

try:
    import cupy as xp  # arrays (and all analysis) on the GPU
except ImportError:
    xp = np  # CPU fallback - everything below is backend-agnostic

from commkit import analysis, plotting
from commkit.backend import to_device
from commkit.impairments import apply_awgn, generate_phase_noise


def cpu(a):
    """Device->host helper for hand-made matplotlib overlays."""
    return to_device(a, "cpu")


plotting.apply_default_theme()

# %% [markdown]
# ## 1. Simulate the measurement - step by step
#
# A 200 kHz white-FM laser (`τ_c = 1.6 µs`) against a **25 km spool**
# (`τ_d = 122.5 µs`, coherence factor `π·Δν·τ_d ≈ 77` -> safely incoherent).
#
# **Step 1 - the laser.**  The canonical laser phase-noise model is **white
# FM noise**: the instantaneous frequency is white, so the phase is a Wiener
# process (random walk) and the field spectrum is Lorentzian with FWHM `Δν`:
#
# $$\varphi[n] = \sum_k \delta_k,\qquad \delta_k \sim
# \mathcal{N}\!\left(0,\; \tfrac{2\pi\,\Delta\nu}{f_s}\right).$$
#
# `impairments.generate_phase_noise` implements exactly this (its
# `linewidth` parameter is `Δν`) and returns the trajectory itself, keeping
# the ground truth in hand for every comparison below.
#
# **Step 2 - the interferometer.**  `analysis.dsh_beat` is the forward model
# the estimators invert: delay by `τ_d` (rounded to `m = τ_d·f_s` samples),
# beat the two copies, modulate onto the AOM carrier -
#
# $$z[n] = \exp\!\big(j\,(2\pi f_\mathrm{shift}\, n/f_s + \varphi[n] -
# \varphi[n-m])\big),$$
#
# returning both the beat and the true differential phase `Δφ`.
#
# **Step 3 - detection noise** `w[n]` (shot/thermal), added with
# `impairments.apply_awgn` (at 1 sample/symbol, `esn0_db` *is* the beat SNR
# in the full Nyquist band).

# %%
FS = 500e6  # ADC sampling rate [Hz]
N = 1 << 21  # record length (≈ 4.2 ms)
DNU_TRUE = 200e3  # laser linewidth to be recovered [Hz]
F_AOM = 80e6  # AOM shift [Hz]
SNR_DB = 25  # beat SNR in the full Nyquist band

M_DELAY = int(round(122.5e-6 * FS))  # 25 km SMF
TAU_D = M_DELAY / FS

phi = generate_phase_noise(N + M_DELAY, FS, linewidth=DNU_TRUE, seed=42)
z_dsh, dphi_true = analysis.dsh_beat(phi, FS, TAU_D, f_shift=F_AOM)
z_dsh = apply_awgn(z_dsh, sps=1, esn0_db=SNR_DB, seed=1)

print(
    f"τ_d = {TAU_D * 1e6:.1f} µs,  τ_c = 1/(πΔν) = {1e6 / (np.pi * DNU_TRUE):.2f} µs, "
    f"τ_d/τ_c = {np.pi * DNU_TRUE * TAU_D:.0f}  (≫ 6 => incoherent regime)"
)

# %% [markdown]
# ## 2. Look at the beat spectrum first
#
# Always plot the raw beat PSD before trusting any number.  In the incoherent
# regime the line at `f_shift` is the **self-convolution of the laser line**:
# a smooth Lorentzian of FWHM `2Δν`, no carrier spike, no fringes.  (Spike +
# fringes would mean the delay is too short for the classic width recipe -
# see §7.)

# %%
plotting.plot_psd(
    z_dsh,
    sampling_rate=FS,
    nperseg=1 << 15,
    xlim=((F_AOM - 3e6) / 1e6, (F_AOM + 3e6) / 1e6),
    title=f"Incoherent-regime beat: Lorentzian, FWHM = 2Δν = {2 * DNU_TRUE / 1e3:.0f} kHz",
)
plt.show()

# %% [markdown]
# ## 3. Extract the differential phase
#
# `dsh_phase` forms the analytic signal (Hilbert transform if the input is
# real), removes the beat carrier, and unwraps in float64.  If `f_shift` is
# not given, the mean beat frequency is estimated (coarse Kay estimate +
# exact least-squares slope removal) - that also *linearly detrends* Δφ,
# absorbing the mean laser drift into the returned `f_shift_hz`.  Pass the
# known AOM frequency when you want drift to stay visible in Δφ.
#
# The extracted Δφ **is not** the laser phase φ - it is φ filtered by the
# interferometer response `H(f) = 1 - e^{-j2πfτ_d}`.  Every estimator below
# has to undo (or sidestep) that response.
#
# Sanity check: in the incoherent regime `Var[Δφ] = 2πΔν·τ_d`.

# %%
dphi_est, f_beat = analysis.dsh_phase(z_dsh, FS, f_shift=F_AOM)
print(f"removed carrier: {f_beat / 1e6:.3f} MHz")
print(
    f"Var[Δφ] measured = {float(xp.var(dphi_est)):.1f} rad², "
    f"theory 2πΔν·τ_d = {2.0 * np.pi * DNU_TRUE * TAU_D:.1f} rad²"
)

# %% [markdown]
# ## 4. Method 1 - FM-noise PSD deconvolution
#
# The interferometer maps the laser FM-noise PSD through
#
# $$S_{f,\mathrm{beat}}(f) = 4\sin^2(\pi f \tau_d)\; S_{f,\mathrm{laser}}(f),$$
#
# so dividing the measured beat FM PSD by `4 sin²(πfτ_d)` recovers the laser
# FM PSD - except at the **notches** `f = k/τ_d` (including DC) where the
# interferometer is blind.  `dsh_fm_noise_psd` masks those bins (NaN + a
# `valid` mask).  With this long delay the comb is dense (`1/τ_d ≈ 8.2 kHz`),
# so the Welch segment must resolve it: `nperseg ≳ 8·f_s·τ_d ≈ 490 000`
# (the library warns if not).
#
# For a white-FM laser the recovered PSD is flat at `S_f = Δν/π`, and the
# linewidth is read off the plateau.  The plateau region is **auto-detected**
# (log-binned minimum-level search): the median spans every valid bin across
# as many lobes as the detection-noise knee allows - the green markers in the
# plot are the bins actually used - and a rising low-frequency region
# (drift, flicker, fiber acoustics) is excluded automatically.  Pass
# `f_min`/`f_max` to take manual control of the band instead.
#
# One statistical subtlety is handled internally: Welch bins are
# χ²-distributed, so a *median*-read floor sits `≈ 1 - 1/(3K)` below the true
# level for `K` averaged segments.  The reported linewidth divides that
# factor back out (`res['n_segments']` is K).  It matters exactly here: the
# comb-resolving `nperseg` leaves this record only K = 7 segments - a ~5 %
# correction - which is why the `Δν/π` floor guide sits slightly *above* the
# (uncorrected) log-binned median curve in the plot.

# %%
res_fm = analysis.linewidth_dsh(
    z_dsh, FS, TAU_D, f_shift=F_AOM, method="fm_psd", nperseg=1 << 19
)

fig, ax = plotting.plot_frequency_noise_psd(
    res_fm["f"],
    res_fm["S_f"],
    floor=res_fm["linewidth"],
    band=res_fm["band"],
    used=res_fm["used"],
    title="Laser FM-noise PSD - gaps are the interferometer notches k/τ_d",
)
ax.axhline(DNU_TRUE / np.pi, color="C3", ls="--", label=r"Truth $\Delta\nu/\pi$")
ax.legend()
plt.show()

print(
    f"fm_psd:    Δν = {res_fm['linewidth'] / 1e3:.1f} kHz   (truth {DNU_TRUE / 1e3:.0f} kHz, "
    f"plateau {res_fm['band'][0] / 1e3:.1f} kHz - {res_fm['band'][1] / 1e6:.1f} MHz)"
)

# %% [markdown]
# **Reading this plot**
#
# * flat plateau at `Δν/π` -> white-FM noise, height *is* the linewidth;
# * a `1/f` rise at low `f` would be flicker FM (technical noise, current
#   source, temperature, **spool acoustics**) - see §8; the auto-detected
#   plateau (green markers) starts above such a rise;
# * the rise toward Nyquist is detection noise (`f²` tail) - the plateau
#   detector stops below it.
#
# **Limitations to keep in mind**
#
# * bins near `k/τ_d` are unrecoverable (masked);
# * `τ_d` must be known accurately - below the first notch the discriminator
#   gain goes as `τ_d²`, so a 1 % delay error is a 2 % linewidth error.  If in
#   doubt, calibrate `τ_d` from the measured notch spacing
#   (`measurement_laser_linewidth_dsh.py` does this automatically);
# * everything the interferometer adds (fiber acoustics/thermal drift in the
#   delay arm, AOM synthesizer phase noise) is charged to the laser.

# %% [markdown]
# ## 5. Method 2 - increment-variance slope (AWGN-immune scalar)
#
# For a Wiener laser phase, the variance of the lag-`a` increment of the
# *measured* Δφ is linear in the lag (for `a ≤ τ_d`, so the two Wiener
# increments do not overlap):
#
# $$\operatorname{Var}[\Delta\varphi(t) - \Delta\varphi(t-a)] =
# 4\pi\,\Delta\nu\, a \;+\; 2\sigma_w^2 ,$$
#
# The detection noise lands entirely in the **intercept**, so a straight-line
# fit needs *no SNR knowledge* - the slope alone gives `Δν = slope/(4π)`.
# Default lags auto-shrink for long delays (small lags give the variance
# estimator many more independent averages).

# %%
res_inc = analysis.linewidth_dsh(z_dsh, FS, TAU_D, f_shift=F_AOM, method="increment")
print(f"increment: Δν = {res_inc['linewidth'] / 1e3:.1f} kHz")

# %% [markdown]
# ## 6. Method 3 - the textbook Lorentzian width
#
# In the incoherent regime the beat line is Lorentzian with FWHM `2Δν` - the
# classic recipe applies directly to the §2 spectrum.  Two practical
# subtleties:
#
# * the **-3 dB width** is corrupted by 1/f noise, which builds a Gaussian
#   core around the carrier (and keeps growing with observation time);
# * the customary fix is to measure the width far down the wings - at
#   `-20 dB`: for a Lorentzian, `W₋₂₀ = √99 · FWHM`, so
#   `Δν = W₋₂₀ / (2√99)`.
#
# `linewidth_dsh(method="lorentzian")` measures both and reports their ratio
# as a lineshape diagnostic: `W₂₀/W₃ ≈ 9.95` for a Lorentzian, `≈ 2.6` for a
# Gaussian - anything in between says "mixed", trust the deep width.
# `debug_plot=True` draws the beat line with both width contours.

# %%
res_lor = analysis.linewidth_dsh(
    z_dsh, FS, TAU_D, method="lorentzian", nperseg=1 << 15, debug_plot=True
)
print(
    f"lorentzian: Δν(-20 dB) = {res_lor['linewidth'] / 1e3:.1f} kHz, "
    f"Δν(-3 dB) = {res_lor['linewidth_3db'] / 1e3:.1f} kHz, "
    f"W₂₀/W₃ = {res_lor['lineshape_ratio']:.1f} (Lorentzian ≈ 9.95), "
    f"τ_d/τ_c = {res_lor['coherence_factor']:.0f}"
)

# %% [markdown]
# ## 7. Sidebar - when the spool cannot be long: the coherent regime
#
# The incoherent condition `τ_d ≫ τ_c` scales inversely with the linewidth:
# a **10 kHz laser needs τ_d ≫ 32 µs -> hundreds of km** once a safety factor
# is applied - impractical (0.2 dB/km loss, and every meter of fiber is a
# microphone).  For narrow-linewidth lasers the modern approach is therefore
# a *short* delay analyzed with the phase-based estimators, which do not
# care about the regime.  Same 200 kHz laser, 1 km spool
# (`τ_d/τ_c ≈ 3` -> coherent):
#
# * the beat spectrum shows the tell-tale carrier spike + fringes spaced
#   `1/τ_d` - a Lorentzian fit to *this* is meaningless: most of the power
#   sits in the un-broadened spike whose width is the Welch **resolution**,
#   and the estimator warns (`coherence_factor < 6`);
# * `fm_psd` (and `increment`) on the *same capture* still recover Δν - for
#   `f ≪ 1/τ_d` the interferometer is a frequency discriminator with known
#   gain `(2πfτ_d)²`, regime-independent.

# %%
M_SHORT = int(round(4.9e-6 * FS))  # 1 km spool
TAU_SHORT = M_SHORT / FS

phi_short = generate_phase_noise(N + M_SHORT, FS, linewidth=DNU_TRUE, seed=44)
z_short, _ = analysis.dsh_beat(phi_short, FS, TAU_SHORT, f_shift=F_AOM)
z_short = apply_awgn(z_short, sps=1, esn0_db=SNR_DB, seed=3)

plotting.plot_psd(
    z_short,
    sampling_rate=FS,
    nperseg=1 << 15,
    xlim=((F_AOM - 3e6) / 1e6, (F_AOM + 3e6) / 1e6),
    title=f"Coherent-regime beat: spike + fringes every 1/τ_d = {1e-6 / TAU_SHORT:.3f} MHz",
)
plt.show()

res_bad = analysis.linewidth_dsh(
    z_short, FS, TAU_SHORT, method="lorentzian", nperseg=1 << 15
)
res_good = analysis.linewidth_dsh(
    z_short, FS, TAU_SHORT, f_shift=F_AOM, method="fm_psd", nperseg=1 << 15
)
print(
    f"coherent-regime 'lorentzian' answer: {res_bad['linewidth'] / 1e3:.1f} kHz "
    f"(truth {DNU_TRUE / 1e3:.0f} kHz) - coherence factor "
    f"{res_bad['coherence_factor']:.2f} < 6, do not trust it."
)
print(f"fm_psd on the same capture         : {res_good['linewidth'] / 1e3:.1f} kHz")

# %% [markdown]
# ## 8. What 1/f (flicker) FM noise does
#
# Real lasers are never purely white-FM: below some corner frequency the FM
# PSD rises as `~1/f`.  `generate_phase_noise` models both terms of the
# standard power-law FM noise in one call,
#
# $$S_f(f) = \underbrace{\Delta\nu/\pi}_{\texttt{linewidth}} +
#            \underbrace{h_{-1}/f}_{\texttt{flicker}},$$
#
# so the laser below is the same white-FM laser plus a flicker term `h₋₁`
# chosen to cross the white plateau at ≈ 63 kHz (`flicker_f_min` caps the
# 1/f divergence at the lowest shaped frequency).  Consequences:
#
# * there is **no unique "linewidth"** anymore - the measured width grows
#   with observation time;
# * the -3 dB width inflates (Gaussian core), while the far wings - and the
#   white-FM plateau in the FM PSD - still reflect the intrinsic Lorentzian
#   part;
# * on the FM-PSD plot the flicker shows up directly as the low-frequency
#   rise, which is *the* most honest way to report it.
#
# The plateau auto-detection earns its keep here: a *naive* median over the
# first lobe reads the flicker level; the detected plateau (green markers)
# starts above the 1/f corner and recovers the white part without any
# manual fencing.

# %%
H_M1 = 4e9
phi_mix = generate_phase_noise(
    N + M_SHORT, FS, linewidth=DNU_TRUE, flicker=H_M1, flicker_f_min=1e3, seed=45
)
z_mix, _ = analysis.dsh_beat(phi_mix, FS, TAU_SHORT, f_shift=F_AOM)
z_mix = apply_awgn(z_mix, sps=1, esn0_db=SNR_DB, seed=4)

res_mix = analysis.linewidth_dsh(
    z_mix, FS, TAU_SHORT, f_shift=F_AOM, method="fm_psd", nperseg=1 << 15
)
res_naive = analysis.linewidth_dsh(
    z_mix,
    FS,
    TAU_SHORT,
    f_shift=F_AOM,
    method="fm_psd",
    nperseg=1 << 15,
    f_max=1.0 / TAU_SHORT,  # manual fence: first lobe only (the naive read)
)

fig, ax = plotting.plot_frequency_noise_psd(
    res_mix["f"],
    res_mix["S_f"],
    floor=res_mix["linewidth"],
    used=res_mix["used"],
    title="Flicker rises above the white plateau at low f - report the PSD, not one number",
)
fline = np.geomspace(2e3, 2e5, 50)
ax.loglog(fline, H_M1 / fline, "--", label=r"$h_{-1}/f$ injected flicker")
ax.axhline(DNU_TRUE / np.pi, color="C3", ls=":", label=r"White plateau $\Delta\nu/\pi$")
ax.legend()
plt.show()

print(
    f"naive first-lobe median      : Δν = {res_naive['linewidth'] / 1e3:.0f} kHz "
    "(flicker-inflated)"
)
print(
    f"auto-detected plateau median : Δν = {res_mix['linewidth'] / 1e3:.0f} kHz "
    f"(plateau {res_mix['band'][0] / 1e3:.0f} kHz - {res_mix['band'][1] / 1e6:.1f} MHz)"
)
print(f"truth (white-FM part only)   : Δν = {DNU_TRUE / 1e3:.0f} kHz")

# %% [markdown]
# ## 9. Long-term stability - Allan deviation from the discriminator
#
# Below the first notch the interferometer is a frequency discriminator:
# `Δφ(t)/(2πτ_d)` is the laser's instantaneous frequency averaged over the
# τ_d window.  Feeding it to `analysis.allan_deviation` gives the laser
# frequency stability σ_y(τ) - **valid for τ ≫ τ_d**.
#
# The short-delay capture is used here deliberately: its validity floor
# (`≈ 5·τ_d = 25 µs`) leaves two decades of usable τ in this record, whereas
# the 25 km spool would push the floor to 0.6 ms - with only 4.2 ms of data.
# A wider Allan range is one more genuine advantage of short delays.
#
# For pure white FM the expected curve is `σ_y(τ) = √(Δν/(2πτ))` - the
# `τ^{-1/2}` guide line of the plot.  A real laser flattens (flicker) and
# turns up (random-walk / temperature drift) at long τ; extend the record,
# not the sampling rate, to see that region.

# %%
dphi_short, _ = analysis.dsh_phase(z_short, FS, f_shift=F_AOM)
df_disc = dphi_short / (2.0 * np.pi * TAU_SHORT)  # discriminator output [Hz]
allan = analysis.allan_deviation(df_disc, FS, n_taus=40)

fig, ax = plotting.plot_allan_deviation(allan["tau_s"], allan["adev"])
tau_valid = allan["tau_s"][allan["tau_s"] > 5 * TAU_SHORT]
ax.loglog(
    tau_valid,
    np.sqrt(DNU_TRUE / (2.0 * np.pi * tau_valid)),
    "--",
    color="C3",
    label=r"White-FM theory $\sqrt{\Delta\nu/2\pi\tau}$",
)
ax.axvspan(
    allan["tau_s"][0],
    5 * TAU_SHORT,
    color="gray",
    alpha=0.4,
    label=r"$\tau \lesssim 5\tau_d$: invalid",
)
ax.legend()
plt.show()

# %% [markdown]
# ## 10. Summary
#
# | method | validity | needs | robust against |
# |---|---|---|---|
# | `fm_psd` | any regime | accurate τ_d, `nperseg ≳ 8·f_s·τ_d` | regime, AWGN + flicker (auto plateau), reveals flicker |
# | `increment` | any regime, white-FM laser | accurate τ_d | AWGN (intercept) |
# | `lorentzian` | `τ_d/τ_c ≳ 6` only | resolution + ≳30 dB dynamic range | 1/f core (use deep width) |
#
# Practical checklist for a real measurement
# (`measurement_laser_linewidth_dsh.py` is the fill-in template):
#
# 1. **Plot the beat PSD first** - a smooth line => incoherent, all three
#    methods apply; spike + fringes => coherent regime => forget the
#    Lorentzian fit.
# 2. Choose `τ_d` for the *laser*: 100 kHz-class -> a 25-50 km spool is the
#    standard incoherent setup; kHz-class -> decoherence is impractical, use a
#    short delay + `fm_psd`/`increment` (§7).
# 3. Sample fast enough that the beat (line + wings) fits inside `(0, f_s/2)`
#    - the AOM must clear the beat half-bandwidth on both sides.  This
#    constrains the *real* single-PD capture only: an IQ receiver (with or
#    without the AOM) delivers complex samples with no such margin.  (No AOM
#    at all?  See `laser_linewidth_homodyne_iq.py`.)
# 4. Check `dphi_var` against `2πΔν·τ_d` and the FM-PSD plateau against
#    `Δν/π` - self-consistency across estimators is the best sanity check.
# 5. Inspect the auto-detected plateau (green markers / `res['used']`): if it
#    is narrow or sits on a slope, the "linewidth" is not a clean white-FM
#    number - report the PSD.
# 6. Everything the interferometer adds (fiber acoustics, AOM driver noise,
#    polarization fading) is billed to the laser - measure the setup floor
#    with a much better laser, or with a shorted (τ_d -> 0) interferometer.

# %%
print("ground truth  :", f"{DNU_TRUE / 1e3:.0f} kHz")
print("fm_psd        :", f"{res_fm['linewidth'] / 1e3:.1f} kHz")
print("increment     :", f"{res_inc['linewidth'] / 1e3:.1f} kHz")
print("lorentzian    :", f"{res_lor['linewidth'] / 1e3:.1f} kHz")
