# %% [markdown]
# # Laser linewidth from a decoherence interferometer with IQ detection
#
# **The setup**: delayed **self-homodyne** with a 90° optical hybrid
# (phase-diversity receiver) - no AOM:
#
# ```
#                  +---- fiber delay τ_d (decoherence spool) ----+   +----------+--> I -> ADC
#   laser -> 50/50 +                                             +-->|90° hybrid|
#                  +---------------------------------------------+   +----------+--> Q -> ADC
# ```
#
# The delayed copy decoheres (`τ_d ≫ τ_c = 1/(πΔν)`), so the laser beats
# against an effectively independent replica of itself, and the hybrid +
# balanced detectors deliver the **complex** beat directly at baseband:
#
# $$z(t) \propto \exp\big(j\,\Delta\varphi(t)\big) + w(t), \qquad
# \Delta\varphi(t) = \varphi(t) - \varphi(t-\tau_d).$$
#
# Why this configuration instead of the classic AOM self-heterodyne:
#
# * **no AOM / RF synthesizer** - one less phase-noise source billed to the
#   laser, one less driver to calibrate;
# * **both quadratures** - the phase is recovered directly (a single
#   photodiode at `f_shift = 0` would observe only `cos Δφ`, which is not
#   invertible);
# * **modest ADC rates** - the spectrum only has to cover the beat line and
#   its wings around 0 Hz, not an 80 MHz carrier.
#
# In the incoherent regime the beat line at 0 Hz is the self-convolution of
# the laser line: a **Lorentzian of FWHM 2Δν** (for white FM noise).  All
# three `linewidth_dsh` estimators apply here; this notebook runs the full
# chain including the front-end calibration that this receiver *requires*.
# (For the AOM/heterodyne variant, the coherent short-delay regime, and
# flicker-noise effects, see `laser_linewidth_dsh.py`.)

# %% [markdown]
# ## 0. Backend and reproducibility

# CommKit dispatches on the array type: build the arrays with CuPy and
# every `analysis` call below runs on the GPU; build them with NumPy and it
# runs on the CPU.  Hand-made matplotlib overlays go through
# `to_device(x, "cpu")`; the library plot functions handle that internally.

# All randomness flows through *seeded library generators*
# (`generate_phase_noise`, `apply_awgn`) - no module-level RNG.  That buys
# two things: every cell is individually re-runnable with identical results
# (no hidden generator state advancing between cells), and trajectories are
# generated on the CPU and transferred, so CPU and GPU runs of this notebook
# produce bit-identical data.

# %%
import matplotlib.pyplot as plt
import numpy as np

try:
    import cupy as xp  # arrays (and all analysis) on the GPU
except ImportError:
    xp = np  # CPU fallback - everything below is backend-agnostic

from commkit import analysis, plotting
from commkit.backend import to_device
from commkit.impairments import (
    apply_awgn,
    apply_iq_imbalance,
    compensate_iq_imbalance_gram_schmidt,
    generate_phase_noise,
)
from commkit.spectral import welch_psd


def cpu(a):
    """Device->host helper for hand-made matplotlib overlays."""
    return to_device(a, "cpu")


plotting.apply_default_theme()

# %% [markdown]
# ## 1. Simulate the measurement - step by step

# A 100 kHz white-FM laser (`τ_c = 3.2 µs`) against a 25 km spool
# (`τ_d = 122.5 µs`, coherence factor `π·Δν·τ_d ≈ 38` -> safely incoherent).
# Note the ADC: **50 MS/s is enough** - the -20 dB width of the beat line is
# only `√99·2Δν ≈ 2 MHz`.

# **Step 1 - the laser.**  `generate_phase_noise` draws the white-FM (Wiener)
# phase trajectory `φ[n]` with per-sample increments `N(0, 2πΔν/f_s)`.  The
# generator returns the trajectory itself, so the ground truth stays in hand
# for every comparison below.

# **Step 2 - the interferometer: where the beat happens.**  The 50/50 splitter
# feeds the hybrid two copies of the field `E(t) = exp(jφ(t))`: the direct arm
# `E(t)` and the spool arm `E(t-τ_d)`.  The 90° hybrid mixes the two into four
# outputs; the two balanced photodiode pairs subtract them pairwise, giving

# $$I = \mathrm{Re}\{E_{\rm sig} E_{\rm del}^*\}, \qquad
#   Q = \mathrm{Im}\{E_{\rm sig} E_{\rm del}^*\},$$

# i.e. the receiver physically computes the **complex field product**
# `z = E_sig · E_del*`.  For unit-amplitude fields that is exactly
# `exp(jΔφ)` with `Δφ(t) = φ(t) - φ(t-τ_d)` - the "beat" is this product;
# with no AOM it sits at 0 Hz instead of at a carrier.  (The same forward
# model is available as `analysis.dsh_beat(phi, FS, TAU_D)`, used in the AOM
# notebook - spelled out here once so the physics is visible.)

# **Step 3 - detection noise**, at the beat SNR (`apply_awgn`; at 1 sps,
# `esn0_db` is the SNR in the full Nyquist band).

# **Step 4 - receiver imperfections** every real hybrid has:

# * **DC offsets** from imperfectly balanced photodiode pairs - and in
#   self-homodyne they land *exactly at the beat line center*;
# * **IQ imbalance** (hybrid not exactly 90°, unequal arm gains) - mixes in
#   the conjugate image of the beat.

# %%
FS = 50e6  # ADC sampling rate [Hz] - baseband capture, no carrier
N = 1 << 21  # record length (≈ 42 ms)
DNU_TRUE = 100e3  # laser linewidth to recover [Hz]
SNR_DB = 25  # beat SNR in the Nyquist band

M_DELAY = int(round(122.5e-6 * FS))  # 25 km SMF
TAU_D = M_DELAY / FS
COH = np.pi * DNU_TRUE * TAU_D
print(
    f"τ_d = {TAU_D * 1e6:.1f} µs, τ_c = {1e6 / (np.pi * DNU_TRUE):.1f} µs, τ_d/τ_c = {COH:.0f}"
)

# Step 1: laser phase trajectory (long enough to cover the delayed copy too).
phi = generate_phase_noise(N + M_DELAY, FS, linewidth=DNU_TRUE, seed=42)

# Step 2: the two interferometer arms beat in the hybrid - complex field
# product of the direct and delayed copies (I + jQ from the balanced pairs).
E = xp.exp(1j * phi)  # laser field, unit amplitude
z_beat = E[M_DELAY:] * xp.conj(E[:-M_DELAY])  # = exp(jΔφ), the 0 Hz beat

# Step 3: detection (shot/thermal) noise.
z_ideal = apply_awgn(z_beat, sps=1, esn0_db=SNR_DB, seed=1)

# Step 4: receiver imperfections.
DC_TRUE = 0.18 - 0.12j  # per-quadrature offsets, relative to |z| = 1
z_meas = apply_iq_imbalance(z_ideal, 1.0, 5.0) + DC_TRUE

# %% [markdown]
# ## 2. Front-end calibration - the step this receiver cannot skip
#
# **DC first.**  Block the signal path (or the delay arm) and record a dark
# capture: whatever the ADCs report is detector/electronics offset.  Subtract
# it from every measurement.  Skipping this is the classic self-homodyne
# mistake - the spur sits mid-line where it hijacks any spectral peak search.
#
# **Then IQ imbalance.**  Gram-Schmidt orthogonalisation (GSOP) fixes the
# hybrid imbalance blindly.  Its circular-signal assumption is comfortably
# satisfied here (`Var[Δφ] = 2πΔν·τ_d ≈ 77 rad²` - the phase wraps many
# times); only for very short delays would you calibrate with a test tone
# instead.

# %%
# Dark capture -> DC calibration.  With the signal blocked the ADCs record
# offset + detector noise; the noise level does not depend on the blocked
# signal, so pin it with signal_power=1.0 - the level a unit-amplitude beat
# sees (the record's own near-zero power cannot define it).
z_dark = apply_awgn(
    xp.full(1 << 16, DC_TRUE, dtype=xp.complex128),
    sps=1,
    esn0_db=SNR_DB,
    signal_power=1.0,
    seed=2,
)
dc_cal = complex(xp.mean(z_dark))
print(f"calibrated DC = {dc_cal:.4f}   (true {DC_TRUE})")

z = compensate_iq_imbalance_gram_schmidt(z_meas - dc_cal)

# %% [markdown]
# ## 3. Look at the beat spectrum
#
# Complex data -> two-sided spectrum around 0 Hz.  The corrected line is the
# expected Lorentzian; on the raw capture the DC spur pokes out exactly at
# the line center (and the IQ image adds a faint mirror asymmetry).

# %%
f_b, P_raw = welch_psd(z_meas, sampling_rate=FS, nperseg=1 << 15, return_onesided=False)
_, P_cor = welch_psd(z, sampling_rate=FS, nperseg=1 << 15, return_onesided=False)

fig, ax = plt.subplots()
pk = float(xp.max(P_cor))
ax.plot(
    cpu(f_b) / 1e6,
    10 * np.log10(cpu(P_raw) / pk),
    alpha=0.7,
    label="Raw (DC spur mid-line)",
)
ax.plot(cpu(f_b) / 1e6, 10 * np.log10(cpu(P_cor) / pk), label="DC + GSOP corrected")
ax.set_xlim(-2, 2)
ax.set_ylim(-45, 12)
ax.set_xlabel("Frequency [MHz]")
ax.set_ylabel("PSD [dB rel. corrected peak]")
ax.set_title(
    f"Self-homodyne beat at 0 Hz - Lorentzian, FWHM = 2Δν = {2 * DNU_TRUE / 1e3:.0f} kHz"
)
ax.legend()
plt.show()

# %% [markdown]
# ## 4. Differential phase and its sanity checks
#
# `dsh_phase` with `f_shift=0.0` (physically known - there is no AOM):
# nothing is removed or detrended, so slow laser drift stays visible in Δφ.
# Passing `f_shift=None` would instead subtract the record's best-fit mean
# frequency - use that when an uncalibrated receiver LO offset must go.
#
# First check: `Var[Δφ] = 2πΔν·τ_d` in the incoherent regime.

# %%
dphi, _ = analysis.dsh_phase(z, FS, f_shift=0.0)
print(
    f"Var[Δφ] = {float(xp.var(dphi)):.1f} rad²   "
    f"(theory 2πΔν·τ_d = {2 * np.pi * DNU_TRUE * TAU_D:.1f} rad²)"
)

# %% [markdown]
# ## 5. Linewidth, three ways
#
# All three estimators are valid in this regime; agreement between them is
# the real quality check of the measurement.
#
# * **`fm_psd`** - deconvolve the interferometer response to get the laser
#   FM-noise PSD; the white-FM plateau at `Δν/π` *is* the linewidth, and any
#   1/f rise (laser flicker + **fiber acoustics of the 25 km spool**) is
#   visible instead of silently folded into one number.  Long-delay caveat:
#   the response period is `1/τ_d ≈ 8 kHz`, so the Welch bin must resolve it -
#   `nperseg ≳ 8·f_s·τ_d ≈ 49000` (the library warns if not).  The plateau
#   region is **auto-detected** (log-binned minimum-level search) and spans
#   as many lobes as the detection-noise knee allows, skipping the notch
#   neighborhoods and any rising low-frequency region; the green markers in
#   the plot are the bins actually used (`f_min`/`f_max` switch to manual).
# * **`increment`** - `Var[Δφ(t) - Δφ(t-a)] = 4πΔν·a + 2σ_w²`; slope -> Δν,
#   detection noise -> intercept.  Default lags auto-shrink for long delays
#   (smaller lags give the variance estimator many more independent
#   averages).
# * **`lorentzian`** - the line sits at 0 Hz; `f_peak` should come out ≈ 0.
#   Report the deep (-20 dB) width; the `lineshape_ratio` (≈ 9.95 for a pure
#   Lorentzian) diagnoses 1/f contamination.

# %%
res_fm = analysis.linewidth_dsh(
    z, FS, TAU_D, f_shift=0.0, method="fm_psd", nperseg=1 << 16
)
res_inc = analysis.linewidth_dsh(z, FS, TAU_D, f_shift=0.0, method="increment")
res_lor = analysis.linewidth_dsh(z, FS, TAU_D, method="lorentzian", debug_plot=True)

fig, ax = plotting.plot_frequency_noise_psd(
    res_fm["f"],
    res_fm["S_f"],
    floor=res_fm["linewidth"],
    band=res_fm["band"],
    used=res_fm["used"],
    title="Deconvolved laser FM-noise PSD (gaps: notch comb at k/τ_d)",
)
ax.axhline(DNU_TRUE / np.pi, color="C3", ls="--", label=r"Truth $\Delta\nu/\pi$")
ax.legend()
plt.show()

print(f"fm_psd     : Δν = {res_fm['linewidth'] / 1e3:6.1f} kHz")
print(f"increment  : Δν = {res_inc['linewidth'] / 1e3:6.1f} kHz")
print(
    f"lorentzian : Δν = {res_lor['linewidth'] / 1e3:6.1f} kHz   "
    f"(f_peak = {res_lor['f_peak'] / 1e3:.1f} kHz, "
    f"W₂₀/W₃ = {res_lor['lineshape_ratio']:.1f}, "
    f"τ_d/τ_c = {res_lor['coherence_factor']:.0f})"
)
print(f"truth      : Δν = {DNU_TRUE / 1e3:6.1f} kHz")

# %% [markdown]
# ### What skipping the DC calibration costs
#
# Same record, no correction: the spur mid-line hijacks the lorentzian peak
# search and the width collapses toward the Welch resolution (the built-in
# warnings fire).  The phase-based estimators shrug it off - the ripple a
# static offset adds to `angle(z)` is bounded, while the decohered phase
# walks over many radians.

# %%
res_lor_bad = analysis.linewidth_dsh(z_meas, FS, TAU_D, method="lorentzian")
res_fm_bad = analysis.linewidth_dsh(
    z_meas, FS, TAU_D, f_shift=0.0, method="fm_psd", nperseg=1 << 16
)
print(
    f"uncalibrated lorentzian : {res_lor_bad['linewidth'] / 1e3:6.1f} kHz  <- DC spur"
)
print(f"uncalibrated fm_psd     : {res_fm_bad['linewidth'] / 1e3:6.1f} kHz  (robust)")

# %% [markdown]
# ## 6. Long-term stability - Allan deviation
#
# Below the first notch the interferometer is a frequency discriminator:
# `Δφ(t)/(2πτ_d)` is the laser's instantaneous frequency averaged over the
# τ_d window.  The Allan deviation of that series is valid for **τ ≫ τ_d**
# (≈ 0.6 ms and up here).  Pure white FM follows `σ_y(τ) = √(Δν/(2πτ))`; a
# real laser flattens (flicker) and turns up (temperature drift) at long τ -
# extend the *record*, not the sampling rate, to see that region.

# %%
df_disc = dphi / (2.0 * np.pi * TAU_D)  # discriminator output [Hz]
allan = analysis.allan_deviation(df_disc, FS, n_taus=40)

fig, ax = plotting.plot_allan_deviation(allan["tau_s"], allan["adev"])
tau_valid = allan["tau_s"][allan["tau_s"] > 5 * TAU_D]
ax.loglog(
    tau_valid,
    np.sqrt(DNU_TRUE / (2.0 * np.pi * tau_valid)),
    "--",
    color="C3",
    label=r"White-FM theory $\sqrt{\Delta\nu/2\pi\tau}$",
)
ax.axvspan(
    allan["tau_s"][0],
    5 * TAU_D,
    color="gray",
    alpha=0.4,
    label=r"$\tau \lesssim 5\tau_d$: invalid",
)
ax.legend()
plt.show()

# %% [markdown]
# ## 7. Checklist for this setup
#
# 1. **Dark-capture the DC offsets** (arms blocked) and subtract - the spur
#    sits exactly mid-line; it is the one error this receiver does not
#    forgive in spectral estimates.
# 2. **GSOP the IQ imbalance** - one line, safe in the decoherence regime.
# 3. **Confirm the regime**: `π·Δν·τ_d ≳ 6`; the `lorentzian` result carries
#    it as `coherence_factor`, and warnings fire when violated.
# 4. **Resolve the notch comb for `fm_psd`**: `nperseg ≳ 8·f_s·τ_d`.  The
#    plateau auto-detection then spans the lobes and stops below the
#    detection-noise knee on its own - inspect the used-bin markers
#    (`res['used']`) to confirm what the number was read from.
# 5. **Cross-check all three estimators** - they share no assumptions beyond
#    the setup, so agreement (a few %) is strong evidence the number is the
#    laser and not the receiver.
# 6. The 25 km spool is a microphone: acoustic/thermal fiber noise appears as
#    a low-frequency rise in the FM PSD and as flattening in the Allan plot.
#    Report the PSD, not just Δν; isolate the spool mechanically/thermally.
# 7. Polarization fades on a long spool - use a Faraday mirror or a
#    polarization scrambler, and watch `|z|` for dropouts.

# %%
print("truth      :", f"{DNU_TRUE / 1e3:.0f} kHz")
print("fm_psd     :", f"{res_fm['linewidth'] / 1e3:.1f} kHz")
print("increment  :", f"{res_inc['linewidth'] / 1e3:.1f} kHz")
print("lorentzian :", f"{res_lor['linewidth'] / 1e3:.1f} kHz")


# f, S_f = res["f"], res["S_f"]
# L_dbc = 10 * np.log10(S_f / f**2 / 2.0)   # SSB phase noise, dBc/Hz
