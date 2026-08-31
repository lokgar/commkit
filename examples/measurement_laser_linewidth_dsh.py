# %% [markdown]
# # MEASUREMENT TEMPLATE - laser linewidth, delayed self-heterodyne (AOM)
#
# Fill **§1 System parameters**, point `DATA_FILE` at your capture, run all
# cells.  The chain (and the physics behind every choice) is documented in
# `laser_linewidth_dsh.py` - this file stays lean on purpose.
#
# **What you need from the lab**
#
# * the digitized beat - **real** samples from a single PD, or **complex**
#   samples from a coherent (90°-hybrid IQ) receiver; both pass through the
#   same calls below unchanged;
# * the ADC sampling rate and the AOM shift `f_shift`;
# * the delay-arm length (or better: τ_d itself - §3 refines it from the
#   measured notch comb).

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    import cupy as xp  # analysis runs on the GPU when available
except ImportError:
    xp = np

from commkit import analysis, plotting
from commkit.backend import to_device

plotting.apply_default_theme()


def cpu(a):
    return to_device(a, "cpu")


# %% [markdown]
# ## 1. System parameters - EDIT THIS CELL

# %%
FS = 500e6  # ADC sampling rate [Hz]
F_AOM = 80e6  # AOM shift [Hz] - pass the *known* value; None -> estimated
FIBER_KM = 1.0  # delay-arm length [km] (SMF ≈ 4.9 µs/km) ...
TAU_D = FIBER_KM * 4.9e-6  # ... or overwrite with the calibrated delay [s]
CALIBRATE_TAU_D = True  # refine τ_d from the measured notch comb (§3)

DATA_FILE = Path("captures/dsh_beat.npy")  # (N,) beat - real (PD) or complex (IQ)
DATA_KEY = "beat"  # array name inside an .npz (ignored for .npy)

# Welch segment for the FM-PSD estimator: nperseg ≥ 8·FS·τ_d resolves the comb;
# keep the record ≥ ~5·NPERSEG so the floor median averages K ≳ 10 segments.
NPERSEG = 1 << 16


def load_capture(path: Path, key: str = DATA_KEY):
    """Loads a .npy / .npz capture as a 1-D array (raw scope exports: adapt
    here - e.g. np.fromfile(path, dtype=np.int16) with the vendor scaling)."""
    if path.suffix == ".npz":
        return np.load(path)[key].ravel()
    return np.load(path).ravel()


# %% [markdown]
# ## 2. Load the capture (or synthesize a demo if the file is missing)

# %%
if DATA_FILE.exists():
    beat = xp.asarray(load_capture(DATA_FILE))
    print(f"loaded {DATA_FILE}: {beat.shape[-1]} samples, {beat.dtype}")
else:
    print(f"*** {DATA_FILE} not found - synthesizing a DEMO capture ***")
    print("*** 200 kHz white-FM laser on the AOM carrier:                 ***")
    print("*** all three estimators below should agree on Δν ≈ 200 kHz.  ***")
    from commkit.impairments import apply_awgn, generate_phase_noise

    m = int(round(TAU_D * FS))
    phi_demo = generate_phase_noise((1 << 21) + m, FS, linewidth=200e3, seed=42)
    z_demo, _ = analysis.dsh_beat(phi_demo, FS, TAU_D, f_shift=F_AOM)
    beat = apply_awgn(z_demo, sps=1, esn0_db=25, seed=1).real  # single PD -> real

T_REC = beat.shape[-1] / FS
K_WELCH = max((beat.shape[-1] - 1 - NPERSEG) // (NPERSEG // 2) + 1, 1)
print(
    f"τ_d (nominal) = {TAU_D * 1e6:.2f} µs | record {T_REC * 1e3:.2f} ms | "
    f"resolution floor ≈ {FS / NPERSEG:.0f} Hz | Welch averages K = {K_WELCH}"
)

# %% [markdown]
# ## 3. Sanity view + τ_d calibration from the notch comb
#
# The beat PSD must show the line at `f_shift` fully inside `(0, FS/2)` -
# spectral folding corrupts the phase silently.  (Real single-PD captures
# only; a complex IQ capture just needs the line inside ±FS/2.)  The
# differential-phase FM
# PSD has notches at exactly `k/τ_d`: fitting their spacing calibrates τ_d
# (a 1 % delay error is a 2 % linewidth error through the τ_d² gain).

# %%
plotting.plot_psd(
    beat,
    sampling_rate=FS,
    nperseg=1 << 15,
    title="Raw beat spectrum - line at f_shift, fringes/wings visible",
)
plt.show()

dphi, f_hat = analysis.dsh_phase(beat, FS, f_shift=F_AOM)
print(f"beat carrier removed at {f_hat / 1e6:.4f} MHz")

if CALIBRATE_TAU_D:
    # The *un-deconvolved* FM PSD of Δφ carries the 4sin²(πfτ_d) comb; the
    # first minimum above f = 0 sits at 1/τ_d.  Search ±20 % around nominal,
    # then refine to sub-bin accuracy with a 3-point parabola on log(S) -
    # without it the estimate is quantized to the Welch bin (±FS/2·NPERSEG,
    # easily worse than the nominal fiber length).
    f_c_, S_c_ = analysis.fm_noise_psd(dphi, FS, nperseg=NPERSEG)
    f_np, S_np = np.asarray(cpu(f_c_)), np.asarray(cpu(S_c_))
    win = np.flatnonzero((f_np > 0.8 / TAU_D) & (f_np < 1.2 / TAU_D))
    if win.size >= 3:
        i = win[np.argmin(S_np[win])]
        y0, y1, y2 = np.log(S_np[i - 1 : i + 2])
        delta = 0.5 * (y0 - y2) / (y0 - 2.0 * y1 + y2)  # vertex offset in bins
        f_notch = float(f_np[i] + np.clip(delta, -1, 1) * (f_np[1] - f_np[0]))
        tau_cal = 1.0 / f_notch
        print(
            f"first notch at {f_notch / 1e3:.3f} kHz -> τ_d = {tau_cal * 1e6:.4f} µs "
            f"(nominal {TAU_D * 1e6:.4f} µs, Δ = {(tau_cal / TAU_D - 1) * 100:+.2f} %)"
        )
        TAU_D = tau_cal
    else:
        print("notch search window empty - check FS/τ_d/NPERSEG; keeping nominal τ_d")

# %% [markdown]
# ## 4. Linewidth, three ways - agreement is the quality check

# %%
res_fm = analysis.linewidth_dsh(
    beat, FS, TAU_D, f_shift=F_AOM, method="fm_psd", nperseg=NPERSEG
)
res_inc = analysis.linewidth_dsh(beat, FS, TAU_D, f_shift=F_AOM, method="increment")
res_lor = analysis.linewidth_dsh(beat, FS, TAU_D, method="lorentzian")

fig, ax = plotting.plot_frequency_noise_psd(
    res_fm["f"],
    res_fm["S_f"],
    floor=res_fm["linewidth"],
    band=res_fm["band"],
    used=res_fm["used"],  # auto-detected plateau: the bins the median ran over
    title="Laser FM-noise PSD (deconvolved; gaps = notch comb at k/τ_d)",
)
plt.show()

# Same PSD in the RF convention: SSB phase noise ℒ(f) [dBc/Hz].
sel = (res_fm["f"] > 0) & np.isfinite(res_fm["S_f"])
fig, ax = plt.subplots()
ax.semilogx(
    res_fm["f"][sel],
    10 * np.log10(res_fm["S_f"][sel] / res_fm["f"][sel] ** 2 / 2.0),
)
ax.set_xlabel("Offset frequency [Hz]")
ax.set_ylabel(r"$\mathcal{L}(f)$ [dBc/Hz]")
ax.set_title("SSB phase noise")
ax.grid(True, which="both")
plt.show()

print(f"fm_psd     : Δν = {res_fm['linewidth'] / 1e3:8.2f} kHz")
print(f"increment  : Δν = {res_inc['linewidth'] / 1e3:8.2f} kHz")
print(
    f"lorentzian : Δν = {res_lor['linewidth'] / 1e3:8.2f} kHz  "
    f"(τ_d/τ_c = {res_lor['coherence_factor']:.1f}; trust only if ≳ 6, "
    f"W₂₀/W₃ = {res_lor['lineshape_ratio']:.1f}, Lorentzian ≈ 9.95)"
)

# %% [markdown]
# ## 5. Long-term stability - Allan deviation (valid for τ ≳ 5·τ_d)

# %%
allan = analysis.allan_deviation(dphi / (2.0 * np.pi * TAU_D), FS, n_taus=40)
fig, ax = plotting.plot_allan_deviation(allan["tau_s"], allan["adev"])
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
# ## 6. Report - quote Δν *with* its context, not alone

# %%
print("== laser linewidth report (delayed self-heterodyne) ==")
print(
    f"record            : {T_REC * 1e3:.2f} ms @ {FS / 1e6:.0f} MS/s, AOM {F_AOM / 1e6:.1f} MHz"
)
print(
    f"delay             : τ_d = {TAU_D * 1e6:.3f} µs (1/τ_d = {1 / TAU_D / 1e3:.2f} kHz)"
)
print(f"coherence factor  : {res_lor['coherence_factor']:.1f}")
print(f"Δν (fm_psd)       : {res_fm['linewidth'] / 1e3:.2f} kHz")
print(f"Δν (increment)    : {res_inc['linewidth'] / 1e3:.2f} kHz")
print(f"Δν (lorentzian)   : {res_lor['linewidth'] / 1e3:.2f} kHz")
print("spread across estimators ≳ 20 % => inspect the FM PSD before quoting.")
