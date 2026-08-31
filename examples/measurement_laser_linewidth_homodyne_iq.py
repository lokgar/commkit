# %% [markdown]
# # MEASUREMENT TEMPLATE - laser linewidth, self-homodyne IQ (90° hybrid)
#
# Fill **§1 System parameters**, point `DATA_FILE` at your capture, run all
# cells.  Everything downstream (calibration -> spectra -> three estimators ->
# Allan) is the chain from `laser_linewidth_homodyne_iq.py`, which also
# explains the physics and the choices - this file stays lean on purpose.
#
# **What you need from the lab**
#
# * complex baseband capture `z[n] = I[n] + jQ[n]` from the hybrid's two
#   balanced pairs (no AOM: the beat sits at 0 Hz);
# * the ADC sampling rate;
# * the delay-arm length (or better: τ_d itself);
# * strongly recommended: a **dark capture** (signal blocked) for DC
#   calibration - in self-homodyne the DC offset lands exactly mid-line.

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
from commkit.impairments import compensate_iq_imbalance_gram_schmidt
from commkit.spectral import welch_psd

plotting.apply_default_theme()


def cpu(a):
    return to_device(a, "cpu")


# %% [markdown]
# ## 1. System parameters - EDIT THIS CELL

# %%
FS = 50e6  # ADC sampling rate [Hz]
FIBER_KM = 25.0  # delay-arm length [km] (SMF ≈ 4.9 µs/km) ...
TAU_D = FIBER_KM * 4.9e-6  # ... or overwrite with the calibrated delay [s]

DATA_FILE = Path("captures/homodyne_iq.npy")  # complex (N,) array, I + jQ
DATA_KEY = "z"  # array name inside an .npz (ignored for .npy)
DARK_FILE = Path("captures/homodyne_dark.npy")  # optional dark capture

# Welch segment for the FM-PSD estimator.  Must resolve the notch comb:
# nperseg ≥ 8·FS·τ_d (the library warns if violated).
NPERSEG = 1 << 17


def load_capture(path: Path, key: str = DATA_KEY):
    """Loads a .npy / .npz capture as a 1-D array (raw scope exports: adapt
    here - e.g. np.fromfile(path, dtype=np.int16) -> (I + 1j·Q) with scaling)."""
    if path.suffix == ".npz":
        return np.load(path)[key].ravel()
    return np.load(path).ravel()


# %% [markdown]
# ## 2. Load the capture (or synthesize a demo if the file is missing)

# %%
if DATA_FILE.exists():
    z_meas = xp.asarray(load_capture(DATA_FILE))
    print(f"loaded {DATA_FILE}: {z_meas.shape[-1]} samples, {z_meas.dtype}")
else:
    print(f"*** {DATA_FILE} not found - synthesizing a DEMO capture ***")
    print("*** 100 kHz white-FM laser + DC offset + IQ imbalance:          ***")
    print("*** all three estimators below should agree on Δν ≈ 100 kHz.   ***")
    from commkit.impairments import (
        apply_awgn,
        apply_iq_imbalance,
        generate_phase_noise,
    )

    m = int(round(TAU_D * FS))
    phi_demo = generate_phase_noise((1 << 21) + m, FS, linewidth=100e3, seed=42)
    z_demo, _ = analysis.dsh_beat(phi_demo, FS, TAU_D)  # f_shift=0: homodyne
    z_meas = apply_iq_imbalance(
        apply_awgn(z_demo, sps=1, esn0_db=25, seed=1), 1.0, 5.0
    ) + (0.18 - 0.12j)

if not xp.iscomplexobj(z_meas):
    raise ValueError("Homodyne IQ analysis needs complex I + jQ samples.")

T_REC = z_meas.shape[-1] / FS
print(
    f"τ_d = {TAU_D * 1e6:.1f} µs | record {T_REC * 1e3:.1f} ms | "
    f"resolution floor ≈ {FS / NPERSEG:.0f} Hz | first notch 1/τ_d = {1 / TAU_D / 1e3:.1f} kHz"
)

# %% [markdown]
# ## 3. Front-end calibration - DC (dark capture) then GSOP
#
# Without a dark capture the DC estimate falls back to the record mean -
# valid only in the decoherence regime (`π·Δν·τ_d ≫ 1`, where `E[z] ≈ 0`).

# %%
if DARK_FILE.exists():
    dc_cal = complex(xp.mean(xp.asarray(load_capture(DARK_FILE))))
    print(f"DC from dark capture : {dc_cal:.4f}")
else:
    dc_cal = complex(xp.mean(z_meas))
    print(
        f"DC from record mean  : {dc_cal:.4f}  (no dark capture - decoherence-regime fallback)"
    )

z = compensate_iq_imbalance_gram_schmidt(z_meas - dc_cal)

# %% [markdown]
# ## 4. Sanity views before any number: beat spectrum and |z| dropouts
#
# * The corrected line should be a smooth Lorentzian-ish peak at 0 Hz - a
#   residual spike mid-line means the DC calibration failed.
# * Dips in |z| are polarization fades (long spool): trim or re-measure.

# %%
f_b, P_raw = welch_psd(z_meas, sampling_rate=FS, nperseg=1 << 15, return_onesided=False)
_, P_cor = welch_psd(z, sampling_rate=FS, nperseg=1 << 15, return_onesided=False)
fig, ax = plt.subplots()
pk = float(xp.max(P_cor))
ax.plot(cpu(f_b) / 1e6, 10 * np.log10(cpu(P_raw) / pk), alpha=0.6, label="Raw")
ax.plot(cpu(f_b) / 1e6, 10 * np.log10(cpu(P_cor) / pk), label="DC + GSOP corrected")
ax.set_xlabel("Frequency [MHz]")
ax.set_ylabel("PSD [dB rel. peak]")
ax.legend()
plt.show()

env = xp.abs(z)
print(
    f"|z| mean = {float(xp.mean(env)):.3f}, min = {float(xp.min(env)):.3f} (≪ mean => fades)"
)

# %% [markdown]
# ## 5. Linewidth, three ways - agreement is the quality check

# %%
res_fm = analysis.linewidth_dsh(
    z, FS, TAU_D, f_shift=0.0, method="fm_psd", nperseg=NPERSEG
)
res_inc = analysis.linewidth_dsh(z, FS, TAU_D, f_shift=0.0, method="increment")
res_lor = analysis.linewidth_dsh(z, FS, TAU_D, method="lorentzian")

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
# ## 6. Long-term stability - Allan deviation (valid for τ ≳ 5·τ_d)

# %%
dphi, _ = analysis.dsh_phase(z, FS, f_shift=0.0)
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
# ## 7. Report - quote Δν *with* its context, not alone

# %%
print("== laser linewidth report (self-homodyne IQ) ==")
print(f"record            : {T_REC * 1e3:.1f} ms @ {FS / 1e6:.0f} MS/s")
print(
    f"delay             : τ_d = {TAU_D * 1e6:.2f} µs (1/τ_d = {1 / TAU_D / 1e3:.1f} kHz)"
)
print(f"coherence factor  : {res_lor['coherence_factor']:.1f}")
print(f"Δν (fm_psd)       : {res_fm['linewidth'] / 1e3:.2f} kHz")
print(f"Δν (increment)    : {res_inc['linewidth'] / 1e3:.2f} kHz")
print(f"Δν (lorentzian)   : {res_lor['linewidth'] / 1e3:.2f} kHz")
print("spread across estimators ≳ 20 % => inspect the FM PSD before quoting.")
