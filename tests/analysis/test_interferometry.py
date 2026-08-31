"""Tests for delayed self-heterodyne / self-homodyne laser characterization.

Beats are synthesized from a discrete Wiener laser phase of known linewidth
(``Δφ(t) = φ(t) - φ(t-τ_d)`` modulated onto an AOM carrier) so every estimator
is checked against analytic ground truth.  Inputs are built on the active
backend via the ``xp`` fixture.
"""

import numpy as np
import pytest

from commkit import analysis
from commkit.backend import to_device
from commkit.impairments import apply_awgn, generate_phase_noise

FS = 500e6  # beat sampling rate (Hz)


def _dsh_beat(dnu, n, m, f_aom, snr_db=None, seed=0):
    """Complex DSH beat + true differential phase (NumPy, float64).

    Library forward model: ``generate_phase_noise`` -> ``analysis.dsh_beat``
    -> ``apply_awgn``, pulled to the host so tests control the device via
    ``xp.asarray``.
    """
    phi = to_device(generate_phase_noise(n + m, FS, linewidth=dnu, seed=seed), "cpu")
    z, dphi = analysis.dsh_beat(phi, FS, m / FS, f_shift=f_aom)
    if snr_db is not None:
        z = apply_awgn(z, sps=1, esn0_db=snr_db, seed=seed + 100)
    return z, dphi


def test_dsh_beat_forward_model(xp, xpt):
    """dsh_beat: unit amplitude, exact differential phase, homodyne case."""
    n, m = 1 << 14, 250
    phi = xp.asarray(
        to_device(generate_phase_noise(n + m, FS, linewidth=1e6, seed=42), "cpu")
    )
    z, dphi = analysis.dsh_beat(phi, FS, m / FS, f_shift=80e6)
    assert z.shape == (n,)
    assert dphi.shape == (n,)
    xpt.assert_allclose(xp.abs(z), xp.ones(n), atol=1e-12)
    xpt.assert_allclose(dphi, phi[m:] - phi[:-m], atol=0.0)
    # Homodyne (f_shift = 0): the beat is exp(j·Δφ) exactly.
    z0, dphi0 = analysis.dsh_beat(phi, FS, m / FS)
    xpt.assert_allclose(z0, xp.exp(1j * dphi0), atol=1e-12)


def test_dsh_beat_rejects_bad_delay(xp):
    phi = xp.zeros(100, dtype=xp.float64)
    with pytest.raises(ValueError, match="delay"):
        analysis.dsh_beat(phi, FS, 0.0)
    with pytest.raises(ValueError, match="delay"):
        analysis.dsh_beat(phi, FS, 100 / FS)


def test_dsh_phase_recovers_differential_phase(xp, xpt):
    n, m = 1 << 18, 500
    z, dphi = _dsh_beat(2e6, n, m, 80e6, snr_db=None, seed=1)
    dp, f_used = analysis.dsh_phase(xp.asarray(z), FS, f_shift=80e6)
    assert f_used == pytest.approx(80e6)
    # Constant offsets are irrelevant; the increments must match exactly.
    xpt.assert_allclose(xp.diff(dp), xp.asarray(np.diff(dphi)), atol=1e-6)


def test_dsh_phase_estimated_carrier_leaves_no_ramp(xp):
    """Kay + LS two-stage carrier removal: Var[Δφ] must match 2πΔν·τ_d."""
    n, m = 1 << 20, 500
    td = m / FS
    z, _ = _dsh_beat(2e6, n, m, 80e6, snr_db=25, seed=2)
    dp, f_hat = analysis.dsh_phase(xp.asarray(z), FS)
    assert f_hat == pytest.approx(80e6, abs=5e3)
    # A residual carrier ramp would inflate this variance by an order of
    # magnitude (LS detrend regression guard).
    assert float(xp.var(dp)) == pytest.approx(2.0 * np.pi * 2e6 * td, rel=0.15)


def test_dsh_phase_real_input_hilbert(xp):
    # Narrow line vs f_aom so the beat spectrum is one-sided (Hilbert-exact).
    n, m = 1 << 18, 500
    z, dphi = _dsh_beat(2e5, n, m, 80e6, snr_db=None, seed=3)
    dp, f_hat = analysis.dsh_phase(xp.asarray(z.real), FS, f_shift=80e6)
    edge = 1000  # Hilbert edge transients
    resid = (xp.asarray(dphi) - dp)[edge:-edge]
    err = resid - xp.mean(resid)
    assert float(xp.std(err)) < 0.05


def test_dsh_phase_real_homodyne_rejected(xp):
    x = xp.asarray(np.cos(np.linspace(0.0, 20.0, 1000)))
    with pytest.raises(ValueError, match="cannot be inverted"):
        analysis.dsh_phase(x, FS, f_shift=0.0)


def test_linewidth_dsh_fm_psd(xp):
    n, m = 1 << 20, 500
    z, _ = _dsh_beat(2e6, n, m, 80e6, snr_db=25, seed=4)
    res = analysis.linewidth_dsh(
        xp.asarray(z), FS, m / FS, method="fm_psd", nperseg=1 << 14
    )
    assert res["linewidth"] == pytest.approx(2e6, rel=0.15)
    # Notch bins are masked and NaN.
    k1 = int(np.argmin(np.abs(res["f"] - FS / m)))
    assert not res["valid"][k1]
    assert np.isnan(res["S_f"][k1])
    # Auto plateau detection spans lobes: the band extends past the first
    # notch, and the 'used' mask reproduces the reported band extent.
    assert res["band"][1] > FS / m
    f_used = res["f"][res["used"]]
    assert res["band"] == (float(f_used[0]), float(f_used[-1]))


def test_linewidth_dsh_fm_psd_auto_band_dodges_flicker(xp):
    """Auto plateau detection excludes a rising low-frequency 1/f region.

    With the flicker corner inside the first lobe, the naive first-lobe
    median is strongly inflated; the auto-detected plateau reads the white
    part.
    """
    dnu, n, m = 2e5, 1 << 21, 2450  # τ_d = 4.9 µs, corner ≈ 63 kHz
    phi = to_device(
        generate_phase_noise(
            n + m, FS, linewidth=dnu, flicker=4e9, flicker_f_min=1e3, seed=45
        ),
        "cpu",
    )
    z, _ = analysis.dsh_beat(phi, FS, m / FS, f_shift=80e6)
    z = apply_awgn(z, sps=1, esn0_db=25, seed=145)
    auto = analysis.linewidth_dsh(
        xp.asarray(z), FS, m / FS, f_shift=80e6, method="fm_psd", nperseg=1 << 15
    )
    naive = analysis.linewidth_dsh(
        xp.asarray(z),
        FS,
        m / FS,
        f_shift=80e6,
        method="fm_psd",
        nperseg=1 << 15,
        f_max=FS / m,
    )
    assert auto["linewidth"] == pytest.approx(dnu, rel=0.25)
    assert abs(auto["linewidth"] - dnu) < abs(naive["linewidth"] - dnu)
    # The rising flicker region is excluded: the plateau starts above the
    # first Welch bin.
    assert auto["band"][0] > FS / (1 << 15)


def test_linewidth_dsh_fm_psd_real_capture_band_capped(xp):
    """Real (single-PD) captures: auto band stops at min(f_aom, nyq - f_aom).

    Beyond the receiver's FM detection bandwidth the Hilbert-derived beat has
    no sideband support and the deconvolved PSD reads fake-low; the plateau
    detector must not latch onto it.  Includes a realistic calibrated-delay
    error (regression: this failed with the band latched above 150 MHz).
    """
    dnu, m, f_aom = 2e5, 2450, 80e6  # τ_d = 4.9 µs true
    phi = to_device(
        generate_phase_noise((1 << 20) + m, FS, linewidth=dnu, seed=42), "cpu"
    )
    z, _ = analysis.dsh_beat(phi, FS, m / FS, f_shift=f_aom)
    beat = apply_awgn(z, sps=1, esn0_db=25, seed=1).real
    tau_cal = 4.8978e-6  # -0.05 % delay-calibration error
    res = analysis.linewidth_dsh(
        xp.asarray(beat), FS, tau_cal, f_shift=f_aom, method="fm_psd", nperseg=1 << 15
    )
    assert res["band"][1] <= min(f_aom, FS / 2 - f_aom)
    assert res["linewidth"] == pytest.approx(dnu, rel=0.2)


def test_linewidth_dsh_fm_psd_manual_fence_is_literal(xp):
    """Explicit f_min/f_max bypass auto detection: the band is the fence."""
    n, m = 1 << 20, 500
    z, _ = _dsh_beat(2e6, n, m, 80e6, snr_db=25, seed=4)
    res = analysis.linewidth_dsh(
        xp.asarray(z),
        FS,
        m / FS,
        f_shift=80e6,
        method="fm_psd",
        nperseg=1 << 14,
        f_max=FS / m,
    )
    f_used = res["f"][res["used"]]
    assert float(f_used[-1]) <= FS / m
    assert res["linewidth"] == pytest.approx(2e6, rel=0.2)


def test_linewidth_dsh_fm_psd_coherent_regime(xp):
    """Short delay (τ_d ≪ τ_c): the discriminator regime, Lorentzian invalid."""
    n, m = 1 << 20, 500
    z, _ = _dsh_beat(50e3, n, m, 80e6, snr_db=None, seed=5)
    res = analysis.linewidth_dsh(
        xp.asarray(z), FS, m / FS, method="fm_psd", nperseg=1 << 14
    )
    assert np.pi * 50e3 * (m / FS) < 1.0  # deeply coherent
    assert res["linewidth"] == pytest.approx(50e3, rel=0.15)


def test_linewidth_dsh_increment_awgn_immune(xp):
    n, m = 1 << 20, 500
    z, _ = _dsh_beat(2e6, n, m, 80e6, snr_db=20, seed=6)
    res = analysis.linewidth_dsh(xp.asarray(z), FS, m / FS, method="increment")
    assert res["linewidth"] == pytest.approx(2e6, rel=0.15)
    assert res["dphi_var"] == pytest.approx(2.0 * np.pi * 2e6 * m / FS, rel=0.2)


def test_linewidth_dsh_lorentzian_incoherent(xp):
    n, m = 1 << 20, 2000  # τ_d = 4 µs, τ_d/τ_c ≈ 63
    dnu = 5e6
    z, _ = _dsh_beat(dnu, n, m, 80e6, snr_db=30, seed=7)
    res = analysis.linewidth_dsh(xp.asarray(z), FS, m / FS, method="lorentzian")
    assert res["linewidth"] == pytest.approx(dnu, rel=0.20)
    assert res["linewidth_3db"] == pytest.approx(dnu, rel=0.20)
    # Pure white FM -> Lorentzian wings: W₂₀/W₃ ≈ √99.
    assert res["lineshape_ratio"] == pytest.approx(np.sqrt(99.0), rel=0.25)
    assert res["coherence_factor"] > 6.0
    assert res["f_peak"] == pytest.approx(80e6, abs=5e5)


def test_linewidth_dsh_mimo_channels(xp):
    n, m = 1 << 19, 500
    z0, _ = _dsh_beat(1e6, n, m, 80e6, snr_db=None, seed=8)
    z1, _ = _dsh_beat(3e6, n, m, 80e6, snr_db=None, seed=9)
    z = np.stack([z0, z1])
    res = analysis.linewidth_dsh(
        xp.asarray(z), FS, m / FS, f_shift=80e6, method="increment"
    )
    assert res["linewidth"].shape == (2,)
    assert res["linewidth"][0] == pytest.approx(1e6, rel=0.2)
    assert res["linewidth"][1] == pytest.approx(3e6, rel=0.2)


def test_linewidth_dsh_rejects_bad_args(xp):
    z, _ = _dsh_beat(1e6, 1 << 12, 100, 80e6, seed=10)
    z = xp.asarray(z)
    with pytest.raises(ValueError, match="positive"):
        analysis.linewidth_dsh(z, FS, -1e-6)
    with pytest.raises(ValueError, match="unresolvable"):
        analysis.linewidth_dsh(z, FS, 0.1 / FS)
    with pytest.raises(ValueError, match="Unknown method"):
        analysis.linewidth_dsh(z, FS, 100 / FS, method="nope")


def test_linewidth_dsh_debug_plots(xp):
    """debug_plot paths render (and close) without error for every method."""
    from unittest.mock import patch

    import matplotlib.pyplot as plt

    n, m = 1 << 16, 200
    z, _ = _dsh_beat(2e6, n, m, 80e6, snr_db=25, seed=12)
    z = xp.asarray(z)
    with patch("matplotlib.pyplot.show"):
        for method in ("fm_psd", "increment", "lorentzian"):
            analysis.linewidth_dsh(z, FS, m / FS, method=method, debug_plot=True)
        dp, _ = analysis.dsh_phase(z, FS, f_shift=80e6)
        analysis.dsh_fm_noise_psd(dp, FS, m / FS, debug_plot=True)
    plt.close("all")
