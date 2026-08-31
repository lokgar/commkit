"""Tests for the linewidth estimators (increment-slope, β-separation).

Each estimator is checked against a synthetic Wiener phase of known linewidth
with calibrated AWGN.  Inputs are built on the active backend via the ``xp``
fixture.
"""

import numpy as np
import pytest

from commkit import analysis
from commkit.backend import to_device
from commkit.impairments import generate_phase_noise

R = 32e9  # symbol rate (Baud)
T = 1.0 / R


def _wiener_phase(linewidth, n, seed=0):
    """Discrete Wiener phase walk at the symbol rate (NumPy, float64)."""
    return to_device(generate_phase_noise(n, R, linewidth=linewidth, seed=seed), "cpu")


def test_linewidth_increment_slope_awgn_free(xp):
    """Lag-slope linewidth must reject AWGN (intercept) and recover Δν."""
    n = 1 << 18
    dnu = 2e6
    phi = _wiener_phase(dnu, n, seed=7)
    rng = np.random.default_rng(8)
    sigma2 = 10 ** (-25 / 10)  # heavy AWGN angle noise
    awgn = rng.normal(0, np.sqrt(sigma2 / 2), n)  # small-angle phase error
    res = analysis.linewidth_increment(xp.asarray(phi + awgn), R, method="slope")
    assert res["linewidth"] == pytest.approx(dnu, rel=0.10)
    # The fitted intercept ≈ 2σ_φ² ≈ σ_n² for unit-power QPSK.
    assert res["awgn_var"] == pytest.approx(sigma2, rel=0.30)


def test_linewidth_increment_subtract_matches_slope(xp):
    n = 1 << 18
    dnu = 1.5e6
    phi = _wiener_phase(dnu, n, seed=11)
    res = analysis.linewidth_increment(
        xp.asarray(phi), R, method="subtract", noise_var=0.0
    )
    assert res["linewidth"] == pytest.approx(dnu, rel=0.10)


def test_linewidth_beta_separation_floor(xp):
    n = 1 << 19
    dnu = 1.5e6
    phi = _wiener_phase(dnu, n, seed=13)
    out = analysis.linewidth_beta_separation(
        xp.asarray(phi), R, nperseg=1 << 13, f_min=5e6, f_max=2e8
    )
    # White-FM floor Δν = π·S_f is the robust estimator at high baud.
    assert out["linewidth_floor"] == pytest.approx(dnu, rel=0.10)


def test_linewidth_beta_area_recovers_white_fm(xp):
    """β-area is exact for white FM when the line crossing f_c is resolved.

    The line is constructed so a flat plateau Δν/π integrated up to
    f_c = πΔν/(8ln2) returns Δν: √(8ln2 · (Δν/π)·f_c) = Δν.  Sample slowly
    enough that f_c spans many Welch bins.
    """
    fs, dnu, n = 100e6, 2e5, 1 << 20  # f_c ≈ 113 kHz, bin ≈ 6.1 kHz
    phi = to_device(generate_phase_noise(n, fs, linewidth=dnu, seed=19), "cpu")
    out = analysis.linewidth_beta_separation(
        xp.asarray(phi), fs, nperseg=1 << 14, f_max=5e6
    )
    assert out["linewidth"] == pytest.approx(dnu, rel=0.15)


def test_linewidth_beta_area_region_is_line_gated(xp):
    """'above' is the exact integrated region: S_f > β within the band fence.

    The region is gated by the line, not by [f_min, f_max]; the area must be
    reproducible from the returned mask alone.
    """
    fs, n = 100e6, 1 << 19
    phi = to_device(
        generate_phase_noise(
            n, fs, linewidth=1e5, flicker=4e9, flicker_f_min=1e3, seed=23
        ),
        "cpu",
    )
    out = analysis.linewidth_beta_separation(
        xp.asarray(phi), fs, nperseg=1 << 14, f_max=5e6
    )
    f, s_f, beta, above = out["f"], out["S_f"], out["beta_line"], out["above"]
    fmin, fmax = out["band"]
    assert above.dtype == bool and above.shape == f.shape
    expected = (s_f > beta) & (f >= fmin) & (f <= fmax)
    np.testing.assert_array_equal(above, expected)
    # The reported area is exactly the trapezoid over the masked PSD.
    area = float(np.trapezoid(np.where(above, s_f, 0.0), f))
    assert out["area_hz2"] == pytest.approx(area, rel=1e-9)
    # With flicker the plateau crossing is noisy: the gate may open/close
    # several times - the region is a union of intervals, not one band.
    assert above.any()


def test_linewidth_beta_floor_auto_dodges_awgn_tail(xp):
    """Unfenced floor auto-detects the plateau, excluding the AWGN f² tail.

    Previously the unfenced floor median ran over the full band and was
    AWGN-dominated; the plateau detector must recover the white-FM level
    without a manual f_max.
    """
    n, dnu = 1 << 19, 1.5e6
    phi = _wiener_phase(dnu, n, seed=13)
    rng = np.random.default_rng(8)
    phi = phi + rng.normal(0, np.sqrt(10 ** (-2.0) / 2), n)  # AWGN angle noise
    out = analysis.linewidth_beta_separation(xp.asarray(phi), R, nperseg=1 << 13)
    assert out["linewidth_floor"] == pytest.approx(dnu, rel=0.15)
    # The tail region is excluded: the used band ends well below Nyquist.
    f_used = out["f"][out["used"]]
    assert float(f_used[-1]) < 0.5 * R / 2


def test_linewidth_increment_debug_plot(xp):
    """debug_plot renders the Var(k) fit (slope) and the point (subtract)."""
    from unittest.mock import patch

    import matplotlib.pyplot as plt

    phi = _wiener_phase(2e6, 1 << 14, seed=17)
    with patch("matplotlib.pyplot.show"):
        analysis.linewidth_increment(
            xp.asarray(phi), R, method="slope", debug_plot=True
        )
        analysis.linewidth_increment(
            xp.asarray(phi), R, method="subtract", noise_var=0.0, debug_plot=True
        )
    plt.close("all")
