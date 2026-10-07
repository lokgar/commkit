"""Tests for estimate_linewidth and its methods.

The phase-trajectory methods are checked against a synthetic Wiener phase of
known linewidth with calibrated AWGN, the DSH methods against beats
synthesized from one (``Δφ(t) = φ(t) - φ(t-τ_d)`` on an AOM carrier). Inputs are built on the active backend via the ``xp``
fixture.
"""

import numpy as np
import pytest

from commkit import analysis
from commkit.analysis import (
    BetaSeparation,
    DshFmPsd,
    DshIncrement,
    DshLorentzian,
    IncrementSlope,
    IncrementSubtract,
)
from commkit.backend import to_device
from commkit.core import Signal
from commkit.impairments import apply_awgn, generate_phase_noise
from tests.common.signals import make_dsh_beat, make_wiener_phase

R = 32e9  # symbol rate (Baud)
T = 1.0 / R
FS = 500e6  # DSH beat sampling rate (Hz)


def _wiener_phase(linewidth, n, seed=0):
    """Discrete Wiener phase walk at the symbol rate (NumPy, float64)."""
    return make_wiener_phase(
        num_symbols=n, linewidth=linewidth, sample_rate=R, seed=seed, dtype=np.float64
    )


class TestLinewidthEstimation:
    """Linewidth estimation via increment slope/subtract and beta separation."""

    def test_increment_slope_awgn_free(self, xp):
        """Lag-slope linewidth must reject AWGN (intercept) and recover Δν."""
        n = 1 << 18
        dnu = 2e6
        phi = _wiener_phase(dnu, n, seed=7)
        rng = np.random.default_rng(8)
        sigma2 = 10 ** (-25 / 10)  # heavy AWGN angle noise
        awgn = rng.normal(0, np.sqrt(sigma2 / 2), n)  # small-angle phase error
        res = analysis.estimate_linewidth(
            xp.asarray(phi + awgn), IncrementSlope(), sampling_rate=R
        )
        assert res.value == pytest.approx(dnu, rel=0.10)
        # The fitted intercept ≈ 2σ_φ² ≈ σ_n² for unit-power QPSK.
        assert res.awgn_var == pytest.approx(sigma2, rel=0.30)

    def test_increment_subtract_noise_free(self, xp):
        n = 1 << 18
        dnu = 1.5e6
        phi = _wiener_phase(dnu, n, seed=11)
        res = analysis.estimate_linewidth(
            xp.asarray(phi), IncrementSubtract(noise_var=0.0), sampling_rate=R
        )
        assert res.value == pytest.approx(dnu, rel=0.10)

    def test_beta_separation_floor(self, xp):
        n = 1 << 19
        dnu = 1.5e6
        phi = _wiener_phase(dnu, n, seed=13)
        out = analysis.estimate_linewidth(
            xp.asarray(phi),
            BetaSeparation(nperseg=1 << 13, f_min=5e6, f_max=2e8),
            sampling_rate=R,
        )
        # White-FM floor Δν = π·S_f is the robust estimator at high baud.
        assert out.linewidth_floor == pytest.approx(dnu, rel=0.10)

    def test_beta_area_recovers_white_fm(self, xp):
        """β-area is exact for white FM when the line crossing f_c is resolved.

        The line is constructed so a flat plateau Δν/π integrated up to
        f_c = πΔν/(8ln2) returns Δν: √(8ln2 · (Δν/π)·f_c) = Δν. Sample slowly
        enough that f_c spans many Welch bins.
        """
        fs, dnu, n = 100e6, 2e5, 1 << 20  # f_c ≈ 113 kHz, bin ≈ 6.1 kHz
        phi = to_device(
            generate_phase_noise(
                num_samples=n, sampling_rate=fs, linewidth=dnu, rng=19
            ),
            "cpu",
        )
        out = analysis.estimate_linewidth(
            xp.asarray(phi),
            BetaSeparation(nperseg=1 << 14, f_max=5e6),
            sampling_rate=fs,
        )
        assert out.value == pytest.approx(dnu, rel=0.15)

    def test_beta_area_region_is_line_gated(self, xp):
        """'above' is the exact integrated region: S_f > β within the band fence.

        The region is gated by the line, not by [f_min, f_max]; the area must be
        reproducible from the returned mask alone.
        """
        fs, n = 100e6, 1 << 19
        phi = to_device(
            generate_phase_noise(
                num_samples=n,
                sampling_rate=fs,
                linewidth=1e5,
                flicker=4e9,
                flicker_f_min=1e3,
                rng=23,
            ),
            "cpu",
        )
        out = analysis.estimate_linewidth(
            xp.asarray(phi),
            BetaSeparation(nperseg=1 << 14, f_max=5e6),
            sampling_rate=fs,
        )
        f, s_f, beta, above = out.f, out.S_f, out.beta_line, out.above
        fmin, fmax = out.band
        assert above.dtype == bool and above.shape == f.shape
        expected = (s_f > beta) & (f >= fmin) & (f <= fmax)
        np.testing.assert_array_equal(above, expected)
        # The reported area is exactly the trapezoid over the masked PSD.
        area = float(np.trapezoid(np.where(above, s_f, 0.0), f))
        assert out.area_hz2 == pytest.approx(area, rel=1e-9)
        # With flicker the plateau crossing is noisy: the gate may open/close
        # several times - the region is a union of intervals, not one band.
        assert above.any()

    def test_beta_floor_auto_dodges_awgn_tail(self, xp):
        """Unfenced floor auto-detects the plateau, excluding the AWGN f² tail.

        Previously the unfenced floor median ran over the full band and was
        AWGN-dominated; the plateau detector must recover the white-FM level
        without a manual f_max.
        """
        n, dnu = 1 << 19, 1.5e6
        phi = _wiener_phase(dnu, n, seed=13)
        rng = np.random.default_rng(8)
        phi = phi + rng.normal(0, np.sqrt(10 ** (-2.0) / 2), n)  # AWGN angle noise
        out = analysis.estimate_linewidth(
            xp.asarray(phi), BetaSeparation(nperseg=1 << 13), sampling_rate=R
        )
        assert out.linewidth_floor == pytest.approx(dnu, rel=0.15)
        # The tail region is excluded: the used band ends well below Nyquist.
        f_used = out.f[out.used]
        assert float(f_used[-1]) < 0.5 * R / 2

    def test_increment_returns_plot_data(self, xp):
        """The returned fit data plots directly: Var(k) fit (slope), point (subtract)."""
        from commkit import plotting

        phi = xp.asarray(_wiener_phase(2e6, 1 << 14, seed=17))
        res = analysis.estimate_linewidth(phi, IncrementSlope(), sampling_rate=R)
        assert res.var.shape == (1, res.lag_s.size)
        plotting.plot_increment_variance(res)
        res = analysis.estimate_linewidth(
            phi, IncrementSubtract(noise_var=0.0), sampling_rate=R
        )
        plotting.plot_increment_variance(res)

    def test_increment_subtract_known_noise(self, xp):
        """Subtracting the known AWGN term recovers Δν under heavy noise.

        Angle noise of variance σ²/2 adds 2·σ²/2 = σ² to the lag-1 increment
        variance, the term ``noise_var=σ²`` removes.
        """
        n, dnu = 1 << 18, 2e6
        sigma2 = 10 ** (-25 / 10)
        phi = _wiener_phase(dnu, n, seed=7)
        phi = phi + np.random.default_rng(8).normal(0, np.sqrt(sigma2 / 2), n)
        res = analysis.estimate_linewidth(
            xp.asarray(phi), IncrementSubtract(noise_var=sigma2), sampling_rate=R
        )
        assert res.value == pytest.approx(dnu, rel=0.10)
        assert res.awgn_var == pytest.approx(sigma2)

    def test_increment_subtract_per_channel_noise(self, xp):
        """One noise variance per channel; a count mismatch raises."""
        n, sigma2 = 1 << 17, np.array([10**-2.5, 10**-2.0])
        rng = np.random.default_rng(3)
        phi = np.stack(
            [
                _wiener_phase(2e6, n, seed=s) + rng.normal(0, np.sqrt(v / 2), n)
                for s, v in zip((1, 2), sigma2, strict=True)
            ]
        )
        res = analysis.estimate_linewidth(
            xp.asarray(phi), IncrementSubtract(noise_var=sigma2), sampling_rate=R
        )
        assert res.value == pytest.approx([2e6, 2e6], rel=0.15)
        with pytest.raises(ValueError, match="3 values for 2 channels"):
            analysis.estimate_linewidth(
                xp.asarray(phi),
                IncrementSubtract(noise_var=[1e-3, 1e-3, 1e-3]),
                sampling_rate=R,
            )


class TestDshLinewidth:
    """DSH linewidth via the FM-PSD, increment and Lorentzian methods."""

    def test_fm_psd(self, xp):
        n, m = 1 << 20, 500
        z, _ = make_dsh_beat(2e6, n, m, 80e6, snr_db=25, seed=4)
        res = analysis.estimate_linewidth(
            xp.asarray(z), DshFmPsd(delay=m / FS, nperseg=1 << 14), sampling_rate=FS
        )
        assert res.value == pytest.approx(2e6, rel=0.15)
        # Notch bins are masked and NaN.
        k1 = int(np.argmin(np.abs(res.f - FS / m)))
        assert not res.valid[k1]
        assert np.isnan(res.S_f[k1])
        # Auto plateau detection spans lobes: the band extends past the first
        # notch, and the 'used' mask reproduces the reported band extent.
        assert res.band[1] > FS / m
        f_used = res.f[res.used]
        assert res.band == (float(f_used[0]), float(f_used[-1]))

    def test_fm_psd_auto_band_dodges_flicker(self, xp):
        """Auto plateau detection excludes a rising low-frequency 1/f region.

        With the flicker corner inside the first lobe, the naive first-lobe
        median is strongly inflated; the auto-detected plateau reads the white
        part.
        """
        dnu, n, m = 2e5, 1 << 21, 2450  # τ_d = 4.9 µs, corner ≈ 63 kHz
        phi = to_device(
            generate_phase_noise(
                num_samples=n + m,
                sampling_rate=FS,
                linewidth=dnu,
                flicker=4e9,
                flicker_f_min=1e3,
                rng=45,
            ),
            "cpu",
        )
        z, _ = analysis.dsh_beat(phi, sampling_rate=FS, delay=m / FS, f_shift=80e6)
        z = xp.asarray(apply_awgn(z, sps=1, esn0_db=25, rng=145))
        auto = analysis.estimate_linewidth(
            z, DshFmPsd(delay=m / FS, f_shift=80e6, nperseg=1 << 15), sampling_rate=FS
        )
        naive = analysis.estimate_linewidth(
            z,
            DshFmPsd(delay=m / FS, f_shift=80e6, nperseg=1 << 15, f_max=FS / m),
            sampling_rate=FS,
        )
        assert auto.value == pytest.approx(dnu, rel=0.25)
        assert abs(auto.value - dnu) < abs(naive.value - dnu)
        # The rising flicker region is excluded: the plateau starts above the
        # first Welch bin.
        assert auto.band[0] > FS / (1 << 15)

    def test_fm_psd_real_capture_band_capped(self, xp):
        """Real (single-PD) captures: auto band stops at min(f_aom, nyq - f_aom).

        Beyond the receiver's FM detection bandwidth the Hilbert-derived beat has
        no sideband support and the deconvolved PSD reads fake-low; the plateau
        detector must not latch onto it. Includes a realistic calibrated-delay
        error (regression: this failed with the band latched above 150 MHz).
        """
        dnu, m, f_aom = 2e5, 2450, 80e6  # τ_d = 4.9 µs true
        phi = to_device(
            generate_phase_noise(
                num_samples=(1 << 20) + m, sampling_rate=FS, linewidth=dnu, rng=42
            ),
            "cpu",
        )
        z, _ = analysis.dsh_beat(phi, sampling_rate=FS, delay=m / FS, f_shift=f_aom)
        beat = apply_awgn(z, sps=1, esn0_db=25, rng=1).real
        tau_cal = 4.8978e-6  # -0.05 % delay-calibration error
        res = analysis.estimate_linewidth(
            xp.asarray(beat),
            DshFmPsd(delay=tau_cal, f_shift=f_aom, nperseg=1 << 15),
            sampling_rate=FS,
        )
        assert res.band[1] <= min(f_aom, FS / 2 - f_aom)
        assert res.value == pytest.approx(dnu, rel=0.2)

    def test_fm_psd_manual_fence_is_literal(self, xp):
        """Explicit f_min/f_max bypass auto detection: the band is the fence."""
        n, m = 1 << 20, 500
        z, _ = make_dsh_beat(2e6, n, m, 80e6, snr_db=25, seed=4)
        res = analysis.estimate_linewidth(
            xp.asarray(z),
            DshFmPsd(delay=m / FS, f_shift=80e6, nperseg=1 << 14, f_max=FS / m),
            sampling_rate=FS,
        )
        f_used = res.f[res.used]
        assert float(f_used[-1]) <= FS / m
        assert res.value == pytest.approx(2e6, rel=0.2)

    def test_fm_psd_coherent_regime(self, xp):
        """Short delay (τ_d ≪ τ_c): the discriminator regime, Lorentzian invalid."""
        n, m = 1 << 20, 500
        z, _ = make_dsh_beat(50e3, n, m, 80e6, snr_db=None, seed=5)
        res = analysis.estimate_linewidth(
            xp.asarray(z), DshFmPsd(delay=m / FS, nperseg=1 << 14), sampling_rate=FS
        )
        assert np.pi * 50e3 * (m / FS) < 1.0  # deeply coherent
        assert res.value == pytest.approx(50e3, rel=0.15)

    def test_increment_awgn_immune(self, xp):
        n, m = 1 << 20, 500
        z, _ = make_dsh_beat(2e6, n, m, 80e6, snr_db=20, seed=6)
        res = analysis.estimate_linewidth(
            xp.asarray(z), DshIncrement(delay=m / FS), sampling_rate=FS
        )
        assert res.value == pytest.approx(2e6, rel=0.15)
        assert res.dphi_var == pytest.approx(2.0 * np.pi * 2e6 * m / FS, rel=0.2)

    def test_lorentzian_incoherent(self, xp):
        n, m = 1 << 20, 2000  # τ_d = 4 µs, τ_d/τ_c ≈ 63
        dnu = 5e6
        z, _ = make_dsh_beat(dnu, n, m, 80e6, snr_db=30, seed=7)
        res = analysis.estimate_linewidth(
            xp.asarray(z), DshLorentzian(delay=m / FS), sampling_rate=FS
        )
        assert res.value == pytest.approx(dnu, rel=0.20)
        assert res.linewidth_3db == pytest.approx(dnu, rel=0.20)
        # Pure white FM -> Lorentzian wings: W₂₀/W₃ ≈ √99.
        assert res.lineshape_ratio == pytest.approx(np.sqrt(99.0), rel=0.25)
        assert res.coherence_factor > 6.0
        assert res.f_peak == pytest.approx(80e6, abs=5e5)

    def test_mimo_channels(self, xp):
        n, m = 1 << 19, 500
        z0, _ = make_dsh_beat(1e6, n, m, 80e6, snr_db=None, seed=8)
        z1, _ = make_dsh_beat(3e6, n, m, 80e6, snr_db=None, seed=9)
        z = np.stack([z0, z1])
        res = analysis.estimate_linewidth(
            xp.asarray(z), DshIncrement(delay=m / FS, f_shift=80e6), sampling_rate=FS
        )
        assert res.value.shape == (2,)
        assert res.value[0] == pytest.approx(1e6, rel=0.2)
        assert res.value[1] == pytest.approx(3e6, rel=0.2)

    def test_delay_below_one_sample_raises(self, xp):
        z, _ = make_dsh_beat(1e6, 1 << 12, 100, 80e6, seed=10)
        with pytest.raises(ValueError, match="unresolvable"):
            analysis.estimate_linewidth(
                xp.asarray(z), DshFmPsd(delay=0.1 / FS), sampling_rate=FS
            )

    def test_results_plot_directly(self, xp):
        """Every method returns the data its diagnostic plot needs."""
        from commkit import plotting

        n, m = 1 << 16, 200
        z, _ = make_dsh_beat(2e6, n, m, 80e6, snr_db=25, seed=12)
        z = xp.asarray(z)
        fm = analysis.estimate_linewidth(z, DshFmPsd(delay=m / FS), sampling_rate=FS)
        plotting.plot_frequency_noise_psd(fm)
        inc = analysis.estimate_linewidth(
            z, DshIncrement(delay=m / FS), sampling_rate=FS
        )
        plotting.plot_increment_variance(inc)
        lor = analysis.estimate_linewidth(
            z, DshLorentzian(delay=m / FS), sampling_rate=FS
        )
        plotting.plot_dsh_beat_psd(lor)

    def test_signal_input(self, xp):
        n, m = 1 << 18, 500
        z, _ = make_dsh_beat(2e6, n, m, 80e6, snr_db=25, seed=4)
        sig = Signal(samples=xp.asarray(z), sampling_rate=FS, symbol_rate=FS)
        method = DshFmPsd(delay=m / FS, nperseg=1 << 13)
        res_sig = analysis.estimate_linewidth(sig, method)
        res_arr = analysis.estimate_linewidth(xp.asarray(z), method, sampling_rate=FS)
        assert res_sig.value == pytest.approx(res_arr.value)

    def test_conflicting_sampling_rate_raises(self, xp):
        """sampling_rate is a fact: a value that disagrees with the Signal raises."""
        z, _ = make_dsh_beat(2e6, 1 << 12, 50, 80e6, snr_db=None, seed=1)
        sig = Signal(samples=xp.asarray(z), sampling_rate=FS, symbol_rate=FS)
        with pytest.raises(ValueError, match="conflicts"):
            analysis.estimate_linewidth(
                sig, DshFmPsd(delay=50 / FS), sampling_rate=FS / 2
            )


class TestEstimateLinewidthContract:
    """Dispatch, input kinds and method validation."""

    def test_result_is_frozen_and_names_its_method(self, xp):
        method = IncrementSlope(lags=(1, 2, 3))
        phi = xp.asarray(_wiener_phase(1e6, 1 << 12, seed=1))
        res = analysis.estimate_linewidth(phi, method, sampling_rate=R)
        assert isinstance(res, analysis.LinewidthEstimate)
        assert res.method is method
        assert res.f is None and res.psd is None  # other methods' fields
        with pytest.raises(AttributeError):
            res.value = 0.0  # type: ignore[misc]

    def test_unknown_method_raises(self, xp):
        with pytest.raises(TypeError, match="IncrementSlope"):
            analysis.estimate_linewidth(xp.zeros(64), "slope", sampling_rate=R)

    def test_phase_methods_reject_complex_input(self, xp):
        """A beat record passed to a phase method raises instead of guessing."""
        z = xp.ones(1024, dtype=xp.complex128)
        for method in (IncrementSlope(), IncrementSubtract(), BetaSeparation()):
            with pytest.raises(ValueError, match="real phase trajectory"):
                analysis.estimate_linewidth(z, method, sampling_rate=R)

    @pytest.mark.parametrize(
        "method",
        [DshIncrement(delay=500 / FS, f_shift=80e6), DshLorentzian(delay=500 / FS)],
    )
    def test_phase_trajectory_to_dsh_method_warns(self, xp, caplog, method):
        """A real phase trajectory cannot be told from a beat by type; its
        power near DC flags it, with and without a known carrier."""
        phi = to_device(
            generate_phase_noise(
                num_samples=1 << 16, sampling_rate=FS, linewidth=2e6, rng=3
            ),
            "cpu",
        )
        with caplog.at_level("WARNING", logger="commkit"):
            analysis.estimate_linewidth(xp.asarray(phi), method, sampling_rate=FS)
        assert "looks like a phase trajectory" in caplog.text

    @pytest.mark.parametrize(
        ("f_shift", "snr_db", "offset", "real"),
        [
            (80e6, 0.0, 0.0, True),  # low SNR
            (None, 25.0, 3.0, True),  # strong DC offset, carrier unknown
            (None, 10.0, 0.0, True),
            (80e6, 25.0, 0.0, False),  # IQ capture
        ],
    )
    def test_beats_do_not_warn(self, xp, caplog, f_shift, snr_db, offset, real):
        z, _ = make_dsh_beat(2e6, 1 << 16, 500, 80e6, snr_db=snr_db, seed=2)
        z = (z.real if real else z) + offset
        with caplog.at_level("WARNING", logger="commkit"):
            analysis.estimate_linewidth(
                xp.asarray(z),
                DshIncrement(delay=500 / FS, f_shift=f_shift),
                sampling_rate=FS,
            )
        assert "phase trajectory" not in caplog.text

    def test_array_needs_sampling_rate(self, xp):
        with pytest.raises(ValueError, match="sampling_rate"):
            analysis.estimate_linewidth(xp.zeros(64), IncrementSlope())

    @pytest.mark.parametrize(
        ("build", "match"),
        [
            (lambda: IncrementSlope(lags=(1, 1, 0)), "two distinct lags"),
            (lambda: IncrementSlope(edge_trim=-1), "edge_trim"),
            (lambda: IncrementSubtract(noise_var=-1.0), "noise_var"),
            (lambda: BetaSeparation(f_min=2e6, f_max=1e6), "f_min"),
            (lambda: BetaSeparation(nperseg=1), "nperseg"),
            (lambda: DshFmPsd(delay=0.0), "positive"),
            (lambda: DshIncrement(delay=1e-6, lags=(0, 5)), "positive"),
            (lambda: DshIncrement(delay=1e-6, lags=(5, 5)), "two distinct"),
            (lambda: DshLorentzian(delay=-1e-6), "positive"),
            (lambda: DshLorentzian(delay=1e-6, level_db=0.0), "level_db"),
        ],
    )
    def test_invalid_methods_raise_on_construction(self, build, match):
        with pytest.raises(ValueError, match=match):
            build()

    def test_subtract_arrays_are_read_only(self):
        method = IncrementSubtract(noise_var=[1e-3, 2e-3], reference=np.ones(8))
        assert not method.noise_var.flags.writeable
        assert not method.reference.flags.writeable
