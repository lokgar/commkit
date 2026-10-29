"""Tests for nonlinear fiber propagation (apply_fiber_propagation).

Expectations are derived independently of the split-step solver: the
fundamental soliton, the exact Kerr phase of a continuous wave, the linear
dispersion operator, the noise-figure formula of an amplifier and the
Maxwellian mean of the differential group delay.
"""

import math

import numpy as np
import pytest

from commkit.backend import to_device
from commkit.core import Signal
from commkit.impairments import (
    Fiber,
    apply_chromatic_dispersion,
    apply_fiber_propagation,
)

FS = 64e9
C_LIGHT = 299792458.0
H_PLANCK = 6.62607015e-34


def _field(num_channels, n=4096, seed=1, dtype=np.complex64):
    """Band-limited complex Gaussian field (occupies |f| < f_s/4)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((num_channels, n)) + 1j * rng.standard_normal(
        (num_channels, n)
    )
    spectrum = np.fft.fft(x, axis=-1)
    spectrum[:, np.abs(np.fft.fftfreq(n)) > 0.25] = 0.0
    x = np.fft.ifft(spectrum, axis=-1)
    return (x / np.sqrt(np.mean(np.abs(x) ** 2))).astype(dtype)


def _dbm(power_w):
    return 10.0 * math.log10(power_w / 1e-3)


def _rel(a, b):
    a, b = np.asarray(to_device(a, "cpu")), np.asarray(to_device(b, "cpu"))
    return float(np.sqrt(np.mean(np.abs(a - b) ** 2) / np.mean(np.abs(b) ** 2)))


class TestLinearLimits:
    def test_dispersion_only_matches_the_linear_channel(self, xp):
        """γ = 0, no loss: the split-step solver is the dispersion operator."""
        x = xp.asarray(_field(2))
        fiber = Fiber(attenuation_db_km=0.0, nonlinearity_per_w_km=0.0)
        y = apply_fiber_propagation(
            x,
            center_wavelength_nm=1550.0,
            fiber=fiber,
            span_length_km=80.0,
            num_spans=2,
            launch_power_dbm=0.0,
            steps_per_span=7,
            sampling_rate=FS,
        )
        ref = apply_chromatic_dispersion(
            x,
            dispersion_ps_nm_km=17.0,
            fiber_length_km=160.0,
            center_wavelength_nm=1550.0,
            sampling_rate=FS,
        )
        assert _rel(y, ref) < 1e-5

    def test_amplifiers_restore_the_span_loss(self, xp):
        """With loss, each amplifier's gain e^{αL} restores the span exactly."""
        x = xp.asarray(_field(1)[0])
        kw = dict(
            center_wavelength_nm=1550.0,
            span_length_km=80.0,
            num_spans=3,
            launch_power_dbm=0.0,
            steps_per_span=11,
            sampling_rate=FS,
        )
        lossy = apply_fiber_propagation(
            x, fiber=Fiber(attenuation_db_km=0.25, nonlinearity_per_w_km=0.0), **kw
        )
        lossless = apply_fiber_propagation(
            x, fiber=Fiber(attenuation_db_km=0.0, nonlinearity_per_w_km=0.0), **kw
        )
        assert _rel(lossy, lossless) < 1e-5

    def test_dispersion_slope_is_the_third_order_term(self, xp):
        """With S given, the extra phase over the β₂-only channel is
        -β₃ ω³ L / 6, with β₃ = (S + 2D/λ)(λ²/2πc)² (dω/dλ = -2πc/λ²)."""
        n = 4096
        delta = np.zeros(n, np.complex128)
        delta[0] = 1.0
        kw = dict(
            center_wavelength_nm=1550.0,
            span_length_km=50.0,
            num_spans=1,
            launch_power_dbm=0.0,
            steps_per_span=3,
            sampling_rate=FS,
        )
        base = Fiber(attenuation_db_km=0.0, nonlinearity_per_w_km=0.0)
        sloped = Fiber(
            attenuation_db_km=0.0,
            nonlinearity_per_w_km=0.0,
            dispersion_slope_ps_nm2_km=0.057,
        )
        h2 = np.fft.fft(
            to_device(
                apply_fiber_propagation(xp.asarray(delta), fiber=base, **kw), "cpu"
            )
        )
        h3 = np.fft.fft(
            to_device(
                apply_fiber_propagation(xp.asarray(delta), fiber=sloped, **kw), "cpu"
            )
        )
        lam, d, s = 1550e-9, 17e-6, 0.057e3  # m; s/m²; s/m³
        beta3 = (s + 2.0 * d / lam) * (lam**2 / (2.0 * np.pi * 299_792_458.0)) ** 2
        omega = 2.0 * np.pi * np.fft.fftfreq(n, 1.0 / FS)
        expected = np.exp(-1j * beta3 * omega**3 * 50e3 / 6.0)
        np.testing.assert_allclose(h3 / h2, expected, atol=1e-9)


class TestKerrEffect:
    @pytest.mark.parametrize("num_channels", [1, 2])
    def test_continuous_wave_phase(self, xp, num_channels):
        """A CW field keeps its power and turns by -γ_eff P L_eff per span,
        L_eff = (1 - e^{-αL})/α, γ_eff = γ (one polarization) or 8γ/9
        (Manakov); exact for any step count."""
        n, p_w, spans = 1024, 10e-3, 3
        x = xp.ones((num_channels, n), dtype=xp.complex64)
        fiber = Fiber(dispersion_ps_nm_km=0.0)
        y = apply_fiber_propagation(
            x if num_channels == 2 else x[0],
            center_wavelength_nm=1550.0,
            fiber=fiber,
            span_length_km=80.0,
            num_spans=spans,
            launch_power_dbm=_dbm(p_w),
            steps_per_span=10,
            sampling_rate=FS,
        )
        alpha = 0.2 * math.log(10.0) / 10.0
        l_eff = -math.expm1(-alpha * 80.0) / alpha
        gamma = 1.3 * (8.0 / 9.0 if num_channels == 2 else 1.0)
        # The Kerr phase follows the power of the field: P / C per channel,
        # summed over the channels in the Manakov term.
        phase = -gamma * p_w * l_eff * spans
        y = np.asarray(to_device(y, "cpu")).reshape(num_channels, n)
        np.testing.assert_allclose(np.abs(y), 1.0, atol=1e-5)
        np.testing.assert_allclose(np.angle(y), np.angle(np.exp(1j * phase)), atol=1e-5)

    def test_fundamental_soliton_keeps_its_shape(self, xp):
        """N = 1 soliton, P0 = |β₂| / (γ T0²), in lossless anomalous fiber:
        |A| is unchanged after 5 dispersion lengths, while a quarter of its
        power (no longer a soliton) reshapes.  Holds only if the Kerr and
        dispersion signs match."""
        fs, n, t0 = 2e12, 4096, 10e-12
        beta2 = (
            17e-6 * (1550e-9) ** 2 / (2.0 * np.pi * 299_792_458.0) * 1e3
        )  # |β₂| s²/km
        gamma = 1.3
        p0 = beta2 / (gamma * t0**2)
        length = 5.0 * t0**2 / beta2  # 5 L_D, km
        t = (np.arange(n) - n // 2) / fs
        pulse = (1.0 / np.cosh(t / t0)).astype(np.complex64)
        fiber = Fiber(attenuation_db_km=0.0, nonlinearity_per_w_km=gamma)

        def run(peak):
            y = apply_fiber_propagation(
                xp.asarray(pulse),
                fiber=fiber,
                span_length_km=length,
                num_spans=1,
                launch_power_dbm=_dbm(peak * np.mean(np.abs(pulse) ** 2)),
                steps_per_span=400,
                center_wavelength_nm=1550.0,
                sampling_rate=fs,
            )
            return np.abs(np.asarray(to_device(y, "cpu")))

        assert _rel(run(p0), np.abs(pulse)) < 5e-3
        assert _rel(run(p0 / 4.0), np.abs(pulse)) > 0.1

    def test_second_order_convergence(self):
        """Halving the step quarters the error of the symmetric scheme."""
        x = _field(2, n=2048, dtype=np.complex128)
        kw = dict(
            center_wavelength_nm=1550.0,
            fiber=Fiber(),
            span_length_km=80.0,
            num_spans=1,
            launch_power_dbm=10.0,
            sampling_rate=FS,
        )
        ref = apply_fiber_propagation(x, steps_per_span=800, **kw)
        e1 = _rel(apply_fiber_propagation(x, steps_per_span=10, **kw), ref)
        e2 = _rel(apply_fiber_propagation(x, steps_per_span=20, **kw), ref)
        assert 3.0 < e1 / e2 < 5.0

    def test_lossless_propagation_conserves_power(self):
        x = _field(2, dtype=np.complex128)
        y = apply_fiber_propagation(
            x,
            center_wavelength_nm=1550.0,
            fiber=Fiber(attenuation_db_km=0.0, dispersion_slope_ps_nm2_km=0.06),
            span_length_km=100.0,
            num_spans=2,
            launch_power_dbm=8.0,
            steps_per_span=30,
            sampling_rate=FS,
        )
        assert np.mean(np.abs(y) ** 2) == pytest.approx(
            np.mean(np.abs(x) ** 2), rel=1e-12
        )


class TestAmplifierNoise:
    @pytest.mark.parametrize("num_channels", [1, 2])
    def test_ase_power(self, xp, num_channels):
        """Each of the spans' amplifiers adds (F G - 1) hν/2 · f_s per
        polarization; γ = 0 keeps the noise additive.  In the input's units
        (input power = launch power) that is P_ase · P_in / P_launch."""
        x = xp.asarray(_field(num_channels, n=1 << 16, seed=3))
        if num_channels == 1:
            x = x[0]
        nf_db, spans, launch_dbm, span_km = 5.0, 4, -3.0, 80.0
        kw = dict(
            center_wavelength_nm=1550.0,
            fiber=Fiber(nonlinearity_per_w_km=0.0),
            span_length_km=span_km,
            num_spans=spans,
            launch_power_dbm=launch_dbm,
            steps_per_span=4,
            sampling_rate=FS,
        )
        noisy = apply_fiber_propagation(x, noise_figure_db=nf_db, rng=7, **kw)
        clean = apply_fiber_propagation(x, **kw)
        noise = np.asarray(to_device(noisy - clean, "cpu"))
        gain = 10.0 ** (0.2 * span_km / 10.0)
        nu = C_LIGHT / 1550e-9
        p_ase = spans * (10.0 ** (nf_db / 10.0) * gain - 1.0) * H_PLANCK * nu / 2 * FS
        p_in = (
            float(np.mean(np.abs(np.asarray(to_device(x, "cpu"))) ** 2)) * num_channels
        )
        expected = p_ase * p_in / (1e-3 * 10.0 ** (launch_dbm / 10.0))
        measured = np.mean(np.abs(noise.reshape(num_channels, -1)) ** 2, axis=-1)
        np.testing.assert_allclose(measured, expected, rtol=0.03)

    @staticmethod
    def _ase_power(x, attenuation_db_km, nf_db, span_km=80.0, launch_dbm=-3.0):
        """Mean ASE power per sample after one span, in the input's units."""
        kw = dict(
            center_wavelength_nm=1550.0,
            fiber=Fiber(attenuation_db_km=attenuation_db_km, nonlinearity_per_w_km=0.0),
            span_length_km=span_km,
            num_spans=1,
            launch_power_dbm=launch_dbm,
            steps_per_span=4,
            sampling_rate=FS,
        )
        noisy = apply_fiber_propagation(x, noise_figure_db=nf_db, rng=11, **kw)
        clean = apply_fiber_propagation(x, **kw)
        return float(np.mean(np.abs(np.asarray(to_device(noisy - clean, "cpu"))) ** 2))

    def test_ase_power_scales_with_the_span_gain(self, xp):
        """The amplifier's gain is the span loss, G = 10^(a L / 10), so at a
        fixed noise figure F the ASE power follows (F G - 1): ratios between
        attenuations cancel hν, f_s and the launch power."""
        x = xp.asarray(_field(1, n=1 << 16, seed=5))[0]
        nf_db, span_km = 5.0, 80.0
        f = 10.0 ** (nf_db / 10.0)
        attenuations = (0.0, 0.1, 0.25)  # G = 0, 8, 20 dB
        powers = [self._ase_power(x, a, nf_db, span_km) for a in attenuations]
        for a, p in zip(attenuations[1:], powers[1:], strict=True):
            gain = 10.0 ** (a * span_km / 10.0)
            assert p / powers[0] == pytest.approx(
                (f * gain - 1.0) / (f - 1.0), rel=0.03
            )

    def test_ideal_amplifier_on_a_lossless_span_adds_no_noise(self, xp):
        """F = 1 (0 dB) and G = 1: (F G - 1) hν / 2 = 0."""
        x = xp.asarray(_field(1, n=1 << 12, seed=5))[0]
        assert self._ase_power(x, 0.0, 0.0) == 0.0

    def test_noise_needs_a_seed_only_for_its_realization(self, xp):
        x = xp.asarray(_field(2))
        kw = dict(
            center_wavelength_nm=1550.0,
            fiber=Fiber(),
            span_length_km=50.0,
            num_spans=2,
            launch_power_dbm=0.0,
            noise_figure_db=5.0,
            steps_per_span=5,
            sampling_rate=FS,
        )
        a = apply_fiber_propagation(x, rng=1, **kw)
        b = apply_fiber_propagation(x, rng=1, **kw)
        c = apply_fiber_propagation(x, rng=2, **kw)
        assert _rel(a, b) == 0.0
        assert _rel(a, c) > 0.0


class TestPMD:
    def test_mean_dgd(self):
        """Over many waveplate realizations the DGD at the carrier,
        Δτ = 2 sqrt(|det dJ/dω|) of the link's Jones matrix J(ω), averages
        pmd_ps_sqrt_km · sqrt(L) (Maxwellian)."""
        n, length = 2048, 100.0
        fiber = Fiber(
            attenuation_db_km=0.0,
            dispersion_ps_nm_km=0.0,
            nonlinearity_per_w_km=0.0,
            pmd_ps_sqrt_km=0.5,
        )
        impulses = np.zeros((2, 2, n), np.complex128)
        impulses[0, 0, 0] = impulses[1, 1, 0] = 1.0
        dw = 2.0 * np.pi * FS / n
        dgd = []
        for seed in range(64):
            cols = [
                np.fft.fft(
                    apply_fiber_propagation(
                        impulses[k],
                        center_wavelength_nm=1550.0,
                        fiber=fiber,
                        span_length_km=length / 10,
                        num_spans=10,
                        launch_power_dbm=0.0,
                        steps_per_span=20,
                        sampling_rate=FS,
                        rng=seed,
                    ),
                    axis=-1,
                )
                for k in range(2)
            ]
            J = np.stack(cols, axis=-1)  # (2, n, 2): J[:, f, :]
            dJ = (J[:, 1] - J[:, -1]) / (2.0 * dw)  # around ω = 0
            np.testing.assert_allclose(
                np.abs(np.linalg.det(J[:, 0])), 1.0, atol=1e-9
            )  # unitary
            dgd.append(2.0 * np.sqrt(abs(np.linalg.det(dJ))))
        expected = 0.5e-12 * math.sqrt(length)
        assert np.mean(dgd) == pytest.approx(expected, rel=0.2)

    def test_pmd_needs_two_polarizations(self, xp):
        with pytest.raises(ValueError, match="pmd"):
            apply_fiber_propagation(
                xp.asarray(_field(1)[0]),
                center_wavelength_nm=1550.0,
                fiber=Fiber(pmd_ps_sqrt_km=0.1),
                span_length_km=10.0,
                num_spans=1,
                launch_power_dbm=0.0,
                sampling_rate=FS,
            )


class TestValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"attenuation_db_km": -0.1},
            {"nonlinearity_per_w_km": -1.0},
            {"pmd_ps_sqrt_km": -0.1},
            {"dispersion_ps_nm_km": float("nan")},
        ],
        ids=str,
    )
    def test_fiber_rejects_bad_values(self, kwargs):
        with pytest.raises(ValueError, match=next(iter(kwargs))):
            Fiber(**kwargs)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"num_spans": 0}, "num_spans"),
            ({"steps_per_span": 2.5}, "steps_per_span"),
            ({"span_length_km": -1.0}, "span_length_km"),
            ({"noise_figure_db": -1.0}, "noise_figure_db"),
            ({"fiber": "ssmf"}, "Fiber"),
        ],
        ids=str,
    )
    def test_bad_link_raises(self, xp, kwargs, match):
        kw = (
            dict(
                fiber=Fiber(),
                span_length_km=10.0,
                num_spans=1,
                launch_power_dbm=0.0,
                center_wavelength_nm=1550.0,
                sampling_rate=FS,
            )
            | kwargs
        )
        with pytest.raises((ValueError, TypeError), match=match):
            apply_fiber_propagation(xp.asarray(_field(2)), **kw)

    def test_three_channels_raise(self, xp):
        with pytest.raises(ValueError, match=r"\(2, N\)"):
            apply_fiber_propagation(
                xp.asarray(_field(3)),
                center_wavelength_nm=1550.0,
                fiber=Fiber(),
                span_length_km=10.0,
                num_spans=1,
                launch_power_dbm=0.0,
                sampling_rate=FS,
            )

    def test_aliasing_warning(self, xp, caplog):
        """A field filling the band broadens into aliasing; it is flagged."""
        rng = np.random.default_rng(0)
        x = (
            rng.standard_normal((2, 4096)) + 1j * rng.standard_normal((2, 4096))
        ).astype(np.complex64)
        with caplog.at_level("WARNING", logger="commkit"):
            apply_fiber_propagation(
                xp.asarray(x),
                center_wavelength_nm=1550.0,
                fiber=Fiber(),
                span_length_km=10.0,
                num_spans=1,
                launch_power_dbm=0.0,
                steps_per_span=2,
                sampling_rate=FS,
            )
        assert "may alias" in caplog.text


class TestSignalInput:
    def test_signal_keeps_its_metadata(self, xp, xpt):
        x = xp.asarray(_field(2))
        sig = Signal(samples=x, sampling_rate=FS, symbol_rate=FS / 2)
        kw = dict(
            center_wavelength_nm=1550.0,
            fiber=Fiber(),
            span_length_km=40.0,
            num_spans=2,
            launch_power_dbm=3.0,
            steps_per_span=5,
        )
        out = apply_fiber_propagation(sig, **kw)
        assert isinstance(out, Signal) and out.sampling_rate == FS
        xpt.assert_allclose(
            out.samples, apply_fiber_propagation(x, sampling_rate=FS, **kw)
        )
