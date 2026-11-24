"""Definition-level checks of the chromatic dispersion model (``_dispersion``).

A round trip (apply, then compensate) cannot catch an error shared by both
directions, so these check the sign and the units against independently
derived values.
"""

from typing import Any

import numpy as np
import pytest

from commkit._dispersion import apply_dispersion, beta2_length, beta3_length
from commkit.impairments import apply_chromatic_dispersion

C = 299_792_458.0  # m/s, exact
LAMBDA_NM = 1550.0


class TestUnits:
    def test_beta2_of_standard_fiber(self) -> None:
        """D = 17 ps/(nm km) at 1550 nm is β₂ ≈ -21.68 ps²/km."""
        b2l = beta2_length(dispersion_ps_nm=17.0, center_wavelength_nm=1550.0)
        assert b2l == pytest.approx(-21.68e-24, rel=2e-3)  # s² over 1 km

    def test_linear_in_length_and_dispersion(self) -> None:
        def b2l(d: float, length: float) -> float:
            return beta2_length(
                dispersion_ps_nm=d * length, center_wavelength_nm=LAMBDA_NM
            )

        assert b2l(17.0, 80.0) == pytest.approx(80.0 * b2l(17.0, 1.0), rel=1e-12)
        assert b2l(-17.0, 80.0) == pytest.approx(-b2l(17.0, 80.0), rel=1e-12)

    def test_beta3_is_the_frequency_derivative_of_beta2(self) -> None:
        """β₃ = dβ₂/dω with D(λ) = D0 + S (λ - λ0): central difference of
        β₂(λ) = -D(λ) λ² / (2πc) over ω = 2πc/λ (SSMF: about 0.13 ps³/km)."""
        d0, s = 17.0, 0.057  # ps/(nm km), ps/(nm² km)

        def beta2(lam_nm: float) -> float:
            return beta2_length(
                dispersion_ps_nm=d0 + s * (lam_nm - LAMBDA_NM),
                center_wavelength_nm=lam_nm,
            )

        def omega(lam_nm: float) -> float:
            return 2.0 * np.pi * 299_792_458.0 / (lam_nm * 1e-9)

        dl = 0.01
        numeric = (beta2(LAMBDA_NM + dl) - beta2(LAMBDA_NM - dl)) / (
            omega(LAMBDA_NM + dl) - omega(LAMBDA_NM - dl)
        )
        b3 = beta3_length(
            dispersion_ps_nm=d0,
            dispersion_slope_ps_nm2=s,
            center_wavelength_nm=LAMBDA_NM,
        )
        assert b3 == pytest.approx(numeric, rel=1e-6)
        assert b3 == pytest.approx(0.13e-36, rel=0.05)  # s³ over 1 km


def _tone_burst(xp: Any, n: int, fs: float, f0: float) -> Any:
    """Gaussian envelope at baseband frequency f0, centred in the record."""
    t = (np.arange(n) - n / 2) / fs
    envelope = np.exp(-0.5 * (t / 40e-12) ** 2)
    return xp.asarray(envelope * np.exp(2j * np.pi * f0 * t))


def _centroid(x: Any) -> float:
    p = np.abs(np.asarray(x if isinstance(x, np.ndarray) else x.get())) ** 2
    return float(np.sum(np.arange(p.size) * p) / np.sum(p))


class TestSign:
    """Group delay of a narrow-band tone: Δτ = D · L · Δλ."""

    FS = 200e9
    N = 1 << 14
    D = 17.0  # ps/(nm km), anomalous
    L = 80.0  # km

    def _expected_delay_samples(self, f0: float) -> float:
        # A positive baseband offset is a shorter wavelength: Δλ = -λ² f0 / c.
        lam = LAMBDA_NM * 1e-9
        d_lambda_nm = -(lam**2) * f0 / C * 1e9
        delay_s = self.D * self.L * d_lambda_nm * 1e-12  # ps/(nm km) * km * nm
        return delay_s * self.FS

    @pytest.mark.parametrize("f0", [-30e9, 30e9])
    def test_forward_delay_matches_d_l_dlambda(self, xp: Any, f0: float) -> None:
        x = _tone_burst(xp, self.N, self.FS, f0)
        y = apply_chromatic_dispersion(
            x,
            sampling_rate=self.FS,
            dispersion_ps_nm_km=self.D,
            fiber_length_km=self.L,
            center_wavelength_nm=LAMBDA_NM,
        )
        delay = _centroid(y) - _centroid(x)
        expected = self._expected_delay_samples(f0)
        assert abs(expected) > 50  # the check is meaningful
        assert delay == pytest.approx(expected, rel=2e-3)

    def test_blue_arrives_first_in_anomalous_fiber(self, xp: Any) -> None:
        kw = dict(
            sampling_rate=self.FS,
            dispersion_ps_nm=self.D * self.L,
            center_wavelength_nm=LAMBDA_NM,
        )
        blue = apply_dispersion(
            _tone_burst(xp, self.N, self.FS, 30e9), inverse=False, **kw
        )
        red = apply_dispersion(
            _tone_burst(xp, self.N, self.FS, -30e9), inverse=False, **kw
        )
        assert _centroid(blue) < _centroid(red)

    def test_inverse_delays_the_other_way(self, xp: Any) -> None:
        x = _tone_burst(xp, self.N, self.FS, 30e9)
        kw = dict(
            sampling_rate=self.FS,
            dispersion_ps_nm=self.D * self.L,
            center_wavelength_nm=LAMBDA_NM,
        )
        forward = _centroid(apply_dispersion(x, inverse=False, **kw)) - _centroid(x)
        inverse = _centroid(apply_dispersion(x, inverse=True, **kw)) - _centroid(x)
        assert inverse == pytest.approx(-forward, rel=1e-6)


class TestRealInputOpticalTransforms:
    """Optical linear transforms return complex output consistently on real input."""

    def test_real_input_returns_complex_without_warning(self, xp: Any) -> None:
        import warnings

        from commkit.equalization import correct_chromatic_dispersion
        from commkit.impairments import apply_pmd

        rng = np.random.default_rng(7)
        x1 = xp.asarray(rng.standard_normal(512).astype(np.float32))
        x2 = xp.asarray(rng.standard_normal((2, 512)).astype(np.float32))
        fs = 64e9

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            y_cd = apply_chromatic_dispersion(
                x1,
                dispersion_ps_nm_km=17.0,
                fiber_length_km=10.0,
                center_wavelength_nm=1550.0,
                sampling_rate=fs,
            )
            y_cdc_wr = correct_chromatic_dispersion(
                x1,
                dispersion_ps_nm=170.0,
                center_wavelength_nm=1550.0,
                sampling_rate=fs,
                method="whole_record",
            )
            y_cdc_fir = correct_chromatic_dispersion(
                x1,
                dispersion_ps_nm=170.0,
                center_wavelength_nm=1550.0,
                sampling_rate=fs,
                method="fir",
            )
            y_pmd = apply_pmd(x2, dgd=5e-12, theta=0.3, sampling_rate=fs)

        for y in (y_cd, y_cdc_wr, y_cdc_fir, y_pmd):
            assert y.dtype == np.complex64
            assert bool(xp.any(xp.abs(y.imag) > 1e-4))
