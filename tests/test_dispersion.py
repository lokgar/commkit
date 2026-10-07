"""Definition-level checks of the chromatic dispersion model (``_dispersion``).

A round trip (apply, then compensate) cannot catch an error shared by both
directions, so these check the sign and the units against independently
derived values.
"""

from typing import Any

import numpy as np
import pytest

from commkit._dispersion import apply_dispersion, beta2_length
from commkit.impairments import apply_chromatic_dispersion

C = 299_792_458.0  # m/s, exact
LAMBDA_NM = 1550.0


class TestUnits:
    def test_beta2_of_standard_fiber(self) -> None:
        """D = 17 ps/(nm km) at 1550 nm is β₂ ≈ -21.68 ps²/km."""
        b2l = beta2_length(
            dispersion_ps_nm_km=17.0, fiber_length_km=1.0, center_wavelength_nm=1550.0
        )
        assert b2l == pytest.approx(-21.68e-24, rel=2e-3)  # s² over 1 km

    def test_linear_in_length_and_dispersion(self) -> None:
        def b2l(d: float, length: float) -> float:
            return beta2_length(
                dispersion_ps_nm_km=d,
                fiber_length_km=length,
                center_wavelength_nm=LAMBDA_NM,
            )

        assert b2l(17.0, 80.0) == pytest.approx(80.0 * b2l(17.0, 1.0), rel=1e-12)
        assert b2l(-17.0, 80.0) == pytest.approx(-b2l(17.0, 80.0), rel=1e-12)


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
            dispersion_ps_nm_km=self.D,
            fiber_length_km=self.L,
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
            dispersion_ps_nm_km=self.D,
            fiber_length_km=self.L,
            center_wavelength_nm=LAMBDA_NM,
        )
        forward = _centroid(apply_dispersion(x, inverse=False, **kw)) - _centroid(x)
        inverse = _centroid(apply_dispersion(x, inverse=True, **kw)) - _centroid(x)
        assert inverse == pytest.approx(-forward, rel=1e-6)
