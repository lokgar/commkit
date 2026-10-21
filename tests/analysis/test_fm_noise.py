"""Tests for fm_noise_psd against analytic FM-noise levels."""

import numpy as np
import pytest

from commkit.analysis import fm_noise_psd
from commkit.backend import to_device

FS = 1e9


def _host(f, s):
    return np.asarray(to_device(f, "cpu")), np.asarray(to_device(s, "cpu"))


class TestFmNoisePsdLevels:
    """The three cases of the Notes: white FM, white phase, band-limited FM."""

    def test_white_fm_is_flat_to_nyquist(self, xp):
        """A sampled Wiener phase has white increments: S_f = Δν/π at every
        frequency, including next to Nyquist."""
        dnu, n = 1e6, 1 << 21
        steps = np.random.default_rng(0).standard_normal(n)
        phi = xp.asarray(np.cumsum(steps * np.sqrt(2 * np.pi * dnu / FS)))
        f, s = _host(*fm_noise_psd(phi, sampling_rate=FS, nperseg=1 << 12))
        low = (f > 0.01 * FS) & (f < 0.1 * FS)
        high = f > 0.4 * FS
        assert np.mean(s[low]) == pytest.approx(dnu / np.pi, rel=0.02)
        assert np.mean(s[high]) == pytest.approx(dnu / np.pi, rel=0.02)

    def test_white_phase_noise_shape(self, xp):
        """Additive white phase noise σ² per sample gives
        (2σ²·f_s/π²)·sin²(πf/f_s)."""
        sigma2, n = 1e-3, 1 << 21
        phi = xp.asarray(np.random.default_rng(1).standard_normal(n) * np.sqrt(sigma2))
        f, s = _host(*fm_noise_psd(phi, sampling_rate=FS, nperseg=1 << 12))
        sel = f > 0.05 * FS
        expected = 2 * sigma2 * FS / np.pi**2 * np.sin(np.pi * f[sel] / FS) ** 2
        np.testing.assert_allclose(
            np.convolve(s[sel] / expected, np.ones(64) / 64, mode="valid"),
            1.0,
            rtol=0.05,
        )

    def test_band_limited_fm_needs_correction(self, xp):
        """A frequency tone well inside Nyquist droops by sinc²(fT) in the raw
        estimate; bias_correction=True restores its power."""
        f_mod, dev, n = 0.4 * FS, 1e6, 1 << 18
        t = np.arange(n) / FS
        # Phase of a sinusoidal frequency deviation dev·cos(2π f_mod t).
        phi = xp.asarray(dev / f_mod * np.sin(2 * np.pi * f_mod * t))
        kw = dict(sampling_rate=FS, nperseg=1 << 12, detrend=False)
        f, raw = _host(*fm_noise_psd(phi, **kw))
        _, corr = _host(*fm_noise_psd(phi, bias_correction=True, **kw))
        df = f[1] - f[0]
        # A tone of amplitude dev carries dev²/2 of frequency variance.
        assert np.sum(corr) * df == pytest.approx(dev**2 / 2, rel=0.02)
        assert np.sum(raw) * df == pytest.approx(
            dev**2 / 2 * np.sinc(f_mod / FS) ** 2, rel=0.02
        )
