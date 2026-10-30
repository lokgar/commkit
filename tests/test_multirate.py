"""Tests for multirate signal processing routines (upsampling, decimation, resampling)."""

from typing import Any

import numpy as np
import pytest

from commkit import multirate
from commkit.core import Signal
from tests.common.conversions import to_numpy


class TestUpsampleAndDecimate:
    """Tests for integer rate conversion and decimation to symbol rate."""

    @staticmethod
    def _tone(n: int, f: float, phase: float = 0.4, t0: float = 0.0) -> np.ndarray:
        """``exp(j(2π f (k + t0) + phase))`` for ``k < n``, ``f`` in cycles
        per sample of that grid."""
        return np.exp(1j * (2 * np.pi * f * (np.arange(n) + t0) + phase))

    @pytest.mark.parametrize(
        ("fn", "kw", "ratio"),
        [
            ("upsample", {"factor": 3}, 3.0),
            ("decimate", {"factor": 4}, 0.25),
            ("decimate", {"factor": 4, "method": "polyphase"}, 0.25),
            ("decimate", {"factor": 4, "ftype": "iir"}, 0.25),
            ("resample", {"up": 3, "down": 2}, 1.5),
            ("resample", {"up": 2, "down": 5}, 0.4),
        ],
        ids=["up3", "dec4", "dec4-poly", "dec4-iir", "res3/2", "res2/5"],
    )
    def test_in_band_tone_is_preserved(
        self, xp: Any, fn: str, kw: dict, ratio: float
    ) -> None:
        """A tone well inside both bands comes out as the same tone sampled
        on the new grid, amplitude and phase (output k at input time
        k / ratio), within 1 % away from the filter transients at the ends.
        Two channels carry different tones, so mixing them up fails."""
        n = 4000
        f_in = (0.03, -0.05)  # cycles per input sample, |f| < 0.4 * min(1, ratio)/2
        x = np.stack([self._tone(n, f) for f in f_in]).astype(np.complex64)
        out = to_numpy(getattr(multirate, fn)(xp.asarray(x), **kw))
        m = out.shape[-1]
        expected = np.stack([self._tone(m, f / ratio) for f in f_in])
        edge = int(200 * max(1.0, ratio))
        np.testing.assert_allclose(
            out[:, edge:-edge], expected[:, edge:-edge], atol=1e-2
        )

    @pytest.mark.parametrize(
        ("fn", "kw", "f_out"),
        [
            ("decimate", {"factor": 4}, 0.4),
            ("decimate", {"factor": 4, "method": "polyphase"}, 0.4),
            ("resample", {"up": 2, "down": 5}, 0.35),
        ],
        ids=["dec4", "dec4-poly", "res2/5"],
    )
    def test_out_of_band_tone_is_rejected(
        self, xp: Any, fn: str, kw: dict, f_out: float
    ) -> None:
        """A tone above the new Nyquist frequency would alias into the band;
        the anti-aliasing filter leaves under -30 dB of it."""
        n = 8000
        x = self._tone(n, f_out).astype(np.complex64)
        out = to_numpy(getattr(multirate, fn)(xp.asarray(x), **kw))[200:-200]
        assert 10 * np.log10(np.mean(np.abs(out) ** 2)) < -30

    def test_upsampling_images_are_rejected(self, xp: Any) -> None:
        """Interpolating by 4 a tone at 0.4 cycles per input sample puts it at
        0.1 on the new grid; the images at 0.15, 0.35, ... stay under -30 dB
        of it."""
        n = 4000
        out = to_numpy(
            multirate.upsample(
                xp.asarray(self._tone(n, 0.4).astype(np.complex64)), factor=4
            )
        )
        seg = out[400:-400]
        spec = np.abs(np.fft.fft(seg * np.hanning(seg.size))) ** 2
        f = np.fft.fftfreq(seg.size)
        main = spec[np.abs(f - 0.1) < 0.01].sum()
        images = spec[np.abs(f - 0.1) > 0.02].sum()
        assert 10 * np.log10(images / main) < -30

    @pytest.mark.parametrize(
        ("fn", "kw", "gain"),
        [
            ("upsample", {"factor": 4}, 0.25),
            ("decimate", {"factor": 2}, 2.0),
            ("resample", {"up": 3, "down": 2}, 2 / 3),
        ],
    )
    def test_correct_power(self, xp: Any, fn: str, kw: dict, gain: float) -> None:
        """``correct_power=True`` scales the power by the rate ratio's inverse
        (``E[|x|²] = 1/sps`` holds), ``False`` keeps the tone's power; a
        Signal defaults to True, an array to False."""
        x = self._tone(4000, 0.02).astype(np.complex64)
        g = getattr(multirate, fn)

        def power(y: Any) -> float:
            y = to_numpy(y.samples if isinstance(y, Signal) else y)
            return float(np.mean(np.abs(y[..., 300:-300]) ** 2))

        assert power(g(xp.asarray(x), **kw)) == pytest.approx(1.0, rel=0.02)
        assert power(g(xp.asarray(x), **kw, correct_power=True)) == pytest.approx(
            gain, rel=0.02
        )
        sig = Signal(samples=xp.asarray(x), sampling_rate=4e9, symbol_rate=1e9)
        assert power(g(sig, **kw)) == pytest.approx(gain, rel=0.02)

    def test_decimate_polyphase(self, xp: Any) -> None:
        """Verify decimation using the polyphase method."""
        data = xp.ones(100)
        factor = 2
        out = multirate.decimate(data, factor=factor, method="polyphase")

        assert isinstance(out, xp.ndarray)
        assert out.size == 50

        with pytest.raises(ValueError, match="Unknown decimation method"):
            multirate.decimate(data, factor=factor, method="unknown")

    def test_decimate_to_symbol_rate(self, xp: Any, xpt: Any) -> None:
        """Verify downsampling (picking) symbols from an upsampled stream."""
        sps = 4
        data = xp.array(
            [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4], dtype="float32"
        )

        syms = multirate.decimate_to_symbol_rate(data, sps=sps, offset=0)
        xpt.assert_array_equal(syms, xp.array([1, 2, 3, 4]))

        data_mimo = xp.stack([data, data * 10])
        syms_mimo = multirate.decimate_to_symbol_rate(data_mimo, sps=sps)
        assert syms_mimo.shape == (2, 4)
        xpt.assert_array_equal(syms_mimo[1], xp.array([10, 20, 30, 40]))


class TestResample:
    """Tests for rational and SPS-targeted resampling."""

    def test_resample_sps(self, xp: Any) -> None:
        """Verify resampling based on input/output samples per symbol ratios."""
        data = xp.ones(100)
        out = multirate.resample(data, sps_in=4, sps_out=8)
        assert out.size == 200

        out_frac = multirate.resample(data, sps_in=10, sps_out=25)
        assert out_frac.size == 250

    def test_resample_errors(self, xp: Any) -> None:
        """Verify inconsistent resampling parameters raise ValueError."""
        data = xp.zeros(10)
        with pytest.raises(ValueError, match="not both"):
            multirate.resample(data, up=2, down=1, sps_out=4)
        with pytest.raises(ValueError, match="give either"):
            multirate.resample(data)
        with pytest.raises(ValueError, match="together"):
            multirate.resample(data, up=2)
        with pytest.raises(ValueError, match="requires sps_in"):
            multirate.resample(data, sps_out=4)
        with pytest.raises(ValueError, match="positive integers"):
            multirate.upsample(data, factor=1.5)
        with pytest.raises(ValueError, match="offset"):
            multirate.decimate_to_symbol_rate(data, sps=2, offset=2)


def _tones(n: int, bins: list[int], phases: list[float]) -> np.ndarray:
    """Sum of complex tones with integer cycles per ``n`` samples."""
    t = np.arange(n) / n
    return sum(
        np.exp(1j * (2 * np.pi * k * t + p)) for k, p in zip(bins, phases)
    ).astype(np.complex64)


class TestPeriodicResampling:
    """upsample/resample(periodic=True): one period at the new rate."""

    @pytest.mark.parametrize("up, down", [(23, 16), (3, 2), (2, 3), (4, 1)])
    def test_band_limited_period_is_exact(self, xp: Any, up: int, down: int) -> None:
        """Tones with whole cycles per record are resampled to the same tones
        sampled at the new rate, ends included."""
        n, bins, phases = 96, [-7, 2, 11], [0.3, 1.0, -2.0]
        out = multirate.resample(
            xp.asarray(_tones(n, bins, phases)), up=up, down=down, periodic=True
        )
        expected = _tones(n * up // down, bins, phases)
        assert out.dtype == xp.complex64
        np.testing.assert_allclose(to_numpy(out), expected, atol=1e-4)

    def test_upsample_matches_resample(self, xp: Any) -> None:
        x = xp.asarray(_tones(64, [3, -9], [0.0, 0.5]).reshape(1, -1).repeat(2, 0))
        up = multirate.upsample(x, factor=3, periodic=True)
        np.testing.assert_allclose(
            to_numpy(up),
            to_numpy(multirate.resample(x, up=3, down=1, periodic=True)),
            atol=1e-6,
        )
        assert up.shape == (2, 192)

    def test_signal_power_and_rate(self) -> None:
        sig = Signal(
            samples=_tones(128, [5], [0.0]) / np.sqrt(2),
            sampling_rate=2e9,
            symbol_rate=1e9,
        )
        out = multirate.resample(sig, sps_out=3.0, periodic=True)
        assert out.sampling_rate == pytest.approx(3e9)
        assert out.samples.shape == (192,)
        assert np.mean(np.abs(out.samples) ** 2) == pytest.approx(1 / 3, rel=1e-5)

    def test_non_integer_period_raises(self, xp: Any) -> None:
        with pytest.raises(ValueError, match="periodic"):
            multirate.resample(
                xp.ones(100, dtype=xp.complex64), up=3, down=8, periodic=True
            )
