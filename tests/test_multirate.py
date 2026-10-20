"""Tests for multirate signal processing routines (upsampling, decimation, resampling)."""

from typing import Any

import numpy as np
import pytest

from commkit import multirate
from commkit.core import Signal
from tests.common.conversions import to_numpy


class TestUpsampleAndDecimate:
    """Tests for integer rate conversion and decimation to symbol rate."""

    def test_upsample(self, xp: Any) -> None:
        """Verify integer upsampling increases signal length correctly."""
        data = xp.array([1.0, 2.0, 3.0])
        factor = 2
        out = multirate.upsample(data, factor=factor)

        assert isinstance(out, xp.ndarray)
        assert out.size >= data.size * factor - factor

    def test_decimate(self, xp: Any) -> None:
        """Verify integer decimation reduces signal length correctly."""
        data = xp.zeros(100)
        data[::2] = 1.0
        factor = 2
        out = multirate.decimate(data, factor=factor)

        assert isinstance(out, xp.ndarray)
        assert out.size <= data.size // factor + 1

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

    def test_resample(self, xp: Any) -> None:
        """Verify rational resampling produce expected output size."""
        data = xp.ones(100)
        up, down = 3, 2
        out = multirate.resample(data, up=up, down=down)

        assert isinstance(out, xp.ndarray)
        expected_size = int(data.size * up / down)
        assert abs(out.size - expected_size) < 5

    def test_resample_sps(self, xp: Any) -> None:
        """Verify resampling based on input/output samples per symbol ratios."""
        data = xp.ones(100)
        out = multirate.resample(data, sps_in=4, sps_out=8)
        assert out.size == 200

        out_frac = multirate.resample(data, sps_in=10, sps_out=25)
        assert out_frac.size == 250

    def test_resample_multidim(self, xp: Any) -> None:
        """Verify resample handles multi-dimensional input correctly."""
        samples = xp.ones((2, 10))
        res = multirate.resample(samples, up=2, down=1)

        assert res.shape == (2, 20)
        assert isinstance(res, xp.ndarray)
        assert res.size == 40

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
