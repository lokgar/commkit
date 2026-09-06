"""Tests for multirate signal processing routines (upsampling, decimation, resampling)."""

from typing import Any

import pytest

from commkit import multirate


class TestUpsampleAndDecimate:
    """Tests for integer rate conversion and decimation to symbol rate."""

    def test_upsample(self, backend_device: str, xp: Any) -> None:
        """Verify integer upsampling increases signal length correctly."""
        data = xp.array([1.0, 2.0, 3.0])
        factor = 2
        out = multirate.upsample(data, factor)

        assert isinstance(out, xp.ndarray)
        assert out.size >= data.size * factor - factor

    def test_decimate(self, backend_device: str, xp: Any) -> None:
        """Verify integer decimation reduces signal length correctly."""
        data = xp.zeros(100)
        data[::2] = 1.0
        factor = 2
        out = multirate.decimate(data, factor)

        assert isinstance(out, xp.ndarray)
        assert out.size <= data.size // factor + 1

    def test_decimate_polyphase(self, backend_device: str, xp: Any) -> None:
        """Verify decimation using the polyphase method."""
        data = xp.ones(100)
        factor = 2
        out = multirate.decimate(data, factor, method="polyphase")

        assert isinstance(out, xp.ndarray)
        assert out.size == 50

        with pytest.raises(ValueError, match="Unknown decimation method"):
            multirate.decimate(data, factor, method="unknown")

    def test_decimate_to_symbol_rate(
        self, backend_device: str, xp: Any, xpt: Any
    ) -> None:
        """Verify downsampling (picking) symbols from an upsampled stream."""
        sps = 4
        data = xp.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4], dtype="float32")

        syms = multirate.decimate_to_symbol_rate(data, sps=sps, offset=0)
        xpt.assert_array_equal(syms, xp.array([1, 2, 3, 4]))

        data_mimo = xp.stack([data, data * 10])
        syms_mimo = multirate.decimate_to_symbol_rate(data_mimo, sps=sps, axis=-1)
        assert syms_mimo.shape == (2, 4)
        xpt.assert_array_equal(syms_mimo[1], xp.array([10, 20, 30, 40]))


class TestResample:
    """Tests for rational and SPS-targeted resampling."""

    def test_resample(self, backend_device: str, xp: Any) -> None:
        """Verify rational resampling produce expected output size."""
        data = xp.ones(100)
        up, down = 3, 2
        out = multirate.resample(data, up, down)

        assert isinstance(out, xp.ndarray)
        expected_size = int(data.size * up / down)
        assert abs(out.size - expected_size) < 5

    def test_resample_sps(self, backend_device: str, xp: Any) -> None:
        """Verify resampling based on input/output samples per symbol ratios."""
        data = xp.ones(100)
        out = multirate.resample(data, sps_in=4, sps_out=8)
        assert out.size == 200

        out_frac = multirate.resample(data, sps_in=10, sps_out=25)
        assert out_frac.size == 250

    def test_resample_multidim(self, backend_device: str, xp: Any) -> None:
        """Verify resample handles multi-dimensional input correctly."""
        samples = xp.ones((2, 10))
        res = multirate.resample(samples, up=2, down=1, axis=-1)

        assert res.shape == (2, 20)
        assert isinstance(res, xp.ndarray)
        assert res.size == 40

    def test_resample_errors(self, backend_device: str, xp: Any) -> None:
        """Verify inconsistent resampling parameters raise ValueError."""
        data = xp.zeros(10)
        with pytest.raises(ValueError, match="Cannot specify both"):
            multirate.resample(data, up=2, sps_in=4)

        with pytest.raises(ValueError, match="Must specify either"):
            multirate.resample(data)
