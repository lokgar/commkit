"""Tests for equalizer result and frequency response plotting."""

from typing import Any
from unittest.mock import patch

import matplotlib.pyplot as plt

from commkit import equalization, generate_psk
from commkit.plotting import plot_equalizer_result
from commkit.plotting.equalizer import plot_zf_equalizer_response


class TestPlotEqualizer:
    """Tests for equalizer convergence and response plotting."""

    def test_equalizer_result_mimo_weights(self, xp: Any) -> None:
        """equalizer_result with MIMO error/weights plots per-channel error and weights."""
        n_symbols = 400
        sig = generate_psk(
            symbol_rate=1e6,
            num_symbols=n_symbols,
            order=4,
            pulse_shape="rrc",
            sps=2,
            num_streams=2,
            seed=0,
        )
        rx_mimo = xp.asarray(sig.samples)
        train_mimo = xp.asarray(sig.source_symbols)

        result = equalization.lms(
            rx_mimo,
            training_symbols=train_mimo,
            num_taps=7,
            step_size=0.05,
            modulation="psk",
            order=4,
            backend="numba",
        )

        fig, axes = plot_equalizer_result(result, smoothing=10)
        assert fig is not None
        assert len(axes) == 2

    def test_equalizer_result_custom_axes(self, xp: Any) -> None:
        """equalizer_result with pre-existing axes uses them rather than creating new figures."""
        sig = generate_psk(
            symbol_rate=1e6, num_symbols=200, order=4, pulse_shape="rrc", sps=2, seed=0
        )
        result = equalization.lms(
            xp.asarray(sig.samples),
            training_symbols=xp.asarray(sig.source_symbols),
            num_taps=7,
            step_size=0.05,
            modulation="psk",
            order=4,
            backend="numba",
        )

        fig0, axes0 = plt.subplots(1, 2)
        fig_ret, axes_ret = plot_equalizer_result(result, ax=axes0)
        assert fig_ret is not None

    def test_equalizer_result_show(self, xp: Any) -> None:
        """equalizer_result with show=True calls plt.show() and returns None."""
        sig = generate_psk(
            symbol_rate=1e6, num_symbols=200, order=4, pulse_shape="rrc", sps=2, seed=0
        )
        result = equalization.lms(
            xp.asarray(sig.samples),
            training_symbols=xp.asarray(sig.source_symbols),
            num_taps=7,
            step_size=0.05,
            modulation="psk",
            order=4,
            backend="numba",
        )

        with patch("matplotlib.pyplot.show"):
            ret = plot_equalizer_result(result, show=True)
        assert ret is None

    def test_equalizer_result_short_smoothing_siso(self, xp: Any) -> None:
        """plot_equalizer_result() SISO where len(mse) <= smoothing uses raw mse."""
        sig = generate_psk(
            symbol_rate=1e6, num_symbols=50, order=4, pulse_shape="rrc", sps=2, seed=0
        )
        result = equalization.lms(
            xp.asarray(sig.samples),
            training_symbols=xp.asarray(sig.source_symbols),
            num_taps=5,
            step_size=0.05,
            modulation="psk",
            order=4,
            backend="numba",
        )
        fig, axes = plot_equalizer_result(result, smoothing=1000)
        assert fig is not None

    def test_equalizer_result_short_smoothing_mimo(self, xp: Any) -> None:
        """plot_equalizer_result() MIMO where len(mse) <= smoothing uses raw mse."""
        sig = generate_psk(
            symbol_rate=1e6,
            num_symbols=50,
            order=4,
            pulse_shape="rrc",
            sps=2,
            num_streams=2,
            seed=0,
        )
        result = equalization.lms(
            xp.asarray(sig.samples),
            training_symbols=xp.asarray(sig.source_symbols),
            num_taps=5,
            step_size=0.05,
            modulation="psk",
            order=4,
            backend="numba",
        )
        fig, axes = plot_equalizer_result(result, smoothing=1000)
        assert fig is not None

    def test_equalizer_result_with_phase_trajectory(self, xp: Any) -> None:
        """equalizer_result with phase_trajectory renders the 3-panel CPR layout."""
        sig = generate_psk(
            symbol_rate=1e6, num_symbols=100, order=4, pulse_shape="rrc", sps=2, seed=0
        )
        result = equalization.lms(
            xp.asarray(sig.samples),
            training_symbols=xp.asarray(sig.source_symbols),
            num_taps=5,
            step_size=0.05,
            modulation="psk",
            order=4,
            backend="numba",
        )
        phase_1d = xp.linspace(0, 0.5, 100)
        result.phase_trajectory = phase_1d
        fig1, axes1 = plot_equalizer_result(result, show=False)
        assert fig1 is not None
        assert len(axes1) == 3

        phase_2d = xp.stack([phase_1d, phase_1d * 0.5])
        result.phase_trajectory = phase_2d
        fig2, axes2 = plot_equalizer_result(result, show=False)
        assert fig2 is not None
        assert len(axes2) == 3

    def test_zf_equalizer_response_siso(self, xp: Any) -> None:
        """Verify plot_zf_equalizer_response for SISO channel impulse response."""
        h = xp.array([0.2, 1.0, -0.3, 0.1], dtype=xp.float32)
        fig, axes = plot_zf_equalizer_response(
            h, noise_variance=0.01, sampling_rate=1e6, show=False
        )
        assert fig is not None
        assert len(axes) == 3

    def test_zf_equalizer_response_mimo(self, xp: Any) -> None:
        """Verify plot_zf_equalizer_response for 2x2 MIMO channel impulse response."""
        h = xp.random.randn(2, 2, 8).astype(xp.float32)
        fig, axes = plot_zf_equalizer_response(
            h, noise_variance=0.005, sampling_rate=1e6, show=False
        )
        assert fig is not None
        assert len(axes) == 4
        assert len(axes[0]) == 3

        # Test with pre-allocated axes
        fig_custom, raw_ax = plt.subplots(4, 3)
        fig_ret, axes_ret = plot_zf_equalizer_response(h, ax=raw_ax, show=False)
        assert fig_ret is fig_custom

    def test_zf_equalizer_response_show(self, xp: Any) -> None:
        """Verify plot_zf_equalizer_response with show=True calls plt.show() and returns None."""
        h = xp.array([1.0, 0.5])
        with patch("matplotlib.pyplot.show"):
            ret = plot_zf_equalizer_response(h, show=True)
        assert ret is None
