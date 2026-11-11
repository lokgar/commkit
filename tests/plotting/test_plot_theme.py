"""Tests for default theme configuration and subplot grid computations."""

from typing import Any
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest

from commkit.plotting import apply_default_theme
from commkit.plotting.theme import _create_subplot_grid


class TestPlotThemeAndGridUtilities:
    """Tests for default theme configuration and subplot grid computations."""

    def test_apply_theme(self) -> None:
        """Verify applying the visual theme."""
        apply_default_theme()

    @patch("matplotlib.font_manager.findfont")
    def test_apply_theme_fallback(self, mock_find: Any) -> None:
        """Trigger the font fallback in apply_default_theme."""
        mock_find.side_effect = ValueError("Font not found")
        apply_default_theme()

    def test_subplot_grid(self, xp: Any) -> None:
        """Verify subplot grid calculation."""
        assert _create_subplot_grid(1) == (1, 1)
        assert _create_subplot_grid(2) == (1, 2)
        assert _create_subplot_grid(3) == (2, 2)
        assert _create_subplot_grid(5) == (3, 2)


class TestSIAxisUnits:
    """The SI prefix of an axis goes into its label's unit, not its ticks."""

    @staticmethod
    def _draw(x: Any, y: Any, xlabel: str, *, logx: bool = False) -> Any:
        from commkit.plotting.theme import _set_eng_formatter

        fig, ax = plt.subplots()
        (ax.semilogx if logx else ax.plot)(x, y)
        ax.set_xlabel(xlabel)
        _set_eng_formatter(ax, "x")
        fig.canvas.draw()
        return fig, ax

    def test_prefix_in_the_label_and_plain_ticks(self) -> None:
        _, ax = self._draw(np.linspace(-20e9, 20e9, 5), np.zeros(5), "Frequency [Hz]")
        assert ax.get_xlabel() == "Frequency [GHz]"
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert "10" in labels and "\N{MINUS SIGN}10" in labels
        assert not any("G" in t for t in labels)

    def test_new_limits_rechoose_the_prefix(self) -> None:
        fig, ax = self._draw(np.linspace(0, 3e-6, 5), np.zeros(5), "Time [s]")
        assert ax.get_xlabel() == "Time [µs]"
        ax.set_xlim(0, 4e-9)
        fig.canvas.draw()
        assert ax.get_xlabel() == "Time [ns]"
        ax.set_xlabel("Delay [s]")  # a new label is the new base
        fig.canvas.draw()
        assert ax.get_xlabel() == "Delay [ns]"

    def test_decimals_follow_the_step(self) -> None:
        fig, ax = self._draw(np.linspace(0, 2e6, 5), np.zeros(5), "Frequency [Hz]")
        ax.set_xticks([0, 0.25e6, 0.5e6])
        ax.set_xlim(0, 0.5e6)
        fig.canvas.draw()
        assert [t.get_text() for t in ax.get_xticklabels()] == ["0", "250", "500"]
        assert ax.get_xlabel() == "Frequency [kHz]"

    def test_text_after_the_unit_is_kept(self) -> None:
        _, ax = self._draw(np.linspace(0, 3e3, 4), np.zeros(4), "Δf [Hz] (÷M)")
        assert ax.get_xlabel() == "Δf [kHz] (÷M)"

    def test_unitless_label_keeps_the_prefix_on_ticks(self) -> None:
        _, ax = self._draw(np.linspace(0, 3e9, 4), np.zeros(4), "Index")
        assert ax.get_xlabel() == "Index"
        assert any(t.get_text().endswith(" G") for t in ax.get_xticklabels())

    def test_log_axis_keeps_per_tick_prefixes(self) -> None:
        _, ax = self._draw(
            np.geomspace(1e3, 1e9, 7), np.ones(7), "Frequency [Hz]", logx=True
        )
        assert ax.get_xlabel() == "Frequency [Hz]"
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert any("k" in t for t in labels) and any("M" in t for t in labels)


class TestDeviceReduction:
    """Plots of sample-rate traces reduce on the device: a long record moves
    only the drawn points to the host."""

    N = 1 << 18

    @staticmethod
    def _cases(xp: Any) -> dict[str, Any]:
        from types import SimpleNamespace as NS

        from commkit import plotting
        from commkit.recovery import PilotTone

        n = TestDeviceReduction.N
        rng = np.random.default_rng(0)
        phase = xp.asarray(np.cumsum(rng.standard_normal((2, n)), axis=-1) * 1e-2)
        samples = xp.asarray(
            (rng.standard_normal((2, n)) + 1j * rng.standard_normal((2, n))).astype(
                np.complex64
            )
        )
        value = xp.asarray([1.0, 2.0])
        return {
            "trajectory": lambda: plotting.plot_carrier_phase_trajectory(phase),
            "decomposition": lambda: plotting.plot_carrier_phase_decomposition(
                phase, phase * 0.5, symbol_rate=1e9
            ),
            "drift": lambda: plotting.plot_frequency_drift(
                NS(df=phase), sampling_rate=1e9
            ),
            "spectrum": lambda: plotting.plot_frequency_offset_spectrum(
                NS(
                    spectrum=xp.abs(phase),
                    power=4,
                    value=value,
                    spectrum_frequencies=np.arange(n) - n / 2.0,
                )  # fmt: skip
            ),
            "autocorrelation": lambda: plotting.plot_mm_autocorrelation(
                NS(autocorrelation=samples, power=4, value=value), sampling_rate=1e9
            ),
            "timing": lambda: plotting.plot_timing_correlation(
                NS(correlation=samples, integer=xp.asarray([50, 900]), search_start=10)
            ),
            "pilot tone": lambda: plotting.plot_pilot_tone_phase_estimate(
                NS(tone_frequencies=np.array([1e6, 1e6]), value=phase),
                samples,
                method=PilotTone(frequency=1e6, bandwidth=1e5),
                sampling_rate=1e9,
            ),
        }

    @pytest.mark.gpu_only
    @pytest.mark.parametrize(
        "case",
        ["trajectory", "decomposition", "drift", "spectrum", "autocorrelation",
         "timing", "pilot tone"],
    )  # fmt: skip
    def test_moves_only_the_drawn_points(self, xp: Any, case: str) -> None:
        """Every device-to-host copy is at most the 4000 points of a trace
        and its remainder, for each of the two channels (2 x 2^18 points
        before)."""
        from commkit import backend

        sizes: list[int] = []
        real = backend.to_device

        def counting(data: Any, device: str) -> Any:
            if device == "cpu" and isinstance(data, xp.ndarray):
                sizes.append(int(data.size))
            return real(data, device)

        import commkit.plotting.analysis as analysis
        import commkit.plotting.frequency as frequency
        import commkit.plotting.recovery as recovery
        import commkit.plotting.theme as theme

        with (
            patch.object(theme, "to_device", counting),
            patch.object(recovery, "to_device", counting),
            patch.object(frequency, "to_device", counting),
            patch.object(analysis, "to_device", counting),
        ):
            self._cases(xp)[case]()
        plt.close("all")
        assert sizes and max(sizes) <= 2 * (4000 + 2000), sizes
