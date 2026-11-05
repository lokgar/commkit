"""Tests for default theme configuration and subplot grid computations."""

from typing import Any
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np

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
