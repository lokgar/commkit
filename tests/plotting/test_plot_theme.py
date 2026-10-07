"""Tests for default theme configuration and subplot grid computations."""

from typing import Any
from unittest.mock import patch

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
