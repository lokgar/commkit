"""Tests for default theme configuration and subplot grid computations."""

from typing import Any
from unittest.mock import patch

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pytest

import commkit.plotting.theme as theme
from commkit.plotting import apply_default_theme
from commkit.plotting.theme import _create_subplot_grid


@pytest.fixture(autouse=True)
def _restore_rcparams() -> Any:
    """The themes these tests apply do not leak into other tests."""
    with mpl.rc_context():
        yield


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
        assert plt.rcParams["font.family"] == ["sans"]
        assert plt.rcParams["mathtext.rm"] == "sans"

    def test_font_found_after_cache_was_written(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A font missing from Matplotlib's cache is found by scanning the font directories.

        Models a font installed after the cache file was written: the family
        is dropped from the font list, and the scan (here the files shipped
        with Matplotlib) restores it for all four faces.
        """
        dejavu = [e for e in fm.fontManager.ttflist if e.name == "DejaVu Sans"]
        assert dejavu
        files = sorted({str(e.fname) for e in dejavu})
        monkeypatch.setattr(
            fm.fontManager,
            "ttflist",
            [e for e in fm.fontManager.ttflist if e.name != "DejaVu Sans"],
        )
        monkeypatch.setattr(fm, "findSystemFonts", lambda *a, **k: files)
        monkeypatch.setattr(theme, "_FONT_CANDIDATES", ("No Such Font", "DejaVu Sans"))
        assert theme._pick_font_family() == "DejaVu Sans"
        found = {
            str(
                fm.findfont(fm.FontProperties(family="DejaVu Sans", style=st, weight=w))
            )
            for st, w in theme._FONT_STYLES
        }
        assert len(found) == 4

    def test_family_missing_a_face_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A family without its bold face is not picked, so bold text never falls back silently."""
        monkeypatch.setattr(
            fm.fontManager,
            "ttflist",
            [
                e
                for e in fm.fontManager.ttflist
                if not (e.name == "DejaVu Sans" and e.weight >= 700)
            ],
        )
        monkeypatch.setattr(fm, "findSystemFonts", lambda *a, **k: [])
        assert not theme._family_installed("DejaVu Sans")

    def test_math_fonts_follow_text_font(self) -> None:
        """Every math face uses the text family, with no generic fallback left behind."""
        apply_default_theme()
        name = plt.rcParams["font.family"][0]
        assert name in theme._FONT_CANDIDATES or name == "sans"
        assert plt.rcParams["mathtext.bfit"] == f"{name}:italic:bold"
        for role in ("rm", "cal", "sf", "tt"):
            assert plt.rcParams[f"mathtext.{role}"] == name

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

        from commkit import plotting
        from commkit.analysis import FrequencyDrift
        from commkit.equalization import EqualizerResult
        from commkit.frequency import (
            BlockwiseFrequencyEstimate,
            MengaliMorelliEstimate,
            MthPowerEstimate,
        )
        from commkit.recovery import (
            PilotTone,
            PilotToneEstimate,
            PilotTonesEstimate,
        )
        from commkit.timing import TimingEstimate

        n = TestDeviceReduction.N
        rng = np.random.default_rng(0)
        phase = xp.asarray(np.cumsum(rng.standard_normal((2, n)), axis=-1) * 1e-2)
        samples = xp.asarray(
            (rng.standard_normal((2, n)) + 1j * rng.standard_normal((2, n))).astype(
                np.complex64
            )
        )
        real_wave = xp.asarray(rng.standard_normal(n).astype(np.float32))
        llrs = xp.asarray(rng.standard_normal(n).astype(np.float32))
        bits = xp.asarray(rng.integers(0, 2, n).astype(np.int8))
        value = xp.asarray([1.0, 2.0])
        return {
            "trajectory": lambda: plotting.plot_carrier_phase_trajectory(phase),
            "decomposition": lambda: plotting.plot_carrier_phase_decomposition(
                phase, phase * 0.5, symbol_rate=1e9
            ),
            "drift": lambda: plotting.plot_frequency_drift(
                FrequencyDrift(df=phase, std=0.0, peak_to_peak=0.0, max_abs=0.0),
                sampling_rate=1e9,
            ),
            "spectrum": lambda: plotting.plot_frequency_offset_spectrum(
                MthPowerEstimate(
                    value=value,
                    weights=value,
                    power=4,
                    spectrum=xp.abs(phase),
                    spectrum_frequencies=np.arange(n) - n / 2.0,
                )
            ),
            "blockwise frequency": lambda: (
                plotting.plot_frequency_offset_blockwise_result(
                    BlockwiseFrequencyEstimate(
                        value=value,
                        weights=value,
                        block_centers=np.array([2500.0, 7500.0]),
                        block_values=xp.asarray([[1e6, 2e6], [1e6, 2e6]]),
                    ),
                    num_samples=n,
                    sampling_rate=1e9,
                )
            ),
            "autocorrelation": lambda: plotting.plot_mm_autocorrelation(
                MengaliMorelliEstimate(
                    value=value, weights=value, power=4, autocorrelation=samples
                ),
                sampling_rate=1e9,
            ),
            "timing": lambda: plotting.plot_timing_correlation(
                TimingEstimate(
                    integer=xp.asarray([50, 900]),
                    fractional=xp.asarray([0.0, 0.0]),
                    metric=xp.asarray([10.0, 10.0]),
                    coherence=xp.asarray([1.0, 1.0]),
                    correlation=samples,
                    search_start=10,
                )
            ),
            "pilot tone": lambda: plotting.plot_pilot_tone_phase_estimate(
                PilotToneEstimate(value=phase, tone_frequencies=np.array([1e6, 1e6])),
                samples,
                method=PilotTone(frequency=1e6, bandwidth=1e5),
                sampling_rate=1e9,
            ),
            "pilot tones": lambda: plotting.plot_pilot_tones_phase_estimate(
                PilotTonesEstimate(
                    value=phase[0],
                    tone_frequencies=np.array([1e6, 2e6]),
                    tone_snr_db=np.array([20.0, 20.0]),
                    differential_phase=phase,
                    reference_tone=0,
                    used_tones=(0, 1),
                )
            ),
            "waveform": lambda: plotting.plot_time_domain(samples, sampling_rate=1e9),
            "psd": lambda: plotting.plot_psd(samples, sampling_rate=1e9, nperseg=1024),
            "llr distribution": lambda: plotting.plot_llr_distribution(llrs, bits),
            "equalizer": lambda: plotting.plot_equalizer_result(
                EqualizerResult(
                    y_hat=samples,
                    weights=xp.zeros((2, 2, 11), dtype=xp.complex64),
                    error=samples,
                    phase_trajectory=phase,
                )
            ),
            "constellation": lambda: plotting.plot_constellation(samples[0]),
            "eye": lambda: plotting.plot_eye_diagram(real_wave, sps=8),
            "spectrogram": lambda: plotting.plot_spectrogram(
                samples[0], sampling_rate=1e9, nperseg=256
            ),
        }

    # Image plots transfer a fixed-size 2-D grid (e.g. 500x500), independent of N.
    _IMAGE_LIMITS: dict[str, int] = {
        "constellation": 300 * 300,
        "eye": 500 * 500,
        "spectrogram": 256 * 2500,
    }

    @pytest.mark.gpu_only
    @pytest.mark.parametrize(
        "case",
        [
            "trajectory",
            "decomposition",
            "drift",
            "spectrum",
            "blockwise frequency",
            "autocorrelation",
            "timing",
            "pilot tone",
            "pilot tones",
            "waveform",
            "psd",
            "llr distribution",
            "equalizer",
            "constellation",
            "eye",
            "spectrogram",
        ],
    )  # fmt: skip
    def test_moves_only_the_drawn_points(self, xp: Any, case: str) -> None:
        """Every device-to-host copy is plot-sized (reduced trace or image),
        well below the 2 x 2^18 sample record."""
        from commkit import backend

        sizes: list[int] = []
        real = backend.to_device

        def counting(data: Any, device: str) -> Any:
            if device == "cpu" and isinstance(data, xp.ndarray):
                sizes.append(int(data.size))
            return real(data, device)

        import commkit.plotting.coding as coding_plot
        import commkit.plotting.constellation as const_plot
        import commkit.plotting.equalizer as eq_plot
        import commkit.plotting.eye as eye_plot
        import commkit.plotting.recovery as recovery
        import commkit.plotting.spectral as spec_plot

        # backend._to_host calls backend.to_device; modules that import
        # to_device directly need patching too.
        with (
            patch.object(backend, "to_device", counting),
            patch.object(recovery, "to_device", counting),
            patch.object(coding_plot, "to_device", counting),
            patch.object(const_plot, "to_device", counting),
            patch.object(eq_plot, "to_device", counting),
            patch.object(eye_plot, "to_device", counting),
            patch.object(spec_plot, "to_device", counting),
        ):
            self._cases(xp)[case]()
        plt.close("all")
        limit = self._IMAGE_LIMITS.get(case, 2 * (4000 + 2000))
        assert sizes and max(sizes) <= limit, (case, max(sizes), limit)


# -----------------------------------------------------------------------------
# dark and light themes
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("dark", [True, False])
def test_theme_variants(dark: bool) -> None:
    """Each variant sets matching background, text, cycle and color map, and
    the light one resets what an earlier dark call set."""
    from matplotlib.colors import to_hex, to_rgb

    apply_default_theme(dark=not dark)
    apply_default_theme(dark=dark)
    rc = plt.rcParams
    background, text = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
    if not dark:
        background, text = text, background
    for key in ("figure.facecolor", "axes.facecolor"):
        assert to_rgb(rc[key]) == background, key
    for key in ("text.color", "axes.labelcolor", "xtick.color", "axes.edgecolor"):
        assert to_rgb(rc[key]) == text, key
    assert rc["image.cmap"] == ("afmhot" if dark else "afmhot_r")
    cycle = rc["axes.prop_cycle"].by_key()["color"]
    assert (to_hex(cycle[0]) == "#1f77b4") is (not dark)  # tab10 on white
    assert rc["errorbar.capsize"] == 3
    assert rc["lines.linewidth"] == 2  # shared settings in both variants


def test_density_low_end_matches_the_background() -> None:
    """The color map starts at the background's luminance in both variants."""
    for dark in (True, False):
        apply_default_theme(dark=dark)
        low = np.mean(plt.get_cmap(plt.rcParams["image.cmap"])(0.0)[:3])
        assert (low < 0.1) if dark else (low > 0.8)


@pytest.mark.parametrize("dark", [True, False])
def test_plots_follow_the_theme(xp: Any, backend_device: str, dark: bool) -> None:
    """Guides take the accent color, overlay markers contrast with the
    background, density images use the theme's map with empty cells clear."""
    from matplotlib.colors import to_rgba

    import commkit as ck
    from commkit.mapping import Constellation

    apply_default_theme(dark=dark)
    fg = to_rgba(plt.rcParams["text.color"])
    bg = to_rgba(plt.rcParams["axes.facecolor"])
    sig = ck.generate(
        Constellation.qam(16),
        num_symbols=4000,
        symbol_rate=1e9,
        sps=1,
        rng=1,
        device=backend_device,
    )
    sig = ck.impairments.apply_awgn(sig, esn0_db=25, rng=2)

    _, ax = ck.plotting.plot_constellation(sig, overlay_ideal=True)
    image = ax.images[0]
    assert image.get_cmap().name == plt.rcParams["image.cmap"]
    data = image.get_array()
    assert np.ma.is_masked(data) and data.mask[0, 0]  # corner: no samples
    ideal = ax.collections[-1]
    assert tuple(ideal.get_facecolor()[0]) == fg
    assert tuple(ideal.get_edgecolor()[0]) == bg
    axis_lines = [line for line in ax.lines if line.get_alpha() == 0.4]
    assert axis_lines and all(
        to_rgba(line.get_color())[:3] == fg[:3] for line in axis_lines
    )

    _, ax = ck.plotting.plot_eye_diagram(xp.asarray(np.sin(np.arange(4000) / 3)), sps=8)
    assert ax.images[0].get_cmap().name == plt.rcParams["image.cmap"]

    _, ax = ck.plotting.plot_filter_response(
        [0.25, 0.5, 0.25], quantity="magnitude", relative=True, annotations=True
    )
    guide = ax.lines[-1]
    assert guide.get_color() == theme._accent()
    assert guide.get_linewidth() == pytest.approx(
        0.75 * plt.rcParams["lines.linewidth"]
    )
    assert guide.get_alpha() is None  # full opacity, unlike the grid
    plt.close("all")


def test_marker_and_line_sizes_scale_with_the_theme() -> None:
    with mpl.rc_context({"lines.markersize": 10, "lines.linewidth": 3}):
        assert theme._marker_area(1.5) == pytest.approx(225.0)
        assert theme._line_width(0.5) == pytest.approx(1.5)


_FORBIDDEN_STYLE_PATTERNS = [
    # color keywords or dict entries with a literal other than "C<n>" or "gray"
    r"""\b(?:color|colors|c|ecolor|edgecolors?|facecolors?|mfc|mec)\s*=\s*["'](?!C\d["']|gray["'])""",
    r"""["'](?:color|colors|c|ecolor|edgecolors?|facecolors?|mfc|mec)["']\s*:\s*["'](?!C\d["']|gray["'])""",
    r"""["']#[0-9a-fA-F]{3,8}["']""",  # hex anywhere
    # format strings with a color letter in either order ("r--", "--r", "o-k")
    r"""["'][rgbcmykw](?:--|-\.|-|:|o|\.|v|\^|s|x|\+|\*)+["']""",
    r"""["'](?:--|-\.|-|:|o|\.|v|\^|s|x|\+|\*)+[rgbcmykw]["']""",
    # font / marker / cap sizes (keyword or dict entry, including .5 floats and ms)
    r"""\b(?:fontsize|labelsize|titlesize|capsize|markersize|ms)\s*=\s*(?:\d|\.\d)""",
    r"""["'](?:fontsize|labelsize|titlesize|capsize|markersize|ms)["']\s*:\s*(?:\d|\.\d)""",
    # line widths other than 0 / 0.0 (no edge)
    r"""\b(?:linewidths?|lw)\s*=\s*(?!0(?:\.0*)?\b)(?:\d|\.\d)""",
    r"""["'](?:linewidths?|lw)["']\s*:\s*(?!0(?:\.0*)?\b)(?:\d|\.\d)""",
    r"""\bs\s*=\s*(?:\d|\.\d)""",  # scatter areas
    r"""["']s["']\s*:\s*(?:\d|\.\d)""",
    r"""\bcmap\s*=\s*["']""",
    r"""["']cmap["']\s*:\s*["']""",
    r"""\bfigsize\s*=\s*\(\s*(?:\d|\.\d)""",
    r"""["']figsize["']\s*:\s*\(\s*(?:\d|\.\d)""",
]


def test_plot_sources_take_style_from_rcparams() -> None:
    """No plot module names a color (other than cycle colors and gray), a
    font size, a fixed line width or marker size, or a color map."""
    import re
    from pathlib import Path

    root = Path(theme.__file__).parent
    offenders = []
    for path in sorted(root.glob("*.py")):
        if path.name == "theme.py":
            continue
        for number, line in enumerate(path.read_text().splitlines(), 1):
            code = line.split("#", 1)[0] if not re.search(r"""["']#""", line) else line
            for pattern in _FORBIDDEN_STYLE_PATTERNS:
                if re.search(pattern, code):
                    offenders.append(f"{path.name}:{number}: {line.strip()}")
    assert not offenders, "\n".join(offenders)


def test_style_scan_catches_tricky_forms() -> None:
    """The source scan catches dict literals, reversed formats, ms= and .5 floats."""
    import re

    bad_lines = [
        'kw = {"color": "red"}',
        'kw = {"lw": .5}',
        'kw = {"ms": 4}',
        'kw = {"fontsize": 10}',
        'ax.plot(x, y, "--r")',
        'ax.plot(x, y, "o-k")',
        "ax.plot(x, y, ms=6)",
        "ax.plot(x, y, lw=.5)",
        "ax.scatter(x, y, s=.5)",
    ]
    for line in bad_lines:
        assert any(re.search(p, line) for p in _FORBIDDEN_STYLE_PATTERNS), line

    good_lines = [
        'kw = {"color": "C0", "lw": 0}',
        'kw = {"color": "gray", "linewidth": 0.0}',
        'ax.plot(x, y, "--", color="C1", linewidth=_line_width(0.8))',
    ]
    for line in good_lines:
        assert not any(re.search(p, line) for p in _FORBIDDEN_STYLE_PATTERNS), line


def test_accent_follows_the_background_and_stands_apart() -> None:
    """The guide color is a light red on a dark background and a deeper one
    on a light background (a user's own style included): a saturated hue
    against the gray grid, readable against either background."""
    from matplotlib.colors import rgb_to_hsv, to_rgb

    def luminance(color: Any) -> float:
        r, g, b = to_rgb(color)
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    for dark in (True, False):
        apply_default_theme(dark=dark)
        accent = theme._accent()
        assert accent == (theme._ACCENT_ON_DARK if dark else theme._ACCENT_ON_LIGHT)
        hue, saturation, _ = rgb_to_hsv(to_rgb(accent))
        assert min(hue, 1 - hue) < 0.03 and saturation > 0.5  # a red
        assert rgb_to_hsv(to_rgb(plt.rcParams["grid.color"]))[1] < 0.1  # gray
        # Luminance at least 0.3 away from the background's.
        gap = abs(luminance(accent) - luminance(plt.rcParams["axes.facecolor"]))
        assert gap > 0.3
    with mpl.rc_context({"axes.facecolor": "#202020"}):
        assert theme._accent() == theme._ACCENT_ON_DARK
    with mpl.rc_context({"axes.facecolor": "#f0f0e8"}):
        assert theme._accent() == theme._ACCENT_ON_LIGHT


def test_si_formatter_preserves_fractional_tick_offsets() -> None:
    """Ticks with an integer step and fractional offset keep the needed decimal."""
    fig, ax = plt.subplots()
    ax.set_xlabel("Frequency [Hz]")
    ax.set_xlim(0.0, 3.0e6)
    fmt = theme._SIAxisFormatter()
    ax.xaxis.set_major_formatter(fmt)
    locs = [0.5e6, 1.5e6, 2.5e6]
    fmt.set_locs(locs)
    labels = [fmt(v) for v in locs]
    assert labels == ["0.5", "1.5", "2.5"]
    assert ax.get_xlabel() == "Frequency [MHz]"
    plt.close(fig)
