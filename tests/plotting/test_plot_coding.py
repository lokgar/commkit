"""Tests for the channel-coding plots (plotting/coding.py)."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytest

from commkit.coding import (
    ExitCurve,
    LDPCCode,
    LinearBlockCode,
    exit_chart,
    ldpc_exit_curves,
)
from commkit.plotting import plot_exit_chart, plot_llr_distribution, plot_parity_check


def _bpsk_llrs(xp: Any, n: int, sigma: float, scale: float = 1.0, seed: int = 0):
    """True LLRs 2y/sigma² of BPSK on AWGN, times ``scale``, and their bits."""
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, n)
    y = (1 - 2 * bits) + sigma * rng.standard_normal(n)
    llrs = (scale * 2 * y / sigma**2).astype(np.float32)
    return xp.asarray(llrs), xp.asarray(bits.astype(np.int8))


def _slope(line: Any) -> float:
    x, y = line.get_data()
    return float(np.polyfit(x, y, 1)[0])


class TestPlotLLRDistribution:
    def test_true_llrs_are_consistent(self, xp: Any) -> None:
        """log p(L|0)/p(L|1) = L for true LLRs: the measured slope is 1."""
        llrs, bits = _bpsk_llrs(xp, 200_000, sigma=0.8)
        _, (_, ax_cons) = plot_llr_distribution(llrs, bits)
        measured = ax_cons.get_lines()[0]
        assert _slope(measured) == pytest.approx(1.0, abs=0.05)
        plt.close("all")

    def test_overconfident_llrs_have_a_flat_slope(self, xp: Any) -> None:
        """LLRs scaled by 3 (noise variance a third of the true one) have a
        slope of 1/3."""
        llrs, bits = _bpsk_llrs(xp, 200_000, sigma=0.8, scale=3.0)
        _, (_, ax_cons) = plot_llr_distribution(llrs, bits)
        assert _slope(ax_cons.get_lines()[0]) == pytest.approx(1 / 3, abs=0.03)
        plt.close("all")

    def test_densities_integrate_to_one(self, xp: Any) -> None:
        llrs, bits = _bpsk_llrs(xp, 50_000, sigma=1.0)
        _, (ax_pdf, _) = plot_llr_distribution(llrs, bits, bins=50)
        for patch in ax_pdf.patches:  # one StepPatch per bit value
            values, edges, _ = patch.get_data()
            # Up to the 0.1 % of LLRs outside the histogram range.
            assert np.sum(values * np.diff(edges)) == pytest.approx(1.0, abs=2e-3)
        plt.close("all")

    def test_one_curve_per_bit_position(self, xp: Any) -> None:
        llrs, bits = _bpsk_llrs(xp, 4 * 20_000, sigma=0.8)
        llrs, bits = llrs.reshape(2, -1), bits.reshape(2, -1)
        _, (_, ax_cons) = plot_llr_distribution(llrs, bits, bits_per_symbol=4)
        labels = [line.get_label() for line in ax_cons.get_lines()]
        assert labels == ["bit 0", "bit 1", "bit 2", "bit 3", "consistent"]
        plt.close("all")

    def test_axes_and_title(self, xp: Any) -> None:
        llrs, bits = _bpsk_llrs(xp, 10_000, sigma=0.8)
        fig, axes = plt.subplots(1, 2)
        out_fig, out_axes = plot_llr_distribution(llrs, bits, ax=axes, title="LLRs")
        assert out_fig is fig and out_axes[0] is axes[0]
        assert fig._suptitle.get_text() == "LLRs"
        plt.close("all")

    def test_errors(self, xp: Any) -> None:
        llrs, bits = _bpsk_llrs(xp, 1000, sigma=0.8)
        with pytest.raises(ValueError, match="shape"):
            plot_llr_distribution(llrs, bits[:-1])
        with pytest.raises(ValueError, match="0 or 1"):
            plot_llr_distribution(llrs, 2 * bits)
        with pytest.raises(ValueError, match="both 0 and 1"):
            plot_llr_distribution(llrs, xp.zeros_like(bits))
        with pytest.raises(ValueError, match="bits_per_symbol"):
            plot_llr_distribution(llrs, bits, bits_per_symbol=3)
        with pytest.raises(TypeError, match="real"):
            plot_llr_distribution(llrs.astype(xp.complex64), bits)
        with pytest.raises(TypeError, match="integers or bool"):
            plot_llr_distribution(llrs, bits.astype(xp.float32))
        plt.close("all")


class TestPlotParityCheck:
    def test_degrees_are_the_node_fractions(self) -> None:
        """The bars are the fractions of columns and rows of each degree."""
        code = LinearBlockCode.hamming(4)
        h = np.asarray(code.parity_check)
        _, (_, ax_deg) = plot_parity_check(code)
        var_bars, chk_bars = ax_deg.containers
        for bars, degrees in ((var_bars, h.sum(axis=0)), (chk_bars, h.sum(axis=1))):
            values, counts = np.unique(degrees, return_counts=True)
            heights = [p.get_height() for p in bars.patches]
            np.testing.assert_allclose(heights, counts / degrees.size)
            centers = [p.get_x() + p.get_width() / 2 for p in bars.patches]
            np.testing.assert_allclose(np.round(centers), values)
        plt.close("all")

    def test_small_matrix_is_an_image(self) -> None:
        h = np.array([[1, 1, 0, 1], [0, 1, 1, 1]])
        _, (ax_h, _) = plot_parity_check(h)
        np.testing.assert_array_equal(ax_h.get_images()[0].get_array(), h)
        assert ax_h.get_title() == "H: 2 x 4, 6 ones"
        plt.close("all")

    def test_large_matrix_marks_each_edge(self) -> None:
        code = LDPCCode.dvbs2("1/2", "short")
        _, (ax_h, _) = plot_parity_check(code)
        (marks,) = ax_h.get_lines()
        cols, rows = marks.get_data()
        assert len(cols) == code.checks.size
        assert set(zip(rows, cols, strict=True)) == set(
            zip(code.checks, code.variables, strict=True)
        )
        assert marks.get_rasterized()
        plt.close("all")

    def test_punctured_columns_are_shaded(self) -> None:
        h = LDPCCode.from_parity_check(
            np.array([[1, 1, 1, 0, 0], [0, 1, 0, 1, 1], [1, 0, 0, 1, 0]])
        ).parity_check()
        code = LDPCCode.from_parity_check(h, punctured=1)
        _, (ax_h, _) = plot_parity_check(code)
        (span,) = ax_h.patches
        x0, x1 = span.get_x(), span.get_x() + span.get_width()
        assert (x0, x1) == pytest.approx((3.5, 4.5))
        plt.close("all")

    def test_errors(self) -> None:
        with pytest.raises(ValueError, match="only 0 and 1"):
            plot_parity_check(np.array([[1, 2], [0, 1]]))
        plt.close("all")


class TestPlotExitChart:
    GRID = np.linspace(0.0, 1.0, 101)

    def test_draws_the_chart(self) -> None:
        chart = exit_chart(
            *ldpc_exit_curves(LDPCCode.dvbs2("1/2", "short"), channel_mi=0.55)
        )
        _, ax = plot_exit_chart(chart)
        first, second, trajectory = ax.get_lines()
        np.testing.assert_array_equal(first.get_xydata()[:, 1], chart.first.extrinsic)
        # The second curve is drawn with swapped axes.
        np.testing.assert_array_equal(second.get_xydata()[:, 0], chart.second.extrinsic)
        np.testing.assert_array_equal(second.get_xydata()[:, 1], chart.second.a_priori)
        np.testing.assert_array_equal(trajectory.get_xydata(), chart.trajectory)
        assert first.get_label() == chart.first.label
        assert "tunnel open" in ax.get_title()
        plt.close("all")

    def test_closed_tunnel_marks_the_stall(self) -> None:
        chart = exit_chart(
            ExitCurve(self.GRID, 0.2 + 0.5 * self.GRID), ExitCurve(self.GRID, self.GRID)
        )
        _, ax = plot_exit_chart(chart)
        stall = ax.get_lines()[-1]
        assert stall.get_label() == "stalls"
        np.testing.assert_allclose(stall.get_xydata()[0], chart.trajectory[-1])
        assert "tunnel closed" in ax.get_title()
        plt.close("all")

    def test_without_trajectory(self) -> None:
        chart = exit_chart(
            ExitCurve(self.GRID, 0.3 + 0.7 * self.GRID), ExitCurve(self.GRID, self.GRID)
        )
        _, ax = plot_exit_chart(chart, trajectory=False)
        assert len(ax.get_lines()) == 2
        plt.close("all")

    def test_errors(self) -> None:
        with pytest.raises(TypeError, match="ExitChart"):
            plot_exit_chart(np.zeros(3))
