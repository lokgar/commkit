"""Tests for the channel-coding plots (plotting/coding.py)."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytest

from commkit import Constellation, generate
from commkit.coding import (
    ErrorRateCurve,
    ExitCurve,
    LDPCCode,
    LinearBlockCode,
    MaximumLikelihood,
    decode,
    exit_chart,
    ldpc_exit_curves,
)
from commkit.impairments import apply_awgn
from commkit.mapping import compute_llr
from commkit.plotting import (
    plot_error_rate,
    plot_exit_chart,
    plot_llr_distribution,
    plot_parity_check,
)


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

    def test_decode_result_with_its_reference(self, backend_device: str) -> None:
        """A DecodeResult draws its a-posteriori LLRs against the transmitted
        bits of the Signal, without the uncoded filler (1500 symbols of
        16-QAM carry 6000 bits: 857 Hamming(7, 4) blocks and 1 filler bit)."""
        code = LinearBlockCode.hamming(3)
        tx = generate(Constellation.qam(16), num_symbols=15 * 100, symbol_rate=1e9, code=code,
                      pad=True, rng=1, device=backend_device)  # fmt: skip
        rx = apply_awgn(tx, esn0_db=8.0, rng=2)
        llrs = compute_llr(rx, noise_var=10**-0.8)[: tx.reference.num_blocks * code.n]
        res = decode(llrs, code=code, method=MaximumLikelihood())
        assert res.llrs.shape[-1] < tx.reference.bits.shape[-1]
        _, (ax_pdf, _) = plot_llr_distribution(res, tx)
        assert len(ax_pdf.patches) == 2
        plot_llr_distribution(res, tx.reference, quantity="density")
        with pytest.raises(ValueError, match="has no LLRs"):
            plot_llr_distribution(decode(llrs, code=code), tx)
        with pytest.raises(ValueError, match="needs the transmitted Signal"):
            plot_llr_distribution(res, None)
        plt.close("all")


def _curve(label: str = "", checked: bool = True) -> ErrorRateCurve:
    return ErrorRateCurve(
        esn0_db=[1.0, 2.0, 3.0],
        num_blocks=[100, 100, 1000],
        bit_errors=[40, 8, 0],
        block_errors=[20, 4, 0],
        pre_fec_bit_errors=[300, 200, 1000],
        detected_failures=[15, 3, 0] if checked else [0, 0, 0],
        undetected_errors=[5, 1, 0] if checked else [20, 4, 0],
        total_iterations=[100, 100, 1000],
        k=4,
        n=8,
        bits_per_symbol=2,
        checked=checked,
        label=label,
    )


class TestPlotErrorRate:
    def test_rates_and_intervals(self) -> None:
        """BER, FER and pre-FEC BER of one curve, zeros left out; the FER
        interval is Clopper-Pearson (95 %): 20 of 100 gives [0.127, 0.292],
        and the zero-error point marks its upper bound 3.7e-3 (1000 blocks)."""
        _, (ax, ax_out) = plot_error_rate(_curve())
        ber, fer, pre = ax.get_lines()[:3]
        np.testing.assert_allclose(ber.get_ydata(), [0.1, 0.02, np.nan])
        np.testing.assert_allclose(fer.get_ydata(), [0.2, 0.04, np.nan])
        np.testing.assert_allclose(pre.get_ydata(), [0.375, 0.25, 0.125])
        bars = ax.collections[0].get_segments()
        np.testing.assert_allclose(
            [bars[0][0][1], bars[0][1][1]], [0.1266, 0.2918], atol=1e-4
        )
        (bound,) = [line for line in ax.get_lines() if line.get_marker() == "v"]
        np.testing.assert_allclose(bound.get_ydata(), [3.682e-3], rtol=1e-3)
        assert [t.get_text() for t in ax.get_legend().get_texts()] == [
            "BER", "FER", "pre-FEC BER",
        ]  # fmt: skip
        undetected, detected = ax_out.get_lines()[:2]
        np.testing.assert_allclose(undetected.get_ydata(), [0.05, 0.01, np.nan])
        np.testing.assert_allclose(detected.get_ydata(), [0.15, 0.03, np.nan])
        plt.close("all")

    def test_several_curves_one_entry_each(self) -> None:
        """Curves overlay with one legend entry each and a gray style key; an
        unchecked decoder has no detected failures."""
        fig, ax = plot_error_rate(
            [_curve("a"), _curve("b", checked=False)], quantity="outcome"
        )
        texts = [t.get_text() for t in ax.get_legend().get_texts()]
        assert texts == ["a", "b", "undetected errors", "detected failures"]
        # a: undetected + detected; b: undetected only; two key entries.
        assert len(ax.get_lines()) == 2 + 1 + 2
        plt.close("all")

    def test_ebn0_axis(self) -> None:
        """Eb/N0 = Es/N0 - 10 log10(k/n * bits per symbol) = Es/N0 here."""
        _, ax = plot_error_rate(_curve(), x_axis="ebn0", quantity="error_rate")
        np.testing.assert_allclose(ax.get_lines()[0].get_xdata(), [1.0, 2.0, 3.0])
        assert ax.get_xlabel() == "Eb/N0 [dB]"
        plt.close("all")

    def test_unknown_x_axis_raises(self) -> None:
        with pytest.raises(ValueError, match="x_axis"):
            plot_error_rate(_curve(), x_axis="snr")

    def test_overlay_on_given_axes(self) -> None:
        fig, ax = plt.subplots()
        plot_error_rate(_curve("a"), quantity="error_rate", ax=ax, pre_fec=False)
        plot_error_rate(_curve("b"), quantity="error_rate", ax=ax, intervals=False)
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert "a" in labels and "b" in labels
        plt.close("all")

    def test_errors(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            plot_error_rate([])
        with pytest.raises(TypeError, match="ErrorRateCurve"):
            plot_error_rate([np.zeros(3)])
        with pytest.raises(ValueError, match="single curve"):
            plot_error_rate([_curve(), _curve()], label="x")
        with pytest.raises(ValueError, match="require a selected quantity"):
            plot_error_rate(_curve(), xlim=(0, 1))
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


def test_llr_selected_density_overlay(xp: Any) -> None:
    """Two demappers can be compared on a single selected density panel."""
    bits = xp.asarray([0, 1] * 300, dtype=xp.int8)
    llrs = xp.asarray([2.0, -2.0] * 300, dtype=xp.float32)
    fig, ax = plot_llr_distribution(llrs, bits, quantity="density", label="first")
    fig2, ax2 = plot_llr_distribution(
        llrs, bits, quantity="density", label="second", ax=ax, legend=True
    )
    assert fig2 is fig and ax2 is ax
    assert [t.get_text() for t in ax.get_legend().get_texts()] == [
        "first b = 0",
        "first b = 1",
        "second b = 0",
        "second b = 1",
    ]


def test_parity_check_selected_panels() -> None:
    """The matrix and degree distribution can be displayed independently."""
    h = np.asarray([[1, 1, 0], [0, 1, 1]], dtype=np.int8)
    _, matrix = plot_parity_check(h, quantity="matrix", xlim=(-0.5, 2.5))
    _, degrees = plot_parity_check(h, quantity="degrees", legend=False)
    assert matrix.get_xlim() == (-0.5, 2.5)
    assert len(degrees.patches) > 0
    assert degrees.get_legend() is None
    _, degrees = plot_parity_check(h, quantity="degrees", label="first")
    _, degrees = plot_parity_check(
        h, quantity="degrees", label="second", ax=degrees, legend=True
    )
    assert [t.get_text() for t in degrees.get_legend().get_texts()] == [
        "first variable nodes",
        "first check nodes",
        "second variable nodes",
        "second check nodes",
    ]
