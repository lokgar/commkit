"""Tests for EXIT curves and charts (coding/exit.py)."""

import logging
from typing import Any

import numpy as np
import pytest
from scipy import integrate

from commkit import generate, metrics
from commkit.backend import to_device
from commkit.coding import (
    BCJR,
    ConvolutionalCode,
    ExitChart,
    ExitCurve,
    LinearBlockCode,
    Syndrome,
    TurboCode,
    decoder_exit_curve,
    demapper_exit_curve,
    exit_chart,
)
from commkit.coding.exit import _information, _j
from commkit.impairments import apply_awgn
from commkit.mapping import Constellation

GRID = np.linspace(0.0, 1.0, 101)


def _gaussian_information(sigma: float) -> float:
    """J(sigma) by adaptive quadrature: the information of N(s²/2, s²) LLRs."""
    mu = sigma * sigma / 2

    def pdf(v: float) -> float:
        return np.exp(-((v - mu) ** 2) / (2 * sigma**2)) / np.sqrt(2 * np.pi * sigma**2)

    loss = integrate.quad(
        lambda v: pdf(v) * np.logaddexp(0, -v) / np.log(2),
        mu - 30 * sigma,
        mu + 30 * sigma,
        limit=200,
    )
    return 1.0 - loss[0]


def _natural_16qam() -> Constellation:
    """16-QAM with natural binary labels over the grid (not Gray)."""
    pts = Constellation.qam(16).points
    labels = np.zeros((16, 4), dtype=np.int8)
    labels[np.lexsort((pts.imag, pts.real))] = (
        np.arange(16)[:, None] >> np.arange(3, -1, -1)
    ) & 1
    return Constellation(pts, bit_labels=labels)


class TestInformation:
    @pytest.mark.parametrize("sigma", [0.5, 1.5, 3.0, 6.0])
    def test_gaussian_llrs(self, xp: Any, sigma: float) -> None:
        """The histogram estimate and the J table both match quadrature."""
        rng = np.random.default_rng(0)
        bits = rng.integers(0, 2, 200_000)
        llrs = (1 - 2 * bits) * sigma**2 / 2 + sigma * rng.standard_normal(bits.size)
        expected = _gaussian_information(sigma)
        assert float(_j(sigma)) == pytest.approx(expected, abs=1e-6)
        got = _information(xp.asarray(llrs, dtype=xp.float32), xp.asarray(bits))
        assert got == pytest.approx(expected, abs=5e-3)

    def test_independent_of_scaling(self, xp: Any) -> None:
        """Information is a property of the LLRs' distributions, not their
        calibration: overconfident and underconfident LLRs carry the same."""
        rng = np.random.default_rng(1)
        bits = xp.asarray(rng.integers(0, 2, 100_000))
        llrs = (1 - 2 * bits) * 2.0 + 2.0 * xp.asarray(rng.standard_normal(bits.size))
        base = _information(llrs, bits)
        for scale in (0.2, 5.0):
            assert _information(scale * llrs, bits) == pytest.approx(base, abs=2e-3)


class TestExitCurve:
    def test_stores_read_only_host_arrays(self, xp: Any) -> None:
        curve = ExitCurve(xp.asarray(GRID), xp.asarray(GRID**2), "square")
        assert isinstance(curve.a_priori, np.ndarray)
        assert not curve.extrinsic.flags.writeable
        assert curve.label == "square"

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="one length"):
            ExitCurve(GRID, GRID[:-1])
        with pytest.raises(ValueError, match="increase"):
            ExitCurve(GRID[::-1], GRID)
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            ExitCurve(GRID, 2 * GRID)
        with pytest.raises(ValueError, match="finite"):
            ExitCurve(GRID, np.full(GRID.size, np.nan))


class TestExitChart:
    def test_open_tunnel(self) -> None:
        """First curve 0.3 + 0.7 x against the identity: x_n = 1 - 0.7^n,
        which passes 0.99 after 13 iterations."""
        chart = exit_chart(ExitCurve(GRID, 0.3 + 0.7 * GRID), ExitCurve(GRID, GRID))
        assert isinstance(chart, ExitChart)
        assert chart.converges
        assert chart.iterations == 13
        assert chart.trajectory[-1, 0] >= 0.99
        np.testing.assert_allclose(chart.trajectory[0], [0, 0])

    def test_closed_tunnel_stalls_at_the_crossing(self, caplog: Any) -> None:
        """0.2 + 0.5 x against the identity crosses at x = 0.4."""
        first = ExitCurve(GRID, 0.2 + 0.5 * GRID, "inner")
        with caplog.at_level(logging.INFO, logger="commkit"):
            chart = exit_chart(first, ExitCurve(GRID, GRID, "outer"))
        assert not chart.converges
        assert chart.narrowest == pytest.approx(0.4, abs=1e-3)
        assert chart.trajectory[-1, 0] == pytest.approx(0.4, abs=1e-3)
        (message,) = [r.message for r in caplog.records]
        assert "inner against outer: tunnel closed at I_A = 0.400" in message

    def test_validation(self) -> None:
        short = ExitCurve(np.linspace(0.0, 0.5, 11), np.linspace(0.0, 0.5, 11))
        with pytest.raises(ValueError, match="span"):
            exit_chart(short, ExitCurve(GRID, GRID))
        with pytest.raises(TypeError, match="ExitCurve"):
            exit_chart(GRID, ExitCurve(GRID, GRID))  # type: ignore[arg-type]


class TestDecoderExitCurve:
    CODE = ConvolutionalCode.k7(info_bits=200)

    def test_area_theorem(self, backend_device: str, caplog: Any) -> None:
        """For an exact a-posteriori decoder with only a-priori input, the
        area under the curve is 1 - k / n (Ashikhmin, Kramer, ten Brink)."""
        with caplog.at_level(logging.INFO, logger="commkit"):
            curve = decoder_exit_curve(
                self.CODE,
                method=BCJR(method="exact"),
                num_blocks=100,
                rng=0,
                device=backend_device,
            )
        area = np.trapezoid(curve.extrinsic, curve.a_priori)
        assert area == pytest.approx(1 - self.CODE.k / self.CODE.n, abs=0.02)
        assert curve.extrinsic[0] == pytest.approx(0.0, abs=0.01)
        assert curve.extrinsic[-1] == pytest.approx(1.0, abs=0.01)
        assert "area" in caplog.records[-1].message

    def test_reproducible(self) -> None:
        kwargs = dict(method=BCJR(), num_blocks=20, points=3, rng=5)
        a = decoder_exit_curve(self.CODE, **kwargs)
        b = decoder_exit_curve(self.CODE, **kwargs)
        np.testing.assert_array_equal(a.extrinsic, b.extrinsic)

    def test_needs_soft_output(self) -> None:
        with pytest.raises(TypeError, match="no a-posteriori LLRs"):
            decoder_exit_curve(LinearBlockCode.hamming(3), method=Syndrome(), points=2)
        with pytest.raises(TypeError, match="no a-posteriori LLRs"):
            decoder_exit_curve(TurboCode.lte(40), num_blocks=2, points=2)

    def test_validation(self) -> None:
        with pytest.raises(TypeError, match="code object"):
            decoder_exit_curve("ldpc")
        with pytest.raises(ValueError, match="points"):
            decoder_exit_curve(self.CODE, points=1)
        with pytest.raises(ValueError, match="num_blocks"):
            decoder_exit_curve(self.CODE, num_blocks=0)
        with pytest.raises(ValueError, match="device"):
            decoder_exit_curve(self.CODE, device="tpu")


class TestDemapperExitCurve:
    def test_gray_qpsk_is_flat(self, backend_device: str) -> None:
        """The two bits of Gray QPSK are independent BPSK channels: a-priori
        information on one says nothing about the other, and each carries
        J(2 / sqrt(noise_var)) (LLR mean 2/nv, variance 4/nv)."""
        nv = 0.5
        curve = demapper_exit_curve(
            Constellation.qam(4),
            noise_var=nv,
            num_symbols=1 << 16,
            rng=1,
            device=backend_device,
        )
        expected = _gaussian_information(2 / np.sqrt(nv))
        np.testing.assert_allclose(curve.extrinsic, expected, atol=8e-3)

    def test_without_a_priori_it_is_the_gmi_per_bit(self, xp: Any) -> None:
        """At I_A = 0 the demapper's LLRs are the plain exact LLRs, whose
        information per bit is the GMI over the bits per symbol."""
        nv = 10**-0.8
        c = Constellation.qam(16)
        curve = demapper_exit_curve(
            c, noise_var=nv, points=2, num_symbols=1 << 15, rng=2
        )
        sig = generate(c, num_symbols=1 << 15, symbol_rate=1e9, rng=3)
        rx = apply_awgn(sig, noise_var=nv, rng=4)
        gmi = metrics.gmi(rx, noise_var=nv, method="exact")
        assert curve.extrinsic[0] == pytest.approx(gmi / 4, abs=0.01)

    def test_natural_labels_gain_more_from_a_priori(self) -> None:
        """Gray labels give the most information without a-priori input and
        gain little from it; natural labels start lower and gain more, which
        is what iterative demapping exploits."""
        nv = 10**-0.6
        gray = demapper_exit_curve(Constellation.qam(16), noise_var=nv, rng=5)
        natural = demapper_exit_curve(_natural_16qam(), noise_var=nv, rng=5)
        assert gray.extrinsic[0] > natural.extrinsic[0]
        assert natural.extrinsic[-1] > gray.extrinsic[-1]
        assert natural.extrinsic[-1] - natural.extrinsic[0] > 0.1
        assert gray.extrinsic[-1] - gray.extrinsic[0] < 0.03

    def test_maxlog_loses_information(self) -> None:
        kwargs = dict(noise_var=10**-0.6, points=2, rng=6)
        exact = demapper_exit_curve(_natural_16qam(), **kwargs)
        maxlog = demapper_exit_curve(_natural_16qam(), method="maxlog", **kwargs)
        assert maxlog.extrinsic[0] <= exact.extrinsic[0] + 5e-3
        assert "max-log" in maxlog.label

    def test_chart_with_a_decoder_converges_only_with_natural_labels(self) -> None:
        """At 6 dB the K = 7 code ends higher with natural labels than with
        Gray labels after iterating."""
        nv = 10**-0.6
        decoder = decoder_exit_curve(
            ConvolutionalCode.k7(info_bits=200), method=BCJR(method="exact"), rng=7
        )
        gray = exit_chart(
            demapper_exit_curve(Constellation.qam(16), noise_var=nv), decoder
        )
        natural = exit_chart(
            demapper_exit_curve(_natural_16qam(), noise_var=nv), decoder
        )
        assert natural.trajectory[-1, 0] > gray.trajectory[-1, 0] + 0.1

    def test_validation(self) -> None:
        c = Constellation.qam(16)
        with pytest.raises(TypeError, match="Constellation"):
            demapper_exit_curve("16qam", noise_var=0.1)
        with pytest.raises(ValueError, match="shaped"):
            demapper_exit_curve(c.shaped(nu=0.1), noise_var=0.1)
        with pytest.raises(ValueError, match="noise_var"):
            demapper_exit_curve(c, noise_var=0.0)
        with pytest.raises(ValueError, match="method"):
            demapper_exit_curve(c, noise_var=0.1, method="approx")
        with pytest.raises(ValueError, match="num_symbols"):
            demapper_exit_curve(c, noise_var=0.1, num_symbols=0)
        with pytest.raises(ValueError, match="device"):
            demapper_exit_curve(c, noise_var=0.1, device="tpu")


def test_curves_on_the_gpu_match_the_cpu(backend_device: str) -> None:
    """Realizations differ between devices; the measured information agrees."""
    code = ConvolutionalCode.k7(info_bits=200)
    kwargs = dict(method=BCJR(method="exact"), num_blocks=100, points=5, rng=8)
    cpu = decoder_exit_curve(code, **kwargs)
    here = decoder_exit_curve(code, device=backend_device, **kwargs)
    np.testing.assert_allclose(
        to_device(here.extrinsic, "cpu"), cpu.extrinsic, atol=0.02
    )
