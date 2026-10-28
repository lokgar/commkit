"""Tests for soft-decision (LLR) demapping."""

from typing import Any

import numpy as np
import pytest

from commkit import mapping
from commkit.core import Signal
from commkit.mapping import Constellation
from tests.common.conversions import to_numpy


class TestComputeLLRCore:
    """Tests for algorithmic accuracy, exact vs max-log approximations, and MIMO support."""

    def test_compute_llr_sign_correctness(self, xp: Any, xpt: Any) -> None:
        """LLR sign should match hard decision at high SNR (noiseless symbols)."""
        order = 16
        constellation = Constellation.qam(order)

        bits = xp.array(
            [0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0], dtype=xp.int32
        )
        symbols = mapping.map_bits(bits, constellation=constellation)

        llrs = mapping.compute_llr(
            symbols, noise_var=1e-6, constellation=constellation, method="maxlog"
        )
        hard_from_llr = (to_numpy(llrs) < 0).astype("int32")
        xpt.assert_array_equal(xp.asarray(hard_from_llr), bits)

    def test_compute_llr_roundtrip(self, xp: Any, xpt: Any) -> None:
        """Hard decision from LLR matches direct hard demapping at high SNR."""
        order = 8
        constellation = Constellation.psk(order)

        bits = xp.array([0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0], dtype=xp.int32)
        symbols = mapping.map_bits(bits, constellation=constellation)

        llrs = mapping.compute_llr(symbols, noise_var=1e-6, constellation=constellation)
        hard_from_llr = (to_numpy(llrs) < 0).astype("int32")
        hard_direct = mapping.demap_symbols_hard(symbols, constellation=constellation)

        xpt.assert_array_equal(xp.asarray(hard_from_llr), hard_direct)

    def test_compute_llr_exact_vs_maxlog(self, xp: Any) -> None:
        """Exact and max-log methods agree on sign and remain close in magnitude."""
        order = 4
        constellation = Constellation.qam(order)

        bits = xp.array([0, 0, 0, 1, 1, 0, 1, 1], dtype=xp.int32)
        symbols = mapping.map_bits(bits, constellation=constellation)

        llrs_maxlog = to_numpy(
            mapping.compute_llr(
                symbols, noise_var=0.01, constellation=constellation, method="maxlog"
            )
        )
        llrs_exact = to_numpy(
            mapping.compute_llr(
                symbols, noise_var=0.01, constellation=constellation, method="exact"
            )
        )

        assert np.array_equal(np.sign(llrs_maxlog), np.sign(llrs_exact))
        ratio = np.abs(llrs_maxlog) / (np.abs(llrs_exact) + 1e-10)
        np.testing.assert_array_equal((ratio > 0.5) & (ratio < 2.0), True)

    def test_compute_llr_mimo_shape(self, xp: Any) -> None:
        """compute_llr preserves multi-channel MIMO structure."""
        order = 4
        constellation = Constellation.qam(order)

        bits = xp.zeros(16, dtype=xp.int32)
        symbols = mapping.map_bits(bits, constellation=constellation).reshape(2, 4)

        llrs = mapping.compute_llr(symbols, noise_var=0.1, constellation=constellation)
        assert llrs.shape == (2, 8)

    def test_compute_llr_methods_agree(self, xp: Any) -> None:
        """Verify maxlog and exact methods agree on sign for 16-QAM."""
        order = 16
        constellation = Constellation.qam(order)
        bits = xp.array([0, 0, 1, 1, 0, 1, 0, 1], dtype=xp.int32)
        symbols = mapping.map_bits(bits, constellation=constellation)
        noise_var = 0.1

        llrs_maxlog = to_numpy(
            mapping.compute_llr(
                symbols,
                noise_var=noise_var,
                constellation=constellation,
                method="maxlog",
            )
        )
        llrs_exact = to_numpy(
            mapping.compute_llr(
                symbols,
                noise_var=noise_var,
                constellation=constellation,
                method="exact",
            )
        )
        assert np.array_equal(np.sign(llrs_exact), np.sign(llrs_maxlog))

    def test_compute_llr_real_symbols(self, xp: Any) -> None:
        """Real-valued (PAM) symbols give float32 LLRs on the input's device."""
        bits = np.array([0, 0, 0, 1, 1, 0, 1, 1], dtype="int32")
        symbols = xp.asarray(mapping.map_bits(bits, constellation=Constellation.pam(4)))
        assert not xp.iscomplexobj(symbols)

        llrs = mapping.compute_llr(
            symbols, noise_var=0.1, constellation=Constellation.pam(4)
        )
        assert isinstance(llrs, xp.ndarray)
        assert llrs.dtype == xp.float32
        assert llrs.shape == (len(bits),)

    def test_qpsk_llr_matches_closed_form(self, xp: Any, xpt: Any) -> None:
        """Gray QPSK: each bit depends on one quadrature only, so exact and
        max-log coincide and |LLR| = 2*sqrt(2)*|y_I or y_Q| / sigma^2."""
        rng = np.random.default_rng(3)
        y = (rng.standard_normal(64) + 1j * rng.standard_normal(64)).astype(
            np.complex64
        )
        sigma2 = 0.3
        exact = mapping.compute_llr(
            xp.asarray(y),
            noise_var=sigma2,
            constellation=Constellation.qam(4),
            method="exact",
        )
        maxlog = mapping.compute_llr(
            xp.asarray(y),
            noise_var=sigma2,
            constellation=Constellation.qam(4),
            method="maxlog",
        )
        xpt.assert_allclose(exact, maxlog, rtol=1e-4, atol=1e-4)
        llr = to_numpy(exact).reshape(-1, 2)
        expected = (
            2 * np.sqrt(2) / sigma2 * np.stack([np.abs(y.real), np.abs(y.imag)], 1)
        )
        np.testing.assert_allclose(
            np.sort(np.abs(llr), axis=1), np.sort(expected, axis=1), rtol=1e-4
        )

    def test_compute_llr_validation(self, xp: Any) -> None:
        """An unknown method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown method"):
            mapping.compute_llr(
                xp.ones(1, dtype=xp.complex64),
                noise_var=0.1,
                constellation=Constellation.qam(4),
                method="magic",
            )


class TestComputeLLRDevice:
    """LLRs are float32 and stay on the input's device."""

    def test_output_on_input_device(self, xp: Any) -> None:
        bits = np.array([0, 0, 1, 1, 0, 1, 0, 1], dtype="int32")
        symbols = xp.asarray(
            mapping.map_bits(bits, constellation=Constellation.qam(16))
        )
        for method in ("maxlog", "exact"):
            llrs = mapping.compute_llr(
                symbols,
                noise_var=0.1,
                constellation=Constellation.qam(16),
                method=method,
            )
            assert isinstance(llrs, xp.ndarray)
            assert llrs.dtype == xp.float32


def _noisy(constellation: Any, shape: tuple[int, ...], sd: float, seed: int) -> Any:
    """Points of ``constellation`` plus Gaussian noise, in single precision."""
    rng = np.random.default_rng(seed)
    pts = constellation.points
    x = pts[rng.integers(0, pts.size, shape)] + sd * rng.standard_normal(shape)
    if np.iscomplexobj(pts):
        x = x + 1j * sd * rng.standard_normal(shape)
    return x.astype(np.complex64 if np.iscomplexobj(x) else np.float32)


@pytest.mark.gpu_only
class TestLLRKernel:
    """The CUDA LLR kernels against their CPU references (Numba)."""

    @pytest.mark.parametrize("method", ["maxlog", "exact"])
    @pytest.mark.parametrize(
        ("constellation", "noise_var"),
        [
            (Constellation.psk(8), 0.02),
            (Constellation.qam(128), 0.02),
            (Constellation.qam(8), 0.02),
            (Constellation.pam(4), 0.02),
            # Most half-sums underflow relative to the symbol's peak metric:
            # the exact method takes the per-half fallback.
            (Constellation.qam(128), 2e-4),
        ],
        ids=["8psk", "cross128qam", "8qam", "pam4", "cross128qam-high-snr"],
    )
    def test_kernel_matches_cpu(
        self, xp: Any, xpt: Any, method: str, constellation: Any, noise_var: float
    ) -> None:
        x = _noisy(constellation, (2, 5000), 0.1, seed=3)
        kw = dict(noise_var=noise_var, constellation=constellation, method=method)
        cpu = mapping.compute_llr(x, **kw)
        gpu = mapping.compute_llr(xp.asarray(x), **kw)
        assert gpu.shape == cpu.shape
        xpt.assert_allclose(gpu, xp.asarray(cpu), rtol=1e-4, atol=1e-3)

    @pytest.mark.parametrize("method", ["maxlog", "exact"])
    @pytest.mark.parametrize(
        "constellation",
        [Constellation.qam(16), Constellation.qam(64).shaped(nu=0.05)],
        ids=["16qam", "shaped64qam"],
    )
    def test_point_kernel_on_separable_qam(
        self, xp: Any, xpt: Any, method: str, constellation: Any
    ) -> None:
        """The M-point kernel, which separable QAM bypasses, gives its LLRs too."""
        from commkit.mapping.llr import _llr

        x = _noisy(constellation, (777,), 0.1, seed=5)
        c = constellation
        args = (c.points, c.bit_labels, c.pmf, 0.02, method)
        gpu = _llr(xp.asarray(x), *args)
        xpt.assert_allclose(gpu, xp.asarray(_llr(x, *args)), rtol=1e-4, atol=1e-3)

    @pytest.mark.parametrize("method", ["maxlog", "exact"])
    @pytest.mark.parametrize(
        ("constellation", "noise_var"),
        [
            (Constellation.qam(4), 0.02),
            (Constellation.qam(16), 0.02),
            (Constellation.qam(256), 0.02),
            (Constellation.qam(1024), 0.002),
            (Constellation.qam(64).shaped(nu=0.05), 0.02),
            (Constellation.qam(256), 2e-4),
        ],
        ids=["4qam", "16qam", "256qam", "1024qam", "shaped64qam", "256qam-high-snr"],
    )
    def test_separable_kernel_matches_cpu(
        self, xp: Any, xpt: Any, method: str, constellation: Any, noise_var: float
    ) -> None:
        x = _noisy(constellation, (2, 5000), 0.1, seed=3)
        kw = dict(noise_var=noise_var, constellation=constellation, method=method)
        cpu = mapping.compute_llr(x, **kw)
        gpu = mapping.compute_llr(xp.asarray(x), **kw)
        assert gpu.shape == cpu.shape
        xpt.assert_allclose(gpu, xp.asarray(cpu), rtol=1e-4, atol=1e-3)

    def test_more_than_16_bits_raises(self, xp: Any) -> None:
        """The kernels' limit; there is no slower GPU path behind it."""
        c = Constellation.pam(2**17)
        with pytest.raises(ValueError, match="at most 16 bits"):
            mapping.compute_llr(
                xp.zeros(4, dtype=xp.float32), noise_var=0.1, constellation=c
            )

    def test_tables_are_cached_per_constellation(self, xp: Any) -> None:
        from commkit.mapping import llr as llr_module

        c = Constellation.qam(64).shaped(nu=0.02)
        x = xp.asarray(_noisy(c, (100,), 0.1, seed=1))
        mapping.compute_llr(x, noise_var=0.05, constellation=c)
        tables = dict(llr_module._DEVICE_TABLES[c])
        mapping.compute_llr(x, noise_var=0.01, constellation=c, method="exact")
        assert llr_module._DEVICE_TABLES[c].keys() == tables.keys()
        for key, value in tables.items():
            assert llr_module._DEVICE_TABLES[c][key] is value


class TestComputeLLRSignalIntegration:
    """Tests for compute_llr when supplied with a Signal object."""

    def test_compute_llr_signal_input_uses_metadata(self, xpt: Any) -> None:
        """Signal input resolves symbols and modulation parameters from metadata."""
        bits = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0], dtype="int32")
        symbols = mapping.map_bits(bits, constellation=Constellation.qam(16))
        sig = Signal(
            samples=symbols,
            sampling_rate=1.0,
            symbol_rate=1.0,
            constellation=mapping.Constellation.qam(16),
        )
        llrs_sig = mapping.compute_llr(sig, noise_var=1e-6)
        llrs_arr = mapping.compute_llr(
            symbols, noise_var=1e-6, constellation=Constellation.qam(16)
        )
        xpt.assert_allclose(llrs_sig, llrs_arr)

    def test_compute_llr_oversampled_signal_raises(self) -> None:
        """Signal input must be at one sample per symbol."""
        bits = np.array([0, 1, 0, 1], dtype="int32")
        symbols = mapping.map_bits(bits, constellation=Constellation.qam(4))
        sig = Signal(
            samples=symbols,
            sampling_rate=2.0,
            symbol_rate=1.0,
            constellation=Constellation.qam(4),
        )

        with pytest.raises(ValueError, match="one sample per symbol"):
            mapping.compute_llr(sig, noise_var=0.1)


class TestLLRBruteForce:
    """compute_llr against a float64 brute force from points, labels and prior."""

    @pytest.mark.parametrize("method", ["maxlog", "exact"])
    @pytest.mark.parametrize(
        ("constellation", "nv"),
        [
            (Constellation.qam(64).shaped(nu=0.05), 0.045),
            (Constellation.pam(4), 0.045),
            (Constellation.psk(8), 0.045),
            (Constellation.qam(256), 0.045),
            (Constellation.qam(128), 0.045),
            # High SNR: most half-sums underflow relative to the symbol's
            # peak metric, so exact LLRs take the per-half fallback.
            (Constellation.qam(256), 2e-4),
            (Constellation.qam(128), 2e-4),
        ],
        ids=[
            "shaped64qam",
            "pam4",
            "8psk",
            "256qam",
            "cross128qam",
            "256qam-high-snr",
            "cross128qam-high-snr",
        ],
    )
    def test_matches_brute_force(
        self, xp: Any, method: str, constellation: Any, nv: float
    ) -> None:
        rng = np.random.default_rng(9)
        pts = constellation.points
        x = pts[rng.integers(0, pts.size, 3000)]
        sd = np.sqrt(nv / 2)
        noise = sd * rng.standard_normal(x.shape)
        if np.iscomplexobj(pts):
            x = x + 1j * sd * rng.standard_normal(x.shape) + noise
        else:
            x = x + noise
        x = x.astype(np.complex64 if np.iscomplexobj(x) else np.float32)
        got = to_numpy(
            mapping.compute_llr(
                xp.asarray(x), noise_var=nv, constellation=constellation, method=method
            )
        ).reshape(x.size, -1)

        prior = np.log(constellation.pmf) if constellation.pmf is not None else 0.0
        metric = -(np.abs(x[:, None] - pts[None, :]) ** 2) / nv + prior  # (N, M)
        labels = constellation.bit_labels
        expected = np.empty_like(got, dtype=np.float64)
        for b in range(labels.shape[1]):
            m0, m1 = metric[:, labels[:, b] == 0], metric[:, labels[:, b] == 1]
            if method == "maxlog":
                expected[:, b] = m0.max(1) - m1.max(1)
            else:
                expected[:, b] = np.logaddexp.reduce(m0, 1) - np.logaddexp.reduce(m1, 1)
        np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-3)


class TestSeparableLLR:
    """Square QAM with per-axis labels and prior is demapped per axis."""

    @pytest.mark.parametrize(
        "constellation",
        [
            Constellation.qam(16),
            Constellation.qam(256),
            Constellation.qam(64).shaped(nu=0.1),
        ],
        ids=["16qam", "256qam", "shaped64qam"],
    )
    def test_separable_detected(self, constellation: Any) -> None:
        axes = constellation._axes
        assert axes is not None
        bits = np.sort(np.concatenate([a.bits for a in axes]))
        np.testing.assert_array_equal(bits, np.arange(constellation.bits_per_symbol))

    @pytest.mark.parametrize(
        "constellation",
        [
            Constellation.qam(8),
            Constellation.qam(32),
            Constellation.psk(8),
            Constellation.pam(4),
        ],
        ids=["8qam", "cross32qam", "8psk", "pam4"],
    )
    def test_not_a_square_lattice(self, constellation: Any) -> None:
        assert constellation._axes is None

    def test_labels_mixing_the_axes(self) -> None:
        """Swapping two points' labels makes some bit depend on both axes."""
        c = Constellation.qam(16)
        labels = c.bit_labels.copy()
        labels[[0, 5]] = labels[[5, 0]]
        assert Constellation(c.points, bit_labels=labels)._axes is None

    def test_prior_not_a_product(self) -> None:
        c = Constellation.qam(16)
        pmf = np.random.default_rng(0).uniform(0.5, 1.5, 16)
        pmf /= pmf.sum()
        assert Constellation(c.points, bit_labels=c.bit_labels, pmf=pmf)._axes is None


def _axis_labelled(order: int, labeling: str, seed: int = 0) -> Constellation:
    """Square QAM with per-axis labels: Gray (the factory's), natural binary,
    or a random permutation of each axis's labels."""
    if labeling == "gray":
        return Constellation.qam(order)
    side = int(round(order**0.5))
    kb = side.bit_length() - 1
    perm = (
        np.arange(side)
        if labeling == "natural"
        else np.random.default_rng(seed).permutation(side)
    )
    lev = np.arange(-(side - 1), side, 2.0)
    bits = (perm[:, None] >> np.arange(kb - 1, -1, -1)) & 1  # (side, kb)
    points = (lev[:, None] + 1j * lev[None, :]).ravel()
    labels = np.concatenate(
        [np.repeat(bits, side, axis=0), np.tile(bits, (side, 1))], axis=1
    )
    return Constellation(
        points / np.sqrt(np.mean(np.abs(points) ** 2)), bit_labels=labels
    )


class TestClosedFormMaxLog:
    """Max-log LLRs of uniform separable QAM in closed form: equal to the level
    metrics they replace, on every labeling of the axes."""

    @pytest.mark.parametrize("labeling", ["gray", "natural", "random"])
    def test_opposite_levels_definition(self, labeling: str) -> None:
        from commkit.mapping.llr import _opposite_levels

        labels = _axis_labelled(256, labeling, seed=2)._axes[0].labels
        lo, hi = _opposite_levels(labels)
        L, k = labels.shape
        for b in range(k):
            for m in range(L):
                other = np.flatnonzero(labels[:, b] != labels[m, b])
                below, above = other[other < m], other[other > m]
                assert lo[b, m] == (below.max() if below.size else -1)
                assert hi[b, m] == (above.min() if above.size else -1)

    @pytest.mark.parametrize("labeling", ["gray", "natural", "random"])
    @pytest.mark.parametrize("order", [4, 16, 64, 256, 1024])
    def test_matches_level_metrics(
        self, xp: Any, xpt: Any, monkeypatch: Any, order: int, labeling: str
    ) -> None:
        """Bit for bit on the CPU (the closed form forced at every order);
        to float rounding on the GPU.  Symbols beyond the outer levels too."""
        from commkit.mapping import llr as llr_module

        c = _axis_labelled(order, labeling, seed=order)
        x = xp.asarray(1.3 * _noisy(c, (2, 3000), 0.05, seed=order))
        noise_var = 0.5 / order
        monkeypatch.setattr(llr_module, "_CLOSED_FORM_MIN_LEVELS_CPU", 2)
        closed = mapping.compute_llr(x, noise_var=noise_var, constellation=c)
        monkeypatch.setattr(llr_module, "_closed_form", lambda *a: False)
        level = mapping.compute_llr(x, noise_var=noise_var, constellation=c)
        if xp is np:
            np.testing.assert_array_equal(closed, level)
        else:
            xpt.assert_allclose(closed, level, rtol=1e-6, atol=1e-6)

    @pytest.mark.parametrize(
        ("constellation", "method", "cpu_closed"),
        [
            (Constellation.qam(256), "maxlog", True),
            (Constellation.qam(64), "maxlog", False),  # CPU: below 16 levels per axis
            (Constellation.qam(256), "exact", False),
            (Constellation.qam(256).shaped(nu=0.02), "maxlog", False),
        ],
        ids=["256qam", "64qam", "256qam-exact", "shaped256qam"],
    )
    def test_routing(
        self,
        xp: Any,
        monkeypatch: Any,
        constellation: Any,
        method: str,
        cpu_closed: bool,
    ) -> None:
        """Only max-log without a prior takes the closed form (on the CPU from
        16 levels per axis, on the GPU always); the rest the level metrics."""
        from commkit import _cuda
        from commkit.mapping import llr as llr_module

        used: list[str] = []
        if xp is np:
            real = llr_module._get_numba_llr
            monkeypatch.setattr(
                llr_module,
                "_get_numba_llr",
                lambda name="llr": used.append(name) or real(name),
            )
            expect_closed = cpu_closed
        else:
            real = _cuda.get_kernel
            monkeypatch.setattr(
                _cuda,
                "get_kernel",
                lambda name, **kw: used.append(name) or real(name, **kw),
            )
            expect_closed = method == "maxlog" and constellation.pmf is None
        x = xp.asarray(_noisy(constellation, (500,), 0.05, seed=1))
        mapping.compute_llr(
            x, noise_var=0.01, constellation=constellation, method=method
        )
        closed = "llr_separable_maxlog"
        assert used == [closed if expect_closed else "llr_separable"]

    @pytest.mark.gpu_only
    def test_kernel_rejects_bad_tables(self, xp: Any) -> None:
        from commkit import _cuda

        launch = _cuda.get_kernel("llr_separable_maxlog")
        lev = xp.asarray([-1.0, 1.0], dtype=xp.float32)
        lab = xp.asarray([0, 1], dtype=xp.int32)
        nb = xp.asarray([[-1, 0]], dtype=xp.int32)
        pos = xp.asarray([0], dtype=xp.int32)
        axis = (lev, lab, nb, nb, pos)
        x = xp.zeros(4, dtype=xp.complex64)
        with pytest.raises(ValueError, match="at least 2 levels"):
            launch(x, (lev[:1], lab[:1], nb[:, :1], nb[:, :1], pos), axis, inv_s2=1.0)
        with pytest.raises(TypeError, match="re levels"):
            launch(x, (lev.astype(xp.float64), *axis[1:]), axis, inv_s2=1.0)
        assert launch(
            x, axis, (lev, lab, nb, nb, xp.asarray([1], xp.int32)), inv_s2=1.0
        ).shape == (4, 2)
