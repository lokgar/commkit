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
                xp.ones(1),
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

    def test_long_records_are_chunked_consistently(
        self, xp: Any, xpt: Any, monkeypatch: Any
    ) -> None:
        """Results do not depend on how the record is split into chunks.

        The chunked path is the GPU fallback when the CUDA kernel is
        unavailable; the kernel is switched off to exercise it.
        """
        from commkit.mapping import llr as llr_module

        monkeypatch.setattr(llr_module._cuda, "get_kernel", lambda *a, **k: None)

        rng = np.random.default_rng(4)
        const = mapping.Constellation.qam(64).points
        y = xp.asarray(const[rng.integers(0, 64, 3000)].astype(np.complex64))
        full = mapping.compute_llr(
            y, noise_var=0.05, constellation=Constellation.qam(64), method="exact"
        )
        original = llr_module._CHUNK_ELEMENTS
        try:
            llr_module._CHUNK_ELEMENTS = 6 * 64 * 7  # 7 symbols per chunk
            chunked = mapping.compute_llr(
                y, noise_var=0.05, constellation=Constellation.qam(64), method="exact"
            )
        finally:
            llr_module._CHUNK_ELEMENTS = original
        xpt.assert_array_equal(full, chunked)


@pytest.mark.gpu_only
class TestLLRKernel:
    """The CUDA LLR kernel against its CPU reference (Numba)."""

    @pytest.mark.requires_kernel("llr")
    @pytest.mark.parametrize("method", ["maxlog", "exact"])
    @pytest.mark.parametrize(
        "constellation",
        [
            Constellation.qam(16),
            Constellation.qam(256),
            Constellation.qam(64).shaped(nu=0.05),
            Constellation.pam(4),
        ],
        ids=["16qam", "256qam", "shaped64qam", "pam4"],
    )
    def test_kernel_matches_cpu(
        self, xp: Any, xpt: Any, method: str, constellation: Any
    ) -> None:
        rng = np.random.default_rng(3)
        pts = constellation.points
        x = pts[rng.integers(0, pts.size, (2, 5000))]
        x = x + 0.1 * rng.standard_normal(x.shape)
        if np.iscomplexobj(pts):
            x = x + 0.1j * rng.standard_normal(x.shape)
        kw = dict(noise_var=0.02, constellation=constellation, method=method)
        cpu = mapping.compute_llr(x, **kw)
        gpu = mapping.compute_llr(xp.asarray(x), **kw)
        assert gpu.shape == cpu.shape
        xpt.assert_allclose(gpu, xp.asarray(cpu), rtol=1e-4, atol=1e-3)


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
        "constellation",
        [
            Constellation.qam(64).shaped(nu=0.05),
            Constellation.pam(4),
            Constellation.psk(8),
        ],
        ids=["shaped64qam", "pam4", "8psk"],
    )
    def test_matches_brute_force(
        self, xp: Any, method: str, constellation: Any
    ) -> None:
        rng = np.random.default_rng(9)
        pts = constellation.points
        x = pts[rng.integers(0, pts.size, 3000)]
        noise = 0.15 * rng.standard_normal(x.shape)
        if np.iscomplexobj(pts):
            x = x + 0.15j * rng.standard_normal(x.shape) + noise
        else:
            x = x + noise
        nv = 0.045
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
