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

    def test_long_records_are_chunked_consistently(self, xp: Any, xpt: Any) -> None:
        """Results do not depend on how the record is split into chunks."""
        from commkit.mapping import llr as llr_module

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
        sig = sig.replace(resolved_symbols=symbols)

        llrs_sig = mapping.compute_llr(sig, noise_var=1e-6)
        llrs_arr = mapping.compute_llr(
            symbols, noise_var=1e-6, constellation=Constellation.qam(16)
        )
        xpt.assert_allclose(llrs_sig, llrs_arr)

    def test_compute_llr_signal_input_raises_without_resolved(self) -> None:
        """Signal input missing resolved_symbols raises ValueError."""
        bits = np.array([0, 1, 0, 1], dtype="int32")
        symbols = mapping.map_bits(bits, constellation=Constellation.qam(4))
        sig = Signal(
            samples=symbols,
            sampling_rate=1.0,
            symbol_rate=1.0,
            constellation=Constellation.qam(4),
        )

        with pytest.raises(ValueError, match="resolved symbols"):
            mapping.compute_llr(sig, noise_var=0.1)
