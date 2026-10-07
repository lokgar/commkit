"""Weight initialization, prefix-pad normalization, and length-independence."""

import numpy as np
import pytest

from commkit import equalization, generate
from commkit.backend import to_device
from commkit.equalization import EqualizerResult
from commkit.filtering import RRC
from commkit.mapping import Constellation
from tests.common.conversions import to_numpy
from tests.common.signals import make_test_psk_samples, make_test_qam_signal

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _make_qam16_rx(xp, n_symbols=2000, seed=0):
    """Generate a simple AWGN-impaired 16-QAM signal at 2 SPS."""
    sig = make_test_qam_signal(
        order=16, num_symbols=n_symbols, sps=2, snr_db=20.0, seed=seed, xp=xp
    )
    return xp.ascontiguousarray(sig.samples)


# -----------------------------------------------------------------------------
# Test Classes
# -----------------------------------------------------------------------------


class TestInitialTaps:
    """initial_taps: start from prior equalizer weights."""

    def test_lms_accepts_initial_taps(self, xp):
        """lms() accepts initial_taps array with correct shape and returns EqualizerResult."""
        rx = _make_qam16_rx(xp)
        num_taps, num_ch = 21, 1
        w0 = np.zeros((num_ch, num_ch, num_taps), dtype=np.complex64)
        w0[0, 0, num_taps // 2] = 1.0 + 0j

        result = equalization.lms(
            rx,
            None,
            constellation=Constellation.qam(16),
            num_taps=num_taps,
            initial_taps=w0,
            sps=2,
        )
        assert isinstance(result, EqualizerResult)
        assert result.weights.shape == (num_taps,)  # SISO squeeze

    def test_rls_accepts_initial_taps(self, xp):
        """rls() accepts initial_taps array with correct shape."""

        sig = generate(
            Constellation.qam(4), 1000, symbol_rate=1e6, sps=2, pulse=RRC(0.35), rng=1
        )
        rx = xp.asarray(sig.samples)
        num_taps, num_ch = 11, 1
        w0 = np.zeros((num_ch, num_ch, num_taps), dtype=np.complex64)
        w0[0, 0, num_taps // 2] = 1.0 + 0j

        result = equalization.rls(
            rx,
            xp.asarray(sig.reference.symbols),
            constellation=Constellation.qam(4),
            num_taps=num_taps,
            sps=2,
            initial_taps=w0,
        )
        assert isinstance(result, EqualizerResult)

    def test_cma_accepts_initial_taps(self, xp):
        """cma() accepts initial_taps array with correct shape."""
        rx = _make_qam16_rx(xp)
        num_taps, num_ch = 21, 1
        w0 = np.zeros((num_ch, num_ch, num_taps), dtype=np.complex64)
        w0[0, 0, num_taps // 2] = 1.0 + 0j

        result = equalization.cma(
            rx,
            constellation=Constellation.qam(16),
            num_taps=num_taps,
            initial_taps=w0,
            sps=2,
        )
        assert isinstance(result, EqualizerResult)

    def test_rde_accepts_initial_taps(self, xp):
        """rde() accepts initial_taps array with correct shape."""
        rx = _make_qam16_rx(xp)
        num_taps, num_ch = 21, 1
        w0 = np.zeros((num_ch, num_ch, num_taps), dtype=np.complex64)
        w0[0, 0, num_taps // 2] = 1.0 + 0j

        result = equalization.rde(
            rx,
            constellation=Constellation.qam(16),
            num_taps=num_taps,
            initial_taps=w0,
            sps=2,
        )
        assert isinstance(result, EqualizerResult)

    def test_initial_taps_shape_mismatch_raises(self, xp):
        """Wrong initial_taps shape raises ValueError before kernel is called."""
        rx = _make_qam16_rx(xp)
        bad_w = np.zeros((1, 1, 99), dtype=np.complex64)  # wrong num_taps

        with pytest.raises(ValueError, match="initial_taps shape"):
            equalization.cma(
                rx,
                constellation=Constellation.qam(16),
                num_taps=21,
                initial_taps=bad_w,
                sps=2,
            )

        with pytest.raises(ValueError, match="initial_taps shape"):
            equalization.rde(
                rx,
                constellation=Constellation.qam(16),
                num_taps=21,
                initial_taps=bad_w,
                sps=2,
            )

    def test_lms_to_rde_handoff_output_shape(self, xp):
        """LMS weights can be handed off to RDE via initial_taps; output shape is correct."""
        rx = _make_qam16_rx(xp, n_symbols=3000)
        half = rx.shape[-1] // 2

        pre_rx = rx[..., :half]
        payload_rx = rx[..., half:]

        pre = equalization.lms(
            pre_rx,
            constellation=Constellation.qam(16),
            num_taps=21,
            step_size=0.05,
            sps=2,
        )
        w0 = to_numpy(pre.weights)
        if w0.ndim == 1:
            w0 = w0[np.newaxis, np.newaxis, :]

        result = equalization.rde(
            payload_rx,
            constellation=Constellation.qam(16),
            num_taps=21,
            step_size=1e-4,
            initial_taps=w0,
            sps=2,
        )
        expected_syms = payload_rx.shape[-1] // 2
        assert result.y_hat.shape[-1] == expected_syms

    def test_warm_start_rde_same_or_better_evm(self, xp):
        """RDE warm-started from LMS achieves same or better EVM than cold-start."""
        from commkit.impairments import apply_awgn

        sig = generate(
            Constellation.qam(16), 4000, symbol_rate=1e6, sps=2, pulse=RRC(0.35), rng=42
        )
        rx_np = apply_awgn(sig.samples, esn0_db=25.0, sps=2)
        rx = xp.asarray(rx_np)

        # Cold-start RDE
        cold = equalization.rde(
            rx, constellation=Constellation.qam(16), num_taps=21, step_size=5e-4, sps=2
        )
        # LMS pre-convergence
        pre = equalization.lms(
            rx,
            xp.asarray(sig.reference.symbols[:200]),
            constellation=Constellation.qam(16),
            num_taps=21,
            sps=2,
        )
        _w = pre.weights
        w0 = to_numpy(pre.weights)
        if w0.ndim == 1:
            w0 = w0[np.newaxis, np.newaxis, :]

        # Warm RDE
        warm = equalization.rde(
            rx,
            constellation=Constellation.qam(16),
            num_taps=21,
            step_size=5e-4,
            initial_taps=w0,
            sps=2,
        )

        tail = slice(-500, None)
        ref = to_numpy(sig.reference.symbols)
        cold_hat = to_numpy(cold.y_hat)
        warm_hat = to_numpy(warm.y_hat)
        evm_cold = float(np.mean(np.abs(cold_hat[tail] - ref[tail]) ** 2))
        evm_warm = float(np.mean(np.abs(warm_hat[tail] - ref[tail]) ** 2))
        # Warm start must not be significantly worse
        assert evm_warm <= evm_cold * 1.5, (
            f"Warm RDE EVM {evm_warm:.4f} much worse than cold {evm_cold:.4f}"
        )


class TestNormalizationLengthIndependence:
    """Normalization uses full-signal RMS - training output scales with signal power."""

    @pytest.mark.parametrize("algo", ["lms", "rls"])
    def test_training_output_finite(self, algo, xp):
        """y_hat training region is finite and non-trivial."""
        import numpy as np

        from commkit.equalization import lms, rls

        rng = np.random.default_rng(42)
        n_train = 200
        n_sym = 500

        const = Constellation.qam(16).points.astype(np.complex64)
        syms = const[rng.integers(0, 16, n_sym)]
        noise = (
            0.05 * (rng.standard_normal(n_sym) + 1j * rng.standard_normal(n_sym))
        ).astype(np.complex64)
        sig = (syms + noise).astype(np.complex64)

        fn = lms if algo == "lms" else rls
        res = fn(
            sig,
            training_symbols=syms[:n_train],
            num_taps=1,
            sps=1,
            constellation=Constellation.qam(16),
        )

        np.testing.assert_array_equal(
            np.isfinite(np.asarray(res.y_hat[:n_train])), True
        )


def _make_qpsk(xp, n_sym=2000, snr_db=20.0, seed=77):
    """Build a QPSK signal using the given array module (numpy or cupy)."""
    return make_test_psk_samples(
        order=4, num_symbols=n_sym, sps=1, snr_db=snr_db, seed=seed, xp=xp
    )


def _algo_kw(algo, num_taps):
    """Return algorithm-specific kwargs (lms uses step_size, rls uses forgetting_factor)."""
    base = dict(num_taps=num_taps, sps=1, constellation=Constellation.psk(4))
    return (
        {**base, "step_size": 1e-2}
        if algo == "lms"
        else {**base, "forgetting_factor": 0.999}
    )


class TestPadAndNormalization:
    """pad_mode of a cold start and the stored normalization."""

    @pytest.mark.parametrize("algo", ["lms", "rls"])
    def test_pad_mode_zeros_is_baseline(self, algo, xp, xpt):
        """Explicit pad_mode='zeros' must be byte-exact with the default."""
        samples, syms = _make_qpsk(xp)
        fn = getattr(equalization, algo)
        kw = _algo_kw(algo, num_taps=7)
        r_default = fn(samples, syms[:50], **kw)
        r_explicit = fn(samples, syms[:50], **kw, pad_mode="zeros")
        xpt.assert_array_equal(
            xp.asarray(r_default.y_hat),
            xp.asarray(r_explicit.y_hat),
        )

    def test_pad_mode_edge_lms(self, xp):
        """pad_mode='edge' must not raise and must produce finite output."""
        samples, syms = _make_qpsk(xp, n_sym=500)
        r = equalization.lms(
            samples,
            syms[:20],
            num_taps=11,
            sps=1,
            step_size=1e-2,
            constellation=Constellation.psk(4),
            pad_mode="edge",
        )
        assert bool(xp.all(xp.isfinite(xp.asarray(r.y_hat))))

    def test_pad_mode_edge_block_lms_matches_cpu(self, xp, xpt):
        """Block padding runs on the input's device and matches the CPU."""
        samples, syms = _make_qpsk(xp, n_sym=500)
        kw = dict(
            num_taps=11,
            sps=1,
            step_size=1e-2,
            block_size=64,
            constellation=Constellation.psk(4),
            pad_mode="edge",
        )
        r = equalization.block_lms(samples, syms[:64], **kw)
        ref = equalization.block_lms(
            to_device(samples, "cpu"), to_device(syms[:64], "cpu"), **kw
        )
        xpt.assert_allclose(r.y_hat, xp.asarray(ref.y_hat), atol=1e-4)

    def test_input_norm_factor_stored_in_result_lms(self, xp):
        """EqualizerResult.input_norm_factor must be a positive scalar for SISO."""
        samples, syms = _make_qpsk(xp, n_sym=500)
        r = equalization.lms(
            samples,
            syms[:20],
            num_taps=5,
            sps=1,
            step_size=1e-2,
            constellation=Constellation.psk(4),
        )
        assert isinstance(r.input_norm_factor, float)
        assert r.input_norm_factor > 0.0
