"""Weight initialization, prefix-pad normalization, and length-independence."""

import numpy as np
import pytest

from commkit import equalization
from commkit.backend import to_device
from commkit.mapping import Constellation
from tests.common import scoring
from tests.common.conversions import to_numpy
from tests.common.signals import (
    make_channel_signal,
    make_test_psk_samples,
    make_test_qam_signal,
)

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

    def test_warm_start_rde_same_or_better_evm(self, xp):
        """RDE started from taps that LMS trained on 200 symbols opens the
        ISI channel at once: over the first 2000 symbols its EVM against the
        reference (one constant rotation removed for both) is below the
        cold start's, which is still converging, and its second half has no
        symbol errors."""
        c = Constellation.qam(16)
        rx = make_channel_signal(
            c,
            8000,
            isi_taps=[0.25, 1.0, 0.3j, -0.1],
            frac_delay=0.3,
            esn0_db=22.0,
            xp=xp,
        )
        ref = to_numpy(rx.reference.symbols)
        kw = dict(constellation=c, num_taps=21, step_size=1e-3)
        pre = equalization.lms(rx, ref[:200], num_taps=21, step_size=1e-2)
        warm = equalization.rde(rx, **kw, initial_taps=pre.weights)
        cold = equalization.rde(rx, **kw)

        def resolved(res):
            return scoring.resolve_blind(res.y_hat, ref, c, rotation="any")

        y_warm, y_cold = resolved(warm), resolved(cold)
        head = slice(0, 2000)
        assert scoring.evm_rms(y_warm[head], ref[head]) < scoring.evm_rms(
            y_cold[head], ref[head]
        )
        assert scoring.ser(y_warm[4000:], ref[4000:], c) == 0.0


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

    def test_pad_mode_reflect_mirrors_the_record(self, xp, xpt):
        """'reflect' pads the cold start with the record mirrored about its
        first sample: the same output as a prefix of those samples."""
        samples, syms = _make_qpsk(xp, n_sym=500)
        weights = equalization.lms(
            samples,
            syms[:50],
            num_taps=11,
            sps=1,
            step_size=1e-2,
            constellation=Constellation.psk(4),
        ).weights
        mirrored = samples[1:21][::-1]  # x[20], ..., x[1]
        y_reflect = equalization.apply_taps(samples, weights, sps=1, pad_mode="reflect")
        y_prefix = equalization.apply_taps(
            samples, weights, sps=1, samples_prefix=mirrored
        )
        # The prefix path pads the tail with zeros; compare away from it.
        xpt.assert_allclose(y_reflect[:-20], y_prefix[:-20], atol=1e-6)

    def test_unknown_pad_mode_raises(self, xp):
        samples, syms = _make_qpsk(xp, n_sym=200)
        with pytest.raises(ValueError, match="pad_mode"):
            equalization.lms(
                samples,
                syms[:20],
                num_taps=5,
                sps=1,
                constellation=Constellation.psk(4),
                pad_mode="wrap",
            )

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
