"""Tests for the time-domain block-update equalizers.

Covers ``update_mode='block'`` on ``lms``/``cma``/``rde`` across the
``backend='xp'`` (array-native NumPy/CuPy) and ``backend='jax'`` (chunked
``lax.scan``) paths, including pilot-aided masking, the validation contract,
remainder handling, and floor parity vs. the per-symbol reference.
"""

import numpy as np
import pytest

from commkit import equalization, generate_psk, generate_qam
from commkit.equalization import build_pilot_ref
from commkit.mapping import gray_constellation


@pytest.fixture(autouse=True)
def _enable_jax_x64():
    try:
        import jax

        jax.config.update("jax_enable_x64", True)
    except ImportError:
        pass


def _to_np(arr):
    return arr.get() if hasattr(arr, "get") else np.asarray(arr)


def _isi_signal(xp, mod, order, n_symbols, seed, channel, noise=0.02):
    """Build a pulse-shaped, ISI-distorted, noisy signal on the ``xp`` device."""
    factory = generate_qam if mod == "qam" else generate_psk
    sig = factory(
        symbol_rate=1e6,
        num_symbols=n_symbols,
        order=order,
        pulse_shape="rrc",
        sps=2,
        seed=seed,
    )
    tx = xp.asarray(sig.source_symbols)
    rx = xp.convolve(xp.asarray(sig.samples), xp.asarray(channel), mode="same")
    rng = xp.random.RandomState(seed)
    rx = rx + noise * (rng.randn(len(rx)) + 1j * rng.randn(len(rx))).astype(
        xp.complex64
    )
    return tx, rx.astype(xp.complex64)


def _tail_mse_db(error, n=400):
    e = _to_np(error)[-n:]
    return 10.0 * np.log10(float(np.mean(np.abs(e) ** 2)) + 1e-30)


def _dispersion(y, order, mod="qam"):
    """Phase-blind radial dispersion: mean squared distance of |y|^2 to the
    nearest constellation ring power.  A fair convergence metric for blind
    equalizers (CMA/RDE) that carry a residual phase ambiguity."""
    const = gray_constellation(mod, order)
    const = const / np.sqrt(np.mean(np.abs(const) ** 2))
    r2 = np.abs(const) ** 2
    a2 = np.abs(_to_np(y)) ** 2
    return float(np.mean(np.min((a2[:, None] - r2[None, :]) ** 2, axis=1)))


_BACKENDS = ["xp", "jax"]


class TestBlockUpdateValidation:
    def test_numba_block_raises(self):
        rx = np.zeros(2048, np.complex64)
        with pytest.raises(ValueError, match="backend='jax'"):
            equalization.lms(
                rx, modulation="qam", order=16, backend="numba", update_mode="block"
            )

    def test_cpr_with_block_raises(self):
        rx = np.zeros(2048, np.complex64)
        with pytest.raises(ValueError, match="cpr_type"):
            equalization.lms(
                rx,
                modulation="qam",
                order=16,
                backend="xp",
                update_mode="block",
                cpr_type="bps",
            )

    def test_store_weights_with_block_raises(self):
        rx = np.zeros(2048, np.complex64)
        with pytest.raises(ValueError, match="store_weights"):
            equalization.cma(
                rx,
                modulation="psk",
                order=4,
                backend="xp",
                update_mode="block",
                store_weights=True,
            )

    def test_bad_update_mode_raises(self):
        rx = np.zeros(2048, np.complex64)
        with pytest.raises(ValueError, match="update_mode"):
            equalization.lms(rx, modulation="qam", order=16, update_mode="delayed")

    def test_numpy_input_jax_gpu_returns_numpy(self):
        """NumPy input + JAX ``device='gpu'`` must return NumPy (output on the
        input's device), not raise on the from_jax -> CuPy mismatch."""
        pytest.importorskip("jax")
        from commkit.backend import is_cupy_available

        if not is_cupy_available():
            pytest.skip("requires a CUDA device for device='gpu'")
        rng = np.random.RandomState(0)
        const = gray_constellation("qam", 16)
        tx = const[rng.randint(0, 16, 2000)].astype(np.complex64)
        tx /= np.sqrt(np.mean(np.abs(tx) ** 2))
        up = np.zeros(4000, np.complex64)
        up[::2] = tx
        rx = np.convolve(up, np.array([0.1, 0.9, 0.2], np.complex64))[:4000].astype(
            np.complex64
        )
        out = equalization.lms(
            rx,
            training_symbols=tx[:1000],
            modulation="qam",
            order=16,
            num_taps=15,
            step_size=2e-3,
            backend="jax",
            device="gpu",
            update_mode="block",
            block_len=16,
        )
        assert isinstance(out.y_hat, np.ndarray)


class TestBlockUpdateLMS:
    def test_converges_and_matches_sequential_floor(self, backend_device, xp):
        channel = np.array([0.1, 0.9, 0.9, 0.1], np.complex64)
        tx, rx = _isi_signal(xp, "qam", 16, 6000, 7, channel)
        seq = equalization.lms(
            _to_np(rx),
            training_symbols=_to_np(tx)[:2000],
            modulation="qam",
            order=16,
            num_taps=15,
            step_size=2e-3,
            backend="numba",
        )
        for be in _BACKENDS:
            blk = equalization.lms(
                rx,
                training_symbols=tx[:2000],
                modulation="qam",
                order=16,
                num_taps=15,
                step_size=2e-3,
                backend=be,
                update_mode="block",
                block_len=16,
            )
            seq_db = _tail_mse_db(seq.error)
            blk_db = _tail_mse_db(blk.error)
            assert blk_db < -20.0, f"{be} block LMS did not converge: {blk_db:.1f} dB"
            # Same mu => same floor (block update is the summed gradient).
            assert blk_db - seq_db < 0.5, (
                f"{be} block floor {blk_db:.2f} dB worse than sequential "
                f"{seq_db:.2f} dB by > 0.5 dB"
            )

    def test_jax_xp_agree(self, backend_device, xp, xpt):
        channel = np.array([0.1, 0.9, 0.9, 0.1], np.complex64)
        tx, rx = _isi_signal(xp, "qam", 16, 4000, 11, channel)
        kw = dict(
            training_symbols=tx[:1500],
            modulation="qam",
            order=16,
            num_taps=15,
            step_size=2e-3,
            update_mode="block",
            block_len=16,
        )
        y_xp = equalization.lms(rx, backend="xp", **kw).y_hat
        y_jax = equalization.lms(rx, backend="jax", **kw).y_hat
        # XLA vs CuPy/NumPy reductions differ only at float32 rounding.
        xpt.assert_allclose(y_xp, y_jax, atol=1e-4, rtol=1e-4)

    def test_mu0_matches_sequential_forward(self, backend_device, xp, xpt):
        """With mu=0 the frozen-identity forward path is bit-exact vs sequential."""
        channel = np.array([0.2, 1.0, 0.3], np.complex64)
        tx, rx = _isi_signal(xp, "qam", 16, 2000, 5, channel, noise=0.01)
        seq = equalization.lms(
            _to_np(rx),
            training_symbols=_to_np(tx)[:10],
            modulation="qam",
            order=16,
            num_taps=15,
            step_size=0.0,
            backend="numba",
        )
        blk = equalization.lms(
            rx,
            training_symbols=tx[:10],
            modulation="qam",
            order=16,
            num_taps=15,
            step_size=0.0,
            backend="xp",
            update_mode="block",
            block_len=16,
        )
        xpt.assert_allclose(blk.y_hat, xp.asarray(seq.y_hat), atol=1e-5, rtol=1e-5)

    def test_remainder_length_and_tail(self, backend_device, xp, xpt):
        """N not divisible by block_len yields full-length output; the tail
        matches a divisible-length run over the shared prefix."""
        channel = np.array([0.2, 1.0, 0.3], np.complex64)
        n_odd = 4007  # n_sym not a multiple of 16
        tx, rx = _isi_signal(xp, "qam", 16, n_odd, 9, channel)
        r_odd = equalization.lms(
            rx,
            training_symbols=tx[:1500],
            modulation="qam",
            order=16,
            num_taps=15,
            step_size=2e-3,
            backend="xp",
            update_mode="block",
            block_len=16,
        )
        assert r_odd.y_hat.shape[-1] == n_odd
        # Truncate the input to a divisible symbol count; the equalized symbols
        # over the common prefix must match (same frozen-per-chunk trajectory).
        n_even = 4000
        r_even = equalization.lms(
            rx[: 2 * n_even],
            training_symbols=tx[:1500],
            modulation="qam",
            order=16,
            num_taps=15,
            step_size=2e-3,
            backend="xp",
            update_mode="block",
            block_len=16,
        )
        # The interior (away from the right edge) is unaffected by total length.
        xpt.assert_allclose(
            r_odd.y_hat[: n_even - 32],
            r_even.y_hat[: n_even - 32],
            atol=1e-4,
            rtol=1e-4,
        )


class TestBlockUpdateCMA:
    def test_blind_converges_and_matches_sequential(self, backend_device, xp):
        channel = np.array([0.08, 1.0, -0.3, 0.1], np.complex64)
        _, rx = _isi_signal(xp, "psk", 4, 8000, 2, channel)
        seq = equalization.cma(
            _to_np(rx),
            modulation="psk",
            order=4,
            num_taps=15,
            step_size=1e-3,
            backend="numba",
        )
        seq_disp = _dispersion(seq.y_hat[4000:], 4, "psk")
        for be, D in [("xp", 8), ("jax", 16), ("xp", 32)]:
            blk = equalization.cma(
                rx,
                modulation="psk",
                order=4,
                num_taps=15,
                step_size=1e-3,
                backend=be,
                update_mode="block",
                block_len=D,
            )
            blk_disp = _dispersion(blk.y_hat[4000:], 4, "psk")
            assert blk_disp < 0.05, f"{be} D={D} CMA blind diverged: {blk_disp:.4f}"
            assert blk_disp < seq_disp * 3 + 0.01

    def test_pilot_aided_resolves_phase(self, backend_device, xp):
        channel = np.array([0.08, 1.0, -0.3, 0.1], np.complex64)
        n = 8000
        tx, rx = _isi_signal(xp, "psk", 4, n, 2, channel)
        pmask = np.zeros(n, bool)
        pmask[::8] = True
        pref, pu8 = build_pilot_ref(_to_np(tx)[pmask], pmask, n, 1)
        seq = equalization.cma(
            _to_np(rx),
            modulation="psk",
            order=4,
            num_taps=15,
            step_size=1e-3,
            backend="numba",
            pilot_ref=pref,
            pilot_mask=pu8,
        )
        for be in _BACKENDS:
            blk = equalization.cma(
                rx,
                modulation="psk",
                order=4,
                num_taps=15,
                step_size=1e-3,
                backend=be,
                update_mode="block",
                block_len=16,
                pilot_ref=pref,
                pilot_mask=pu8,
            )
            seq_db = _tail_mse_db(seq.error)
            blk_db = _tail_mse_db(blk.error)
            # Pilot error locks phase: block floor tracks the sequential floor.
            assert blk_db - seq_db < 0.6, (
                f"{be} PA-CMA block floor {blk_db:.2f} vs seq {seq_db:.2f} dB"
            )


class TestBlockUpdateRDE:
    def test_blind_converges_and_matches_sequential(self, backend_device, xp):
        channel = np.array([0.08, 1.0, -0.3, 0.1], np.complex64)
        _, rx = _isi_signal(xp, "qam", 16, 12000, 2, channel)
        seq = equalization.rde(
            _to_np(rx),
            modulation="qam",
            order=16,
            num_taps=15,
            step_size=4e-4,
            backend="numba",
        )
        seq_disp = _dispersion(seq.y_hat[6000:], 16, "qam")
        for be in _BACKENDS:
            blk = equalization.rde(
                rx,
                modulation="qam",
                order=16,
                num_taps=15,
                step_size=4e-4,
                backend=be,
                update_mode="block",
                block_len=16,
            )
            blk_disp = _dispersion(blk.y_hat[6000:], 16, "qam")
            assert blk_disp < seq_disp + 0.02, (
                f"{be} RDE block disp {blk_disp:.4f} vs seq {seq_disp:.4f}"
            )


class TestBlockUpdateMIMO:
    def test_butterfly_block_runs(self, backend_device, xp):
        """2x2 butterfly block LMS runs and converges on both channels."""
        channel = np.array([0.1, 0.9, 0.2], np.complex64)
        tx, rx = _isi_signal(xp, "qam", 16, 4000, 3, channel)
        rx2 = xp.stack([rx, xp.roll(rx, 1)])
        tx2 = xp.stack([tx, tx])
        blk = equalization.lms(
            rx2,
            training_symbols=tx2[:, :1500],
            modulation="qam",
            order=16,
            num_taps=15,
            step_size=2e-3,
            backend="xp",
            update_mode="block",
            block_len=16,
        )
        assert blk.y_hat.shape == (2, 4000)
        for ch in range(2):
            assert _tail_mse_db(blk.error[ch]) < -15.0
