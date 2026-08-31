"""Tests for the blind frequency-domain equalizers block_cma / block_rde.

These are the blind, phase-directed siblings of ``block_lms``: the same
overlap-save FDAF engine (shared via ``_fdaf_forward`` / ``_fdaf_gradient_update``)
driven by the Godard / ring-radius error instead of a trained/DD slicer.
"""

import numpy as np
import pytest

from commkit import generate_psk, generate_qam
from commkit.equalization import (
    block_cma,
    block_lms,
    block_rde,
    build_pilot_ref,
)
from commkit.mapping import gray_constellation


def _to_np(arr):
    return arr.get() if hasattr(arr, "get") else np.asarray(arr)


def _isi_signal(xp, mod, order, n_symbols, seed, channel, noise=0.02):
    factory = generate_qam if mod == "qam" else generate_psk
    sig = factory(
        symbol_rate=1e6,
        num_symbols=n_symbols,
        order=order,
        pulse_shape="rrc",
        sps=2,
        seed=seed,
    )
    # Signal may default to GPU when CuPy is present; coerce to host then to xp
    # so an explicit xp=np reference works regardless of the global default.
    tx = xp.asarray(_to_np(sig.source_symbols))
    rx = xp.convolve(xp.asarray(_to_np(sig.samples)), xp.asarray(channel), mode="same")
    rng = xp.random.RandomState(seed)
    rx = rx + noise * (rng.randn(len(rx)) + 1j * rng.randn(len(rx))).astype(
        xp.complex64
    )
    return tx, rx.astype(xp.complex64)


def _dispersion(y, order, mod):
    const = gray_constellation(mod, order)
    const = const / np.sqrt(np.mean(np.abs(const) ** 2))
    r2 = np.abs(const) ** 2
    a2 = np.abs(_to_np(y)) ** 2
    return float(np.mean(np.min((a2[:, None] - r2[None, :]) ** 2, axis=1)))


class TestBlockFDAFEngine:
    def test_all_pilots_block_cma_matches_block_lms(self, backend_device, xp, xpt):
        """With a full pilot mask every position uses the LMS residual error, so
        block_cma must reproduce block_lms (training) bit-for-bit - the strongest
        check that the shared FDAF primitives match block_lms's engine."""
        channel = np.array([0.06, 1.0, -0.25, 0.08], np.complex64)
        tx, rx = _isi_signal(xp, "qam", 16, 8000, 6, channel)
        n = 8000
        pmask = np.ones(n, bool)
        pref, pu8 = build_pilot_ref(_to_np(tx), pmask, n, 1)
        r_cma = block_cma(
            rx,
            modulation="qam",
            order=16,
            num_taps=15,
            step_size=2e-4,
            block_size=256,
            pilot_ref=pref,
            pilot_mask=pu8,
        )
        r_lms = block_lms(
            rx,
            training_symbols=tx,
            modulation="qam",
            order=16,
            num_taps=15,
            step_size=2e-4,
            block_size=256,
        )
        xpt.assert_allclose(r_cma.y_hat, r_lms.y_hat, atol=1e-4, rtol=1e-4)

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_divergence_raises(self, backend_device, xp):
        channel = np.array([0.1, 1.0, 0.2], np.complex64)
        _, rx = _isi_signal(xp, "psk", 4, 4000, 1, channel)
        with pytest.raises(RuntimeError, match="diverged"):
            block_cma(
                rx,
                modulation="psk",
                order=4,
                num_taps=15,
                step_size=5.0,  # far above the stability ceiling
                block_size=256,
            )


class TestBlockCMA:
    def test_blind_converges(self, backend_device, xp):
        channel = np.array([0.06, 1.0, -0.25, 0.08], np.complex64)
        _, rx = _isi_signal(xp, "psk", 4, 40000, 5, channel)
        r = block_cma(
            rx,
            modulation="psk",
            order=4,
            num_taps=15,
            step_size=1e-3,
            block_size=256,
        )
        assert r.y_hat.shape[-1] == 40000
        assert _dispersion(r.y_hat[20000:], 4, "psk") < 0.02

    def test_cpu_gpu_consistent(self, backend_device, xp):
        """block_cma output is consistent across CPU/GPU (within float32)."""
        channel = np.array([0.06, 1.0, -0.25, 0.08], np.complex64)
        _, rx = _isi_signal(np, "psk", 4, 8000, 5, channel)
        ref = block_cma(
            rx, modulation="psk", order=4, num_taps=15, step_size=1e-3, block_size=256
        )
        cur = block_cma(
            xp.asarray(rx),
            modulation="psk",
            order=4,
            num_taps=15,
            step_size=1e-3,
            block_size=256,
        )
        assert np.max(np.abs(_to_np(cur.y_hat) - ref.y_hat)) < 1e-3

    def test_pilot_aided_resolves_phase(self, backend_device, xp):
        channel = np.array([0.06, 1.0, -0.25, 0.08], np.complex64)
        n = 20000
        tx, rx = _isi_signal(xp, "psk", 4, n, 5, channel)
        pmask = np.zeros(n, bool)
        pmask[::8] = True
        pref, pu8 = build_pilot_ref(_to_np(tx)[pmask], pmask, n, 1)
        r = block_cma(
            rx,
            modulation="psk",
            order=4,
            num_taps=15,
            step_size=1e-3,
            block_size=128,
            pilot_ref=pref,
            pilot_mask=pu8,
        )
        # Phase resolved by pilots => low MSE vs the true symbols (no ambiguity).
        e = _to_np(r.y_hat)[10000:] - _to_np(tx)[10000 : r.y_hat.shape[-1]]
        mse_db = 10 * np.log10(float(np.mean(np.abs(e) ** 2)) + 1e-30)
        assert mse_db < -10.0, f"PA block_cma did not resolve phase: {mse_db:.1f} dB"


class TestBlockRDE:
    def test_blind_converges_16qam(self, backend_device, xp):
        channel = np.array([0.06, 1.0, -0.25, 0.08], np.complex64)
        _, rx = _isi_signal(xp, "qam", 16, 16000, 5, channel)
        r = block_rde(
            rx,
            modulation="qam",
            order=16,
            num_taps=15,
            step_size=1e-4,
            block_size=256,
        )
        assert r.y_hat.shape[-1] == 16000
        # Multi-ring 16-QAM: RDE drives |y| onto the rings (blind, phase-ambiguous).
        assert _dispersion(r.y_hat[8000:], 16, "qam") < 0.05

    def test_remainder_block_size(self, backend_device, xp):
        """n_sym not divisible by block_size yields full-length output."""
        channel = np.array([0.1, 1.0, 0.2], np.complex64)
        n_odd = 8003
        _, rx = _isi_signal(xp, "qam", 16, n_odd, 7, channel)
        r = block_rde(
            rx,
            modulation="qam",
            order=16,
            num_taps=15,
            step_size=1e-4,
            block_size=256,
        )
        assert r.y_hat.shape[-1] == n_odd


class TestBlockBlindMIMO:
    def test_butterfly_runs(self, backend_device, xp):
        channel = np.array([0.06, 1.0, -0.25], np.complex64)
        _, rx = _isi_signal(xp, "psk", 4, 20000, 3, channel)
        rx2 = xp.stack([rx, xp.roll(rx, 1)])
        r = block_cma(
            rx2,
            modulation="psk",
            order=4,
            num_taps=15,
            step_size=1e-3,
            block_size=128,
        )
        assert r.y_hat.shape == (2, 20000)
        for ch in range(2):
            assert _dispersion(r.y_hat[ch, 10000:], 4, "psk") < 0.05
