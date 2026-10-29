"""Tests for the blind frequency-domain equalizers block_cma / block_rde.

These are the blind, phase-directed siblings of ``block_lms``: the same
overlap-save FDAF engine (shared via ``_fdaf_forward`` / ``_fdaf_gradient_update``)
driven by the Godard / ring-radius error instead of a trained/DD slicer.
"""

import numpy as np
import pytest

from commkit.core import Signal
from commkit.equalization import (
    EqualizerPilots,
    block_cma,
    block_lms,
    block_rde,
)
from commkit.equalization._block._blind import _cupy_rde_error
from commkit.equalization._common import _rde_ring_radii, _rde_rings
from commkit.mapping import Constellation
from tests.common import skip_unless_gpu
from tests.common.conversions import to_numpy
from tests.common.metrics import calc_dispersion
from tests.common.signals import make_isi_distorted_signal

_isi_signal = make_isi_distorted_signal


class TestBlockFDAFEngine:
    def test_all_pilots_block_cma_matches_block_lms(self, xp, xpt):
        """With a full pilot mask every position uses the LMS residual error, so
        block_cma must reproduce block_lms (training) bit-for-bit - the strongest
        check that the shared FDAF primitives match block_lms's engine."""
        channel = np.array([0.06, 1.0, -0.25, 0.08], np.complex64)
        tx, rx = _isi_signal(xp, "qam", 16, 8000, 6, channel)
        n = 8000
        pmask = np.ones(n, bool)
        pilots = EqualizerPilots(to_numpy(tx), pmask)
        r_cma = block_cma(
            rx,
            constellation=Constellation.qam(16),
            num_taps=15,
            step_size=2e-4,
            block_size=256,
            pilots=pilots,
            sps=2,
        )
        r_lms = block_lms(
            rx,
            tx,
            constellation=Constellation.qam(16),
            num_taps=15,
            step_size=2e-4,
            block_size=256,
            sps=2,
        )
        xpt.assert_allclose(r_cma.y_hat, r_lms.y_hat, atol=1e-4, rtol=1e-4)

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_divergence_raises(self, xp):
        channel = np.array([0.1, 1.0, 0.2], np.complex64)
        _, rx = _isi_signal(xp, "psk", 4, 4000, 1, channel)
        with pytest.raises(RuntimeError, match="diverged"):
            block_cma(
                rx,
                constellation=Constellation.psk(4),
                num_taps=15,
                step_size=5.0,
                block_size=256,
                sps=2,
            )

    @pytest.mark.parametrize("fn", [block_lms, block_cma, block_rde])
    @pytest.mark.parametrize("block_size", [0, -4, 256.0])
    def test_block_size_must_be_positive_integer(self, xp, fn, block_size):
        rx = xp.ones(2000, dtype=xp.complex64)
        data = (rx[:10],) if fn is block_lms else ()
        with pytest.raises(ValueError, match="block_size must be a positive integer"):
            fn(
                rx,
                *data,
                constellation=Constellation.qam(16),
                num_taps=5,
                block_size=block_size,
                sps=2,
            )


class TestBlockCMA:
    def test_blind_converges(self, xp):
        channel = np.array([0.06, 1.0, -0.25, 0.08], np.complex64)
        _, rx = _isi_signal(xp, "psk", 4, 40000, 5, channel)
        r = block_cma(
            rx,
            constellation=Constellation.psk(4),
            num_taps=15,
            step_size=1e-3,
            block_size=256,
            sps=2,
        )
        assert r.y_hat.shape[-1] == 40000
        assert calc_dispersion(r.y_hat[20000:], 4, "psk") < 0.02

    def test_cpu_gpu_consistent(self, xp):
        """block_cma output is consistent across CPU/GPU (within float32)."""
        channel = np.array([0.06, 1.0, -0.25, 0.08], np.complex64)
        _, rx = _isi_signal(np, "psk", 4, 8000, 5, channel)
        ref = block_cma(
            rx,
            constellation=Constellation.psk(4),
            num_taps=15,
            step_size=1e-3,
            block_size=256,
            sps=2,
        )
        cur = block_cma(
            xp.asarray(rx),
            constellation=Constellation.psk(4),
            num_taps=15,
            step_size=1e-3,
            block_size=256,
            sps=2,
        )
        assert np.max(np.abs(to_numpy(cur.y_hat) - ref.y_hat)) < 1e-3

    def test_pilot_aided_resolves_phase(self, xp):
        channel = np.array([0.06, 1.0, -0.25, 0.08], np.complex64)
        n = 20000
        tx, rx = _isi_signal(xp, "psk", 4, n, 5, channel)
        pmask = np.zeros(n, bool)
        pmask[::8] = True
        pilots = EqualizerPilots(to_numpy(tx)[pmask], pmask)
        r = block_cma(
            rx,
            constellation=Constellation.psk(4),
            num_taps=15,
            step_size=1e-3,
            block_size=128,
            pilots=pilots,
            sps=2,
        )
        # Phase resolved by pilots => low MSE vs the true symbols (no ambiguity).
        e = to_numpy(r.y_hat)[10000:] - to_numpy(tx)[10000 : r.y_hat.shape[-1]]
        mse_db = 10 * np.log10(float(np.mean(np.abs(e) ** 2)) + 1e-30)
        assert mse_db < -10.0, f"PA block_cma did not resolve phase: {mse_db:.1f} dB"

    def test_signal_input(self, xp, xpt):
        """Signal input: sps is taken from the signal, y_hat becomes a Signal
        at the symbol rate."""
        channel = np.array([0.06, 1.0, -0.25, 0.08], np.complex64)
        _, rx = _isi_signal(xp, "psk", 4, 4000, 5, channel)
        sig = Signal(samples=rx, sampling_rate=2e6, symbol_rate=1e6)
        kw = dict(
            constellation=Constellation.psk(4),
            num_taps=15,
            step_size=1e-3,
            block_size=256,
        )

        result_sig = block_cma(sig, **kw)
        result_arr = block_cma(rx, sps=2, **kw)

        assert isinstance(result_sig.signal, Signal)
        assert result_sig.signal.sampling_rate == 1e6
        xpt.assert_allclose(result_sig.signal.samples, result_arr.y_hat)
        xpt.assert_allclose(result_sig.y_hat, result_arr.y_hat)


class TestBlockRDE:
    def test_blind_converges_16qam(self, xp):
        channel = np.array([0.06, 1.0, -0.25, 0.08], np.complex64)
        _, rx = _isi_signal(xp, "qam", 16, 16000, 5, channel)
        r = block_rde(
            rx,
            constellation=Constellation.qam(16),
            num_taps=15,
            step_size=1e-4,
            block_size=256,
            sps=2,
        )
        assert r.y_hat.shape[-1] == 16000
        # Multi-ring 16-QAM: RDE drives |y| onto the rings (blind, phase-ambiguous).
        assert calc_dispersion(r.y_hat[8000:], 16, "qam") < 0.05

    def test_default_step_converges_from_cold_start(self, xp, xpt):
        """The default step, min(1e-3, 0.5/block_size), opens a dual-pol
        16-QAM eye under a 30-degree polarization rotation from a cold start
        (the previous fixed 2e-4 had not converged after 10^5 symbols)."""
        from commkit import RRC, generate
        from commkit.impairments import apply_awgn, apply_polarization_mixing
        from commkit.recovery import resolve_phase_ambiguity

        c16 = Constellation.qam(16)
        n_sym, tail, block_size = 49152, 16384, 256
        tx = generate(
            c16, n_sym, symbol_rate=1e9, sps=2, pulse=RRC(0.1), num_channels=2, rng=1
        )
        rx = apply_polarization_mixing(tx.samples, theta=np.deg2rad(30.0))
        rx = xp.asarray(apply_awgn(rx, esn0_db=20.0, sps=2, rng=3))
        kw = dict(sps=2, num_taps=21, constellation=c16, block_size=block_size)
        r = block_rde(rx, **kw)
        r_explicit = block_rde(rx, step_size=min(1e-3, 0.5 / block_size), **kw)
        xpt.assert_array_equal(r.y_hat, r_explicit.y_hat)

        ref = tx.reference.symbols[:, -tail:]
        y = resolve_phase_ambiguity(
            to_numpy(r.y_hat)[:, -tail:], ref, constellation=c16
        )
        ser = float(np.mean(c16.demap(y) != c16.demap(ref)))
        assert ser < 1e-3, f"SER {ser:.2e}: default step did not converge"

    def test_remainder_block_size(self, xp):
        """n_sym not divisible by block_size yields full-length output."""
        channel = np.array([0.1, 1.0, 0.2], np.complex64)
        n_odd = 8003
        _, rx = _isi_signal(xp, "qam", 16, n_odd, 7, channel)
        r = block_rde(
            rx,
            constellation=Constellation.qam(16),
            num_taps=15,
            step_size=1e-4,
            block_size=256,
            sps=2,
        )
        assert r.y_hat.shape[-1] == n_odd

    def test_signal_input(self, xp, xpt):
        """Signal input: sps is taken from the signal, y_hat becomes a Signal
        at the symbol rate."""
        channel = np.array([0.06, 1.0, -0.25, 0.08], np.complex64)
        _, rx = _isi_signal(xp, "qam", 16, 4000, 5, channel)
        sig = Signal(samples=rx, sampling_rate=2e6, symbol_rate=1e6)
        kw = dict(
            constellation=Constellation.qam(16),
            num_taps=15,
            step_size=1e-4,
            block_size=256,
        )

        result_sig = block_rde(sig, **kw)
        result_arr = block_rde(rx, sps=2, **kw)

        assert isinstance(result_sig.signal, Signal)
        assert result_sig.signal.sampling_rate == 1e6
        xpt.assert_allclose(result_sig.signal.samples, result_arr.y_hat)
        xpt.assert_allclose(result_sig.y_hat, result_arr.y_hat)


class TestRDERings:
    @pytest.mark.parametrize("order", [16, 64, 256])
    def test_squared_midpoints_pick_the_nearest_radius(self, order):
        """Counting the squared midpoints below |y|² picks the radius nearest
        |y|, the definition of the RDE decision, without a square root."""
        radii = _rde_ring_radii(Constellation.qam(order))
        rings = _rde_rings(radii)
        np.testing.assert_array_equal(rings[0], radii * radii)
        assert rings[1, -1] == np.finfo(np.float32).max
        a = np.random.default_rng(0).uniform(0.0, 1.2 * radii[-1], 100_000)
        a = a.astype(np.float32)
        nearest = np.argmin(np.abs(a[:, None] - radii[None, :]), axis=1)
        counted = np.searchsorted(rings[1, :-1], a * a)
        np.testing.assert_array_equal(counted, nearest)

    def test_tie_at_midpoint_goes_to_inner_ring(self):
        rings = _rde_rings(np.array([1.0, 3.0], dtype=np.float32))
        assert rings[1, 0] == 4.0
        assert np.searchsorted(rings[1, :-1], np.float32(4.0)) == 0

    @pytest.mark.gpu_only
    def test_kernel_matches_cpu(self, backend_device, xp, xpt):
        skip_unless_gpu(backend_device)
        rings = _rde_rings(_rde_ring_radii(Constellation.qam(64)))
        rng = np.random.default_rng(1)
        y = 0.7 * (rng.standard_normal((2, 4099)) + 1j * rng.standard_normal((2, 4099)))
        y = y.astype(np.complex64)
        a2 = (y * np.conj(y)).real
        expected = y * (rings[0][np.searchsorted(rings[1, :-1], a2)] - a2)
        y_d = xp.asarray(y)
        e = _cupy_rde_error()(
            y_d,
            xp.real(y_d * xp.conj(y_d)),
            xp.asarray(rings),
            np.int32(rings.shape[1]),
        )
        xpt.assert_allclose(e, xp.asarray(expected), rtol=1e-6, atol=1e-7)


class TestBlockBlindMIMO:
    def test_butterfly_runs(self, xp):
        channel = np.array([0.06, 1.0, -0.25], np.complex64)
        _, rx = _isi_signal(xp, "psk", 4, 20000, 3, channel)
        rx2 = xp.stack([rx, xp.roll(rx, 1)])
        r = block_cma(
            rx2,
            constellation=Constellation.psk(4),
            num_taps=15,
            step_size=1e-3,
            block_size=128,
            sps=2,
        )
        assert r.y_hat.shape == (2, 20000)
        for ch in range(2):
            assert calc_dispersion(r.y_hat[ch, 10000:], 4, "psk") < 0.05
