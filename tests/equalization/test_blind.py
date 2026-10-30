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
from tests.common import scoring, skip_unless_gpu
from tests.common.conversions import to_numpy
from tests.common.signals import make_channel_signal, make_isi_distorted_signal

_isi_signal = make_isi_distorted_signal


def _channel_record(constellation, num_symbols, *, esn0_db, seed=0, xp=None):
    """RRC record at 2 sps through a 4-tap complex ISI channel and a
    0.3-sample delay, with AWGN."""
    return make_channel_signal(
        constellation,
        num_symbols,
        isi_taps=[0.25, 1.0, 0.3j, -0.1],
        frac_delay=0.3,
        esn0_db=esn0_db,
        seed=seed,
        xp=xp,
    )


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
        """Block CMA on QPSK through a 4-tap complex ISI channel and a
        0.3-sample delay (Es/N0 18 dB): after one constant rotation the second
        half of 2e4 symbols has no errors and an EVM within 1.4x the AWGN
        floor (0.126); frozen taps stay above twice it."""
        c = Constellation.psk(4)
        rx = _channel_record(c, 20000, esn0_db=18.0, xp=xp)
        ref = to_numpy(rx.reference.symbols)
        floor = 10 ** (-18 / 20)

        def score(step):
            y = block_cma(
                rx, constellation=c, num_taps=21, step_size=step, block_size=256
            )
            y = scoring.resolve_blind(y.y_hat, ref, c, skip=10000, rotation="any")
            return scoring.ser(y[10000:], ref[10000:], c), scoring.evm_rms(
                y[10000:], ref[10000:]
            )

        ser, evm = score(1e-3)
        assert ser == 0.0 and evm < 1.4 * floor
        assert score(0.0)[1] > 2 * floor

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
        """Pilots every 8th symbol fix the absolute phase: through the ISI
        channel with a 1 rad carrier rotation the last quarter of 4e4 symbols
        matches the transmitted symbols with no ambiguity resolution; without
        pilots it does not (SER ~1)."""
        c = Constellation.qam(16)
        rx = _channel_record(c, 40000, esn0_db=22.0, seed=2)
        rx = rx.replace(samples=(rx.samples * np.exp(1j)).astype(np.complex64))
        rx = rx.to("gpu" if xp is not np else "cpu")
        ref = to_numpy(rx.reference.symbols)
        pmask = np.zeros(ref.size, bool)
        pmask[::8] = True
        kw = dict(constellation=c, num_taps=21, step_size=1e-3, block_size=128)
        aided = block_cma(rx, **kw, pilots=EqualizerPilots(ref[pmask], pmask))
        blind = block_cma(rx, **kw)
        assert scoring.ser(to_numpy(aided.y_hat)[30000:], ref[30000:], c) < 1e-3
        assert scoring.ser(to_numpy(blind.y_hat)[30000:], ref[30000:], c) > 0.5

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
        """Block RDE on 16-QAM through the ISI channel (Es/N0 22 dB): after one
        constant rotation the second half has no errors and an EVM within
        1.4x the AWGN floor (0.079); frozen taps leave SER above 0.3.  (The
        radial distance to the rings, used before, passed at SER 0.12.)"""
        c = Constellation.qam(16)
        rx = _channel_record(c, 20000, esn0_db=22.0, xp=xp)
        ref = to_numpy(rx.reference.symbols)

        def resolved(step):
            y = block_rde(
                rx, constellation=c, num_taps=21, step_size=step, block_size=256
            )
            return scoring.resolve_blind(y.y_hat, ref, c, skip=10000, rotation="any")[
                10000:
            ]

        y = resolved(1e-3)
        assert scoring.ser(y, ref[10000:], c) == 0.0
        assert scoring.evm_rms(y, ref[10000:]) < 1.4 * 10 ** (-22 / 20)
        assert scoring.ser(resolved(0.0), ref[10000:], c) > 0.3

    def test_default_step_converges_from_cold_start(self, xp, xpt):
        """The default step, min(1e-3, 0.5/block_size), opens a dual-pol
        16-QAM eye under a 30-degree polarization rotation from a cold start
        (the previous fixed 2e-4 had not converged after 10^5 symbols)."""
        from commkit import RRC, generate
        from commkit.impairments import apply_awgn, apply_polarization_mixing

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

        # One permutation and one rotation per stream; every stream must
        # carry its own payload (a permutation is bijective).  At 20 dB the
        # AWGN-only 16-QAM SER is 2e-5.
        ref = tx.reference.symbols[:, -tail:]
        ser = scoring.blind_ser(to_numpy(r.y_hat)[:, -tail:], ref, c16)
        assert np.all(ser < 1e-3), f"SER {ser}: default step did not converge"

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
    def test_butterfly_separates_independent_streams(self, xp):
        """Block CMA demultiplexes two independent QPSK streams through half a
        symbol of DGD and a 30-degree rotation: one permutation and one
        rotation per stream leave no errors, and each output correlates with
        its own stream only."""
        c = Constellation.psk(4)
        rx = make_channel_signal(
            c,
            20000,
            dgd=0.5e-9,
            dgd_theta=np.pi / 8,
            theta=np.deg2rad(30.0),
            esn0_db=18.0,
            num_channels=2,
            seed=4,
            xp=xp,
        )
        ref = to_numpy(rx.reference.symbols)
        r = block_cma(rx, constellation=c, num_taps=21, step_size=1e-3, block_size=128)
        y = scoring.resolve_blind(r.y_hat, ref, c, skip=10000, rotation="any")
        np.testing.assert_array_equal(scoring.ser(y[:, 10000:], ref[:, 10000:], c), 0.0)
        corr = scoring.stream_correlation(y[:, 10000:], ref[:, 10000:])
        assert corr[0, 1] < 0.05 and corr[1, 0] < 0.05, corr
