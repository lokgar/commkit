"""Butterfly (2x2) MIMO equalization: stream separation.

Every test sends independent streams and checks that each output carries
its own one: SER against the transmitted streams after at most one
bijective channel permutation and one constant rotation per stream (blind
receivers only), and a near-identity stream-correlation matrix, so that two
outputs locked onto the same source fail.
"""

import numpy as np

from commkit import equalization
from commkit.mapping import Constellation
from tests.common import scoring
from tests.common.conversions import to_numpy
from tests.common.signals import make_channel_signal

ISI_TAPS = [0.25, 1.0, 0.3j, -0.1]
THETA = np.deg2rad(30.0)


def _dual_pol(constellation, num_symbols, *, esn0_db, seed=3, xp=None, **channel):
    """Two independent RRC streams at 2 sps through a 30-degree rotation and
    the given channel (ISI taps or DGD), with AWGN."""
    return make_channel_signal(
        constellation,
        num_symbols,
        theta=THETA,
        esn0_db=esn0_db,
        num_channels=2,
        seed=seed,
        xp=xp,
        **channel,
    )


def _assert_separated(y, ref, constellation, skip, rotation="any"):
    y = scoring.resolve_blind(y, ref, constellation, skip=skip, rotation=rotation)
    ref = to_numpy(ref)[:, : y.shape[-1]]
    np.testing.assert_array_equal(
        scoring.ser(y[:, skip:], ref[:, skip:], constellation), 0.0
    )
    corr = scoring.stream_correlation(y[:, skip:], ref[:, skip:])
    assert np.all(np.diag(corr) > 0.95) and corr[0, 1] < 0.05 and corr[1, 0] < 0.05, (
        corr
    )


class TestButterflyMIMO:
    """Trained and blind butterflies separate independent streams."""

    def test_lms_2x2_cross_channel(self, xp):
        """Trained LMS on ISI plus rotation: after 1000 training symbols each
        output carries its own stream (held-out half); frozen taps leave the
        streams mixed."""
        c = Constellation.qam(16)
        rx = _dual_pol(c, 8000, esn0_db=22.0, isi_taps=ISI_TAPS, xp=xp)
        ref = to_numpy(rx.reference.symbols)
        res = equalization.lms(rx, ref[:, :1000], num_taps=21, step_size=1e-3)
        _assert_separated(res.y_hat, ref, c, 4000, rotation="symmetry")
        frozen = equalization.lms(rx, ref[:, :1000], num_taps=21, step_size=0.0)
        y = to_numpy(frozen.y_hat)
        assert np.all(scoring.ser(y[:, 4000:], ref[:, 4000:], c) > 0.3)

    def test_lms_undoes_differential_group_delay(self, xp):
        """Half a symbol of DGD at π/8 plus a rotation is frequency selective;
        trained LMS removes it (SER 0 on both streams; frozen taps 0.73)."""
        c = Constellation.qam(16)
        rx = _dual_pol(
            c, 8000, esn0_db=22.0, dgd=0.5e-9, dgd_theta=np.pi / 8, seed=4, xp=xp
        )
        ref = to_numpy(rx.reference.symbols)
        res = equalization.lms(rx, ref[:, :1000], num_taps=21, step_size=1e-3)
        _assert_separated(res.y_hat, ref, c, 4000, rotation="symmetry")
        frozen = to_numpy(
            equalization.lms(rx, ref[:, :1000], num_taps=21, step_size=0.0).y_hat
        )
        assert np.all(scoring.ser(frozen[:, 4000:], ref[:, 4000:], c) > 0.3)

    def test_cma_2x2_polarization_demux(self, xp):
        """Blind CMA demultiplexes two QPSK streams through ISI and a
        rotation: one permutation and one rotation per stream leave no
        errors and a cross-correlation under 0.05 (frozen taps: 0.48)."""
        c = Constellation.psk(4)
        rx = _dual_pol(c, 20000, esn0_db=18.0, isi_taps=ISI_TAPS, xp=xp)
        ref = to_numpy(rx.reference.symbols)
        res = equalization.cma(rx, num_taps=21, step_size=1e-3, constellation=c)
        _assert_separated(res.y_hat, ref, c, 10000)
        frozen = equalization.cma(rx, num_taps=21, step_size=0.0, constellation=c)
        y = scoring.resolve_blind(frozen.y_hat, ref, c, skip=10000, rotation="any")
        assert scoring.stream_correlation(y[:, 10000:], ref[:, 10000:])[0, 1] > 0.3

    def test_zf_mimo_channel_matrix(self, xp, xpt):
        """ZF with full (C, C, L) channel matrix should invert MIMO channel."""
        n = 128
        rng = xp.random.RandomState(42)

        # 2x2 single-tap channel matrix
        # H[rx, tx, delay]
        H = xp.zeros((2, 2, 1), dtype=xp.complex64)
        H[0, 0, 0] = 1.0
        H[0, 1, 0] = 0.3
        H[1, 0, 0] = 0.2
        H[1, 1, 0] = 1.0

        # Transmit 2 independent streams
        tx = (rng.randn(2, n) + 1j * rng.randn(2, n)).astype(xp.complex64)

        # Apply MIMO channel in frequency domain
        H_f = xp.fft.fft(H, n=n, axis=-1)  # (2, 2, N)
        Tx_f = xp.fft.fft(tx, axis=-1)  # (2, N)
        Rx_f = xp.zeros_like(Tx_f)

        # Manually compute matrix mul at each freq bin?
        # Einsum could work: 'r t k, t k -> r k'
        # CuPy supports einsum.
        # But for loop is also fine for test clarity.
        # Actually H_f is (Rx, Tx, Freq), Tx_f is (Tx, Freq).
        # We want (Rx, Freq).

        # Vectorized matmul broadcasting over freq?
        # Move axis to make it (N, Rx, Tx) @ (N, Tx, 1) -> (N, Rx, 1)
        # H_f_T = H_f.transpose(2, 0, 1) # (N, 2, 2)
        # Tx_f_T = Tx_f.T[..., None]     # (N, 2, 1)
        # Rx_f_T = H_f_T @ Tx_f_T        # (N, 2, 1)
        # Rx_f = Rx_f_T.squeeze(-1).T    # (2, N)

        H_f_T = xp.moveaxis(H_f, -1, 0)
        Tx_f_T = xp.moveaxis(Tx_f, -1, 0)[..., None]
        Rx_f_T = H_f_T @ Tx_f_T
        Rx_f = xp.moveaxis(Rx_f_T.squeeze(-1), 0, -1)

        rx = xp.fft.ifft(Rx_f, axis=-1).astype(xp.complex64)

        equalized = equalization.zf_equalizer(rx, H)

        xpt.assert_allclose(equalized, tx, atol=1e-3)


class TestButterflyMIMOExtended:
    """The two-stage blind receiver on 16-QAM."""

    def test_rde_2x2_butterfly_numba(self, xp):
        """CMA pre-convergence, then RDE from its taps (the standard blind
        receiver for 16-QAM): two streams through ISI and a rotation are
        separated without errors, with a lower EVM than CMA alone."""
        c = Constellation.qam(16)
        rx = _dual_pol(c, 20000, esn0_db=22.0, isi_taps=ISI_TAPS, xp=xp)
        ref = to_numpy(rx.reference.symbols)
        cma = equalization.cma(rx, num_taps=21, step_size=5e-4, constellation=c)
        rde = equalization.rde(
            rx, num_taps=21, step_size=5e-4, constellation=c, initial_taps=cma.weights
        )
        _assert_separated(rde.y_hat, ref, c, 10000)
        evm = [
            scoring.evm_rms(
                scoring.resolve_blind(r.y_hat, ref, c, skip=10000, rotation="any")[
                    :, 10000:
                ],
                ref[:, 10000:],
            )
            for r in (cma, rde)
        ]
        assert np.all(evm[1] < evm[0]), evm
