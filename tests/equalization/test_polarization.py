"""Static and dynamic polarization-tone demultiplexing."""

import numpy as np
import pytest

from commkit import equalization
from commkit.core import Signal
from commkit.impairments import apply_polarization_mixing
from commkit.spectral import add_pilot_tone, grid_frequency


def _demux(samples, *, trim_edges=False, **estimate):
    """estimate_polarization_tones, then correct_polarization."""
    track = equalization.estimate_polarization_tones(samples, **estimate)
    return equalization.correct_polarization(samples, track, trim_edges=trim_edges)


class TestDemultiplexPolarizationTones:
    """A static estimate: one unmixing matrix for the record."""

    # Data occupies |f| < 0.3*fs; tones live in the clean guard band beyond it,
    # mirroring a real pilot-tone placement so single-bin extraction is exact.
    TONES = [35.0, -42.0]  # Hz, both inside the guard for fs=100

    @staticmethod
    def _streams(xp, C=2, N=8192, seed=0, fs=100.0, band_frac=0.3):
        """C independent, band-limited, unit-power complex streams, shape (C, N)."""
        rng = xp.random.default_rng(seed)
        s = (rng.standard_normal((C, N)) + 1j * rng.standard_normal((C, N))).astype(
            xp.complex64
        )
        # Brick-wall low-pass so a guard band is left for the pilot tones.
        freqs = xp.fft.fftfreq(N, d=1.0 / fs)
        mask = (xp.abs(freqs) < band_frac * fs)[None, :]
        s = xp.fft.ifft(xp.fft.fft(s, axis=-1) * mask, axis=-1).astype(xp.complex64)
        s /= xp.sqrt(xp.mean(xp.abs(s) ** 2, axis=-1, keepdims=True))
        return s

    @staticmethod
    def _corr(xp, a, b):
        """Magnitude of the normalised complex correlation (scale-invariant)."""
        num = xp.abs(xp.sum(a * xp.conj(b)))
        den = xp.sqrt(xp.sum(xp.abs(a) ** 2) * xp.sum(xp.abs(b) ** 2))
        return float(num / den)

    def test_demux_recovers_streams_no_permutation(self, xp):
        """A static rotation is inverted; row j tracks tone j (no permutation)."""

        fs = 100.0
        s = self._streams(xp)
        tx = add_pilot_tone(
            s, sampling_rate=fs, frequency=self.TONES, power_ratio_db=-10.0
        )
        f_used = grid_frequency(self.TONES, sampling_rate=fs, num_samples=s.shape[-1])
        rx = apply_polarization_mixing(tx, theta=0.6)  # frequency-flat Jones mix

        demuxed = _demux(rx, sampling_rate=fs, tone_frequencies=f_used)

        assert demuxed.shape == tx.shape
        assert demuxed.dtype == rx.dtype
        for j in range(2):
            # Row j recovers the transmitted stream j (data+tone) up to a scale.
            assert self._corr(xp, demuxed[j], tx[j]) > 0.999
            # ...and rejects the other stream (cross-talk suppressed).
            assert self._corr(xp, demuxed[j], tx[1 - j]) < 0.05

    def test_demux_robust_to_noise(self, xp):
        """The tone-phasor estimate still locks under moderate AWGN."""

        fs = 100.0
        s = self._streams(xp, N=8192)
        tx = add_pilot_tone(
            s, sampling_rate=fs, frequency=self.TONES, power_ratio_db=-8.0
        )
        f_used = grid_frequency(self.TONES, sampling_rate=fs, num_samples=s.shape[-1])
        rx = apply_polarization_mixing(tx, theta=-0.9)
        rng = xp.random.RandomState(123)
        std = 0.1
        rx = rx + std * (rng.randn(*rx.shape) + 1j * rng.randn(*rx.shape)).astype(
            rx.dtype
        )

        demuxed = _demux(rx, sampling_rate=fs, tone_frequencies=f_used)
        for j in range(2):
            assert self._corr(xp, demuxed[j], tx[j]) > 0.99

    def test_static_track(self, xp, xpt):
        """A static estimate holds one (K, C) complex128 unmixing matrix."""

        fs = 100.0
        s = self._streams(xp)
        tx = add_pilot_tone(
            s, sampling_rate=fs, frequency=self.TONES, power_ratio_db=-10.0
        )
        f_used = grid_frequency(self.TONES, sampling_rate=fs, num_samples=s.shape[-1])
        track = equalization.estimate_polarization_tones(
            tx, sampling_rate=fs, tone_frequencies=f_used
        )
        W = track.matrix
        assert W.shape == (2, 2)
        assert W.dtype == xp.complex128
        assert track.edge == 0 and track.matrix_grid.shape == (1, 2, 2)
        assert track.tone_frequencies == pytest.approx(f_used, abs=0.05)
        # A bare matrix is applied like the track.
        xpt.assert_allclose(
            equalization.correct_polarization(tx, W),
            equalization.correct_polarization(tx, track),
        )

    def test_overdetermined_pinv(self, xp):
        """C > K: a (3, 2) mixing is unmixed to 2 streams via the pseudo-inverse."""

        fs = 100.0
        s = self._streams(xp, C=2, N=8192)
        tx = add_pilot_tone(
            s, sampling_rate=fs, frequency=self.TONES, power_ratio_db=-10.0
        )
        f_used = grid_frequency(self.TONES, sampling_rate=fs, num_samples=s.shape[-1])
        rng = xp.random.RandomState(7)
        J = (rng.randn(3, 2) + 1j * rng.randn(3, 2)).astype(xp.complex64)  # (3, 2)
        rx = (J @ tx.astype(xp.complex64)).astype(xp.complex64)  # (3, N)

        demuxed = _demux(rx, sampling_rate=fs, tone_frequencies=f_used)
        assert demuxed.shape == (2, tx.shape[-1])
        for j in range(2):
            assert self._corr(xp, demuxed[j], tx[j]) > 0.999

    def test_more_tones_than_channels_raises(self, xp):
        """K > C is rejected (cannot unmix more streams than receive channels)."""
        fs = 100.0
        s = self._streams(xp, C=2)
        with pytest.raises(ValueError, match=r"need K <= C"):
            _demux(s, sampling_rate=fs, tone_frequencies=[10.0, 20.0, 30.0])

    def test_requires_2d(self, xp):
        """A 1-D (SISO) input is rejected - demux is inherently MIMO."""
        fs = 100.0
        s = self._streams(xp, C=1)[0]  # (N,)
        with pytest.raises(ValueError, match=r"2-D \(C, N\)"):
            _demux(s, sampling_rate=fs, tone_frequencies=[10.0])

    def test_signal_input_returns_signal(self, xp, xpt):
        """Signal input: sampling_rate is taken from the signal."""

        fs = 100.0
        s = self._streams(xp)
        tx = add_pilot_tone(
            s, sampling_rate=fs, frequency=self.TONES, power_ratio_db=-10.0
        )
        f_used = grid_frequency(self.TONES, sampling_rate=fs, num_samples=s.shape[-1])
        rx = apply_polarization_mixing(tx, theta=0.6)
        sig = Signal(samples=rx, sampling_rate=fs, symbol_rate=fs)

        demuxed_sig = _demux(sig, tone_frequencies=f_used)
        demuxed_arr = _demux(rx, sampling_rate=fs, tone_frequencies=f_used)

        assert isinstance(demuxed_sig, Signal)
        xpt.assert_allclose(demuxed_sig.samples, demuxed_arr)

    def test_signal_input_gives_the_same_estimate(self, xp, xpt):
        """Signal input: the same estimate as the array with sampling_rate."""

        fs = 100.0
        s = self._streams(xp)
        tx = add_pilot_tone(
            s, sampling_rate=fs, frequency=self.TONES, power_ratio_db=-10.0
        )
        f_used = grid_frequency(self.TONES, sampling_rate=fs, num_samples=s.shape[-1])
        sig = Signal(samples=tx, sampling_rate=fs, symbol_rate=fs)

        W_sig = equalization.estimate_polarization_tones(
            sig, tone_frequencies=f_used
        ).matrix
        W_arr = equalization.estimate_polarization_tones(
            tx, sampling_rate=fs, tone_frequencies=f_used
        ).matrix
        xpt.assert_allclose(W_sig, W_arr)
        with pytest.raises(ValueError, match="track_bandwidth"):
            equalization.estimate_polarization_tones(
                sig, tone_frequencies=f_used, num_taps=101
            )


class TestDemultiplexPolarizationTonesDynamic:
    """A tracked estimate (track_bandwidth): a drifting SOP."""

    TONES = TestDemultiplexPolarizationTones.TONES
    _streams = staticmethod(TestDemultiplexPolarizationTones._streams)
    _corr = staticmethod(TestDemultiplexPolarizationTones._corr)

    def test_tracks_drifting_sop_where_static_fails(self, xp):
        """A rotating SOP defeats the static one-shot inverse but is tracked here."""

        fs = 100.0
        N = 16384
        s = self._streams(xp, N=N)
        tx = add_pilot_tone(
            s, sampling_rate=fs, frequency=self.TONES, power_ratio_db=-8.0
        )
        f_used = grid_frequency(self.TONES, sampling_rate=fs, num_samples=s.shape[-1])
        # SOP sweeps a full ~pi/2 across the capture - the static average Jones
        # matrix matches no instant, so its cross-talk rejection collapses.
        drift = (np.pi / 2) / N * fs  # rad/s
        rx = apply_polarization_mixing(
            tx, theta=0.2, drift_rate=drift, sampling_rate=fs
        )

        static = _demux(rx, sampling_rate=fs, tone_frequencies=f_used)
        dynamic = _demux(
            rx, sampling_rate=fs, tone_frequencies=f_used, track_bandwidth=2.0
        )

        assert dynamic.shape == tx.shape
        assert dynamic.dtype == rx.dtype
        for j in range(2):
            # Dynamic recovers stream j and suppresses the other stream.
            assert self._corr(xp, dynamic[j], tx[j]) > 0.99
            assert self._corr(xp, dynamic[j], tx[1 - j]) < 0.1
        # The static inverse smears the recovered stream (its fixed W cannot
        # follow J(n)), so its signal fidelity is materially worse - this is the
        # symptom the dynamic path is meant to cure.
        min_static = min(self._corr(xp, static[j], tx[j]) for j in range(2))
        min_dynamic = min(self._corr(xp, dynamic[j], tx[j]) for j in range(2))
        assert min_static < 0.95  # static degraded by the drift
        assert min_dynamic > min_static + 0.04

    def test_tracks_with_long_tracking_filter(self, xp):
        """A narrow tracking bandwidth (fs/B in the hundreds, as in real
        captures) gives a tracking FIR of hundreds of taps; it still tracks."""

        fs = 100.0
        N = 16384
        s = self._streams(xp, N=N)
        tx = add_pilot_tone(
            s, sampling_rate=fs, frequency=self.TONES, power_ratio_db=-8.0
        )
        f_used = grid_frequency(self.TONES, sampling_rate=fs, num_samples=s.shape[-1])
        rx = apply_polarization_mixing(
            tx, theta=0.2, drift_rate=(np.pi / 2) / N * fs, sampling_rate=fs
        )

        # Default num_taps = round(3.3 fs / B) = 660 at B = 0.5 Hz.
        dynamic = _demux(
            rx, sampling_rate=fs, tone_frequencies=f_used, track_bandwidth=0.5
        )

        for j in range(2):
            assert self._corr(xp, dynamic[j], tx[j]) > 0.99
            assert self._corr(xp, dynamic[j], tx[1 - j]) < 0.1

    def test_static_sop_still_recovered(self, xp):
        """With no drift the dynamic path matches the static result (no regression)."""

        fs = 100.0
        s = self._streams(xp, N=8192)
        tx = add_pilot_tone(
            s, sampling_rate=fs, frequency=self.TONES, power_ratio_db=-10.0
        )
        f_used = grid_frequency(self.TONES, sampling_rate=fs, num_samples=s.shape[-1])
        rx = apply_polarization_mixing(tx, theta=0.6)

        demuxed = _demux(
            rx, sampling_rate=fs, tone_frequencies=f_used, track_bandwidth=2.0
        )
        for j in range(2):
            assert self._corr(xp, demuxed[j], tx[j]) > 0.99
            assert self._corr(xp, demuxed[j], tx[1 - j]) < 0.1

    def test_track_shapes(self, xp):
        """A tracked estimate holds the (G, K, C) stack and (G,) grid positions."""

        fs = 100.0
        N = 8192
        s = self._streams(xp, N=N)
        tx = add_pilot_tone(
            s, sampling_rate=fs, frequency=self.TONES, power_ratio_db=-10.0
        )
        f_used = grid_frequency(self.TONES, sampling_rate=fs, num_samples=s.shape[-1])
        track = equalization.estimate_polarization_tones(
            tx, sampling_rate=fs, tone_frequencies=f_used, track_bandwidth=2.0
        )
        Wg, grid = track.matrix_grid, track.grid_positions
        assert track.edge > 0
        with pytest.raises(ValueError, match="matrices"):
            _ = track.matrix
        assert Wg.ndim == 3 and Wg.shape[1:] == (2, 2)
        assert Wg.shape[0] == grid.shape[0]
        assert Wg.dtype == xp.complex128
        assert int(grid[-1]) == N - 1  # last sample is pinned on the grid

    def test_invalid_track_bandwidth_raises(self, xp):
        """A non-positive tracking bandwidth is rejected."""
        fs = 100.0
        s = self._streams(xp, N=4096)
        with pytest.raises(ValueError, match=r"track_bandwidth must be positive"):
            _demux(
                s, sampling_rate=fs, tone_frequencies=self.TONES, track_bandwidth=0.0
            )

    def test_trim_edges_returns_valid_interior(self, xp):
        """trim_edges drops the track's edge, num_taps//2, at each side."""

        fs = 100.0
        N = 8192
        num_taps = 165
        s = self._streams(xp, N=N)
        tx = add_pilot_tone(
            s, sampling_rate=fs, frequency=self.TONES, power_ratio_db=-10.0
        )
        f_used = grid_frequency(self.TONES, sampling_rate=fs, num_samples=s.shape[-1])
        track = equalization.estimate_polarization_tones(
            tx,
            sampling_rate=fs,
            tone_frequencies=f_used,
            track_bandwidth=2.0,
            num_taps=num_taps,
        )
        demuxed = equalization.correct_polarization(tx, track, trim_edges=True)
        g = num_taps // 2
        assert track.edge == g
        valid = slice(g, N - g)
        assert demuxed.shape == (2, N - 2 * g)
        # The retained interior must align with the input over `valid`.
        for j in range(2):
            assert self._corr(xp, demuxed[j], tx[j, valid]) > 0.99

    def test_signal_input_returns_signal(self, xp, xpt):
        """Signal input: sampling_rate is taken from the signal."""

        fs = 100.0
        s = self._streams(xp, N=8192)
        tx = add_pilot_tone(
            s, sampling_rate=fs, frequency=self.TONES, power_ratio_db=-10.0
        )
        f_used = grid_frequency(self.TONES, sampling_rate=fs, num_samples=s.shape[-1])
        rx = apply_polarization_mixing(tx, theta=0.6)
        sig = Signal(samples=rx, sampling_rate=fs, symbol_rate=fs)

        demuxed_sig = _demux(sig, tone_frequencies=f_used, track_bandwidth=2.0)
        demuxed_arr = _demux(
            rx, sampling_rate=fs, tone_frequencies=f_used, track_bandwidth=2.0
        )

        assert isinstance(demuxed_sig, Signal)
        xpt.assert_allclose(demuxed_sig.samples, demuxed_arr)

    def test_signal_input_gives_the_same_track(self, xp, xpt):
        """The JonesTrack is the same for Signal and array input."""

        fs = 100.0
        s = self._streams(xp, N=8192)
        tx = add_pilot_tone(
            s, sampling_rate=fs, frequency=self.TONES, power_ratio_db=-10.0
        )
        f_used = grid_frequency(self.TONES, sampling_rate=fs, num_samples=s.shape[-1])
        sig = Signal(samples=tx, sampling_rate=fs, symbol_rate=fs)

        track_sig = equalization.estimate_polarization_tones(
            sig, tone_frequencies=f_used, track_bandwidth=2.0
        )
        track_arr = equalization.estimate_polarization_tones(
            tx, sampling_rate=fs, tone_frequencies=f_used, track_bandwidth=2.0
        )

        assert isinstance(track_sig, equalization.JonesTrack)
        xpt.assert_allclose(track_sig.matrix_grid, track_arr.matrix_grid)
        xpt.assert_allclose(track_sig.grid_positions, track_arr.grid_positions)

    def test_signal_input_trim_edges_raises(self, xp):
        """Trimming samples would misalign a Signal's reference."""
        fs = 100.0
        s = self._streams(xp, N=8192)
        sig = Signal(samples=s, sampling_rate=fs, symbol_rate=fs)
        with pytest.raises(ValueError, match="trim_edges"):
            _demux(
                sig, tone_frequencies=self.TONES, track_bandwidth=2.0, trim_edges=True
            )


class TestCorrectPolarization:
    """correct_polarization with a JonesTrack built by hand: the
    grid-interpolated time-varying matrix apply."""

    def test_matches_per_sample_reference(self, xp, xpt):
        """Block-GEMM apply == per-sample linear-interp einsum (to complex64)."""
        rng = xp.random.RandomState(0)
        N, step = 20000, 4096
        idx = xp.arange(0, N, step)
        if int(idx[-1]) != N - 1:
            idx = xp.concatenate([idx, xp.asarray([N - 1])])
        gp = idx.astype(xp.float64)
        G = int(gp.shape[0])
        M = (
            rng.standard_normal((G, 2, 2)) + 1j * rng.standard_normal((G, 2, 2))
        ).astype(xp.complex64)
        x = (rng.standard_normal((2, N)) + 1j * rng.standard_normal((2, N))).astype(
            xp.complex64
        )
        # per-sample linear-interpolation reference (double precision)
        Md = M.astype(xp.complex128)
        n = xp.arange(N, dtype=xp.float64)
        lo = xp.clip(xp.searchsorted(gp, n, side="right") - 1, 0, G - 2)
        frac = (n - gp[lo]) / (gp[lo + 1] - gp[lo])
        M_full = Md[lo] + (Md[lo + 1] - Md[lo]) * frac[:, None, None]
        ref = xp.einsum("lkc,cl->kl", M_full, x.astype(xp.complex128))
        out = equalization.correct_polarization(
            x, equalization.JonesTrack(M, gp), normalize=False
        )
        assert out.shape == (2, N)
        xpt.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)

    def test_constant_grid_is_static_matmul(self, xp, xpt):
        """A grid of identical matrices applies that one matrix everywhere."""
        N = 10000
        M2 = xp.asarray([[1.0, 0.5j], [0.2, -1.0]], dtype=xp.complex64)
        M = xp.broadcast_to(M2, (5, 2, 2)).copy()
        gp = xp.linspace(0.0, N - 1, 5)
        rng = xp.random.RandomState(1)
        x = (rng.standard_normal((2, N)) + 1j * rng.standard_normal((2, N))).astype(
            xp.complex64
        )
        out = equalization.correct_polarization(
            x, equalization.JonesTrack(M, gp), normalize=False
        )
        xpt.assert_allclose(out, M2 @ x, rtol=1e-4, atol=1e-4)

    def test_signal_input_returns_signal(self, xp, xpt):
        """Signal input returns a Signal with the interpolated-matrix output."""
        N = 4096
        M2 = xp.asarray([[1.0, 0.5j], [0.2, -1.0]], dtype=xp.complex64)
        M = xp.broadcast_to(M2, (5, 2, 2)).copy()
        gp = xp.linspace(0.0, N - 1, 5)
        rng = xp.random.RandomState(2)
        data = (rng.standard_normal((2, N)) + 1j * rng.standard_normal((2, N))).astype(
            xp.complex64
        )
        sig = Signal(samples=data, sampling_rate=1.0, symbol_rate=1.0)

        track = equalization.JonesTrack(M, gp)
        out_sig = equalization.correct_polarization(sig, track)
        out_arr = equalization.correct_polarization(data, track)

        assert isinstance(out_sig, Signal)
        xpt.assert_allclose(out_sig.samples, out_arr)

    def test_channel_count_must_match(self, xp):
        x = xp.ones((3, 64), dtype=xp.complex64)
        with pytest.raises(ValueError, match="C = 2"):
            equalization.correct_polarization(x, xp.eye(2))
        with pytest.raises(ValueError, match="JonesTrack or a"):
            equalization.correct_polarization(x, xp.ones(3))
