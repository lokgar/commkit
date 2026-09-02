"""Static and dynamic polarization-tone demultiplexing."""

import numpy as np
import pytest

from commkit import equalization
from commkit.core import Signal


@pytest.fixture(autouse=True)
def _enable_jax_x64():
    """Enable JAX x64 mode for all tests in this module.

    JAX RLS requires complex128 for P-matrix stability; LMS CPR requires float64
    for phase accumulation. Enabling x64 globally is safe - it only affects
    precision when 64-bit dtypes are explicitly requested.
    """
    try:
        import jax

        jax.config.update("jax_enable_x64", True)
    except ImportError:
        pass


jax = pytest.importorskip("jax", reason="JAX not installed")


class TestDemultiplexPolarizationTones:
    """Tests for equalization.demultiplex_polarization_tones_static."""

    # Data occupies |f| < 0.3*fs; tones live in the clean guard band beyond it,
    # mirroring a real pilot-tone placement so single-bin extraction is exact.
    TONES = [35.0, -42.0]  # Hz, both inside the guard for fs=100

    @staticmethod
    def _streams(xp, C=2, N=8192, seed=0, fs=100.0, band_frac=0.3):
        """C independent, band-limited, unit-power complex streams, shape (C, N)."""
        rng = xp.random.RandomState(seed)
        s = (rng.randn(C, N) + 1j * rng.randn(C, N)).astype(xp.complex64)
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

    def test_demux_recovers_streams_no_permutation(self, backend_device, xp):
        """A static rotation is inverted; row j tracks tone j (no permutation)."""
        from commkit.impairments import apply_polarization_mixing
        from commkit.spectral import add_pilot_tone

        fs = 100.0
        s = self._streams(xp)
        tx, f_used = add_pilot_tone(s, fs, self.TONES, power_ratio_db=-10.0)
        rx = apply_polarization_mixing(tx, theta=0.6)  # frequency-flat Jones mix

        demuxed = equalization.demultiplex_polarization_tones_static(rx, fs, f_used)

        assert demuxed.shape == tx.shape
        assert demuxed.dtype == rx.dtype
        for j in range(2):
            # Row j recovers the transmitted stream j (data+tone) up to a scale.
            assert self._corr(xp, demuxed[j], tx[j]) > 0.999
            # ...and rejects the other stream (cross-talk suppressed).
            assert self._corr(xp, demuxed[j], tx[1 - j]) < 0.05

    def test_demux_robust_to_noise(self, backend_device, xp):
        """The tone-phasor estimate still locks under moderate AWGN."""
        from commkit.impairments import apply_polarization_mixing
        from commkit.spectral import add_pilot_tone

        fs = 100.0
        s = self._streams(xp, N=8192)
        tx, f_used = add_pilot_tone(s, fs, self.TONES, power_ratio_db=-8.0)
        rx = apply_polarization_mixing(tx, theta=-0.9)
        rng = xp.random.RandomState(123)
        std = 0.1
        rx = rx + std * (rng.randn(*rx.shape) + 1j * rng.randn(*rx.shape)).astype(
            rx.dtype
        )

        demuxed = equalization.demultiplex_polarization_tones_static(rx, fs, f_used)
        for j in range(2):
            assert self._corr(xp, demuxed[j], tx[j]) > 0.99

    def test_return_matrix_shape(self, backend_device, xp):
        """return_matrix yields the (K, C) complex128 unmixing matrix."""
        from commkit.spectral import add_pilot_tone

        fs = 100.0
        s = self._streams(xp)
        tx, f_used = add_pilot_tone(s, fs, self.TONES, power_ratio_db=-10.0)
        demuxed, W = equalization.demultiplex_polarization_tones_static(
            tx, fs, f_used, return_matrix=True
        )
        assert W.shape == (2, 2)
        assert W.dtype == xp.complex128

    def test_overdetermined_pinv(self, backend_device, xp):
        """C > K: a (3, 2) mixing is unmixed to 2 streams via the pseudo-inverse."""
        from commkit.spectral import add_pilot_tone

        fs = 100.0
        s = self._streams(xp, C=2, N=8192)
        tx, f_used = add_pilot_tone(s, fs, self.TONES, power_ratio_db=-10.0)
        rng = xp.random.RandomState(7)
        J = (rng.randn(3, 2) + 1j * rng.randn(3, 2)).astype(xp.complex64)  # (3, 2)
        rx = (J @ tx.astype(xp.complex64)).astype(xp.complex64)  # (3, N)

        demuxed = equalization.demultiplex_polarization_tones_static(rx, fs, f_used)
        assert demuxed.shape == (2, tx.shape[-1])
        for j in range(2):
            assert self._corr(xp, demuxed[j], tx[j]) > 0.999

    def test_more_tones_than_channels_raises(self, backend_device, xp):
        """K > C is rejected (cannot unmix more streams than receive channels)."""
        fs = 100.0
        s = self._streams(xp, C=2)
        with pytest.raises(ValueError, match=r"need K <= C"):
            equalization.demultiplex_polarization_tones_static(
                s, fs, [10.0, 20.0, 30.0]
            )

    def test_requires_2d(self, backend_device, xp):
        """A 1-D (SISO) input is rejected - demux is inherently MIMO."""
        fs = 100.0
        s = self._streams(xp, C=1)[0]  # (N,)
        with pytest.raises(ValueError, match=r"2-D \(C, N\)"):
            equalization.demultiplex_polarization_tones_static(s, fs, [10.0])

    def test_signal_input_returns_signal(self, backend_device, xp, xpt):
        """Signal input: sampling_rate is taken from the signal."""
        from commkit.impairments import apply_polarization_mixing
        from commkit.spectral import add_pilot_tone

        fs = 100.0
        s = self._streams(xp)
        tx, f_used = add_pilot_tone(s, fs, self.TONES, power_ratio_db=-10.0)
        rx = apply_polarization_mixing(tx, theta=0.6)
        sig = Signal(samples=rx, sampling_rate=fs, symbol_rate=fs)

        demuxed_sig = equalization.demultiplex_polarization_tones_static(
            sig, tone_frequencies=f_used
        )
        demuxed_arr = equalization.demultiplex_polarization_tones_static(rx, fs, f_used)

        assert isinstance(demuxed_sig, Signal)
        xpt.assert_allclose(demuxed_sig.samples, demuxed_arr)

    def test_signal_input_with_return_matrix(self, backend_device, xp, xpt):
        """Signal input + return_matrix=True: (Signal, W) tuple."""
        from commkit.spectral import add_pilot_tone

        fs = 100.0
        s = self._streams(xp)
        tx, f_used = add_pilot_tone(s, fs, self.TONES, power_ratio_db=-10.0)
        sig = Signal(samples=tx, sampling_rate=fs, symbol_rate=fs)

        demuxed_sig, W_sig = equalization.demultiplex_polarization_tones_static(
            sig, tone_frequencies=f_used, return_matrix=True
        )
        demuxed_arr, W_arr = equalization.demultiplex_polarization_tones_static(
            tx, fs, f_used, return_matrix=True
        )

        assert isinstance(demuxed_sig, Signal)
        xpt.assert_allclose(demuxed_sig.samples, demuxed_arr)
        xpt.assert_allclose(W_sig, W_arr)


class TestDemultiplexPolarizationTonesDynamic:
    """Tests for equalization.demultiplex_polarization_tones_dynamic (drifting SOP)."""

    TONES = TestDemultiplexPolarizationTones.TONES
    _streams = staticmethod(TestDemultiplexPolarizationTones._streams)
    _corr = staticmethod(TestDemultiplexPolarizationTones._corr)

    def test_tracks_drifting_sop_where_static_fails(self, backend_device, xp):
        """A rotating SOP defeats the static one-shot inverse but is tracked here."""
        from commkit.impairments import apply_polarization_mixing
        from commkit.spectral import add_pilot_tone

        fs = 100.0
        N = 16384
        s = self._streams(xp, N=N)
        tx, f_used = add_pilot_tone(s, fs, self.TONES, power_ratio_db=-8.0)
        # SOP sweeps a full ~pi/2 across the capture - the static average Jones
        # matrix matches no instant, so its cross-talk rejection collapses.
        drift = (np.pi / 2) / N
        rx = apply_polarization_mixing(tx, theta=0.2, drift_rate_rad_per_sym=drift)

        static = equalization.demultiplex_polarization_tones_static(rx, fs, f_used)
        dynamic = equalization.demultiplex_polarization_tones_dynamic(
            rx, fs, f_used, track_bandwidth=2.0
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

    def test_static_sop_still_recovered(self, backend_device, xp):
        """With no drift the dynamic path matches the static result (no regression)."""
        from commkit.impairments import apply_polarization_mixing
        from commkit.spectral import add_pilot_tone

        fs = 100.0
        s = self._streams(xp, N=8192)
        tx, f_used = add_pilot_tone(s, fs, self.TONES, power_ratio_db=-10.0)
        rx = apply_polarization_mixing(tx, theta=0.6)

        demuxed = equalization.demultiplex_polarization_tones_dynamic(
            rx, fs, f_used, track_bandwidth=2.0
        )
        for j in range(2):
            assert self._corr(xp, demuxed[j], tx[j]) > 0.99
            assert self._corr(xp, demuxed[j], tx[1 - j]) < 0.1

    def test_return_matrix_shapes(self, backend_device, xp):
        """return_matrix yields the (G, K, C) stack and (G,) grid positions."""
        from commkit.spectral import add_pilot_tone

        fs = 100.0
        N = 8192
        s = self._streams(xp, N=N)
        tx, f_used = add_pilot_tone(s, fs, self.TONES, power_ratio_db=-10.0)
        demuxed, Wg, grid = equalization.demultiplex_polarization_tones_dynamic(
            tx, fs, f_used, track_bandwidth=2.0, return_matrix=True
        )
        assert Wg.ndim == 3 and Wg.shape[1:] == (2, 2)
        assert Wg.shape[0] == grid.shape[0]
        assert Wg.dtype == xp.complex128
        assert int(grid[-1]) == N - 1  # last sample is pinned on the grid

    def test_invalid_track_bandwidth_raises(self, backend_device, xp):
        """A non-positive tracking bandwidth is rejected."""
        fs = 100.0
        s = self._streams(xp, N=4096)
        with pytest.raises(ValueError, match=r"track_bandwidth must be positive"):
            equalization.demultiplex_polarization_tones_dynamic(
                s, fs, self.TONES, track_bandwidth=0.0
            )

    def test_trim_edges_returns_valid_interior(self, backend_device, xp):
        """trim_edges drops num_taps//2 each side and reports the original range."""
        from commkit.spectral import add_pilot_tone

        fs = 100.0
        N = 8192
        num_taps = 165
        s = self._streams(xp, N=N)
        tx, f_used = add_pilot_tone(s, fs, self.TONES, power_ratio_db=-10.0)
        demuxed, valid = equalization.demultiplex_polarization_tones_dynamic(
            tx, fs, f_used, track_bandwidth=2.0, num_taps=num_taps, trim_edges=True
        )
        g = num_taps // 2
        assert isinstance(valid, slice)
        assert (valid.start, valid.stop) == (g, N - g)
        assert demuxed.shape == (2, N - 2 * g)
        # The retained interior must align with the input over `valid`.
        for j in range(2):
            assert self._corr(xp, demuxed[j], tx[j, valid]) > 0.99

    def test_trim_edges_with_return_matrix_order(self, backend_device, xp):
        """Combined flags return (demuxed, valid, W_grid, grid) in that order."""
        from commkit.spectral import add_pilot_tone

        fs = 100.0
        N = 8192
        s = self._streams(xp, N=N)
        tx, f_used = add_pilot_tone(s, fs, self.TONES, power_ratio_db=-10.0)
        result = equalization.demultiplex_polarization_tones_dynamic(
            tx, fs, f_used, track_bandwidth=2.0, trim_edges=True, return_matrix=True
        )
        demuxed, valid, Wg, grid = result
        assert isinstance(valid, slice)
        assert demuxed.shape[-1] == valid.stop - valid.start
        # W_grid / grid still span the full record.
        assert int(grid[-1]) == N - 1
        assert Wg.shape[1:] == (2, 2)

    def test_signal_input_returns_signal(self, backend_device, xp, xpt):
        """Signal input: sampling_rate is taken from the signal."""
        from commkit.impairments import apply_polarization_mixing
        from commkit.spectral import add_pilot_tone

        fs = 100.0
        s = self._streams(xp, N=8192)
        tx, f_used = add_pilot_tone(s, fs, self.TONES, power_ratio_db=-10.0)
        rx = apply_polarization_mixing(tx, theta=0.6)
        sig = Signal(samples=rx, sampling_rate=fs, symbol_rate=fs)

        demuxed_sig = equalization.demultiplex_polarization_tones_dynamic(
            sig, tone_frequencies=f_used, track_bandwidth=2.0
        )
        demuxed_arr = equalization.demultiplex_polarization_tones_dynamic(
            rx, fs, f_used, track_bandwidth=2.0
        )

        assert isinstance(demuxed_sig, Signal)
        xpt.assert_allclose(demuxed_sig.samples, demuxed_arr)

    def test_signal_input_apply_false_returns_raw_tuple(self, backend_device, xp, xpt):
        """apply=False: no signal-shaped output, so the tuple stays raw arrays."""
        from commkit.spectral import add_pilot_tone

        fs = 100.0
        s = self._streams(xp, N=8192)
        tx, f_used = add_pilot_tone(s, fs, self.TONES, power_ratio_db=-10.0)
        sig = Signal(samples=tx, sampling_rate=fs, symbol_rate=fs)

        Wg_sig, grid_sig = equalization.demultiplex_polarization_tones_dynamic(
            sig, tone_frequencies=f_used, track_bandwidth=2.0, apply=False
        )
        Wg_arr, grid_arr = equalization.demultiplex_polarization_tones_dynamic(
            tx, fs, f_used, track_bandwidth=2.0, apply=False
        )

        assert not isinstance(Wg_sig, Signal)
        xpt.assert_allclose(Wg_sig, Wg_arr)
        xpt.assert_allclose(grid_sig, grid_arr)

    def test_signal_input_trim_edges_with_return_matrix(self, backend_device, xp, xpt):
        """Signal input + trim_edges + return_matrix: (Signal, valid, W_grid, grid)."""
        from commkit.spectral import add_pilot_tone

        fs = 100.0
        N = 8192
        s = self._streams(xp, N=N)
        tx, f_used = add_pilot_tone(s, fs, self.TONES, power_ratio_db=-10.0)
        sig = Signal(samples=tx, sampling_rate=fs, symbol_rate=fs)

        demuxed_sig, valid, Wg, grid = (
            equalization.demultiplex_polarization_tones_dynamic(
                sig,
                tone_frequencies=f_used,
                track_bandwidth=2.0,
                trim_edges=True,
                return_matrix=True,
            )
        )

        assert isinstance(demuxed_sig, Signal)
        assert isinstance(valid, slice)
        assert demuxed_sig.samples.shape[-1] == valid.stop - valid.start
        assert Wg.shape[1:] == (2, 2)


class TestApplyInterpolatedMatrix:
    """Generic grid-interpolated time-varying matrix apply (the demux apply core)."""

    def test_matches_per_sample_reference(self, backend_device, xp, xpt):
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
        out = equalization.apply_interpolated_matrix(x, M, gp)
        assert out.shape == (2, N)
        xpt.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)

    def test_constant_grid_is_static_matmul(self, backend_device, xp, xpt):
        """A grid of identical matrices applies that one matrix everywhere."""
        N = 10000
        M2 = xp.asarray([[1.0, 0.5j], [0.2, -1.0]], dtype=xp.complex64)
        M = xp.broadcast_to(M2, (5, 2, 2)).copy()
        gp = xp.linspace(0.0, N - 1, 5)
        rng = xp.random.RandomState(1)
        x = (rng.standard_normal((2, N)) + 1j * rng.standard_normal((2, N))).astype(
            xp.complex64
        )
        out = equalization.apply_interpolated_matrix(x, M, gp)
        xpt.assert_allclose(out, M2 @ x, rtol=1e-4, atol=1e-4)

    def test_signal_input_returns_signal(self, backend_device, xp, xpt):
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

        out_sig = equalization.apply_interpolated_matrix(sig, M, gp)
        out_arr = equalization.apply_interpolated_matrix(data, M, gp)

        assert isinstance(out_sig, Signal)
        xpt.assert_allclose(out_sig.samples, out_arr)
