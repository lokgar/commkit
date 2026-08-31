"""Tests for source impairments (laser linewidth phase noise)."""

import math

import numpy as np

from commkit.backend import to_device
from commkit.impairments import apply_phase_noise, generate_phase_noise


class TestApplyPhaseNoise:
    """Tests for apply_phase_noise."""

    def test_siso_output_shape(self, backend_device, xp):
        """SISO: output shape matches input."""
        samples = xp.ones(1024, dtype=xp.complex64)
        out = apply_phase_noise(samples, linewidth=100e3, sampling_rate=64e9, seed=1)
        assert out.shape == (1024,)

    def test_mimo_output_shape(self, backend_device, xp):
        """MIMO (C, N): output shape matches input."""
        samples = xp.ones((4, 512), dtype=xp.complex64)
        out = apply_phase_noise(samples, linewidth=100e3, sampling_rate=64e9, seed=2)
        assert out.shape == (4, 512)

    def test_modifies_signal(self, backend_device, xp):
        """Phase noise should change the signal."""
        samples = xp.ones(1024, dtype=xp.complex64)
        out = apply_phase_noise(samples, linewidth=1e6, sampling_rate=64e9, seed=3)
        diff = float(xp.max(xp.abs(out - samples)))
        assert diff > 1e-4

    def test_preserves_amplitude(self, backend_device, xp):
        """Phase rotation must not change sample amplitude."""
        rng = xp.random.RandomState(42)
        samples = (rng.randn(2048) + 1j * rng.randn(2048)).astype(xp.complex64)
        out = apply_phase_noise(samples, linewidth=100e3, sampling_rate=64e9, seed=4)
        amp_in = xp.abs(samples)
        amp_out = xp.abs(out)
        assert float(xp.max(xp.abs(amp_out - amp_in))) < 1e-4

    def test_phase_variance_matches_linewidth(self, backend_device, xp):
        """Incremental phase variance per sample should equal 2π·Δν/fs.

        Extracts phase increments from a long single-channel trajectory and
        measures their variance.  A flat-amplitude input of ones means
        angle(out[n]) = cumulative phase, so diff(angle(out)) = increments.
        """
        import math

        N = 200000
        linewidth = 100e3
        fs = 64e9
        expected_variance = 2.0 * math.pi * linewidth / fs

        # Unit-amplitude input so abs(out)=1 and angle(out) = cumulative phase
        samples = xp.ones(N, dtype=xp.complex128)
        out = apply_phase_noise(samples, linewidth=linewidth, sampling_rate=fs, seed=42)

        cumphase = xp.angle(out)  # (N,) wrapped, but increments are tiny
        increments = xp.diff(cumphase)  # (N-1,)

        # Small increments so wrapping is not an issue
        measured_var = float(xp.var(increments))
        assert abs(measured_var - expected_variance) / expected_variance < 0.05

    def test_shared_lo_all_channels_equal(self, backend_device, xp):
        """shared_lo=True: all channels receive identical phase trajectory."""
        C, N = 4, 512
        samples = xp.ones((C, N), dtype=xp.complex128)
        out = apply_phase_noise(
            samples, linewidth=1e6, sampling_rate=64e9, seed=7, shared_lo=True
        )
        # All channels should have identical output since same phase is applied
        for c in range(1, C):
            assert float(xp.max(xp.abs(out[c] - out[0]))) < 1e-10

    def test_independent_lo_channels_differ(self, backend_device, xp):
        """shared_lo=False (default): channels have independent phase trajectories."""
        C, N = 2, 512
        samples = xp.ones((C, N), dtype=xp.complex128)
        out = apply_phase_noise(
            samples, linewidth=10e6, sampling_rate=64e9, seed=5, shared_lo=False
        )
        # Independent trajectories should differ
        diff = float(xp.max(xp.abs(out[0] - out[1])))
        assert diff > 1e-4

    def test_flicker_modifies_signal_preserves_amplitude(self, backend_device, xp):
        """Flicker-only phase noise rotates without changing amplitude."""
        samples = xp.ones(4096, dtype=xp.complex128)
        out = apply_phase_noise(
            samples, linewidth=0.0, flicker=1e9, sampling_rate=500e6, seed=6
        )
        assert float(xp.max(xp.abs(xp.abs(out) - 1.0))) < 1e-9
        assert float(xp.max(xp.abs(out - samples))) > 1e-4


class TestGeneratePhaseNoise:
    """Tests for generate_phase_noise."""

    def test_shapes_and_dtype(self, backend_device, xp):
        """SISO (N,) for one stream, (C, N) for several; float64 on device."""
        phi = generate_phase_noise(256, 64e9, linewidth=1e6, seed=1)
        assert isinstance(phi, xp.ndarray)
        assert phi.shape == (256,)
        assert phi.dtype == xp.float64
        phi2 = generate_phase_noise(256, 64e9, linewidth=1e6, num_streams=3, seed=1)
        assert phi2.shape == (3, 256)

    def test_seed_reproducible(self, backend_device, xp):
        """Same seed yields the identical trajectory."""
        a = generate_phase_noise(1024, 64e9, linewidth=1e6, flicker=1e9, seed=42)
        b = generate_phase_noise(1024, 64e9, linewidth=1e6, flicker=1e9, seed=42)
        assert float(xp.max(xp.abs(a - b))) == 0.0

    def test_wiener_increment_variance(self, backend_device, xp):
        """White-FM increments have variance 2π·Δν/fs."""
        fs, dnu = 64e9, 100e3
        phi = generate_phase_noise(200_000, fs, linewidth=dnu, seed=7)
        expected = 2.0 * math.pi * dnu / fs
        measured = float(xp.var(xp.diff(phi)))
        assert abs(measured - expected) / expected < 0.05

    def test_flicker_fm_psd_level(self, backend_device, xp):
        """Flicker-only FM PSD follows S_f(f) = h_-1 / f in the shaped band."""
        from scipy import signal as sp_sig

        fs, h_m1 = 500e6, 4e9
        phi = generate_phase_noise(1 << 20, fs, flicker=h_m1, seed=9)
        df = np.diff(to_device(phi, "cpu")) * fs / (2.0 * np.pi)
        f, s_f = sp_sig.welch(df, fs=fs, nperseg=1 << 15)
        band = (f > 1e5) & (f < 1e7)
        # S_f·f is flat at h_-1 for a 1/f PSD; median over two decades.
        assert abs(np.median(s_f[band] * f[band]) - h_m1) / h_m1 < 0.2

    def test_matches_apply_phase_noise_trajectory(self, backend_device, xp):
        """apply_phase_noise(ones) reproduces the generate_phase_noise walk."""
        fs, dnu, n = 64e9, 100e3, 8192
        phi = generate_phase_noise(n, fs, linewidth=dnu, seed=11)
        out = apply_phase_noise(
            xp.ones(n, dtype=xp.complex128), sampling_rate=fs, linewidth=dnu, seed=11
        )
        # Wrapped comparison: increments are tiny, so unwrap-free diff works.
        err = xp.abs(xp.exp(1j * phi) - out)
        assert float(xp.max(err)) < 1e-9
