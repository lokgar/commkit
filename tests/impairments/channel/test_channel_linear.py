"""Tests for linear fiber-channel impairments (CD, PMD, polarization mixing)."""

import numpy as np
import pytest

from commkit.core import Signal
from commkit.impairments import (
    apply_chromatic_dispersion,
    apply_pmd,
    apply_polarization_mixing,
)


class TestApplyPMD:
    """Tests for the apply_pmd function."""

    def test_identity_no_dgd_no_rotation(self, xp, xpt):
        """PMD with dgd=0, theta=0 should return the input unchanged."""
        N = 256
        rng = xp.random.RandomState(42)
        samples = (rng.randn(2, N) + 1j * rng.randn(2, N)).astype(xp.complex64)
        fs = 56e9

        out = apply_pmd(samples, dgd=0.0, theta=0.0, sampling_rate=fs)

        xpt.assert_allclose(out, samples, atol=1e-5)

    def test_energy_conservation(self, xp, xpt):
        """PMD is unitary - output power should equal input power."""
        N = 1024
        rng = xp.random.RandomState(7)
        samples = (rng.randn(2, N) + 1j * rng.randn(2, N)).astype(xp.complex64)
        fs = 56e9

        out = apply_pmd(samples, dgd=5e-12, theta=np.pi / 5, sampling_rate=fs)

        power_in = float(xp.sum(xp.abs(samples) ** 2))
        power_out = float(xp.sum(xp.abs(out) ** 2))

        xpt.assert_allclose(power_out, power_in, rtol=1e-4)

    def test_pure_rotation(self, xp, xpt):
        """With dgd=0, H(f)=R(+θ)·I·R(-θ)=I - output equals input regardless of theta."""
        N = 128
        theta = np.pi / 4
        fs = 56e9

        # Constant signal: X-pol = 1+0j, Y-pol = 0+0j
        samples = xp.zeros((2, N), dtype=xp.complex64)
        samples[0, :] = 1.0 + 0j

        out = apply_pmd(samples, dgd=0.0, theta=theta, sampling_rate=fs)

        # With correct Jones matrix H=R(+θ)·D(f)·R(-θ), DGD=0 => D(f)=I => H=I.
        # Output must equal input for any theta.
        xpt.assert_allclose(out, samples, atol=1e-5)

    def test_pure_rotation_with_dgd_zero_coupling(self, xp, xpt):
        """With theta=0, DGD applies a phase shift but no polarization coupling."""
        N = 256
        rng = xp.random.RandomState(99)
        samples = (rng.randn(2, N) + 1j * rng.randn(2, N)).astype(xp.complex64)
        fs = 56e9
        dgd = 5e-12

        out = apply_pmd(samples, dgd=dgd, theta=0.0, sampling_rate=fs)

        # With theta=0, R=I so H=diag(D0, D1) - only phase shift per pol
        # Output power should still be preserved
        power_in = float(xp.sum(xp.abs(samples) ** 2))
        power_out = float(xp.sum(xp.abs(out) ** 2))

        xpt.assert_allclose(power_out, power_in, rtol=1e-4)

    def test_output_shape(self, xp):
        """Output shape should match input shape."""
        samples = xp.ones((2, 512), dtype=xp.complex64)
        out = apply_pmd(samples, dgd=1e-12, theta=0.3, sampling_rate=56e9)

        assert out.shape == (2, 512)

    def test_output_dtype(self, xp):
        """Output dtype should match input dtype."""
        samples = xp.ones((2, 128), dtype=xp.complex64)
        out = apply_pmd(samples, dgd=1e-12, theta=0.3, sampling_rate=56e9)

        assert out.dtype == xp.complex64

    def test_rejects_siso(self, xp):
        """Should raise ValueError for 1D (SISO) input."""
        samples = xp.ones(100, dtype=xp.complex64)
        with pytest.raises(ValueError, match="dual-pol"):
            apply_pmd(samples, dgd=1e-12, sampling_rate=56e9)

    def test_rejects_wrong_channels(self, xp):
        """Should raise ValueError for non-2 first dimension."""
        samples = xp.ones((3, 100), dtype=xp.complex64)
        with pytest.raises(ValueError, match="dual-pol"):
            apply_pmd(samples, dgd=1e-12, sampling_rate=56e9)

    def test_nonzero_dgd_causes_distortion(self, xp):
        """Non-zero DGD with non-zero theta should change the signal."""
        N = 512
        rng = xp.random.RandomState(42)
        samples = (rng.randn(2, N) + 1j * rng.randn(2, N)).astype(xp.complex64)
        fs = 56e9

        out = apply_pmd(samples, dgd=10e-12, theta=np.pi / 4, sampling_rate=fs)

        # Signal should be different from input
        diff = float(xp.max(xp.abs(out - samples)))
        assert diff > 0.01, "PMD with DGD should modify the signal"

    def test_round_trip_nonzero_theta(self, xp, xpt):
        """Apply PMD then invert it: applying with -theta and negated DGD phase reverses CD.

        Simpler correctness check: apply PMD forward then backward (theta, -theta with
        complex conjugate of H) to verify the Jones matrix fix gives a round-trip near identity.
        """
        N = 512
        rng = xp.random.RandomState(13)
        samples = (rng.randn(2, N) + 1j * rng.randn(2, N)).astype(xp.complex64)
        fs = 56e9
        dgd = 5e-12
        theta = np.pi / 6

        # Forward: apply PMD (adds DGD at angle theta)
        distorted = apply_pmd(samples, dgd=dgd, theta=theta, sampling_rate=fs)

        # Inverse: apply PMD with same theta but negated DGD (conjugate filter)
        # H_inv(f) = H*(f)  ->  apply_pmd with dgd=-dgd
        recovered = apply_pmd(distorted, dgd=-dgd, theta=theta, sampling_rate=fs)

        xpt.assert_allclose(recovered, samples, atol=1e-4)


class TestApplyPolarizationMixing:
    """Tests for apply_polarization_mixing."""

    def test_static_identity(self, xp, xpt):
        """theta=0 should return the input unchanged."""
        N = 256
        rng = xp.random.RandomState(1)
        samples = (rng.randn(2, N) + 1j * rng.randn(2, N)).astype(xp.complex64)
        out = apply_polarization_mixing(samples, theta=0.0)
        xpt.assert_allclose(out, samples, atol=1e-5)

    def test_static_90deg_rotation(self, xp, xpt):
        """θ=π/2 swaps X<->Y with a sign flip: Ex' = -Ey, Ey' = Ex."""
        N = 128
        samples = xp.zeros((2, N), dtype=xp.complex64)
        samples[0, :] = 1.0 + 0j  # only X-pol
        out = apply_polarization_mixing(samples, theta=np.pi / 2)
        # cos(π/2)=0, sin(π/2)=1 -> Ex'=0-1*Ey=0, Ey'=1*Ex+0*Ey=1
        xpt.assert_allclose(out[0], xp.zeros(N, dtype=xp.complex64), atol=1e-5)
        xpt.assert_allclose(out[1], xp.ones(N, dtype=xp.complex64), atol=1e-5)

    def test_static_energy_preserved(self, xp, xpt):
        """Static rotation is unitary - total power preserved."""
        N = 512
        rng = xp.random.RandomState(7)
        samples = (rng.randn(2, N) + 1j * rng.randn(2, N)).astype(xp.complex64)
        out = apply_polarization_mixing(samples, theta=np.pi / 3)
        power_in = float(xp.sum(xp.abs(samples) ** 2))
        power_out = float(xp.sum(xp.abs(out) ** 2))
        xpt.assert_allclose(power_out, power_in, rtol=1e-4)

    def test_variable_theta_energy_preserved(self, xp, xpt):
        """Time-varying rotation must preserve energy at every sample."""
        N = 256
        rng = xp.random.RandomState(3)
        samples = (rng.randn(2, N) + 1j * rng.randn(2, N)).astype(xp.complex64)
        thetas = xp.linspace(0, 2 * np.pi, N)
        out = apply_polarization_mixing(samples, theta=thetas)
        # Per-sample energy is conserved: |Ex'[n]|² + |Ey'[n]|² = |Ex[n]|² + |Ey[n]|²
        power_in_per_sample = xp.abs(samples[0]) ** 2 + xp.abs(samples[1]) ** 2
        power_out_per_sample = xp.abs(out[0]) ** 2 + xp.abs(out[1]) ** 2
        xpt.assert_allclose(power_out_per_sample, power_in_per_sample, rtol=1e-4)

    def test_drift_rate_produces_varying_output(self, xp, xpt):
        """Non-zero drift_rate should produce time-varying output."""
        N = 256
        samples = xp.zeros((2, N), dtype=xp.complex64)
        samples[0, :] = 1.0 + 0j
        out = apply_polarization_mixing(
            samples, theta=0.0, drift_rate=1e4, sampling_rate=1e6
        )
        # theta = 1e-2 rad per sample: the last sample is rotated by 2.55 rad.
        xpt.assert_allclose(out[0, -1], xp.asarray(np.cos(2.55 + 0j)), atol=1e-5)
        sig = Signal(samples=samples, sampling_rate=1e6, symbol_rate=1e6)
        xpt.assert_allclose(
            apply_polarization_mixing(sig, theta=0.0, drift_rate=1e4).samples, out
        )
        with pytest.raises(ValueError, match="sampling_rate"):
            apply_polarization_mixing(samples, theta=0.0, drift_rate=1e4)
        with pytest.raises(ValueError, match="scalar theta"):
            apply_polarization_mixing(
                samples, theta=xp.zeros(N), drift_rate=1e4, sampling_rate=1e6
            )

    def test_rejects_siso(self, xp):
        """Should raise ValueError for 1D input."""
        with pytest.raises(ValueError, match="dual-pol"):
            apply_polarization_mixing(xp.ones(100, dtype=xp.complex64), theta=0.1)

    def test_rejects_wrong_channels(self, xp):
        """Should raise ValueError when first axis != 2."""
        with pytest.raises(ValueError, match="dual-pol"):
            apply_polarization_mixing(xp.ones((3, 100), dtype=xp.complex64), theta=0.1)

    def test_output_shape(self, xp):
        """Output shape should match input shape."""
        samples = xp.ones((2, 512), dtype=xp.complex64)
        out = apply_polarization_mixing(samples, theta=np.pi / 4)
        assert out.shape == (2, 512)


class TestApplyChomaticDispersion:
    """Tests for apply_chromatic_dispersion and round-trip with correct_chromatic_dispersion."""

    def test_round_trip_siso(self, xp, xpt):
        """Apply CD then compensate: output should be ~equal to input."""
        from commkit.filtering import correct_chromatic_dispersion

        N = 1024
        rng = xp.random.RandomState(42)
        samples = (rng.randn(N) + 1j * rng.randn(N)).astype(xp.complex64)
        fs = 64e9
        D, L, lam = 17.0, 80.0, 1550.0

        distorted = apply_chromatic_dispersion(
            samples,
            sampling_rate=fs,
            dispersion_ps_nm_km=D,
            fiber_length_km=L,
            center_wavelength_nm=lam,
        )
        recovered = correct_chromatic_dispersion(
            distorted,
            sampling_rate=fs,
            dispersion_ps_nm_km=D,
            fiber_length_km=L,
            center_wavelength_nm=lam,
        )

        xpt.assert_allclose(recovered, samples, atol=1e-3)

    def test_round_trip_mimo(self, xp, xpt):
        """Round-trip (MIMO): each channel should recover to input."""
        from commkit.filtering import correct_chromatic_dispersion

        C, N = 2, 512
        rng = xp.random.RandomState(7)
        samples = (rng.randn(C, N) + 1j * rng.randn(C, N)).astype(xp.complex64)
        fs = 64e9
        D, L, lam = 17.0, 80.0, 1550.0

        distorted = apply_chromatic_dispersion(
            samples,
            sampling_rate=D,
            dispersion_ps_nm_km=L,
            fiber_length_km=lam,
            center_wavelength_nm=fs,
        )
        recovered = correct_chromatic_dispersion(
            distorted,
            sampling_rate=D,
            dispersion_ps_nm_km=L,
            fiber_length_km=lam,
            center_wavelength_nm=fs,
        )

        xpt.assert_allclose(recovered, samples, atol=1e-3)

    def test_output_shape_siso(self, xp):
        """SISO output shape matches input."""
        samples = xp.ones(512, dtype=xp.complex64)
        out = apply_chromatic_dispersion(
            samples,
            sampling_rate=17.0,
            dispersion_ps_nm_km=80.0,
            fiber_length_km=1550.0,
            center_wavelength_nm=64e9,
        )
        assert out.shape == (512,)

    def test_output_shape_mimo(self, xp):
        """MIMO output shape matches input."""
        samples = xp.ones((2, 512), dtype=xp.complex64)
        out = apply_chromatic_dispersion(
            samples,
            sampling_rate=17.0,
            dispersion_ps_nm_km=80.0,
            fiber_length_km=1550.0,
            center_wavelength_nm=64e9,
        )
        assert out.shape == (2, 512)

    def test_modifies_signal(self, xp):
        """CD should change the signal (non-trivial dispersion)."""
        rng = xp.random.RandomState(99)
        samples = (rng.randn(512) + 1j * rng.randn(512)).astype(xp.complex64)
        out = apply_chromatic_dispersion(
            samples,
            sampling_rate=17.0,
            dispersion_ps_nm_km=80.0,
            fiber_length_km=1550.0,
            center_wavelength_nm=64e9,
        )
        diff = float(xp.max(xp.abs(out - samples)))
        assert diff > 1e-3

    def test_energy_preserved(self, xp, xpt):
        """CD is an all-pass filter: energy must be preserved."""
        rng = xp.random.RandomState(11)
        samples = (rng.randn(1024) + 1j * rng.randn(1024)).astype(xp.complex64)
        out = apply_chromatic_dispersion(
            samples,
            sampling_rate=17.0,
            dispersion_ps_nm_km=80.0,
            fiber_length_km=1550.0,
            center_wavelength_nm=64e9,
        )
        power_in = float(xp.sum(xp.abs(samples) ** 2))
        power_out = float(xp.sum(xp.abs(out) ** 2))
        xpt.assert_allclose(power_out, power_in, rtol=1e-4)

    def test_signal_input_returns_signal(self, xp, xpt):
        """Signal input: sampling_rate is taken from the signal."""
        fs = 64e9
        data = xp.ones(512, dtype=xp.complex64)
        sig = Signal(samples=data, sampling_rate=fs, symbol_rate=fs / 2)

        out_sig = apply_chromatic_dispersion(
            sig,
            dispersion_ps_nm_km=17.0,
            fiber_length_km=80.0,
            center_wavelength_nm=1550.0,
        )
        out_arr = apply_chromatic_dispersion(
            data,
            sampling_rate=fs,
            dispersion_ps_nm_km=17.0,
            fiber_length_km=80.0,
            center_wavelength_nm=1550.0,
        )

        assert isinstance(out_sig, Signal)
        xpt.assert_allclose(out_sig.samples, out_arr)
        xpt.assert_allclose(sig.samples, data)  # original untouched


class TestSignalInputPMDAndPolarizationMixing:
    """Signal-awareness for apply_pmd and apply_polarization_mixing."""

    def test_apply_pmd_signal_input(self, xp, xpt):
        """Signal input: sampling_rate is taken from the signal."""
        fs = 56e9
        rng = xp.random.RandomState(3)
        data = (rng.randn(2, 256) + 1j * rng.randn(2, 256)).astype(xp.complex64)
        sig = Signal(samples=data, sampling_rate=fs, symbol_rate=fs / 2)

        out_sig = apply_pmd(sig, dgd=5e-12, theta=np.pi / 5)
        out_arr = apply_pmd(data, sampling_rate=fs, dgd=5e-12, theta=np.pi / 5)

        assert isinstance(out_sig, Signal)
        xpt.assert_allclose(out_sig.samples, out_arr)

    def test_apply_polarization_mixing_signal_input(self, xp, xpt):
        """Signal input returns a Signal with the rotated samples."""
        rng = xp.random.RandomState(4)
        data = (rng.randn(2, 128) + 1j * rng.randn(2, 128)).astype(xp.complex64)
        sig = Signal(samples=data, sampling_rate=1.0, symbol_rate=1.0)

        out_sig = apply_polarization_mixing(sig, theta=0.6)
        out_arr = apply_polarization_mixing(data, theta=0.6)

        assert isinstance(out_sig, Signal)
        xpt.assert_allclose(out_sig.samples, out_arr)


class TestOpticalCarrier:
    """The fiber functions read the wavelength from an optical Signal."""

    CD = dict(dispersion_ps_nm_km=17.0, fiber_length_km=20.0)

    def _signal(self, xp, **carrier):
        rng = np.random.default_rng(0)
        x = (rng.standard_normal(512) + 1j * rng.standard_normal(512)).astype(
            np.complex64
        )
        return Signal(
            samples=xp.asarray(x), sampling_rate=64e9, symbol_rate=32e9, **carrier
        )

    def test_optical_signal_supplies_the_wavelength(self, xp, xpt):
        sig = self._signal(xp, domain="OPT", center_frequency=193.4e12)
        lam = 299_792_458.0 / 193.4e12 * 1e9
        assert sig.center_wavelength_nm == pytest.approx(lam)
        out = apply_chromatic_dispersion(sig, **self.CD)
        ref = apply_chromatic_dispersion(
            sig.samples, sampling_rate=64e9, center_wavelength_nm=lam, **self.CD
        )
        xpt.assert_allclose(out.samples, ref)
        # An agreeing value is accepted (0.01 %), a conflicting one raises.
        apply_chromatic_dispersion(sig, center_wavelength_nm=round(lam, 2), **self.CD)
        with pytest.raises(ValueError, match="conflicts with the optical"):
            apply_chromatic_dispersion(sig, center_wavelength_nm=1310.0, **self.CD)

    def test_other_signals_need_the_wavelength(self, xp):
        for sig in (
            self._signal(xp),
            self._signal(xp, domain="RF", center_frequency=10e9),
            self._signal(xp, domain="OPT"),  # no carrier
        ):
            assert sig.center_wavelength_nm is None
            with pytest.raises(ValueError, match="requires center_wavelength_nm"):
                apply_chromatic_dispersion(sig, **self.CD)
            apply_chromatic_dispersion(sig, center_wavelength_nm=1550.0, **self.CD)
