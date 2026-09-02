"""Pilot-aided and pilot-tone carrier phase / frequency recovery."""

import numpy as np
import pytest

from commkit import generate_qam, recovery, spectral
from commkit.backend import to_device
from commkit.core import Signal
from commkit.impairments import apply_awgn

FS = 1e6  # 1 MHz sampling rate, common to all tests


SNR_DB = 30  # generous SNR so numerical algorithms converge reliably


def _apply_phase_ramp(xp, samples, phase_per_sample):
    """Apply a linear phase ramp: samples[n] *= exp(j * n * phase_per_sample)."""
    n = xp.arange(samples.shape[-1], dtype=xp.float64)
    ramp = xp.exp(1j * n * phase_per_sample).astype(samples.dtype)
    return samples * ramp


def _rms_phase_error(xp, phase_est, phase_true):
    """RMS of the phase error, after removing any constant offset (M-fold ambiguity)."""
    err = phase_est - phase_true
    # Remove the mean bias (accounts for the irreducible global ambiguity)
    err = err - float(xp.mean(err))
    return float(xp.sqrt(xp.mean(err**2)))


class TestCprPilots:
    def _pilot_setup(self, xp, n_symbols=512, pilot_period=16, phase_per_sym=0.001):
        """Return (noisy+rotated samples, pilot_indices, pilot_values, true_phase)."""
        sig = generate_qam(order=16, num_symbols=n_symbols, sps=1, symbol_rate=FS)
        # Save ideal symbols before adding noise
        ideal_symbols = xp.asarray(sig.samples.copy())
        sig.samples = apply_awgn(sig.samples, esn0_db=SNR_DB, sps=1)
        # Apply a slow linear phase ramp on top of noise
        sig.samples = _apply_phase_ramp(xp, sig.samples, phase_per_sym)

        pilot_indices = np.arange(0, n_symbols, pilot_period)
        # Known pilot values = noiseless, unrotated symbols at pilot positions
        pilot_values = ideal_symbols[pilot_indices]
        true_phase = phase_per_sym * xp.arange(n_symbols, dtype=xp.float64)

        return sig.samples, pilot_indices, pilot_values, true_phase

    @pytest.mark.parametrize("pilot_period", [8, 16, 32])
    def test_phase_residual(self, backend_device, xp, pilot_period):
        """Pilot CPR: RMS residual < 0.05 rad for a linear phase ramp."""
        samples, pilot_indices, pilot_values, true_phase = self._pilot_setup(
            xp, n_symbols=512, pilot_period=pilot_period, phase_per_sym=0.001
        )
        phase_est = recovery.recover_carrier_phase_pilot_symbols(
            samples, pilot_indices=pilot_indices, pilot_values=pilot_values
        )
        err = _rms_phase_error(xp, phase_est, true_phase)
        assert err < 0.05

    def test_output_shape_siso(self, backend_device, xp):
        """Pilot CPR: 1D input -> 1D phase output."""
        samples, pilot_indices, pilot_values, _ = self._pilot_setup(xp)
        phase = recovery.recover_carrier_phase_pilot_symbols(
            samples, pilot_indices=pilot_indices, pilot_values=pilot_values
        )
        assert phase.shape == samples.shape

    def test_output_shape_mimo(self, backend_device, xp):
        """Pilot CPR: 2D input (C, N) -> 2D phase output (C, N)."""
        samples_a, pilot_indices, pilot_values, _ = self._pilot_setup(xp, n_symbols=256)
        samples_b, _, _, _ = self._pilot_setup(xp, n_symbols=256)
        mimo = xp.stack([samples_a, samples_b])  # (2, 256)
        phase = recovery.recover_carrier_phase_pilot_symbols(
            mimo, pilot_indices=pilot_indices, pilot_values=pilot_values
        )
        assert phase.shape == mimo.shape

    def test_large_phase_unwrap(self, backend_device, xp):
        """Pilot CPR: unwrapping handles cumulative phase > 2π correctly."""
        n_symbols = 512
        # Phase ramp of 4π total (spans two full cycles)
        phase_per_sym = 4 * np.pi / n_symbols
        samples, pilot_indices, pilot_values, true_phase = self._pilot_setup(
            xp, n_symbols=n_symbols, pilot_period=8, phase_per_sym=phase_per_sym
        )
        phase_est = recovery.recover_carrier_phase_pilot_symbols(
            samples, pilot_indices=pilot_indices, pilot_values=pilot_values
        )
        err = _rms_phase_error(xp, phase_est, true_phase)
        assert err < 0.1  # relaxed tolerance for large phase

    def test_cubic_interpolation(self, backend_device, xp):
        """Cubic interpolation works on both CPU and GPU."""
        samples, pilot_indices, pilot_values, _ = self._pilot_setup(xp)
        phase = recovery.recover_carrier_phase_pilot_symbols(
            samples,
            pilot_indices=pilot_indices,
            pilot_values=pilot_values,
            interpolation="cubic",
        )
        assert phase.shape == samples.shape

    def test_invalid_interpolation_raises(self, backend_device, xp):
        """Unknown interpolation method raises ValueError."""
        samples, pilot_indices, pilot_values, _ = self._pilot_setup(xp)
        with pytest.raises(ValueError, match="Unknown interpolation method"):
            recovery.recover_carrier_phase_pilot_symbols(
                samples,
                pilot_indices=pilot_indices,
                pilot_values=pilot_values,
                interpolation="spline_42",
            )


class TestCprPilotTone:
    """Pilot-tone CPR on an oversampled RRC waveform with a guard-band tone."""

    SPS = 8
    BETA = 0.1  # RRC roll-off -> data edge at (1+β)/2 · Rs
    F_TONE = 2.0e6  # guard band: > (1.1/2)·Rs (=0.55 MHz) + B, and < fs/2 (=4 MHz)
    BW = 0.3e6  # extraction half-width B
    EDGE = 600  # trim FFT-window edge ringing before comparing

    def _setup(
        self,
        xp,
        n_symbols=2000,
        df=0.0,
        linewidth=0.0,
        psr_db=-12.0,
        num_streams=1,
        seed=7,
    ):
        """Return (samples, fs, true_common_phase) for a tone-bearing waveform.

        The common phase = frequency-offset ramp + Wiener phase noise is applied
        *after* the tone, so the tone carries exactly the phase to be recovered.
        """
        fs = self.SPS * FS  # symbol rate = FS
        sig = generate_qam(
            order=16,
            num_symbols=n_symbols,
            sps=self.SPS,
            symbol_rate=FS,
            rrc_rolloff=self.BETA,
            num_streams=num_streams,
            seed=seed,
        )
        samples = xp.asarray(sig.samples)
        samples, _ = spectral.add_pilot_tone(
            samples, fs, self.F_TONE, power_ratio_db=psr_db
        )

        N = samples.shape[-1]
        n = xp.arange(N, dtype=xp.float64)
        if linewidth > 0.0:
            rng = xp.random.RandomState(seed)
            incr = rng.normal(0.0, float(np.sqrt(2 * np.pi * linewidth / fs)), N)
            pn = xp.cumsum(incr)
        else:
            pn = xp.zeros(N, dtype=xp.float64)
        common = 2 * np.pi * df * n / fs + pn  # (N,) float64
        samples = samples * xp.exp(1j * common).astype(samples.dtype)
        return samples, fs, common

    def _interior_rms(self, xp, phase_est, common):
        g = slice(self.EDGE, -self.EDGE)
        return _rms_phase_error(xp, phase_est[..., g], common[..., g])

    @pytest.mark.parametrize("df", [0.0, 0.05e6, 0.1e6])
    def test_phase_residual_foe(self, backend_device, xp, df):
        """Recovers a frequency-offset ramp (< B) to < 0.1 rad RMS."""
        samples, fs, common = self._setup(xp, df=df)
        theta = recovery.recover_carrier_phase_pilot_tone(
            samples, fs, self.F_TONE, bandwidth=self.BW
        )
        assert self._interior_rms(xp, theta, common) < 0.1

    def test_phase_residual_foe_and_phase_noise(self, backend_device, xp):
        """Recovers joint frequency offset + Wiener phase noise."""
        samples, fs, common = self._setup(xp, df=0.05e6, linewidth=5e3)
        theta = recovery.recover_carrier_phase_pilot_tone(
            samples, fs, self.F_TONE, bandwidth=self.BW
        )
        assert self._interior_rms(xp, theta, common) < 0.15

    @pytest.mark.parametrize(
        "window", ["tukey", ("tukey", 0.3), "boxcar", ("gaussian", 250), "hann"]
    )
    def test_window_options(self, backend_device, xp, window):
        """Any scipy.get_window spec tracks the common phase to < 0.12 rad RMS."""
        samples, fs, common = self._setup(xp, df=0.05e6)
        theta = recovery.recover_carrier_phase_pilot_tone(
            samples, fs, self.F_TONE, bandwidth=self.BW, window=window
        )
        assert self._interior_rms(xp, theta, common) < 0.12

    def test_output_shape_siso(self, backend_device, xp):
        """1D input -> 1D phase output of matching length."""
        samples, fs, _ = self._setup(xp)
        theta = recovery.recover_carrier_phase_pilot_tone(
            samples, fs, self.F_TONE, bandwidth=self.BW
        )
        assert theta.shape == samples.shape

    def test_output_shape_mimo(self, backend_device, xp):
        """2D input (C, N) -> 2D phase output (C, N)."""
        samples, fs, _ = self._setup(xp, num_streams=2)
        theta = recovery.recover_carrier_phase_pilot_tone(
            samples, fs, self.F_TONE, bandwidth=self.BW
        )
        assert theta.shape == samples.shape

    def test_joint_rows_identical(self, backend_device, xp):
        """joint_channels broadcasts a single trajectory to all rows."""
        samples, fs, _ = self._setup(xp, num_streams=2, df=0.05e6)
        theta = recovery.recover_carrier_phase_pilot_tone(
            samples, fs, self.F_TONE, bandwidth=self.BW, joint_channels=True
        )
        assert bool(xp.allclose(theta[0], theta[1]))

    def test_remove_frequency_offset_false_leaves_pure_pn(self, backend_device, xp):
        """With FOE removal off, the estimate matches the detrended common phase."""
        samples, fs, common = self._setup(xp, df=0.08e6, linewidth=5e3)
        theta = recovery.recover_carrier_phase_pilot_tone(
            samples,
            fs,
            self.F_TONE,
            bandwidth=self.BW,
            remove_frequency_offset=False,
        )
        N = samples.shape[-1]
        n = np.arange(N, dtype=np.float64)
        common_np = np.asarray(to_device(common, "cpu"))
        common_detrended = common_np - np.polyval(np.polyfit(n, common_np, 1), n)
        theta_np = np.asarray(to_device(theta, "cpu"))
        g = slice(self.EDGE, -self.EDGE)
        err = theta_np[g] - common_detrended[g]
        err -= err.mean()
        assert float(np.sqrt(np.mean(err**2))) < 0.15

    def test_refine_relocates_large_offset(self, backend_device, xp):
        """An offset > B is tracked when the search band covers the shifted tone."""
        # df = 0.6 MHz exceeds B = 0.3 MHz: a window centred at nominal would miss
        # the tone, but refine + a wide search band relocates it.
        samples, fs, common = self._setup(xp, df=0.6e6)
        theta = recovery.recover_carrier_phase_pilot_tone(
            samples, fs, self.F_TONE, bandwidth=self.BW, search_band=1.0e6
        )
        assert self._interior_rms(xp, theta, common) < 0.1

    def test_refine_off_fails_when_tone_leaves_window(self, backend_device, xp):
        """Without refinement, an offset > B drags the tone out of the window."""
        samples, fs, common = self._setup(xp, df=0.6e6)
        theta = recovery.recover_carrier_phase_pilot_tone(
            samples, fs, self.F_TONE, bandwidth=self.BW, refine_tone=False
        )
        # The tone is outside the nominal window -> estimate is garbage, not tracking.
        assert self._interior_rms(xp, theta, common) > 1.0

    def test_invalid_window_raises(self, backend_device, xp):
        samples, fs, _ = self._setup(xp, n_symbols=256)
        with pytest.raises(ValueError, match="Invalid window"):
            recovery.recover_carrier_phase_pilot_tone(
                samples, fs, self.F_TONE, bandwidth=self.BW, window="brick"
            )

    def test_invalid_bandwidth_raises(self, backend_device, xp):
        samples, fs, _ = self._setup(xp, n_symbols=256)
        with pytest.raises(ValueError, match="bandwidth must be > 0"):
            recovery.recover_carrier_phase_pilot_tone(
                samples, fs, self.F_TONE, bandwidth=0.0
            )

    def test_tone_frequency_out_of_range_raises(self, backend_device, xp):
        samples, fs, _ = self._setup(xp, n_symbols=256)
        with pytest.raises(ValueError, match=r"must lie in \(-fs/2, fs/2\)"):
            recovery.recover_carrier_phase_pilot_tone(
                samples, fs, fs, bandwidth=self.BW
            )


class TestCprPilotTones:
    """Multi-pilot common-phase CPR: SNR-weighted MRC + slow inter-tone tracking."""

    SPS = 8
    BETA = 0.1
    F0 = 1.5e6  # tone 0 - guard band, below fs/2 = 4 MHz
    F1 = 2.0e6  # tone 1 - distinct frequency, 0.5 MHz away
    BW = 0.3e6  # per-tone extraction half-width
    DIFF_BW = 5e3  # slow-δ low-pass cut-off - narrow, so δ averages out the noise
    EDGE = 600

    def _setup(
        self,
        xp,
        df=0.0,
        linewidth=0.0,
        snr_db=SNR_DB,
        n_symbols=2000,
        psr_db=(-12.0, -12.0),
        delta=0.4,
        seed=7,
    ):
        """Two isolated tones (one per channel) sharing a common carrier phase.

        Models the post-demux case: channel 0 carries tone ``F0``, channel 1
        carries tone ``F1``; both ride the *same* common phase (FOE ramp + Wiener
        phase noise).  ``delta`` is a static inter-tone offset applied to tone 1
        (the differential the slow-δ tracker must absorb).  Returns
        ``(samples, fs, common)``.
        """
        fs = self.SPS * FS
        sig = generate_qam(
            order=16,
            num_symbols=n_symbols,
            sps=self.SPS,
            symbol_rate=FS,
            rrc_rolloff=self.BETA,
            num_streams=2,
            seed=seed,
        )
        samples = apply_awgn(
            xp.asarray(sig.samples), esn0_db=snr_db, sps=self.SPS, seed=seed
        )
        # One tone per channel (channel c gets frequency[c]); tone 1 gets a static
        # phase offset δ so the inter-tone differential is non-trivial.
        samples, _ = spectral.add_pilot_tone(
            samples,
            fs,
            [self.F0, self.F1],
            power_ratio_db=list(psr_db),
            phase_init=0.0,
        )
        n = xp.arange(samples.shape[-1], dtype=xp.float64)
        samples[1] = samples[1] * xp.exp(1j * float(delta)).astype(samples.dtype)

        if linewidth > 0.0:
            rng = xp.random.RandomState(seed)
            incr = rng.normal(
                0.0, float(np.sqrt(2 * np.pi * linewidth / fs)), samples.shape[-1]
            )
            pn = xp.cumsum(incr)
        else:
            pn = xp.zeros(samples.shape[-1], dtype=xp.float64)
        common = 2 * np.pi * df * n / fs + pn
        samples = samples * xp.exp(1j * common).astype(samples.dtype)
        return samples, fs, common

    def _interior_rms(self, xp, phase_est, common):
        g = slice(self.EDGE, -self.EDGE)
        return _rms_phase_error(xp, phase_est[..., g], common[..., g])

    @pytest.mark.parametrize("df", [0.0, 0.05e6])
    def test_recovers_common_phase(self, backend_device, xp, df):
        """Two-pilot MRC tracks the shared FOE + phase noise to < 0.15 rad RMS."""
        samples, fs, common = self._setup(xp, df=df, linewidth=5e3)
        phi = recovery.recover_carrier_phase_pilot_tones(
            samples,
            fs,
            [self.F0, self.F1],
            bandwidth=self.BW,
            differential_bandwidth=self.DIFF_BW,
            per_tone_channel=[0, 1],
        )
        assert self._interior_rms(xp, phi[0], common) < 0.15

    def test_mrc_beats_single_tone(self, backend_device, xp):
        """Combining two equal-SNR pilots lowers the residual vs a single tone.

        Run in a noise-limited regime (low tone SNR) where the √2 of the combine
        is visible; a *narrow* δ low-pass is essential - a wide one tracks noise
        into δ and the combine can lose to single-tone.
        """
        samples, fs, common = self._setup(
            xp,
            df=0.0,
            linewidth=0.0,
            snr_db=6,
            psr_db=(-10.0, -10.0),
        )
        single = recovery.recover_carrier_phase_pilot_tone(
            samples,
            fs,
            self.F0,
            bandwidth=self.BW,
            joint_channels=False,
        )
        both, diag = recovery.recover_carrier_phase_pilot_tones(
            samples,
            fs,
            [self.F0, self.F1],
            bandwidth=self.BW,
            differential_bandwidth=self.DIFF_BW,
            per_tone_channel=[0, 1],
            return_diagnostics=True,
        )
        assert diag["used"] == [0, 1]  # both tones genuinely combined
        # single-tone reads tone 0 on channel 0; compare like-for-like on row 0.
        rms_single = self._interior_rms(xp, single[0], common)
        rms_both = self._interior_rms(xp, both[0], common)
        assert rms_both < rms_single

    def test_output_shape_and_common_broadcast(self, backend_device, xp):
        """(C, N) input -> (C, N) output; the common track is identical on all rows."""
        samples, fs, _ = self._setup(xp, df=0.05e6)
        phi = recovery.recover_carrier_phase_pilot_tones(
            samples,
            fs,
            [self.F0, self.F1],
            bandwidth=self.BW,
            differential_bandwidth=self.DIFF_BW,
            per_tone_channel=[0, 1],
        )
        assert phi.shape == samples.shape
        assert bool(xp.allclose(phi[0], phi[1]))

    def test_gating_drops_faded_tone(self, backend_device, xp):
        """A weak pilot (low tone-to-noise) is gated out -> single-tone fallback.

        Fade tone 1 by lowering its PSR (not by scaling the whole channel, which
        is SNR-invariant): a 48 dB weaker tone falls below the SNR gate.
        """
        samples, fs, _ = self._setup(xp, psr_db=(-12.0, -60.0))
        _, diag = recovery.recover_carrier_phase_pilot_tones(
            samples,
            fs,
            [self.F0, self.F1],
            bandwidth=self.BW,
            differential_bandwidth=self.DIFF_BW,
            per_tone_channel=[0, 1],
            snr_gate_db=3.0,
            return_diagnostics=True,
        )
        assert diag["used"] == [diag["ref"]]
        assert diag["ref"] == 0  # the strong tone

    def test_single_tone_K1_tracks(self, backend_device, xp):
        """K=1 reduces to a single-tone tracker and still recovers the phase."""
        samples, fs, common = self._setup(xp, df=0.05e6)
        phi = recovery.recover_carrier_phase_pilot_tones(
            samples,
            fs,
            [self.F0],
            bandwidth=self.BW,
            differential_bandwidth=self.DIFF_BW,
            per_tone_channel=[0],
        )
        assert self._interior_rms(xp, phi[0], common) < 0.15

    def test_joint_pre_demux_combine(self, backend_device, xp):
        """per_tone_channel=None coherently sums each tone across channels."""
        samples, fs, common = self._setup(xp, df=0.05e6)
        phi = recovery.recover_carrier_phase_pilot_tones(
            samples,
            fs,
            [self.F0, self.F1],
            bandwidth=self.BW,
            differential_bandwidth=self.DIFF_BW,
            per_tone_channel=None,
        )
        assert self._interior_rms(xp, phi[0], common) < 0.15

    def test_diagnostics_keys(self, backend_device, xp):
        samples, fs, _ = self._setup(xp)
        _, diag = recovery.recover_carrier_phase_pilot_tones(
            samples,
            fs,
            [self.F0, self.F1],
            bandwidth=self.BW,
            differential_bandwidth=self.DIFF_BW,
            per_tone_channel=[0, 1],
            return_diagnostics=True,
        )
        assert set(diag) == {"delta", "snr_db", "ref", "used", "f_centers"}
        assert len(diag["snr_db"]) == 2

    def test_per_tone_channel_length_mismatch_raises(self, backend_device, xp):
        samples, fs, _ = self._setup(xp, n_symbols=256)
        with pytest.raises(ValueError, match="per_tone_channel must have one entry"):
            recovery.recover_carrier_phase_pilot_tones(
                samples,
                fs,
                [self.F0, self.F1],
                bandwidth=self.BW,
                per_tone_channel=[0],
            )

    def test_invalid_bandwidth_raises(self, backend_device, xp):
        samples, fs, _ = self._setup(xp, n_symbols=256)
        with pytest.raises(ValueError, match="bandwidth must be > 0"):
            recovery.recover_carrier_phase_pilot_tones(
                samples,
                fs,
                [self.F0, self.F1],
                bandwidth=0.0,
            )

    def test_empty_tone_list_raises(self, backend_device, xp):
        samples, fs, _ = self._setup(xp, n_symbols=256)
        with pytest.raises(ValueError, match="at least one frequency"):
            recovery.recover_carrier_phase_pilot_tones(
                samples,
                fs,
                [],
                bandwidth=self.BW,
            )


class TestPilotsCPREnhancements:
    """Pilots CPR joint_channels and cycle_slip_correction parameters."""

    def _pilot_setup(
        self, xp, n_symbols=512, pilot_period=8, phase_per_sym=0.001, seed=42
    ):
        """Return (samples, pilot_indices, pilot_values) for a 16-QAM signal."""
        sig = generate_qam(
            order=16, num_symbols=n_symbols, sps=1, symbol_rate=FS, seed=seed
        )
        ideal = xp.asarray(sig.samples.copy())
        sig.samples = apply_awgn(sig.samples, esn0_db=SNR_DB, sps=1, seed=seed)
        sig.samples = _apply_phase_ramp(xp, sig.samples, phase_per_sym)
        pilot_indices = np.arange(0, n_symbols, pilot_period)
        pilot_values = ideal[pilot_indices]
        return sig.samples, pilot_indices, pilot_values

    def test_joint_rows_identical(self, backend_device, xp):
        """joint_channels=True: both phi_full rows are bitwise identical."""
        samples_a, pilot_indices, pilot_values = self._pilot_setup(xp, seed=1)
        samples_b, _, _ = self._pilot_setup(xp, seed=2)
        mimo = xp.stack([samples_a, samples_b], axis=0)  # (2, N)
        phi = recovery.recover_carrier_phase_pilot_symbols(
            mimo,
            pilot_indices=pilot_indices,
            pilot_values=pilot_values,
            joint_channels=True,
            cycle_slip_correction=False,
        )
        assert phi.shape == (2, mimo.shape[-1])
        phi_np = phi if xp is np else phi.get()
        np.testing.assert_array_equal(phi_np[0], phi_np[1])

    def test_joint_siso_noop(self, backend_device, xp):
        """joint_channels=True on SISO returns identical result to False."""
        samples, pilot_indices, pilot_values = self._pilot_setup(xp, seed=3)
        phi_a = recovery.recover_carrier_phase_pilot_symbols(
            samples,
            pilot_indices=pilot_indices,
            pilot_values=pilot_values,
            joint_channels=False,
            cycle_slip_correction=False,
        )
        phi_b = recovery.recover_carrier_phase_pilot_symbols(
            samples,
            pilot_indices=pilot_indices,
            pilot_values=pilot_values,
            joint_channels=True,
            cycle_slip_correction=False,
        )
        phi_a_np = phi_a if xp is np else phi_a.get()
        phi_b_np = phi_b if xp is np else phi_b.get()
        np.testing.assert_allclose(phi_a_np, phi_b_np, atol=1e-10)

    def test_cycle_slip_shape(self, backend_device, xp):
        """cycle_slip_correction=True returns correct shape."""
        samples, pilot_indices, pilot_values = self._pilot_setup(xp)
        phi = recovery.recover_carrier_phase_pilot_symbols(
            samples,
            pilot_indices=pilot_indices,
            pilot_values=pilot_values,
            cycle_slip_correction=True,
        )
        assert phi.shape == samples.shape

    def test_cycle_slip_standalone_symmetry_1(self, backend_device, xp):
        """correct_cycle_slips with symmetry=1 corrects an injected 2π slip."""
        B = 200
        phi_u = np.linspace(0.0, 3.0, B)
        phi_slipped = phi_u.copy()
        phi_slipped[100:] += 2.0 * np.pi
        phi_out = recovery.correct_cycle_slips(
            phi_slipped, symmetry=1, history_length=50
        )
        np.testing.assert_allclose(phi_out, phi_u, atol=0.1)

    def test_joint_cycle_slip_mimo_rows_identical(self, backend_device, xp):
        """joint_channels=True + cycle_slip_correction=True: rows remain identical."""
        samples_a, pilot_indices, pilot_values = self._pilot_setup(xp, seed=4)
        samples_b, _, _ = self._pilot_setup(xp, seed=5)
        mimo = xp.stack([samples_a, samples_b], axis=0)
        phi = recovery.recover_carrier_phase_pilot_symbols(
            mimo,
            pilot_indices=pilot_indices,
            pilot_values=pilot_values,
            joint_channels=True,
            cycle_slip_correction=True,
        )
        assert phi.shape == (2, mimo.shape[-1])
        phi_np = phi if xp is np else phi.get()
        np.testing.assert_array_equal(phi_np[0], phi_np[1])


class TestSignalInputPilotFunctions:
    """Signal-awareness for pilot-symbol and pilot-tone CPR."""

    def test_pilot_symbols_signal_input(self, backend_device, xp, xpt):
        """Signal input: symbols come from .samples, output stays a raw array."""
        samples, pilot_indices, pilot_values, _ = TestCprPilots()._pilot_setup(xp)
        sig = Signal(samples=samples, sampling_rate=FS, symbol_rate=FS)

        phi_sig = recovery.recover_carrier_phase_pilot_symbols(
            sig, pilot_indices=pilot_indices, pilot_values=pilot_values
        )
        phi_arr = recovery.recover_carrier_phase_pilot_symbols(
            samples, pilot_indices=pilot_indices, pilot_values=pilot_values
        )

        assert not isinstance(phi_sig, Signal)
        xpt.assert_allclose(phi_sig, phi_arr)

    def test_pilot_tone_signal_input_uses_sampling_rate(self, backend_device, xp, xpt):
        """Signal input: sampling_rate is taken from the signal."""
        cpr = TestCprPilotTone()
        samples, fs, _ = cpr._setup(xp, df=0.05e6)
        sig = Signal(samples=samples, sampling_rate=fs, symbol_rate=FS)

        theta_sig = recovery.recover_carrier_phase_pilot_tone(
            sig, tone_frequency=cpr.F_TONE, bandwidth=cpr.BW
        )
        theta_arr = recovery.recover_carrier_phase_pilot_tone(
            samples, fs, cpr.F_TONE, bandwidth=cpr.BW
        )

        assert not isinstance(theta_sig, Signal)
        xpt.assert_allclose(theta_sig, theta_arr)

    def test_pilot_tones_signal_input_uses_sampling_rate(self, backend_device, xp, xpt):
        """Signal input: sampling_rate is taken from the signal."""
        cpr = TestCprPilotTones()
        samples, fs, _ = cpr._setup(xp, df=0.05e6, linewidth=5e3)
        sig = Signal(samples=samples, sampling_rate=fs, symbol_rate=FS)

        phi_sig = recovery.recover_carrier_phase_pilot_tones(
            sig, tone_frequencies=[cpr.F0, cpr.F1], bandwidth=cpr.BW
        )
        phi_arr = recovery.recover_carrier_phase_pilot_tones(
            samples, fs, [cpr.F0, cpr.F1], bandwidth=cpr.BW
        )

        assert not isinstance(phi_sig, Signal)
        xpt.assert_allclose(phi_sig, phi_arr)
