"""Phase corrections, cycle-slip correction, and phase-ambiguity resolution."""

import numpy as np
import pytest

from commkit import generate_qam, recovery, spectral
from commkit.impairments import apply_awgn
from commkit.mapping import gray_constellation

FS = 1e6  # 1 MHz sampling rate, common to all tests


SNR_DB = 30  # generous SNR so numerical algorithms converge reliably


def _qam_signal(xp, order, n_symbols, fo_hz=0.0, snr_db=SNR_DB, fs=FS, seed=42):
    """Generate a 1-SPS QAM signal with optional frequency offset and AWGN."""
    sig = generate_qam(
        order=order, num_symbols=n_symbols, sps=1, symbol_rate=fs, seed=seed
    )
    sig.samples = apply_awgn(sig.samples, esn0_db=snr_db, sps=1, seed=seed)
    if fo_hz != 0.0:
        sig.samples, _ = spectral.shift_frequency(sig.samples, fo_hz, fs)
    return sig


class TestCorrectionFunctions:
    def test_correct_carrier_phase_dtype_preserved(self, backend_device, xp):
        """correct_carrier_phase: complex64 input -> complex64 output."""
        sig = _qam_signal(xp, 4, 512)
        phase = xp.zeros(512, dtype=xp.float64)
        corrected = recovery.correct_carrier_phase(sig.samples, phase)
        assert corrected.dtype == xp.complex64

    def test_correct_carrier_phase_zero_phase_identity(self, backend_device, xp):
        """Applying zero phase correction leaves samples unchanged."""
        sig = _qam_signal(xp, 4, 512)
        phase = xp.zeros(512, dtype=xp.float64)
        corrected = recovery.correct_carrier_phase(sig.samples, phase)
        assert float(xp.max(xp.abs(corrected - sig.samples))) < 1e-5


class TestCycleSlipCorrection:
    """correct_cycle_slips() detects and corrects injected slips."""

    def test_standalone_no_slip(self, backend_device, xp):
        """Smooth linear ramp with no slips is returned unchanged."""
        B = 200
        phi_u = np.linspace(0.0, 2.0, B)
        phi_out = recovery.correct_cycle_slips(
            phi_u.copy(), symmetry=4, history_length=50
        )
        np.testing.assert_allclose(phi_out, phi_u, atol=1e-10)

    def test_standalone_single_slip(self, backend_device, xp):
        """A single injected pi/2 slip is corrected back to the original ramp."""
        B = 300
        phi_u = np.linspace(0.0, 1.0, B)
        phi_slipped = phi_u.copy()
        phi_slipped[150:] += np.pi / 2
        phi_out = recovery.correct_cycle_slips(
            phi_slipped, symmetry=4, history_length=100
        )
        np.testing.assert_allclose(phi_out, phi_u, atol=0.05)

    def test_standalone_multiple_slips(self, backend_device, xp):
        """Multiple +/-pi/2 slips are all corrected."""
        B = 500
        phi_u = np.linspace(0.0, 1.5, B)
        phi_slipped = phi_u.copy()
        phi_slipped[100:] += np.pi / 2
        phi_slipped[300:] -= np.pi / 2
        phi_out = recovery.correct_cycle_slips(
            phi_slipped, symmetry=4, history_length=80
        )
        np.testing.assert_allclose(phi_out, phi_u, atol=0.05)

    def test_bps_correction_bounded_output(self, backend_device, xp):
        """BPS cycle_slip_correction=True returns phase within reasonable bounds."""
        sig = _qam_signal(xp, 16, 2048, snr_db=SNR_DB)
        phi = recovery.recover_carrier_phase_bps(
            sig.samples, "qam", 16, cycle_slip_correction=True
        )
        assert phi.shape == sig.samples.shape
        phi_np = phi if xp is np else phi.get()
        assert np.max(np.abs(phi_np)) < 10 * np.pi

    def test_vv_correction_shape(self, backend_device, xp):
        """VV cycle_slip_correction=True returns correct shape."""
        sig = _qam_signal(xp, 16, 2048, snr_db=SNR_DB)
        phi = recovery.recover_carrier_phase_viterbi_viterbi(
            sig.samples, "qam", 16, cycle_slip_correction=True
        )
        assert phi.shape == sig.samples.shape

    def test_tikhonov_correction_shape(self, backend_device, xp):
        """Tikhonov cycle_slip_correction=True returns correct shape."""
        sig = _qam_signal(xp, 16, 2048, snr_db=SNR_DB)
        phi = recovery.recover_carrier_phase_tikhonov(
            sig.samples,
            "qam",
            16,
            linewidth_symbol_periods=1e-4,
            snr_db=SNR_DB,
            cycle_slip_correction=True,
        )
        assert phi.shape == sig.samples.shape


class TestResolvePhaseAmbiguity:
    """resolve_phase_ambiguity selects the rotation with lowest SER."""

    N = 2048

    def test_best_rotation_is_zero(self, backend_device, xp):
        """Already-aligned symbols: k=0 chosen and SER is minimal."""
        from commkit.helpers import normalize
        from commkit.metrics import ser

        sig = _qam_signal(xp, 16, self.N, snr_db=30, seed=5)
        sym = normalize(sig.samples, "average_power")
        ref = normalize(xp.asarray(sig.source_symbols), "average_power")
        resolved = recovery.resolve_phase_ambiguity(sym, ref, "qam", 16)
        s0 = float(ser(resolved, ref, "qam", 16))
        for k in range(1, 4):
            sk = float(
                ser(
                    resolved * xp.exp(1j * k * np.pi / 2).astype(sym.dtype),
                    ref,
                    "qam",
                    16,
                )
            )
            assert s0 <= sk + 1e-6

    def test_corrects_pi_half_rotation(self, backend_device, xp):
        """Symbols rotated by pi/2 are corrected; post-resolution SER is low."""
        from commkit.helpers import normalize
        from commkit.metrics import ser

        sig = _qam_signal(xp, 16, self.N, snr_db=30, seed=5)
        sym = normalize(sig.samples, "average_power")
        ref = normalize(xp.asarray(sig.source_symbols), "average_power")
        rotated = sym * xp.exp(1j * np.pi / 2).astype(sym.dtype)
        resolved = recovery.resolve_phase_ambiguity(rotated, ref, "qam", 16)
        assert float(ser(resolved, ref, "qam", 16)) < 0.05

    def test_mimo_independent_per_channel(self, backend_device, xp):
        """MIMO: channels with different rotations are each independently corrected."""
        from commkit.helpers import normalize
        from commkit.metrics import ser

        sig_a = _qam_signal(xp, 16, self.N, seed=1)
        sig_b = _qam_signal(xp, 16, self.N, seed=2)
        sym_a = normalize(sig_a.samples, "average_power")
        sym_b = normalize(sig_b.samples, "average_power")
        ref_a = normalize(xp.asarray(sig_a.source_symbols), "average_power")
        ref_b = normalize(xp.asarray(sig_b.source_symbols), "average_power")
        mimo = xp.stack(
            [
                sym_a * xp.exp(1j * np.pi / 2).astype(sym_a.dtype),
                sym_b * xp.exp(1j * np.pi).astype(sym_b.dtype),
            ],
            axis=0,
        )
        ref_mimo = xp.stack([ref_a, ref_b], axis=0)
        resolved = recovery.resolve_phase_ambiguity(mimo, ref_mimo, "qam", 16)
        assert resolved.shape == (2, self.N)
        s = ser(resolved, ref_mimo, "qam", 16)
        s_np = s if xp is np else s.get()
        assert float(s_np[0]) < 0.05
        assert float(s_np[1]) < 0.05

    def test_signal_method_in_place(self, backend_device, xp):
        """Signal.resolve_phase_ambiguity() updates resolved_symbols in place."""
        from commkit.helpers import normalize
        from commkit.metrics import ser

        sig = generate_qam(order=16, num_symbols=self.N, sps=1, symbol_rate=1e6, seed=9)
        sig.samples = apply_awgn(sig.samples, esn0_db=30, sps=1, seed=9)
        sym = normalize(sig.samples, "average_power")
        sig.resolved_symbols = sym * xp.exp(1j * np.pi / 2).astype(sym.dtype)
        sig = recovery.resolve_phase_ambiguity(sig)
        assert sig.resolved_symbols is not None
        ref = normalize(xp.asarray(sig.source_symbols), "average_power")
        assert float(ser(sig.resolved_symbols, ref, "qam", 16)) < 0.1

    def test_signal_method_raises_without_resolved(self, backend_device, xp):
        """Raises ValueError when resolved_symbols is None."""
        sig = generate_qam(order=16, num_symbols=256, sps=1, symbol_rate=1e6, seed=0)
        with pytest.raises(ValueError, match="resolved_symbols"):
            sig = recovery.resolve_phase_ambiguity(sig)

    def test_signal_method_raises_without_source(self, backend_device, xp):
        """Raises ValueError when source_symbols is None."""
        sig = generate_qam(order=16, num_symbols=256, sps=1, symbol_rate=1e6, seed=0)
        sig.resolved_symbols = sig.samples
        sig.source_symbols = None
        with pytest.raises(ValueError, match="source_symbols"):
            sig = recovery.resolve_phase_ambiguity(sig)


def _make_ambiguous_qam16(n_sym=2000, corrupt_head=500, seed=0):
    """Return (symbols, ref) where the first corrupt_head symbols are rotated by π/2."""
    rng = np.random.default_rng(seed)
    const = gray_constellation("qam", 16).astype(np.complex64)
    const /= np.sqrt(np.mean(np.abs(const) ** 2))
    ref = const[rng.integers(0, 16, n_sym)]
    # True ambiguity k=1: rotate entire stream by π/2
    rot1 = np.exp(1j * np.pi / 2).astype(np.complex64)
    symbols = ref * rot1
    # Corrupt only the first corrupt_head symbols with an additional π/2 (total π)
    symbols[:corrupt_head] = ref[:corrupt_head] * np.exp(1j * np.pi).astype(
        np.complex64
    )
    return symbols, ref


def test_resolve_phase_ambiguity_skip(backend_device, xp, xpt):
    """num_skip_symbols bypasses the corrupt head and picks the correct rotation."""
    n_sym, corrupt_head = 2000, 500
    symbols_np, ref_np = _make_ambiguous_qam16(n_sym=n_sym, corrupt_head=corrupt_head)
    symbols, ref = xp.asarray(symbols_np), xp.asarray(ref_np)

    out_no_skip = recovery.resolve_phase_ambiguity(
        symbols, ref, "qam", 16, num_skip_symbols=0
    )
    out_skip = recovery.resolve_phase_ambiguity(
        symbols, ref, "qam", 16, num_skip_symbols=corrupt_head
    )

    from commkit.metrics import ser as _ser_fn

    def _ser(y, r):
        return float(xp.mean(xp.asarray(_ser_fn(y, r, "qam", 16))))

    ser_skip_tail = _ser(out_skip[corrupt_head:], ref[corrupt_head:])
    ser_no_skip_tail = _ser(out_no_skip[corrupt_head:], ref[corrupt_head:])
    assert ser_skip_tail <= ser_no_skip_tail, (
        f"Skip should improve tail SER: {ser_skip_tail:.4f} vs {ser_no_skip_tail:.4f}"
    )


def test_resolve_phase_ambiguity_skip_zero_is_baseline(backend_device, xp, xpt):
    """num_skip_symbols=0 must produce identical output to the default call."""
    symbols_np, ref_np = _make_ambiguous_qam16(n_sym=1000, corrupt_head=0)
    symbols, ref = xp.asarray(symbols_np), xp.asarray(ref_np)

    out_default = recovery.resolve_phase_ambiguity(symbols, ref, "qam", 16)
    out_skip0 = recovery.resolve_phase_ambiguity(
        symbols, ref, "qam", 16, num_skip_symbols=0
    )

    assert bool(xp.all(out_default == out_skip0))


def test_resolve_phase_ambiguity_skip_ge_n_raises(backend_device, xp):
    """num_skip_symbols >= N must raise ValueError."""
    symbols_np, ref_np = _make_ambiguous_qam16(n_sym=100, corrupt_head=0)
    symbols, ref = xp.asarray(symbols_np), xp.asarray(ref_np)

    with pytest.raises(ValueError, match="num_skip_symbols"):
        recovery.resolve_phase_ambiguity(symbols, ref, "qam", 16, num_skip_symbols=100)

    with pytest.raises(ValueError, match="num_skip_symbols"):
        recovery.resolve_phase_ambiguity(symbols, ref, "qam", 16, num_skip_symbols=200)


def _clean_qam16(xp, n, seed=0):
    """Noiseless unit-power 16-QAM symbols."""
    rng = np.random.default_rng(seed)
    const = gray_constellation("qam", 16).astype(np.complex64)
    const /= np.sqrt(np.mean(np.abs(const) ** 2))
    return xp.asarray(const[rng.integers(0, 16, n)])


class TestCorrectPhaseRotation:
    """correct_phase_rotation corrects arbitrary constant per-channel rotation."""

    N = 2048

    def test_arbitrary_rotation_corrected_siso(self, backend_device, xp, xpt):
        """Arbitrary non-grid rotation is removed; residual angle is near zero."""
        ref = _clean_qam16(xp, self.N, seed=0)
        theta_true = 0.7  # ~40°, not a π/2 multiple
        rotated = ref * xp.array(np.exp(1j * theta_true), dtype=ref.dtype)
        out = recovery.correct_phase_rotation(rotated, ref)
        residual = float(xp.abs(xp.angle(xp.mean(out * xp.conj(ref)))))
        assert residual < 0.02

    def test_short_ref_applies_to_full_sequence(self, backend_device, xp):
        """Estimation from first N_pre symbols; correction spans the full N sequence."""
        N, N_pre = self.N, 256
        ref_full = _clean_qam16(xp, N, seed=1)
        rotated = ref_full * xp.array(np.exp(1j * 1.2), dtype=ref_full.dtype)
        out = recovery.correct_phase_rotation(rotated, ref_full[:N_pre])
        assert out.shape == rotated.shape
        residual = float(xp.abs(xp.angle(xp.mean(out * xp.conj(ref_full)))))
        assert residual < 0.02

    def test_mimo_independent_channels(self, backend_device, xp):
        """Each MIMO channel gets its own rotation corrected independently."""
        ref_a = _clean_qam16(xp, self.N, seed=2)
        ref_b = _clean_qam16(xp, self.N, seed=3)
        ref = xp.stack([ref_a, ref_b])
        rotated = xp.stack(
            [
                ref_a * xp.array(np.exp(1j * 0.4), dtype=ref_a.dtype),
                ref_b * xp.array(np.exp(1j * -1.1), dtype=ref_b.dtype),
            ]
        )
        out = recovery.correct_phase_rotation(rotated, ref)
        assert out.shape == (2, self.N)
        for ch in range(2):
            residual = float(xp.abs(xp.angle(xp.mean(out[ch] * xp.conj(ref[ch])))))
            assert residual < 0.02

    def test_num_skip_symbols_excludes_transient(self, backend_device, xp):
        """Corrupted head is excluded; tail correction uses the clean portion only."""
        N, skip = self.N, 200
        ref = _clean_qam16(xp, N, seed=4)
        rotated = ref * xp.array(np.exp(1j * 0.9), dtype=ref.dtype)
        corrupted = xp.array(rotated)
        corrupted[:skip] = ref[:skip] * xp.array(np.exp(1j * 2.5), dtype=ref.dtype)
        out = recovery.correct_phase_rotation(corrupted, ref, num_skip_symbols=skip)
        residual = float(xp.abs(xp.angle(xp.mean(out[skip:] * xp.conj(ref[skip:])))))
        assert residual < 0.02

    def test_num_skip_ge_nref_raises(self, backend_device, xp):
        """num_skip_symbols >= N_ref must raise ValueError."""
        ref = _clean_qam16(xp, 100, seed=0)
        symbols = _clean_qam16(xp, 500, seed=1)
        with pytest.raises(ValueError, match="num_skip_symbols"):
            recovery.correct_phase_rotation(symbols, ref, num_skip_symbols=100)
        with pytest.raises(ValueError, match="num_skip_symbols"):
            recovery.correct_phase_rotation(symbols, ref, num_skip_symbols=200)

    def test_dtype_preserved(self, backend_device, xp):
        """complex64 input -> complex64 output."""
        ref = _clean_qam16(xp, 256, seed=0)
        out = recovery.correct_phase_rotation(
            ref * xp.array(np.exp(1j * 0.5), dtype=ref.dtype), ref
        )
        assert out.dtype == ref.dtype

    def test_1d_input_returns_1d(self, backend_device, xp):
        """1-D input returns 1-D output."""
        ref = _clean_qam16(xp, 256, seed=0)
        out = recovery.correct_phase_rotation(
            ref * xp.array(np.exp(1j * 0.3), dtype=ref.dtype), ref
        )
        assert out.ndim == 1


class TestResolveChannelPermutation:
    """Stream-assignment resolution under both scoring metrics."""

    @staticmethod
    def _dual_pol(xp, n=4096, seed=7):
        """Two independent QPSK streams and the equalizer's swapped output."""
        rng = np.random.default_rng(seed)
        s = rng.choice([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j], (2, n)) / np.sqrt(2.0)
        return xp.asarray(s), xp.asarray(s[::-1].copy())

    @pytest.mark.parametrize("metric", ["coherence", "phase_increment"])
    def test_resolves_swap(self, backend_device, xp, xpt, metric):
        """A swapped output is reordered back to reference order."""
        ref, swapped = self._dual_pol(xp)
        out = recovery.resolve_channel_permutation(swapped, ref, metric=metric)
        xpt.assert_allclose(out, ref)

    @pytest.mark.parametrize("metric", ["coherence", "phase_increment"])
    def test_identity_is_a_no_op(self, backend_device, xp, xpt, metric):
        """Already-aligned streams are returned in the same order."""
        ref, _ = self._dual_pol(xp)
        out = recovery.resolve_channel_permutation(ref, ref, metric=metric)
        xpt.assert_allclose(out, ref)

    def test_phase_increment_survives_a_frequency_offset(self, backend_device, xp, xpt):
        """The metric that exists for carrier-phase-intact records.

        With the carrier left on, ``|Σ y·conj(s)|`` averages a rotating phasor
        toward zero for *every* pairing, so the coherence scores collapse below
        even the weak-match threshold and the assignment is arbitrary.  The
        phase-increment metric differences the phase error first and is
        unaffected.
        """
        from commkit.recovery.corrections import _pairing_scores

        ref, swapped = self._dual_pol(xp)
        n = ref.shape[-1]
        # 0.01 cycles/symbol: many full rotations across the record.
        ramp = xp.exp(2j * np.pi * 0.01 * xp.arange(n, dtype=xp.float64))
        spun = (swapped * ramp[None, :]).astype(ref.dtype)

        coh = _pairing_scores(spun, ref, xp, "coherence")
        assert float(np.max(coh)) < 0.3, "coherence should collapse under a FOE"

        out = recovery.resolve_channel_permutation(spun, ref, metric="phase_increment")
        # Correct pairing leaves only the common ramp on each channel.
        xpt.assert_allclose(out * xp.conj(ramp)[None, :], ref, atol=1e-6)

    def test_rejects_unknown_metric(self, backend_device, xp):
        ref, swapped = self._dual_pol(xp, n=64)
        with pytest.raises(ValueError, match="metric"):
            recovery.resolve_channel_permutation(swapped, ref, metric="nonsense")
