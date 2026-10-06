"""Phase corrections, cycle-slip correction, and phase-ambiguity resolution."""

from typing import Any

import numpy as np
import pytest

from commkit import generate, recovery
from commkit.filtering import RRC
from commkit.impairments import apply_awgn
from commkit.mapping import Constellation
from tests.common.conversions import device_of, to_numpy
from tests.common.signals import (
    make_ambiguous_qam16,
    make_test_mimo_samples,
    make_test_qam_signal,
    make_test_symbols,
)

FS = 1e6  # 1 MHz sampling rate, common to all tests
SNR_DB = 30  # generous SNR so numerical algorithms converge reliably


class TestCorrectionFunctions:
    def test_correct_carrier_phase_dtype_preserved(self, xp):
        """correct_carrier_phase: complex64 input -> complex64 output."""
        sig = make_test_qam_signal(
            order=4, num_symbols=512, sps=1, symbol_rate=FS, xp=xp
        )
        phase = xp.zeros(512, dtype=xp.float64)
        corrected = recovery.correct_carrier_phase(sig.samples, phase)
        assert corrected.dtype == xp.complex64

    def test_correct_carrier_phase_zero_phase_identity(self, xp):
        """Applying zero phase correction leaves samples unchanged."""
        sig = make_test_qam_signal(
            order=4, num_symbols=512, sps=1, symbol_rate=FS, xp=xp
        )
        phase = xp.zeros(512, dtype=xp.float64)
        corrected = recovery.correct_carrier_phase(sig.samples, phase)
        assert float(xp.max(xp.abs(corrected - sig.samples))) < 1e-5


class TestCycleSlipCorrection:
    """correct_cycle_slips() detects and corrects injected slips."""

    def test_standalone_no_slip(self, xp, xpt):
        """Smooth linear ramp with no slips is returned unchanged."""
        B = 200
        phi_u = np.linspace(0.0, 2.0, B)
        phi_out = recovery.correct_cycle_slips(phi_u.copy(), symmetry=4, history=50)
        xpt.assert_allclose(phi_out, phi_u, atol=1e-10)

    def test_standalone_single_slip(self, xp, xpt):
        """A single injected pi/2 slip is corrected back to the original ramp."""
        B = 300
        phi_u = np.linspace(0.0, 1.0, B)
        phi_slipped = phi_u.copy()
        phi_slipped[150:] += np.pi / 2
        phi_out = recovery.correct_cycle_slips(phi_slipped, symmetry=4, history=100)
        xpt.assert_allclose(phi_out, phi_u, atol=0.05)

    def test_standalone_multiple_slips(self, xp, xpt):
        """Multiple +/-pi/2 slips are all corrected."""
        B = 500
        phi_u = np.linspace(0.0, 1.5, B)
        phi_slipped = phi_u.copy()
        phi_slipped[100:] += np.pi / 2
        phi_slipped[300:] -= np.pi / 2
        phi_out = recovery.correct_cycle_slips(phi_slipped, symmetry=4, history=80)
        xpt.assert_allclose(phi_out, phi_u, atol=0.05)

    def test_returns_new_array_on_input_device(self, xp, xpt):
        """The input is left as it was; the result stays on its device."""
        phi_slipped = np.linspace(0.0, 1.0, 300)
        phi_slipped[150:] += np.pi / 2
        x = xp.asarray(phi_slipped)
        phi_out = recovery.correct_cycle_slips(x, symmetry=4, history=100)
        xpt.assert_array_equal(x, xp.asarray(phi_slipped))
        assert type(phi_out) is type(x)
        assert float(xp.max(xp.abs(phi_out - x))) == pytest.approx(np.pi / 2)

    def test_bps_correction_bounded_output(self, xp):
        """BPS cycle_slip_correction=True returns phase within reasonable bounds."""
        sig = make_test_qam_signal(
            order=16, num_symbols=2048, sps=1, snr_db=SNR_DB, xp=xp
        )
        phi = recovery.estimate_carrier_phase(
            sig.samples,
            recovery.BPS(cycle_slip=recovery.CycleSlip()),
            constellation=Constellation.qam(16),
        ).value
        assert phi.shape == sig.samples.shape
        phi_np = to_numpy(phi)
        assert np.max(np.abs(phi_np)) < 10 * np.pi

    def test_vv_correction_shape(self, xp):
        """VV cycle_slip_correction=True returns correct shape."""
        sig = make_test_qam_signal(
            order=16, num_symbols=2048, sps=1, snr_db=SNR_DB, xp=xp
        )
        phi = recovery.estimate_carrier_phase(
            sig.samples,
            recovery.ViterbiViterbi(cycle_slip=recovery.CycleSlip()),
            constellation=Constellation.qam(16),
        ).value
        assert phi.shape == sig.samples.shape

    def test_tikhonov_correction_shape(self, xp):
        """Tikhonov cycle_slip_correction=True returns correct shape."""
        sig = make_test_qam_signal(
            order=16, num_symbols=2048, sps=1, snr_db=SNR_DB, xp=xp
        )
        phi = recovery.estimate_carrier_phase(
            sig.samples,
            recovery.Tikhonov(
                linewidth_symbol_periods=1e-4,
                snr_db=SNR_DB,
                cycle_slip=recovery.CycleSlip(),
            ),
            constellation=Constellation.qam(16),
        ).value
        assert phi.shape == sig.samples.shape


class TestResolvePhaseAmbiguity:
    """resolve_phase_ambiguity selects the rotation with lowest SER."""

    N = 2048

    def test_best_rotation_is_zero(self, xp):
        """Already-aligned symbols: k=0 chosen and SER is minimal."""
        from commkit.math import normalize
        from commkit.metrics import ser

        sig = make_test_qam_signal(
            order=16, num_symbols=self.N, sps=1, snr_db=30, seed=5, xp=xp
        )
        sym = normalize(sig.samples, mode="average_power")
        ref = normalize(xp.asarray(sig.source_symbols), mode="average_power")
        resolved = recovery.resolve_phase_ambiguity(
            sym, ref, constellation=Constellation.qam(16)
        )
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

    def test_corrects_pi_half_rotation(self, xp):
        """Symbols rotated by pi/2 are corrected; post-resolution SER is low."""
        from commkit.math import normalize
        from commkit.metrics import ser

        sig = make_test_qam_signal(
            order=16, num_symbols=self.N, sps=1, snr_db=30, seed=5, xp=xp
        )
        sym = normalize(sig.samples, mode="average_power")
        ref = normalize(xp.asarray(sig.source_symbols), mode="average_power")
        rotated = sym * xp.exp(1j * np.pi / 2).astype(sym.dtype)
        resolved = recovery.resolve_phase_ambiguity(
            rotated, ref, constellation=Constellation.qam(16)
        )
        assert float(ser(resolved, ref, "qam", 16)) < 0.05

    def test_mimo_independent_per_channel(self, xp):
        """MIMO: channels with different rotations are each independently corrected."""
        from commkit.math import normalize
        from commkit.metrics import ser

        mimo, ref_mimo = make_test_mimo_samples(
            num_channels=2,
            order=16,
            num_symbols=self.N,
            sps=1,
            snr_db=30,
            seed=1,
            xp=xp,
        )
        sym_a = normalize(mimo[0], mode="average_power")
        sym_b = normalize(mimo[1], mode="average_power")
        ref_a = normalize(ref_mimo[0], mode="average_power")
        ref_b = normalize(ref_mimo[1], mode="average_power")
        mimo_rot = xp.stack(
            [
                sym_a * xp.exp(1j * np.pi / 2).astype(sym_a.dtype),
                sym_b * xp.exp(1j * np.pi).astype(sym_b.dtype),
            ],
            axis=0,
        )
        ref_mimo_norm = xp.stack([ref_a, ref_b], axis=0)
        resolved = recovery.resolve_phase_ambiguity(
            mimo_rot, ref_mimo_norm, constellation=Constellation.qam(16)
        )
        assert resolved.shape == (2, self.N)
        s = ser(resolved, ref_mimo_norm, "qam", 16)
        s_np = to_numpy(s)
        assert float(s_np[0]) < 0.05
        assert float(s_np[1]) < 0.05

    def test_signal_method_in_place(self, xp):
        """Signal.resolve_phase_ambiguity() updates resolved_symbols in place."""
        from commkit.math import normalize
        from commkit.metrics import ser

        sig = generate(
            Constellation.qam(16),
            self.N,
            symbol_rate=1e6,
            sps=1,
            pulse=RRC(0.35),
            rng=9,
        ).to(device_of(xp))
        sig = sig.replace(samples=apply_awgn(sig.samples, esn0_db=30, sps=1, rng=9))
        sym = normalize(sig.samples, mode="average_power")
        sig = sig.replace(
            resolved_symbols=sym * xp.exp(1j * np.pi / 2).astype(sym.dtype)
        )
        sig = recovery.resolve_phase_ambiguity(sig)
        assert sig.resolved_symbols is not None
        ref = normalize(xp.asarray(sig.source_symbols), mode="average_power")
        assert float(ser(sig.resolved_symbols, ref, "qam", 16)) < 0.1

    def test_signal_method_raises_without_resolved(self, xp):
        """Raises ValueError when resolved_symbols is None."""
        sig = generate(
            Constellation.qam(16), 256, symbol_rate=1e6, sps=1, pulse=RRC(0.35), rng=0
        )
        with pytest.raises(ValueError, match="resolved_symbols"):
            sig = recovery.resolve_phase_ambiguity(sig)

    def test_signal_method_raises_without_source(self, xp):
        """Raises ValueError when source_symbols is None."""
        sig = generate(
            Constellation.qam(16), 256, symbol_rate=1e6, sps=1, pulse=RRC(0.35), rng=0
        )
        sig = sig.replace(resolved_symbols=sig.samples)
        sig = sig.replace(reference=None)
        with pytest.raises(ValueError, match="source_symbols"):
            sig = recovery.resolve_phase_ambiguity(sig)

    def test_resolve_phase_ambiguity_skip(self, xp: Any, xpt: Any) -> None:
        """num_skip_symbols bypasses the corrupt head and picks the correct rotation."""
        n_sym, corrupt_head = 2000, 500
        symbols_np, ref_np = make_ambiguous_qam16(
            n_sym=n_sym, corrupt_head=corrupt_head
        )
        symbols, ref = xp.asarray(symbols_np), xp.asarray(ref_np)

        out_no_skip = recovery.resolve_phase_ambiguity(
            symbols, ref, constellation=Constellation.qam(16), num_skip_symbols=0
        )
        out_skip = recovery.resolve_phase_ambiguity(
            symbols,
            ref,
            constellation=Constellation.qam(16),
            num_skip_symbols=corrupt_head,
        )

        from commkit.metrics import ser as _ser_fn

        def _ser(y, r):
            return float(xp.mean(xp.asarray(_ser_fn(y, r, "qam", 16))))

        ser_skip_tail = _ser(out_skip[corrupt_head:], ref[corrupt_head:])
        ser_no_skip_tail = _ser(out_no_skip[corrupt_head:], ref[corrupt_head:])
        assert ser_skip_tail <= ser_no_skip_tail, (
            f"Skip should improve tail SER: {ser_skip_tail:.4f} vs {ser_no_skip_tail:.4f}"
        )

    def test_resolve_phase_ambiguity_skip_zero_is_baseline(
        self, xp: Any, xpt: Any
    ) -> None:
        """num_skip_symbols=0 must produce identical output to the default call."""
        symbols_np, ref_np = make_ambiguous_qam16(n_sym=1000, corrupt_head=0)
        symbols, ref = xp.asarray(symbols_np), xp.asarray(ref_np)

        out_default = recovery.resolve_phase_ambiguity(
            symbols, ref, constellation=Constellation.qam(16)
        )
        out_skip0 = recovery.resolve_phase_ambiguity(
            symbols, ref, constellation=Constellation.qam(16), num_skip_symbols=0
        )

        assert bool(xp.all(out_default == out_skip0))

    def test_resolve_phase_ambiguity_skip_ge_n_raises(self, xp: Any) -> None:
        """num_skip_symbols >= N must raise ValueError."""
        symbols_np, ref_np = make_ambiguous_qam16(n_sym=100, corrupt_head=0)
        symbols, ref = xp.asarray(symbols_np), xp.asarray(ref_np)

        with pytest.raises(ValueError, match="num_skip_symbols"):
            recovery.resolve_phase_ambiguity(
                symbols, ref, constellation=Constellation.qam(16), num_skip_symbols=100
            )

        with pytest.raises(ValueError, match="num_skip_symbols"):
            recovery.resolve_phase_ambiguity(
                symbols, ref, constellation=Constellation.qam(16), num_skip_symbols=200
            )


def _clean_qam16(xp, n, seed=0):
    """Noiseless unit-power 16-QAM symbols."""
    return make_test_symbols(scheme="qam", order=16, num_symbols=n, seed=seed, xp=xp)


class TestSlipQuantum:
    """Cycle slips are repaired in steps of 2π/M, M the rotational symmetry."""

    @pytest.mark.parametrize(
        "method",
        [
            recovery.BPS(cycle_slip=recovery.CycleSlip()),
            recovery.ViterbiViterbi(cycle_slip=recovery.CycleSlip()),
            recovery.Tikhonov(1e-4, 20, cycle_slip=recovery.CycleSlip()),
            recovery.PLL(mu=1e-2, cycle_slip=recovery.CycleSlip()),
        ],
        ids=["bps", "vv", "tikhonov", "pll"],
    )
    def test_methods_pass_the_constellation_symmetry(self, xp, method, monkeypatch):
        from commkit.recovery import corrections

        seen = []
        repair = corrections.correct_cycle_slips

        def spy(phase, **kwargs):
            seen.append(kwargs["symmetry"])
            return repair(phase, **kwargs)

        monkeypatch.setattr(corrections, "correct_cycle_slips", spy)
        c = Constellation.psk(8)
        rng = np.random.default_rng(0)
        x = xp.asarray(c.points[rng.integers(0, 8, 512)].astype(np.complex64))
        recovery.estimate_carrier_phase(x, method, constellation=c)
        assert seen == [8]


class TestDataAided:
    """DataAided corrects an arbitrary constant per-channel rotation."""

    N = 2048

    def test_arbitrary_rotation_corrected_siso(self, xp, xpt):
        """Arbitrary non-grid rotation is removed; residual angle is near zero."""
        ref = _clean_qam16(xp, self.N, seed=0)
        theta_true = 0.7  # ~40°, not a π/2 multiple
        rotated = ref * xp.array(np.exp(1j * theta_true), dtype=ref.dtype)
        out = recovery.correct_carrier_phase(rotated, recovery.DataAided(symbols=ref))
        residual = float(xp.abs(xp.angle(xp.mean(out * xp.conj(ref)))))
        assert residual < 0.02

    def test_short_ref_applies_to_full_sequence(self, xp):
        """Estimation from first N_pre symbols; correction spans the full N sequence."""
        N, N_pre = self.N, 256
        ref_full = _clean_qam16(xp, N, seed=1)
        rotated = ref_full * xp.array(np.exp(1j * 1.2), dtype=ref_full.dtype)
        out = recovery.correct_carrier_phase(
            rotated, recovery.DataAided(symbols=ref_full[:N_pre])
        )
        assert out.shape == rotated.shape
        residual = float(xp.abs(xp.angle(xp.mean(out * xp.conj(ref_full)))))
        assert residual < 0.02

    def test_mimo_independent_channels(self, xp):
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
        out = recovery.correct_carrier_phase(rotated, recovery.DataAided(symbols=ref))
        assert out.shape == (2, self.N)
        for ch in range(2):
            residual = float(xp.abs(xp.angle(xp.mean(out[ch] * xp.conj(ref[ch])))))
            assert residual < 0.02

    def test_num_skip_symbols_excludes_transient(self, xp):
        """Corrupted head is excluded; tail correction uses the clean portion only."""
        N, skip = self.N, 200
        ref = _clean_qam16(xp, N, seed=4)
        rotated = ref * xp.array(np.exp(1j * 0.9), dtype=ref.dtype)
        corrupted = xp.array(rotated)
        corrupted[:skip] = ref[:skip] * xp.array(np.exp(1j * 2.5), dtype=ref.dtype)
        out = recovery.correct_carrier_phase(
            corrupted, recovery.DataAided(symbols=ref, num_skip_symbols=skip)
        )
        residual = float(xp.abs(xp.angle(xp.mean(out[skip:] * xp.conj(ref[skip:])))))
        assert residual < 0.02

    def test_num_skip_ge_nref_raises(self, xp):
        """num_skip_symbols >= N_ref must raise ValueError."""
        ref = _clean_qam16(xp, 100, seed=0)
        symbols = _clean_qam16(xp, 500, seed=1)
        with pytest.raises(ValueError, match="num_skip_symbols"):
            recovery.correct_carrier_phase(
                symbols, recovery.DataAided(symbols=ref, num_skip_symbols=100)
            )
        with pytest.raises(ValueError, match="num_skip_symbols"):
            recovery.correct_carrier_phase(
                symbols, recovery.DataAided(symbols=ref, num_skip_symbols=200)
            )

    def test_dtype_preserved(self, xp):
        """complex64 input -> complex64 output."""
        ref = _clean_qam16(xp, 256, seed=0)
        out = recovery.correct_carrier_phase(
            ref * xp.array(np.exp(1j * 0.5), dtype=ref.dtype),
            recovery.DataAided(symbols=ref),
        )
        assert out.dtype == ref.dtype

    def test_1d_input_returns_1d(self, xp):
        """1-D input returns 1-D output."""
        ref = _clean_qam16(xp, 256, seed=0)
        out = recovery.correct_carrier_phase(
            ref * xp.array(np.exp(1j * 0.3), dtype=ref.dtype),
            recovery.DataAided(symbols=ref),
        )
        assert out.ndim == 1

    def test_signal_input_uses_reference(self, xp, xpt):
        """Signal input: the samples are rotated against ``sig.reference``."""
        sig = generate(
            Constellation.qam(16),
            self.N,
            symbol_rate=1e6,
            sps=1,
            pulse=RRC(0.35),
            rng=9,
        ).to(device_of(xp))
        ref = sig.reference.symbols
        sig = sig.replace(samples=ref * xp.array(np.exp(1j * 0.7), dtype=ref.dtype))
        sig = apply_awgn(sig, esn0_db=30, rng=9)

        out_sig = recovery.correct_carrier_phase(sig, recovery.DataAided())

        assert out_sig is not sig
        assert out_sig.reference is sig.reference
        residual = float(xp.abs(xp.angle(xp.mean(out_sig.samples * xp.conj(ref)))))
        assert residual < 0.02

    def test_signal_input_needs_one_sample_per_symbol(self, xp):
        """An oversampled Signal raises instead of misaligning the reference."""
        sig = generate(
            Constellation.qam(16), 256, symbol_rate=1e6, sps=2, pulse=RRC(0.35), rng=0
        )
        with pytest.raises(ValueError, match="sps=2"):
            recovery.correct_carrier_phase(sig, recovery.DataAided())

    def test_signal_input_raises_without_reference(self, xp):
        """Raises ValueError when neither DataAided nor the Signal has symbols."""
        sig = generate(
            Constellation.qam(16), 256, symbol_rate=1e6, sps=1, pulse=RRC(0.35), rng=0
        )
        sig = sig.replace(reference=None)
        with pytest.raises(ValueError, match="known symbols"):
            recovery.correct_carrier_phase(sig, recovery.DataAided())


class TestResolveChannelPermutation:
    """Stream-assignment resolution under both scoring metrics."""

    @staticmethod
    def _dual_pol(xp, n=4096, seed=7):
        """Two independent QPSK streams and the equalizer's swapped output."""
        s, _ = make_test_mimo_samples(
            num_channels=2, order=4, num_symbols=n, sps=1, snr_db=None, seed=seed, xp=xp
        )
        return s, s[::-1].copy()

    @pytest.mark.parametrize("metric", ["coherence", "phase_increment"])
    def test_resolves_swap(self, xp, xpt, metric):
        """A swapped output is reordered back to reference order."""
        ref, swapped = self._dual_pol(xp)
        out = recovery.resolve_channel_permutation(swapped, ref, metric=metric)
        xpt.assert_allclose(out, ref)

    @pytest.mark.parametrize("metric", ["coherence", "phase_increment"])
    def test_identity_is_a_no_op(self, xp, xpt, metric):
        """Already-aligned streams are returned in the same order."""
        ref, _ = self._dual_pol(xp)
        out = recovery.resolve_channel_permutation(ref, ref, metric=metric)
        xpt.assert_allclose(out, ref)

    def test_phase_increment_survives_a_frequency_offset(self, xp, xpt):
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

    def test_rejects_unknown_metric(self, xp):
        ref, swapped = self._dual_pol(xp, n=64)
        with pytest.raises(ValueError, match="metric"):
            recovery.resolve_channel_permutation(swapped, ref, metric="nonsense")


class TestLogPhaseSummary:
    """Shared "phase mean/std in degrees" INFO summary (recovery._common)."""

    def test_logs_formatted_mean_and_std(self, xp, caplog):
        """Emits the prefix/mean/std/suffix in the expected combined format."""
        import logging

        from commkit.recovery.corrections import _log_phase_summary

        phi = xp.full(4, np.pi / 2)  # constant -> mean=90 deg, std=0 deg
        with caplog.at_level(logging.INFO, logger="commkit"):
            _log_phase_summary(phi, "CPR (test, %s)", ("alg",), "[C=%s]", (1,))
        assert len(caplog.records) == 1
        msg = caplog.records[0].message
        assert "CPR (test, alg)" in msg
        assert "[C=1]" in msg
        assert "mean=90.00" in msg
        assert "std=0.00" in msg

    def test_no_op_when_info_disabled(self, xp, caplog):
        """No log line and no host transfer when INFO is off."""
        import logging
        from unittest.mock import patch

        from commkit.recovery import corrections

        phi = xp.asarray([0.0, 1.0, 2.0])
        with (
            caplog.at_level(logging.WARNING, logger="commkit"),
            patch.object(corrections, "to_device") as transfer,
        ):
            corrections._log_phase_summary(phi, "CPR (test)", (), "[]", ())
        assert len(caplog.records) == 0
        transfer.assert_not_called()


class TestVvBlockPhase:
    """Shared Viterbi-Viterbi block-phase estimator (recovery._common)."""

    def test_matches_qpsk_no_noise(self, xp, xpt):
        """Noiseless QPSK at a fixed phase offset must recover that offset exactly."""
        from commkit.recovery._common import _vv_block_phase

        offset = 0.3  # radians
        symbols = xp.asarray(
            [np.exp(1j * (k * np.pi / 2 + offset)) for k in range(64)],
            dtype=xp.complex64,
        )[None, :]  # (1, 64)

        phi_u, block_centers, all_positions = _vv_block_phase(
            symbols,
            xp,
            M=4,
            project=False,
            bias=0.0,
            block_size=16,
            joint_channels=False,
        )
        assert phi_u.shape == (1, 4)
        assert block_centers.shape == (4,)
        assert all_positions.shape == (64,)
        xpt.assert_allclose(phi_u, offset, atol=1e-4)

    def test_joint_channels_broadcasts_identical_rows(self, xp, xpt):
        """joint_channels=True must return broadcast-identical rows across C."""
        from commkit.recovery._common import _vv_block_phase

        offset = -0.2
        base = xp.asarray(
            [np.exp(1j * (k * np.pi / 2 + offset)) for k in range(32)],
            dtype=xp.complex64,
        )
        symbols = xp.stack([base, base])  # (2, 32) - identical channels

        phi_u, _, _ = _vv_block_phase(
            symbols,
            xp,
            M=4,
            project=False,
            bias=0.0,
            block_size=16,
            joint_channels=True,
        )
        assert phi_u.shape == (2, 2)
        xpt.assert_array_equal(phi_u[0], phi_u[1])
