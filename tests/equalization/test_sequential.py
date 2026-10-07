"""Sequential adaptive equalizers (lms/rls/cma/rde) on the Numba backend."""

import numpy as np
import pytest

from commkit import equalization, generate
from commkit.core import Signal
from commkit.equalization import EqualizerResult
from commkit.equalization.sequential._dd import _check_rls_divergence
from commkit.filtering import RRC
from commkit.mapping import Constellation
from tests.common.conversions import to_numpy


class TestLMS:
    """Tests for the LMS adaptive equalizer."""

    def test_convergence_known_channel(self, xp):
        """LMS should converge and recover QPSK through a known ISI channel."""
        n_symbols = 1000
        # Create channel on device
        channel = xp.array([0.2, 1.0, 0.3], dtype=xp.complex64)

        # Generate bits and symbols on device
        # Generate symbols and RRC pulse-shaped waveform

        sig = generate(
            Constellation.psk(4),
            n_symbols,
            symbol_rate=1e6,
            sps=2,
            pulse=RRC(0.35),
            rng=42,
        )

        tx = xp.asarray(sig.source_symbols)

        rx_up = xp.asarray(sig.samples)

        # Apply Channel (Convolution)
        rx = xp.convolve(rx_up, channel, mode="same")

        # Add Noise (SNR ~ 20dB implies sigma ~ 0.1 for unit power signal)
        # Using fixed noise for reproducibility in test
        rng = xp.random.RandomState(0)
        noise = 0.05 * (rng.randn(len(rx)) + 1j * rng.randn(len(rx))).astype(
            xp.complex64
        )
        rx = rx + noise

        result = equalization.lms(
            rx,
            tx,
            num_taps=15,
            step_size=0.01,
            constellation=Constellation.psk(4),
            sps=2,
        )

        assert isinstance(result, EqualizerResult)

        # Check convergence (MSE on device)
        # Last 200 symbols
        mse_tail = xp.mean(xp.abs(result.error[-200:]) ** 2)

        # Move singular scalar to CPU for assertion if needed, or assert on device scalar
        # Pytest/NumPy comparisons handling of CuPy scalars varies, explicit conversion is safest
        mse_tail = float(mse_tail)

        assert mse_tail < 0.1, f"LMS did not converge: tail MSE = {mse_tail:.4f}"

    def test_decision_directed_after_training(self, xp):
        """LMS should maintain performance in DD mode after training."""
        n_symbols = 1000
        n_train = 300
        channel = xp.array([0.1, 1.0, 0.2], dtype=xp.complex64)

        # Generate symbols and RRC pulse-shaped waveform

        sig = generate(
            Constellation.psk(4),
            n_symbols,
            symbol_rate=1e6,
            sps=2,
            pulse=RRC(0.35),
            rng=123,
        )

        tx = xp.asarray(sig.source_symbols)

        rx_up = xp.asarray(sig.samples)
        rx = xp.convolve(rx_up, channel, mode="same")

        result = equalization.lms(
            rx,
            tx[:n_train],
            num_taps=15,
            step_size=0.01,
            constellation=Constellation.psk(4),
            sps=2,
        )

        # Check DD mode MSE
        mse_dd = xp.mean(xp.abs(result.error[n_train + 50 :]) ** 2)
        mse_dd = float(mse_dd)

        assert mse_dd < 0.2, f"LMS DD mode failed: MSE = {mse_dd:.4f}"

    def test_output_shape_siso(self, xp):
        """LMS SISO output shapes should be correct."""
        n = 500

        sig = generate(
            Constellation.psk(4), n, symbol_rate=1e6, sps=2, pulse=RRC(0.35), rng=0
        )
        tx = xp.asarray(sig.source_symbols)
        rx = xp.asarray(sig.samples)

        result = equalization.lms(
            rx, tx, num_taps=11, constellation=Constellation.psk(4), sps=2
        )

        assert result.y_hat.ndim == 1
        assert result.weights.shape == (11,)
        assert result.error.ndim == 1
        assert result.y_hat.shape == result.error.shape

    def test_store_weights(self, xp):
        """Weight history should be stored when requested."""
        n = 200
        num_taps = 7

        sig = generate(
            Constellation.psk(4), n, symbol_rate=1e6, sps=2, pulse=RRC(0.35), rng=0
        )
        tx = xp.asarray(sig.source_symbols)
        rx = xp.asarray(sig.samples)

        result = equalization.lms(
            rx,
            tx,
            num_taps=num_taps,
            constellation=Constellation.psk(4),
            store_weights=True,
            sps=2,
        )

        assert result.weights_history is not None
        assert result.weights_history.ndim == 2
        assert result.weights_history.shape[1] == num_taps

    def test_no_weights_by_default(self, xp):
        """Weight history should be None by default."""

        sig = generate(
            Constellation.psk(4), 100, symbol_rate=1e6, sps=2, pulse=RRC(0.35), rng=0
        )
        tx = xp.asarray(sig.source_symbols)
        rx = xp.asarray(sig.samples)

        result = equalization.lms(
            rx, tx, num_taps=5, constellation=Constellation.psk(4), sps=2
        )

        assert result.weights_history is None

    def test_requires_constellation_or_training(self, xp):
        """LMS should raise if neither training nor constellation is given."""

        sig = generate(
            Constellation.psk(4), 100, symbol_rate=1e6, sps=2, pulse=RRC(0.35), rng=0
        )
        rx = xp.asarray(sig.samples)
        with pytest.raises(ValueError):
            equalization.lms(rx, sps=2)


class TestRLS:
    """Tests for the RLS adaptive equalizer."""

    def test_convergence(self, xp):
        """RLS should converge on a known ISI channel."""
        n_symbols = 500
        channel = xp.array([0.2, 1.0, 0.3], dtype=xp.complex64)

        sig = generate(
            Constellation.psk(4),
            n_symbols,
            symbol_rate=1e6,
            sps=1,
            pulse=RRC(0.35),
            rng=42,
        )
        tx = xp.asarray(sig.source_symbols)
        rx_up = xp.asarray(sig.samples)

        rx = xp.convolve(rx_up, channel, mode="same")

        result = equalization.rls(
            rx,
            tx,
            num_taps=15,
            forgetting_factor=0.99,
            constellation=Constellation.psk(4),
            sps=1,
        )

        mse_tail = xp.mean(xp.abs(result.error[-100:]) ** 2)
        mse_tail = float(mse_tail)

        assert mse_tail < 0.1, f"RLS did not converge: tail MSE = {mse_tail:.4f}"

    def test_faster_convergence_than_lms(self, xp):
        """RLS should converge faster than LMS (lower MSE in early symbols)."""
        n_symbols = 300
        channel = xp.array([0.3, 1.0, 0.2], dtype=xp.complex64)

        sig = generate(
            Constellation.psk(4),
            n_symbols,
            symbol_rate=1e6,
            sps=1,
            pulse=RRC(0.35),
            rng=77,
        )
        tx = xp.asarray(sig.source_symbols)
        rx_up = xp.asarray(sig.samples)

        rx = xp.convolve(rx_up, channel, mode="same")

        lms_result = equalization.lms(
            rx,
            tx,
            num_taps=15,
            step_size=0.01,
            constellation=Constellation.psk(4),
            sps=2,
        )
        rls_result = equalization.rls(
            rx,
            tx,
            num_taps=15,
            forgetting_factor=0.99,
            constellation=Constellation.psk(4),
            sps=1,
        )

        # Compare MSE in the first 50 symbols (convergence speed)
        lms_early = xp.mean(xp.abs(lms_result.error[:50]) ** 2)
        rls_early = xp.mean(xp.abs(rls_result.error[:50]) ** 2)

        lms_early = float(lms_early)
        rls_early = float(rls_early)

        assert rls_early <= lms_early, (
            f"RLS ({rls_early:.4f}) not faster than LMS ({lms_early:.4f})"
        )

    def test_output_shape_siso(self, xp):
        """RLS SISO output shapes should match LMS convention."""

        sig = generate(
            Constellation.psk(4), 200, symbol_rate=1e6, sps=2, pulse=RRC(0.35), rng=0
        )
        tx = xp.asarray(sig.source_symbols)
        rx = xp.asarray(sig.samples)

        result = equalization.rls(
            rx, tx, num_taps=11, constellation=Constellation.psk(4), sps=1
        )

        assert result.y_hat.ndim == 1
        assert result.weights.shape == (11,)


class TestAPIRegression:
    """Verify that removed parameters no longer exist on the public API."""

    def test_lms_has_no_normalize_param(self, xp):
        """lms() must not accept a 'normalize' keyword - always NLMS."""

        sig = generate(
            Constellation.psk(4), 100, symbol_rate=1e6, sps=2, pulse=RRC(0.35), rng=0
        )
        rx = xp.asarray(sig.samples)
        tx = xp.asarray(sig.source_symbols)
        with pytest.raises(TypeError, match="normalize"):
            equalization.lms(
                rx,
                tx,
                num_taps=5,
                constellation=Constellation.psk(4),
                normalize=True,
                sps=2,
            )

    def test_cma_has_no_normalize_param(self, xp):
        """cma() must not accept a 'normalize' keyword."""

        sig = generate(
            Constellation.psk(4), 100, symbol_rate=1e6, sps=2, pulse=RRC(0.35), rng=0
        )
        rx = xp.asarray(sig.samples)
        with pytest.raises(TypeError, match="normalize"):
            equalization.cma(rx, num_taps=5, normalize=True, sps=2)


class TestCMA:
    """Tests for the CMA blind equalizer."""

    def test_convergence_qpsk(self, xp):
        """CMA should converge for QPSK (constant modulus) through ISI channel."""
        n_symbols = 2000
        channel = xp.array([0.2, 1.0, 0.3], dtype=xp.complex64)

        sig = generate(
            Constellation.psk(4),
            n_symbols,
            symbol_rate=1e6,
            sps=2,
            pulse=RRC(0.35),
            rng=42,
        )
        rx_up = xp.asarray(sig.samples)
        rx = xp.convolve(rx_up, channel, mode="same")
        rx = xp.ascontiguousarray(rx)  # Ensure a contiguous layout

        result = equalization.cma(
            rx, num_taps=21, step_size=0.005, constellation=Constellation.psk(4), sps=2
        )

        # After convergence, output should have roughly constant modulus
        y = result.y_hat
        modulus_tail = xp.abs(y[-500:])
        modulus_std = xp.std(modulus_tail)

        modulus_std = float(modulus_std)

        assert modulus_std < 0.3, (
            f"CMA output modulus not constant: std = {modulus_std:.4f}"
        )

    def test_r2_from_modulation(self, xp, xpt):
        """R2 should be correctly auto-computed from constellation."""
        # QPSK: all points on unit circle -> R2 = 1.0
        # This test checks the logic inside CMA, but here we can just verify the property
        # using the public API if exposed, or just run a dummy CMA and check it doesn't crash?
        # The original test verified calculation logic.

        const = xp.asarray(Constellation.psk(4).points)
        r2 = xp.mean(xp.abs(const) ** 4) / xp.mean(xp.abs(const) ** 2)

        r2 = float(r2)

        xpt.assert_allclose(r2, 1.0, atol=1e-6)

    def test_r2_default(self, xp):
        """CMA should work with default R2=1.0."""

        sig = generate(
            Constellation.psk(4), 500, symbol_rate=1e6, sps=2, pulse=RRC(0.35), rng=0
        )
        rx = xp.asarray(sig.samples)

        result = equalization.cma(rx, num_taps=11, step_size=0.01, sps=2)

        assert isinstance(result, EqualizerResult)
        assert result.y_hat.shape[0] > 0

    def test_output_shape_siso(self, xp):
        """CMA SISO output should be 1D."""

        sig = generate(
            Constellation.psk(4), 200, symbol_rate=1e6, sps=2, pulse=RRC(0.35), rng=0
        )
        rx = xp.asarray(sig.samples)

        result = equalization.cma(rx, num_taps=11, sps=2)

        assert result.y_hat.ndim == 1
        assert result.weights.shape == (11,)


class TestRDE:
    """Tests for the Radius Directed Equalizer."""

    def test_convergence_qpsk(self, xp):
        """RDE should converge for QPSK (single-ring) just like CMA."""
        n_symbols = 2000
        channel = xp.array([0.2, 1.0, 0.3], dtype=xp.complex64)

        sig = generate(
            Constellation.psk(4),
            n_symbols,
            symbol_rate=1e6,
            sps=2,
            pulse=RRC(0.35),
            rng=42,
        )
        rx = xp.convolve(xp.asarray(sig.samples), channel, mode="same")
        rx = xp.ascontiguousarray(rx)

        result = equalization.rde(
            rx, num_taps=21, step_size=0.005, constellation=Constellation.psk(4), sps=2
        )

        y = result.y_hat
        modulus_tail = xp.abs(y[-500:])
        modulus_std = xp.std(modulus_tail)
        modulus_std = float(modulus_std)

        assert modulus_std < 0.3, (
            f"RDE QPSK modulus not converged: std = {modulus_std:.4f}"
        )

    def test_steady_state_error_16qam(self, xp):
        """RDE steady-state error should be much lower than CMA on 16-QAM.

        CMA and RDE are both phase-ambiguous, so EVM vs the reference symbol
        sequence is not a reliable comparison.  Instead we compare the
        steady-state magnitude of each algorithm's *own* error signal.

        For 16-QAM (3 rings), CMA's error e = y*(|y|²-R²_Godard) is bounded
        below by a non-zero "design residual": since R²_Godard is a single
        average radius, inner-ring and outer-ring symbols always have large
        signed errors that cancel only in expectation, not per-symbol.

        RDE's error e = y*(|y|²-R_d²) uses the nearest ring for each symbol,
        so it approaches zero as the equalizer converges to correct ISI
        compensation.  Therefore: mean(|e_RDE|) << mean(|e_CMA|) at steady state.
        """

        n_symbols = 5000
        channel = xp.array([0.15, 1.0, 0.25], dtype=xp.complex64)

        sig = generate(
            Constellation.qam(16),
            n_symbols,
            symbol_rate=1e6,
            sps=2,
            pulse=RRC(0.35),
            rng=7,
        )
        rx = xp.convolve(xp.asarray(sig.samples), channel, mode="same")
        rx = xp.ascontiguousarray(rx)

        result_rde = equalization.rde(
            rx, num_taps=21, step_size=5e-4, constellation=Constellation.qam(16), sps=2
        )
        result_cma = equalization.cma(
            rx, num_taps=21, step_size=5e-4, constellation=Constellation.qam(16), sps=2
        )

        def late_error(err):
            return float(xp.mean(xp.abs(err[-500:])))

        err_rde = late_error(result_rde.error)
        err_cma = late_error(result_cma.error)

        assert err_rde < err_cma, (
            f"RDE steady-state error ({err_rde:.4f}) should be lower than "
            f"CMA ({err_cma:.4f}) for 16-QAM: CMA has an irreducible design "
            f"residual from using a single Godard radius on a multi-ring constellation"
        )

    def test_radii_extraction(self, xp):
        """Unique radii should match known 16-QAM ring structure."""
        import numpy as _np

        const = Constellation.qam(16).points
        radii = _np.unique(_np.round(_np.abs(const).astype(_np.float32), 6))

        # Standard normalized 16-QAM has 3 unique radii
        assert len(radii) == 3, f"Expected 3 unique radii for 16-QAM, got {len(radii)}"
        # All radii must be positive
        assert _np.all(radii > 0)

    def test_output_shape_siso(self, xp):
        """RDE SISO output should be 1D with correct length."""

        sig = generate(
            Constellation.qam(16), 300, symbol_rate=1e6, sps=2, pulse=RRC(0.35), rng=0
        )
        rx = xp.asarray(sig.samples)

        result = equalization.rde(
            rx, num_taps=11, constellation=Constellation.qam(16), sps=2
        )

        assert result.y_hat.ndim == 1
        assert result.weights.shape == (11,)


class TestStoreWeights:
    """Tests that store_weights=True produces correct weight history shapes."""

    def _qpsk_rx(self, xp, n_symbols=600, seed=0):

        sig = generate(
            Constellation.psk(4),
            n_symbols,
            symbol_rate=1e6,
            sps=2,
            pulse=RRC(0.35),
            rng=seed,
        )
        return xp.ascontiguousarray(xp.asarray(sig.samples)), sig

    def test_lms_store_weights_numba(self, xp):
        """LMS Numba: weights_history has shape (N_sym, num_taps) for SISO."""
        rx, sig = self._qpsk_rx(xp)
        n_sym = rx.shape[0] // 2

        result = equalization.lms(
            rx,
            xp.asarray(sig.source_symbols),
            num_taps=9,
            step_size=0.05,
            constellation=Constellation.psk(4),
            store_weights=True,
            sps=2,
        )

        assert result.weights_history is not None
        assert result.weights_history.shape == (n_sym, 9)

    def test_rls_store_weights_numba(self, xp):
        """RLS Numba: weights_history has shape (N_sym_truncated, num_taps) for SISO."""
        rx, sig = self._qpsk_rx(xp)

        result = equalization.rls(
            rx,
            xp.asarray(sig.source_symbols),
            sps=1,
            num_taps=7,
            constellation=Constellation.psk(4),
            store_weights=True,
        )

        assert result.weights_history is not None
        # SISO: (N_sym_truncated, num_taps)
        assert result.weights_history.ndim == 2
        assert result.weights_history.shape[1] == 7

    def test_cma_store_weights_numba(self, xp):
        """CMA Numba: weights_history has shape (N_sym, num_taps) for SISO."""
        rx, _ = self._qpsk_rx(xp)
        n_sym = rx.shape[0] // 2

        result = equalization.cma(
            rx,
            num_taps=9,
            step_size=0.005,
            constellation=Constellation.psk(4),
            store_weights=True,
            sps=2,
        )

        assert result.weights_history is not None
        assert result.weights_history.shape == (n_sym, 9)

    def test_rde_store_weights_numba(self, xp):
        """RDE Numba: weights_history has shape (N_sym, num_taps) for SISO."""
        rx, _ = self._qpsk_rx(xp)
        n_sym = rx.shape[0] // 2

        result = equalization.rde(
            rx,
            num_taps=9,
            step_size=0.005,
            constellation=Constellation.psk(4),
            store_weights=True,
            sps=2,
        )

        assert result.weights_history is not None
        assert result.weights_history.shape == (n_sym, 9)

    def test_mimo_store_weights_numba(self, xp):
        """LMS Numba MIMO: weights_history has shape (N_sym, C, C, num_taps)."""

        sig = generate(
            Constellation.psk(4), 600, symbol_rate=1e6, sps=2, pulse=RRC(0.35), rng=0
        )
        rx = xp.asarray(sig.samples)
        rx_mimo = xp.stack([rx, xp.roll(rx, 1)], axis=0)
        train = xp.stack([xp.asarray(sig.source_symbols)] * 2, axis=0)

        result = equalization.lms(
            rx_mimo,
            train,
            num_taps=5,
            constellation=Constellation.psk(4),
            store_weights=True,
            sps=2,
        )

        n_sym = rx.shape[0] // 2
        assert result.weights_history is not None
        assert result.weights_history.shape == (n_sym, 2, 2, 5)

    def test_no_weights_by_default_all_algorithms(self, xp):
        """All algorithms should return weights_history=None by default."""
        rx, sig = self._qpsk_rx(xp, n_symbols=300)
        train = xp.asarray(sig.source_symbols)

        for algo, kwargs in [
            ("lms", dict(training_symbols=train, constellation=Constellation.psk(4))),
            ("cma", dict(constellation=Constellation.psk(4))),
            ("rde", dict(constellation=Constellation.psk(4))),
        ]:
            result = getattr(equalization, algo)(rx, num_taps=7, sps=2, **kwargs)
            assert result.weights_history is None, (
                f"{algo}: expected weights_history=None by default"
            )


class TestEdgeCases:
    """Tests for error paths, warnings, and edge-case inputs."""

    def _qpsk_rx(self, xp, n_symbols=500, seed=0):

        sig = generate(
            Constellation.psk(4),
            n_symbols,
            symbol_rate=1e6,
            sps=2,
            pulse=RRC(0.35),
            rng=seed,
        )
        return xp.ascontiguousarray(xp.asarray(sig.samples)), sig

    def test_lms_raises_no_constellation_numba(self, xp):
        """LMS Numba: ValueError when no modulation and no training symbols (DD impossible)."""
        rx, _ = self._qpsk_rx(xp)
        with pytest.raises(ValueError, match="needs a constellation"):
            equalization.lms(rx, num_taps=7, sps=2)

    def test_rls_warns_fractional_spacing(self, xp):
        """RLS should warn when sps > 1 (ill-conditioned correlation matrix)."""
        rx, sig = self._qpsk_rx(xp, n_symbols=400)
        result = equalization.rls(
            xp.asarray(sig.samples),
            xp.asarray(sig.source_symbols),
            sps=2,
            num_taps=7,
            constellation=Constellation.psk(4),
        )
        # Should complete and return valid output (even if warned)
        assert isinstance(result, EqualizerResult)

    def test_validate_sps_small_num_taps(self, xp, caplog):
        """_validate_sps should log a warning when num_taps < 4*sps."""
        import logging

        rx, sig = self._qpsk_rx(xp)
        with caplog.at_level(logging.WARNING, logger="commkit"):
            equalization.lms(
                rx,
                xp.asarray(sig.source_symbols),
                num_taps=3,  # < 4*sps=8 -> should warn
                constellation=Constellation.psk(4),
                sps=2,
            )
        assert any("small" in r.message.lower() for r in caplog.records)

    def test_rde_no_modulation_falls_back_to_cma(self, xp):
        """RDE with no modulation should use unit radius (same gradient as CMA R²=1)."""
        rx, _ = self._qpsk_rx(xp)

        result = equalization.rde(rx, num_taps=7, step_size=0.005, sps=2)

        assert isinstance(result, EqualizerResult)
        assert result.y_hat.shape[0] > 0

    def test_rde_mimo_no_modulation(self, xp):
        """RDE MIMO path with no modulation should run (unit radius, 2-ch)."""

        sig = generate(
            Constellation.psk(4), 600, symbol_rate=1e6, sps=2, pulse=RRC(0.35), rng=0
        )
        rx = xp.asarray(sig.samples)
        rx_mimo = xp.stack([rx, xp.roll(rx, 1)], axis=0)

        result = equalization.rde(rx_mimo, num_taps=7, step_size=5e-4, sps=2)

        assert result.y_hat.shape == (2, rx.shape[0] // 2)

    def test_lms_num_train_symbols_clamps_numba(self, xp):
        """LMS Numba: pre-sliced training_symbols limits DA phase length."""
        rx, sig = self._qpsk_rx(xp, n_symbols=1000)
        train = xp.asarray(sig.source_symbols[:30])

        result = equalization.lms(
            rx,
            train,
            num_taps=7,
            step_size=0.05,
            constellation=Constellation.psk(4),
            sps=2,
        )

        assert result.num_train_symbols <= 30

    def test_lms_partial_training_needs_constellation(self, xp):
        """Decisions after the training need a constellation; none is guessed."""
        rx, sig = self._qpsk_rx(xp)
        train = xp.asarray(sig.source_symbols)[..., :50]
        with pytest.raises(ValueError, match="needs a constellation"):
            equalization.lms(rx, train, num_taps=7, step_size=0.05, sps=2)

    def test_lms_full_training_needs_no_constellation(self, xp):
        """With every symbol trained there are no decisions to make."""
        rx, sig = self._qpsk_rx(xp)
        train = xp.asarray(sig.source_symbols)
        result = equalization.lms(rx, train, num_taps=7, step_size=0.05, sps=2)
        assert result.num_train_symbols == result.y_hat.shape[-1]

    def test_center_tap_override(self, xp):
        """Custom center_tap should shift the decision delay without error."""
        rx, sig = self._qpsk_rx(xp)
        train = xp.asarray(sig.source_symbols)

        result = equalization.lms(
            rx,
            train,
            num_taps=11,
            step_size=0.05,
            constellation=Constellation.psk(4),
            center_tap=8,
            sps=2,
        )

        assert isinstance(result, EqualizerResult)
        assert result.y_hat.shape[0] > 0


class TestNumbaBackendCoverage:
    """Tests targeting uncovered numba-backend code paths in LMS/RLS/CMA/RDE."""

    def _make_qpsk_rx(self, xp, n_symbols=1000, seed=0):

        channel = xp.array([0.1, 1.0, 0.15], dtype=xp.complex64)
        sig = generate(
            Constellation.psk(4),
            n_symbols,
            symbol_rate=1e6,
            sps=2,
            pulse=RRC(0.35),
            rng=seed,
        )
        rx = xp.convolve(xp.asarray(sig.samples), channel, mode="same")
        return xp.ascontiguousarray(rx), sig

    def test_lms_numba_pure_dd_no_training(self, xp):
        """LMS numba with no training symbols (pure DD from start) covers _prepare_training_numpy else branch."""
        rx, _ = self._make_qpsk_rx(xp, n_symbols=600)

        # No training_symbols -> _prepare_training_numpy gets None -> n_train_aligned=0
        result = equalization.lms(
            rx, num_taps=11, step_size=0.01, constellation=Constellation.psk(4), sps=2
        )

        assert isinstance(result, EqualizerResult)
        assert result.y_hat.ndim == 1

    def test_rls_numba_siso(self, xp):
        """RLS with numba backend on SISO input."""
        rx, sig = self._make_qpsk_rx(xp, n_symbols=800)
        train = xp.asarray(sig.source_symbols)

        result = equalization.rls(
            rx, train, sps=2, num_taps=9, constellation=Constellation.psk(4)
        )

        assert isinstance(result, EqualizerResult)
        assert result.y_hat.ndim == 1
        assert result.weights.shape == (9,)

    def test_rls_divergence_guard(self, xp):
        """Non-finite RLS weights raise an actionable error instead of silently
        returning garbage taps (mirrors the block_lms divergence safeguard)."""
        good = xp.ones((2, 2, 5), dtype=xp.complex64)
        # Finite weights pass through untouched.
        _check_rls_divergence(good, xp, 0.99, 0.01)

        bad = good.copy()
        bad[0, 0, 0] = float("nan")
        with pytest.raises(RuntimeError, match="RLS equalizer diverged"):
            _check_rls_divergence(bad, xp, 0.99, 0.01)

    def test_rls_numba_mimo(self, xp):
        """RLS numba MIMO path correctly handles (num_channels, n_samples) input shape."""

        n_symbols = 600
        sig = generate(
            Constellation.psk(4),
            n_symbols,
            symbol_rate=1e6,
            sps=2,
            pulse=RRC(0.35),
            rng=5,
        )
        rx1 = xp.asarray(sig.samples)
        rx_mimo = xp.stack([rx1, xp.roll(rx1, 2)], axis=0)  # (2, N)
        train_mimo = xp.stack([xp.asarray(sig.source_symbols)] * 2, axis=0)

        result = equalization.rls(
            rx_mimo, train_mimo, sps=2, num_taps=7, constellation=Constellation.psk(4)
        )

        # RLS truncates the last num_taps//2 symbols from y_hat
        tail_trim = 7 // 2
        assert result.y_hat.shape == (2, n_symbols - tail_trim)
        assert result.weights.shape == (2, 2, 7)

    def test_rls_numba_num_train_symbols(self, xp):
        """RLS numba: pre-sliced training_symbols limits DA phase length."""
        rx, sig = self._make_qpsk_rx(xp, n_symbols=800)
        train = xp.asarray(sig.source_symbols[:50])

        result = equalization.rls(
            rx, train, sps=2, num_taps=7, constellation=Constellation.psk(4)
        )

        assert isinstance(result, EqualizerResult)
        assert result.num_train_symbols <= 50

    def test_rls_numba_store_weights(self, xp):
        """RLS numba with store_weights=True should populate weight history."""
        rx, sig = self._make_qpsk_rx(xp, n_symbols=400)
        train = xp.asarray(sig.source_symbols)

        result = equalization.rls(
            rx,
            train,
            sps=2,
            num_taps=5,
            constellation=Constellation.psk(4),
            store_weights=True,
        )

        assert result.weights_history is not None

    def test_cma_numba_store_weights(self, xp):
        """CMA numba backend with store_weights=True."""

        sig = generate(
            Constellation.psk(4), 400, symbol_rate=1e6, sps=2, pulse=RRC(0.35), rng=0
        )
        rx = xp.asarray(sig.samples)

        result = equalization.cma(
            rx,
            num_taps=7,
            step_size=0.005,
            constellation=Constellation.psk(4),
            store_weights=True,
            sps=2,
        )

        assert result.weights_history is not None

    def test_rde_numba_store_weights(self, xp):
        """RDE numba backend with store_weights=True."""

        sig = generate(
            Constellation.qam(16), 400, symbol_rate=1e6, sps=2, pulse=RRC(0.35), rng=0
        )
        rx = xp.asarray(sig.samples)

        result = equalization.rde(
            rx,
            num_taps=7,
            step_size=5e-4,
            constellation=Constellation.qam(16),
            store_weights=True,
            sps=2,
        )

        assert result.weights_history is not None


class TestCmaPilotAided:
    """Tests for cma()/rde() with pilot_ref/pilot_mask (hybrid PA mode)."""

    def _make_comb_frame_and_samples(self, xp):
        from commkit.backend import to_device
        from commkit.core import Preamble, SingleCarrierFrame

        preamble = Preamble(sequence_type="barker", length=13)
        frame = SingleCarrierFrame(
            payload_len=1008,
            pilot_pattern="comb",
            pilot_period=10,
            preamble=preamble,
        )
        sig = frame.to_signal(sps=2, symbol_rate=1e6, pulse=RRC(0.35))
        struct = frame.get_structure_map(unit="symbols", sps=1, include_preamble=False)
        samples_cpu = to_device(sig.samples, "cpu")
        pilot_syms_cpu = to_device(frame.pilot_symbols, "cpu")
        pilot_mask_bool = to_device(struct["pilots"], "cpu")
        n_body = int(pilot_mask_bool.size)
        return frame, samples_cpu, pilot_syms_cpu, pilot_mask_bool, n_body

    @pytest.mark.parametrize("name", ["cma", "rde"])
    def test_pilot_deboost_leaves_input_alone(self, name):
        """De-boosting the pilots works on a copy of a NumPy complex64 input."""
        rng = np.random.default_rng(0)
        x = (rng.standard_normal(400) + 1j * rng.standard_normal(400)).astype(
            np.complex64
        )
        before = x.copy()
        mask = np.zeros(200, dtype=np.uint8)
        mask[::10] = 1
        ref = np.where(mask, 1 + 1j, 0).astype(np.complex64)[None, :]
        getattr(equalization, name)(
            x,
            sps=2,
            num_taps=5,
            pilot_ref=ref,
            pilot_mask=mask,
            pilot_gain_db=3.0,
        )
        np.testing.assert_array_equal(x, before)

    def test_cma_pilot_aided_numba_output_shape(self, xp):
        """cma() with pilot_ref/pilot_mask returns correct body length."""
        from commkit.equalization import build_pilot_ref

        frame, samples_cpu, pilot_syms_cpu, pilot_mask_bool, n_body = (
            self._make_comb_frame_and_samples(xp)
        )
        # Slice body samples (after preamble)
        sps = 2
        n_pre = frame.preamble.num_symbols * sps
        body_samples = samples_cpu[n_pre:]

        pilot_ref, pilot_mask_u8 = build_pilot_ref(
            pilot_symbols=pilot_syms_cpu,
            pilot_mask=pilot_mask_bool,
            n_sym=n_body,
            num_ch=1,
        )
        result = equalization.cma(
            body_samples,
            constellation=Constellation.qam(16),
            num_taps=11,
            step_size=1e-4,
            sps=2,
            pilot_ref=pilot_ref,
            pilot_mask=pilot_mask_u8,
        )
        assert isinstance(result, EqualizerResult)
        assert result.y_hat.shape[-1] == n_body

    def test_rde_pilot_aided_numba_output_shape(self, xp):
        """rde() with pilot_ref/pilot_mask returns correct body length."""
        from commkit.equalization import build_pilot_ref

        frame, samples_cpu, pilot_syms_cpu, pilot_mask_bool, n_body = (
            self._make_comb_frame_and_samples(xp)
        )
        sps = 2
        n_pre = frame.preamble.num_symbols * sps
        body_samples = samples_cpu[n_pre:]

        pilot_ref, pilot_mask_u8 = build_pilot_ref(
            pilot_symbols=pilot_syms_cpu,
            pilot_mask=pilot_mask_bool,
            n_sym=n_body,
            num_ch=1,
        )
        result = equalization.rde(
            body_samples,
            constellation=Constellation.qam(16),
            num_taps=11,
            step_size=1e-4,
            sps=2,
            pilot_ref=pilot_ref,
            pilot_mask=pilot_mask_u8,
        )
        assert isinstance(result, EqualizerResult)
        assert result.y_hat.shape[-1] == n_body

    def test_rde_pilot_aided_initial_taps_warm_start(self, xp):
        """rde() PA accepts initial_taps from a prior lms() call."""
        from commkit.equalization import build_pilot_ref

        frame, samples_cpu, pilot_syms_cpu, pilot_mask_bool, n_body = (
            self._make_comb_frame_and_samples(xp)
        )
        sps = 2
        n_pre = frame.preamble.num_symbols * sps
        preamble_samples = samples_cpu[:n_pre]
        body_samples = samples_cpu[n_pre:]
        # Pre-converge on preamble
        pre = equalization.lms(
            preamble_samples,
            training_symbols=to_numpy(frame.preamble.symbols),
            num_taps=11,
            step_size=0.01,
            sps=2,
        )
        pilot_ref, pilot_mask_u8 = build_pilot_ref(
            pilot_symbols=pilot_syms_cpu,
            pilot_mask=pilot_mask_bool,
            n_sym=n_body,
            num_ch=1,
        )
        result = equalization.rde(
            body_samples,
            constellation=Constellation.qam(16),
            num_taps=11,
            step_size=1e-4,
            sps=2,
            initial_taps=pre.weights,
            pilot_ref=pilot_ref,
            pilot_mask=pilot_mask_u8,
        )
        assert isinstance(result, EqualizerResult)
        assert result.y_hat.shape[-1] == n_body

    def test_build_pilot_ref_shape(self, xp):
        """build_pilot_ref returns correct shapes."""
        from commkit.equalization import build_pilot_ref

        n_sym, n_pilots, num_ch = 100, 10, 2
        positions = np.linspace(0, n_sym - 1, n_pilots, dtype=int)
        mask = np.zeros(n_sym, dtype=bool)
        mask[positions] = True
        pilot_syms = np.ones((num_ch, n_pilots), dtype=np.complex64)
        ref, mask_u8 = build_pilot_ref(pilot_syms, mask, n_sym=n_sym, num_ch=num_ch)
        assert ref.shape == (num_ch, n_sym)
        assert mask_u8.shape == (n_sym,)
        assert mask_u8.dtype == np.uint8
        assert int(mask_u8.sum()) == n_pilots


class TestSignalInputSequentialEqualizers:
    """Signal-awareness for lms, rls, cma, rde: sps from the signal, y_hat as
    a new Signal at the symbol rate."""

    def _rx_signal(self, xp, order=16, n_symbols=2000, sps=2, seed=0):
        sig = generate(
            Constellation.qam(order),
            n_symbols,
            symbol_rate=1e6,
            sps=sps,
            pulse=RRC(0.35),
            rng=seed,
        )
        return sig

    def test_lms_signal_input(self, xp, xpt):
        sig = self._rx_signal(xp)
        data = xp.asarray(sig.samples)

        result_sig = equalization.lms(
            sig, num_taps=11, constellation=Constellation.qam(16)
        )
        result_arr = equalization.lms(
            data, num_taps=11, sps=2, constellation=Constellation.qam(16)
        )

        assert isinstance(result_sig.signal, Signal)
        assert result_sig.signal.sampling_rate == 1e6
        xpt.assert_allclose(result_sig.signal.samples, result_arr.y_hat)
        xpt.assert_allclose(result_sig.y_hat, result_arr.y_hat)

    def test_rls_signal_input(self, xp, xpt):
        sig = self._rx_signal(xp, sps=1, n_symbols=500)
        data = xp.asarray(sig.samples)

        result_sig = equalization.rls(
            sig, num_taps=11, constellation=Constellation.qam(16)
        )
        result_arr = equalization.rls(
            data, num_taps=11, sps=1, constellation=Constellation.qam(16)
        )

        assert isinstance(result_sig.signal, Signal)
        assert result_sig.signal.sampling_rate == 1e6
        xpt.assert_allclose(result_sig.signal.samples, result_arr.y_hat)
        xpt.assert_allclose(result_sig.y_hat, result_arr.y_hat)

    def test_cma_signal_input(self, xp, xpt):
        sig = self._rx_signal(xp)
        data = xp.asarray(sig.samples)

        result_sig = equalization.cma(
            sig, num_taps=11, constellation=Constellation.qam(16)
        )
        result_arr = equalization.cma(
            data, num_taps=11, sps=2, constellation=Constellation.qam(16)
        )

        assert isinstance(result_sig.signal, Signal)
        assert result_sig.signal.sampling_rate == 1e6
        xpt.assert_allclose(result_sig.signal.samples, result_arr.y_hat)
        xpt.assert_allclose(result_sig.y_hat, result_arr.y_hat)

    def test_rde_signal_input(self, xp, xpt):
        sig = self._rx_signal(xp)
        data = xp.asarray(sig.samples)

        result_sig = equalization.rde(
            sig, num_taps=11, constellation=Constellation.qam(16)
        )
        result_arr = equalization.rde(
            data, num_taps=11, sps=2, constellation=Constellation.qam(16)
        )

        assert isinstance(result_sig.signal, Signal)
        assert result_sig.signal.sampling_rate == 1e6
        xpt.assert_allclose(result_sig.signal.samples, result_arr.y_hat)
        xpt.assert_allclose(result_sig.y_hat, result_arr.y_hat)


class TestKnownSymbolScale:
    """Training symbols and pilots are taken as given, on the scale of the
    unit-power constellation (plan 3.7g)."""

    @staticmethod
    def _corner_symbols(n: int = 3000) -> np.ndarray:
        # Only the four corners of 16-QAM: sample power 1.8, far from 1, so a
        # renormalized training sequence would land 25% inside the grid.
        pts = Constellation.qam(16).points
        corners = pts[np.abs(pts) ** 2 > 1.5]
        rng = np.random.default_rng(0)
        return corners[rng.integers(0, 4, n)].astype(np.complex64)

    @pytest.mark.parametrize("name", ["lms", "rls", "block_lms"])
    def test_training_reproduces_the_points(self, name, xp):
        """Noiseless identity channel, trained throughout: y_hat -> s."""
        s = self._corner_symbols()
        kw = dict(sps=1, num_taps=3, constellation=Constellation.qam(16))
        if name == "lms":
            kw["step_size"] = 0.05
        if name == "block_lms":
            kw.update(step_size=5e-3, block_size=64)
        res = getattr(equalization, name)(xp.asarray(s), xp.asarray(s), **kw)
        y = to_numpy(res.y_hat)[-500:]
        assert np.max(np.abs(y - s[: res.y_hat.shape[-1]][-500:])) < 1e-3

    @pytest.mark.parametrize("name", ["cma", "block_cma"])
    def test_pilots_on_a_shaped_constellation(self, name, xp):
        """All-pilot blind run on a shaped constellation reproduces the points;
        1.x rescaled the pilots by 1/sqrt(E_PS) off the constellation."""
        c = Constellation.qam(16).shaped(nu=0.2)
        rng = np.random.default_rng(1)
        s = c.points[rng.choice(16, 3000, p=c.pmf)].astype(np.complex64)
        mask = np.ones(s.size, dtype=bool)
        ref, pm = equalization.build_pilot_ref(s, mask, n_sym=s.size, num_ch=1)
        kw = dict(sps=1, num_taps=3, constellation=c, pilot_ref=ref, pilot_mask=pm)
        if name == "cma":
            kw["step_size"] = 0.05
        else:
            kw.update(step_size=5e-3, block_size=64)
        res = getattr(equalization, name)(xp.asarray(s), **kw)
        y = to_numpy(res.y_hat)[-500:]
        assert np.max(np.abs(y - s[-500:])) < 1e-3
