"""Tests for block_lms - frequency-domain block LMS equalizer.

Coverage:
  1. Output shapes - SISO and MIMO, with/without CPR
  2. Gradient descent - MSE decreases over blocks (identity channel, noise)
  3. Identity channel parity - with sufficient training, output ≈ input symbols
  4. w_init passthrough - warm-start weights are used
  5. store_weights shape - weights_history has expected layout
  6. num_train_symbols boundary - DA/DD switch is respected
  7. Last-block edge - n_sym not a multiple of block_size
  8. cpr_type validation - pll raises ValueError
  9. BPS + CPR - phase_trajectory shape and MSE better than no CPR under phase noise
 10. BPS block_size vs cpr_bps_block_size independence - different values accepted
 11. MIMO butterfly convergence - 2x2, training on both channels
 12. CPRState warm-start - second block_lms call resumes BPS state seamlessly
 13. input_norm_factor - supplied norm factor skips RMS recomputation
"""

import numpy as np
import pytest

from commkit.backend import to_device
from commkit.core import Signal
from commkit.equalization import CPRState, block_lms
from commkit.helpers import normalize
from commkit.mapping import gray_constellation

# -----------------------------------------------------------------------------
# Helpers (NumPy only - tests convert to xp before calling block_lms)
# -----------------------------------------------------------------------------


def _qam16(n_sym=4096, snr_db=25.0, sps=2, rng=None):
    """Return (samples_sps, symbols) for 16-QAM with AWGN, repeat upsampling."""
    if rng is None:
        rng = np.random.default_rng(42)
    const = gray_constellation("qam", 16).astype(np.complex64)
    const = normalize(const, "average_power").astype(np.complex64)
    syms = const[rng.integers(0, 16, n_sym)]
    samples = np.repeat(syms, sps).astype(np.complex64)
    noise_std = np.sqrt(10 ** (-snr_db / 10) / 2)
    samples += noise_std * (
        rng.standard_normal(len(samples)) + 1j * rng.standard_normal(len(samples))
    ).astype(np.complex64)
    return samples, syms


def _qpsk(n_sym=4096, snr_db=20.0, sps=2, rng=None):
    """Return (samples_sps, symbols) for QPSK with AWGN, repeat upsampling."""
    if rng is None:
        rng = np.random.default_rng(7)
    const = gray_constellation("psk", 4).astype(np.complex64)
    syms = const[rng.integers(0, 4, n_sym)]
    samples = np.repeat(syms, sps).astype(np.complex64)
    noise_std = np.sqrt(10 ** (-snr_db / 10) / 2)
    samples += noise_std * (
        rng.standard_normal(len(samples)) + 1j * rng.standard_normal(len(samples))
    ).astype(np.complex64)
    return samples, syms


# -----------------------------------------------------------------------------
# Test 1: Output shapes - SISO
# -----------------------------------------------------------------------------


def test_output_shape_siso(backend_device, xp):
    samples, syms = _qam16(n_sym=1024, sps=2)
    r = block_lms(
        xp.asarray(samples),
        xp.asarray(syms),
        num_taps=11,
        sps=2,
        modulation="qam",
        order=16,
        block_size=128,
    )
    n_sym = len(syms)
    assert r.y_hat.shape == (n_sym,), f"y_hat shape {r.y_hat.shape}"
    assert r.weights.shape == (11,), f"weights shape {r.weights.shape}"
    assert r.error.shape == (n_sym,)
    assert r.phase_trajectory is None


# -----------------------------------------------------------------------------
# Test 2: Output shapes - MIMO (C=2)
# -----------------------------------------------------------------------------


def test_output_shape_mimo(backend_device, xp):
    rng = np.random.default_rng(1)
    s1, t1 = _qam16(n_sym=1024, sps=2, rng=rng)
    s2, t2 = _qam16(n_sym=1024, sps=2, rng=rng)
    r = block_lms(
        xp.asarray(np.stack([s1, s2])),
        xp.asarray(np.stack([t1, t2])),
        num_taps=11,
        sps=2,
        modulation="qam",
        order=16,
        block_size=128,
    )
    assert r.y_hat.shape == (2, 1024)
    assert r.weights.shape == (2, 2, 11)
    assert r.error.shape == (2, 1024)


# -----------------------------------------------------------------------------
# Test 3: Output shapes - BPS CPR
# -----------------------------------------------------------------------------


def test_output_shape_bps(backend_device, xp):
    samples, syms = _qam16(n_sym=1024, sps=2)
    r = block_lms(
        xp.asarray(samples),
        xp.asarray(syms),
        num_taps=11,
        sps=2,
        modulation="qam",
        order=16,
        block_size=128,
        cpr_type="bps",
        cpr_bps_test_phases=16,
        cpr_bps_block_size=32,
    )
    assert r.y_hat.shape == (1024,)
    assert r.phase_trajectory is not None
    assert r.phase_trajectory.shape == (1024,), (
        f"phase_trajectory shape {r.phase_trajectory.shape}"
    )


def test_output_shape_bps_mimo(backend_device, xp):
    rng = np.random.default_rng(2)
    s1, t1 = _qam16(n_sym=1024, sps=2, rng=rng)
    s2, t2 = _qam16(n_sym=1024, sps=2, rng=rng)
    r = block_lms(
        xp.asarray(np.stack([s1, s2])),
        xp.asarray(np.stack([t1, t2])),
        num_taps=11,
        sps=2,
        modulation="qam",
        order=16,
        block_size=128,
        cpr_type="bps",
        cpr_bps_test_phases=16,
    )
    assert r.phase_trajectory.shape == (2, 1024)


# -----------------------------------------------------------------------------
# Test 4: Gradient descent - MSE decreases
# -----------------------------------------------------------------------------


def test_mse_decreases(backend_device, xp):
    """MSE in the last quarter of the signal must be less than in the first quarter."""
    samples, syms = _qam16(n_sym=8192, snr_db=25.0, sps=2)
    r = block_lms(
        xp.asarray(samples),
        xp.asarray(syms[:512]),
        num_taps=11,
        sps=2,
        step_size=5e-4,
        modulation="qam",
        order=16,
        block_size=128,
    )
    err = xp.abs(r.error) ** 2
    n = len(err)
    assert float(err[: n // 4].mean()) > float(err[3 * n // 4 :].mean()), (
        "MSE did not decrease from first to last quarter"
    )


# -----------------------------------------------------------------------------
# Test 5: Identity channel - output close to training symbols
# -----------------------------------------------------------------------------


def test_identity_channel_convergence(backend_device, xp):
    """On a near-identity channel, symbols after training should match reference."""
    samples, syms = _qam16(n_sym=4096, snr_db=30.0, sps=2)
    n_train = 1024
    syms_xp = xp.asarray(syms)
    r = block_lms(
        xp.asarray(samples),
        syms_xp,
        num_taps=11,
        sps=2,
        step_size=5e-4,
        modulation="qam",
        order=16,
        block_size=128,
    )
    y_eval = r.y_hat[n_train:]
    s_eval = syms_xp[n_train:]
    evm = float(
        xp.sqrt(xp.mean(xp.abs(y_eval - s_eval) ** 2) / xp.mean(xp.abs(s_eval) ** 2))
    )
    assert evm < 0.15, f"EVM {evm:.3f} too high - equalizer did not converge"


# -----------------------------------------------------------------------------
# Test 6: w_init passthrough
# -----------------------------------------------------------------------------


def test_w_init_used(backend_device, xp):
    """Warm-starting from converged weights should give lower initial MSE."""
    samples, syms = _qam16(n_sym=4096, snr_db=25.0, sps=2)
    samples_xp = xp.asarray(samples)
    syms_xp = xp.asarray(syms)
    # First pass: converge weights
    r1 = block_lms(
        samples_xp,
        syms_xp,
        num_taps=11,
        sps=2,
        step_size=5e-4,
        modulation="qam",
        order=16,
        block_size=128,
    )
    # Second pass: warm-start; MSE at start should be low
    r2 = block_lms(
        samples_xp,
        syms_xp,
        num_taps=11,
        sps=2,
        step_size=5e-4,
        modulation="qam",
        order=16,
        block_size=128,
        w_init=r1.weights,
    )
    mse_cold_start = float(xp.mean(xp.abs(r1.error[:128]) ** 2))
    mse_warm_start = float(xp.mean(xp.abs(r2.error[:128]) ** 2))
    assert mse_warm_start < mse_cold_start, (
        f"warm-start MSE ({mse_warm_start:.4f}) not better than cold ({mse_cold_start:.4f})"
    )


# -----------------------------------------------------------------------------
# Test 7: store_weights shape
# -----------------------------------------------------------------------------


def test_store_weights_shape_siso(backend_device, xp):
    samples, syms = _qam16(n_sym=512, sps=2)
    r = block_lms(
        xp.asarray(samples),
        xp.asarray(syms),
        num_taps=11,
        sps=2,
        modulation="qam",
        order=16,
        block_size=64,
        store_weights=True,
    )
    assert r.weights_history is not None
    assert r.weights_history.shape == (512, 11), (
        f"weights_history shape {r.weights_history.shape}"
    )


def test_store_weights_shape_mimo(backend_device, xp):
    rng = np.random.default_rng(3)
    s1, t1 = _qam16(n_sym=512, sps=2, rng=rng)
    s2, t2 = _qam16(n_sym=512, sps=2, rng=rng)
    r = block_lms(
        xp.asarray(np.stack([s1, s2])),
        xp.asarray(np.stack([t1, t2])),
        num_taps=11,
        sps=2,
        modulation="qam",
        order=16,
        block_size=64,
        store_weights=True,
    )
    assert r.weights_history.shape == (512, 2, 2, 11)


# -----------------------------------------------------------------------------
# Test 8: num_train_symbols - DA/DD boundary
# -----------------------------------------------------------------------------


def test_num_train_symbols_respected(backend_device, xp):
    """Training length is determined by the length of training_symbols passed in."""
    samples, syms = _qam16(n_sym=2048, snr_db=30.0, sps=2)
    samples_xp = xp.asarray(samples)
    syms_xp = xp.asarray(syms)
    # Pure DA (all training)
    r_da = block_lms(
        samples_xp,
        syms_xp,
        num_taps=11,
        sps=2,
        step_size=5e-4,
        modulation="qam",
        order=16,
        block_size=128,
    )
    assert r_da.num_train_symbols == 2048

    # Pre-sliced training to 256
    r_clip = block_lms(
        samples_xp,
        syms_xp[..., :256],
        num_taps=11,
        sps=2,
        step_size=5e-4,
        modulation="qam",
        order=16,
        block_size=128,
    )
    assert r_clip.num_train_symbols == 256


# -----------------------------------------------------------------------------
# Test 9: Last-block edge - n_sym not a multiple of block_size
# -----------------------------------------------------------------------------


def test_non_multiple_block_size(backend_device, xp):
    samples, syms = _qam16(n_sym=1000, sps=2)  # 1000 is not a multiple of 128
    r = block_lms(
        xp.asarray(samples),
        xp.asarray(syms),
        num_taps=11,
        sps=2,
        modulation="qam",
        order=16,
        block_size=128,
    )
    assert r.y_hat.shape == (1000,)
    assert r.error.shape == (1000,)


# -----------------------------------------------------------------------------
# Test 10: cpr_type='pll' raises ValueError
# -----------------------------------------------------------------------------


def test_pll_raises(backend_device, xp):
    samples, syms = _qam16(n_sym=512, sps=2)
    with pytest.raises(ValueError, match="pll"):
        block_lms(
            xp.asarray(samples),
            xp.asarray(syms),
            num_taps=11,
            sps=2,
            modulation="qam",
            order=16,
            cpr_type="pll",
        )


# -----------------------------------------------------------------------------
# Test 11: BPS block_size independent of cpr_bps_block_size
# -----------------------------------------------------------------------------


def test_bps_block_size_independent(backend_device, xp):
    """block_size=256 with cpr_bps_block_size=16 must produce per-symbol phi."""
    samples, syms = _qam16(n_sym=1024, sps=2)
    r = block_lms(
        xp.asarray(samples),
        xp.asarray(syms),
        num_taps=11,
        sps=2,
        modulation="qam",
        order=16,
        block_size=256,
        cpr_type="bps",
        cpr_bps_test_phases=16,
        cpr_bps_block_size=16,
    )
    assert r.phase_trajectory.shape == (1024,)
    phi = r.phase_trajectory
    assert not bool(xp.all(phi == phi[0])), (
        "All phi identical - expected per-symbol variation with cpr_bps_block_size=16"
    )


# -----------------------------------------------------------------------------
# Test 12: BPS under phase noise - CPR reduces MSE vs no CPR
# -----------------------------------------------------------------------------


def test_bps_reduces_mse_under_phase_noise(backend_device, xp):
    """With strong phase noise, BPS should produce lower steady-state MSE."""
    rng = np.random.default_rng(99)
    n_sym = 4096
    sps = 2
    const = gray_constellation("psk", 4).astype(np.complex64)
    syms = const[rng.integers(0, 4, n_sym)]

    # Add random-walk phase noise
    phase_noise = np.cumsum(0.03 * rng.standard_normal(n_sym)).astype(np.float32)
    samples_pn = (
        np.repeat(syms * np.exp(1j * phase_noise), sps)
        + 0.1
        * (rng.standard_normal(2 * n_sym) + 1j * rng.standard_normal(2 * n_sym)).astype(
            np.complex64
        )
    ).astype(np.complex64)

    n_eval = n_sym // 2
    r_no_cpr = block_lms(
        xp.asarray(samples_pn),
        xp.asarray(syms[:512]),
        num_taps=7,
        sps=sps,
        step_size=5e-4,
        block_size=128,
        modulation="psk",
        order=4,
    )
    r_bps = block_lms(
        xp.asarray(samples_pn),
        xp.asarray(syms[:512]),
        num_taps=7,
        sps=sps,
        step_size=5e-4,
        block_size=128,
        modulation="psk",
        order=4,
        cpr_type="bps",
        cpr_bps_test_phases=32,
        cpr_bps_block_size=32,
    )

    mse_no_cpr = float(xp.mean(xp.abs(r_no_cpr.error[n_eval:]) ** 2))
    mse_bps = float(xp.mean(xp.abs(r_bps.error[n_eval:]) ** 2))
    assert mse_bps < mse_no_cpr, (
        f"BPS MSE ({mse_bps:.4f}) not better than no-CPR ({mse_no_cpr:.4f}) "
        "under phase noise"
    )


# -----------------------------------------------------------------------------
# Test 13: MIMO butterfly convergence (2x2)
# -----------------------------------------------------------------------------


def test_mimo_convergence(backend_device, xp):
    """2x2 MIMO: both channels should converge to low EVM."""
    rng = np.random.default_rng(5)
    n_sym = 4096
    sps = 1
    const = gray_constellation("qam", 16).astype(np.complex64)
    const = normalize(const, "average_power").astype(np.complex64)
    t1 = const[rng.integers(0, 16, n_sym)]
    t2 = const[rng.integers(0, 16, n_sym)]
    noise = 0.05 * (
        rng.standard_normal((2, n_sym)) + 1j * rng.standard_normal((2, n_sym))
    ).astype(np.complex64)
    training_np = np.stack([t1, t2])
    samples = xp.asarray(training_np + noise)
    training = xp.asarray(training_np)

    r = block_lms(
        samples,
        training,
        num_taps=5,
        sps=sps,
        step_size=2e-3,
        modulation="qam",
        order=16,
        block_size=64,
    )
    n_eval = n_sym // 2
    for ch in range(2):
        evm = float(
            xp.sqrt(
                xp.mean(xp.abs(r.y_hat[ch, n_eval:] - training[ch, n_eval:]) ** 2)
                / xp.mean(xp.abs(training[ch, n_eval:]) ** 2)
            )
        )
        assert evm < 0.15, f"MIMO ch{ch} EVM {evm:.3f} too high"


# -----------------------------------------------------------------------------
# Tests 14-15: edge cases and ISI channel
# -----------------------------------------------------------------------------


def test_single_tap(backend_device, xp):
    """num_taps=1: degenerate equalizer - must not crash, output shape correct."""
    rng = np.random.default_rng(42)
    n_sym = 64
    const = gray_constellation("qam", 4).astype(np.complex64)
    const = normalize(const, "average_power").astype(np.complex64)
    syms = xp.asarray(const[rng.integers(0, 4, n_sym)].astype(np.complex64))
    noise = 0.05 * xp.asarray(
        (rng.standard_normal(n_sym) + 1j * rng.standard_normal(n_sym)).astype(
            np.complex64
        )
    )
    samples = syms + noise

    r = block_lms(
        samples,
        syms,
        num_taps=1,
        sps=1,
        step_size=1e-2,
        modulation="qam",
        order=4,
        block_size=16,
    )
    assert r.y_hat.shape == (n_sym,)


def test_isi_channel_convergence(backend_device, xp):
    """Known 3-tap ISI channel: equalizer must converge across block boundaries."""
    rng = np.random.default_rng(99)
    n_sym = 2048
    sps = 1
    block_size = 32

    channel = np.array([0.1, 1.0, 0.1], dtype=np.complex64)
    const = gray_constellation("qam", 4).astype(np.complex64)
    const = normalize(const, "average_power").astype(np.complex64)
    syms_np = const[rng.integers(0, 4, n_sym)].astype(np.complex64)
    received_np = np.convolve(syms_np, channel, mode="full")[:n_sym].astype(
        np.complex64
    )
    received = xp.asarray(received_np)
    syms = xp.asarray(syms_np)

    r = block_lms(
        received,
        syms,
        num_taps=5,
        sps=sps,
        step_size=5e-3,
        modulation="qam",
        order=4,
        block_size=block_size,
    )
    n_eval = n_sym // 2
    evm = float(
        xp.sqrt(
            xp.mean(xp.abs(r.y_hat[n_eval:] - syms[n_eval:]) ** 2)
            / xp.mean(xp.abs(syms[n_eval:]) ** 2)
        )
    )
    assert evm < 0.10, (
        f"EVM {evm:.3f} - ISI equalization failed across block boundaries"
    )


# -----------------------------------------------------------------------------
# 12. CPRState warm-start
# -----------------------------------------------------------------------------


def _wiener_qam16_block(n_sym=4096, snr_db=25.0, linewidth=5e3, seed=11):
    """Return (samples, symbols) for 16-QAM under Wiener phase noise."""
    rng = np.random.default_rng(seed)
    const = gray_constellation("qam", 16).astype(np.complex64)
    const = normalize(const, "average_power").astype(np.complex64)
    syms = const[rng.integers(0, 16, n_sym)]
    sigma_phi = float(np.sqrt(2 * np.pi * linewidth))
    phase = np.cumsum(rng.normal(0.0, sigma_phi, n_sym)).astype(np.float64)
    samples = (syms * np.exp(1j * phase)).astype(np.complex64)
    noise_std = np.sqrt(10 ** (-snr_db / 10) / 2)
    samples += noise_std * (
        rng.standard_normal(n_sym) + 1j * rng.standard_normal(n_sym)
    ).astype(np.complex64)
    return samples, syms


def test_cpr_state_warmstart_block_lms_bps(backend_device, xp):
    """block_lms with cpr_state should populate and accept CPRState."""
    n_sym = 4096
    half = n_sym // 2
    samples_np, syms_np = _wiener_qam16_block(n_sym=n_sym)
    s1, s2 = samples_np[:half], samples_np[half:]
    t1, t2 = syms_np[:half], syms_np[half:]

    r1 = block_lms(
        xp.asarray(s1),
        xp.asarray(t1),
        num_taps=11,
        sps=1,
        step_size=5e-4,
        modulation="qam",
        order=16,
        cpr_type="bps",
        cpr_bps_test_phases=32,
        cpr_bps_block_size=16,
    )
    assert r1.cpr_state is not None, "cpr_state must be set when cpr_type='bps'"
    assert isinstance(r1.cpr_state, CPRState)
    assert r1.cpr_state.bps_prev4 is not None
    assert r1.cpr_state.bps_offset4 is not None
    assert r1.cpr_state.bps_d2_hist is not None
    # CPRState contract: exported arrays are CPU NumPy on every backend.
    assert isinstance(r1.cpr_state.bps_prev4, np.ndarray)
    assert isinstance(r1.cpr_state.bps_offset4, np.ndarray)
    assert isinstance(r1.cpr_state.bps_d2_hist, np.ndarray)

    r2 = block_lms(
        xp.asarray(s2),
        xp.asarray(t2[:50]),
        num_taps=11,
        sps=1,
        step_size=5e-4,
        modulation="qam",
        order=16,
        cpr_type="bps",
        cpr_bps_test_phases=32,
        cpr_bps_block_size=16,
        w_init=r1.weights,
        cpr_state=r1.cpr_state,
        input_norm_factor=r1.input_norm_factor,
    )
    assert r2.cpr_state is not None
    assert r2.cpr_state.cpr_type == "bps"
    assert r2.phase_trajectory is not None


def test_cpr_state_none_is_baseline_block_lms(backend_device, xp):
    """cpr_state=None must be byte-exact with the default (no cpr_state) call."""
    samples_np, syms_np = _wiener_qam16_block(n_sym=2048)
    kw = dict(
        num_taps=11,
        sps=1,
        step_size=5e-4,
        modulation="qam",
        order=16,
        cpr_type="bps",
        cpr_bps_test_phases=32,
        cpr_bps_block_size=16,
    )
    r_default = block_lms(samples_np, syms_np[:100], **kw)
    r_none = block_lms(
        samples_np, syms_np[:100], **kw, cpr_state=None, input_norm_factor=None
    )
    np.testing.assert_array_equal(
        np.asarray(r_default.y_hat),
        np.asarray(r_none.y_hat),
    )


@pytest.mark.parametrize("cs_corr", [False, True])
@pytest.mark.parametrize("n_sym", [100_000, 100_137])
def test_block_lms_cuda_graph_matches_eager(cs_corr, n_sym, backend_device, xp):
    """GPU: cuda_graph=True must be bit-exact with cuda_graph=False.

    Uses a short training prefix so the bulk of the run is decision-directed
    (the only blocks the graph captures), and an odd ``n_sym`` to exercise the
    eager partial last block alongside the captured full blocks.  Graph replay
    recomputes every block from the same inputs and state, so the two paths
    must agree to the last bit - any divergence flags a capture/replay aliasing
    bug.
    """
    if xp is np:
        pytest.skip("CUDA-graph capture is GPU-only")
    from commkit import _cuda

    if not _cuda.is_available():
        pytest.skip("CUDA device below compute capability 7.0")

    rng = np.random.default_rng(5)
    const = normalize(gray_constellation("qam", 16), "average_power").astype(
        np.complex64
    )
    syms = const[rng.integers(0, 16, n_sym)]
    phase = np.cumsum(rng.normal(0.0, 0.01, n_sym))
    samples = (syms * np.exp(1j * phase)).astype(np.complex64)
    noise = np.sqrt(10 ** (-25.0 / 10) / 2)
    samples += noise * (
        rng.standard_normal(n_sym) + 1j * rng.standard_normal(n_sym)
    ).astype(np.complex64)

    kw = dict(
        num_taps=21,
        sps=1,
        step_size=5e-4,
        block_size=256,
        modulation="qam",
        order=16,
        cpr_type="bps",
        cpr_cycle_slip_correction=cs_corr,
    )
    x = xp.asarray(samples)
    t = xp.asarray(syms[:512])  # short data-aided preamble; rest is graph-eligible

    r_graph = block_lms(x, t, **kw, cuda_graph=True)
    r_eager = block_lms(x, t, **kw, cuda_graph=False)

    np.testing.assert_array_equal(
        np.asarray(to_device(r_graph.y_hat, "cpu")),
        np.asarray(to_device(r_eager.y_hat, "cpu")),
    )
    np.testing.assert_array_equal(
        np.asarray(to_device(r_graph.phase_trajectory, "cpu")),
        np.asarray(to_device(r_eager.phase_trajectory, "cpu")),
    )
    np.testing.assert_array_equal(
        np.asarray(to_device(r_graph.weights, "cpu")),
        np.asarray(to_device(r_eager.weights, "cpu")),
    )


def _wiener_qam16_trackable(n_sym, snr_db=25.0, sigma_phi=0.005, seed=33):
    """16-QAM under slow Wiener phase noise that BPS can actually track.

    The second half of the symbol sequence is a permutation of the first half,
    so both halves and the full sequence share the same symbol multiset - the
    per-call training-symbol normalization then applies the same scale to the
    uninterrupted run and to each half-run (up to float summation order).
    """
    rng = np.random.default_rng(seed)
    const = gray_constellation("qam", 16).astype(np.complex64)
    const = normalize(const, "average_power").astype(np.complex64)
    half = n_sym // 2
    syms_h = const[rng.integers(0, 16, half)]
    syms = np.concatenate([syms_h, rng.permutation(syms_h)])
    phase = np.cumsum(rng.normal(0.0, sigma_phi, n_sym))
    samples = (syms * np.exp(1j * phase)).astype(np.complex64)
    noise_std = float(np.sqrt(10 ** (-snr_db / 10) / 2))
    samples += noise_std * (
        rng.standard_normal(n_sym) + 1j * rng.standard_normal(n_sym)
    ).astype(np.complex64)
    return samples, syms


@pytest.mark.parametrize("cs_corr", [False, True])
def test_cpr_state_roundtrip_matches_uninterrupted(cs_corr, backend_device, xp):
    """Split run (export CPRState at half, resume) matches one uninterrupted run.

    With num_taps=1 (no padding, no look-ahead) and the split aligned to a
    block boundary, the two runs see identical per-block inputs - the only
    state crossing the split is what w_init + cpr_state + input_norm_factor
    carry.  This gates the full unwrap/offset/d2-history/cycle-slip round
    trip through the CPU-NumPy CPRState contract on both backends.
    """
    n_sym = 2048
    half = n_sym // 2
    samples_np, syms_np = _wiener_qam16_trackable(n_sym)

    kw = dict(
        num_taps=1,
        sps=1,
        step_size=5e-4,
        block_size=256,
        modulation="qam",
        order=16,
        cpr_type="bps",
        cpr_bps_test_phases=32,
        cpr_bps_block_size=16,
        cpr_cycle_slip_correction=cs_corr,
    )

    r_full = block_lms(xp.asarray(samples_np), xp.asarray(syms_np), **kw)

    r1 = block_lms(
        xp.asarray(samples_np[:half]),
        xp.asarray(syms_np[:half]),
        **kw,
        input_norm_factor=r_full.input_norm_factor,
    )
    r2 = block_lms(
        xp.asarray(samples_np[half:]),
        xp.asarray(syms_np[half:]),
        **kw,
        w_init=r1.weights,
        cpr_state=r1.cpr_state,
        input_norm_factor=r_full.input_norm_factor,
    )

    y_full = np.asarray(to_device(r_full.y_hat, "cpu"))
    y_split = np.concatenate(
        [np.asarray(to_device(r1.y_hat, "cpu")), np.asarray(to_device(r2.y_hat, "cpu"))]
    )
    np.testing.assert_allclose(y_split, y_full, rtol=1e-5, atol=1e-5)

    assert r_full.phase_trajectory is not None
    p_full = np.asarray(to_device(r_full.phase_trajectory, "cpu"))
    p_split = np.concatenate(
        [
            np.asarray(to_device(r1.phase_trajectory, "cpu")),
            np.asarray(to_device(r2.phase_trajectory, "cpu")),
        ]
    )
    # Unwrapped trajectories must agree absolutely - a lost carry shows up as
    # an O(π/2) jump at the split point.
    np.testing.assert_allclose(p_split, p_full, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("cs_corr", [False, True])
def test_block_lms_bps_loop_transfer_count_constant(
    cs_corr, backend_device, xp, monkeypatch
):
    """GPU: D2H/H2D transfers must not scale with the number of blocks.

    Spies on to_device in the equalization module and asserts the call count
    is identical for a 4-block and a 16-block run - i.e. the unwrap-carry and
    cycle-slip transfers happen at entry/exit only, never inside the block
    loop.
    """
    if xp is np:
        pytest.skip("GPU-only host-sync hygiene check")
    if cs_corr:
        from commkit import _cuda

        if _cuda.get_kernel("cs_block") is None:
            pytest.skip("cs_block CUDA kernel unavailable - fallback transfers")

    # block_lms lives in the equalization._block submodule and resolves
    # to_device in that namespace; spy there to count host-device transfers.
    import commkit.equalization._block as eqmod

    real_to_device = eqmod.to_device
    counts = {"n": 0}

    def spy(data, device):
        counts["n"] += 1
        return real_to_device(data, device)

    monkeypatch.setattr(eqmod, "to_device", spy)

    def run(n_sym):
        samples_np, syms_np = _wiener_qam16_trackable(n_sym)
        counts["n"] = 0
        block_lms(
            xp.asarray(samples_np),
            xp.asarray(syms_np[:128]),
            num_taps=11,
            sps=1,
            step_size=5e-4,
            block_size=128,
            modulation="qam",
            order=16,
            cpr_type="bps",
            cpr_bps_test_phases=32,
            cpr_bps_block_size=16,
            cpr_cycle_slip_correction=cs_corr,
        )
        return counts["n"]

    n_small = run(512)  # 4 blocks
    n_large = run(2048)  # 16 blocks
    assert n_small == n_large, (
        f"to_device call count scales with block count: "
        f"{n_small} (4 blocks) vs {n_large} (16 blocks)"
    )


def test_block_lms_cycle_slip_kernel_matches_cpu_fallback(
    backend_device, xp, monkeypatch
):
    """GPU: cs_block CUDA kernel path matches the D2H + Numba fallback path.

    Runs the same bps+cs workload twice on GPU - once with the CUDA detector,
    once with get_kernel('cs_block') forced to None - and requires matching
    outputs.  A disagreement in any slip decision would show up as an O(π/2)
    phase divergence.
    """
    if xp is np:
        pytest.skip("GPU-only kernel-vs-fallback check")

    from commkit import _cuda

    if _cuda.get_kernel("cs_block") is None:
        pytest.skip("cs_block CUDA kernel unavailable")

    samples_np, syms_np = _wiener_qam16_trackable(2048, sigma_phi=0.02)
    kw = dict(
        num_taps=11,
        sps=1,
        step_size=5e-4,
        block_size=128,
        modulation="qam",
        order=16,
        cpr_type="bps",
        cpr_bps_test_phases=32,
        cpr_bps_block_size=16,
        cpr_cycle_slip_correction=True,
    )

    r_kernel = block_lms(xp.asarray(samples_np), xp.asarray(syms_np[:128]), **kw)

    real_get_kernel = _cuda.get_kernel

    def no_cs_kernel(name, **spec):
        if name == "cs_block":
            return None
        return real_get_kernel(name, **spec)

    monkeypatch.setattr(_cuda, "get_kernel", no_cs_kernel)
    r_fallback = block_lms(xp.asarray(samples_np), xp.asarray(syms_np[:128]), **kw)

    np.testing.assert_allclose(
        np.asarray(to_device(r_kernel.phase_trajectory, "cpu")),
        np.asarray(to_device(r_fallback.phase_trajectory, "cpu")),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(to_device(r_kernel.y_hat, "cpu")),
        np.asarray(to_device(r_fallback.y_hat, "cpu")),
        rtol=1e-5,
        atol=1e-6,
    )
    for attr in ("cs_buf_y", "cs_buf_ptr", "cs_buf_n", "cs_stats"):
        np.testing.assert_allclose(
            getattr(r_kernel.cpr_state, attr),
            getattr(r_fallback.cpr_state, attr),
            rtol=1e-8,
            atol=1e-8,
            err_msg=attr,
        )


# -----------------------------------------------------------------------------
# 13. Per-symbol cycle-slip correction in block_lms
# -----------------------------------------------------------------------------


def test_block_lms_cycle_slip_correction(backend_device, xp):
    """block_lms with cpr_cycle_slip_correction=True recovers through deliberate π/2 phase steps."""
    rng = np.random.default_rng(77)
    n_sym = 4096
    const = gray_constellation("qam", 16).astype(np.complex64)
    const = normalize(const, "average_power").astype(np.complex64)
    syms = const[rng.integers(0, 16, n_sym)]

    # Inject a π/2 phase step every 512 symbols (well within block_size=256 boundaries)
    phase = np.zeros(n_sym, dtype=np.float64)
    for step_idx in range(512, n_sym, 512):
        phase[step_idx:] += np.pi / 2

    samples = (syms * np.exp(1j * phase).astype(np.complex64)).astype(np.complex64)
    noise_std = float(np.sqrt(10 ** (-25.0 / 10) / 2))
    samples += noise_std * (
        rng.standard_normal(n_sym) + 1j * rng.standard_normal(n_sym)
    ).astype(np.complex64)

    res = block_lms(
        xp.asarray(samples),
        training_symbols=xp.asarray(syms[:256]),
        num_taps=1,
        sps=1,
        step_size=1e-3,
        modulation="qam",
        order=16,
        block_size=128,
        cpr_type="bps",
        cpr_bps_test_phases=64,
        cpr_bps_block_size=32,
        cpr_cycle_slip_correction=True,
        cpr_cycle_slip_threshold=np.pi / 4,
    )

    assert res.phase_trajectory is not None
    assert res.cpr_state is not None
    assert res.cpr_state.cs_buf_y is not None  # regression buffer populated

    # Steady-state MSE on the last quarter (well past training); y_hat is 1D for SISO
    y_tail = res.y_hat[-n_sym // 4 :]
    const_xp = xp.asarray(const)
    d2 = xp.abs(y_tail[:, None] - const_xp[None, :]) ** 2
    decisions = const_xp[xp.argmin(d2, axis=1)]
    mse = float(xp.mean(xp.abs(y_tail - decisions) ** 2))
    assert mse < 0.05, (
        f"Steady-state MSE too large after cycle-slip correction: {mse:.4f}"
    )


def test_block_lms_cycle_slip_regression_warmstart(backend_device, xp):
    """Regression buffer is saved into CPRState and correctly restored on warm-start."""
    samples_np, syms_np = _wiener_qam16_block(n_sym=2048)
    half = 1024
    kw = dict(
        num_taps=11,
        sps=1,
        step_size=5e-4,
        modulation="qam",
        order=16,
        cpr_type="bps",
        cpr_bps_test_phases=32,
        cpr_bps_block_size=16,
        cpr_cycle_slip_correction=True,
    )

    r1 = block_lms(xp.asarray(samples_np[:half]), xp.asarray(syms_np[:50]), **kw)
    assert r1.cpr_state is not None
    assert r1.cpr_state.cs_buf_y is not None
    assert r1.cpr_state.cs_buf_n is not None

    # Warm-start: regression buffer must be accepted and produce finite output
    r2 = block_lms(
        xp.asarray(samples_np[half:]),
        xp.asarray(syms_np[half : half + 50]),
        **kw,
        w_init=r1.weights,
        cpr_state=r1.cpr_state,
        input_norm_factor=r1.input_norm_factor,
    )
    assert r2.cpr_state is not None
    assert r2.cpr_state.cs_buf_y is not None
    assert bool(xp.all(xp.isfinite(xp.asarray(r2.y_hat))))


# -----------------------------------------------------------------------------
# 14. input_norm_factor
# -----------------------------------------------------------------------------


def test_input_norm_factor_block_lms(backend_device, xp, xpt):
    """Supplying input_norm_factor reproduces the same output as auto-computed."""
    samples_np, syms_np = _wiener_qam16_block(n_sym=2048)
    samples, syms = xp.asarray(samples_np), xp.asarray(syms_np)
    kw = dict(num_taps=11, sps=1, step_size=5e-4, modulation="qam", order=16)

    r_auto = block_lms(samples, syms[:100], **kw)
    nf = r_auto.input_norm_factor

    r_supplied = block_lms(samples, syms[:100], **kw, input_norm_factor=nf)
    xpt.assert_allclose(
        xp.asarray(r_supplied.y_hat),
        xp.asarray(r_auto.y_hat),
        rtol=1e-5,
        atol=1e-6,
    )


# -----------------------------------------------------------------------------
# 15. Signal input
# -----------------------------------------------------------------------------


def test_block_lms_signal_input(backend_device, xp, xpt):
    """Signal input: sps is taken from the signal, y_hat becomes a Signal at
    the symbol rate."""
    samples_np, syms_np = _qam16(n_sym=2048, sps=2)
    samples, syms = xp.asarray(samples_np), xp.asarray(syms_np)
    sig = Signal(samples=samples, sampling_rate=2e6, symbol_rate=1e6)
    kw = dict(num_taps=11, modulation="qam", order=16, block_size=128)

    result_sig = block_lms(sig, syms[:200], **kw)
    result_arr = block_lms(samples, syms[:200], sps=2, **kw)

    assert isinstance(result_sig.y_hat, Signal)
    assert result_sig.y_hat.sampling_rate == 1e6
    xpt.assert_allclose(result_sig.y_hat.samples, result_arr.y_hat)
