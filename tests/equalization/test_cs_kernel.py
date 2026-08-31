"""Tests for the cs_block CUDA kernel (block_lms cycle-slip correction).

Kernel-level tests compare against a pure-Python float64 reference of the
same online-OLS slip detector, sequenced over multiple uneven blocks so the
circular history buffer is exercised through fill, wrap, and steady state.
The end-to-end kernel-vs-fallback comparison lives in test_block_lms.py.
"""

import numpy as np
import pytest

from commkit import _cuda

QUANTUM = float(2.0 * np.pi / 4.0)
THRESHOLD = float(np.pi / 4.0)


def _skip_unless_kernel(backend_device):
    if backend_device != "gpu":
        pytest.skip("requires a CUDA device")
    if not _cuda.is_available():
        pytest.skip("CUDA device below compute capability 7.0")


def _reference_cs_block(
    phi_blk, phi_corr, cs_buf_y, cs_buf_ptr, cs_buf_n, cs_stats, quantum, threshold, H
):
    """Pure-Python float64 mirror of the cs_block detector (one block)."""
    C, B = phi_blk.shape
    H_f = float(H)
    for ci in range(C):
        for i in range(B):
            y_b = phi_blk[ci, i]
            n_b = int(cs_buf_n[ci])
            ptr = int(cs_buf_ptr[ci])

            if n_b == 0:
                phi_expected = y_b
            elif n_b < 10:
                phi_expected = cs_buf_y[ci, (ptr - 1 + H) % H]
            else:
                sy = cs_stats[ci, 0]
                sxy = cs_stats[ci, 1]
                n_f = float(n_b)
                if n_b < H:
                    Sx_c = n_f * (n_f - 1.0) / 2.0
                    Sxx_c = n_f * (n_f - 1.0) * (2.0 * n_f - 1.0) / 6.0
                    denom = n_f * Sxx_c - Sx_c * Sx_c
                else:
                    Sx_c = H_f * (H_f - 1.0) / 2.0
                    Sxx_c = H_f * (H_f - 1.0) * (2.0 * H_f - 1.0) / 6.0
                    denom = H_f * Sxx_c - Sx_c * Sx_c
                if abs(denom) > 1e-30:
                    slope = (n_f * sxy - Sx_c * sy) / denom
                    intercept = (sy - slope * Sx_c) / n_f
                else:
                    slope = 0.0
                    intercept = sy / n_f
                phi_expected = slope * n_f + intercept

            diff = y_b - phi_expected
            k_slip = int(round(diff / quantum))
            if abs(diff) > threshold and k_slip != 0:
                y_b -= float(k_slip) * quantum
            phi_corr[ci, i] = y_b

            write_pos = ptr % H
            if n_b == H:
                old_y = cs_buf_y[ci, write_pos]
                old_sy = cs_stats[ci, 0]
                cs_stats[ci, 1] = cs_stats[ci, 1] - old_sy + old_y + (H_f - 1.0) * y_b
                cs_stats[ci, 0] = old_sy - old_y + y_b
            else:
                cs_stats[ci, 1] += float(n_b) * y_b
                cs_stats[ci, 0] += y_b
            cs_buf_y[ci, write_pos] = y_b
            cs_buf_ptr[ci] = ptr + 1
            if n_b < H:
                cs_buf_n[ci] = n_b + 1


def _slip_workload(C=3, n_total=1000, seed=7):
    """Slow ramp + noise with deliberate per-channel pi/2 slips."""
    rng = np.random.default_rng(seed)
    phi = 0.3 * np.sin(np.linspace(0.0, 3.0, n_total))[None, :]
    phi = phi + 0.001 * np.arange(n_total)[None, :]
    phi = phi + 0.02 * rng.standard_normal((C, n_total))
    for ch, pos in ((0, 300), (1, 500), (2, 750), (0, 760)):
        phi[ch % C, pos:] += QUANTUM
    return np.ascontiguousarray(phi)


def test_cs_block_matches_reference_across_blocks(backend_device, xp):
    """Kernel output and state match the reference through fill/wrap/slips."""
    _skip_unless_kernel(backend_device)
    kern = _cuda.get_kernel("cs_block")
    assert kern is not None

    C, H = 3, 100
    phi = _slip_workload(C=C)

    buf_y_ref = np.zeros((C, H))
    ptr_ref = np.zeros(C, np.int64)
    n_ref = np.zeros(C, np.int64)
    st_ref = np.zeros((C, 4))

    buf_y_dev = xp.zeros((C, H))
    ptr_dev = xp.zeros(C, xp.int64)
    n_dev = xp.zeros(C, xp.int64)
    st_dev = xp.zeros((C, 4))

    # Uneven block edges: tiny blocks during fill, then past the H=100 wrap.
    edges = [0, 7, 8, 64, 200, 333, 700, 1000]
    out_ref, out_dev = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        blk = np.ascontiguousarray(phi[:, a:b])
        corr_ref = np.empty_like(blk)
        _reference_cs_block(
            blk, corr_ref, buf_y_ref, ptr_ref, n_ref, st_ref, QUANTUM, THRESHOLD, H
        )
        out_ref.append(corr_ref)

        blk_dev = xp.asarray(blk)
        corr_dev = xp.empty_like(blk_dev)
        kern(
            blk_dev, corr_dev, buf_y_dev, ptr_dev, n_dev, st_dev, QUANTUM, THRESHOLD, H
        )
        out_dev.append(np.asarray(corr_dev.get()))

    corr_full_ref = np.concatenate(out_ref, axis=1)
    corr_full_dev = np.concatenate(out_dev, axis=1)
    np.testing.assert_allclose(corr_full_dev, corr_full_ref, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(buf_y_dev.get(), buf_y_ref, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(st_dev.get(), st_ref, rtol=1e-8, atol=1e-8)
    np.testing.assert_array_equal(ptr_dev.get(), ptr_ref)
    np.testing.assert_array_equal(n_dev.get(), n_ref)

    # The injected slips must actually be corrected: after correction the
    # remaining trajectory is slip-free (no inter-symbol jump near pi/2).
    jumps = np.abs(np.diff(corr_full_dev, axis=1))
    assert float(jumps.max()) < QUANTUM / 2.0


def test_cs_block_rejects_bad_inputs(backend_device, xp):
    """Wrapper must reject wrong dtype, wrong shape, and non-contiguous state."""
    _skip_unless_kernel(backend_device)
    kern = _cuda.get_kernel("cs_block")
    assert kern is not None

    C, B, H = 2, 16, 100
    phi = xp.zeros((C, B), dtype=xp.float64)
    corr = xp.empty_like(phi)
    buf_y = xp.zeros((C, H))
    ptr = xp.zeros(C, xp.int64)
    n = xp.zeros(C, xp.int64)
    st = xp.zeros((C, 4))

    with pytest.raises(TypeError, match="phi_blk"):
        kern(phi.astype(xp.float32), corr, buf_y, ptr, n, st, QUANTUM, THRESHOLD, H)
    with pytest.raises(ValueError, match="cs_buf_y"):
        kern(phi, corr, buf_y[:, : H // 2], ptr, n, st, QUANTUM, THRESHOLD, H)
    with pytest.raises(ValueError, match="C-contiguous"):
        kern(
            phi,
            corr,
            xp.asfortranarray(xp.zeros((C, H))),
            ptr,
            n,
            st,
            QUANTUM,
            THRESHOLD,
            H,
        )
