"""Tests for the cs_block CUDA kernel (block_lms cycle-slip correction).

Kernel-level tests compare against a pure-Python float64 reference of the
same online-OLS slip detector, sequenced over multiple uneven blocks so the
circular history buffer is exercised through fill, wrap, and steady state.
The end-to-end kernel-vs-fallback comparison lives in test_block_lms.py.
"""

import numpy as np
import pytest

from commkit import _cuda
from tests.common.conversions import to_numpy
from tests.common.kernel_utils import (
    reference_cs_block,
    skip_unless_kernel_available,
    slip_workload,
)

QUANTUM = float(2.0 * np.pi / 4.0)
THRESHOLD = float(np.pi / 4.0)


@pytest.mark.gpu_only
class TestCSBlockKernel:
    """Cycle-slip correction kernel validation against float64 reference."""

    def test_cs_block_matches_reference_across_blocks(self, backend_device, xp, xpt):
        """Kernel output and state match the reference through fill/wrap/slips."""
        skip_unless_kernel_available("cs_block", backend_device=backend_device)
        kern = _cuda.get_kernel("cs_block")
        assert kern is not None

        C, H = 3, 100
        phi = slip_workload(C=C)

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
            reference_cs_block(
                blk, corr_ref, buf_y_ref, ptr_ref, n_ref, st_ref, QUANTUM, THRESHOLD, H
            )
            out_ref.append(corr_ref)

            blk_dev = xp.asarray(blk)
            corr_dev = xp.empty_like(blk_dev)
            kern(
                blk_dev,
                corr_dev,
                buf_y_dev,
                ptr_dev,
                n_dev,
                st_dev,
                QUANTUM,
                THRESHOLD,
                H,
            )
            out_dev.append(to_numpy(corr_dev))

        corr_full_ref = np.concatenate(out_ref, axis=1)
        corr_full_dev = np.concatenate(out_dev, axis=1)
        xpt.assert_allclose(
            xp.asarray(corr_full_dev), xp.asarray(corr_full_ref), rtol=1e-9, atol=1e-9
        )
        xpt.assert_allclose(buf_y_dev, xp.asarray(buf_y_ref), rtol=1e-9, atol=1e-9)
        xpt.assert_allclose(st_dev, xp.asarray(st_ref), rtol=1e-8, atol=1e-8)
        xpt.assert_array_equal(ptr_dev, xp.asarray(ptr_ref))
        xpt.assert_array_equal(n_dev, xp.asarray(n_ref))

        # The injected slips must actually be corrected
        jumps = np.abs(np.diff(corr_full_dev, axis=1))
        assert float(jumps.max()) < QUANTUM / 2.0

    def test_cs_block_rejects_bad_inputs(self, backend_device, xp):
        """Wrapper must reject wrong dtype, wrong shape, and non-contiguous state."""
        skip_unless_kernel_available("cs_block", backend_device=backend_device)
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
