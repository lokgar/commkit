"""Tests for the cs_block CUDA kernel (block_lms cycle-slip correction).

Kernel-level tests compare against a pure-Python float64 reference of the
same online-OLS slip detector, sequenced over multiple uneven blocks so the
circular history buffer is exercised through fill, wrap, and steady state.
The end-to-end comparison with the CPU lives in test_block.py.
"""

import numpy as np
import pytest

from commkit import _cuda
from commkit.equalization._block._dd import _get_numba_cs_block
from tests.common.conversions import to_numpy
from tests.common.kernel_utils import (
    reference_cs_block,
    skip_unless_gpu,
    slip_workload,
)

QUANTUM = float(2.0 * np.pi / 4.0)
THRESHOLD = float(np.pi / 4.0)


@pytest.mark.gpu_only
class TestCSBlockKernel:
    """Cycle-slip correction kernel validation against float64 reference."""

    def test_cs_block_matches_reference_across_blocks(self, backend_device, xp, xpt):
        """Kernel output and state match the reference through fill/wrap/slips."""
        skip_unless_gpu(backend_device)
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
        kern = _cuda.get_kernel("cs_block")(
            buf_y_dev, ptr_dev, n_dev, st_dev,
            quantum=QUANTUM, threshold=THRESHOLD, cs_H=H, max_block=367,
        )  # fmt: skip
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
            kern(blk_dev, corr_dev)
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

    def test_cs_block_more_than_1024_channels(self, backend_device, xp, xpt):
        """One thread block per channel: wider than one block's thread limit,
        the kernel matches the Numba path that the CPU runs."""
        skip_unless_gpu(backend_device)
        cs_numba = _get_numba_cs_block()

        C, H = 1100, 100
        phi = slip_workload(C=C, n_total=400)
        phi[C - 1, 250:] += QUANTUM  # a slip on the last thread block

        bufs_ref = [np.zeros((C, H)), np.zeros(C, np.int64), np.zeros(C, np.int64)]
        st_ref = np.zeros((C, 4))
        bufs_dev = [xp.asarray(b) for b in bufs_ref]
        st_dev = xp.zeros((C, 4))
        kern = _cuda.get_kernel("cs_block")(
            *bufs_dev, st_dev, quantum=QUANTUM, threshold=THRESHOLD, cs_H=H,
            max_block=250,
        )  # fmt: skip
        out_ref, out_dev = [], []
        for a, b in ((0, 150), (150, 400)):  # fill, then past the H=100 wrap
            blk = np.ascontiguousarray(phi[:, a:b])
            corr_ref = blk.copy()
            cs_numba(blk, corr_ref, *bufs_ref, st_ref, QUANTUM, THRESHOLD)
            out_ref.append(corr_ref)
            corr_dev = xp.empty_like(xp.asarray(blk))
            kern(xp.asarray(blk), corr_dev)
            out_dev.append(corr_dev)

        corr_dev_full = xp.concatenate(out_dev, axis=1)
        xpt.assert_allclose(
            corr_dev_full, xp.asarray(np.concatenate(out_ref, axis=1)), atol=1e-9
        )
        xpt.assert_allclose(bufs_dev[0], xp.asarray(bufs_ref[0]), atol=1e-9)
        assert float(xp.abs(xp.diff(corr_dev_full[C - 1])).max()) < QUANTUM / 2.0

    def test_cs_block_rejects_bad_inputs(self, backend_device, xp):
        """The binding rejects wrong state buffers, the launch wrong phases."""
        skip_unless_gpu(backend_device)
        bind = _cuda.get_kernel("cs_block")

        C, B, H = 2, 16, 100
        phi = xp.zeros((C, B), dtype=xp.float64)
        corr = xp.empty_like(phi)
        buf_y = xp.zeros((C, H))
        ptr = xp.zeros(C, xp.int64)
        n = xp.zeros(C, xp.int64)
        st = xp.zeros((C, 4))
        kw = dict(quantum=QUANTUM, threshold=THRESHOLD, cs_H=H, max_block=B)

        with pytest.raises(ValueError, match="cs_buf_y"):
            bind(buf_y[:, : H // 2], ptr, n, st, **kw)
        with pytest.raises(TypeError, match="cs_buf_n"):
            bind(buf_y, ptr, n.astype(xp.int32), st, **kw)
        with pytest.raises(ValueError, match="C-contiguous"):
            bind(xp.asfortranarray(xp.zeros((C, H))), ptr, n, st, **kw)

        launch = bind(buf_y, ptr, n, st, **kw)
        with pytest.raises(ValueError, match="phi_blk"):
            launch(phi.astype(xp.float32), corr)
        with pytest.raises(ValueError, match="phi_blk"):
            launch(xp.zeros((C, B + 1)), xp.zeros((C, B + 1)))  # beyond max_block
        with pytest.raises(ValueError, match="phi_corr"):
            launch(phi, corr[:, :-1])


class TestWarmupMatchesCorrectCycleSlips:
    """The equalizers' repair is the recursion of correct_cycle_slips."""

    @pytest.mark.parametrize("H", [1, 4, 9, 30])
    def test_blocks_match_the_standalone_repair(self, backend_device, xp, xpt, H):
        """cs_block, fed in uneven blocks (Numba on the CPU, the CUDA kernel
        on the GPU), repairs a trajectory as correct_cycle_slips does with
        the same history, the warmup included: the newest value predicts
        while fewer than min(10, H) values are known, then the line.  A ramp
        of 0.9 rad per value separates the two predictors (the newest value
        is off by more than the threshold at every step, the line is not),
        so a warmup of 10 values for every history (the equalizers' former
        rule) fails at H = 4 and 9.
        """
        from commkit.recovery import correct_cycle_slips

        rng = np.random.default_rng(3)
        n = 400
        steps = np.zeros(n)
        steps[15:19] = [0.2, 0.4, 0.6, 0.8]  # flat, an accelerating turn, flat
        phi = np.cumsum(steps) + 0.01 * rng.standard_normal(n)
        phi[250:] += QUANTUM
        phi = np.ascontiguousarray(np.stack([phi, phi[::-1].copy()]))
        expected = correct_cycle_slips(phi, symmetry=4, history=H, threshold=THRESHOLD)

        C = phi.shape[0]
        bufs = [xp.zeros((C, H)), xp.zeros(C, xp.int64), xp.zeros(C, xp.int64)]
        stats = xp.zeros((C, 4))
        if backend_device == "gpu":
            kern = _cuda.get_kernel("cs_block")(
                *bufs, stats, quantum=QUANTUM, threshold=THRESHOLD, cs_H=H,
                max_block=200,
            )  # fmt: skip
        else:
            numba_kernel = _get_numba_cs_block()
        out = []
        for a, b in ((0, 3), (3, 11), (11, 200), (200, 400)):
            blk = xp.asarray(np.ascontiguousarray(phi[:, a:b]))
            corr = xp.empty_like(blk)
            if backend_device == "gpu":
                kern(blk, corr)
            else:
                numba_kernel(blk, corr, *bufs, stats, QUANTUM, THRESHOLD)
            out.append(to_numpy(corr))
        got = np.concatenate(out, axis=1)
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-9)
