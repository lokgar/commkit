"""Launch wrappers of the carrier-recovery kernels (private).

``bps_min_d2`` (the BPS block metric of :mod:`commkit.recovery.bps`), and
the two per-block scans of ``block_lms``: ``cs_block`` (cycle slips) and
``bps_anchor`` (the data-aided BPS anchor).
"""

from collections.abc import Callable
from typing import Any


def _bps_min_d2_factory(mode: str = "table", return_argmin: bool = False) -> Callable:
    """Wrapper factory for the fused BPS minimum-distance kernel.

    Computes ``min_d2[p, c, n] = min_m |x[c, n] * phasor[p] - const[m]|**2``
    in one pass. ``mode="table"`` searches an explicit constellation table;
    ``mode="grid"`` snaps to the uniform square-QAM level grid. With
    ``return_argmin=True`` (TABLE only) the nearest-point indices are
    returned alongside the distances.
    """
    import cupy as cp

    from . import compiler

    mode_id = {"table": 0, "grid": 1}[mode]
    if mode_id == 1 and return_argmin:
        raise ValueError("return_argmin is only supported in TABLE mode")
    kern = compiler.get_raw_kernel(
        "bps_min_d2",
        f"bps_min_d2<{mode_id}, {'true' if return_argmin else 'false'}>",
    )
    block_n = 128
    _empty_i32 = cp.empty(0, dtype=cp.int32)

    def launch(
        x: Any,
        phasor: Any,
        constellation: Any = None,
        lev_min: float = 0.0,
        d_grid: float = 1.0,
        side: int = 0,
    ) -> Any:
        """Return min_d2 (P, C, N) float32; with argmin, an (min_d2, idx) tuple.

        `x` is (C, N) complex64 with time on the last axis; `phasor` is
        (P,) complex64, P <= 128. TABLE mode requires `constellation`
        ((M,) complex64, M <= 1024); GRID mode requires the level-grid
        scalars `lev_min`, `d_grid`, `side`.
        """
        x = cp.ascontiguousarray(x)
        phasor = cp.ascontiguousarray(phasor)
        if x.ndim != 2:
            raise ValueError(f"x must be 2-D (C, N), got shape {x.shape}")
        if x.dtype != cp.complex64 or phasor.dtype != cp.complex64:
            raise TypeError(
                f"x and phasor must be complex64, got {x.dtype}/{phasor.dtype}"
            )
        C, N = x.shape
        P = int(phasor.size)
        if P < 1 or P > 128:
            raise ValueError(f"phasor count must be in [1, 128], got {P}")

        if mode_id == 0:
            constellation = cp.ascontiguousarray(constellation)
            if constellation.dtype != cp.complex64:
                raise TypeError(
                    f"constellation must be complex64, got {constellation.dtype}"
                )
            M = int(constellation.size)
            if M < 1 or M > 1024:
                raise ValueError(f"constellation size must be in [1, 1024], got {M}")
            shared_mem = M * 8
        else:
            if side < 2:
                raise ValueError(f"GRID mode requires side >= 2, got {side}")
            constellation = _empty_i32  # never dereferenced (M == 0)
            M = 0
            shared_mem = 0

        min_d2 = cp.empty((P, C, N), dtype=cp.float32)
        argmin = cp.empty((P, C, N), dtype=cp.int32) if return_argmin else _empty_i32
        if C == 0 or N == 0:  # a zero-size grid is a launch error
            return (min_d2, argmin) if return_argmin else min_d2
        # Channels beyond the grid's z limit (65535) run in the kernel's
        # channel loop.
        grid = ((N + block_n - 1) // block_n, P, min(C, 65535))
        kern(
            grid,
            (block_n,),
            (
                x,
                phasor,
                constellation,
                cp.float32(lev_min),
                cp.float32(d_grid),
                cp.int32(side),
                cp.int32(M),
                cp.int32(C),
                cp.int64(N),
                min_d2,
                argmin,
            ),
            shared_mem=shared_mem,
        )
        if return_argmin:
            return min_d2, argmin
        return min_d2

    return launch


def _cs_block_factory() -> Callable:
    """Wrapper factory for the block_lms cycle-slip correction kernel.

    Per-channel slip detector (one block of threads per channel, decisions
    speculated in parallel; see ``cs_block.cu``) operating in place on
    device-resident state buffers. One launch processes one equalizer block,
    instead of a round trip per block through the CPU path (device to host,
    the Numba kernel, host to device).

    The wrapper binds one run: it checks the state buffers (float64 or
    int64, changed in place, so non-contiguous ones are rejected instead of
    copied silently) and allocates the kernel's scratch for the largest
    block once, and returns the per-block launch, which checks only the
    block's phases.
    """
    import cupy as cp

    from . import compiler

    kern = compiler.get_raw_kernel("cs_block", "cs_block")

    def bind(
        cs_buf_y: Any,
        cs_buf_ptr: Any,
        cs_buf_n: Any,
        cs_stats: Any,
        *,
        quantum: float,
        threshold: float,
        cs_H: int,
        max_block: int,
    ) -> Callable[[Any, Any], Any]:
        """Bind the state buffers of one run.

        ``C`` channels (the length of ``cs_buf_ptr``) and blocks of at most
        ``max_block`` symbols.
        """
        C = cs_buf_ptr.shape[0] if cs_buf_ptr.ndim == 1 else -1
        cs_H = int(cs_H)
        for label, arr, dtype, shape in (
            ("cs_buf_y", cs_buf_y, cp.float64, (C, cs_H)),
            ("cs_buf_ptr", cs_buf_ptr, cp.int64, (C,)),
            ("cs_buf_n", cs_buf_n, cp.int64, (C,)),
            ("cs_stats", cs_stats, cp.float64, (C, 4)),
        ):
            if arr.dtype != dtype:
                raise TypeError(f"{label} must be {dtype}, got {arr.dtype}")
            if arr.shape != shape:
                raise ValueError(f"{label} must have shape {shape}, got {arr.shape}")
            if not arr.flags.c_contiguous:
                raise ValueError(f"{label} must be C-contiguous (mutated in place)")
        if max_block < 1:
            raise ValueError(f"max_block must be >= 1, got {max_block}")
        # Scratch for the largest block: the kernel strides by the actual B.
        L = cs_H + max_block
        z = cp.empty(C * L, dtype=cp.float64)
        p1 = cp.empty(C * (L + 1), dtype=cp.float64)
        p2 = cp.empty(C * (L + 1), dtype=cp.float64)
        guess = cp.empty(C * max_block, dtype=cp.int32)
        decided = cp.empty(C * max_block, dtype=cp.int32)
        scalars = (cp.float64(quantum), cp.float64(threshold), cp.int32(cs_H))

        def launch(phi_blk: Any, phi_corr: Any) -> Any:
            """Repair the cycle slips of ``phi_blk`` into ``phi_corr``.

            Both are ``(C, B)`` float64, ``1 <= B <= max_block``; the bound
            history buffers are updated in place.
            """
            shape = phi_blk.shape
            if (
                phi_blk.dtype != cp.float64
                or len(shape) != 2
                or shape[0] != C
                or not 1 <= shape[1] <= max_block
                or not phi_blk.flags.c_contiguous
            ):
                raise ValueError(
                    f"phi_blk must be C-contiguous float64 (C={C}, B<={max_block}), "
                    f"got {phi_blk.dtype} {shape}"
                )
            if (
                phi_corr.dtype != cp.float64
                or phi_corr.shape != shape
                or not phi_corr.flags.c_contiguous
            ):
                raise ValueError(
                    f"phi_corr must be C-contiguous float64 {shape}, got "
                    f"{phi_corr.dtype} {phi_corr.shape}"
                )
            kern(
                (C,),
                (256,),  # CS_THREADS in cs_block.cu
                (
                    phi_blk, phi_corr, cs_buf_y, cs_buf_ptr, cs_buf_n, cs_stats,
                    z, p1, p2, guess, decided, *scalars, cp.int32(shape[1]),
                ),
            )  # fmt: skip
            return phi_corr

        return launch

    return bind


def _bps_anchor_factory() -> Callable:
    """Wrapper factory for the block_lms data-aided BPS anchor kernel.

    One launch scans the training columns of one equalizer block (one block
    of 256 threads), keeping the block loop free of the host round
    trip of the NumPy reference (``_BlockBps._anchor``).  ``offset4`` and ``prev4``
    are updated in place; the anchored phase and the new trailing products
    (complex64) are returned.  The window sums are float32 and the unwrap
    float64.
    """
    import cupy as cp

    from . import compiler

    kern = compiler.get_raw_kernel("bps_anchor", "bps_anchor")

    def launch(
        y: Any,
        d: Any,
        hist: Any,
        offset4: Any,
        prev4: Any,
        *,
        symmetry: int,
        joint: bool,
    ) -> tuple[Any, Any]:
        """Anchored phase ``(C, n)`` float64 and the new ``(C, K-1)`` history."""
        if y.ndim != 2:
            raise ValueError(f"y must be 2-D (C, n), got shape {y.shape}")
        C, n = y.shape
        if n < 1:
            raise ValueError("y must have at least one training column")
        H = hist.shape[-1]
        for label, arr, dtype, shape in (
            ("y", y, cp.complex64, (C, n)),
            ("d", d, cp.complex64, (C, n)),
            ("hist", hist, cp.complex64, (C, H)),
            ("offset4", offset4, cp.float64, (C,)),
            ("prev4", prev4, cp.float64, (C,)),
        ):
            if arr.dtype != dtype:
                raise TypeError(f"{label} must be {dtype}, got {arr.dtype}")
            if arr.shape != shape:
                raise ValueError(f"{label} must have shape {shape}, got {arr.shape}")
            if not arr.flags.c_contiguous:
                raise ValueError(f"{label} must be C-contiguous")
        phi = cp.empty((C, n), dtype=cp.float64)
        hist_out = cp.empty((C, H), dtype=cp.complex64)
        total = cp.zeros(n if joint else 0, dtype=cp.complex64)
        kern(
            (1,),
            (256,),
            (
                y,
                d,
                hist,
                hist_out,
                total,
                offset4,
                prev4,
                phi,
                cp.int32(C),
                cp.int32(n),
                cp.int32(H + 1),
                cp.int32(1 if joint else 0),
                cp.float64(symmetry),
            ),
        )
        return phi, hist_out

    return launch
