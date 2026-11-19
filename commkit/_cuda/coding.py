"""Launch wrappers of the decoder kernels of :mod:`commkit.coding` (private).

``ldpc_layered`` (layered belief propagation, one block per codeword) and
``algebraic_correct`` (BCH and Reed-Solomon, one thread per codeword).
"""

from collections.abc import Callable
from typing import Any


def _ldpc_layered_factory(sum_product: bool = False) -> Callable:
    """Wrapper factory for the layered LDPC decoder (``coding.ldpc``).

    One thread block per codeword (see ``ldpc.cu``); ``sum_product``
    selects the check-node rule (min-sum otherwise).  The launch decodes the
    codewords of ``app`` in place and returns the iterations run and the
    syndrome check per codeword.
    """
    import cupy as cp

    from . import compiler

    kern = compiler.get_raw_kernel(
        "ldpc", f"ldpc_layered<{'true' if sum_product else 'false'}>"
    )

    def launch(
        app: Any,
        row_ptr: Any,
        cols: Any,
        layer_ptr: Any,
        *,
        scale: float,
        offset: float,
        max_iterations: int,
        early_stop: bool,
    ) -> tuple[Any, Any]:
        """Decode ``app`` ``(B, N)`` float32 in place; ``(iters, valid)``."""
        compiler.check_arrays(
            ("app", app, cp.float32, (None, None)),
            ("row_ptr", row_ptr, cp.int32, (None,)),
            ("cols", cols, cp.int32, (None,)),
            ("layer_ptr", layer_ptr, cp.int32, (None,)),
        )
        B, N = app.shape
        m, E = row_ptr.size - 1, cols.size
        iters = cp.zeros(B, dtype=cp.int32)
        valid = cp.zeros(B, dtype=cp.uint8)
        if B == 0:
            return iters, valid.astype(bool)
        msg = cp.empty((B, E), dtype=cp.float32)
        kern(
            (B,),
            (256,),  # LDPC_THREADS in ldpc.cu
            (
                app, msg, row_ptr, cols, layer_ptr, cp.int32(layer_ptr.size - 1),
                cp.int32(m), cp.int32(N), cp.int32(E), cp.float32(scale),
                cp.float32(offset), cp.int32(max_iterations), cp.int32(early_stop),
                iters, valid,
            ),
        )  # fmt: skip
        return iters, valid.astype(bool)

    return launch


def _algebraic_correct_factory() -> Callable:
    """Wrapper factory for the BCH / Reed-Solomon decoder (``coding.algebraic``).

    One thread per codeword (see ``algebraic.cu``).  The launch corrects
    the hard words in place from their syndromes and returns the success
    per codeword.
    """
    import cupy as cp

    from . import compiler

    kern = compiler.get_raw_kernel("algebraic", "algebraic_correct")

    def launch(
        s: Any,
        exp: Any,
        log: Any,
        locator_logs: Any,
        words: Any,
        *,
        t: int,
        first_root: int,
        m: int,
        binary: bool,
    ) -> Any:
        """Correct ``words`` ``(B, n m)`` int8 in place; ``valid`` ``(B,)``."""
        B = words.shape[0]
        n = locator_logs.size
        compiler.check_arrays(
            ("s", s, cp.int64, (B, 2 * t)),
            ("exp", exp, cp.int64, (None,)),
            ("log", log, cp.int64, (None,)),
            ("locator_logs", locator_logs, cp.int64, (None,)),
            ("words", words, cp.int8, (B, n * m)),
        )
        if exp.size != 2 * log.size:
            raise ValueError("exp must be the doubled antilog table of log")
        valid = cp.zeros(B, dtype=cp.uint8)
        if B == 0:
            return valid.astype(bool)
        work = cp.empty((10 * t + 3, B), dtype=cp.int64)
        threads = 128  # ALG_THREADS in algebraic.cu
        kern(
            ((B + threads - 1) // threads,),
            (threads,),
            (
                s, exp, log, locator_logs, work, words, valid, cp.int32(B),
                cp.int32(n), cp.int32(t), cp.int32(first_root), cp.int32(m),
                cp.int32(binary), cp.int64(log.size - 1),
            ),
        )  # fmt: skip
        return valid.astype(bool)

    return launch
