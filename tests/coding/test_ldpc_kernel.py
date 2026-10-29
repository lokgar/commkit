"""The CUDA layered LDPC decoder against its Numba reference."""

from typing import Any

import numpy as np
import pytest

from commkit import _cuda
from commkit.backend import to_device
from commkit.coding import LDPCCode, MinSum, SumProduct, decode
from commkit.coding import ldpc as ldpc_module

pytestmark = pytest.mark.gpu_only


def _llrs(code: LDPCCode, rows: int, mean: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (mean + 1.5 * rng.standard_normal((rows, code.n))).astype(np.float32)


CODES = [LDPCCode.dvbs2("1/2", "short"), LDPCCode.pon()]


@pytest.mark.parametrize("code", CODES, ids=["dvbs2-short-1/2", "pon"])
@pytest.mark.parametrize("early_stop", [False, True])
def test_min_sum_bit_exact(backend_device: str, xp: Any, code, early_stop) -> None:
    llrs = _llrs(code, 6, 1.8, seed=1)
    method = MinSum(scale=0.75, offset=0.05, max_iterations=12, early_stop=early_stop)
    cpu = decode(llrs, code=code, method=method)
    gpu = decode(xp.asarray(llrs), code=code, method=method)
    np.testing.assert_array_equal(to_device(gpu.llrs, "cpu"), cpu.llrs)
    np.testing.assert_array_equal(to_device(gpu.bits, "cpu"), cpu.bits)
    np.testing.assert_array_equal(to_device(gpu.iterations, "cpu"), cpu.iterations)
    np.testing.assert_array_equal(to_device(gpu.valid, "cpu"), cpu.valid)


def test_sum_product_matches(backend_device: str, xp: Any) -> None:
    code = CODES[0]
    llrs = _llrs(code, 4, 0.8, seed=2)
    method = SumProduct(max_iterations=5, early_stop=False)
    cpu = decode(llrs, code=code, method=method)
    gpu = decode(xp.asarray(llrs), code=code, method=method)
    np.testing.assert_allclose(
        to_device(gpu.llrs, "cpu"), cpu.llrs, rtol=1e-3, atol=1e-3
    )


def test_chunks_agree(backend_device: str, xp: Any, monkeypatch) -> None:
    code = CODES[0]
    llrs = xp.asarray(_llrs(code, 5, 1.8, seed=3))
    method = MinSum(max_iterations=6, early_stop=False)
    whole = decode(llrs, code=code, method=method)
    monkeypatch.setattr(ldpc_module, "_GPU_MESSAGES", code.checks.size * 2)  # 2 rows
    part = decode(llrs, code=code, method=method)
    np.testing.assert_array_equal(
        to_device(part.llrs, "cpu"), to_device(whole.llrs, "cpu")
    )


def test_wrapper_checks(backend_device: str, xp: Any) -> None:
    code = CODES[0]
    row_ptr, cols, layer_ptr = (xp.asarray(a) for a in code._schedule)
    launch = _cuda.get_kernel("ldpc_layered")
    kw = dict(scale=1.0, offset=0.0, max_iterations=1, early_stop=True)
    with pytest.raises(ValueError, match="float32"):
        launch(xp.zeros((1, code.num_variables)), row_ptr, cols, layer_ptr, **kw)
    with pytest.raises(ValueError, match="int32"):
        launch(
            xp.zeros((1, code.num_variables), xp.float32), row_ptr.astype(xp.int64), cols,
            layer_ptr, **kw,
        )  # fmt: skip
    iters, valid = launch(
        xp.zeros((0, code.num_variables), xp.float32), row_ptr, cols, layer_ptr, **kw
    )
    assert iters.shape == (0,) and valid.shape == (0,)
