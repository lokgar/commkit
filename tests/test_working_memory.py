"""Bounded working memory of the batched paths (overlap-save, MI).

Each test shrinks the module's element budget so that a modest record spans
many batches, warms the call up (FFT plans and caches are allocated once),
and checks on the CPU the peak memory beyond the output against the input's
size: whole-record intermediates would be several times larger.
"""

import numpy as np
import pytest

from commkit import _overlap_save, metrics
from commkit.equalization import zf_equalizer
from commkit.filtering import ols_fir_filter
from commkit.mapping import Constellation
from tests.common.memory import peak_bytes


def _workspace(fn) -> int:
    """Peak bytes of ``fn()`` beyond its output, after one warm-up call."""
    fn()
    peak, out = peak_bytes(fn)
    return peak - (out.nbytes if isinstance(out, np.ndarray) else 0)


def _samples(n: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    return (rng.standard_normal((2, n)) + 1j * rng.standard_normal((2, n))).astype(
        np.complex64
    )


@pytest.fixture
def small_ols_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_overlap_save, "_OLS_ELEMENTS", 1 << 14)


@pytest.mark.usefixtures("small_ols_budget")
def test_ols_fir_filter_workspace() -> None:
    """About one padded input copy (whole-record batches: 7x the input)."""
    x = _samples(1 << 19)
    taps = np.hanning(63).astype(np.float32)
    assert _workspace(lambda: ols_fir_filter(x, taps, nfft=1024)) < 3 * x.nbytes


@pytest.mark.usefixtures("small_ols_budget")
def test_zf_equalizer_workspace() -> None:
    x = _samples(1 << 19)
    h = np.zeros((2, 2, 9), dtype=np.complex64)
    h[0, 0, 4] = h[1, 1, 4] = 1.0
    h[0, 1, 3] = h[1, 0, 5] = 0.2
    assert _workspace(lambda: zf_equalizer(x, h)) < 3 * x.nbytes


def test_mi_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    """Chunk intermediates of C x chunk x M (whole record: 190x the input)."""
    monkeypatch.setattr(metrics, "_MI_ELEMENTS", 1 << 16)
    c = Constellation.qam(64)
    rng = np.random.default_rng(0)
    tx = c.points[rng.integers(0, c.order, (2, 1 << 14))]
    noise = 0.05 * (rng.standard_normal(tx.shape) + 1j * rng.standard_normal(tx.shape))
    y = (tx + noise).astype(np.complex64)
    assert _workspace(lambda: metrics.mi(y, noise_var=0.005, constellation=c)) < (
        20 * y.nbytes
    )
