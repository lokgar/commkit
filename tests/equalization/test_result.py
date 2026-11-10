"""The scaling chain of ``EqualizerResult``: every factor between the received
samples and ``y_hat`` is in the result.

``input_norm_factor`` is the per-channel RMS of the input as given, and

    y_hat[i, n] = Σ_j Σ_t conj(W_n[i, j, t]) · x_j[n·sps - T_c + t] / a_j

with ``W_n`` the taps that produced symbol ``n``: row ``n - 1`` of
``weights_history`` for the sequential equalizers (row ``n`` holds the taps
after the update at ``n``), row ``n`` for ``block_lms``.  Rebuilt here in
float64 from the raw samples; measured agreement 1e-7 (float32 rounding).
"""

import numpy as np
import pytest

from commkit import equalization
from commkit.mapping import Constellation
from tests.common.conversions import to_numpy
from tests.common.signals import make_channel_signal

T = 7
SPS = 2
TRAINED = {"lms", "rls", "block_lms"}
SEQUENTIAL = {"lms", "rls", "cma", "rde"}


@pytest.mark.allow_warnings
@pytest.mark.parametrize("name", ["lms", "rls", "cma", "rde", "block_lms"])
@pytest.mark.parametrize("num_channels", [1, 2])
def test_output_is_rebuilt_from_raw_samples(xp, name, num_channels):
    c = Constellation.qam(16)
    rx = make_channel_signal(
        c,
        400,
        isi_taps=[0.2, 1.0, 0.1j],
        esn0_db=25.0,
        num_channels=num_channels,
        theta=0.3 if num_channels == 2 else None,
        seed=1,
        xp=xp,
    )
    kw = {"num_taps": T, "store_weights": True}
    if name != "rls":
        kw["step_size"] = 1e-3
    if name == "block_lms":
        kw["block_size"] = 16
    data = (rx.reference.symbols[..., :100],) if name in TRAINED else ()
    res = getattr(equalization, name)(rx, *data, **kw)

    x = np.atleast_2d(to_numpy(rx.samples)).astype(np.complex128)
    a = np.atleast_1d(np.asarray(res.input_norm_factor, dtype=np.float64))
    np.testing.assert_allclose(a, np.sqrt(np.mean(np.abs(x) ** 2, axis=-1)), rtol=1e-6)

    hist = to_numpy(res.weights_history).astype(np.complex128)
    hist = hist.reshape(hist.shape[0], num_channels, num_channels, T)
    if name in SEQUENTIAL:  # symbol n was produced by row n - 1
        hist = hist[:-1]
        first = 1
    else:
        first = 0
    n = np.arange(first, first + hist.shape[0])
    padded = np.pad(x / a[:, None], ((0, 0), (T // 2, T)))
    windows = padded[:, n[:, None] * SPS + np.arange(T)]  # (C, N, T)
    rebuilt = np.einsum("nijt,jnt->in", np.conj(hist), windows)
    y = np.atleast_2d(to_numpy(res.y_hat))[:, n]
    np.testing.assert_allclose(y, rebuilt, atol=1e-5)
