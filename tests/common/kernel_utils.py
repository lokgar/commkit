"""CUDA/Numba kernel testing helpers."""

from typing import Any

import pytest

from commkit import _cuda


def skip_unless_kernel_available(
    arg1: str | None = None,
    arg2: str | None = None,
    *,
    backend_device: str | None = None,
    kernel_name: str | None = None,
) -> None:
    """Skip test unless CUDA kernel infrastructure is available and device is GPU.

    Supports both:
    - ``skip_unless_kernel_available(backend_device, kernel_name)``
    - ``skip_unless_kernel_available(kernel_name, backend_device=backend_device)``
    """
    dev = backend_device
    kname = kernel_name

    if arg1 is not None:
        if arg1 in ("cpu", "gpu"):
            dev = arg1
            if arg2 is not None:
                kname = arg2
        else:
            kname = arg1
            if arg2 is not None and arg2 in ("cpu", "gpu"):
                dev = arg2

    if dev is None:
        dev = "gpu"

    if dev != "gpu":
        pytest.skip("Test requires a GPU (CUDA device)")
    if not _cuda.is_available():
        pytest.skip("CUDA device below compute capability 7.0 or driver unavailable")
    if kname is not None:
        kern = _cuda.get_kernel(kname)
        if kern is None:
            pytest.skip(f"Kernel {kname!r} not available on this platform")


def reset_warned_kernels(monkeypatch: Any) -> None:
    """Isolate the warn-once bookkeeping between tests."""
    monkeypatch.setattr(_cuda, "_warned_kernels", set())


def reference_cs_block(
    phi_blk: Any,
    phi_corr: Any,
    cs_buf_y: Any,
    cs_buf_ptr: Any,
    cs_buf_n: Any,
    cs_stats: Any,
    quantum: float,
    threshold: float,
    H: int,
) -> None:
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


def slip_workload(C: int = 3, n_total: int = 1000, seed: int = 7) -> Any:
    """Slow ramp + noise with deliberate per-channel pi/2 slips."""
    import numpy as np

    quantum = float(2.0 * np.pi / 4.0)
    rng = np.random.default_rng(seed)
    phi = 0.3 * np.sin(np.linspace(0.0, 3.0, n_total))[None, :]
    phi = phi + 0.001 * np.arange(n_total)[None, :]
    phi = phi + 0.02 * rng.standard_normal((C, n_total))
    for ch, pos in ((0, 300), (1, 500), (2, 750), (0, 760)):
        phi[ch % C, pos:] += quantum
    return np.ascontiguousarray(phi)


def reference_bps_d2(x: Any, phasor: Any, const: Any) -> Any:
    """Full (P, C, N, M) candidate-distance tensor in float64."""
    import numpy as np

    xr = x[None, :, :].astype(np.complex128) * phasor[:, None, None].astype(
        np.complex128
    )
    return np.abs(xr[..., None] - const[None, None, None, :].astype(np.complex128)) ** 2
