"""Plain-Python reference implementations ("oracles") for sequential algorithms.

Each oracle is written directly from the textbook update equations, in
float64/complex128 throughout, with explicit per-symbol loops and no shared code
with the library.  Library kernels (Numba today, possibly CUDA later) are tested
against these on small inputs.  They exist to catch errors in the optimized
kernels *and* in the surrounding preparation (normalization, padding, initial
taps, training alignment), so the oracles re-derive that preparation from its
documented definition rather than calling library helpers.

Conventions shared with the library's documented contract:

- Input normalization: each channel is divided by ``rms(x_ch) * sqrt(sps)`` so
  the symbol-rate power is 1.  Training symbols are scaled to unit average power
  per channel.
- Regressor for output symbol ``n``: samples ``[n*sps, n*sps + T)`` of the input
  padded with ``min(T // 2, pad_total)`` leading zeros, where
  ``pad_total = n_sym*sps - n_samples + T - 1``.
- Butterfly filter: ``y_i = sum_j sum_t conj(W[i, j, t]) * X[j, t]``; initial
  taps are the centre-tap identity ``W[i, i, T // 2] = 1``.
"""

from __future__ import annotations

import numpy as np


def prepare_input(
    samples: np.ndarray, sps: int, num_taps: int
) -> tuple[np.ndarray, int, np.ndarray]:
    """Normalize and pad ``samples`` as the equalizers document.

    Returns ``(x_padded (C, N_pad) complex128, n_sym, norm (C,))``.
    """
    x = np.atleast_2d(np.asarray(samples)).astype(np.complex128)
    _, n_samples = x.shape
    n_sym = n_samples // sps
    norm = np.sqrt(np.mean(np.abs(x) ** 2, axis=-1)) * np.sqrt(sps)
    x = x / norm[:, None]
    pad_total = max(0, n_sym * sps - n_samples + num_taps - 1)
    pad_left = min(num_taps // 2, pad_total)
    x = np.pad(x, ((0, 0), (pad_left, pad_total - pad_left)))
    return x, n_sym, norm


def normalize_training(training: np.ndarray, num_ch: int) -> np.ndarray:
    """Unit average power per channel; a shared 1-D sequence is tiled to C."""
    t = np.atleast_2d(np.asarray(training)).astype(np.complex128)
    if t.shape[0] == 1 and num_ch > 1:
        t = np.repeat(t, num_ch, axis=0)
    return t / np.sqrt(np.mean(np.abs(t) ** 2, axis=-1, keepdims=True))


def identity_taps(num_ch: int, num_taps: int) -> np.ndarray:
    """Centre-tap identity butterfly ``(C, C, T)``."""
    w = np.zeros((num_ch, num_ch, num_taps), dtype=np.complex128)
    for i in range(num_ch):
        w[i, i, num_taps // 2] = 1.0
    return w


def _window(x_padded: np.ndarray, n: int, sps: int, num_taps: int) -> np.ndarray:
    return x_padded[:, n * sps : n * sps + num_taps]


def _filter(w: np.ndarray, window: np.ndarray) -> np.ndarray:
    """Butterfly output ``y_i = sum_{j,t} conj(W[i,j,t]) X[j,t]``."""
    return np.array([np.sum(np.conj(w[i]) * window) for i in range(w.shape[0])])


def _nearest(y: complex, constellation: np.ndarray) -> complex:
    return constellation[np.argmin(np.abs(y - constellation))]


def lms_reference(
    samples: np.ndarray,
    training: np.ndarray | None,
    constellation: np.ndarray,
    *,
    num_taps: int,
    sps: int,
    step_size: float,
) -> dict[str, np.ndarray]:
    """Data-aided then decision-directed butterfly LMS.

    ``e = d - y`` and ``W[i] <- W[i] + mu * conj(e_i) * X``.
    Returns ``y``, ``e`` of shape ``(C, n_sym)`` and final ``w`` ``(C, C, T)``.
    """
    x, n_sym, _ = prepare_input(samples, sps, num_taps)
    num_ch = x.shape[0]
    d_train = None if training is None else normalize_training(training, num_ch)
    n_train = 0 if d_train is None else min(d_train.shape[-1], n_sym)
    w = identity_taps(num_ch, num_taps)
    y_out = np.zeros((num_ch, n_sym), dtype=np.complex128)
    e_out = np.zeros_like(y_out)
    for n in range(n_sym):
        win = _window(x, n, sps, num_taps)
        y = _filter(w, win)
        for i in range(num_ch):
            d = d_train[i, n] if n < n_train else _nearest(y[i], constellation)
            e = d - y[i]
            w[i] = w[i] + step_size * np.conj(e) * win
            y_out[i, n], e_out[i, n] = y[i], e
    return {"y": y_out, "e": e_out, "w": w}


def rls_reference(
    samples: np.ndarray,
    training: np.ndarray | None,
    constellation: np.ndarray,
    *,
    num_taps: int,
    sps: int,
    forgetting_factor: float,
    delta: float,
    leakage: float = 0.0,
) -> dict[str, np.ndarray]:
    """Exponentially weighted (leaky) RLS on the stacked butterfly regressor.

    ``k = P x / (lam + x^H P x)``, ``W_i <- (1 - gamma) W_i + k conj(e_i)``,
    ``P <- (P - k x^H P) / lam`` with ``P(0) = I / delta``.  Updates stop
    ``T // 2`` symbols before the end and the output is truncated there (the
    trailing zero-pad zone), matching the documented ``tail_trim``.
    """
    x, n_sym, _ = prepare_input(samples, sps, num_taps)
    num_ch = x.shape[0]
    d_train = None if training is None else normalize_training(training, num_ch)
    n_train = 0 if d_train is None else min(d_train.shape[-1], n_sym)
    n_halt = max(0, n_sym - num_taps // 2)
    dim = num_ch * num_taps
    w = identity_taps(num_ch, num_taps).reshape(num_ch, dim)
    p = np.eye(dim, dtype=np.complex128) / delta
    lam = forgetting_factor
    y_out = np.zeros((num_ch, n_sym), dtype=np.complex128)
    e_out = np.zeros_like(y_out)
    for n in range(n_sym):
        xr = _window(x, n, sps, num_taps).reshape(dim)
        y = np.conj(w) @ xr
        e = np.empty(num_ch, dtype=np.complex128)
        for i in range(num_ch):
            d = d_train[i, n] if n < n_train else _nearest(y[i], constellation)
            e[i] = d - y[i]
        px = p @ xr
        k = px / (lam + np.real(np.conj(xr) @ px))
        if n < n_halt:
            w = (1.0 - leakage) * w + np.outer(np.conj(e), k)
            p = (p - np.outer(k, np.conj(xr) @ p)) / lam
        y_out[:, n], e_out[:, n] = y, e
    return {
        "y": y_out[:, :n_halt],
        "e": e_out[:, :n_halt],
        "w": w.reshape(num_ch, num_ch, num_taps),
    }


def cma_reference(
    samples: np.ndarray,
    *,
    num_taps: int,
    sps: int,
    step_size: float,
    r2: float,
) -> dict[str, np.ndarray]:
    """Godard p=2 CMA: ``e = y (|y|^2 - R2)``, ``W[i] <- W[i] - mu conj(e_i) X``."""
    return _blind_reference(
        samples, num_taps=num_taps, sps=sps, step_size=step_size, radii=None, r2=r2
    )


def rde_reference(
    samples: np.ndarray,
    *,
    num_taps: int,
    sps: int,
    step_size: float,
    radii: np.ndarray,
) -> dict[str, np.ndarray]:
    """Radius-directed equalizer: CMA with the target radius nearest to ``|y|``."""
    return _blind_reference(
        samples, num_taps=num_taps, sps=sps, step_size=step_size, radii=radii, r2=None
    )


def _blind_reference(samples, *, num_taps, sps, step_size, radii, r2):
    x, n_sym, _ = prepare_input(samples, sps, num_taps)
    num_ch = x.shape[0]
    w = identity_taps(num_ch, num_taps)
    y_out = np.zeros((num_ch, n_sym), dtype=np.complex128)
    e_out = np.zeros_like(y_out)
    for n in range(n_sym):
        win = _window(x, n, sps, num_taps)
        y = _filter(w, win)
        for i in range(num_ch):
            if radii is None:
                target2 = r2
            else:
                target2 = radii[np.argmin(np.abs(np.abs(y[i]) - radii))] ** 2
            e = y[i] * (np.abs(y[i]) ** 2 - target2)
            w[i] = w[i] - step_size * np.conj(e) * win
            y_out[i, n], e_out[i, n] = y[i], e
    return {"y": y_out, "e": e_out, "w": w}


# -----------------------------------------------------------------------------
# Carrier-phase recovery
# -----------------------------------------------------------------------------
#
# Shared contract: symbols are 1-SPS; BPS and the PLL normalize each channel to
# unit average power before estimation (Viterbi-Viterbi does not); the result is
# a per-symbol phase trajectory in radians, shape (C, N).


def _unit_power(symbols: np.ndarray) -> np.ndarray:
    s = np.atleast_2d(np.asarray(symbols)).astype(np.complex128)
    return s / np.sqrt(np.mean(np.abs(s) ** 2, axis=-1, keepdims=True))


def _block_interp(phi_blocks: np.ndarray, n: int, block_size: int) -> np.ndarray:
    """Linear interpolation between block centres, held flat at the edges."""
    centers = np.arange(phi_blocks.shape[-1]) * block_size + block_size / 2
    return np.interp(np.arange(n, dtype=np.float64), centers, phi_blocks)


def cycle_slip_reference(
    phi: np.ndarray, *, symmetry: int, history_length: int, threshold: float
) -> np.ndarray:
    """Sequential cycle-slip removal by linear extrapolation.

    For block ``b`` the expected phase is extrapolated from the last
    ``min(b, history_length)`` *corrected* blocks: the previous value while
    fewer than ``min(10, history_length)`` are available, otherwise a
    least-squares line evaluated one step past the newest block.  A deviation
    larger than ``threshold`` is removed in whole quanta of ``2*pi/symmetry``.
    """
    quantum = 2 * np.pi / symmetry
    out = np.array(phi, dtype=np.float64)
    for b in range(1, out.size):
        hist = out[max(0, b - history_length) : b]
        if hist.size < min(10, history_length):
            pred = hist[-1]
        else:
            slope, intercept = np.polyfit(np.arange(hist.size), hist, 1)
            pred = slope * hist.size + intercept
        diff = out[b] - pred
        k = round(diff / quantum)
        if abs(diff) > threshold and k != 0:
            out[b] -= k * quantum
    return out


def viterbi_viterbi_reference(
    symbols: np.ndarray,
    *,
    modulation: str,
    order: int,
    block_size: int,
    joint_channels: bool = False,
) -> np.ndarray:
    """M-th power block phase estimator (normalized Viterbi-Viterbi).

    QAM symbols are projected onto the unit circle and raised to the 4th
    power, M-PSK to the M-th.  Per block, ``angle(sum s^M) / M`` is unwrapped
    M-fold; QAM is bias-corrected by ``-pi/M``.  Independent MIMO channels are
    moved to channel 0's M-fold branch.

    Unlike BPS and the PLL, the input is *not* power-normalized: in joint mode
    each channel's M-th-power phasors are weighted by its amplitude^M.
    """
    s = np.atleast_2d(np.asarray(symbols)).astype(np.complex128)
    num_ch, n = s.shape
    m = order if modulation == "psk" else 4
    n_blocks = n // block_size
    blocks = s[:, : n_blocks * block_size].reshape(num_ch, n_blocks, block_size)
    if modulation == "qam":
        blocks = blocks / np.abs(blocks)
    sums = np.sum(blocks**m, axis=-1)
    if joint_channels and num_ch > 1:
        sums = np.repeat(np.sum(sums, axis=0, keepdims=True), num_ch, axis=0)
    phi = np.unwrap(np.angle(sums), axis=-1) / m
    if modulation == "qam":
        phi = phi - np.pi / m
    if not joint_channels:
        for c in range(1, num_ch):
            k = np.round(np.mean(phi[c] - phi[0]) * m / (2 * np.pi))
            phi[c] -= k * 2 * np.pi / m
    return np.stack([_block_interp(p, n, block_size) for p in phi])


def bps_reference(
    symbols: np.ndarray,
    constellation: np.ndarray,
    *,
    num_test_phases: int,
    block_size: int,
    joint_channels: bool = False,
) -> np.ndarray:
    """Blind phase search over ``[0, pi/2)``.

    Candidate ``b`` is ``b * pi / (2B)``.  Per block the candidate minimizing
    ``sum_n min_m |s_n exp(-j phi_b) - c_m|^2`` wins (summed over channels in
    joint mode); block phases are 4-fold unwrapped and interpolated.
    """
    s = _unit_power(symbols)
    num_ch, n = s.shape
    candidates = np.arange(num_test_phases) * (np.pi / 2 / num_test_phases)
    n_blocks = n // block_size
    metric = np.zeros((num_ch, n_blocks, num_test_phases))
    for c in range(num_ch):
        for blk in range(n_blocks):
            seg = s[c, blk * block_size : (blk + 1) * block_size]
            for k, phi_k in enumerate(candidates):
                rot = seg * np.exp(-1j * phi_k)
                d2 = np.abs(rot[:, None] - constellation[None, :]) ** 2
                metric[c, blk, k] = np.sum(np.min(d2, axis=-1))
    if joint_channels and num_ch > 1:
        metric = np.repeat(np.sum(metric, axis=0, keepdims=True), num_ch, axis=0)
    phi = np.unwrap(4 * candidates[np.argmin(metric, axis=-1)], axis=-1) / 4
    return np.stack([_block_interp(p, n, block_size) for p in phi])


def pll_reference(
    symbols: np.ndarray,
    constellation: np.ndarray,
    *,
    mu: float,
    beta: float,
    phase_init: float = 0.0,
) -> np.ndarray:
    """Decision-directed 2nd-order PLL, one loop per channel.

    ``y = s exp(-j phi)``, ``d = nearest(y)``, ``e = Im(y conj(d))``; the
    phase used for symbol ``n`` is recorded before
    ``phi += mu e + nu``, ``nu += beta e``.
    """
    s = _unit_power(symbols)
    out = np.zeros(s.shape)
    for c in range(s.shape[0]):
        phi, nu = phase_init, 0.0
        for n in range(s.shape[1]):
            y = s[c, n] * np.exp(-1j * phi)
            e = np.imag(y * np.conj(_nearest(y, constellation)))
            out[c, n] = phi
            phi = phi + mu * e + nu
            nu = nu + beta * e
    return out
