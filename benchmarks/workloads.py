"""Canonical fixed-seed workload generators for the benchmark suite.

All generation happens in NumPy on the host; benchmarks move data to the
active device with ``xp.asarray`` during setup (excluded from timing).
Workload IDs referenced by design-doc acceptance criteria are encoded in the
benchmark parametrization, e.g. ``bps/128cross/N1e6/C2``.
"""

import numpy as np

from commkit.helpers import normalize
from commkit.impairments import apply_awgn, apply_phase_noise
from commkit.mapping import gray_constellation

FS = 1e6  # nominal symbol rate / sampling rate at 1 SPS [Hz]


def qam_symbols(order: int, n_sym: int, num_ch: int = 1, seed: int = 0) -> np.ndarray:
    """Random unit-average-power QAM symbols. Shape (num_ch, n_sym) complex64
    (or (n_sym,) when num_ch == 1)."""
    rng = np.random.default_rng(seed)
    const = normalize(gray_constellation("qam", order), "average_power").astype(
        np.complex64
    )
    syms = const[rng.integers(0, order, (num_ch, n_sym))].astype(np.complex64)
    return syms if num_ch > 1 else syms[0]


def bps_workload(
    order: int,
    n_sym: int,
    num_ch: int = 2,
    snr_db: float = 25.0,
    linewidth_hz: float = 1e5,
    seed: int = 0,
) -> np.ndarray:
    """1-SPS QAM with Wiener laser phase noise + AWGN (CPR benchmark input)."""
    syms = qam_symbols(order, n_sym, num_ch, seed)
    x = apply_phase_noise(syms, sampling_rate=FS, linewidth=linewidth_hz, seed=seed)
    x = apply_awgn(x, esn0_db=snr_db, sps=1, seed=seed + 1)
    return np.asarray(x, dtype=np.complex64)


def mimo_equalizer_workload(
    n_sym: int,
    order: int = 16,
    sps: int = 2,
    snr_db: float = 25.0,
    mix_deg: float = 30.0,
    linewidth_hz: float = 0.0,
    seed: int = 0,
):
    """Dual-pol QAM at ``sps`` with static 2x2 unitary mixing + AWGN.

    Returns ``(samples (2, n_sym*sps) complex64, syms (2, n_sym) complex64)``.
    ``linewidth_hz > 0`` adds laser phase noise (for CPR-enabled equalizer
    benches).
    """
    syms = qam_symbols(order, n_sym, num_ch=2, seed=seed)
    x = np.repeat(syms, sps, axis=-1)  # (2, n_sym*sps), rectangular upsampling
    th = np.deg2rad(mix_deg)
    R = np.array(
        [[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]], dtype=np.complex64
    )
    x = (R @ x).astype(np.complex64)
    if linewidth_hz > 0:
        x = apply_phase_noise(
            x, sampling_rate=FS * sps, linewidth=linewidth_hz, seed=seed
        )
    x = apply_awgn(x, esn0_db=snr_db, sps=sps, seed=seed + 1)
    return np.asarray(x, dtype=np.complex64), syms


def rotated_symbols_workload(
    order: int = 16,
    n_sym: int = 100_000,
    num_ch: int = 2,
    rot_quadrants: int = 1,
    snr_db: float = 25.0,
    seed: int = 0,
):
    """Symbols with a fixed pi/2-quadrant rotation + AWGN, plus the clean
    reference - input for ``resolve_phase_ambiguity`` / EVM benches."""
    ref = qam_symbols(order, n_sym, num_ch, seed)
    rot = np.exp(1j * rot_quadrants * np.pi / 2).astype(np.complex64)
    y = apply_awgn(ref * rot, esn0_db=snr_db, sps=1, seed=seed + 1)
    return np.asarray(y, dtype=np.complex64), ref
