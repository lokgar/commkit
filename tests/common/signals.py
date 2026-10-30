"""Shared signal generators, waveforms, and channel impairment helpers for tests."""

from typing import Any

import numpy as np

from commkit import backend, generate
from commkit.core import Preamble, Signal, SingleCarrierFrame
from commkit.filtering import RRC
from commkit.mapping import Constellation
from commkit.math import normalize

from .conversions import device_of


def make_test_qam_samples(
    order: int = 16,
    num_symbols: int = 1000,
    sps: int = 2,
    snr_db: float | None = None,
    seed: int = 42,
    xp: Any = None,
) -> tuple[Any, Any]:
    """QAM symbols repeated ``sps`` times, with optional AWGN: no pulse and no
    channel.  For plumbing (shapes, layouts, state); an equalizer starting
    from the centre-tap identity already outputs these symbols up to a gain,
    so convergence tests use :func:`make_channel_signal` instead."""
    rng = np.random.default_rng(seed)
    const = Constellation.qam(order).points.astype(np.complex64)
    const = normalize(const, mode="average_power").astype(np.complex64)
    sym_indices = rng.integers(0, order, num_symbols)
    syms_np = const[sym_indices]

    if sps > 1:
        samples_np = np.repeat(syms_np, sps)
    else:
        samples_np = syms_np.copy()

    if snr_db is not None:
        noise_std = float(np.sqrt(10 ** (-snr_db / 10) / 2))
        noise = noise_std * (
            rng.standard_normal(len(samples_np))
            + 1j * rng.standard_normal(len(samples_np))
        ).astype(np.complex64)
        samples_np = samples_np + noise

    if xp is not None:
        return xp.asarray(samples_np), xp.asarray(syms_np)
    return samples_np, syms_np


def make_test_psk_samples(
    order: int = 4,
    num_symbols: int = 1000,
    sps: int = 2,
    snr_db: float | None = None,
    seed: int = 42,
    xp: Any = None,
) -> tuple[Any, Any]:
    """PSK symbols repeated ``sps`` times, with optional AWGN: no pulse and no
    channel (plumbing only, see :func:`make_test_qam_samples`)."""
    rng = np.random.default_rng(seed)
    const = Constellation.psk(order).points.astype(np.complex64)
    sym_indices = rng.integers(0, order, num_symbols)
    syms_np = const[sym_indices]

    if sps > 1:
        samples_np = np.repeat(syms_np, sps)
    else:
        samples_np = syms_np.copy()

    if snr_db is not None:
        noise_std = float(np.sqrt(10 ** (-snr_db / 10) / 2))
        noise = noise_std * (
            rng.standard_normal(len(samples_np))
            + 1j * rng.standard_normal(len(samples_np))
        ).astype(np.complex64)
        samples_np = samples_np + noise

    if xp is not None:
        return xp.asarray(samples_np), xp.asarray(syms_np)
    return samples_np, syms_np


def make_test_mimo_samples(
    num_channels: int = 2,
    order: int = 16,
    num_symbols: int = 1000,
    sps: int = 2,
    snr_db: float | None = None,
    seed: int = 42,
    xp: Any = None,
) -> tuple[Any, Any]:
    """Independent streams of :func:`make_test_qam_samples`, stacked with no
    mixing between them (plumbing only; see :func:`make_channel_signal`)."""
    samples_list = []
    symbols_list = []
    for ch in range(num_channels):
        s_samp, s_sym = make_test_qam_samples(
            order=order,
            num_symbols=num_symbols,
            sps=sps,
            snr_db=snr_db,
            seed=seed + ch * 1000,
            xp=None,
        )
        samples_list.append(s_samp)
        symbols_list.append(s_sym)

    samples_arr = np.stack(samples_list, axis=0)
    symbols_arr = np.stack(symbols_list, axis=0)

    if xp is not None:
        return xp.asarray(samples_arr), xp.asarray(symbols_arr)
    return samples_arr, symbols_arr


def make_channel_signal(
    constellation: Constellation,
    num_symbols: int,
    *,
    sps: int = 2,
    rolloff: float = 0.2,
    isi_taps: Any = None,
    frac_delay: float = 0.0,
    theta: float | None = None,
    dgd: float = 0.0,
    dgd_theta: float = np.pi / 8,
    esn0_db: float | None = None,
    num_channels: int = 1,
    symbol_rate: float = 1e9,
    seed: int = 0,
    xp: Any = None,
) -> Signal:
    """A pulse-shaped record through a channel an equalizer must undo.

    Independent streams (``num_channels``) of RRC-shaped symbols at ``sps``,
    generated periodic so that the circular channel below has no edge, then
    in order:

    - ``isi_taps``: circular convolution at the sample spacing; the largest
      tap is the main path, so reference symbol ``k`` stays at period ``k``.
    - ``frac_delay``: a common delay in samples (:func:`fft_fractional_delay`).
    - ``theta``: a polarization rotation (two channels).
    - ``dgd``: differential group delay in seconds at ``dgd_theta``.
    - ``esn0_db``: AWGN at that Es/N0.

    Returns the received :class:`Signal` (complex64), whose reference holds
    the transmitted symbols, on ``xp``'s device.
    """
    from commkit.impairments import (
        apply_awgn,
        apply_pmd,
        apply_polarization_mixing,
    )
    from commkit.timing import fft_fractional_delay

    sig = generate(
        constellation,
        num_symbols,
        symbol_rate=symbol_rate,
        sps=sps,
        pulse=RRC(rolloff),
        num_channels=num_channels,
        periodic=True,
        rng=seed,
    )
    x = np.asarray(sig.samples).astype(np.complex128)
    if isi_taps is not None:
        h = np.asarray(isi_taps, dtype=np.complex128)
        main = int(np.argmax(np.abs(h)))
        kernel = np.zeros(x.shape[-1], dtype=np.complex128)
        kernel[: h.size] = h
        kernel = np.roll(kernel, -main)
        x = np.fft.ifft(np.fft.fft(x, axis=-1) * np.fft.fft(kernel), axis=-1)
    x = x.astype(np.complex64)
    if frac_delay:
        x = fft_fractional_delay(x, delay=frac_delay)
    if theta is not None:
        x = apply_polarization_mixing(x, theta=theta)
    if dgd:
        x = apply_pmd(x, dgd=dgd, theta=dgd_theta, sampling_rate=sig.sampling_rate)
    if esn0_db is not None:
        x = apply_awgn(x, esn0_db=esn0_db, sps=sps, rng=seed + 1)
    out = sig.replace(samples=np.asarray(x).astype(np.complex64))
    if xp is not None:
        out = out.to(device_of(xp))
    return out


def make_test_qam_signal(
    order: int = 16,
    num_symbols: int = 1000,
    sps: int = 1,
    symbol_rate: float = 1e6,
    fo_hz: float = 0.0,
    snr_db: float | None = 30.0,
    seed: int = 42,
    xp: Any = None,
) -> Signal:
    """Generate a Signal container populated with QAM samples, optional frequency offset and AWGN."""
    from commkit import spectral
    from commkit.impairments import apply_awgn

    sig = generate(
        Constellation.qam(order),
        num_symbols,
        symbol_rate=symbol_rate,
        sps=sps,
        pulse=RRC(0.35),
        rng=seed,
    )
    if snr_db is not None:
        sig = sig.replace(
            samples=apply_awgn(sig.samples, esn0_db=snr_db, sps=sps, rng=seed)
        )
    if fo_hz != 0.0:
        shifted = spectral.shift_frequency(
            sig.samples, frequency=fo_hz, sampling_rate=symbol_rate * sps
        )
        sig = sig.replace(samples=shifted)
    if xp is not None:
        sig = sig.to(device_of(xp))
    return sig


def make_test_psk_signal(
    order: int = 4,
    num_symbols: int = 1000,
    sps: int = 1,
    symbol_rate: float = 1e6,
    fo_hz: float = 0.0,
    snr_db: float | None = 30.0,
    seed: int = 42,
    xp: Any = None,
) -> Signal:
    """Generate a Signal container populated with PSK samples, optional frequency offset and AWGN."""
    from commkit import spectral
    from commkit.impairments import apply_awgn

    sig = generate(
        Constellation.psk(order),
        num_symbols,
        symbol_rate=symbol_rate,
        sps=sps,
        pulse=RRC(0.35),
        rng=seed,
    )
    if snr_db is not None:
        sig = sig.replace(
            samples=apply_awgn(sig.samples, esn0_db=snr_db, sps=sps, rng=seed)
        )
    if fo_hz != 0.0:
        shifted = spectral.shift_frequency(
            sig.samples, frequency=fo_hz, sampling_rate=symbol_rate * sps
        )
        sig = sig.replace(samples=shifted)
    if xp is not None:
        sig = sig.to(device_of(xp))
    return sig


def make_test_mimo_signal(
    num_channels: int = 2,
    order: int = 16,
    num_symbols: int = 1000,
    sps: int = 2,
    symbol_rate: float = 1e9,
    seed: int = 42,
    xp: Any = None,
) -> Signal:
    """Generate a 2x2 or NxN MIMO Signal."""

    sig = generate(
        Constellation.qam(order),
        num_symbols,
        symbol_rate=symbol_rate,
        sps=sps,
        pulse=RRC(0.35),
        num_channels=num_channels,
        rng=seed,
    )
    if xp is not None:
        sig = sig.to(device_of(xp))
    return sig


def make_test_frame_signal(
    payload_num_symbols: int = 203,
    preamble_num_symbols: int = 13,
    sps: int = 4,
    symbol_rate: float = 1e9,
    payload_mod_order: int = 16,
    seed: int = 42,
    xp: Any = None,
) -> Signal:
    """Generate a SingleCarrierFrame converted to a Signal."""
    frame = SingleCarrierFrame(
        payload_num_symbols=payload_num_symbols,
        payload_constellation=Constellation.qam(payload_mod_order),
        preamble=Preamble(sequence_type="barker", num_symbols=preamble_num_symbols),
        pilot_pattern="comb",
        pilot_period=8,
        pilot_constellation=Constellation.psk(4),
        guard_type="zero",
        guard_num_symbols=4,
    )
    sig = frame.to_signal(sps=sps, symbol_rate=symbol_rate, pulse=RRC(0.35))
    if xp is not None:
        sig = sig.to(device_of(xp))
    return sig


def make_wiener_phase(
    num_symbols: int = 1000,
    linewidth: float = 100e3,
    sample_rate: float = 10e9,
    seed: int = 42,
    dtype: Any = np.float32,
    xp: Any = None,
) -> Any:
    """Generate Wiener phase noise trajectory theta[n]."""
    rng = np.random.default_rng(seed)
    var_per_sample = 2.0 * np.pi * linewidth / sample_rate
    increments = rng.normal(0.0, np.sqrt(var_per_sample), num_symbols)
    phase = np.cumsum(increments).astype(dtype)
    if xp is not None:
        return xp.asarray(phase)
    return phase


def make_isi_distorted_signal(
    xp: Any,
    mod: str,
    order: int,
    n_symbols: int,
    seed: int,
    channel: Any,
    noise: float = 0.02,
) -> tuple[Any, Any]:
    """Build a pulse-shaped, ISI-distorted, noisy signal on the xp device."""
    from commkit import RRC, generate
    from commkit.mapping import Constellation
    from tests.common.conversions import to_numpy

    constellation = (
        Constellation.qam(order) if mod == "qam" else Constellation.psk(order)
    )
    sig = generate(
        constellation, n_symbols, symbol_rate=1e6, sps=2, pulse=RRC(0.35), rng=seed
    )
    # Known symbols on the constellation's scale, as the equalizers take them.
    tx = xp.asarray(to_numpy(sig.reference.symbols))
    rx = xp.convolve(
        xp.asarray(to_numpy(sig.samples)), xp.asarray(channel), mode="same"
    )
    rng = xp.random.RandomState(seed)
    rx = rx + noise * (rng.randn(len(rx)) + 1j * rng.randn(len(rx))).astype(
        xp.complex64
    )
    return tx, rx.astype(xp.complex64)


def make_ambiguous_qam16(
    n_sym: int = 2000, corrupt_head: int = 500, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Return (symbols, ref) where the first corrupt_head symbols are rotated by pi/2."""
    rng = np.random.default_rng(seed)
    const = Constellation.qam(16).points.astype(np.complex64)
    const /= np.sqrt(np.mean(np.abs(const) ** 2))
    ref = const[rng.integers(0, 16, n_sym)]
    rot1 = np.exp(1j * np.pi / 2).astype(np.complex64)
    symbols = ref * rot1
    symbols[:corrupt_head] = ref[:corrupt_head] * np.exp(1j * np.pi).astype(
        np.complex64
    )
    return symbols, ref


def apply_phase_ramp(
    symbols: Any, delta_omega: float, phi0: float = 0.0, xp: Any = None
) -> Any:
    """Multiply symbols by exp(j * (delta_omega * n + phi0))."""
    if xp is None:
        xp = backend.get_array_module(symbols)
    n = xp.arange(symbols.shape[-1], dtype=xp.float64)
    ramp = xp.exp(1j * (delta_omega * n + phi0)).astype(symbols.dtype)
    return symbols * ramp


def make_dsh_beat(
    linewidth: float = 2e6,
    num_samples: int = 1 << 14,
    delay_samples: int = 250,
    f_shift: float = 80e6,
    snr_db: float | None = None,
    seed: int = 0,
    sample_rate: float = 500e6,
    xp: Any = None,
) -> tuple[Any, Any]:
    """Generate complex DSH beat + true differential phase."""
    from commkit import analysis
    from commkit.backend import to_device
    from commkit.impairments import apply_awgn, generate_phase_noise

    phi = to_device(
        generate_phase_noise(
            num_samples=num_samples + delay_samples,
            sampling_rate=sample_rate,
            linewidth=linewidth,
            rng=seed,
        ),
        "cpu",
    )
    z, dphi = analysis.dsh_beat(
        phi,
        sampling_rate=sample_rate,
        delay=delay_samples / sample_rate,
        f_shift=f_shift,
    )
    if snr_db is not None:
        z = apply_awgn(z, sps=1, esn0_db=snr_db, rng=seed + 100)
    if xp is not None:
        return xp.asarray(z), xp.asarray(dphi)
    return z, dphi


def make_adapter_test_signal(xp: Any = None, **metadata: Any) -> Signal:
    """Construct a dummy Signal for adapter testing."""
    if xp is None:
        xp = np
    return Signal(
        samples=xp.ones(16, dtype=xp.complex64),
        sampling_rate=2e6,
        symbol_rate=1e6,
        **metadata,
    )


def make_test_symbols(
    scheme: str = "qam",
    order: int = 16,
    num_symbols: int = 1000,
    seed: int = 42,
    xp: Any = None,
) -> Any:
    """Generate normalized constellation symbols (QAM or PSK)."""
    rng = np.random.default_rng(seed)
    const = getattr(Constellation, scheme)(order).points.astype(np.complex64)
    const = normalize(const, mode="average_power").astype(np.complex64)
    syms = const[rng.integers(0, order, num_symbols)]
    if xp is not None:
        return xp.asarray(syms)
    return syms


def make_dual_pol_dsh_beat(
    linewidth: float = 2e5,
    num_samples: int = 1 << 19,
    delay_samples: int = 2450,
    f_shift: float = 80e6,
    snr_db: float | None = None,
    sop_rate: float = 2e3,
    seed: int = 0,
    sample_rate: float = 500e6,
) -> tuple[np.ndarray, np.ndarray]:
    """Two-polarization DSH beat whose SOP fades each channel in turn.

    The beat arrives in the Jones vector ``(cos α, sin α·e^{jβ})``: ``α``
    swings between 0 and π/2 (a full fade of one channel, then the other)
    and ``β`` rotates, both at about ``sop_rate``.  Independent complex
    noise of variance ``10^(-snr_db/10)`` (relative to the total beat power)
    is added per channel.  Returns ``(z, delta_phi)``: ``(2, N)`` beat and the
    true differential phase.
    """
    z, dphi = make_dsh_beat(
        linewidth, num_samples, delay_samples, f_shift, None, seed, sample_rate
    )
    t = np.arange(z.shape[-1]) / sample_rate
    alpha = np.pi / 2 * np.sin(2 * np.pi * sop_rate * t + seed) ** 2
    beta = 2 * np.pi * 1.7 * sop_rate * t
    jones = np.stack([np.cos(alpha), np.sin(alpha) * np.exp(1j * beta)])
    z2 = jones * z
    if snr_db is not None:
        rng = np.random.default_rng(seed + 1000)
        std = float(np.sqrt(10 ** (-snr_db / 10) / 2))
        z2 = z2 + std * (
            rng.standard_normal(z2.shape) + 1j * rng.standard_normal(z2.shape)
        )
    return z2, dphi
