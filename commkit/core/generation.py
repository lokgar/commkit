"""
Signal generation factories.

Free functions that construct :class:`Signal` instances for the supported
modulation formats (``generate`` plus the ``generate_pam``/``generate_psk``/
``generate_qam``/``generate_psqam`` wrappers); they are re-exported at the
package top level (``commkit.generate_qam(...)`` etc.).

All factories follow a bit-first architecture: random bits are generated, mapped
to symbols, upsampled, and pulse-shaped, with samples normalized to unit symbol
power (Es = 1, average sample power = 1/sps).
"""

from typing import Literal, cast

import numpy as np

from .. import filtering, helpers, mapping
from ..backend import ArrayType, is_cupy_available, to_device
from ..logger import logger
from .signal import Signal


def generate(
    num_symbols: int,
    sps: int,
    symbol_rate: float,
    modulation: str,
    order: int,
    unipolar: bool = False,
    rz: bool = False,
    pulse_shape: str = "none",
    num_streams: int = 1,
    seed: int | None = None,
    duty_cycle: float = 1.0,
    filter_span: int = 10,
    rrc_rolloff: float = 0.35,
    rc_rolloff: float = 0.35,
    rise_time: float = 0.0,
) -> "Signal":
    """
    Generates a generic baseband waveform with specified modulation.

    This is the primary factory method for creating synthetic signals.
    It follows a bit-first architecture: random bits are generated,
    mapped to symbols, upsampled, and pulse-shaped.

    Parameters
    ----------
    num_symbols : int
        Number of symbols to generate per stream.
    sps : float
        Samples per symbol.
    symbol_rate : float
        Symbol rate in symbols per second (Baud).
    modulation : {"psk", "qam", "ask"}
        The modulation scheme identifier.
    order : int
        Modulation order (e.g., 4, 16, 64).
    unipolar : bool, default False
        If True, uses a unipolar constellation.
    rz : bool, default False
        If True, uses Return-to-Zero signaling.
    pulse_shape : str, default "none"
        Pulse shaping filter type (e.g., ``'rrc'``, ``'rect'``).
    num_streams : int, default 1
        Number of independent streams (MIMO).
    seed : int, optional
        Seed for reproducible random generation.
    duty_cycle : float, default 1.0
        Fraction of the symbol period occupied by the pulse (rect/smoothrect).
        Overridden to 0.5 when ``rz=True``.
    filter_span : int, default 10
        Filter span in symbols for smoothrect/gaussian/rrc/rc/sinc.
    rrc_rolloff : float, default 0.35
        Roll-off factor for the RRC filter.
    rc_rolloff : float, default 0.35
        Roll-off factor for the RC filter.
    rise_time : float, default 0.22
        10%-90% edge transition duration in symbol periods for smoothrect.
    duty_cycle : float, default 1.0
        FWHM of the Gaussian pulse in symbol periods.

    Returns
    -------
    Signal
        A new `Signal` instance.

    Notes
    -----
    Samples are normalized to unit symbol power (Es = 1, average sample power = 1/sps).
    Call ``resolve_symbols()`` before demapping or computing metrics.
    """

    if sps != int(sps) or sps < 1:
        logger.warning(
            "sps=%r is not a positive integer. Non-integer sps is valid for "
            "captured/resampled signals but not for generation: "
            "resample_poly requires an integer upsampling factor, so the "
            "sample buffer would not match the stored sampling_rate "
            "metadata. To generate at a fractional sps, generate at an "
            "integer sps then call Signal.resample(up=..., down=...).",
            sps,
        )
        raise ValueError(
            f"sps must be a positive integer for signal generation, got {sps!r}."
        )
    sps = int(sps)

    # When rz=True and the caller hasn't specified a custom duty_cycle,
    # default to 50% (canonical RZ). Explicit duty_cycle values are preserved.
    if rz and duty_cycle == 1.0:
        duty_cycle = 0.5

    # Bit-first architecture: generate bits -> map to symbols
    k = int(np.log2(order))  # bits per symbol
    total_symbols = num_symbols * num_streams
    total_bits = total_symbols * k

    # Generate source bits
    bits = helpers.generate_bits(total_bits, seed=seed)

    # Map bits to symbols
    symbols_flat = mapping.map_bits(bits, modulation, order, unipolar)

    if num_streams > 1:
        # Shape: (Channels, Time)
        symbols = symbols_flat.reshape(num_streams, num_symbols)
        bits = bits.reshape(num_streams, num_symbols * k)
    else:
        symbols = symbols_flat

    if is_cupy_available():
        symbols = to_device(symbols, "gpu")
        bits = to_device(bits, "gpu")

    # Apply pulse shaping
    # shape_pulse defaults to axis=-1 (Time) which is correct for (C, T)
    samples = filtering.shape_pulse(
        symbols=symbols,
        sps=sps,
        pulse_shape=pulse_shape,
        rz=rz,
        duty_cycle=duty_cycle,
        filter_span=filter_span,
        rrc_rolloff=rrc_rolloff,
        rc_rolloff=rc_rolloff,
        rise_time=rise_time,
    )

    logger.info(
        "Generated %s-%s signal: %s symbols x %s stream(s), sps=%s, "
        "pulse_shape=%s, %s samples/stream @ %.3g Sa/s.",
        modulation.upper(),
        order,
        num_symbols,
        num_streams,
        sps,
        pulse_shape,
        samples.shape[-1],
        symbol_rate * sps,
    )

    return Signal(
        samples=samples,
        sampling_rate=symbol_rate * sps,
        symbol_rate=symbol_rate,
        mod_scheme=modulation.upper(),
        mod_order=order,
        mod_unipolar=unipolar,
        mod_rz=rz,
        source_bits=bits,
        source_symbols=symbols,
        pulse_shape=pulse_shape,
        filter_span=filter_span,
        rrc_rolloff=rrc_rolloff,
        rc_rolloff=rc_rolloff,
        rise_time=rise_time,
        duty_cycle=duty_cycle,
    )


def generate_pam(
    num_symbols: int,
    sps: int,
    symbol_rate: float,
    order: int,
    unipolar: bool = False,
    rz: bool = False,
    pulse_shape: Literal["rect", "smoothrect"] = "rect",
    num_streams: int = 1,
    seed: int | None = None,
    duty_cycle: float = 1.0,
    filter_span: int = 10,
    rise_time: float = 0.0,
) -> "Signal":
    """
    Generates a Pulse Amplitude Modulation (PAM) baseband waveform.

    Supports both NRZ (Non-Return-to-Zero) and RZ (Return-to-Zero)
    signaling, with configurable pulse shaping and bipolar/unipolar
    constellations.

    Parameters
    ----------
    num_symbols : int
        Total number of symbols to generate per stream.
    sps : int
        Samples per symbol. For RZ mode, this must be an even integer.
    symbol_rate : float
        Symbol rate in symbols per second (Baud).
    order : int
        Modulation order (e.g., 2, 4, 8).
    unipolar : bool, default False
        If True, uses a unipolar constellation starting from 0 (e.g., 0, 1).
        If False, uses a symmetric bipolar constellation (e.g., -1, +1).
    rz : bool, default False
        If True, uses Return-to-Zero signaling.
    pulse_shape : {"rect", "smoothrect"}, default "rect"
        Pulse shaping filter type. Default is "rect" for PAM.
    num_streams : int, default 1
        Number of independent streams (channels) to generate.
    seed : int, optional
        Random seed for reproducible bit and symbol generation.
    duty_cycle : float, default 1.0
        Fraction of the symbol period occupied by the pulse. Overridden to
        0.5 when ``rz=True``.
    filter_span : int, default 10
        Filter span in symbols (smoothrect only).
    rise_time : float, default 0.22
        10%-90% edge transition duration in symbol periods (smoothrect only).

    Returns
    -------
    Signal
        A `Signal` object containing the generated PAM waveform.

    Notes
    -----
    Samples are normalized to unit symbol power (Es = 1, average sample power = 1/sps).
    Call ``resolve_symbols()`` before demapping or computing metrics.
    """
    if rz:
        if sps % 2 != 0:
            raise ValueError("For correct RZ duty cycle, `sps` must be even")

        allowed_rz_pulses = ["rect", "smoothrect"]
        if pulse_shape not in allowed_rz_pulses:
            raise ValueError(
                f"Pulse shape '{pulse_shape}' is not allowed for RZ PAM. "
                f"Allowed: {allowed_rz_pulses}"
            )

    return generate(
        num_symbols=num_symbols,
        sps=sps,
        symbol_rate=symbol_rate,
        modulation="PAM",
        order=order,
        unipolar=unipolar,
        rz=rz,
        pulse_shape=pulse_shape,
        num_streams=num_streams,
        seed=seed,
        filter_span=filter_span,
        rise_time=rise_time,
        duty_cycle=duty_cycle,
    )


def generate_psk(
    num_symbols: int,
    sps: int,
    symbol_rate: float,
    order: int,
    unipolar: bool = False,
    rz: bool = False,
    pulse_shape: str = "rrc",
    num_streams: int = 1,
    seed: int | None = None,
    filter_span: int = 10,
    rrc_rolloff: float = 0.35,
    rc_rolloff: float = 0.35,
    rise_time: float = 0.0,
    duty_cycle: float = 1.0,
) -> "Signal":
    """
    Generates a Phase Shift Keying (PSK) baseband waveform.

    Parameters
    ----------
    num_symbols : int
        Total number of symbols to generate per stream.
    sps : float
        Samples per symbol.
    symbol_rate : float
        Symbol rate in symbols per second (Baud).
    order : int
        Modulation order (e.g., 2 for BPSK, 4 for QPSK, 8 for 8-PSK).
    unipolar : bool, default False
        If True, uses a unipolar constellation.
    rz : bool, default False
        If True, uses Return-to-Zero signaling.
    pulse_shape : str, default "rrc"
        Pulse shaping filter type.
    num_streams : int, default 1
        Number of independent streams (channels) to generate.
    seed : int, optional
        Random seed for bit and symbol generation.
    duty_cycle : float, default 1.0
        Fraction of the symbol period occupied by the pulse (rect/smoothrect).
        Only meaningful when ``rz=True``.
    filter_span : int, default 10
        Filter span in symbols.
    rrc_rolloff : float, default 0.35
        Roll-off factor for the RRC filter.
    rc_rolloff : float, default 0.35
        Roll-off factor for the RC filter.
    rise_time : float, default 0.22
        10%-90% edge transition duration in symbol periods (smoothrect only).
    duty_cycle : float, default 1.0
        FWHM of the Gaussian pulse in symbol periods (gaussian only).

    Returns
    -------
    Signal
        A `Signal` object containing the PSK waveform.

    Notes
    -----
    Samples are normalized to unit symbol power (Es = 1, average sample power = 1/sps).
    Call ``resolve_symbols()`` before demapping or computing metrics.
    """
    return generate(
        modulation="psk",
        order=order,
        num_symbols=num_symbols,
        sps=sps,
        symbol_rate=symbol_rate,
        pulse_shape=pulse_shape,
        num_streams=num_streams,
        seed=seed,
        unipolar=unipolar,
        rz=rz,
        filter_span=filter_span,
        rrc_rolloff=rrc_rolloff,
        rc_rolloff=rc_rolloff,
        rise_time=rise_time,
        duty_cycle=duty_cycle,
    )


def generate_qam(
    num_symbols: int,
    sps: int,
    symbol_rate: float,
    order: int,
    unipolar: bool = False,
    rz: bool = False,
    pulse_shape: str = "rrc",
    num_streams: int = 1,
    seed: int | None = None,
    filter_span: int = 10,
    rrc_rolloff: float = 0.35,
    rc_rolloff: float = 0.35,
    rise_time: float = 0.0,
    duty_cycle: float = 1.0,
) -> "Signal":
    """
    Generates a Quadrature Amplitude Modulation (QAM) baseband waveform.

    Parameters
    ----------
    num_symbols : int
        Number of symbols to generate per stream.
    sps : float
        Samples per symbol.
    symbol_rate : float
        Symbol rate in symbols per second (Baud).
    order : int
        Modulation order (e.g., 16, 64, 256).
    unipolar : bool, default False
        If True, uses a unipolar constellation.
    rz : bool, default False
        If True, uses Return-to-Zero signaling.
    pulse_shape : str, default "rrc"
        Pulse shaping filter type.
    num_streams : int, default 1
        Number of MIMO streams.
    seed : int, optional
        Seed for random generation.
    duty_cycle : float, default 1.0
        Fraction of the symbol period occupied by the pulse (rect/smoothrect).
        Only meaningful when ``rz=True``.
    filter_span : int, default 10
        Filter span in symbols.
    rrc_rolloff : float, default 0.35
        Roll-off factor for the RRC filter.
    rc_rolloff : float, default 0.35
        Roll-off factor for the RC filter.
    rise_time : float, default 0.22
        10%-90% edge transition duration in symbol periods (smoothrect only).
    duty_cycle : float, default 1.0
        FWHM of the Gaussian pulse in symbol periods (gaussian only).

    Returns
    -------
    Signal
        A `Signal` object containing the QAM waveform.

    Notes
    -----
    Samples are normalized to unit symbol power (Es = 1, average sample power = 1/sps).
    Call ``resolve_symbols()`` before demapping or computing metrics.
    """
    return generate(
        modulation="qam",
        order=order,
        num_symbols=num_symbols,
        sps=sps,
        symbol_rate=symbol_rate,
        pulse_shape=pulse_shape,
        num_streams=num_streams,
        seed=seed,
        unipolar=unipolar,
        rz=rz,
        filter_span=filter_span,
        rrc_rolloff=rrc_rolloff,
        rc_rolloff=rc_rolloff,
        rise_time=rise_time,
        duty_cycle=duty_cycle,
    )


def generate_psqam(
    num_symbols: int,
    sps: int,
    symbol_rate: float,
    order: int,
    *,
    nu: float | None = None,
    entropy: float | None = None,
    pulse_shape: str = "rrc",
    num_streams: int = 1,
    seed: int | None = None,
    filter_span: int = 10,
    rrc_rolloff: float = 0.35,
    rc_rolloff: float = 0.35,
    duty_cycle: float = 1.0,
) -> "Signal":
    """
    Generates a Probabilistically Shaped QAM (PS-QAM) baseband waveform.

    Symbols are drawn from a Maxwell-Boltzmann (MB) distribution over the
    normalized QAM constellation, giving inner (low-energy) points higher
    probability. This recovers up to 1.53 dB shaping gain over uniform QAM.

    Exactly one of ``nu`` or ``entropy`` must be specified.

    Parameters
    ----------
    num_symbols : int
        Number of symbols to generate per stream.
    sps : float
        Samples per symbol.
    symbol_rate : float
        Symbol rate in symbols per second (Baud).
    order : int
        QAM modulation order (e.g. 16, 64, 256).
    nu : float, optional
        MB shaping parameter nu >= 0. nu = 0 is uniform QAM.
        Larger values apply stronger shaping (lower entropy, lower power).
    entropy : float, optional
        Target per-symbol entropy in bits, in the range (0, log2(order)].
        optimal_nu is called to solve for the corresponding nu.
    pulse_shape : str, default "rrc"
        Pulse shaping filter type.
    num_streams : int, default 1
        Number of independent streams (MIMO).
    seed : int, optional
        Random seed for reproducible symbol generation.
    filter_span : int, default 10
        Filter span in symbols.
    rrc_rolloff : float, default 0.35
        Roll-off factor for the RRC filter.
    rc_rolloff : float, default 0.35
        Roll-off factor for the RC filter.

    Returns
    -------
    Signal
        A ``Signal`` with ``mod_scheme="PS-QAM"``, ``ps_pmf`` set to the MB
        distribution, and both ``source_symbols`` and ``source_bits`` populated.

    Notes
    -----
    ``source_bits`` carry the non-uniform MB statistics (correct for BER/GMI
    estimation, not a full coded PAS transmitter). Average symbol energy is
    below 1 for nu > 0; pass ``pmf=signal.ps_pmf`` to ``metrics.mi`` and
    ``compute_llr`` for correct soft-demapping.

    Examples
    --------
    >>> sig = generate_psqam(10000, sps=4, symbol_rate=32e9, order=64, entropy=6.0)
    >>> sig = generate_psqam(10000, sps=4, symbol_rate=32e9, order=64, nu=0.3)
    """

    if sps != int(sps) or sps < 1:
        logger.warning(
            "sps=%r is not a positive integer. Non-integer sps is valid for "
            "captured/resampled signals but not for generation: "
            "resample_poly requires an integer upsampling factor, so the "
            "sample buffer would not match the stored sampling_rate "
            "metadata. To generate at a fractional sps, generate at an "
            "integer sps then call Signal.resample(up=..., down=...).",
            sps,
        )
        raise ValueError(
            f"sps must be a positive integer for signal generation, got {sps!r}."
        )
    sps = int(sps)

    if (nu is None) == (entropy is None):
        raise ValueError("Exactly one of `nu` or `entropy` must be specified.")

    if entropy is not None:
        nu_val, _ = mapping.optimal_nu(order, entropy)
    else:
        assert nu is not None
        nu_val = float(nu)
        if nu_val < 0:
            raise ValueError("`nu` must be non-negative.")

    pmf = mapping.maxwell_boltzmann(order, nu_val)
    k = int(np.log2(order))
    total_symbols = num_symbols * num_streams

    # Sample symbols from MB distribution (NumPy, CPU)
    symbols_flat = mapping.sample_ps_symbols(total_symbols, order, pmf, seed=seed)

    # Derive source bits by demapping noiseless shaped symbols (lossless).
    # Array input -> array output (the Signal-dispatch branch is not taken).
    bits_flat = cast(ArrayType, mapping.demap_symbols_hard(symbols_flat, "qam", order))

    if num_streams > 1:
        symbols = symbols_flat.reshape(num_streams, num_symbols)
        bits = bits_flat.reshape(num_streams, num_symbols * k)
    else:
        symbols = symbols_flat
        bits = bits_flat

    if is_cupy_available():
        symbols = to_device(symbols, "gpu")
        bits = to_device(bits, "gpu")

    samples = filtering.shape_pulse(
        symbols=symbols,
        sps=sps,
        pulse_shape=pulse_shape,
        filter_span=filter_span,
        rrc_rolloff=rrc_rolloff,
        rc_rolloff=rc_rolloff,
        duty_cycle=duty_cycle,
    )

    _ps_tag = f"entropy={entropy:.3g}" if entropy is not None else f"ν={nu_val:.3g}"
    logger.info(
        "Generated PS-QAM-%s signal: %s symbols x %s stream(s), sps=%s, %s, "
        "pulse_shape=%s, %s samples/stream @ %.3g Sa/s.",
        order,
        num_symbols,
        num_streams,
        sps,
        _ps_tag,
        pulse_shape,
        samples.shape[-1],
        symbol_rate * sps,
    )

    return Signal(
        samples=samples,
        sampling_rate=symbol_rate * sps,
        symbol_rate=symbol_rate,
        mod_scheme="PS-QAM",
        mod_order=order,
        source_bits=bits,
        source_symbols=symbols,
        pulse_shape=pulse_shape,
        ps_pmf=pmf,
        ps_nu=nu_val,
        filter_span=filter_span,
        rrc_rolloff=rrc_rolloff,
    )
