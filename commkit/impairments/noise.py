"""Additive white Gaussian noise: :func:`apply_awgn`."""

from typing import Any

from .._random import RNG, as_generator, complex_normal, standard_normal
from ..backend import ArrayType, dispatch
from ..core._signal_adapter import S, adapt_signal
from ..logger import logger
from ..math import db_to_linear

__all__ = ["apply_awgn"]


def apply_awgn(
    samples: S,
    *,
    esn0_db: float | None = None,
    osnr_db: float | None = None,
    sps: float | None = None,
    signal_power: float | ArrayType | None = None,
    reference_bandwidth: float = 12.5e9,
    noise_modes: int | None = None,
    rng: RNG = None,
) -> S:
    """Add white Gaussian noise at an Es/N0, or at an OSNR for an optical field.

    Powers are per sample and per channel, in the squared units of the
    samples: ``P_c = mean(|x_c[n]|²)`` over the samples of channel ``c``.
    A symbol lasts ``T = sps / f_s``, so its energy is
    ``Es_c = P_c · T = sps · P_c / f_s``; white noise of variance ``σ²``
    per sample fills the band ``f_s`` with the density ``N0 = σ² / f_s``.
    Hence ``Es/N0 = sps · P_c / σ²``: in sample units, ``Es`` is the power
    summed over the ``sps`` samples of a symbol and ``N0`` the noise
    variance of one sample.

    With ``esn0_db`` every channel gets the requested Es/N0 on its own:
    channel ``c`` gets noise of variance ``σ_c² = sps · P_c / (Es/N0)`` per
    sample.  That is the Es/N0 per symbol after a matched filter and
    decimation, whatever the oversampling.  A channel can be anything with
    its own receiver noise: a polarization, an antenna, a wire pair, a lane.

    With ``osnr_db`` (an optical Signal) the noise is amplified spontaneous
    emission specified as optical links are: the total signal power of all
    channels over the ASE power in ``reference_bandwidth``, summed over the
    ``M = noise_modes`` modes the medium guides (polarizations included).
    ASE has the same density in every mode and is white over the simulated
    band ``f_s``, so complex noise of variance ``σ²`` per mode puts
    ``M σ² B_ref / f_s`` into ``B_ref``; every channel therefore gets the
    same

        σ² = (Σ_c P_c) · f_s / (M · B_ref · OSNR),

    which for equal powers gives ``Es/N0 = M B_ref / (C R_s) · OSNR`` per
    channel, whatever the oversampling: the textbook
    ``2 B_ref / (p R_s) · OSNR`` for ``p`` polarizations of a single-mode
    fiber.

    Parameters
    ----------
    samples : array_like or Signal
        Input samples, time on the last axis, ``(N,)`` or ``(C, N)``; for
        ``osnr_db`` a complex optical Signal (``domain="OPT"``) with one
        channel per guided mode it occupies (polarization, spatial mode or
        core).
    esn0_db : float, optional
        Symbol energy over noise spectral density, Es/N0, in dB, for each
        channel.
    osnr_db : float, optional
        Optical signal-to-noise ratio in dB in ``reference_bandwidth``,
        instead of ``esn0_db``.
    sps : float, optional
        Samples per symbol, for ``esn0_db``.  Taken from the Signal; required
        for array input.  A value that disagrees with the Signal raises.
    signal_power : float or array_like, optional
        The power ``P_c`` per sample of each channel, ``mean(|x_c|²)`` in
        the squared units of the samples, that the Es/N0 or OSNR refers to:
        ``Es_c = sps · P_c``.  A scalar is the power of *each* channel (not
        their sum or a total), so every channel gets the same absolute
        noise level, as identical receivers with one noise floor; ``(C,)``
        gives one power per channel.  Defaults to each channel's measured
        mean power over the whole record, which counts any silent stretch.
        Pass the nominal power instead when that is the reference: e.g.
        ``signal_power=1.0`` for unit-power symbols at one sample per
        symbol sent in bursts separated by zero guard intervals, whose
        measured mean falls below the power of the symbols.  For a
        pulse-shaped signal the per-sample power of unit-energy symbols is
        ``1 / sps`` (``signal_power=1 / sig.sps``).  With ``osnr_db`` the
        signal power is the sum over channels, ``C · signal_power`` for a
        scalar.
    reference_bandwidth : float, default 12.5e9
        Noise reference bandwidth ``B_ref`` of ``osnr_db`` in Hz.  The
        standard reference is 0.1 nm of optical bandwidth, ``c · 0.1 nm /
        λ²``: 12.5 GHz in the C band (12.48 GHz at 1550 nm), 17.5 GHz at
        1310 nm.  Outside the C band pass the value for the carrier, e.g.
        ``reference_bandwidth=17.5e9`` for an O-band OSNR in 0.1 nm.
    noise_modes : int, optional
        For ``osnr_db``: the number ``M`` of guided modes, polarizations
        included, whose ASE the OSNR counts.  Defaults to ``max(C, 2)``: both
        polarizations of a single-mode fiber for one or two channels (an
        OSNR meter's definition; a single-polarization signal receives the
        half of the ASE in its own polarization), and one mode per channel
        beyond that (few-mode or multicore fiber with every mode carrying
        signal).  Pass it when the medium guides modes that carry no signal,
        e.g. ``noise_modes=6`` for 4 channels in a 3-mode fiber.
    rng : int, numpy.random.Generator or None
        Random source.  The noise is drawn on the data's device: from this
        Generator on the CPU, and on the GPU from a CuPy Generator seeded from
        it, so a seed is reproducible per device.

    Returns
    -------
    ndarray or Signal
        The noisy samples, same shape, dtype and device as the input.

    Raises
    ------
    ValueError
        If neither or both of ``esn0_db`` and ``osnr_db`` are given,
        ``osnr_db`` is given for anything but a complex optical Signal,
        ``reference_bandwidth`` is not positive, ``noise_modes`` is not an
        integer of at least ``C``, or a per-channel ``signal_power`` does not
        have ``C`` entries.

    Notes
    -----
    Complex input gets circular complex noise, its variance split equally
    between I and Q; real input gets real noise of the full variance.  At
    ``sps = 1`` Es/N0 equals the per-sample SNR.

    OSNR is a property of the whole optical field, so pass every channel in
    one call: calling once per polarization of a dual-polarization signal
    counts only that polarization's power and adds half the intended
    noise.  Modes left out of a simulation carry no noise, which matters
    for a receiver without a polarizer (direct detection, where the
    orthogonal polarization adds ASE-ASE beat noise); model such a link
    with both polarizations.

    Examples
    --------
    >>> sig = generate(Constellation.pam(4), 1000, symbol_rate=1e6, sps=4)
    >>> noisy = apply_awgn(sig.samples, esn0_db=20, sps=4)
    >>> noisy = apply_awgn(sig, esn0_db=20)  # Signal input: sps taken from sig
    >>> field = sig.replace(domain="OPT", center_frequency=193.4e12)
    >>> noisy = apply_awgn(field, osnr_db=18)  # in 12.5 GHz, both polarizations
    """
    signal_adapter = adapt_signal(samples, function_name="apply_awgn()")
    if (esn0_db is None) == (osnr_db is None):
        raise ValueError("apply_awgn(): give one of esn0_db and osnr_db.")
    x, xp, _ = dispatch(signal_adapter.array)
    num_channels = 1 if x.ndim == 1 else x.shape[0]
    power = _channel_power(x, xp, signal_power, num_channels)

    if osnr_db is not None:
        sig = signal_adapter.signal
        if sig is None or sig.domain != "OPT" or not xp.iscomplexobj(x):
            raise ValueError(
                "apply_awgn(): osnr_db needs a complex optical Signal "
                "(domain='OPT'), whose sampling rate sets the noise bandwidth."
            )
        if not reference_bandwidth > 0:
            raise ValueError(
                "apply_awgn(): reference_bandwidth must be positive (Hz), got "
                f"{reference_bandwidth!r}."
            )
        if noise_modes is None:
            noise_modes = max(num_channels, 2)
        if (
            isinstance(noise_modes, bool)
            or not isinstance(noise_modes, int)
            or noise_modes < num_channels
        ):
            raise ValueError(
                f"apply_awgn(): noise_modes must be an integer of at least the "
                f"{num_channels} channels, got {noise_modes!r}."
            )
        logger.info(
            "Adding ASE noise (OSNR target: %.2f dB, %d modes).", osnr_db, noise_modes
        )
        # The ASE density is the same in every mode: one variance for all
        # channels, from their total power.
        noise_power = (
            xp.sum(xp.broadcast_to(power, (num_channels, 1)))
            * sig.sampling_rate
            / (noise_modes * reference_bandwidth * db_to_linear(osnr_db, power=True))
        )
        return signal_adapter.wrap_samples(x + _noise(x, xp, noise_power, rng))

    sps = signal_adapter.resolve_fact("sps", sps)
    logger.info("Adding AWGN (Es/N0 target: %.2f dB).", esn0_db)

    # Es/N0 to the noise variance per sample: Es = sps · P_c is the energy of
    # a symbol over its sps samples; white noise of variance σ² over the band
    # f_s = sps · R_s has N0 = σ² / f_s, so Es/N0 = P_c · sps / σ² per
    # symbol, i.e. a per-sample SNR of Es/N0 / sps.
    esn0_linear = max(db_to_linear(esn0_db, power=True), 1e-20)
    noise_power = power * (sps / esn0_linear)
    return signal_adapter.wrap_samples(x + _noise(x, xp, noise_power, rng))


def _channel_power(
    x: ArrayType, xp: Any, signal_power: Any, num_channels: int
) -> ArrayType:
    """Signal power per channel, broadcastable against ``x``: measured,
    ``(C, 1)`` (``(1,)`` for 1-D input), or the given scalar or ``(C,)``."""
    if signal_power is None:
        return xp.mean(xp.abs(x) ** 2, axis=-1, keepdims=True)
    power = xp.asarray(signal_power, dtype=xp.float64)
    if power.ndim == 0:
        return power
    if power.shape != (num_channels,):
        raise ValueError(
            f"apply_awgn(): signal_power must be a scalar or have one entry per "
            f"channel ({num_channels},), got shape {power.shape}."
        )
    return power[:, None] if x.ndim == 2 else power


def _noise(x: Any, xp: Any, noise_power: Any, rng: RNG) -> Any:
    """White Gaussian noise of ``noise_power`` per sample, shaped like ``x``.

    Complex noise splits its power equally between I and Q.
    """
    gen = as_generator(rng)
    real_dtype = x.real.dtype if x.real.dtype.kind == "f" else xp.dtype(xp.float64)
    if xp.iscomplexobj(x):
        std = xp.sqrt(noise_power / 2).astype(real_dtype)
        return complex_normal(gen, x.shape, std=std, dtype=real_dtype, xp=xp)
    std = xp.sqrt(noise_power).astype(real_dtype)
    return std * standard_normal(gen, x.shape, dtype=real_dtype, xp=xp)
