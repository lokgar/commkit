"""Additive white Gaussian noise: :func:`apply_awgn`, amplified spontaneous
emission at an OSNR: :func:`apply_ase`, and the noise of coherent detection
at absolute optical powers: :func:`apply_coherent_detection`.

Powers are per sample and per channel, in the squared units of the samples.
Es/N0 is the channel's: complex samples get noise of variance ``N0`` per
sample, real samples its in-phase half, ``N0 / 2``, so error rates at an
Es/N0 do not depend on the samples' dtype.  An SNR is a power ratio in the
samples themselves.
The signal's symbol energy is measured as energy per symbol: the energy of
each channel over the number of symbols it carries, so silence around the
signal (zero padding, a capture before the frame arrives) lowers neither.
"""

import logging
from typing import Any

import numpy as np

from .._array import as_2d, restore_1d
from .._constants import ELEMENTARY_CHARGE
from .._random import RNG, as_generator, complex_normal, standard_normal
from ..backend import ArrayType, dispatch, to_device
from ..core._signal_adapter import S, SignalAdapter, adapt_signal
from ..logger import logger
from ..math import db_to_linear

__all__ = ["apply_ase", "apply_awgn", "apply_coherent_detection"]


def apply_awgn(
    samples: S,
    *,
    esn0_db: float | None = None,
    snr_db: float | None = None,
    noise_var: float | ArrayType | None = None,
    sps: float | None = None,
    per_channel: bool = True,
    rng: RNG = None,
) -> S:
    """Add white Gaussian noise: at an Es/N0, at an SNR, or of an absolute
    variance.

    Give exactly one of:

    - ``esn0_db``, the channel's *Es/N0*, the textbook quantity of digital
      communications: the symbol energy ``Es`` is measured, and channel
      ``c`` gets noise of density ``N0_c = Es_c / (Es/N0)``.  Complex
      samples get circular complex noise of variance ``N0_c`` per sample
      (``N0_c / 2`` in I and in Q); real samples get its in-phase part,
      real noise of variance ``N0_c / 2``.  So a constellation's error
      rates at an Es/N0 are the same whether its samples are real or
      complex, and the textbook formulas apply (BPSK:
      ``Q(sqrt(2 Es/N0))``).
    - ``snr_db``, the *signal-to-noise ratio in the samples*: channel ``c``
      gets noise of variance ``σ_c² = P_c / SNR`` per sample, with ``P_c =
      mean(|x_c|²)`` the channel's mean power over the record, complex for
      complex samples and real for real ones.  What a measurement quotes
      (a photocurrent's beat SNR) and what :func:`~commkit.metrics.snr`
      reads back on symbols.  For complex samples at one sample per symbol
      it equals Es/N0; for real samples ``snr_db = esn0_db + 3.01 dB``.
    - ``noise_var``, the *absolute* noise variance ``σ² = E|n|²`` per
      sample, in the squared units of the samples (complex or real noise,
      after the samples); nothing is measured.  A fixed noise floor, such
      as a receiver's thermal noise: scaling the signal leaves the noise as
      it is, and the SNR changes.

    With ``esn0_db`` or ``snr_db`` the noise scales with the signal.  The
    symbol energy is energy per symbol, ``Es_c = E_c / K`` with ``E_c =
    Σ|x_c[n]|²`` the energy of channel ``c`` and ``K`` the number of symbols
    the record carries: a Signal's frame length in symbols (without a zero
    guard), else its reference's symbol count, else (and for arrays, or a
    record shorter than that) ``N / sps``.  Without silence this is
    ``sps · mean(|x_c|²)``, the power summed over the ``sps`` samples of a
    symbol; zero padding adds no energy and no symbols, so it changes
    nothing.  ``Es/N0`` is then the SNR per symbol after a matched filter
    and decimation, whatever the oversampling.

    Parameters
    ----------
    samples : array_like or Signal
        Input samples, time on the last axis, ``(N,)`` or ``(C, N)``.  A
        channel is anything with its own receiver noise: a polarization, an
        antenna, a wire pair, a lane.
    esn0_db : float, optional
        Symbol energy over noise spectral density, Es/N0, in dB.
    snr_db : float, optional
        Mean signal power over noise variance per sample, in dB.
    noise_var : float or array_like, optional
        Noise variance ``E|n|²`` per sample: a scalar for every channel, or
        ``(C,)`` per channel.
    sps : float, optional
        Samples per symbol, for ``esn0_db``.  Taken from the Signal; required
        for array input.  A value that disagrees with the Signal raises.
    per_channel : bool, default True
        For ``esn0_db`` and ``snr_db``: ``True`` gives every channel the
        requested ratio from its own energy; ``False`` gives every channel
        the same noise, from the mean over channels, as identical receivers
        with one noise floor (a channel that arrives weaker, e.g. through
        polarization-dependent loss, then has a lower ratio).
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
        If not exactly one of ``esn0_db``, ``snr_db`` and ``noise_var`` is
        given, or a per-channel ``noise_var`` does not have ``C`` entries.

    See Also
    --------
    commkit.mapping.compute_llr : Takes the variance of the noise in the
        symbols, or the same ``esn0_db``.
    commkit.metrics.estimate_noise_variance : Measures that variance.

    Examples
    --------
    >>> sig = generate(Constellation.pam(4), num_symbols=1000, symbol_rate=1e6, sps=4)
    >>> noisy = apply_awgn(sig, esn0_db=20)  # sps taken from sig
    >>> noisy = apply_awgn(sig.samples, esn0_db=20, sps=4)
    >>> noisy = apply_awgn(sig, snr_db=25)  # power ratio in the samples
    >>> floor = apply_awgn(sig, noise_var=1e-3)  # absolute, whatever the signal
    """
    signal_adapter = adapt_signal(samples, function_name="apply_awgn()")
    given = [esn0_db is not None, snr_db is not None, noise_var is not None]
    if sum(given) != 1:
        raise ValueError(
            "apply_awgn(): give exactly one of esn0_db, snr_db and noise_var."
        )
    x, xp, _ = dispatch(signal_adapter.array)
    num_channels = 1 if x.ndim == 1 else x.shape[0]

    if noise_var is not None:
        logger.info("Adding AWGN (noise variance given).")
        var = xp.asarray(noise_var, dtype=xp.float64)
        if var.ndim != 0:
            if var.shape != (num_channels,):
                raise ValueError(
                    "apply_awgn(): noise_var must be a scalar or have one entry "
                    f"per channel ({num_channels},), got shape {var.shape}."
                )
            var = var[:, None] if x.ndim == 2 else var
        return signal_adapter.wrap_samples(x + _noise(x, xp, var, rng))

    if snr_db is not None:
        logger.info("Adding AWGN (SNR target: %.2f dB).", snr_db)
        power = xp.mean(xp.abs(x) ** 2, axis=-1, keepdims=True)
        if not per_channel:
            power = xp.mean(power)
        snr_linear = max(db_to_linear(snr_db, power=True), 1e-20)
        return signal_adapter.wrap_samples(x + _noise(x, xp, power / snr_linear, rng))

    sps = signal_adapter.resolve_fact("sps", sps)
    logger.info("Adding AWGN (Es/N0 target: %.2f dB).", esn0_db)
    es = _symbol_energy(signal_adapter, x, xp, sps)
    if not per_channel:
        es = xp.mean(es)
    esn0_linear = max(db_to_linear(esn0_db, power=True), 1e-20)
    n0 = es / esn0_linear
    # Real samples carry the in-phase half of complex noise of density N0.
    return signal_adapter.wrap_samples(
        x + _noise(x, xp, n0 if xp.iscomplexobj(x) else n0 / 2, rng)
    )


def apply_ase(
    samples: S,
    *,
    osnr_db: float,
    reference_bandwidth: float = 12.5e9,
    noise_modes: int | None = None,
    rng: RNG = None,
) -> S:
    """Add amplified spontaneous emission at an optical signal-to-noise ratio.

    OSNR is specified as optical links are: the total signal power of all
    channels over the ASE power in ``reference_bandwidth``, summed over the
    ``M = noise_modes`` modes the medium guides (polarizations included).
    ASE has the same density in every mode and is white over the simulated
    band ``f_s``, so complex noise of variance ``σ²`` per mode puts
    ``M σ² B_ref / f_s`` into ``B_ref``, and every channel gets the same

        σ² = P · f_s / (M · B_ref · OSNR),

    with ``P = Σ_c Es_c / sps`` the total signal power while the signal is
    present (the symbol energy as in :func:`apply_awgn`, so silence around
    the signal does not lower it).  For equal channel powers this gives
    ``Es/N0 = M B_ref / (C R_s) · OSNR`` per channel, whatever the
    oversampling: the textbook ``2 B_ref / (p R_s) · OSNR`` for ``p``
    polarizations of a single-mode fiber.

    Parameters
    ----------
    samples : Signal
        A complex optical Signal (``domain="OPT"``), ``(N,)`` or ``(C, N)``
        with one channel per guided mode it occupies (polarization, spatial
        mode or core); its sampling rate sets the noise bandwidth.
    osnr_db : float
        Optical signal-to-noise ratio in dB in ``reference_bandwidth``.
    reference_bandwidth : float, default 12.5e9
        Noise reference bandwidth ``B_ref`` in Hz.  The standard reference
        is 0.1 nm of optical bandwidth, ``c · 0.1 nm / λ²``: 12.5 GHz in the
        C band (12.48 GHz at 1550 nm), 17.5 GHz at 1310 nm.  Outside the C
        band pass the value for the carrier, e.g. ``17.5e9`` for an O-band
        OSNR in 0.1 nm.
    noise_modes : int, optional
        The number ``M`` of guided modes, polarizations included, whose ASE
        the OSNR counts.  Defaults to ``max(C, 2)``: both polarizations of a
        single-mode fiber for one or two channels (an OSNR meter's
        definition; a single-polarization signal receives the half of the
        ASE in its own polarization), and one mode per channel beyond that
        (few-mode or multicore fiber with every mode carrying signal).  Pass
        it when the medium guides modes that carry no signal, e.g.
        ``noise_modes=6`` for 4 channels in a 3-mode fiber.
    rng : int, numpy.random.Generator or None
        Random source, as in :func:`apply_awgn`.

    Returns
    -------
    Signal
        The noisy Signal, same shape, dtype and device.

    Raises
    ------
    ValueError
        If ``samples`` is not a complex optical Signal,
        ``reference_bandwidth`` is not positive, or ``noise_modes`` is not
        an integer of at least ``C``.

    Notes
    -----
    OSNR is a property of the whole optical field, so pass every channel in
    one call: calling once per polarization of a dual-polarization signal
    counts only that polarization's power and adds half the intended
    noise.  Modes left out of a simulation carry no noise, which matters
    for a receiver without a polarizer (direct detection, where the
    orthogonal polarization adds ASE-ASE beat noise); model such a link
    with both polarizations.

    Examples
    --------
    >>> field = sig.replace(domain="OPT", center_frequency=193.4e12)
    >>> noisy = apply_ase(field, osnr_db=18)  # in 12.5 GHz, both polarizations
    """
    signal_adapter = adapt_signal(samples, function_name="apply_ase()")
    x, xp, _ = dispatch(signal_adapter.array)
    sig = signal_adapter.signal
    if sig is None or sig.domain != "OPT" or not xp.iscomplexobj(x):
        raise ValueError(
            "apply_ase(): needs a complex optical Signal (domain='OPT'), whose "
            "sampling rate sets the noise bandwidth."
        )
    if not reference_bandwidth > 0:
        raise ValueError(
            "apply_ase(): reference_bandwidth must be positive (Hz), got "
            f"{reference_bandwidth!r}."
        )
    num_channels = 1 if x.ndim == 1 else x.shape[0]
    if noise_modes is None:
        noise_modes = max(num_channels, 2)
    if (
        isinstance(noise_modes, bool)
        or not isinstance(noise_modes, int)
        or noise_modes < num_channels
    ):
        raise ValueError(
            f"apply_ase(): noise_modes must be an integer of at least the "
            f"{num_channels} channels, got {noise_modes!r}."
        )
    logger.info(
        "Adding ASE noise (OSNR target: %.2f dB, %d modes).", osnr_db, noise_modes
    )
    sps = float(sig.sps)
    # The ASE density is the same in every mode: one variance for all
    # channels, from their total power while the signal is present.
    power = xp.sum(_symbol_energy(signal_adapter, x, xp, sps)) / sps
    noise_power = (
        power
        * sig.sampling_rate
        / (noise_modes * reference_bandwidth * db_to_linear(osnr_db, power=True))
    )
    return signal_adapter.wrap_samples(x + _noise(x, xp, noise_power, rng))


def apply_coherent_detection(
    samples: S,
    *,
    received_power_dbm: float,
    lo_power_dbm: float,
    responsivity: float = 1.0,
    thermal_noise_density: float = 0.0,
    sampling_rate: float | None = None,
    rng: RNG = None,
) -> S:
    """Add the noise of a phase-diversity coherent receiver at absolute powers.

    Each channel (polarization) mixes with a local oscillator in a 90°
    hybrid whose four outputs feed two balanced photodiode pairs, one per
    quadrature.  The beat term ``R·sqrt(P_c·P_LO)`` carries the signal; the
    shot noise of the light on the diodes, ``R (P_c + P_LO)`` in total, and
    the receiver's thermal noise add to it.  Per complex sample, the noise
    variance relative to the signal power is

        1 / SNR = f_s · (q R (P_c + P_LO) + i_n²) / (R² P_c P_LO),

    with ``P_c`` the channel's share of ``received_power_dbm`` (in proportion
    to its mean sample power), ``q`` the elementary charge and ``i_n`` the
    thermal noise density.  The output keeps the input's units: the noise is
    added to the samples at that ratio, with the signal component
    unchanged.  With a strong LO and no thermal noise, ``Es/N0 = R P_c T / q``,
    ``η`` photons per symbol for a responsivity ``R = η q / (hν)``.

    Parameters
    ----------
    samples : array_like or Signal
        Received optical field, ``(N,)`` or ``(C, N)`` with one channel per
        polarization, real or complex.
    received_power_dbm : float
        Mean optical power the input stands for, in dBm, summed over the
        channels.
    lo_power_dbm : float
        Local-oscillator power at each channel's hybrid, in dBm.
    responsivity : float, default 1.0
        Photodiode responsivity ``R`` in A/W, positive (``η q / (hν)``:
        1.25 A/W for a quantum efficiency of 1 at 1550 nm).
    thermal_noise_density : float, default 0.0
        Input-referred thermal noise current density ``i_n`` of each
        balanced pair in A/sqrt(Hz) (one-sided), non-negative; typical
        receivers are near 10-20 pA/sqrt(Hz).
    sampling_rate : float, optional
        Sampling rate in Hz, the noise bandwidth.  Taken from the Signal;
        required for array input.  A value that disagrees with the Signal
        raises.
    rng : int, numpy.random.Generator or None
        Random source, as in :func:`apply_awgn`.

    Returns
    -------
    ndarray or Signal
        The detected field, complex in the input's precision, same shape and
        device; a Signal keeps its domain and carrier.

    Raises
    ------
    ValueError
        If ``responsivity`` is not positive, ``thermal_noise_density`` is
        negative, a channel is all zeros, or the sampling rate is missing
        or conflicts with the Signal's.

    Notes
    -----
    A balanced pair sees a quarter of the signal and LO light on each
    diode, whose shot noises (two-sided density ``q I``) add: per quadrature
    ``q R (P_c + P_LO) f_s / 2`` per sample.  Thermal noise of one-sided
    density ``i_n²`` adds ``i_n² f_s / 2``.  The beat term of one quadrature
    is ``R sqrt(P_c P_LO)`` times the field's component of unit mean power,
    so the complex signal power is ``R² P_c P_LO``.  The orthogonal
    quadratures make the shot-noise limit ``Es/N0 = η·n_p``, half that of a
    single-quadrature homodyne receiver.

    Examples
    --------
    >>> rx = apply_coherent_detection(
    ...     field, received_power_dbm=-30.0, lo_power_dbm=10.0,
    ...     responsivity=1.0, thermal_noise_density=15e-12,
    ... )
    """
    if not responsivity > 0:
        raise ValueError(
            f"apply_coherent_detection(): responsivity must be positive, got {responsivity!r}."
        )
    if not thermal_noise_density >= 0:
        raise ValueError(
            "apply_coherent_detection(): thermal_noise_density must be non-negative, "
            f"got {thermal_noise_density!r}."
        )
    function_name = "apply_coherent_detection()"
    signal_adapter = adapt_signal(samples, function_name=function_name)
    fs = signal_adapter.resolve_fact("sampling_rate", sampling_rate)
    x, xp, _ = dispatch(signal_adapter.array)
    if not xp.iscomplexobj(x):
        x = x.astype(xp.complex64 if x.dtype == xp.float32 else xp.complex128)
    x, was_1d = as_2d(x, name="samples")
    channel_power = xp.mean(xp.abs(x) ** 2, axis=-1, dtype=xp.float64)  # (C,)
    host_power = np.asarray(to_device(channel_power, "cpu"), dtype=np.float64)
    if np.any(host_power == 0):
        raise ValueError(f"{function_name}: a channel is all zeros.")
    p_total = 1e-3 * 10 ** (received_power_dbm / 10)
    p_lo = 1e-3 * 10 ** (lo_power_dbm / 10)
    p_c = p_total * host_power / host_power.sum()
    r = responsivity
    inv_snr = (
        fs
        * (ELEMENTARY_CHARGE * r * (p_c + p_lo) + thermal_noise_density**2)
        / (r * r * p_c * p_lo)
    )
    if logger.isEnabledFor(logging.INFO):
        sig = signal_adapter.signal
        snr_db = -10 * np.log10(inv_snr)
        if sig is not None and sig.symbol_rate:
            esn0_db = snr_db + 10 * np.log10(fs / sig.symbol_rate)
            logger.info(
                "Coherent detection: Es/N0 %s dB per channel (%.2f dBm received, "
                "%.2f dBm LO).",
                np.array2string(esn0_db, precision=2),
                received_power_dbm,
                lo_power_dbm,
            )
        else:
            logger.info(
                "Coherent detection: SNR %s dB per sample.",
                np.array2string(snr_db, precision=2),
            )
    noise_var = xp.asarray(host_power * inv_snr)[:, None]
    y = x + _noise(x, xp, noise_var, rng)
    return signal_adapter.wrap_samples(restore_1d(was_1d, y))


def _symbol_energy(
    signal_adapter: SignalAdapter[Any], x: ArrayType, xp: Any, sps: float
) -> ArrayType:
    """Energy per symbol of each channel in sample units, ``(C, 1)`` (``(1,)``
    for 1-D input): the channel's energy over the symbols the record
    carries (see :func:`apply_awgn`)."""
    num_periods = x.shape[-1] / sps
    num_symbols = num_periods
    sig = signal_adapter.signal
    if sig is not None:
        if sig.frame is not None:
            frame = sig.frame
            zero_guard = frame.guard_num_symbols if frame.guard_type == "zero" else 0
            num_symbols = frame.num_symbols - zero_guard
        elif sig.reference is not None and sig.reference.symbols is not None:
            num_symbols = sig.reference.symbols.shape[-1]
    # A record shorter than its frame or reference holds only part of the
    # signal: its own length counts.
    num_symbols = min(num_symbols, num_periods)
    mean_power = xp.mean(xp.abs(x) ** 2, axis=-1, keepdims=True)
    return mean_power * (x.shape[-1] / num_symbols)


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
