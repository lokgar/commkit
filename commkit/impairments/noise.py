"""Additive white Gaussian noise: :func:`apply_awgn`, amplified spontaneous
emission at an OSNR: :func:`apply_ase`, and the noise of coherent and
square-law detection at absolute optical powers:
:func:`apply_coherent_detection`, :func:`apply_photodetection`.

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
import math
from typing import Any

import numpy as np

from .._array import as_2d, restore_1d
from .._constants import ELEMENTARY_CHARGE
from .._random import RNG, as_generator, complex_normal, standard_normal
from ..backend import ArrayType, dispatch, to_device
from ..core._signal_adapter import S, SignalAdapter, adapt_signal
from ..logger import logger
from ..math import db_to_linear

__all__ = [
    "apply_ase",
    "apply_awgn",
    "apply_coherent_detection",
    "apply_photodetection",
]


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
    cmrr_db: float | None = None,
    lo_rin_db_hz: float | None = None,
    balanced: bool = True,
    sampling_rate: float | None = None,
    rng: RNG = None,
) -> S:
    """Add the noise of a phase-diversity coherent receiver at absolute powers.

    Each channel (polarization) mixes with a local oscillator in a 90°
    hybrid whose four outputs feed two balanced photodiode pairs, one per
    quadrature.  The beat term ``R·sqrt(P_c·P_LO)`` carries the signal; the
    shot noise of the light on the diodes, ``R (P_c + P_LO)`` in total, and
    the receiver's thermal noise add to it.  Per complex sample, the noise
    variance relative to the signal power of an ideal balanced receiver is

        1 / SNR = f_s · (q R (P_c + P_LO) + i_n²) / (R² P_c P_LO),

    with ``P_c`` the channel's share of ``received_power_dbm`` (in proportion
    to its mean sample power), ``q`` the elementary charge and ``i_n`` the
    thermal noise density.  The output keeps the input's units: the noise is
    added to the samples at that ratio, with the signal component
    unchanged.  With a strong LO and no thermal noise, ``Es/N0 = R P_c T / q``,
    ``η`` photons per symbol for a responsivity ``R = η q / (hν)``.

    A balanced pair rejects what both diodes see: the signal's own power
    ``|s|²`` (signal-signal beat) and the LO's intensity noise.  With a
    finite ``cmrr_db`` a fraction of them leaks through, and a single-ended
    receiver (``balanced=False``) passes them in full while losing half the
    signal light.

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
    cmrr_db : float, optional
        Common-mode rejection ratio of the balanced pairs in dB,
        ``20 log10((R1 + R2) / |R1 - R2|)`` for the diodes' responsivities.
        ``None`` (default) is perfect balance.  Ignored with
        ``balanced=False``.
    lo_rin_db_hz : float, optional
        Relative intensity noise of the LO in dB/Hz (one-sided), white over
        the simulated band.  ``None`` (default) is a noiseless LO.  It
        reaches the output only through imperfect balance or a single-ended
        receiver.
    balanced : bool, default True
        Balanced photodiode pairs; ``False`` detects one output of each pair
        with a single diode.
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

    With diode responsivities ``R(1 ± δ)``, ``δ = 10^(-CMRR/20)``, the
    common-mode light ``(|s(t)|² + P_LO(1 + r(t)))/4`` on each diode leaks
    as ``δ R (|s(t)|² + P_LO r(t)) / 2`` into each quadrature; a single
    diode passes ``R (|s(t)|² + P_LO r(t)) / 4`` and carries the beat at half
    the amplitude, ``R sqrt(P_c P_LO) / 2``, with the shot noise of
    ``R (P_c + P_LO) / 4``.  In units of the beat, the leak is
    ``κ (sqrt(P_c/P_LO) (|x̃|² - 1) + sqrt(P_LO/P_c) r(t))`` on both
    quadratures, ``κ = δ/2`` (balanced) or ``1/2`` (single-ended), for the
    field ``x̃`` of unit mean power: the receiver is AC coupled, so the mean
    of ``|s|²`` and the LO's DC are removed.  ``r(t)`` is white with
    variance ``RIN · f_s / 2``; the RIN's effect on the beat itself,
    second order, is left out.

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
    if cmrr_db is not None and not cmrr_db >= 0:
        raise ValueError(
            f"apply_coherent_detection(): cmrr_db must be non-negative, got {cmrr_db!r}."
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
    # Beat amplitude per quadrature, the shot-noise current of the diodes
    # of one quadrature, and the common-mode leak (module Notes).
    gain = 1.0 if balanced else 0.5
    shot_current = r * (p_c + p_lo) * (0.5 if balanced else 0.25)
    if balanced:
        leak = 0.0 if cmrr_db is None else 0.5 * 10 ** (-cmrr_db / 20)
    else:
        leak = 0.5
    inv_snr = (
        2.0
        * (ELEMENTARY_CHARGE * shot_current * fs + thermal_noise_density**2 * fs / 2)
        / (gain * gain * r * r * p_c * p_lo)
    )
    rin_var = 0.0
    if leak and lo_rin_db_hz is not None:
        rin_var = 10 ** (lo_rin_db_hz / 10) * fs / 2
        inv_snr = inv_snr + 2.0 * leak * leak * (p_lo / p_c) * rin_var
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
    gen = as_generator(rng)
    shot_thermal = inv_snr - (2.0 * leak * leak * (p_lo / p_c) * rin_var)
    y = x + _noise(x, xp, xp.asarray(host_power * shot_thermal)[:, None], gen)
    if leak:
        # Common-mode leak on both quadratures, in the units of x.
        scale = xp.asarray(np.sqrt(host_power))[:, None]
        real = x.real.dtype
        term = xp.zeros(x.shape, dtype=real)
        direct = leak * np.sqrt(p_c / p_lo)
        intensity = xp.abs(x) ** 2 / xp.asarray(host_power, dtype=real)[:, None]
        term = term + xp.asarray(direct, dtype=real)[:, None] * (intensity - 1.0)
        if rin_var:
            lo_noise = math.sqrt(rin_var) * standard_normal(
                gen, (x.shape[-1],), dtype=real, xp=xp
            )
            term = (
                term
                + xp.asarray(leak * np.sqrt(p_lo / p_c), dtype=real)[:, None] * lo_noise
            )
        y = y + (scale.astype(real) * term) * (1 + 1j)
    return signal_adapter.wrap_samples(restore_1d(was_1d, y))


def apply_photodetection(
    samples: S,
    *,
    received_power_dbm: float,
    responsivity: float = 1.0,
    thermal_noise_density: float = 0.0,
    dark_current: float = 0.0,
    saturation_current: float | None = None,
    shot_noise: bool = True,
    sum_channels: bool = False,
    sampling_rate: float | None = None,
    rng: RNG = None,
) -> S:
    r"""Detect an optical field with a photodiode: the square law and its noise.

    The output is ``|x|²``, in the input's power units, plus the noise of
    the photocurrent.  With ``P(t)`` the optical power the samples stand for
    (``received_power_dbm`` at their mean), the diode delivers
    ``I(t) = R P(t) + I_d``, limited at ``saturation_current``, with shot
    noise of variance ``q I(t) f_s`` per sample (two-sided density ``q I``)
    and thermal noise of ``i_n² f_s / 2``.  The current is mapped back to
    the input's units by the same scale, so the output's mean is the input's
    mean power plus the dark current's equivalent.

    Parameters
    ----------
    samples : array_like or Signal
        Optical field, ``(N,)`` or ``(C, N)`` (one channel per polarization).
    received_power_dbm : float
        Mean optical power the input stands for, in dBm, summed over the
        channels.
    responsivity : float, default 1.0
        Photodiode responsivity ``R`` in A/W, positive.
    thermal_noise_density : float, default 0.0
        Input-referred thermal noise current density ``i_n`` in
        A/sqrt(Hz) (one-sided), non-negative.
    dark_current : float, default 0.0
        Dark current ``I_d`` in A, non-negative.
    saturation_current : float, optional
        Photocurrent at which the diode or its amplifier saturates, in A;
        ``None`` for none.  A hard limit, before the noise.
    shot_noise : bool, default True
        Add shot noise.
    sum_channels : bool, default False
        One polarization-insensitive diode for all channels: the output is
        ``Σ_c |x_c|²``, ``(N,)``.  ``False`` detects each channel on its own
        diode (a polarization-diverse receiver), ``(C, N)``.
    sampling_rate : float, optional
        Sampling rate in Hz, the noise bandwidth.  Taken from the Signal;
        required for array input.  A value that disagrees with the Signal
        raises.
    rng : int, numpy.random.Generator or None
        Random source, as in :func:`apply_awgn`.

    Returns
    -------
    ndarray or Signal
        Real samples in the input's precision.  A Signal becomes a digital
        baseband one (``domain="DIG"``, ``center_frequency=0``) and keeps its
        reference; with ``sum_channels`` a reference of several streams is
        dropped.

    Raises
    ------
    ValueError
        If ``responsivity`` is not positive, a current or noise density is
        negative, the input is all zeros, or the sampling rate is missing
        or conflicts with the Signal's.

    Notes
    -----
    The square law doubles the field's bandwidth: a field of band
    ``±(1 + ρ) R_s / 2`` gives an intensity of band ``±(1 + ρ) R_s``, so the
    field needs at least ``2 (1 + ρ)`` samples per symbol (2.2 for
    ``ρ = 0.1``) for ``|x|²`` not to alias.  The Gaussian shot-noise model
    holds while a sample carries many photoelectrons, ``I / (q f_s) ≫ 1``.
    Per channel, ``P_c(t) = P_rx · |x_c(t)|² / Σ_c E|x_c|²``, and

    .. math::

        y = \frac{\min(R P(t) + I_d, I_{sat}) + n_{shot} + n_{th}}{R k},
        \quad k = P_{rx} / \textstyle\sum_c E|x_c|^2.

    Examples
    --------
    >>> y = apply_photodetection(field, received_power_dbm=-10.0,
    ...                          thermal_noise_density=15e-12, rng=1)
    """
    function_name = "apply_photodetection()"
    if not responsivity > 0:
        raise ValueError(
            f"{function_name}: responsivity must be positive, got {responsivity!r}."
        )
    for name, value in (
        ("thermal_noise_density", thermal_noise_density),
        ("dark_current", dark_current),
    ):
        if not value >= 0:
            raise ValueError(
                f"{function_name}: {name} must be non-negative, got {value!r}."
            )
    if saturation_current is not None and not saturation_current > 0:
        raise ValueError(
            f"{function_name}: saturation_current must be positive, got {saturation_current!r}."
        )
    signal_adapter = adapt_signal(samples, function_name=function_name)
    fs = signal_adapter.resolve_fact("sampling_rate", sampling_rate)
    x, xp, _ = dispatch(signal_adapter.array)
    x, was_1d = as_2d(x, name="samples")
    real = x.real.dtype if xp.iscomplexobj(x) else x.dtype
    intensity = (xp.abs(x) ** 2).astype(real, copy=False)
    if sum_channels:
        intensity = xp.sum(intensity, axis=0, keepdims=True)
    total = float(xp.sum(xp.mean(intensity, axis=-1, dtype=xp.float64)))
    if total == 0:
        raise ValueError(f"{function_name}: the input is all zeros.")
    p_rx = 1e-3 * 10 ** (received_power_dbm / 10)
    to_amps = responsivity * p_rx / total  # current per unit of |x|²
    current = intensity * to_amps + dark_current
    if saturation_current is not None:
        current = xp.minimum(current, saturation_current)
    gen = as_generator(rng)
    var = xp.zeros_like(current) + thermal_noise_density**2 * fs / 2
    if shot_noise:
        var = var + ELEMENTARY_CHARGE * fs * xp.maximum(current, 0.0)
    if shot_noise or thermal_noise_density:
        current = current + xp.sqrt(var) * standard_normal(
            gen, current.shape, dtype=real, xp=xp
        )
    y = (current / to_amps).astype(real, copy=False)
    logger.info(
        "Photodetection at %.2f dBm: mean photocurrent %.3g A%s.",
        received_power_dbm,
        responsivity * p_rx
        + dark_current * (1 if sum_channels else intensity.shape[0]),
        "" if not sum_channels else " (one diode)",
    )
    out = y[0] if (was_1d or sum_channels) else y
    sig = signal_adapter.signal
    if sig is None:
        return signal_adapter.wrap_samples(out)
    reference = sig.reference
    if sum_channels and reference is not None and reference.symbols.ndim > 1:
        reference = None
    return signal_adapter.wrap_samples(
        out, domain="DIG", center_frequency=0.0, reference=reference
    )


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
