"""Additive white Gaussian noise: :func:`apply_awgn`, and amplified
spontaneous emission at an OSNR: :func:`apply_ase`.

Powers are per sample and per channel, in the squared units of the samples.
The signal's symbol energy is measured as energy per symbol: the energy of
each channel over the number of symbols it carries, so silence around the
signal (zero padding, a capture before the frame arrives) lowers neither.
"""

from typing import Any

from .._random import RNG, as_generator, complex_normal, standard_normal
from ..backend import ArrayType, dispatch
from ..core._signal_adapter import S, SignalAdapter, adapt_signal
from ..logger import logger
from ..math import db_to_linear

__all__ = ["apply_ase", "apply_awgn"]


def apply_awgn(
    samples: S,
    *,
    esn0_db: float | None = None,
    noise_var: float | ArrayType | None = None,
    sps: float | None = None,
    per_channel: bool = True,
    rng: RNG = None,
) -> S:
    """Add white Gaussian noise, relative to the signal or of an absolute
    variance.

    Give exactly one of:

    - ``esn0_db``, the noise *relative to the signal*: the symbol energy
      ``Es`` is measured, and channel ``c`` gets noise of variance
      ``σ_c² = Es_c / (Es/N0)`` per sample.  Scaling the signal scales the
      noise with it; the Es/N0 stays.
    - ``noise_var``, the *absolute* noise variance ``σ² = E|n|²`` per sample,
      in the squared units of the samples; nothing is measured.  A fixed
      noise floor, such as a receiver's thermal noise: scaling the signal
      leaves the noise as it is, and the SNR changes.

    The symbol energy is energy per symbol, ``Es_c = E_c / K`` with ``E_c =
    Σ|x_c[n]|²`` the energy of channel ``c`` and ``K`` the number of symbols
    the record carries: a Signal's frame length in symbols (without a zero
    guard), else its reference's symbol count, else (and for arrays, or a
    record shorter than that) ``N / sps``.  Without silence this is
    ``sps · mean(|x_c|²)``, the power summed over the ``sps`` samples of a
    symbol; zero padding adds no energy and no symbols, so it changes
    nothing.  With ``Es`` in these units the noise density is ``N0 = σ²``,
    the variance of one sample, and ``Es/N0`` is the SNR per symbol after a
    matched filter and decimation, whatever the oversampling.

    Parameters
    ----------
    samples : array_like or Signal
        Input samples, time on the last axis, ``(N,)`` or ``(C, N)``.  A
        channel is anything with its own receiver noise: a polarization, an
        antenna, a wire pair, a lane.
    esn0_db : float, optional
        Symbol energy over noise spectral density, Es/N0, in dB.
    noise_var : float or array_like, optional
        Noise variance ``E|n|²`` per sample: a scalar for every channel, or
        ``(C,)`` per channel.  Instead of ``esn0_db``.
    sps : float, optional
        Samples per symbol, for ``esn0_db``.  Taken from the Signal; required
        for array input.  A value that disagrees with the Signal raises.
    per_channel : bool, default True
        For ``esn0_db``: ``True`` gives every channel the requested Es/N0
        from its own symbol energy; ``False`` gives every channel the same
        noise, from the mean symbol energy over channels, as identical
        receivers with one noise floor (a channel that arrives weaker, e.g.
        through polarization-dependent loss, then has a lower Es/N0).
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
        If neither or both of ``esn0_db`` and ``noise_var`` are given, or a
        per-channel ``noise_var`` does not have ``C`` entries.

    Notes
    -----
    Complex input gets circular complex noise, its variance split equally
    between I and Q; real input gets real noise of the full variance.

    Examples
    --------
    >>> sig = generate(Constellation.pam(4), 1000, symbol_rate=1e6, sps=4)
    >>> noisy = apply_awgn(sig, esn0_db=20)  # sps taken from sig
    >>> noisy = apply_awgn(sig.samples, esn0_db=20, sps=4)
    >>> floor = apply_awgn(sig, noise_var=1e-3)  # absolute, whatever the signal
    """
    signal_adapter = adapt_signal(samples, function_name="apply_awgn()")
    if (esn0_db is None) == (noise_var is None):
        raise ValueError("apply_awgn(): give one of esn0_db and noise_var.")
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

    sps = signal_adapter.resolve_fact("sps", sps)
    logger.info("Adding AWGN (Es/N0 target: %.2f dB).", esn0_db)
    es = _symbol_energy(signal_adapter, x, xp, sps)
    if not per_channel:
        es = xp.mean(es)
    esn0_linear = max(db_to_linear(esn0_db, power=True), 1e-20)
    return signal_adapter.wrap_samples(x + _noise(x, xp, es / esn0_linear, rng))


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
