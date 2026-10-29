"""Additive white Gaussian noise: :func:`apply_awgn`."""

from typing import Any

from .._random import RNG, as_generator, complex_normal, standard_normal
from ..backend import dispatch
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
    signal_power: float | None = None,
    reference_bandwidth: float = 12.5e9,
    rng: RNG = None,
) -> S:
    """Add white Gaussian noise at an Es/N0, or at an OSNR for an optical field.

    With ``esn0_db`` the noise variance per sample is
    ``signal_power · sps / (Es/N0)``: with ``Es = sps · signal_power``, that
    is ``Es/N0`` per symbol after a matched filter and decimation, whatever
    the oversampling.

    With ``osnr_db`` (an optical Signal) the noise is amplified spontaneous
    emission specified as optical links are: the total signal power over the
    ASE power in ``reference_bandwidth``, counted in both polarizations.  ASE
    is white over the simulated band ``f_s``, so complex noise of variance
    ``σ²`` per polarization has the density ``σ²/f_s`` and puts
    ``2 σ² B_ref / f_s`` into ``B_ref`` over both polarizations; each
    channel therefore gets

        σ² = C · signal_power · f_s / (2 · B_ref · OSNR),

    which gives the textbook ``Es/N0 = 2 B_ref / (p R_s) · OSNR`` for ``p``
    polarizations, whatever the oversampling.  A single-polarization Signal
    (``C = 1``) still counts the ASE of both polarizations, as an OSNR meter
    does, and receives the half that falls into its own: the noise a
    single-polarization coherent receiver sees, 3 dB less than a
    dual-polarization signal of the same total power at the same OSNR.

    OSNR is a property of the whole optical field, so pass every
    polarization in one call.  Calling once per polarization of a
    dual-polarization signal counts only that polarization's power and
    adds half the intended noise.  The orthogonal polarization's ASE is not
    simulated for ``C = 1``, which matters for a receiver without a
    polarizer (direct detection, where it adds ASE-ASE beat noise); model
    such a link with both polarizations.

    Parameters
    ----------
    samples : array_like or Signal
        Input samples, time on the last axis; for ``osnr_db`` a complex
        optical Signal (``domain="OPT"``), ``(N,)`` or ``(C, N)`` with one
        channel per polarization.
    esn0_db : float, optional
        Symbol energy over noise spectral density, Es/N0, in dB.
    osnr_db : float, optional
        Optical signal-to-noise ratio in dB in ``reference_bandwidth``,
        instead of ``esn0_db``.
    sps : float, optional
        Samples per symbol, for ``esn0_db``.  Taken from the Signal; required
        for array input.  A value that disagrees with the Signal raises.
    signal_power : float, optional
        Signal power per channel for the noise scaling.  Defaults to the mean
        power of ``samples`` over all channels, which counts any silent
        stretch.  Pass the nominal power instead when that is the reference:
        e.g. ``signal_power=1.0`` for a unit-power constellation sent in
        bursts separated by zero guard intervals, whose measured mean falls
        below the power of the symbols.
    reference_bandwidth : float, default 12.5e9
        Noise reference bandwidth ``B_ref`` of ``osnr_db`` in Hz.  The
        standard reference is 0.1 nm of optical bandwidth, ``c · 0.1 nm /
        λ²``: 12.5 GHz in the C band (12.48 GHz at 1550 nm), 17.5 GHz at
        1310 nm.  Outside the C band pass the value for the carrier, e.g.
        ``reference_bandwidth=17.5e9`` for an O-band OSNR in 0.1 nm.
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
        ``osnr_db`` is given for anything but a complex optical Signal, or
        ``reference_bandwidth`` is not positive.

    Notes
    -----
    Complex input gets circular complex noise, its variance split equally
    between I and Q; real input gets real noise of the full variance.  At
    ``sps = 1`` Es/N0 equals the per-sample SNR.

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
    if signal_power is None:
        signal_power = xp.mean(xp.abs(x) ** 2)
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
        logger.info("Adding ASE noise (OSNR target: %.2f dB).", osnr_db)
        num_channels = 1 if x.ndim == 1 else x.shape[0]
        total_power = num_channels * signal_power
        noise_power = (
            total_power
            * sig.sampling_rate
            / (2.0 * reference_bandwidth * db_to_linear(osnr_db, power=True))
        )
        return signal_adapter.wrap_samples(x + _noise(x, xp, noise_power, rng))

    sps = signal_adapter.resolve_fact("sps", sps)
    logger.info("Adding AWGN (Es/N0 target: %.2f dB).", esn0_db)

    # === Es/N0 to sample-level SNR conversion ===
    #
    # Es/N0 = Symbol Energy / Noise Spectral Density
    #
    # For oversampled signals:
    #   - Symbol energy Es = sps * (average sample power)  [sum over sps samples]
    #   - Noise power in full bandwidth = N0 * fs = N0 * (sps * symbol_rate)
    #   - Noise power per sample = N0 * symbol_rate
    #
    # Therefore:
    #   Es/N0 = (sps * P_sample) / (N0 * symbol_rate)
    #   Sample_SNR = P_sample / P_noise_per_sample = P_sample / (N0 * symbol_rate)
    #
    # Relationship: Sample_SNR = Es/N0 / sps
    #
    # Or equivalently: P_noise = P_signal * sps / Es_N0_linear

    esn0_linear = db_to_linear(esn0_db, power=True)

    # Noise power accounting for oversampling
    if esn0_linear <= 1e-20:
        noise_power = signal_power * sps / 1e-20
    else:
        noise_power = signal_power * sps / esn0_linear

    return signal_adapter.wrap_samples(x + _noise(x, xp, noise_power, rng))


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
