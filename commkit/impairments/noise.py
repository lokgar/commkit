"""Additive measurement noise (ASE / thermal) impairments."""

from ..backend import ArrayType, dispatch
from ..core.signal import Signal
from ..helpers import db_to_linear, rewrap_signal, unwrap_signal
from ..logger import logger

__all__ = ["apply_awgn"]


def apply_awgn(
    samples: ArrayType | Signal,
    sps: float | None = None,
    esn0_db: float | None = None,
    seed: int | None = None,
    signal_power: float | None = None,
) -> ArrayType | Signal:
    """
    Adds Additive White Gaussian Noise (AWGN) to a signal based on Es/N0.

    Uses the standard communications definition where Es/N0 is the ratio of
    symbol energy to noise spectral density. This accounts for oversampling
    so the specified Es/N0 matches what you'd measure in the signal bandwidth.

    Parameters
    ----------
    samples : array_like or Signal
        The input signal samples. Shape: (..., N_samples)
    sps : float, optional
        Samples per symbol.  Required for array input; ignored for
        :class:`Signal` input, which always uses the signal's own ``sps``.
    esn0_db : float
        Symbol energy to noise spectral density ratio (Es/N0) in dB.
    seed : int, optional
        Random seed for reproducible noise generation. When ``None`` (default),
        the global RNG state is used.
    signal_power : float, optional
        Reference signal power for the noise scaling.  Defaults to the
        measured mean power of ``samples``.  Pass an explicit value to add
        noise at an absolute level - e.g. ``signal_power=1.0`` on a dark
        (signal-blocked) capture adds the noise a unit-power signal would
        see, which the record's own (near-zero) power cannot define.

    Returns
    -------
    array_like or Signal
        The noisy signal with the same type and backend as the input.  A
        :class:`Signal` returns a new noisy :class:`Signal`.

    Notes
    -----
    - For symbol-rate signals (sps=1), Es/N0 equals the sample-level SNR.
    - For oversampled signals, noise power is scaled by `sps` to maintain
      the correct Es/N0 in the signal bandwidth.
    - For complex signals, noise power is split equally between I and Q.

    Examples
    --------
    >>> sig = generate_pam(order=4, num_symbols=1000, sps=4, symbol_rate=1e6)
    >>> noisy = apply_awgn(sig.samples, esn0_db=20, sps=sig.sps)
    >>> noisy = apply_awgn(sig, esn0_db=20)  # Signal input: sps taken from sig
    """
    x, sig = unwrap_signal(samples)
    if sig is not None:
        # sig.sps is always populated (derived from the required sampling_rate
        # / symbol_rate fields), so the Signal's own value always wins over a
        # supplied sps - see CLAUDE.md, "Signal-Awareness".
        if sps is not None:
            logger.warning(
                "apply_awgn(): ignoring supplied sps=%r for Signal input; "
                "using the signal's own sps=%r instead.",
                sps,
                sig.sps,
            )
        result = apply_awgn(
            x,
            sig.sps,
            esn0_db,
            seed=seed,
            signal_power=signal_power,
        )
        return rewrap_signal(sig, result)

    if sps is None:
        raise ValueError("apply_awgn() requires sps for array input.")
    if esn0_db is None:
        raise ValueError("apply_awgn() requires esn0_db.")

    logger.info("Adding AWGN (Es/N0 target: %.2f dB).", esn0_db)

    samples, xp, _ = dispatch(samples)

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

    if signal_power is None:
        signal_power = xp.mean(xp.abs(samples) ** 2)
    esn0_linear = db_to_linear(esn0_db, power=True)

    # Noise power accounting for oversampling
    if esn0_linear <= 1e-20:
        noise_power = signal_power * sps / 1e-20
    else:
        noise_power = signal_power * sps / esn0_linear

    # Handle complex signals (split power between I and Q)
    is_complex = xp.iscomplexobj(samples)

    rng = xp.random.RandomState(seed) if seed is not None else xp.random
    if is_complex:
        noise_std_component = xp.sqrt(noise_power / 2)
        real_dtype = samples.real.dtype
        noise = rng.normal(0, noise_std_component, samples.shape).astype(
            real_dtype
        ) + 1j * rng.normal(0, noise_std_component, samples.shape).astype(real_dtype)
    else:
        noise_std = xp.sqrt(noise_power)
        noise = rng.normal(0, noise_std, samples.shape).astype(samples.dtype)

    noisy_samples = samples + noise

    return noisy_samples
