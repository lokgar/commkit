"""
Signal generation.

:func:`generate` draws random symbols from a :class:`Constellation` and
pulse-shapes them into a :class:`Signal`, with samples normalized to unit
symbol power (Es = 1, average sample power = 1/sps).  ``generate_psqam`` is a
bridge for the 1.x PS-QAM scale until module pass 3.2.
"""

from typing import Any

import numpy as np

from .. import filtering, helpers, mapping
from ..backend import ArrayType, dispatch
from ..logger import logger
from ._signal_adapter import require_integer_sps
from .signal import Reference, Signal

# -----------------------------------------------------------------------------
# WAVEFORM SYNTHESIS PRIMITIVES
# -----------------------------------------------------------------------------
# expand:      Zero-stuffing upsample (pre-shaping primitive for shape_pulse)
# shape_pulse: Symbol sequence -> pulse-shaped waveform (used by generate*
#              below and by Preamble/SingleCarrierFrame.to_signal())
#
# Both operate on the raw symbol array a Signal gets *built from*, not on an
# existing Signal's samples, so they live here rather than in filtering.py /
# multirate.py (see CLAUDE.md, "Signal-Awareness").


def expand(samples: ArrayType, factor: int, axis: int = -1) -> ArrayType:
    """
    Inserts zeros between samples (up-sampling by zero-stuffing).

    This operation increases the sampling rate by an integer factor by
    inserting `factor - 1` zeros between each original sample. This is the
    first step in traditional interpolation but requires subsequent
    filtering to remove spectral images.

    Parameters
    ----------
    samples : array_like
        Input signal samples. Shape: (..., N_samples).
    factor : int
        The expansion factor (number of output samples per input sample).
    axis : int, default -1
        The axis along which to perform expansion.

    Returns
    -------
    array_like
        The expanded sample array with zeros inserted.
        Shape: (..., N_samples * factor).
    """
    logger.debug("Inserting zeros (expansion factor=%s).", factor)
    samples, xp, _ = dispatch(samples)

    n_in = samples.shape[axis]
    n_out = n_in * factor

    # Construct output shape
    out_shape = list(samples.shape)
    out_shape[axis] = n_out

    out = xp.zeros(out_shape, dtype=samples.dtype)

    # Slice logic to insert
    # We want out[..., ::factor, ...] = samples
    # Construct slices dynamically
    slices = [slice(None)] * samples.ndim
    slices[axis] = slice(None, None, factor)
    out[tuple(slices)] = samples

    return out


def shape_pulse(
    symbols: ArrayType,
    sps: float,
    pulse_shape: str = "none",
    *,
    duty_cycle: float = 1.0,
    rise_time: float = 0.0,
    filter_span: int = 10,
    rrc_rolloff: float = 0.35,
    rc_rolloff: float = 0.35,
    rz: bool = False,
) -> ArrayType:
    """
    Applies pulse shaping to a symbol sequence.

    Parameters
    ----------
    symbols : array_like
        Input symbol sequence. Shape: (..., N_symbols).
    sps : float
        Samples per symbol (upsampling factor).
    pulse_shape : {"none", "rect", "smoothrect", "gaussian", "rrc", "rc", "sinc"}, default "none"
        Identifier for the pulse shaping filter type.
    duty_cycle : float, default 1.0
        Pulse width in symbol periods, in the range ``(0, 1]``.

        - ``"rect"``, ``"smoothrect"``: total on-time of the pulse (including
          ramps for rect, underlying rect width for smoothrect).
        - ``"gaussian"``: Full-Width at Half-Maximum (FWHM) of the Gaussian.
        - NRZ signals always use 1.0; use 0.5 for canonical RZ.
    rise_time : float, default 0.0
        Edge transition duration in symbol periods. Applies to ``"rect"`` and
        ``"smoothrect"`` only; ignored for all other pulse types.

        - ``"rect"``: duration of each linear ramp. The flat top width is
          ``duty_cycle - 2 * rise_time``. Must satisfy
          ``rise_time <= duty_cycle / 2``.
        - ``"smoothrect"``: 10%-90% erf-edge duration. Smaller values give
          sharper edges; larger values give softer Gaussian-like transitions.
        - ``0.0`` (default): hard rectangular edges for ``"rect"``.
    filter_span : int, default 10
        Filter span in symbols for FIR tap generators
        (``"smoothrect"``, ``"gaussian"``, ``"rrc"``, ``"rc"``, ``"sinc"``).
    rrc_rolloff : float, default 0.35
        Roll-off factor for the Root-Raised-Cosine filter (``"rrc"``). Range [0, 1].
    rc_rolloff : float, default 0.35
        Roll-off factor for the Raised-Cosine filter (``"rc"``). Range [0, 1].
    rz : bool, default False
        Convenience flag for Return-to-Zero signaling. When ``True``, overrides
        ``duty_cycle`` to 0.5 (if not already set below 1.0) and converts
        ``pulse_shape="none"`` to ``"rect"`` automatically.

    Returns
    -------
    array_like
        The pulse-shaped waveform at rate ``sps * symbol_rate``, normalized to
        **unit symbol power** (Es = 1). Average sample power = 1/sps.

    Notes
    -----
    All pulse types produce output satisfying E[|x|²] * sps = 1 (symbol-power
    convention). For peak-normalized samples (e.g. eye diagrams), apply
    ``normalize(..., "peak")`` after.
    """
    logger.debug("Applying pulse shaping: %s", pulse_shape)
    sps = require_integer_sps(sps, "shape_pulse()")

    if rz:
        duty_cycle = 0.5

    symbols, xp, sp = dispatch(symbols)

    if pulse_shape == "none":
        if rz:
            logger.debug("RZ signaling requested, using rect pulse shape")
            pulse_shape = "rect"
        else:
            logger.debug("Pulse shaping disabled, expanding symbols by sps")
            return helpers.normalize(
                expand(symbols, sps, axis=-1),
                "symbol_power",
                sps=sps,
                axis=-1,
            )

    if pulse_shape == "rect":
        h = filtering.rect_taps(sps, duty_cycle=duty_cycle, rise_time=rise_time)
    elif pulse_shape == "smoothrect":
        h = filtering.smoothrect_taps(
            sps, span=filter_span, rise_time=rise_time, duty_cycle=duty_cycle
        )
    elif pulse_shape == "gaussian":
        h = filtering.gaussian_taps(sps, span=filter_span, duty_cycle=duty_cycle)
    elif pulse_shape == "rrc":
        h = filtering.rrc_taps(sps, span=filter_span, rolloff=rrc_rolloff)
    elif pulse_shape == "rc":
        h = filtering.rc_taps(sps, span=filter_span, rolloff=rc_rolloff)
    elif pulse_shape == "sinc":
        # Sinc pulse shaping is equivalent to RRC with rolloff=0
        h = filtering.rrc_taps(sps, span=filter_span, rolloff=0.0)
    else:
        raise ValueError(f"Not implemented pulse shape: {pulse_shape}")

    # Ensure h is on the correct backend and matches symbol precision.
    # Tap generators return float64; casting here prevents scipy's resample_poly
    # from promoting complex64 symbols to complex128.
    h = xp.asarray(h).astype(symbols.real.dtype)

    # Apply Pulse Shaping via Polyphase Resampling
    res = sp.signal.resample_poly(symbols, sps, 1, window=h, axis=-1)
    if res.dtype != symbols.dtype:
        res = res.astype(symbols.dtype)

    return helpers.normalize(res, "symbol_power", sps=sps, axis=-1)


def _legacy_pulse(
    pulse_shape: str,
    *,
    rz: bool = False,
    duty_cycle: float = 1.0,
    rise_time: float = 0.0,
    filter_span: int = 10,
    rrc_rolloff: float = 0.35,
    rc_rolloff: float = 0.35,
) -> filtering.Pulse | None:
    """The pulse object for 1.x ``pulse_shape`` arguments (as ``shape_pulse``).

    Bridge for the string-based factories; removed with them in 2.6.
    """
    if rz:
        duty_cycle = 0.5
        if pulse_shape == "none":
            pulse_shape = "rect"
    if pulse_shape == "none":
        return None
    if pulse_shape == "rect":
        return filtering.Rect(duty_cycle, rise_time)
    if pulse_shape == "smoothrect":
        return filtering.SmoothRect(rise_time, duty_cycle, filter_span)
    if pulse_shape == "gaussian":
        return filtering.Gaussian(duty_cycle, filter_span)
    if pulse_shape == "rrc":
        return filtering.RRC(rrc_rolloff, filter_span)
    if pulse_shape == "rc":
        return filtering.RC(rc_rolloff, filter_span)
    if pulse_shape == "sinc":
        return filtering.RRC(0.0, filter_span)
    raise ValueError(f"Not implemented pulse shape: {pulse_shape}")


# -----------------------------------------------------------------------------
# SIGNAL FACTORIES
# -----------------------------------------------------------------------------


def generate(
    constellation: "mapping.Constellation",
    num_symbols: int,
    *,
    symbol_rate: float,
    sps: int = 1,
    pulse: "filtering.Pulse | ArrayType | None" = None,
    num_channels: int = 1,
    rng: int | np.random.Generator | None = None,
) -> Signal:
    """
    Random symbols from ``constellation``, pulse-shaped into a Signal.

    Parameters
    ----------
    constellation : Constellation
        The constellation to draw from.  Uniform constellations draw random
        bits and map them; a shaped constellation (``pmf`` set) draws points
        with probabilities ``pmf`` and takes their labels as the bits.
    num_symbols : int
        Symbols per channel.
    symbol_rate : float
        Symbol rate in Hz.
    sps : int, default 1
        Samples per symbol (integer).
    pulse : Pulse or array_like, optional
        Transmit pulse (``RRC(0.1)``, ``Rect(0.5)``, ...) or raw taps at
        ``sps``.  ``None`` inserts ``sps - 1`` zeros after each symbol.
    num_channels : int, default 1
        Number of independent channels.  Samples are ``(N,)`` for one
        channel and ``(C, N)`` otherwise; the reference has the same layout.
    rng : int, numpy.random.Generator or None
        Seed or generator (SciPy SPEC 7).  Data is always generated on the
        CPU; move the Signal with ``.to("gpu")``.

    Returns
    -------
    Signal
        Samples normalized to unit symbol power (average sample power
        ``1/sps``), with ``constellation``, ``pulse`` (when a Pulse was given)
        and ``reference`` set.

    Examples
    --------
    >>> sig = generate(Constellation.qam(16), 10_000, symbol_rate=32e9,
    ...                sps=2, pulse=RRC(0.1), rng=0)
    """
    if not isinstance(constellation, mapping.Constellation):
        raise TypeError(
            "generate(): constellation must be a Constellation, e.g. "
            f"Constellation.qam(16); got {type(constellation).__name__}."
        )
    for name, value in (("num_symbols", num_symbols), ("num_channels", num_channels)):
        if isinstance(value, bool) or not isinstance(value, int | np.integer):
            raise ValueError(f"generate(): {name} must be an integer, got {value!r}.")
        if value < 1:
            raise ValueError(f"generate(): {name} must be >= 1, got {value}.")
    sps = require_integer_sps(sps, "generate()")
    gen = np.random.default_rng(rng)

    c = constellation
    k = c.bits_per_symbol
    total = int(num_symbols) * int(num_channels)
    if c.pmf is None:
        bits = gen.integers(0, 2, size=total * k, dtype="int8")
        symbols = c.map(bits)
    else:
        idx = gen.choice(c.order, size=total, p=c.pmf)
        symbols = c.points.astype(c._storage_dtype())[idx]
        bits = c.bit_labels[idx].reshape(-1)

    if num_channels > 1:
        symbols = symbols.reshape(num_channels, num_symbols)
        bits = bits.reshape(num_channels, num_symbols * k)

    samples = _shape(symbols, sps, pulse)

    logger.info(
        "Generated %r: %s symbols x %s channel(s), sps=%s, pulse=%r.",
        c,
        num_symbols,
        num_channels,
        sps,
        pulse if isinstance(pulse, filtering.Pulse) or pulse is None else "taps",
    )
    return Signal(
        samples=samples,
        sampling_rate=symbol_rate * sps,
        symbol_rate=symbol_rate,
        constellation=c,
        pulse=pulse if isinstance(pulse, filtering.Pulse) else None,
        reference=Reference(symbols=symbols, bits=bits),
    )


def _shape(symbols: ArrayType, sps: int, pulse: Any) -> ArrayType:
    """Pulse-shape ``symbols`` to unit symbol power (see :func:`shape_pulse`)."""
    if pulse is None:
        return helpers.normalize(
            expand(symbols, sps, axis=-1), "symbol_power", sps=sps, axis=-1
        )
    taps = pulse.taps(sps) if isinstance(pulse, filtering.Pulse) else pulse
    symbols, xp, sp = dispatch(symbols)
    h = xp.asarray(taps).astype(symbols.real.dtype)
    res = sp.signal.resample_poly(symbols, sps, 1, window=h, axis=-1)
    if res.dtype != symbols.dtype:
        res = res.astype(symbols.dtype)
    return helpers.normalize(res, "symbol_power", sps=sps, axis=-1)


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
) -> Signal:
    """
    PS-QAM bridge with the 1.x scale; removed in module pass 3.2.

    Equivalent to ``generate(Constellation.gray("qam", order, pmf=pmf), ...)``
    with a Maxwell-Boltzmann ``pmf`` for ``nu`` (or the ``nu`` reaching
    ``entropy``).  The pmf is attached without rescaling, so the reference
    symbols have average power below 1.  In 2.0 use
    ``generate(Constellation.qam(order).shaped(nu=...), ...)``.
    """
    sps = require_integer_sps(sps, "generate_psqam()")
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
    return generate(
        mapping.Constellation.gray("qam", order, pmf=pmf),
        num_symbols,
        symbol_rate=symbol_rate,
        sps=sps,
        pulse=_legacy_pulse(
            pulse_shape,
            duty_cycle=duty_cycle,
            filter_span=filter_span,
            rrc_rolloff=rrc_rolloff,
            rc_rolloff=rc_rolloff,
        ),
        num_channels=num_streams,
        rng=seed,
    )
