"""
Signal generation.

:func:`generate` draws random symbols from a :class:`Constellation` and
pulse-shapes them into a :class:`Signal`, with samples normalized to unit
symbol power (Es = 1, average sample power = 1/sps).  PS-QAM is a shaped
constellation: ``generate(Constellation.qam(64).shaped(nu=0.1), ...)``.
"""

import numpy as np

from .. import filtering, mapping
from ..backend import ArrayType, dispatch, to_device
from ..logger import logger
from ..math import normalize
from ._signal_adapter import require_device, require_integer_sps
from .signal import Reference, Signal

__all__ = ["expand", "generate", "shape_pulse"]

# -----------------------------------------------------------------------------
# WAVEFORM SYNTHESIS PRIMITIVES
# -----------------------------------------------------------------------------
# expand:      Zero-stuffing upsample (pre-shaping primitive for shape_pulse)
# shape_pulse: Symbol sequence -> pulse-shaped waveform (used by generate
#              below and by Preamble/SingleCarrierFrame.to_signal())
#
# Both operate on the raw symbol array a Signal gets *built from*, not on an
# existing Signal's samples, so they live here rather than in filtering.py /
# multirate.py.


def expand(samples: ArrayType, *, factor: int) -> ArrayType:
    """
    Insert ``factor - 1`` zeros after every sample (zero-stuffing upsample).

    The first step of interpolation; the spectral images it creates are
    removed by a following filter.

    Parameters
    ----------
    samples : array_like
        Input samples, time on the last axis.
    factor : int
        Output samples per input sample.

    Returns
    -------
    array_like
        Expanded samples, ``(..., N * factor)``, same dtype and device.
    """
    logger.debug("Inserting zeros (expansion factor=%s).", factor)
    samples, xp, _ = dispatch(samples)
    out = xp.zeros((*samples.shape[:-1], samples.shape[-1] * factor), samples.dtype)
    out[..., ::factor] = samples
    return out


def shape_pulse(
    symbols: ArrayType,
    *,
    sps: int,
    pulse: filtering.Pulse | ArrayType | None = None,
) -> ArrayType:
    """
    Pulse-shape a symbol sequence to unit symbol power.

    Polyphase interpolation by ``sps`` with the pulse taps
    (``scipy.signal.resample_poly``), then ``E[|x|²] = 1/sps``.

    Parameters
    ----------
    symbols : array_like
        Symbols, ``(N,)`` or ``(C, N)``.
    sps : int
        Samples per symbol (integer).
    pulse : Pulse or array_like, optional
        Pulse object or taps.  ``None`` zero-stuffs without shaping.

    Returns
    -------
    array_like
        Waveform ``(..., N * sps)``, same dtype and device as ``symbols``.
    """
    sps = require_integer_sps(sps, "shape_pulse()")
    if pulse is None:
        return normalize(
            expand(symbols, factor=sps), mode="symbol_power", sps=sps, axis=-1
        )
    taps = pulse.taps(sps) if isinstance(pulse, filtering.Pulse) else pulse
    symbols, xp, sp = dispatch(symbols)
    # Cast the float64 taps to the symbol precision so resample_poly does not
    # promote complex64 symbols to complex128.
    h = xp.asarray(taps).astype(symbols.real.dtype)
    res = sp.signal.resample_poly(symbols, sps, 1, window=h, axis=-1)
    if res.dtype != symbols.dtype:
        res = res.astype(symbols.dtype)
    return normalize(res, mode="symbol_power", sps=sps, axis=-1)


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
    device: str = "cpu",
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
        Seed or generator (SciPy SPEC 7).
    device : {"cpu", "gpu"}, default "cpu"
        Where the samples and the reference are built.  The random bits are
        always drawn on the host, so a seed gives the same symbols on either
        device; mapping and pulse shaping run on ``device``.  Prefer
        ``device="gpu"`` to ``generate(...).to("gpu")`` for long records:
        shaping on the GPU is an order of magnitude faster.

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
    device = require_device(device, "generate()")
    gen = np.random.default_rng(rng)

    c = constellation
    k = c.bits_per_symbol
    total = int(num_symbols) * int(num_channels)
    # Draw on the host (same data on every device), move the small draw,
    # build symbols and samples on the target device.
    if c.pmf is None:
        bits = to_device(gen.integers(0, 2, size=total * k, dtype="int8"), device)
        symbols = c.map(bits)
    else:
        idx, xp, _ = dispatch(
            to_device(gen.choice(c.order, size=total, p=c.pmf), device)
        )
        symbols = xp.asarray(c.points.astype(c._storage_dtype()))[idx]
        bits = xp.asarray(c.bit_labels)[idx].reshape(-1)

    if num_channels > 1:
        symbols = symbols.reshape(num_channels, num_symbols)
        bits = bits.reshape(num_channels, num_symbols * k)

    samples = shape_pulse(symbols, sps=sps, pulse=pulse)

    logger.info(
        "Generated %r: %s symbols x %s channel(s), sps=%s, pulse=%r, on %s.",
        c,
        num_symbols,
        num_channels,
        sps,
        pulse if isinstance(pulse, filtering.Pulse) or pulse is None else "taps",
        device,
    )
    return Signal(
        samples=samples,
        sampling_rate=symbol_rate * sps,
        symbol_rate=symbol_rate,
        constellation=c,
        pulse=pulse if isinstance(pulse, filtering.Pulse) else None,
        reference=Reference(symbols=symbols, bits=bits),
    )
