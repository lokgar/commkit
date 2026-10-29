"""Signal generation: :func:`generate`, :func:`shape_pulse` and :func:`expand`.

:func:`generate` draws random symbols from a :class:`Constellation` and
pulse-shapes them into a :class:`Signal`, with samples normalized to unit
symbol power (``Es = 1``, average sample power ``1/sps``).  PS-QAM is a
shaped constellation: ``generate(Constellation.qam(64).shaped(nu=0.1),
...)``, whose random bits go through a distribution matcher.  ``code=``
encodes random information bits with a code of :mod:`commkit.coding`
(channel code, matcher or both) before mapping.  ``periodic=True`` shapes the record as one period of a periodic
waveform, for looping on a streaming AWG.  On arrays, :func:`shape_pulse`
pulse-shapes a symbol sequence to unit symbol power and :func:`expand`
inserts zeros between samples.
"""

import math
from typing import Any

import numpy as np

from .. import filtering, mapping
from .._circular import convolve_periodic
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
# existing Signal's samples, so they live here rather than in filtering /
# multirate.py.


def expand(samples: ArrayType, *, factor: int) -> ArrayType:
    """Insert ``factor - 1`` zeros after every sample (zero-stuffing upsample).

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
    ndarray
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
    periodic: bool = False,
) -> ArrayType:
    """Pulse-shape a symbol sequence to unit symbol power.

    Polyphase interpolation by ``sps`` with the pulse taps
    (``scipy.signal.resample_poly``), then ``E[|x|²] = 1/sps``.  With
    ``periodic=True`` the zero-stuffed symbols are convolved circularly with
    the taps instead; away from the record ends both give the same samples.

    Parameters
    ----------
    symbols : array_like
        Symbols, ``(N,)`` or ``(C, N)``.
    sps : int
        Samples per symbol (integer).
    pulse : Pulse or array_like, optional
        Pulse object or taps.  ``None`` zero-stuffs without shaping.
    periodic : bool, default False
        Treat ``symbols`` as one period of a periodic sequence (circular
        convolution): the waveform loops seamlessly, e.g. on a streaming AWG,
        and has no filter transients at its ends.

    Returns
    -------
    ndarray
        Waveform ``(..., N * sps)``, normalized per channel to
        ``E[|x|²] = 1/sps``, same dtype and device as ``symbols``.
    """
    sps = require_integer_sps(sps, "shape_pulse()")
    if pulse is None:
        return normalize(
            expand(symbols, factor=sps), mode="symbol_power", sps=sps, axis=-1
        )
    taps = pulse.taps(sps) if isinstance(pulse, filtering.Pulse) else pulse
    if periodic:
        return normalize(
            convolve_periodic(expand(symbols, factor=sps), taps),
            mode="symbol_power",
            sps=sps,
            axis=-1,
        )
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
    num_symbols: int | None = None,
    *,
    symbol_rate: float,
    sps: int = 1,
    pulse: "filtering.Pulse | ArrayType | None" = None,
    num_channels: int = 1,
    periodic: bool = False,
    code: Any = None,
    num_blocks: int | None = None,
    pad: bool = False,
    rng: int | np.random.Generator | None = None,
    device: str = "cpu",
) -> Signal:
    """Generate random symbols from a constellation, pulse-shaped into a Signal.

    Parameters
    ----------
    constellation : Constellation
        The constellation to draw from.  Every symbol comes from bits: a
        uniform constellation maps random bits; a shaped constellation
        (``pmf`` set) passes random bits through a constant-composition
        distribution matcher, ``CCDM(constellation, block_length=num_symbols)``
        per channel, unless ``code`` gives another matcher.  The shaped
        record then holds each point exactly as often as the composition
        says: the pmf quantized to ``1 / num_symbols``.
    num_symbols : int, optional
        Symbols per channel.  Give either ``num_symbols`` or ``num_blocks``.
        With a ``code``, the symbols must carry whole code blocks unless
        ``pad`` is set (see Notes).
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
    periodic : bool, default False
        Shape the symbols as one period of a periodic sequence (circular
        convolution): the record loops seamlessly, e.g. on a streaming AWG,
        and has no filter transients at its ends.  Later Tx steps that filter
        or resample the record (``fir_filter``, ``upsample``, ``resample``,
        ``apply_phase_noise``) need ``periodic=True`` too; the Signal does not
        record it.
    code : code object, optional
        A code of :mod:`commkit.coding`: random information bits are encoded
        with :func:`commkit.coding.encode`, and the code's output is mapped.
        A FEC code (``LDPCCode``, ...) takes a uniform constellation; a
        distribution matcher (``CCDM``) or probabilistic amplitude shaping
        (``PAS``) carries the constellation it matches onto, which must be
        ``constellation``.  The reference then holds the bits on the air as
        ``bits`` and the information bits as ``info_bits``, with
        ``code_block = (n, k)``.  Matchers run on the host before the
        payload moves to ``device``.
    num_blocks : int, optional
        Code blocks (codewords of a FEC code) per channel, instead of
        ``num_symbols``; needs ``code``.  The record has
        ``num_blocks * code.n / bits_per_symbol`` symbols, which must be
        whole unless ``pad`` is set.
    pad : bool, default False
        With a FEC ``code``: fill the symbols after the last whole block
        with random uncoded bits, so any ``num_symbols`` (an instrument's
        record length, say) or ``num_blocks`` works.  The reference's
        ``bits`` include the filler, ``info_bits`` cover the whole blocks
        only (``reference.num_blocks`` of them).  Not with a distribution
        matcher, whose composition the filler would change.
    rng : int, numpy.random.Generator or None
        Seed or generator (SciPy SPEC 7).
    device : {"cpu", "gpu"}, default "cpu"
        Where the samples and the reference are built.  The random bits are
        always drawn on the host, so a seed gives the same symbols on either
        device; encoding, mapping and pulse shaping run on ``device``.  Prefer
        ``device="gpu"`` to ``generate(...).to("gpu")`` for long records:
        shaping on the GPU is an order of magnitude faster.

    Returns
    -------
    Signal
        Samples normalized per channel to unit symbol power (average sample
        power ``1/sps``), complex64 (float32 for a real constellation such as
        bipolar PAM), with ``constellation``, ``pulse`` (when a Pulse was
        given) and ``reference`` set.

    Raises
    ------
    ValueError
        On a non-positive count, both or neither of ``num_symbols`` and
        ``num_blocks``, ``num_blocks`` or ``pad`` without a code, a record
        that does not hold whole code blocks without ``pad``, ``pad`` with a
        distribution matcher, a FEC code with a shaped constellation, or a
        matcher onto another constellation.

    Notes
    -----
    **Coded payloads.**  A code works on blocks: a FEC code turns every
    ``k`` information bits into a codeword of ``n`` bits (``code.k``,
    ``code.n``), a distribution matcher every ``k`` bits into ``n`` bits of
    point labels.  The decoder needs whole blocks, so the record holds a
    whole number of them; a block may start and end inside a symbol when
    ``n`` is not a multiple of the bits per symbol.  Size the record by
    what matters to the experiment:

    - a number of codewords: ``num_blocks=100`` (about 100 frame errors
      bound the FER statistics, so take more blocks than ``1 / FER``);
    - a fixed record length, such as an arbitrary waveform generator's
      memory: ``num_symbols=N`` with ``pad=True``, which carries
      ``N * bits_per_symbol // n`` blocks and random filler after them.

    Decode the whole blocks, ``decode(llrs[..., : ref.num_blocks * code.n],
    code=code)``, and compare with ``ref.info_bits``.

    **Shaped constellations** without a ``code`` go through one CCDM block
    of ``num_symbols`` per channel: the ideal-matcher limit, with
    negligible rate loss and the pmf quantized to ``1 / num_symbols``, so
    the composition follows the record length.  A transmitter's matcher is
    shorter (a few hundred to a few thousand symbols, at a rate loss of
    about 0.16 bit per symbol for 1024 symbols of PS-64-QAM at 5 bits);
    model one with ``code=CCDM(constellation, block_length=1024)`` and
    ``num_blocks``.

    Examples
    --------
    >>> sig = generate(Constellation.qam(16), 10_000, symbol_rate=32e9,
    ...                sps=2, pulse=RRC(0.1), rng=0)
    >>> coded = generate(Constellation.psk(4), symbol_rate=32e9,
    ...                  code=LDPCCode.dvbs2("1/2"), num_blocks=20, rng=0)
    """
    if not isinstance(constellation, mapping.Constellation):
        raise TypeError(
            "generate(): constellation must be a Constellation, e.g. "
            f"Constellation.qam(16); got {type(constellation).__name__}."
        )
    if (num_symbols is None) == (num_blocks is None):
        raise ValueError("generate(): give exactly one of num_symbols and num_blocks.")
    if code is None and (num_blocks is not None or pad):
        raise ValueError(
            "generate(): num_blocks and pad size a coded record; they need code=."
        )
    counts = {"num_symbols": num_symbols, "num_blocks": num_blocks}
    for name, value in (*counts.items(), ("num_channels", num_channels)):
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int | np.integer):
            raise ValueError(f"generate(): {name} must be an integer, got {value!r}.")
        if value < 1:
            raise ValueError(f"generate(): {name} must be >= 1, got {value}.")
    sps = require_integer_sps(sps, "generate()")
    device = require_device(device, "generate()")
    gen = np.random.default_rng(rng)

    c = constellation
    k = c.bits_per_symbol
    info_bits = None
    # Draw on the host (same data on every device), move the small draw,
    # build symbols and samples on the target device.
    if code is not None or c.pmf is not None:
        from .. import coding

        if code is None:
            # Every shaped symbol comes from bits: one matcher block per channel.
            assert num_symbols is not None
            code = coding.CCDM(c, block_length=int(num_symbols))
        codec = coding.base._codec(code, "generate()")
        matched = getattr(code, "constellation", None)
        if matched is not None and matched != c:
            raise ValueError(
                f"generate(): {code!r} matches onto another constellation than {c!r}."
            )
        if c.pmf is not None and matched is None:
            raise ValueError(
                "generate(): a shaped constellation with a FEC code needs a "
                "distribution matcher: use code=PAS(code, constellation)."
            )
        if pad and matched is not None:
            raise ValueError(
                f"generate(): pad fills with uniform random bits, which would change "
                f"the composition of {code!r}; size the record in whole blocks."
            )
        num_symbols, num_blocks = _coded_layout(num_symbols, num_blocks, k, code.n, pad)
        info_bits = gen.integers(
            0, 2, size=(num_channels, num_blocks * code.k), dtype="int8"
        )
        filler = num_symbols * k - num_blocks * code.n
        if codec.host:
            # Matchers are sequential: encode on the host, move the labels.
            bits = to_device(coding.encode(info_bits, code=code), device)
            info_bits = to_device(info_bits, device)
        else:
            info_bits = to_device(info_bits, device)
            bits = coding.encode(info_bits, code=code)
        if filler:
            extra = gen.integers(0, 2, size=(num_channels, filler), dtype="int8")
            bits = dispatch(bits)[1].concatenate(
                [bits, to_device(extra, device)], axis=1
            )
        symbols = c.map(bits)
        if num_channels == 1:
            info_bits, bits, symbols = info_bits[0], bits[0], symbols[0]
    else:
        assert num_symbols is not None
        total = int(num_symbols) * int(num_channels)
        bits = to_device(gen.integers(0, 2, size=total * k, dtype="int8"), device)
        symbols = c.map(bits)

    if num_channels > 1 and code is None:
        symbols = symbols.reshape(num_channels, num_symbols)
        bits = bits.reshape(num_channels, num_symbols * k)

    samples = shape_pulse(symbols, sps=sps, pulse=pulse, periodic=periodic)

    logger.info(
        "Generated %r%s: %s symbols x %s channel(s), sps=%s, pulse=%r, "
        "periodic=%s, on %s.",
        c,
        "" if code is None else f" coded with {code!r}",
        num_symbols,
        num_channels,
        sps,
        pulse if isinstance(pulse, filtering.Pulse) or pulse is None else "taps",
        periodic,
        device,
    )
    return Signal(
        samples=samples,
        sampling_rate=symbol_rate * sps,
        symbol_rate=symbol_rate,
        constellation=c,
        pulse=pulse if isinstance(pulse, filtering.Pulse) else None,
        reference=Reference(
            symbols=symbols,
            bits=bits,
            info_bits=info_bits,
            code_block=None if code is None else (code.n, code.k),
        ),
    )


def _coded_layout(
    num_symbols: int | None, num_blocks: int | None, m: int, n: int, pad: bool
) -> tuple[int, int]:
    """``(num_symbols, num_blocks)`` of a coded record of ``n``-bit blocks.

    From ``num_blocks``, the symbols that hold them (rounded up with
    ``pad``); from ``num_symbols``, the whole blocks they hold (raising on
    a remainder without ``pad``).  The error names the nearest exact sizes.
    """
    step_symbols = n // math.gcd(n, m)  # the shortest record of whole blocks
    step_blocks = m // math.gcd(n, m)
    if num_blocks is not None:
        bits = int(num_blocks) * n
        if bits % m and not pad:
            low = num_blocks // step_blocks * step_blocks
            valid = f"{low} or {low + step_blocks}" if low else f"{step_blocks}"
            raise ValueError(
                f"generate(): num_blocks={num_blocks} fills {bits / m:g} symbols of "
                f"{m} bits; use num_blocks={valid}, or pad=True to fill the last "
                "symbol with random bits."
            )
        return -(-bits // m), int(num_blocks)
    assert num_symbols is not None
    total = int(num_symbols) * m
    blocks = total // n
    if blocks == 0:
        raise ValueError(
            f"generate(): num_symbols={num_symbols} carries {total} bits, fewer than "
            f"one code block of {n}."
        )
    if total % n and not pad:
        low = num_symbols // step_symbols * step_symbols
        valid = f"{low} or {low + step_symbols}" if low else f"{step_symbols}"
        raise ValueError(
            f"generate(): num_symbols={num_symbols} carries {total} bits, not a "
            f"multiple of the code's block length {n}; use num_symbols={valid}, "
            "num_blocks= to size the record in blocks, or pad=True to fill the "
            "rest with random bits."
        )
    return int(num_symbols), blocks
